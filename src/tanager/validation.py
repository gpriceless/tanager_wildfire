"""Validation and accuracy assessment for Tanager-1 analysis products.

This module compares Tanager-derived MESMA fractions and severity maps against
external reference datasets (AVIRIS-3 fractions, USGS BARC severity, EMIT/PRISMA
sensor cross-comparisons) and computes accuracy metrics for both continuous and
classified products.

Public API (lazy-imported via :mod:`tanager`):

* :func:`load_aviris3_reference` — load AVIRIS-3 fraction product, aggregate to
  Tanager 30 m grid.
* :func:`load_barc_reference` — load USGS BARC classified severity GeoTIFF and
  align to a Tanager scene grid.
* :func:`compute_accuracy` — R², RMSE, MAE, bias for continuous predictions;
  overall accuracy, Cohen's kappa, F1, confusion matrix for classified
  predictions.
* :func:`compare_sensors` — comparative metrics and improvement ratios for
  Tanager vs. a reference sensor (EMIT / PRISMA), used for the +5 competition
  tie-breaker.

Heavy ML imports (scikit-learn) and raster I/O (rasterio) are deferred to
function bodies so that importing :mod:`tanager.validation` stays cheap.

Import direction:

* validation.py MAY import from any other tanager module — it sits at the top
  of the dependency tree.
* No other tanager module should import FROM validation.py.

References:
    Roy, D. P., Boschetti, L., Trigg, S. N. (2006). Remote sensing of fire
        severity: assessing the performance of the normalized burn ratio. IEEE
        Geoscience and Remote Sensing Letters.
    Eidenshink, J., Schwind, B., Brewer, K., Zhu, Z.-L., Quayle, B., Howard, S.
        (2007). A project for monitoring trends in burn severity. Fire Ecology.
"""

from __future__ import annotations

import logging
from os import PathLike
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

FilePath = Union[str, PathLike]

# AVIRIS-3 native ground sample distance is ~3-4 m. Default Tanager-1 ortho_sr
# resolution is 30 m, so the typical aggregation factor is ~8x8 to ~10x10 pixels.
_DEFAULT_TARGET_RESOLUTION_M = 30.0

# Mapping from common AVIRIS-3 fraction product band names to the canonical
# variable names used by tanager.unmixing output (char/pv/npv/soil/shade).
# Different processing pipelines (e.g. ISOFIT-Tetracorder, custom MESMA runs)
# label these columns differently, so we accept several aliases.
_AVIRIS3_FRACTION_ALIASES: Mapping[str, str] = {
    "char": "char",
    "burn": "char",
    "burned": "char",
    "ash": "char",
    "pv": "pv",
    "gv": "pv",
    "vegetation": "pv",
    "green_vegetation": "pv",
    "npv": "npv",
    "non_photosynthetic_vegetation": "npv",
    "litter": "npv",
    "soil": "soil",
    "bare_soil": "soil",
    "rock": "soil",
    "shade": "shade",
    "shadow": "shade",
}

# BARC severity codes used by USGS/MTBS rasters (Eidenshink et al. 2007). Some
# products use the "thematic" 1-5 scheme; others use 0-4 or even floating-point.
# We normalise to the same 0..4 scheme as :func:`tanager.severity.predict_severity`.
_BARC_CODE_MAP: Mapping[int, int] = {
    0: 0,  # Unburned (or no data treated as unburned in some products)
    1: 1,  # Low
    2: 2,  # Moderate-Low
    3: 3,  # Moderate-High
    4: 4,  # High
    5: 4,  # MTBS "Increased Greenness" sometimes coded 5 — fold into High class
}


def load_aviris3_reference(
    filepath: FilePath,
    target_resolution: float = _DEFAULT_TARGET_RESOLUTION_M,
) -> xr.Dataset:
    """Load an AVIRIS-3 fraction reference product, aggregated to Tanager grid.

    AVIRIS-3 ships per-pixel fractional abundance products at native 3-4 m GSD.
    For competition validation we compare against Tanager fractions on the
    canonical 30 m ortho_sr grid, so this loader spatially aggregates the
    AVIRIS-3 raster (mean reflectance / fraction within each 30 m cell) and
    renames AVIRIS-3 fraction columns to the canonical Tanager schema
    (``char``, ``pv``, ``npv``, ``soil``, ``shade``).

    Both GeoTIFF (multi-band) and NetCDF inputs are supported. GeoTIFF bands
    are mapped to fraction names via ``band_descriptions`` tags (rasterio
    exposes these as ``descriptions``); if descriptions are missing, the
    function expects the canonical 4-band order char/pv/npv/soil.

    Args:
        filepath: Path to the AVIRIS-3 fraction product (GeoTIFF or NetCDF).
        target_resolution: Target ground sample distance in metres. Defaults
            to 30 m to match Tanager ortho_sr.

    Returns:
        xr.Dataset with:

        * Data variables with canonical names (char, pv, npv, soil and
          optionally shade), dtype float32, dims ``(y, x)``.
        * ``y``, ``x`` coordinates in target CRS.
        * Attribute ``source = "aviris3"``.
        * Attribute ``target_resolution = float`` (metres).

    Raises:
        FileNotFoundError: If ``filepath`` does not exist on disk.
        ValueError: If the file cannot be parsed as either GeoTIFF or NetCDF
            with a recognisable fraction schema.
    """
    import os

    from pathlib import Path

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"AVIRIS-3 reference product not found: {filepath}. "
            "Pre-fire and Eaton-Fire AVIRIS-3 overflights must be staged "
            "locally before running validation (see ORNL DAAC docs)."
        )

    suffix = path.suffix.lower()
    if suffix in (".nc", ".cdf", ".nc4"):
        ds = _load_aviris3_netcdf(path)
    else:
        ds = _load_aviris3_raster(path)

    aggregated = _aggregate_to_resolution(ds, target_resolution)
    aggregated.attrs["source"] = "aviris3"
    aggregated.attrs["target_resolution"] = float(target_resolution)
    aggregated.attrs["original_path"] = os.fspath(filepath)
    return aggregated


def _load_aviris3_raster(path: "Path") -> xr.Dataset:  # type: ignore[name-defined]
    """Open a GeoTIFF / ENVI fraction raster and label bands as fraction names."""
    import rasterio

    with rasterio.open(path) as src:
        descriptions = list(src.descriptions or ())
        bands = src.read().astype(np.float32)  # shape (n_bands, rows, cols)
        transform = src.transform
        crs = src.crs

    n_bands = bands.shape[0]
    if descriptions and any(descriptions):
        var_names = [
            _AVIRIS3_FRACTION_ALIASES.get((d or "").lower().strip(), (d or f"band_{i + 1}").lower())
            for i, d in enumerate(descriptions)
        ]
    elif n_bands >= 4:
        defaults = ["char", "pv", "npv", "soil", "shade"]
        var_names = defaults[:n_bands]
    else:
        raise ValueError(
            f"AVIRIS-3 raster {path} has {n_bands} bands and no band descriptions; "
            "cannot infer fraction schema. Expected at least 4 bands "
            "(char, pv, npv, soil)."
        )

    rows, cols = bands.shape[1], bands.shape[2]
    if transform is not None:
        # Pixel-centre coordinates so coarsen can compute simple group means.
        xs = transform.c + (np.arange(cols) + 0.5) * transform.a
        ys = transform.f + (np.arange(rows) + 0.5) * transform.e
    else:
        xs = np.arange(cols, dtype=np.float64)
        ys = np.arange(rows, dtype=np.float64)

    data_vars: dict[str, tuple] = {}
    for i, name in enumerate(var_names):
        data_vars[name] = (("y", "x"), bands[i])

    ds = xr.Dataset(
        data_vars,
        coords={"y": ys, "x": xs},
    )
    if crs is not None:
        ds.attrs["crs"] = str(crs)
    return ds


def _load_aviris3_netcdf(path: "Path") -> xr.Dataset:  # type: ignore[name-defined]
    """Open a NetCDF fraction product and rename to canonical schema."""
    ds = xr.open_dataset(path)
    rename: dict[str, str] = {}
    for var in list(ds.data_vars):
        canonical = _AVIRIS3_FRACTION_ALIASES.get(var.lower().strip())
        if canonical and canonical != var:
            rename[var] = canonical
    if rename:
        ds = ds.rename(rename)
    # Keep only canonical fraction variables we recognise.
    canonical_names = set(_AVIRIS3_FRACTION_ALIASES.values())
    drop = [v for v in ds.data_vars if v not in canonical_names]
    if drop:
        ds = ds.drop_vars(drop)
    return ds


def _aggregate_to_resolution(
    ds: xr.Dataset,
    target_resolution: float,
) -> xr.Dataset:
    """Coarsen a fraction Dataset to a target ground sample distance.

    Uses xarray ``coarsen`` with ``boundary="trim"`` so trailing pixels that
    don't fit a full coarsening window are dropped. Means are computed with
    ``skipna=True`` so masked / NaN pixels don't bias the aggregate.
    """
    if "x" not in ds.coords or "y" not in ds.coords:
        return ds

    x_vals = ds["x"].values
    y_vals = ds["y"].values
    if len(x_vals) < 2 or len(y_vals) < 2:
        return ds

    src_dx = float(abs(x_vals[1] - x_vals[0]))
    src_dy = float(abs(y_vals[1] - y_vals[0]))
    if src_dx <= 0 or src_dy <= 0:
        return ds

    factor_x = max(1, int(round(target_resolution / src_dx)))
    factor_y = max(1, int(round(target_resolution / src_dy)))
    if factor_x == 1 and factor_y == 1:
        return ds

    logger.info(
        "Aggregating AVIRIS-3 fractions: %d×%d → factor (y=%d, x=%d) for %.1f m target",
        len(y_vals),
        len(x_vals),
        factor_y,
        factor_x,
        target_resolution,
    )
    coarsened = ds.coarsen(y=factor_y, x=factor_x, boundary="trim").mean(skipna=True)
    return coarsened


def load_barc_reference(
    filepath: FilePath,
    *,
    code_map: Optional[Mapping[int, int]] = None,
    target_grid: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Load a USGS BARC classified-severity GeoTIFF as a Tanager-aligned DataArray.

    BARC (Burned Area Reflectance Classification) products are integer-coded
    rasters where each pixel is one of: 0=Unburned, 1=Low, 2=Moderate-Low,
    3=Moderate-High, 4=High. Some MTBS variants use 5 for "increased greenness"
    or use no-data values; ``code_map`` lets the caller override the default
    mapping to the canonical 0..4 scheme used by
    :func:`tanager.severity.predict_severity`.

    When ``target_grid`` is provided, the BARC raster is reprojected via
    nearest-neighbour resampling onto the target grid's CRS / transform so the
    output can be subtracted from a Tanager severity map directly.

    Args:
        filepath: Path to the BARC GeoTIFF (or any rasterio-readable raster).
        code_map: Optional override for translating BARC integer codes to the
            canonical 0..4 severity scheme. Pixels with codes not present in
            ``code_map`` are passed through unchanged.
        target_grid: Optional DataArray whose ``y`` and ``x`` coordinates and
            ``crs`` attribute / spatial_ref describe the destination grid.
            When supplied the BARC raster is reprojected to that grid using
            nearest-neighbour resampling.

    Returns:
        xr.DataArray with integer dtype, dims ``(y, x)``, and an attribute
        ``source = "barc"``. NoData pixels are encoded as ``-1`` so callers
        can mask them prior to computing accuracy metrics.

    Raises:
        FileNotFoundError: If ``filepath`` does not exist.
    """
    from pathlib import Path

    import rasterio

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"BARC reference product not found: {filepath}. "
            "BARC severity GeoTIFFs are obtained from USGS / MTBS for the "
            "fire of interest (e.g. Eaton 2025)."
        )

    with rasterio.open(path) as src:
        data = src.read(1)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs

    mapping = dict(_BARC_CODE_MAP)
    if code_map is not None:
        mapping.update({int(k): int(v) for k, v in code_map.items()})

    arr = np.asarray(data)
    out = arr.astype(np.int16, copy=True)
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        out = np.where(arr == nodata, -1, out)
    elif np.issubdtype(arr.dtype, np.floating):
        out = np.where(np.isnan(arr), -1, out.astype(np.int16))

    if mapping:
        # Vectorised remap: build a lookup over the unique codes present so
        # we don't allocate a 2**16-element table.
        unique_codes = np.unique(out)
        remapped = out.copy()
        for code in unique_codes:
            if int(code) in mapping:
                remapped = np.where(out == code, mapping[int(code)], remapped)
        out = remapped

    rows, cols = out.shape
    if transform is not None:
        xs = transform.c + (np.arange(cols) + 0.5) * transform.a
        ys = transform.f + (np.arange(rows) + 0.5) * transform.e
    else:
        xs = np.arange(cols, dtype=np.float64)
        ys = np.arange(rows, dtype=np.float64)

    da = xr.DataArray(
        out,
        dims=("y", "x"),
        coords={"y": ys, "x": xs},
        attrs={"source": "barc", "nodata": -1},
    )
    if crs is not None:
        da.attrs["crs"] = str(crs)

    if target_grid is not None:
        da = _align_to_target_grid(da, target_grid)

    return da


def _align_to_target_grid(
    array: xr.DataArray,
    target: xr.DataArray,
) -> xr.DataArray:
    """Align ``array`` onto the spatial coordinates of ``target`` (nearest)."""
    if "y" not in target.coords or "x" not in target.coords:
        return array
    return array.interp(
        y=target["y"],
        x=target["x"],
        method="nearest",
        kwargs={"fill_value": -1},
    ).astype(array.dtype)


_ArrayLike = Union[np.ndarray, xr.DataArray, Sequence[float]]


def compute_accuracy(
    predicted: _ArrayLike,
    observed: _ArrayLike,
    metric_type: str = "continuous",
    *,
    nodata: Optional[int] = -1,
) -> dict[str, Any]:
    """Compute accuracy metrics for predicted vs. observed values.

    For ``metric_type="continuous"`` the function returns standard regression
    metrics (R², RMSE, MAE, mean bias, Spearman correlation, valid-pair count).
    For ``metric_type="classified"`` it returns overall accuracy, Cohen's
    kappa, per-class F1 scores (unweighted/macro and per-class array), and the
    raw confusion matrix.

    NaN values (continuous) and ``nodata`` codes (classified) are excluded
    pairwise: a sample is dropped if either ``predicted`` or ``observed`` is
    invalid at that index. The number of valid pairs used is included in the
    result dict for downstream auditing.

    Args:
        predicted: Predicted values from a Tanager product. Accepts a NumPy
            array, xarray DataArray, or any 1-D sequence of floats / ints.
        observed: Reference / ground-truth values. Same accepted types as
            ``predicted``.
        metric_type: Either ``"continuous"`` or ``"classified"``.
        nodata: NoData sentinel for classified inputs (default ``-1``, which
            matches the encoding used by :func:`load_barc_reference`).

    Returns:
        Dict with metric names → values. Always contains ``n_valid``.

    Raises:
        ValueError: If ``metric_type`` is not ``"continuous"`` or
            ``"classified"``, or if shapes are incompatible / no valid pairs
            remain after masking.
    """
    pred = np.asarray(predicted).ravel()
    obs = np.asarray(observed).ravel()

    if pred.shape != obs.shape:
        raise ValueError(
            f"Shape mismatch between predicted ({pred.shape}) and observed "
            f"({obs.shape}). Both inputs must have the same number of elements."
        )

    if metric_type == "continuous":
        valid = ~(np.isnan(pred.astype(np.float64)) | np.isnan(obs.astype(np.float64)))
    elif metric_type == "classified":
        valid = np.ones(pred.shape, dtype=bool)
        if nodata is not None:
            valid &= pred != nodata
            valid &= obs != nodata
        if np.issubdtype(pred.dtype, np.floating):
            valid &= ~np.isnan(pred)
        if np.issubdtype(obs.dtype, np.floating):
            valid &= ~np.isnan(obs)
    else:
        raise ValueError(
            f"Unknown metric_type {metric_type!r}; expected 'continuous' or 'classified'."
        )

    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError(
            "No valid (non-NaN, non-nodata) pairs remain after masking; "
            "cannot compute accuracy metrics."
        )

    pred_v = pred[valid]
    obs_v = obs[valid]

    if metric_type == "continuous":
        return _continuous_metrics(pred_v.astype(np.float64), obs_v.astype(np.float64), n_valid)
    return _classified_metrics(pred_v, obs_v, n_valid)


def _continuous_metrics(pred: np.ndarray, obs: np.ndarray, n_valid: int) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    r2 = float(r2_score(obs, pred))
    rmse = float(np.sqrt(mean_squared_error(obs, pred)))
    mae = float(mean_absolute_error(obs, pred))
    bias = float(np.mean(pred - obs))

    spearman = _spearman(pred, obs)

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "spearman": spearman,
        "n_valid": n_valid,
    }


def _classified_metrics(pred: np.ndarray, obs: np.ndarray, n_valid: int) -> dict[str, Any]:
    from sklearn.metrics import (
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
    )

    pred_int = pred.astype(np.int64)
    obs_int = obs.astype(np.int64)

    accuracy = float(np.mean(pred_int == obs_int))
    kappa = float(cohen_kappa_score(obs_int, pred_int))
    labels = np.unique(np.concatenate([obs_int, pred_int]))
    cm = confusion_matrix(obs_int, pred_int, labels=labels)
    per_class_f1 = f1_score(
        obs_int,
        pred_int,
        labels=labels,
        average=None,
        zero_division=0,
    )
    macro_f1 = float(
        f1_score(
            obs_int,
            pred_int,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "kappa": kappa,
        "f1_macro": macro_f1,
        "f1_per_class": np.asarray(per_class_f1, dtype=np.float64),
        "labels": labels.astype(np.int64),
        "confusion_matrix": np.asarray(cm, dtype=np.int64),
        "n_valid": n_valid,
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman rank correlation without requiring SciPy."""
    if len(a) < 2:
        return float("nan")
    rank_a = _rank_average_ties(a)
    rank_b = _rank_average_ties(b)
    if np.std(rank_a) == 0 or np.std(rank_b) == 0:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def _rank_average_ties(values: np.ndarray) -> np.ndarray:
    """Return ranks with average-of-ties tie-breaking (matches scipy.stats.rankdata)."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    # Adjust for ties: replace each tied group with its mean rank.
    sorted_vals = values[order]
    i = 0
    while i < len(sorted_vals):
        j = i + 1
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            mean_rank = float(np.mean(ranks[order[i:j]]))
            ranks[order[i:j]] = mean_rank
        i = j
    return ranks


def compare_sensors(
    tanager_result: _ArrayLike,
    reference_result: _ArrayLike,
    ground_truth: _ArrayLike,
    sensor_name: str,
    *,
    metric_type: str = "continuous",
) -> dict[str, Any]:
    """Compare Tanager-derived predictions to a reference sensor's predictions.

    Both ``tanager_result`` and ``reference_result`` are evaluated against the
    same ``ground_truth`` reference. Improvement ratios indicate how much
    better Tanager performs versus the reference sensor (positive = Tanager
    wins) and feed the +5 competition tie-breaker for Tanager vs. EMIT or
    Tanager vs. PRISMA comparisons.

    Args:
        tanager_result: Tanager prediction array (continuous or classified).
        reference_result: Reference sensor prediction (e.g. EMIT, PRISMA).
        ground_truth: Reference / ground-truth labels (e.g. AVIRIS-3, BARC,
            field CBI).
        sensor_name: Human-readable name for the reference sensor (used in
            the comparison table).
        metric_type: ``"continuous"`` or ``"classified"`` — passed through to
            :func:`compute_accuracy`.

    Returns:
        Dict with keys:

        * ``tanager_metrics`` — :func:`compute_accuracy` output for Tanager.
        * ``reference_metrics`` — :func:`compute_accuracy` output for the
          reference sensor.
        * ``improvement_ratios`` — dict of relative improvements (continuous:
          ``r2_improvement``, ``rmse_reduction_pct``, ``mae_reduction_pct``,
          ``bias_change``; classified: ``accuracy_gain``, ``kappa_gain``,
          ``f1_macro_gain``).
        * ``comparison_table`` — a pandas DataFrame suitable for inclusion in
          a competition submission appendix (one row per metric).
        * ``sensor_name`` — echoed for traceability.

    Raises:
        ValueError: If ``metric_type`` is unsupported or if shapes mismatch.
    """
    tanager_metrics = compute_accuracy(tanager_result, ground_truth, metric_type=metric_type)
    reference_metrics = compute_accuracy(reference_result, ground_truth, metric_type=metric_type)

    if metric_type == "continuous":
        ref_rmse = reference_metrics["rmse"]
        ref_mae = reference_metrics["mae"]
        improvements: dict[str, float] = {
            "r2_improvement": tanager_metrics["r2"] - reference_metrics["r2"],
            "rmse_reduction_pct": (
                (ref_rmse - tanager_metrics["rmse"]) / ref_rmse * 100.0
                if ref_rmse > 0
                else 0.0
            ),
            "mae_reduction_pct": (
                (ref_mae - tanager_metrics["mae"]) / ref_mae * 100.0
                if ref_mae > 0
                else 0.0
            ),
            "bias_change": tanager_metrics["bias"] - reference_metrics["bias"],
        }
    else:
        improvements = {
            "accuracy_gain": tanager_metrics["accuracy"] - reference_metrics["accuracy"],
            "kappa_gain": tanager_metrics["kappa"] - reference_metrics["kappa"],
            "f1_macro_gain": tanager_metrics["f1_macro"] - reference_metrics["f1_macro"],
        }

    comparison_table = _build_comparison_table(
        tanager_metrics,
        reference_metrics,
        sensor_name,
        metric_type,
    )

    return {
        "tanager_metrics": tanager_metrics,
        "reference_metrics": reference_metrics,
        "improvement_ratios": improvements,
        "comparison_table": comparison_table,
        "sensor_name": sensor_name,
    }


def _build_comparison_table(
    tanager_metrics: Mapping[str, Any],
    reference_metrics: Mapping[str, Any],
    sensor_name: str,
    metric_type: str,
) -> Any:
    """Build a per-metric comparison DataFrame for competition submissions."""
    import pandas as pd

    if metric_type == "continuous":
        scalar_keys = ("r2", "rmse", "mae", "bias", "spearman")
    else:
        scalar_keys = ("accuracy", "kappa", "f1_macro")

    rows = []
    for key in scalar_keys:
        if key in tanager_metrics and key in reference_metrics:
            t_val = float(tanager_metrics[key])
            r_val = float(reference_metrics[key])
            rows.append(
                {
                    "metric": key,
                    "tanager": t_val,
                    sensor_name: r_val,
                    "delta": t_val - r_val,
                }
            )

    return pd.DataFrame(rows)
