"""Burn severity products from MESMA fraction maps.

This module turns per-pixel fractional abundance maps (char/PV/NPV/soil) into
burn severity products following Quintano et al. (2023):

* :func:`train_severity_model` — fit a regressor (default RF) from a 4-feature
  fraction matrix to ground-truth Composite Burn Index (CBI) values, with
  5-fold cross-validation R² / RMSE.
* :func:`predict_severity` — apply a trained model to produce a continuous CBI
  map (clipped to [0, 3]) plus a 5-class BARC severity map (Unburned / Low /
  Moderate-Low / Moderate-High / High).
* :func:`calibrate_nbr_thresholds` — derive NBR → severity-class thresholds
  from a co-registered classified reference product (e.g. a BAER Soil Burn
  Severity raster) via per-class NBR medians and their midpoints.
* :func:`classify_severity_from_nbr` — apply calibrated thresholds to a
  single-date NBR map to produce a classified severity map.
* :func:`compute_trajectories` — run MESMA on a dictionary of dated scenes and
  stack the fraction outputs into a single time-series Dataset with dims
  (time, y, x).
* :func:`compare_severity_methods` — Pearson correlation, RMSE, bias, and
  difference map between a MESMA-derived severity product and a dNBR baseline.

Heavy ML imports (scikit-learn) are deferred to function bodies so that
importing :mod:`tanager.severity` stays cheap when only :func:`compute_trajectories`
is used.

Public API (lazy-imported via :mod:`tanager`):

* :func:`train_severity_model`
* :func:`predict_severity`
* :func:`calibrate_nbr_thresholds`
* :func:`classify_severity_from_nbr`
* :func:`compute_trajectories`
* :func:`compare_severity_methods`

Import direction:

* severity.py MAY import from :mod:`tanager.config`, :mod:`tanager.spectral`,
  and :mod:`tanager.unmixing` (for :func:`run_mesma` inside
  :func:`compute_trajectories`).
* severity.py MUST NOT import from :mod:`tanager.lfmc`,
  :mod:`tanager.endmembers`, or :mod:`tanager.validation`.

References:
    Quintano, C., Fernández-Manso, A., Roberts, D. A. (2023). Multiple
        Endmember Spectral Mixture Analysis (MESMA) for monitoring burn
        severity. Remote Sensing of Environment.
    Key, C. H., Benson, N. C. (2006). Landscape Assessment (LA): Sampling and
        Analysis Methods. USDA Forest Service General Technical Report.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BARC classification thresholds (Composite Burn Index → discrete severity)
# ---------------------------------------------------------------------------
# Standard BARC thresholds per Key & Benson (2006), used by USGS/USFS for
# operational classified burn severity maps. Codes are stored as int8 in the
# output severity_map; NaN inputs propagate to a sentinel value of -1 (handled
# explicitly by the caller).
_BARC_THRESHOLDS: Tuple[Tuple[float, int], ...] = (
    (0.10, 0),  # Unburned: CBI < 0.10
    (1.00, 1),  # Low:        0.10 <= CBI < 1.00
    (1.50, 2),  # Moderate-Low:  1.00 <= CBI < 1.50
    (2.25, 3),  # Moderate-High: 1.50 <= CBI < 2.25
    # High: CBI >= 2.25 → 4
)

# Fraction classes used as the feature vector when training/predicting CBI.
# Matches the canonical MESMA fraction order minus shade (which is removed
# by :func:`tanager.unmixing.normalize_fractions` before severity work).
_SEVERITY_FEATURES: Tuple[str, ...] = ("char", "pv", "npv", "soil")

# Default RF hyperparameters per spec; deliberately conservative to avoid
# overfitting the small CBI ground-truth set typical for a single fire.
_DEFAULT_RF_N_ESTIMATORS: int = 200
_DEFAULT_RF_RANDOM_STATE: int = 42
_DEFAULT_CV_FOLDS: int = 5

# CBI is bounded in [0, 3] by Key & Benson (2006); RF can extrapolate
# slightly outside training range so we clip on prediction.
_CBI_MIN: float = 0.0
_CBI_MAX: float = 3.0


# ---------------------------------------------------------------------------
# Feature-matrix helpers
# ---------------------------------------------------------------------------


def _validate_features(
    fractions: xr.Dataset,
    feature_names: Sequence[str],
) -> None:
    """Check the requested feature variables are present in the fractions Dataset."""
    missing = [name for name in feature_names if name not in fractions.data_vars]
    if missing:
        raise ValueError(
            f"fractions Dataset is missing required variable(s): {missing}. "
            f"Available variables: {list(fractions.data_vars)}"
        )


def _flatten_fractions(
    fractions: xr.Dataset,
    feature_names: Sequence[str],
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Stack the named fraction variables into an ``(n_pixels, n_features)`` matrix.

    Args:
        fractions: xarray Dataset with each feature variable shaped (y, x).
        feature_names: Variable names to stack, in feature order.

    Returns:
        Tuple of:
            X: ``(n_pixels, n_features)`` float64 array.
            spatial_shape: original ``(y, x)`` shape so callers can reshape
                predictions back to the scene grid.
    """
    _validate_features(fractions, feature_names)
    first = fractions[feature_names[0]]
    spatial_shape = tuple(first.shape)
    X = np.stack(
        [np.asarray(fractions[name].values, dtype=np.float64).ravel() for name in feature_names],
        axis=1,
    )
    return X, spatial_shape


def _coerce_target(
    target: Union[np.ndarray, xr.DataArray, xr.Dataset, Sequence[float]],
    expected_size: int,
) -> np.ndarray:
    """Coerce a CBI target into a flat float64 array and validate length."""
    if isinstance(target, xr.Dataset):
        raise TypeError(
            "ground_truth_cbi must be a 1-D array or DataArray, not a Dataset"
        )
    if isinstance(target, xr.DataArray):
        arr = np.asarray(target.values, dtype=np.float64).ravel()
    else:
        arr = np.asarray(target, dtype=np.float64).ravel()
    if arr.size != expected_size:
        raise ValueError(
            f"ground_truth_cbi has {arr.size} entries but fractions have "
            f"{expected_size} pixels; sizes must match (NaN target rows are "
            "filtered automatically)."
        )
    return arr


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_severity_model(
    fractions: xr.Dataset,
    ground_truth_cbi: Union[np.ndarray, xr.DataArray, Sequence[float]],
    method: str = "random_forest",
    *,
    n_estimators: int = _DEFAULT_RF_N_ESTIMATORS,
    random_state: int = _DEFAULT_RF_RANDOM_STATE,
    cv_folds: int = _DEFAULT_CV_FOLDS,
    feature_names: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Train a fraction → CBI severity regressor with cross-validated metrics.

    Builds a per-pixel feature matrix from the supplied fraction maps
    (default features: char, pv, npv, soil) and fits a regressor that maps
    those four fractions to a Composite Burn Index value in ``[0, 3]``.
    Cross-validated R² and RMSE are reported on the held-out folds; the
    final model is then re-fit on all valid pixels and returned for use
    with :func:`predict_severity`.

    Args:
        fractions: xarray Dataset with each feature variable shaped (y, x).
            Typically the output of :func:`tanager.unmixing.normalize_fractions`
            (shade removed, remaining fractions sum to 1.0).
        ground_truth_cbi: 1-D array (or DataArray) of CBI values aligned with
            the flattened pixel order of ``fractions``. Length must equal
            ``y * x``. NaN entries in either ``fractions`` or
            ``ground_truth_cbi`` are masked out before training.
        method: Currently ``"random_forest"`` is the only supported method.
        n_estimators: Number of trees in the RF regressor. Default 200.
        random_state: Seed for RF determinism. Default 42.
        cv_folds: K for K-fold cross-validation. Default 5.
        feature_names: Optional override of the feature variables. Defaults
            to ``("char", "pv", "npv", "soil")``.

    Returns:
        Dict with keys:
            ``model``: trained ``RandomForestRegressor`` fit on all valid pixels.
            ``r2``: mean cross-validated R² (float).
            ``rmse``: cross-validated RMSE (float, in CBI units).
            ``method``: the method string used.
            ``feature_names``: tuple of feature variable names used.
            ``n_samples``: number of valid pixels used for training.

    Raises:
        ValueError: If ``method`` is unsupported, the fractions Dataset is
            missing a required feature variable, ``ground_truth_cbi`` length
            does not match the pixel count, or fewer than ``cv_folds`` valid
            pixels remain after NaN filtering.
    """
    if method != "random_forest":
        raise ValueError(
            f"unsupported method {method!r}; only 'random_forest' is implemented"
        )

    feats = tuple(feature_names) if feature_names is not None else _SEVERITY_FEATURES
    if not feats:
        raise ValueError("feature_names must contain at least one variable")

    X, _ = _flatten_fractions(fractions, feats)
    y = _coerce_target(ground_truth_cbi, X.shape[0])

    finite = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    n_valid = int(finite.sum())
    if n_valid < cv_folds:
        raise ValueError(
            f"only {n_valid} valid pixels after NaN filtering; need >= {cv_folds} "
            "for cross-validation"
        )

    X_train = X[finite]
    y_train = y[finite]

    # Heavy ML imports happen here so a bare `import tanager.severity` stays cheap.
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    from .config import parallel_jobs

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=parallel_jobs(),
    )

    r2_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="r2")
    neg_mse_scores = cross_val_score(
        model, X_train, y_train, cv=cv_folds, scoring="neg_mean_squared_error"
    )
    r2 = float(np.mean(r2_scores))
    rmse = float(np.sqrt(-np.mean(neg_mse_scores)))

    # Refit on all valid samples so callers can immediately call predict_severity.
    model.fit(X_train, y_train)

    logger.info(
        "train_severity_model: method=%s n_samples=%d cv_r2=%.4f cv_rmse=%.4f",
        method,
        n_valid,
        r2,
        rmse,
    )

    return {
        "model": model,
        "r2": r2,
        "rmse": rmse,
        "method": method,
        "feature_names": feats,
        "n_samples": n_valid,
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def _resolve_model(
    model: Any,
    feature_names: Optional[Sequence[str]],
) -> Tuple[Any, Tuple[str, ...]]:
    """Accept either the dict returned by :func:`train_severity_model` or a bare estimator."""
    if isinstance(model, Mapping):
        estimator = model.get("model")
        if estimator is None:
            raise ValueError(
                "model dict must contain a 'model' key with a fitted estimator"
            )
        feats = (
            tuple(feature_names)
            if feature_names is not None
            else tuple(model.get("feature_names", _SEVERITY_FEATURES))
        )
    else:
        estimator = model
        feats = tuple(feature_names) if feature_names is not None else _SEVERITY_FEATURES
    return estimator, feats


def predict_severity(
    fractions: xr.Dataset,
    model: Any,
    *,
    feature_names: Optional[Sequence[str]] = None,
) -> dict[str, xr.DataArray]:
    """Apply a trained severity model to a fraction Dataset.

    Produces:

    * a continuous CBI map (clipped to ``[0, 3]``), and
    * a 5-class BARC severity map using the thresholds:

      - 0 (Unburned):       CBI < 0.10
      - 1 (Low):            0.10 <= CBI < 1.00
      - 2 (Moderate-Low):   1.00 <= CBI < 1.50
      - 3 (Moderate-High):  1.50 <= CBI < 2.25
      - 4 (High):           CBI >= 2.25

    Pixels where any feature variable is NaN are propagated as NaN through
    both output maps (the severity map uses ``float64`` so NaN can be
    preserved alongside the integer-valued class codes).

    Args:
        fractions: xarray Dataset containing the feature variables (default
            ``char``, ``pv``, ``npv``, ``soil``) shaped ``(y, x)``.
        model: Either the dict returned by :func:`train_severity_model`
            (preferred) or a bare fitted scikit-learn regressor.
        feature_names: Optional override of the feature variable names.
            Defaults to the value stored on the model dict, or
            ``("char", "pv", "npv", "soil")`` for a bare estimator.

    Returns:
        Dict with keys:
            ``cbi_map``: DataArray of continuous CBI values in ``[0, 3]``
                (NaN where input was NaN), preserving the input ``(y, x)`` dims
                and coordinates.
            ``severity_map``: DataArray of class codes 0..4 as ``float64``
                (NaN where input was NaN), same dims/coords.

    Raises:
        ValueError: If the model dict is missing the ``"model"`` key, or if
            a required feature variable is absent from ``fractions``.
    """
    estimator, feats = _resolve_model(model, feature_names)

    X, spatial_shape = _flatten_fractions(fractions, feats)
    nan_mask = ~np.all(np.isfinite(X), axis=1)

    # Replace NaN with 0 for prediction so sklearn does not warn / raise; we
    # restore NaN on those pixels immediately after prediction.
    X_safe = np.where(np.isnan(X), 0.0, X)
    cbi_flat = np.asarray(estimator.predict(X_safe), dtype=np.float64)
    cbi_flat = np.clip(cbi_flat, _CBI_MIN, _CBI_MAX)
    cbi_flat[nan_mask] = np.nan

    # BARC classification via np.digitize (left-closed bins by default).
    edges = np.array([thresh for thresh, _code in _BARC_THRESHOLDS], dtype=np.float64)
    severity_flat = np.full(cbi_flat.shape, np.nan, dtype=np.float64)
    valid = ~nan_mask
    severity_flat[valid] = np.digitize(cbi_flat[valid], edges, right=False).astype(np.float64)

    # Re-attach (y, x) dims and coords from the first feature variable.
    template = fractions[feats[0]]
    out_dims = template.dims
    out_coords = {name: template.coords[name] for name in out_dims if name in template.coords}

    cbi_map = xr.DataArray(
        cbi_flat.reshape(spatial_shape),
        dims=out_dims,
        coords=out_coords,
        name="cbi",
        attrs={
            "long_name": "composite_burn_index",
            "valid_range": [_CBI_MIN, _CBI_MAX],
            "scale": "CBI (0-3)",
            "reference": "Key & Benson (2006)",
        },
    )
    severity_map = xr.DataArray(
        severity_flat.reshape(spatial_shape),
        dims=out_dims,
        coords=out_coords,
        name="barc_severity",
        attrs={
            "long_name": "barc_severity_class",
            "classification_system": "BARC",
            "class_codes": "0=unburned, 1=low, 2=moderate-low, 3=moderate-high, 4=high",
            "thresholds": "0.10, 1.00, 1.50, 2.25",
            "reference": "Key & Benson (2006)",
        },
    )

    logger.info(
        "predict_severity: predicted %d pixels (%d NaN), CBI range [%.3f, %.3f]",
        int(valid.sum()),
        int(nan_mask.sum()),
        float(np.nanmin(cbi_flat)) if valid.any() else float("nan"),
        float(np.nanmax(cbi_flat)) if valid.any() else float("nan"),
    )

    return {"cbi_map": cbi_map, "severity_map": severity_map}


# ---------------------------------------------------------------------------
# Reference-calibrated NBR classification
# ---------------------------------------------------------------------------


def calibrate_nbr_thresholds(
    nbr_map: Union[xr.DataArray, np.ndarray],
    reference_classes: Union[xr.DataArray, np.ndarray],
    *,
    min_pixels: int = 50,
) -> dict[str, Any]:
    """Calibrate NBR → severity-class thresholds against a reference severity raster.

    Given a single-date NBR map and a co-registered classified reference
    product (e.g. a BAER Soil Burn Severity raster loaded via
    :func:`tanager.validation.load_barc_reference`), compute the median NBR
    within each reference class and place classification thresholds at the
    midpoints between consecutive class medians. Because burned surfaces
    depress NBR, medians must decrease as severity increases; the resulting
    thresholds are returned in descending-NBR order.

    Args:
        nbr_map: 2-D NBR values (DataArray or ndarray).
        reference_classes: Integer class codes on the same grid, ordered so
            larger codes mean higher severity (e.g. 0=Unburned … 3=Mod-High).
            Negative codes are treated as nodata. Classes that a single-date
            NBR cannot separate (e.g. BAER High with few pixels) should be
            merged into a neighbour via the ``code_map`` argument of
            :func:`~tanager.validation.load_barc_reference` before calling.
        min_pixels: Classes with fewer jointly-valid pixels than this are
            dropped from the calibration (logged as a warning).

    Returns:
        Dict with keys:
            ``class_codes``: tuple of retained class codes, ascending severity.
            ``medians``: mapping of class code → median NBR (float).
            ``n_pixels``: mapping of class code → jointly-valid pixel count.
            ``thresholds``: ``(n_classes - 1,)`` float64 array of midpoint
                thresholds in descending NBR order — ``thresholds[i]`` is the
                boundary between ``class_codes[i]`` and ``class_codes[i+1]``.
            ``n_valid``: total jointly-valid pixels used.

    Raises:
        ValueError: If the grids differ in shape, fewer than 2 classes
            survive ``min_pixels`` filtering, or class medians are not
            strictly decreasing with severity (merge the offending classes
            and recalibrate).
    """
    nbr_v = np.asarray(
        nbr_map.values if isinstance(nbr_map, xr.DataArray) else nbr_map,
        dtype=np.float64,
    )
    ref_v = np.asarray(
        reference_classes.values
        if isinstance(reference_classes, xr.DataArray)
        else reference_classes
    )
    if nbr_v.shape != ref_v.shape:
        raise ValueError(
            f"shape mismatch: nbr_map={nbr_v.shape} vs "
            f"reference_classes={ref_v.shape} — align with "
            "load_barc_reference(target_grid=...) first"
        )

    valid = np.isfinite(nbr_v) & np.isfinite(ref_v.astype(np.float64)) & (ref_v >= 0)
    n_valid = int(valid.sum())

    medians: dict[int, float] = {}
    n_pixels: dict[int, int] = {}
    for code in np.unique(ref_v[valid]).astype(int):
        class_mask = valid & (ref_v == code)
        n = int(class_mask.sum())
        if n < min_pixels:
            logger.warning(
                "calibrate_nbr_thresholds: dropping class %d (%d px < min_pixels=%d)",
                code,
                n,
                min_pixels,
            )
            continue
        medians[int(code)] = float(np.median(nbr_v[class_mask]))
        n_pixels[int(code)] = n

    codes = tuple(sorted(medians))
    if len(codes) < 2:
        raise ValueError(
            f"only {len(codes)} class(es) with >= {min_pixels} valid pixels; "
            "need at least 2 to place a threshold"
        )

    med_seq = [medians[c] for c in codes]
    if any(a <= b for a, b in zip(med_seq, med_seq[1:])):
        raise ValueError(
            "class NBR medians are not strictly decreasing with severity: "
            + ", ".join(f"class {c}: {medians[c]:+.3f}" for c in codes)
            + " — merge the non-separable classes (via load_barc_reference's "
            "code_map) and recalibrate"
        )

    thresholds = np.array(
        [(a + b) / 2.0 for a, b in zip(med_seq, med_seq[1:])], dtype=np.float64
    )

    logger.info(
        "calibrate_nbr_thresholds: %d classes over %d px; medians=%s thresholds=%s",
        len(codes),
        n_valid,
        {c: round(medians[c], 3) for c in codes},
        np.round(thresholds, 3).tolist(),
    )

    return {
        "class_codes": codes,
        "medians": medians,
        "n_pixels": n_pixels,
        "thresholds": thresholds,
        "n_valid": n_valid,
    }


def classify_severity_from_nbr(
    nbr_map: xr.DataArray,
    calibration: Mapping[str, Any],
) -> xr.DataArray:
    """Classify a single-date NBR map using reference-calibrated thresholds.

    Args:
        nbr_map: 2-D NBR DataArray (dims/coords are carried to the output).
        calibration: Dict from :func:`calibrate_nbr_thresholds` — only the
            ``class_codes`` and ``thresholds`` keys are required.

    Returns:
        Float64 DataArray of class codes (NaN where the input NBR is NaN),
        same dims/coords as ``nbr_map``. Pixels exactly on a threshold are
        assigned to the less-severe class.

    Raises:
        ValueError: If the calibration dict is missing keys or its sizes are
            inconsistent (``len(thresholds) != len(class_codes) - 1``).
    """
    try:
        codes = tuple(int(c) for c in calibration["class_codes"])
        thresholds = np.asarray(calibration["thresholds"], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(
            f"calibration dict is missing required key {exc}; expected the "
            "dict returned by calibrate_nbr_thresholds"
        ) from None
    if thresholds.size != len(codes) - 1:
        raise ValueError(
            f"calibration has {len(codes)} class codes but {thresholds.size} "
            f"thresholds; expected len(class_codes) - 1"
        )

    nbr_v = np.asarray(nbr_map.values, dtype=np.float64)
    finite = np.isfinite(nbr_v)

    # thresholds are descending in NBR; digitize needs ascending bins. With
    # k = len(thresholds), digitize returns k for NBR above every threshold
    # (least severe) down to 0 below every threshold (most severe), so the
    # class index is k - digitize(...).
    ascending = thresholds[::-1]
    class_flat = np.full(nbr_v.shape, np.nan, dtype=np.float64)
    idx = np.digitize(nbr_v[finite], ascending, right=False)
    code_lookup = np.asarray(codes, dtype=np.float64)
    class_flat[finite] = code_lookup[thresholds.size - idx]

    out = xr.DataArray(
        class_flat,
        dims=nbr_map.dims,
        coords={k: nbr_map.coords[k] for k in nbr_map.dims if k in nbr_map.coords},
        name="nbr_severity",
        attrs={
            "long_name": "nbr_threshold_severity_class",
            "classification_system": "reference-calibrated single-date NBR",
            "class_codes": ", ".join(str(c) for c in codes),
            "thresholds_nbr_descending": ", ".join(f"{t:.4f}" for t in thresholds),
        },
    )

    logger.info(
        "classify_severity_from_nbr: classified %d px (%d NaN) into %d classes",
        int(finite.sum()),
        int((~finite).sum()),
        len(codes),
    )
    return out


# ---------------------------------------------------------------------------
# Multi-temporal trajectories
# ---------------------------------------------------------------------------


def _scenes_share_grid(scenes: Sequence[xr.Dataset]) -> bool:
    """Return True iff every scene has identical ``y`` / ``x`` coordinates."""
    if len(scenes) < 2:
        return True
    first = scenes[0]
    if "y" not in first.coords or "x" not in first.coords:
        return False
    y0 = np.asarray(first.coords["y"].values)
    x0 = np.asarray(first.coords["x"].values)
    for ds in scenes[1:]:
        if "y" not in ds.coords or "x" not in ds.coords:
            return False
        if ds.coords["y"].size != y0.size or ds.coords["x"].size != x0.size:
            return False
        if not (
            np.array_equal(np.asarray(ds.coords["y"].values), y0)
            and np.array_equal(np.asarray(ds.coords["x"].values), x0)
        ):
            return False
    return True


def _coerce_time_coord(keys: Sequence[Any]) -> np.ndarray:
    """Convert the scenes_dict keys into a time coordinate array.

    Tries ``np.datetime64`` first so downstream xarray operations get real
    datetime semantics; falls back to keeping the keys as objects (typically
    strings) when conversion fails for any key.
    """
    try:
        return np.array([np.datetime64(k) for k in keys], dtype="datetime64[ns]")
    except (TypeError, ValueError):
        return np.asarray(keys)


def compute_trajectories(
    scenes_dict: Mapping[str, xr.Dataset],
    library: xr.DataArray,
    *,
    constraints: Optional[Mapping[str, float]] = None,
    bands: Optional[np.ndarray] = None,
    align: bool = True,
) -> xr.Dataset:
    """Run MESMA on a dictionary of dated scenes and stack the results.

    Each value in ``scenes_dict`` is unmixed against the same endmember
    ``library`` (so fraction maps are directly comparable across dates), and
    the resulting per-class fraction Datasets are concatenated along a new
    ``time`` dimension. The output preserves the canonical fraction schema
    from :func:`tanager.unmixing.run_mesma` — variables ``char``, ``pv``,
    ``npv``, ``soil``, ``shade``, ``rmse`` — but with dims ``(time, y, x)``.

    Args:
        scenes_dict: Mapping of date label → ``xr.Dataset`` containing a
            ``reflectance`` variable. Keys are typically ISO-8601 datetime
            strings (e.g. ``"2024-12-15T18:00:00"``); they are passed through
            ``np.datetime64`` so the output ``time`` coord supports the usual
            xarray time selection. Non-datetime-parseable keys are stored
            verbatim.
        library: Endmember library DataArray as produced by
            :func:`tanager.endmembers.build_fire_library`. The same library
            is used for every scene to keep fractions comparable.
        constraints: Optional MESMA constraints dict (forwarded to
            :func:`tanager.unmixing.run_mesma`).
        bands: Optional 1-D wavelength array selecting a band subset for
            unmixing (forwarded to :func:`tanager.unmixing.run_mesma`).
        align: If True (default), check whether all scenes share an identical
            ``(y, x)`` grid and call
            :func:`tanager.io.reproject_to_common_grid` when they don't. Pass
            ``False`` to skip both the check and the reproject (e.g., for
            unit tests on hand-built scenes that already share coords).

    Returns:
        ``xr.Dataset`` with dims ``(time, y, x)``, the canonical MESMA
        fraction variables, ``rmse``, and a ``time`` coordinate carrying the
        parsed datetimes (or the raw keys if parsing failed).

    Raises:
        ValueError: If ``scenes_dict`` is empty, or if ``align=True`` and
            ``reproject_to_common_grid`` rejects the scenes (overlap below
            threshold, missing CRS metadata, etc.).
    """
    if not scenes_dict:
        raise ValueError("scenes_dict cannot be empty")

    keys = list(scenes_dict.keys())
    scenes = [scenes_dict[k] for k in keys]

    if align and len(scenes) >= 2 and not _scenes_share_grid(scenes):
        from tanager.io import reproject_to_common_grid

        logger.info(
            "compute_trajectories: aligning %d scenes onto a common grid via reproject_to_common_grid",
            len(scenes),
        )
        scenes = reproject_to_common_grid(scenes)

    # Heavy import here so tests that mock unmixing don't pay for it at import time.
    from tanager.unmixing import run_mesma

    fraction_datasets: list[xr.Dataset] = []
    for key, scene in zip(keys, scenes):
        logger.info("compute_trajectories: unmixing scene %s", key)
        fractions = run_mesma(scene, library, constraints=constraints, bands=bands)
        fraction_datasets.append(fractions)

    stacked = xr.concat(fraction_datasets, dim="time")
    stacked = stacked.assign_coords(time=("time", _coerce_time_coord(keys)))
    stacked = stacked.transpose("time", ...)

    # Carry forward the engine attr from the first scene (consistent across all
    # scenes since they share the library and constraints).
    if "unmixing_engine" in fraction_datasets[0].attrs:
        stacked.attrs["unmixing_engine"] = fraction_datasets[0].attrs["unmixing_engine"]
    stacked.attrs["n_scenes"] = len(fraction_datasets)

    return stacked


# ---------------------------------------------------------------------------
# Severity-method comparison
# ---------------------------------------------------------------------------


def compare_severity_methods(
    mesma_severity: xr.DataArray,
    dnbr_map: xr.DataArray,
) -> dict[str, Any]:
    """Compare a MESMA-derived severity map against a dNBR baseline.

    Computes the standard agreement metrics between two co-registered
    severity products on the pixels where both are finite. The two inputs
    must already be on a common spatial grid; if grids differ, align with
    :func:`tanager.io.reproject_to_common_grid` before calling.

    Args:
        mesma_severity: MESMA-derived severity DataArray (typically the
            ``cbi_map`` returned by :func:`predict_severity`, or any
            continuous burn-severity index sharing its spatial dims).
        dnbr_map: dNBR (delta NBR) map for the same scene pair, e.g. the
            output of :func:`tanager.spectral.dnbr`.

    Returns:
        Dict with keys:
            ``correlation``: Pearson r over the jointly-finite pixels
                (float, NaN if fewer than 2 finite pixels).
            ``rmse``: Root mean squared error between MESMA and dNBR (float).
            ``bias``: Mean of ``mesma_severity − dnbr_map`` (float).
            ``difference_map``: Per-pixel ``mesma_severity − dnbr_map``
                DataArray with the same dims/coords as the inputs (NaN
                where either input was NaN).
            ``n_valid``: Number of jointly-finite pixels used for the metrics.

    Raises:
        ValueError: If the two DataArrays do not share dimensionality. The
            dim/coord-equality check uses ``.shape`` plus, when both arrays
            have ``y`` and ``x`` coords, exact coord-array equality.
    """
    if mesma_severity.shape != dnbr_map.shape:
        raise ValueError(
            f"shape mismatch: mesma_severity={mesma_severity.shape} vs "
            f"dnbr_map={dnbr_map.shape} — align with reproject_to_common_grid first"
        )
    for axis in ("y", "x"):
        if axis in mesma_severity.coords and axis in dnbr_map.coords:
            if not np.array_equal(
                np.asarray(mesma_severity.coords[axis].values),
                np.asarray(dnbr_map.coords[axis].values),
            ):
                raise ValueError(
                    f"coordinate mismatch on '{axis}' axis — align with "
                    "reproject_to_common_grid first"
                )

    a = np.asarray(mesma_severity.values, dtype=np.float64)
    b = np.asarray(dnbr_map.values, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    n_valid = int(finite.sum())

    if n_valid < 2:
        correlation = float("nan")
        rmse = float("nan")
        bias = float("nan")
    else:
        a_v = a[finite]
        b_v = b[finite]
        diff_v = a_v - b_v
        bias = float(diff_v.mean())
        rmse = float(np.sqrt(np.mean(diff_v**2)))
        # np.corrcoef returns nan for constant inputs; guard explicitly.
        if a_v.std() == 0.0 or b_v.std() == 0.0:
            correlation = float("nan")
        else:
            correlation = float(np.corrcoef(a_v, b_v)[0, 1])

    diff_full = a - b
    diff_full[~finite] = np.nan
    difference_map = xr.DataArray(
        diff_full,
        dims=mesma_severity.dims,
        coords={k: mesma_severity.coords[k] for k in mesma_severity.dims if k in mesma_severity.coords},
        name="severity_difference",
        attrs={
            "long_name": "mesma_severity_minus_dnbr",
            "n_valid": n_valid,
        },
    )

    logger.info(
        "compare_severity_methods: n_valid=%d corr=%.4f rmse=%.4f bias=%+.4f",
        n_valid,
        correlation,
        rmse,
        bias,
    )

    return {
        "correlation": correlation,
        "rmse": rmse,
        "bias": bias,
        "difference_map": difference_map,
        "n_valid": n_valid,
    }
