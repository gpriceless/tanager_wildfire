"""Live Fuel Moisture Content (LFMC) products from Tanager-1 reflectance.

This module estimates per-pixel live fuel moisture from hyperspectral
reflectance using two complementary approaches:

* **Tier 1 — Spectral indices** (Quan et al. 2021):
  Eight water-sensitive indices computed by :func:`compute_lfmc_indices`,
  including SAI (Spectral Absorption Index) at 970, 1200, and 1660 nm,
  three NDWI variants (1240, 1640, 2130 nm), the Water Index WI = R900/R970,
  and continuum-removal band depths at the four water-absorption wavelengths.
  Indices are interpretable proxies for canopy water content but do not
  yield an absolute LFMC percent on their own.

* **Tier 2 — PLSRegression** (Peterson & Roberts 2014):
  :func:`train_lfmc_plsr` fits a Partial Least Squares regression from full
  ~330-band reflectance (bad bands removed) to ground-truth LFMC observations
  from Globe-LFMC 2.0. Returns the trained model plus 5-fold-CV R² / RMSE
  and per-band VIP (Variable Importance in Projection) scores so callers can
  verify the model is keying on water-absorption bands rather than spurious
  features.

* :func:`load_globe_lfmc` provides the ground-truth side: filtered
  GeoDataFrame of Globe-LFMC 2.0 observations, optionally restricted to a
  bounding box, vegetation types, or co-located with Tanager scene dates.

* :func:`predict_lfmc` applies a trained model to a scene and returns a
  per-pixel LFMC DataArray plus an uncertainty DataArray and a
  ``low_lfmc < 60%`` flag (the nonlinear regime per Roberts et al. 2006).

Heavy ML / vector imports (scikit-learn, geopandas) are deferred to function
bodies so importing :mod:`tanager.lfmc` stays cheap.

Public API (lazy-imported via :mod:`tanager`):

* :func:`compute_lfmc_indices`
* :func:`load_globe_lfmc`
* :func:`train_lfmc_plsr`
* :func:`predict_lfmc`

Import direction:

* lfmc.py MAY import from :mod:`tanager.config` and :mod:`tanager.spectral`
  (for ``continuum_removal``, ``select_bands``, and the
  ``_normalized_difference`` helper).
* lfmc.py MUST NOT import from :mod:`tanager.severity`,
  :mod:`tanager.unmixing`, :mod:`tanager.endmembers`, or
  :mod:`tanager.validation`.

References:
    Peterson, S. H., Roberts, D. A. (2014). Mapping live fuel moisture using
        partial least squares regression. Remote Sensing of Environment.
    Quan, X., He, B., Yebra, M., et al. (2021). A spectral absorption index
        for live fuel moisture content estimation. Remote Sensing.
    Roberts, D. A., Dennison, P. E., Roth, K. L. (2006). Methods for mapping
        live fuel moisture using AVIRIS data. International Journal of
        Wildland Fire.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from tanager.spectral import scene_reflectance as _scene_reflectance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Water-absorption band targets (nm)
# ---------------------------------------------------------------------------
# Centre wavelengths of the canonical liquid-water absorption features in the
# 0.9–2.2 µm region. These are nominal targets — actual band selection uses
# nearest-neighbour matching on the Tanager 5 nm grid.
_WATER_ABSORPTION_TARGETS_NM: Tuple[float, ...] = (970.0, 1200.0, 1660.0, 2100.0)

# Default left/right shoulder offsets for SAI computation, in nm.
# SAI fits a straight-line continuum between local maxima flanking the
# absorption feature. Offsets are searched within these windows.
_SAI_LEFT_WINDOW_NM: Tuple[float, float] = (-150.0, -20.0)
_SAI_RIGHT_WINDOW_NM: Tuple[float, float] = (20.0, 150.0)

# LFMC physical bounds (percent dry-mass basis).
# LFMC > 300% is unphysical (some succulents reach ~250%, but anything above
# 300% in a remotely-sensed pixel is a model artifact). Used to clip
# predict_lfmc output.
_LFMC_MIN_PERCENT: float = 0.0
_LFMC_MAX_PERCENT: float = 300.0

# The "low LFMC" flag threshold per Roberts et al. (2006):
# below 60% is the nonlinear / fire-prone regime where small moisture changes
# produce large changes in flammability. Used by predict_lfmc to set the
# `low_lfmc` boolean DataArray.
_LFMC_LOW_THRESHOLD_PERCENT: float = 60.0

# NDWI variants (numerator/denominator wavelength pairs in nm).
# Output convention: (R860 - Rtarget) / (R860 + Rtarget) so positive values
# indicate dry conditions and negative values wet.
_NDWI_PAIRS: Tuple[Tuple[str, float, float], ...] = (
    ("NDWI_1240", 860.0, 1240.0),
    ("NDWI_1640", 860.0, 1640.0),
    ("NDWI_2130", 860.0, 2130.0),
)

# Water Index numerator/denominator (Peñuelas et al. 1993):
# WI = R900 / R970. Values < 1 indicate water absorption.
_WI_NUMERATOR_NM: float = 900.0
_WI_DENOMINATOR_NM: float = 970.0


# ---------------------------------------------------------------------------
# Spectral Absorption Index (SAI) — core single-feature computation
# ---------------------------------------------------------------------------


def _compute_sai(
    reflectance: np.ndarray,
    wavelengths: np.ndarray,
    target_wl: float,
    left_shoulder: float,
    right_shoulder: float,
) -> float:
    """Compute the Spectral Absorption Index for a single feature.

    The SAI is the relative depth of a continuum-removed absorption feature:

    .. math::

        \\mathrm{SAI} = \\frac{R_c(\\lambda_t) - R(\\lambda_t)}{R_c(\\lambda_t)}

    where ``R_c`` is the straight-line continuum fit between the left and
    right shoulder reflectances and ``R(λ_t)`` is the measured reflectance
    at the absorption feature minimum.

    The output is clipped to ``[0, 1]``: 0 indicates no absorption (flat
    spectrum, or measured reflectance at or above the continuum), 1 indicates
    total absorption (zero reflectance at the feature minimum).

    Args:
        reflectance: 1-D array of reflectance values for a single pixel.
        wavelengths: 1-D wavelength array (nm), same length as ``reflectance``.
        target_wl: Wavelength of the absorption feature minimum (nm).
        left_shoulder: Approximate wavelength of the left continuum anchor (nm).
            Nearest-neighbour band matching (Tanager 5 nm grid) is applied so
            the value need only be approximate.
        right_shoulder: Approximate wavelength of the right continuum anchor (nm).

    Returns:
        SAI value in ``[0, 1]``. Returns ``NaN`` when the feature cannot be
        evaluated, so masked / invalid pixels do not silently inflate
        downstream "valid pixel" coverage statistics:

        * shoulders do not bracket the target (``left < target < right`` violated),
        * any of the target / shoulder wavelengths fall outside the supplied
          spectrum's range,
        * any of the three reflectance values is NaN,
        * the linearly-interpolated continuum is non-positive,
        * the resulting SAI value itself is non-finite.

        A genuine flat spectrum (``R_target == R_continuum``) still returns
        ``0.0`` — that is a valid measurement of "no absorption", not a mask.

    Raises:
        ValueError: If ``reflectance`` and ``wavelengths`` have different shapes.
    """
    refl = np.asarray(reflectance, dtype=np.float64)
    wl = np.asarray(wavelengths, dtype=np.float64)

    if refl.shape != wl.shape:
        raise ValueError(
            f"reflectance shape {refl.shape} does not match wavelengths shape {wl.shape}"
        )
    if refl.ndim != 1:
        raise ValueError(f"_compute_sai expects 1-D arrays; got {refl.ndim}-D reflectance")
    if refl.size == 0:
        return float("nan")

    if not (left_shoulder < target_wl < right_shoulder):
        return float("nan")

    wl_min = float(wl.min())
    wl_max = float(wl.max())
    if left_shoulder < wl_min or right_shoulder > wl_max:
        return float("nan")

    idx_target = int(np.argmin(np.abs(wl - target_wl)))
    idx_left = int(np.argmin(np.abs(wl - left_shoulder)))
    idx_right = int(np.argmin(np.abs(wl - right_shoulder)))

    r_target = float(refl[idx_target])
    r_left = float(refl[idx_left])
    r_right = float(refl[idx_right])
    if not (np.isfinite(r_target) and np.isfinite(r_left) and np.isfinite(r_right)):
        return float("nan")

    wl_left = float(wl[idx_left])
    wl_right = float(wl[idx_right])
    wl_target = float(wl[idx_target])
    if wl_right == wl_left:
        return float("nan")

    continuum = r_left + (r_right - r_left) * (wl_target - wl_left) / (wl_right - wl_left)
    if continuum <= 0.0:
        return float("nan")

    sai = (continuum - r_target) / continuum
    if not np.isfinite(sai):
        return float("nan")
    return float(np.clip(sai, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Multi-pixel index maps
# ---------------------------------------------------------------------------


def _sai_map(
    reflectance: xr.DataArray,
    target_wl: float,
    left_shoulder: float,
    right_shoulder: float,
) -> xr.DataArray:
    """Vectorized SAI over a (wavelength, y, x) reflectance cube.

    The math is identical to :func:`_compute_sai`; the implementation skips
    the per-pixel Python loop by selecting the three relevant bands once
    and operating on the resulting ``(y, x)`` arrays. All edge-case guards
    that drive ``_compute_sai`` to ``NaN`` produce the same per-pixel value
    here (NaN-filled cells in the output map), so masked / invalid pixels
    do not silently inflate "valid pixel" coverage statistics.
    """
    if not (left_shoulder < target_wl < right_shoulder):
        template = reflectance.isel(wavelength=0, drop=True)
        return xr.full_like(template, np.nan, dtype=np.float64)

    wl_axis = reflectance.coords["wavelength"]
    wl_min = float(wl_axis.min())
    wl_max = float(wl_axis.max())
    if left_shoulder < wl_min or right_shoulder > wl_max:
        template = reflectance.isel(wavelength=0, drop=True)
        return xr.full_like(template, np.nan, dtype=np.float64)

    r_target = reflectance.sel(wavelength=target_wl, method="nearest").astype(np.float64)
    r_left = reflectance.sel(wavelength=left_shoulder, method="nearest").astype(np.float64)
    r_right = reflectance.sel(wavelength=right_shoulder, method="nearest").astype(np.float64)

    wl_target = float(r_target.coords["wavelength"].values)
    wl_left = float(r_left.coords["wavelength"].values)
    wl_right = float(r_right.coords["wavelength"].values)
    if wl_right == wl_left:
        return xr.full_like(r_target, np.nan, dtype=np.float64)

    # Drop the residual wavelength scalar coords so the arithmetic broadcasts
    # over (y, x) without alignment churn.
    r_target = r_target.drop_vars("wavelength", errors="ignore")
    r_left = r_left.drop_vars("wavelength", errors="ignore")
    r_right = r_right.drop_vars("wavelength", errors="ignore")

    continuum = r_left + (r_right - r_left) * (wl_target - wl_left) / (wl_right - wl_left)
    raw_sai = (continuum - r_target) / continuum
    # Replicate _compute_sai's guards element-wise — NaN, not zero, marks
    # masked or non-physical pixels.
    sai = xr.where(continuum <= 0.0, np.nan, raw_sai)
    sai = xr.where(np.isfinite(sai), sai, np.nan)
    return sai.clip(0.0, 1.0).rename(None)


def _continuum_removal_depths(
    scene: Any,
    target_wls: Sequence[float],
    *,
    wavelength_range: Tuple[float, float] = (800.0, 2400.0),
) -> xr.DataArray:
    """Compute upper-hull continuum-removal depths at the requested wavelengths.

    Returns a DataArray with dims ``(cr_target, y, x)`` where each slice along
    ``cr_target`` is the absorption depth (``1 − R/R_continuum``) at the
    corresponding wavelength. The hull is fit on ``wavelength_range`` to keep
    cost bounded; this window contains all of the standard liquid-water
    absorption features (970, 1200, 1660, 2100 nm). Reflectance is clipped to
    ``[0, 1]`` before hull fitting to avoid negative ISOFIT shadow values
    from interfering with the continuum.

    Delegates the upper-hull math to :func:`tanager.spectral.continuum_removal`
    so the algorithm lives in one place.
    """
    from tanager.spectral import continuum_removal as _continuum_removal

    refl = _scene_reflectance(scene).clip(0.0, 1.0).astype(np.float64)
    cr = _continuum_removal(
        xr.Dataset({"reflectance": refl}),
        wavelength_range=wavelength_range,
    )

    # Drop per-band aux coords (`fwhm`, `good_wavelengths`) along with the
    # scalar `wavelength`: each `sel(method="nearest")` slice carries the
    # nearest band's fwhm/good_wavelengths, and those values differ per
    # target so concat would otherwise raise MergeError.
    aux_coords = ("wavelength", "fwhm", "good_wavelengths")
    depth_slices = []
    for tgt in target_wls:
        slice_da = (1.0 - cr.sel(wavelength=tgt, method="nearest")).astype(np.float64)
        depth_slices.append(slice_da.drop_vars(aux_coords, errors="ignore"))
    stacked = xr.concat(depth_slices, dim="cr_target")
    stacked = stacked.assign_coords(cr_target=("cr_target", list(map(float, target_wls))))
    return stacked.transpose("cr_target", ...)


def compute_lfmc_indices(scene: Any) -> xr.Dataset:
    """Compute eight water-sensitive indices for an LFMC scene.

    The returned Dataset contains:

    * ``SAI970``, ``SAI1200``, ``SAI1660`` — Spectral Absorption Indices at
      the three liquid-water absorption features (Quan et al. 2021).
    * ``NDWI_1240``, ``NDWI_1640``, ``NDWI_2130`` — three NDWI variants
      (Gao 1996 et al.) using R860 as the NIR reference: each is computed
      via the epsilon-guarded normalized difference from
      :mod:`tanager.spectral` so near-zero denominators produce NaN rather
      than blow-up values.
    * ``WI`` — Peñuelas et al. (1993) Water Index, ``R900 / R970``.
    * ``CR_depths`` — convex-hull continuum-removal absorption depths at
      970, 1200, 1700, and 2100 nm. Stored as a 3-D variable with dims
      ``(cr_target, y, x)`` and a ``cr_target`` coordinate naming each
      wavelength so callers can ``isel(cr_target=...)`` or
      ``sel(cr_target=970.0)``.

    Reflectance is clamped to ``[0, 1]`` before any index math (see Phase 2
    finding LGT-311: real Tanager ISOFIT surface reflectance has ~13%
    negative values that would corrupt SAI continua and ratio-based indices).

    Args:
        scene: xr.Dataset with a ``reflectance`` variable shaped
            ``(wavelength, y, x)`` and a ``wavelength`` coordinate (nm), or
            an equivalent DataArray.

    Returns:
        xr.Dataset with the 8 index variables described above. The
        ``(y, x)`` dims and coords from the input scene are preserved on
        every variable.

    Raises:
        ValueError: If ``scene`` is a Dataset without a ``reflectance``
            variable, or if the wavelength axis does not span the bands
            required by these indices (roughly 800–2400 nm).
    """
    refl = _scene_reflectance(scene)

    if "wavelength" not in refl.coords:
        raise ValueError("scene reflectance must have a 'wavelength' coordinate")

    # Real Tanager DataArrays carry per-band aux coords (`fwhm`,
    # `good_wavelengths`) along the wavelength dim. Each `sel(method="nearest")`
    # below grabs the nearest band's aux value — and those differ per pick, so
    # the resulting variables conflict on `xr.Dataset(...)` construction
    # (LGT-333). Strip them once up front; they're not meaningful on per-band
    # selections or on derived index outputs.
    refl = refl.drop_vars(("fwhm", "good_wavelengths"), errors="ignore")

    wl = refl.coords["wavelength"]
    wl_min = float(wl.min())
    wl_max = float(wl.max())
    required_min, required_max = 860.0, 2130.0
    if wl_min > required_min or wl_max < required_max:
        raise ValueError(
            f"scene wavelength axis [{wl_min:.0f}, {wl_max:.0f}] nm does not span "
            f"the bands required for LFMC indices (need at least "
            f"[{required_min:.0f}, {required_max:.0f}] nm)"
        )

    # Phase 2 mitigation: clamp before index math.
    refl = refl.clip(0.0, 1.0).astype(np.float64)

    # Heavy import here so module import stays cheap.
    from tanager.spectral import _normalized_difference

    out_vars: dict[str, xr.DataArray] = {}

    # SAI indices — shoulder windows chosen to bracket each absorption feature
    # with continuum anchors that fall outside the feature wing.
    sai_targets = (
        ("SAI970", 970.0, 850.0, 1100.0),
        ("SAI1200", 1200.0, 1080.0, 1320.0),
        ("SAI1660", 1660.0, 1530.0, 1790.0),
    )
    for name, target, left, right in sai_targets:
        out_vars[name] = _sai_map(refl, target, left, right)

    # NDWI variants — R860 as the NIR baseline (per Gao 1996 NDWI convention).
    nir = refl.sel(wavelength=860.0, method="nearest").drop_vars("wavelength", errors="ignore")
    for name, _, target_wl in _NDWI_PAIRS:
        b_target = refl.sel(wavelength=target_wl, method="nearest").drop_vars(
            "wavelength", errors="ignore"
        )
        out_vars[name] = _normalized_difference(nir, b_target).astype(np.float64)

    # Water Index — R900 / R970, with a near-zero-denominator guard.
    r900 = refl.sel(wavelength=_WI_NUMERATOR_NM, method="nearest").drop_vars(
        "wavelength", errors="ignore"
    )
    r970 = refl.sel(wavelength=_WI_DENOMINATOR_NM, method="nearest").drop_vars(
        "wavelength", errors="ignore"
    )
    wi = xr.where(np.abs(r970) < 1e-3, np.nan, r900 / r970).astype(np.float64)
    out_vars["WI"] = wi

    # Continuum-removal depths at the four standard water-absorption features.
    out_vars["CR_depths"] = _continuum_removal_depths(
        scene if isinstance(scene, xr.Dataset) else refl,
        target_wls=(970.0, 1200.0, 1700.0, 2100.0),
    )

    indices = xr.Dataset(out_vars)
    logger.info(
        "compute_lfmc_indices: emitted %d variables (%s)",
        len(indices.data_vars),
        ", ".join(indices.data_vars),
    )
    return indices


# ---------------------------------------------------------------------------
# Globe-LFMC 2.0 ground truth loader
# ---------------------------------------------------------------------------


# Tolerant column-name lookup for Globe-LFMC 2.0 distributions. The DOI-hosted
# CSV uses ``Latitude``/``Longitude``/``Date``/``LFMC_value`` etc.; some
# downstream redistributions strip casing or rename. We normalize once at load.
_GLOBE_LFMC_COLUMN_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "lon", "long", "lng"),
    "date": ("date", "observation_date", "sampling_date", "obs_date"),
    "lfmc_percent": (
        "lfmc_percent",
        "lfmc",
        "lfmc_value",
        "lfmc_%",
        "live_fuel_moisture",
    ),
    "species": ("species", "species_name", "scientific_name"),
    "site_name": ("site_name", "site", "sitename", "location_name", "location"),
    "vegetation_type": (
        "vegetation_type",
        "veg_type",
        "vegetation",
        "land_cover",
        "vegetation_class",
    ),
}


def _normalize_globe_lfmc_columns(columns: Iterable[str]) -> dict[str, str]:
    """Map an input CSV's columns onto our canonical schema using lowercase aliases."""
    cols_l = {c.strip().lower(): c for c in columns}
    rename: dict[str, str] = {}
    for canonical, aliases in _GLOBE_LFMC_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in cols_l and cols_l[alias] != canonical:
                rename[cols_l[alias]] = canonical
                break
    return rename


def load_globe_lfmc(
    data_path: Any,
    *,
    region_bbox: Optional[Tuple[float, float, float, float]] = None,
    vegetation_types: Optional[Sequence[str]] = None,
    tanager_scene_dates: Optional[Sequence[Any]] = None,
    colocation_window_days: int = 30,
) -> Any:
    """Load Globe-LFMC 2.0 observations as a filtered GeoDataFrame.

    Globe-LFMC 2.0 (Yebra et al. 2024, DOI 10.1038/s41597-024-03159-6) is the
    canonical global database of in-situ live fuel moisture observations. This
    function reads the published CSV (or any file ``pandas.read_csv`` can
    consume), normalizes the column names against
    ``_GLOBE_LFMC_COLUMN_ALIASES``, applies optional spatial / vegetation
    filters, and returns a GeoPandas GeoDataFrame ready for
    :func:`train_lfmc_plsr` ground-truth assembly.

    Args:
        data_path: Path to the Globe-LFMC CSV (or any file ``pandas.read_csv``
            can read). Must exist on disk; this loader does not download.
        region_bbox: Optional ``(west, south, east, north)`` bounding box in
            WGS84 degrees. Observations outside the box are dropped.
        vegetation_types: Optional list of vegetation-type strings used as
            case-insensitive substring filters against the
            ``vegetation_type`` column (or ``species`` if vegetation_type is
            not present). Matches any pattern in the list.
        tanager_scene_dates: Optional iterable of Tanager scene capture dates.
            When supplied, each row gets a ``tanager_colocated`` boolean set
            True iff its observation date falls within
            ``colocation_window_days`` of *any* scene date.
        colocation_window_days: Half-window for the colocation flag. Default 30.

    Returns:
        ``geopandas.GeoDataFrame`` with EPSG:4326 geometry and at least the
        canonical columns ``longitude``, ``latitude``, ``date``,
        ``lfmc_percent``, plus ``species``, ``site_name``,
        ``vegetation_type``, and ``tanager_colocated`` when available in the
        source.

    Raises:
        FileNotFoundError: If ``data_path`` does not point to an existing file.
        ValueError: If the source CSV is missing any of the required columns
            after alias normalization (``latitude``, ``longitude``, ``date``,
            ``lfmc_percent``).
    """
    from pathlib import Path

    import geopandas as gpd
    import pandas as pd

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Globe-LFMC data not found at {path}. Download from "
            "https://doi.org/10.6084/m9.figshare.24745469 or pass an "
            "alternative local CSV path."
        )

    df = pd.read_csv(path)
    rename = _normalize_globe_lfmc_columns(df.columns)
    if rename:
        df = df.rename(columns=rename)

    required = {"latitude", "longitude", "date", "lfmc_percent"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Globe-LFMC CSV missing required columns after alias normalization: "
            f"{sorted(missing)}. Found columns: {sorted(df.columns.tolist())}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "latitude", "longitude", "lfmc_percent"])
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    df["lfmc_percent"] = df["lfmc_percent"].astype(float)

    if region_bbox is not None:
        west, south, east, north = region_bbox
        in_bbox = (
            (df["longitude"] >= west)
            & (df["longitude"] <= east)
            & (df["latitude"] >= south)
            & (df["latitude"] <= north)
        )
        df = df[in_bbox]

    if vegetation_types is not None and len(vegetation_types) > 0:
        veg_col = (
            "vegetation_type"
            if "vegetation_type" in df.columns
            else ("species" if "species" in df.columns else None)
        )
        if veg_col is None:
            logger.warning(
                "load_globe_lfmc: vegetation_types filter requested but neither "
                "'vegetation_type' nor 'species' column present; skipping filter."
            )
        else:
            patterns = [str(v).strip().lower() for v in vegetation_types]
            haystack = df[veg_col].astype(str).str.lower()
            mask = haystack.apply(lambda s: any(p in s for p in patterns))
            df = df[mask]

    if tanager_scene_dates is not None and len(list(tanager_scene_dates)) > 0:
        scene_dates = pd.to_datetime(list(tanager_scene_dates))
        window = pd.Timedelta(days=int(colocation_window_days))

        def _is_colocated(d: Any) -> bool:
            return bool(((scene_dates - d).to_series().abs() <= window).any())

        df["tanager_colocated"] = df["date"].apply(_is_colocated)
    else:
        df["tanager_colocated"] = False

    geom = gpd.points_from_xy(df["longitude"], df["latitude"])
    gdf = gpd.GeoDataFrame(df.reset_index(drop=True), geometry=geom, crs="EPSG:4326")

    logger.info(
        "load_globe_lfmc: %d observations after filtering (bbox=%s, veg_types=%s)",
        len(gdf),
        region_bbox,
        vegetation_types,
    )
    return gdf


# ---------------------------------------------------------------------------
# Tier 2 — PLSR
# ---------------------------------------------------------------------------


def _compute_vip(model: Any) -> np.ndarray:
    """Variable Importance in Projection (VIP) scores for a fitted PLSRegression.

    Standard PLS-VIP formula::

        VIP_j = sqrt(p * sum_h(SS_h * (w_{jh} / ||w_h||)^2) / sum_h(SS_h))

    where ``p`` is the number of predictor variables, ``h`` indexes
    components, ``SS_h`` is the explained sum-of-squares of component ``h``
    (``t_h^T t_h * q_h^T q_h``), and ``w_{jh}`` is the weight of variable
    ``j`` in component ``h``.
    """
    T = np.asarray(model.x_scores_, dtype=np.float64)
    W = np.asarray(model.x_weights_, dtype=np.float64)
    Q = np.asarray(model.y_loadings_, dtype=np.float64)

    p, h = W.shape
    ss = np.zeros(h, dtype=np.float64)
    for i in range(h):
        ss[i] = (T[:, i] @ T[:, i]) * (Q[:, i] @ Q[:, i])
    total_ss = float(ss.sum())
    if total_ss <= 0.0:
        return np.zeros(p, dtype=np.float64)

    vip = np.zeros(p, dtype=np.float64)
    for j in range(p):
        weighted = 0.0
        for i in range(h):
            w_norm_sq = float(W[:, i] @ W[:, i])
            if w_norm_sq > 0.0:
                weighted += ss[i] * (W[j, i] ** 2) / w_norm_sq
        vip[j] = np.sqrt(p * weighted / total_ss)
    return vip


def train_lfmc_plsr(
    spectra: np.ndarray,
    lfmc_values: np.ndarray,
    n_components: int = 10,
    *,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Fit a Partial Least Squares regression from reflectance to LFMC.

    Performs k-fold cross-validated component selection over
    ``[1, n_components]`` (clipped against sample / feature counts) and
    returns the model fit at the best component count along with R² / RMSE
    on the held-out folds and per-band VIP scores. VIP highlights which
    wavelengths drove the regression — a sanity check that the model is
    keying on water-absorption bands rather than spurious correlations.

    Args:
        spectra: 2-D array of shape ``(n_samples, n_bands)``. Bad bands
            should be removed by the caller before training (Tanager
            ortho_sr has ~330 good bands after the standard bad-band mask).
        lfmc_values: 1-D array of LFMC values in percent (typical 30–200 %).
        n_components: Upper bound on PLS components to search. Default 10.
            The actual maximum is clipped to ``min(n_samples-1, n_features)``.
        cv_folds: K for K-fold cross-validation. Default 5; clipped to the
            sample count.

    Returns:
        Dict with keys:
            ``model``: fitted ``sklearn.cross_decomposition.PLSRegression``
                refit on all valid samples at the best component count.
            ``r2``: cross-validated R² at the best component count (float).
            ``rmse``: cross-validated RMSE at the best component count
                (float, in LFMC units = percent).
            ``n_components_optimal``: selected number of components (int).
            ``vip_scores``: 1-D array of length ``n_features`` carrying the
                VIP score for each wavelength.

    Raises:
        ValueError: If shapes mismatch, fewer than two valid samples remain
            after NaN filtering, or ``n_components`` is non-positive.
    """
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1; got {n_components}")

    X = np.asarray(spectra, dtype=np.float64)
    y = np.asarray(lfmc_values, dtype=np.float64).ravel()
    if X.ndim != 2:
        raise ValueError(f"spectra must be 2-D (n_samples, n_bands); got {X.ndim}-D")
    if X.shape[0] != y.size:
        raise ValueError(f"spectra has {X.shape[0]} samples but lfmc_values has {y.size}")

    finite_rows = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X = X[finite_rows]
    y = y[finite_rows]
    n_samples, n_features = X.shape

    if n_samples < 2:
        raise ValueError(
            f"need at least 2 valid samples for PLSR; got {n_samples} after NaN filter"
        )

    max_components = min(n_components, n_samples - 1, n_features)
    if max_components < 1:
        raise ValueError(f"max_components clipped to {max_components}; need at least one component")

    cv_actual = max(2, min(cv_folds, n_samples))

    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import cross_val_score

    best_rmse = float("inf")
    best_n = 1
    for n in range(1, max_components + 1):
        candidate = PLSRegression(n_components=n)
        try:
            neg_mse = cross_val_score(
                candidate, X, y, cv=cv_actual, scoring="neg_mean_squared_error"
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug(
                "train_lfmc_plsr: PLS with n_components=%d failed CV (%s); skipping",
                n,
                exc,
            )
            continue
        candidate_rmse = float(np.sqrt(-np.mean(neg_mse)))
        if candidate_rmse < best_rmse:
            best_rmse = candidate_rmse
            best_n = n

    model = PLSRegression(n_components=best_n)
    r2_cv = float(np.mean(cross_val_score(model, X, y, cv=cv_actual, scoring="r2")))
    rmse_cv = float(
        np.sqrt(
            -np.mean(cross_val_score(model, X, y, cv=cv_actual, scoring="neg_mean_squared_error"))
        )
    )
    model.fit(X, y)
    vip = _compute_vip(model)

    logger.info(
        "train_lfmc_plsr: n_samples=%d n_features=%d n_components_optimal=%d cv_r2=%.4f cv_rmse=%.4f",
        n_samples,
        n_features,
        best_n,
        r2_cv,
        rmse_cv,
    )

    return {
        "model": model,
        "r2": r2_cv,
        "rmse": rmse_cv,
        "n_components_optimal": best_n,
        "vip_scores": vip,
    }


# ---------------------------------------------------------------------------
# Per-pixel LFMC prediction
# ---------------------------------------------------------------------------


def predict_lfmc(
    scene: Any,
    model: Any,
    method: str = "plsr",
) -> dict[str, xr.DataArray]:
    """Apply a trained LFMC model to a scene, with uncertainty + low-LFMC flag.

    Flattens the scene reflectance into ``(n_pixels, n_bands)``, runs the
    model's ``predict``, clips the result to the physical ``[0, 300]`` %
    range, and reshapes back to ``(y, x)``. Two companion DataArrays are
    returned alongside the LFMC map:

    * **uncertainty_map** — per-pixel uncertainty estimate. When ``model``
      is the dict returned by :func:`train_lfmc_plsr`, the cross-validated
      RMSE (in % LFMC) is used as a uniform global uncertainty floor; this
      is a coarse approximation, intended as a Tier-1 placeholder until the
      Phase 4 bootstrap-based prediction interval becomes available. Pixels
      with NaN input propagate to NaN.
    * **low_lfmc_flag** — boolean DataArray, ``True`` where predicted LFMC
      is below 60 % (the nonlinear / fire-prone regime per Roberts et al.
      2006). NaN-input pixels are ``False`` (unknown rather than low).

    Args:
        scene: ``xr.Dataset`` with a ``reflectance`` variable shaped
            ``(wavelength, y, x)``, or a DataArray of the same shape. The
            band layout MUST match the layout used to train ``model``
            (caller is responsible for selecting bands consistently —
            typically by passing the same `bands` array through
            :func:`tanager.spectral.select_bands` before training and
            prediction).
        model: Either the dict returned by :func:`train_lfmc_plsr`
            (preferred — the ``rmse`` key drives the uncertainty floor) or
            a bare fitted PLSRegression. With a bare estimator the
            uncertainty is reported as 0 (caller should construct the
            full dict to get a meaningful uncertainty).
        method: Currently ``"plsr"`` only. Reserved for future
            ``"indices"`` mode driven by :func:`compute_lfmc_indices`.

    Returns:
        Dict with keys:
            ``lfmc_map``: ``(y, x)`` DataArray of predicted LFMC %, NaN on
                input-NaN pixels, clipped to ``[0, 300]``.
            ``uncertainty_map``: ``(y, x)`` DataArray of LFMC uncertainty %,
                NaN on input-NaN pixels.
            ``low_lfmc_flag``: ``(y, x)`` boolean DataArray, True where
                LFMC < 60 % and the input was finite.

    Raises:
        ValueError: If ``method`` is unsupported, the model dict is missing
            the ``"model"`` key, or the scene's band count does not match
            the model's expected feature count.
    """
    if method != "plsr":
        raise ValueError(f"unsupported method {method!r}; only 'plsr' is implemented")

    if isinstance(model, Mapping):
        estimator = model.get("model")
        if estimator is None:
            raise ValueError("model dict must contain a 'model' key with a fitted estimator")
        cv_rmse = float(model.get("rmse", 0.0))
    else:
        estimator = model
        cv_rmse = 0.0

    refl = _scene_reflectance(scene)
    if "wavelength" not in refl.dims:
        raise ValueError("scene reflectance must have a 'wavelength' dim leading the array")
    refl = refl.transpose("wavelength", ...)
    nb = refl.sizes["wavelength"]

    # Spatial dims after wavelength.
    spatial_dims = tuple(d for d in refl.dims if d != "wavelength")
    if not spatial_dims:
        raise ValueError("scene reflectance must have at least one spatial dim")
    spatial_shape = tuple(refl.sizes[d] for d in spatial_dims)
    n_pixels = int(np.prod(spatial_shape))

    expected_features = getattr(estimator, "n_features_in_", None)
    if expected_features is not None and nb != int(expected_features):
        raise ValueError(
            f"scene has {nb} bands but model expects {int(expected_features)} features; "
            "select bands consistently between training and prediction"
        )

    # (n_bands, n_pixels) → (n_pixels, n_bands) for sklearn.
    X = np.asarray(refl.values, dtype=np.float64).reshape(nb, n_pixels).T

    nan_mask = ~np.all(np.isfinite(X), axis=1)
    X_safe = np.where(np.isnan(X), 0.0, X)

    pred = np.asarray(estimator.predict(X_safe), dtype=np.float64).ravel()
    pred = np.clip(pred, _LFMC_MIN_PERCENT, _LFMC_MAX_PERCENT)
    pred[nan_mask] = np.nan

    uncertainty = np.full(pred.shape, cv_rmse, dtype=np.float64)
    uncertainty[nan_mask] = np.nan

    low_flag = (pred < _LFMC_LOW_THRESHOLD_PERCENT) & ~nan_mask

    template_coords = {d: refl.coords[d] for d in spatial_dims if d in refl.coords}

    lfmc_map = xr.DataArray(
        pred.reshape(spatial_shape),
        dims=spatial_dims,
        coords=template_coords,
        name="lfmc_percent",
        attrs={
            "long_name": "live_fuel_moisture_content",
            "units": "percent",
            "valid_range": [_LFMC_MIN_PERCENT, _LFMC_MAX_PERCENT],
            "method": method,
        },
    )
    uncertainty_map = xr.DataArray(
        uncertainty.reshape(spatial_shape),
        dims=spatial_dims,
        coords=template_coords,
        name="lfmc_uncertainty",
        attrs={
            "long_name": "lfmc_prediction_uncertainty",
            "units": "percent",
            "source": "cv_rmse" if cv_rmse > 0 else "none",
        },
    )
    low_lfmc_flag = xr.DataArray(
        low_flag.reshape(spatial_shape),
        dims=spatial_dims,
        coords=template_coords,
        name="low_lfmc",
        attrs={
            "threshold_percent": _LFMC_LOW_THRESHOLD_PERCENT,
            "regime_note": "below threshold = nonlinear/fire-prone (Roberts et al. 2006)",
        },
    )

    logger.info(
        "predict_lfmc: %d pixels (%d NaN), LFMC range [%.1f, %.1f] %%, low_lfmc=%d (%.1f%%)",
        int((~nan_mask).sum()),
        int(nan_mask.sum()),
        float(np.nanmin(pred)) if (~nan_mask).any() else float("nan"),
        float(np.nanmax(pred)) if (~nan_mask).any() else float("nan"),
        int(low_flag.sum()),
        100.0 * float(low_flag.sum()) / max(1, n_pixels),
    )

    return {
        "lfmc_map": lfmc_map,
        "uncertainty_map": uncertainty_map,
        "low_lfmc_flag": low_lfmc_flag,
    }
