"""Validation and accuracy assessment for Tanager-1 analysis products.

This module compares Tanager-derived MESMA fractions and severity maps against
external reference datasets (AVIRIS-3 fractions, USGS BARC severity, EMIT/PRISMA
sensor cross-comparisons) and computes accuracy metrics for both continuous and
classified products.

Public API (lazy-imported via :mod:`tanager`):

* :func:`load_aviris3_reference` — load AVIRIS-3 fraction product, aggregate to
  Tanager 30 m grid.
* :func:`load_aviris3_reflectance` — load AVIRIS-3 L2A surface-reflectance
  NetCDF (284-band cube) as an xr.Dataset compatible with :func:`run_mesma`.
* :func:`cross_validate_aviris3` — run MESMA on AVIRIS-3 reflectance and compare
  the resulting char fraction against Tanager MESMA char (R²/RMSE/bias).
* :func:`load_barc_reference` — load USGS BARC classified severity GeoTIFF and
  align to a Tanager scene grid. Also accepts BAER Soil Burn Severity (SBS)
  rasters via the ``code_map`` kwarg (use :data:`SBS_CODE_MAP`).
* :data:`SBS_CODE_MAP` — class-code mapping for BAER SBS 4-class rasters.
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
    Ward-Baranyay, L. K., Coleman, R. C. (2026). AVIRIS-3 char/ash mapping of
        the 2025 Los Angeles fires. Geophysical Research Letters,
        doi:10.1029/2025GL118756.
"""

from __future__ import annotations

import logging
import os
from os import PathLike
from pathlib import Path
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

# BAER Soil Burn Severity (SBS) products use the same 4-class BARC scheme after
# field correction.  The raster codes are identical to BARC (1=Low through
# 4=High); 0 or NoData = outside mapped area.  Pass this as ``code_map`` to
# :func:`load_barc_reference` when loading SBS rasters; pixels outside the
# mapped perimeter (code 0) are treated as nodata (-1) so they don't pollute
# accuracy metrics.
#
# WARNING: BAER SBS encodings vary by product — some rasters use additional
# codes (e.g. 15 for unburned-inside-perimeter).  Always verify the code
# distribution of each raster (``np.unique(data, return_counts=True)``) and
# supply a per-raster ``code_map`` if codes outside 0..4 are present.
SBS_CODE_MAP: Mapping[int, int] = {
    0: -1,  # Outside mapped perimeter → nodata
    1: 1,   # Low (field-corrected)
    2: 2,   # Moderate-Low (field-corrected)
    3: 3,   # Moderate-High (field-corrected)
    4: 4,   # High (field-corrected)
}

# CAL FIRE DINS (Damage INSpection) structure-damage categories, ordered by
# increasing damage. Keys match the raw ``DAMAGE`` field values in DINS
# GeoJSON exports; values are the ordinal scale used for rank correlation
# against continuous severity products (dNBR, char fraction).
DINS_DAMAGE_ORDINAL: Mapping[str, int] = {
    "No Damage": 0,
    "Affected (1-9%)": 1,
    "Minor (10-25%)": 2,
    "Major (26-50%)": 3,
    "Destroyed (>50%)": 4,
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


def _load_aviris3_raster(path: Path) -> xr.Dataset:
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


def _load_aviris3_netcdf(path: Path) -> xr.Dataset:
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


# ---------------------------------------------------------------------------
# AVIRIS-3 L2A reflectance loader (284-band cube for cross-sensor MESMA)
# ---------------------------------------------------------------------------

# AVIRIS-3 L2A OE products use these NetCDF variable names. The reflectance
# cube is stored as ``Rfl`` with dims (bands, y, x) or (number_of_bands, y, x).
# Wavelengths are stored as a 1-D coordinate or in ``wavelength``/``wl``.
_AVIRIS3_RFL_CANDIDATES = ("Rfl", "rfl", "reflectance", "Reflectance", "surface_reflectance")
_AVIRIS3_WL_CANDIDATES = ("wavelength", "wl", "Wavelength", "WL", "wavelengths")
_AVIRIS3_UNC_CANDIDATES = ("Rfl_unc", "rfl_unc", "uncertainty", "Uncertainty")


def load_aviris3_reflectance(
    filepath: FilePath,
    *,
    wavelength_range: Optional[tuple[float, float]] = None,
) -> xr.Dataset:
    """Load an AVIRIS-3 L2A surface-reflectance NetCDF as a scene Dataset.

    AVIRIS-3 L2A OE products (DOI 10.3334/ORNLDAAC/2357) ship as
    orthocorrected NetCDF files with ~284 spectral bands at ~1.8-2.8 m GSD.
    This loader reads the reflectance cube and returns an xr.Dataset with dims
    ``(wavelength, y, x)`` — the same schema as :func:`tanager.io.load_scene`
    — so the cube can be fed directly to :func:`tanager.unmixing.run_mesma`.

    Args:
        filepath: Path to an AVIRIS-3 ``*_RFL_ORT.nc`` file.
        wavelength_range: Optional ``(min_wl, max_wl)`` in nanometres. When
            supplied, only bands within the range are returned.

    Returns:
        xr.Dataset with:

        * Data variable ``reflectance`` with dims ``(wavelength, y, x)``,
          dtype float32, reflectance in [0, 1].
        * Coords ``wavelength`` (nm), ``y``, ``x`` in the file's native CRS.
        * Attrs ``crs``, ``source = "aviris3_l2a"``, ``spatial_resolution_m``.

    Raises:
        FileNotFoundError: If ``filepath`` does not exist.
        ValueError: If the NetCDF lacks a recognisable reflectance variable
            or wavelength coordinate.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"AVIRIS-3 reflectance file not found: {filepath}")

    ds = xr.open_dataset(path)

    rfl_var = _find_variable(ds, _AVIRIS3_RFL_CANDIDATES)
    if rfl_var is None:
        # AVIRIS-3 L2A OE products (DOI 10.3334/ORNLDAAC/2357) store the
        # reflectance cube inside a NetCDF group (e.g. "/reflectance"), while
        # the spatial coords (easting/northing) and the CRS grid-mapping
        # variable live in the root group. xr.open_dataset reads only the root
        # — which surfaces just the grid-mapping var — so fall back to a
        # group-aware open that merges the group's cube with the root coords.
        grouped = _open_aviris3_grouped(path, ds)
        if grouped is not None:
            ds.close()
            ds = grouped
            rfl_var = _find_variable(ds, _AVIRIS3_RFL_CANDIDATES)
    if rfl_var is None:
        raise ValueError(
            f"No reflectance variable found in {filepath}. "
            f"Expected one of {_AVIRIS3_RFL_CANDIDATES}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    rfl = ds[rfl_var]

    wavelengths = _extract_aviris3_wavelengths(ds, rfl)
    if wavelengths is None:
        raise ValueError(
            f"No wavelength coordinate found in {filepath}. "
            f"Searched variable coords, dataset coords, and attrs."
        )

    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    # AVIRIS-3 wavelengths may be in micrometres; auto-detect.
    if wavelengths.max() < 50.0:
        wavelengths = wavelengths * 1000.0

    rfl_arr = np.asarray(rfl.values, dtype=np.float32)

    # Determine dimension order. AVIRIS-3 NetCDF may use (bands, y, x) or
    # (y, x, bands). Normalise to (wavelength, y, x).
    if rfl_arr.ndim != 3:
        raise ValueError(
            f"Reflectance variable {rfl_var!r} has {rfl_arr.ndim} dims, expected 3"
        )

    band_dim = _identify_band_dim(rfl, wavelengths)
    if band_dim != 0:
        rfl_arr = np.moveaxis(rfl_arr, band_dim, 0)
        spatial_dims = [d for i, d in enumerate(rfl.dims) if i != band_dim]
    else:
        spatial_dims = list(rfl.dims[1:])

    n_bands = rfl_arr.shape[0]
    if n_bands != len(wavelengths):
        raise ValueError(
            f"Band count mismatch: reflectance has {n_bands} bands but "
            f"wavelength array has {len(wavelengths)} entries"
        )

    # Build y/x coordinates from the spatial dims.
    y_coords, x_coords = _extract_spatial_coords(ds, rfl, spatial_dims)

    # Replace fill values with NaN.
    fill_candidates = [-9999.0, -9999, -999.0]
    for fv in fill_candidates:
        rfl_arr = np.where(rfl_arr == fv, np.nan, rfl_arr)
    # Reflectance should be in [0, 1]; AVIRIS-3 values > 1 are artefacts.
    rfl_arr = np.where(np.isfinite(rfl_arr), rfl_arr, np.nan)

    # Subset wavelengths if requested.
    if wavelength_range is not None:
        min_wl, max_wl = wavelength_range
        mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
        if not mask.any():
            raise ValueError(
                f"wavelength_range ({min_wl}, {max_wl}) nm selects no bands "
                f"from AVIRIS-3 wavelengths [{wavelengths.min():.1f}, "
                f"{wavelengths.max():.1f}] nm"
            )
        rfl_arr = rfl_arr[mask]
        wavelengths = wavelengths[mask]

    crs = _extract_crs(ds)
    res = _estimate_resolution(y_coords, x_coords)

    out_ds = xr.Dataset(
        {
            "reflectance": (("wavelength", "y", "x"), rfl_arr),
        },
        coords={
            "wavelength": wavelengths.astype(np.float32),
            "y": y_coords,
            "x": x_coords,
        },
        attrs={
            "source": "aviris3_l2a",
            "data_var": "reflectance",
        },
    )
    if crs is not None:
        out_ds.attrs["crs"] = crs
    if res is not None:
        out_ds.attrs["spatial_resolution_m"] = res

    ds.close()
    return out_ds


def _find_variable(
    ds: xr.Dataset, candidates: tuple[str, ...],
) -> Optional[str]:
    """Return the first matching variable name from candidates."""
    for name in candidates:
        if name in ds.data_vars:
            return name
    return None


def _open_aviris3_grouped(
    path: Path, root: xr.Dataset,
) -> Optional[xr.Dataset]:
    """Open a grouped AVIRIS-3 L2A NetCDF as a single flat Dataset.

    AVIRIS-3 L2A OE products place the reflectance cube in a NetCDF group and
    keep the spatial coordinates (``easting``/``northing``) and CRS
    grid-mapping variable in the root group. This locates the group holding a
    recognised reflectance variable, then merges the root-level spatial coords
    and CRS onto it so the rest of the loader can treat it as a flat Dataset.

    Args:
        path: Path to the AVIRIS-3 ``*_RFL_ORT.nc`` file.
        root: The already-opened root-group Dataset.

    Returns:
        A flattened Dataset containing the reflectance cube with spatial coords
        and a ``crs`` attr, or ``None`` if no group holds a reflectance cube.
    """
    try:
        import netCDF4  # noqa: PLC0415 — optional, only needed for group discovery
    except ImportError:
        return None

    with netCDF4.Dataset(path) as nc:
        group_names = list(nc.groups.keys())
    if not group_names:
        return None

    # Prefer a group named like a reflectance candidate (e.g. "reflectance").
    ordered = [g for g in group_names if g in _AVIRIS3_RFL_CANDIDATES]
    ordered += [g for g in group_names if g not in ordered]

    for gname in ordered:
        try:
            grp = xr.open_dataset(path, group=gname)
        except (OSError, ValueError):
            continue
        if _find_variable(grp, _AVIRIS3_RFL_CANDIDATES) is None:
            grp.close()
            continue

        # Merge root-level spatial coords onto the group by shared dim name.
        for cname in ("easting", "northing", "x", "y", "lat", "lon",
                      "latitude", "longitude"):
            if cname in root.variables and cname in grp.dims:
                grp = grp.assign_coords({cname: root[cname].values})

        _promote_aviris3_crs(grp, root)
        return grp

    return None


def _promote_aviris3_crs(grp: xr.Dataset, root: xr.Dataset) -> None:
    """Copy the CRS from the root grid-mapping variable to ``grp.attrs['crs']``.

    The reflectance variable's ``grid_mapping`` attr names a scalar variable in
    the root group (e.g. ``transverse_mercator``) whose attrs carry the CRS WKT
    (``spatial_ref`` / ``crs_wkt``). ``_extract_crs`` only inspects dataset-level
    attrs, so promote the WKT string up so it is discoverable.
    """
    if "crs" in grp.attrs:
        return

    gm_name: Optional[str] = None
    rfl_var = _find_variable(grp, _AVIRIS3_RFL_CANDIDATES)
    if rfl_var is not None:
        gm_name = grp[rfl_var].attrs.get("grid_mapping")
    if gm_name is None or gm_name not in root.variables:
        for cand in ("transverse_mercator", "crs", "spatial_ref"):
            if cand in root.variables:
                gm_name = cand
                break
    if gm_name is None or gm_name not in root.variables:
        return

    gm_attrs = root[gm_name].attrs
    for attr in ("spatial_ref", "crs_wkt"):
        if attr in gm_attrs:
            grp.attrs["crs"] = str(gm_attrs[attr])
            return


def _extract_aviris3_wavelengths(
    ds: xr.Dataset, rfl: xr.DataArray,
) -> Optional[np.ndarray]:
    """Extract wavelength array from AVIRIS-3 NetCDF (multiple conventions)."""
    # Check variable coords first.
    for name in _AVIRIS3_WL_CANDIDATES:
        if name in rfl.coords:
            return np.asarray(rfl.coords[name].values, dtype=np.float64)
        if name in ds.coords:
            return np.asarray(ds.coords[name].values, dtype=np.float64)
        if name in ds.data_vars:
            return np.asarray(ds[name].values, dtype=np.float64).ravel()

    # Check reflectance variable attrs.
    for attr_name in ("wavelengths", "wavelength", "wl", "center_wavelengths"):
        if attr_name in rfl.attrs:
            return np.asarray(rfl.attrs[attr_name], dtype=np.float64)

    # Check global attrs.
    for attr_name in ("wavelengths", "wavelength", "center_wavelengths"):
        if attr_name in ds.attrs:
            return np.asarray(ds.attrs[attr_name], dtype=np.float64)

    return None


def _identify_band_dim(rfl: xr.DataArray, wavelengths: np.ndarray) -> int:
    """Identify which dimension of rfl is the band/spectral dimension."""
    n_wl = len(wavelengths)
    for i, dim_size in enumerate(rfl.shape):
        if dim_size == n_wl:
            return i
    # Fallback: assume first dim is bands (most common convention).
    return 0


def _extract_spatial_coords(
    ds: xr.Dataset,
    rfl: xr.DataArray,
    spatial_dims: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract y and x coordinate arrays from AVIRIS-3 spatial dims."""
    y_coords = None
    x_coords = None

    # Try named coords first.
    for name in ("y", "northing", "lat", "latitude"):
        if name in ds.coords:
            y_coords = np.asarray(ds.coords[name].values, dtype=np.float64)
            break
    for name in ("x", "easting", "lon", "longitude"):
        if name in ds.coords:
            x_coords = np.asarray(ds.coords[name].values, dtype=np.float64)
            break

    # Fall back to dim coords.
    if y_coords is None and len(spatial_dims) >= 1 and spatial_dims[0] in ds.coords:
        y_coords = np.asarray(ds.coords[spatial_dims[0]].values, dtype=np.float64)
    if x_coords is None and len(spatial_dims) >= 2 and spatial_dims[1] in ds.coords:
        x_coords = np.asarray(ds.coords[spatial_dims[1]].values, dtype=np.float64)

    # Ultimate fallback: pixel indices.
    if y_coords is None:
        n_y = rfl.shape[1] if len(rfl.shape) > 1 else rfl.shape[0]
        y_coords = np.arange(n_y, dtype=np.float64)
    if x_coords is None:
        n_x = rfl.shape[2] if len(rfl.shape) > 2 else rfl.shape[-1]
        x_coords = np.arange(n_x, dtype=np.float64)

    return y_coords.ravel(), x_coords.ravel()


def _extract_crs(ds: xr.Dataset) -> Optional[str]:
    """Extract CRS string from AVIRIS-3 dataset metadata."""
    for attr in ("crs", "spatial_ref", "crs_wkt", "proj4"):
        if attr in ds.attrs:
            return str(ds.attrs[attr])
    if "spatial_ref" in ds.coords:
        return str(ds.coords["spatial_ref"].item())
    # AVIRIS-3 L2A ORT products are projected in UTM; look for EPSG in attrs.
    for attr in ("epsg", "EPSG", "epsg_code"):
        if attr in ds.attrs:
            return f"EPSG:{ds.attrs[attr]}"
    return None


def _estimate_resolution(
    y_coords: np.ndarray, x_coords: np.ndarray,
) -> Optional[float]:
    """Estimate ground sample distance from coordinate arrays."""
    if len(x_coords) < 2 or len(y_coords) < 2:
        return None
    dx = float(abs(x_coords[1] - x_coords[0]))
    dy = float(abs(y_coords[1] - y_coords[0]))
    return (dx + dy) / 2.0


# ---------------------------------------------------------------------------
# Cross-sensor MESMA validation (AVIRIS-3 reflectance → char fraction)
# ---------------------------------------------------------------------------


def _aggregate_fractions_to_grid(
    fractions: xr.Dataset,
    target_y: np.ndarray,
    target_x: np.ndarray,
    target_resolution: float,
    *,
    target_crs: Optional[str] = None,
) -> xr.Dataset:
    """Aggregate fine-resolution fractions onto a coarser target grid.

    Performs CRS-aware reprojection when the source and target CRS differ,
    then area-weighted mean aggregation using xarray coarsen with NaN-correct
    skipna=True semantics. Pixels with no valid fine-resolution data within
    a coarse cell become NaN.

    Args:
        fractions: MESMA output Dataset with fraction variables and dims (y, x).
        target_y: Target grid y-coordinates (pixel centres).
        target_x: Target grid x-coordinates (pixel centres).
        target_resolution: Target pixel size in CRS units (metres).
        target_crs: CRS of the target grid (e.g. ``"EPSG:32611"``).

    Returns:
        xr.Dataset aligned to the target grid, with the same fraction
        variables.
    """
    # Step 1: coarsen to approximate target resolution.
    if "x" in fractions.coords and "y" in fractions.coords:
        src_x = np.asarray(fractions.coords["x"].values)
        src_y = np.asarray(fractions.coords["y"].values)
        if len(src_x) >= 2 and len(src_y) >= 2:
            src_dx = float(abs(src_x[1] - src_x[0]))
            src_dy = float(abs(src_y[1] - src_y[0]))
            if src_dx > 0 and src_dy > 0:
                factor_x = max(1, int(round(target_resolution / src_dx)))
                factor_y = max(1, int(round(target_resolution / src_dy)))
                if factor_x > 1 or factor_y > 1:
                    logger.info(
                        "Aggregating fractions: factor (y=%d, x=%d) for %.1f m target",
                        factor_y, factor_x, target_resolution,
                    )
                    fractions = fractions.coarsen(
                        y=factor_y, x=factor_x, boundary="trim",
                    ).mean(skipna=True)

    # Step 2: interpolate onto the exact target grid (nearest-neighbour for
    # fraction data to avoid creating out-of-[0,1] artefacts from linear
    # interpolation).
    aligned = fractions.interp(
        y=target_y, x=target_x,
        method="nearest",
        kwargs={"fill_value": np.nan},
    )

    if target_crs is not None:
        aligned.attrs["crs"] = target_crs

    n_total = aligned.sizes["y"] * aligned.sizes["x"]
    frac_vars = [v for v in aligned.data_vars if v != "rmse"]
    if frac_vars:
        first = aligned[frac_vars[0]]
        n_valid = int(np.sum(np.isfinite(first.values)))
        coverage = n_valid / n_total if n_total > 0 else 0.0
        logger.info(
            "_aggregate_fractions_to_grid: coverage %.1f%% (%d/%d valid)",
            coverage * 100.0, n_valid, n_total,
        )
        if coverage < 0.05:
            logger.warning(
                "Less than 5%% coverage after aggregation — check that the "
                "AVIRIS-3 swath overlaps the Tanager footprint"
            )

    return aligned


def cross_validate_aviris3(
    aviris3_reflectance: xr.Dataset,
    tanager_fractions: xr.Dataset,
    library: "xr.DataArray",
    *,
    target_resolution: float = _DEFAULT_TARGET_RESOLUTION_M,
    fraction_variable: str = "char",
    mesma_constraints: Optional[Mapping[str, float]] = None,
) -> dict[str, Any]:
    """Cross-sensor MESMA validation: AVIRIS-3 reflectance vs Tanager fractions.

    Runs MESMA on AVIRIS-3 surface reflectance using the same endmember library
    (resampled to the AVIRIS-3 wavelength grid), aggregates the resulting
    fine-resolution fractions to the Tanager 30 m grid, and computes
    continuous accuracy metrics (R², RMSE, MAE, bias) on the overlap area.

    This is cross-sensor validation — the same MESMA algorithm runs on both
    sensors — not independent ground truth. Frame results accordingly.

    Args:
        aviris3_reflectance: AVIRIS-3 reflectance Dataset from
            :func:`load_aviris3_reflectance` with dims (wavelength, y, x).
        tanager_fractions: Tanager MESMA output Dataset from
            :func:`tanager.unmixing.run_mesma` with fraction variables and
            dims (y, x).
        library: Endmember library DataArray (output of
            :func:`tanager.endmembers.build_fire_library`). Will be
            resampled to the AVIRIS-3 wavelength grid.
        target_resolution: Target grid resolution in metres for comparison.
        fraction_variable: Which fraction to compare (default ``"char"``).
        mesma_constraints: Optional MESMA constraint overrides.

    Returns:
        Dict with:

        * ``accuracy`` — output of :func:`compute_accuracy` (R², RMSE, etc.)
        * ``aviris3_fractions`` — aggregated AVIRIS-3 MESMA fractions on the
          Tanager grid.
        * ``overlap_area_pixels`` — number of valid comparison pixels.
        * ``method`` — ``"cross_sensor_mesma"`` for honest framing.
        * ``fraction_variable`` — which fraction was compared.

    Raises:
        ValueError: If the overlap area has fewer than 10 valid pixels, or if
            required variables are missing.
    """
    from tanager.endmembers import resample_library
    from tanager.unmixing import run_mesma

    if fraction_variable not in tanager_fractions.data_vars:
        raise ValueError(
            f"Tanager fractions Dataset missing {fraction_variable!r}. "
            f"Available: {list(tanager_fractions.data_vars)}"
        )

    # Resample endmember library to AVIRIS-3 wavelength grid.
    aviris3_wl = np.asarray(
        aviris3_reflectance.coords["wavelength"].values, dtype=np.float64
    )
    aviris3_library = resample_library(library, aviris3_wl)
    logger.info(
        "Resampled endmember library to AVIRIS-3 grid: %d bands",
        len(aviris3_wl),
    )

    # Run MESMA on the AVIRIS-3 reflectance.
    aviris3_mesma = run_mesma(
        aviris3_reflectance, aviris3_library, constraints=mesma_constraints,
    )
    logger.info(
        "AVIRIS-3 MESMA complete: engine=%s, shape=(%d, %d)",
        aviris3_mesma.attrs.get("unmixing_engine", "unknown"),
        aviris3_mesma.sizes.get("y", 0),
        aviris3_mesma.sizes.get("x", 0),
    )

    # Carry CRS from the reflectance Dataset to the fractions.
    if "crs" in aviris3_reflectance.attrs:
        aviris3_mesma.attrs["crs"] = aviris3_reflectance.attrs["crs"]

    # Aggregate AVIRIS-3 fractions to the Tanager grid.
    tgt_y = np.asarray(tanager_fractions.coords["y"].values, dtype=np.float64)
    tgt_x = np.asarray(tanager_fractions.coords["x"].values, dtype=np.float64)
    tgt_crs = tanager_fractions.attrs.get("crs")

    aviris3_aggregated = _aggregate_fractions_to_grid(
        aviris3_mesma,
        target_y=tgt_y,
        target_x=tgt_x,
        target_resolution=target_resolution,
        target_crs=tgt_crs,
    )

    # Compute accuracy on the overlap area.
    pred = np.asarray(tanager_fractions[fraction_variable].values, dtype=np.float64).ravel()
    obs = np.asarray(aviris3_aggregated[fraction_variable].values, dtype=np.float64).ravel()

    valid = np.isfinite(pred) & np.isfinite(obs)
    n_valid = int(valid.sum())
    if n_valid < 10:
        raise ValueError(
            f"Only {n_valid} valid comparison pixels in the overlap area "
            f"(minimum 10 required). Check that the AVIRIS-3 swath covers "
            f"the Tanager footprint."
        )

    accuracy = compute_accuracy(pred[valid], obs[valid], metric_type="continuous")

    logger.info(
        "Cross-sensor validation (%s): R²=%.3f, RMSE=%.4f, bias=%.4f, n=%d",
        fraction_variable,
        accuracy["r2"],
        accuracy["rmse"],
        accuracy["bias"],
        accuracy["n_valid"],
    )

    return {
        "accuracy": accuracy,
        "aviris3_fractions": aviris3_aggregated,
        "overlap_area_pixels": n_valid,
        "method": "cross_sensor_mesma",
        "fraction_variable": fraction_variable,
    }


def load_barc_reference(
    filepath: FilePath,
    *,
    code_map: Optional[Mapping[int, int]] = None,
    target_grid: Optional[xr.DataArray] = None,
    strict: bool = True,
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
            canonical 0..4 severity scheme. When ``strict=True`` (default),
            any code present in the raster but absent from the active code map
            raises :class:`ValueError`.
        target_grid: Optional DataArray whose ``y`` and ``x`` coordinates and
            ``crs`` attribute / spatial_ref describe the destination grid.
            When supplied the BARC raster is reprojected to that grid using
            nearest-neighbour resampling.
        strict: If ``True`` (default), raise :class:`ValueError` when the
            raster contains codes not covered by the active code map. Set to
            ``False`` to pass unmapped codes through with a warning instead.

    Returns:
        xr.DataArray with integer dtype, dims ``(y, x)``, and an attribute
        ``source = "barc"``. NoData pixels are encoded as ``-1`` so callers
        can mask them prior to computing accuracy metrics.

    Raises:
        FileNotFoundError: If ``filepath`` does not exist.
        ValueError: If ``strict=True`` and unmapped codes are found.
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
        unique_codes = np.unique(out)
        unmapped = [
            int(c)
            for c in unique_codes
            if int(c) not in mapping and int(c) != -1
        ]
        if unmapped:
            counts = {
                c: int(np.sum(out == c)) for c in unmapped
            }
            msg = (
                f"Raster contains codes not in code_map: {counts}. "
                "These codes will produce bogus classes in accuracy metrics. "
                "Supply a complete code_map covering all codes in the raster."
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
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
    """Align ``array`` onto the spatial coordinates of ``target`` (nearest).

    When both arrays carry a ``crs`` attribute and the CRS differs, the source
    array is reprojected via ``rasterio.warp.reproject`` (nearest-neighbour,
    appropriate for integer-coded severity classes).  When the CRS matches (or
    is absent), a lightweight ``xarray.interp`` is used instead.

    After alignment, a coverage fraction is computed and logged — if less than
    10 % of the target grid is covered by valid (non-nodata) source pixels, a
    warning is emitted so partial-overlap situations are caught early.
    """
    if "y" not in target.coords or "x" not in target.coords:
        return array

    src_crs = array.attrs.get("crs")
    tgt_crs = target.attrs.get("crs")

    if src_crs and tgt_crs and str(src_crs) != str(tgt_crs):
        aligned = _reproject_classified(array, target, src_crs, tgt_crs)
    else:
        if src_crs and tgt_crs:
            logger.info(
                "_align_to_target_grid: CRS match (%s), using coordinate interpolation",
                src_crs,
            )
        elif not src_crs or not tgt_crs:
            logger.warning(
                "_align_to_target_grid: CRS metadata missing on %s; "
                "assuming grids share the same CRS",
                "source" if not src_crs else "target",
            )
        aligned = array.interp(
            y=target["y"],
            x=target["x"],
            method="nearest",
            kwargs={"fill_value": -1},
        ).astype(array.dtype)

    nodata_val = array.attrs.get("nodata", -1)
    n_total = aligned.size
    n_valid = int(np.sum(np.asarray(aligned.values) != nodata_val))
    coverage = n_valid / n_total if n_total > 0 else 0.0
    logger.info(
        "_align_to_target_grid: coverage %.1f%% (%d/%d valid pixels)",
        coverage * 100.0,
        n_valid,
        n_total,
    )
    if coverage < 0.10:
        logger.warning(
            "_align_to_target_grid: only %.1f%% of target grid covered by "
            "reference data — check that the reference footprint overlaps "
            "the Tanager scene extent",
            coverage * 100.0,
        )

    return aligned


def _reproject_classified(
    source: xr.DataArray,
    target: xr.DataArray,
    src_crs: str,
    dst_crs: str,
) -> xr.DataArray:
    """Reproject an integer-coded raster onto a target grid via rasterio.

    Uses nearest-neighbour resampling (the only valid method for classified
    data) and fills areas outside the source extent with the nodata sentinel.
    """
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject

    nodata_val = source.attrs.get("nodata", -1)

    src_y = np.asarray(source.coords["y"].values, dtype=np.float64)
    src_x = np.asarray(source.coords["x"].values, dtype=np.float64)
    src_dy = abs(float(src_y[1] - src_y[0])) if len(src_y) > 1 else 30.0
    src_dx = abs(float(src_x[1] - src_x[0])) if len(src_x) > 1 else 30.0
    src_transform = from_bounds(
        float(src_x.min()) - src_dx / 2,
        float(src_y.min()) - src_dy / 2,
        float(src_x.max()) + src_dx / 2,
        float(src_y.max()) + src_dy / 2,
        len(src_x),
        len(src_y),
    )

    dst_y = np.asarray(target.coords["y"].values, dtype=np.float64)
    dst_x = np.asarray(target.coords["x"].values, dtype=np.float64)
    dst_dy = abs(float(dst_y[1] - dst_y[0])) if len(dst_y) > 1 else 30.0
    dst_dx = abs(float(dst_x[1] - dst_x[0])) if len(dst_x) > 1 else 30.0
    dst_transform = from_bounds(
        float(dst_x.min()) - dst_dx / 2,
        float(dst_y.min()) - dst_dy / 2,
        float(dst_x.max()) + dst_dx / 2,
        float(dst_y.max()) + dst_dy / 2,
        len(dst_x),
        len(dst_y),
    )

    src_data = np.asarray(source.values, dtype=np.int16)
    dst_data = np.full((len(dst_y), len(dst_x)), nodata_val, dtype=np.int16)

    logger.info(
        "_reproject_classified: reprojecting %s → %s (%d×%d → %d×%d)",
        src_crs,
        dst_crs,
        len(src_y),
        len(src_x),
        len(dst_y),
        len(dst_x),
    )

    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=CRS.from_user_input(src_crs),
        dst_transform=dst_transform,
        dst_crs=CRS.from_user_input(dst_crs),
        resampling=Resampling.nearest,
        src_nodata=nodata_val,
        dst_nodata=nodata_val,
    )

    return xr.DataArray(
        dst_data,
        dims=("y", "x"),
        coords={"y": dst_y, "x": dst_x},
        attrs={**source.attrs, "crs": dst_crs},
    )


def load_dins_reference(
    filepath: FilePath,
    target_crs: str = "EPSG:32611",
    *,
    damage_field: str = "DAMAGE",
) -> "Any":
    """Load a CAL FIRE DINS structure-damage GeoJSON as a projected GeoDataFrame.

    DINS (Damage INSpection) points are per-structure field assessments
    published by CAL FIRE after major fires. Each point carries a ``DAMAGE``
    category (No Damage → Destroyed); this loader reprojects the points to
    the raster CRS used by Tanager products and attaches a ``damage_ordinal``
    column (0..4, see :data:`DINS_DAMAGE_ORDINAL`) for rank-based comparison
    against continuous severity products.

    Args:
        filepath: Path to the DINS GeoJSON (WGS84 point geometry). Files
            without an embedded CRS are assumed to be EPSG:4326.
        target_crs: CRS to reproject the points into. Defaults to UTM 11N
            (``EPSG:32611``), the grid used by the Palisades/Eaton Tanager
            scenes.
        damage_field: Name of the damage-category attribute. Rows whose
            value is not a known :data:`DINS_DAMAGE_ORDINAL` category are
            dropped with a warning.

    Returns:
        geopandas.GeoDataFrame in ``target_crs`` with the original attribute
        columns plus ``damage_ordinal`` (int). Rows with missing geometry or
        missing/unknown damage category are dropped.

    Raises:
        FileNotFoundError: If ``filepath`` does not exist.
        ValueError: If ``damage_field`` is absent from the file.
    """
    import geopandas as gpd

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"DINS reference file not found: {filepath}. DINS GeoJSON exports "
            "are published by CAL FIRE per incident (e.g. Palisades 2025) via "
            "https://data.ca.gov / the CAL FIRE DINS viewer."
        )

    gdf = gpd.read_file(path)
    if damage_field not in gdf.columns:
        raise ValueError(
            f"Damage field {damage_field!r} not found in {path.name}; "
            f"available columns: {list(gdf.columns)}"
        )

    n_raw = len(gdf)
    gdf = gdf[gdf.geometry.notna() & gdf[damage_field].notna()].copy()

    if gdf.crs is None:
        logger.warning(
            "load_dins_reference: %s has no CRS; assuming EPSG:4326 (WGS84).",
            path.name,
        )
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(target_crs)

    ordinal = gdf[damage_field].map(DINS_DAMAGE_ORDINAL)
    unknown = ordinal.isna()
    if unknown.any():
        logger.warning(
            "load_dins_reference: dropping %d point(s) with unrecognised %s "
            "values %s (known categories: %s).",
            int(unknown.sum()),
            damage_field,
            sorted(gdf.loc[unknown, damage_field].unique().tolist()),
            list(DINS_DAMAGE_ORDINAL),
        )
        gdf = gdf[~unknown].copy()
        ordinal = ordinal[~unknown]
    gdf["damage_ordinal"] = ordinal.astype(int)

    logger.info(
        "load_dins_reference: %d/%d points loaded from %s → %s (categories: %s)",
        len(gdf),
        n_raw,
        path.name,
        target_crs,
        gdf[damage_field].value_counts().to_dict(),
    )
    return gdf


def cross_check_dins(
    dins_gdf: "Any",
    product_raster: Union[xr.DataArray, FilePath],
    product_name: str = "dNBR",
    *,
    threshold: float = 0.1,
    damaged_min_ordinal: int = 1,
    damage_field: str = "DAMAGE",
) -> dict[str, Any]:
    """Cross-check a continuous severity product against DINS structure damage.

    Samples ``product_raster`` at each DINS point (nearest pixel) and
    summarises how the product value relates to the field-assessed damage
    category:

    * per-category mean / median / std / count of the sampled product value,
    * binary detection metrics (accuracy, precision, recall, F1) where
      "observed damaged" is ``damage_ordinal >= damaged_min_ordinal`` and
      "predicted damaged" is ``product value >= threshold``,
    * Spearman rank correlation between the damage ordinal and the product
      value.

    DINS is a structure-level (not pixel-level) reference: a destroyed house
    inside a defended lot can sit on a low-dNBR pixel, so expect the rank
    correlation to be modest even for a good product — the monotonic trend of
    the per-category means is the more meaningful signal.

    Args:
        dins_gdf: GeoDataFrame from :func:`load_dins_reference` (must contain
            point geometry; ``damage_ordinal`` is recomputed from
            ``damage_field`` if absent). Must be in a projected CRS matching
            the raster, or carry a CRS so it can be reprojected.
        product_raster: Continuous-valued DataArray with ``y``/``x``
            coordinates (e.g. dNBR from :func:`tanager.indices.nbr`
            differencing) or a path to a single-band GeoTIFF.
        product_name: Label for the product, echoed in the result dict.
        threshold: Product value at/above which a point counts as "predicted
            damaged". Default ``0.1``, the Key & Benson (2006)
            unburned/burned dNBR boundary.
        damaged_min_ordinal: Minimum ``damage_ordinal`` that counts as
            "observed damaged". Default ``1`` (any damage, i.e. Affected+).
        damage_field: Damage-category column used to recompute ordinals when
            ``damage_ordinal`` is missing.

    Returns:
        Dict with keys ``product_name``, ``threshold``, ``n_points``,
        ``n_valid``, ``n_outside``, ``per_category`` (category →
        ``{ordinal, count, mean, median, std}``), ``binary``
        (``{threshold, damaged_min_ordinal, tp, fp, fn, tn, accuracy,
        precision, recall, f1}``), and ``spearman_rho``.

    Raises:
        ValueError: If no DINS point falls on a valid raster pixel.
    """
    if isinstance(product_raster, (str, PathLike)):
        product_raster = _load_continuous_raster(Path(product_raster))

    gdf = dins_gdf
    raster_crs = product_raster.attrs.get("crs")
    if raster_crs is not None and gdf.crs is not None and str(gdf.crs) != str(raster_crs):
        logger.info(
            "cross_check_dins: reprojecting DINS points %s → raster CRS %s",
            gdf.crs,
            raster_crs,
        )
        gdf = gdf.to_crs(raster_crs)

    if "damage_ordinal" in gdf.columns:
        ordinals = np.asarray(gdf["damage_ordinal"].values, dtype=np.int64)
    else:
        mapped = gdf[damage_field].map(DINS_DAMAGE_ORDINAL)
        if mapped.isna().any():
            raise ValueError(
                f"Unrecognised {damage_field!r} categories "
                f"{sorted(gdf.loc[mapped.isna(), damage_field].unique().tolist())}; "
                "load the points with load_dins_reference() first."
            )
        ordinals = np.asarray(mapped.values, dtype=np.int64)

    px = np.asarray(gdf.geometry.x.values, dtype=np.float64)
    py = np.asarray(gdf.geometry.y.values, dtype=np.float64)
    values = _sample_raster_at_points(product_raster, px, py)

    outside = np.isnan(values)
    n_outside = int(outside.sum())
    valid = ~outside
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError(
            f"No DINS point falls on a valid {product_name} pixel — check that "
            "the raster and the points share a CRS and actually overlap."
        )

    vals_v = values[valid]
    ords_v = ordinals[valid]
    categories = np.asarray(gdf[damage_field].values)[valid]

    per_category: dict[str, dict[str, Any]] = {}
    for cat in sorted(np.unique(categories), key=lambda c: DINS_DAMAGE_ORDINAL.get(str(c), -1)):
        cat_vals = vals_v[categories == cat]
        per_category[str(cat)] = {
            "ordinal": int(DINS_DAMAGE_ORDINAL.get(str(cat), -1)),
            "count": int(cat_vals.size),
            "mean": float(np.mean(cat_vals)),
            "median": float(np.median(cat_vals)),
            "std": float(np.std(cat_vals)),
        }

    obs_damaged = ords_v >= damaged_min_ordinal
    pred_damaged = vals_v >= threshold
    tp = int(np.sum(pred_damaged & obs_damaged))
    fp = int(np.sum(pred_damaged & ~obs_damaged))
    fn = int(np.sum(~pred_damaged & obs_damaged))
    tn = int(np.sum(~pred_damaged & ~obs_damaged))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    result = {
        "product_name": product_name,
        "threshold": float(threshold),
        "n_points": int(len(gdf)),
        "n_valid": n_valid,
        "n_outside": n_outside,
        "per_category": per_category,
        "binary": {
            "threshold": float(threshold),
            "damaged_min_ordinal": int(damaged_min_ordinal),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "accuracy": float((tp + tn) / n_valid),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
        "spearman_rho": _spearman(vals_v, ords_v.astype(np.float64)),
    }
    logger.info(
        "cross_check_dins: %s vs %d DINS points — spearman_rho=%.3f, "
        "recall=%.3f @ threshold=%.2f (%d outside raster)",
        product_name,
        n_valid,
        result["spearman_rho"],
        recall,
        threshold,
        n_outside,
    )
    return result


def _load_continuous_raster(path: Path) -> xr.DataArray:
    """Load band 1 of a rasterio-readable file as a float DataArray (NaN nodata)."""
    import rasterio

    if not path.exists():
        raise FileNotFoundError(f"Product raster not found: {path}")

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        if src.nodata is not None and not np.isnan(src.nodata):
            data = np.where(data == src.nodata, np.nan, data)
        transform = src.transform
        crs = src.crs
        rows, cols = data.shape
        xs = transform.c + (np.arange(cols) + 0.5) * transform.a
        ys = transform.f + (np.arange(rows) + 0.5) * transform.e

    da = xr.DataArray(data, dims=("y", "x"), coords={"y": ys, "x": xs})
    if crs is not None:
        da.attrs["crs"] = str(crs)
    return da


def _sample_raster_at_points(
    raster: xr.DataArray,
    px: np.ndarray,
    py: np.ndarray,
) -> np.ndarray:
    """Sample ``raster`` at point coordinates via nearest pixel.

    Points falling more than half a pixel outside the grid extent return NaN.
    """
    ys = np.asarray(raster["y"].values, dtype=np.float64)
    xs = np.asarray(raster["x"].values, dtype=np.float64)
    data = np.asarray(raster.values, dtype=np.float64)

    iy, ok_y = _nearest_coord_index(ys, py)
    ix, ok_x = _nearest_coord_index(xs, px)

    out = np.full(px.shape, np.nan, dtype=np.float64)
    ok = ok_y & ok_x
    out[ok] = data[iy[ok], ix[ok]]
    return out


def _nearest_coord_index(
    coords: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest index into a monotonic 1-D coordinate axis, with bounds mask.

    Handles ascending and descending axes (raster ``y`` is typically
    descending). Targets beyond half a pixel outside the axis extent are
    flagged False in the returned mask.
    """
    if coords.size < 2:
        idx = np.zeros(targets.shape, dtype=np.int64)
        ok = np.isfinite(targets) if coords.size else np.zeros(targets.shape, dtype=bool)
        return idx, ok

    ascending = coords[0] <= coords[-1]
    c = coords if ascending else coords[::-1]

    pos = np.clip(np.searchsorted(c, targets), 1, len(c) - 1)
    left = c[pos - 1]
    right = c[pos]
    nearest = np.where(np.abs(targets - left) <= np.abs(right - targets), pos - 1, pos)

    half_step = float(np.median(np.abs(np.diff(c)))) / 2.0
    ok = (targets >= c[0] - half_step) & (targets <= c[-1] + half_step)

    if not ascending:
        nearest = len(coords) - 1 - nearest
    return nearest.astype(np.int64), ok


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


# ---------------------------------------------------------------------------
# Spectral degradation simulation (sensor cross-comparison)
#
# Convolve a Tanager-1 native scene (426 bands, ~5.5 nm FWHM) into the band
# centres + FWHM of a coarser reference sensor (EMIT, PRISMA, Sentinel-2).
# Used by the +5 competition tie-breaker — see compare_sensors() above.
# ---------------------------------------------------------------------------

# Reflectance is bounded in [0, 1] across the pipeline; mirror the constants
# used in endmembers.py for resample_library() to avoid drift.
_REFLECTANCE_MIN: float = 0.0
_REFLECTANCE_MAX: float = 1.0
# Mean Tanager-1 per-band FWHM. Spectral characterisation shows per-band
# values in 5.20-6.81 nm; 5.5 nm is the operating mean and matches
# endmembers._DEFAULT_TARGET_FWHM_NM.
_DEFAULT_TANAGER_FWHM_NM: float = 5.5


def simulate_sensor(
    scene: Union[xr.DataArray, xr.Dataset],
    target_centers: Union[np.ndarray, Sequence[float]],
    target_fwhm: Union[float, np.ndarray, Sequence[float]],
    sensor_name: str,
    *,
    source_fwhm: Union[float, np.ndarray, Sequence[float]] = _DEFAULT_TANAGER_FWHM_NM,
) -> Union[xr.DataArray, xr.Dataset]:
    """Spectrally degrade a Tanager scene to simulate a reference sensor.

    Convolves the input scene's reflectance from Tanager's 426 native bands
    onto ``target_centers`` using :class:`spectral.BandResampler` (Gaussian
    overlap integration). Pattern follows :func:`tanager.endmembers.resample_library`
    but operates on multi-pixel scenes (DataArray with ``(wavelength, y, x)``
    dims, or Dataset with one or more wavelength-bearing variables).

    Args:
        scene: Source Tanager scene. Either an :class:`xr.DataArray` with a
            ``wavelength`` dim, or an :class:`xr.Dataset` whose data variables
            optionally carry a ``wavelength`` dim. Variables without a
            ``wavelength`` dim are passed through unchanged.
        target_centers: 1D array of target band centres in nanometres
            (e.g. ``EMIT_SENSOR`` 285 wavelengths).
        target_fwhm: Target sensor FWHM in nanometres. Scalar broadcasts to
            every target band, or a per-band 1D array matching
            ``target_centers``.
        sensor_name: Human-readable label written to the output's
            ``sensor_name`` attribute (e.g. ``"EMIT"``, ``"PRISMA"``,
            ``"Sentinel-2"``).
        source_fwhm: Source (Tanager) FWHM in nanometres. Defaults to
            ``5.5`` nm — the mean across Tanager's 5.20-6.81 nm
            per-band range. Pass the per-band ``coords["fwhm"]`` array when
            available for higher accuracy.

    Returns:
        Same xarray container type as ``scene`` with:

        * ``wavelength`` coord replaced by ``target_centers`` (float32).
        * Reflectance clipped to ``[0, 1]``.
        * Spatial / non-wavelength coordinates preserved.
        * Attributes: original ``attrs`` plus ``sensor_name`` and
          ``target_fwhm_nm`` (scalar when ``target_fwhm`` was scalar; a
          ``(min, max)`` tuple when it was a per-band array).

    Raises:
        ValueError: If ``target_centers`` is not a non-empty 1D array, or
            if ``scene`` (DataArray) has no ``wavelength`` dim.
    """
    target_centers_arr = np.asarray(target_centers, dtype=np.float64).ravel()
    if target_centers_arr.ndim != 1 or target_centers_arr.size == 0:
        raise ValueError("target_centers must be a non-empty 1D array")

    if isinstance(scene, xr.Dataset):
        return _simulate_sensor_dataset(
            scene,
            target_centers_arr,
            target_fwhm,
            sensor_name,
            source_fwhm,
        )
    return _simulate_sensor_dataarray(
        scene,
        target_centers_arr,
        target_fwhm,
        sensor_name,
        source_fwhm,
    )


def _simulate_sensor_dataset(
    scene: xr.Dataset,
    target_centers: np.ndarray,
    target_fwhm: Union[float, np.ndarray, Sequence[float]],
    sensor_name: str,
    source_fwhm: Union[float, np.ndarray, Sequence[float]],
) -> xr.Dataset:
    """Apply :func:`simulate_sensor` to every wavelength-bearing variable."""
    new_vars: dict[str, xr.DataArray] = {}
    for var_name, var in scene.data_vars.items():
        if "wavelength" in var.dims:
            new_vars[var_name] = _simulate_sensor_dataarray(
                var, target_centers, target_fwhm, sensor_name, source_fwhm,
            )
        else:
            new_vars[var_name] = var

    # Carry forward non-wavelength coords (y, x, time, ...). The new
    # wavelength coord is contributed by each resampled DataArray.
    extra_coords = {
        name: coord
        for name, coord in scene.coords.items()
        if "wavelength" not in coord.dims and name != "wavelength"
    }

    target_fwhm_attr = _format_target_fwhm_attr(target_fwhm, target_centers.size)
    out_attrs = {
        **scene.attrs,
        "sensor_name": sensor_name,
        "target_fwhm_nm": target_fwhm_attr,
    }

    return xr.Dataset(new_vars, coords=extra_coords, attrs=out_attrs)


def _simulate_sensor_dataarray(
    da: xr.DataArray,
    target_centers: np.ndarray,
    target_fwhm: Union[float, np.ndarray, Sequence[float]],
    sensor_name: str,
    source_fwhm: Union[float, np.ndarray, Sequence[float]],
) -> xr.DataArray:
    """Resample a single DataArray onto ``target_centers``."""
    if "wavelength" not in da.dims:
        raise ValueError(
            f"scene DataArray must have a 'wavelength' dim, got dims={da.dims}"
        )
    if "wavelength" not in da.coords:
        raise ValueError("scene DataArray must have a 'wavelength' coordinate")

    source_centers = np.asarray(da.coords["wavelength"].values, dtype=np.float64)
    n_source = source_centers.size
    n_target = target_centers.size

    fwhm_target = np.broadcast_to(
        np.asarray(target_fwhm, dtype=np.float64), (n_target,),
    ).copy()
    fwhm_source = np.broadcast_to(
        np.asarray(source_fwhm, dtype=np.float64), (n_source,),
    ).copy()

    from spectral import BandResampler  # heavy dep — defer import

    resampler = BandResampler(
        centers1=source_centers,
        centers2=target_centers,
        fwhm1=fwhm_source,
        fwhm2=fwhm_target,
    )

    # Move wavelength to the trailing axis so we can flatten everything else
    # into rows of length n_source and resample row-by-row.
    other_dims = tuple(d for d in da.dims if d != "wavelength")
    da_trans = da.transpose(*other_dims, "wavelength")
    arr = np.asarray(da_trans.values, dtype=np.float64)

    other_shape = arr.shape[:-1]
    flat = np.nan_to_num(arr.reshape(-1, n_source), nan=0.0)

    out_flat = np.empty((flat.shape[0], n_target), dtype=np.float32)
    for i in range(flat.shape[0]):
        resampled = np.asarray(resampler(flat[i]), dtype=np.float32)
        # BandResampler emits NaN for any target band with no source overlap;
        # treat those as 0 so reflectance bounds hold downstream.
        resampled = np.where(np.isnan(resampled), np.float32(0.0), resampled)
        out_flat[i] = resampled
    out_flat = np.clip(out_flat, _REFLECTANCE_MIN, _REFLECTANCE_MAX)

    out = out_flat.reshape((*other_shape, n_target))

    new_coords: dict[str, Any] = {}
    for d in other_dims:
        if d in da_trans.coords:
            new_coords[d] = da_trans.coords[d]
    new_coords["wavelength"] = target_centers.astype(np.float32)

    target_fwhm_attr = _format_target_fwhm_attr(target_fwhm, n_target)
    new_attrs = {
        **da.attrs,
        "sensor_name": sensor_name,
        "target_fwhm_nm": target_fwhm_attr,
    }

    out_da = xr.DataArray(
        out,
        dims=(*other_dims, "wavelength"),
        coords=new_coords,
        attrs=new_attrs,
        name=da.name,
    )
    # Restore the caller's original axis order for ergonomic chaining.
    return out_da.transpose(*da.dims)


def _format_target_fwhm_attr(
    target_fwhm: Union[float, np.ndarray, Sequence[float]],
    n_target: int,
) -> Union[float, tuple[float, float]]:
    """Return a JSON/NetCDF-friendly summary of the target FWHM input."""
    arr = np.asarray(target_fwhm, dtype=np.float64).ravel()
    if arr.size == 1:
        return float(arr[0])
    return (float(arr.min()), float(arr.max()))


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
