"""Spectral band operations for Tanager-1 hyperspectral imagery.

This module provides wavelength-based band selection, bad-band masking,
normalized-difference spectral indices (NBR, NDVI, NDWI, dNBR), and convex-hull
continuum removal for xarray Datasets produced by the tanager.io loader.  All
operations return new objects; the input is never modified in place.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from tanager.config import BAD_BAND_RANGES, BAND_ALIASES

logger = logging.getLogger(__name__)

_EXPECTED_GOOD_BAND_MIN = 320
_EXPECTED_GOOD_BAND_MAX = 360

# Threshold below which |NIR + SWIR| is treated as zero in normalized-difference
# indices. Real Tanager surface reflectance contains negative ISOFIT retrievals
# and near-zero values that produce extreme ratios (NBR=-99) without a guard.
_DENOMINATOR_EPSILON = 1e-3

# HDF5 path used by Tanager ortho_sr products; mirrors tanager.io._ORTHO_SR_DATASET.
_ORTHO_SR_DATASET_PATH = "HDFEOS/GRIDS/HYP/Data Fields/surface_reflectance"


def select_bands(
    dataset: xr.Dataset,
    *,
    min_wl: Optional[float] = None,
    max_wl: Optional[float] = None,
    wavelengths: Optional[Sequence[float]] = None,
) -> tuple[xr.Dataset, np.ndarray] | xr.Dataset:
    """Select a subset of spectral bands from a hyperspectral dataset.

    Exactly one of the two selection modes must be specified:

    * **Range mode** (``min_wl`` + ``max_wl``): returns all bands whose centre
      wavelength falls within the closed interval ``[min_wl, max_wl]``.
    * **Nearest-neighbor mode** (``wavelengths``): maps each requested
      wavelength to the closest band present in the dataset and returns a
      ``(dataset, matched_wavelengths)`` tuple so callers know which bands were
      actually selected.

    Args:
        dataset: xarray Dataset with a ``wavelength`` coordinate (units: nm).
        min_wl: Lower bound of wavelength range (nm), inclusive.  Must be
            provided together with ``max_wl``.
        max_wl: Upper bound of wavelength range (nm), inclusive.  Must be
            provided together with ``min_wl``.
        wavelengths: Sequence of target wavelengths (nm) for nearest-neighbor
            matching.  Duplicates are preserved in the output order.

    Returns:
        Range mode: an ``xr.Dataset`` containing only the selected bands.
        Nearest-neighbor mode: a ``(xr.Dataset, np.ndarray)`` tuple where the
        second element is the array of matched centre wavelengths.

    Raises:
        ValueError: If both modes are specified simultaneously, if neither mode
            is specified, or if a range selection yields zero matching bands.
    """
    range_provided = (min_wl is not None) or (max_wl is not None)
    nn_provided = wavelengths is not None

    if range_provided and nn_provided:
        raise ValueError(
            "Specify either (min_wl, max_wl) for range selection "
            "or wavelengths for nearest-neighbor selection, not both."
        )
    if not range_provided and not nn_provided:
        raise ValueError("One of (min_wl, max_wl) or wavelengths must be provided.")

    if range_provided:
        if min_wl is None or max_wl is None:
            raise ValueError("Both min_wl and max_wl must be provided for range selection.")
        return _select_by_range(dataset, min_wl, max_wl)

    return _select_by_nearest(dataset, wavelengths)  # type: ignore[arg-type]


def _select_by_range(dataset: xr.Dataset, min_wl: float, max_wl: float) -> xr.Dataset:
    """Return bands in [min_wl, max_wl] using boolean indexing.

    Args:
        dataset: Dataset with ``wavelength`` coordinate (nm).
        min_wl: Inclusive lower bound (nm).
        max_wl: Inclusive upper bound (nm).

    Returns:
        Subset Dataset restricted to the specified wavelength range.

    Raises:
        ValueError: If no bands fall within the specified range.
    """
    wl = dataset.coords["wavelength"]
    mask = (wl >= min_wl) & (wl <= max_wl)
    if not mask.any():
        raise ValueError(
            f"No bands found in wavelength range [{min_wl}, {max_wl}] nm. "
            f"Dataset covers {float(wl.min()):.1f}–{float(wl.max()):.1f} nm."
        )
    return dataset.sel(wavelength=mask)


def _select_by_nearest(
    dataset: xr.Dataset, wavelengths: Sequence[float]
) -> tuple[xr.Dataset, np.ndarray]:
    """Return nearest-neighbor band matches for each requested wavelength.

    Args:
        dataset: Dataset with ``wavelength`` coordinate (nm).
        wavelengths: Requested centre wavelengths (nm).

    Returns:
        Tuple of (subset Dataset, array of actual matched wavelengths).
    """
    target = xr.DataArray(list(wavelengths), dims="wavelength")
    subset = dataset.sel(wavelength=target, method="nearest")
    matched = subset.coords["wavelength"].values.copy()
    return subset, matched


def _read_good_wavelengths_from_hdf5(
    hdf5_filepath: Union[str, "os.PathLike[str]"],
) -> np.ndarray:
    """Read the per-band ``good_wavelengths`` flag array from a Tanager HDF5 file.

    The Tanager ``ortho_sr`` HDF5 product stores a uint8 array of length 426
    on the ``surface_reflectance`` dataset's ``good_wavelengths`` attribute,
    where ``1`` marks a sensor-validated band and ``0`` marks a band the
    sensor flags as bad (predominantly water-vapour absorption regions:
    ~1342–1437 nm and ~1782–1967 nm).

    Args:
        hdf5_filepath: Path to a Tanager ``.h5`` ortho_sr file.

    Returns:
        Boolean array of shape ``(n_bands,)`` where ``True`` indicates a
        sensor-good band.

    Raises:
        ValueError: If the file cannot be opened, the surface_reflectance
            dataset is missing, or the ``good_wavelengths`` attribute is
            absent.
    """
    try:
        import h5py  # heavy dep — imported lazily
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ValueError("h5py is required to read good_wavelengths from HDF5") from exc

    path_str = os.fspath(hdf5_filepath)
    try:
        h5 = h5py.File(path_str, "r")
    except OSError as exc:
        raise ValueError(f"Cannot read Tanager HDF5 file {path_str!r}: {exc}") from exc

    with h5:
        if _ORTHO_SR_DATASET_PATH not in h5:
            raise ValueError(
                f"File {path_str!r} is missing surface_reflectance dataset "
                f"at {_ORTHO_SR_DATASET_PATH!r}"
            )
        sr_attrs = dict(h5[_ORTHO_SR_DATASET_PATH].attrs)
        if "good_wavelengths" not in sr_attrs:
            raise ValueError(
                f"surface_reflectance in {path_str!r} is missing the 'good_wavelengths' attribute"
            )
        # Sensor convention: 1 = good, 0 = bad.
        return np.asarray(sr_attrs["good_wavelengths"]).astype(bool)


def mask_bad_bands(
    dataset: xr.Dataset,
    *,
    zones: Optional[list[tuple[float, float]]] = None,
    hdf5_filepath: Optional[Union[str, "os.PathLike[str]"]] = None,
) -> xr.Dataset:
    """Remove bands that fall within known atmospheric-absorption and sensor-edge ranges.

    By default the four standard Tanager-1 exclusion zones defined in
    ``tanager.config.BAD_BAND_RANGES`` are applied:

    * 0–400 nm   — sensor edge / below reliable detector response
    * 1340–1480 nm — water vapour absorption band 1
    * 1780–1970 nm — water vapour absorption band 2
    * 2350–2500 nm — CO₂ / H₂O absorption at long-wave sensor edge

    When ``zones`` is provided it **replaces** the defaults entirely; the
    caller is responsible for specifying all exclusion zones they want applied.

    The function also honours the sensor-provided ``good_wavelengths`` flag
    when available.  A band is considered bad if it falls inside any
    exclusion zone OR the sensor has flagged it (``good_wavelengths`` value of
    ``False``/``0``).  The flag is sourced from, in priority order:

    1. ``hdf5_filepath`` (when provided) — read directly from the
       ``surface_reflectance.good_wavelengths`` attribute of the HDF5 file.
    2. The dataset's own ``good_wavelengths`` coordinate, populated by
       ``tanager.io.load_ortho_scene`` for ortho_sr products.

    Args:
        dataset: xarray Dataset with a ``wavelength`` coordinate (units: nm).
        zones: Optional list of ``(low_nm, high_nm)`` tuples.  Each band whose
            centre wavelength falls within any zone (inclusive on both ends) is
            excluded.  When provided this argument replaces the default
            ``BAD_BAND_RANGES`` entirely.
        hdf5_filepath: Optional path to the source Tanager HDF5 file.  When
            provided the ``good_wavelengths`` attribute is read from the file
            and OR-combined with the wavelength-zone mask.  Length must match
            the dataset's ``wavelength`` dimension.

    Returns:
        A new Dataset with bad bands removed.  The ``wavelength`` coordinate
        is a contiguous sorted sub-array of the input coordinate.

    Raises:
        ValueError: If ``hdf5_filepath`` cannot be read or its
            ``good_wavelengths`` length does not match the dataset.

    Warns:
        Logs a WARNING if the remaining band count is outside the expected
        330–346 range (only when default zones are applied to a 426-band
        dataset).
    """
    exclusion_zones = zones if zones is not None else BAD_BAND_RANGES

    wl = dataset.coords["wavelength"]
    n_input = int(wl.sizes["wavelength"])

    good_mask = np.ones(n_input, dtype=bool)
    for low, high in exclusion_zones:
        good_mask &= ~((wl.values >= low) & (wl.values <= high))

    sensor_good: Optional[np.ndarray] = None
    sensor_source: Optional[str] = None
    if hdf5_filepath is not None:
        sensor_good = _read_good_wavelengths_from_hdf5(hdf5_filepath)
        sensor_source = f"hdf5_filepath={os.fspath(hdf5_filepath)!r}"
    elif "good_wavelengths" in dataset.coords:
        sensor_good = np.asarray(dataset.coords["good_wavelengths"].values).astype(bool)
        sensor_source = "dataset.coords['good_wavelengths']"

    if sensor_good is not None:
        if sensor_good.shape != (n_input,):
            raise ValueError(
                f"good_wavelengths length {sensor_good.shape[0]} does not match "
                f"dataset wavelength dimension {n_input} (source: {sensor_source})"
            )
        sensor_excluded = int(np.sum(good_mask & ~sensor_good))
        good_mask &= sensor_good
        logger.debug(
            "mask_bad_bands: applied sensor good_wavelengths from %s "
            "(%d bands flagged bad by sensor not already excluded by zones)",
            sensor_source,
            sensor_excluded,
        )

    n_excluded = int(np.sum(~good_mask))
    n_remaining = int(np.sum(good_mask))

    logger.info(
        "mask_bad_bands: excluded %d bands, %d bands remaining (of %d input)",
        n_excluded,
        n_remaining,
        n_input,
    )

    if zones is None and sensor_good is None and n_input == 426:
        if not (_EXPECTED_GOOD_BAND_MIN <= n_remaining <= _EXPECTED_GOOD_BAND_MAX):
            logger.warning(
                "mask_bad_bands: expected ~330–346 good bands (real data) after "
                "masking 426-band dataset but got %d; verify BAD_BAND_RANGES "
                "and wavelength grid.",
                n_remaining,
            )

    return dataset.sel(wavelength=good_mask)


# ---------------------------------------------------------------------------
# Spectral indices
# ---------------------------------------------------------------------------


_REFLECTANCE_VARIABLE_PRIORITY: tuple[str, ...] = (
    "reflectance",
    "surface_reflectance",
    "toa_radiance",
)


def _reflectance_var(dataset: xr.Dataset) -> xr.DataArray:
    """Locate the reflectance/radiance cube on a Tanager Dataset.

    Tanager ortho_sr products expose ``surface_reflectance`` (with a
    ``toa_radiance`` alias for swath-path compatibility); synthetic test
    fixtures use ``reflectance``.  Pick the first variable that exists in
    a fixed priority order so spectral indices work transparently across
    both layouts.
    """
    explicit = dataset.attrs.get("data_var")
    if isinstance(explicit, str) and explicit in dataset.data_vars:
        return dataset[explicit]
    for name in _REFLECTANCE_VARIABLE_PRIORITY:
        if name in dataset.data_vars:
            return dataset[name]
    raise ValueError(
        "Dataset has no reflectance variable; expected one of "
        f"{_REFLECTANCE_VARIABLE_PRIORITY}, got {list(dataset.data_vars)}"
    )


def scene_reflectance(scene: Union[xr.Dataset, xr.DataArray]) -> xr.DataArray:
    """Return the reflectance DataArray for a scene.

    Accepts either a Dataset (resolved via ``attrs['data_var']`` first, then
    walking :data:`_REFLECTANCE_VARIABLE_PRIORITY`) or a bare DataArray
    (returned unchanged).  Used by downstream pipelines (lfmc, unmixing) so
    callers can pass a ``load_ortho_scene`` Dataset without manual variable
    extraction.
    """
    if isinstance(scene, xr.Dataset):
        return _reflectance_var(scene)
    if isinstance(scene, xr.DataArray):
        return scene
    raise TypeError(f"scene must be xr.Dataset or xr.DataArray, got {type(scene).__name__}")


def _normalized_difference(band1: xr.DataArray, band2: xr.DataArray) -> xr.DataArray:
    """Compute (band1 - band2) / (band1 + band2) with NaN where the denominator is near zero.

    Args:
        band1: First spectral band DataArray.
        band2: Second spectral band DataArray.

    Returns:
        DataArray with normalized difference values in [-1, 1], NaN where
        ``|band1 + band2| < _DENOMINATOR_EPSILON``.  The epsilon guard is
        critical for real ISOFIT surface reflectance: small/negative values
        sum to a near-zero denominator and would otherwise blow up the ratio.
    """
    numerator = band1 - band2
    denominator = band1 + band2
    return xr.where(
        np.abs(denominator) < _DENOMINATOR_EPSILON,
        np.nan,
        numerator / denominator,
    )


def clamp_reflectance(
    data: Union[xr.Dataset, xr.DataArray],
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> Union[xr.Dataset, xr.DataArray]:
    """Clamp reflectance values to ``[vmin, vmax]``.

    Tanager-1 surface-reflectance products from ISOFIT atmospheric correction
    contain a substantial fraction of negative values (~13% on the LA-area
    scenes used for this competition) and occasional extreme positive
    outliers.  Both violate the physical [0, 1] range of reflectance and
    produce nonsense in spectral indices such as NBR.  Clamping each input
    band to a physical range before the index ratio is the standard
    mitigation.

    Args:
        data: Either an ``xr.Dataset`` containing a ``reflectance`` variable
            or a single ``xr.DataArray`` of reflectance values.  When a
            Dataset is passed, only the ``reflectance`` variable is clamped;
            other variables and coordinates are preserved unchanged.
        vmin: Lower clamp bound (inclusive).  Default ``0.0``.
        vmax: Upper clamp bound (inclusive).  Default ``1.0``.

    Returns:
        A new object of the same type as ``data`` with values outside
        ``[vmin, vmax]`` clipped to the bound.  The input is not modified.

    Raises:
        ValueError: If ``vmin > vmax`` or if a Dataset without a
            ``reflectance`` variable is passed.
    """
    if vmin > vmax:
        raise ValueError(f"vmin ({vmin}) must be <= vmax ({vmax})")

    if isinstance(data, xr.Dataset):
        if "reflectance" not in data:
            raise ValueError("Dataset must contain a 'reflectance' variable")
        refl = data["reflectance"]
        n_clamped = int(((refl < vmin) | (refl > vmax)).sum())
        clamped = data.copy()
        clamped["reflectance"] = refl.clip(min=vmin, max=vmax)
    else:
        n_clamped = int(((data < vmin) | (data > vmax)).sum())
        clamped = data.clip(min=vmin, max=vmax)

    if n_clamped > 0:
        logger.debug(
            "clamp_reflectance: clamped %d values to [%g, %g]",
            n_clamped,
            vmin,
            vmax,
        )
    return clamped


def nbr(dataset: xr.Dataset) -> xr.DataArray:
    """Compute Normalized Burn Ratio (NBR).

    NBR = (NIR - SWIR2) / (NIR + SWIR2)

    Uses 860 nm as NIR and 2200 nm as SWIR2, matched to nearest available band
    in the dataset (Tanager-1 5 nm spacing; match within 2.5 nm guaranteed).

    Args:
        dataset: xarray Dataset with a ``wavelength`` coordinate (nm) and a
            ``reflectance`` variable of shape (wavelength, y, x).

    Returns:
        DataArray of NBR values with spatial dimensions (y, x).  Values are in
        [-1, 1]; pixels where ``|NIR + SWIR2| < _DENOMINATOR_EPSILON`` are set
        to NaN.  Both bands are clamped to ``[0, 1]`` before the ratio is
        computed so ISOFIT negative-reflectance retrievals do not propagate.
    """
    refl = _reflectance_var(dataset)
    nir = refl.sel(wavelength=BAND_ALIASES["NIR"], method="nearest")
    swir2 = refl.sel(wavelength=BAND_ALIASES["SWIR2"], method="nearest")
    nir = clamp_reflectance(nir)
    swir2 = clamp_reflectance(swir2)
    return _normalized_difference(nir, swir2)


def ndvi(dataset: xr.Dataset) -> xr.DataArray:
    """Compute Normalized Difference Vegetation Index (NDVI).

    NDVI = (NIR - Red) / (NIR + Red)

    Uses 860 nm as NIR and 660 nm as Red, matched to nearest available band.

    Args:
        dataset: xarray Dataset with a ``wavelength`` coordinate (nm) and a
            ``reflectance`` variable of shape (wavelength, y, x).

    Returns:
        DataArray of NDVI values with spatial dimensions (y, x).  Values are in
        [-1, 1]; pixels where ``|NIR + Red| < _DENOMINATOR_EPSILON`` are set
        to NaN.  Both bands are clamped to ``[0, 1]`` before the ratio is
        computed.
    """
    refl = _reflectance_var(dataset)
    nir = refl.sel(wavelength=BAND_ALIASES["NIR"], method="nearest")
    red = refl.sel(wavelength=BAND_ALIASES["RED"], method="nearest")
    nir = clamp_reflectance(nir)
    red = clamp_reflectance(red)
    return _normalized_difference(nir, red)


def ndwi(dataset: xr.Dataset) -> xr.DataArray:
    """Compute Normalized Difference Water Index (NDWI).

    NDWI = (Green - NIR) / (Green + NIR)

    Uses 560 nm as Green and 860 nm as NIR, matched to nearest available band.

    Args:
        dataset: xarray Dataset with a ``wavelength`` coordinate (nm) and a
            ``reflectance`` variable of shape (wavelength, y, x).

    Returns:
        DataArray of NDWI values with spatial dimensions (y, x).  Values are in
        [-1, 1]; pixels where ``|Green + NIR| < _DENOMINATOR_EPSILON`` are set
        to NaN.  Both bands are clamped to ``[0, 1]`` before the ratio is
        computed.
    """
    refl = _reflectance_var(dataset)
    green = refl.sel(wavelength=BAND_ALIASES["GREEN"], method="nearest")
    nir = refl.sel(wavelength=BAND_ALIASES["NIR"], method="nearest")
    green = clamp_reflectance(green)
    nir = clamp_reflectance(nir)
    return _normalized_difference(green, nir)


def _scenes_are_aligned(pre: xr.Dataset, post: xr.Dataset) -> bool:
    """True when ``pre`` and ``post`` already share spatial dims and y/x coords.

    Two scenes are considered aligned when they have the same ``y`` and ``x``
    sizes AND, if both expose projected y/x coordinate arrays, those arrays are
    element-wise equal.  Synthetic test datasets that only carry integer pixel
    indices on y/x still satisfy the size check and are treated as aligned —
    the reprojection path is reserved for real ortho_sr scenes whose UTM grids
    differ.
    """
    if pre.sizes.get("y") != post.sizes.get("y"):
        return False
    if pre.sizes.get("x") != post.sizes.get("x"):
        return False
    if "x" in pre.coords and "x" in post.coords:
        if not np.array_equal(pre.coords["x"].values, post.coords["x"].values):
            return False
    if "y" in pre.coords and "y" in post.coords:
        if not np.array_equal(pre.coords["y"].values, post.coords["y"].values):
            return False
    return True


def dnbr(
    pre: xr.Dataset,
    post: xr.Dataset,
    *,
    auto_align: bool = True,
) -> xr.DataArray:
    """Compute differenced Normalized Burn Ratio (dNBR).

    dNBR = NBR(pre) - NBR(post)

    Positive values indicate burn severity (pre-fire vegetation vs post-fire
    bare/charred ground).

    Args:
        pre: Pre-fire xarray Dataset with ``wavelength`` coordinate and
            ``reflectance`` (or ``surface_reflectance`` / ``toa_radiance``)
            variable of shape (wavelength, y, x).
        post: Post-fire xarray Dataset with the same structure.
        auto_align: When ``True`` (default), if the two scenes do not share an
            identical spatial grid, transparently call
            :func:`tanager.io.reproject_to_common_grid` to put them on a common
            grid before differencing.  When ``False``, mismatched grids raise
            ``ValueError`` (legacy behaviour).

    Returns:
        DataArray of dNBR values with spatial dimensions (y, x).

    Raises:
        ValueError: If the spatial dimensions of ``pre`` and ``post`` differ
            and ``auto_align=False``, or if ``auto_align=True`` and the scenes
            do not overlap enough for co-registration to succeed.
    """
    if not _scenes_are_aligned(pre, post):
        if not auto_align:
            raise ValueError(
                f"Spatial dimensions of pre and post datasets must match: "
                f"pre is ({pre.sizes.get('y')}, {pre.sizes.get('x')}), "
                f"post is ({post.sizes.get('y')}, {post.sizes.get('x')}). "
                f"Pass auto_align=True (default) or call "
                f"tanager.io.reproject_to_common_grid first."
            )
        # Defer the import to avoid a circular module load at import time
        # (io.py does not import spectral.py, but spectral.py importing io.py
        # at module scope would still work; keep the lazy import to make the
        # alignment dependency obvious to readers).
        from tanager.io import get_spatial_info, reproject_to_common_grid

        pre_info = get_spatial_info(pre)
        post_info = get_spatial_info(post)
        logger.warning(
            "dnbr: pre/post scenes are not on the same grid (pre %s, post %s); "
            "auto-aligning via reproject_to_common_grid",
            pre_info["shape"],
            post_info["shape"],
        )
        pre, post = reproject_to_common_grid([pre, post])
        aligned_info = get_spatial_info(pre)
        logger.info(
            "dnbr: aligned grid shape=%s bounds=%s",
            aligned_info["shape"],
            aligned_info["bounds"],
        )

    return nbr(pre) - nbr(post)


# ---------------------------------------------------------------------------
# Continuum removal
# ---------------------------------------------------------------------------


def _continuum_removal_spectrum(reflectance: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Apply convex hull continuum removal to a single spectrum.

    Computes the *upper* convex hull of the (wavelength, reflectance) point
    set using Andrew's monotone-chain algorithm restricted to right turns,
    interpolates it back to every wavelength to form the continuum, and
    divides the spectrum by it. The full ``scipy.spatial.ConvexHull`` returns
    both upper and lower hull vertices and so produces a continuum that
    drops into absorption features rather than spanning over them; the
    monotone chain keeps only the upper boundary.

    Args:
        reflectance: 1-D array of reflectance values (unitless, typically 0–1).
        wavelengths: 1-D array of centre wavelengths (nm), same length as
            ``reflectance``.

    Returns:
        1-D array of continuum-removed reflectance values in [0, 1].
    """
    n = len(reflectance)
    if n < 3:
        # Cannot form a hull with fewer than 3 points; return ones.
        return np.ones_like(reflectance, dtype=np.float64)

    try:
        # Operate in wavelength-sorted space, then restore caller's order.
        order = np.argsort(wavelengths)
        wl_s = wavelengths[order].astype(np.float64)
        r_s = reflectance[order].astype(np.float64)

        upper: list[Tuple[float, float]] = []
        for i in range(n):
            while len(upper) >= 2:
                ox, oy = upper[-2]
                ax, ay = upper[-1]
                bx, by = wl_s[i], r_s[i]
                # Cross product of (a − o) and (b − o). For the upper hull we
                # keep only strict right turns (cross < 0); discard collinear
                # and left-turning vertices that would dip into the spectrum.
                cross = (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)
                if cross >= 0.0:
                    upper.pop()
                else:
                    break
            upper.append((wl_s[i], r_s[i]))

        hull_x = np.array([p[0] for p in upper], dtype=np.float64)
        hull_y = np.array([p[1] for p in upper], dtype=np.float64)
        continuum_sorted = np.interp(wl_s, hull_x, hull_y)

        # Restore caller's wavelength order.
        inv = np.empty_like(order)
        inv[order] = np.arange(n)
        continuum = continuum_sorted[inv]
    except Exception:
        # If the algorithm fails (e.g., all-NaN or pathological spectrum),
        # use the maximum value as a flat continuum to avoid dividing by zero.
        max_val = float(np.max(reflectance))
        continuum = np.full_like(reflectance, max_val if max_val > 0 else 1.0, dtype=np.float64)

    # Avoid division by zero in the continuum
    continuum = np.where(continuum == 0, np.nan, continuum)
    result = reflectance.astype(np.float64) / continuum
    return np.minimum(result, 1.0)


def _continuum_removal_batched(
    reflectance: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """Vectorized upper-hull continuum removal across a batch of pixels.

    Runs Andrew's monotone-chain upper hull simultaneously over P pixels by
    treating each step of the chain as a numpy operation across the pixel
    axis. The wavelength axis is shared, so x-comparisons re-use a single
    sorted array; only the y-values vary per pixel. NaN comparisons evaluate
    False, so NaN-containing spectra simply skip pops and propagate NaN
    through interpolation — matching the per-pixel implementation's output.

    Args:
        reflectance: Array shaped ``(n, P)`` with n wavelength bands and P
            pixels (any spatial layout flattened by the caller).
        wavelengths: 1-D array of n centre wavelengths (nm). Need not be
            sorted; sorted internally.

    Returns:
        ``(n, P)`` continuum-removed reflectance, clipped to ``<= 1.0``,
        in the caller's original wavelength order.
    """
    n, P = reflectance.shape
    if n < 3 or P == 0:
        return np.ones((n, P), dtype=np.float64)

    R_orig = reflectance.astype(np.float64, copy=False)

    # Sort wavelengths ascending so the monotone chain is well-defined; we
    # restore the caller's order at the end.
    order = np.argsort(wavelengths)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n)
    wl = wavelengths[order].astype(np.float64)
    R = R_orig[order]  # (n, P)

    # stack[k, p] is the k-th hull-vertex index (into wl/R) for pixel p.
    # int32 keeps memory bounded (4 B × n × P) for typical Tanager scenes.
    stack = np.empty((n, P), dtype=np.int32)
    sizes = np.zeros(P, dtype=np.int64)
    p_arange = np.arange(P)

    for i in range(n):
        x_i = wl[i]
        y_i = R[i, :]

        # Pop until each pixel's top-two hull vertices form a strict right
        # turn with the new point, or the stack has fewer than two entries.
        while True:
            mask_can_pop = sizes >= 2
            if not mask_can_pop.any():
                break
            top_k = np.maximum(sizes - 1, 0)
            below_k = np.maximum(sizes - 2, 0)
            top_pos = stack[top_k, p_arange]
            below_pos = stack[below_k, p_arange]
            ox = wl[below_pos]
            oy = R[below_pos, p_arange]
            ax = wl[top_pos]
            ay = R[top_pos, p_arange]
            cross = (ax - ox) * (y_i - oy) - (ay - oy) * (x_i - ox)
            pop = mask_can_pop & (cross >= 0.0)
            if not pop.any():
                break
            sizes = np.where(pop, sizes - 1, sizes)

        stack[sizes, p_arange] = i
        sizes += 1

    # hull_mask[i, p] = True iff i is a hull vertex of pixel p.
    k_grid = np.arange(n)[:, None]
    valid = k_grid < sizes[None, :]
    hull_mask = np.zeros((n, P), dtype=bool)
    valid_positions = stack[valid]
    valid_p_idx = np.broadcast_to(p_arange, (n, P))[valid]
    hull_mask[valid_positions, valid_p_idx] = True

    # For every wavelength i we need the bracketing hull indices:
    # prev_hull[i, p] = largest hull index <= i (always 0 at i=0 since 0 is
    # pushed first and never popped).
    # next_hull[i, p] = smallest hull index >= i (always exists since n-1 is
    # the last push).
    i_grid = np.arange(n, dtype=np.int32)[:, None]
    prev_vals = np.where(hull_mask, i_grid, 0)
    prev_hull = np.maximum.accumulate(prev_vals, axis=0)

    next_vals = np.where(hull_mask, i_grid, n - 1)
    next_hull = np.minimum.accumulate(next_vals[::-1], axis=0)[::-1]

    # Linearly interpolate the upper hull onto the full wavelength grid.
    prev_hull_idx = prev_hull.astype(np.int64, copy=False)
    next_hull_idx = next_hull.astype(np.int64, copy=False)
    xp = wl[prev_hull]
    xn = wl[next_hull]
    yp = np.take_along_axis(R, prev_hull_idx, axis=0)
    yn = np.take_along_axis(R, next_hull_idx, axis=0)

    denom = xn - xp
    denom_safe = np.where(denom == 0, 1.0, denom)
    t = (wl[:, None] - xp) / denom_safe
    t = np.where(denom == 0, 0.0, t)
    continuum_sorted = yp + t * (yn - yp)

    # Restore the caller's wavelength order.
    continuum = continuum_sorted[inv_order]

    # Avoid division by zero (matches single-spectrum behaviour).
    continuum_safe = np.where(continuum == 0, np.nan, continuum)
    result = R_orig / continuum_safe
    return np.minimum(result, 1.0)


def continuum_removal(
    dataset: Union[xr.Dataset, xr.DataArray],
    wavelength_range: Optional[Tuple[float, float]] = None,
) -> xr.DataArray:
    """Apply convex hull continuum removal to every pixel spectrum.

    Continuum removal normalises each pixel spectrum by dividing it by a
    convex-hull continuum fitted to the (wavelength, reflectance) curve.  The
    output represents relative absorption depth rather than absolute reflectance
    and is useful for comparing spectral features across illumination conditions.

    Args:
        dataset: xarray Dataset with a ``wavelength`` coordinate (nm) and a
            ``reflectance`` variable with dimensions (wavelength, y, x).
        wavelength_range: Optional ``(min_nm, max_nm)`` tuple.  When provided,
            continuum removal is applied only to bands within the closed
            interval ``[min_nm, max_nm]``; bands outside the range are not
            included in the output.  When ``None`` (default), the full spectrum
            is used.

    Returns:
        DataArray of continuum-removed reflectance values in [0, 1] with
        dimensions (wavelength, y, x).  The wavelength coordinate reflects the
        selected range if ``wavelength_range`` was specified.
    """
    refl = _reflectance_var(dataset) if isinstance(dataset, xr.Dataset) else dataset

    if wavelength_range is not None:
        min_wl, max_wl = wavelength_range
        wl_coord = refl.coords["wavelength"]
        mask = (wl_coord >= min_wl) & (wl_coord <= max_wl)
        refl = refl.sel(wavelength=mask)

    wavelengths = refl.coords["wavelength"].values.astype(np.float64)

    # Move the wavelength axis to the front, flatten the remaining spatial
    # axes into a single pixel dimension, run the vectorized hull, then
    # restore the original layout. This converts the per-pixel Python loop
    # into a handful of numpy ops per outer-step over (n × P) arrays.
    refl_t = refl.transpose("wavelength", ...)
    spatial_dims = [d for d in refl_t.dims if d != "wavelength"]
    spatial_shape = tuple(refl_t.sizes[d] for d in spatial_dims)
    arr = refl_t.values
    n_wl = arr.shape[0]
    flat = arr.reshape(n_wl, -1)
    P = flat.shape[1]

    # Process pixels in chunks: the batched core allocates ~15 (n_wl, P_chunk)
    # arrays during the monotone-chain sweep, so a 426-band, 564K-pixel scene
    # needs ~28 GB without chunking. 25K pixels keeps each worker's peak
    # working set near ~1.3 GB and gives enough chunks (~12-25 per scene) to
    # saturate a multi-core CPU. Chunks are independent (the hull math is
    # per-pixel), so we parallelise across them with joblib for full-scene
    # workloads and stay sequential for small inputs to avoid worker overhead.
    P_chunk = 25_000
    chunks = [(start, min(start + P_chunk, P)) for start in range(0, P, P_chunk)]

    def _process(start: int, stop: int) -> np.ndarray:
        return _continuum_removal_batched(flat[:, start:stop], wavelengths)

    if len(chunks) <= 1:
        cr_flat = _process(*chunks[0]) if chunks else np.empty_like(flat, dtype=np.float64)
    else:
        from joblib import Parallel, delayed

        cr_chunks = Parallel(n_jobs=-1, backend="loky")(
            delayed(_process)(start, stop) for start, stop in chunks
        )
        cr_flat = np.concatenate(cr_chunks, axis=1)
    cr_arr = cr_flat.reshape((n_wl,) + spatial_shape)

    coords = {d: refl_t.coords[d] for d in refl_t.coords if d in refl_t.dims}
    coords["wavelength"] = refl_t.coords["wavelength"]
    result = xr.DataArray(
        cr_arr,
        dims=("wavelength",) + tuple(spatial_dims),
        coords=coords,
        name=refl.name,
    )
    return result
