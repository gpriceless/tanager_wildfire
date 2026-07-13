"""Pixel-quality masking utilities for Tanager-1 hyperspectral imagery.

All mask functions return ``xr.DataArray`` booleans with the same spatial
dimensions as the input dataset (``y``, ``x``).  The convention throughout is
**True = valid / keep**, **False = masked / discard**.  This matches the
``xr.Dataset.where(mask)`` contract: only True pixels are preserved.

Example::

    mask = nodata_mask(ds) & cloud_mask(ds) & water_mask(ds)
    clean = apply_masks(ds, [nodata_mask(ds), cloud_mask(ds), water_mask(ds)])
"""

from __future__ import annotations

import logging
from functools import reduce
from os import PathLike
from typing import Optional, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

FilePath = Union[str, PathLike]


def nodata_mask(
    dataset: xr.Dataset,
    fill_value: Optional[float] = None,
) -> xr.DataArray:
    """Return a boolean mask where True indicates valid (non-fill, non-NaN) pixels.

    A pixel is valid only when **all** bands for that spatial location contain
    finite, non-NaN reflectance values.  When ``fill_value`` is provided, any
    pixel where any band equals that value is also marked invalid.

    Args:
        dataset: xarray Dataset with a ``wavelength`` dimension containing
            one or more reflectance-like data variables.
        fill_value: Sentinel value that represents missing data (e.g. -9999).
            When ``None``, only NaN / non-finite values are treated as nodata.

    Returns:
        2-D boolean DataArray with dims ``(y, x)``.  True = valid pixel.
    """
    arr = dataset.to_array(dim="variable")

    # Finite (includes NaN check) over all variables and wavelengths
    finite_mask: xr.DataArray = (
        np.isfinite(arr).all(dim="variable").all(dim="wavelength")
    )

    if fill_value is None:
        return finite_mask

    # Also mark any pixel where any band equals fill_value as invalid
    fill_mask: xr.DataArray = (arr != fill_value).all(dim="variable").all(dim="wavelength")
    return finite_mask & fill_mask


def cloud_mask(
    dataset: xr.Dataset,
    filepath: Optional[FilePath] = None,
) -> xr.DataArray:
    """Return a boolean mask where True indicates cloud-free pixels.

    Reads the ``beta_cirrus_mask`` and ``beta_cloud_mask`` quality flags from
    the source HDF5 file and OR-combines them: a pixel is flagged cloudy if
    either field flags it.  The lookup order is:

    1. ``'beta_cirrus_mask'`` and/or ``'beta_cloud_mask'`` data variables
       already present in *dataset* (each contributes; missing fields are
       treated as all-clear).
    2. HDF5 fields read from the source file, resolved via (in order):
       a. the *filepath* argument,
       b. ``dataset.encoding.get("source")``,
       c. a ``"filepath"`` attribute on *dataset*.
       Both SWATHS (``/HDFEOS/SWATHS/HYP/...``) and ortho GRIDS
       (``/HDFEOS/GRIDS/HYP/Data Fields/...``) layouts are searched.
    3. If neither field is found by either route, returns an all-True mask
       with a WARNING log so downstream steps are not silently broken.

    Args:
        dataset: xarray Dataset produced by ``tanager.io.load_scene``.
        filepath: Optional explicit path to the source ``.h5`` file.  Useful
            when the dataset was sliced or processed and has lost its encoding.

    Returns:
        2-D boolean DataArray with dims ``(y, x)``.  True = clear sky.
    """
    y_size = dataset.sizes.get("y", dataset.sizes.get("lat", None))
    x_size = dataset.sizes.get("x", dataset.sizes.get("lon", None))

    # --- 1. Check if already data variables ---------------------------------
    cirrus_var = dataset["beta_cirrus_mask"] if "beta_cirrus_mask" in dataset.data_vars else None
    cloud_var = dataset["beta_cloud_mask"] if "beta_cloud_mask" in dataset.data_vars else None
    if cirrus_var is not None or cloud_var is not None:
        # Convention: 0 = clear, 1 = cloud (follow Planet QA flag semantics).
        # OR-combine the two cloudy flags, then invert to clear-sky mask.
        cloudy = None
        if cirrus_var is not None:
            cloudy = cirrus_var != 0
        if cloud_var is not None:
            cloudy = (cloud_var != 0) if cloudy is None else (cloudy | (cloud_var != 0))
        logger.debug(
            "cloud_mask: using beta_cirrus_mask=%s, beta_cloud_mask=%s data variables from dataset",
            cirrus_var is not None,
            cloud_var is not None,
        )
        return (~cloudy).rename("cloud_mask")

    # --- 2. Try to read from HDF5 file --------------------------------------
    source_path = _resolve_source_path(dataset, filepath)

    if source_path is not None and y_size is not None and x_size is not None:
        cirrus_arr = _read_mask_field_from_hdf5(source_path, "beta_cirrus_mask")
        cloud_arr = _read_mask_field_from_hdf5(source_path, "beta_cloud_mask")

        cloudy_arr: Optional[np.ndarray] = None
        sources_used: list[str] = []
        for name, arr in (("beta_cirrus_mask", cirrus_arr), ("beta_cloud_mask", cloud_arr)):
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.shape != (y_size, x_size):
                logger.warning(
                    "cloud_mask: %s shape %s does not match dataset %s; ignoring",
                    name, arr.shape, (y_size, x_size),
                )
                continue
            flag = arr != 0
            cloudy_arr = flag if cloudy_arr is None else (cloudy_arr | flag)
            sources_used.append(name)

        if cloudy_arr is not None:
            da = xr.DataArray(
                ~cloudy_arr,
                dims=["y", "x"],
                name="cloud_mask",
            )
            logger.debug(
                "cloud_mask: loaded %s from HDF5 file %s",
                "+".join(sources_used), source_path,
            )
            return da

    # --- 3. Fallback: all-True (assume clear) --------------------------------
    logger.warning(
        "cloud_mask: neither beta_cirrus_mask nor beta_cloud_mask found in "
        "dataset or source HDF5. Returning all-True mask (all pixels assumed "
        "cloud-free). Pass filepath= or ensure the dataset retains "
        "encoding['source']."
    )
    y_size = dataset.sizes.get("y", dataset.sizes.get("lat", 1))
    x_size = dataset.sizes.get("x", dataset.sizes.get("lon", 1))
    return xr.DataArray(
        np.ones((y_size, x_size), dtype=bool),
        dims=["y", "x"],
        name="cloud_mask",
    )


def water_mask(
    dataset: xr.Dataset,
    threshold: float = 0.3,
) -> xr.DataArray:
    """Return a boolean mask where True indicates land pixels.

    Uses the Normalised Difference Water Index (NDWI) to distinguish water
    (NDWI > threshold) from land (NDWI <= threshold).  NDWI is computed via
    ``tanager.spectral.ndwi``.

    Args:
        dataset: xarray Dataset with a ``wavelength`` coordinate (nm)
            containing Green (~560 nm) and NIR (~860 nm) bands.
        threshold: NDWI threshold above which pixels are classified as water.
            Default 0.3 follows the McFeeters (1996) recommendation.

    Returns:
        2-D boolean DataArray with dims ``(y, x)``.  True = land, False = water.
    """
    from tanager.spectral import ndwi  # imported here to avoid circular import at module level

    ndwi_values: xr.DataArray = ndwi(dataset)
    land_mask: xr.DataArray = ndwi_values <= threshold
    land_mask.name = "water_mask"
    return land_mask


def apply_masks(
    dataset: xr.Dataset,
    mask_list: list[xr.DataArray],
) -> xr.Dataset:
    """Apply a logical AND of all masks to the dataset, setting masked pixels to NaN.

    Args:
        dataset: xarray Dataset to mask.
        mask_list: List of boolean DataArrays (True = valid).  Must not be
            empty.  All masks must be broadcastable to the dataset's spatial
            dimensions.

    Returns:
        A new Dataset where pixels that fail any mask are set to NaN.
        The spatial structure and coordinates of the input are preserved.

    Raises:
        ValueError: If ``mask_list`` is empty.
    """
    if not mask_list:
        raise ValueError("mask_list must contain at least one mask.")

    combined: xr.DataArray = reduce(lambda a, b: a & b, mask_list)
    return dataset.where(combined)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_source_path(
    dataset: xr.Dataset,
    filepath: Optional[FilePath],
) -> Optional[FilePath]:
    """Return the best available source file path for the dataset, or None."""
    if filepath is not None:
        return filepath

    # Try dataset.encoding["source"] — set by xarray when opened with open_dataset
    enc_source = dataset.encoding.get("source")
    if enc_source:
        return enc_source

    # Try a "filepath" attribute attached by load_scene
    attr_path = dataset.attrs.get("filepath")
    if attr_path:
        return attr_path

    return None


def _candidate_mask_paths(field_name: str) -> list[str]:
    """Return HDF5 paths to search for a Planet QA mask field.

    Covers both SWATHS-based products (legacy / non-ortho) and GRIDS-based
    ortho SR products, plus a couple of flat fallbacks.
    """
    return [
        f"/HDFEOS/GRIDS/HYP/Data Fields/{field_name}",
        f"/HDFEOS/SWATHS/HYP/Data Fields/{field_name}",
        f"/HDFEOS/SWATHS/HYP/Metadata/{field_name}",
        f"/Metadata/{field_name}",
        f"/{field_name}",
    ]


def _read_mask_field_from_hdf5(
    filepath: FilePath,
    field_name: str,
) -> Optional[np.ndarray]:
    """Read a named QA mask field from an HDF5 file, returning None if absent.

    Searches both SWATHS and GRIDS layouts so swath products and ortho SR
    products both resolve.
    """
    try:
        import h5py
    except ImportError:
        logger.warning("cloud_mask: h5py not installed; cannot read HDF5 metadata")
        return None

    try:
        with h5py.File(filepath, "r") as f:
            for path in _candidate_mask_paths(field_name):
                if path in f:
                    data = f[path][()]
                    logger.debug("cloud_mask: found %s at HDF5 path %s", field_name, path)
                    return np.asarray(data)
    except OSError as exc:
        logger.warning("cloud_mask: could not open HDF5 file %s: %s", filepath, exc)

    return None


def _read_beta_cirrus_from_hdf5(filepath: FilePath) -> Optional[np.ndarray]:
    """Backward-compat shim: read beta_cirrus_mask via _read_mask_field_from_hdf5."""
    return _read_mask_field_from_hdf5(filepath, "beta_cirrus_mask")
