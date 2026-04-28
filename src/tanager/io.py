"""Scene I/O for Planet Tanager-1 HDF5 hyperspectral data.

Public API
----------
load_scene(filepath, wavelength_range=None) -> xarray.Dataset
    Load a Tanager scene, optionally subsetting by wavelength range.
    Auto-detects swath vs. ortho product layouts.

load_ortho_scene(filepath, wavelength_range=None) -> xarray.Dataset
    Load an ortho-rectified Tanager surface-reflectance product directly
    via h5py. Used by ``load_scene`` when ``hypercoast.read_tanager`` cannot
    handle the file (ortho products lack lat/lon arrays).

get_spatial_info(dataset) -> dict
    Extract CRS, bounds, resolution, and shape from a loaded Dataset.

reproject_to_common_grid(datasets, target_bounds=None, target_resolution=30.0,
                         min_overlap_fraction=0.10, resampling="nearest")
    Reproject a set of Tanager scenes onto a single shared grid so that
    multi-temporal operations (dNBR, change detection, trajectories) can
    subtract / index aligned arrays.

Notes
-----
Loading a wavelength subset (``wavelength_range``) on swath products reads
the full scene from disk first then slices in memory. HyperCoast's ``bands``
parameter accepts integer indices, not wavelength values, so a two-pass
approach (read metadata → compute indices → slice) would still require
opening the file twice. The ortho path slices contiguous band ranges
directly via h5py and avoids reading the full cube when a narrow range is
requested.
"""

import logging
import re
from os import PathLike
from typing import Optional, Union

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

# Type alias matching common filepath arguments.
FilePath = Union[str, PathLike]

# HDF5 group/dataset paths used by ortho SR products.
_ORTHO_GRID_GROUP = "HDFEOS/GRIDS/HYP"
_ORTHO_SR_DATASET = "HDFEOS/GRIDS/HYP/Data Fields/surface_reflectance"
_ORTHO_STRUCT_METADATA = "HDFEOS INFORMATION/StructMetadata.0"


def load_scene(
    filepath: FilePath,
    wavelength_range: Optional[tuple[float, float]] = None,
) -> xr.Dataset:
    """Load a Planet Tanager-1 HDF5 scene as an xarray Dataset.

    Auto-detects the HDF5 layout. Swath/basic products are read via
    ``hypercoast.read_tanager``; ortho-rectified surface-reflectance products
    are read via :func:`load_ortho_scene` (HyperCoast cannot read those because
    they lack the ``Latitude``/``Longitude`` datasets it requires).

    When ``wavelength_range`` is supplied the swath path reads the full scene
    into memory and then slices; the ortho path slices contiguous band ranges
    directly on disk (see module docstring).

    Args:
        filepath: Path to a Tanager ``.h5`` file (local or HTTPS URL).
        wavelength_range: Optional ``(min_wl, max_wl)`` tuple in nanometres
            (nm). When supplied only bands whose centre wavelength falls within
            ``[min_wl, max_wl]`` (inclusive) are retained.

    Returns:
        xr.Dataset: Dataset with dims ``(wavelength, y, x)``.  The wavelength
        coordinate is in nm.  Data variables depend on the product type
        (``toa_radiance`` for radiance products, ``surface_reflectance`` plus a
        ``toa_radiance`` alias for SR products).

    Raises:
        ValueError: If the file cannot be read, is not a valid Tanager HDF5
            file, or ``wavelength_range`` does not select any bands.
    """
    import hypercoast  # heavy dep — imported here to keep module fast at import time

    path_str = str(filepath)
    log.debug("Loading Tanager scene from %s", path_str)

    try:
        ds: xr.Dataset = hypercoast.read_tanager(filepath)
    except (OSError, KeyError) as exc:
        raise ValueError(
            f"Cannot read Tanager HDF5 file {path_str!r}: {exc}"
        ) from exc
    except ValueError as exc:
        # HyperCoast raises ValueError on ortho products because they have no
        # Latitude/Longitude datasets. Fall back to the direct ortho reader.
        if _looks_like_missing_latlon_error(exc):
            log.debug(
                "HyperCoast could not locate lat/lon in %s; falling back to "
                "load_ortho_scene (likely an ortho-rectified product)",
                path_str,
            )
            try:
                return load_ortho_scene(filepath, wavelength_range=wavelength_range)
            except ValueError as ortho_exc:
                raise ValueError(
                    f"Cannot read Tanager HDF5 file {path_str!r}: HyperCoast "
                    f"swath read failed ({exc}) and ortho fallback failed "
                    f"({ortho_exc})"
                ) from ortho_exc
        raise ValueError(
            f"Invalid or corrupted Tanager HDF5 file {path_str!r}: {exc}"
        ) from exc

    # HyperCoast guarantees (wavelength, y, x); assert so callers catch drift.
    expected = ("wavelength", "y", "x")
    data_var = ds.attrs.get("data_var", next(iter(ds.data_vars)))
    actual = tuple(ds[data_var].dims)
    if actual != expected:
        raise ValueError(
            f"Unexpected dataset dimensions from HyperCoast: expected "
            f"{expected}, got {actual} in {path_str!r}"
        )

    n_bands = ds.sizes["wavelength"]
    log.debug("Loaded scene: %d bands, shape y=%d x=%d", n_bands, ds.sizes["y"], ds.sizes["x"])

    if wavelength_range is not None:
        min_wl, max_wl = wavelength_range
        wl = ds["wavelength"].values
        mask = (wl >= min_wl) & (wl <= max_wl)
        n_selected = int(mask.sum())
        if n_selected == 0:
            raise ValueError(
                f"wavelength_range ({min_wl}, {max_wl}) nm selects no bands from "
                f"scene with wavelengths [{wl.min():.1f}, {wl.max():.1f}] nm in "
                f"{path_str!r}"
            )
        ds = ds.isel(wavelength=np.where(mask)[0])
        log.debug(
            "Subset to %d bands in range [%.1f, %.1f] nm",
            n_selected,
            min_wl,
            max_wl,
        )

    return ds


def _looks_like_missing_latlon_error(exc: BaseException) -> bool:
    """Detect HyperCoast's 'no lat/lon' error so we can fall back to ortho path.

    HyperCoast's ``read_tanager`` raises a generic ``ValueError`` when it
    cannot locate Latitude/Longitude datasets — exactly the case for ortho
    grid products. Match on stable substrings rather than exact text so we
    survive minor wording changes upstream.
    """
    msg = str(exc).lower()
    return ("latitude" in msg or "lat/lon" in msg) and (
        "could not" in msg or "not found" in msg or "locate" in msg
    )


_STRUCT_METADATA_PATTERNS = {
    "x_dim": re.compile(r"\bXDim\s*=\s*(\d+)"),
    "y_dim": re.compile(r"\bYDim\s*=\s*(\d+)"),
    "upper_left": re.compile(
        r"UpperLeftPointMtrs\s*=\s*\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)"
    ),
    "lower_right": re.compile(
        r"LowerRightMtrs\s*=\s*\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)"
    ),
    "zone_code": re.compile(r"\bZoneCode\s*=\s*(-?\d+)"),
    "grid_origin": re.compile(r"\bGridOrigin\s*=\s*([A-Za-z0-9_]+)"),
    "projection": re.compile(r"\bProjection\s*=\s*([A-Za-z0-9_]+)"),
}


def _parse_struct_metadata(text: str) -> dict:
    """Parse the relevant fields out of HDF-EOS5 ``StructMetadata.0`` text.

    Only the fields needed to construct UTM x/y coords are extracted. The
    metadata blob is INI-like but the official HDF-EOS5 parsers are heavyweight,
    so a small regex pass keeps this module free of extra dependencies.
    """
    result: dict = {}
    for key in ("x_dim", "y_dim", "zone_code"):
        m = _STRUCT_METADATA_PATTERNS[key].search(text)
        if m:
            result[key] = int(m.group(1))

    for key in ("upper_left", "lower_right"):
        m = _STRUCT_METADATA_PATTERNS[key].search(text)
        if m:
            result[key] = (float(m.group(1)), float(m.group(2)))

    for key in ("grid_origin", "projection"):
        m = _STRUCT_METADATA_PATTERNS[key].search(text)
        if m:
            result[key] = m.group(1)

    required = {"x_dim", "y_dim", "upper_left", "lower_right"}
    missing = required - result.keys()
    if missing:
        raise ValueError(
            f"StructMetadata.0 is missing required fields: {sorted(missing)}"
        )
    return result


def load_ortho_scene(
    filepath: FilePath,
    wavelength_range: Optional[tuple[float, float]] = None,
) -> xr.Dataset:
    """Load an ortho-rectified Tanager surface-reflectance product.

    Reads ``HDFEOS/GRIDS/HYP/Data Fields/surface_reflectance`` directly with
    h5py and constructs an xarray Dataset matching the schema expected by the
    rest of the pipeline. Used as the fallback for files where
    ``hypercoast.read_tanager`` raises ``ValueError`` because the lat/lon
    datasets it requires are absent (true for all ortho grid products).

    Args:
        filepath: Path to a Tanager ``ortho_sr`` ``.h5`` file.
        wavelength_range: Optional ``(min_wl, max_wl)`` tuple in nanometres.
            When supplied, only the contiguous band slice covering the range
            is read from disk, keeping memory bounded for narrow subsets.

    Returns:
        xr.Dataset with:

        - dims ``(wavelength, y, x)``
        - data variables ``surface_reflectance`` and ``toa_radiance`` (alias)
        - coords: ``wavelength`` (nm), ``y`` and ``x`` as UTM-metre pixel
          centres, plus ``fwhm`` (nm) and ``good_wavelengths`` (uint8) along
          the wavelength dimension
        - attrs: ``crs`` (``"EPSG:<code>"``), ``epsg``, ``data_var``,
          ``product``, ``source``, plus the ``strip_id`` and ``created_at``
          metadata copied from the HDF5 group

        Fill-value pixels (``-9999``) are converted to NaN.

    Raises:
        ValueError: If the file is missing required ortho-product structure
            (surface_reflectance dataset, StructMetadata.0, wavelengths
            attribute) or ``wavelength_range`` selects zero bands.
    """
    import h5py  # heavy dep — imported lazily

    path_str = str(filepath)
    log.debug("Loading ortho Tanager scene from %s", path_str)

    try:
        h5 = h5py.File(path_str, "r")
    except OSError as exc:
        raise ValueError(
            f"Cannot read Tanager HDF5 file {path_str!r}: {exc}"
        ) from exc

    with h5:
        if _ORTHO_SR_DATASET not in h5:
            raise ValueError(
                f"File {path_str!r} is not a Tanager ortho SR product "
                f"(missing dataset {_ORTHO_SR_DATASET!r})"
            )
        sr = h5[_ORTHO_SR_DATASET]
        sr_attrs = dict(sr.attrs)

        if "wavelengths" not in sr_attrs:
            raise ValueError(
                f"surface_reflectance dataset in {path_str!r} is missing the "
                f"'wavelengths' attribute"
            )

        wavelengths = np.asarray(sr_attrs["wavelengths"], dtype=np.float64)
        fwhm = (
            np.asarray(sr_attrs["fwhm"], dtype=np.float64)
            if "fwhm" in sr_attrs
            else None
        )
        good_wavelengths = (
            np.asarray(sr_attrs["good_wavelengths"], dtype=np.uint8)
            if "good_wavelengths" in sr_attrs
            else None
        )
        fill_value = (
            float(sr_attrs["_FillValue"]) if "_FillValue" in sr_attrs else -9999.0
        )

        n_bands = wavelengths.shape[0]
        if sr.shape[0] != n_bands:
            raise ValueError(
                f"surface_reflectance shape mismatch in {path_str!r}: dataset has "
                f"{sr.shape[0]} bands but 'wavelengths' attribute has {n_bands}"
            )

        # Decide which contiguous band slice to read.
        if wavelength_range is not None:
            min_wl, max_wl = wavelength_range
            mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
            n_selected = int(mask.sum())
            if n_selected == 0:
                raise ValueError(
                    f"wavelength_range ({min_wl}, {max_wl}) nm selects no bands "
                    f"from scene with wavelengths "
                    f"[{wavelengths.min():.1f}, {wavelengths.max():.1f}] nm in "
                    f"{path_str!r}"
                )
            indices = np.where(mask)[0]
            band_start = int(indices[0])
            band_stop = int(indices[-1]) + 1  # half-open for slicing
            band_slice = slice(band_start, band_stop)
            log.debug(
                "Subset to %d bands in range [%.1f, %.1f] nm (h5 slice [%d:%d])",
                n_selected,
                min_wl,
                max_wl,
                band_start,
                band_stop,
            )
        else:
            band_slice = slice(0, n_bands)

        # Read the cube. h5py returns a numpy array; cast fills to NaN.
        cube = sr[band_slice, :, :].astype(np.float32, copy=False)
        wavelengths_sel = wavelengths[band_slice]
        fwhm_sel = fwhm[band_slice] if fwhm is not None else None
        good_sel = (
            good_wavelengths[band_slice] if good_wavelengths is not None else None
        )

        # Mask fill values to NaN.
        cube = np.where(cube == fill_value, np.nan, cube)

        # Parse grid metadata.
        if _ORTHO_STRUCT_METADATA not in h5:
            raise ValueError(
                f"Ortho file {path_str!r} is missing {_ORTHO_STRUCT_METADATA!r}"
            )
        sm_raw = h5[_ORTHO_STRUCT_METADATA][()]
        sm_text = sm_raw.decode("utf-8") if isinstance(sm_raw, bytes) else str(sm_raw)
        grid = _parse_struct_metadata(sm_text)

        # EPSG: prefer explicit epsg_code on the HYP grid group; fall back to
        # UTM zone code from StructMetadata if present.
        hyp_attrs = (
            dict(h5[_ORTHO_GRID_GROUP].attrs) if _ORTHO_GRID_GROUP in h5 else {}
        )

    # Coords -----------------------------------------------------------------
    x_dim = grid["x_dim"]
    y_dim = grid["y_dim"]
    if cube.shape[1] != y_dim or cube.shape[2] != x_dim:
        raise ValueError(
            f"Cube shape {cube.shape[1:]} does not match StructMetadata grid "
            f"({y_dim}, {x_dim}) in {path_str!r}"
        )
    ulx, uly = grid["upper_left"]
    lrx, lry = grid["lower_right"]
    x_res = (lrx - ulx) / x_dim
    y_res = (uly - lry) / y_dim  # positive number; y descends north→south
    x_coords = ulx + (np.arange(x_dim) + 0.5) * x_res
    y_coords = uly - (np.arange(y_dim) + 0.5) * y_res

    # CRS --------------------------------------------------------------------
    epsg: Optional[int] = None
    if "epsg_code" in hyp_attrs:
        epsg = int(np.asarray(hyp_attrs["epsg_code"]).item())
    elif (
        grid.get("projection") == "HE5_GCTP_UTM"
        and "zone_code" in grid
        and grid["zone_code"] > 0
    ):
        # Northern-hemisphere UTM EPSG = 32600 + zone (Tanager fire scenes are
        # all northern hemisphere; if a southern scene is encountered the
        # explicit epsg_code attribute should be present).
        epsg = 32600 + grid["zone_code"]
    crs = f"EPSG:{epsg}" if epsg is not None else None

    # Build dataset ----------------------------------------------------------
    coords: dict = {
        "wavelength": wavelengths_sel,
        "y": y_coords,
        "x": x_coords,
    }
    if fwhm_sel is not None:
        coords["fwhm"] = (("wavelength",), fwhm_sel)
    if good_sel is not None:
        coords["good_wavelengths"] = (("wavelength",), good_sel)

    attrs: dict = {
        "source": "Planet Tanager HDF5",
        "product": "ortho_sr",
        "data_var": "surface_reflectance",
    }
    if crs is not None:
        attrs["crs"] = crs
    if epsg is not None:
        attrs["epsg"] = epsg
    for k in ("strip_id", "created_at"):
        if k in hyp_attrs:
            v = hyp_attrs[k]
            attrs[k] = v.decode() if isinstance(v, bytes) else v

    sr_da = xr.DataArray(
        cube,
        dims=("wavelength", "y", "x"),
        coords=coords,
        name="surface_reflectance",
    )

    ds = xr.Dataset(
        {
            "surface_reflectance": sr_da,
            # Keep the toa_radiance alias to match the swath-path schema so
            # downstream code that probes either name keeps working.
            "toa_radiance": sr_da,
        },
        coords=coords,
        attrs=attrs,
    )
    return ds


def get_spatial_info(dataset: xr.Dataset) -> dict:
    """Extract spatial metadata from a loaded Tanager Dataset.

    Inspects ``dataset.attrs`` and coordinate arrays to assemble a dict
    describing the coordinate reference system, spatial extent, pixel spacing,
    and raster dimensions.

    CRS lookup order:
    1. ``dataset.attrs["crs"]``
    2. ``dataset.attrs["spatial_ref"]``
    3. ``dataset.attrs["epsg"]`` (wrapped as ``EPSG:<value>``)
    4. A ``"spatial_ref"`` coordinate variable if present
    5. ``None`` if none of the above are found

    Bounds and resolution are derived from the ``y`` and ``x`` dimension
    coordinates when present, or from the ``latitude`` / ``longitude``
    ancillary coordinates as a fallback (bearing in mind that lat/lon may be
    curvilinear, so the fallback bounds are approximate).

    Args:
        dataset: xarray.Dataset returned by :func:`load_scene`.

    Returns:
        dict with the following keys:

        - ``crs`` (str or None): CRS string (WKT, PROJ, or ``"EPSG:XXXX"``),
          or ``None`` if not determinable.
        - ``bounds`` (tuple[float, float, float, float]): Spatial extent as
          ``(x_min, y_min, x_max, y_max)`` in dataset coordinates.
        - ``resolution`` (tuple[float, float] or None): Pixel spacing
          ``(x_res, y_res)`` in dataset coordinate units (absolute values).
          ``None`` if fewer than two pixels exist along an axis.
        - ``shape`` (tuple[int, int]): Raster dimensions as ``(n_rows, n_cols)``
          i.e. ``(y_size, x_size)``.
    """
    # ------------------------------------------------------------------ CRS
    attrs = dataset.attrs
    crs: Optional[str] = None

    if "crs" in attrs:
        crs = str(attrs["crs"])
    elif "spatial_ref" in attrs:
        crs = str(attrs["spatial_ref"])
    elif "epsg" in attrs:
        crs = f"EPSG:{attrs['epsg']}"
    elif "spatial_ref" in dataset.coords:
        # rasterio writes CRS as a scalar coordinate named "spatial_ref"
        crs = str(dataset.coords["spatial_ref"].item())

    # ------------------------------------------------------------- Bounds / resolution
    n_rows = dataset.sizes["y"]
    n_cols = dataset.sizes["x"]
    shape = (n_rows, n_cols)

    if "x" in dataset.coords and "y" in dataset.coords:
        x_vals = dataset.coords["x"].values
        y_vals = dataset.coords["y"].values
        x_min = float(x_vals.min())
        x_max = float(x_vals.max())
        y_min = float(y_vals.min())
        y_max = float(y_vals.max())

        x_res: Optional[float] = (
            float(abs(np.diff(x_vals).mean())) if len(x_vals) >= 2 else None
        )
        y_res: Optional[float] = (
            float(abs(np.diff(y_vals).mean())) if len(y_vals) >= 2 else None
        )
        resolution: Optional[tuple] = (
            (x_res, y_res) if (x_res is not None and y_res is not None) else None
        )
    elif "longitude" in dataset.coords and "latitude" in dataset.coords:
        # Curvilinear fallback — bounds are approximate.
        lon_vals = dataset.coords["longitude"].values
        lat_vals = dataset.coords["latitude"].values
        x_min = float(np.nanmin(lon_vals))
        x_max = float(np.nanmax(lon_vals))
        y_min = float(np.nanmin(lat_vals))
        y_max = float(np.nanmax(lat_vals))
        resolution = None
        log.debug(
            "get_spatial_info: no projected x/y coords; bounds derived from "
            "curvilinear latitude/longitude (approximate)"
        )
    else:
        x_min = x_max = y_min = y_max = float("nan")
        resolution = None
        log.warning(
            "get_spatial_info: no spatial coordinate found; bounds set to NaN"
        )

    bounds = (x_min, y_min, x_max, y_max)

    return {
        "crs": crs,
        "bounds": bounds,
        "resolution": resolution,
        "shape": shape,
    }


# ---------------------------------------------------------------------------
# Multi-temporal alignment
# ---------------------------------------------------------------------------

# Tanager-1 ortho_sr products land on a 30 m UTM grid. We default to that so
# resampling onto the common grid is a near no-op when the target CRS already
# matches the source CRS.
_DEFAULT_TARGET_RESOLUTION_M = 30.0

# Below this fractional overlap we refuse to reproject — the resulting common
# grid would contain mostly NaN and dNBR-style operations would be meaningless.
_DEFAULT_MIN_OVERLAP_FRACTION = 0.10


def _resampling_from_str(name: str):
    """Translate a resampling-method string to a ``rasterio.enums.Resampling`` value.

    A small allow-list mirrors the methods that make sense for reflectance:
    nearest preserves pixel values exactly (recommended), bilinear / cubic are
    available for callers that explicitly want continuous interpolation.
    """
    from rasterio.enums import Resampling

    table = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }
    if name not in table:
        raise ValueError(
            f"Unsupported resampling method {name!r}; expected one of {sorted(table)}"
        )
    return table[name]


def _intersect_bounds(
    bounds_list: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """Return the geometric intersection (xmin, ymin, xmax, ymax) of N rectangles.

    If the rectangles do not overlap, the returned ``(xmin, ymin, xmax, ymax)``
    has ``xmin >= xmax`` or ``ymin >= ymax`` (zero/negative width or height);
    callers must check.
    """
    xmin = max(b[0] for b in bounds_list)
    ymin = max(b[1] for b in bounds_list)
    xmax = min(b[2] for b in bounds_list)
    ymax = min(b[3] for b in bounds_list)
    return xmin, ymin, xmax, ymax


def _bounds_area(bounds: tuple[float, float, float, float]) -> float:
    """Area of an (xmin, ymin, xmax, ymax) rectangle, clamped to >= 0."""
    xmin, ymin, xmax, ymax = bounds
    return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)


def _pixel_edge_extent(info: dict) -> tuple[float, float, float, float]:
    """Convert pixel-centre bounds (from ``get_spatial_info``) to pixel-edge bounds.

    ``get_spatial_info`` reports ``bounds`` using the min/max of the pixel-centre
    coordinate arrays. For overlap/intersection logic we want the actual
    rectangular extent of the raster, which extends a half-pixel beyond the
    centres at every edge. When no resolution is available (curvilinear or
    single-pixel datasets) the centre bounds are returned unchanged.
    """
    xmin, ymin, xmax, ymax = info["bounds"]
    res = info["resolution"]
    if res is None:
        return (xmin, ymin, xmax, ymax)
    x_res, y_res = res
    return (
        xmin - x_res / 2.0,
        ymin - y_res / 2.0,
        xmax + x_res / 2.0,
        ymax + y_res / 2.0,
    )


def reproject_to_common_grid(
    datasets: list[xr.Dataset],
    target_bounds: Optional[tuple[float, float, float, float]] = None,
    target_resolution: float = _DEFAULT_TARGET_RESOLUTION_M,
    min_overlap_fraction: float = _DEFAULT_MIN_OVERLAP_FRACTION,
    resampling: str = "nearest",
) -> list[xr.Dataset]:
    """Reproject Tanager scenes onto a single shared grid.

    The Tanager-1 ortho_sr products have per-acquisition grids of differing
    extent, origin, and even pixel count.  Multi-temporal analysis (dNBR,
    burn-recovery trajectories) requires arrays with matching shape and
    coordinates so they can be subtracted directly.  This function computes
    the geographic intersection of all input scenes, builds a regular target
    grid over that intersection at ``target_resolution`` metres, and resamples
    each scene onto that grid using nearest-neighbour resampling by default.

    Args:
        datasets: List of two or more xarray Datasets produced by
            :func:`load_ortho_scene` (or any source that exposes ``crs`` /
            ``epsg`` in attrs and projected ``x``/``y`` coordinates).  All
            datasets must share the same CRS.
        target_bounds: Optional ``(xmin, ymin, xmax, ymax)`` in the common CRS
            specifying the exact rectangle to resample onto.  When omitted the
            intersection of input bounds is used.
        target_resolution: Pixel size in CRS units (metres for UTM).  Defaults
            to 30 m, the native ortho_sr spacing.
        min_overlap_fraction: Minimum fraction of the smallest input scene's
            area that must be covered by the intersection.  When the
            intersection is smaller than this fraction the function raises
            ``ValueError`` rather than producing a near-empty common grid.
            Ignored when ``target_bounds`` is supplied (the caller has
            explicitly opted into a chosen extent).
        resampling: Resampling method — ``"nearest"`` (default), ``"bilinear"``,
            or ``"cubic"``.  Nearest-neighbour preserves the original
            reflectance values without interpolation artefacts.

    Returns:
        List of new Datasets in the same order as the input.  Each dataset
        has identical ``y`` and ``x`` coordinate arrays (verified before
        return), the original ``wavelength`` coordinate / per-band coords
        (``fwhm``, ``good_wavelengths``) preserved, and ``attrs["crs"]``,
        ``attrs["epsg"]``, plus a new ``attrs["aligned_to"]`` recording the
        target grid extent.

    Raises:
        ValueError: If fewer than 2 datasets are passed, if any dataset is
            missing CRS metadata, if CRSs differ across inputs, if the
            intersection is empty, or if the overlap is below
            ``min_overlap_fraction`` of the smallest input scene.

    Notes:
        Uses ``rioxarray`` (a thin wrapper over ``rasterio.warp.reproject``)
        for the actual warp.  Datasets that already match the requested grid
        exactly are still warped — the cost is negligible because the warp
        loop becomes a memcpy when source and destination grids align.
    """
    if not isinstance(datasets, (list, tuple)) or len(datasets) < 2:
        raise ValueError(
            f"reproject_to_common_grid requires at least 2 datasets; got {len(datasets) if hasattr(datasets, '__len__') else type(datasets)}"
        )

    # ------------------------------------------------------------------ CRS
    spatial_infos = [get_spatial_info(ds) for ds in datasets]
    crs_values = [info["crs"] for info in spatial_infos]
    if any(c is None for c in crs_values):
        raise ValueError(
            "All input datasets must have a CRS in attrs (crs/epsg) or as a "
            "spatial_ref coord; received: " + repr(crs_values)
        )
    if len(set(crs_values)) > 1:
        raise ValueError(
            "All input datasets must share the same CRS; got " + repr(crs_values)
        )
    common_crs = crs_values[0]

    # ----------------------------------------------------------------- Bounds
    # Use pixel-edge extent (not pixel-centre min/max) so the intersection
    # geometry matches the raster footprint stamped in StructMetadata.
    bounds_list = [_pixel_edge_extent(info) for info in spatial_infos]

    if target_bounds is None:
        intersection = _intersect_bounds(bounds_list)
        x0, y0, x1, y1 = intersection
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                "Input scenes do not overlap: intersection bounds are "
                f"({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}). Per-scene bounds: "
                + ", ".join(repr(b) for b in bounds_list)
            )

        smallest_scene_area = min(_bounds_area(b) for b in bounds_list)
        overlap_area = _bounds_area(intersection)
        overlap_fraction = (
            overlap_area / smallest_scene_area if smallest_scene_area > 0 else 0.0
        )
        if overlap_fraction < min_overlap_fraction:
            raise ValueError(
                f"Scene overlap is {overlap_fraction:.1%} of the smallest input "
                f"scene, below the required {min_overlap_fraction:.0%} threshold. "
                f"Scenes are too far apart to align meaningfully. Intersection: "
                f"({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})"
            )
        log.info(
            "reproject_to_common_grid: %d scenes, intersection area %.0f m² (%.1f%% of smallest scene)",
            len(datasets),
            overlap_area,
            overlap_fraction * 100.0,
        )
        target = intersection
    else:
        x0, y0, x1, y1 = target_bounds
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f"target_bounds must satisfy xmin<xmax and ymin<ymax; got {target_bounds}"
            )
        target = target_bounds

    # ------------------------------------------------------------- Build grid
    if target_resolution <= 0:
        raise ValueError(f"target_resolution must be positive; got {target_resolution}")

    from rasterio.transform import from_origin

    x_min, y_min, x_max, y_max = target
    width = int(round((x_max - x_min) / target_resolution))
    height = int(round((y_max - y_min) / target_resolution))
    if width < 1 or height < 1:
        raise ValueError(
            f"Target grid would be empty: width={width}, height={height} from "
            f"bounds {target} at resolution {target_resolution}"
        )
    # Origin convention: y descends from y_max (north) to y_min (south).
    transform = from_origin(x_min, y_max, target_resolution, target_resolution)

    resampling_method = _resampling_from_str(resampling)

    # ------------------------------------------------------------- Reproject
    import rioxarray  # noqa: F401  # registers the .rio accessor

    aligned: list[xr.Dataset] = []
    for ds, info in zip(datasets, spatial_infos):
        # Pick the spatial DataArray to reproject. surface_reflectance and
        # toa_radiance share the same underlying buffer in load_ortho_scene
        # output; reproject one and reuse for both var aliases below.
        primary_name = ds.attrs.get("data_var") or (
            "surface_reflectance"
            if "surface_reflectance" in ds.data_vars
            else "toa_radiance"
            if "toa_radiance" in ds.data_vars
            else next(iter(ds.data_vars))
        )
        da = ds[primary_name]

        # Rebuild the DataArray with rio-friendly metadata. We strip wavelength
        # auxiliary coords that share the wavelength dim because rioxarray.reproject
        # only cares about the spatial dims; we re-attach them after the warp.
        wl_aux = {
            name: ds.coords[name]
            for name in ("fwhm", "good_wavelengths")
            if name in ds.coords
        }

        rio_da = da.drop_vars(list(wl_aux.keys()), errors="ignore")
        rio_da = rio_da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
        rio_da = rio_da.rio.write_crs(info["crs"], inplace=False)

        warped = rio_da.rio.reproject(
            dst_crs=common_crs,
            shape=(height, width),
            transform=transform,
            resampling=resampling_method,
            nodata=np.nan,
        )

        # rioxarray writes its own spatial_ref scalar coord; keep it but also
        # mirror crs/epsg in attrs so the rest of the pipeline (which reads
        # attrs["crs"]) keeps working without depending on rioxarray.
        new_attrs = dict(ds.attrs)
        new_attrs["crs"] = common_crs
        if "epsg" not in new_attrs:
            try:
                new_attrs["epsg"] = int(str(common_crs).split(":")[-1])
            except (ValueError, IndexError):
                pass
        new_attrs["aligned_to"] = {
            "bounds": tuple(float(v) for v in target),
            "resolution": float(target_resolution),
            "shape": (height, width),
            "crs": common_crs,
        }

        warped = warped.rename(primary_name)

        # Reassemble Dataset preserving the swath/ortho data_var aliases.
        data_vars: dict = {primary_name: warped}
        if "surface_reflectance" in ds.data_vars and "toa_radiance" in ds.data_vars:
            # Preserve the alias relationship from load_ortho_scene.
            other = "toa_radiance" if primary_name == "surface_reflectance" else "surface_reflectance"
            data_vars[other] = warped.rename(other)

        coords: dict = {
            "wavelength": ds.coords["wavelength"],
        }
        for name, coord in wl_aux.items():
            coords[name] = coord

        new_ds = xr.Dataset(data_vars, coords=coords, attrs=new_attrs)
        aligned.append(new_ds)

    # Sanity check: x/y coords must be identical across all aligned scenes.
    ref_x = aligned[0].coords["x"].values
    ref_y = aligned[0].coords["y"].values
    for i, ds in enumerate(aligned[1:], start=1):
        if not np.array_equal(ds.coords["x"].values, ref_x):
            raise RuntimeError(
                f"Internal error: aligned scene {i} x-coordinates do not match scene 0"
            )
        if not np.array_equal(ds.coords["y"].values, ref_y):
            raise RuntimeError(
                f"Internal error: aligned scene {i} y-coordinates do not match scene 0"
            )

    return aligned
