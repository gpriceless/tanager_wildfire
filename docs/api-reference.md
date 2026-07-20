# Tanager API Reference

Public API summary for the `tanager` package, generated from the lazy-export
table in `src/tanager/__init__.py`. Every name listed here is importable
directly from the top-level package (`from tanager import ...`) — the
submodule is loaded on first access (PEP 562 lazy import), so `import tanager`
stays fast even though heavy dependencies (rasterio, hypercoast, sklearn,
matplotlib) live behind these functions.

**Scope.** This document covers all 63 public exports, organized by module
in the same order as `_LAZY_EXPORTS`. Every function has its signature and
parameter summary; the 12 functions judges are most likely to call directly
(`run_mesma`, `predict_severity`, `compute_lfmc_indices`, `predict_lfmc`,
`compute_trajectories`, `compare_sensors`, `plot_before_after`,
`plot_severity_summary`, `plot_temporal_trajectory`, `save_figure`,
`load_ortho_scene`, `list_fire_scenes`) get a runnable usage example.

For architecture and internal design rationale, see `docs/patterns.md`.
This document is a *reference*, not a tutorial — keep it in sync with
`__init__.py`'s `_LAZY_EXPORTS` table whenever a public symbol is added,
renamed, or removed.

---

## Table of Contents

1. [config](#config) — sensor constants, fire-scene catalog, data directory
2. [catalog](#catalog) — STAC browsing and scene download
3. [io](#io) — HDF5 scene loading
4. [spectral](#spectral) — band selection, spectral indices, continuum removal
5. [masks](#masks) — pixel-quality masking
6. [endmembers](#endmembers) — endmember library construction
7. [unmixing](#unmixing) — MESMA spectral unmixing
8. [severity](#severity) — burn severity products
9. [lfmc](#lfmc) — live fuel moisture content
10. [validation](#validation) — accuracy assessment, sensor comparison
11. [visualization](#visualization) — maps, figures, interactive display

---

## config

Stdlib-only module (no heavy deps) with sensor parameters, bad-band ranges,
the hardcoded fire-scene catalog, band-name aliases, and the data directory
path. These are constants, not functions — imported the same way as any
other lazy export.

### `SENSOR`

`SimpleNamespace` of Tanager-1 sensor parameters (dot-accessible):
`name`, `n_bands` (426), `wavelength_min_nm` (380), `wavelength_max_nm`
(2500), `spectral_resolution_nm` (5), `spatial_resolution_m` (30),
`swath_width_km` (18).

```python
import tanager
print(tanager.SENSOR.n_bands, tanager.SENSOR.spatial_resolution_m)
# 426 30
```

### `BAD_BAND_RANGES`

`list[tuple[int, int]]` — four `(low_nm, high_nm)` wavelength ranges to
exclude before spectral analysis (sensor edges + water-vapour absorption).
Default input to `spectral.mask_bad_bands`.

### `FIRE_SCENES`

`dict[str, dict]` — hardcoded catalog of known fire-collection scene IDs to
metadata (`datetime`, `phase`, `fire_complex`, `days_relative_to_ignition`,
`notes`, `bbox`).  `fire_complex` is `"palisades"`, `"hughes"`, or `None`
(unverified); `days_relative_to_ignition` is relative to that complex's
ignition date (see `FIRE_IGNITION_DATES`).
Use `catalog.list_fire_scenes()` for the live, authoritative source; this
dict is a static reference snapshot.

### `FIRE_IGNITION_DATES`

`dict[str, str]` — maps fire-complex name to its ignition date (ISO format).
Palisades/Eaton ignited 2025-01-07; Hughes ignited 2025-01-22.

### `BAND_ALIASES`

`dict[str, int]` — common band name → nominal centre wavelength (nm), e.g.
`{"NIR": 860, "RED": 660, "SWIR2": 2200, ...}`. Used by `spectral.py` index
functions for nearest-neighbour band lookup.

### `DATA_DIR`

`pathlib.Path` — root directory for raw fire-scene HDF5 files. Defaults to
`<project_root>/data/raw/fire/`; override with the `TANAGER_DATA_DIR`
environment variable.

---

## catalog

STAC catalog browsing and streaming download for Tanager fire scenes
(connects to Planet's static STAC catalog via `pystac`).

### `list_fire_scenes(start_date=None, end_date=None) -> list[pystac.Item]`

Return STAC items from the Tanager fire collection, optionally filtered by
acquisition datetime.

| Parameter | Type | Description |
|---|---|---|
| `start_date` | `str \| None` | Inclusive lower bound, ISO-8601 (e.g. `"2025-01-01"`). `None` = no lower bound. |
| `end_date` | `str \| None` | Inclusive upper bound, ISO-8601. `None` = no upper bound. |

Returns a list of `pystac.Item` objects exposing `.id`, `.datetime`,
`.bbox`, and `.assets`.

Raises `ConnectionError` if the catalog is unreachable, `ValueError` if a
date string cannot be parsed.

```python
import tanager

# All post-fire scenes over the LA fires (2025-01-01 onward)
scenes = tanager.list_fire_scenes(start_date="2025-01-01", end_date="2025-02-01")
for item in scenes:
    print(item.id, item.datetime, item.bbox)
```

### `get_scene_metadata(item: pystac.Item) -> dict`

Extract structured metadata (`scene_id`, `datetime`, `bbox`, `product_types`,
`file_size_mb`) from a STAC item returned by `list_fire_scenes`.

```python
meta = tanager.get_scene_metadata(scenes[0])
print(meta["product_types"], meta["file_size_mb"])
```

### `download_scene(item, product_type, output_dir, overwrite=False) -> Path`

Stream-download a single Tanager scene asset to `output_dir`.

| Parameter | Type | Description |
|---|---|---|
| `item` | `pystac.Item` | Scene item from `list_fire_scenes`. |
| `product_type` | `str` | Asset key to download (e.g. `"analytic"`), must be in `item.assets`. |
| `output_dir` | `Path` | Destination directory; created if missing. |
| `overwrite` | `bool` | Skip download if the file already exists (default `False`). |

Raises `KeyError` if `product_type` is not a valid asset key, `ConnectionError`
on network failure.

---

## io

Scene I/O for Planet Tanager-1 HDF5 hyperspectral data. Auto-detects
swath-vs-ortho product layout.

### `load_scene(filepath, wavelength_range=None) -> xr.Dataset`

Load a Tanager scene, auto-detecting swath (via `hypercoast.read_tanager`)
vs. ortho-rectified (via `load_ortho_scene`) HDF5 layout.

| Parameter | Type | Description |
|---|---|---|
| `filepath` | `str \| PathLike` | Path to a Tanager `.h5` file (local or HTTPS URL). |
| `wavelength_range` | `tuple[float, float] \| None` | Optional `(min_wl, max_wl)` nm; when set, only bands in range are retained. |

Returns an `xr.Dataset` with dims `(wavelength, y, x)`. Raises `ValueError`
if the file cannot be read or the wavelength range selects zero bands.

### `load_ortho_scene(filepath, wavelength_range=None) -> xr.Dataset`

Load an ortho-rectified Tanager surface-reflectance product directly via
`h5py`. Used as the fallback path when `hypercoast.read_tanager` cannot
handle the file (ortho products lack lat/lon arrays required by HyperCoast).

| Parameter | Type | Description |
|---|---|---|
| `filepath` | `str \| PathLike` | Path to a Tanager `ortho_sr` `.h5` file. |
| `wavelength_range` | `tuple[float, float] \| None` | Optional `(min_wl, max_wl)` nm band subset; reads only the matching contiguous slice from disk. |

Returns an `xr.Dataset` with data variables `surface_reflectance` and
`toa_radiance` (alias), dims `(wavelength, y, x)`, coords `wavelength` (nm),
`y`/`x` (UTM metres), `fwhm` (nm), `good_wavelengths` (uint8). Attrs include
`crs`, `epsg`, `data_var`, `product`, `source`. Fill-value pixels (`-9999`)
become NaN. Raises `ValueError` on missing required HDF5 structure.

```python
import tanager

ds = tanager.load_ortho_scene(
    "data/raw/fire/20250123_185507_64_4001_ortho_sr.h5",
    wavelength_range=(400.0, 2400.0),
)
print(ds["surface_reflectance"].dims, ds.sizes)
print(ds.attrs["crs"])
```

### `get_spatial_info(dataset: xr.Dataset) -> dict`

Extract CRS, spatial bounds, pixel resolution, and raster shape from a
loaded Dataset.

| Parameter | Type | Description |
|---|---|---|
| `dataset` | `xr.Dataset` | Dataset returned by `load_scene` / `load_ortho_scene`. |

Returns `{"crs": str | None, "bounds": (xmin, ymin, xmax, ymax), "resolution": (x_res, y_res) | None, "shape": (n_rows, n_cols)}`.

```python
info = tanager.get_spatial_info(ds)
print(info["crs"], info["shape"], info["resolution"])
```

> **Note:** `io.reproject_to_common_grid` (multi-temporal grid alignment) is
> available via `tanager.io.reproject_to_common_grid` but is not part of the
> top-level lazy-export API; several functions above (`dnbr`,
> `compute_trajectories`) call it internally when scenes don't share a grid.

---

## spectral

Wavelength-based band selection, bad-band masking, normalized-difference
spectral indices, and convex-hull continuum removal. All functions return
new objects — inputs are never modified in place.

### `select_bands(dataset, *, min_wl=None, max_wl=None, wavelengths=None)`

Select a subset of spectral bands. Exactly one mode: **range** (`min_wl` +
`max_wl`) or **nearest-neighbor** (`wavelengths`).

| Parameter | Type | Description |
|---|---|---|
| `dataset` | `xr.Dataset` | Must have a `wavelength` coordinate (nm). |
| `min_wl`, `max_wl` | `float \| None` | Inclusive range bounds (nm); provide both together. |
| `wavelengths` | `Sequence[float] \| None` | Target wavelengths for nearest-neighbor matching. |

Range mode returns `xr.Dataset`; nearest-neighbor mode returns
`(xr.Dataset, np.ndarray)` (matched wavelengths). Raises `ValueError` if
both/neither mode given, or range selects zero bands.

```python
subset = tanager.select_bands(ds, min_wl=800.0, max_wl=1000.0)
```

### `mask_bad_bands(dataset, *, zones=None, hdf5_filepath=None) -> xr.Dataset`

Remove bands in atmospheric-absorption / sensor-edge ranges (default:
`config.BAD_BAND_RANGES`), optionally OR-combined with the sensor's own
`good_wavelengths` flag.

| Parameter | Type | Description |
|---|---|---|
| `dataset` | `xr.Dataset` | Must have `wavelength` coordinate. |
| `zones` | `list[tuple] \| None` | Overrides (replaces) the default exclusion zones. |
| `hdf5_filepath` | `str \| PathLike \| None` | Source HDF5 to read `good_wavelengths` from, if not already on `dataset.coords`. |

Returns a new `xr.Dataset` with bad bands removed.

### `clamp_reflectance(data, vmin=0.0, vmax=1.0)`

Clamp reflectance values to `[vmin, vmax]` (real Tanager ISOFIT SR has ~13%
negative values that break spectral-index math otherwise).

| Parameter | Type | Description |
|---|---|---|
| `data` | `xr.Dataset \| xr.DataArray` | Dataset (clamps its `reflectance` var) or bare DataArray. |
| `vmin`, `vmax` | `float` | Clamp bounds, default `[0.0, 1.0]`. |

### `nbr(dataset: xr.Dataset) -> xr.DataArray`

Normalized Burn Ratio: `(NIR - SWIR2) / (NIR + SWIR2)` using 860/2200 nm
bands, clamped to `[0, 1]` before the ratio. Returns `(y, x)` DataArray in
`[-1, 1]`, NaN where the denominator is near zero.

### `ndvi(dataset: xr.Dataset) -> xr.DataArray`

Normalized Difference Vegetation Index: `(NIR - Red) / (NIR + Red)` using
860/660 nm bands. Same NaN/clamp semantics as `nbr`.

### `ndwi(dataset: xr.Dataset) -> xr.DataArray`

Normalized Difference Water Index: `(Green - NIR) / (Green + NIR)` using
560/860 nm bands. Same NaN/clamp semantics as `nbr`.

### `dnbr(pre, post, *, auto_align=True) -> xr.DataArray`

Differenced NBR: `NBR(pre) - NBR(post)`. Positive values indicate burn
severity.

| Parameter | Type | Description |
|---|---|---|
| `pre` | `xr.Dataset` | Pre-fire scene. |
| `post` | `xr.Dataset` | Post-fire scene. |
| `auto_align` | `bool` | When `True` (default), auto-reproject onto a common grid via `io.reproject_to_common_grid` if the two scenes don't already share one. `False` raises `ValueError` on mismatch instead. |

```python
dnbr_map = tanager.dnbr(pre_ds, post_ds)
```

### `continuum_removal(dataset, wavelength_range=None) -> xr.DataArray`

Apply convex-hull continuum removal to every pixel spectrum (upper-hull
normalization; output is relative absorption depth, not absolute
reflectance).

| Parameter | Type | Description |
|---|---|---|
| `dataset` | `xr.Dataset \| xr.DataArray` | Must have a `wavelength` coordinate. |
| `wavelength_range` | `tuple[float, float] \| None` | Restrict the hull fit + output to this band range. |

Returns `xr.DataArray` with dims `(wavelength, y, x)`, values in `[0, 1]`.

---

## masks

Pixel-quality masking utilities. All functions return `xr.DataArray`
booleans with dims `(y, x)`. Convention: **True = valid/keep**, **False =
masked/discard** — matches `xr.Dataset.where(mask)`.

### `nodata_mask(dataset, fill_value=None) -> xr.DataArray`

True where all bands at a pixel are finite (and, if `fill_value` given, not
equal to the sentinel).

| Parameter | Type | Description |
|---|---|---|
| `dataset` | `xr.Dataset` | Dataset with `wavelength` dim. |
| `fill_value` | `float \| None` | Sentinel missing-data value (e.g. `-9999`). |

### `cloud_mask(dataset, filepath=None) -> xr.DataArray`

True where the pixel is cloud-free, sourced from `beta_cirrus_mask` /
`beta_cloud_mask` (dataset variables, then HDF5 fields, then an all-True
fallback with a warning).

| Parameter | Type | Description |
|---|---|---|
| `dataset` | `xr.Dataset` | Dataset produced by `io.load_scene`. |
| `filepath` | `str \| PathLike \| None` | Explicit source `.h5` path, useful if the dataset lost its encoding. |

### `water_mask(dataset, threshold=0.3) -> xr.DataArray`

True where land (NDWI ≤ `threshold`), False where water. Wraps
`spectral.ndwi`.

| Parameter | Type | Description |
|---|---|---|
| `dataset` | `xr.Dataset` | Must contain Green (~560 nm) and NIR (~860 nm) bands. |
| `threshold` | `float` | NDWI cutoff above which pixels are water (default `0.3`, McFeeters 1996). |

### `apply_masks(dataset, mask_list) -> xr.Dataset`

Logical AND of all masks in `mask_list`; masked-out pixels become NaN.

```python
from tanager.masks import nodata_mask, cloud_mask, water_mask, apply_masks

clean = apply_masks(ds, [nodata_mask(ds), cloud_mask(ds), water_mask(ds)])
```

---

## endmembers

Assembles a fire-relevant endmember library from USGS, ECOSTRESS, FRAMES
SoCal, and image-derived spectral sources for MESMA unmixing. All loaders
return `xr.DataArray` with dims `(spectrum_id, wavelength)` and per-spectrum
coords `name`, `category`, `source`; reflectance is in `[0, 1]`.

### `load_usgs_library(categories=None, data_dir=None, *, sensor_hint="ASD", category_map=None) -> xr.DataArray`

Load USGS splib07a spectra from local ASCII files (must be downloaded
manually — see docstring for the ScienceBase DOI).

| Parameter | Type | Description |
|---|---|---|
| `categories` | `Sequence[str] \| None` | Keep only these category strings; `None` = all. |
| `data_dir` | `str \| PathLike` | Directory of extracted `ASCIIdata_splib07a` files. **Required.** |
| `sensor_hint` | `str` | Substring selecting spectrum/wavelength files, default `"ASD"`. |
| `category_map` | `Mapping[str, str] \| None` | Explicit filename-stem → category overrides. |

Source tag: `"usgs_v7"`. Raises `FileNotFoundError` / `ValueError` on bad
inputs.

### `load_ecostress_library(categories=None, *, sqlite_path=None, db=None) -> xr.DataArray`

Load ECOSTRESS spectra via SPy `EcostressDatabase`, filtered to VSWIR
(350–2500 nm).

| Parameter | Type | Description |
|---|---|---|
| `categories` | `Sequence[str] \| None` | Keep only these categories. |
| `sqlite_path` | `str \| PathLike \| None` | Pre-built ECOSTRESS SQLite database path. |
| `db` | `object \| None` | Pre-instantiated duck-typed DB (for tests); overrides `sqlite_path`. |

Source tag: `"ecostress"`. Raises `RuntimeError` if neither `sqlite_path`
nor `db` given.

### `load_frames_library(data_dir, *, category_map=None, file_pattern="*.txt") -> xr.DataArray`

Load the FRAMES SoCal chaparral spectral library from local two-column
ASCII files (download manually from frames.gov).

| Parameter | Type | Description |
|---|---|---|
| `data_dir` | `str \| PathLike` | Directory of FRAMES ASCII spectrum files. |
| `category_map` | `Mapping[str, str] \| None` | Filename-stem → category overrides. |
| `file_pattern` | `str` | Glob pattern for spectrum files, default `"*.txt"`. |

Source tag: `"frames"`. Auto-detects µm-vs-nm and percent-vs-fraction units.

### `resample_library(library, target_wavelengths, *, fwhm=5.5, source_fwhm=1.0) -> xr.DataArray`

Resample a library to a target wavelength grid via `spectral.BandResampler`
(Gaussian convolution).

| Parameter | Type | Description |
|---|---|---|
| `library` | `xr.DataArray` | Output of one of the loaders above; must have `wavelength` coord. |
| `target_wavelengths` | `np.ndarray` | 1D array of target band centres (nm), e.g. Tanager's 426 bands. |
| `fwhm` | `float \| np.ndarray` | Target sensor FWHM (nm), scalar or per-band. |
| `source_fwhm` | `float \| np.ndarray` | Source spectrometer FWHM (nm), default 1.0 (ASD field-spec). |

### `build_hybrid_library(usgs=None, ecostress=None, frames=None, image_derived=None) -> xr.DataArray`

Merge multiple resampled libraries (must already share a wavelength grid)
into one DataArray with source-tracking, deduplicating `spectrum_id`s.

### `select_endmembers_incob(library, max_per_class) -> xr.DataArray`

Reduce a library to at most `max_per_class` spectra per class, ranked by
within-class spectral standard deviation (a simplified In-CoB proxy).

| Parameter | Type | Description |
|---|---|---|
| `library` | `xr.DataArray` | Must carry a `category` coordinate. |
| `max_per_class` | `int` | Cap per category, must be ≥ 1. |

### `prune_endmembers_ear_masa(library, *, threshold_ear=0.025, threshold_masa=10.0) -> xr.DataArray`

Prune redundant/noisy endmembers via EAR (Endmember Average RMSE) / MASA
(Mean Spectral Angle) — Roberts et al. 2018. A spectrum exceeding **both**
thresholds is dropped.

| Parameter | Type | Description |
|---|---|---|
| `library` | `xr.DataArray` | Must carry a `category` coordinate. |
| `threshold_ear` | `float` | Max acceptable EAR (default 0.025). |
| `threshold_masa` | `float` | Max acceptable MASA in degrees (default 10.0). |

Adds `ear` / `masa_deg` coords to retained spectra.

### `extract_image_endmembers(scene, method="spatial", regions=None, *, n_pure_pixels=30, n_iterations=1000, seed=None) -> xr.DataArray`

Extract image-derived endmembers from a scene: `"spatial"` (average
reflectance within named ROI windows) or `"ppi"` (SPy Pixel Purity Index).

| Parameter | Type | Description |
|---|---|---|
| `scene` | `xr.Dataset \| xr.DataArray` | Scene with dims `(wavelength, y, x)`. |
| `method` | `str` | `"spatial"` or `"ppi"`. |
| `regions` | `Mapping[str, tuple[slice, slice]] \| None` | Category → `(y_slice, x_slice)`, required for `"spatial"`. |
| `n_pure_pixels` | `int` | Purest pixels to return for `"ppi"` (default 30). |
| `n_iterations` | `int` | PPI random-projection iterations (default 1000). |
| `seed` | `int \| None` | RNG seed for reproducible PPI runs. |

Source tag: `"image"`.

### `build_fire_library(*, scene_pre=None, scene_post=None, target_wavelengths=None, target_fwhm=5.5, usgs_dir=None, ecostress_sqlite=None, frames_dir=None, pre_regions=None, post_regions=None, max_per_class=15, threshold_ear=0.025, threshold_masa=10.0, add_shade=True) -> xr.DataArray`

Orchestrates the full pipeline: load sources → resample → merge → In-CoB
select → EAR/MASA prune → optional zero-reflectance shade spectrum. Skips
any source left as `None`. Logs a warning if the final library falls
outside the recommended 50–80 spectra range.

Raises `ValueError` if no source is provided at all.

```python
from tanager.endmembers import build_fire_library

library = build_fire_library(
    usgs_dir="data/reference/splib07a",
    frames_dir="data/reference/frames_socal",
    scene_pre=pre_scene,
    scene_post=post_scene,
    pre_regions={"pv": (slice(100, 150), slice(200, 250))},
    post_regions={"char": (slice(300, 350), slice(400, 450))},
    max_per_class=15,
)
```

---

## unmixing

MESMA (Multiple Endmember Spectral Mixture Analysis) unmixing pipeline.
Backend priority: `mesma` package → `hysup` FCLS → `scipy.optimize.nnls`.
Chosen backend recorded in `Dataset.attrs["unmixing_engine"]`.

### `run_mesma(scene, library, constraints=None, bands=None) -> xr.Dataset`

Main unmixing entry point. Produces per-pixel fractional abundance maps for
the canonical classes plus RMSE.

| Parameter | Type | Description |
|---|---|---|
| `scene` | `xr.Dataset \| xr.DataArray` | Scene with `wavelength`, `y`, `x` coords. |
| `library` | `xr.DataArray` | Endmember library (output of `endmembers.build_fire_library`), must carry `category` and be resampled to the scene's wavelength grid. |
| `constraints` | `Mapping[str, float] \| None` | Overrides for `min_fraction, max_fraction, min_shade, max_shade, max_rmse` (see `DEFAULT_CONSTRAINTS`). |
| `bands` | `Sequence[float] \| np.ndarray \| None` | Optional wavelength subset to run unmixing on (pair with `select_bands_uszu`). |

Returns `xr.Dataset` with variables `char, pv, npv, soil, shade, rmse`, dims
`(y, x)`. `ds.attrs["unmixing_engine"]` is `"mesma"`, `"hysup"`, or `"nnls"`.

```python
from tanager import run_mesma

fractions = run_mesma(post_scene, library)
print(fractions.attrs["unmixing_engine"])
print(fractions["char"].mean().item())
```

### `select_bands_uszu(scene, library, n_bands=40) -> tuple[xr.Dataset, np.ndarray]`

Select the most class-discriminatory bands via Uniform Spectral Zone
Unmixing (Somers et al. 2010): partitions the spectrum into `n_bands` equal
zones and picks the highest Fisher-separability band per zone.

| Parameter | Type | Description |
|---|---|---|
| `scene` | `xr.Dataset \| xr.DataArray` | Bad bands should already be masked (`spectral.mask_bad_bands`). |
| `library` | `xr.DataArray` | Endmember library with `category` coord, used as class-statistics proxy. |
| `n_bands` | `int` | Number of bands to select (default 40). |

Returns `(scene_subset, selected_indices)`.

### `normalize_fractions(fractions, remove_shade=True) -> xr.Dataset`

Remove the shade fraction and rescale remaining fractions to sum to 1.0
(Roberts et al. 2018 standard practice).

| Parameter | Type | Description |
|---|---|---|
| `fractions` | `xr.Dataset` | Output of `run_mesma`. |
| `remove_shade` | `bool` | Default `True`; `False` returns an unmodified copy. |

### `plot_fraction_maps(fractions, figsize=None, cmap="viridis") -> Figure`

Multi-panel matplotlib figure of per-class fraction maps (excludes `rmse`).

### `plot_rgb_composite(fractions, r="char", g="pv", b="npv", figsize=None) -> Figure`

False-colour RGB composite from three fraction maps, each independently
normalized to `[0, 1]`.

---

## severity

Burn severity products from MESMA fraction maps, following Quintano et al.
(2023) and Key & Benson (2006) BARC thresholds.

### `train_severity_model(fractions, ground_truth_cbi, method="random_forest", *, n_estimators=200, random_state=42, cv_folds=5, feature_names=None) -> dict`

Fit a regressor (default RandomForest) mapping the 4-feature fraction
vector (`char, pv, npv, soil`) to ground-truth Composite Burn Index (CBI),
with 5-fold cross-validated R²/RMSE.

| Parameter | Type | Description |
|---|---|---|
| `fractions` | `xr.Dataset` | Feature variables shaped `(y, x)`, typically `unmixing.normalize_fractions` output. |
| `ground_truth_cbi` | `np.ndarray \| xr.DataArray \| Sequence[float]` | CBI values aligned with the flattened pixel order; length must equal `y * x`. |
| `method` | `str` | Only `"random_forest"` supported. |
| `n_estimators` | `int` | RF tree count (default 200). |
| `random_state` | `int` | RF seed (default 42). |
| `cv_folds` | `int` | K-fold CV count (default 5). |
| `feature_names` | `Sequence[str] \| None` | Override default `("char", "pv", "npv", "soil")`. |

Returns `{"model", "r2", "rmse", "method", "feature_names", "n_samples"}`.

### `predict_severity(fractions, model, *, feature_names=None) -> dict[str, xr.DataArray]`

Apply a trained severity model to produce a continuous CBI map (clipped to
`[0, 3]`) and a 5-class BARC severity map.

| Parameter | Type | Description |
|---|---|---|
| `fractions` | `xr.Dataset` | Feature variables shaped `(y, x)`. |
| `model` | `dict \| estimator` | Preferably the dict from `train_severity_model`; a bare fitted estimator also works. |
| `feature_names` | `Sequence[str] \| None` | Override the model's stored feature names. |

Returns `{"cbi_map": DataArray, "severity_map": DataArray}`. BARC classes:
0=Unburned (CBI<0.10), 1=Low (<1.00), 2=Moderate-Low (<1.50),
3=Moderate-High (<2.25), 4=High (≥2.25).

```python
from tanager import train_severity_model, predict_severity
from tanager.unmixing import normalize_fractions

norm_fractions = normalize_fractions(fractions)
trained = train_severity_model(norm_fractions, ground_truth_cbi)
print(f"CV R2={trained['r2']:.3f}  CV RMSE={trained['rmse']:.3f}")

result = predict_severity(norm_fractions, trained)
cbi_map = result["cbi_map"]
severity_map = result["severity_map"]
```

### `compute_trajectories(scenes_dict, library, *, constraints=None, bands=None, align=True) -> xr.Dataset`

Run MESMA on a dictionary of dated scenes against the same endmember
library and stack the fraction outputs into a single `(time, y, x)`
Dataset for burn-recovery trajectory analysis.

| Parameter | Type | Description |
|---|---|---|
| `scenes_dict` | `Mapping[str, xr.Dataset]` | Date label (ISO-8601 string, typically) → scene Dataset. |
| `library` | `xr.DataArray` | Same endmember library used for every scene (keeps fractions comparable). |
| `constraints` | `Mapping[str, float] \| None` | Forwarded to `unmixing.run_mesma`. |
| `bands` | `np.ndarray \| None` | Forwarded to `unmixing.run_mesma`. |
| `align` | `bool` | Auto-reproject onto a common grid if scenes don't already share one (default `True`). |

```python
from tanager import compute_trajectories

scenes_dict = {
    "2024-12-15T18:59:16": prefire_scene,
    "2025-01-23T18:55:07": postfire_scene,
    "2025-04-07T19:22:35": early_recovery_scene,
}
trajectories = compute_trajectories(scenes_dict, library)
print(trajectories.dims, trajectories["char"].sizes)
char_over_time = trajectories["char"].mean(dim=("y", "x"))
```

### `compare_severity_methods(mesma_severity, dnbr_map) -> dict`

Compare a MESMA-derived severity map against a dNBR baseline on jointly
finite pixels: Pearson correlation, RMSE, bias, and a difference map.

| Parameter | Type | Description |
|---|---|---|
| `mesma_severity` | `xr.DataArray` | Typically `predict_severity(...)["cbi_map"]`. |
| `dnbr_map` | `xr.DataArray` | Output of `spectral.dnbr`, same spatial grid. |

Returns `{"correlation", "rmse", "bias", "difference_map", "n_valid"}`.

---

## lfmc

Live Fuel Moisture Content (LFMC) products from Tanager-1 reflectance —
Tier 1 spectral indices (Quan et al. 2021) and Tier 2 PLSR (Peterson &
Roberts 2014).

### `compute_lfmc_indices(scene) -> xr.Dataset`

Compute the water-sensitive index stack: `SAI970`, `SAI1200` (Spectral
Absorption Index), `NDWI_1240`/`NDWI_1640`/`NDWI_2130`, `WI` (R900/R970),
and `CR_depths` (continuum-removal depths at 970/1200/1700/2100 nm, dims
`(cr_target, y, x)`).

| Parameter | Type | Description |
|---|---|---|
| `scene` | `xr.Dataset \| xr.DataArray` | Must have `reflectance`/`surface_reflectance` shaped `(wavelength, y, x)` spanning at least 860–2130 nm. |

Returns an `xr.Dataset` with 8 index variables (reflectance is clamped to
`[0, 1]` before all index math). Raises `ValueError` if the wavelength axis
doesn't span the required range.

```python
from tanager import compute_lfmc_indices

indices = compute_lfmc_indices(scene)
print(list(indices.data_vars))
sai970_mean = indices["SAI970"].mean().item()
cr_1700 = indices["CR_depths"].sel(cr_target=1700.0)
```

### `load_globe_lfmc(data_path, *, region_bbox=None, vegetation_types=None, tanager_scene_dates=None, colocation_window_days=30) -> GeoDataFrame`

Load Globe-LFMC 2.0 in-situ observations as a filtered GeoDataFrame for use
as `train_lfmc_plsr` ground truth.

| Parameter | Type | Description |
|---|---|---|
| `data_path` | `str \| PathLike` | Path to the Globe-LFMC CSV (must be downloaded manually). |
| `region_bbox` | `tuple[float, float, float, float] \| None` | `(west, south, east, north)` WGS84 filter. |
| `vegetation_types` | `Sequence[str] \| None` | Case-insensitive substring filters against vegetation_type/species. |
| `tanager_scene_dates` | `Sequence \| None` | When given, flags rows within `colocation_window_days` of any scene date. |
| `colocation_window_days` | `int` | Half-window for colocation flag (default 30). |

### `train_lfmc_plsr(spectra, lfmc_values, n_components=10, *, cv_folds=5) -> dict`

Fit a Partial Least Squares regression from full-spectrum reflectance to
ground-truth LFMC, with CV component selection and per-band VIP scores.

| Parameter | Type | Description |
|---|---|---|
| `spectra` | `np.ndarray` | Shape `(n_samples, n_bands)`; bad bands should already be removed. |
| `lfmc_values` | `np.ndarray` | 1D LFMC ground-truth values (percent). |
| `n_components` | `int` | Upper bound on PLS components to search (default 10). |
| `cv_folds` | `int` | K for CV, clipped to sample count (default 5). |

Returns `{"model", "r2", "rmse", "n_components_optimal", "vip_scores"}`.

### `predict_lfmc(scene, model, method="plsr") -> dict[str, xr.DataArray]`

Apply a trained LFMC model to a scene, producing an LFMC map plus
uncertainty and a low-LFMC (<60%, fire-prone regime) flag.

| Parameter | Type | Description |
|---|---|---|
| `scene` | `xr.Dataset \| xr.DataArray` | Reflectance shaped `(wavelength, y, x)`; band layout must match training. |
| `model` | `dict \| estimator` | Preferably the dict from `train_lfmc_plsr` (drives the uncertainty floor). |
| `method` | `str` | Only `"plsr"` currently implemented. |

Returns `{"lfmc_map", "uncertainty_map", "low_lfmc_flag"}` DataArrays,
clipped to `[0, 300]` percent.

```python
from tanager import train_lfmc_plsr, predict_lfmc
from tanager.spectral import mask_bad_bands, scene_reflectance
import numpy as np

masked = mask_bad_bands(scene)
refl = scene_reflectance(masked)
spectra = refl.values.reshape(refl.sizes["wavelength"], -1).T  # (n_samples, n_bands)

trained = train_lfmc_plsr(spectra, ground_truth_lfmc_percent, n_components=10)
print(f"CV R2={trained['r2']:.3f}  n_components={trained['n_components_optimal']}")

result = predict_lfmc(masked, trained)
lfmc_map = result["lfmc_map"]
low_lfmc_pixels = result["low_lfmc_flag"]
```

---

## validation

Accuracy assessment against external reference data (AVIRIS-3, USGS BARC)
and Tanager-vs-reference-sensor comparisons (EMIT/PRISMA), including the +5
competition tie-breaker workflow.

### `load_aviris3_reference(filepath, target_resolution=30.0) -> xr.Dataset`

Load an AVIRIS-3 fraction reference product (GeoTIFF or NetCDF), spatially
aggregated to the target resolution and renamed to the canonical
`char/pv/npv/soil/shade` schema.

| Parameter | Type | Description |
|---|---|---|
| `filepath` | `str \| PathLike` | Path to the AVIRIS-3 product. |
| `target_resolution` | `float` | Target GSD in metres (default 30.0, matching Tanager ortho_sr). |

### `load_barc_reference(filepath, *, code_map=None, target_grid=None) -> xr.DataArray`

Load a USGS BARC classified-severity GeoTIFF, normalized to the canonical
0–4 severity code scheme; NoData encoded as `-1`.

| Parameter | Type | Description |
|---|---|---|
| `filepath` | `str \| PathLike` | Path to the BARC GeoTIFF. |
| `code_map` | `Mapping[int, int] \| None` | Override the default BARC code translation. |
| `target_grid` | `xr.DataArray \| None` | When given, reprojects (nearest-neighbour) onto this grid's coords. |

### `compute_accuracy(predicted, observed, metric_type="continuous", *, nodata=-1) -> dict`

Accuracy metrics for predicted vs. observed values, pairwise NaN/nodata
filtered.

| Parameter | Type | Description |
|---|---|---|
| `predicted` | `np.ndarray \| xr.DataArray \| Sequence[float]` | Predicted values. |
| `observed` | `np.ndarray \| xr.DataArray \| Sequence[float]` | Reference/ground-truth values, same shape as `predicted`. |
| `metric_type` | `str` | `"continuous"` (R², RMSE, MAE, bias, Spearman) or `"classified"` (accuracy, kappa, F1, confusion matrix). |
| `nodata` | `int \| None` | NoData sentinel for classified inputs (default -1). |

### `simulate_sensor(scene, target_centers, target_fwhm, sensor_name, *, source_fwhm=5.5)`

Spectrally degrade a Tanager scene to simulate a reference sensor (EMIT,
PRISMA, Sentinel-2) via Gaussian-overlap band convolution
(`spectral.BandResampler`). Powers the +5 competition tie-breaker together
with `compare_sensors`.

| Parameter | Type | Description |
|---|---|---|
| `scene` | `xr.DataArray \| xr.Dataset` | Source Tanager scene; DataArray needs a `wavelength` dim, Dataset variables without one pass through unchanged. |
| `target_centers` | `np.ndarray \| Sequence[float]` | Target sensor band centres (nm), e.g. `EMIT_SENSOR` wavelengths. |
| `target_fwhm` | `float \| np.ndarray \| Sequence[float]` | Target sensor FWHM (nm), scalar or per-band. |
| `sensor_name` | `str` | Label written to the output's `sensor_name` attribute (e.g. `"EMIT"`). |
| `source_fwhm` | `float \| np.ndarray \| Sequence[float]` | Source (Tanager) FWHM (nm), default 5.5 (mean across the 5.20-6.81 nm per-band range). |

Returns the same container type as `scene`, with `wavelength` replaced by
`target_centers` and reflectance clipped to `[0, 1]`.

### `compare_sensors(tanager_result, reference_result, ground_truth, sensor_name, *, metric_type="continuous") -> dict`

Compute accuracy for both Tanager and a reference sensor against the same
ground truth, plus improvement ratios and a comparison table — the +5
competition tie-breaker.

| Parameter | Type | Description |
|---|---|---|
| `tanager_result` | array-like | Tanager prediction (continuous or classified). |
| `reference_result` | array-like | Reference sensor prediction (e.g. EMIT, PRISMA). |
| `ground_truth` | array-like | Reference labels (e.g. AVIRIS-3, BARC, field CBI). |
| `sensor_name` | `str` | Human-readable reference sensor name, used in the comparison table. |
| `metric_type` | `str` | `"continuous"` or `"classified"`, passed to `compute_accuracy`. |

Returns `{"tanager_metrics", "reference_metrics", "improvement_ratios", "comparison_table", "sensor_name"}`.

```python
from tanager import compare_sensors, simulate_sensor
from tanager.config import EMIT_SENSOR

emit_scene = simulate_sensor(
    scene["surface_reflectance"],
    target_centers=emit_band_centers,   # e.g. from EMIT_SENSOR metadata
    target_fwhm=EMIT_SENSOR.fwhm_nm,
    sensor_name="EMIT",
)
# ... run the same severity/fraction pipeline on emit_scene to get emit_result ...

comparison = compare_sensors(
    tanager_result=tanager_cbi_predictions,
    reference_result=emit_cbi_predictions,
    ground_truth=field_cbi_values,
    sensor_name="EMIT",
)
print(comparison["improvement_ratios"])
print(comparison["comparison_table"])
```

---

## visualization

Map-making, diagnostic plotting, and interactive visualization for the
Tanager product suite. Heavy rendering deps (matplotlib, contextily,
geopandas, rioxarray, leafmap/folium) are imported lazily inside each
function.

### `PRODUCT_STYLES`

`dict[str, ProductStyle]` — colormap/scale presets keyed by product name
(`"nbr"`, `"ndvi"`, `"ndwi"`, `"dnbr"`, `"cbi"`, `"severity"`, `"char"`,
`"pv"`, `"npv"`, `"soil"`, `"lfmc"`). Each `ProductStyle` has `cmap`, `vmin`,
`vmax`, `label`, `class_ticks`. Consumed by every `plot_*` function below
via their `product_name` parameter.

### `plot_map(da, title="", cmap=None, vmin=None, vmax=None, product_name=None, publication=False, figsize=(10, 8), basemap=False, ax=None) -> Figure`

Render a single-band raster as a georeferenced map with UTM axes (km
labels), NaN pixels transparent.

| Parameter | Type | Description |
|---|---|---|
| `da` | `xr.DataArray` | 2-D array with `x`/`y` (or `easting`/`northing`) coords in metres. |
| `title` | `str` | Figure title. |
| `cmap`, `vmin`, `vmax` | | Colour-scale overrides; default to `PRODUCT_STYLES[product_name]` when set. |
| `product_name` | `str \| None` | Key into `PRODUCT_STYLES`. |
| `publication` | `bool` | DPI 300 + larger fonts when `True`. |
| `figsize` | `tuple` | Ignored when `ax` is given. |
| `basemap` | `bool` | Calls `add_basemap` after rendering. |
| `ax` | `Axes \| None` | Draw into an existing axes instead of creating a new figure. |

### `plot_before_after(pre, post, product_name="nbr", pre_label=None, post_label=None, fire_perimeters=None, basemap=False, publication=False, figsize=(16, 8)) -> Figure`

Side-by-side pre/post fire comparison with a single shared colorbar.

| Parameter | Type | Description |
|---|---|---|
| `pre`, `post` | `xr.DataArray` | Pre-fire / post-fire rasters; each rendered at its own extent. |
| `product_name` | `str` | `PRODUCT_STYLES` key for colormap/scale/label (default `"nbr"`). |
| `pre_label`, `post_label` | `str \| None` | Panel titles (default `"Pre-Fire"` / `"Post-Fire"`). |
| `fire_perimeters` | `GeoDataFrame \| None` | Overlaid via `overlay_perimeters` on both panels. |
| `basemap` | `bool` | Overlay a basemap on both panels. |
| `publication` | `bool` | DPI 300 + larger fonts. |
| `figsize` | `tuple` | Figure size in inches (default `(16, 8)`). |

```python
from tanager import plot_before_after, save_figure
from tanager.spectral import nbr

pre_nbr = nbr(pre_scene)
post_nbr = nbr(post_scene)

fig = plot_before_after(
    pre_nbr, post_nbr,
    product_name="nbr",
    pre_label="Dec 2024 (pre-fire)",
    post_label="Jan 2025 (post-fire)",
)
save_figure(fig, "outputs/eaton_fire_nbr_before_after", formats=["png", "pdf"])
```

### `plot_temporal_trajectory(dates, values, product_name="NBR", fire_date="2025-01-07", error_bands=None, ax=None, publication=False, figsize=(12, 6)) -> Figure`

Time-series trajectory plot for a pixel/ROI, with fire-ignition marker and
shaded pre-/post-fire regions.

| Parameter | Type | Description |
|---|---|---|
| `dates` | `list` | Date values — `str`, `datetime`, or anything `pandas.to_datetime` accepts. |
| `values` | `list[float]` | Index values, one per date. |
| `product_name` | `str` | Y-axis label / legend entry (e.g. `"NBR"`). |
| `fire_date` | `str \| datetime \| None` | Ignition date; draws a red dashed line + shaded pre/post regions. `None` disables. |
| `error_bands` | `list[float] \| None` | Per-date ±uncertainty; drawn as a shaded band. |
| `ax` | `Axes \| None` | Draw into an existing axes. |
| `publication` | `bool` | DPI 300 + larger fonts. |
| `figsize` | `tuple` | Default `(12, 6)`. |

```python
from tanager import plot_temporal_trajectory, save_figure

dates = ["2024-12-15", "2025-01-23", "2025-04-07", "2025-07-26", "2025-09-20"]
nbr_values = [0.65, 0.15, 0.22, 0.38, 0.51]
uncertainty = [0.03, 0.05, 0.04, 0.04, 0.05]

fig = plot_temporal_trajectory(
    dates, nbr_values,
    product_name="NBR",
    fire_date="2025-01-07",
    error_bands=uncertainty,
)
save_figure(fig, "outputs/eaton_fire_nbr_trajectory")
```

### `plot_severity_summary(fractions, cbi, severity_class, publication=False, figsize=(18, 12)) -> Figure`

2×3 multi-panel summary grid: char / PV / NPV fractions (top row), soil
fraction / CBI / severity class (bottom row), each with its own colormap
and colorbar drawn from `PRODUCT_STYLES`.

| Parameter | Type | Description |
|---|---|---|
| `fractions` | `xr.Dataset` | Must contain `char`, `pv`, `npv`, `soil` variables, all sharing UTM `x`/`y` coords. |
| `cbi` | `xr.DataArray` | Composite Burn Index map, same spatial coords. |
| `severity_class` | `xr.DataArray` | Integer burn-severity class map, same spatial coords. |
| `publication` | `bool` | DPI 300 + larger fonts. |
| `figsize` | `tuple` | Default `(18, 12)`. |

```python
from tanager import plot_severity_summary, save_figure

fig = plot_severity_summary(
    normalize_fractions(fractions),
    cbi=result["cbi_map"],
    severity_class=result["severity_map"],
    publication=True,
)
save_figure(fig, "outputs/eaton_fire_severity_summary", formats=["png"])
```

### `plot_difference_map(diff_da, product_name="dnbr", class_boundaries=None, publication=False, figsize=(10, 8)) -> Figure`

Styled difference raster (e.g. dNBR) rendered via `plot_map`, with labelled
severity-class contour lines overlaid.

| Parameter | Type | Description |
|---|---|---|
| `diff_da` | `xr.DataArray` | Difference product with `x`/`y` UTM coords. |
| `product_name` | `str` | `PRODUCT_STYLES` key (default `"dnbr"`). |
| `class_boundaries` | `dict[str, float] \| None` | Class name → threshold; defaults to USGS dNBR thresholds when `product_name == "dnbr"`. |
| `publication` | `bool` | DPI 300 + larger fonts. |
| `figsize` | `tuple` | Default `(10, 8)`. |

### `interactive_map(layers=None, center=None, zoom=12, perimeters=None, basemap="satellite") -> Map`

Return a `leafmap` (primary) or `folium` (fallback) interactive map widget
for Jupyter notebooks.

| Parameter | Type | Description |
|---|---|---|
| `layers` | `list[tuple[xr.DataArray, str]] \| None` | `(DataArray, product_name)` pairs; reprojected to EPSG:4326 automatically. |
| `center` | `tuple[float, float] \| None` | `(lat, lon)`; defaults to the first layer's centroid or CONUS. |
| `zoom` | `int` | Initial zoom level (default 12). |
| `perimeters` | `str \| Path \| GeoDataFrame \| None` | Fire perimeter overlay (path loaded via `load_fire_perimeters`, or a GeoDataFrame directly). |
| `basemap` | `str` | `"satellite"`, `"terrain"`, or `"osm"` (leafmap only; folium fallback always uses OSM). |

Raises `ImportError` if neither `leafmap` nor `folium` is installed.

### `show_product(da, product_name=None, scene_date=None, interactive=False) -> Figure | Map`

Convenience wrapper: display a named product as a static map (`plot_map` +
basemap) or, when `interactive=True`, an `interactive_map` widget.

| Parameter | Type | Description |
|---|---|---|
| `da` | `xr.DataArray` | 2-D array to display. |
| `product_name` | `str \| None` | `PRODUCT_STYLES` key; falls back to `da.name` when `None`. |
| `scene_date` | `str \| None` | Appended to the static-map title. |
| `interactive` | `bool` | Return an interactive widget instead of a static figure. |

### `save_figure(fig, path, formats=["png"]) -> list[Path]`

Save a matplotlib figure to disk in one or more formats, at DPI 300 with
tight bounding box.

| Parameter | Type | Description |
|---|---|---|
| `fig` | `Figure` | Figure to save. |
| `path` | `str \| Path` | Base output path **without** extension. |
| `formats` | `list[str]` | Format strings matplotlib supports (default `["png"]`). |

Returns resolved `Path` objects, one per format, in the same order.

```python
from tanager import save_figure

paths = save_figure(fig, "outputs/eaton_fire_dnbr", formats=["png", "pdf", "svg"])
# -> [Path("outputs/eaton_fire_dnbr.png"), Path("outputs/eaton_fire_dnbr.pdf"), Path("outputs/eaton_fire_dnbr.svg")]
```

### `add_basemap(ax, source="satellite", alpha=0.3, crs="EPSG:32611") -> Axes`

Overlay a web-tile basemap (via `contextily`) on an axes that already has a
projected raster rendered on it.

| Parameter | Type | Description |
|---|---|---|
| `ax` | `Axes` | Axes with an existing raster (sets the extent). |
| `source` | `str` | `"satellite"` (Esri World Imagery, default), `"terrain"` (Stadia), or `"osm"` (OSM Mapnik). |
| `alpha` | `float` | Basemap tile opacity (default 0.3). |
| `crs` | `str` | Axes CRS (default `"EPSG:32611"`, UTM Zone 11N). |

Network failures are caught and logged as a warning; `ax` is returned
unchanged in that case.

### `load_fire_perimeters(path, *, crs=None) -> GeoDataFrame`

Load NIFC/GeoMAC fire perimeter polygons from a GeoJSON, Shapefile, or
GeoPackage.

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Perimeter vector file. |
| `crs` | `Any \| None` | Reproject to this CRS after loading. |

### `overlay_perimeters(ax, perimeters, color="red", linestyle="--", linewidth=2.0, label=True) -> Axes`

Draw fire perimeter polygon boundaries on `ax` (reprojected to
`EPSG:32611`).

| Parameter | Type | Description |
|---|---|---|
| `ax` | `Axes` | Target axes, must already have a projected extent. |
| `perimeters` | `GeoDataFrame` | Any CRS; reprojected internally. |
| `color` | `str` | Line/label colour (default `"red"`). |
| `linestyle` | `str` | Line style (default `"--"`). |
| `linewidth` | `float` | Outline width in points (default 2.0). |
| `label` | `bool` | Add text labels at polygon centroids from a `name`/`incident_name` column. |

### `add_scalebar(ax, length_km=5.0, location="lower left") -> Axes`

Add a distance scale-bar (matplotlib-only, no external deps) to a map
axes, aligned exactly with UTM metre-unit projected axes.

| Parameter | Type | Description |
|---|---|---|
| `ax` | `Axes` | Target axes with limits already set. |
| `length_km` | `float` | Bar length in km (default 5.0). |
| `location` | `str` | `"lower left"` (default), `"lower right"`, `"upper left"`, `"upper right"`. |

---

## Maintenance

This document tracks `_LAZY_EXPORTS` in `src/tanager/__init__.py`. When
adding, renaming, or removing a public symbol:

1. Update `_LAZY_EXPORTS` in `__init__.py`.
2. Add/update the corresponding section here (signature, parameter table,
   and an example if the function is judge-facing).
3. Update the module export count in the header summary if it changed.
