# Change: 002-data-pipeline

## Open Questions — RESOLVED

1. **HyperCoast version pinning:** `>=0.22.0,<1.0` — floor at latest tested, cap at major version.
2. **Data directory convention:** `data/raw/fire/` in project root, override with `TANAGER_DATA_DIR` env var. Add `data/raw/fire/*.h5` to `.gitignore`.
3. **SPy vs mesma package:** Phase 2 includes only `spectral` (SPy). MESMA package decision deferred to Phase 3.

---

## Wave 1: Foundation
<!-- execution: sequential -->

### Section 1: Project Scaffolding
<!-- execution_mode: sequential -->
<!-- network: none — all local file creation -->

- [x] Create `pyproject.toml` with project metadata, Python 3.10+ requirement, and all dependencies (hypercoast>=0.22.0,<1.0; spectral; rasterio; xarray; geopandas; scikit-learn; pystac; requests; spyndex; h5py) plus dev dependencies (pytest, ruff, mypy)
  <!-- files: pyproject.toml (new) -->
  <!-- gotcha: pin hypercoast>=0.22.0,<1.0 — research tested 0.20.2, latest is 0.22.0, API may shift pre-1.0. spyndex is now 0.10.0 (not 0.6.0 as in research). Add h5py — needed by masks.py for reading beta_cirrus_mask from raw HDF5 -->
  <!-- test: pip install -e . succeeds -->

- [x] Create `src/tanager/__init__.py` with package version and top-level imports
  <!-- files: src/tanager/__init__.py (new) -->
  <!-- gotcha: export the public API symbols: config (SENSOR, BAD_BAND_RANGES, FIRE_SCENES, BAND_ALIASES, DATA_DIR), catalog (list_fire_scenes, download_scene, get_scene_metadata), io (load_scene, get_spatial_info), spectral (select_bands, mask_bad_bands, nbr, ndvi, ndwi, dnbr, continuum_removal), masks (nodata_mask, cloud_mask, water_mask, apply_masks). Use lazy imports to avoid heavy dep loading at import time. -->
  <!-- test: `import tanager; print(tanager.__version__)` works -->

- [x] Create `src/tanager/config.py` with sensor parameters (SENSOR), bad band ranges (BAD_BAND_RANGES), fire scene catalog (FIRE_SCENES), band wavelength aliases (BAND_ALIASES), and data directory path (DATA_DIR with TANAGER_DATA_DIR env override)
  <!-- files: src/tanager/config.py (new) -->
  <!-- gotcha: FIRE_SCENES — the data access evaluation lists 11 named scene IDs but proposal says 12. Query the live STAC catalog to get the authoritative count before hardcoding. If 11 scenes are found, use 11 and update the spec. The scene table from research: 20241215_185916_33_4001 (pre-fire), 20250123_185507_64_4001, 20250123_185518_92_4001 (post-fire), 20250407_192235_24_4001, 20250407_192229_16_4001 (early recovery), 20250724_190927_83_4001, 20250726_192343_21_4001, 20250726_192422_87_4001 (mid recovery), 20250902_190116_02_4001, 20250902_190121_86_4001 (late recovery), 20250920_193207_61_4001 (N. Arizona). Include bbox per scene. -->
  <!-- gotcha: DATA_DIR must use Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "fire" to find project root relative to installed package. Override with os.environ.get("TANAGER_DATA_DIR"). -->
  <!-- gotcha: SENSOR should be a SimpleNamespace or dataclass, not a plain dict, so fields are accessible via dot notation (SENSOR.n_bands, not SENSOR["n_bands"]). Spec says "namespace or dictionary" — prefer namespace. -->
  <!-- test: from tanager.config import SENSOR; assert SENSOR.n_bands == 426 -->

- [x] Create `data/raw/fire/.gitkeep` and add `data/raw/fire/*.h5` to `.gitignore`
  <!-- files: data/raw/fire/.gitkeep (new), .gitignore (modify — append data/raw/fire/*.h5) -->
  <!-- gotcha: .gitignore already has `data/raw/` which covers all files under data/raw/fire/. It also has `*.hdf5` but NOT `*.h5`. Tanager files use .h5 extension (HDF-EOS5). The existing `data/raw/` glob already provides coverage, but adding the explicit `*.h5` pattern at top level is safer and self-documenting for files stored elsewhere. -->
  <!-- test: ls data/raw/fire/.gitkeep exists -->

- [x] Verify: `pip install -e .` succeeds and `import tanager` works
  <!-- verify: manual — run `pip install -e .` in a clean venv, then `python -c "from tanager.config import SENSOR; print(SENSOR.n_bands)"` -->
  <!-- network: requires pip to resolve and download dependencies -->

## Wave 2: Data Access + Spectral Processing
<!-- execution: parallel -->
<!-- PARALLEL SAFETY CHECK
  Track A files: src/tanager/catalog.py (new)
  Track B files: src/tanager/io.py (new)
  Track C files: src/tanager/spectral.py (new)
  Overlap: NONE — all new files, no shared modifications
  Verdict: SAFE for parallel (3 tracks, all file-disjoint)
-->

### Track A: STAC Catalog
<!-- execution_mode: sequential (within track) -->
<!-- network: REQUIRED — tasks 1-5 can be coded offline, but task 6 (verify) requires internet to reach planet.com STAC catalog -->

- [x] Create `src/tanager/catalog.py` with `list_fire_scenes()` that traverses the static STAC catalog via pystac, returns scene items with ID, datetime, bbox, and asset keys
  <!-- files: src/tanager/catalog.py (new) -->
  <!-- pattern: use pystac.Catalog.from_file("https://www.planet.com/data/stac/tanager-core-imagery/catalog.json"), then catalog.get_child("fire"), then iterate items. See research/tanager-data-access-evaluation.md Section 1 and Section 7 for exact code pattern. -->
  <!-- gotcha: this is a STATIC catalog — use pystac, NOT pystac-client. No /search endpoint exists. -->

- [x] Add `list_fire_scenes(start_date, end_date)` date range filtering
  <!-- files: src/tanager/catalog.py (modify) -->
  <!-- gotcha: STAC item.datetime is a Python datetime object. Parse start_date/end_date strings to datetime for comparison. Handle timezone awareness — STAC datetimes may be tz-aware (UTC). -->

- [x] Add `get_scene_metadata(item)` returning structured metadata dict (scene_id, datetime, bbox, product_types, file_size_mb)
  <!-- files: src/tanager/catalog.py (modify) -->
  <!-- gotcha: file_size_mb may not be available in STAC metadata for all assets. Return None for file_size_mb if not present rather than erroring. product_types = list(item.assets.keys()). -->

- [x] Add `download_scene(item, product_type, output_dir, overwrite)` with streaming download, skip-existing logic, and progress logging
  <!-- files: src/tanager/catalog.py (modify) -->
  <!-- pattern: follow research/tanager-data-access-evaluation.md Section 7 Step 2 for download pattern. Use requests.get(url, stream=True) with iter_content(chunk_size=8192). Log via Python logging module, not print(). -->
  <!-- gotcha: no auth required. Files are ~480 MB each. Use Content-Length header for progress percentage if available. Default overwrite=False. Return Path object for downloaded file. -->

- [x] Add `ConnectionError` handling when catalog URL is unreachable
  <!-- files: src/tanager/catalog.py (modify) -->
  <!-- gotcha: wrap pystac.Catalog.from_file() call in try/except for requests.exceptions.ConnectionError and pystac errors. Re-raise as ConnectionError with descriptive message including the URL that failed. -->

- [x] Verify: `catalog.list_fire_scenes()` returns items from the live STAC catalog
  <!-- verify: manual — requires network. Run `python -c "from tanager.catalog import list_fire_scenes; items = list_fire_scenes(); print(f'{len(items)} scenes found')"`. Expect 11-12 items. Record actual count and update FIRE_SCENES in config.py if needed. -->
  <!-- network: REQUIRED — live STAC catalog query -->

### Track B: Scene I/O
<!-- execution_mode: sequential (within track) -->
<!-- network: none for coding — verify step requires a downloaded .h5 file (local) -->

- [x] Create `src/tanager/io.py` with `load_scene(filepath)` wrapping HyperCoast `read_tanager()`, returning xarray.Dataset with (wavelength, y, x) dims
  <!-- files: src/tanager/io.py (new) -->
  <!-- pattern: `import hypercoast; ds = hypercoast.read_tanager(filepath)`. Verify returned dims are (wavelength, y, x). If HyperCoast returns different dim names, rename to our convention. -->
  <!-- gotcha: HyperCoast read_tanager() signature: filepath, bands=None, stac_url=None, wavelengths=None, product=None, **kwargs. Returns xarray.Dataset with wavelength coordinate in nm. -->

- [x] Add `load_scene(filepath, wavelength_range)` for loading a band subset
  <!-- files: src/tanager/io.py (modify) -->
  <!-- gotcha: HyperCoast's `bands` parameter takes band INDICES, not wavelength values. To support wavelength_range=(min_wl, max_wl), you must: (1) load wavelength metadata first (or load full dataset), (2) find band indices where wavelength is in range, (3) either pass those indices to read_tanager(bands=indices) or load full and slice. Option 2 (load then slice) is simpler and more reliable. If memory is a concern for the caller, document that the full dataset is loaded temporarily. -->

- [x] Add `get_spatial_info(dataset)` returning CRS, bounds, resolution, shape
  <!-- files: src/tanager/io.py (modify) -->
  <!-- gotcha: CRS and geotransform may be stored in dataset.attrs or in coordinate metadata. Check ds.attrs for 'crs', 'epsg', or 'spatial_ref'. Bounds from y/x coordinate min/max. Resolution from coordinate spacing. Shape from ds.dims. Return a dict with keys: crs, bounds, resolution, shape. -->

- [x] Add `ValueError` for invalid/corrupted HDF5 files
  <!-- files: src/tanager/io.py (modify) -->
  <!-- gotcha: catch OSError (h5py/HDF5 read errors) and any HyperCoast exceptions. Re-raise as ValueError with descriptive message including the filepath. -->

- [x] Verify: Load a downloaded fire scene, confirm 426 bands x spatial grid, wavelength coordinate spans 380-2500nm
  <!-- verify: manual — requires a downloaded .h5 file. Run after Track A verify step provides a file. -->
  <!-- network: none — uses local file, but depends on Track A having downloaded a scene first -->

### Track C: Spectral Band Operations
<!-- execution_mode: sequential (within track) -->
<!-- network: none — all operations on in-memory xarray datasets -->

- [x] Create `src/tanager/spectral.py` with `select_bands(dataset, min_wl, max_wl)` for wavelength range selection
  <!-- files: src/tanager/spectral.py (new) -->
  <!-- pattern: use xarray .sel(wavelength=slice(min_wl, max_wl)) or boolean indexing on wavelength coordinate. Return a new xarray.Dataset (do not modify in place). -->

- [x] Add `select_bands(dataset, wavelengths=[...])` for nearest-neighbor band matching
  <!-- files: src/tanager/spectral.py (modify) -->
  <!-- pattern: use xarray .sel(wavelength=wavelengths, method="nearest"). Return both the dataset and the actual matched wavelengths so the caller knows which bands were selected. -->

- [x] Add `ValueError` when no bands match the specified range
  <!-- files: src/tanager/spectral.py (modify) -->

- [x] Add `mask_bad_bands(dataset)` removing sensor edge (<400nm), water vapor (1340-1480nm, 1790-1960nm), and CO2/H2O (2350-2500nm) bands, with logging of excluded/remaining band count
  <!-- files: src/tanager/spectral.py (modify) -->
  <!-- pattern: import BAD_BAND_RANGES from tanager.config. Build boolean mask over wavelength coordinate. Use Python logging (not print) for band count messages. -->
  <!-- gotcha: BAD_BAND_RANGES default is [(0, 400), (1340, 1480), (1790, 1960), (2350, 2500)]. After masking 426 bands, expect ~330-346 remaining. If result is outside this range, log a warning. -->

- [x] Add `mask_bad_bands(dataset, zones=[...])` for custom exclusion zones
  <!-- files: src/tanager/spectral.py (modify) -->
  <!-- gotcha: when zones parameter is provided, it REPLACES the defaults entirely (not additive). This matches the spec scenario "overriding defaults". -->

- [x] Verify: `mask_bad_bands()` on a 426-band dataset returns ~330-346 bands, wavelength coordinate is contiguous
  <!-- verify: can be tested with synthetic data from conftest.py fixture. Does not require real data. -->
  <!-- network: none -->

## Wave 3: Indices, Masks, and Tests
<!-- execution: parallel -->
<!-- PARALLEL SAFETY CHECK
  Track D files: src/tanager/spectral.py (modify — indices + continuum)
  Track E files: src/tanager/masks.py (new)
  Track F files: tests/conftest.py (new), tests/test_spectral.py (new), tests/test_masks.py (new), tests/test_catalog.py (new), tests/test_io.py (new)
  Overlap: Track D modifies spectral.py, Track F tests spectral.py — Track F READS but does not MODIFY spectral.py. Track E creates masks.py, Track F tests masks.py — same read-only relationship.
  HOWEVER: Track F cannot be written until Track D and E are complete (tests reference functions that don't exist yet).
  Verdict: Track D + Track E are SAFE for parallel (file-disjoint). Track F MUST be sequential AFTER D and E.
-->

### Track D: Spectral Indices and Continuum Removal
<!-- execution_mode: sequential (within track) -->
<!-- network: none — all operations on in-memory xarray datasets -->

- [ ] Add `nbr(dataset)` computing (NIR_860 - SWIR_2200) / (NIR_860 + SWIR_2200), returning DataArray
  <!-- files: src/tanager/spectral.py (modify) -->
  <!-- pattern: import BAND_ALIASES from tanager.config to get NIR=860nm, SWIR2=2200nm. Use select_bands(dataset, wavelengths=[860]) to get nearest band. Compute normalized difference. -->
  <!-- gotcha: use .sel(wavelength=860, method="nearest") for band lookup. Tanager's 5nm spacing means the nearest band to 860nm will be within 2.5nm. -->

- [ ] Add `ndvi(dataset)` computing (NIR_860 - Red_660) / (NIR_860 + Red_660)
  <!-- files: src/tanager/spectral.py (modify) -->

- [ ] Add `ndwi(dataset)` computing (Green_560 - NIR_860) / (Green_560 + NIR_860)
  <!-- files: src/tanager/spectral.py (modify) -->

- [ ] Add `dnbr(pre, post)` computing NBR_pre - NBR_post
  <!-- files: src/tanager/spectral.py (modify) -->
  <!-- gotcha: pre and post are separate xarray.Datasets. They must have compatible spatial dimensions. Do NOT assume they are co-registered — add a shape check and raise ValueError if spatial dims differ. -->

- [ ] Ensure all index functions return NaN (not Inf) when denominator is zero
  <!-- files: src/tanager/spectral.py (modify) -->
  <!-- pattern: use np.where(denominator == 0, np.nan, numerator / denominator) or xarray's built-in where(). Apply to all normalized difference functions. -->

- [ ] Add `continuum_removal(dataset, wavelength_range)` using convex hull continuum fitting
  <!-- files: src/tanager/spectral.py (modify) -->
  <!-- gotcha: convex hull continuum fitting: (1) extract reflectance spectrum for wavelength range, (2) compute upper convex hull of (wavelength, reflectance) points, (3) interpolate hull to all wavelengths, (4) divide reflectance by hull values. Use scipy.spatial.ConvexHull or an iterative approach. Result should be in [0, 1] but floating-point may produce values slightly > 1.0 at hull vertices — clip with np.minimum(result, 1.0). When wavelength_range is None, apply to full spectrum. -->
  <!-- gotcha: this must work per-pixel. For a spatial dataset, vectorize over (y, x) dimensions. Consider np.apply_along_axis or xarray.apply_ufunc for performance. -->
  <!-- dep: add scipy to pyproject.toml dependencies if not already present — needed for ConvexHull -->

- [ ] Verify: Compute NBR on a real fire scene, confirm values in [-1, 1] range, NaN only where masked
  <!-- verify: manual — requires downloaded scene. Can also be partially verified with synthetic data. -->
  <!-- network: none — uses local file -->

### Track E: Masking Utilities
<!-- execution_mode: sequential (within track) -->
<!-- network: none — all operations on in-memory xarray datasets -->

- [ ] Create `src/tanager/masks.py` with `nodata_mask(dataset, fill_value)` returning boolean DataArray (True=valid)
  <!-- files: src/tanager/masks.py (new) -->
  <!-- pattern: check all bands — a pixel is valid only if ALL bands have finite, non-NaN values. Use xarray's .notnull().all(dim="wavelength"). If fill_value is provided, also mask pixels where any band equals fill_value. -->
  <!-- gotcha: fill_value parameter should default to None (NaN-only check). When fill_value=-9999, check with == comparison before the NaN check. -->

- [ ] Add `cloud_mask(dataset)` reading beta_cirrus_mask from HDF5 metadata, with fallback to all-True when field is absent
  <!-- files: src/tanager/masks.py (modify) -->
  <!-- gotcha: HyperCoast's read_tanager() may NOT expose beta_cirrus_mask in the xarray dataset. The coder needs to: (1) check if the field exists in dataset or dataset.attrs, (2) if not, open the source HDF5 file directly with h5py to read /HDFEOS/SWATHS/HYP/Metadata/beta_cirrus_mask (or similar path), (3) if the field doesn't exist in HDF5 either, return all-True mask with a warning log. This requires h5py (added to dependencies in Section 1 task 1). The function signature should accept either a dataset (with a .encoding["source"] or filepath attr) or a filepath string. -->

- [ ] Add `water_mask(dataset, threshold)` using NDWI computation
  <!-- files: src/tanager/masks.py (modify) -->
  <!-- pattern: import ndwi from tanager.spectral. Compute NDWI, return boolean DataArray where True = land (NDWI <= threshold). Default threshold=0.3 per spec. -->
  <!-- gotcha: circular dependency risk — masks.py imports from spectral.py. This is fine as long as spectral.py does NOT import from masks.py. Verify this remains true. -->

- [ ] Add `apply_masks(dataset, mask_list)` applying logical AND of all masks, setting masked pixels to NaN
  <!-- files: src/tanager/masks.py (modify) -->
  <!-- pattern: combine = reduce(lambda a, b: a & b, mask_list). Then dataset.where(combine). This sets all masked pixels to NaN across all bands. -->

- [ ] Verify: Apply combined mask to a fire scene, confirm masked pixels are NaN and unmasked pixels retain original values
  <!-- verify: manual with real data, or with synthetic data using known mask regions. -->
  <!-- network: none -->

### Track F: Test Suite (sequential — depends on Tracks D + E)
<!-- execution_mode: sequential -->
<!-- blocked_by: Track D (spectral indices), Track E (masks) -->
<!-- network: none — all tests use synthetic data or mocked HTTP -->

- [ ] Create `tests/conftest.py` with `synthetic_tanager_dataset()` fixture: 426 bands, 380-2500nm, 50x50 pixels, Float32 reflectance [0,1]
  <!-- files: tests/conftest.py (new) -->
  <!-- pattern: use xarray.Dataset with coords: wavelength=np.linspace(380, 2500, 426), y=range(50), x=range(50). Data variable "reflectance" with random Float32 values clipped to [0, 1]. Add wavelength as a proper coordinate. -->

- [ ] Add `synthetic_tanager_dataset(signatures=["vegetation", "char", "soil"])` with known spectral profiles in specific pixel regions
  <!-- files: tests/conftest.py (modify) -->
  <!-- gotcha: vegetation signature — high NIR plateau (750-1300nm ~0.4-0.5), chlorophyll absorption dip at 680nm (~0.05), cellulose features at 2100nm. Char signature — low flat reflectance (~0.02-0.05), slight rise in SWIR. Soil signature — monotonic increase VNIR to SWIR (~0.1-0.3). Place each signature in a distinct pixel block (e.g., vegetation at y[0:15], char at y[15:30], soil at y[30:45], random at y[45:50]). -->

- [ ] Create `tests/test_spectral.py` — test band selection, bad band masking, spectral indices (NBR, NDVI, NDWI, dNBR), continuum removal, division-by-zero handling
  <!-- files: tests/test_spectral.py (new) -->
  <!-- test: verify select_bands returns correct wavelength range; mask_bad_bands removes ~80-96 bands; NBR of vegetation signature is positive (healthy veg has high NIR); NBR of char is negative or near-zero; division by zero returns NaN not Inf; continuum_removal output in [0, 1]. -->

- [ ] Create `tests/test_masks.py` — test nodata, cloud, water, and combined mask application
  <!-- files: tests/test_masks.py (new) -->
  <!-- test: nodata_mask correctly identifies NaN pixels; fill_value=-9999 is caught; water_mask with NDWI threshold separates water pixels; apply_masks with multiple masks produces logical AND; masked pixels are NaN in output. -->

- [ ] Create `tests/test_catalog.py` — test STAC browsing, date filtering, metadata extraction with mocked HTTP responses
  <!-- files: tests/test_catalog.py (new) -->
  <!-- pattern: use unittest.mock.patch or pytest-mock to mock pystac.Catalog.from_file(). Create mock STAC items with known IDs, datetimes, bboxes, and assets. Test list_fire_scenes returns all items, date filtering works, get_scene_metadata extracts correct fields, ConnectionError is raised when catalog is unreachable. -->

- [ ] Create `tests/test_io.py` — test scene loading, band subsetting, spatial info extraction, and invalid file handling with mocked HyperCoast
  <!-- files: tests/test_io.py (new) -->
  <!-- pattern: use unittest.mock.patch to mock hypercoast.read_tanager(). Return a synthetic xarray.Dataset from the mock. Test load_scene returns correct dims (wavelength, y, x); load_scene with wavelength_range returns subset; get_spatial_info extracts CRS, bounds, resolution, shape; load_scene with invalid path raises ValueError. -->

- [ ] Verify: `pytest tests/` passes with all tests green
  <!-- verify: automated — `pytest tests/ -v` should pass. -->
  <!-- network: none — all tests use mocks or synthetic data -->
