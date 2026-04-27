# Change: 002-data-pipeline

## Section 1: Project Scaffolding
<!-- execution_mode: sequential -->

- [ ] Create `pyproject.toml` with project metadata, Python 3.10+ requirement, and all dependencies (hypercoast, spectral, rasterio, xarray, geopandas, scikit-learn, pystac, requests, spyndex) plus dev dependencies (pytest, ruff, mypy)
- [ ] Create `src/tanager/__init__.py` with package version and top-level imports
- [ ] Create `src/tanager/config.py` with sensor parameters (SENSOR), bad band ranges (BAD_BAND_RANGES), fire scene catalog (FIRE_SCENES), band wavelength aliases (BAND_ALIASES), and data directory path (DATA_DIR with TANAGER_DATA_DIR env override)
- [ ] Create `data/raw/fire/.gitkeep` and add `data/raw/fire/*.h5` to `.gitignore`
- [ ] Verify: `pip install -e .` succeeds and `import tanager` works

## Section 2: Data Access — Discover and Download Scenes
<!-- execution_mode: sequential -->

- [ ] Create `src/tanager/catalog.py` with `list_fire_scenes()` that traverses the static STAC catalog via pystac, returns scene items with ID, datetime, bbox, and asset keys
- [ ] Add `list_fire_scenes(start_date, end_date)` date range filtering
- [ ] Add `get_scene_metadata(item)` returning structured metadata dict (scene_id, datetime, bbox, product_types, file_size_mb)
- [ ] Add `download_scene(item, product_type, output_dir, overwrite)` with streaming download, skip-existing logic, and progress logging
- [ ] Add `ConnectionError` handling when catalog URL is unreachable
- [ ] Verify: `catalog.list_fire_scenes()` returns 12 items from the live STAC catalog

## Section 3: Data Access — Load and Inspect Scenes
<!-- execution_mode: sequential -->

- [ ] Create `src/tanager/io.py` with `load_scene(filepath)` wrapping HyperCoast `read_tanager()`, returning xarray.Dataset with (wavelength, y, x) dims
- [ ] Add `load_scene(filepath, wavelength_range)` for loading a band subset
- [ ] Add `get_spatial_info(dataset)` returning CRS, bounds, resolution, shape
- [ ] Add `ValueError` for invalid/corrupted HDF5 files
- [ ] Verify: Load a downloaded fire scene, confirm 426 bands x spatial grid, wavelength coordinate spans 380-2500nm

## Section 4: Spectral Preprocessing — Band Operations
<!-- execution_mode: sequential -->

- [ ] Create `src/tanager/spectral.py` with `select_bands(dataset, min_wl, max_wl)` for wavelength range selection
- [ ] Add `select_bands(dataset, wavelengths=[...])` for nearest-neighbor band matching
- [ ] Add `ValueError` when no bands match the specified range
- [ ] Add `mask_bad_bands(dataset)` removing sensor edge (<400nm), water vapor (1340-1480nm, 1790-1960nm), and CO2/H2O (2350-2500nm) bands, with logging of excluded/remaining band count
- [ ] Add `mask_bad_bands(dataset, zones=[...])` for custom exclusion zones
- [ ] Verify: `mask_bad_bands()` on a 426-band dataset returns ~330-346 bands, wavelength coordinate is contiguous

## Section 5: Spectral Preprocessing — Indices and Continuum Removal
<!-- execution_mode: sequential -->

- [ ] Add `nbr(dataset)` computing (NIR_860 - SWIR_2200) / (NIR_860 + SWIR_2200), returning DataArray
- [ ] Add `ndvi(dataset)` computing (NIR_860 - Red_660) / (NIR_860 + Red_660)
- [ ] Add `ndwi(dataset)` computing (Green_560 - NIR_860) / (Green_560 + NIR_860)
- [ ] Add `dnbr(pre, post)` computing NBR_pre - NBR_post
- [ ] Ensure all index functions return NaN (not Inf) when denominator is zero
- [ ] Add `continuum_removal(dataset, wavelength_range)` using convex hull continuum fitting
- [ ] Verify: Compute NBR on a real fire scene, confirm values in [-1, 1] range, NaN only where masked

## Section 6: Masking — Quality Flags
<!-- execution_mode: sequential -->

- [ ] Create `src/tanager/masks.py` with `nodata_mask(dataset, fill_value)` returning boolean DataArray (True=valid)
- [ ] Add `cloud_mask(dataset)` reading beta_cirrus_mask from HDF5 metadata, with fallback to all-True when field is absent
- [ ] Add `water_mask(dataset, threshold)` using NDWI computation
- [ ] Add `apply_masks(dataset, mask_list)` applying logical AND of all masks, setting masked pixels to NaN
- [ ] Verify: Apply combined mask to a fire scene, confirm masked pixels are NaN and unmasked pixels retain original values

## Section 7: Test Suite
<!-- execution_mode: sequential -->

- [ ] Create `tests/conftest.py` with `synthetic_tanager_dataset()` fixture: 426 bands, 380-2500nm, 50x50 pixels, Float32 reflectance [0,1]
- [ ] Add `synthetic_tanager_dataset(signatures=["vegetation", "char", "soil"])` with known spectral profiles in specific pixel regions
- [ ] Create `tests/test_spectral.py` — test band selection, bad band masking, spectral indices (NBR, NDVI, NDWI, dNBR), continuum removal, division-by-zero handling
- [ ] Create `tests/test_masks.py` — test nodata, cloud, water, and combined mask application
- [ ] Create `tests/test_catalog.py` — test STAC browsing, date filtering, metadata extraction with mocked HTTP responses
- [ ] Verify: `pytest tests/` passes with all tests green
