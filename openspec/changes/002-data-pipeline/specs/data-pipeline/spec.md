# Capability: Data Pipeline

**Capability ID:** data-pipeline
**Status:** Proposed (via change 002-data-pipeline)

---

## ADDED Requirements

### Requirement: STAC Catalog Browsing

The system SHALL provide a Python interface to browse the Planet Tanager static STAC catalog and discover available scenes.

#### Scenario: List fire collection scenes

**WHEN** a user calls `catalog.list_fire_scenes()`
**THEN** the function returns a list of STAC items from the fire collection, each containing: scene ID, datetime, bbox, and available asset keys.

#### Scenario: Filter scenes by date range

**WHEN** a user calls `catalog.list_fire_scenes(start_date="2025-01-01", end_date="2025-04-30")`
**THEN** only scenes with datetime within the specified range are returned.

#### Scenario: Get scene metadata

**WHEN** a user calls `catalog.get_scene_metadata(item)`
**THEN** the function returns a dictionary with keys: `scene_id`, `datetime`, `bbox`, `product_types` (list of available asset keys), and `file_size_mb` (if available from STAC metadata).

#### Scenario: Catalog unavailable

**WHEN** the STAC catalog URL is unreachable
**THEN** the function raises a `ConnectionError` with a message indicating the catalog URL that failed and suggesting the user check network connectivity.

---

### Requirement: Scene Download

The system SHALL support downloading Tanager scene files from the STAC catalog without authentication.

#### Scenario: Download ortho surface reflectance

**WHEN** a user calls `catalog.download_scene(item, product_type="ortho_surface_reflectance", output_dir="data/raw")`
**THEN** the HDF5 file is downloaded to `output_dir/` with the original filename, and the function returns the local file path.

#### Scenario: Skip existing downloads

**WHEN** a file with the same name already exists in `output_dir` and `overwrite=False` (default)
**THEN** the download is skipped and the existing file path is returned, with a log message indicating the file was already present.

#### Scenario: Download progress

**WHEN** a download is in progress
**THEN** progress SHALL be reported via Python logging at INFO level, including file size and percentage if Content-Length is available from the server.

---

### Requirement: Scene Loading

The system SHALL load Tanager HDF5 scenes into xarray.Dataset using HyperCoast.

#### Scenario: Load a single scene

**WHEN** a user calls `io.load_scene(filepath)` with a valid Tanager HDF5 file
**THEN** the function returns an `xarray.Dataset` with dimensions `(wavelength, y, x)`, where `wavelength` contains center wavelengths in nanometers and spatial coordinates are in the scene's native CRS.

#### Scenario: Load with band subset

**WHEN** a user calls `io.load_scene(filepath, wavelength_range=(900, 1400))`
**THEN** only bands with center wavelengths between 900nm and 1400nm are loaded, reducing memory usage.

#### Scenario: Extract scene spatial metadata

**WHEN** a user calls `io.get_spatial_info(dataset)`
**THEN** the function returns a dictionary with keys: `crs` (EPSG code), `bounds` (xmin, ymin, xmax, ymax), `resolution` (pixel size in meters), and `shape` (rows, cols).

#### Scenario: Invalid file

**WHEN** a user calls `io.load_scene(filepath)` with a non-HDF5 file or corrupted file
**THEN** the function raises a `ValueError` with a descriptive error message.

---

### Requirement: Band Selection

The system SHALL support selecting spectral bands by wavelength range from an xarray.Dataset.

#### Scenario: Select bands by wavelength range

**WHEN** a user calls `spectral.select_bands(dataset, min_wl=900, max_wl=1300)`
**THEN** the function returns a new xarray.Dataset containing only bands whose center wavelength falls within [900, 1300] nm inclusive.

#### Scenario: Select bands by specific wavelengths

**WHEN** a user calls `spectral.select_bands(dataset, wavelengths=[970, 1200, 1680])`
**THEN** the function returns a dataset with the bands closest to each requested wavelength (nearest-neighbor matching), along with the actual matched wavelengths.

#### Scenario: No bands in range

**WHEN** a user specifies a wavelength range that contains no bands (e.g., `min_wl=2600, max_wl=2700`)
**THEN** the function raises a `ValueError` indicating no bands were found in the specified range.

---

### Requirement: Bad Band Masking

The system SHALL mask or remove bands in known low-quality spectral regions, including sensor edge noise and atmospheric absorption zones.

#### Scenario: Apply default bad band mask

**WHEN** a user calls `spectral.mask_bad_bands(dataset)`
**THEN** bands in the following regions are removed: sensor edge (<400nm), water vapor 1 (1340-1480nm), water vapor 2 (1790-1960nm), and CO2/H2O overlap (2350-2500nm). A log message reports how many bands were excluded and how many remain (~330-346 usable bands expected).

#### Scenario: Custom exclusion zones

**WHEN** a user calls `spectral.mask_bad_bands(dataset, zones=[(1340, 1480), (1790, 1960)])`
**THEN** only the specified zones are excluded (overriding defaults), allowing the user to customize bad band masking.

#### Scenario: Preserve wavelength metadata

**WHEN** bad bands are excluded
**THEN** the `wavelength` coordinate in the returned dataset accurately reflects the remaining bands, with no gaps or incorrect values.

#### Scenario: Get bad band definitions

**WHEN** a user calls `config.BAD_BAND_RANGES`
**THEN** a list of tuples is returned defining the default bad band regions: `[(0, 400), (1340, 1480), (1790, 1960), (2350, 2500)]`.

---

### Requirement: Spectral Index Computation

The system SHALL compute standard spectral indices from hyperspectral data.

#### Scenario: Compute NBR

**WHEN** a user calls `spectral.nbr(dataset)`
**THEN** the function computes the Normalized Burn Ratio using bands nearest to 860nm (NIR) and 2200nm (SWIR), returning an `xarray.DataArray` with the same spatial dimensions.

#### Scenario: Compute dNBR

**WHEN** a user calls `spectral.dnbr(pre_fire_dataset, post_fire_dataset)`
**THEN** the function computes `NBR_pre - NBR_post`, returning an `xarray.DataArray` where positive values indicate burn severity.

#### Scenario: Compute NDVI

**WHEN** a user calls `spectral.ndvi(dataset)`
**THEN** the function computes `(NIR - Red) / (NIR + Red)` using bands nearest to 860nm and 660nm.

#### Scenario: Compute NDWI

**WHEN** a user calls `spectral.ndwi(dataset)`
**THEN** the function computes `(Green - NIR) / (Green + NIR)` using bands nearest to 560nm and 860nm.

#### Scenario: Division by zero handling

**WHEN** NIR + SWIR (or equivalent denominator) equals zero for a pixel
**THEN** the index value for that pixel SHALL be NaN, not infinity or an error.

---

### Requirement: Continuum Removal

The system SHALL implement continuum removal for absorption feature analysis.

#### Scenario: Apply continuum removal

**WHEN** a user calls `spectral.continuum_removal(dataset, wavelength_range=(900, 1100))`
**THEN** a convex hull continuum is fitted to the specified wavelength range and the reflectance is divided by the continuum, returning values in [0, 1] where absorption features appear as dips below 1.0.

#### Scenario: Full spectrum continuum removal

**WHEN** a user calls `spectral.continuum_removal(dataset)` without a wavelength range
**THEN** continuum removal is applied across the full available wavelength range.

---

### Requirement: No-Data Masking

The system SHALL detect and mask no-data pixels in Tanager scenes.

#### Scenario: Mask NaN values

**WHEN** a user calls `masks.nodata_mask(dataset)`
**THEN** the function returns a boolean xarray.DataArray (True = valid, False = no-data) that is True only for pixels where all bands have finite, non-NaN values.

#### Scenario: Mask fill values

**WHEN** a user calls `masks.nodata_mask(dataset, fill_value=-9999)`
**THEN** pixels where any band equals the fill value are also masked as no-data.

---

### Requirement: Cloud and Cirrus Masking

The system SHALL provide cloud masking for Tanager scenes.

#### Scenario: Cloud mask from HDF5 metadata

**WHEN** a user calls `masks.cloud_mask(dataset)` and the source HDF5 file contains a `beta_cirrus_mask` field
**THEN** the function returns a boolean DataArray where True = clear, False = cloud/cirrus.

#### Scenario: Cloud mask fallback

**WHEN** the HDF5 file does not contain a `beta_cirrus_mask` field
**THEN** the function logs a warning and returns an all-True mask (no masking applied), rather than raising an error.

---

### Requirement: Water Body Masking

The system SHALL identify and mask water body pixels.

#### Scenario: NDWI-based water mask

**WHEN** a user calls `masks.water_mask(dataset, threshold=0.3)`
**THEN** the function computes NDWI and returns a boolean DataArray where True = land, False = water (NDWI > threshold).

---

### Requirement: Combined Mask Application

The system SHALL support applying multiple masks to a dataset simultaneously.

#### Scenario: Apply combined mask

**WHEN** a user calls `masks.apply_masks(dataset, [nodata_mask, cloud_mask, water_mask])`
**THEN** the function applies the logical AND of all masks (pixel must pass all masks to be retained), setting masked pixels to NaN across all bands.

---

### Requirement: Configuration Constants

The system SHALL provide centralized configuration for Tanager-1 sensor parameters, fire scene metadata, and standard wavelength aliases.

#### Scenario: Access sensor parameters

**WHEN** a user imports `from tanager.config import SENSOR`
**THEN** a namespace or dictionary is available with keys: `n_bands` (426), `spectral_range` ((380, 2500)), `spectral_sampling` (5.0), `fwhm` (5.5), `gsd` (30), and `data_format` ("HDF-EOS5").

#### Scenario: Access fire scene catalog

**WHEN** a user imports `from tanager.config import FIRE_SCENES`
**THEN** a list of dictionaries is returned, each containing: `scene_id`, `date`, `phase` (pre-fire, post-fire, early-recovery, mid-recovery, late-recovery), and `bbox`. The catalog MUST contain all 12 known fire scenes.

#### Scenario: Access band aliases

**WHEN** a user imports `from tanager.config import BAND_ALIASES`
**THEN** a dictionary mapping common names to center wavelengths is available: `Red` (660nm), `Red_Edge` (720nm), `NIR` (860nm), `SWIR1` (1650nm), `SWIR2` (2200nm), `Green` (560nm), `Blue` (480nm).

#### Scenario: Access default data directory

**WHEN** a user imports `from tanager.config import DATA_DIR`
**THEN** a `Path` object pointing to `data/raw/fire/` relative to the project root is returned. The path can be overridden by setting the `TANAGER_DATA_DIR` environment variable.

---

### Requirement: Test Fixtures for Spectral Data

The test suite SHALL provide synthetic xarray.Dataset fixtures that mimic Tanager scene structure.

#### Scenario: Synthetic Tanager dataset

**WHEN** a test calls `synthetic_tanager_dataset()` from conftest.py
**THEN** the function returns an xarray.Dataset with 426 bands spanning 380-2500nm at ~5nm spacing, spatial dimensions of 50x50 pixels, and Float32 reflectance values in [0, 1].

#### Scenario: Known spectral signatures

**WHEN** a test calls `synthetic_tanager_dataset(signatures=["vegetation", "char", "soil"])`
**THEN** specific pixel regions contain reflectance profiles matching published spectral signatures for those materials, enabling deterministic testing of spectral indices and classification.
