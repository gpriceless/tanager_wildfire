# Change: Add Data Pipeline & Project Scaffolding

**Change ID:** 002-data-pipeline
**Plane Issue:** TANAGER-5
**Status:** Proposed
**Author:** Product Queen
**Date:** 2026-04-27

---

## Why

FireSpec has no code yet. Phase 1 research is complete — we know what to build (MESMA burn severity + PLSR LFMC) and how to access the data (static STAC catalog, HyperCoast for I/O, ~6 GB of ortho SR fire scenes). Before any analysis can happen, we need a working Python package that can discover Tanager scenes, download them, load them into xarray, and perform basic spectral preprocessing (band selection, atmospheric exclusion, masking, spectral indices). This change establishes the foundation that every subsequent Phase 3 analysis module depends on.

## What Changes

### Phase 2A: Project Scaffolding
- Python package at `src/tanager/` with `pyproject.toml`
- All dependencies declared: hypercoast, spectral, rasterio, xarray, geopandas, scikit-learn, pystac, requests, spyndex
- Dev dependencies: pytest, ruff, mypy
- Package is installable via `pip install -e .`

### Phase 2B: Data Access Pipeline
- `src/tanager/catalog.py` — STAC catalog interface using pystac
  - Browse static catalog collections
  - Filter fire scenes by date range and product type
  - Extract scene metadata (dates, bbox, product type, scene ID)
  - Download ortho SR assets (no auth required)
- `src/tanager/io.py` — Tanager data I/O module
  - Load scenes via HyperCoast `read_tanager()` into xarray.Dataset
  - Scene metadata extraction from xarray (wavelengths, spatial extent, CRS)
  - Batch loading support for temporal analysis

### Phase 2C: Spectral Preprocessing
- `src/tanager/spectral.py` — Spectral preprocessing utilities
  - Band selection by wavelength range (return subset of xarray along wavelength dim)
  - Bad band masking: sensor edge (<400nm), water vapor (1340-1480nm, 1790-1960nm), CO2/H2O overlap (2350-2500nm)
  - Continuum removal for absorption feature analysis
  - Spectral index computation: NBR, NDVI, NDWI, dNBR
- `src/tanager/masks.py` — Masking utilities
  - No-data masking (NaN and fill-value detection)
  - Cloud/cirrus masking (from HDF5 beta_cirrus_mask if available)
  - Water body masking (NDWI threshold)
  - Combined mask application to xarray datasets

### Phase 2D: Configuration
- `src/tanager/config.py` — Project configuration and constants
  - Tanager-1 sensor parameters (426 bands, 5nm spacing, 380-2500nm, FWHM 5.5nm)
  - Bad band definitions (sensor edge, atmospheric absorption zones)
  - Fire collection scene catalog (12 scenes with IDs, dates, phases, bboxes)
  - Band wavelength aliases for common spectral features (Red, NIR, SWIR1, SWIR2, etc.)
  - Default data directory paths

### Phase 2E: Validation Foundation
- `tests/conftest.py` — Shared test fixtures
  - Synthetic xarray.Dataset mimicking Tanager structure (426 bands, 380-2500nm)
  - Known-signature spectral profiles (vegetation, char, soil, water)
- `tests/test_spectral.py` — Unit tests for spectral preprocessing
- `tests/test_masks.py` — Unit tests for masking utilities
- `tests/test_catalog.py` — Unit tests for STAC catalog (mocked HTTP)

## Impact

- **Affected specs:** None (first spec being created)
- **Affected code:** All new files — no existing code to modify
- **Dependencies introduced:** hypercoast, spectral, rasterio, xarray, geopandas, scikit-learn, pystac, requests, spyndex, pytest, ruff, mypy
- **Storage:** Fire collection download will require ~6 GB disk space

## Research Summary

Two research reports inform this change:

1. **Literature Review** (`research/wildfire-hyperspectral-literature-review.md`): Established the MESMA + PLSR approach as proven. Key wavelengths identified: 970nm, 1200nm (water absorption), 1680nm (lignin), 2100nm, 2280nm (cellulose). Bad band zones confirmed from research-memory: sensor edge (<400nm), 1340-1480nm, 1790-1960nm, 2350-2500nm. After masking, ~330-346 usable bands remain.

2. **Data Access Evaluation** (`research/tanager-data-access-evaluation.md`): Confirmed static STAC catalog at `planet.com/data/stac/tanager-core-imagery/catalog.json`, no auth required. Use `pystac` (NOT `pystac-client`). HyperCoast `read_tanager()` returns xarray.Dataset with dims `(wavelength, y, x)`. 12 fire scenes spanning Dec 2024 - Sep 2025 across 5 temporal phases. Ortho Surface Reflectance is the primary product (~480 MB/scene).

## Production Risk

Not applicable — this is a research project, not a production service.

## Open Questions for EM

1. **HyperCoast version pinning:** Should we pin hypercoast to a specific version (0.22.0 tested in research) or use latest? HyperCoast is under active development and the `read_tanager()` API may shift.
2. **Data directory convention:** Research recommends `data/` in project root. Should downloaded scenes go to `data/raw/fire/` with a `.gitignore`? Or use an environment variable for data path?
3. **SPy vs mesma package:** Literature review identified both `spectral` (SPy) and `mesma` v1.0.8 as MESMA candidates. For Phase 2 we only need SPy (spectral indices, band math). The MESMA package decision can be deferred to Phase 3. Confirm this is acceptable.
