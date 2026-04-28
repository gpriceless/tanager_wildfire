# Engineering Memory: Tanager Competition

> Long-term memory for Engineering Manager. Tracks architecture, tech debt, and code quality.

**Location:** `/docs/engineering-memory.md`
**Owner:** Engineering Manager (Crenshaw)
**Updated:** 2026-04-28
**Version:** 3.3 (Phase 3 — Wave 4 Validation & Integration in progress; validation module + 5 test files merged)

---

## Purpose

This document is the Engineering Manager's working memory. It tracks:
1. **What exists** — Prevent duplicate implementations
2. **Architecture decisions** — What to use, what to avoid
3. **Tech debt** — Known issues and their status
4. **Patterns** — How things should be done

**All coding agents should check this before building new features.**

---

## Architecture Overview

### Status: Phase 3 — Core Analysis (003-core-analysis) — Wave 4 in progress (validation.py landed, full pytest 331/332 green)

Python package at `src/tanager/`, editable install via `pip install -e .`.

### Module Registry

```
src/tanager/
├── __init__.py       # Package version, lazy public API imports
├── config.py         # Sensor params (SENSOR), bad bands, fire scene catalog, band aliases, DATA_DIR
├── catalog.py        # STAC catalog interface — list, filter, download fire scenes via pystac
├── io.py             # Scene I/O — load HDF5 via HyperCoast read_tanager(), spatial info extraction
├── spectral.py       # Band selection, bad band masking, spectral indices (NBR/NDVI/NDWI/dNBR), continuum removal
├── masks.py          # No-data, cloud/cirrus, water body masking, combined mask application
├── endmembers.py     # [Phase 3] Spectral library loading (USGS/ECOSTRESS/FRAMES), resampling, In-CoB/EAR-MASA selection
├── unmixing.py       # [Phase 3] MESMA spectral unmixing (mesma v1.0.8 primary, HySUPP FCLS fallback), band selection, fraction maps
├── severity.py       # [Phase 3] Burn severity mapping — RF regression fractions→CBI, classification, temporal trajectories
├── lfmc.py           # [Phase 3] Live fuel moisture content — SAI indices, PLSR regression, Globe-LFMC integration
└── validation.py     # [Phase 3] Accuracy metrics (R², RMSE, Kappa), AVIRIS-3/BARC reference loading, sensor comparison

tests/
├── conftest.py           # Synthetic 426-band xarray.Dataset fixtures with known spectral signatures
├── test_spectral.py      # Band selection, bad bands, indices, continuum removal, div-by-zero
├── test_masks.py         # No-data, cloud, water, combined mask tests
├── test_catalog.py       # STAC browsing/filtering with mocked HTTP
├── test_io.py            # Scene loading with mocked HyperCoast
├── test_endmembers.py    # [Phase 3] Library loading (mocked), resampling dims, selection, pruning
├── test_unmixing.py      # [Phase 3] MESMA on synthetic pure pixels, constraint filtering, shade norm
├── test_severity.py      # [Phase 3] RF training, prediction ranges, classification, trajectories
├── test_lfmc.py          # [Phase 3] SAI computation, PLSR, Globe-LFMC loader
└── test_validation.py    # [Phase 3] Accuracy metrics, spatial aggregation, sensor comparison
```

### Key Dependencies

| Library | Purpose | Version Constraint | Notes |
|---------|---------|-------------------|-------|
| HyperCoast | Tanager HDF5 I/O | `>=0.22.0,<1.0` | `read_tanager()` — API may shift pre-1.0 |
| spectral (SPy) | Spectral algorithms | Latest | SAM, BandResampler, EcostressDatabase, PPI |
| rasterio | Raster I/O | >=1.3 | Geospatial raster handling |
| xarray | N-dim arrays | Latest | Hyperspectral cube handling |
| geopandas | Vector ops | >=0.12 | Output geometries, Globe-LFMC GeoDataFrame |
| pystac | STAC catalog | Latest | Static catalog traversal (NOT pystac-client) |
| requests | HTTP downloads | Latest | Scene file downloads, no auth required |
| spyndex | Spectral indices | Latest (0.10.0+) | Reference/validation, not core computation |
| h5py | HDF5 access | Latest | Required for cloud_mask beta_cirrus_mask reading |
| scikit-learn | ML | Latest | RF (severity), PLSRegression (LFMC), cross-validation |
| scipy | Scientific computing | >=1.10 | ConvexHull for continuum removal |
| mesma | MESMA unmixing | >=1.0.8 | **OPTIONAL DEP** — may not be numpy 2.x compatible; HySUPP fallback |
| spectral-libraries | Endmember selection | >=1.1.3 | EAR/MASA/CoB pruning via EarMasaCob class |
| splib07-loader | USGS v7 loader | git+https | Third-party, pure Python, GitHub only |
| joblib | Model serialization | Latest | Transitive dep of scikit-learn; used for RF/PLSR persistence |

**Dev dependencies:** pytest, ruff, mypy

### Data Convention

- **Default data directory:** `data/raw/fire/` relative to project root
- **Override:** `TANAGER_DATA_DIR` environment variable
- **File extension:** `.h5` (HDF-EOS5), not `.hdf5`
- **Storage:** ~480 MB per scene, ~6 GB for full fire collection (ortho SR only)
- **gitignore:** `data/raw/` glob covers all raw data; explicit `*.h5` also added
- **Endmember libraries:** ~100 MB (USGS, ECOSTRESS, FRAMES combined)
- **Globe-LFMC database:** ~50 MB
- **AVIRIS-3 validation data:** ~2 GB (if available from ORNL DAAC)

---

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Package layout | `src/tanager/` (src layout) | Editable install, clean namespace |
| Data format | xarray for hyperspectral cubes | 426 bands = N-dimensional, xarray is standard |
| I/O layer | HyperCoast `read_tanager()` | Already handles HDF-EOS5 layout discovery |
| STAC access | pystac (static catalog) | No STAC API exists — must use static catalog reader |
| Spectral analysis | SPy (spectral-python) | Mature, MESMA/SAM implementations |
| MESMA engine | mesma v1.0.8 (primary), HySUPP FCLS (fallback) | mesma is standard but untested at 426 bands; HySUPP is robust fallback. Engine detection at import time. |
| Endmember library format | xarray DataArray (spectrum_id, wavelength) | Consistent with pipeline conventions; metadata via attrs |
| Endmember selection | In-CoB + EAR/MASA (spectral-libraries v1.1.3) | Roberts et al. (2018) standard methodology |
| Severity regression | scikit-learn RandomForestRegressor | 4-feature problem (fractions→CBI), RF handles nonlinearities |
| LFMC estimation | Tier 1 (spectral indices) + Tier 2 (PLSR) | Indices for interpretability, PLSR for accuracy |
| Shade endmember | Single zero-reflectance spectrum | Standard practice; partial shade as follow-up if needed |
| Sensor config | SimpleNamespace (dot notation) | `SENSOR.n_bands` not `SENSOR["n_bands"]` |
| Index computation | Direct band math (not spyndex) | Full control over band selection; spyndex for validation only |
| Continuum removal | scipy ConvexHull | Standard approach; per-pixel via apply_ufunc |
| Output format | GeoPackage + GeoZarr (Phase 4) | OGC-interoperable, cloud-native |
| Notebooks | Jupyter (Phase 4) | Competition deliverable format |
| HyperCoast version | `>=0.22.0,<1.0` | Floor at latest tested; cap at major version boundary |

---

## Patterns

### Spectral Data Handling
- Always preserve wavelength metadata alongside pixel values
- Use xarray DataArrays with `wavelength` coordinate, not raw numpy
- Wavelengths in nanometers (nm), not micrometers
- Band lookup by wavelength uses `method="nearest"` (5nm spacing = max 2.5nm error)

### Endmember Library Schema
- xarray DataArray with dims (spectrum_id, wavelength)
- Wavelength coordinate in nm, matching target sensor bands after resampling
- Metadata attributes per spectrum: name, category, source
- Category values: char, ash, pv, npv, soil, shade

### Fraction Map Schema
- xarray Dataset with dims (y, x)
- Variables: char, pv, npv, soil, shade, rmse (before normalization)
- Variables: char, pv, npv, soil (after shade normalization)
- Fractions sum to 1.0 within tolerance of 0.01
- NaN for pixels where no valid model was found
- Metadata attribute: unmixing_engine ("mesma" or "hysup" or "nnls")

### Import Direction (dependency rule)
- `config.py` — leaf module, no tanager imports
- `io.py` — independent (only imports HyperCoast)
- `catalog.py` MAY import from `config.py`
- `spectral.py` MAY import from `config.py`
- `masks.py` MAY import from `spectral.py` (for ndwi)
- `spectral.py` MUST NOT import from `masks.py` (circular dependency)
- `endmembers.py` MAY import from `config.py`, `spectral.py`
- `unmixing.py` MAY import from `config.py`, `spectral.py`, `endmembers.py`
- `severity.py` MAY import from `config.py`, `spectral.py`, `unmixing.py`
- `lfmc.py` MAY import from `config.py`, `spectral.py`
- `validation.py` MAY import from any tanager module (top of dependency tree)
- **No module may import FROM validation.py**

### SPy Integration
- SPy expects numpy arrays with shape (rows, cols, bands)
- Our xarray convention is (wavelength, y, x)
- Transpose for SPy: `data.values.transpose(1, 2, 0)`
- SPy BandResampler for spectral resampling (Gaussian FWHM convolution)
- SPy EcostressDatabase for library access — returns wavelengths in micrometers (convert to nm)

### Error Handling
- Division by zero in normalized difference indices: return NaN, never Inf
- Network failures (STAC catalog): catch and re-raise as ConnectionError with URL context
- Invalid HDF5 files: catch and re-raise as ValueError with filepath context
- Missing HDF5 fields (beta_cirrus_mask): log warning, return permissive default (all-True mask)
- Missing optional dependencies (mesma): log info, use fallback engine
- Data unavailable (FRAMES, AVIRIS-3, Globe-LFMC): FileNotFoundError with helpful message

### Logging
- Use Python `logging` module, never `print()`
- All functions that perform I/O or filtering should log at INFO level
- Warnings for unexpected data shapes or missing fields
- Progress logging for slow operations (MESMA per-pixel unmixing, large library resampling)

### Validation
- Compare against Sentinel-2 dNBR as baseline
- Compare against AVIRIS-3 fractions as high-res reference
- Compare against USGS BARC maps for classified severity
- Use known fire perimeters (NIFC) for spatial validation
- Report R2, RMSE, and bias for quantitative comparisons
- Include sensor comparison (Tanager vs EMIT/PRISMA) for +5 tie-breaker

---

## Tech Debt Tracking

| ID | Issue | Severity | Status |
|----|-------|----------|--------|
| TD-1 | Scene count ambiguity (11 vs 12 fire scenes) | Low | Will resolve at build time via live STAC query |
| TD-2 | HyperCoast wavelength_range: must load-then-slice (no native wavelength filter) | Low | Documented in io.py gotcha |
| TD-3 | cloud_mask may require direct h5py access (HyperCoast may not expose beta_cirrus_mask) | Medium | h5py added as dependency; fallback documented |
| TD-4 | In-CoB selection is simplified (spectral variability ranking) until MESMA exists | Low | Full In-CoB requires unmixing loop; deferred to post-Wave 2 refinement |
| TD-5 | ~~mesma v1.0.8 may not be numpy 2.x compatible~~ | ~~High~~ | **RESOLVED (2026-04-28) — mesma 1.0.8 verified on Python 3.12.3 / numpy 2.4.4. API: MesmaCore.execute(). Stays as optional dep, primary engine.** |
| TD-6 | FRAMES SoCal library bulk download mechanism unverified | Medium | Manual download acceptable for competition; loader handles local dir |

---

## Recent Changes

| Date | Change | Status |
|------|--------|--------|
| 2026-04-27 | Project initialized | **DONE** |
| 2026-04-27 | 002-data-pipeline tasks.md enriched (EM audit) | **DONE** |
| 2026-04-27 | 002-data-pipeline built and merged to main | **DONE** |
| 2026-04-27 | 003-core-analysis tasks.md enriched (EM audit) | **DONE** |
| 2026-04-28 | Phase 2 remediation: 5/6 bugs resolved, LGT-298 closed, LGT-299 awaiting last QA | **IN PROGRESS** |
| 2026-04-28 | Phase 3 (003-core-analysis) EM validation: READY, no blockers | **DONE** |
| 2026-04-28 | Phase 2 findings integrated into Phase 3 tasks.md (epsilon guard, FWHM, reflectance clamp) | **DONE** |
| 2026-04-28 | Phase 3 Wave 1 Section 1 complete: deps added, mesma verified (PASS), spectral-libraries verified (PASS, corrected import path), splib07-loader incompatible (custom ASCII parser needed) | **DONE** |
| 2026-04-28 | tasks.md mid-execution enrichment: splib07-loader gotcha, mesma API details from QA, spectral-libraries import path correction, shade shape (bands,1) | **DONE** |
| 2026-04-28 | Wave 4 validation module landed: validation.py (load_aviris3_reference, load_barc_reference, compute_accuracy, compare_sensors). Continuous metrics (R²/RMSE/MAE/bias/Spearman) and classified metrics (accuracy/Cohen κ/F1/confusion) with NaN/nodata pairwise masking. compare_sensors emits an improvement_ratios dict + pandas comparison_table for the +5 EMIT/PRISMA tie-breaker. | **DONE** |
| 2026-04-28 | Wave 4 test suite landed: tests/test_endmembers.py (25), tests/test_unmixing.py expanded (+3 → 22), tests/test_severity.py (12), tests/test_lfmc.py (16), tests/test_validation.py (19). Full pytest tests/ → 331 passed, 1 skipped (spectral_libraries optional dep). | **DONE** |
| 2026-04-28 | Public API expanded with 12 new lazy exports across severity / lfmc / validation; `import tanager; tanager.<symbol>` resolves for every Wave 1–4 public function. | **DONE** |
