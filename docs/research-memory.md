# Research Memory: Tanager Competition

> Long-term memory for research agents. Tracks literature, experiments, data sources, and scientific findings.

**Location:** `/docs/research-memory.md`
**Owner:** Tobler (Research Lead)
**Updated:** 2026-04-27
**Version:** 2.2 (Phase 3 open questions resolved)

---

## Purpose

This is the third tier of the memory system, specific to research-heavy projects. It tracks:
1. **Literature** — Key papers, their findings, and relevance
2. **Data sources** — Datasets, spectral libraries, reference data
3. **Experiments** — What was tried, what worked, what didn't
4. **Scientific context** — Domain knowledge critical for the project

---

## Tanager-1 Sensor Characterization

| Parameter | Value |
|-----------|-------|
| Bands | 426 (contiguous VSWIR) |
| Spectral range | 380-2500 nm |
| Spectral sampling | 5 nm |
| FWHM | 5.5 nm |
| GSD | 30 m nominal (actual 32-50 m, varies with collection geometry) |
| Swath width | 18-24 km (nadir ~19 km) |
| SNR (2200 nm) | 310-655 (mode-dependent, 4 sensitivity modes) |
| Atmospheric correction | ISOFIT v2.9.5 (~1% RMSE reflectance accuracy) |
| Orbit | 406 km sun-synchronous |
| Operator | Planet Labs / Carbon Mapper |
| Launch | August 16, 2024 |
| Detector | JPL Dyson spectrometer, MCT 640x480 FPA |
| Data format | HDF-EOS5 (HDF5) |
| License | CC BY 4.0 |

### Sensor Comparison

| Parameter | Tanager-1 | EMIT | PRISMA | EnMAP | AVIRIS-3 |
|-----------|-----------|------|--------|-------|----------|
| Spectral range | 380-2500 nm | 381-2493 nm | 400-2505 nm | 418-2446 nm | 390-2500 nm |
| Bands | 426 | 285 | 239 | 224 | 285 |
| Spectral sampling | 5 nm | ~7.4 nm | ~12 nm | ~6.5/10 nm | 7.4 nm |
| FWHM | 5.5 nm | 8.5 nm | <12 nm | 6.5/10 nm | ~7.4 nm |
| GSD | 30 m | 60 m | 30 m | 30 m | 0.5-13 m |
| SNR (SWIR) | 310-655 | >500 | ~200 | ~180 | Very high |
| Tasking | On-demand | ISS orbital | Request | Request | Campaign |

**Key insight:** Tanager-1 has the finest spectral sampling (5 nm) of any spaceborne imaging spectrometer. It is essentially an orbital AVIRIS-NG, making 20+ years of AVIRIS research directly transferable.

### Bad Bands / Atmospheric Windows to Mask

| Window | Wavelength Range | Cause | ~Bands Affected |
|--------|-----------------|-------|-----------------|
| Sensor edge | <400 nm | Low SNR, stray light | ~1-5 |
| Water vapor 1 | 1340-1480 nm | H2O absorption | ~193-221 |
| Water vapor 2 | 1790-1960 nm | H2O absorption | ~283-317 |
| CO2/H2O overlap | 2350-2500 nm | Strong absorption | ~395-426 |

After masking: **~330-346 usable bands** remain. Adaptive per-scene SNR filtering recommended (threshold SNR < 20-30).

---

## Available Data

### LA Wildfire Time Series (Fire Collection)

| Parameter | Value |
|-----------|-------|
| Total fire scenes | 12 (LA + Northern Arizona) |
| LA fires covered | Palisades Fire, Eaton Fire (both started Jan 7, 2025) |
| Date range | Dec 2024 - Jul 2025 |
| Pre-fire baseline | ~Dec 2024 |
| Immediate post-fire | ~Late Jan 2025 |
| Early recovery | ~Feb-Mar 2025 |
| Recovery monitoring | ~Apr-Jul 2025 |
| Products confirmed | Basic Radiance HDF5 + Basic Beta UDM (cloud mask) |
| Products claimed | Ortho Surface Reflectance (NOT confirmed for fire collection -- verify!) |
| Access | Static STAC catalog (no API), pystac traversal, no auth required |
| Storage | GCS: storage.googleapis.com/open-cogs/planet-stac/ |
| Status | 11 scenes across 7 dates identified (see data access report) |

### LA Fire Scene Inventory (11 scenes, 7 dates)

| Scene ID | Date | GSD (m) | Phase |
|----------|------|---------|-------|
| `20241215_185916_33_4001` | 2024-12-15 | 37.45 | PRE-FIRE (23 days before) |
| `20250123_185507_64_4001` | 2025-01-23 | 39.03 | POST-FIRE (16 days after) |
| `20250123_185518_92_4001` | 2025-01-23 | ~39 | POST-FIRE (adjacent strip) |
| `20250407_192235_24_4001` | 2025-04-07 | 50.22 | RECOVERY 3mo |
| `20250407_192229_16_4001` | 2025-04-07 | ~50 | RECOVERY 3mo (adjacent) |
| `20250726_192343_21_4001` | 2025-07-26 | 38.35 | RECOVERY 6mo |
| `20250726_192422_87_4001` | 2025-07-26 | ~38 | RECOVERY 6mo (adjacent) |
| `20250724_190927_83_4001` | 2025-07-24 | ~38 | RECOVERY 6mo |
| `20250902_190116_02_4001` | 2025-09-02 | TBD | LATE RECOVERY / N. Arizona? |
| `20250902_190121_86_4001` | 2025-09-02 | TBD | LATE RECOVERY / N. Arizona? |
| `20250920_193207_61_4001` | 2025-09-20 | 34 | N. Arizona confirmed |

**Critical pair:** Dec 15 pre-fire + Jan 23 post-fire (brackets Palisades + Eaton fires)

### Cross-Validation Data

| Source | Data | Resolution | Availability |
|--------|------|-----------|-------------|
| AVIRIS-3 Eaton Fire | MESMA char/ash fractions, Jan 10-16, 2025 | 3-4 m | ORNL DAAC (public) |
| USGS Burn Severity Portal v11.0 | BARC maps for LA fires | 30 m | usgs.gov (April 2025 update) |
| Globe-LFMC 2.0 | 287,551 field LFMC observations | Point | DOI: 10.1038/s41597-024-03159-6 |
| Sentinel-2 dNBR | Broadband burn severity baseline | 10-20 m | Copernicus Open Access Hub |
| Varga & Jones SoCal LFMC | 32-yr species-specific chaparral LFMC (chamise, sage, ceanothus) | 1 km / semi-monthly | Dryad (NetCDF, Dec 1987-Jun 2019) |

### Other Available Scenes

| Collection | Scenes | Key Geographies |
|------------|--------|-----------------|
| Agriculture | ~43 | Germany (10), Kenya (8), California, India, etc. |
| Natural Lands | ~77 | Global (largest collection) |
| Urban | ~60 | Los Angeles, Buenos Aires, others |
| Fire | 12 | Southern California, Northern Arizona |
| Coastal & Water | Present | San Francisco Bay |
| GHG Plumes | Present | Turkmenistan, Texas, Algeria |

---

## Key Literature

### Hyperspectral Burn Severity (MESMA)

| Paper | Year | Sensor | Key Finding | Relevance |
|-------|------|--------|-------------|-----------|
| Quintano et al. | 2023 | PRISMA | First spaceborne hyperspectral fire severity. MESMA R2=0.64-0.79 vs CBI; Sentinel-2 only R2=0.27-0.53. ~2x improvement. | **Critical** -- validates our approach; PRISMA is closest Tanager analog |
| Roberts et al. | 2018 | AVIRIS | IES + uSZU band selection for post-fire MESMA. r2=0.74 (ash), 0.84 (GV) vs WorldView-2. | **High** -- endmember selection methodology (In-CoB + uSZU) |
| Veraverbeke et al. | 2018 | Review | Comprehensive hyperspectral fire RS review. VSWIR is primary domain. | **High** -- authoritative review framing our approach |
| Veraverbeke et al. | 2014 | AVIRIS | Burned fraction vs GeoCBI R2=0.86. Significantly better than Landsat. | **High** -- upper bound for severity accuracy |
| AVIRIS-3 Rapid Response | 2025 | AVIRIS-3 | MESMA char/ash fractions over Eaton Fire. 86.3% structural damage accuracy at 3-4m. | **Critical** -- direct LA fire validation source |
| Quintano et al. | 2017 | Landsat+LST | MESMA fractions + Land Surface Temperature improves severity mapping. | **Moderate** -- ancillary data fusion idea |
| Quintano et al. | 2013 | Landsat TM | First MESMA for burn severity in Mediterranean. Kappa >0.75. | **Moderate** -- established MESMA-fire methodology |
| Fernandez-Manso et al. | 2016 | Landsat | Multi-temporal MESMA fraction time series for recovery | **Moderate** -- validates temporal trajectory approach |
| Robichaud et al. | 2007 | AVIRIS | MTMF unmixing of Hayman Fire. r2=0.21-0.48. | **Moderate** -- MESMA outperforms MTMF |
| Dennison & Roberts | 2003 | AVIRIS | Foundational EAR/MASA metrics for chaparral MESMA. | **Moderate** -- core endmember algorithms |

**Key DOIs:**
- Quintano et al. 2023: 10.1016/j.rse.2023.113670
- Roberts et al. 2018: 10.3390/rs10030389
- Veraverbeke et al. 2018: 10.1016/j.rse.2018.06.020
- AVIRIS-3 LA 2025: 10.1029/2025GL118756

### Live Fuel Moisture Content (LFMC)

| Paper | Year | Sensor | Approach | Accuracy | Relevance |
|-------|------|--------|----------|----------|-----------|
| Roberts, Dennison, Peterson et al. | 2006 | AVIRIS+MODIS | Spectral indices (WI, NDWI, VARI), SMA | High (AVIRIS), lower (MODIS) | **Critical** -- SoCal chaparral, directly transferable |
| Peterson & Roberts | 2014 | Lab spectrometer | PLSR on full spectra (350-2500nm) | R2=0.94 (new needles), R2=0.91 (sagebrush) | **High** -- PLSR approach matches Tanager range |
| Yebra, Dennison, Chuvieco et al. | 2013 | Review | Global review: indices vs RTM | Varies by method/biome | **High** -- definitive LFMC review |
| Dennison et al. | 2003-2004 | AVIRIS | EWT from SMA, seasonal canopy moisture | Seasonal EWT mapped in Santa Monica Mtns | **Critical** -- LA region chaparral |
| Quan et al. | 2021 | Lab | SAI970, SAI1200, SAI1660 indices | SAI1200: R2cv=0.845 (EWT); SAI1660: R2cv=0.637 (FMC) | **High** -- validated narrowband indices |
| Yebra et al. | 2024 | Field+MODIS | Globe-LFMC 2.0: 287,551 field observations | Training dataset | **Critical** -- ground truth source |
| Leite et al. | 2025 | Review | Next-gen spaceborne for fuel monitoring (EMIT, SBG, CHIME, Tanager) | Qualitative | **Critical** -- frames Tanager in context |
| Varga & Jones | 2026 | Landsat+WRF | 32-yr species-specific SoCal chaparral LFMC; chamise MAE=9.68%, R²=0.76 | Training/validation | **High** -- direct SoCal ground truth for LFMC |
| Qi et al. | 2014 | Various | PLSR R2=0.72-0.94 leaf level; dry mass confounds water | PLSR methodology | **High** -- cautionary wavelength interpretation |
| Danson & Bowyer | 2004 | Various | GA-PLS R2=0.82-0.89; bands at 1144, 1304, 1670, 1750nm | Optimal SWIR bands | **High** -- key band selection |
| Riano et al. | 2005 | Various | PROSPECT inversion for EWT+DM | RTM approach | **Moderate** -- RTM reference |

**Key DOIs:**
- Roberts et al. 2006: 10.1029/2005JG000113
- Yebra et al. 2013: 10.1016/j.rse.2013.02.029
- Quan et al. 2021: 10.1371/journal.pone.0249351
- Globe-LFMC 2.0: 10.1038/s41597-024-03159-6
- Leite et al. 2025: 10.1002/rse2.416
- Varga & Jones 2026: 10.1038/s41597-026-06794-3 (Dryad data repository)

### Accuracy Comparison: Narrowband vs Broadband

| Platform | Bands | Resolution | LFMC R2 | LFMC RMSE |
|----------|-------|------------|---------|-----------|
| AVIRIS (airborne) | 224 @ ~10nm | 4-20m | 0.72-0.94 | 5-21% |
| **Tanager-1 (est.)** | **426 @ ~5nm** | **30m** | **est. 0.70-0.85** | **est. 10-25%** |
| Sentinel-2 | 12 bands, 20-180nm | 10-20m | 0.55-0.75 | 8-20% |
| MODIS | 7 bands, broad | 500m | 0.49-0.55 | 23-28% |
| Landsat 8/9 | 6 VNIR-SWIR | 30m | 0.50-0.65 | 15-25% |

---

## Key Wavelength Regions

### Burn Severity Diagnostic Bands

| Region | Wavelength | Signal | Importance |
|--------|-----------|--------|------------|
| Red | 660-670 nm | Chlorophyll absorption loss | High |
| Red Edge | 700-750 nm | Vegetation stress/mortality | High |
| NIR Plateau | 810-900 nm | Highest discriminatory power (M=0.73) | **Critical** |
| SWIR-1 | 1550-1700 nm | Moisture loss, exposed mineral soil | High |
| SWIR-2 | 2000-2300 nm | Char/ash features, cellulose destruction | **Critical** |

### LFMC Water Absorption Features

| Feature (nm) | Absorption Cause | Tanager Bands (~) | Strength | Best Index |
|--------------|-----------------|-------------------|----------|------------|
| 970 | O-H stretch overtone | ~118 bands | Weak | SAI970 |
| 1200 | O-H stretch+bend combination | ~36 bands | Moderate | SAI1200 (R2cv=0.845 for EWT) |
| 1450 | O-H stretch+bend | ~40 bands | Strong | Often atmospheric interference |
| 1660-1700 | O-H stretch overtone | ~32 bands | Moderate | SAI1660 (R2cv=0.637 for FMC) |
| 2100 | O-H + cellulose | ~30 bands | Moderate | Dry matter + water |

---

## Spectral Libraries & Reference Data

| Resource | Purpose | Fire Relevance | Format | Spectra |
|----------|---------|----------------|--------|---------|
| USGS Spectral Library v7 | Minerals, soils, vegetation, organics | Char/charcoal (Organics ch.), heated soils/iron oxides (Soils ch.), chaparral veg incl. manzanita (Vegetation ch.) | SPECPR + ASCII, 350-2500nm at 1nm (s07ASD) | 1000+ |
| ECOSTRESS Spectral Library v1.0 | 3400+ spectra (plants, minerals, NPV) | 541 vegetation VIS/SWIR + 51 NPV spectra. SPy native via EcostressDatabase | SQLite via SPy, 0.35-15.4µm | 3400+ |
| FRAMES Burn Severity Library (SoCal) | Fire-specific field endmembers from chaparral | **Primary**: 7 char/ash, 36 GV (chamise, manzanita, ceanothus, sage, oak), 13 NPV, 10 soil. Old Fire + Simi Fire. | ASCII (350-2500nm) | 66 |
| Globe-LFMC 2.0 | 287,551 field LFMC observations | Ground truth for LFMC training | Database | 287,551 |
| spectral-libraries v1.1.3 | EAR/MASA/CoB endmember selection | Library pruning for MESMA | pip (Python) | N/A |
| splib07-loader | ~~Python loader for USGS v7~~ | ~~Programmatic access to USGS spectra~~ | ~~pip (GitHub)~~ | **INCOMPATIBLE — nptyping pins numpy<2.0. Use custom ASCII parser instead.** |

### USGS v7 ASCII Data Access (Verified 2026-04-28)

**Download:** [ScienceBase DOI:10.5066/F7RR1WDJ](https://dx.doi.org/10.5066/F7RR1WDJ) → child item "Spectra of materials in ASCII format"

| Archive | Size | Contents | Recommended? |
|---------|------|----------|--------------|
| ASCIIdata_splib07a.zip | 20.8 MB | Measured spectra, native spectrometer resolution | **YES — primary** |
| ASCIIdata_splib07b.zip | 41.4 MB | Oversampled (cubic spline to finer grid) | Optional |
| ASCIIdata_splib07b_cvASD.zip | 27.1 MB | Convolved to ASD standard resolution | Alternative |

**File format:** Single-column text files (.txt), one reflectance value per line per band. Wavelength/bandpass in separate files. Bad values = `-1.23e34` → mask as NaN.

**Folder structure:** Chapter subfolders — `ChapterO_OrganicCompounds` (charcoal), `ChapterS_SoilsAndMixtures` (heated soils), `ChapterV_Vegetation` (chaparral), `ChapterM_Minerals` (goethite, hematite).

**File prefix convention:** `s07ASD` = ASD spectrometer (350-2500nm, 2151 channels). This is the standard for VSWIR comparison.

**Parser implementation:** ~30-50 lines Python with `numpy.loadtxt()`. No external dependency needed. SPy `BandResampler` handles 2151→426 channel convolution to Tanager bands.

### Fire Endmember Types (Recommended)

**Primary (required):**
1. **Char** -- charred plant material, spectrally dark, rising SWIR
2. **Photosynthetic Vegetation (PV)** -- green veg, chlorophyll + NIR + cellulose
3. **Non-Photosynthetic Vegetation (NPV)** -- dead litter/bark, no chlorophyll, cellulose retained
4. **Soil/Rock** -- bare mineral soil
5. **Shade** -- photometric shade (zero reflectance)

**Secondary (recommended for LA WUI context):**
6. **Ash (white/light)** -- complete combustion indicator, higher VNIR than char
7. **Urban/Impervious** -- for wildland-urban interface pixels

**Target library size:** ~52-78 spectra after pruning (Char 10-15, Ash 3-5, PV 20-30, NPV 10-15, Soil 8-12, Shade 1).

**Selection strategy:** In-CoB (count-based within-class) for initial selection + EAR/MASA joint pruning + uSZU band reduction per Roberts et al. (2018). Hybrid library: FRAMES (primary) + USGS v7 + ECOSTRESS + image-derived endmembers from pre/post-fire Tanager scenes. Implemented via `spectral-libraries` v1.1.3 EarMasaCob class.

**Resampling:** SPy `BandResampler` with Gaussian FWHM convolution (1nm library → ~5nm Tanager band centers). Extract actual band centers from HDF5 metadata.

---

## Tools Evaluated

| Tool | Version | Purpose | Verdict | Notes |
|------|---------|---------|---------|-------|
| HyperCoast | 0.20.2 | Tanager data I/O | **USE** -- primary loader | `read_tanager()`, STAC search, leafmap viz. Ortho products only. |
| Spectral Python (SPy) | 0.24 | Spectral analysis | **USE** -- most mature | SAM, BandResampler, ECOSTRESS DB. Needs numpy bridge from xarray. |
| MESMA | 1.0.8 | Spectral unmixing | **USE** — confirmed compatible | Tested: Python 3.12 + numpy 2.4. API: `MesmaCore.execute()`. Image=(bands,pixels), Library=(bands,spectra). |
| HySUPP | Current | 20+ unmixing algorithms | **FALLBACK** | Alternative if MESMA chokes on 426 bands |
| spyndex | 0.6.0 | 232+ spectral indices | **USE** | Any numpy array input |
| splib07-loader | 0.5.1 | USGS Spectral Library v7 | **DO NOT USE** | Requires nptyping which pins numpy<2.0 — breaks rasterio/rioxarray. Correct repo: brianschubert/splib07-loader (NOT RobertSparworthy). Use custom ASCII parser instead. |
| pystac | Current | STAC catalog traversal | **USE** | Static catalog -- use pystac, NOT pystac-client |
| spectral-libraries | 1.1.3 | EAR/MASA/CoB endmember selection | **USE** | Same author as mesma. IES, CRES, MUSIC, AMUSES algorithms. |
| pysptools | 0.15.0 | FCLS/NNLS unmixing, endmember extraction | **FALLBACK** | N-FINDR, PPI, FCLS. Unmaintained, Python 3.6 era. |

### Tool Integration Architecture

```
HyperCoast (HDF5 -> xarray) -> numpy bridge -> SPy / MESMA / spyndex
                                    |
                            BandResampler (spectral libraries -> Tanager bands)
```

### Known Issues / Blockers

| ID | Issue | Impact | Mitigation | Status |
|----|-------|--------|------------|--------|
| B0 | ~~Surface reflectance not confirmed for fire collection~~ | ~~Critical~~ | ~~Verify by crawling STAC~~ | **RESOLVED — ortho_sr_hdf5 available for all 11 scenes** |
| B1 | ~~HyperCoast Basic products are ungridded~~ | ~~High~~ | ~~Use Ortho products only~~ | **RESOLVED — ortho products confirmed working (direct h5py)** |
| B2 | MESMA scalability at 426 bands untested | Medium | Test small subset; MNF reduction; HySUPP fallback | Open |
| B3 | Static STAC catalog (no /search endpoint) | Low | pystac traversal; cache locally | Working |
| B4 | SPy does not read HDF5 or xarray natively | Low | Extract .values from xarray to numpy | Working |
| B5 | GEE has only basic radiance (no L2) | Low | Direct HDF5 download is primary workflow | N/A |
| B6 | ~~USGS Spectral Library v7 no Python-native loader~~ | ~~Low~~ | ~~splib07-loader or numpy ASCII parsing~~ | **RESOLVED (2026-04-28) — splib07-loader incompatible (numpy<2); use custom ASCII parser on ASCIIdata_splib07a.zip (20.8 MB from ScienceBase DOI:10.5066/F7RR1WDJ). s07ASD prefix = ASD spectrometer, 2151 channels, 350-2500nm.** |
| B7 | **HyperCoast read_tanager() fails on ortho products** | **Critical** | Direct h5py loader needed (prototype validated) | **NEW — LGT-296** |
| B8 | **cloud_mask() wrong HDF5 paths for ortho** | **High** | Add GRIDS paths to candidate list | **NEW — LGT-297** |
| B9 | **Spectral indices produce impossible values** | **High** | Clamp reflectance + epsilon guard | **NEW — LGT-298** |
| B10 | **No spatial co-registration for multi-temporal** | **High** | rasterio reproject to common grid | **NEW — LGT-299** |

---

## Recommended Approaches

### Burn Severity Pipeline (MESMA)

Based on Quintano et al. (2023) methodology:
1. Load pre/post-fire Tanager L2 surface reflectance
2. Apply bad band mask (~330 usable bands)
3. uSZU band selection -> ~30-50 diagnostic bands
4. Build hybrid endmember library (FRAMES + ECOSTRESS + image-derived)
5. In-CoB endmember selection to prune library
6. Run MESMA -> char, PV, NPV, soil, shade fraction maps
7. Train Random Forest regression: fractions -> CBI-equivalent severity
8. Validate against AVIRIS-3 (aggregated to 30m) and BARC maps
9. Multi-temporal analysis: fraction trajectories Dec 2024 -> Jul 2025

**Expected performance:** R2=0.64-0.79 vs CBI (based on PRISMA precedent).

### LFMC Estimation Pipeline

Three-tier approach (recommended: Tier 1 + Tier 2 for competition):

**Tier 1 -- Spectral Water Indices (simplest, most interpretable):**
- SAI1200 (for EWT), SAI1660 (for FMC)
- NDWI variants: NDWI(860,1240), NDWI(860,1640), NDWI(860,2130)
- Water Index: R900/R970
- Continuum-removal band depth at 970, 1200, 1700, 2100 nm
- Expected: R2=0.60-0.85 for EWT; R2=0.50-0.64 for direct FMC

**Tier 2 -- PLSR / Machine Learning:**
- PLSR using full 330-band reflectance
- Random Forest with bands + derived indices
- Train on Globe-LFMC 2.0 co-located observations
- Expected: R2=0.70-0.94 depending on veg type and calibration

**Tier 3 -- Physically-based RTM (deferred):**
- PROSAIL inversion for EWT + LAI -> CWC
- High effort, publishable but not needed for competition

**Key insight:** No satellite hyperspectral LFMC product exists. We would be first.

### Preprocessing Workflow

1. Download L2 Ortho Surface Reflectance from STAC (HDF5)
2. Load via `hypercoast.read_tanager()`
3. Interpolate to regular grid
4. Apply bad band mask (<400nm, 1340-1480nm, 1790-1960nm, 2350-2500nm)
5. Apply cloud/cirrus mask (beta_cirrus_mask layer)
6. Per-scene SNR filtering
7. Validate reflectance against known targets (water, bare soil)
8. Proceed to analysis

---

## Open Questions

### Resolved

1. ~~What atmospheric correction is needed?~~ **RESOLVED:** ISOFIT v2.9.5 in L2 products is analysis-ready (~1% RMSE). No additional correction needed.
2. ~~How does Tanager 30m GSD compare to airborne for burn severity?~~ **RESOLVED:** 30m = landscape-scale severity (matching Landsat dNBR standard). Loses structure-level damage. 426-band spectral advantage compensates.
3. ~~Can we validate against BARC maps?~~ **YES:** USGS Burn Severity Portal v11.0 (April 2025). AVIRIS-3 Eaton Fire data (ORNL DAAC) provides additional validation.
4. ~~What is the best MESMA software?~~ **RESOLVED:** mesma v1.0.8 is primary; HySUPP as fallback. Test 426-band performance early.

### Partially Resolved

5. ~~Which endmember spectra for LA fire vegetation?~~ **RESOLVED:** FRAMES SoCal chaparral library (66 spectra: 7 char/ash, 36 GV, 13 NPV, 10 soil from Old Fire + Simi Fire) as primary. Supplemented by USGS v7 charcoal/heated soils and ECOSTRESS vegetation/NPV. Hybrid strategy with image-derived endmembers. In-CoB + EAR/MASA selection via spectral-libraries v1.1.3.
6. ~~Optimal band subset for LFMC?~~ **RESOLVED:** SAI1200 (1200nm, R²cv=0.845 for EWT), SAI1660 (1660nm, R²cv=0.637 for FMC), NDWI variants. Full-spectrum PLSR for Tier 2.

### Open

7. ~~Exact scene IDs and dates for LA fire collection~~ **RESOLVED:** 11 scenes across 7 dates. See scene inventory above.
8. MESMA performance at 426 bands -- needs empirical testing on small spatial subset.
9. ~~AVIRIS-3 Eaton Fire data public availability~~ **RESOLVED (2026-04-27):** Publicly available at ORNL DAAC. AVIRIS-3 L2A surface reflectance (DOI: 10.3334/ORNLDAAC/2357) has 511+ granules, date range 2023-07-05 to present. AVIRIS-3 flew over LA fires on Jan 10 and Jan 16, 2025 at 3-4m resolution. Published in GRL (DOI: 10.1029/2025GL118756): MESMA char/ash fraction mapping achieved 86.3% structural damage classification accuracy. Access via NASA Earthdata Search with free account. Search by flight line ID with wildcards in Granule IDs box.
10. ~~Globe-LFMC 2.0 SoCal coverage~~ **RESOLVED (2026-04-27):** Strong SoCal coverage. Globe-LFMC 2.0 at Figshare (DOI: 10.6084/m9.figshare.c.6980418) has 287,551 observations across 2000+ sites in 15 countries, 1977-2023. Additionally, Varga & Jones (2026, Scientific Data 13:438) published a 32-year species-specific SoCal chaparral LFMC dataset with >10,000 observations covering chamise (new+old growth), black sage, and bigpod ceanothus. Available on Dryad as NetCDF, 1km semi-monthly resolution, Dec 1987-Jun 2019. Chamise model MAE=9.68%, R²=0.76. This is an excellent supplementary training/validation source.
11. LFMC below 60% nonlinear regime -- Roberts et al. (2006) found this; needs calibration attention.
12. Can we obtain field CBI measurements for the LA fires?
13. ~~mesma v1.0.8 compatibility with Python 3.10+ and numpy 2.x~~ **RESOLVED (2026-04-27):** Empirically tested and confirmed working. mesma 1.0.8 installs and runs correctly with Python 3.12.3 + numpy 2.4.4. Full MESMA unmixing test passed: accurate fraction recovery on synthetic 2-class mixtures. API: `MesmaCore.execute(image, library, look_up_table, em_per_class, constraints)`. Image format: (n_bands, n_pixels). Library format: (n_bands, n_spectra). Look-up table: `{level: {class_model_tuple: np.array([[em_indices]])}}}`. No HySUPP fallback needed.
14. ~~FRAMES SoCal library bulk download~~ **RESOLVED (2026-04-27):** No bulk download archive exists. Individual ASCII file downloads only from frames.gov/assessing-burn-severity/spectral-library/southern-california. 67 spectra: 37 green vegetation, 13 NPV, 10 mineral soil/rock, 7 char/ash. Each spectrum has a digital photo, spectral graph (.jpg), and ASCII data file. Will need a download script to automate retrieval.
15. Ash vs. char spectral separation — only 7 combined spectra in FRAMES; may need supplementary lab measurements.

---

## Competition T&C Compliance (Reviewed 2026-04-27)

**Status:** Fully compliant. Full review at `research/competition-tc-compliance-review.md`.

### Judging Criteria (100 pts + 15 tie-breaker)

| Category | Points | Weight |
|----------|--------|--------|
| Scientific Integrity & Innovation | 30 | Methodology, novelty, limitations |
| Application or Use Case | 30 | Relevance, feasibility, real-world impact |
| Workflow & Tool Development | 20 | Clean code, reproducibility, STAC usage, open-source |
| Visualization & Storytelling | 20 | Maps, plots, narrative, broad audience appeal |
| Tie-breaker: Planet verticals/impact | +5 | Wildfire = eligible |
| Tie-breaker: Tanager vs EMIT/PRISMA | +5 | Quantitative comparison required |
| Tie-breaker: Open-source/AI-ML | +5 | New library or cutting-edge AI/ML |

### Submission Requirements

- **Required:** Tanager imagery from Open STAC catalog + one of: technical memo (1-3 pg), Jupyter notebooks, GitHub repo, slides, or video (<3 min)
- **Optional:** Public repo with reproducibility docs, Zenodo data derivatives, figures/maps
- **Third-party code:** Allowed if properly licensed (all our libs are MIT/BSD/Apache)
- **One submission per participant** (team members may also submit individually)
- **English only**
- **Deadline:** August 31, 2026, 11:59 PM PST
- **Submit via:** surveymonkey.com/r/tanager-competition (after registration)

### Key IP Terms

- We retain full IP ownership
- Planet gets irrevocable royalty-free promotional license
- Tanager data remains Planet's under CC license

### Action Items

- **Human must register** for competition (opened April 14, 2026)
- Verify FRAMES SoCal library redistribution rights
- Invest in visualization & storytelling (20% of score)
- Plan open-source packaging for +5 tie-breaker
- Include Tanager vs EMIT/PRISMA quantitative comparison for +5

---

## Experiments Log

| Date | Experiment | Result | Notes |
|------|-----------|--------|-------|
| 2026-04-27 | Real data download (3 ortho SR scenes) | **PASS** | catalog.py works; 3.4 GB downloaded |
| 2026-04-27 | HyperCoast read_tanager() on ortho SR | **FAIL** | Requires lat/lon arrays; ortho uses UTM grid |
| 2026-04-27 | Direct h5py loading of ortho SR | **PASS** | Full 426-band cube loaded successfully |
| 2026-04-27 | Bad band masking on real data | **PASS** | 328/426 bands retained; sensor flags slightly differ |
| 2026-04-27 | Spectral indices (NBR/NDVI/NDWI) on real data | **FAIL** | Values outside [-1,1] due to extreme reflectance |
| 2026-04-27 | Cloud mask on real data | **FAIL** | Searches wrong HDF5 paths for ortho products |
| 2026-04-27 | dNBR (pre vs post) | **FAIL** | Different spatial grids; scenes barely overlap |
| 2026-04-27 | Scene geographic verification | **FAIL** | 3 "LA" scenes are actually in Utah |

---

## Real Data Findings (Phase 2 Defensibility Review)

**Status:** Review complete (2026-04-27). Full report: `research/phase2-defensibility-review.md`
**Verdict:** NOT DEFENSIBLE — 4 blocking issues must be resolved before Phase 3.

### Confirmed Sensor Parameters (from real HDF5 metadata)

| Parameter | Config Value | Actual Value | Status |
|-----------|-------------|-------------|--------|
| Bands | 426 | 426 | Correct |
| Wavelength range | 380-2500 nm | 376.44-2499.00 nm | Slightly off |
| Spectral sampling | 5 nm | 4.95-5.02 nm (mean 4.99) | Correct |
| FWHM | 5.5 nm | 5.20-6.81 nm (varies by band) | Not constant |
| GSD | 30 m | 30.0 m (confirmed from grid metadata) | Correct |
| Projection | N/A | UTM Zone 11, EPSG:32611 | New finding |
| Fill value | N/A | -9999.0 | New finding |

### Corrected Scene Inventory (8 LA, 3 Utah)

**LA area scenes (usable for FireSpec):**

| Scene | Date | Phase | Grid | Notes |
|-------|------|-------|------|-------|
| 20241215_185916_33_4001 | 2024-12-15 | pre-fire | 713×791 | Malibu/Palisades coast |
| 20250123_185507_64_4001 | 2025-01-23 | post-fire | 1047×961 | Eaton Fire area (north/east LA) |
| 20250123_185518_92_4001 | 2025-01-23 | post-fire | TBD | Adjacent swath (likely overlaps pre-fire) |
| 20250407_192235_24_4001 | 2025-04-07 | early-recovery | 869×1039 | North LA area |
| 20250407_192229_16_4001 | 2025-04-07 | early-recovery | TBD | Adjacent swath |
| 20250726_192343_21_4001 | 2025-07-26 | mid-recovery | TBD | LA area |
| 20250726_192422_87_4001 | 2025-07-26 | mid-recovery | TBD | Adjacent swath |
| 20250920_193207_61_4001 | 2025-09-20 | late-recovery | TBD | LA coast area |

**Utah scenes (NOT LA fires):**

| Scene | Date | Location |
|-------|------|----------|
| 20250724_190927_83_4001 | 2025-07-24 | Utah (38.5°N, -112°W) |
| 20250902_190116_02_4001 | 2025-09-02 | Utah (38.5°N, -112°W) |
| 20250902_190121_86_4001 | 2025-09-02 | Utah (38.5°N, -112°W) |

### Key Resolved Questions

- **B0 RESOLVED:** Surface reflectance IS available for all fire scenes (ortho_sr_hdf5 product type)
- **Scene overlap CONCERN:** Pre-fire (Dec 15) and post-fire (Jan 23 primary) cover different areas. Must verify Jan 23 second swath overlaps pre-fire scene.
- **Ortho products use GRIDS, not SWATHS:** All HDF5 paths in the pipeline need updating.

---

## Phase 1 Research Summary

**Status:** Complete (2026-04-27)

**Conclusion:** The FireSpec approach is strongly validated by literature. Tanager-1's sensor characteristics (426 bands, 5nm, 380-2500nm) are ideal for both burn severity (MESMA) and LFMC estimation. The LA wildfire time series provides a compelling case study. Tools exist but need integration work. No satellite hyperspectral LFMC product exists -- high novelty opportunity.

**Ready for Phase 2:** Data pipeline development and exploratory spectral analysis.
