# Deep Literature Review: Hyperspectral Remote Sensing for Wildfire Analysis

**Date:** 2026-04-27

---

## Executive Summary

This review covers the state of hyperspectral wildfire analysis with focus on MESMA burn severity mapping and LFMC estimation — the two core capabilities for FireSpec. The literature is unambiguous: **MESMA with hyperspectral data dramatically outperforms index-based multispectral approaches** for burn severity (R^2 = 0.86 vs 0.52, Veraverbeke 2014). The Quintano et al. (2023) study with PRISMA at 30m GSD provides the closest methodological template for what FireSpec will do with Tanager. For LFMC, spaceborne estimation at 5nm resolution is unprecedented — all existing products use coarse spectral resolution (MODIS) or broadband sensors (Landsat/Sentinel-2).

**FireSpec would be the first published wildfire analysis from Tanager-1, filling a significant gap in the literature.**

---

## 1. MESMA for Burn Severity Mapping

### Key Papers

#### P1. Quintano et al. (2023) — "First evaluation of fire severity retrieval from PRISMA hyperspectral data"
- **Journal:** Remote Sensing of Environment, DOI: 10.1016/j.rse.2023.113670
- **Finding:** PRISMA MESMA fire severity (R^2 = 0.64-0.79, RMSE = 0.33-0.41) substantially outperformed Sentinel-2 (R^2 = 0.27-0.53, RMSE = 0.54-0.60) for the Sierra de la Culebra megafire (28,046 ha), Spain, 2022. PRISMA prevented the underestimation of high fire severity that Sentinel-2 systematically exhibits.
- **Endmembers:** Char, Photosynthetic Vegetation (PV), Non-Photosynthetic Vegetation and Soil (NPVS). Random Forest Regression against field-measured CBI.
- **Relevance:** **CRITICAL** — Most directly comparable to FireSpec. Spaceborne hyperspectral (30m) with MESMA against CBI.

#### P2. Quintano & Fernandez-Manso (2023/2024) — MESMA PRISMA fire severity in Mediterranean forests
- **Journal:** Proc. SPIE 12688; follow-up 2024 with field integration
- **Finding:** Overall accuracy 92%, Kappa = 0.80 for categorized fire severity. Overcame the typical confusion between moderate/low/high severity seen in broadband data.
- **Relevance:** **CRITICAL** — Demonstrates operational viability at 92%/0.80 kappa.

#### P3. Veraverbeke, Stavros & Hook (2014) — "Assessing fire severity using imaging spectroscopy data from AVIRIS"
- **Journal:** Remote Sensing of Environment, Vol. 154, pp. 153-163
- **Finding:** AVIRIS all-band SMA burned fraction achieved R^2 = 0.86 with Geo-CBI for the Rim Fire, California. Landsat yielded only R^2 = 0.52. This is the foundational paper establishing the hyperspectral advantage.
- **Endmembers:** Char, green vegetation, substrate (soil/rock).
- **Relevance:** **CRITICAL** — The quantitative baseline: R^2 = 0.86 vs 0.52.

#### P4. Quintano et al. (2013) — "MESMA to map burn severity levels from Landsat images in Mediterranean countries"
- **Journal:** Remote Sensing of Environment, DOI: 10.1016/j.rse.2013.04.024
- **Finding:** MESMA with Landsat broadband, four endmembers (char, GV, NPVS, shade), Kappa > 0.75. Established the MESMA-for-fire methodology later extended to hyperspectral.
- **Relevance:** **Important** — Methodological foundation.

#### P5. Dennison, Qi et al. (2018) — "Evaluating Endmember and Band Selection Techniques for MESMA"
- **Journal:** Remote Sensing, Vol. 10(3), 389
- **Finding:** Compared four endmember selection methods (IES, In-CoB, EAR, MASA) with uncorrelated Stable Zone Unmixing (uSZU) band selection on AVIRIS Rim Fire data. **In-CoB + uSZU** provided best accuracy-efficiency tradeoff. Char fractions R^2 = 0.741, green vegetation fractions R^2 = 0.841.
- **Relevance:** **CRITICAL** — Directly informs our endmember selection strategy.

#### P6. Rao et al. (2020) — "Enhanced burn severity estimation using fine resolution ET and MESMA fraction images"
- **Journal:** Remote Sensing of Environment, DOI: 10.1016/j.rse.2020.101801
- **Finding:** Combining MESMA fractions with evapotranspiration data and environmental variables through ML enhanced severity estimation. MESMA fractions have clear physical meaning.
- **Relevance:** **Important** — Value of combining MESMA with ancillary data.

### Endmember Selection Strategy

Standard fire MESMA endmember library:

| Endmember | Description | Spectral Features |
|-----------|-------------|-------------------|
| White Ash | High reflectance mineral residue | High broadband reflectance |
| Black Char | Low reflectance carbon-rich | Very low reflectance, flat spectrum |
| Green Vegetation (PV) | Photosynthesizing | Chlorophyll absorption 680nm, NIR plateau |
| Non-Photosynthetic Veg (NPV) | Dry grass, dead wood, litter | Cellulose features 1700nm, 2100nm |
| Bare Soil/Substrate | Exposed mineral soil | Iron oxide features, clay minerals |
| Shade | Photometric shade | Zero vector (standard MESMA) |

**Recommended approach:** In-CoB endmember selection + uSZU band reduction (Dennison 2018).

### Most Diagnostic Spectral Regions

| Region | Wavelength | Feature | Application |
|--------|-----------|---------|-------------|
| Red absorption | 680 nm | Chlorophyll | Live vegetation detection |
| Red-edge | 700-750 nm | Red-edge position | Recovery monitoring |
| NIR plateau | 800-900 nm | Leaf structure | Reference band for indices |
| Water-1 | 970 nm | O-H overtone | LFMC, canopy water |
| Optimal dNBR-NIR | 962 nm | — | Best dNBR NIR band (van Gerrevink 2021) |
| Water-2 | 1200 nm | O-H combination | LFMC |
| NPV / Cellulose | 1700 nm | C-H stretch | NPV discrimination, LFMC |
| Cellulose | 2100 nm | C-O stretch | CAI (R^2 = 0.808), dry matter |
| Clay minerals | 2200 nm | Al-OH | Post-fire soil mineralogy |
| Optimal dNBR-SWIR | 2246 nm | — | Best dNBR SWIR band (van Gerrevink 2021) |

---

## 2. LFMC Estimation from Hyperspectral Imagery

### Key Papers

#### P7. Yebra, Dennison, Chuvieco et al. (2013) — "A global review of remote sensing of live fuel moisture content"
- **Journal:** Remote Sensing of Environment, DOI: 10.1016/j.rse.2013.05.029
- **Finding:** Foundational review. Liquid water has strong NIR/SWIR absorption features providing physical basis for LFMC estimation. Methods split into empirical (statistical indices) and physically-based (RTM inversion). Empirical relationships are site-specific; RTM inversion is more generalizable but complex.
- **Relevance:** **CRITICAL** — Essential background.

#### P8. Veraverbeke, Dennison, Gitas et al. (2018) — "Hyperspectral remote sensing of fire: State-of-the-art and future perspectives"
- **Journal:** Remote Sensing of Environment, Vol. 216, pp. 105-121
- **Finding:** Definitive field review covering fuel characterization, active fire detection, burn severity, and recovery. Hyperspectral enables detailed discrimination of fuel types/condition, fire temperatures, severity, and recovery. Anticipated spaceborne imaging spectroscopy would scale airborne methods.
- **Relevance:** **CRITICAL** — Maps the research landscape FireSpec enters.

#### P9. Qi, Dennison, Jolly et al. (2014) — "Spectroscopic analysis of seasonal changes in live fuel moisture content and leaf dry mass"
- **Journal:** Remote Sensing of Environment
- **Finding:** PLSR models for LFMC achieved R^2 = 0.94 (new needles), R^2 = 0.72 (old needles), R^2 = 0.91 (sagebrush). **Critical insight: dry mass contribution to LFMC variation equals or exceeds water's contribution.** SWIR features related to dry matter are also important, not just water bands.
- **Relevance:** **CRITICAL** — Must account for both water AND dry matter spectral contributions.

#### P10. Spectral Absorption Indices for Water Content
- **Sources:** Multiple studies (PLOS ONE 2021, BMC Ecology 2019)
- **Key water absorption features:**
  - **SAI970**: 970 nm (weak O-H overtone)
  - **SAI1200**: 1200 nm (O-H combination)
  - **SAI1660**: ~1660 nm (water/cellulose)
  - Additional: 1700 nm (C-H stretch), 2100 nm (cellulose/starch)
- NDWI (~860nm / ~1240nm): R^2 = 0.39-0.80 for LFMC (vs R^2 = 0.25-0.60 for NDVI)
- **Relevance:** **Important** — Exact spectral features Tanager's 5nm resolution resolves.

### Advantage of 5nm Spectral Resolution for LFMC

1. **Precise absorption depth measurement** — water features at 970nm, 1200nm, 1700nm are narrow; 5nm resolves shape and depth. Sentinel-2's 20nm bands partially capture these at best.
2. **Water vs dry matter separation** — Qi et al. (2014) showed dry mass equals water in LFMC spectral expression; narrow bands needed to separate overlapping absorptions.
3. **Continuum removal precision** — narrow-band continuum removal yields more accurate absorption depth than broadband ratios.

### Operational LFMC Products and Their Limitations

#### P11. Quan, Yebra et al. (2021) — "Global fuel moisture content mapping from MODIS"
- **Journal:** Int. J. Applied Earth Observation & Geoinformation, Vol. 101
- **Finding:** First daily global FMC at 500m from MODIS using PROSPECT-5/4SAIL/GeoSail RTM inversion. Three fuel classes. Limitation: 500m misses within-stand heterogeneity.
- **Relevance:** **Important** — The baseline FireSpec would improve upon (Tanager 30m/5nm vs MODIS 500m/broadband).

#### P12. Varga & Jones (2026) — "A 32-year species-specific live fuel moisture content dataset for southern California chaparral"
- **Journal:** Scientific Data
- **Finding:** RF models for chamise (MAE = 9.68%, R^2 = 0.76), 4 chaparral species, San Luis Obispo to LA County, 32-year record.
- **Relevance:** **CRITICAL** — Directly usable for FireSpec LFMC validation over the LA fire area.

---

## 3. Hyperspectral vs Multispectral — Quantitative Comparison

### R^2 Summary Table

| Method | Sensor | R^2 with CBI | Source |
|--------|--------|--------------|--------|
| SMA burned fraction | AVIRIS (224 bands) | **0.86** | Veraverbeke 2014 |
| MESMA fractions (RFR) | PRISMA (hyperspectral) | **0.64-0.79** | Quintano 2023 |
| dNBR (optimal bands) | AVIRIS (hyperspectral) | 0.71 | van Gerrevink 2021 |
| dNBR (standard) | Landsat (multispectral) | 0.67 | van Gerrevink 2021 |
| dNBR | Sentinel-2 | 0.67 | Howe 2022 |
| MESMA fractions (RFR) | Sentinel-2 (multispectral) | **0.27-0.53** | Quintano 2023 |
| SMA burned fraction | Landsat (multispectral) | **0.52** | Veraverbeke 2014 |

### What Benefits Most from 426 Bands

1. **MESMA / SMA** — The biggest beneficiary. R^2 improvement: 0.86 vs 0.52 (65% relative).
2. **Cellulose Absorption Index** — CAI requires precise 2100nm measurement. R^2 = 0.808 from PRISMA (Quintano 2023).
3. **LFMC estimation** — Water absorption depth measurement at 970nm, 1200nm, 1700nm.
4. **Post-fire mineral characterization** — Heated soils produce diagnostic iron oxide/clay mineral transformations.
5. **Vegetation recovery** — Red-edge position benefits from 5nm sampling in 680-750nm.

### What Does NOT Benefit Much

Standard dNBR: only marginal improvement (R^2 0.71 vs 0.67). Two-band ratio doesn't leverage 426 bands. **This validates our MESMA-first approach.**

---

## 4. Relevant Sensor Heritage

### Published Fire Analysis by Platform

| Sensor | Type | Resolution | Fire Studies |
|--------|------|-----------|--------------|
| AVIRIS/AVIRIS-NG | Airborne | 4-20m, 224 bands, 10nm | Extensive — Veraverbeke 2014, Dennison 2018, van Gerrevink 2021 |
| PRISMA | Spaceborne | 30m, 239 bands, 12nm | Quintano 2023 — most comparable to FireSpec |
| EnMAP | Spaceborne | 30m, 242 bands, 6.5-10nm | Inventoried fire areas (enmap.org 2022), no MESMA results |
| EMIT | Spaceborne (ISS) | 60m, 285 bands, 7.4nm | No fire MESMA yet; vegetation/mineral products in development |
| **Tanager-1** | Spaceborne | 30m, 426 bands, ~5nm | **No published fire work — FireSpec fills this gap** |

### Cross-Sensor Calibration

Tanager's ISOFIT v2.9.5 atmospheric correction is the same algorithm developed at JPL for AVIRIS/EMIT, providing cross-sensor consistency.

#### P14. Leite et al. (2025) — "Leveraging next generation spaceborne observations for fuel monitoring and wildland fire management"
- **Journal:** Remote Sensing in Ecology and Conservation
- **Finding:** Most comprehensive recent review positioning spaceborne imaging spectroscopy for fire management. Covers DESIS, PRISMA, HISUI, EnMAP, EMIT. Identified imaging spectroscopy as complementary to lidar (structure) and SAR (moisture correction).
- **Relevance:** **CRITICAL** — Directly contextualizes Tanager-1 for fire management.

---

## 5. Validation Data Sources

### Burn Severity

| Source | Description | Access |
|--------|-------------|--------|
| MTBS | Landsat dNBR severity maps, all US fires >1000ac since 1984 | mtbs.gov, GEE |
| BARC | USGS rapid-response dNBR, within 7 days of containment | burnseverity.cr.usgs.gov |
| RAVG | USFS post-fire vegetation assessment | USFS data |
| CBI plots | Compiled field measurements (0.0-3.0), 1994-2018 CONUS | ScienceBase, GEE |

### LFMC

| Source | Description | Access |
|--------|-------------|--------|
| Globe-LFMC 2.0 | 280,000+ measurements, 2,000+ sites, 15 countries, 47 years | nature.com/articles/s41597-024-03159-6 |
| Varga & Jones 2026 | 32-year SoCal chaparral LFMC, 4 species, SLO to LA County | Scientific Data |
| NFMD | National Fuel Moisture Database (US) | wfas.net |

### Comparison Baselines

| Source | Resolution | Bands | Access |
|--------|-----------|-------|--------|
| Sentinel-2 | 10-20m | 13 | Copernicus Data Space |
| Landsat 8/9 | 30m | 11 | USGS EarthExplorer |
| MODIS | 500m-1km | 7-36 | NASA LPDAAC |

---

## 6. Synthesis — Recommended Approach for FireSpec

### Primary Method: MESMA for Burn Severity

The literature is unambiguous — MESMA outperforms dNBR by a wide margin with hyperspectral data. Recommended pipeline:

1. **Endmember library:** White Ash, Black Char, Green Vegetation, NPV, Bare Soil, Shade. Source from USGS Spectral Library v7 and ECOSTRESS, resampled to Tanager band centers.
2. **Endmember selection:** In-CoB method (Dennison 2018)
3. **Band selection:** uSZU (uncorrelated Stable Zone Unmixing) for efficiency (Dennison 2018)
4. **Severity mapping:** Random Forest Regression of MESMA fraction images against CBI (Quintano 2023 approach)
5. **Comparison:** Reproduce with Sentinel-2 bands simulated from Tanager to quantify hyperspectral advantage

### LFMC Estimation

1. **Spectral features:** Continuum-removed absorption depth at 970nm, 1200nm, 1700nm water features + 2100nm, 2300nm dry matter features
2. **Model:** PLSR or Random Forest regression (Qi 2014 approach)
3. **Validation:** Globe-LFMC 2.0 + Varga 2026 SoCal chaparral dataset
4. **Critical insight:** Account for dry mass contribution — not just water bands (Qi 2014)

### Temporal Analysis (Unique Strength)

Exploit the 7-date LA time series for:
- Pre-fire LFMC conditions (Dec 15, 2024)
- Immediate post-fire severity (Jan 23, 2025)
- Recovery trajectory (Apr, Jul 2025)

No published spaceborne hyperspectral multi-temporal MESMA fire study exists. This would be novel.

### Validation Strategy

1. **Burn severity:** BARC rapid-response maps (likely available) + MTBS dNBR (if published) + CBI field plots (if available)
2. **LFMC:** Globe-LFMC 2.0 site co-location + Varga 2026 chaparral species-specific data
3. **Baseline comparison:** Sentinel-2 dNBR and simulated broadband from Tanager (spectral resampling)

### Literature Gaps FireSpec Addresses

1. **First Tanager-1 wildfire analysis** — no published work exists
2. **First spaceborne LFMC at 5nm resolution** — unprecedented
3. **First multi-temporal spaceborne hyperspectral MESMA for fire** — only done with Landsat broadband before
4. **First 426-band vs broadband MESMA at same GSD** — perfectly controlled comparison

### Risk Factors

| Risk | Mitigation |
|------|------------|
| 30m GSD smooths fine-scale severity | MESMA handles mixed pixels; 30m is MTBS standard |
| Atmospheric correction artifacts in water bands | ISOFIT v2.9.5 is state-of-art; verify empirically |
| No Tanager-specific endmember library | Build from USGS/ECOSTRESS + image endmembers |
| CBI field plots may not be available for LA fires | Use BARC + Sentinel-2 as fallback validation |

---

## Full Citation Index

| ID | Citation | Relevance |
|----|----------|-----------|
| P1 | Quintano, C. et al. (2023). First evaluation of fire severity retrieval from PRISMA. *RSE*, 113670. | Critical |
| P2 | Quintano, C. & Fernandez-Manso, A. (2023/24). MESMA PRISMA fire severity. *Proc. SPIE 12688*. | Critical |
| P3 | Veraverbeke, S. et al. (2014). Assessing fire severity using AVIRIS. *RSE*, 154, 153-163. | Critical |
| P4 | Quintano, C. et al. (2013). MESMA burn severity from Landsat. *RSE*. | Important |
| P5 | Dennison, P.E. et al. (2018). Endmember and band selection for MESMA. *Remote Sensing*, 10(3), 389. | Critical |
| P6 | Rao, K. et al. (2020). Enhanced burn severity with ET + MESMA fractions. *RSE*. | Important |
| P7 | Yebra, M. et al. (2013). Global review of RS of LFMC. *RSE*. | Critical |
| P8 | Veraverbeke, S. et al. (2018). Hyperspectral RS of fire: State-of-the-art. *RSE*, 216, 105-121. | Critical |
| P9 | Qi, Y. et al. (2014). Spectroscopic LFMC and leaf dry mass. *RSE*. | Critical |
| P10 | Various (2019-2021). Spectral absorption indices 970nm, 1200nm, 1660nm. | Important |
| P11 | Quan, X. et al. (2021). Global FMC from MODIS. *IJAEOG*, 101. | Important |
| P12 | Varga, K. & Jones, C. (2026). 32-yr SoCal chaparral LFMC. *Scientific Data*. | Critical |
| P13 | van Gerrevink, M.J. & Veraverbeke, S. (2021). Hyperspectral dNBR sensitivity. *Remote Sensing*, 13(22), 4611. | Critical |
| P14 | Leite, R.V. et al. (2025). Spaceborne observations for fire management. *RSEC*. | Critical |
| P15 | Yebra, M. et al. (2024). Globe-LFMC 2.0. *Scientific Data*. | Critical |
| P16 | Robichaud, P. et al. (2007). Postfire soil burn severity with hyperspectral unmixing. *RSE*. | Useful |
| P17 | Howe, A. et al. (2022). Sentinel-2 vs Landsat burn severity. *Remote Sensing*, 14(20), 5249. | Important |
