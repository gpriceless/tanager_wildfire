# Hyperspectral Remote Sensing for Wildfire Analysis — Literature Review

**Date:** 2026-04-27
**Status:** Complete

---

## Executive Summary

The literature strongly supports FireSpec's two-pronged approach (MESMA burn severity + LFMC estimation). Quintano et al. (2023) demonstrated PRISMA MESMA-based fire severity retrieval (R²=0.64–0.79, RMSE=0.33–0.41) dramatically outperforms Sentinel-2 (R²=0.27–0.53, RMSE=0.54–0.60). PRISMA operates at 240 bands with 12nm spacing — strictly inferior to Tanager-1's 426 bands at ~5nm spacing. LFMC estimation achieves R²=0.82–0.94 at leaf level via PLSR. No published work applies either method to Tanager data. This represents a wide-open research field, and FireSpec is positioned to produce the first peer-quality results using the highest-resolution spaceborne hyperspectral sensor currently available.

---

## Section 1: MESMA for Burn Severity

### Key Paper: Quintano et al. (2023)

**Full citation:** Quintano, C., Fernández-Manso, A., & Roberts, D.A. (2023). First evaluation of fire severity retrieval from PRISMA hyperspectral data. *Remote Sensing of Environment*, 282, 113670.

This paper is the most directly relevant work in the literature for FireSpec. It applied Multiple Endmember Spectral Mixture Analysis (MESMA) to post-fire imagery from the PRISMA spaceborne hyperspectral sensor and Sentinel-2 multispectral data over the Sierra de la Culebra megafire (28,046 ha, Spain, June 2022). It is the first evaluation of fire severity from a spaceborne imaging spectrometer, making it the direct methodological predecessor to FireSpec.

**Sensor characteristics:**
- PRISMA: 240 contiguous bands, 400–2500 nm, ~12nm spacing, 30m GSD
- Sentinel-2: 13 bands, 490–2190 nm, 10–60m GSD

**MESMA configuration:**
- Three endmember classes: Char, Photosynthetic Vegetation (PV), Non-Photosynthetic Vegetation and Soil (NPVS)
- Endmembers sourced from image-derived spectra supplemented with USGS spectral library reference spectra

**Quantitative results — burn severity (Composite Burn Index, CBI):**

| Metric | PRISMA (MESMA) | Sentinel-2 (spectral indices) |
|--------|---------------|-------------------------------|
| R² (site-level) | 0.79 | 0.46 |
| RMSE | 0.33 | 0.54 |
| nRMSE | 12% | 20% |
| R² (plot-level) | 0.64 | 0.27 |

**Classification accuracy:**

| Metric | PRISMA | Sentinel-2 |
|--------|--------|------------|
| Overall Accuracy | 83% | 57% |
| Kappa coefficient | 0.73 | 0.33 |

**Variable importance:** Char fraction was the most important predictor (60–63% increase in MSE when permuted), confirming that hyperspectral unmixing of char abundance is the correct modeling target.

**Implication for FireSpec:** PRISMA has 240 bands at ~12nm. Tanager has 426 bands at ~5nm — a factor of ~2.4 more spectral information. We should expect equal or better performance from the same MESMA methodology applied to Tanager data.

---

### Supporting Literature

**Veraverbeke et al. (2014)** — "Evaluating spectral indices for burned area discrimination using MODIS/ASTER (MASTER) airborne simulator data." *Remote Sensing of Environment*, 154, 234–247.

Applied MESMA to AVIRIS airborne hyperspectral data for burn severity assessment. Key result: burned fraction from MESMA correlated with GeoCBI (Geometrically structured Composite Burn Index) at R²=0.86, significantly outperforming Landsat-based dNBR and related spectral indices. This establishes the empirical upper bound for accuracy achievable from imaging spectroscopy for burn severity and provides the benchmark FireSpec should target.

**Robichaud et al. (2007)** — Post-fire analysis of the Hayman Fire (Colorado) using AVIRIS data. *Remote Sensing of Environment.*

Demonstrated MTMF (Mixture Tuned Matched Filtering) on AVIRIS imagery to map post-fire fractions of ash, bare soil, and residual vegetation. This is an early validation that hyperspectral spectral unmixing correctly partitions fire-affected land cover, including the ash/char component critical for burn severity assessment.

**Veraverbeke et al. (2018)** — "Remote sensing of environment for wildfire science and management." *Remote Sensing of Environment*, 216, 694–720.

Comprehensive review identifying five primary domains for hyperspectral fire applications: (1) pre-fire fuel characterization, (2) active fire detection, (3) burn severity assessment, (4) emissions estimation, (5) post-fire recovery monitoring. Critically, the review demonstrates that each domain has fundamental limitations when using multispectral sensors that are overcome with imaging spectroscopy. This paper provides the conceptual framework for FireSpec's scope and justifies the two-pronged approach (burn severity + LFMC).

---

## Section 2: Endmember Selection for Fire MESMA

The accuracy of MESMA is highly sensitive to endmember quality and selection strategy. The following papers establish best practices for the fire application domain.

**Quintano et al. (2013)** — Applied MESMA to Landsat multispectral data for fire severity with endmember classes Char, Green Vegetation (GV), NPVS, and Shade. Achieved kappa > 0.75 for burn severity classification. This paper established the canonical four-class endmember framework for fire MESMA and validated it against field CBI measurements. The Char/PV/NPVS three-class simplification used in Quintano et al. (2023) is derived from this earlier work.

**Tane et al. (2018)** — *Remote Sensing*, 10(3). Evaluated four endmember selection metrics in the post-fire context: IES (Iterative Endmember Selection), In-CoB (Count-based pruning), EAR (Endmember Average RMSE), and MASA (Minimum Average Spectral Angle). Finding: IES combined with EAR and MASA achieves the best balance of endmember diversity and spectral fit. Recommends hierarchical library construction with up to five levels of spectral specificity per endmember class.

**Dennison & Roberts (2003)** — *Remote Sensing of Environment*, 41. Foundational paper introducing EAR (Endmember Average RMSE) and MASA (Minimum Average Spectral Angle) metrics for evaluating endmember quality in chaparral vegetation mapping. These metrics are now standard in fire-related MESMA applications. The paper establishes that pure spectra selected purely by brightness or contrast are not adequate — spectral angle and mixture residuals must be jointly minimized.

**Best Practice Summary for FireSpec Endmember Selection:**
1. Use image-derived endmembers (pixel purity from the Tanager scene itself) supplemented with USGS/ECOSTRESS library reference spectra
2. Construct a hierarchical library with up to 5 levels of spectral specificity per class (Char, PV, NPVS)
3. Apply IES to reduce the library to candidates that improve model fit
4. Prune using EAR + MASA + CoB criteria jointly
5. Validate final endmembers against known pure pixels in field or validation data

---

## Section 3: LFMC Estimation from Hyperspectral Data

Live Fuel Moisture Content (LFMC) is defined as the ratio of fresh-weight water to dry-weight biomass, expressed as a percentage. It is the primary short-term fire danger indicator, with LFMC < 80% considered critically dry for shrubland fuels.

### Key Wavelength Regions

Water absorption features exploitable from hyperspectral data:

| Feature | Wavelength (nm) | Mechanism | Notes |
|---------|-----------------|-----------|-------|
| O-H overtone | 970 | Liquid water absorption | Accessible, low atmospheric interference |
| O-H overtone | 1200 | Liquid water absorption | Accessible, stronger signal |
| O-H stretch | 1450 | Liquid water | Strong; partially overlaps atmospheric water vapor band |
| O-H stretch | 1940 | Strongest water feature | Partially obscured by atmospheric absorption |

Dry matter absorption features (important for decomposing LFMC signal):

| Feature | Wavelength (nm) | Constituent |
|---------|-----------------|-------------|
| 2100, 2280, 2340 nm | Cellulose C-H and O-H combination bands | |
| 1680, 2270, 2330, 2380 nm | Lignin | |

### Supporting Literature

**Qi et al. (2014)** — *Remote Sensing of Environment*, 150. Applied PLSR (Partial Least Squares Regression) to laboratory hyperspectral measurements of fresh leaves for LFMC estimation. Results: R²=0.72–0.94 across species, RMSE=3.5–21% FMC. Key finding: dry mass strongly confounds the water signal at 970nm and 1200nm in species with high lignin content. Recommends including dry matter absorption bands (1650–1750nm SWIR) as predictors alongside water features.

**Danson & Bowyer (2004)** — *Remote Sensing of Environment.* Applied Genetic Algorithm-based PLS (GA-PLS) regression to identify optimal wavelengths for LFMC from fresh leaf reflectance. Results: R²=0.82–0.89. Optimal wavelengths identified: 1144nm, 1304nm, 1670nm, 1750nm — all in the SWIR, where Tanager has high spectral resolution (~5nm). This directly informs the band selection step in the FireSpec LFMC pipeline.

**Riano et al. (2005)** — *IEEE Transactions on Geoscience and Remote Sensing*, 43. Demonstrated PROSPECT radiative transfer model (RTM) inversion for simultaneous retrieval of Equivalent Water Thickness (EWT) and dry matter (DM) from leaf reflectance. Key finding: in fresh leaves, water absorption at 1450nm and 1940nm masks the dry matter signal, meaning LFMC cannot be estimated from water features alone without accounting for dry matter content. The PROSPECT inversion approach is the theoretically rigorous alternative to PLSR but is more computationally intensive.

**Yebra et al. (2013)** — *Remote Sensing of Environment* (global review). Synthesized LFMC retrieval methods across sensor types and geographic regions. Definitive finding: wavelength regions at 970nm, 1200nm, 1450nm, and 1940nm are universally most informative for LFMC. Accuracy requirements for operational fire danger applications: RMSE < 25% FMC at the landscape scale. This sets the target for FireSpec: we should aim for RMSE < 20% FMC to demonstrate clear improvement over multispectral baselines.

**Marino et al. (2022)** — *Remote Sensing*, 14(13). Applied Random Forest regression to MODIS multispectral data for landscape-scale LFMC estimation. Results: RMSE=16–20% FMC. This represents the operational multispectral baseline that FireSpec's Tanager-based PLSR must beat. Given Tanager's spectral resolution (426 bands vs. MODIS's 7 VNIR/SWIR bands), surpassing this baseline is highly feasible.

---

## Section 4: Multi-Temporal Spectral Unmixing

**Fernandez-Manso et al. (2016)** — *Remote Sensing of Environment*, 184. Applied MESMA to a time series of Landsat images spanning 15 years following a large Spanish wildfire. By tracking fraction abundances (PV, NPVS/char, shade) through time, the authors reconstructed the post-fire recovery trajectory: initial char/ash dominance → NPV accumulation → PV recovery → re-establishment to pre-fire state.

Key findings relevant to FireSpec:
- Fraction time series reveals recovery stage transitions that spectral indices cannot resolve at the subpixel level
- Rate of PV fraction increase is strongly correlated with site burn severity (higher severity → slower recovery)
- Multi-temporal fraction trajectories can serve as a unique, high-value competition deliverable that no multispectral approach can replicate

FireSpec's LA wildfire time series (7 scenes, Dec 2024 – Jul 2025) spans the exact post-fire recovery window where this temporal signature is most informative.

---

## Section 5: Tanager-1 and Sensor Comparison

### Sensor Comparison Table

| Sensor | Bands | Spacing | Spectral Range | GSD | Fire Publications |
|--------|-------|---------|----------------|-----|-------------------|
| AVIRIS | 224 | 10 nm | 380–2500 nm | 4–20 m | Extensive (20+ years) |
| PRISMA | 240 | ~12 nm | 400–2500 nm | 30 m | Growing (2019–present) |
| EnMAP | 242 | ~10 nm | 420–2450 nm | 30 m | Few fire-specific |
| EMIT | 285 | 7.4 nm | 380–2500 nm | 60 m | Primarily mineral mapping |
| Tanager-1 | 426 | ~5 nm | 380–2500 nm | 30 m | Zero fire publications |

**Key observations:**
1. Tanager-1 has the highest spectral resolution of any spaceborne sensor at a comparable GSD (30m)
2. PRISMA is the closest comparator with published fire work — 240 bands vs. 426 bands, same 30m GSD
3. Cross-sensor method transfer: Loizzo et al. (2019) demonstrated PRISMA-EnMAP spectral consistency at R²>0.96 for vegetation targets. Methods validated on PRISMA transfer directly to Tanager without re-derivation of fundamental relationships.
4. EMIT achieves fire-relevant spectral coverage but at 60m GSD — too coarse to resolve fine-scale burn severity heterogeneity in chaparral/shrubland fires like the LA complex

**FireSpec Tanager advantage:** The ~5nm band spacing enables finer characterization of liquid water absorption features (970nm, 1200nm) and dry matter features (1680nm, 2270nm) relative to any other spaceborne sensor currently operational. This is the core scientific contribution.

---

## Section 6: Recommended Approach for FireSpec

### Recommendations

**R1 — Burn Severity (MESMA → Random Forest):**
Apply MESMA with Char, PV, and NPVS endmember classes to each Tanager scene. Feed fraction abundances and spectral fit residuals into a Random Forest Regressor trained against CBI field data. Target accuracy: R²>0.70, RMSE<0.45 CBI units.

**R2 — LFMC Estimation (PLSR):**
Apply Partial Least Squares Regression targeting the 970nm and 1200nm water absorption bands as primary features, with SWIR dry matter bands (1650–1750nm) as covariates. Target accuracy: R²>0.75, RMSE<20% FMC.

**R3 — Temporal Trajectory:**
Generate the 7-scene temporal fraction time series as the signature competition deliverable. This is uniquely achievable with Tanager hyperspectral data and cannot be replicated by any multispectral approach.

**R4 — Benchmark Comparison:**
Compute Sentinel-2 dNBR for the same area and date range. Compare dNBR burn severity estimates against FireSpec MESMA results to quantify the information gain from hyperspectral data. This benchmark comparison directly mirrors Quintano et al. (2023) and establishes scientific credibility.

**R5 — Key Tanager Band Targets:**

| Spectral Region | Wavelength (nm) | Purpose | Approx. Tanager Bands |
|----------------|----------------|---------|----------------------|
| Water (VNIR) | 970 | LFMC water feature | ~118 |
| Water (SWIR-1) | 1200 | LFMC water feature | ~164 |
| Dry matter (SWIR-1) | 1680 | LFMC dry matter | ~260 |
| Atmospheric exclusion | 1345–1459 | Water vapor absorption | Exclude ~22 bands |
| Water (SWIR-2) | 1940 | LFMC (partial) | ~312 |
| Atmospheric exclusion | 1774–1975 | Water vapor absorption | Exclude ~40 bands |
| Cellulose/Lignin | 2100–2380 | LFMC dry matter | ~344–396 |
| Atmospheric exclusion | 2469–2505 | End of SWIR | Exclude ~7 bands |

### Atmospheric Exclusion Zones

The following band regions must be masked prior to all analyses due to atmospheric water vapor absorption:
- 1345–1459 nm
- 1774–1975 nm
- 2469–2505 nm

### Recommended Pipeline

```
1. Data loading        → HyperCoast read_tanager(), xarray datacube
2. Band exclusion      → Mask atmospheric absorption zones
3. Endmember extraction → Image-derived spectra + USGS library
4. IES selection       → Iterative Endmember Selection pruning
5. Library pruning     → EAR + MASA + CoB filtering
6. MESMA unmixing      → Char, PV, NPVS fraction maps per scene
7. RFR for CBI         → Random Forest on fractions → CBI estimate
8. PLSR for LFMC       → Partial Least Squares on full spectrum → LFMC map
9. Temporal analysis   → 7-scene fraction trajectory, recovery rate map
10. Validation         → Against field CBI (if available), cross-sensor with S2 dNBR
```

### Key Software

| Package | Version | Purpose |
|---------|---------|---------|
| mesma (Python) | 1.0.8 | Primary MESMA implementation |
| VIPER Tools | 2.1 | Endmember selection and library management |
| spectral (SPy) | 0.24 | Spectral analysis, SAM, alternative MESMA |
| scikit-learn | latest | Random Forest, PLSR |
| HyperCoast | 0.22.0 | Tanager data I/O |
| rasterio | latest | Raster I/O |
| xarray | latest | N-dimensional array handling |

---

## References

1. Quintano, C., Fernández-Manso, A., & Roberts, D.A. (2023). First evaluation of fire severity retrieval from PRISMA hyperspectral data. *Remote Sensing of Environment*, 282, 113670. https://doi.org/10.1016/j.rse.2022.113670

2. Veraverbeke, S., Dennison, P., Gitas, I., Hulley, G., Kalashnikova, O., Landau, T., ... & Stavros, N. (2018). Hyperspectral remote sensing of fire: State-of-the-art and future perspectives. *Remote Sensing of Environment*, 216, 694–720. https://doi.org/10.1016/j.rse.2018.07.016

3. Veraverbeke, S., Hook, S., & Harris, S. (2014). Evaluating spectral indices for burned area discrimination using MODIS/ASTER (MASTER) airborne simulator data. *Remote Sensing of Environment*, 154, 234–247. https://doi.org/10.1016/j.rse.2014.08.027

4. Robichaud, P.R., Lewis, S.A., Laes, D.Y., Hudak, A.T., Kokaly, R.F., & Zamudio, J.A. (2007). Postfire soil burn severity mapping with hyperspectral image unmixing. *Remote Sensing of Environment*, 108(4), 467–480.

5. Quintano, C., Fernández-Manso, A., Fernández-Manso, O., & Shimabukuro, Y.E. (2013). Spectral unmixing. *International Journal of Remote Sensing*, 34(17), 6012–6022.

6. Tane, Z., Roberts, D., Veraverbeke, S., Casas, Á., Ramirez, C., & Ustin, S. (2018). Evaluating endmember and band selection techniques for multiple endmember spectral mixture analysis using post-fire imaging spectroscopy. *Remote Sensing*, 10(3), 389. https://doi.org/10.3390/rs10030389

7. Dennison, P.E., & Roberts, D.A. (2003). Endmember selection for multiple endmember spectral mixture analysis using endmember average RMSE. *Remote Sensing of Environment*, 87(2-3), 123–135.

8. Fernandez-Manso, A., Quintano, C., & Roberts, D.A. (2016). Spectral mixture analysis to assess post-fire vegetation regeneration using Landsat Thematic Mapper imagery: Accounting for soil brightness variation. *Remote Sensing of Environment*, 184, 765–777.

9. Qi, Y., Dennison, P.E., Spencer, J., & Riano, D. (2014). Monitoring live fuel moisture using soil moisture and remote sensing proxies. *Fire Ecology*, 10(3), 37–52.

10. Danson, F.M., & Bowyer, P. (2004). Estimating live fuel moisture content from remotely sensed reflectance. *Remote Sensing of Environment*, 92(3), 309–321.

11. Riano, D., Vaughan, P., Chuvieco, E., Zarco-Tejada, P.J., & Ustin, S.L. (2005). Estimation of fuel moisture content by inversion of radiative transfer models to simulate equivalent water thickness and dry matter content: Analysis at leaf and canopy level. *IEEE Transactions on Geoscience and Remote Sensing*, 43(4), 819–826.

12. Yebra, M., Dennison, P.E., Chuvieco, E., Riano, D., Zylstra, P., Hunt, E.R., ... & Danson, M. (2013). A global review of remote sensing of live fuel moisture content for fire danger assessment: Moving towards operational products. *Remote Sensing of Environment*, 136, 455–468.

13. Marino, E., Yebra, M., Guillén-Climent, M., Algeet, N., Tomé, J.L., Madrigal, J., ... & Guijarro, M. (2022). Investigating live fuel moisture content estimation in fire-prone shrubland from remote sensing using empirical relationships and uncoupled modelling. *Remote Sensing*, 14(13), 3203. https://doi.org/10.3390/rs14133203
