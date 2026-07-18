# FireSpec: Hyperspectral Burn Severity and Live Fuel Moisture Mapping with Tanager-1

**Technical Memo — Planet Tanager Open Data Competition**
**Author:** gpriceless · **Repository:** github.com/gpriceless/tanager_wildfire (MIT License)
**Study area:** Palisades (ignited January 7, 2025) and Hughes (ignited January 22, 2025) Fires, Los Angeles, CA

---

## Abstract

FireSpec is an open-source Python toolkit that turns Tanager-1's 426-band, 5 nm, 30 m
hyperspectral imagery into wildfire products for the 2025 LA fires. We map burn severity with
Multiple Endmember Spectral Mixture Analysis (MESMA) and a Random-Forest classifier trained
against a Hughes-fire BAER Soil Burn Severity reference (provisional — see below), cross-check
the dNBR map against an independent CAL FIRE DINS structure-damage survey (F1 = 0.774 for
any-damage detection; Spearman rho = 0.358, a modest but positive structure-to-pixel rank
signal), map canopy water content from resolved 970/1200/2100 nm absorption features, and
quantify recovery across four acquisitions (Dec 2024–Apr 2025). A spectral-degradation
experiment shows Tanager recovers MESMA char fraction at R² = 0.991 vs native (EMIT-equivalent)
but only R² = 0.361 at Sentinel-2's 10 broadband channels — an absolute R² gap of ~0.63 (Sentinel-2
leaves roughly two-thirds of the char-fraction variance unexplained) that broadband indices
(NBR, NDVI; R² > 0.99 for all sensors) fail to expose. Because the BAER reference is of
unverifiable provenance and no colocated field LFMC exists for these scenes, we report a
provisional severity kappa and ship spectral water indices rather than a calibrated LFMC
percentage; each caveat travels with its number below.

## 1. Introduction

Operational burn-severity mapping relies on the differenced Normalized Burn Ratio (dNBR), a
two-band ratio that collapses ground condition into one scalar: a 50%-char/50%-canopy pixel
and a 100%-moderately-scorched pixel can yield identical dNBR despite very different
post-fire ecology. Tanager-1's 426 contiguous ~5 nm bands (380–2500 nm) resolve the distinct
absorption features of char, ash, photosynthetic vegetation (PV), non-photosynthetic
vegetation (NPV), and soil, enabling spectral unmixing (MESMA; Roberts et al. 1998) to
recover *what is on the ground*, not just *how much changed*. The same fine sampling resolves
narrow water-absorption features (970, 1200, 1700, 2100 nm) — the physical basis of live fuel
moisture retrieval, critical for pre-fire risk since LFMC below ~60% marks the flammable
regime for chaparral (Roberts et al. 2006).

We apply this to the January 2025 Palisades and Hughes Fires using four Tanager-1 acquisitions
(Dec 15, 2024 pre-fire; Jan 23, 2025 post-fire, two swaths; Apr 7, 2025 recovery), and
quantify Tanager's advantage over EMIT, PRISMA, and Sentinel-2 via spectral-response
degradation simulation.

## 2. Methodology

**Burn severity (MESMA).** Pre-fire (Dec 15) and post-fire (Jan 23, swath 2 — the swath
overlapping the pre-fire footprint by 434 km², 85.6% of scene area) cubes are masked
(nodata/cloud/water), reduced to 11 diagnostic bands, and reprojected to a common UTM grid.
dNBR locates the burn scar; a 300×284 px crop (66,167 valid pixels) centered on dNBR≥0.44
(USGS moderate-high threshold, Key & Benson 2006) is used for interactive MESMA. No external
endmember library was available locally, so a 4-class library (char, PV, NPV, soil) is
extracted directly from the image via percentile-based region selection. Fractions are
normalized (shade removed, clamped, renormalized). A Random-Forest classifier maps the
4-fraction feature vector (char, PV, NPV, soil) to severity classes, trained against a
**BAER Soil Burn Severity (SBS)** raster for the neighbouring Hughes fire — nominally a
field-corrected severity product imaged in the same Tanager overpass, though this particular file
carries no metadata and its provenance could not be verified (see Limitations). MESMA is run on
the Hughes scene with the same endmember library; cross-validated accuracy, Cohen's kappa,
and per-class F1 are reported against the BAER reference as provisional figures. BAER code 4 ("High", ~141 px) is kept
as its own class rather than merged, despite a median NBR brighter than code 3 that violates the
single-date-NBR ordering (see Limitations). An earlier revision used a synthetic CBI proxy
(CBI ≈ 3 × char fraction) as the training target; that has been replaced with the BAER reference.
We additionally cross-check the dNBR map against the CAL FIRE DINS structure-damage survey — a
field-collected reference that *does* cover the Palisades footprint (§7.1).

**Canopy water content (spectral indices).** On the Apr 7, 2025 scene (902,891 px, 75.8% valid),
we compute narrowband water indices (SAI970, SAI1200, three NDWI variants, the Peñuelas Water
Index) and convex-hull continuum-removal band depths at the water-absorption centers directly
from Tanager reflectance. These are physically-interpretable *relative* water-content indicators,
not a calibrated LFMC percentage. We assess whether a calibrated LFMC regression is possible by
counting how many Globe-LFMC 2.0 field sites colocate with the scene footprint: on the order of
one site per swath (area ratio: ~60 SoCal sites over a 230×165 km box vs an ~18–30 km swath) and
the SoCal record ends 2023-01-30 while the scene is April 2025 — far below the sites needed to
train a full-spectrum PLSR. We therefore do not ship a calibrated LFMC map; the
`train_lfmc_plsr`/`predict_lfmc` functions remain in the toolkit for when colocated field data
exists, and PROSPECT/PROSAIL inversion is the training-free path (Future work).

**Temporal recovery.** Four of eleven cataloged scenes were locally downloaded, yielding two
overlap pairs: Palisades (Dec 15 → Jan 23 swath 2, burn onset) and a second pairing (Jan 23
swath 1 → Apr 7, 74 days, "Hughes" footprint). NBR, NDVI, and SAI970 trajectories use each
overlap's common grid only, since the pre-fire scene does not overlap the recovery footprints.

**Sensor comparison.** Native Tanager reflectance (128×128 px crop, 12,539 valid px) is
convolved with **literature-derived** spectral response functions — Gaussian band models built
from each sensor's published band centers and FWHM, not the manufacturers' official measured SRF
files — to simulate EMIT (285 bd), PRISMA (239 bd), and Sentinel-2 MSI (10 bd). Broadband indices, narrowband absorption-feature
depths, and MESMA fractions on each degraded product are compared against native Tanager (R²
vs. native).

## 3. Results

**Burn severity.** dNBR over the Palisades overlap (n=111,568) shows mean +0.211, median
+0.113. The MESMA crop yields normalized mean fractions char=0.383, NPV=0.259, PV=0.248,
soil=0.111. A Random-Forest classifier trained on MESMA fractions against the Hughes-fire BAER
Soil Burn Severity reference (same overpass) is cross-validated on that reference; the trained
model is then applied to the Palisades fractions. Its cross-validated Cohen's kappa is reported
as a **provisional** figure: the BAER raster's class coding is inferred (no metadata), its
identity is unverified (per-class proportions are near-inverted vs the published USDA BAER
summary), and the scene is ~1 day post-ignition while BAER SBS is mapped after containment — the
kappa cannot be read as a validated accuracy and must be recomputed under the 5-class coding
(the prior 4-class value was 0.527). An NBR-threshold baseline (single-date NBR calibrated from
BAER class medians) is reported alongside; it is expected to misplace the anomalous code-4 class,
which the fraction-based RF classifier is not bound to NBR ordering to reproduce. As an
internal-consistency check (not ground-truth validation), MESMA char fraction vs. dNBR
(n=21,495) shows correlation r=0.365 — MESMA fractions and dNBR measure related but distinct
signals, and char fraction alone is not a severity product (dark non-vegetated surfaces confuse
the image-derived char endmember).

**Structure-level cross-check (DINS).** The CAL FIRE DINS survey (12,081 inspected structures,
No-Damage→Destroyed) is an independent, field-collected reference covering the Palisades
footprint. Against the dNBR map it gives Spearman rho = 0.358 between damage ordinal and dNBR
and F1 = 0.774 for "any damage" detection at the Key & Benson burned/unburned boundary
(dNBR ≥ 0.1). The rho is deliberately modest and expected to be: DINS is *structure*-level truth
scored against a 30 m *pixel* product, so a destroyed house on a defended lot can sit in a
low-dNBR pixel. The value that matters is the monotonic climb of mean dNBR with each damage step
and the high any-damage recall — both of which hold. This is the memo's only fully independent
field validation and belongs beside the provisional BAER kappa, not in place of it.

**Canopy water content.** SAI970 and SAI1200 average 0.073 and 0.076 across 683,948 valid
pixels; convex-hull continuum-removal depth peaks at 2100 nm (mean 0.234, p95 0.432), consistent
with strong cellulose/water absorption. These indices are *relative* water-content indicators
computed directly from Tanager reflectance — they rank pixels by absorption depth but do not
report a moisture percentage. No calibrated LFMC map is produced: the Globe-LFMC 2.0
data-availability assessment finds on the order of one field site colocated with the scene
footprint (an area-ratio limit, not fixable with more data), and the SoCal field record predates
the scene by ~2 years, so no honest field-calibrated regression can be trained here. An earlier
revision filled this gap with a synthetic PLSR (training targets generated in-notebook via
`inject_water_absorption`, scaling injected 970/1200/1700/2100 nm depressions by a randomly
assigned LFMC); its reported CV R² measured that injection formula rather than spectroscopy, and
its per-pixel LFMC map — mean near the dry extreme, most pixels flagged "fire-prone" on an April
scene when the Globe-LFMC April SoCal shrub median is ~119% — was implausible against external
climatology. Both the synthetic regression and that map have been removed.

**Temporal recovery.** NBR on the Hughes-footprint pair rises from 0.0070 (Jan 23) to 0.2098
(Apr 7), +0.0822 NBR/30 days; linear extrapolation reaches the Palisades pre-fire NBR
baseline (0.2352) in an estimated 9.3 further days. NDVI and SAI970 (LFMC proxy) recover
faster, already exceeding pre-fire baselines by Apr 7 (extrapolated overshoot 10.1 and 19.5
days). Palisades-overlap BARC classes from dNBR (n=111,568) show 46.7% unburned, 11.3% high
severity.

**Sensor comparison.** Broadband indices are essentially sensor-invariant: NBR/NDVI mean R²
vs. native is 0.999 (EMIT), 0.995 (PRISMA), 0.996 (Sentinel-2). Narrowband absorption-feature
depth is far more sensitive: mean R² is 0.965 (EMIT), 0.904 (PRISMA), and **-0.623
(Sentinel-2)** — negative R² meaning Sentinel-2's 10 bands perform worse than predicting the
mean. MESMA char-fraction fidelity follows the same pattern: R²=0.991 (EMIT), 0.957 (PRISMA),
**0.361 (Sentinel-2)** — an absolute R² gap of ~0.63 between EMIT-equivalent hyperspectral
resolution and Sentinel-2 broadband, i.e. Sentinel-2's 10 bands leave roughly two-thirds of the
native char-fraction variance unexplained. (A ratio of the two R² values would not be a
meaningful "fidelity multiple" — R² is a variance-explained fraction, not a linear scale — so we
report the gap in R², not a ratio.) The R² gap between broadband-index and absorption-feature
performance (spectral-degradation cost) is 0.034 (EMIT), 0.091 (PRISMA), 1.619 (Sentinel-2).

| Product | Tanager (native) | EMIT (285 bd) | PRISMA (239 bd) | Sentinel-2 (10 bd) |
|---|---|---|---|---|
| NBR/NDVI (mean R² vs. native) | 1.00 | 0.999 | 0.995 | 0.996 |
| Absorption depth (mean R² vs. native) | 1.00 | 0.965 | 0.904 | -0.623 |
| MESMA char fraction (R² vs. native) | 1.00 | 0.991 | 0.957 | 0.361 |

## 4. Discussion

**Significance.** The sensor comparison gives direct quantitative evidence for
Tanager-vs-EMIT/PRISMA differentiation: broadband fire indices are commodity-grade across
sensors, but sub-pixel material unmixing (char fraction, absorption depth) degrades sharply
below hyperspectral resolution — exactly what Tanager's 426 bands are built to preserve. The
magnitude (R² 0.99→0.36 for char fraction) is consistent with published PRISMA-vs-Sentinel-2
MESMA gaps (Quintano et al. 2023: R²=0.64–0.79 hyperspectral vs. 0.27–0.53 broadband against
field CBI). On the water-content side, Tanager resolves the individual 970/1200/2100 nm
absorption features that a 10-band sensor blurs together — the physical prerequisite for
hyperspectral LFMC retrieval — even though this study stops at relative indices rather than a
calibrated percentage for want of colocated field data.

**Limitations.** (1) **BAER reference is provisional and of unverifiable provenance.** The
Hughes `hughes_sbs.tif` ships with no colormap or metadata; its class coding was inferred from
spatial structure and per-class NBR/NDVI, and its per-class proportions are near-inverted
relative to the published USDA BAER summary, so we cannot confirm it is the final BAER product.
BAER code 4 ("High") is retained as its own class despite a median NBR brighter than code 3 — an
ordering anomaly we document rather than merge away. The reported kappa is therefore provisional
and must be recomputed under the 5-class coding. (2) **Cross-fire transfer.** The RF classifier
is trained on Hughes and applied to Palisades — different fuel/terrain — and Hughes has no
pre-fire scene, so its NBR-threshold baseline uses single-date NBR, strictly harder than dNBR;
these numbers are lower bounds. (3) **Temporal mismatch.** The Hughes swath is ~1 day
post-ignition with the fire still burning, while BAER SBS is mapped after containment, so the
scene may not register the final burn state the reference encodes. (4) **No calibrated LFMC.** No
Globe-LFMC 2.0 field site colocates with the scene in useful numbers (area ratio ≈ 1 site/swath;
SoCal record ends 2023-01-30 vs an April-2025 scene), so a field-calibrated regression cannot be
trained honestly; the notebook ships relative water indices and an explicit data-availability
assessment instead of a moisture percentage. The earlier synthetic PLSR (CV R² measuring an
in-notebook injection formula) and its implausible LFMC map have been removed. None of (1)–(4)
affects the sensor-comparison results, which validate against native Tanager, not field data.
(5) **Coverage.** Only 4 of 11 cataloged scenes were downloaded, limiting the trajectory to two
overlap pairs; the pre-fire scene does not overlap the Apr footprint, forcing the post-Jan
recovery segment onto a secondary ("Hughes" footprint) pair.

**Future work.** Retrieve LFMC via physics-based PROSPECT/PROSAIL inversion, which recovers leaf
equivalent water thickness from reflectance with no scene-specific field training and sidesteps
the colocation problem entirely; source a verified BAER/MTBS product (or field CBI plots) with
recoverable provenance to replace the provisional Hughes reference and enable same-fire Palisades
scoring; and, as Tanager revisits and contemporaneous Globe-LFMC observations accumulate,
calibrate the retained `train_lfmc_plsr`/`predict_lfmc` functions against genuinely colocated
field data. Download the remaining cataloged scenes to complete the recovery series on one
consistent footprint.

## References

- Roberts, D. A., et al. (1998). Mapping chaparral using multiple endmember spectral mixture
  models. *Remote Sensing of Environment*, 65(3), 267–279.
- Quintano, C., et al. (2023). Fire severity mapping from PRISMA hyperspectral imagery via
  MESMA. *Remote Sensing of Environment*, 293, 113670. DOI: 10.1016/j.rse.2023.113670.
- Roberts, D. A., et al. (2006). Evaluation of live fuel moisture via imaging spectrometry.
  *J. Geophys. Res.* DOI: 10.1029/2005JG000113.
- Peterson, S. H., & Roberts, D. A. (2014). PLSR for live fuel moisture estimation. R²=0.94
  (needles), 0.91 (sagebrush).
- Danson, F. M., & Bowyer, P. (2004). Estimating live fuel moisture from reflectance.
  *Remote Sensing of Environment*, 92(3), 309–321.
- Key, C. H., & Benson, N. C. (2006). Composite Burn Index and Normalized Burn Ratio. FIREMON,
  USDA Forest Service.
- Veraverbeke, S., et al. (2018). Hyperspectral remote sensing of fire: review.
  *Remote Sensing of Environment*, 216, 105–121. DOI: 10.1016/j.rse.2018.06.020.
- Ward-Baranyay, Coleman et al. (2026). AVIRIS-3 Rapid Response to January 2025 Los Angeles
  Wildfires. *Geophysical Research Letters*. DOI: 10.1029/2025GL118756.
- Yebra, M., et al. (2024). Globe-LFMC 2.0: training/validation dataset for LFMC. *Scientific
  Data*. DOI: 10.1038/s41597-024-03159-6.
