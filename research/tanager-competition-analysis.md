# Tanager Competition & Alternative Competitions Analysis

**Date**: 2026-04-27 (updated with deep-dive)
**Author**: Tobler (Geospatial Data Scientist)
**Issue**: LGT-288 (parent: LGT-287 Training / Fine-Tuning)

---

## Executive Summary

The Planet Tanager Open Data Competition (deadline August 31, 2026) is a genuine opportunity to build open-source tools that serve the broader OGC and geospatial community — not just a quick notebook exercise. Tanager-1's 426-band VSWIR imagery (380-2500nm, 30m GSD) enables material-level discrimination that RGB cannot touch: mineral identification, soil carbon mapping, burn severity analysis, coral reef health assessment, crop phenology tracking, and cross-sensor calibration.

The key insight from deep research: **the tooling gap is at the application layer, not the data layer.** HyperCoast solves data I/O. SPy solves individual algorithms. But nobody has built domain-specific, pip-installable pipelines that combine data loading + preprocessing + validated algorithms + OGC-compliant outputs. This is where novel, community-valuable contributions live.

This report presents **6 research-grade submission ideas** ranked by community impact, scientific novelty, and feasibility with available open data (150+ scenes across 9 collections).

---

## Part 1: Competition & Satellite Overview

### Competition Details

| Field | Value |
|-------|-------|
| Organizer | Planet Labs PBC / Carbon Mapper |
| Launch | April 14, 2026 |
| Deadline | August 31, 2026 |
| Winners announced | November 2, 2026 |
| Prize | 3 winners each select up to 10 Tanager images for public release (30 total) + AGU 2026 recognition |
| Tracks | Lightning Case Studies, Code & Scripts, Technical Analysis |

### Tanager-1 Specifications

| Parameter | Value |
|-----------|-------|
| Bands | ~426 (VSWIR) |
| Spectral range | 380-2500 nm |
| Spectral resolution | ~5 nm spacing |
| Spatial resolution | 30m GSD |
| Swath | 18 km |
| Data format | HDF5 (radiance + surface reflectance) |
| Open data | 150+ scenes, CC BY 4.0, via STAC catalog |
| Atmospheric correction | ISOFIT v2.9.5 (by Planet) |
| SNR | 300-600 (varies by collection mode) |
| Detection | Designed by NASA JPL, launched August 16, 2024 |

### Available Open Data (9 Collections)

| Collection | Scenes | Key Geographies |
|------------|--------|-----------------|
| Agriculture | ~43 | Germany (10), Kenya (8), California, India, Netherlands, Argentina, Brazil, Indonesia, Australia |
| Natural Lands | ~77 | Global (largest collection) |
| Urban | ~60 | Los Angeles, Buenos Aires, Netherlands, Germany, Japan, Middle East |
| Fire | 12 | Southern California (LA fires, Dec 2024-Jul 2025), Northern Arizona |
| Energy & Mining | Present | Details TBD |
| Coastal & Water | Present | San Francisco Bay featured |
| Snow & Ice | Present | High-latitude scenes |
| GHG Plumes | Present | Turkmenistan, Texas Permian Basin, Algeria, South Africa |
| ROCX 2025 | Present | Calibration/validation campaign |

### Published Research Is Almost Entirely Methane

- 5,392 methane plumes detected across 44 countries in year one
- 1,234 CO2 plumes across 763 sources
- Only ONE non-methane peer-reviewed paper: coral reef mapping in Hawaii (MDPI Remote Sensing, Jan 2026)
- Zero published work on: mineral mapping, crop analysis, water quality, soil, or fire from Tanager data

**This is a wide-open field for differentiated submissions.**

---

## Part 2: OGC Standards Landscape for Hyperspectral

### Current Standards

| Layer | Standard | Status | Hyperspectral Relevance |
|-------|----------|--------|------------------------|
| Data encoding | GeoZarr | In development (SWG formed Jan 2024, ABR summer 2026) | **Explicitly targets hyperspectral** |
| Data encoding | CIS (Coverage Implementation Schema) | Adopted | Supports arbitrary spectral dimensions |
| Data access | WCS 2.0 + EO Profile | Adopted | Can serve hyperspectral coverages |
| Cloud processing | openEO | Community Standard candidate | Datacube model with native spectral band operations |
| Data catalog | STAC + EO extension | Community standard | Per-band wavelength metadata |
| Data catalog | STAC HSI extension | **Proposal (immature, 5 commits)** | wavelength_min/max summary only |
| Emissions | EmissionML/MethaneML | SWG formed March 2025 | Links satellite methane observations to sources |

### Key Gaps Where Contributions Would Be Valued

1. **STAC HSI extension is nearly empty** — only defines `hsi:wavelength_min` and `hsi:wavelength_max`. No spectral response curves, atmospheric correction provenance, or per-band noise characteristics. Contributing richer metadata support would create visibility in the STAC community.

2. **No cross-mission spectral calibration standard** — EMIT, EnMAP, PRISMA, Tanager, and upcoming CHIME/SBG all produce VSWIR data with different band configurations. No OGC standard addresses cross-mission spectral interoperability.

3. **GeoZarr Architecture Board review is summer 2026** — timing aligns perfectly with a competition submission that demonstrates hyperspectral data in Zarr format.

4. **OGC-OSGeo relationship is formal and deep** — MoU since 2008, joint code sprints, GDAL certified as OGC reference implementation. Open-source contributions to the geospatial stack are directly valued.

---

## Part 3: Six Research-Grade Submission Ideas

### Idea 1: "TanagerMinerals" — Open-Source Critical Mineral Mapper

**Track**: Code & Scripts
**OGC Value**: High — standardized mineral map outputs in GeoPackage/GeoJSON, potential STAC HSI extension contributions
**Novelty**: Very High — no pip-installable Python package does satellite mineral mapping

**Concept**: A Python library that reads Tanager/EMIT/PRISMA/EnMAP surface reflectance data, matches pixel spectra against the USGS Spectral Library v7 using Spectral Angle Mapper (SAM) and spectral feature fitting, and outputs georeferenced mineral abundance maps with OGC-compliant formats.

**Why this matters for OGC**: Critical mineral exploration from space is a hot policy topic — the US, EU, and Australia all have critical mineral strategies. USGS is actively using VSWIR for lithium clay detection. The existing tool (Tetracorder) is Fortran legacy code. PyHAT is USGS-internal and planetary-focused. No modern, community-maintained tool exists.

**Technical approach**:
1. Load Tanager surface reflectance via HyperCoast `read_tanager()`
2. Implement continuum removal for absorption feature normalization
3. Match against USGS Spectral Library v7 endmembers using SAM, SFF, and band ratios
4. Output mineral probability maps for key groups:
   - Iron oxides (hematite, goethite) — Fe3+ absorption at ~900nm
   - Clays (kaolinite, montmorillonite, illite) — Al-OH at ~2200nm
   - Carbonates (calcite, dolomite) — CO3 at ~2300nm
   - Lithium-bearing clays (hectorite) — features at 1900nm, 2200nm
   - REE-bearing minerals — Nd absorptions (validated by 2024 EnMAP study at Mountain Pass, CA)
5. Export as GeoPackage with per-pixel mineral class + confidence + matched library spectrum ID
6. Interactive visualization with leafmap

**Study areas from available data**: Energy & Mining collection scenes, arid Natural Lands scenes. If we win, request scenes over Atacama lithium triangle, Pilbara iron belt, or Mountain Pass REE deposit.

**Effort**: 8-10 weeks. Requires SAM/SFF implementation, USGS library integration, validation against published EnMAP mineral maps.

**Why it stands out**: Every other submission will be methane-focused or a simple spectral index notebook. A mineral mapping tool addresses a real capability gap and serves geological surveys, mining companies, and environmental regulators globally.

---

### Idea 2: "TanagerFire" — Hyperspectral Burn Severity Toolkit

**Track**: Lightning Case Study (LA fires) + Code & Scripts (reusable toolkit)
**OGC Value**: High — standardized burn severity output, temporal change detection patterns
**Novelty**: High — no open-source satellite hyperspectral fire toolkit exists

**Concept**: A pre/post-fire spectral unmixing toolkit that decomposes burned landscape pixels into sub-pixel abundances of ash, charcoal, scorched vegetation, green vegetation, bare soil, and new litter. Quantifiably outperforms Sentinel-2 dNBR (published evidence: 2x better R2).

**Why this matters**: The LA wildfire time series in the open data (7 scenes from Dec 2024 to Jul 2025) is a compelling, media-relevant story. Fire science is a massively funded field (USFS, CalFire, FEMA). Hyperspectral unmixing reveals material composition changes invisible to broadband indices, yet most fire science still uses Sentinel-2 dNBR despite proven superiority.

**Technical approach**:
1. Load pre-fire (Dec 15, 2024) and post-fire (Jan 23, 2025 + recovery series) scenes
2. Apply Multiple Endmember Spectral Mixture Analysis (MESMA) using fire-specific spectral libraries:
   - Green vegetation endmember (photosynthetically active)
   - Non-photosynthetic vegetation (dry grass, dead wood)
   - Ash/char endmember (white ash vs black char — distinct spectral signatures)
   - Bare soil endmember
3. Generate sub-pixel abundance fraction maps for each endmember at each time step
4. Compute change trajectories: vegetation loss → char/ash → recovery (new vegetation fraction)
5. Compare with Sentinel-2 dNBR over same area and time period
6. Demonstrate that hyperspectral unmixing captures burn severity gradations invisible to dNBR

**Study area**: LA wildfire time series (7 scenes, Dec 2024-Jul 2025). Northern Arizona fire scenes (Sep 2025) for validation.

**Effort**: 6-8 weeks. MESMA Python package exists (v1.0.8 on PyPI). Need to build fire-specific spectral libraries and wrap the pipeline.

**Why it stands out**: Visually dramatic (fire imagery), socially relevant (LA fires were major news), scientifically rigorous (2x improvement over multispectral is publishable), and practically useful for fire agencies worldwide. The time series showing vegetation recovery adds a narrative arc.

---

### Idea 3: "TanagerReef" — Coral and Coastal Habitat Classifier

**Track**: Lightning Case Study (specific coastal site) + Code & Scripts
**OGC Value**: Medium-High — blue carbon assessment, marine protected area monitoring
**Novelty**: Medium-High — extends validated Tanager coral research into reusable tooling

**Concept**: A classification module for benthic habitat mapping that takes Tanager coastal scenes and classifies bottom types: live coral, macroalgae, seagrass, sand, rubble. Builds on the only existing non-methane Tanager paper (Hawaii coral mapping, MDPI Remote Sensing Jan 2026).

**Why this matters**: A January 2026 paper demonstrated that Tanager-1 produced verifiable estimates of live coral cover on Hawaiian reefs, with spaceborne and airborne sensors producing "comparable geospatial distributions." Coral reef monitoring is critical for conservation (US Coral Reef Task Force, NOAA). Blue carbon assessment programs need standardized benthic habitat maps. No automated tool exists.

**Technical approach**:
1. Load Tanager coastal scenes via HyperCoast (already designed for coastal hyperspectral)
2. Implement water column correction to remove depth-dependent attenuation
3. Extract benthic spectral signatures using spectral libraries for coral, algae, seagrass, sand
4. Apply supervised classification (SAM + ML ensemble) to map bottom types
5. Validate against the published Hawaii coral paper results
6. Export habitat maps as GeoPackage with ICZM-compatible attributes

**Study areas**: Coastal & Water Bodies collection (San Francisco Bay featured). If we win, request scenes over Hawaiian reef systems, Florida Keys, or Great Barrier Reef.

**Effort**: 8-10 weeks. HyperCoast handles data loading. Water column correction algorithms are published but need implementation. Training data from the existing coral paper.

**Why it stands out**: Builds on proven Tanager results (lower risk). Blue carbon and coral conservation are high-visibility topics. Pacific Island nations and NOAA would be natural adopters. Integrates with the opengeos ecosystem.

---

### Idea 4: "TanagerSoil" — Soil Organic Carbon Mapping Pipeline

**Track**: Code & Scripts + Technical Analysis
**OGC Value**: Very High — carbon credit verification standards, soil monitoring for EU Soil Health Law
**Novelty**: High — no unified open-source satellite soil spectral analysis tool exists

**Concept**: A Python module that estimates soil organic carbon (SOC), clay mineral content, iron oxide content, and moisture from hyperspectral surface reflectance using validated spectral indices and machine learning models. Designed for the voluntary carbon market's need for satellite-verified SOC.

**Why this matters**: Carbon credit markets are a multi-billion dollar industry. SOC verification is the bottleneck — ground-truth sampling is expensive and sparse. Hyperspectral satellite estimation of SOC has been validated with R2 = 0.64-0.79 at 30m (EnMAP, 2025). Every research group that does this builds a custom pipeline from scratch. The EU Soil Health Law (2023) mandates soil monitoring across member states. Yet no standardized, open-source tool exists for satellite-based soil spectral analysis.

**Technical approach**:
1. Load Tanager surface reflectance over agricultural areas (bare soil periods)
2. Mask vegetated pixels using NDVI threshold
3. Extract soil spectral features: organic matter absorptions (450nm, 600nm, 2200nm), clay mineral OH (1400nm, 1900nm, 2200nm), iron oxide (480nm, 550nm, 900nm)
4. Apply validated ML models (PLSR, Random Forest, CNN) trained on ground-truth SOC databases
5. Produce georeferenced SOC concentration maps with uncertainty estimates
6. Compare with existing soil maps (SoilGrids, LUCAS) over the same areas

**Study areas**: Germany agriculture collection (10 scenes, Apr-Jun 2025 — some with bare soil exposure). Kenya agriculture (8 scenes, seasonal variation). Brazil Mato Grosso (soil exposure in deforested areas).

**Effort**: 10-12 weeks. Requires curating soil training data, building ML pipeline, validation. Higher effort but very high impact.

**Why it stands out**: Carbon markets are the most commercially relevant application of hyperspectral remote sensing. This tool would serve agricultural scientists, carbon credit verifiers, soil conservation agencies, and EU policy compliance — a massive potential user base.

---

### Idea 5: Cross-Sensor Spectral Validation Framework

**Track**: Technical Analysis
**OGC Value**: Very High — directly addresses the OGC gap in cross-mission spectral calibration standards
**Novelty**: Very High — no Tanager surface reflectance validation study has been published

**Concept**: A systematic comparison of Tanager-1 vs EMIT vs EnMAP surface reflectance products over the same geographic areas, computing spectral index agreement, band-to-band correlation, and identifying sensor-specific biases. The first independent validation of Tanager's surface reflectance quality.

**Why this matters**: CHIME (ESA, ~2028) and SBG (NASA, ~2028) will generate hyperspectral data at global scale. No OGC standard currently addresses cross-mission spectral interoperability. The GRSS-IEEE Standards Committee works on this but has no adopted standard. A framework that quantifies how well the same spectral indices reproduce across Tanager, EMIT, and EnMAP would be a foundational contribution — cited by every future cross-sensor study.

**Technical approach**:
1. Identify areas where Tanager and EMIT/EnMAP have overlapping coverage (same location, similar dates)
2. Load both datasets, resample to common grid
3. Compute identical spectral indices (NDVI, clay minerals, iron oxide, moisture) from both sensors
4. Quantify agreement: R2, RMSE, bias, spectral angle between full-spectrum signatures
5. Identify systematic differences (band misalignment, atmospheric correction artifacts, SNR effects)
6. Produce a cross-sensor compatibility matrix: "which indices transfer reliably between sensors?"
7. Publish the framework as a reusable tool for future sensor comparisons

**Study areas**: Find overlap between Tanager open data and EMIT coverage (EMIT covers vast areas from ISS). GHG Plumes collection likely has EMIT overlap (both target emission sources).

**Effort**: 6-8 weeks. Data access is straightforward (both are open data). Analysis is computationally light. The scientific value is in the methodology and results.

**Why it stands out**: The Technical Analysis track explicitly invites "comparing Tanager's performance to other sensors." This is the most academically valuable submission — it would be cited in every subsequent Tanager paper and directly informs the OGC cross-sensor standardization gap. First-mover advantage is strong.

---

### Idea 6: "TanagerCrops" — Multi-Temporal Crop Phenology Tracker

**Track**: Lightning Case Study (Germany or Kenya) + Code & Scripts
**OGC Value**: Medium-High — agricultural monitoring standards, food security
**Novelty**: Medium — crop phenology from hyperspectral is well-studied from airborne platforms, but sparse from satellite

**Concept**: A temporal analysis pipeline that tracks crop growth stages using narrow-band spectral indices derived from Tanager's multi-temporal agricultural coverage. Demonstrates capabilities beyond what Sentinel-2's 13 bands can offer for precision agriculture.

**Why this matters**: Tanager has unusually dense temporal coverage over Germany (10 scenes, Apr-Jun 2025) and Kenya (8 scenes, Apr-Sep spanning wet/dry seasons). Sentinel-2 provides 10-day revisit with 13 bands. Tanager provides ~monthly revisit with 426 bands. The question is: does 30x more spectral information compensate for lower temporal resolution? This has not been answered for satellite-based crop monitoring.

**Technical approach**:
1. Load multi-temporal Tanager scenes over Germany agriculture plots
2. Compute temporal profiles of narrow-band vegetation indices:
   - Red-edge chlorophyll indices (MCARI, TCARI) — 680-750nm region
   - Water stress indices — 970nm and 1200nm water absorption depth
   - Nitrogen content proxies — red-edge position derivative
   - Lignin/cellulose maturity — 2100-2300nm features
3. Map crop phenological stages (emergence, growth, heading, senescence) using spectral trajectory analysis
4. Compare with Sentinel-2 NDVI/EVI temporal profiles over same fields
5. Identify what crop health parameters are accessible only to hyperspectral (stress detection, disease early warning, species discrimination)

**Study areas**: Germany agriculture (10 scenes, best temporal coverage). Kenya agriculture (8 scenes, tropical crop diversity + food security angle).

**Effort**: 6-8 weeks. Straightforward spectral analysis. Multi-temporal alignment and visualization are the main challenges.

**Why it stands out**: Food security is a universal concern. The Germany time series enables a direct comparison with Copernicus Sentinel-2 over the same fields — this is a compelling "what can 426 bands tell us that 13 cannot?" story. The Kenya angle adds food security relevance for developing nations.

---

## Part 4: Recommendation Matrix

| Idea | Track | OGC Value | Novelty | Risk | Effort | Recommended? |
|------|-------|-----------|---------|------|--------|-------------|
| 1. TanagerMinerals | Code & Scripts | High | Very High | Medium | 8-10 wk | **YES — most differentiated** |
| 2. TanagerFire | Lightning + Code | High | High | Low | 6-8 wk | **YES — compelling story** |
| 3. TanagerReef | Lightning + Code | Medium-High | Medium-High | Medium | 8-10 wk | YES — builds on validated work |
| 4. TanagerSoil | Code + Technical | Very High | High | High | 10-12 wk | YES if resources allow |
| 5. Cross-Sensor Validation | Technical Analysis | Very High | Very High | Low | 6-8 wk | **YES — highest academic value** |
| 6. TanagerCrops | Lightning + Code | Medium-High | Medium | Low | 6-8 wk | YES — broadest appeal |

### If choosing just two submissions:

**Primary**: **Idea 5 (Cross-Sensor Validation)** for the Technical Analysis track — lowest risk, highest academic impact, directly addresses OGC standards gaps, first-mover advantage.

**Secondary**: **Idea 2 (TanagerFire)** for Lightning Case Study — the LA wildfire data is ready-made, the science is proven (2x better than Sentinel-2), and the story has natural media appeal.

### If choosing three submissions (one per track):

1. **Technical Analysis**: Idea 5 (Cross-Sensor Validation)
2. **Lightning Case Study**: Idea 2 (TanagerFire — LA wildfires)
3. **Code & Scripts**: Idea 1 (TanagerMinerals)

---

## Part 5: Natural-Fit Competitions for detr_geo (Retained from v1)

While the Tanager competition is the primary focus, these competitions remain relevant for detr_geo's object detection mission:

| Competition | Status | GSD | detr_geo Fit | Weights Distributable? | Priority |
|-------------|--------|-----|--------------|----------------------|----------|
| **RarePlanes** | Dataset (AWS) | 0.3m | **DIRECT** | **Yes (CC BY-SA 4.0)** | High |
| **DOTA v2** | Active leaderboard | Sub-meter | **DIRECT (HBB task)** | No (academic) | High |
| **SpaceNet 2** | Dataset | 0.3-0.5m | Needs polygon→bbox | **Yes (CC BY-SA 4.0)** | Medium |

These can proceed in parallel with Tanager competition work — they serve a different user base and use detr_geo's core detection pipeline, not hyperspectral analysis.

---

## Part 6: Hyperspectral Tools Ecosystem

### Key Libraries for Building Submissions

| Tool | Version | Key Capability | Role in Submissions |
|------|---------|----------------|-------------------|
| **HyperCoast** | 0.22.0 | `read_tanager()`, STAC search, visualization | Data loading (all ideas) |
| **Spectral Python (SPy)** | 0.24 | SAM, PCA, MNF, classification | Mineral mapping (Idea 1) |
| **MESMA** | 1.0.8 | Multiple Endmember Spectral Mixture Analysis | Fire unmixing (Idea 2) |
| **spyndex** | 0.6.0 | 232 spectral indices | Vegetation/crop indices (Ideas 4, 6) |
| **ACOLITE** | 20260421 | Atmospheric correction (aquatic) | Coastal analysis (Idea 3) |
| **leafmap** | Latest | Interactive web maps | Visualization (all ideas) |

### Spectral Libraries Available

| Library | Spectra | Coverage | Use Case |
|---------|---------|----------|----------|
| USGS Spectral Library v7 | 1000+ | 0.2-200um | Minerals, soils, vegetation |
| ASTER Spectral Library | 2000+ | VNIR-TIR | Minerals, rocks, coatings |
| RockSL (integrated) | Combined | VNIR-SWIR-TIR | Mineral identification |

---

## Part 7: OGC Contribution Opportunities

Beyond the competition, these are concrete ways to contribute to OGC standards:

1. **STAC HSI Extension** — Currently Proposal with 5 commits. Adding spectral response metadata, atmospheric correction provenance, and per-band quality flags would be directly useful for Tanager/EMIT/CHIME.

2. **GeoZarr hyperspectral examples** — The SWG is targeting Architecture Board review summer 2026. Demonstrating Tanager data in Zarr format would be timely.

3. **OGC-OSGeo Code Sprint** — Annual joint event. Demonstrating any of the above tools at a code sprint creates direct community connections.

4. **EmissionML integration** — If the methane detection angle is pursued, EmissionML provides the interoperability standard for encoding results.

---

## Sources

### Tanager Competition & Data
- Planet: Tanager Open Data Competition (planet.com/pulse/announcing-the-tanager-open-data-competition/)
- Planet: Competition Registration (learn.planet.com/2026-Tanager-Open-Data-Competition.html)
- Planet: Tanager Documentation (docs.planet.com/data/imagery/tanager/)
- Planet: Tanager STAC Catalog (planet.com/data/stac/tanager-core-imagery/catalog.json)
- Carbon Mapper: Tanager-1 One Year (carbonmapper.org/articles/tanager-1-one-year-in-space)
- AMT: Carbon Mapper Emissions System (amt.copernicus.org/articles/18/6933/2025/)
- GEE Community Catalog (gee-community-catalog.org/projects/tanager/)

### Application Science
- EnMAP REE Detection (Nature Scientific Reports, 2024) — doi:10.1038/s41598-024-71395-2
- PRISMA Fire Severity (Remote Sensing of Environment, 2023) — 2x R2 improvement over Sentinel-2
- Tanager Coral Mapping (MDPI Remote Sensing, Jan 2026) — doi:10.3390/rs18030435
- EnMAP SOC Mapping (MDPI Remote Sensing, 2025) — soil organic carbon at 30m
- Soil Heavy Metals Hyperspectral (Digital Earth, 2025) — doi:10.1080/17538947.2025.2520474
- HAB Hyperspectral Classification (MDPI Remote Sensing, 2025) — 90% algal bloom accuracy
- USGS Spectral Library (usgs.gov/labs/spectroscopy-lab/science/spectral-library)
- USGS Lithium Playa Mapping (usgs.gov/centers/gggsc)
- PyHAT User Guide (USGS Open-File Report 2025-1038)
- MESMA Python Package (mesma.readthedocs.io)

### OGC Standards
- OGC GeoZarr SWG (ogc.org, Jan 2024 announcement)
- STAC HSI Extension (github.com/stac-extensions/hsi)
- OGC EmissionML SWG (github.com/opengeospatial/EmissionML, March 2025)
- OGC-OSGeo MoU (ogc.org, since 2008)
- OGC openEO Community Standard CFP (OGC 24-059, 24-060)
- OGC CIS Standard (09-146r6)
- Joint OGC-OSGeo-ASF Code Sprint 2026 (USGS Fort Collins, CO)

### Hyperspectral Tools
- HyperCoast (github.com/opengeos/HyperCoast) — v0.22.0, 266 stars
- Spectral Python (github.com/spectralpython/spectral) — v0.24, 664 stars
- ISOFIT (github.com/isofit/isofit) — v3.7.5, 121 stars
- ACOLITE (github.com/acolite/acolite) — v20260421, 234 stars
- HyperGas 1.0 (egusphere.copernicus.org/preprints/2026/egusphere-2025-6127/)
- spyndex (github.com/awesome-spectral-indices/spyndex)
- HySUPP (github.com/inria-thoth/HySUPP) — 20+ unmixing algorithms

### Object Detection Competitions (detr_geo core)
- DOTA (captain-whu.github.io/DOTA/) — 11,268 images, 1.79M instances
- RarePlanes (registry.opendata.aws/rareplanes/) — CC BY-SA 4.0, 14.7K aircraft
- SpaceNet (spacenet.ai/challenges/) — full challenge history
