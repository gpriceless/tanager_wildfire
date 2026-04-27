# Tanager Open Data Competition: Deep Research & Project Ideas

**Date**: 2026-04-27
**Author**: Tobler (Geospatial Data Scientist)
**Issue**: LGT-288 (parent: LGT-287 Training / Fine-Tuning)
**Supersedes**: `tanager-competition-analysis.md` (initial quick analysis)

---

## Executive Summary

The Planet Tanager Open Data Competition (deadline August 31, 2026) is an opportunity to build research-grade, open-source tools that serve the broader geospatial community — particularly the OGC ecosystem, which has significant gaps in hyperspectral data handling. Rather than forcing detr_geo's object detection into Tanager's 30m GSD (a poor fit), the strongest strategy is to build genuinely useful hyperspectral analysis tools and workflows that could eventually bridge into the detr_geo ecosystem.

This report presents **6 research project ideas** ranked by OGC community value, novelty, and feasibility. The top 3 are all viable as standalone competition entries in the Code & Scripts track. Additionally, this report covers **competition alternatives** where detr_geo's core capabilities are a natural fit (RarePlanes, DOTA v2).

**Key insight**: The biggest gap in the hyperspectral ecosystem isn't algorithms — it's interoperability. The STAC Hyperspectral Imagery extension has only 2 fields and is stuck at Proposal stage. GeoZarr V1 RC is due May 2026 but lacks hyperspectral test data. Python tooling is fragmented across 7+ libraries with no integration layer. A competition entry that addresses these structural gaps will have more lasting impact than any single analysis notebook.

---

## Part 1: The Opportunity Landscape

### Why Tanager Matters for OGC

Tanager-1 is one of five operational spaceborne imaging spectrometers (alongside EMIT, PRISMA, EnMAP, HISUI), but unique in its commercial tasking capability and CC BY 4.0 open data program. The 150+ freely available scenes represent the largest freely accessible spaceborne hyperspectral dataset outside of EMIT.

Yet OGC standards have not kept pace with this data:

| Standard | Hyperspectral Readiness |
|----------|------------------------|
| STAC HSI Extension | Proposal stage, 2 fields only (`hsi:wavelength_min/max`) |
| STAC `eo:bands` | Designed for multispectral (~tens of bands), scales poorly to 426 |
| GeoZarr | V1 RC targeted May 2026, no hyperspectral test datasets yet |
| COG (Cloud Optimized GeoTIFF) | Poor fit for 426-band data (massive file sizes, slow band access) |
| OGC API Coverages | Draft standard, sparse hyperspectral implementations |
| CoverageJSON | "Strongly incompatible" with OGC/ISO coverage standards per Coverages WG |

### Published Research Gap

First-year Tanager publications are overwhelmingly methane/CO2 detection (5,392 plumes across 44 countries). Non-methane applications — mineral mapping, water quality, vegetation health, soil characterization — are conspicuously absent despite being well-established in airborne hyperspectral research. This is the gap the competition should target.

### Python Tooling Fragmentation

| Tool | Focus | Tanager Support | OGC Integration |
|------|-------|-----------------|-----------------|
| HyperCoast 0.22.0 | Visualization, STAC search | Direct (`read_tanager()`) | STAC only |
| ISOFIT 3.7.5 | Atmospheric correction | Used internally by Planet | None |
| Spectral Python 0.24 | Classification, PCA/MNF | Needs adapter (ENVI only) | None |
| PyHAT (USGS, 2025) | Spectral analysis, GUI | Needs adapter | None |
| HySUPP | Spectral unmixing (20+ methods) | Generic array input | None |
| spyndex 0.6.0 | 232 spectral indices | Any array data | None |
| ACOLITE 20260421 | Atmospheric correction | Via HyperCoast | None |

Only HyperCoast uses STAC for data discovery. None produce OGC-compliant outputs. None implement OGC API Processes. The gap is not in individual capabilities but in the **glue** between them.

---

## Part 2: Six Research Project Ideas

### Project 1: "TanagerFlow" — An OGC-Interoperable Hyperspectral Analysis Toolkit

**Track**: Code & Scripts
**Effort**: 6-8 weeks (competition-ready MVP in 4 weeks)
**OGC Community Value**: Very High
**Novelty**: High

**What it does**: A Python library/notebook collection that provides the missing integration layer between Tanager data and OGC standards. End-to-end workflow from STAC discovery to standards-compliant output.

**Technical approach**:
1. **STAC Discovery** — Search Tanager catalog (`planet.com/data/stac/browser/tanager-core-imagery/catalog.json`) using HyperCoast's earthaccess integration. Enrich item metadata with proper spectral descriptions beyond the 2-field HSI extension.
2. **Data Loading** — HyperCoast `read_tanager()` to xarray.Dataset. Handle both Basic Radiance and Ortho Surface Reflectance products.
3. **Spectral Analysis Modules** — Pluggable analysis backends:
   - Band selection presets (true color, SWIR composites, vegetation, mineral, water quality)
   - PCA/MNF dimensionality reduction via Spectral Python
   - Spectral index computation via spyndex (232 indices)
   - Spectral Angle Mapper (SAM) matching against USGS Spectral Library v7
   - Linear spectral unmixing for sub-pixel material fractions
4. **OGC-Compliant Output** — Export results as:
   - GeoPackage (vector results: classified pixels, material maps)
   - Cloud-Optimized GeoTIFF (raster results: index maps, composites)
   - GeoZarr (experimental: full datacube output following draft V1 spec)
   - STAC Item metadata for each derived product
5. **Interactive Visualization** — leafmap/HyperCoast for web-based exploration

**Why it matters for OGC**: This directly addresses the #1 gap — no tool bridges STAC discovery to OGC-compliant analysis output for hyperspectral data. By prototyping GeoZarr output aligned with the V1 RC (due May 2026), this positions the project at the bleeding edge of standards adoption. Contributing practical feedback to the STAC HSI extension based on real 426-band workflows would be highly visible.

**How it fits detr_geo**: The band selection and dimensionality reduction modules become the bridge between hyperspectral data and detection. A future `detr_geo.io.read_hyperspectral()` function could consume TanagerFlow's 3-band composites directly.

---

### Project 2: "SpectralMiner" — Mineral Exploration & Mining Monitoring from Space

**Track**: Code & Scripts or Technical Analysis
**Effort**: 4-6 weeks
**OGC Community Value**: High
**Novelty**: High (no Tanager mineral mapping publications exist)

**What it does**: An open-source mineral mapping workflow using Tanager's 426-band VSWIR data, validated against the USGS Spectral Library v7 and applied to real mining regions in the open data catalog.

**Technical approach**:
1. **Spectral Library Integration** — Ingest USGS Spectral Library v7 (thousands of mineral spectra, 0.2-200 micrometers) and ECOSTRESS/ASTER Spectral Library (2,300+ spectra). Resample to Tanager's band centers.
2. **Target Minerals** — Focus on the "EMIT-10" minerals validated by NASA's EMIT mission: calcite, chlorite, dolomite, goethite, gypsum, hematite, illite+muscovite, kaolinite, montmorillonite, vermiculite. These have diagnostic absorption features in the VSWIR range:
   - Iron oxides: 480nm, 670nm, 870nm (electronic transitions)
   - Clays: 1400nm, 1900nm, 2200nm (Al-OH vibrational)
   - Carbonates: 2300-2350nm (CO3 fundamentals)
   - Sulfates: 1480nm, 1760nm, 2170nm (OH and SO4)
3. **Detection Algorithms** — Implement and compare:
   - SAM (Spectral Angle Mapper): pixel-to-reference angle in n-dimensional space
   - MTMF (Mixture-Tuned Matched Filtering): sub-pixel abundance estimation
   - ACE (Adaptive Cosine Estimator): statistical target detection
   - MNF + k-means classification pipeline
4. **Validation** — Apply to Tanager scenes over known mining regions (South Africa, Mexico, Turkmenistan scenes are in the open catalog). Cross-validate against EMIT L2B mineral products where scene overlap exists.
5. **Output** — Mineral abundance maps as GeoTIFF + GeoPackage with per-pixel confidence scores and mineral identification.

**Why it matters**: 30m GSD is more than adequate for mining — operations span hectares to square kilometers. Acid mine drainage monitoring (tracking jarosite/schwertmannite/ferrihydrite mineral progression) has direct environmental value. Critical mineral exploration is a major government priority (USGS is actively expanding spectral library coverage for rare earth elements). Zero Tanager mineral mapping publications exist — this would be genuinely first.

**Real-world use cases**:
- Exploration: hydrothermal alteration halo mapping (clay-sericite-propylite zonation around porphyry copper deposits)
- Environmental: acid mine drainage detection and severity progression
- Monitoring: tailings storage facility mineralogical weathering over time
- Critical minerals: rare earth element surface expression detection

---

### Project 3: "HyperWater" — Hyperspectral Water Quality Monitoring Pipeline

**Track**: Code & Scripts or Lightning Case Study
**Effort**: 4-5 weeks
**OGC Community Value**: Very High (direct public health impact)
**Novelty**: High (no Tanager water quality publications exist)

**What it does**: An end-to-end pipeline for retrieving water quality parameters from Tanager hyperspectral imagery, targeting harmful algal bloom (HAB) detection, chlorophyll-a estimation, and turbidity mapping for large inland water bodies and coastal zones.

**Technical approach**:
1. **Atmospheric Correction** — ACOLITE integration (via HyperCoast) for aquatic-optimized atmospheric correction. ACOLITE doesn't require MODTRAN and handles adjacency effects critical for water pixels.
2. **Water Quality Parameters**:
   - Chlorophyll-a: NDCI, MCI, Three-Band Algorithm, FLH (Fluorescence Line Height at 681nm)
   - Cyanobacteria (HABs): phycocyanin index at 620nm absorption
   - Turbidity/TSM: 700-900nm scattering signal
   - CDOM: 400-450nm absorption ratios
   - Red-Edge Peak Width Index (REPWI) — novel 2025 index, overcomes limitations in optically complex waters
3. **Tanager's Advantage** — 426 bands at ~5nm spacing vs. PRISMA (239 bands, ~12nm) enables finer phytoplankton functional type discrimination. The narrow bands around 681nm enable true fluorescence line height retrieval that multispectral sensors cannot do.
4. **Target Scenes** — Identify coastal and large water body scenes in the open catalog. Cross-validate against Sentinel-2 and in-situ water quality data where available.
5. **Output** — Water quality parameter maps with uncertainty estimation, GeoPackage vector output for threshold exceedances (e.g., "HAB risk zones"), time-series dashboard template.

**Why it matters**: HAB monitoring is a direct public health application — cyanobacteria produce toxins that contaminate drinking water. EPA and state agencies monitor HABs using Sentinel-2/Landsat (6-13 bands), but hyperspectral data enables far more specific phytoplankton identification. An open-source pipeline tailored to Tanager's 426 bands would serve water management agencies globally.

**30m adequacy**: Strong for large lakes, reservoirs, and coastal zones. Validated by PRISMA studies (30m, 239 bands) achieving 70-95% Chl-a estimation accuracy in turbid lakes.

---

### Project 4: "CarbonSpec" — Soil Organic Carbon Mapping from Spaceborne Hyperspectral

**Track**: Technical Analysis
**Effort**: 5-7 weeks
**OGC Community Value**: Very High (climate science priority)
**Novelty**: Very High (nascent field, Tanager's dense SWIR is uniquely suited)

**What it does**: Demonstrates spaceborne soil organic carbon (SOC) estimation using Tanager's dense SWIR sampling, targeting the climate science community's need for global soil carbon inventories.

**Technical approach**:
1. **Diagnostic Bands** — SOC has well-characterized absorption features that Tanager's ~5nm SWIR sampling captures precisely:
   - 400-570nm: organic matter broad absorption
   - 1434-1476nm: O-H overtone
   - 1819-2001nm: C-H stretch and C=O combinations
   - 2198-2206nm: Al-OH (clay-organic complexes)
   - 2365-2373nm: C-O in carbonates vs. organic carbon
   - 2481-2500nm: additional organic features
2. **Continuum Removal Analysis** — Apply continuum-removal spectral indices (D2200, D1900/D2200, A2200) that have shown strong SOC correlation in ground-based studies.
3. **ML Regression** — Train SOC estimation models using:
   - Spectral features from bare-soil Tanager pixels
   - Validation against soil survey databases (ISRIC World Soil Information, USDA SSURGO)
   - Comparison with Sentinel-2-derived SOC estimates (13 bands) to quantify the value-add of 426 bands
4. **Multi-Sensor Fusion** — Explore combining Tanager SWIR with Sentinel-1 SAR (soil moisture/roughness correction) to reduce coupling effects on SOC estimation.
5. **Target Scenes** — Agricultural regions with available soil survey data. The open catalog includes scenes over agricultural areas in Brazil, Mexico, and other regions.

**Why it matters**: Soil organic carbon is the largest terrestrial carbon pool (~1,500 Gt in the top 1m). Accurate SOC mapping is critical for carbon credit verification, agricultural policy, and IPCC reporting. Current satellite SOC estimates rely on Sentinel-2's 13 broad bands — Tanager's 426 narrow bands in the diagnostic SWIR region provide the most spectrally detailed spaceborne SOC characterization possible today. A 2025 study demonstrated the approach but used spaceborne composites, not targeted hyperspectral. This would advance the field significantly.

**30m adequacy**: Excellent. Standard soil maps operate at much coarser resolution. 30m captures field-to-landscape scale SOC variation, which is the resolution needed for carbon accounting.

---

### Project 5: "FireSpec" — Live Fuel Moisture & Post-Fire Assessment from Hyperspectral

**Track**: Lightning Case Study or Technical Analysis
**Effort**: 3-5 weeks
**OGC Community Value**: High (wildfire agency relevance)
**Novelty**: High (LFMC from spaceborne VSWIR at 5nm is under-published)

**What it does**: Two-part analysis: (1) pre-fire live fuel moisture content (LFMC) estimation from Tanager's SWIR water absorption features, and (2) post-fire burn severity characterization using hyperspectral mineral analysis of heated soils.

**Technical approach**:
1. **LFMC Estimation** — Water has absorption features at 970nm, 1200nm, 1700nm, and 2100nm. Tanager's ~5nm spectral resolution enables precise measurement of absorption depth and shape, which correlate with vegetation water content. AVIRIS has validated this approach from aircraft; Tanager brings it to satellite scale with on-demand tasking.
2. **Burn Severity Mapping** — Beyond standard NBR (NIR - SWIR2)/(NIR + SWIR2):
   - Characterize post-fire mineral assemblages (heated soils produce distinctive iron oxide and clay mineral transformations)
   - Map ash/char distribution using broadband + SWIR features
   - Detect soil hydrophobicity indicators from organic compound volatilization signatures
3. **Comparison** — Quantify the information gain of 426-band hyperspectral burn severity mapping vs. standard Landsat/Sentinel-2 broadband indices. AVIRIS research shows hyperspectral band ratios are more sensitive to fire effects than traditional NBR.
4. **Target Scenes** — Identify Tanager scenes over recent fire scars. The open catalog may include relevant scenes given the global coverage.

**Why it matters**: Wildfire is the most media-visible and politically urgent environmental issue in the western US, Australia, and Mediterranean Europe. LFMC estimation is the holy grail of fire behavior prediction — knowing how dry vegetation is determines fire spread rate and intensity. Current operational LFMC maps use Landsat/MODIS with coarse spectral resolution. A spaceborne VSWIR LFMC product at 5nm spectral resolution would be directly useful to USFS, CAL FIRE, and equivalents worldwide.

**30m adequacy**: For landscape-scale burn severity: yes (standard dNBR is produced at 30m from Landsat). For LFMC at management scales: adequate for stand-level variation.

---

### Project 6: "CoastalSpec" — Hyperspectral Coastal Ecosystem Health Assessment

**Track**: Lightning Case Study
**Effort**: 3-4 weeks
**OGC Community Value**: Moderate-High
**Novelty**: Moderate (mangrove/coral work exists with PRISMA, but not Tanager)

**What it does**: Demonstrate Tanager's utility for coastal ecosystem monitoring — mangrove health assessment, submerged aquatic vegetation mapping, and coastal water quality in a single integrated notebook.

**Technical approach**:
1. **Mangrove Health** — Use Enhanced Mangrove Vegetation Index (EMVI) and red-edge position analysis for species-level discrimination within mangrove forests. Tanager's narrow bands enable finer species discrimination than Sentinel-2.
2. **Submerged Aquatic Vegetation (SAV)** — Water-penetrating visible bands (400-600nm) for benthic habitat mapping in shallow, clear coastal waters.
3. **Integrated Water-Land Analysis** — Combine terrestrial vegetation indices with aquatic water quality parameters in the coastal transition zone — an analysis that requires both VNIR vegetation features and SWIR water/soil features in the same scene.
4. **Target Scenes** — Coastal scenes from the open catalog (likely available given geographic diversity of 25+ countries).

**Why it matters**: Coastal ecosystems (mangroves, seagrass, coral reefs) are among the most productive and threatened on Earth. OGC has multiple coastal observation initiatives. An integrated land-water analysis notebook showcases Tanager's unique ability to characterize both domains simultaneously — something narrow multispectral sensors cannot do as effectively.

**30m adequacy**: Mixed. Mangrove stands and large reef systems: adequate. Individual reef features: too coarse. Best scoped as stand/zone-level analysis.

---

## Part 3: Project Ranking & Recommended Strategy

### Ranking by Competition Fit

| Rank | Project | Track | OGC Value | Novelty | Effort | Risk |
|------|---------|-------|-----------|---------|--------|------|
| 1 | TanagerFlow (toolkit) | Code & Scripts | Very High | High | 6-8 wk | Low |
| 2 | SpectralMiner (minerals) | Code & Scripts | High | High | 4-6 wk | Low |
| 3 | HyperWater (water quality) | Code & Scripts | Very High | High | 4-5 wk | Medium |
| 4 | CarbonSpec (soil carbon) | Technical Analysis | Very High | Very High | 5-7 wk | Medium |
| 5 | FireSpec (wildfire/LFMC) | Lightning/Technical | High | High | 3-5 wk | Medium |
| 6 | CoastalSpec (coastal) | Lightning | Moderate-High | Moderate | 3-4 wk | High |

### Recommended Competition Strategy

**Option A: Single Deep Submission (Recommended)**

Submit **TanagerFlow + SpectralMiner as a combined entry** in the Code & Scripts track. TanagerFlow provides the standards-compliant infrastructure; SpectralMiner provides the compelling application. Together, they tell a story: "Here's a reusable toolkit for OGC-interoperable hyperspectral analysis, demonstrated through mineral mapping of mining regions."

- Timeline: May-July development, August polish and submission
- Deliverable: GitHub repo + Jupyter notebook + mineral maps of 3-5 mining regions
- Community value: Reusable toolkit + novel Tanager application + STAC HSI extension feedback

**Option B: Two Submissions Across Tracks**

Submit **TanagerFlow** (Code & Scripts) + **CarbonSpec** (Technical Analysis). The two entries address different tracks and different audiences (developers vs. scientists), maximizing coverage.

**Option C: Maximum Ambition**

Submit TanagerFlow (Code & Scripts) + SpectralMiner + HyperWater (as Lightning Case Studies). Three entries, but the risk of spreading too thin is real with a 4-month timeline.

---

## Part 4: Better-Fit Competitions for detr_geo Core Capabilities

While the Tanager projects above serve the broader geospatial community, these competitions play directly to detr_geo's RGB object detection strengths:

### RarePlanes Fine-Tuning (HIGHEST PRIORITY)

| Field | Detail |
|-------|--------|
| Dataset | 253 Maxar WorldView-3 scenes, 0.3m GSD, 14,700 aircraft |
| License | **CC BY-SA 4.0** — can distribute trained weights |
| Format | Pre-tiled 512x512 PNGs + COCO JSON annotations (direct ingest) |
| Access | `aws s3 cp --no-sign-request s3://rareplanes-public/` |
| Baselines | Faster R-CNN: 73.3 mAP, Mask R-CNN: 73.7 mAP (COCO AP) |
| DETR baselines | **None exist** — detr_geo would be the first |
| Effort | 1-2 weeks (pipeline proven on xView) |

**Why #1**: The only major satellite detection dataset where detr_geo can legally publish trained weights. No DETR benchmark exists. The data is already in COCO format at the right tile size. RF-DETR Medium should be competitive with or exceed Faster R-CNN's 73.3 mAP given the architectural advantages of DETR for dense detection. Deliverable: `detr-geo-rareplanes` weights on HuggingFace.

### DOTA v2 HBB Leaderboard (HIGH PRIORITY)

| Field | Detail |
|-------|--------|
| Dataset | 11,268 images (up to 20K x 20K px), 1.79M instances, 18 classes |
| License | Academic/research only — **cannot distribute weights** |
| Evaluation | Active server at bed4rs.net (~2 day turnaround) |
| SOTA (OBB) | LSKNet: 81.85% mAP; ARS-DETR: 75.84% mAP |
| DETR on HBB | ~72% mAP (13% above CNN baselines in some evaluations) |
| Effort | 3-4 weeks (new data loader + training) |

**Why important**: Most-cited aerial detection benchmark. A competitive result is the single strongest credibility signal for the library in the research community. detr_geo's tiling + cross-tile NMS handles DOTA's enormous images natively — this is exactly the capability gap other teams struggle with.

### SpaceNet 2 Buildings (MODERATE PRIORITY, DEFER)

CC BY-SA 4.0, 302K buildings across 5 cities. No active evaluation server. Completes the "vehicles + aircraft + buildings" weight portfolio. Defer to Phase 3 — RarePlanes and DOTA deliver higher ROI first.

### Emerging: KFGOD (MONITOR)

880K instances, 33 fine-grained classes from KOMPSAT imagery at 55-70cm GSD. Published November 2025 with SOTA mAP of 63.9%. If data access opens under a permissive license, this would be an excellent showcase for detr_geo's capabilities across many categories. Monitor for availability.

---

## Part 5: Recommended Action Plan

### Phase 1: Now - June 2026 (Immediate)
- **RarePlanes fine-tuning** — ship `detr-geo-rareplanes` weights to HuggingFace
- **Begin TanagerFlow development** — STAC discovery, HyperCoast loading, band selection presets

### Phase 2: June - July 2026
- **DOTA v2 HBB submission** — build data loader, train, submit to evaluation server
- **TanagerFlow + SpectralMiner** — mineral mapping workflow, USGS library integration, GeoZarr output prototype

### Phase 3: July - August 2026
- **Tanager competition submission** — polish TanagerFlow + SpectralMiner, write narrative, submit by August 31
- **Optional**: HyperWater or CarbonSpec as additional entry if time permits

### Phase 4: September+ 2026
- SpaceNet 2 building detection (complete weight portfolio)
- Evaluate KFGOD if data becomes accessible

---

## Appendix A: Tanager-1 Data Access Reference

| Resource | URL/Path |
|----------|----------|
| STAC Catalog | `planet.com/data/stac/browser/tanager-core-imagery/catalog.json` |
| Open Data Program | 150+ scenes, CC BY 4.0 |
| Google Earth Engine | `projects/sat-io/open-datasets/PLANET/TANAGER_HYPERSPECTRAL` |
| Products | Basic Radiance (HDF5), Ortho Surface Reflectance (ISOFIT v2.9.5) |
| Spectral Range | 380-2500nm, ~426 bands, ~5nm spacing |
| Spatial Resolution | 30m GSD |
| Swath | 18-481km (sensitivity mode dependent) |
| Competition Registration | `learn.planet.com/2026-Tanager-Open-Data-Competition.html` |
| Carbon Mapper Data API | `carbonmapper.org/data` (plumes, 30-day latency) |

## Appendix B: Key Spectral Libraries

| Library | Spectra | Coverage | URL |
|---------|---------|----------|-----|
| USGS Spectral Library v7 | Thousands | 0.2-200 um (minerals, rocks, soils, vegetation, man-made) | `pubs.usgs.gov/ds/1035/` |
| ECOSTRESS/ASTER | 2,300+ | 0.4-15.4 um (160 JPL mineral samples in 3 particle sizes) | `speclib.jpl.nasa.gov` |
| GHISA (USGS) | Crop-focused | VSWIR | Global agricultural crop spectra |

## Appendix C: Dataset Licensing Summary

| Dataset | License | Distribute Weights? | Access |
|---------|---------|---------------------|--------|
| Tanager Open Data | CC BY 4.0 | Yes (attribution) | STAC catalog, no auth |
| RarePlanes | CC BY-SA 4.0 | **Yes** (share-alike) | AWS S3, no auth |
| SpaceNet 1-9 | CC BY-SA 4.0 | **Yes** (share-alike) | AWS S3, no auth |
| DOTA v1/v2 | Academic | **No** | Application required |
| xView | CC BY-NC-SA 4.0 | Risky (NC clause) | Application required |
| FAIR1M | Unknown | Unknown | Unreliable access |
| KFGOD | Unknown | Unknown | Not confirmed |

## Appendix D: OGC Standards Timeline (Competition-Relevant)

| Standard | Status | Competition Relevance |
|----------|--------|-----------------------|
| STAC Core | Published Oct 2025 (OGC Community Standard) | Discovery layer for all Tanager data |
| STAC HSI Extension | Proposal, 2 fields | Gap to address — contribute improvements |
| GeoZarr | V1 RC target May 2026 | Prototype output format — bleeding edge |
| OGC API Coverages | Draft | Future datacube access standard |
| SensorML 2.1 | Published 2024 | Sensor description (can model Tanager spectrometer) |

---

## Sources

### Tanager & Planet
- Planet: Announcing the Tanager Open Data Competition (planet.com/pulse/announcing-the-tanager-open-data-competition/)
- Planet: Competition Registration (learn.planet.com/2026-Tanager-Open-Data-Competition.html)
- Planet: Tanager Documentation (docs.planet.com/data/imagery/tanager/)
- Carbon Mapper: Product Guide (carbonmapper.org/articles/product-guide)
- Carbon Mapper: Tanager-1 One Year in Space (carbonmapper.org/articles/tanager-1-one-year-in-space)
- GEE Community Catalog: Tanager (gee-community-catalog.org/projects/tanager/)

### OGC Standards
- OGC GeoZarr SWG Formation (ogc.org/announcement/ogc-forms-new-geozarr-standards-working-group/)
- STAC HSI Extension (github.com/stac-extensions/hsi)
- STAC Bands RFC (github.com/radiantearth/stac-spec/discussions/1213)
- STAC Community Standard Publication (spatialists.ch/posts/2025/10/28-publication-of-the-spatiotemporal-asset-catalog-community-standards/)
- GeoZarr Spec (github.com/zarr-developers/geozarr-spec)

### Hyperspectral Science
- USGS Spectral Library v7 (pubs.usgs.gov/ds/1035/)
- ECOSTRESS/ASTER Spectral Library (speclib.jpl.nasa.gov)
- EMIT L2B Mineral Detection ATBD (lpdaac.usgs.gov/documents/1659/EMITL2B_ATBD_v1.pdf)
- EnMAP Mineral Exploration 2025 (sciencedirect.com/science/article/pii/S016913682500472X)
- PRISMA Water Quality in Turbid Lakes (mdpi.com/2072-4292/12/23/3984)
- Novel REPWI Chl-a Index 2025 (sciencedirect.com/science/article/abs/pii/S0034425725002512)
- SOC from Spaceborne Hyperspectral 2025 (sciencedirect.com/science/article/pii/S1569843225001517)
- Post-Fire AVIRIS vs Multispectral (sciencedirect.com/science/article/abs/pii/S0034425714003162)

### Competitions & Datasets
- RarePlanes on AWS (registry.opendata.aws/rareplanes/)
- RarePlanes Paper (arxiv.org/abs/2006.02963)
- DOTA Official (captain-whu.github.io/DOTA/)
- SpaceNet Challenges (spacenet.ai/challenges/)
- KFGOD Dataset (mdpi.com/2072-4292/17/22/3774)

### Python Tools
- HyperCoast (github.com/opengeos/HyperCoast)
- Spectral Python (spectralpython.net)
- ISOFIT (github.com/isofit/isofit)
- spyndex (github.com/awesome-spectral-indices/spyndex)
- PyHAT (usgs.gov/centers/astrogeology-science-center/science/python-hyperspectral-analysis-tool-pyhat)
- HySUPP (github.com/inria-thoth/HySUPP)
- ACOLITE (github.com/acolite/acolite)
