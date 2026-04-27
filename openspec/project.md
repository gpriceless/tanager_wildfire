# Tanager Competition — OpenSpec Project Definition

## Project Overview

**Tanager Competition** is a research and development project targeting the [Planet Tanager Open Data Competition](https://www.planet.com/pulse/announcing-the-tanager-open-data-competition/). The goal is to submit a robust, research-grade wildfire analysis solution leveraging Tanager-1's 426-band hyperspectral imagery.

This is a research-first project. The deliverable is a compelling, open-source submission to the competition — not a production SaaS product. Quality of research, scientific rigor, and OGC community value are the success metrics.

## Vision

Build a wildfire-focused hyperspectral analysis toolkit that:
- Processes Tanager-1 426-band imagery for wildfire-related spectral analysis
- Implements live fuel moisture content (LFMC) estimation from hyperspectral signatures
- Provides post-fire burn severity mapping using spectral unmixing (MESMA)
- Leverages the LA wildfire time series (7 scenes, Dec 2024 – Jul 2025) as a case study
- Produces OGC-interoperable outputs (STAC, GeoZarr, GeoPackage)
- Contributes to the broader open-source geospatial ecosystem

## Competition Context

| Field | Detail |
|-------|--------|
| Competition | Planet Tanager Open Data Competition |
| Sensor | Tanager-1 (426-band hyperspectral, 30m GSD) |
| Tracks | Lightning Case Studies, Code & Scripts, Technical Analysis |
| Deadline | August 31, 2026 |
| Prize | AGU 2026 visibility + Planet relationship |
| Our Focus | FireSpec — wildfire analysis (burn severity + fuel moisture) |

## Prior Research

Tobler's deep research (completed in detr_geo project) identified 6 viable project ideas. The board selected **wildfire/hyperspectral** as the focus. Key findings:

1. **FireSpec** — Live fuel moisture + post-fire burn severity using MESMA spectral unmixing. Published evidence shows 2x better R² than Sentinel-2 dNBR.
2. The LA wildfire time series (7 Tanager scenes) provides a compelling, timely case study.
3. The tooling gap is at the **application layer** — HyperCoast handles I/O, SPy handles algorithms, but no one has built domain-specific pipelines.
4. Non-methane applications are underserved in Tanager research — wildfire is a gap to fill.

## Target Users

| Persona | Background | Primary JTBD |
|---------|-----------|-------------|
| Wildfire Researcher | Remote sensing / fire ecology | Analyze burn severity with hyperspectral precision |
| Emergency Manager | Government / disaster response | Assess post-fire conditions rapidly |
| Remote Sensing Scientist | PhD/MS, hyperspectral familiar | Build on open tools for spectral analysis |
| OGC Community Member | Standards-aware geospatial developer | Leverage OGC-interoperable outputs |

## Tech Stack (Preliminary)

- **Python 3.10+**
- **spectral (SPy)** — spectral analysis algorithms (MESMA, SAM, band math)
- **HyperCoast** — Tanager data I/O (`read_tanager()`)
- **rasterio** — raster I/O
- **xarray** — N-dimensional array handling
- **geopandas** — vector output
- **leafmap** — visualization
- **STAC / pystac** — data discovery and OGC interoperability
- **Jupyter** — notebook-based deliverables

## Key Technical Domains

### Hyperspectral Analysis
- Spectral unmixing (MESMA — Multiple Endmember Spectral Mixture Analysis)
- Spectral Angle Mapper (SAM)
- Band ratios and indices (NBR, NDVI, NDWI, custom)
- Endmember extraction and spectral library matching
- Dimensionality reduction (PCA, MNF)

### Wildfire Science
- Live Fuel Moisture Content (LFMC) estimation
- Burn severity classification (CBI correlation)
- Pre/post-fire change detection
- Vegetation recovery monitoring
- Composite burn index mapping

### Data Standards
- STAC (SpatioTemporal Asset Catalog)
- GeoZarr (cloud-native multidimensional)
- OGC API standards
- GeoPackage / GeoJSON

## Phasing (Preliminary)

### Phase 1 — Foundation & Literature Review
- Deep literature review of hyperspectral wildfire analysis
- Tanager-1 sensor characterization and data access
- Tool evaluation (SPy, HyperCoast, EMIT tools)
- Endmember library research (USGS Spectral Library v7)

### Phase 2 — Data Pipeline & Exploration
- Tanager data download and preprocessing pipeline
- Exploratory spectral analysis on LA wildfire scenes
- Band selection for wildfire-relevant indices
- Atmospheric correction assessment

### Phase 3 — Core Analysis
- MESMA implementation for burn severity mapping
- LFMC estimation pipeline
- Multi-temporal change detection
- Validation against Sentinel-2 / MODIS baselines

### Phase 4 — Packaging & Submission
- Jupyter notebook preparation (competition deliverable)
- OGC-interoperable output formats
- Documentation and reproducibility
- Community value narrative

## Development Approach

This project uses **spec-driven development** via OpenSpec. Research phases begin as proposals in `openspec/changes/` and research findings are tracked in `docs/research-memory.md`.

All research and implementation is done by AI agents, coordinated through the Paperclip pipeline. Tobler leads research; Product Queen synthesizes findings into specs; coders implement tooling.

## Repository Structure

```
tanager/
├── openspec/           # Spec-driven development
│   ├── changes/        # Active change proposals
│   ├── archive/        # Completed changes
│   ├── specs/          # Current capability specs
│   ├── deferred/       # Deferred proposals
│   ├── project.md      # This file
│   ├── AGENTS.md       # AI assistant instructions
│   └── PROCESS.md      # Process guide
├── docs/               # Memory system
│   ├── product-memory.md
│   ├── engineering-memory.md
│   └── research-memory.md
├── research/           # Research reports and findings
├── notebooks/          # Jupyter notebooks (competition deliverables)
├── src/                # Source code (when implementation begins)
│   └── tanager/        # Python package
├── tests/              # Test suite
├── data/               # Sample data and spectral libraries
├── .claude/            # Agent configurations
│   └── agents/
├── CLAUDE.md           # Project instructions
└── README.md           # Project overview
```

## Key References

- Planet Tanager Competition: https://www.planet.com/pulse/announcing-the-tanager-open-data-competition/
- HyperCoast: https://github.com/opengeos/HyperCoast
- spectral (SPy): https://github.com/spectralpython/spectral
- USGS Spectral Library v7: https://www.usgs.gov/labs/spectroscopy-lab/science/spectral-library
- MESMA: Multiple Endmember Spectral Mixture Analysis
- Prior research: `research/tanager-deep-research.md`, `research/tanager-competition-analysis.md`

## Status

**Phase: Project Onboarding — Setting up research infrastructure**
