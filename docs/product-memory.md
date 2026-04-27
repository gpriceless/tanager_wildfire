# Product Memory: Tanager Competition

> Long-term memory for Product Queen. Tracks product evolution, research synthesis, and competition strategy.

**Location:** `/docs/product-memory.md`
**Owner:** Product Queen
**Updated:** 2026-04-27
**Version:** 2.0

---

## Executive Summary

Tanager Competition is a research project targeting the Planet Tanager Open Data Competition (deadline: August 31, 2026). We are building a wildfire-focused hyperspectral analysis toolkit using Tanager-1's 426-band imagery. The deliverable is an open-source, OGC-interoperable submission demonstrating burn severity mapping and live fuel moisture estimation.

---

## Current State

**Last Updated:** 2026-04-27
**Current Phase:** Phase 2 — Data Pipeline & Project Scaffolding (spec complete, awaiting EM review)

### What Exists
| Capability | Status | Key Details |
|-----------|--------|-------------|
| Prior Research | Complete | Competition analysis + deep research (from detr_geo) |
| Literature Review | Complete | MESMA burn severity + PLSR LFMC methodology validated |
| Data Access Research | Complete | STAC catalog, HyperCoast I/O, scene inventory confirmed |
| Project Setup | In Progress | OpenSpec initialized, memory system created, change 002 proposed |
| Data Pipeline (code) | Not Started | Spec ready at `openspec/changes/002-data-pipeline/` |
| Analysis Pipeline | Not Started | Phase 3 — depends on data pipeline |

### Remaining Gaps
- No Tanager data downloaded or accessible locally
- No spectral analysis code
- No Jupyter notebook structure
- Competition track not formally selected (but FireSpec direction confirmed)

---

## Research Findings

### Phase 1 Research (Complete)

**Literature Review** (Tobler, 2026-04-27):
- MESMA with Char/PV/NPVS endmembers is proven: PRISMA R²=0.79 vs Sentinel-2 R²=0.46 (Quintano 2023)
- Tanager's 426 bands at ~5nm should equal or exceed PRISMA's 240 bands at ~12nm
- PLSR achieves R²=0.72-0.94 for LFMC at leaf level
- Key wavelengths: 970nm, 1200nm (water), 1680nm (lignin), 2100nm, 2280nm (cellulose)
- Atmospheric exclusion zones: 1345-1459nm, 1774-1975nm, 2469-2505nm
- 7-scene temporal trajectory is FireSpec's signature differentiator
- Target accuracy: R²>0.70 CBI, RMSE<20% FMC

**Data Access Evaluation** (Tobler, 2026-04-27):
- Static STAC catalog, no auth: `planet.com/data/stac/tanager-core-imagery/catalog.json`
- Use `pystac` (NOT pystac-client) for static catalog
- HDF-EOS5 format, HyperCoast `read_tanager()` → xarray.Dataset
- 12 fire scenes: pre-fire (Dec 15), post-fire (Jan 23), recovery (Apr, Jul, Sep)
- Ortho Surface Reflectance is primary product (~6 GB for fire collection)

### 6 Project Ideas (Tobler, April 2026)

1. **TanagerFlow** — OGC-interoperable hyperspectral analysis toolkit (8-10 wk)
2. **SpectralMiner** — Mineral mapping with USGS Spectral Library v7 (8-10 wk)
3. **HyperWater** — Water quality pipeline (HAB detection, Chl-a) (8-10 wk)
4. **CarbonSpec** — Soil organic carbon mapping (10-12 wk)
5. **FireSpec** — Live fuel moisture + post-fire burn severity (6-8 wk) ← **SELECTED**
6. **CoastalSpec** — Land-water coastal ecosystem assessment (8-10 wk)

### Why FireSpec
- Published evidence: 2x better R² than Sentinel-2 dNBR for burn severity
- LA wildfire time series (7 Tanager scenes, Dec 2024 – Jul 2025) provides compelling case study
- Timely, fundable, operationally relevant to wildfire agencies
- 6-8 week estimate fits competition timeline with room for depth

---

## Roadmap

| Phase | Focus | Timeline | Status |
|-------|-------|----------|--------|
| 1 | Foundation & Literature Review | Apr 2026 | **Complete** |
| 2 | Data Pipeline & Project Scaffolding | Apr-May 2026 | **Specced** (002-data-pipeline) |
| 3 | Core Analysis (MESMA, LFMC) | May-Jun 2026 | Planned |
| 4 | Packaging & Submission | Jul-Aug 2026 | Planned |

---

## Recent Activity

| Date | Event | Details |
|------|-------|---------|
| 2026-04-27 | Phase 1 research complete | Literature review + data access evaluation finished |
| 2026-04-27 | Phase 2 spec proposed | OpenSpec change 002-data-pipeline created, TANAGER-5 |
| 2026-04-27 | Project initialized | OpenSpec, memory system, and research infrastructure set up |
| 2026-04-27 | FireSpec selected | Board chose wildfire focus from 6 candidates |

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-27 | Focus on FireSpec (wildfire) | Board direction + timely case study + OGC gap |
| 2026-04-27 | Separate project from detr_geo | Different sensor, different tech stack, different deliverable |
| 2026-04-27 | Use HyperCoast for I/O, SPy for spectral | Research evaluation: HyperCoast handles Tanager format, SPy is mature for analysis |
| 2026-04-27 | Static STAC + pystac (not pystac-client) | Data access research confirmed static catalog, no API available |
| 2026-04-27 | Ortho SR as primary product | Pre-corrected by Planet (ISOFIT v2.9.5), no additional atm correction needed |

---

## Open Threads

1. **HyperCoast version pinning** — Pin to 0.22.0 or use latest? (EM to decide)
2. **Data directory convention** — `data/raw/fire/` vs env var? (EM to decide)
3. **MESMA package choice** — SPy vs mesma v1.0.8 for Phase 3 (deferred)
4. **Competition track selection** — Lightning Case Studies vs Code & Scripts vs Technical Analysis
5. **Field CBI data** — Can we obtain ground truth for LA fires?
6. **EMIT cross-validation** — Are EMIT scenes available over LA for comparison?
