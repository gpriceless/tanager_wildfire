# Product Memory: Tanager Competition

> Long-term memory for Product Queen. Tracks product evolution, research synthesis, and competition strategy.

**Location:** `/docs/product-memory.md`
**Owner:** Product Queen
**Updated:** 2026-04-27
**Version:** 1.0

---

## Executive Summary

Tanager Competition is a research project targeting the Planet Tanager Open Data Competition (deadline: August 31, 2026). We are building a wildfire-focused hyperspectral analysis toolkit using Tanager-1's 426-band imagery. The deliverable is an open-source, OGC-interoperable submission demonstrating burn severity mapping and live fuel moisture estimation.

---

## Current State

**Last Updated:** 2026-04-27
**Current Phase:** Project onboarding — research infrastructure being set up

### What Exists
| Capability | Status | Key Details |
|-----------|--------|-------------|
| Prior Research | Complete | 2 reports in detr_geo/research/ (competition analysis + deep research) |
| Project Setup | In Progress | OpenSpec initialized, memory system created |
| Data Access | Not Started | Tanager scenes not yet downloaded |
| Analysis Pipeline | Not Started | No code yet |

### Remaining Gaps
- No Tanager data downloaded or accessible locally
- No spectral analysis code
- No Jupyter notebook structure
- Competition track not formally selected

---

## Research Findings (from detr_geo phase)

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

### Key Technical Insights
- Tooling gap is at the **application layer** — HyperCoast + SPy exist but no domain-specific pipelines
- STAC HSI extension stuck at Proposal stage — contributing improvements = high OGC visibility
- GeoZarr V1 RC due May 2026 — prototyping Tanager→GeoZarr = bleeding edge
- Non-methane Tanager applications are underserved — wildfire is a gap

---

## Roadmap

| Phase | Focus | Timeline | Status |
|-------|-------|----------|--------|
| 1 | Foundation & Literature Review | May 2026 | Next |
| 2 | Data Pipeline & Exploration | May-Jun 2026 | Planned |
| 3 | Core Analysis (MESMA, LFMC) | Jun-Jul 2026 | Planned |
| 4 | Packaging & Submission | Jul-Aug 2026 | Planned |

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-27 | Focus on FireSpec (wildfire) | Board direction + timely case study + OGC gap |
| 2026-04-27 | Separate project from detr_geo | Different sensor, different tech stack, different deliverable |
