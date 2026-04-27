# Research Memory: Tanager Competition

> Long-term memory for research agents. Tracks literature, experiments, data sources, and scientific findings.

**Location:** `/docs/research-memory.md`
**Owner:** Tobler (Research Lead)
**Updated:** 2026-04-27
**Version:** 1.0

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
| Bands | 426 (contiguous) |
| Spectral range | VNIR + SWIR (~400-2500 nm) |
| GSD | 30 meters |
| Swath width | ~30 km |
| Operator | Planet Labs |
| Launch | 2024 |

---

## Available Data

### LA Wildfire Time Series
- **Scenes:** 7 acquisitions (Dec 2024 – Jul 2025)
- **Coverage:** Los Angeles region
- **Significance:** Pre-fire, active fire period, and post-fire recovery
- **Status:** Not yet downloaded — access via Planet API

### Other Available Scenes (from Tobler's research)
- Germany agricultural area (10 scenes)
- Kenya (8 scenes)
- Hawaii coral reef area

---

## Key Literature (To Be Populated)

### Hyperspectral Wildfire Analysis
| Paper/Source | Key Finding | Relevance |
|-------------|-------------|-----------|
| (TBD — Phase 1 research) | | |

### Spectral Unmixing (MESMA)
| Paper/Source | Key Finding | Relevance |
|-------------|-------------|-----------|
| (TBD — Phase 1 research) | | |

### Live Fuel Moisture Content (LFMC)
| Paper/Source | Key Finding | Relevance |
|-------------|-------------|-----------|
| (TBD — Phase 1 research) | | |

---

## Spectral Libraries & Reference Data

| Resource | URL | Purpose |
|----------|-----|---------|
| USGS Spectral Library v7 | usgs.gov/labs/spectroscopy-lab | Endmember spectra for unmixing |
| ECOSTRESS Spectral Library | speclib.jpl.nasa.gov | Additional vegetation spectra |
| (More TBD) | | |

---

## Tools Evaluated

| Tool | Purpose | Verdict | Notes |
|------|---------|---------|-------|
| HyperCoast | Tanager data I/O | **USE** | Has `read_tanager()`, maintained by opengeos |
| spectral (SPy) | Spectral analysis | **USE** | MESMA, SAM, mature library |
| EMIT tools | Hyperspectral processing | **EVALUATE** | NASA's EMIT mission tools, may be adaptable |
| pysptools | Spectral unmixing | **EVALUATE** | Alternative MESMA implementation |

---

## Experiments Log

| Date | Experiment | Result | Notes |
|------|-----------|--------|-------|
| (TBD — Phase 2) | | | |

---

## Open Questions

1. What atmospheric correction is needed for Tanager data? (Planet may provide L2 products)
2. Which endmember spectra best represent LA fire-affected vegetation?
3. What is the optimal band subset for LFMC estimation?
4. How does Tanager's 30m GSD compare to airborne hyperspectral for burn severity?
5. Can we validate against BARC (Burned Area Reflectance Classification) maps?
