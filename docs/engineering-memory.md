# Engineering Memory: Tanager Competition

> Long-term memory for Engineering Manager. Tracks architecture, tech debt, and code quality.

**Location:** `/docs/engineering-memory.md`
**Owner:** Engineering Manager (Crenshaw)
**Updated:** 2026-04-27
**Version:** 1.0

---

## Purpose

This document is the Engineering Manager's working memory. It tracks:
1. **What exists** — Prevent duplicate implementations
2. **Architecture decisions** — What to use, what to avoid
3. **Tech debt** — Known issues and their status
4. **Patterns** — How things should be done

**All coding agents should check this before building new features.**

---

## Architecture Overview

### Status: Pre-implementation

No code exists yet. This section will be populated as implementation begins.

### Planned Architecture

```
tanager/
├── src/tanager/          # Python package
│   ├── __init__.py
│   ├── io.py             # Tanager data I/O (via HyperCoast)
│   ├── spectral.py       # Spectral analysis (MESMA, SAM, indices)
│   ├── fire.py           # Wildfire-specific analysis (burn severity, LFMC)
│   ├── validation.py     # Validation against reference data
│   └── viz.py            # Visualization helpers
├── notebooks/            # Competition deliverables
├── data/                 # Sample data, spectral libraries
└── tests/                # Test suite
```

### Key Dependencies

| Library | Purpose | Version | Notes |
|---------|---------|---------|-------|
| spectral (SPy) | Spectral algorithms | Latest | MESMA, SAM, endmember extraction |
| HyperCoast | Tanager I/O | Latest | `read_tanager()` function |
| rasterio | Raster I/O | >=1.3 | Geospatial raster handling |
| xarray | N-dim arrays | Latest | Hyperspectral cube handling |
| geopandas | Vector ops | >=0.12 | Output geometries |

---

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data format | xarray for hyperspectral cubes | 426 bands = N-dimensional, xarray is standard |
| I/O layer | HyperCoast | Already has `read_tanager()`, maintained by opengeos |
| Spectral analysis | SPy (spectral-python) | Mature, MESMA/SAM implementations |
| Output format | GeoPackage + GeoZarr | OGC-interoperable, cloud-native |
| Notebooks | Jupyter | Competition deliverable format |

---

## Patterns

### Spectral Data Handling
- Always preserve wavelength metadata alongside pixel values
- Use xarray DataArrays with `wavelength` coordinate, not raw numpy
- Wavelengths in nanometers (nm), not micrometers

### Validation
- Compare against Sentinel-2 dNBR as baseline
- Use known fire perimeters (NIFC) for spatial validation
- Report R², RMSE, and bias for quantitative comparisons

---

## Tech Debt Tracking

No tech debt yet — project is pre-implementation.

---

## Recent Changes

| Date | Change | Status |
|------|--------|--------|
| 2026-04-27 | Project initialized | **DONE** |
