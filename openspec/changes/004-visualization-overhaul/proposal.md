# Change: Visualization Overhaul — Geographic Basemaps, Fire Perimeters, Publication-Quality Figures

**Change ID:** 004-visualization-overhaul
**Plane Issue:** TBD (created by /run-phase)
**Status:** Approved
**Author:** Product Queen
**Date:** 2026-05-04

---

## Why

Visualization & Storytelling is **20% of the competition score** (20 points). Our current output is
12 matplotlib PNGs with pixel coordinates (x/y pixel labels), no geographic context, no basemaps,
no fire perimeter overlays, no before/after comparison panels, and auto-scaled colormaps without
labeled severity classes. Tobler's triage (LGT-397) estimates this gap costs us **8-12 points**.

The judges will evaluate "maps, plots, narrative, broad audience appeal." Right now our maps look
like lab diagnostics — technically correct data rendered without any cartographic context. A reviewer
cannot visually locate the fire, compare pre- vs post-fire conditions, or follow the recovery
trajectory. This is the single highest-impact improvement remaining before submission packaging.

Phase 3 delivered all the analysis (MESMA, LFMC, severity, validation). The data exists as
georeferenced GeoTIFFs with EPSG:32611 CRS. We just need to render it properly.

## What Changes

### New Module: `src/tanager/visualization.py`

A dedicated visualization module that replaces the current pixel-coordinate plotting with
geographic, publication-quality cartographic output. All functions return matplotlib Figure objects
(notebook-friendly) or write HTML (for interactive maps).

### Section 1: Geographic Foundation

- **Geo-aware raster rendering:** All map functions accept xarray DataArrays with CRS metadata
  and render with geographic coordinates (UTM easting/northing or lat/lon) instead of pixel indices
- **Publication colorbars:** Labeled, fixed-range colorbars for each product type:
  - NBR/NDVI/NDWI: diverging colormap, [-1, 1] range, labeled ticks
  - dNBR: diverging (green=regrowth, red=burn), labeled severity thresholds
  - CBI: sequential (0-3), labeled severity classes
  - MESMA fractions: sequential (0-1), per-class colorbars
  - LFMC: sequential, labeled moisture thresholds
- **Consistent figure style:** Shared defaults for font sizes, DPI (300 for publication),
  figure dimensions, title formatting, coordinate label formatting

### Section 2: Geographic Context Layers

- **Basemap tiles:** contextily integration for adding OpenStreetMap/satellite tiles behind
  raster overlays on static matplotlib figures (requires reprojection to Web Mercator for tile
  fetch, then back to UTM for overlay)
- **Fire perimeter overlays:** Load NIFC fire perimeter GeoJSON/shapefile (Palisades, Eaton,
  Hughes fires) and overlay as vector boundaries on any map
- **Scale bar and north arrow:** Standard cartographic elements for publication figures
- **CRS-aware extent labeling:** UTM grid labels with km formatting

### Section 3: Comparison & Temporal Panels

- **Before/after comparison panels:** Side-by-side or swipe-style 2-panel figures comparing
  pre-fire (Dec 15) and post-fire (Jan 23) for any index (NBR, NDVI, true-color RGB)
- **Temporal trajectory charts:** Line plots showing mean index values (NBR, NDVI, LFMC)
  across all available dates (Dec 15, Jan 23, Apr 7, Jul 24/26, Sep 2/20), with fire event
  marker and error bands (std dev or percentile range)
- **Multi-panel severity maps:** 2x2 or 3x2 grid showing fraction maps + CBI + severity class
  with shared geographic extent and consistent colorbars
- **Difference maps:** Styled dNBR/dNDVI maps with USGS severity class boundaries overlaid

### Section 4: Interactive Maps & Notebook Integration

- **Interactive leafmap/folium maps:** HTML output with layer toggle (basemap, raster overlay,
  fire perimeters, severity classes) for Jupyter notebook embedding
- **Notebook helper:** `show_map(product, scene_date)` convenience function that auto-selects
  colormap, range, and overlay based on product type
- **Export utilities:** Save figures as PNG (300 DPI), PDF, and SVG for competition submission

## Impact

- **New file:** `src/tanager/visualization.py` (~600-800 lines)
- **Modified file:** `src/tanager/__init__.py` (add lazy exports for visualization functions)
- **Data dependency:** NIFC fire perimeter shapefile (~1-5 MB, public domain)
- **New dependencies:** contextily (basemap tiles), possibly matplotlib-scalebar
- **Affected specs:** None existing — new capability spec created
- **No changes to existing analysis modules** — visualization wraps existing outputs

## Dependencies on Existing Code

| Module | What We Use | Notes |
|--------|-------------|-------|
| `spectral.py` | NBR, NDVI, NDWI, dNBR output DataArrays | Geographic metadata via rioxarray |
| `unmixing.py` | Fraction maps (xarray Dataset) | Existing plot functions remain but are supplemented |
| `severity.py` | CBI predictions, severity classifications, trajectories | `compute_trajectories()` provides temporal data |
| `lfmc.py` | LFMC index DataArrays | SAI, NDWI variants, CR depths |
| `io.py` | `load_ortho_scene()`, `get_spatial_info()` | CRS, bounds, transform metadata |
| `config.py` | FIRE_SCENES for date/scene mapping | Temporal ordering |

## Research Summary

No new research needed. This change applies established cartographic practices:

1. **contextily** is the standard Python library for adding basemap tiles to matplotlib figures.
   It fetches tiles from OpenStreetMap/Stamen/ESRI providers and reprojects to match the plot CRS.
2. **NIFC fire perimeters** are publicly available from the National Interagency Fire Center
   (nifc.gov) as GeoJSON/shapefile. The 2025 LA fires (Palisades, Eaton) are included.
3. **leafmap** is already a dependency via HyperCoast. It provides interactive Jupyter map widgets
   with layer control, split-map comparison, and raster overlay capabilities.
4. **Publication figure standards** follow AGU/EGU conventions: 300 DPI, labeled axes, consistent
   fonts, explicit colorbars with units.

## Production Risk

Not applicable — research project.

## Open Questions for EM

1. **contextily tile provider:** Should we default to OpenStreetMap, ESRI World Imagery (satellite),
   or Stamen Terrain? Satellite tiles may clash visually with raster overlays.
2. **NIFC fire perimeter source:** Direct download from nifc.gov vs. bundled in `data/` directory?
   Need to verify exact URL and format for 2025 LA fires.
3. **leafmap version compatibility:** HyperCoast pins `hypercoast>=0.22.0`. Confirm that the
   leafmap version bundled with HyperCoast supports `split_map()` and raster overlay.
4. **Figure size for notebooks:** Jupyter renders at screen resolution. Should we default to
   screen-friendly sizes (8x6) with a `publication=True` flag for 300 DPI export?
