# Capability: Visualization — Geographic Basemaps, Publication-Quality Figures, Interactive Maps

**Capability ID:** visualization
**Version:** 1.0.0
**Change:** 004-visualization-overhaul
**Date:** 2026-05-04

---

## Overview

This capability adds geographic visualization, publication-quality figures, and interactive maps
to the Tanager pipeline. It transforms georeferenced GeoTIFF outputs into cartographically
contextualized maps with basemaps, fire perimeters, comparison panels, and temporal trajectory
charts suitable for competition submission (20% of judging score).

---

## ADDED Requirements

### Requirement: Geo-Aware Raster Rendering

The module SHALL provide a `plot_map()` function that renders any xarray DataArray with
geographic coordinates (UTM easting/northing), product-appropriate colormaps, and labeled colorbars.

#### Scenario: Render NBR map with geographic coordinates

WHEN `plot_map(nbr_da, product_name="nbr")` is called on a DataArray with EPSG:32611 CRS
THEN the figure axes SHALL show UTM easting/northing labels (not pixel indices)
AND the colorbar SHALL use RdYlGn colormap with range [-1, 1] and label "NBR"
AND the function SHALL return a matplotlib Figure object

#### Scenario: Render with explicit colormap override

WHEN `plot_map(da, cmap="hot", vmin=0, vmax=3)` is called with explicit parameters
THEN the explicit parameters SHALL override any PRODUCT_STYLES defaults

#### Scenario: Publication mode rendering

WHEN `plot_map(da, publication=True)` is called
THEN the figure SHALL render at 300 DPI with 12pt title font
AND the output SHALL be suitable for journal or competition submission at printed size

---

### Requirement: Product-Type Colormap Presets

The module SHALL define a `PRODUCT_STYLES` dictionary mapping product names to default rendering
parameters (colormap, value range, colorbar label, class tick positions) for all pipeline products.

#### Scenario: Product style lookup for dNBR

WHEN `PRODUCT_STYLES["dnbr"]` is accessed
THEN it SHALL return cmap="RdYlGn_r", vmin=-0.5, vmax=1.0, label="dNBR"
AND class tick positions for USGS severity thresholds

#### Scenario: All pipeline products covered

WHEN iterating over all product types (nbr, ndvi, ndwi, dnbr, cbi, severity, char, pv, npv, soil, lfmc)
THEN each SHALL have an entry in PRODUCT_STYLES with valid colormap, range, and label

---

### Requirement: Basemap Tile Integration

The module SHALL provide an `add_basemap()` function that overlays contextily basemap tiles
on matplotlib axes with CRS-aware reprojection.

#### Scenario: Add satellite basemap behind raster overlay

WHEN `add_basemap(ax, source="satellite", alpha=0.3)` is called on axes showing a UTM raster
THEN satellite imagery tiles SHALL appear behind the raster overlay
AND the raster data SHALL remain fully visible above the tiles (zorder ordering)

#### Scenario: Graceful degradation without network

WHEN `add_basemap(ax)` is called without network connectivity
THEN a warning SHALL be logged ("Basemap tiles unavailable")
AND the axes SHALL be returned unchanged with no error raised

---

### Requirement: Fire Perimeter Overlay

The module SHALL provide functions to load NIFC fire perimeter geometries and overlay them
as vector boundaries on any map axes.

#### Scenario: Load NIFC fire perimeter GeoJSON

WHEN `load_fire_perimeters(path)` is called with a valid GeoJSON or shapefile path
THEN a GeoDataFrame SHALL be returned with geometry and fire name attributes

#### Scenario: Overlay fire perimeter on severity map

WHEN `overlay_perimeters(ax, perimeters, color="red", linestyle="--")` is called
THEN fire boundary polygons SHALL appear as styled outlines on the map
AND perimeters SHALL be reprojected to match the axes CRS if needed

---

### Requirement: Scale Bar

The module SHALL provide an `add_scalebar()` function that adds a distance scale bar
to geo-referenced map axes.

#### Scenario: Add 5 km scale bar to UTM map

WHEN `add_scalebar(ax, length_km=5)` is called on axes with EPSG:32611 projection
THEN a 5 km scale bar SHALL appear in the lower-left corner with distance label

---

### Requirement: Before/After Comparison Panel

The module SHALL provide a `plot_before_after()` function that creates side-by-side 2-panel
figures comparing pre-fire and post-fire maps with shared colormap and colorbar.

#### Scenario: NBR before/after comparison

WHEN `plot_before_after(pre_nbr, post_nbr, "nbr")` is called
THEN a 2-panel figure SHALL be produced with identical colormap and value range
AND the left panel SHALL be titled with the pre-fire date
AND the right panel SHALL be titled with the post-fire date
AND a single shared colorbar SHALL span both panels

#### Scenario: Before/after with fire perimeter overlay

WHEN `plot_before_after(pre, post, "nbr", fire_perimeters=gdf)` is called
THEN fire perimeter boundaries SHALL appear on both panels

---

### Requirement: Temporal Trajectory Chart

The module SHALL provide a `plot_temporal_trajectory()` function that creates line charts
showing index evolution over time with fire event markers and error bands.

#### Scenario: NBR temporal trajectory across 5 dates

WHEN `plot_temporal_trajectory(dates, nbr_means, "NBR", fire_date="2025-01-07", error_bands=stds)` is called
THEN a line chart SHALL show NBR values at each date with markers
AND a vertical dashed red line SHALL mark the fire ignition date
AND shaded bands SHALL show variability around each point

#### Scenario: Multiple series on same axes

WHEN `plot_temporal_trajectory` is called with an existing axes parameter
THEN the new series SHALL be overlaid on the existing axes
AND the legend SHALL include both series

---

### Requirement: Multi-Panel Severity Summary

The module SHALL provide a `plot_severity_summary()` function that creates a multi-panel grid
showing MESMA fractions alongside CBI and classified severity.

#### Scenario: Full severity summary figure

WHEN `plot_severity_summary(fractions, cbi, severity_class)` is called
THEN a 6-panel grid SHALL be produced: char, pv, npv, soil, CBI, severity class
AND each panel SHALL use its product-specific colormap from PRODUCT_STYLES
AND all panels SHALL share the same geographic extent

---

### Requirement: Difference Map with Severity Boundaries

The module SHALL provide a `plot_difference_map()` function for styled dNBR/dNDVI maps
with USGS severity class boundary contours.

#### Scenario: dNBR with USGS severity class boundaries

WHEN `plot_difference_map(dnbr, "dnbr")` is called
THEN the dNBR raster SHALL be displayed with diverging colormap
AND contour lines SHALL appear at USGS severity thresholds [0.1, 0.27, 0.44, 0.66]
AND contours SHALL be labeled with class names

---

### Requirement: Interactive Leafmap Map

The module SHALL provide an `interactive_map()` function that creates HTML-based interactive
maps for Jupyter notebook embedding with toggleable layers.

#### Scenario: Interactive burn severity explorer

WHEN `interactive_map([(cbi, "cbi"), (nbr, "nbr")], perimeters=path)` is called
THEN an interactive map widget SHALL be returned with satellite basemap
AND CBI and NBR SHALL be available as toggleable overlay layers
AND fire perimeters SHALL appear as vector outlines

#### Scenario: Graceful fallback without leafmap

WHEN leafmap is not installed
THEN the function SHALL attempt to use folium as a fallback
AND raise ImportError with a helpful message if neither is available

---

### Requirement: Notebook Convenience Helper

The module SHALL provide a `show_product()` function that auto-selects rendering parameters
and dispatches to either static or interactive output.

#### Scenario: Quick notebook display

WHEN `show_product(nbr, "nbr", "2025-01-23")` is called
THEN a publication-quality matplotlib figure SHALL be returned with geographic axes and colorbar

#### Scenario: Interactive mode dispatch

WHEN `show_product(nbr, "nbr", interactive=True)` is called
THEN an interactive leafmap widget SHALL be returned

---

### Requirement: Multi-Format Figure Export

The module SHALL provide a `save_figure()` function for exporting figures in multiple formats
at publication quality.

#### Scenario: Export for competition submission

WHEN `save_figure(fig, "outputs/severity_map", formats=["png", "pdf"])` is called
THEN "outputs/severity_map.png" SHALL be written at 300 DPI
AND "outputs/severity_map.pdf" SHALL be written as vector graphics
AND both files SHALL have tight bounding boxes (minimal whitespace)

---

## Module Public API

```python
# Geographic foundation
plot_map(da, title, cmap, vmin, vmax, product_name, publication, basemap, **kwargs) -> Figure
PRODUCT_STYLES: dict[str, ProductStyle]

# Context layers
add_basemap(ax, source, alpha) -> Axes
load_fire_perimeters(path) -> GeoDataFrame
overlay_perimeters(ax, perimeters, color, linestyle, linewidth, label) -> Axes
add_scalebar(ax, length_km, location) -> Axes

# Comparison & temporal
plot_before_after(pre, post, product_name, fire_perimeters, basemap, **kwargs) -> Figure
plot_temporal_trajectory(dates, values, product_name, fire_date, error_bands, **kwargs) -> Figure
plot_severity_summary(fractions, cbi, severity_class, **kwargs) -> Figure
plot_difference_map(diff_da, product_name, class_boundaries, **kwargs) -> Figure

# Interactive & export
interactive_map(layers, center, zoom, perimeters, **kwargs) -> Map
show_product(da, product_name, scene_date, interactive, **kwargs) -> Figure | Map
save_figure(fig, path, formats) -> list[Path]
```

---

## Dependencies

| Package | Purpose | New? |
|---------|---------|------|
| contextily | Basemap tile fetching for static matplotlib figures | **YES** |
| matplotlib | Core plotting (already transitive dep) | No |
| geopandas | Fire perimeter loading and CRS reprojection | No |
| rioxarray | CRS metadata access from xarray DataArrays | No |
| leafmap | Interactive map widgets (via HyperCoast) | No (transitive) |
| folium | HTML map rendering | No (transitive via leafmap) |

---

## Import Dependency Rule

`visualization.py` MAY import from any tanager module (leaf consumer like `validation.py`).
**No module may import FROM visualization.py** — it is output-only, never used as input to analysis.
