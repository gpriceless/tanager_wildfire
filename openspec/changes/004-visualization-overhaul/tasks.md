# Change: 004-visualization-overhaul

## Engineering Decisions (EM-resolved)

1. **Tile provider: Esri.WorldImagery at alpha=0.3** — Satellite context is essential for
   judges to see what burned. The alpha setting keeps the analysis raster dominant. The
   `add_basemap()` function accepts a `source` parameter for alternatives (terrain, osm).

2. **NIFC fire perimeters: bundled locally at `data/reference/fire_perimeters/`** — Download
   once from NIFC, store GeoJSON files in the repo (small, ~1-5 MB, public domain). The loader
   accepts a local path. No live download mechanism — NIFC URLs change without notice. Include
   a README in that directory documenting acquisition date and source URL for reproducibility.

3. **leafmap compatibility: CONFIRMED** — leafmap 0.61.1 is installed via HyperCoast.
   `split_map()` is available (verified). Full interactive map feature can proceed as designed.
   No degradation path needed, but keep the folium fallback in `interactive_map()` for
   portability (costs nothing to implement).

4. **Figure size convention: parameterize DPI and fontscale, not figsize** — Both modes use
   the same figsize (10, 8 default). `publication=False` renders at 150 DPI with standard fonts.
   `publication=True` renders at 300 DPI with 12pt title / 10pt label fonts. The coder should
   define `_PUB_DEFAULTS` and `_SCREEN_DEFAULTS` dicts at module level.

---

## Wave 1: Geographic Foundation
<!-- execution: sequential -->

### Section 1: Dependencies & Module Scaffold
<!-- execution_mode: sequential -->
<!-- network: REQUIRED — pip install contextily -->

- [ ] Add `contextily` to `pyproject.toml` dependencies
  <!-- files: pyproject.toml (modify) -->
  <!-- gotcha: contextily requires network access for tile fetching at runtime, but the pip
       install itself is straightforward. It depends on mercantile, rasterio (already present),
       and requests (already present). No version pin needed — latest stable is fine. -->
  <!-- test: pip install -e . succeeds; python -c "import contextily" works -->
  <!-- acceptance: contextily importable after editable install -->

- [ ] Create `src/tanager/visualization.py` with module docstring and imports
  <!-- files: src/tanager/visualization.py (create) -->
  <!-- pattern: follow existing module pattern — module docstring with public API list,
       TYPE_CHECKING imports for matplotlib types, lazy import of heavy deps (matplotlib,
       contextily, geopandas) inside functions. See unmixing.py lines 1-90 for reference. -->
  <!-- imports: numpy, xarray, logging, pathlib, typing. Heavy: matplotlib (lazy), contextily (lazy),
       geopandas (lazy), rioxarray (lazy) -->
  <!-- acceptance: module exists, imports cleanly, has __all__ with all public function stubs -->

- [ ] Add visualization lazy exports to `src/tanager/__init__.py`
  <!-- files: src/tanager/__init__.py (modify) -->
  <!-- pattern: add entries to _LAZY_EXPORTS dict following existing convention.
       Functions to export: plot_map, plot_before_after, plot_temporal_trajectory,
       plot_severity_summary, plot_difference_map, interactive_map, show_product, save_figure,
       add_basemap, load_fire_perimeters, overlay_perimeters, add_scalebar, PRODUCT_STYLES -->
  <!-- acceptance: tanager.plot_map resolves (may raise NotImplementedError from stub) -->

### Section 2: Product Styles & Core Rendering
<!-- execution_mode: sequential -->

- [ ] Implement `PRODUCT_STYLES` dictionary with colormap presets for all product types
  <!-- files: src/tanager/visualization.py (modify) -->
  <!-- spec: REQ-VIZ-002 — dict mapping product names to NamedTuple/dataclass with
       cmap, vmin, vmax, label, class_ticks fields.
       Products: nbr, ndvi, ndwi, dnbr, cbi, severity, char, pv, npv, soil, lfmc.
       See spec.md table for exact values. -->
  <!-- pattern: use a simple dataclass ProductStyle with fields: cmap, vmin, vmax, label,
       class_ticks (Optional[list]). Instantiate PRODUCT_STYLES as module-level dict. -->
  <!-- acceptance: PRODUCT_STYLES["dnbr"].cmap == "RdYlGn_r"; all 11 products present -->

- [ ] Implement `plot_map(da, ...)` — geo-aware single-panel raster renderer
  <!-- files: src/tanager/visualization.py (modify) -->
  <!-- spec: REQ-VIZ-001, REQ-VIZ-003 -->
  <!-- signature: plot_map(da: xr.DataArray, title: str = "", cmap: str | None = None,
       vmin: float | None = None, vmax: float | None = None, product_name: str | None = None,
       publication: bool = False, figsize: tuple[float, float] = (10, 8),
       basemap: bool = False, ax: Axes | None = None) -> Figure -->
  <!-- logic:
       1. If product_name provided and cmap/vmin/vmax are None, look up from PRODUCT_STYLES
       2. Extract x/y coordinates from DataArray (these are UTM easting/northing in meters)
       3. Use ax.imshow() with extent computed from coordinate bounds
       4. Format axes: UTM labels with km formatting (e.g., "340 km E")
       5. Add colorbar with label from PRODUCT_STYLES or explicit parameter
       6. If publication=True: dpi=300, fontsize bump
       7. If basemap=True: call add_basemap(ax) after raster rendering
       8. Return figure -->
  <!-- gotcha: DataArray.plot.imshow() can be used but has limited control over colorbar.
       Prefer raw ax.imshow(da.values, extent=[xmin, xmax, ymin, ymax], ...) for full control.
       Make sure to handle NaN values — use masked array or set_bad on colormap. -->
  <!-- gotcha: x/y coords in DataArray are in meters (UTM). Format tick labels to show km
       with reasonable precision: formatter = FuncFormatter(lambda v, _: f"{v/1000:.0f}") -->
  <!-- acceptance: plot_map(nbr_da, product_name="nbr") produces figure with UTM axes, not pixel indices -->

- [ ] Implement `save_figure(fig, path, formats)` — multi-format export utility
  <!-- files: src/tanager/visualization.py (modify) -->
  <!-- spec: REQ-VIZ-032 -->
  <!-- signature: save_figure(fig: Figure, path: str | Path, formats: list[str] = ["png"]) -> list[Path] -->
  <!-- logic: for each format, call fig.savefig(f"{path}.{fmt}", dpi=300, bbox_inches="tight").
       Return list of written paths. Create parent directory if needed. -->
  <!-- acceptance: save_figure(fig, "out/test", ["png", "pdf"]) writes both files -->

- [ ] Write tests for PRODUCT_STYLES, plot_map, and save_figure
  <!-- files: tests/test_visualization.py (create) -->
  <!-- pattern: create synthetic xarray DataArray fixtures with CRS metadata (rio.write_crs).
       Test that plot_map returns Figure, axes have non-pixel labels, colorbar exists.
       Test PRODUCT_STYLES has all expected keys with correct types.
       Test save_figure creates files on disk. -->
  <!-- gotcha: use matplotlib Agg backend in tests (import matplotlib; matplotlib.use("Agg")).
       Close figures after assertions to avoid memory leaks: plt.close(fig). -->
  <!-- acceptance: pytest tests/test_visualization.py passes all tests -->

---

## Wave 2: Geographic Context Layers
<!-- execution: sequential -->
<!-- gate: Wave 1 must pass QA before starting Wave 2 -->

### Section 3: Basemap & Fire Perimeters
<!-- execution_mode: sequential -->

- [ ] Implement `add_basemap(ax, source, alpha)` — contextily tile overlay
  <!-- files: src/tanager/visualization.py (modify) -->
  <!-- spec: REQ-VIZ-010 -->
  <!-- signature: add_basemap(ax: Axes, source: str = "satellite", alpha: float = 0.3) -> Axes -->
  <!-- logic:
       1. Map source string to contextily provider: "satellite" → ctx.providers.Esri.WorldImagery,
          "terrain" → ctx.providers.Stamen.Terrain, "osm" → ctx.providers.OpenStreetMap.Mapnik
       2. Call ctx.add_basemap(ax, crs=crs_string, source=provider, alpha=alpha, zorder=0)
       3. The raster overlay should have zorder=1 (rendered on top)
       4. Wrap in try/except: network failure → log warning, return ax unchanged -->
  <!-- gotcha: contextily needs the axes extent set BEFORE calling add_basemap. The raster
       imshow must be rendered first, then basemap tiles are added underneath.
       Set basemap zorder=0 and raster zorder=1. -->
  <!-- gotcha: contextily.add_basemap expects the CRS as an EPSG string like "EPSG:32611"
       or a pyproj CRS object. Extract from DataArray via da.rio.crs. -->
  <!-- acceptance: add_basemap(ax) adds tiles without error; visual: tiles visible behind raster -->

- [ ] Implement `load_fire_perimeters(path)` — NIFC perimeter loader
  <!-- files: src/tanager/visualization.py (modify) -->
  <!-- spec: REQ-VIZ-011 -->
  <!-- signature: load_fire_perimeters(path: str | Path) -> gpd.GeoDataFrame -->
  <!-- logic: gpd.read_file(path), validate geometry column exists, return GeoDataFrame.
       Support GeoJSON and shapefile formats (geopandas handles both). -->
  <!-- acceptance: load_fire_perimeters("perimeter.geojson") returns GeoDataFrame with geometry -->

- [ ] Implement `overlay_perimeters(ax, perimeters, ...)` — vector boundary overlay
  <!-- files: src/tanager/visualization.py (modify) -->
  <!-- spec: REQ-VIZ-011 -->
  <!-- signature: overlay_perimeters(ax: Axes, perimeters: gpd.GeoDataFrame,
       color: str = "red", linestyle: str = "--", linewidth: float = 2.0,
       label: bool = True) -> Axes -->
  <!-- logic:
       1. Reproject perimeters to match axes CRS if needed
       2. perimeters.boundary.plot(ax=ax, color=color, linestyle=linestyle, linewidth=linewidth)
       3. If label=True and "name" or "incident_name" column exists, add text labels at centroid
       4. Return ax -->
  <!-- acceptance: overlay_perimeters(ax, gdf) draws dashed red outlines -->

- [ ] Implement `add_scalebar(ax, length_km, location)` — scale bar element
  <!-- files: src/tanager/visualization.py (modify) -->
  <!-- spec: REQ-VIZ-012 -->
  <!-- signature: add_scalebar(ax: Axes, length_km: float = 5.0,
       location: str = "lower left") -> Axes -->
  <!-- logic: manually draw a rectangle + text label using ax.add_patch and ax.text.
       Calculate pixel width from length_km * 1000 (UTM is in meters). Position in lower-left
       using axes transform. Black bar with white outline for visibility. -->
  <!-- gotcha: do NOT depend on matplotlib-scalebar package — keep deps minimal. A manual
       implementation is ~15 lines and avoids an extra dependency. -->
  <!-- acceptance: add_scalebar(ax, 5) renders a 5 km bar in lower-left -->

- [ ] Write tests for basemap, perimeter, and scalebar functions
  <!-- files: tests/test_visualization.py (modify) -->
  <!-- pattern: test add_basemap with mocked contextily (patch ctx.add_basemap to no-op).
       Test offline graceful degradation (mock network error → no exception).
       Test load_fire_perimeters with a tiny synthetic GeoJSON fixture.
       Test overlay_perimeters draws on axes without error.
       Test add_scalebar adds patches to axes. -->
  <!-- acceptance: pytest tests/test_visualization.py passes all new tests -->

---

## Wave 3: Comparison & Temporal Panels
<!-- execution: sequential -->
<!-- gate: Wave 2 must pass QA before starting Wave 3 -->

### Section 4: Comparison Panels
<!-- execution_mode: sequential -->

- [ ] Implement `plot_before_after(pre, post, product_name, ...)` — side-by-side comparison
  <!-- files: src/tanager/visualization.py (modify) -->
  <!-- spec: REQ-VIZ-020 -->
  <!-- signature: plot_before_after(pre: xr.DataArray, post: xr.DataArray,
       product_name: str = "nbr", pre_label: str | None = None, post_label: str | None = None,
       fire_perimeters: gpd.GeoDataFrame | None = None, basemap: bool = False,
       publication: bool = False, figsize: tuple[float, float] = (16, 8)) -> Figure -->
  <!-- logic:
       1. Create 1x2 subplot grid with shared colorbar space
       2. Render pre on left panel, post on right panel using plot_map logic (reuse internals)
       3. Same cmap, vmin, vmax from PRODUCT_STYLES[product_name]
       4. Panel titles: pre_label or "Pre-Fire (date)", post_label or "Post-Fire (date)"
       5. Single shared colorbar at bottom or right spanning both panels
       6. Optional: overlay fire perimeters on both panels
       7. Optional: add basemap behind both panels -->
  <!-- gotcha: the pre and post scenes may have different spatial extents (Dec 15 is 713x791,
       Jan 23 is 1047x961). The function should either (a) crop to overlapping extent, or
       (b) render each panel at its own extent with the same colormap range. Option (b) is
       simpler and preserves full spatial coverage — recommend this approach. -->
  <!-- acceptance: plot_before_after(pre_nbr, post_nbr, "nbr") produces 2-panel figure -->

- [ ] Implement `plot_difference_map(diff_da, product_name, ...)` — styled dNBR with severity contours
  <!-- files: src/tanager/visualization.py (modify) -->
  <!-- spec: REQ-VIZ-023 -->
  <!-- signature: plot_difference_map(diff_da: xr.DataArray, product_name: str = "dnbr",
       class_boundaries: dict[str, float] | None = None, publication: bool = False,
       figsize: tuple[float, float] = (10, 8)) -> Figure -->
  <!-- logic:
       1. Render diff raster using plot_map with product_name colormap
       2. If class_boundaries provided (or default USGS dNBR thresholds), overlay contour lines
          using ax.contour(x, y, diff_da.values, levels=list(thresholds.values()))
       3. Label contours with class names using ax.clabel()
       4. Default USGS dNBR thresholds: {"Unburned": 0.1, "Low": 0.27, "Mod-Low": 0.44,
          "Mod-High": 0.66} -->
  <!-- acceptance: plot_difference_map(dnbr) shows raster with labeled contour lines at severity boundaries -->

- [ ] Implement `plot_severity_summary(fractions, cbi, severity_class, ...)` — multi-panel grid
  <!-- files: src/tanager/visualization.py (modify) -->
  <!-- spec: REQ-VIZ-022 -->
  <!-- signature: plot_severity_summary(fractions: xr.Dataset, cbi: xr.DataArray,
       severity_class: xr.DataArray, publication: bool = False,
       figsize: tuple[float, float] = (18, 12)) -> Figure -->
  <!-- logic:
       1. Create 2x3 subplot grid
       2. Panel layout: [char, pv, npv] top row, [soil, CBI, severity] bottom row
       3. Each panel uses its own PRODUCT_STYLES colormap and range
       4. Each panel has its own colorbar
       5. Shared geographic extent across all panels
       6. Tight layout with consistent spacing -->
  <!-- gotcha: fractions Dataset has variables char, pv, npv, soil — extract each.
       The existing plot_fraction_maps() from unmixing.py does something similar but with
       pixel coords. This function adds geographic axes and CBI/severity panels. -->
  <!-- acceptance: plot_severity_summary produces 6-panel figure with correct colormaps -->

- [ ] Write tests for comparison panel functions
  <!-- files: tests/test_visualization.py (modify) -->
  <!-- pattern: test plot_before_after with two synthetic DataArrays. Verify 2 axes in figure.
       Test plot_difference_map produces contour lines when boundaries provided.
       Test plot_severity_summary with synthetic fractions Dataset + CBI + severity DataArrays.
       Verify 6 axes in figure. -->
  <!-- acceptance: pytest tests/test_visualization.py passes all new tests -->

### Section 5: Temporal Trajectory Charts
<!-- execution_mode: sequential -->

- [ ] Implement `plot_temporal_trajectory(dates, values, product_name, ...)` — time series chart
  <!-- files: src/tanager/visualization.py (modify) -->
  <!-- spec: REQ-VIZ-021 -->
  <!-- signature: plot_temporal_trajectory(dates: list[datetime], values: list[float],
       product_name: str = "NBR", fire_date: str | datetime | None = "2025-01-07",
       error_bands: list[float] | None = None, ax: Axes | None = None,
       publication: bool = False, figsize: tuple[float, float] = (12, 6)) -> Figure -->
  <!-- logic:
       1. Create line plot: x=dates, y=values, marker="o", linewidth=2
       2. If error_bands provided: ax.fill_between(dates, values-err, values+err, alpha=0.2)
       3. If fire_date: ax.axvline(fire_date, color="red", linestyle="--", label="Fire Ignition")
       4. Add phase labels: "Pre-Fire" region (before fire_date), "Post-Fire" / "Recovery"
          as text annotations or shaded background regions
       5. X-axis: date formatting (auto or monthly ticks)
       6. Y-axis: product_name label with units from PRODUCT_STYLES
       7. Legend with series name and fire event marker -->
  <!-- gotcha: dates may be strings ("2024-12-15") or datetime objects. Parse with
       pd.to_datetime() for robustness. -->
  <!-- gotcha: the existing severity.compute_trajectories() returns a dict with dates and
       mean values — this function should accept that output format directly. -->
  <!-- acceptance: plot_temporal_trajectory produces line chart with fire event marker -->

- [ ] Write tests for temporal trajectory function
  <!-- files: tests/test_visualization.py (modify) -->
  <!-- pattern: test with 5 synthetic date/value pairs. Verify line plotted, fire event marker
       present (axvline), error bands rendered when provided. -->
  <!-- acceptance: pytest tests/test_visualization.py passes temporal trajectory tests -->

---

## Wave 4: Interactive Maps & Integration
<!-- execution: sequential -->
<!-- gate: Wave 3 must pass QA before starting Wave 4 -->

### Section 6: Interactive Maps
<!-- execution_mode: sequential -->

- [ ] Implement `interactive_map(layers, center, zoom, ...)` — leafmap/folium interactive map
  <!-- files: src/tanager/visualization.py (modify) -->
  <!-- spec: REQ-VIZ-030 -->
  <!-- signature: interactive_map(layers: list[tuple[xr.DataArray, str]] | None = None,
       center: tuple[float, float] | None = None, zoom: int = 12,
       perimeters: str | Path | gpd.GeoDataFrame | None = None,
       basemap: str = "satellite") -> Map -->
  <!-- logic:
       1. Create leafmap.Map(center=center, zoom=zoom) with satellite basemap
       2. For each (da, product_name) in layers:
          a. Write DataArray to temporary GeoTIFF (or use in-memory rasterio MemoryFile)
          b. Add as raster layer with colormap from PRODUCT_STYLES
          c. Layer name = product_name
       3. If perimeters provided: add as GeoJSON vector layer with popup attributes
       4. Add LayerControl widget
       5. Return Map object (displayable in Jupyter) -->
  <!-- gotcha: leafmap may need rasters in EPSG:4326 for web display. Reproject if needed.
       Check leafmap.add_raster() API for CRS handling. -->
  <!-- gotcha: if leafmap import fails (HyperCoast not installed), fall back to folium.
       If both fail, raise ImportError with helpful message. -->
  <!-- acceptance: interactive_map([(nbr, "nbr")]) returns displayable Map widget -->

- [ ] Implement `show_product(da, product_name, scene_date, interactive)` — convenience helper
  <!-- files: src/tanager/visualization.py (modify) -->
  <!-- spec: REQ-VIZ-031 -->
  <!-- signature: show_product(da: xr.DataArray, product_name: str,
       scene_date: str | None = None, interactive: bool = False) -> Figure | Map -->
  <!-- logic:
       1. If interactive=True: return interactive_map([(da, product_name)])
       2. If interactive=False: return plot_map(da, product_name=product_name,
          title=f"{product_name.upper()} {scene_date or ''}", basemap=True)
       3. Auto-detect product_name from DataArray name attribute if not provided -->
  <!-- acceptance: show_product(nbr, "nbr", "2025-01-23") returns Figure with basemap -->

- [ ] Write tests for interactive map and show_product functions
  <!-- files: tests/test_visualization.py (modify) -->
  <!-- pattern: mock leafmap.Map to avoid network dependency. Test that interactive_map
       creates a Map object. Test show_product dispatches correctly based on interactive flag.
       Test fallback behavior when leafmap is not importable. -->
  <!-- acceptance: pytest tests/test_visualization.py passes all new tests -->

### Section 7: Integration & Final Polish
<!-- execution_mode: sequential -->

- [ ] Wire `plot_map` into `run_pipeline.py` replacing `_quicklook_png` for all PNG outputs
  <!-- files: scripts/run_pipeline.py (modify) -->
  <!-- logic: replace _quicklook_png(da, path, title, cmap) calls with:
       fig = tanager.plot_map(da, title=title, product_name=product_name, basemap=False)
       tanager.save_figure(fig, path.with_suffix(""), formats=["png"])
       plt.close(fig)
       This gives all pipeline quicklooks geographic coordinates automatically. -->
  <!-- gotcha: run_pipeline.py currently uses _quicklook_png as an internal function.
       Keep _quicklook_png as a deprecated fallback in case visualization.py fails.
       Wrap the new call in try/except with fallback to _quicklook_png. -->
  <!-- acceptance: running the pipeline produces PNGs with UTM coordinate axes -->

- [ ] Update module `__all__` and verify all public API exports resolve
  <!-- files: src/tanager/visualization.py (modify), src/tanager/__init__.py (verify) -->
  <!-- logic: ensure __all__ in visualization.py matches _LAZY_EXPORTS in __init__.py.
       Test: python -c "import tanager; print(tanager.plot_map)" should not raise. -->
  <!-- acceptance: all 13 lazy exports resolve without error -->

- [ ] Write integration test: generate publication figure from synthetic data end-to-end
  <!-- files: tests/test_visualization.py (modify) -->
  <!-- pattern: create a complete synthetic scene (DataArray with CRS, bounds, realistic values),
       call plot_map with publication=True, save_figure to temp dir, verify PNG written at 300 DPI.
       Optionally call plot_before_after and plot_temporal_trajectory with synthetic data. -->
  <!-- acceptance: integration test produces publication-quality PNG files on disk -->

- [ ] smoke: end-to-end exercise — import tanager, call plot_map + save_figure + interactive_map on synthetic data
  <!-- files: tests/test_visualization.py (modify) -->
  <!-- pattern: single test function that exercises the full public API surface:
       1. Create synthetic DataArray with CRS (EPSG:32611)
       2. Call plot_map, plot_before_after, plot_temporal_trajectory, plot_difference_map
       3. Call save_figure to write PNG and PDF
       4. Call interactive_map (mocked leafmap)
       5. Verify all calls succeed without exception and produce expected return types -->
  <!-- acceptance: smoke test passes — all visualization functions callable end-to-end -->

---

## Summary

| Wave | Sections | Tasks | Focus |
|------|----------|-------|-------|
| 1 | 1-2 | 7 | Module scaffold, product styles, core geo-aware rendering |
| 2 | 3 | 5 | Basemap tiles, fire perimeters, scale bar |
| 3 | 4-5 | 6 | Before/after panels, temporal trajectories, severity grids |
| 4 | 6-7 | 7 | Interactive leafmap, notebook helpers, pipeline integration, smoke test |
| **Total** | **7** | **25** | |

<!-- DEFERRED TASKS
(none — all original PQ tasks retained, one smoke task added by EM)
-->
