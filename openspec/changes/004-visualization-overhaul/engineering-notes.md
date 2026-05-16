# Engineering Notes — 004-visualization-overhaul

## Verdict: READY

No blocking gaps. All open questions resolved. File references verified against codebase.
Dependency confirmed (leafmap 0.61.1 with split_map, contextily not yet installed but trivial add).

## Parallelism

- Total sections: 7
- Sequential sections: 7
- Parallel sections: 0
- Max parallel tracks: 1 (all work is in a single file — visualization.py)
- File-disjointness: N/A — no parallel tracks

All sections modify `src/tanager/visualization.py`. No parallelism is possible without
creating merge conflicts. This is by design — a single cohesive module is better than
fragmented files for a ~600-800 line visualization library.

## Execution Plan

- Estimated coder spawns: 7 (one per section, sequential)
- Branch strategy: feature/004-visualization-overhaul (single branch)
- Resource estimate: moderate (4 waves, 25 tasks, each section is one coder session)
- Wave gates: 4 mandatory QA gates between waves

## Open Questions Resolved

1. **Tile provider** → Esri.WorldImagery at alpha=0.3 (satellite context for judges)
2. **NIFC perimeters** → Bundle locally in `data/reference/fire_perimeters/` (no live downloads)
3. **leafmap** → v0.61.1 confirmed, split_map available, full feature set supported
4. **Figure sizes** → Same figsize for both modes; parameterize DPI and fontscale only

## Gotchas

- `data/reference/` directory does not exist yet — coder must `mkdir -p` or the perimeter
  loader should note this as a manual data acquisition step (not automated in code)
- contextily is a NEW pip dependency (not yet installed). Section 1 Task 1 handles this.
- `_quicklook_png` in run_pipeline.py (line 115) should be preserved as deprecated fallback,
  not deleted — the try/except pattern in Section 7 Task 1 handles this correctly
- The existing `plot_fraction_maps()` in unmixing.py remains unchanged — visualization.py
  supplements it with geo-aware versions, does not replace it
- All sections are sequential because they all modify the same file (visualization.py).
  Do NOT attempt parallel execution.

## Dependency Graph Impact

New module `visualization.py` sits at the TOP of the dependency tree alongside `validation.py`:
- MAY import from: config, io, spectral, unmixing, severity, lfmc (all upstream)
- MUST NOT be imported by any existing module
- Only consumers: run_pipeline.py (Section 7), notebooks (future), tests
