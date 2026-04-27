# Tanager Competition

Research and development project for the [Planet Tanager Open Data Competition](https://www.planet.com/pulse/announcing-the-tanager-open-data-competition/).

## Focus

**FireSpec** — Wildfire analysis using Tanager-1's 426-band hyperspectral imagery:
- Post-fire burn severity mapping using MESMA spectral unmixing
- Live Fuel Moisture Content (LFMC) estimation
- Case study: LA wildfire time series (7 scenes, Dec 2024 – Jul 2025)

## Competition

- **Deadline:** August 31, 2026
- **Sensor:** Tanager-1 (426 bands, 30m GSD, VNIR+SWIR)
- **Track:** Code & Scripts / Technical Analysis

## Project Structure

```
tanager/
├── openspec/       # Spec-driven development
├── docs/           # Memory system (product, engineering, research)
├── research/       # Research reports
├── notebooks/      # Jupyter notebooks (competition deliverables)
├── src/tanager/    # Python package (when implementation begins)
├── tests/          # Test suite
├── data/           # Sample data and spectral libraries
└── CLAUDE.md       # AI assistant instructions
```

## Prior Research

- `research/tanager-deep-research.md` — 6 project ideas with OGC community analysis
- `research/tanager-competition-analysis.md` — Competition strategy and tool evaluation
