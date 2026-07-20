# FireSpec — Project Summary

**Hyperspectral Wildfire Analysis for Planet Tanager-1**
Gabriel Price · github.com/gpriceless/tanager_wildfire · MIT License

---

## What FireSpec Does

FireSpec is an open-source Python toolkit that turns Tanager-1's 426-band hyperspectral
imagery into wildfire products no broadband sensor can match. Using the January 2025 Los
Angeles wildfires (Palisades and Hughes fires) as the case study, FireSpec demonstrates:

- **Sub-pixel burn severity mapping** via MESMA spectral unmixing — decomposing each
  30 m pixel into char, vegetation, and soil fractions
- **Spectral water-content mapping** from resolved 970/1200/2100 nm absorption features
- **Multi-temporal recovery tracking** across 4 acquisitions (Dec 2024–Apr 2025)
- **Quantified sensor comparison** — Tanager-1 vs EMIT, PRISMA, and Sentinel-2

## Key Results

| Product | Result |
|---|---|
| Burn severity | RF classifier trained on BAER ground truth; validated against independent CAL FIRE DINS structure-damage survey (F1 = 0.774 any-damage detection) |
| Sensor comparison | MESMA char fraction R² = 0.991 (EMIT), 0.957 (PRISMA), **0.361** (Sentinel-2) — Sentinel-2's 10 bands leave two-thirds of the variance unexplained |
| Water-content mapping | SAI970, SAI1200, continuum-removal depths at 4 absorption centers — resolving individual features a 10-band sensor blurs together |
| Recovery trajectory | NBR recovery rate +0.082/30 days, with severity-stratified trends |

## Why Tanager-1 Matters Here

Operational fire mapping uses dNBR — a two-band ratio that treats a 50%-char/50%-canopy
pixel and a 100%-moderately-scorched pixel as identical. Tanager-1's 426 contiguous 5 nm
bands let MESMA recover *what is on the ground*, not just *how much changed*. The sensor
comparison quantifies this: broadband indices (NBR, NDVI) show R² > 0.99 across all
sensors — they cannot distinguish Tanager from Sentinel-2. But narrowband products
(absorption depths, MESMA fractions) degrade sharply below hyperspectral resolution,
exposing the information that only 426 bands preserve.

## Toolkit Design

FireSpec is pip-installable (`pip install -e .`), STAC-native (no authentication required
for data access), and ships 972 passing tests. Five annotated Jupyter notebooks reproduce
every result from raw data. The API covers the full pipeline: scene I/O, quality masking,
spectral operations, endmember extraction, MESMA unmixing, severity classification,
water-content indices, cross-sensor simulation, and interactive visualization.

## Limitations (stated honestly)

The BAER severity reference is provisional (unverifiable provenance); no calibrated LFMC
map is produced (insufficient colocated field data — an area-ratio limit, not fixable with
more data); temporal recovery uses only 2 overlap pairs from 4 of 11 cataloged scenes.
Each caveat travels with its number throughout the memo and notebooks.

## Links

- **Repository:** github.com/gpriceless/tanager_wildfire
- **Technical memo:** docs/technical-memo.md (full methodology and results)
- **Notebooks:** 5 end-to-end workflows in `notebooks/`
- **API reference:** docs/api-reference.md
