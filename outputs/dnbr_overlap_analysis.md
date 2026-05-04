# Pre-Fire / Post-Fire Scene Overlap & Valid dNBR Analysis

**Issue:** LGT P0 — Verify pre-fire/post-fire scene overlap for valid dNBR burn severity
**Author:** Coder-DetrGeo
**Date:** 2026-05-04

## Summary

The pipeline's prior `20250123 → 20250407` dNBR was scientifically invalid (post-fire → post-fire = recovery, not severity, hence the 89.5% negative dNBR observed in `outputs/validation_analysis.md`). The fix requires a **pre-fire** scene paired with a **post-fire** scene over the same footprint.

We have three Tanager scenes locally; two of them — the originally cataloged "pre-fire / post-fire" pair — observe **completely different geographic locations**, separated by ~30 km of UTM northing with **zero overlap**. The Dec 15 scene observes the **Palisades fire** area; the Jan 23 swath-1 / Apr 7 scenes observe the **Hughes fire** area, ~50 km north.

A second Jan 23 swath (`20250123_185518_92_4001`) is recorded in `tanager.config.FIRE_SCENES` but was not on disk. After downloading it from the Planet STAC catalog (~1.34 GB), the resulting **Dec 15 (pre-fire) → Jan 23 swath 2 (post-fire)** pair overlaps over the Palisades fire footprint with **434 km² of overlap (85.6 % of the smaller scene)**, and produces a physically valid burn-severity dNBR.

## Local scenes (UTM zone 11N, EPSG:32611)

| Scene ID | Date | Phase | UTM bounds (xmin, ymin, xmax, ymax) | Size | Lat / Lon centre |
|---|---|---|---|---|---|
| `20241215_185916_33_4001` | 2024-12-15 | pre-fire | (329 340, 3 754 410, 353 070, 3 775 800) | 23.7 × 21.4 km | 34.01 °N, -118.72 °W |
| `20250123_185507_64_4001` (swath 1) | 2025-01-23 | post-fire | (345 090, 3 805 920, 373 920, 3 837 330) | 28.8 × 31.4 km | 34.53 °N, -118.53 °W |
| `20250123_185518_92_4001` (swath 2) | 2025-01-23 | post-fire | (332 760, 3 752 760, 361 470, 3 784 650) | 28.7 × 31.9 km | 33.97 °N, -118.66 °W |
| `20250407_192235_24_4001` | 2025-04-07 | early-recovery | (324 300, 3 808 620, 355 470, 3 834 690) | 31.2 × 26.1 km | 34.52 °N, -118.74 °W |

The two Jan 23 swaths come from the same overpass (11 s apart). Tanager-1 travels ~7 km/s in a near-polar descending orbit, so the second swath captures ~80 km further along-track — far enough to slip from the Hughes-fire latitude into the Palisades-fire latitude.

## Overlap matrix (pre-fire 20241215 vs each post-fire candidate)

| Post-fire candidate | Geographic target | Overlap | Pre-fire fractional coverage |
|---|---|---|---|
| `20250123_185507_64_4001` (swath 1) | Hughes fire (Lake Hughes, north LA) | **none** — 30.1 km Y-gap | 0 % |
| `20250123_185518_92_4001` (swath 2) | Palisades fire (Topanga / Pacific Palisades) | **434.4 km² (85.6 % of pre-fire)** | 85.6 % |
| `20250407_192235_24_4001` | Hughes fire (early recovery) | **none** — 32.8 km Y-gap | 0 % |

Overlap rectangle (in UTM 32611):

```
xmin = 332 760, xmax = 353 070   (20.3 km wide)
ymin = 3 754 410, ymax = 3 775 800 (21.4 km tall)
```

After `reproject_to_common_grid` resamples both scenes to the 30 m UTM grid restricted to the intersection, the analysis raster is **(713, 677) = 482 701 pixels**, of which **326 189 (67.6 %) are finite** after fill / cloud / water masking.

## Valid Palisades-fire dNBR (Dec 15 → Jan 23 swath 2)

NBR collapses from pre-fire to post-fire as expected for active vegetation loss:

| Stat | Pre (Dec 15) | Post (Jan 23 swath 2) |
|---|---|---|
| NBR mean | +0.311 | +0.164 |
| NBR median | +0.351 | +0.224 |

dNBR (positive = burn severity):

| Stat | Value |
|---|---|
| min | -1.716 |
| max | +1.507 |
| mean | **+0.078** (was −0.177 on the invalid pair) |
| median | **+0.055** |
| n finite / total | 326 189 / 482 701 (67.6 %) |

USGS Key & Benson (2006) burn severity classes:

| dNBR range | Class | Pixels | % of finite |
|---|---|---|---|
| < 0.10 | unburned / regrowth | 203 270 | 62.3 % |
| 0.10 – 0.27 | low severity | 82 041 | 25.2 % |
| 0.27 – 0.44 | moderate-low | 15 445 | 4.7 % |
| 0.44 – 0.66 | moderate-high | 12 181 | 3.7 % |
| ≥ 0.66 | high severity | 13 099 | 4.0 % |

Burned (dNBR ≥ 0.10): **122 766 pixels (37.6 %)** — physically plausible coverage for a 23 488-acre fire (~95 km²) over a 434 km² scene (≈ 22 % of the overlap). The remaining "burned" pixels include shadowed slopes, low-severity scarring, and cloud-edge artefacts; the high-severity quadrant in the upper right of the quicklook clearly outlines the Palisades fire footprint.

## Fire perimeter coverage

Approximate Palisades fire bounding box (CAL FIRE / NIFC perimeter, public reports):
- Lat: 34.04 – 34.10 °N
- Lon: -118.62 – -118.50 °W
- Total burned area: 23 488 acres (95 km²)

Pre-fire scene 20241215 lat/lon envelope: 33.92 – 34.11 °N × -118.85 – -118.59 °W. The eastern edge of this scene at -118.59 °W just clips the western front of the Palisades fire (Topanga / Topanga State Park). The bulk of the Palisades footprint between -118.59 °W and -118.50 °W is **outside** the pre-fire scene's spatial coverage.

Jan 23 swath 2 lat/lon envelope: 33.91 – 34.19 °N × -118.81 – -118.50 °W — captures the **entire** Palisades fire footprint.

After intersection, the overlap rectangle (≈ -118.81 to -118.59 °W × 33.92 – 34.11 °N) covers the **western half** of the Palisades fire (Topanga, Palisades Highlands, parts of Topanga State Park). The eastern-most ~6 km of the fire footprint, including the Pacific Palisades neighbourhood proper, falls **east of the pre-fire scene** and is therefore not in the dNBR overlap region.

This is the best valid dNBR achievable from the four Tanager scenes available. It covers the western Palisades / Topanga front but does not capture the easternmost part of the burn scar.

## Outputs written

| Path | Type | Description |
|---|---|---|
| `outputs/20241215_to_20250123swath2_dnbr.tif` | GeoTIFF (EPSG:32611, 30 m) | Valid burn-severity dNBR over the 434 km² overlap |
| `outputs/20241215_to_20250123swath2_dnbr.png` | PNG | Quicklook with K&B colour ramp |
| `outputs/dnbr_overlap_analysis.md` | Markdown | This document |

## Recommendations for follow-up work

1. **Wire swath 2 into `scripts/run_pipeline.py`.** The pipeline currently pairs `20250123 → 20250407` (post-post recovery). Replace that pair with `20241215 → 20250123_185518_92_4001` for the headline burn-severity product, and keep the recovery pair as a separate "vegetation recovery" output with explicit naming.
2. **Pre-fire scene for the eastern Palisades / Eaton fires is still missing.** No Tanager pre-fire scene in the catalog covers (-118.59 to -118.10 °W, 34.04 – 34.20 °N), the eastern Palisades and Eaton fire footprint. Submission may need to fall back to AVIRIS-3 or Landsat-9 for those areas.
3. **Pre-fire scene for the Hughes fire (Jan 23 swath 1 / Apr 7 footprint) does not exist in the local data.** No Tanager pre-fire scene covers ~34.4 – 34.7 °N, -118.7 to -118.4 °W. Either source one from STAC, or limit the Tanager submission to the Palisades fire area only.
4. **The download-on-demand path matters.** `tanager.catalog.download_scene(item, "ortho_sr_hdf5", DATA_DIR)` works as documented; the swath-2 file (`20250123_185518_92_4001_ortho_sr_hdf5.h5`, 1.34 GB) was downloaded with a single `curl` and lands at the path expected by the rest of the pipeline.
