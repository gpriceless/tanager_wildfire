# Published numbers, recomputed

Generated 2026-10-01T03:33:21Z by scripts/verify_published_numbers.py. Commit: 5ce3630

| Claim | Published | Recomputed | Definition |
|---|---|---|---|
| dNBR median inside Palisades | 0.588 | 0.588 | land pixels inside the WFIGS perimeter, n=32915 |
| dNBR median outside, all perimeters | 0.060 | 0.060 | land outside Palisades, Franklin and Kenneth |
| dNBR median outside, Palisades only | 0.046 | 0.046 | land outside Palisades; includes the Franklin scar |
| dNBR median inside Franklin | - | -0.007 | 18853 land pixels |
| dNBR median inside Kenneth | - | not in dNBR footprint | 0 land pixels |
| char median inside / outside Palisades | 0.389 / 0.000 | 0.389 / 0.000 | outside = outside every perimeter in the reference file |
| char perimeter AUC | 0.795 | 0.795 |  |
| char median inside Franklin | 0.337 | 0.337 | n=11107 |
| char median inside Kenneth | 0.575 | 0.575 | n=2539 |
| char DINS AUC | 0.530 | 0.530 | destroyed/no-damage points with char modeled, n=4114/3257 |
| char vs dNBR DINS AUC, shared points | 0.530 / 0.721 | 0.532 / 0.721 | points where both are defined, n=856/664 |
| urban library debris DINS AUC | 0.672 | 0.672 | median destroyed 0.337 vs no damage 0.000, n=4128/3262 |
| urban library char DINS AUC | 0.506 | 0.506 | median destroyed 0.000 vs no damage 0.000, n=4128/3262 |
| char inside Palisades, wildland vs built-up (30 m) | 0.501 (AUC 0.887) / 0.000 | 0.475 (AUC 0.863) / 0.000 (AUC 0.489) | built-up = within 30 m of a DINS point; AUC vs outside all perimeters; n=53752/11907 |
| char inside Palisades, wildland vs built-up (100 m) | 0.501 (AUC 0.887) / 0.000 | 0.500 (AUC 0.891) / 0.000 (AUC 0.580) | built-up = within 100 m of a DINS point; AUC vs outside all perimeters; n=45458/20201 |
| char inside Palisades, wildland vs built-up (200 m) | 0.501 (AUC 0.887) / 0.000 | 0.500 (AUC 0.893) / 0.000 (AUC 0.643) | built-up = within 200 m of a DINS point; AUC vs outside all perimeters; n=39878/25781 |
| Franklin char mean inside / outside (2024-12-15) | 0.490 / 0.050 or 0.047 | 0.490 / 0.050 (all) / 0.047 (Franklin only) | means; medians 0.580 / 0.000; AUC 0.774; endmembers derived from this image |
| share of valid land pixels modelled | 85.4% | 85.4% | 322,304 of 377,590 pixels left by the nodata, cloud and water masks; 31.7% of the full grid |
| structures in frame / destroyed | 2,046 / 984 | 2,046 / 984 | CAL FIRE DINS, hero-figure frame |
| dNBR median destroyed / undamaged | 0.375 / 0.234 | 0.375 / 0.234 | in-frame points with dNBR, n=869/679 |
| Hughes mean NBR Jan -> Apr | 0.017 -> 0.192 | -0.072 -> 0.205 | all finite pixels in the animation frame |
| Hughes mean NBR Jan -> Apr, inside perimeter | - | -0.330 -> -0.012 | pixels inside WFIGS Hughes perimeter covered by both scenes, n=14877 of 46850 |
## Notes

- **dNBR outside the perimeter.** Both published values are correct, for different
  definitions. 0.060 is land outside every mapped perimeter; 0.046 is land outside
  the Palisades perimeter only, which counts the Franklin scar (median dNBR
  -0.007; it burned before the pre-fire scene) as unburned. The Kenneth perimeter
  lies outside the dNBR footprint and changes neither value.
- **Hughes recovery** does not reproduce. Use the inside-perimeter row, which has
  an external referent: the April scene covers 14,877 of the perimeter's 46,850
  pixels.
- **Wildland vs built-up.** The split has no committed definition. The direction
  holds at every threshold from 0 to 300 m (wildland median 0.43-0.50, AUC
  0.82-0.89; built-up median 0.000), so quote it with its threshold.
- **Franklin** values are means on the 2024-12-15 map, whose endmembers were
  averaged from that image's own low-NBR pixels. The perimeter is external, but
  the fraction largely restates the NBR threshold that defined the char endmember.
