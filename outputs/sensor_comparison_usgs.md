# Sensor comparison with the USGS endmember library

Generated 2026-10-01T03:33:12Z by scripts/sensor_comparison_usgs.py. Commit: 5ce3630

Scene: `20250123_185518_92_4001` (Palisades post-fire swath). Library: USGS
splib07a, `data/reference/endmembers/palisades_fire_library.csv` (23 spectra).
EMIT and PRISMA band centres are uniformly spaced across each sensor's range,
as in notebook 05; Sentinel-2 uses its published centres and bandwidths.

**What R² vs native measures.** Each score compares the char map from
spectrally degraded Tanager data with the char map from the full 426-band
data, same pixels, same library. It measures how much of the native
result survives losing bands, not whether either map is right; the
perimeter and DINS columns in section 3 are the accuracy test.

## 0. Products with no endmembers (128x128 crop, all bands)

NBR, NDVI and continuum-removed absorption depths, each scored against the
same product from native Tanager. Notebook 05 published means of 0.999 / 0.995 /
0.996 (broadband) and 0.965 / 0.904 / -0.623 (absorption) for EMIT / PRISMA /
Sentinel-2.

```
sensor      EMIT  PRISMA  Sentinel-2
product                             
CR 1200 nm 0.987   0.961      -0.351
CR 1700 nm 0.954   0.885      -0.378
CR 2100 nm 0.965   0.912      -0.710
CR 970 nm  0.954   0.858      -1.053
NBR        1.000   1.000       0.997
NDVI       0.998   0.991       0.994

mean by product type
sensor      EMIT  PRISMA  Sentinel-2
demand                              
absorption 0.965   0.904      -0.623
broadband  0.999   0.995       0.996
```

## 1. Control: notebook 05 reproduced (image-derived endmembers, 128x128 crop)

```
fraction    char    pv    npv  soil
sensor                             
EMIT       0.991 1.000  0.974 0.996
PRISMA     0.957 0.996  0.979 0.995
Sentinel-2 0.363 0.898 -0.651 0.611
```

Notebook 05 char R²: EMIT 0.991, PRISMA 0.957, Sentinel-2 0.361. Reproduced within 0.005.

Crop: 12539 valid pixels of 16384; 8829 inside the Palisades perimeter. Native USGS char on the crop: median 0.307, 28.7% of modeled pixels above 0.05.

## 2. Same crop, USGS library

All bands (direct analogue of notebook 05):

```
fraction     char    pv    npv   soil
sensor                               
EMIT        0.843 0.963  1.000  0.906
PRISMA      0.601 0.940 -3.283  0.863
Sentinel-2 -3.888 0.341 -0.016 -0.076
```

Validated band set (`MESMA_BANDS` in `scripts/palisades_mesma.py`; 22 bands, Sentinel-2 all 10):

```
fraction    char    pv   npv  soil
sensor                            
EMIT       0.983 0.966 0.928 0.949
PRISMA     0.983 0.973 0.924 0.942
Sentinel-2 0.648 0.756 0.137 0.155
```

## 3. Full Palisades swath, USGS library, validated band set

Native run vs `20250123swath2_frac_char.tif` (the map behind AUC 0.795): 322304 common pixels, max |diff| 0.00e+00, modeled 322304 vs 322304.

R² against native Tanager, all modeled pixels:

```
fraction    char    pv   npv  soil
sensor                            
EMIT       0.985 0.971 0.935 0.961
PRISMA     0.986 0.980 0.950 0.962
Sentinel-2 0.731 0.727 0.125 0.303
```

R² against native Tanager, inside the Palisades perimeter only:

```
fraction    char    pv   npv  soil
sensor                            
EMIT       0.982 0.968 0.909 0.945
PRISMA     0.980 0.975 0.897 0.932
Sentinel-2 0.675 0.802 0.134 0.214
```

Char against references outside the imagery (perimeter: inside Palisades vs
outside every perimeter in `la_fires_2025.geojson`; DINS: `Destroyed (>50%)`
vs `No Damage` points):

```
            modeled_px  inside_median  outside_median  perimeter_auc  dins_destroyed_median  dins_auc     n_dins
sensor                                                                                                          
Native          322304          0.389           0.000          0.795                  0.000     0.530  4114/3257
EMIT            322178          0.395           0.000          0.795                  0.000     0.530  4114/3260
PRISMA          322032          0.384           0.000          0.793                  0.000     0.531  4114/3252
Sentinel-2      321788          0.447           0.000          0.807                  0.000     0.612  4104/3239
```

## Verdict

**The char-fraction R² values move.** With the external library on the full
swath, EMIT holds (0.991 → 0.985), PRISMA rises to match it (0.957 → 0.986), and
Sentinel-2 roughly doubles (0.361 → 0.731). On the notebook's crop the same
library gives anything from -3.89 to 0.65 for Sentinel-2 depending only on which
bands MESMA sees, so no single char R² is a stable property of a sensor.

**Against references outside the imagery, the extra bands buy nothing for char
extent here.** The Sentinel-2-band char map separates the Palisades perimeter at
AUC 0.807, compared with 0.795 for native Tanager, and does slightly better on
the structure survey (0.612 vs 0.530). EMIT and PRISMA are indistinguishable from
native. These are simulated sensors: only band positions and widths change. Real
Sentinel-2 would also differ in pixel size (10-20 m), noise, and acquisition date.

**What still separates the sensors** is the endmember-free work in section 0. The
narrow absorption depths reproduce exactly (0.965 / 0.904 / -0.623) and collapse
for Sentinel-2, which has no band between 865 and 1610 nm.

**What R² vs native measures:** how closely a product made from fewer, wider
bands matches the same product made from all 426 Tanager bands, on the same
pixels. It measures what is lost relative to Tanager, not whether either map is
correct.
