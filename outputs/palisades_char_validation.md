# Palisades char-fraction validation

Generated 2026-09-24T23:35:10Z by scripts/validate_char_fractions.py.
Commit: c73205e

Both comparisons are against references outside the imagery. The endmember
library is USGS splib07a; scripts/build_fire_endmember_library.py reads the
scene only for its band centres and FWHM (instrument configuration) and
deletes it before building any spectrum, so no image content informs the
endmembers and the perimeter comparison is out-of-sample.

## Core library — outputs/20250123swath2_frac_char.tif
```
20250123swath2_frac_char.tif: 322304 / 1017291 pixels modeled (31.7%)

--- PALISADES perimeter (official interagency) ---
  inside perimeter       n=  65659 median=0.389 mean=0.334 p90=0.699
  outside all perimeters n= 243140 median=0.000 mean=0.023 p90=0.000
  separation             median diff = +0.389   AUC = 0.795

--- CAL FIRE DINS structure damage ---
  destroyed (>50%)       n=   4114 median=0.000 mean=0.017 p90=0.000
  no damage              n=   3257 median=0.000 mean=0.004 p90=0.000
  separation             median diff = +0.000   AUC = 0.530
```

## Urban-extended library — outputs/20250123swath2_urbanlib_frac_char.tif
```
20250123swath2_urbanlib_frac_char.tif: 324314 / 1017291 pixels modeled (31.9%)

--- PALISADES perimeter (official interagency) ---
  inside perimeter       n=  65835 median=0.350 mean=0.315 p90=0.712
  outside all perimeters n= 244844 median=0.000 mean=0.021 p90=0.000
  separation             median diff = +0.350   AUC = 0.757

--- CAL FIRE DINS structure damage ---
  destroyed (>50%)       n=   4128 median=0.000 mean=0.010 p90=0.000
  no damage              n=   3262 median=0.000 mean=0.004 p90=0.000
  separation             median diff = +0.000   AUC = 0.506
```

## Char inside each perimeter in the scene

The library was given no information about where anything burned.

```
perimeter           n   median     mean
Palisades       65659    0.389    0.334
Franklin        11107    0.337    0.310
Kenneth          2539    0.575    0.500
```

## Verdict

Resolves wildland burn: AUC 0.795 against the official Palisades perimeter,
and the same library independently places char inside all three fire
perimeters present in the scene.

Does not resolve structure damage: AUC 0.530 against the CAL FIRE DINS
survey, near chance. A 30 m pixel is wider than a house. The core library
outperforms the urban-extended one on both comparisons (0.795/0.530 vs
0.757/0.506), so the core library is the one in use.
