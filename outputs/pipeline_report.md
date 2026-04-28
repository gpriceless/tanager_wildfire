# Tanager FireSpec Pipeline — End-to-End Run

Generated: 2026-04-28 12:45:16 PDT
Repo: /home/gprice/projects/tanager
Tanager version: 0.1.0

## Summary

- Stages OK: **15**
- Stages errored: **0**

## Per-scene results

### 20241215
- Source: `/home/gprice/projects/tanager/data/raw/fire/20241215_185916_33_4001_ortho_sr_hdf5.h5`
- CRS: `EPSG:32611`
- Shape: `(713, 791)`
- Bounds (xmin, ymin, xmax, ymax): `(329355.0, 3754425.0, 353055.0, 3775785.0)`
- Resolution (m): `(30.0, 30.0)`
- Bands: 426, pixels: 563983

  - **[OK] `load_scene`** — 4.5s
      loaded shape=(wavelength=426, y=713, x=791) crs=EPSG:32611 bounds=(329355.0, 3754425.0, 353055.0, 3775785.0)

  - **[OK] `apply_masks`** — 2.7s
      applied 3 masks; valid pixels 173100/563983 (30.7%)

  - **[OK] `spectral_indices`** — 1.3s
      nbr: n_finite=  173100 min=-0.570 max=+0.819 mean=+0.239 p50=+0.274; ndvi: n_finite=  173100 min=-0.216 max=+1.000 mean=+0.498 p50=+0.545; ndwi: n_finite=  173100 min=-0.807 max=+0.300 mean=-0.467 p50=-0.529
      - `outputs/20241215_nbr.tif`
      - `outputs/20241215_nbr.png`
      - `outputs/20241215_ndvi.tif`
      - `outputs/20241215_ndvi.png`
      - `outputs/20241215_ndwi.tif`
      - `outputs/20241215_ndwi.png`

  - **[OK] `lfmc_indices`** — 24.6s
      SAI970: n_finite=65536 mean=+0.011 p50=+0.000; SAI1200: n_finite=65536 mean=+0.014 p50=+0.000; SAI1660: n_finite=65536 mean=+0.000 p50=+0.000; NDWI_1240: n_finite=18366 mean=-0.101 p50=-0.116; NDWI_1640: n_finite=18366 mean=-0.063 p50=-0.055; NDWI_2130: n_finite=18366 mean=+0.089 p50=+0.125; WI: n_finite=18366 mean=+0.982 p50=+0.974; CR_depths_970nm: n_finite=18366 mean=+0.065 p50=+0.056; CR_depths_1200nm: n_finite=18366 mean=+0.089 p50=+0.076; CR_depths_1700nm: n_finite=18366 mean=+0.079 p50=+0.048; CR_depths_2100nm: n_finite=18366 mean=+0.164 p50=+0.148
      - `outputs/20241215_SAI970.tif`
      - `outputs/20241215_SAI1200.tif`
      - `outputs/20241215_SAI1660.tif`
      - `outputs/20241215_NDWI_1240.tif`
      - `outputs/20241215_NDWI_1640.tif`
      - `outputs/20241215_NDWI_2130.tif`
      - `outputs/20241215_WI.tif`
      - `outputs/20241215_CR_depths_970nm.tif`
      - `outputs/20241215_CR_depths_1200nm.tif`
      - `outputs/20241215_CR_depths_1700nm.tif`
      - `outputs/20241215_CR_depths_2100nm.tif`

  - **[OK] `mesma_image_endmembers`** — 9.7s
      engine=mesma regions={'char': 7871, 'pv': 101770, 'npv': 13763, 'soil': 2236}; frac_char: n_finite=94795 mean=+0.098; frac_pv: n_finite=94795 mean=+0.557; frac_npv: n_finite=94795 mean=+0.208; frac_soil: n_finite=94795 mean=+0.137; mesma_rmse: mean=0.0099 p50=0.0074
      - `outputs/20241215_frac_char.tif`
      - `outputs/20241215_frac_pv.tif`
      - `outputs/20241215_frac_npv.tif`
      - `outputs/20241215_frac_soil.tif`
      - `outputs/20241215_mesma_rmse.tif`

  - **[OK] `burn_severity`** — 17.5s
      trained RF (synthetic CBI) cv_r2=0.995 cv_rmse=0.039; predicted CBI mean=0.319
      - `outputs/20241215_cbi.tif`
      - `outputs/20241215_severity.tif`
      - `outputs/20241215_cbi.png`

### 20250123
- Source: `/home/gprice/projects/tanager/data/raw/fire/20250123_185507_64_4001_ortho_sr_hdf5.h5`
- CRS: `EPSG:32611`
- Shape: `(1047, 961)`
- Bounds (xmin, ymin, xmax, ymax): `(345105.0, 3805935.0, 373905.0, 3837315.0)`
- Resolution (m): `(30.0, 30.0)`
- Bands: 426, pixels: 1006167

  - **[OK] `load_scene`** — 7.1s
      loaded shape=(wavelength=426, y=1047, x=961) crs=EPSG:32611 bounds=(345105.0, 3805935.0, 373905.0, 3837315.0)

  - **[OK] `apply_masks`** — 4.7s
      applied 3 masks; valid pixels 678672/1006167 (67.5%)

  - **[OK] `spectral_indices`** — 1.1s
      nbr: n_finite=  678672 min=-0.850 max=+0.801 mean=+0.083 p50=+0.070; ndvi: n_finite=  678672 min=-0.196 max=+0.824 mean=+0.317 p50=+0.308; ndwi: n_finite=  678672 min=-0.713 max=+0.300 mean=-0.342 p50=-0.365
      - `outputs/20250123_nbr.tif`
      - `outputs/20250123_nbr.png`
      - `outputs/20250123_ndvi.tif`
      - `outputs/20250123_ndvi.png`
      - `outputs/20250123_ndwi.tif`
      - `outputs/20250123_ndwi.png`

  - **[OK] `lfmc_indices`** — 30.6s
      SAI970: n_finite=65536 mean=+0.016 p50=+0.015; SAI1200: n_finite=65536 mean=+0.043 p50=+0.045; SAI1660: n_finite=65536 mean=+0.000 p50=+0.000; NDWI_1240: n_finite=65100 mean=-0.153 p50=-0.158; NDWI_1640: n_finite=65100 mean=-0.151 p50=-0.155; NDWI_2130: n_finite=65100 mean=+0.041 p50=+0.065; WI: n_finite=65100 mean=+0.941 p50=+0.936; CR_depths_970nm: n_finite=65100 mean=+0.028 p50=+0.021; CR_depths_1200nm: n_finite=65100 mean=+0.063 p50=+0.061; CR_depths_1700nm: n_finite=65100 mean=+0.068 p50=+0.068; CR_depths_2100nm: n_finite=65100 mean=+0.191 p50=+0.217
      - `outputs/20250123_SAI970.tif`
      - `outputs/20250123_SAI1200.tif`
      - `outputs/20250123_SAI1660.tif`
      - `outputs/20250123_NDWI_1240.tif`
      - `outputs/20250123_NDWI_1640.tif`
      - `outputs/20250123_NDWI_2130.tif`
      - `outputs/20250123_WI.tif`
      - `outputs/20250123_CR_depths_970nm.tif`
      - `outputs/20250123_CR_depths_1200nm.tif`
      - `outputs/20250123_CR_depths_1700nm.tif`
      - `outputs/20250123_CR_depths_2100nm.tif`

### 20250407
- Source: `/home/gprice/projects/tanager/data/raw/fire/20250407_192235_24_4001_ortho_sr_hdf5.h5`
- CRS: `EPSG:32611`
- Shape: `(869, 1039)`
- Bounds (xmin, ymin, xmax, ymax): `(324315.0, 3808635.0, 355455.0, 3834675.0)`
- Resolution (m): `(30.0, 30.0)`
- Bands: 426, pixels: 902891

  - **[OK] `load_scene`** — 6.7s
      loaded shape=(wavelength=426, y=869, x=1039) crs=EPSG:32611 bounds=(324315.0, 3808635.0, 355455.0, 3834675.0)

  - **[OK] `apply_masks`** — 4.2s
      applied 3 masks; valid pixels 683948/902891 (75.8%)

  - **[OK] `spectral_indices`** — 1.1s
      nbr: n_finite=  683948 min=-0.593 max=+0.964 mean=+0.273 p50=+0.263; ndvi: n_finite=  683948 min=-0.213 max=+1.000 mean=+0.599 p50=+0.616; ndwi: n_finite=  683948 min=-0.945 max=+0.300 mean=-0.553 p50=-0.568
      - `outputs/20250407_nbr.tif`
      - `outputs/20250407_nbr.png`
      - `outputs/20250407_ndvi.tif`
      - `outputs/20250407_ndvi.png`
      - `outputs/20250407_ndwi.tif`
      - `outputs/20250407_ndwi.png`

  - **[OK] `lfmc_indices`** — 30.7s
      SAI970: n_finite=65536 mean=+0.076 p50=+0.076; SAI1200: n_finite=65536 mean=+0.070 p50=+0.067; SAI1660: n_finite=65536 mean=+0.000 p50=+0.000; NDWI_1240: n_finite=65226 mean=-0.052 p50=-0.056; NDWI_1640: n_finite=65226 mean=+0.039 p50=+0.020; NDWI_2130: n_finite=65226 mean=+0.275 p50=+0.259; WI: n_finite=65226 mean=+1.031 p50=+1.029; CR_depths_970nm: n_finite=65226 mean=+0.087 p50=+0.086; CR_depths_1200nm: n_finite=65226 mean=+0.100 p50=+0.096; CR_depths_1700nm: n_finite=65226 mean=+0.044 p50=+0.014; CR_depths_2100nm: n_finite=65226 mean=+0.225 p50=+0.224
      - `outputs/20250407_SAI970.tif`
      - `outputs/20250407_SAI1200.tif`
      - `outputs/20250407_SAI1660.tif`
      - `outputs/20250407_NDWI_1240.tif`
      - `outputs/20250407_NDWI_1640.tif`
      - `outputs/20250407_NDWI_2130.tif`
      - `outputs/20250407_WI.tif`
      - `outputs/20250407_CR_depths_970nm.tif`
      - `outputs/20250407_CR_depths_1200nm.tif`
      - `outputs/20250407_CR_depths_1700nm.tif`
      - `outputs/20250407_CR_depths_2100nm.tif`

## Multi-scene products

- **[OK] `dnbr`** — 0.3s
    shape=(869, 346) n_finite=137601 min=-1.384 max=+0.575 mean=-0.177 p50=-0.151
    - `outputs/20250123_to_20250407_dnbr.tif`
    - `outputs/20250123_to_20250407_dnbr.png`

## Known gaps observed during this run

- **No external endmember libraries installed.** ECOSTRESS SQLite is not present on this machine and the USGS loader is not yet implemented (LGT-330). MESMA was therefore exercised with image-derived endmembers extracted via NBR/NDVI heuristics — useful for plumbing validation but not for publishable severity products.
- **No CBI ground truth.** The burn-severity model was trained on a synthetic CBI proxy (`3 * char`) so the train/predict path is exercised end-to-end. Real CBI plots will replace this once available.
- **LFMC predict_lfmc not exercised.** A trained PLSR model artifact is required and none is checked in. Only `compute_lfmc_indices` was run; once a model is trained the predict_lfmc stage can be slotted in.
- **`compute_lfmc_indices` is per-pixel-Python slow.** The continuum-removal pass uses `xr.apply_ufunc(vectorize=True)`, which iterates over every pixel in Python. The full-scene run was killed after >4 minutes; this stage was exercised on a 256x256 center crop to keep the heartbeat bounded. A vectorized hull or chunked dask path is needed before full-scene LFMC products are practical.
- **`compute_lfmc_indices` requires `reflectance` variable name.** The function rejects a Dataset whose only reflectance variable is named `surface_reflectance` (the canonical Tanager ortho_sr layout). This script works around it by passing the bare DataArray, but the module should follow the `_REFLECTANCE_VARIABLE_PRIORITY` lookup used by `tanager.spectral` for consistency.
- **Three scenes, two locations.** 20241215 sits ~60 km south of 20250123/20250407, so no dNBR is computed against it. The 20250123 → 20250407 pair overlaps and is used for dNBR.