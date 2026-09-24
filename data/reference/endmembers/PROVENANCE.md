# Endmember library provenance

`palisades_fire_library.csv` and `palisades_fire_library_urban.csv` are built by
`scripts/build_fire_endmember_library.py`. Every spectrum is a measured
laboratory/field spectrum from the **USGS Spectral Library Version 7**; nothing
in either file is derived from Tanager imagery, synthesised, or fitted.

- Source archive: `ASCIIdata_splib07a.zip`, 21,812,828 bytes,
  MD5 `bfe74068d85811e52e5e07d017720a17`, ScienceBase item
  `586e8c88e4b0f5ce109fccae` (child of the splib07 data release
  `5807a2a2e4b0841e59e3a18d`). The MD5 is checked before the archive is used.
- Citation: Kokaly, R.F., Clark, R.N., Swayze, G.A., Livo, K.E., Hoefen, T.M.,
  Pearson, N.C., Wise, R.A., Benzel, W.M., Lowers, H.A., Driscoll, R.L., and
  Klein, A.J., 2017, *USGS Spectral Library Version 7*: U.S. Geological Survey
  Data Series 1035, 61 p. DOI: [10.5066/F7RR1WDJ](https://doi.org/10.5066/F7RR1WDJ).
- Spectra are the ASD full-resolution measurements (2151 channels, 350–2500 nm),
  convolved to the Tanager-1 band centres with that scene's per-band FWHM via
  `spectral.BandResampler` (source FWHM 1 nm).
- `Record=` below is the splib07a record number printed in each ASCII file's
  header line, so any spectrum here can be traced back to the archive.

## Why an external library at all

`scripts/run_pipeline.py`'s `stage_mesma_image` builds its endmembers by
thresholding the scene: char is the mean spectrum of pixels with `NBR < -0.1`
and `NDVI < 0.2`. Reading the resulting char fraction as burn extent is
circular — the endmember is defined as "the pixels that look burned", so the
fraction cannot disagree with the threshold that produced it. A library with no
contact with the image removes that loop: the perimeter and DINS comparisons in
`scripts/validate_char_fractions.py` are then genuinely out-of-sample.

## Core library (`palisades_fire_library.csv`, 23 spectra)

| class | splib07a sample | Record | chapter |
|---|---|---|---|
| char | BurnArea_Traverse_WRF00-01_ASDFRb_AREF | 11918 | SoilsAndMixtures |
| char | BurnArea_TopSurface_WRF00-02_ASDFRb_AREF | 11928 | SoilsAndMixtures |
| pv | Chamise_CA01-ADFA-1_bush_1_ASDFRa_AREF | 20999 | Vegetation |
| pv | Chamise_CA01-ADFA-2_bush_2_ASDFRa_AREF | 21015 | Vegetation |
| pv | Buckbrush_CA01-CECU-1_bush_1_ASDFRa_AREF | 20938 | Vegetation |
| pv | Buckbrush_CA01-CECU-2_bush_2_ASDFRa_AREF | 20954 | Vegetation |
| pv | Buckbrush_CA01-CECU-3_bush_3_ASDFRa_AREF | 20970 | Vegetation |
| pv | Manzanita_CA01-ARVI-1_bush_1_ASDFRa_AREF | 21651 | Vegetation |
| pv | Manzanita_CA01-ARVI-4_bush_4_ASDFRa_AREF | 21699 | Vegetation |
| pv | Manzanita_CA01-ARVI-6_bush_6_ASDFRa_AREF | 21731 | Vegetation |
| pv | Oak_QUDU_CA01-QUDU-1_bush_1_ASDFRa_AREF | 21788 | Vegetation |
| pv | Oak_QUDU_CA01-QUDU-3_bush_3_ASDFRa_AREF | 21820 | Vegetation |
| pv | Toyon_CA01-HEAR-1_bush_ASDFRa_AREF | 22121 | Vegetation |
| pv | Yerba_Santa_CA01-ERCA-1_bush_ASDFRa_AREF | 22184 | Vegetation |
| pv | Gray-Pine_CA01-PISA-1_branch_ASDFRa_AREF | 21247 | Vegetation |
| npv | Grass_AETR70_CA01-AETR-2_NPV_ASDFRa_AREF | 21182 | Vegetation |
| npv | Grass_AETR95_CA01-AETR-1_NPV_ASDFRa_AREF | 21166 | Vegetation |
| npv | Grass_CA01-TACA-1_meadow_NPV_ASDFRa_AREF | 21198 | Vegetation |
| npv | Grass_Golden_Dry_GDS480_ASDFRa_AREF | 21213 | Vegetation |
| soil | Sand_GrndIsle1_no_oil_ASDFRa_AREF | 13249 | SoilsAndMixtures |
| soil | Sand_GrndIsle2_no_visibl_oil_ASDFRa_AREF | 13264 | SoilsAndMixtures |
| soil | Stonewall_Playa_Dry_Mud_2001_ASDFRa_AREF | 13304 | SoilsAndMixtures |
| soil | Pyroxene_Basalt_CU01-20A_ASDFRa_AREF | 13145 | SoilsAndMixtures |

### char — 2 spectra, and that is all there is

`BurnArea_Traverse` and `BurnArea_TopSurface` are the only measured
burned-surface spectra in splib07. Both are very dark in the visible and NIR
(R660 0.032 / 0.041, R860 0.040 / 0.059) and rise steadily into the SWIR
(R2200 0.156 / 0.207), giving a strongly negative NBR of −0.59 / −0.56 — the
canonical char signature, and the reason MESMA can find char without being told
where the fire was.

The per-sample descriptions live in the library's `HTMLmetadata.zip`, which is
1.35 GB and was not downloaded; the site and fuel type behind the `WRF00` sample
IDs are therefore **not** established here. Two spectra from one unverified
site is a thin basis for a class, and it is the single biggest weakness of this
library. It is not, however, the reason the product fails at structures — see
the "urban" section.

`Carbon_Black_GDS68` (Record 2320) was **rejected**. It is a manufactured
pigment, essentially flat at 0.015–0.019 across the whole range. As an
endmember it behaves like a second shade term: it can absorb any amount of
brightness variation without changing spectral shape, which lets MESMA fit dark
pixels of any composition and inflates char.

### pv — California chaparral, deliberately

Every pv spectrum is from splib07's `CA01` field campaign: chamise, buckbrush,
manzanita, oak, toyon, yerba santa and gray pine — Californian chaparral and
oak woodland, the fuel type that burned in the Santa Monica Mountains. Where a
species has many samples (manzanita has seven) the subset keeps the spread
rather than the near-duplicates: ARVI-4 is the outlier at NBR +0.46 and is kept
for that reason, while ARVI-1 (+0.66) and ARVI-6 (+0.63) stand in for the tight
+0.63…+0.70 cluster that ARVI-2, -3 and -5 also sit in.

### npv — cured grass and litter

`Grass_AETR70`, `Grass_AETR95` and `Grass_CA01-TACA-1_meadow` are CA01 campaign
spectra the library itself labels `NPV`; `Grass_Golden_Dry_GDS480` is the
library's generic cured-grass spectrum. Candidates that the filename heuristic
would have swept in were rejected on their numbers: `Sagebrush_Sage-Leaves-1_dry`
and `Willow_Willow-Leaves-1_dry` have NDVI +0.42 and +0.66 — still substantially
green, and as "NPV" they would have taken share from pv. The Louisiana marsh
NPV spectra (`P.australis`, `S.alterniflora`, `D.spicata`) are the wrong biome
and far brighter (R660 up to 0.63).

### soil — the weak class, stated plainly

**splib07 contains no California soil.** The `CA01` campaign sampled vegetation
only. The four spectra here are natural surfaces chosen to span soil brightness —
beach sand from the Grand Isle sampling (R660 0.17 and 0.27), playa dry mud
(very bright, R660 0.48) and a pyroxene basalt (R660 0.17) — but none of
them was measured on a Southern California soil. Char/soil confusion is the
failure mode this class most directly controls, so treat the soil fraction from
this library as the least trustworthy of the four.

## Urban extension (`palisades_fire_library_urban.csv`, +9 spectra)

Built with `--with-urban`. This library exists to answer one diagnostic
question and is **not** a shipping product.

| class | splib07a sample | Record | chapter |
|---|---|---|---|
| urban | Concrete_GDS375_Lt_Gry_Road_ASDFRa_AREF | 18568 | ArtificialMaterials |
| urban | Asphalt_GDS376_Blck_Road_old_ASDFRa_AREF | 18269 | ArtificialMaterials |
| urban | Asphalt_Shingle_GDS367_DkGry_ASDFRa_AREF | 18287 | ArtificialMaterials |
| urban | Asphalt_Shingle_GDS368_Lgray_ASDFRa_AREF | 18296 | ArtificialMaterials |
| urban | Brick_GDS350_Dk_Red_Building_ASDFRa_AREF | 18350 | ArtificialMaterials |
| urban | Sheet_Metal_GDS352_crg_Galvn_ASDFRa_AREF | 20078 | ArtificialMaterials |
| debris | WTC_Dust_Debris_WTC01-2_ASDFRa_AREF | 20790 | ArtificialMaterials |
| debris | WTC_Dust_Debris_WTC01-28_ASDFRa_AREF | 20801 | ArtificialMaterials |
| debris | Concrete_WTC01-37A_ASDFRa_AREF | 18578 | ArtificialMaterials |

The four-class library has nothing that can stand for a residential block, so
every roof, road and slab has to be absorbed by `soil`. The question is whether
that mis-assignment is what suppresses char at destroyed structures. Adding
intact built materials and a debris class answers it: it does not. Char at DINS
`Destroyed` points stays at median 0.000 (AUC 0.506 against `No Damage`) while
the damage signal moves onto `debris` (median 0.337 vs 0.000, AUC 0.672).

**`debris` is a stand-in and must be labelled as one.** The WTC samples are
pulverised building materials from a *collapse*, not from a
wildfire. They are the closest measured analogue to a destroyed-structure
surface in any public library, and they are not a measurement of wildfire ash.
No number derived from the `debris` fraction should be quoted as a
burned-structure measurement.
