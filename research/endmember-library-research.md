# Endmember Library Research for FireSpec MESMA

**Date:** 2026-04-27

---

## 1. Executive Summary

Three spectral libraries provide the foundation for FireSpec's MESMA endmember set: (1) the **FRAMES Burn Severity Spectral Library** with 66 field-collected spectra from Southern California chaparral fires (including 7 char/ash, 36 GV, 13 NPV, 10 soil), (2) the **USGS Spectral Library v7** (splib07) with 1000+ spectra across 7 chapters including fire-specific char/ash field measurements, California chaparral vegetation (chamise/Adenostoma, manzanita/Arctostaphylos), and heated soils/iron oxides, and (3) the **ECOSTRESS Spectral Library v1.0** with 3400+ spectra including 541 vegetation and 51 NPV VIS/SWIR spectra. The FRAMES library is the primary source for fire endmembers; USGS v7 and ECOSTRESS supplement with additional within-class variability. For resampling, SPy's `BandResampler` with Gaussian FWHM convolution handles the 1nm library-to-5nm Tanager conversion. For endmember selection, the recommended approach is **In-CoB** (compact libraries, models rare classes) combined with **EAR+MASA** pruning, implemented via the `spectral-libraries` v1.1.3 Python package. For MESMA execution, the `mesma` v1.0.8 package is primary but dormant since Nov 2020 — **pysptools FCLS** serves as fallback, and **HySUPP** as the modern alternative with 20+ unmixing methods.

---

## 2. USGS Spectral Library v7 — Available Endmembers

**Citation:** Kokaly, R.F., Clark, R.N., Swayze, G.A., et al. (2017). USGS Spectral Library Version 7. USGS Data Series 1035. DOI: 10.5066/F7RR1WDJ

**Organization:** Seven chapters — M (Minerals), S (Soils/rocks/mineral mixtures), C (Coatings), L (Liquids/volatiles), O (Organics/plant biochemicals), A (Artificial/man-made), V (Vegetation/biological).

**Wavelength coverage:** 0.2–200 µm (ultraviolet through far-infrared). The s07ASD version provides 1nm-sampled spectra from 350–2500nm (matching our VSWIR range exactly).

**Fire-relevant spectra identified:**

| Chapter | Material Type | Fire Application | Notes |
|---------|--------------|------------------|-------|
| O (Organics) | Charcoal | Char endmember | Field measurements of ash/char following wildland fire confirmed in v7 |
| S (Soils) | Heated soils, iron oxides, clays | Soil endmember, post-fire mineral exposure | Goethite, hematite, montmorillonite, kaolinite |
| V (Vegetation) | California chaparral species | GV endmember | Chamise (Adenostoma), manzanita (Arctostaphylos), plus leaf/plant-level spectra |
| V (Vegetation) | Dry litter, bark, stems | NPV endmember | Leaf/stem components at multiple phenological states |
| M (Minerals) | Calcite | Post-fire mineral soil exposure | Common in SoCal soils |

**Key chaparral vegetation:** The vegetation chapter explicitly includes California chaparral ecosystem spectra, with species listed by common name (40-character SPECPR limit) but cross-referenced to scientific names in metadata. Confirmed species: manzanita (Arctostaphylos viscida), plus general chaparral-ecosystem measurements.

**Data format:** SPECPR binary format with linked HTML metadata. ASCII text versions available. The `splib07-loader` Python package (pip-installable from GitHub) provides programmatic access.

**Download:** https://www.sciencebase.gov/catalog/item/5807a2a2e4b0841e59e3a18d (usgs_splib07.zip)

---

## 3. ECOSTRESS Spectral Library — Supplementary Spectra

**Citation:** Meerdink, S.K., Hook, S.J., Roberts, D.A., & Abbott, E.A. (2019). The ECOSTRESS spectral library version 1.0. Remote Sensing of Environment, 230, 111196.

**Contents:** 3400+ spectra of natural and man-made materials. Key additions in v1.0:
- **541 vegetation VIS/SWIR spectra** (massive increase from original 4 vegetation spectra in ASTER library)
- **472 vegetation TIR spectra**
- **51 NPV VIS/SWIR + TIR spectra** (non-photosynthetic vegetation — dry litter, bark, stems)
- Soils (lunar and terrestrial), minerals, rocks, water/snow/ice, man-made materials

**Wavelength coverage:** 0.35–15.4 µm (VIS through TIR)

**Fire-relevant strengths:**
- Large NPV spectral diversity (51 spectra) — critical for distinguishing dry pre-fire vegetation from char
- Vegetation species diversity supporting within-class endmember variability for MESMA
- SPy native integration via `EcostressDatabase` class (SQLite backend, queryable by name/type/class)

**Access:** https://speclib.jpl.nasa.gov/ — searchable web interface, individual spectrum downloads, bulk download available

**SPy integration code:**
```python
import spectral as spy
db = spy.EcostressDatabase("ecostress_lib.db")
db.print_query("SELECT * FROM Samples WHERE Type LIKE '%vegetation%'")
sig = db.get_signature(spectrum_id)
wavelengths, reflectance = sig.x, sig.y
```

---

## 4. Recommended Endmember Set for LA Fire MESMA

### Primary Source: FRAMES Burn Severity Spectral Library

**URL:** https://www.frames.gov/assessing-burn-severity/spectral-library/southern-california

The FRAMES Southern California library is the most directly relevant source for LA fire MESMA. It contains **66 field-collected spectra** from the Old Fire and Simi Fire (both SoCal chaparral fires), organized into four endmember classes:

| Class | Count | Key Samples |
|-------|-------|-------------|
| Green Vegetation (GV/PV) | 36 | Chamise (Adenostoma), manzanita (Arctostaphylos), ceanothus, California sagebrush, yerba santa, oak (Quercus), lupine |
| Non-Photosynthetic Vegetation (NPV) | 13 | Dead vegetation, pine needles, bark samples |
| Mineral Soil/Rock | 10 | Various soil types |
| Char/Ash | 7 | Charred chamise stem, charred coulter pinebark, charred log, charred rock, charred soil (2 variants from different fires) |

**Collection instrument:** ASD FieldSpec Pro FR (350–2500nm, continuous reflectance)
**Format:** ASCII text files (e.g., `chamise_old.txt`)

### Recommended Hybrid Library Strategy

Following Quintano et al. (2023) methodology:

1. **FRAMES SoCal** (primary) — 66 spectra, field-collected from chaparral fires
2. **Image-derived endmembers** from pre/post-fire Tanager scenes — capture scene-specific spectral variability
3. **USGS v7** (supplementary) — char/charcoal from Organics chapter, heated soils, additional chaparral vegetation
4. **ECOSTRESS** (supplementary) — NPV diversity (51 spectra), additional vegetation species
5. **Shade** — photometric shade (zero reflectance spectrum) per standard MESMA practice

### Target Library Size

| Class | Target Spectra | Sources |
|-------|---------------|---------|
| Char | 10–15 | FRAMES (7) + USGS Organics (charcoal) + image-derived |
| Ash (white) | 3–5 | FRAMES char/ash subset + image-derived |
| PV (green vegetation) | 20–30 | FRAMES (36, pruned via In-CoB) + ECOSTRESS chaparral |
| NPV (dry litter) | 10–15 | FRAMES (13) + ECOSTRESS NPV (subset) |
| Soil/Rock | 8–12 | FRAMES (10) + USGS Soils (heated soils) |
| Shade | 1 | Photometric zero reflectance |
| **Total** | **~52–78** | After In-CoB + EAR/MASA pruning |

---

## 5. Spectral Resampling to Tanager-1 Band Centers

### The Problem

Library spectra are measured at ~1nm sampling (ASD FieldSpec: 350–2500nm, 2151 channels). Tanager-1 has 426 bands at ~5nm spacing with ~5.5nm FWHM. Spectra must be resampled to match exactly — the `mesma` package requires band positions to be an exact match between image and library.

### Method: Gaussian FWHM Convolution via SPy BandResampler

SPy's `BandResampler` treats each destination band as having a Gaussian spectral response function. For each source-to-destination band pair, it computes an integral over the FWHM overlap region, assuming the source band value is constant across its FWHM and the destination band has a Gaussian response.

**Constructor:** `BandResampler(centers1, centers2, fwhm1=None, fwhm2=None)`
- `centers1`: source band centers (e.g., 1nm library wavelengths)
- `centers2`: destination band centers (Tanager 426 band positions)
- `fwhm1`: source FWHM (default: half-distance to adjacent bands, ~1nm for ASD)
- `fwhm2`: destination FWHM (~5.5nm for Tanager)

**Resampling code:**
```python
import numpy as np
import spectral as spy

# Tanager band centers and FWHM (from HDF5 metadata)
tanager_centers = np.linspace(380, 2500, 426)  # approximate; use actual from HDF5
tanager_fwhm = np.full(426, 5.5)  # approximate; use actual from HDF5

# Library spectrum (1nm sampling, 350-2500nm)
lib_centers = np.arange(350, 2501, 1)  # 2151 channels
lib_fwhm = np.ones(2151)  # 1nm FWHM for ASD data

# Create resampler
resampler = spy.BandResampler(lib_centers, tanager_centers, lib_fwhm, tanager_fwhm)

# Resample a single spectrum
resampled_spectrum = resampler(library_spectrum)

# Resample entire library (N spectra x 2151 bands -> N x 426 bands)
resampled_library = np.array([resampler(spec) for spec in library_spectra])
```

**Important:** Actual Tanager band centers and FWHM values must be extracted from HDF5 scene metadata (stored as dataset attributes), not assumed to be uniform.

### Alternative: SpectRes

The `spectres` Python package provides flux-conserving resampling onto arbitrary wavelength grids, which may be preferred when spectral integrated quantities matter.

---

## 6. Endmember Selection Methodology (IES / EAR / MASA / CoB)

### Method Descriptions

**EAR (Endmember Average RMSE):** For each endmember spectrum in a class, EAR computes the average RMSE when that spectrum is used as an endmember to model all other within-class spectra via SMA. Lower EAR = better representative endmember.

**MASA (Minimum Average Spectral Angle):** Similar to EAR but uses spectral angle distance instead of RMSE. Captures shape similarity independent of brightness. Lower MASA = more spectrally representative.

**CoB / In-CoB (Count-Based Selection):** Evaluates how many within-class spectra each candidate endmember can model within constraints. In-CoB selects endmembers with the highest unique count. When tied, selects by minimum EAR. Produces compact libraries.

**IES (Iterative Endmember Selection):** Starts with two endmembers maximizing Cohen's kappa for the full library. Iteratively adds/removes endmembers to maximize kappa. Produces the smallest library that maximizes classification accuracy.

### Comparison from Tane et al. (2018)

| Method | Library Size | Accuracy | Rare Class Handling | Recommended For |
|--------|-------------|----------|---------------------|-----------------|
| IES | Large (2x+ bigger) | Highest overall | Poor (fails rare species) | Large, diverse scenes |
| In-CoB | Compact | Good | Good (retains rare) | **Fire MESMA** (recommended) |
| EAR | Medium | Good | Moderate | Pruning supplement |
| MASA | Medium | Good | Moderate | Pruning supplement |

### Recommended Strategy for FireSpec

**Step 1 — In-CoB selection** per endmember class (Char, PV, NPV, Soil):
- Select all endmembers with unique In-CoB values
- For ties, select by minimum EAR

**Step 2 — EAR + MASA pruning:**
- Remove endmembers with both high EAR and high MASA
- Joint criterion prevents removing spectrally unique but important endmembers

**Step 3 — Validate with uSZU band selection:**
- Apply Stable Zone Unmixing band selection to identify the ~30–50 most diagnostic bands
- Re-evaluate endmembers on the reduced band set

**Software:** `spectral-libraries` v1.1.3 (Python package):
```python
from spectral_libraries import EarMasaCob

ear_masa_cob = EarMasaCob()
ear_masa_cob.execute(
    library=library_array,
    class_list=class_labels,
    constraints=(-0.05, 1.05, 0.025),  # min_frac, max_frac, max_RMSE
    reset=True
)
```

---

## 7. MESMA Software Evaluation (SPy vs mesma vs pysptools)

### Comparison Matrix

| Feature | mesma v1.0.8 | pysptools v0.15.0 | SPy v0.24 | HySUPP |
|---------|-------------|-------------------|-----------|--------|
| MESMA (variable endmembers) | Yes (core) | No | No | No |
| FCLS unmixing | No | Yes | No | Yes |
| NNLS unmixing | No | Yes | No | Yes |
| Endmember extraction (N-FINDR, PPI) | No | Yes | No | Yes |
| Band selection (SZU/uSZU) | Yes | No | No | No |
| EAR/MASA/CoB | Via spectral-libraries | No | No | No |
| Last updated | Nov 2020 | ~2021 | Active | 2024 |
| Python 3.10+ tested | Unknown | Unlikely | Yes | Yes |
| License | GPL v3 | Apache 2.0 | GPL | MIT |

### Verdicts

**mesma v1.0.8 — PRIMARY (with caution)**
- Only package implementing true MESMA (variable endmembers per pixel)
- Companion `spectral-libraries` v1.1.3 provides EAR/MASA/CoB
- Risk: dormant since Nov 2020, Python/numpy compatibility untested

**pysptools v0.15.0 — FALLBACK for fixed SMA**
- FCLS with sum-to-one and non-negativity constraints
- Useful as baseline comparison with fixed endmember set

**SPy v0.24 — SUPPORT ROLE**
- BandResampler, ECOSTRESS database access, SAM classification
- Actively maintained

**HySUPP — MODERN ALTERNATIVE**
- 20+ unmixing algorithms, actively maintained (2024)
- No true MESMA, but FCLS/NNLS fallback if mesma v1.0.8 fails

### Recommended Software Stack

```
spectral-libraries v1.1.3  →  Library management, EAR/MASA/CoB
mesma v1.0.8 (or pysptools FCLS fallback)  →  Unmixing
SPy v0.24  →  BandResampler, ECOSTRESS DB, SAM
HySUPP  →  Fallback unmixing if mesma fails at 426 bands
```

---

## 8. Open Questions and Next Steps

### Resolved by This Research

- **Q2 (endmember spectra):** FRAMES SoCal library (66 spectra from chaparral fires) as primary, supplemented by USGS v7 charcoal/soils and ECOSTRESS vegetation/NPV. Hybrid strategy with image-derived endmembers.
- **Q6 (MESMA software):** mesma v1.0.8 primary, pysptools FCLS fallback, HySUPP modern alternative. spectral-libraries v1.1.3 for endmember selection.

### New Open Questions

1. **mesma v1.0.8 Python 3.10+ compatibility** — needs empirical testing
2. **Tanager exact band centers** — must extract from HDF5 metadata for accurate resampling
3. **FRAMES library download format** — verify if bulk download is available
4. **Image-derived endmember extraction** — implement once data pipeline is operational
5. **Ash vs. char spectral separation** — only 7 combined spectra in FRAMES; may need supplementary measurements

### Next Steps

1. Download FRAMES SoCal ASCII spectra and USGS splib07 archive
2. Load ECOSTRESS database via SPy and query for chaparral/NPV spectra
3. Extract Tanager band centers from HDF5 scene metadata
4. Resample all library spectra to Tanager band positions via BandResampler
5. Test mesma v1.0.8 installation and 426-band compatibility
6. Run In-CoB + EAR/MASA selection on combined library
7. Generate initial MESMA fraction maps on Jan 23 post-fire scene

---

## References

- Kokaly, R.F. et al. (2017). USGS Spectral Library Version 7. USGS Data Series 1035. DOI: 10.5066/F7RR1WDJ
- Meerdink, S.K. et al. (2019). The ECOSTRESS spectral library version 1.0. RSE, 230, 111196.
- Roberts, D.A. et al. (1998). Mapping chaparral in the Santa Monica Mountains using MESMA. RSE, 65(3), 267–279.
- Tane, Z. et al. (2018). Evaluating Endmember and Band Selection for MESMA. Remote Sensing, 10(3), 389.
- FRAMES Burn Severity Spectral Library: https://www.frames.gov/assessing-burn-severity/spectral-library/southern-california
- USGS Spectral Library v7: https://www.sciencebase.gov/catalog/item/5807a2a2e4b0841e59e3a18d
- ECOSTRESS Spectral Library: https://speclib.jpl.nasa.gov/
- SPy BandResampler: https://www.spectralpython.net/algorithms.html
- mesma v1.0.8: https://pypi.org/project/mesma/
- spectral-libraries v1.1.3: https://pypi.org/project/spectral-libraries/
- HySUPP: https://github.com/BehnoodRasti/HySUPP
