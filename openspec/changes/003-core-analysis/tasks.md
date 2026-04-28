# Change: 003-core-analysis

## Open Questions — EM REVIEW COMPLETE

1. **mesma v1.0.8 Python/numpy compatibility:** Last release Nov 2020, 66 weekly downloads. RISK: HIGH.
   Must verify in Wave 1 Section 1 before building pipeline. If `pip install mesma>=1.0.8` fails on
   Python 3.10+ / numpy 2.x, pivot to HySUPP FCLS immediately. The fallback is already specced
   (scenario: "MESMA fallback to HySUPP"). Coder task 2 in Section 1 is the gating test.
   <!-- eng-note: mesma uses numpy C extensions compiled for older numpy ABI. numpy 2.x changed ABI.
        If install succeeds but runtime fails (segfault, dtype errors), that counts as failure too.
        Test with: `python -c "import mesma; print(mesma.__version__)"` AND a small unmixing run. -->

2. **FRAMES SoCal library acquisition:** FRAMES is a USDA Forest Service database. The 66 SoCal
   chaparral spectra (Old Fire + Simi Fire) may be downloadable as bulk ASCII archive or may require
   per-spectrum scraping from https://www.frames.gov/. The `load_frames_library()` function should
   handle both: directory of ASCII files already downloaded, OR a fetch utility.
   <!-- eng-note: for competition purposes, manual download + local directory is acceptable.
        The loader only needs to parse ASCII files from a local dir. Do NOT build a web scraper.
        Document the manual acquisition steps in a README or docstring. -->

3. **Globe-LFMC 2.0 SoCal coverage:** Globe-LFMC 2.0 has 287,551 observations globally. Need to
   verify SoCal chaparral subset is >= 50 observations for meaningful PLSR training. If sparse,
   expand to all California chaparral sites (still within the same vegetation type).
   <!-- eng-note: Globe-LFMC 2.0 is available via DOI 10.1038/s41597-024-03159-6. Data is CSV/Parquet.
        The loader should accept a bbox filter. If SoCal yields < 50 obs, log a warning and suggest
        expanding the bbox. Not blocking — the loader handles any bbox, and training code handles
        any sample size >= 10 (with appropriate warnings). -->

4. **AVIRIS-3 Eaton Fire ORNL DAAC access:** ORNL DAAC hosts AVIRIS-3 data. Need to confirm the
   Eaton Fire overflights (Jan 10-16, 2025) are publicly available and determine file format (ENVI
   binary? NetCDF?). The `load_aviris3_reference()` function must handle whatever format ORNL uses.
   <!-- eng-note: this is a Wave 4 dependency. Does NOT block Waves 1-3. The validation module
        can be built with a placeholder loader that raises NotImplementedError if data isn't available.
        Real data integration happens during Phase 4 (notebooks/submission). -->

5. **Shade endmember construction:** Use single zero-reflectance photometric shade spectrum.
   This is standard practice (Roberts et al. 2018, Quintano et al. 2023). If MESMA fit quality is
   poor (high RMSE on shadow-heavy pixels), add 1-2 partial shade spectra (e.g., 50% illumination).
   <!-- eng-note: RESOLVED for implementation. Start with zero-reflectance shade. The endmember
        library builder should create it as np.zeros(n_bands). If QA testing reveals poor shadow
        fits, adding partial shade is a Phase 3 follow-up, not a blocker. -->

---

## Wave 1: Endmember Library Foundation
<!-- execution: sequential -->

### Section 1: Dependencies & Compatibility Check
<!-- execution_mode: sequential -->
<!-- network: REQUIRED — pip install fetches packages from PyPI and GitHub -->

- [x] Add new dependencies to `pyproject.toml`: mesma>=1.0.8, spectral-libraries>=1.1.3, splib07-loader (git+https://github.com/RobertSparworthy/splib07-loader.git), joblib
  <!-- files: pyproject.toml (modify) -->
  <!-- gotcha: splib07-loader is not on PyPI — must use git+https URL in dependencies list.
       Format: "splib07-loader @ git+https://github.com/.../splib07-loader.git"
       joblib is likely already a transitive dep of scikit-learn but pin it explicitly for
       model serialization in severity.py and lfmc.py. -->
  <!-- gotcha: do NOT add mesma to hard dependencies if it fails the compat check (task 2).
       Instead, make it an optional dependency: [project.optional-dependencies] mesma = ["mesma>=1.0.8"]
       and handle ImportError at runtime. -->
  <!-- test: pip install -e ".[dev]" succeeds -->
  <!-- acceptance: pyproject.toml has all 4 new deps; pip install succeeds without errors -->

- [x] Verify mesma v1.0.8 installs successfully with current Python/numpy versions; document result
  <!-- files: (none — verification only, document in Plane comment) -->
  <!-- gotcha: run BOTH import test AND functional test:
       1. `pip install mesma>=1.0.8 && python -c "import mesma; print(mesma.__version__)"`
       2. Create a tiny 3-endmember, 10-band, 5x5 pixel synthetic test and run mesma.
       If either fails, immediately: (a) move mesma to optional deps, (b) document the failure
       mode, (c) the HySUPP fallback becomes primary in unmixing.py.
       DO NOT spend time debugging mesma internals — the fallback path is the mitigation. -->
  <!-- risk: HIGH — this is the critical path decision point for the entire unmixing pipeline -->
  <!-- acceptance: documented result (PASS or FAIL) with specific Python/numpy versions tested -->

- [x] Verify spectral-libraries v1.1.3 installs and `EarMasaCob` class is importable
  <!-- files: (none — verification only) -->
  <!-- gotcha: `from spectral_libraries import EarMasaCob` — note underscore in package name vs
       hyphen in PyPI name. Verify the class exists and can be instantiated. -->
  <!-- acceptance: import succeeds; EarMasaCob class is callable -->

- [x] If mesma fails to install: document failure, flag HySUPP as primary, and adjust downstream tasks
  <!-- files: pyproject.toml (modify — move mesma to optional deps), Plane comment -->
  <!-- dep: blocked_by task 2 result -->
  <!-- gotcha: if mesma fails, the coder must: (1) move mesma to optional deps in pyproject.toml,
       (2) ensure unmixing.py (Wave 2) treats HySUPP as primary (not fallback),
       (3) update the import detection logic in unmixing.py to prefer HySUPP.
       The spec already has the "MESMA fallback to HySUPP" scenario — just flip the priority. -->
  <!-- acceptance: if triggered, pyproject.toml updated and downstream tasks adjusted -->

### Section 2: Library Loaders
<!-- execution_mode: sequential -->
<!-- network: none for coding — loaders read local files. Verify tasks may need downloaded data. -->

- [x] Create `src/tanager/endmembers.py` with module docstring, logging setup, and type annotations
  <!-- files: src/tanager/endmembers.py (new) -->
  <!-- pattern: follow src/tanager/spectral.py for module structure — __future__ annotations import,
       logging.getLogger(__name__), type imports from typing. Module docstring should list public API. -->
  <!-- gotcha: import direction — endmembers.py MAY import from config.py and spectral.py (for
       continuum_removal). endmembers.py MUST NOT import from io.py or masks.py.
       Other modules (unmixing.py, severity.py) will import FROM endmembers.py. -->
  <!-- acceptance: module imports cleanly with `from tanager import endmembers` -->

- [ ] Implement `load_usgs_library(categories, data_dir)` parsing USGS v7 ASCII files; return xarray DataArray (spectrum_id, wavelength) with metadata attrs
  <!-- files: src/tanager/endmembers.py (modify) -->
  <!-- pattern: return xr.DataArray with dims=(spectrum_id, wavelength), coords={wavelength: nm values},
       attrs={name: str, category: str, source: "usgs_v7"} per spectrum. Use spectrum_id as a
       string coordinate (e.g., "usgs_char_001"). -->
  <!-- gotcha: *** splib07-loader is INCOMPATIBLE (nptyping pins numpy<2.0, breaks rasterio/rioxarray).
       DO NOT attempt to install or use splib07-loader. ***
       Instead: parse USGS v7 ASCII text files directly. Data source: ASCIIdata_splib07a.zip
       (20.8 MB from ScienceBase DOI:10.5066/F7RR1WDJ). Files with s07ASD prefix = ASD
       spectrometer, 2151 channels, 350-2500nm at ~1nm resolution.
       ASCII format: tab-delimited, wavelength (micrometers) + reflectance columns.
       Convert wavelengths from micrometers to nm (* 1000).
       categories parameter should filter by material type: "char", "soil", "vegetation", etc.
       Category assignment from filename/header parsing or a manual mapping dict.
       Exclude spectra with all-NaN in VSWIR range (380-2500nm) per spec. -->
  <!-- gotcha: USGS v7 spectra are at 1nm resolution (ASD measurements). The returned DataArray
       should be at source resolution — resampling to Tanager bands is a separate function. -->
  <!-- dep: requires USGS v7 ASCII data files downloaded to data_dir -->
  <!-- test: tests/test_endmembers.py — mock file I/O, verify output schema -->
  <!-- acceptance: returns DataArray with correct dims, spectra filtered by category, NaN spectra excluded -->

- [x] Implement `load_ecostress_library(categories)` using SPy `EcostressDatabase`; return same schema as USGS loader
  <!-- files: src/tanager/endmembers.py (modify) -->
  <!-- pattern: `from spectral import EcostressDatabase; db = EcostressDatabase()`
       SPy auto-downloads the SQLite database on first call. Filter to VSWIR (0.35-2.5 um = 350-2500nm).
       Exclude TIR spectra (wavelengths > 2500nm). -->
  <!-- gotcha: SPy EcostressDatabase returns spectral.Spectrum objects. Extract wavelength and
       reflectance arrays via spectrum.params.wavelength and spectrum.data.
       ECOSTRESS wavelengths may be in micrometers — convert to nm (* 1000).
       Categories mapping: "vegetation" = plant spectra, "npv" = dry vegetation/litter,
       "mineral" = soils and rocks. SPy uses different category names internally — check SPy docs. -->
  <!-- test: tests/test_endmembers.py — mock SPy EcostressDatabase -->
  <!-- acceptance: same DataArray schema as USGS loader; only VSWIR spectra included -->

- [x] Implement `load_frames_library(data_dir)` parsing ASCII spectral files; categorize as char/ash/pv/npv/soil; return same schema
  <!-- files: src/tanager/endmembers.py (modify) -->
  <!-- gotcha: FRAMES ASCII format is likely tab-delimited or CSV with wavelength + reflectance columns.
       The coder should check actual file format from a sample file. Categorization may need to be
       based on filename patterns or a metadata file in the directory.
       If no metadata file exists, the function should accept an optional `category_map` dict
       mapping filename patterns to categories. -->
  <!-- gotcha: FRAMES SoCal library has 66 spectra: 7 char/ash, 36 GV (=pv), 13 NPV, 10 soil.
       Metadata should include: species name, collection site, fire name (per spec). -->
  <!-- risk: MEDIUM — depends on FRAMES data format which we haven't verified. If ASCII parsing
       fails, this is not blocking — USGS + ECOSTRESS still provide a viable library. -->
  <!-- test: tests/test_endmembers.py — mock file I/O with synthetic ASCII data -->
  <!-- acceptance: returns DataArray with char/ash/pv/npv/soil categories; metadata attributes present -->

- [x] Implement `resample_library(library, target_wavelengths, fwhm=5.5)` using SPy `BandResampler` with Gaussian convolution; clip output to [0, 1]
  <!-- files: src/tanager/endmembers.py (modify) -->
  <!-- pattern: `from spectral import BandResampler; resampler = BandResampler(source_centers, target_centers, fwhm1=source_fwhm, fwhm2=target_fwhm)`
       source_centers = library wavelength coordinate (1nm for USGS/ECOSTRESS).
       target_centers = target_wavelengths parameter (Tanager band centers from scene metadata).
       fwhm parameter is for the TARGET sensor (Tanager FWHM = 5.5nm per SENSOR config). -->
  <!-- gotcha: SPy BandResampler expects numpy arrays for wavelengths. Extract from xarray coords.
       Source FWHM for lab spectra (USGS, ECOSTRESS) is ~1nm (ASD spectrometer).
       After resampling: np.clip(output, 0, 1) — Gaussian convolution can produce slight negatives
       or values > 1 at boundaries. -->
  <!-- gotcha: target_wavelengths should default to np.linspace(380, 2500, 426) if not extracted
       from a specific scene. In practice, use actual band centers from scene metadata via
       dataset.coords["wavelength"].values. -->
  <!-- gotcha: PHASE 2 FINDING — actual per-band FWHM varies 5.20-6.81nm (not constant 5.5nm).
       load_ortho_scene() now stores per-band FWHM as dataset.coords["fwhm"]. The fwhm parameter
       should accept EITHER a scalar (backward compat) OR a numpy array of per-band values.
       When a scene Dataset is available, prefer: dataset.coords["fwhm"].values for accuracy.
       This matters for sharp absorption features where FWHM variation is significant. -->
  <!-- test: tests/test_endmembers.py — verify output has target_wavelengths length, values in [0, 1] -->
  <!-- acceptance: output DataArray has 426 wavelengths; reflectance clipped to [0, 1]; dims preserved; accepts per-band FWHM array -->

- [x] Implement `build_hybrid_library(usgs, ecostress, frames, image_derived)` to merge multiple sources into a single library DataArray with source tracking
  <!-- files: src/tanager/endmembers.py (modify) -->
  <!-- pattern: concatenate along spectrum_id dimension. Add a "source" attribute to each spectrum
       ("usgs_v7", "ecostress", "frames", "image"). All inputs must have same wavelength grid
       (i.e., already resampled to Tanager bands). Validate wavelength alignment before merging. -->
  <!-- gotcha: any of the inputs may be None (e.g., FRAMES data not available). The function
       should gracefully handle None inputs by skipping them. At least one input must be non-None. -->
  <!-- test: tests/test_endmembers.py — merge two synthetic libraries, verify source tracking -->
  <!-- acceptance: merged DataArray with source attribute; wavelength alignment enforced; None inputs handled -->

- [x] Verify: Load USGS library, resample to Tanager bands, confirm output has 426 wavelengths and reflectance in [0, 1]
  <!-- verify: requires USGS v7 ASCII data files in data_dir. Can use mocked data for unit test, real data for integration test. -->
  <!-- network: ASCII data must be pre-downloaded from ScienceBase (DOI:10.5066/F7RR1WDJ) -->
  <!-- acceptance: resampled output has exactly 426 bands; all values in [0, 1] -->

### Section 3: Endmember Selection
<!-- execution_mode: sequential -->
<!-- network: none — all operations on in-memory DataArrays -->

- [x] Implement `select_endmembers_incob(library, max_per_class)` using count-based selection within each class
  <!-- files: src/tanager/endmembers.py (modify) -->
  <!-- gotcha: In-CoB (count-based) selection requires running each candidate endmember against
       image pixels to count how many pixels it "wins" (best-fit model). This means the function
       needs access to image data OR pre-computed model counts.
       For the initial implementation, a simplified version is acceptable: rank endmembers within
       each class by spectral variability (std across wavelengths) and select top max_per_class.
       The full In-CoB requires MESMA runs, which creates a circular dependency with unmixing.py.
       Resolution: implement simplified version first, add full In-CoB as enhancement after
       unmixing.py exists. Document this in the docstring. -->
  <!-- pattern: group spectra by "category" attribute, select top max_per_class per group. -->
  <!-- test: tests/test_endmembers.py — verify selection reduces count, respects max_per_class -->
  <!-- acceptance: output has <= max_per_class per class; class labels preserved -->

- [x] Implement `prune_endmembers_ear_masa(library, threshold_ear, threshold_masa)` wrapping spectral-libraries EarMasaCob
  <!-- files: src/tanager/endmembers.py (modify) -->
  <!-- pattern: *** CORRECTED IMPORT PATH (QA verified) ***
       `from spectral_libraries.core.ear_masa_cob import EarMasaCob` — NOT from top-level package.
       Top-level `spectral_libraries.__init__` is a QGIS plugin classFactory and does NOT export EarMasaCob.
       Public method: `EarMasaCob.execute()`. Requires numpy arrays, not xarray.
       Extract .values and class labels from the DataArray before passing. -->
  <!-- gotcha: EarMasaCob computes EAR (Endmember Average RMSE) and MASA (Minimum Average Spectral
       Angle) for each endmember. Endmembers exceeding BOTH thresholds are removed.
       Default thresholds from Roberts et al. (2018): threshold_ear=0.025, threshold_masa=10.0 (degrees).
       The spec says pruned library should have 50-80 spectra total — log a warning if outside range. -->
  <!-- dep: requires spectral-libraries installed (Section 1 task 3) -->
  <!-- test: tests/test_endmembers.py — verify pruning reduces count -->
  <!-- acceptance: pruned library has fewer spectra; both EAR and MASA thresholds applied -->

- [x] Implement `extract_image_endmembers(scene, method, regions)` for spatial ROI extraction and PPI
  <!-- files: src/tanager/endmembers.py (modify) -->
  <!-- pattern: method="spatial" — extract mean spectrum from each region (dict of {class_name: (y_slice, x_slice)}).
       method="ppi" — use SPy's PPI algorithm: `from spectral import PPI; ppi = PPI(image, n_iterations)`.
       SPy PPI expects a numpy array of shape (rows, cols, bands) — note the axis order differs from
       our xarray convention (wavelength, y, x). Transpose: scene["reflectance"].values.transpose(1, 2, 0). -->
  <!-- gotcha: PPI (Pixel Purity Index) is computationally expensive. Default n_iterations to 1000
       for testing, 10000 for production. Log progress. -->
  <!-- gotcha: output must match library DataArray schema (spectrum_id, wavelength) with metadata.
       For spatial extraction, spectrum_id = class_name. For PPI, spectrum_id = "ppi_001", "ppi_002", etc. -->
  <!-- test: tests/test_endmembers.py — test spatial extraction on synthetic dataset with known signatures -->
  <!-- acceptance: extracted spectra match library schema; spatial method returns mean of ROI pixels -->

- [x] Implement `build_fire_library(scene_pre, scene_post, frames_dir, usgs_dir)` — convenience function that orchestrates the full library build pipeline: load -> resample -> merge -> select -> prune -> return final library (~52-78 spectra)
  <!-- files: src/tanager/endmembers.py (modify) -->
  <!-- pattern: orchestration function that calls the individual functions in sequence:
       1. load_usgs_library(usgs_dir) -> resample_library(target_wl)
       2. load_ecostress_library() -> resample_library(target_wl)
       3. load_frames_library(frames_dir) -> resample_library(target_wl) [if frames_dir provided]
       4. extract_image_endmembers(scene_pre, method="spatial") [pre-fire veg endmembers]
       5. extract_image_endmembers(scene_post, method="spatial") [post-fire char endmembers]
       6. build_hybrid_library(all sources)
       7. select_endmembers_incob(max_per_class=15)
       8. prune_endmembers_ear_masa()
       9. Add shade endmember (zero-reflectance spectrum)
       Return final library. Log each step. -->
  <!-- gotcha: scene_pre and scene_post should be xarray Datasets from load_scene().
       target_wavelengths extracted from scene_pre.coords["wavelength"].values.
       frames_dir and usgs_dir can be None — skip those sources if not available. -->
  <!-- gotcha: shade endmember = np.zeros(n_bands) with category="shade", source="synthetic".
       IMPORTANT for mesma: shade spectrum must be shaped (bands, 1) when passed to MesmaCore.execute().
       In the library DataArray it's stored as a normal spectrum; reshape at call site.
       Per open question 5, single zero-reflectance is sufficient. -->
  <!-- dep: all previous endmember functions must exist -->
  <!-- test: tests/test_endmembers.py — test with mocked loaders, verify output size 50-80 -->
  <!-- acceptance: returns library with 50-80 spectra; all fire-relevant classes represented; shade included -->

- [x] Verify: Full pipeline produces a library with 50-80 spectra across all fire-relevant classes
  <!-- verify: integration test with real or synthetic data -->
  <!-- acceptance: library has spectra in classes: char, ash, pv, npv, soil, shade -->

### Section 1-3 Gate
<!-- gate: qa-review -->
Verify: endmembers module imports cleanly, library loading works with test data, resampling produces correct output dimensions, selection reduces library size to target range. All new functions have type hints and Google-style docstrings.

---

## Wave 2: MESMA Spectral Unmixing
<!-- execution: sequential -->
<!-- dep: Wave 1 must complete (endmember library required for unmixing) -->

### Section 4: Band Selection & MESMA Core
<!-- execution_mode: sequential -->
<!-- network: none — all operations on in-memory data -->

- [x] Create `src/tanager/unmixing.py` with module docstring, logging, and engine detection (mesma vs HySUPP)
  <!-- files: src/tanager/unmixing.py (new) -->
  <!-- pattern: follow src/tanager/spectral.py module structure. Add engine detection at module level:
       ```python
       _MESMA_AVAILABLE = False
       try:
           import mesma
           _MESMA_AVAILABLE = True
       except ImportError:
           logger.info("mesma package not available; will use HySUPP FCLS fallback")
       ```
       This pattern allows runtime fallback per spec scenario "MESMA fallback to HySUPP". -->
  <!-- gotcha: import direction — unmixing.py MAY import from config.py, spectral.py, endmembers.py.
       unmixing.py MUST NOT import from severity.py, lfmc.py, or validation.py. -->
  <!-- acceptance: module imports cleanly; engine detection works for both mesma-present and mesma-absent cases -->

- [x] Implement `select_bands_uszu(scene, library, n_bands=40)` for Uniform SZU band selection to identify most discriminatory bands
  <!-- files: src/tanager/unmixing.py (modify) -->
  <!-- gotcha: uSZU (Uniform Spectral Zone Unmixing) selects bands that maximize class separability.
       The algorithm: (1) divide spectrum into n_bands zones, (2) within each zone, select the band
       with highest between-class variance relative to within-class variance.
       If spectral-libraries provides a uSZU implementation, use it. Otherwise implement from
       Roberts et al. (2018) description. -->
  <!-- gotcha: input scene should already have bad bands masked (use mask_bad_bands() output).
       The function should work on ~330 bands and select the top n_bands from those. -->
  <!-- pattern: return a band-subset xarray Dataset with reduced wavelength dimension.
       Also return the selected wavelength indices for use with the endmember library. -->
  <!-- test: tests/test_unmixing.py — verify output has exactly n_bands wavelengths -->
  <!-- acceptance: output Dataset has n_bands wavelengths; selected bands span VNIR-SWIR range -->

- [x] Implement `run_mesma(scene, library, constraints, bands)` wrapping mesma v1.0.8 API; handle pixel-by-pixel unmixing with constraint filtering
  <!-- files: src/tanager/unmixing.py (modify) -->
  <!-- gotcha: mesma v1.0.8 API — CONFIRMED WORKING (Python 3.12.3 + numpy 2.4.4).
       Full workflow:
       1. `from mesma.core.mesma import MesmaCore, MesmaModels`
       2. `models = MesmaModels(); models.setup(class_list)` — class_list = list of str per endmember
       3. `look_up_table = models.return_look_up_table()` — auto-selects 2-EM and 3-EM combos
       4. `core = MesmaCore(n_cores=N)`
       5. `result = core.execute(image, library, look_up_table, em_per_class, constraints, fusion_value, shade_spectrum)`
       6. Returns tuple: (best_model, best_fractions, best_rmse, residuals_or_None)
       Image shape: (bands, rows, cols) or (bands, n_pixels) — BANDS FIRST.
       Library shape: (bands, n_endmembers) — column per spectrum.
       Shade spectrum: MUST be shape (bands, 1) — passing (bands,) breaks _subtract_shade broadcasting.
       best_fractions shape: (n_classes+1, ny, nx); shade is LAST class; class order is SORTED ALPHABETICAL.
       n_cores>1 uses threading (multiprocessing.dummy.Pool) — GIL-bound, limited speedup. -->
  <!-- gotcha: if bands parameter is provided, subset both scene and library to those bands
       before running MESMA. Use select_bands() for consistent wavelength matching. -->
  <!-- gotcha: constraints is a TUPLE, not dict. Format:
       (min_frac, max_frac, min_shade, max_shade, max_rmse, residual_flag, residual_flag)
       Default per spec: (-0.05, 1.05, 0.0, 0.8, 0.025, -9999, -9999).
       The public API should accept a user-friendly dict and convert internally. -->
  <!-- pattern: output xarray Dataset with variables: char, pv, npv, soil, shade, rmse.
       Dims: (y, x). Fractions sum to 1.0 within tolerance of 0.01.
       Pixels where no valid model found: all fractions = NaN, rmse = NaN.
       Add metadata attribute: unmixing_engine="mesma" or "hysup". -->
  <!-- risk: HIGH — mesma's 426-band performance is untested. If it's too slow (>30 min per scene),
       consider: (1) running on uSZU-selected bands only (~40 bands), (2) spatial subsetting for testing. -->
  <!-- dep: endmembers.py library must exist; mesma compatibility verified (Section 1) -->
  <!-- test: tests/test_unmixing.py — synthetic scene with known pure pixels, verify fractions -->
  <!-- acceptance: fractions sum to 1.0; known-pure pixels have fraction ~1.0 for matching endmember; RMSE stored -->

- [x] Implement `_run_mesma_hysup_fallback(scene, library, constraints)` using HySUPP FCLS as fallback; same output schema
  <!-- files: src/tanager/unmixing.py (modify) -->
  <!-- gotcha: HySUPP implements Fully Constrained Least Squares (FCLS). It's not multi-endmember
       like MESMA — it uses ALL endmembers simultaneously (one model per pixel, not combinatorial).
       This means fraction output will always have values for all classes, not NaN for "no valid model".
       RMSE should still be computed: ||observed - reconstructed||.
       Output schema must match run_mesma exactly: Dataset with char, pv, npv, soil, shade, rmse. -->
  <!-- gotcha: HySUPP may need: `pip install hysup` or clone from GitHub. Check availability.
       If HySUPP is also unavailable, implement a simple NNLS (non-negative least squares) using
       scipy.optimize.nnls as the ultimate fallback. -->
  <!-- pattern: set metadata attribute unmixing_engine="hysup" or "nnls" to track which backend was used. -->
  <!-- test: tests/test_unmixing.py — test fallback produces valid output when mesma unavailable -->
  <!-- acceptance: same output schema as run_mesma; unmixing_engine metadata set correctly -->

- [x] Implement RMSE constraint filtering within `run_mesma`: max_rmse, min_fraction, max_fraction; reject invalid models and mark pixels NaN
  <!-- files: src/tanager/unmixing.py (modify) -->
  <!-- pattern: post-processing step applied to MESMA output.
       For each pixel: if rmse > max_rmse, set all fractions to NaN.
       For each pixel: if any fraction < min_fraction or > max_fraction, reject model (NaN).
       For multi-endmember MESMA: this filtering happens during model selection (best valid model
       with lowest RMSE). For FCLS fallback: apply as post-filter. -->
  <!-- test: tests/test_unmixing.py — verify constraint filtering rejects bad fits -->
  <!-- acceptance: pixels with RMSE > threshold have NaN fractions; fractions outside bounds rejected -->

- [x] Implement `normalize_fractions(fractions, remove_shade=True)` — remove shade, rescale remaining to sum=1.0
  <!-- files: src/tanager/unmixing.py (modify) -->
  <!-- pattern: if remove_shade: drop the "shade" variable from Dataset, then rescale remaining
       fractions so each pixel sums to 1.0. Rescaling: new_frac = old_frac / (1 - shade_frac).
       Handle edge case: shade_frac == 1.0 (fully shaded pixel) → set all fractions to NaN.
       Output variables: char, pv, npv, soil (no shade). -->
  <!-- test: tests/test_unmixing.py — verify shade removed, remaining sum to 1.0 -->
  <!-- acceptance: no shade variable in output; non-shade fractions sum to 1.0 per pixel -->

- [x] Output format: xarray Dataset with variables (char, pv, npv, soil, shade, rmse) and dims (y, x)
  <!-- files: src/tanager/unmixing.py (modify — ensure output format is consistent across all paths) -->
  <!-- pattern: this is a format validation task, not a new function. Ensure run_mesma and
       _run_mesma_hysup_fallback both produce identically-structured output.
       Add a _validate_fraction_output(ds) internal function that checks:
       (1) all expected variables present, (2) dims are (y, x), (3) fractions in [0, 1] or NaN,
       (4) fractions sum to ~1.0 where not NaN. -->
  <!-- test: tests/test_unmixing.py — validate output schema from both backends -->
  <!-- acceptance: both backends produce identical output schema; validation function catches malformed output -->

- [x] Verify: Run MESMA on synthetic scene with known endmembers; confirm fractions sum to 1.0 and known-pure pixels map correctly
  <!-- verify: use synthetic_tanager_dataset_with_signatures fixture from conftest.py.
       Create a small endmember library from the known signatures (vegetation, char, soil).
       Run unmixing. Verify: vegetation pixels → high pv fraction, char pixels → high char fraction, etc. -->
  <!-- acceptance: pure vegetation pixels have pv > 0.8; pure char pixels have char > 0.8; fractions sum to 1.0 -->

### Section 5: Fraction Map Visualization
<!-- execution_mode: sequential -->
<!-- network: none -->

- [x] Implement `plot_fraction_maps(fractions, figsize, cmap)` — matplotlib multi-panel figure showing each fraction as a spatial map
  <!-- files: src/tanager/unmixing.py (modify) -->
  <!-- pattern: use matplotlib.pyplot.subplots(1, n_fractions) to create a row of panels.
       Each panel: imshow of one fraction variable (char, pv, npv, soil).
       Default cmap: "viridis" for continuous fractions. Add colorbars. Title each panel.
       Return the Figure object for Jupyter display. -->
  <!-- gotcha: import matplotlib only inside the function (heavy dep, not needed for headless runs).
       Handle the case where fractions have shade variable (before normalization) or not (after). -->
  <!-- test: tests/test_unmixing.py — verify function returns matplotlib Figure without error -->
  <!-- acceptance: produces Figure with correct number of panels; no errors on test data -->

- [x] Implement `plot_rgb_composite(fractions, r="char", g="pv", b="npv")` — false-color composite from fraction maps
  <!-- files: src/tanager/unmixing.py (modify) -->
  <!-- pattern: stack three fraction arrays as RGB channels. Normalize each to [0, 1] for display.
       Use matplotlib imshow with the stacked array. Add title and legend explaining the color mapping. -->
  <!-- test: tests/test_unmixing.py — verify function returns matplotlib Figure -->
  <!-- acceptance: produces RGB composite Figure; colors correspond to fraction assignments -->

- [x] Verify: Plotting functions produce figures without errors on test data
  <!-- verify: run on synthetic unmixing output -->
  <!-- acceptance: no exceptions; figures render correctly -->

### Section 4-5 Gate
<!-- gate: qa-review -->
Verify: unmixing module works end-to-end on synthetic data, fraction maps are physically reasonable (sum to 1, values in [0,1] after normalization), visualization produces clean outputs. Engine detection and fallback work correctly. All functions have type hints and Google-style docstrings.

---

## Wave 3: Severity & LFMC Products
<!-- execution: parallel -->
<!-- PARALLEL SAFETY CHECK
  Track A files: src/tanager/severity.py (new)
  Track B files: src/tanager/lfmc.py (new)
  Overlap: NONE — both are new files, no shared modifications to existing files
  Verdict: SAFE for parallel (2 tracks, completely file-disjoint)
-->
<!-- dep: Wave 2 must complete (MESMA fractions needed for severity; spectral indices exist for LFMC) -->

### Track A: Burn Severity Mapping
<!-- execution_mode: sequential (within track) -->
<!-- network: none for coding -->

- [x] Create `src/tanager/severity.py` with module docstring and logging
  <!-- files: src/tanager/severity.py (new) -->
  <!-- pattern: follow src/tanager/spectral.py module structure. Import scikit-learn at function level
       (heavy dep). Module docstring should reference Quintano et al. (2023) methodology. -->
  <!-- gotcha: import direction — severity.py MAY import from config.py, spectral.py (for dnbr),
       unmixing.py (for run_mesma in trajectories). severity.py MUST NOT import from lfmc.py or
       validation.py. -->
  <!-- acceptance: module imports cleanly -->

- [x] Implement `train_severity_model(fractions, ground_truth_cbi, method="random_forest")` using scikit-learn RF; return model + metrics (R², RMSE from 5-fold CV)
  <!-- files: src/tanager/severity.py (modify) -->
  <!-- pattern: `from sklearn.ensemble import RandomForestRegressor; from sklearn.model_selection import cross_val_score`
       Features: char, pv, npv, soil fractions (4 features per pixel).
       Target: CBI values (continuous, 0-3 scale).
       Default hyperparams per spec: n_estimators=200, max_depth=None, random_state=42.
       5-fold CV for metrics: R² and RMSE. Return dict with keys: model, r2, rmse. -->
  <!-- gotcha: fractions input is an xarray Dataset with (y, x) dims. Must flatten to 2D array
       for sklearn: shape (n_pixels, 4). ground_truth_cbi is a 1D array matching the pixels.
       Handle NaN pixels: exclude from training (fractions may have NaN where MESMA failed). -->
  <!-- gotcha: ground_truth_cbi must be provided by the user (from USGS BARC or field data).
       This function does NOT load ground truth — that's validation.py's job.
       For synthetic testing, generate CBI as a linear function of char fraction. -->
  <!-- test: tests/test_severity.py — train on synthetic fraction/CBI pairs, verify R² > 0 -->
  <!-- acceptance: returns trained model + R²/RMSE metrics; handles NaN input gracefully -->

- [x] Implement `predict_severity(fractions, model)` — apply trained model to produce continuous CBI map + classified severity map (5 classes using BARC thresholds)
  <!-- files: src/tanager/severity.py (modify) -->
  <!-- pattern: flatten fractions to (n_pixels, 4), predict with model, reshape to (y, x).
       Classification thresholds per spec:
       Unburned: CBI < 0.1 (code 0)
       Low: 0.1 <= CBI < 1.0 (code 1)
       Moderate-Low: 1.0 <= CBI < 1.5 (code 2)
       Moderate-High: 1.5 <= CBI < 2.25 (code 3)
       High: CBI >= 2.25 (code 4)
       Return dict with keys: cbi_map (DataArray, continuous), severity_map (DataArray, integer codes). -->
  <!-- gotcha: predicted CBI should be clipped to [0, 3] range (RF can extrapolate outside training range).
       NaN pixels in input fractions → NaN in output maps. -->
  <!-- test: tests/test_severity.py — verify prediction range [0, 3]; classification boundaries correct -->
  <!-- acceptance: CBI values in [0, 3]; severity classes match BARC thresholds; NaN pixels preserved -->

- [x] Implement `compute_trajectories(scenes_dict, library)` — run MESMA on multiple dates, return time-series Dataset with dims (time, y, x)
  <!-- files: src/tanager/severity.py (modify) -->
  <!-- pattern: scenes_dict = {datetime_str: xr.Dataset, ...}. For each scene, run unmixing
       (import run_mesma from tanager.unmixing), collect fraction Datasets.
       Stack along new "time" dimension. Output: xr.Dataset with dims (time, y, x) and
       variables for each fraction class. Time coordinate from scene datetime metadata. -->
  <!-- gotcha: all scenes must be unmixed with the SAME endmember library for comparability.
       The library parameter ensures this. Scenes may have different spatial extents — use
       tanager.io.reproject_to_common_grid(scenes) to align all scenes to a shared UTM grid
       before unmixing. This function was added during Phase 2 remediation and handles overlap
       detection, reprojection, and coordinate alignment. It raises ValueError if scenes have
       < 10% overlap (too little common area). -->
  <!-- gotcha: this function can be VERY slow (MESMA on N scenes). Log progress per scene.
       For the Dec 2024 → Jul 2025 timeline, expect 5-7 scenes. -->
  <!-- dep: requires unmixing.py run_mesma to exist (Wave 2) -->
  <!-- test: tests/test_severity.py — test with 2 synthetic scenes, verify time dimension -->
  <!-- acceptance: output has time, y, x dims; fraction variables present; time coordinate is datetime -->

- [x] Implement `compare_severity_methods(mesma_severity, dnbr_map)` — correlation, RMSE, bias, difference map between MESMA-derived and dNBR-derived severity
  <!-- files: src/tanager/severity.py (modify) -->
  <!-- pattern: mesma_severity and dnbr_map are both DataArrays with (y, x) dims.
       Compute: Pearson correlation (np.corrcoef), RMSE, bias (mean difference), difference map.
       Return dict with keys: correlation, rmse, bias, difference_map (DataArray). -->
  <!-- gotcha: use spectral.dnbr() to get the dNBR baseline. dnbr() now has auto_align=True
       (Phase 2 fix) which calls reproject_to_common_grid automatically when scenes differ.
       The comparison should demonstrate MESMA's improvement. If MESMA R² > dNBR R², that's
       a strong competition result. -->
  <!-- gotcha: PHASE 2 FINDING — pre-fire (Dec 15) and primary post-fire (Jan 23) scenes have
       minimal spatial overlap (different swath areas). The Jan 23 second swath
       (20250123_185518_92_4001) may overlap better. Verify overlap with
       reproject_to_common_grid before computing dNBR or severity comparison. -->
  <!-- test: tests/test_severity.py — test with synthetic data, verify metric ranges -->
  <!-- acceptance: returns correlation, RMSE, bias, difference map; metrics are physically reasonable -->

- [x] Verify: Train model on synthetic fraction/CBI pairs; predict produces values in [0, 3]; classification boundaries are correct
  <!-- verify: use synthetic data from conftest.py + generated CBI values -->
  <!-- acceptance: all severity functions produce correct output ranges and types -->

### Track B: LFMC Estimation
<!-- execution_mode: sequential (within track) -->
<!-- network: none for coding; Globe-LFMC loading may need downloaded data -->

- [x] Create `src/tanager/lfmc.py` with module docstring and logging
  <!-- files: src/tanager/lfmc.py (new) -->
  <!-- pattern: follow src/tanager/spectral.py module structure. Module docstring should reference
       Peterson & Roberts (2014) for PLSR approach and Quan et al. (2021) for SAI indices. -->
  <!-- gotcha: import direction — lfmc.py MAY import from config.py, spectral.py (for continuum_removal,
       select_bands, _normalized_difference). lfmc.py MUST NOT import from severity.py, unmixing.py,
       endmembers.py, or validation.py. -->
  <!-- acceptance: module imports cleanly -->

- [x] Implement `_compute_sai(reflectance, target_wl, left_shoulder, right_shoulder)` — core SAI computation for a single absorption feature
  <!-- files: src/tanager/lfmc.py (modify) -->
  <!-- pattern: SAI (Spectral Absorption Index) per spec scenario:
       1. Identify absorption feature minimum near target_wl
       2. Identify left and right shoulder wavelengths (local maxima flanking the feature)
       3. Compute straight-line continuum between shoulders
       4. SAI = (continuum_at_target - reflectance_at_target) / continuum_at_target
       Output in [0, 1]. If no clear absorption feature detected, return 0.0. -->
  <!-- gotcha: this is a per-pixel operation. The function should accept a 1D reflectance array
       (single pixel spectrum) and wavelength array. Vectorize over (y, x) using apply_ufunc
       or numpy operations in the calling function. -->
  <!-- gotcha: left_shoulder and right_shoulder are approximate wavelengths. Use nearest-neighbor
       band matching (Tanager 5nm grid). The "local maxima" search should be within a window
       around the target — e.g., search for left shoulder in [target-150nm, target-20nm]. -->
  <!-- test: tests/test_lfmc.py — test SAI on synthetic spectrum with known absorption feature -->
  <!-- acceptance: SAI in [0, 1]; returns 0.0 for flat spectrum; correct value for known absorption -->

- [x] Implement `compute_lfmc_indices(scene)` — compute all 8 water-sensitive indices (SAI970, SAI1200, SAI1660, NDWI_1240, NDWI_1640, NDWI_2130, WI, CR_depths); return xarray Dataset
  <!-- files: src/tanager/lfmc.py (modify) -->
  <!-- pattern: output xr.Dataset with dims (y, x) and variables:
       SAI970, SAI1200, SAI1660 — computed via _compute_sai()
       NDWI_1240 = (R860 - R1240) / (R860 + R1240)
       NDWI_1640 = (R860 - R1640) / (R860 + R1640)
       NDWI_2130 = (R860 - R2130) / (R860 + R2130)
       WI = R900 / R970
       CR_depths = continuum removal band depths at 970, 1200, 1700, 2100 nm -->
  <!-- gotcha: use spectral._normalized_difference() for NDWI variants, or compute inline.
       Band lookup via .sel(wavelength=target, method="nearest").
       CR_depths: use spectral.continuum_removal() then extract depth at target wavelengths.
       Depth = 1 - CR_value (continuum removal gives ratio, depth is the complement). -->
  <!-- gotcha: per spec, all wavelength lookups use nearest-neighbor matching (Tanager 5nm grid).
       Import from tanager.spectral where possible to reuse existing band math. -->
  <!-- gotcha: PHASE 2 FINDING — real Tanager reflectance has 13% negative values (ISOFIT shadow
       artifacts) and 0.09% values > 1.0 (calibration artifacts). Before computing indices,
       clamp reflectance to [0, 1] or use the epsilon-guarded _normalized_difference from
       spectral.py (LGT-311 added epsilon guard for near-zero denominators). SAI computation
       on unclamped reflectance will produce nonsense absorption depths. -->
  <!-- gotcha: this function should be fast (< 60 seconds per scene per spec NFR).
       The SAI computation is the bottleneck — vectorize aggressively. -->
  <!-- test: tests/test_lfmc.py — compute indices on synthetic dataset, verify shapes and ranges -->
  <!-- acceptance: 8 index variables in output; shapes match input spatial dims; NDWI in [-1, 1] -->

- [x] Implement `load_globe_lfmc(region_bbox, vegetation_types)` — load Globe-LFMC 2.0 observations; return GeoDataFrame with location, date, lfmc_percent, species
  <!-- files: src/tanager/lfmc.py (modify) -->
  <!-- gotcha: Globe-LFMC 2.0 data format — likely CSV or Parquet. Download from DOI.
       The function should accept a local file path OR download URL.
       region_bbox = (west, south, east, north) in WGS84 degrees.
       Filter by: bbox (spatial), vegetation_types (list of strings like ["chaparral"]).
       Return GeoDataFrame with columns: longitude, latitude, date, lfmc_percent, species, site_name.
       Add a "tanager_colocated" column: True if observation date is within +-30 days of any
       Tanager scene date (from FIRE_SCENES config). -->
  <!-- gotcha: Globe-LFMC vegetation type names may not match our categories exactly.
       Provide a mapping dict or use case-insensitive substring matching. -->
  <!-- risk: MEDIUM — depends on Globe-LFMC data availability and format. If data format is
       unexpected, the loader may need adjustment. Not blocking — the PLSR function works
       with any GeoDataFrame that has the required columns. -->
  <!-- dep: requires geopandas (already a dependency) -->
  <!-- test: tests/test_lfmc.py — test with mocked CSV data, verify output schema -->
  <!-- acceptance: returns GeoDataFrame with required columns; bbox filtering works; tanager_colocated flag set -->

- [x] Implement `train_lfmc_plsr(spectra, lfmc_values, n_components=10)` using scikit-learn PLSRegression; return model + R² + RMSE + VIP scores
  <!-- files: src/tanager/lfmc.py (modify) -->
  <!-- pattern: `from sklearn.cross_decomposition import PLSRegression`
       Input spectra: 2D array (n_samples, n_bands) — full ~330-band reflectance (bad bands excluded).
       Target: lfmc_values in percent (30-200%).
       Use cross-validation to find optimal n_components (1 to max_components, select by min RMSE).
       VIP (Variable Importance in Projection) scores: computed from PLSR model weights.
       Return dict: model, r2, rmse, n_components_optimal, vip_scores (array of length n_bands). -->
  <!-- gotcha: VIP score computation:
       VIP_j = sqrt(p * sum_h(SS_h * w_jh^2) / sum_h(SS_h))
       where p = number of predictors, h = component, SS_h = explained variance by component h,
       w_jh = weight of variable j in component h.
       Implement as a helper function _compute_vip(model, X, Y). -->
  <!-- gotcha: n_components cannot exceed min(n_samples, n_features). Add a check. -->
  <!-- test: tests/test_lfmc.py — train on synthetic spectra with moisture signal, R² > 0.5 -->
  <!-- acceptance: returns trained model + metrics; VIP scores highlight water absorption bands (970, 1200, 1660 nm) -->

- [x] Implement `predict_lfmc(scene, model, method)` — per-pixel LFMC estimate + uncertainty DataArray; flag pixels with LFMC < 60%
  <!-- files: src/tanager/lfmc.py (modify) -->
  <!-- pattern: flatten scene reflectance to (n_pixels, n_bands), predict with model.
       Reshape to (y, x) DataArray. Clip to physical range [0, 300] (LFMC can exceed 200% for
       some species but > 300% is unphysical).
       Uncertainty: use bootstrap prediction intervals OR model prediction variance.
       For PLSR, a simple approach: train multiple models on bootstrap samples, compute std of predictions.
       Flag: add a "low_lfmc" boolean DataArray where LFMC < 60% (nonlinear regime per Roberts 2006). -->
  <!-- gotcha: method parameter allows future expansion (e.g., method="indices" for index-based
       estimation). For now, method="plsr" is the primary path. -->
  <!-- gotcha: per spec, uncertainty estimates are prediction intervals, not just point estimates.
       Return as a separate DataArray with same dims (y, x). -->
  <!-- test: tests/test_lfmc.py — predict on synthetic data, verify output range and flag -->
  <!-- acceptance: LFMC values in [0, 300]; uncertainty DataArray present; low_lfmc flag at < 60% -->

- [x] Verify: Train PLSR on synthetic spectra with known moisture signal; R² > 0.5 on synthetic data; VIP scores highlight water absorption bands
  <!-- verify: create synthetic spectra where reflectance at 970, 1200, 1660 nm correlates with LFMC value -->
  <!-- acceptance: R² > 0.5; VIP scores highest near water absorption bands -->

### Wave 3 Gate
<!-- gate: qa-review -->
Verify: severity module produces classified maps with correct class boundaries; LFMC indices are physically reasonable for known signatures; both modules handle edge cases (all-NaN pixels, empty regions) gracefully. All functions have type hints and Google-style docstrings. Parallel track outputs do not conflict.

---

## Wave 4: Validation & Integration
<!-- execution: sequential -->
<!-- dep: Waves 1-3 must complete (all analysis modules needed for integration testing) -->

### Section 8: Validation Module
<!-- execution_mode: sequential -->
<!-- network: may need downloaded AVIRIS-3 and BARC data for verify steps -->

- [x] Create `src/tanager/validation.py` with module docstring
  <!-- files: src/tanager/validation.py (new) -->
  <!-- pattern: follow src/tanager/spectral.py module structure. This is a pure computation module
       with data loading helpers. No side effects. -->
  <!-- gotcha: import direction — validation.py MAY import from any tanager module (it sits at the
       top of the dependency tree). No other tanager module should import FROM validation.py. -->
  <!-- acceptance: module imports cleanly -->

- [x] Implement `load_aviris3_reference(filepath, target_resolution=30)` — load and spatially aggregate AVIRIS-3 fractions to 30m
  <!-- files: src/tanager/validation.py (modify) -->
  <!-- gotcha: AVIRIS-3 native resolution is 3-4m. Aggregation to 30m means ~8x8 or ~10x10 pixel
       averaging. Use rasterio or xarray coarsen for spatial averaging.
       File format: likely ENVI binary or GeoTIFF from ORNL DAAC. Use rasterio.open() for generic
       raster I/O. If format is NetCDF, use xarray.open_dataset(). -->
  <!-- gotcha: spatial alignment to Tanager grid: use rasterio.warp.reproject or xarray interp
       with nearest-neighbor resampling. The output must match Tanager's spatial coordinate system. -->
  <!-- gotcha: output must use same variable names as MESMA output (char, pv, npv, soil, shade)
       for direct comparison. AVIRIS-3 may use different names — map them. -->
  <!-- risk: MEDIUM — depends on AVIRIS-3 data availability (open question 4). If data is not
       available, the function should raise FileNotFoundError with a helpful message. -->
  <!-- test: tests/test_validation.py — test with synthetic raster data, verify aggregation math -->
  <!-- acceptance: output resolution matches target; spatial alignment correct; variable names match MESMA output -->

- [x] Implement `load_barc_reference(filepath)` — load USGS BARC classified severity maps, align to Tanager grid
  <!-- files: src/tanager/validation.py (modify) -->
  <!-- gotcha: BARC maps are classified rasters (GeoTIFF) with integer severity codes.
       Load via rasterio. Reproject/align to Tanager grid. Output: xr.DataArray with integer codes
       matching our severity classification: 0=Unburned, 1=Low, 2=Mod-Low, 3=Mod-High, 4=High.
       BARC may use different code values — include a mapping dict. -->
  <!-- test: tests/test_validation.py — test with synthetic classified raster -->
  <!-- acceptance: output is integer-coded DataArray aligned to Tanager grid -->

- [x] Implement `compute_accuracy(predicted, observed, metric_type)` — R², RMSE, MAE, bias for continuous; accuracy, Kappa, confusion matrix, F1 for classified
  <!-- files: src/tanager/validation.py (modify) -->
  <!-- pattern: metric_type="continuous": compute R², RMSE, MAE, bias, Spearman correlation.
       metric_type="classified": compute overall accuracy, Cohen's Kappa, confusion matrix, per-class F1.
       Use sklearn.metrics where available: r2_score, mean_squared_error, mean_absolute_error,
       cohen_kappa_score, confusion_matrix, f1_score.
       Return dict of metric names to values. -->
  <!-- gotcha: handle NaN values — exclude NaN pairs from both predicted and observed before
       computing metrics. Log the number of valid pairs used. -->
  <!-- test: tests/test_validation.py — test with known inputs (perfect prediction → R²=1.0, RMSE=0) -->
  <!-- acceptance: correct metrics for known inputs; NaN handling; both metric_type paths work -->

- [x] Implement `compare_sensors(tanager_result, reference_result, sensor_name)` — comparative metrics and improvement ratios for competition tie-breaker
  <!-- files: src/tanager/validation.py (modify) -->
  <!-- pattern: compute accuracy metrics for both Tanager and reference sensor results against
       the same ground truth. Compute improvement ratios: R²_improvement = tanager_R² - reference_R²,
       RMSE_reduction = (reference_RMSE - tanager_RMSE) / reference_RMSE * 100.
       Return a structured dict with: tanager_metrics, reference_metrics, improvement_ratios.
       Also generate a comparison table (pandas DataFrame) suitable for competition submission. -->
  <!-- gotcha: this is for the +5 tie-breaker (Tanager vs EMIT/PRISMA comparison).
       The sensor_name parameter identifies the reference sensor for the table output. -->
  <!-- test: tests/test_validation.py — test with synthetic data, verify improvement ratio math -->
  <!-- acceptance: returns comparison dict with both metric sets + improvement ratios; table format correct -->

- [x] Verify: Accuracy metrics produce correct values on known inputs (perfect prediction gives R²=1.0, zero RMSE)
  <!-- verify: unit test with exact known values -->
  <!-- acceptance: R²=1.0 for perfect prediction; RMSE=0; Kappa=1.0 for perfect classification -->

### Section 9: Test Suite
<!-- execution_mode: sequential -->
<!-- dep: all 5 new modules must exist before writing tests that reference their functions -->
<!-- network: none — all tests use synthetic data or mocked I/O -->

- [x] Create `tests/test_endmembers.py` — test library loading (with mocked file I/O), resampling dimensions, selection produces target library size, EAR/MASA pruning reduces count
  <!-- files: tests/test_endmembers.py (new) -->
  <!-- pattern: follow tests/test_catalog.py for mock patterns (unittest.mock.patch).
       Create synthetic library DataArrays with known spectra for testing.
       Test cases: load_usgs_library returns correct schema, resample_library produces 426 bands,
       select_endmembers_incob respects max_per_class, prune reduces count, build_hybrid merges. -->
  <!-- gotcha: mock USGS ASCII file I/O and SPy EcostressDatabase — do not require real spectral
       library data in unit tests. Create synthetic spectra matching the expected DataArray schema.
       Note: splib07-loader is NOT used (incompatible with numpy 2.x). USGS loader parses ASCII files directly. -->
  <!-- pattern: use conftest.py fixtures where applicable. Add new fixtures for endmember-specific
       test data (synthetic library with known categories). -->
  <!-- acceptance: all loader, resampler, selector, and pruner functions tested; mocks used for I/O -->

- [ ] Expand `tests/test_unmixing.py` — test MESMA on synthetic pure pixels (expect fraction=1.0 for matching endmember), constraint filtering rejects bad fits, shade normalization sums to 1.0
  <!-- files: tests/test_unmixing.py (modify — FILE ALREADY EXISTS with 19 tests from Wave 2 QA) -->
  <!-- pattern: use synthetic_tanager_dataset_with_signatures fixture. Create a small endmember
       library from the known signatures. Run unmixing. Verify fractions.
       Test constraint filtering: create a result with known bad RMSE, verify rejection.
       Test shade normalization: input with shade=0.3, verify output sums to 1.0. -->
  <!-- gotcha: if mesma is not installed in CI, the test should skip mesma-specific tests and
       only test the HySUPP/NNLS fallback path. Use pytest.importorskip("mesma") or check
       _MESMA_AVAILABLE flag. -->
  <!-- acceptance: pure-pixel fractions tested; constraint filtering tested; both backends covered -->

- [x] Create `tests/test_severity.py` — test RF training on synthetic data, prediction value ranges, classification thresholds, trajectory output shape
  <!-- files: tests/test_severity.py (new) -->
  <!-- pattern: generate synthetic fractions + CBI values (CBI = 2.5 * char_fraction + noise).
       Train model, verify R² > 0 on training data.
       Predict and verify: CBI in [0, 3], severity classes match BARC thresholds.
       Trajectory: create 2-scene dict, verify output has time dimension. -->
  <!-- acceptance: training produces valid model; prediction ranges correct; classification tested -->

- [x] Create `tests/test_lfmc.py` — test SAI computation against known absorption features, PLSR on synthetic data, Globe-LFMC loader with mocked data
  <!-- files: tests/test_lfmc.py (new) -->
  <!-- pattern: create synthetic spectrum with known absorption feature at 1200nm.
       Compute SAI, verify it's > 0 and in [0, 1].
       For PLSR: create synthetic spectra where reflectance at water bands correlates with LFMC.
       Train PLSR, verify R² > 0.5.
       For Globe-LFMC: mock CSV file, verify GeoDataFrame output schema. -->
  <!-- gotcha: the synthetic moisture signal should be strong enough for PLSR to find it.
       Add a clear relationship: LFMC = 200 - 500 * reflectance_at_1200nm + noise. -->
  <!-- acceptance: SAI tested against known features; PLSR R² reasonable; loader schema verified -->

- [x] Create `tests/test_validation.py` — test accuracy metrics against hand-calculated values, spatial aggregation preserves total, sensor comparison format
  <!-- files: tests/test_validation.py (new) -->
  <!-- pattern: test compute_accuracy with known inputs:
       perfect prediction → R²=1.0, RMSE=0, Kappa=1.0
       all-wrong classification → low Kappa
       Test compare_sensors with two synthetic results, verify improvement ratio math. -->
  <!-- acceptance: metrics match hand-calculated values; both continuous and classified tested -->

- [x] Verify: `pytest tests/` passes with all new tests green; no regressions in Phase 2 tests
  <!-- verify: `pytest tests/ -v` -->
  <!-- acceptance: all tests pass; zero failures; zero errors -->

### Section 10: Package Integration
<!-- execution_mode: sequential -->
<!-- dep: all new modules and tests must exist -->

- [x] Update `src/tanager/__init__.py` to export new public API: endmembers (load_usgs_library, load_ecostress_library, load_frames_library, resample_library, build_fire_library), unmixing (run_mesma, select_bands_uszu, normalize_fractions), severity (train_severity_model, predict_severity, compute_trajectories), lfmc (compute_lfmc_indices, train_lfmc_plsr, predict_lfmc), validation (compute_accuracy, compare_sensors)
  <!-- files: src/tanager/__init__.py (modify) -->
  <!-- pattern: add entries to _LAZY_EXPORTS dict following existing pattern:
       "load_usgs_library": "endmembers",
       "run_mesma": "unmixing",
       "train_severity_model": "severity",
       etc.
       Update __all__ (it's auto-derived from _LAZY_EXPORTS, so just adding to the dict suffices). -->
  <!-- gotcha: do NOT import the modules eagerly — keep the lazy import pattern.
       The existing __getattr__ mechanism handles everything. Just add the name→module mapping. -->
  <!-- acceptance: `import tanager; tanager.run_mesma` resolves without error; all new public symbols accessible -->

- [x] Update `docs/engineering-memory.md` — add new modules to Module Registry, update Architecture Decisions with MESMA engine choice
  <!-- files: docs/engineering-memory.md (modify) -->
  <!-- gotcha: EM handles this — do NOT delegate to coder. This task is for the EM pass. -->
  <!-- acceptance: Module Registry includes endmembers.py, unmixing.py, severity.py, lfmc.py, validation.py -->

- [x] Verify: `import tanager; tanager.run_mesma` resolves without error; all lazy imports work
  <!-- verify: `python -c "import tanager; print(tanager.run_mesma)"` -->
  <!-- acceptance: all new public API symbols resolve via lazy import -->

### Wave 4 Gate
<!-- gate: qa-review -->
Verify: full test suite passes, package installs cleanly with new dependencies, all modules are importable, engineering-memory is updated. Phase 3 complete.
