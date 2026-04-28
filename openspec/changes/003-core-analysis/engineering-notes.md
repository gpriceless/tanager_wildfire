# Engineering Notes — 003-core-analysis

## Verdict: READY (with monitored risks)

All dependencies identified, architecture sound, task structure validated against spec scenarios. Five open questions reviewed — none are blocking. Two HIGH risks identified (mesma compatibility, FRAMES acquisition) with mitigations already built into the task plan. Wave 1 Section 1 is a gating compatibility check that determines the unmixing engine before any pipeline code is written.

## Parallelism

- Total waves: 4
- Wave 1 (Endmember Foundation): sequential — 3 sections, 16 tasks, foundational dependency for all subsequent work
- Wave 2 (MESMA Unmixing): sequential — 2 sections, 10 tasks, depends on Wave 1 endmember library
- Wave 3 (Severity + LFMC): parallel — 2 tracks (severity.py, lfmc.py), file-disjoint, 13 tasks total
- Wave 4 (Validation + Integration): sequential — 3 sections, 9 tasks, integration layer
- Max parallel tracks: 2 (in Wave 3)
- File-disjointness: VERIFIED — Track A (severity.py) and Track B (lfmc.py) are both new files with zero shared modifications

## Execution Plan

- Estimated coder spawns: 4 sequential (Waves 1, 2, 4) + 2 parallel (Wave 3) = 6 total sessions
- Branch strategy: feature/003-core-analysis (worktrees for Wave 3 parallel tracks)
- Resource estimate: heavy (4 waves, new dependencies, compatibility gating, 5 new modules)

## Spec Scenario Coverage

| Spec Requirement | Scenarios | Tasks Covering |
|---|---|---|
| Endmember Library Loading | Load USGS, Load ECOSTRESS, Load FRAMES, Resample | Section 2 tasks 2-5 |
| Endmember Selection | In-CoB, EAR/MASA, Image-derived | Section 3 tasks 1-3 |
| MESMA Spectral Unmixing | Run MESMA, uSZU, RMSE constraint, Shade norm, HySUPP fallback | Section 4 tasks 2-7 |
| Burn Severity Mapping | Train, Predict, Trajectories, dNBR comparison | Track A tasks 2-5 |
| LFMC Estimation | SAI computation, Compute indices, PLSR, Globe-LFMC, Predict+uncertainty | Track B tasks 2-6 |
| Validation Framework | AVIRIS-3, BARC, Accuracy metrics, Sensor comparison | Section 8 tasks 2-5 |

All 25 spec scenarios are covered by at least one task. No orphan scenarios.

## Open Question Engineering Assessments

### 1. mesma v1.0.8 Python/numpy compatibility — RISK: HIGH
Last release Nov 2020 (66 weekly downloads). numpy 2.x changed C ABI. Likely failure mode: ImportError or segfault on numpy dtype changes. Mitigation is fully specced: Wave 1 Section 1 task 2 is a gating test. If mesma fails, HySUPP FCLS becomes primary (task 4 adjusts downstream). The "MESMA fallback to HySUPP" spec scenario ensures the output schema is identical regardless of engine. This risk is HIGH but mitigated — it does NOT block starting execution.

### 2. FRAMES SoCal library acquisition — RISK: MEDIUM
66 SoCal chaparral spectra from USDA FRAMES database. Bulk download mechanism unverified. Mitigation: `load_frames_library()` accepts a local directory of pre-downloaded ASCII files. Manual acquisition is acceptable for competition purposes. Even without FRAMES, USGS v7 + ECOSTRESS provide a viable (though less fire-specific) library. Not blocking.

### 3. Globe-LFMC 2.0 SoCal coverage — RISK: LOW
287,551 observations globally. Even if SoCal chaparral subset is small (<50), expanding to all California chaparral sites is a simple bbox adjustment. The loader accepts any bbox, and the PLSR training code handles small sample sizes with appropriate warnings. Not blocking.

### 4. AVIRIS-3 Eaton Fire ORNL DAAC access — RISK: LOW (deferred)
This is a Wave 4 dependency only. Waves 1-3 do not need AVIRIS-3 data. The validation loader can raise FileNotFoundError if data isn't available, and the full pipeline still works for MESMA + LFMC products. Real data integration happens in Phase 4 (notebooks). Not blocking.

### 5. Shade endmember construction — RESOLVED
Single zero-reflectance spectrum is standard practice (Roberts 2018, Quintano 2023). If shadow-heavy pixels show poor MESMA fit quality, adding partial shade is a follow-up, not a blocker. Implemented as np.zeros(n_bands) in build_fire_library().

## Enrichment Changes (this pass)

1. **Added acceptance criteria** to every task — explicit definition of done
2. **Added file annotations** (`<!-- files: -->`) to all tasks specifying exact module paths
3. **Added pattern references** (`<!-- pattern: -->`) pointing to Phase 2 exemplar modules (spectral.py, catalog.py)
4. **Added import direction constraints** (`<!-- gotcha: -->`) — documented which modules can import from which to prevent circular dependencies
5. **Added dependency flags** (`<!-- dep: -->`) for inter-task and inter-wave dependencies
6. **Added risk callouts** for mesma compatibility (HIGH), FRAMES data format (MEDIUM), Globe-LFMC coverage (LOW)
7. **Added network annotations** to all sections indicating whether internet access is needed
8. **Added test file references** (`<!-- test: -->`) pointing to specific test files for each functional task
9. **Enriched open questions** with engineering notes and resolution guidance
10. **Verified parallel safety** — Wave 3 Track A (severity.py) and Track B (lfmc.py) are completely file-disjoint

## New Dependency Additions

| Package | Purpose | Version | Risk |
|---|---|---|---|
| mesma | MESMA spectral unmixing | >=1.0.8 | HIGH — may not be numpy 2.x compatible; optional dep |
| spectral-libraries | EAR/MASA/CoB endmember selection | >=1.1.3 | LOW — same author as mesma, more recently maintained |
| splib07-loader | USGS Spectral Library v7 Python loader | git+https | LOW — small utility, pure Python |
| joblib | Model serialization (RF, PLSR) | Latest | NONE — already transitive dep of scikit-learn |

## Import Direction Map (Phase 3)

```
config.py    <-- endmembers.py
config.py    <-- unmixing.py
config.py    <-- severity.py
config.py    <-- lfmc.py
spectral.py  <-- endmembers.py (for continuum_removal)
spectral.py  <-- lfmc.py (for select_bands, _normalized_difference)
spectral.py  <-- severity.py (for dnbr)
spectral.py  <-- masks.py (existing, for ndwi)
endmembers.py <-- unmixing.py
unmixing.py  <-- severity.py (for run_mesma in trajectories)
(any)        <-- validation.py (top of tree, imports freely)
```

No circular dependencies. validation.py is a leaf — nothing imports from it.

## Gotchas

1. **mesma numpy ABI compatibility:** If mesma installs but segfaults at runtime (numpy C extension mismatch), that counts as failure. Test must include both import AND functional run.

2. **In-CoB circular dependency:** Full In-CoB selection requires running MESMA on image pixels to count "wins" per endmember. But MESMA requires an endmember library. Resolution: implement simplified spectral-variability ranking first, add full In-CoB as a post-unmixing refinement.

3. **SPy array convention mismatch:** SPy expects (rows, cols, bands) numpy arrays. Our xarray convention is (wavelength, y, x). Every SPy interface call needs a transpose: `data.values.transpose(1, 2, 0)`.

4. **ECOSTRESS wavelength units:** SPy EcostressDatabase may return wavelengths in micrometers, not nanometers. Must multiply by 1000 to convert to our nm convention.

5. **HySUPP vs MESMA output semantics:** MESMA is multi-endmember (combinatorial — different endmember sets per pixel, NaN for no valid model). FCLS is single-model (all endmembers always used, always produces fractions). The output schema is identical but the NaN semantics differ. Document this in the unmixing.py module docstring.

6. **PLSR n_components constraint:** sklearn PLSRegression requires n_components <= min(n_samples, n_features). With ~330 bands and potentially < 50 ground truth observations, n_components is limited by sample size, not spectral dimensionality.

7. **LFMC nonlinear regime:** Below 60% LFMC, the spectral-moisture relationship becomes nonlinear (Roberts et al. 2006). The flag in predict_lfmc is informational — the model still produces estimates, but users should interpret with caution.

## Task Count Validation

- Wave 1: 16 tasks (4 + 7 + 5)
- Wave 2: 10 tasks (8 + 2)
- Wave 3: 13 tasks (6 in Track A + 7 in Track B)
- Wave 4: 9 tasks (5 + 6 + 3... wait, let me recount)

Actual:
- Section 1: 4 tasks
- Section 2: 7 tasks
- Section 3: 5 tasks (incl verify)
- Section 4: 8 tasks (incl verify)
- Section 5: 3 tasks (incl verify)
- Track A: 6 tasks (incl verify)
- Track B: 7 tasks (incl verify)
- Section 8: 6 tasks (incl verify)
- Section 9: 6 tasks (incl verify)
- Section 10: 3 tasks (incl verify)

Total: 55 tasks across 4 waves. This is higher than PQ's original 35 because enrichment expanded verification steps and added explicit format validation tasks. Each task is roughly one coder focus area — reasonable scope.

## mesma v1.0.8 Compatibility Verification (2026-04-28)

**Test environment:**
- Python: 3.12.3
- numpy: 2.4.4
- OS: Linux 6.17.0-20-generic (Ubuntu)

**Test 1 — pip install:** PASS

`pip install mesma>=1.0.8` completed successfully; `mesma-1.0.8-py3-none-any.whl` installed without errors. Note: the package is pure-Python (no compiled C extensions), so there is no numpy ABI compilation step. Install will not fail due to ABI mismatch.

**Test 2 — import:** PASS (with caveat)

`import mesma` succeeds. `mesma.__version__` does not exist (AttributeError) — the package has no version attribute in `__init__.py`. `from mesma.core.mesma import MesmaCore, MesmaModels` imports cleanly.

**Test 3 — functional smoke test:** PARTIAL PASS / SOFT-FAIL on shade path

Core unmixing works. A synthetic 5×5 image (10 bands, 60% pv + 40% soil mixture) was unmixed using `MesmaCore.execute()`. Recovered fractions: pv=0.6000, soil=0.4000, shade=0.0000, RMSE=0.0 (exact recovery). The core MESMA algorithm is numerically correct under Python 3.12 / numpy 2.4.4.

**However, `shade_spectrum` parameter is broken under numpy 2.x:**

`_subtract_shade()` attempts `self.image_as_line - shade_spectrum` where `image_as_line` shape is `(n_bands, n_pixels)` and `shade_spectrum` shape is `(n_bands,)`. In numpy 1.x this subtracted shade band-wise (treating `(n_bands,)` as a column). In numpy 2.x this raises `ValueError: operands could not be broadcast together with shapes (10,25) (10,)`. The fix would be `shade_spectrum.reshape(-1, 1)` but we cannot patch the installed package.

**Workaround:** Omit `shade_spectrum` parameter (pass `None`). Pre-subtract shade endmember from library columns before calling `execute()`. This is equivalent — mesma's shade correction is a linear operation. The shade fraction output (last channel of fractions array) will be `1 - sum(other_fractions)` per pixel, which is the standard photometric shade derivation.

**API discovery findings (relevant for unmixing.py implementation):**

- `MesmaCore.execute()` expects `image` as `(n_bands, rows, cols)` or `(n_bands, n_pixels)` — bands-first, NOT `(rows, cols, bands)`.
- `library` expects `(n_bands, n_endmembers)` — bands-first, columns are spectra.
- `look_up_table` must be `dict[int, dict[tuple[int,...], np.ndarray]]` — outer key is level (2=2-EM, 3=3-EM), inner key is a **tuple of class indices** (not a string), value is array of endmember index combinations.
- `em_per_class` is `dict[str, list[int]]` mapping class name to library column indices.
- `MesmaCore.execute()` returns a **4-tuple**: `(model_indices, fractions, rmse_map, residual_image)` — NOT an object with `.get_fractions()`/`.get_rmse()` methods. Unpack as: `_, fractions, rmse_map, _ = core.execute(...)`
- Output `fractions` shape: `(n_classes+1, rows, cols)` — last channel is shade.
- Unmodeled pixels: fractions=0, rmse=9999 (not NaN — requires post-processing to convert to NaN).

**Verdict:** USE MESMA (with shade workaround)

**Implications for Wave 2 unmixing.py:**

The core MESMA engine is usable. `unmixing.py` must use the bands-first array convention for both image and library, build the LUT using tuple-of-integer keys (not string keys), and pass `shade_spectrum=None` while pre-subtracting shade from the library if shade correction is needed. Unmodeled pixels come back with rmse=9999 (not NaN) — `run_mesma()` must convert `rmse >= 9999` to NaN fractions. The HySUPP fallback should still be implemented for robustness, but MESMA is the primary engine. `mesma` stays as an optional dependency in pyproject.toml (the engine detection pattern in `unmixing.py` handles the import gracefully), with the recommendation to promote it to hard deps only after confirming 426-band performance.
