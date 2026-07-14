"""Cross-sensor MESMA validation: Tanager char fraction vs AVIRIS-3 reflectance.

Runs the same MESMA unmixing on both Tanager (30 m) and AVIRIS-3 (2.9 m)
over the Palisades fire footprint using a same-day Jan 23 overpass, then
aggregates AVIRIS-3 fractions to the Tanager grid and computes R²/RMSE/bias.

This is cross-sensor validation (same algorithm, different sensors), not
independent ground truth.
"""

from __future__ import annotations

import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from tanager.endmembers import resample_library
from tanager.io import load_ortho_scene
from tanager.unmixing import run_mesma, select_bands_uszu
from tanager.validation import (
    _aggregate_fractions_to_grid,
    compute_accuracy,
    load_aviris3_reflectance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("aviris3_validation")

AVIRIS3_DIR = REPO_ROOT / "data" / "raw" / "aviris3"
TANAGER_SCENE_PATH = (
    REPO_ROOT / "data" / "raw" / "fire"
    / "20250123_185518_92_4001_ortho_sr_hdf5.h5"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "aviris3_validation"


def build_synthetic_fire_library(
    wavelengths: np.ndarray,
) -> xr.DataArray:
    """Build a minimal fire endmember library from synthetic spectra.

    Uses physically realistic spectral signatures for char, photosynthetic
    vegetation (PV), non-photosynthetic vegetation (NPV), soil, and shade.
    Adequate for cross-sensor validation where the same library is applied to
    both sensors — errors from library simplification cancel in the comparison.
    """
    n_wl = len(wavelengths)

    def _char(wl: np.ndarray) -> np.ndarray:
        spec = np.zeros(len(wl), dtype=np.float32)
        for i, w in enumerate(wl):
            if w < 1500.0:
                spec[i] = 0.02 + (w - 380.0) / (1500.0 - 380.0) * 0.02
            else:
                spec[i] = 0.04 + (w - 1500.0) / (2500.0 - 1500.0) * 0.03
        return spec

    def _pv(wl: np.ndarray) -> np.ndarray:
        spec = np.zeros(len(wl), dtype=np.float32)
        for i, w in enumerate(wl):
            if w < 680.0:
                spec[i] = 0.05 + (w - 380.0) / (680.0 - 380.0) * 0.05
            elif w < 750.0:
                spec[i] = 0.05 + (w - 680.0) / (750.0 - 680.0) * 0.40
            elif w < 1300.0:
                spec[i] = 0.45
            elif w < 1800.0:
                spec[i] = 0.45 - (w - 1300.0) / (1800.0 - 1300.0) * 0.25
            elif w < 2150.0:
                dist = abs(w - 2100.0)
                spec[i] = max(0.08, 0.20 - (1.0 - dist / 200.0) * 0.12)
            else:
                spec[i] = 0.08
        return spec

    def _npv(wl: np.ndarray) -> np.ndarray:
        spec = np.zeros(len(wl), dtype=np.float32)
        for i, w in enumerate(wl):
            if w < 700.0:
                spec[i] = 0.08 + (w - 380.0) / (700.0 - 380.0) * 0.07
            elif w < 1400.0:
                spec[i] = 0.15 + (w - 700.0) / (1400.0 - 700.0) * 0.15
            elif w < 1900.0:
                spec[i] = 0.30 - (w - 1400.0) / (1900.0 - 1400.0) * 0.10
            elif w < 2200.0:
                spec[i] = 0.20 - (w - 1900.0) / (2200.0 - 1900.0) * 0.05
            else:
                spec[i] = 0.15
        return spec

    def _soil(wl: np.ndarray) -> np.ndarray:
        return (
            0.10 + (wl - 380.0) / (2500.0 - 380.0) * 0.20
        ).astype(np.float32)

    builders = {
        "char": _char,
        "pv": _pv,
        "npv": _npv,
        "soil": _soil,
    }

    spectra = []
    ids = []
    names = []
    categories = []
    for cat, fn in builders.items():
        spec = fn(wavelengths)
        spectra.append(spec)
        ids.append(f"synthetic_{cat}_0001")
        names.append(cat)
        categories.append(cat)

    shade = np.zeros(n_wl, dtype=np.float32)
    spectra.append(shade)
    ids.append("synthetic_shade_0001")
    names.append("shade")
    categories.append("shade")

    reflectance = np.vstack(spectra).astype(np.float32)
    library = xr.DataArray(
        reflectance,
        dims=("spectrum_id", "wavelength"),
        coords={
            "spectrum_id": ids,
            "wavelength": wavelengths.astype(np.float32),
            "name": ("spectrum_id", names),
            "category": ("spectrum_id", categories),
            "source": ("spectrum_id", ["synthetic"] * len(ids)),
        },
    )
    return library


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- Step 1: Load Tanager scene ---
    log.info("Loading Tanager scene: %s", TANAGER_SCENE_PATH.name)
    tanager_ds = load_ortho_scene(TANAGER_SCENE_PATH)
    log.info(
        "Tanager scene: %d bands, %d×%d pixels, CRS=%s",
        tanager_ds.sizes["wavelength"],
        tanager_ds.sizes["y"],
        tanager_ds.sizes["x"],
        tanager_ds.attrs.get("crs"),
    )

    # Filter to good wavelengths (drop H2O absorption bands ~1350-1440,
    # ~1780-1970 nm that are NaN across the scene).
    rfl_var = tanager_ds.attrs.get("data_var", "surface_reflectance")
    if rfl_var != "reflectance" and rfl_var in tanager_ds.data_vars:
        tanager_ds = tanager_ds.rename({rfl_var: "reflectance"})

    if "good_wavelengths" in tanager_ds.coords:
        good_mask = tanager_ds.coords["good_wavelengths"].values == 1
        tanager_ds = tanager_ds.isel(wavelength=good_mask)
        log.info("Filtered to %d good bands", tanager_ds.sizes["wavelength"])

    # Clamp reflectance to [0, 1] — required by the mesma backend and
    # prevents NaN propagation from negative atmospheric-correction artefacts.
    rfl_data = tanager_ds["reflectance"].values.copy()
    rfl_data = np.where(np.isfinite(rfl_data), np.clip(rfl_data, 0.0, 1.0), np.nan)
    tanager_ds["reflectance"] = (("wavelength", "y", "x"), rfl_data)

    tanager_wl = np.asarray(tanager_ds.coords["wavelength"].values, dtype=np.float64)

    # --- Step 2: Build endmember library ---
    log.info("Building synthetic fire endmember library")
    library = build_synthetic_fire_library(tanager_wl)
    log.info(
        "Library: %d spectra, %d bands, categories=%s",
        library.sizes["spectrum_id"],
        library.sizes["wavelength"],
        sorted(set(library.coords["category"].values.tolist())),
    )

    # --- Step 3: Run MESMA on Tanager ---
    # Select 40 optimal bands via USZU for fast MESMA (standard practice).
    _, uszu_indices = select_bands_uszu(tanager_ds, library, n_bands=40)
    uszu_wl = tanager_wl[uszu_indices]
    log.info("USZU selected %d bands for MESMA", len(uszu_wl))
    log.info("Running MESMA on Tanager scene...")
    tanager_fractions = run_mesma(tanager_ds, library, bands=uszu_wl)
    log.info(
        "Tanager MESMA complete: engine=%s, vars=%s",
        tanager_fractions.attrs.get("unmixing_engine"),
        list(tanager_fractions.data_vars),
    )
    tanager_fractions.attrs["crs"] = tanager_ds.attrs.get("crs", "EPSG:32611")

    # Save Tanager fractions
    tanager_frac_path = OUTPUT_DIR / "tanager_palisades_fractions.nc"
    tanager_fractions.to_netcdf(tanager_frac_path)
    log.info("Saved Tanager fractions: %s", tanager_frac_path)

    # Free scene memory
    tanager_ds.close()
    del tanager_ds
    gc.collect()

    # --- Step 4: Process AVIRIS-3 granules ---
    aviris3_files = sorted(AVIRIS3_DIR.glob("*_RFL_ORT.nc"))
    log.info("Found %d AVIRIS-3 granules", len(aviris3_files))

    tgt_y = np.asarray(tanager_fractions.coords["y"].values, dtype=np.float64)
    tgt_x = np.asarray(tanager_fractions.coords["x"].values, dtype=np.float64)
    tgt_crs = tanager_fractions.attrs.get("crs")

    # Tanager extent for quick overlap check
    tgt_y_min, tgt_y_max = tgt_y.min(), tgt_y.max()
    tgt_x_min, tgt_x_max = tgt_x.min(), tgt_x.max()

    all_pred = []
    all_obs = []
    per_granule = []

    for i, av3_path in enumerate(aviris3_files):
        log.info(
            "[%d/%d] Processing %s", i + 1, len(aviris3_files), av3_path.name
        )
        try:
            av3_ds = load_aviris3_reflectance(av3_path)
        except Exception as exc:
            log.warning("Failed to load %s: %s", av3_path.name, exc)
            continue

        # Quick overlap check
        av3_y = np.asarray(av3_ds.coords["y"].values)
        av3_x = np.asarray(av3_ds.coords["x"].values)
        if (av3_y.max() < tgt_y_min or av3_y.min() > tgt_y_max
                or av3_x.max() < tgt_x_min or av3_x.min() > tgt_x_max):
            log.info("  No spatial overlap — skipping")
            av3_ds.close()
            continue

        try:
            # Drop H2O absorption bands from AVIRIS-3 (same regions as Tanager)
            av3_wl_all = np.asarray(
                av3_ds.coords["wavelength"].values, dtype=np.float64
            )
            h2o_mask = (
                ((av3_wl_all >= 1340) & (av3_wl_all <= 1440))
                | ((av3_wl_all >= 1780) & (av3_wl_all <= 1970))
            )
            if h2o_mask.any():
                av3_ds = av3_ds.isel(wavelength=~h2o_mask)
                log.info(
                    "  Dropped %d H2O bands, kept %d",
                    h2o_mask.sum(),
                    av3_ds.sizes["wavelength"],
                )

            # Clamp reflectance to [0, 1]
            av3_rfl = av3_ds["reflectance"].values.copy()
            av3_rfl = np.where(
                np.isfinite(av3_rfl), np.clip(av3_rfl, 0.0, 1.0), np.nan
            )
            av3_ds["reflectance"] = (("wavelength", "y", "x"), av3_rfl)

            # Resample library to AVIRIS-3 wavelength grid
            av3_wl = np.asarray(
                av3_ds.coords["wavelength"].values, dtype=np.float64
            )
            av3_library = resample_library(library, av3_wl)
            log.info(
                "  Resampled library to %d AVIRIS-3 bands", len(av3_wl)
            )

            # Run MESMA on AVIRIS-3 using the same USZU band positions
            av3_fractions = run_mesma(av3_ds, av3_library, bands=uszu_wl)
            log.info(
                "  AVIRIS-3 MESMA: engine=%s, shape=(%d, %d)",
                av3_fractions.attrs.get("unmixing_engine"),
                av3_fractions.sizes.get("y", 0),
                av3_fractions.sizes.get("x", 0),
            )

            if "crs" in av3_ds.attrs:
                av3_fractions.attrs["crs"] = av3_ds.attrs["crs"]

            av3_ds.close()
            del av3_ds
            gc.collect()

            # Aggregate to Tanager grid
            av3_aggregated = _aggregate_fractions_to_grid(
                av3_fractions,
                target_y=tgt_y,
                target_x=tgt_x,
                target_resolution=30.0,
                target_crs=tgt_crs,
            )

            # Extract char fractions for comparison
            if "char" not in av3_aggregated.data_vars:
                log.warning("  No 'char' variable in AVIRIS-3 fractions")
                continue

            pred = np.asarray(
                tanager_fractions["char"].values, dtype=np.float64
            ).ravel()
            obs = np.asarray(
                av3_aggregated["char"].values, dtype=np.float64
            ).ravel()

            valid = np.isfinite(pred) & np.isfinite(obs)
            n_valid = int(valid.sum())
            log.info("  Overlap pixels: %d", n_valid)

            if n_valid >= 10:
                granule_acc = compute_accuracy(
                    pred[valid], obs[valid], metric_type="continuous"
                )
                per_granule.append({
                    "granule": av3_path.name,
                    "n_pixels": n_valid,
                    **{k: float(v) for k, v in granule_acc.items()
                       if isinstance(v, (int, float, np.floating, np.integer))},
                })
                log.info(
                    "  R²=%.3f, RMSE=%.4f, bias=%.4f",
                    granule_acc["r2"],
                    granule_acc["rmse"],
                    granule_acc["bias"],
                )
                all_pred.append(pred[valid])
                all_obs.append(obs[valid])

            del av3_fractions, av3_aggregated
            gc.collect()

        except Exception as exc:
            log.warning("  Failed: %s", exc, exc_info=True)
            gc.collect()
            continue

    # --- Step 5: Compute aggregate accuracy ---
    if all_pred:
        combined_pred = np.concatenate(all_pred)
        combined_obs = np.concatenate(all_obs)
        overall = compute_accuracy(
            combined_pred, combined_obs, metric_type="continuous"
        )
        log.info("=" * 60)
        log.info("CROSS-SENSOR VALIDATION RESULTS (char fraction)")
        log.info("=" * 60)
        log.info("Granules with overlap: %d / %d", len(per_granule), len(aviris3_files))
        log.info("Total comparison pixels: %d", len(combined_pred))
        log.info("R²:   %.4f", overall["r2"])
        log.info("RMSE: %.4f", overall["rmse"])
        log.info("MAE:  %.4f", overall["mae"])
        log.info("Bias: %.4f", overall["bias"])
        log.info("=" * 60)

        results = {
            "method": "cross_sensor_mesma",
            "fraction_variable": "char",
            "tanager_scene": TANAGER_SCENE_PATH.name,
            "n_aviris3_granules": len(aviris3_files),
            "n_granules_with_overlap": len(per_granule),
            "total_comparison_pixels": int(len(combined_pred)),
            "overall_accuracy": {
                k: float(v) for k, v in overall.items()
                if isinstance(v, (int, float, np.floating, np.integer))
            },
            "per_granule": per_granule,
            "library": "synthetic_fire_5class",
            "tanager_crs": tgt_crs,
            "elapsed_seconds": round(time.time() - t0, 1),
            "honest_framing": (
                "Cross-sensor validation using the same MESMA algorithm on "
                "both Tanager (30 m) and AVIRIS-3 (2.9 m) with the same "
                "synthetic endmember library. NOT independent ground truth."
            ),
        }

        results_path = OUTPUT_DIR / "cross_validation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info("Results saved: %s", results_path)
    else:
        log.error("No valid granule overlaps — cannot compute cross-sensor accuracy")

    log.info("Total elapsed: %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
