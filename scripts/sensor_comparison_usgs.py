"""Re-run the notebook 05 sensor comparison with the external USGS library.

Notebook 05 scores MESMA char fraction from spectrally degraded Tanager data
against the native 426-band result, using endmembers averaged out of the image
(``extract_image_endmembers``). This script asks whether those numbers survive
when the endmembers come from USGS splib07a instead
(``data/reference/endmembers/palisades_fire_library.csv``), which never saw the
scene.

Three blocks, each written to ``outputs/sensor_comparison_usgs.md``:

1. **Control.** Notebook 05's own construction on its own 128x128 crop
   (image-derived endmembers, all bands). It must reproduce the notebook's
   EMIT/PRISMA/Sentinel-2 char R² before anything else is believed.
2. **Library swap on the same crop.** Identical procedure with the USGS
   library, once on all bands (the direct analogue of the notebook) and once on
   the band set ``scripts/palisades_mesma.py`` uses for the validated map.
3. **Full Palisades swath.** USGS library on the validated band set, each
   sensor scored two ways: R² against native Tanager (agreement), and AUC
   against the official perimeter and the CAL FIRE DINS survey (accuracy
   against references outside the imagery).

Each degraded sensor gets the scene and the library through the same Gaussian
spectral resampling, so the solver always sees matched resolution.

Usage::

    python scripts/sensor_comparison_usgs.py
"""

from __future__ import annotations

import argparse
import datetime
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from palisades_common import (  # noqa: E402
    LIBRARY_CSV,
    OUTPUTS,
    PALISADES_SCENE,
    library_dataarray,
    load_library,
    load_masked_scene,
    point_indices,
    polygon_masks,
)
from palisades_mesma import MESMA_BANDS  # noqa: E402
from validate_char_fractions import auc  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sensor_comparison_usgs")

CROP_PX = 128
FRACTIONS = ("char", "pv", "npv", "soil")

# Notebook 05's published char R² against native Tanager — the control target.
NOTEBOOK05_CHAR_R2 = {"EMIT": 0.990820, "PRISMA": 0.957077, "Sentinel-2": 0.360800}


def sensor_specs() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Band centres and FWHM per sensor, built exactly as notebook 05 builds them."""
    from tanager.config import EMIT_SENSOR, PRISMA_SENSOR, SENTINEL2_BANDS

    def uniform(sensor) -> tuple[np.ndarray, np.ndarray]:
        centers = np.linspace(float(sensor.wavelength_min_nm), float(sensor.wavelength_max_nm),
                              int(sensor.n_bands))
        return centers, np.full(centers.shape, float(sensor.fwhm_nm))

    s2_centers = np.array([b["center_nm"] for b in SENTINEL2_BANDS.values()], dtype=np.float64)
    s2_fwhm = np.array([b["fwhm_nm"] for b in SENTINEL2_BANDS.values()], dtype=np.float64)
    return {"EMIT": uniform(EMIT_SENSOR), "PRISMA": uniform(PRISMA_SENSOR),
            "Sentinel-2": (s2_centers, s2_fwhm)}


def validated_band_subset(centers: np.ndarray, fwhm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each sensor's nearest band to every MESMA_BANDS centre, de-duplicated.

    For EMIT and PRISMA this is 22 distinct bands. Sentinel-2's ten bands all
    sit on or next to a MESMA_BANDS centre and none falls in an atmospheric
    water window, so it keeps all ten.
    """
    if centers.size < MESMA_BANDS.size:
        return centers, fwhm
    idx = np.unique(np.abs(centers[None, :] - MESMA_BANDS[:, None]).argmin(axis=1))
    return centers[idx], fwhm[idx]


def unmix(refl, library, bands=None):
    from tanager.unmixing import SHADE_NORMALIZED_CONSTRAINTS, normalize_fractions, run_mesma

    raw = run_mesma(refl, library, constraints=SHADE_NORMALIZED_CONSTRAINTS, bands=bands)
    return normalize_fractions(raw)


def r2_vs_native(sim, native, mask=None) -> dict:
    from tanager.validation import compute_accuracy

    s = np.asarray(sim, dtype=np.float64)
    n = np.asarray(native, dtype=np.float64)
    if mask is not None:
        s = np.where(mask, s, np.nan)
        n = np.where(mask, n, np.nan)
    acc = compute_accuracy(s, n, metric_type="continuous")
    return {"r2": float(acc["r2"]), "rmse": float(acc["rmse"]), "n": int(acc["n_valid"])}


def crop_of(scene):
    ny, nx = scene.sizes["y"], scene.sizes["x"]
    y0 = max(0, ny // 2 - CROP_PX // 2)
    x0 = max(0, nx // 2 - CROP_PX // 2)
    return scene.isel(y=slice(y0, y0 + CROP_PX), x=slice(x0, x0 + CROP_PX))


def notebook05_image_library(crop):
    """Notebook 05's ``extract_image_endmembers`` construction, verbatim."""
    from tanager.endmembers import extract_image_endmembers
    from tanager.spectral import nbr, ndvi

    nbr_v = np.asarray(nbr(crop).values, dtype=np.float64)
    ndvi_v = np.asarray(ndvi(crop).values, dtype=np.float64)
    finite = np.isfinite(nbr_v) & np.isfinite(ndvi_v)

    def pct(arr, q):
        return np.nanpercentile(np.where(finite, arr, np.nan), q)

    def window(mask, half=8):
        ys, xs = np.where(mask)
        cy, cx = int(np.median(ys)), int(np.median(xs))
        return (slice(max(0, cy - half), min(mask.shape[0], cy + half)),
                slice(max(0, cx - half), min(mask.shape[1], cx + half)))

    class_masks = {
        "char": finite & (nbr_v <= pct(nbr_v, 15)) & (ndvi_v <= pct(ndvi_v, 35)),
        "pv": finite & (ndvi_v >= pct(ndvi_v, 80)),
        "npv": finite & (ndvi_v >= pct(ndvi_v, 40)) & (ndvi_v <= pct(ndvi_v, 60)),
        "soil": finite & (ndvi_v <= pct(ndvi_v, 30)) & (nbr_v >= pct(nbr_v, 30)),
    }
    regions = {k: window(v) for k, v in class_masks.items() if int(v.sum()) >= 25}
    return extract_image_endmembers(crop, method="spatial", regions=regions)


def endmember_free_table(crop, specs) -> pd.DataFrame:
    """Notebook 05 sections 5-6: NBR, NDVI and continuum-removed absorption depths.

    None of these touch an endmember library, so the circularity question does
    not apply; they are recomputed here so the published values have a source.
    """
    from tanager.lfmc import compute_lfmc_indices
    from tanager.spectral import nbr, ndvi
    from tanager.validation import simulate_sensor

    native = {"NBR": nbr(crop), "NDVI": ndvi(crop)}
    native_cr = compute_lfmc_indices(crop)["CR_depths"]
    rows = []
    for name, (centers, fwhm) in specs.items():
        sim = simulate_sensor(crop, centers, fwhm, name)
        for product, fn in (("NBR", nbr), ("NDVI", ndvi)):
            m = r2_vs_native(fn(sim).values, native[product].values)
            rows.append({"sensor": name, "product": product, "demand": "broadband", **m})
        sim_cr = compute_lfmc_indices(sim)["CR_depths"]
        for target in (970.0, 1200.0, 1700.0, 2100.0):
            m = r2_vs_native(sim_cr.sel(cr_target=target).values,
                             native_cr.sel(cr_target=target).values)
            rows.append({"sensor": name, "product": f"CR {target:.0f} nm", "demand": "absorption", **m})
    return pd.DataFrame(rows)


def degraded_runs(refl, library, specs, *, band_subset: bool, library_source_fwhm: float):
    """Unmix native and each degraded sensor; return {label: fractions Dataset}."""
    from tanager.endmembers import resample_library
    from tanager.validation import simulate_sensor

    out = {"Native": unmix(refl, library, bands=MESMA_BANDS if band_subset else None)}
    for name, (centers, fwhm) in specs.items():
        if band_subset:
            centers, fwhm = validated_band_subset(centers, fwhm)
        sim = simulate_sensor(refl, centers, fwhm, name)
        sim_lib = resample_library(library, centers, fwhm=fwhm, source_fwhm=library_source_fwhm)
        started = time.time()
        out[name] = unmix(sim, sim_lib)
        log.warning("%s: %d bands, MESMA %.1f s", name, centers.size, time.time() - started)
        del sim
    return out


def r2_table(runs: dict, mask=None) -> pd.DataFrame:
    native = runs["Native"]
    rows = []
    for name, frac in runs.items():
        if name == "Native":
            continue
        for var in FRACTIONS:
            m = r2_vs_native(frac[var].values, native[var].values, mask)
            rows.append({"sensor": name, "fraction": var, **m})
    return pd.DataFrame(rows)


def pivot(frame: pd.DataFrame, value: str = "r2") -> str:
    return frame.pivot(index="sensor", columns="fraction", values=value)[list(FRACTIONS)] \
        .reindex(["EMIT", "PRISMA", "Sentinel-2"]).to_string(float_format=lambda v: f"{v:.3f}")


def referent_table(runs: dict, inside, outside, dins) -> pd.DataFrame:
    (rd, cd), (rn, cn) = dins
    rows = []
    for name, frac in runs.items():
        char = np.asarray(frac["char"].values, dtype=np.float64)
        fin = np.isfinite(char)
        des, nod = char[rd, cd], char[rn, cn]
        des, nod = des[np.isfinite(des)], nod[np.isfinite(nod)]
        rows.append({
            "sensor": name,
            "modeled_px": int(fin.sum()),
            "inside_median": float(np.median(char[inside & fin])),
            "outside_median": float(np.median(char[outside & fin])),
            "perimeter_auc": auc(char[inside & fin], char[outside & fin]),
            "dins_destroyed_median": float(np.median(des)),
            "dins_auc": auc(des, nod),
            "n_dins": f"{des.size}/{nod.size}",
        })
    return pd.DataFrame(rows).set_index("sensor")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=PALISADES_SCENE)
    parser.add_argument("--out", type=Path, default=OUTPUTS / "sensor_comparison_usgs.md")
    parser.add_argument("--skip-full", action="store_true", help="crop blocks only")
    args = parser.parse_args()

    specs = sensor_specs()
    usgs = library_dataarray(load_library(LIBRARY_CSV))

    scene = load_masked_scene(args.scene)
    primary = scene.attrs["data_var"]
    refl = scene[primary].drop_vars(("fwhm", "good_wavelengths"), errors="ignore")
    crop = crop_of(scene)
    crop_refl = crop[primary].drop_vars(("fwhm", "good_wavelengths"), errors="ignore")

    inside_full, outside_full = polygon_masks(refl, "PALISADES")
    crop_inside = polygon_masks(crop_refl, "PALISADES")[0]
    crop_valid = np.isfinite(crop_refl.isel(wavelength=0).values)

    report: list[str] = []
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout.strip()
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 0. Endmember-free products on the notebook 05 crop.
    free = endmember_free_table(crop, specs)
    free_wide = free.pivot(index="product", columns="sensor", values="r2")[["EMIT", "PRISMA", "Sentinel-2"]]
    free_means = free.groupby(["demand", "sensor"])["r2"].mean().unstack()[["EMIT", "PRISMA", "Sentinel-2"]]

    # 1. Control: notebook 05 exactly, on its own crop.
    image_lib = notebook05_image_library(crop)
    control = degraded_runs(crop_refl, image_lib, specs, band_subset=False, library_source_fwhm=5.5)
    control_r2 = r2_table(control)
    char_r2 = control_r2[control_r2.fraction == "char"].set_index("sensor")["r2"]
    reproduced = all(abs(char_r2[k] - v) < 0.005 for k, v in NOTEBOOK05_CHAR_R2.items())

    # 2. Library swap on the same crop.
    swap_all = degraded_runs(crop_refl, usgs, specs, band_subset=False, library_source_fwhm=5.5)
    swap_sub = degraded_runs(crop_refl, usgs, specs, band_subset=True, library_source_fwhm=5.5)
    crop_char_native = np.asarray(swap_sub["Native"]["char"].values, dtype=np.float64)

    report += [
        "# Sensor comparison with the USGS endmember library",
        "",
        f"Generated {stamp} by scripts/sensor_comparison_usgs.py. Commit: {commit}",
        "",
        "Scene: `20250123_185518_92_4001` (Palisades post-fire swath). Library: USGS",
        "splib07a, `data/reference/endmembers/palisades_fire_library.csv` (23 spectra).",
        "EMIT and PRISMA band centres are uniformly spaced across each sensor's range,",
        "as in notebook 05; Sentinel-2 uses its published centres and bandwidths.",
        "",
        "**What R² vs native measures.** Each score compares the char map from",
        "spectrally degraded Tanager data with the char map from the full 426-band",
        "data, same pixels, same library. It measures how much of the native",
        "result survives losing bands, not whether either map is right; the",
        "perimeter and DINS columns in section 3 are the accuracy test.",
        "",
        "## 0. Products with no endmembers (128x128 crop, all bands)",
        "",
        "NBR, NDVI and continuum-removed absorption depths, each scored against the",
        "same product from native Tanager. Notebook 05 published means of 0.999 / 0.995 /",
        "0.996 (broadband) and 0.965 / 0.904 / -0.623 (absorption) for EMIT / PRISMA /",
        "Sentinel-2.",
        "",
        "```",
        free_wide.to_string(float_format=lambda v: f"{v:.3f}"),
        "",
        "mean by product type",
        free_means.to_string(float_format=lambda v: f"{v:.3f}"),
        "```",
        "",
        "## 1. Control: notebook 05 reproduced (image-derived endmembers, 128x128 crop)",
        "",
        "```",
        pivot(control_r2),
        "```",
        "",
        "Notebook 05 char R²: EMIT 0.991, PRISMA 0.957, Sentinel-2 0.361. "
        + ("Reproduced within 0.005." if reproduced else "**NOT reproduced; the blocks below are not comparable.**"),
        "",
        f"Crop: {int(crop_valid.sum())} valid pixels of {CROP_PX * CROP_PX}; "
        f"{int((crop_inside & crop_valid).sum())} inside the Palisades perimeter. "
        f"Native USGS char on the crop: median {np.nanmedian(crop_char_native):.3f}, "
        f"{100.0 * np.nanmean(crop_char_native > 0.05):.1f}% of modeled pixels above 0.05.",
        "",
        "## 2. Same crop, USGS library",
        "",
        "All bands (direct analogue of notebook 05):",
        "",
        "```",
        pivot(r2_table(swap_all)),
        "```",
        "",
        "Validated band set (`MESMA_BANDS` in `scripts/palisades_mesma.py`; 22 bands, Sentinel-2 all 10):",
        "",
        "```",
        pivot(r2_table(swap_sub)),
        "```",
        "",
    ]
    del control, swap_all, swap_sub

    if not args.skip_full:
        full = degraded_runs(refl, usgs, specs, band_subset=True, library_source_fwhm=5.5)
        dins = (point_indices(refl, "Destroyed (>50%)"), point_indices(refl, "No Damage"))
        r2_all = r2_table(full)
        r2_in = r2_table(full, mask=inside_full)
        ref = referent_table(full, inside_full, outside_full, dins)
        native_char = np.asarray(full["Native"]["char"].values, dtype=np.float64)
        committed = OUTPUTS / "20250123swath2_frac_char.tif"
        match_line = ""
        if committed.exists():
            import rioxarray  # noqa: F401
            import xarray as xr

            prior = np.asarray(
                xr.open_dataarray(committed, engine="rasterio").squeeze("band", drop=True).values,
                dtype=np.float64,
            )
            both = np.isfinite(prior) & np.isfinite(native_char)
            match_line = (
                f"Native run vs `{committed.name}` (the map behind AUC 0.795): "
                f"{int(both.sum())} common pixels, max |diff| "
                f"{float(np.max(np.abs(prior[both] - native_char[both]))):.2e}, "
                f"modeled {int(np.isfinite(native_char).sum())} vs {int(np.isfinite(prior).sum())}."
            )
        report += [
            "## 3. Full Palisades swath, USGS library, validated band set",
            "",
            match_line,
            "",
            "R² against native Tanager, all modeled pixels:",
            "",
            "```",
            pivot(r2_all),
            "```",
            "",
            "R² against native Tanager, inside the Palisades perimeter only:",
            "",
            "```",
            pivot(r2_in),
            "```",
            "",
            "Char against references outside the imagery (perimeter: inside Palisades vs",
            "outside every perimeter in `la_fires_2025.geojson`; DINS: `Destroyed (>50%)`",
            "vs `No Damage` points):",
            "",
            "```",
            ref.to_string(float_format=lambda v: f"{v:.3f}"),
            "```",
            "",
        ]
        r2_all.assign(scope="all_modeled").pipe(
            lambda f: pd.concat([f, r2_in.assign(scope="inside_palisades")])
        ).to_csv(OUTPUTS / "sensor_comparison_usgs_r2.csv", index=False)
        ref.to_csv(OUTPUTS / "sensor_comparison_usgs_referents.csv")

    args.out.write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
