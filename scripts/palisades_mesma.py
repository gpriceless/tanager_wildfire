"""Run MESMA on the Palisades post-fire Tanager swath with the USGS library.

``scripts/run_pipeline.py`` only unmixes the 2024-12-15 scene, and it does so
with endmembers averaged out of the image itself under NBR/NDVI thresholds.
That is circular for a burned-area product — the char endmember is defined as
"the pixels with low NBR" and the char fraction is then read as burn extent.
This script unmixes the Palisades post-fire swath instead, with a library built
entirely from USGS splib07a (see ``scripts/build_fire_endmember_library.py``),
so nothing in the endmembers knows where the fire was.

Usage::

    python scripts/palisades_mesma.py [--scene ...] [--prefix 20250123swath2]

Writes ``outputs/<prefix>_frac_{char,pv,npv,soil}.tif`` and
``outputs/<prefix>_mesma_rmse.tif``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

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
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("palisades_mesma")

# Band centres (nm) MESMA is run on. Chosen to span the features that separate
# the four classes — visible slope and chlorophyll absorption, the red edge and
# NIR plateau, the 1200 nm liquid-water feature, the SWIR shoulders, and the
# 2100-2300 nm cellulose/lignin and clay region where char and soil part ways.
# The two saturated atmospheric water bands (1350-1450, 1800-1950 nm) are left
# out: surface reflectance there is an atmospheric-correction artefact.
MESMA_BANDS = np.array(
    [
        480.0, 560.0, 660.0, 705.0, 740.0, 783.0, 860.0, 960.0,
        1080.0, 1200.0, 1240.0, 1300.0,
        1520.0, 1610.0, 1660.0, 1700.0,
        2050.0, 2100.0, 2150.0, 2200.0, 2250.0, 2300.0,
    ],
    dtype=np.float32,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=PALISADES_SCENE)
    parser.add_argument("--prefix", default="20250123swath2")
    parser.add_argument("--library", type=Path, default=LIBRARY_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()

    from tanager.io import write_product_raster
    from tanager.unmixing import SHADE_NORMALIZED_CONSTRAINTS, normalize_fractions, run_mesma

    library = library_dataarray(load_library(args.library))
    cats, counts = np.unique(np.asarray(library.coords["category"].values, dtype=str), return_counts=True)
    log.info("library: %s", dict(zip(cats.tolist(), counts.tolist())))

    scene = load_masked_scene(args.scene)
    primary = scene.attrs["data_var"]
    refl = scene[primary].drop_vars(("fwhm", "good_wavelengths"), errors="ignore")
    n_px = int(refl.sizes["y"] * refl.sizes["x"])
    log.info("scene %s: %d pixels", args.scene.name, n_px)

    started = time.time()
    fractions = run_mesma(
        refl, library, constraints=SHADE_NORMALIZED_CONSTRAINTS, bands=MESMA_BANDS
    )
    log.info("MESMA finished in %.1f s (engine=%s)", time.time() - started,
             fractions.attrs.get("unmixing_engine"))

    fractions = normalize_fractions(fractions)
    n_outside = int(fractions.attrs.get("n_pixels_outside_unit_interval", 0))
    if n_outside:
        log.warning("%d pixels outside [0, 1] after shade normalization", n_outside)

    crs = scene.attrs.get("crs")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for var in fractions.data_vars:
        name = "mesma_rmse" if var == "rmse" else f"frac_{var}"
        path = args.out_dir / f"{args.prefix}_{name}.tif"
        write_product_raster(fractions[var], path, crs=crs)
        values = np.asarray(fractions[var].values, dtype=np.float64)
        finite = np.isfinite(values)
        log.info(
            "%s: modeled %.1f%% (%d px) mean=%.3f p50=%.3f p90=%.3f",
            name, 100.0 * finite.mean(), int(finite.sum()),
            float(np.nanmean(values)), float(np.nanpercentile(values[finite], 50)),
            float(np.nanpercentile(values[finite], 90)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
