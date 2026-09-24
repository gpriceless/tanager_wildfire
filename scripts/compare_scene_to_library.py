"""Compare Palisades scene spectra against the USGS endmember library.

Before unmixing, check that the library and the image live in the same
radiometric space: pull the mean spectrum of burned and unburned pixels
(labelled by the official perimeter and by CAL FIRE DINS damage class, never by
a spectral threshold) and measure the spectral angle from each to every library
spectrum. If image char does not look like library char, MESMA will not
discover that for us — it will just return a bad fit quietly.

Usage::

    python scripts/compare_scene_to_library.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from palisades_common import (  # noqa: E402
    LIBRARY_CSV,
    PALISADES_SCENE,
    load_library,
    load_masked_scene,
    point_samples,
    polygon_masks,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("compare")

PROBE_WAVELENGTHS = (480.0, 560.0, 660.0, 860.0, 1240.0, 1650.0, 2100.0, 2200.0)


def spectral_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two reflectance vectors."""
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(num / den, -1.0, 1.0))))


def main() -> int:
    library = load_library(LIBRARY_CSV)
    scene = load_masked_scene(PALISADES_SCENE)
    refl = scene[scene.attrs["data_var"]]
    wavelengths = np.asarray(refl.coords["wavelength"].values, dtype=np.float64)

    inside, outside = polygon_masks(scene, "PALISADES")
    groups: dict[str, np.ndarray] = {
        "inside_perimeter": inside,
        "outside_perimeter": outside,
    }
    for label, mask in point_samples(scene).items():
        groups[label] = mask

    rows = []
    mean_spectra: dict[str, np.ndarray] = {}
    for label, mask in groups.items():
        n = int(mask.sum())
        if n == 0:
            log.warning("group %s has no pixels", label)
            continue
        cube = refl.values[:, mask]
        finite = np.isfinite(cube).all(axis=0)
        cube = cube[:, finite]
        if cube.shape[1] == 0:
            log.warning("group %s has no fully finite spectra (%d pixels)", label, n)
            continue
        mean = np.nanmean(cube, axis=1)
        mean_spectra[label] = mean
        row = {"group": label, "n_px": n, "n_finite": int(cube.shape[1])}
        for w in PROBE_WAVELENGTHS:
            row[f"R{int(w)}"] = float(mean[np.argmin(np.abs(wavelengths - w))])
        rows.append(row)

    print("\n=== scene mean reflectance by reference group ===")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== spectral angle (deg) from each group mean to each library spectrum ===")
    lib_values = library["reflectance"]
    ang = pd.DataFrame(
        {
            label: [spectral_angle(mean, lib_values[i]) for i in range(lib_values.shape[0])]
            for label, mean in mean_spectra.items()
        },
        index=pd.MultiIndex.from_arrays(
            [library["category"], library["spectrum_id"]], names=["category", "spectrum_id"]
        ),
    )
    print(ang.to_string(float_format=lambda v: f"{v:6.2f}"))

    print("\n=== best-matching library class per group ===")
    for label in ang.columns:
        per_class = ang[label].groupby(level="category").min().sort_values()
        best = ", ".join(f"{c}={v:.1f}" for c, v in per_class.items())
        print(f"{label:22s} {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
