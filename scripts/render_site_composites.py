#!/usr/bin/env python3
"""Render chrome-free Tanager-1 band composites of the Palisades scene for publication.

Builds the same three composites as the social figure set — true colour, NIR false colour
and SWIR — but over the largest axis-aligned rectangle of the swath in which every band used
is valid, trimmed of open sea below the coastline, at the instrument's native 30 m, with no title, caption, wavelength strip or credit
baked in. Each composite is written as its own file:

    firespec-true-colour.png
    firespec-nir.png
    firespec-swir.png

These are surface-reflectance composites of three bands each. No index, fraction, severity,
moisture or recovery quantity is computed or displayed.

Usage::

    python3 scripts/render_site_composites.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from tanager.io import load_ortho_scene

SCENE = Path("data/raw/fire/20250123_185518_92_4001_ortho_sr_hdf5.h5")
OUT_DIR = Path("outputs/site")

COMPOSITES = {
    "firespec-true-colour": (650.0, 550.0, 470.0),
    "firespec-nir": (860.0, 650.0, 550.0),
    "firespec-swir": (2200.0, 1600.0, 860.0),
}

SEA_MARGIN_ROWS = 20


def _nearest_band(wavelengths: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(wavelengths - target)))


def _largest_rectangle(valid: np.ndarray) -> tuple[int, int, int, int]:
    """Row/column bounds ``(r0, r1, c0, c1)`` of the largest all-True rectangle."""
    rows, cols = valid.shape
    heights = np.zeros(cols, dtype=int)
    best = (0, 0, 0, 0, 0)
    for r in range(rows):
        heights = np.where(valid[r], heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for c in range(cols + 1):
            h = heights[c] if c < cols else 0
            start = c
            while stack and stack[-1][1] >= h:
                start, top = stack.pop()
                area = top * (c - start)
                if area > best[0]:
                    best = (area, r - top + 1, r + 1, start, c)
            stack.append((start, h))
    return best[1:]


def _stretch(
    band: np.ndarray, land: np.ndarray, lo: float = 2.0, hi: float = 98.0
) -> np.ndarray:
    """Percentile stretch with limits taken from land pixels, so open sea sets no contrast."""
    vmin, vmax = np.percentile(band[land], [lo, hi])
    return np.clip((band - vmin) / (vmax - vmin), 0.0, 1.0)


def main() -> None:
    ds = load_ortho_scene(SCENE)
    sr = ds["surface_reflectance"]
    wl = ds["wavelength"].values.astype(float)
    x = ds["x"].values
    y = ds["y"].values

    band_index = {
        name: [_nearest_band(wl, t) for t in targets] for name, targets in COMPOSITES.items()
    }
    used = sorted({b for bands in band_index.values() for b in bands})
    cube = {b: sr.values[b] for b in used}
    valid = np.logical_and.reduce([np.isfinite(cube[b]) for b in used])
    r0, r1, c0, c1 = (int(v) for v in _largest_rectangle(valid))

    # Trim the open Pacific off the bottom of the window: keep rows down to the last one
    # that is mostly land, plus a short strip of sea so the coastline reads.
    green = sr.values[_nearest_band(wl, 560.0), r0:r1, c0:c1]
    nir = sr.values[_nearest_band(wl, 860.0), r0:r1, c0:c1]
    with np.errstate(invalid="ignore", divide="ignore"):
        land = (green - nir) / (green + nir) < 0.0
    mostly_land = np.where(land.mean(axis=1) >= 0.5)[0]
    r1 = min(r1, r0 + int(mostly_land[-1]) + 1 + SEA_MARGIN_ROWS)
    land = land[: r1 - r0]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, bands in band_index.items():
        rgb = np.dstack([_stretch(cube[b][r0:r1, c0:c1], land) for b in bands])
        Image.fromarray((rgb * 255).round().astype(np.uint8)).save(
            OUT_DIR / f"{name}.png", optimize=True
        )

    provenance = {
        "sensor": "Planet Tanager-1 imaging spectrometer, ortho_sr (surface reflectance)",
        "scene": SCENE.stem,
        "strip_id": ds.attrs.get("strip_id"),
        "acquired_utc": f"{SCENE.stem[:4]}-{SCENE.stem[4:6]}-{SCENE.stem[6:8]}T"
        f"{SCENE.stem[9:11]}:{SCENE.stem[11:13]}:{SCENE.stem[13:15]}Z",
        "crs": ds.attrs.get("crs"),
        "gsd_m": 30,
        "window_rows_cols": [r0, r1, c0, c1],
        "window_utm": [float(x[c0]), float(x[c1 - 1]), float(y[r1 - 1]), float(y[r0])],
        "size": [c1 - c0, r1 - r0],
        "bands_nm": {n: [round(float(wl[b]), 2) for b in bs] for n, bs in band_index.items()},
        "stretch": "linear 2-98 percentile per band, limits from land pixels (NDWI < 0) "
        "within the window",
        "licence": "CC BY 4.0. Tanager STAC Data, available at www.planet.com/data/stac "
        "(c) 2025 Planet Labs PBC. All Rights Reserved.",
        "script": "scripts/render_site_composites.py",
    }
    (OUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
