#!/usr/bin/env python3
"""Extract spectra and datacube faces from a Tanager scene for social figures.

Pulls three things out of the 426-band surface-reflectance cube and caches them
to a small ``.npz`` so the figure script stays fast and re-runnable:

1. Mean spectral signatures for burned / unburned / soil pixels, with the pixel
   classes taken from the MESMA fraction rasters (not invented).
2. A true-colour and two false-colour RGB composites built from real bands.
3. The cube faces needed to draw a hyperspectral "datacube" graphic.

Usage::

    python3 scripts/extract_spectral_cache.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from tanager.io import load_ortho_scene

RAW = Path("data/raw/fire")
OUTPUTS = Path("outputs")
CACHE = OUTPUTS / "social_spectral_cache.npz"

# Post-fire Palisades swath — the scene the severity figures are built from.
POST_SCENE = RAW / "20250123_185518_92_4001_ortho_sr_hdf5.h5"

# Band centres used for the RGB composites (nm). Chosen to match how Planet
# presents Tanager first-light imagery: one natural-colour and two false-colour
# renderings, each "3 of the 426 bands".
COMPOSITES = {
    "true_color": (650.0, 550.0, 470.0),
    "nir_false_color": (860.0, 650.0, 550.0),
    "swir_burn": (2200.0, 1600.0, 860.0),
}

# UTM 11N box used for the datacube graphic — the Palisades burn area. Chosen
# so the southern and eastern faces of the cube, which are what the viewer
# actually sees, sit on land rather than on open ocean.
CUBE_WINDOW_UTM = (346000.0, 358000.0, 3768000.0, 3778000.0)

# Deep atmospheric water-vapour absorption regions. Reflectance retrievals are
# unreliable here, so these bands are dropped from the plotted spectra rather
# than drawn as if they were measurements.
WATER_VAPOUR_GAPS = [(1340.0, 1465.0), (1790.0, 1955.0)]


def _nearest_band(wavelengths: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(wavelengths - target)))


def _stretch(band: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Percentile contrast stretch to [0, 1], preserving NaN as NaN."""
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.full_like(band, np.nan)
    vmin, vmax = np.percentile(finite, [lo, hi])
    if vmax <= vmin:
        return np.zeros_like(band)
    return np.clip((band - vmin) / (vmax - vmin), 0.0, 1.0)


def _window_indices(
    x: np.ndarray, y: np.ndarray, bounds: tuple[float, float, float, float]
) -> tuple[int, int, int, int]:
    """Array row/column slice covering a UTM ``(xmin, xmax, ymin, ymax)`` box."""
    xmin, xmax, ymin, ymax = bounds
    c0 = int(np.searchsorted(x, xmin))
    c1 = int(np.searchsorted(x, xmax))
    r0 = int(np.searchsorted(-y, -ymax))
    r1 = int(np.searchsorted(-y, -ymin))
    return r0, r1, c0, c1


def _class_masks(
    sr: np.ndarray, wl: np.ndarray, x: np.ndarray, y: np.ndarray
) -> dict[str, np.ndarray]:
    """Label pixels burned / unburned / soil on the scene's own grid.

    The burn split comes from the pipeline's dNBR product — the same raster the
    severity figures key on — resampled onto the scene grid. The vegetation and
    soil splits use NDVI and NDWI computed directly from this cube, because the
    NDVI product on disk for 2025-01-23 belongs to the other (Hughes) swath and
    does not overlap this footprint at all.
    """
    dnbr = rioxarray.open_rasterio(
        OUTPUTS / "20241215_to_20250123swath2_dnbr.tif"
    ).squeeze("band", drop=True)
    target_x = xr.DataArray(x, dims="x", coords={"x": x})
    target_y = xr.DataArray(y, dims="y", coords={"y": y})
    d = dnbr.interp(x=target_x, y=target_y, method="nearest").values

    green = sr[_nearest_band(wl, 560.0)]
    red = sr[_nearest_band(wl, 665.0)]
    nir = sr[_nearest_band(wl, 860.0)]

    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red)
        ndwi = (green - nir) / (green + nir)

    land = np.isfinite(ndwi) & (ndwi < 0.0)
    unburned = np.isfinite(d) & (d < 0.1) & land

    return {
        # High-severity burn: strong NBR drop between pre- and post-fire.
        "burned": np.isfinite(d) & (d > 0.44) & land,
        # Unburned green vegetation: no NBR change and still vigorous.
        "unburned_veg": unburned & np.isfinite(ndvi) & (ndvi > 0.45),
        # Bare / sparsely vegetated ground, unburned.
        "bare_soil": unburned & np.isfinite(ndvi) & (ndvi < 0.15),
    }


def main() -> None:
    if not POST_SCENE.exists():
        raise SystemExit(f"Scene not found: {POST_SCENE}")

    print(f"Loading {POST_SCENE.name} ...")
    ds = load_ortho_scene(POST_SCENE)
    sr = ds["surface_reflectance"]
    wl = ds["wavelength"].values.astype(float)
    x = ds["x"].values.astype(float)
    y = ds["y"].values.astype(float)
    print(f"  cube: {sr.shape} bands x rows x cols, {wl.min():.0f}-{wl.max():.0f} nm")

    good = (
        ds["good_wavelengths"].values.astype(bool)
        if "good_wavelengths" in ds
        else np.ones_like(wl, dtype=bool)
    )
    for lo, hi in WATER_VAPOUR_GAPS:
        good &= ~((wl >= lo) & (wl <= hi))

    masks = _class_masks(sr.values, wl, x, y)
    for name, m in masks.items():
        print(f"  {name}: {int(m.sum()):,} pixels")

    # --- 1. Mean spectra per class -----------------------------------------
    spectra: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for name, m in masks.items():
        if m.sum() < 50:
            print(f"  ! {name} has too few pixels ({int(m.sum())}), skipping")
            continue
        # Cap the sample so the mean stays cheap on a 1.7 GB cube.
        idx = np.argwhere(m)
        if idx.shape[0] > 4000:
            rng = np.random.default_rng(20260924)
            idx = idx[rng.choice(idx.shape[0], 4000, replace=False)]
        vals = sr.values[:, idx[:, 0], idx[:, 1]]
        spectra[name] = np.nanmean(vals, axis=1)
        counts[name] = int(idx.shape[0])

    # --- 2. RGB composites --------------------------------------------------
    # Built on the same land window as the cube, and stretched within it, so
    # the contrast is set by the scene content rather than by nodata and sea.
    wr0, wr1, wc0, wc1 = _window_indices(x, y, CUBE_WINDOW_UTM)
    composites: dict[str, np.ndarray] = {}
    composite_bands: dict[str, np.ndarray] = {}
    for name, targets in COMPOSITES.items():
        bands = [_nearest_band(wl, t) for t in targets]
        rgb = np.dstack(
            [_stretch(sr.values[b, wr0:wr1, wc0:wc1]) for b in bands]
        )
        composites[name] = rgb.astype(np.float32)
        composite_bands[name] = wl[bands]
        print(f"  {name}: bands at {np.round(wl[bands], 1)} nm")

    # --- 3. Datacube faces --------------------------------------------------
    # The cube's front and right faces are the reflectance profiles along its
    # southern and eastern edges, so the window has to sit entirely on land and
    # entirely inside the swath — otherwise those faces render as black panels
    # of sea-surface and nodata. CUBE_WINDOW_UTM is the Palisades burn area:
    # fully valid, and land along both exposed edges.
    r0, r1, c0, c1 = _window_indices(x, y, CUBE_WINDOW_UTM)
    edge_ok = np.isfinite(sr.values[_nearest_band(wl, 860.0), r0:r1, c0:c1])
    if not edge_ok.all():
        raise SystemExit(
            f"CUBE_WINDOW_UTM contains {int((~edge_ok).sum())} nodata pixels; "
            "adjust the window."
        )
    print(f"  cube window: rows {r0}:{r1}, cols {c0}:{c1}")
    cube_top = composites["true_color"]
    cube_right = sr.values[:, r0:r1, c1 - 1].astype(np.float32)
    cube_bottom = sr.values[:, r1 - 1, c0:c1].astype(np.float32)

    np.savez_compressed(
        CACHE,
        wavelength=wl,
        good_wavelengths=good,
        x=x,
        y=y,
        spectra_names=np.array(list(spectra.keys())),
        spectra=np.array([spectra[k] for k in spectra]),
        spectra_counts=np.array([counts[k] for k in spectra]),
        composite_names=np.array(list(composites.keys())),
        **{f"composite_{k}": v for k, v in composites.items()},
        **{f"composite_bands_{k}": v for k, v in composite_bands.items()},
        cube_top=cube_top,
        cube_right=cube_right,
        cube_bottom=cube_bottom,
        scene_id=np.array(POST_SCENE.stem),
    )
    size_mb = CACHE.stat().st_size / 1e6
    print(f"\nWrote {CACHE} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
