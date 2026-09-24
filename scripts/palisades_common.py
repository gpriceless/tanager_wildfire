"""Shared loading and reference-geometry helpers for the Palisades MESMA work.

Kept in one place so the unmixing run, the library comparison, and the
validation all label pixels the same way. Every "burned" / "unburned" label in
here comes from an external vector source — the official interagency fire
perimeter or the CAL FIRE DINS structure survey — never from a spectral
threshold on the scene being tested.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

log = logging.getLogger(__name__)

PALISADES_SCENE = REPO_ROOT / "data" / "raw" / "fire" / "20250123_185518_92_4001_ortho_sr_hdf5.h5"
FRANKLIN_SCENE = REPO_ROOT / "data" / "raw" / "fire" / "20241215_185916_33_4001_ortho_sr_hdf5.h5"
PERIMETERS = REPO_ROOT / "data" / "reference" / "perimeters" / "la_fires_2025.geojson"
DINS = REPO_ROOT / "data" / "reference" / "dins" / "palisades_dins.geojson"
LIBRARY_CSV = REPO_ROOT / "data" / "reference" / "endmembers" / "palisades_fire_library.csv"
OUTPUTS = REPO_ROOT / "outputs"

SCENE_CRS = "EPSG:32611"

# DINS damage classes. "Destroyed (>50%)" and "No Damage" are the two ends the
# issue asks about; the partial classes are reported but not used as either
# pole, since a structure that is 10-25% damaged sits inside a burned block.
DINS_DESTROYED = "Destroyed (>50%)"
DINS_NO_DAMAGE = "No Damage"


def load_library(path: Path = LIBRARY_CSV) -> dict:
    """Read the endmember library CSV written by build_fire_endmember_library.py."""
    frame = pd.read_csv(path, index_col=0)
    categories = frame.pop("category").to_numpy(dtype=str)
    usgs_names = frame.pop("usgs_name").to_numpy(dtype=str)
    return {
        "spectrum_id": frame.index.to_numpy(dtype=str),
        "category": categories,
        "usgs_name": usgs_names,
        "wavelength": frame.columns.to_numpy(dtype=float),
        "reflectance": frame.to_numpy(dtype=np.float32),
    }


def library_dataarray(library: dict, keep: Iterable[str] | None = None) -> xr.DataArray:
    """Turn the library dict into the ``(spectrum_id, wavelength)`` DataArray MESMA wants."""
    values = library["reflectance"]
    ids = library["spectrum_id"]
    cats = library["category"]
    names = library["usgs_name"]
    if keep is not None:
        keep_set = set(keep)
        sel = np.array([sid in keep_set for sid in ids])
        values, ids, cats, names = values[sel], ids[sel], cats[sel], names[sel]
    return xr.DataArray(
        values,
        dims=("spectrum_id", "wavelength"),
        coords={
            "spectrum_id": ids,
            "wavelength": library["wavelength"].astype(np.float32),
            "category": ("spectrum_id", cats),
            "name": ("spectrum_id", names),
            "source": ("spectrum_id", np.array(["usgs_v7"] * len(ids))),
        },
    )


def load_masked_scene(path: Path) -> xr.Dataset:
    """Load a Tanager ortho_sr scene with the pipeline's standard masks applied."""
    from tanager.io import load_ortho_scene
    from tanager.masks import apply_masks, cloud_mask, nodata_mask, water_mask
    from tanager.spectral import clamp_reflectance

    scene = load_ortho_scene(path)
    masks = [nodata_mask(scene)]
    for name, fn in (("cloud", cloud_mask), ("water", water_mask)):
        try:
            masks.append(fn(scene, filepath=path) if name == "cloud" else fn(scene))
        except Exception as exc:  # pragma: no cover - depends on scene metadata
            log.info("%s mask not applied: %s", name, exc)
    masked = apply_masks(scene, masks)
    primary = masked.attrs.get("data_var") or "surface_reflectance"
    masked[primary] = clamp_reflectance(masked[primary])
    # toa_radiance is an alias of the same cube in the ortho_sr reader and
    # doubles peak memory on a 1063x957x426 scene for no benefit here.
    return masked.drop_vars([v for v in masked.data_vars if v != primary])


def _pixel_grid(scene: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(scene.coords["x"].values, dtype=np.float64),
        np.asarray(scene.coords["y"].values, dtype=np.float64),
    )


def polygon_masks(scene: xr.Dataset | xr.DataArray, incident: str,
                  perimeters: Path = PERIMETERS) -> tuple[np.ndarray, np.ndarray]:
    """Return (inside, outside) boolean masks on the scene grid for one fire.

    ``outside`` excludes every fire in the perimeter file, not just the named
    one, so an adjacent burn scar cannot end up in the unburned reference.
    """
    import geopandas as gpd
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin

    frame = gpd.read_file(perimeters).to_crs(SCENE_CRS)
    match = frame[frame["poly_IncidentName"].str.upper() == incident.upper()]
    if match.empty:
        raise ValueError(f"no perimeter named {incident!r} in {perimeters}")

    x, y = _pixel_grid(scene)
    res_x = float(abs(x[1] - x[0]))
    res_y = float(abs(y[1] - y[0]))
    transform = from_origin(x[0] - res_x / 2.0, y[0] + res_y / 2.0, res_x, res_y)
    shape = (y.size, x.size)

    inside = ~geometry_mask(match.geometry, out_shape=shape, transform=transform, invert=False)
    any_fire = ~geometry_mask(frame.geometry, out_shape=shape, transform=transform, invert=False)
    return inside, ~any_fire


def dins_points(damage: str, dins: Path = DINS) -> "gpd.GeoDataFrame":  # noqa: F821
    """CAL FIRE DINS points of one damage class, projected to the scene CRS."""
    import geopandas as gpd

    frame = gpd.read_file(dins).to_crs(SCENE_CRS)
    return frame[frame["DAMAGE"] == damage]


def point_indices(scene: xr.Dataset | xr.DataArray, damage: str) -> tuple[np.ndarray, np.ndarray]:
    """Row/col indices of the pixels containing DINS points of one damage class."""
    x, y = _pixel_grid(scene)
    res_x = float(abs(x[1] - x[0]))
    res_y = float(abs(y[1] - y[0]))
    pts = dins_points(damage)
    px = np.asarray(pts.geometry.x.values, dtype=np.float64)
    py = np.asarray(pts.geometry.y.values, dtype=np.float64)

    col = np.round((px - x[0]) / res_x).astype(int)
    row = np.round((y[0] - py) / res_y).astype(int)
    keep = (col >= 0) & (col < x.size) & (row >= 0) & (row < y.size)
    return row[keep], col[keep]


def point_samples(scene: xr.Dataset | xr.DataArray) -> dict[str, np.ndarray]:
    """Boolean masks for the DINS destroyed / no-damage pixel sets."""
    out: dict[str, np.ndarray] = {}
    x, y = _pixel_grid(scene)
    for label, damage in (("dins_destroyed", DINS_DESTROYED), ("dins_no_damage", DINS_NO_DAMAGE)):
        row, col = point_indices(scene, damage)
        mask = np.zeros((y.size, x.size), dtype=bool)
        mask[row, col] = True
        out[label] = mask
    return out
