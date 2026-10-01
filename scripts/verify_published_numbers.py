"""Recompute every number quoted in the public figures and write-ups.

Each value is recomputed from the product rasters and the external references
(NIFC/WFIGS perimeters, CAL FIRE DINS) and printed beside the value that was
published, so a stale or mis-defined number shows up as a mismatch instead of
being carried forward.

Usage::

    python scripts/verify_published_numbers.py

Writes ``outputs/published_numbers_check.md``. The Hughes perimeter lives in its
own file, ``data/reference/perimeters/hughes_2025.geojson`` (WFIGS, discovered
2025-01-22, 10,425 ac), so it never enters the Palisades inside/outside masks.
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import social_base as sb  # noqa: E402
from palisades_common import OUTPUTS, PERIMETERS, point_indices, polygon_masks  # noqa: E402
from validate_char_fractions import auc  # noqa: E402

DNBR = "20241215_to_20250123swath2_dnbr.tif"
CHAR = OUTPUTS / "20250123swath2_frac_char.tif"
HUGHES_FRAME = dict(xmin=345150, xmax=355400, ymin=3813800, ymax=3826800)


def grid_mask(da, frame) -> np.ndarray:
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin

    x, y = da.x.values, da.y.values
    rx, ry = abs(x[1] - x[0]), abs(y[1] - y[0])
    transform = from_origin(x[0] - rx / 2, y[0] + ry / 2, rx, ry)
    return ~geometry_mask(frame.geometry, out_shape=(y.size, x.size), transform=transform)


def main() -> int:
    import geopandas as gpd
    import rioxarray  # noqa: F401
    import xarray as xr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hughes-perimeter", type=Path,
                        default=REPO_ROOT / "data" / "reference" / "perimeters" / "hughes_2025.geojson")
    parser.add_argument("--out", type=Path, default=OUTPUTS / "published_numbers_check.md")
    args = parser.parse_args()

    rows: list[tuple[str, str, str, str]] = []  # (claim, published, recomputed, definition)

    # dNBR inside / outside the Palisades perimeter, land only.
    dnbr = sb.load(DNBR, mask_water=True)
    v = dnbr.values
    fin = np.isfinite(v)
    perims = gpd.read_file(PERIMETERS).to_crs(sb.CRS)
    names = perims["poly_IncidentName"].str.upper()
    pal = grid_mask(dnbr, perims[names == "PALISADES"])
    any_fire = grid_mask(dnbr, perims)
    rows.append(("dNBR median inside Palisades", "0.588", f"{np.median(v[pal & fin]):.3f}",
                 f"land pixels inside the WFIGS perimeter, n={int((pal & fin).sum())}"))
    rows.append(("dNBR median outside, all perimeters", "0.060",
                 f"{np.median(v[~any_fire & fin]):.3f}",
                 "land outside Palisades, Franklin and Kenneth"))
    rows.append(("dNBR median outside, Palisades only", "0.046",
                 f"{np.median(v[~pal & fin]):.3f}",
                 "land outside Palisades; includes the Franklin scar"))
    for name in ("FRANKLIN", "KENNETH"):
        m = grid_mask(dnbr, perims[names == name])
        val = f"{np.median(v[m & fin]):.3f}" if (m & fin).any() else "not in dNBR footprint"
        rows.append((f"dNBR median inside {name.title()}", "-", val,
                     f"{int((m & fin).sum())} land pixels"))

    # Char fraction against the perimeters and DINS.
    char = xr.open_dataarray(CHAR, engine="rasterio").squeeze("band", drop=True)
    c = np.asarray(char.values, dtype=np.float64)
    cf = np.isfinite(c)
    inside, outside = polygon_masks(char, "PALISADES")
    rows.append(("char median inside / outside Palisades", "0.389 / 0.000",
                 f"{np.median(c[inside & cf]):.3f} / {np.median(c[outside & cf]):.3f}",
                 "outside = outside every perimeter in the reference file"))
    rows.append(("char perimeter AUC", "0.795", f"{auc(c[inside & cf], c[outside & cf]):.3f}", ""))
    for name, published in (("Franklin", "0.337"), ("KENNETH", "0.575")):
        m, _ = polygon_masks(char, name)
        rows.append((f"char median inside {name.title()}", published,
                     f"{np.median(c[m & cf]):.3f}", f"n={int((m & cf).sum())}"))

    # DINS: char on its own point set, and char vs dNBR on the shared point set.
    post = sb.load("20250123swath2_nbr.tif")
    pre = sb.load("20241215_nbr.tif").rio.reproject_match(post)
    d = (pre - post).values
    rd, cd = point_indices(char, "Destroyed (>50%)")
    rn, cn = point_indices(char, "No Damage")
    md, mn = np.isfinite(c[rd, cd]), np.isfinite(c[rn, cn])
    rows.append(("char DINS AUC", "0.530", f"{auc(c[rd, cd][md], c[rn, cn][mn]):.3f}",
                 f"destroyed/no-damage points with char modeled, n={md.sum()}/{mn.sum()}"))
    md2 = md & np.isfinite(d[rd, cd])
    mn2 = mn & np.isfinite(d[rn, cn])
    rows.append(("char vs dNBR DINS AUC, shared points", "0.530 / 0.721",
                 f"{auc(c[rd, cd][md2], c[rn, cn][mn2]):.3f} / "
                 f"{auc(d[rd, cd][md2], d[rn, cn][mn2]):.3f}",
                 f"points where both are defined, n={md2.sum()}/{mn2.sum()}"))

    # Urban-extended library: does the damage signal move onto debris?
    for var, published in (("debris", "0.672"), ("char", "0.506")):
        arr = np.asarray(xr.open_dataarray(OUTPUTS / f"20250123swath2_urbanlib_frac_{var}.tif",
                                           engine="rasterio").squeeze("band", drop=True).values,
                         dtype=np.float64)
        pos, neg = arr[rd, cd], arr[rn, cn]
        pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
        rows.append((f"urban library {var} DINS AUC", published, f"{auc(pos, neg):.3f}",
                     f"median destroyed {np.median(pos):.3f} vs no damage {np.median(neg):.3f}, "
                     f"n={pos.size}/{neg.size}"))

    # Char inside the perimeter, split by distance to the nearest surveyed structure.
    # "Built-up" = within BUILT_UP_M of any DINS point of any damage class.
    from scipy.ndimage import distance_transform_edt

    survey = sb.load_dins()
    x, y = char.x.values, char.y.values
    res = float(abs(x[1] - x[0]))
    col = np.round((survey.geometry.x.values - x[0]) / res).astype(int)
    row = np.round((y[0] - survey.geometry.y.values) / res).astype(int)
    keep = (col >= 0) & (col < x.size) & (row >= 0) & (row < y.size)
    has_structure = np.zeros(c.shape, dtype=bool)
    has_structure[row[keep], col[keep]] = True
    distance = distance_transform_edt(~has_structure) * res
    for threshold in (30, 100, 200):
        built = distance <= threshold
        wild, urb = inside & cf & ~built, inside & cf & built
        rows.append((f"char inside Palisades, wildland vs built-up ({threshold} m)",
                     "0.501 (AUC 0.887) / 0.000",
                     f"{np.median(c[wild]):.3f} (AUC {auc(c[wild], c[outside & cf]):.3f}) / "
                     f"{np.median(c[urb]):.3f} (AUC {auc(c[urb], c[outside & cf]):.3f})",
                     f"built-up = within {threshold} m of a DINS point; AUC vs outside all "
                     f"perimeters; n={int(wild.sum())}/{int(urb.sum())}"))

    # Franklin on the 2024-12-15 map (image-derived endmembers).
    fr = xr.open_dataarray(OUTPUTS / "20241215_frac_char.tif", engine="rasterio").squeeze("band", drop=True)
    fv = np.asarray(fr.values, dtype=np.float64)
    ff = np.isfinite(fv)
    f_in, f_out_all = polygon_masks(fr, "Franklin")
    f_out_one = ~grid_mask(fr, perims[names == "FRANKLIN"])
    rows.append(("Franklin char mean inside / outside (2024-12-15)", "0.490 / 0.050 or 0.047",
                 f"{fv[f_in & ff].mean():.3f} / {fv[f_out_all & ff].mean():.3f} (all) / "
                 f"{fv[f_out_one & ff].mean():.3f} (Franklin only)",
                 f"means; medians {np.median(fv[f_in & ff]):.3f} / {np.median(fv[f_out_all & ff]):.3f}; "
                 f"AUC {auc(fv[f_in & ff], fv[f_out_all & ff]):.3f}; endmembers derived from this image"))

    # Share of valid land pixels the core-library run modelled.
    from palisades_common import PALISADES_SCENE, load_masked_scene

    scene = load_masked_scene(PALISADES_SCENE)
    valid = np.isfinite(scene[scene.attrs["data_var"]].sel(wavelength=860, method="nearest").values)
    del scene
    rows.append(("share of valid land pixels modelled", "85.4%",
                 f"{100.0 * (cf & valid).sum() / valid.sum():.1f}%",
                 f"{int((cf & valid).sum()):,} of {int(valid.sum()):,} pixels left by the nodata, "
                 f"cloud and water masks; {100.0 * cf.mean():.1f}% of the full grid"))

    # Structures figure: DINS points in the published frame, dNBR at each point.
    dins = sb.load_dins()
    ext = sb.extent_of()
    inframe = dins.cx[ext[0]:ext[1], ext[2]:ext[3]]
    vals = dnbr.sel(x=xr.DataArray(inframe.geometry.x.values, dims="p"),
                    y=xr.DataArray(inframe.geometry.y.values, dims="p"), method="nearest").values
    destroyed = (inframe["DAMAGE"] == "Destroyed (>50%)").values
    undamaged = (inframe["DAMAGE"] == "No Damage").values
    ok = np.isfinite(vals)
    rows.append(("structures in frame / destroyed", "2,046 / 984",
                 f"{len(inframe):,} / {int(destroyed.sum()):,}", "CAL FIRE DINS, hero-figure frame"))
    rows.append(("dNBR median destroyed / undamaged", "0.375 / 0.234",
                 f"{np.median(vals[destroyed & ok]):.3f} / {np.median(vals[undamaged & ok]):.3f}",
                 f"in-frame points with dNBR, n={int((destroyed & ok).sum())}/{int((undamaged & ok).sum())}"))

    # Hughes recovery.
    jan = sb.load("20250123_nbr.tif")
    apr = sb.load("20250407_nbr.tif").rio.reproject_match(jan)
    sl = dict(x=slice(HUGHES_FRAME["xmin"], HUGHES_FRAME["xmax"]),
              y=slice(HUGHES_FRAME["ymax"], HUGHES_FRAME["ymin"]))
    j, a = jan.sel(**sl).values, apr.sel(**sl).values
    rows.append(("Hughes mean NBR Jan -> Apr", "0.017 -> 0.192",
                 f"{np.nanmean(j):.3f} -> {np.nanmean(a):.3f}",
                 "all finite pixels in the animation frame"))
    if args.hughes_perimeter.exists():
        hughes = gpd.read_file(args.hughes_perimeter).to_crs(sb.CRS)
        m = grid_mask(jan, hughes)
        both = m & np.isfinite(jan.values) & np.isfinite(apr.values)
        rows.append(("Hughes mean NBR Jan -> Apr, inside perimeter", "-",
                     f"{jan.values[both].mean():.3f} -> {apr.values[both].mean():.3f}",
                     f"pixels inside WFIGS Hughes perimeter covered by both scenes, "
                     f"n={int(both.sum())} of {int(m.sum())}"))

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout.strip()
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# Published numbers, recomputed",
        "",
        f"Generated {stamp} by scripts/verify_published_numbers.py. Commit: {commit}",
        "",
        "| Claim | Published | Recomputed | Definition |",
        "|---|---|---|---|",
        *[f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows],
        "",
    ]
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
