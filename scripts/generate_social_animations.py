#!/usr/bin/env python3
"""Build blink-comparator GIFs from the Tanager scenes.

**On why there is no single before -> after -> recovery animation.** The four
downloaded scenes form two disjoint pairs, not one time series over one place:

    2024-12-15  Palisades footprint  (lat 33.92-34.11)  pre-fire
    2025-01-23  swath 2, Palisades   (lat 33.92-34.11)  post-fire
    2025-01-23  swath 1, Hughes      (lat 34.38-34.67)  post-fire
    2025-04-07  Hughes footprint     (lat 34.40-34.64)  recovery

The two Jan 23 swaths are 11 seconds apart on the same descending pass, which
puts them ~80 km apart along-track. So Palisades has pre and post but no
recovery, and Hughes has post and recovery but no pre. Animating all four as
one arc would imply we watched one landscape burn and recover; we did not.

Two honest animations instead, each over a single footprint with a single
colour scale. Frames are hard cuts, not crossfades — a crossfade would render
intermediate states nobody observed.

Usage::

    python3 scripts/generate_social_animations.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import rioxarray  # noqa: F401
import social_base as sb
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUTS = Path("outputs")
SWATH2_NBR = OUTPUTS / "20250123swath2_nbr.tif"
POST_SCENE = Path("data/raw/fire/20250123_185518_92_4001_ortho_sr_hdf5.h5")

# NBR: high = intact vegetation, low = burned. Kept identical across both GIFs
# so the two animations can be read against each other.
NBR_CMAP = sb._cmap(
    "nbr",
    ["#1a0d05", "#4a2410", "#8a4a1e", "#c08a4a", "#d9c88a",
     "#9bbf5a", "#4f9b3a", "#1f6b2c"],
)
NBR_VMIN, NBR_VMAX = -0.35, 0.65

# The Jan/Apr overlap is a 10 x 22 km strip, too tall for a landscape canvas.
# Cropped to a 4:5-ish window centred on the recovering burn (centroid
# 349935, 3820305) and rendered portrait, which also suits the feed.
HUGHES_FRAME = dict(xmin=345150, xmax=355400, ymin=3813800, ymax=3826800)


def ensure_swath2_nbr() -> None:
    """Compute post-fire NBR for the Palisades swath from the raw cube.

    Derived from the measured reflectance rather than inverted out of dNBR, so
    the animation shows two measurements rather than one measurement and one
    piece of arithmetic.
    """
    if SWATH2_NBR.exists():
        print(f"  using cached {SWATH2_NBR.name}")
        return
    if not POST_SCENE.exists():
        raise SystemExit(f"Need {POST_SCENE} to compute post-fire NBR.")

    from tanager.io import load_ortho_scene
    from tanager.spectral import nbr

    print(f"  computing NBR from {POST_SCENE.name} ...")
    scene = load_ortho_scene(POST_SCENE)
    values = nbr(scene)
    values = values.rio.write_crs(scene.attrs.get("crs", sb.CRS))
    values.rio.to_raster(SWATH2_NBR)
    print(f"  wrote {SWATH2_NBR}")


def _frame(
    data, extent, frame, date_label, caption, title, perimeter, place_labels,
    scale_km, footnote, figsize=(11, 7.4), headline_width=0.56,
    date_size=26,
):
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    sb.style_dark(fig, ax)

    ax.imshow(
        np.ma.masked_invalid(data), extent=extent, origin="upper",
        cmap=NBR_CMAP, vmin=NBR_VMIN, vmax=NBR_VMAX,
        interpolation="nearest", alpha=0.9, zorder=2,
    )
    sb.add_basemap(ax, alpha=0.45)
    ax.set_xlim(frame["xmin"], frame["xmax"])
    ax.set_ylim(frame["ymin"], frame["ymax"])

    if perimeter:
        sb.overlay_perimeter(ax, perimeter, color="#ffe9a8", linewidth=1.6)

    ax.set_xticks([])
    ax.set_yticks([])
    if place_labels:
        sb.label_places(ax, avoid=[(0.0, 0.80, 0.58, 1.0), (0.62, 0.84, 1.0, 1.0)])
    sb.scalebar(ax, scale_km)
    sb.north_arrow(ax)

    sb.headline(ax, title, caption, title_size=19, width=headline_width)

    # The date chip is the thing that changes between frames, so it is the
    # loudest element on the image.
    ax.text(
        0.975, 0.955, date_label, transform=ax.transAxes, ha="right", va="top",
        fontsize=date_size, fontweight="bold", color="#ffffff", zorder=24,
        path_effects=[pe.withStroke(linewidth=4.5, foreground="#000000cc")],
    )
    ax.text(
        0.975, 0.895, footnote, transform=ax.transAxes, ha="right", va="top",
        fontsize=11, color="#ffe9a8", zorder=24,
        path_effects=[pe.withStroke(linewidth=3.0, foreground="#000000cc")],
    )

    ax.text(
        0.022, 0.035,
        "Normalized Burn Ratio · green = intact vegetation, brown = burned",
        transform=ax.transAxes, fontsize=9.5, color="#d8d8e8", va="bottom",
        zorder=24, path_effects=[pe.withStroke(linewidth=2.6,
                                               foreground="#000000dd")],
    )
    sb.credit(ax)
    fig.tight_layout(pad=0.4)

    fig.canvas.draw()
    image = Image.frombuffer(
        "RGBA", fig.canvas.get_width_height(),
        fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1,
    ).convert("RGB")
    plt.close(fig)
    return image


def write_gif(frames, path: Path, durations, colors: int = 200) -> None:
    """Write a looping GIF, trimming the palette until it fits Bluesky's 1 MB cap."""
    for palette in (colors, 128, 96, 64):
        quantized = [
            f.quantize(colors=palette, method=Image.MEDIANCUT,
                       dither=Image.FLOYDSTEINBERG)
            for f in frames
        ]
        quantized[0].save(
            path, save_all=True, append_images=quantized[1:], loop=0,
            duration=durations, optimize=True, disposal=2,
        )
        if path.stat().st_size <= 1_000_000:
            break
    print(f"      -> {path} ({path.stat().st_size / 1e6:.2f} MB, "
          f"{palette}-colour palette)")
    return

def palisades_gif() -> None:
    print("  [1] Palisades: before -> after ...")
    ensure_swath2_nbr()

    pre = sb.crop(sb.load("20241215_nbr.tif", mask_water=True))
    post_full = sb.load(SWATH2_NBR.name, mask_water=True)
    post = sb.crop(post_full.rio.reproject_match(sb.load("20241215_nbr.tif")))
    ext = sb.extent_of()

    frames = [
        _frame(pre.values, ext, sb.FRAME, "15 Dec 2024",
               "Chaparral on the Santa Monica Mountains, three weeks before\n"
               "the Palisades Fire. The yellow outline is where it will burn.",
               "Before", "Palisades", True, 5,
               "23,448 acres · ignites 7 Jan"),
        _frame(post.values, ext, sb.FRAME, "23 Jan 2025",
               "Sixteen days after ignition. Everything inside the outline has\n"
               "stopped reflecting like living vegetation.",
               "After", "Palisades", True, 5,
               "median dNBR inside: 0.588"),
    ]
    write_gif(frames, sb.SOCIAL / "palisades_before_after.gif", [2200, 2200])


def hughes_gif() -> None:
    print("  [2] Hughes: burned -> recovering ...")
    jan = sb.load("20250123_nbr.tif")
    apr = sb.load("20250407_nbr.tif").rio.reproject_match(jan)

    f = HUGHES_FRAME
    ext = [f["xmin"], f["xmax"], f["ymin"], f["ymax"]]
    jan_c = jan.sel(x=slice(f["xmin"], f["xmax"]), y=slice(f["ymax"], f["ymin"]))
    apr_c = apr.sel(x=slice(f["xmin"], f["xmax"]), y=slice(f["ymax"], f["ymin"]))

    frames = [
        _frame(jan_c.values, ext, f, "23 Jan 2025",
               "The Hughes Fire near Castaic, one day after\n"
               "the fire was stopped. Brown is bare ground.",
               "Burned", None, True, 3, "mean NBR 0.017",
               figsize=(8.2, 10.2), headline_width=0.70, date_size=21),
        _frame(apr_c.values, ext, f, "7 Apr 2025",
               "Seventy-four days later, after a wet spring.\n"
               "Green is vegetation grown back since the fire.",
               "Recovering", None, True, 3, "mean NBR 0.192",
               figsize=(8.2, 10.2), headline_width=0.70, date_size=21),
    ]
    write_gif(frames, sb.SOCIAL / "hughes_recovery.gif", [2200, 2200])


if __name__ == "__main__":
    matplotlib.use("Agg")
    sb.SOCIAL.mkdir(parents=True, exist_ok=True)
    print("Generating blink-comparator animations\n")
    palisades_gif()
    hughes_gif()
    print("\nDone.")
