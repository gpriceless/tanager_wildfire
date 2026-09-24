"""Shared map furniture and styling for the social-media figure suite.

Everything here is about making a map *readable* to someone who has never seen
a remote-sensing product: a land mask so the ocean is not painted with burn
severity, place names that are actually where they say they are, a scale bar, a
north arrow, and a locator inset so the reader knows which part of the planet
they are looking at.
"""
from __future__ import annotations

from pathlib import Path

import contextily as ctx
import matplotlib.patheffects as pe
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Transformer

OUTPUTS = Path("outputs")
SOCIAL = Path("figures/social")

CRS = "EPSG:32611"
BASEMAP = ctx.providers.Esri.WorldImagery

# --- Palette -------------------------------------------------------------
DARK_BG = "#0b0b14"
PANEL_BG = "#12121f"
TEXT_COLOR = "#f7f7fa"
SUBTITLE_COLOR = "#9a9ab5"
ACCENT = "#ff5a5f"
CREDIT_COLOR = "#5a5a75"

CREDIT = "Planet Tanager-1 · CAL FIRE DINS · Esri imagery · github.com/gpriceless/tanager_wildfire"

_to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
_to_wgs = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)


def utm(lon: float, lat: float) -> tuple[float, float]:
    return _to_utm.transform(lon, lat)


# Place names as (lon, lat) so they can be checked against a map, then
# projected. The previous hard-coded UTM values were 7-11 km off.
PLACES_LONLAT: dict[str, tuple[float, float]] = {
    "Malibu": (-118.7798, 34.0259),
    "Pepperdine University": (-118.7096, 34.0400),
    "Malibu Creek State Park": (-118.7297, 34.0975),
    "Topanga": (-118.6015, 34.0936),
    "Las Flores Canyon": (-118.6400, 34.0350),
    "Saddle Peak": (-118.6600, 34.0800),
}

PLACES = {k: utm(*v) for k, v in PLACES_LONLAT.items()}

# The frame used by every severity map. The pre/post Tanager scenes overlap
# over land down to y=3759015, but everything below ~3762500 is open ocean, so
# the frame is trimmed to keep the coastline for context without donating a
# third of the image to empty sea.
FRAME = dict(xmin=333400, xmax=353100, ymin=3762500, ymax=3775200)


# --- Colormaps -----------------------------------------------------------

def _cmap(name: str, colors: list[str]) -> LinearSegmentedColormap:
    cm = LinearSegmentedColormap.from_list(name, colors, N=256)
    cm.set_bad(alpha=0)
    return cm


FIRE_CMAP = _cmap(
    "fire",
    ["#0a0a12", "#2b0f05", "#5c1c06", "#9c3a06", "#d96a0b",
     "#f79d3c", "#ffd08a", "#fff6e0"],
)
SEV_CMAP = _cmap(
    "sev",
    ["#1b3a6b", "#2e6f9e", "#7fb2cf", "#e8e3d3", "#f0a860",
     "#d95f2b", "#9e1b1b"],
)
WATER_CMAP = _cmap(
    "water",
    ["#4a1200", "#8b3a00", "#c98d3f", "#efe0bd", "#7fc4dd",
     "#3a7fb5", "#10306b"],
)
VEG_CMAP = _cmap(
    "veg",
    ["#231b12", "#4a3b1c", "#7a6b22", "#9bab2e", "#5fa03a",
     "#2f7d34", "#123f1f"],
)


# --- Data loading --------------------------------------------------------

_LAND_REFERENCE = "20241215_ndwi.tif"


def land_mask(template: xr.DataArray) -> np.ndarray:
    """Boolean land mask on *template*'s grid.

    The NDWI product is already water-masked by the pipeline, so its finite
    footprint is the land footprint. The dNBR product is *not* water-masked —
    open ocean carries dNBR values up to ~0.3, which a severity colormap
    renders as "moderate burn". Without this mask roughly 62% of the painted
    pixels in the severity figures are sea surface.
    """
    ndwi = rioxarray.open_rasterio(OUTPUTS / _LAND_REFERENCE).squeeze(
        "band", drop=True
    )
    aligned = ndwi.reindex_like(template, method="nearest", tolerance=1)
    return np.isfinite(aligned.values)


def load(name: str, mask_water: bool = False) -> xr.DataArray:
    da = rioxarray.open_rasterio(OUTPUTS / name).squeeze("band", drop=True)
    da = da.where(da != da.rio.nodata)
    if mask_water:
        da = da.where(land_mask(da))
    return da


def crop(da: xr.DataArray, frame: dict | None = None) -> xr.DataArray:
    f = frame or FRAME
    return da.sel(
        x=slice(f["xmin"], f["xmax"]), y=slice(f["ymax"], f["ymin"])
    )


def extent_of(frame: dict | None = None) -> list[float]:
    f = frame or FRAME
    return [f["xmin"], f["xmax"], f["ymin"], f["ymax"]]


def load_dins() -> "object":
    """CAL FIRE Damage Inspection points for the Palisades Fire."""
    import geopandas as gpd

    return gpd.read_file("data/reference/dins/palisades_dins.geojson").to_crs(CRS)


# --- Map furniture -------------------------------------------------------

def style_dark(fig, axes) -> None:
    fig.patch.set_facecolor(DARK_BG)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#1e1e33")


def add_basemap(ax, alpha: float = 0.55) -> None:
    """Satellite basemap without the wall of Esri attribution text.

    The default attribution string is ~150 characters and lands on top of the
    figure credit. It is replaced by a compact source line in :func:`credit`.
    """
    try:
        ctx.add_basemap(
            ax, crs=CRS, source=BASEMAP, alpha=alpha, zorder=0, attribution=False
        )
    except Exception:
        pass


def _halo(lw: float = 2.2):
    return [pe.withStroke(linewidth=lw, foreground="#000000cc")]


def label_places(
    ax,
    fontsize: float = 9.5,
    color: str = "#ffffff",
    avoid: list[tuple[float, float, float, float]] | None = None,
) -> None:
    """Drop place pins on the map.

    Skips pins outside the current view, and pins that fall inside one of the
    *avoid* rectangles — given in axes fraction as ``(x0, y0, x1, y1)`` — which
    is how headline plates keep from being written over by a place name.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    for name, (px, py) in PLACES.items():
        if not (min(x0, x1) < px < max(x0, x1) and min(y0, y1) < py < max(y0, y1)):
            continue
        if avoid:
            fx = (px - x0) / (x1 - x0)
            fy = (py - y0) / (y1 - y0)
            if any(ax0 <= fx <= ax1 and ay0 <= fy <= ay1
                   for ax0, ay0, ax1, ay1 in avoid):
                continue
        ax.plot(
            px, py, "o", color=color, markersize=4.5, zorder=20,
            markeredgecolor="#000000", markeredgewidth=0.8,
        )
        ax.annotate(
            name, (px, py), textcoords="offset points", xytext=(7, 4),
            fontsize=fontsize, color=color, fontweight="bold", zorder=20,
            path_effects=_halo(),
        )


def scalebar(
    ax, length_km: float = 5.0, pad: float = 0.04, corner: str = "right"
) -> None:
    """Scale bar in data (metre) coordinates, anchored to a bottom corner."""
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    w, h = x1 - x0, y1 - y0
    length = length_km * 1000.0
    if corner == "left":
        bx0 = x0 + pad * w
        bx1 = bx0 + length
    else:
        bx1 = x1 - pad * w
        bx0 = bx1 - length
    by = y0 + pad * h
    bh = 0.011 * h
    ax.add_patch(
        __import__("matplotlib").patches.Rectangle(
            (bx0, by), length, bh, facecolor="#ffffff", edgecolor="#000000",
            linewidth=0.8, zorder=22,
        )
    )
    ax.text(
        (bx0 + bx1) / 2, by + bh * 1.5, f"{length_km:g} km",
        ha="center", va="bottom", fontsize=9, color="#ffffff",
        fontweight="bold", zorder=22, path_effects=_halo(1.8),
    )


def north_arrow(ax, pad: float = 0.04, corner: str = "right") -> None:
    """North arrow, anchored above the scale bar.

    Kept out of the top-left corner, which every figure uses for its headline.
    """
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    w, h = x1 - x0, y1 - y0
    if corner == "left":
        ax_x = x0 + pad * w + 0.035 * w
    else:
        ax_x = x1 - pad * w - 0.035 * w
    base = y0 + pad * h + 0.055 * h
    ax.annotate(
        "N", xy=(ax_x, base + 0.075 * h), xytext=(ax_x, base),
        ha="center", va="center", fontsize=12, fontweight="bold",
        color="#ffffff", zorder=22, path_effects=_halo(1.8),
        arrowprops=dict(arrowstyle="-|>", color="#ffffff", lw=1.6,
                        shrinkA=3, shrinkB=0),
    )


def locator_inset(ax, loc=(0.735, 0.055, 0.235, 0.30)) -> None:
    """Small California map with the study footprint marked.

    Coastline comes from Esri's imagery tiles, which need no API key, so this
    works from a clean clone with no extra downloads or credentials.
    """
    ins = ax.inset_axes(loc)
    ins.set_facecolor(PANEL_BG)
    cx, cy = utm(-118.6, 34.05)
    half_x, half_y = 330000, 300000
    ins.set_xlim(cx - half_x, cx + half_x * 0.55)
    ins.set_ylim(cy - half_y * 0.5, cy + half_y)
    try:
        ctx.add_basemap(
            ins, crs=CRS, source=BASEMAP, alpha=0.75, zorder=0,
            attribution=False,
        )
    except Exception:
        pass
    ins.plot(
        cx, cy, marker="o", color=ACCENT, markersize=8, zorder=5,
        markeredgecolor="#ffffff", markeredgewidth=1.2,
    )
    ins.text(
        cx + 22000, cy + 12000, "this map", ha="left", va="bottom",
        fontsize=8, color="#ffffff", fontweight="bold",
        path_effects=_halo(1.8), zorder=5,
    )
    ins.text(
        0.5, 0.965, "SOUTHERN CALIFORNIA", transform=ins.transAxes,
        ha="center", va="top", fontsize=7, color="#ffffff",
        fontweight="bold", path_effects=_halo(1.8), zorder=5,
    )
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_color("#5a5a7a")
        s.set_linewidth(1.2)
        s.set_visible(True)


def headline(
    ax,
    title: str,
    subtitle: str = "",
    *,
    y: float = 0.975,
    title_size: float = 21,
    sub_size: float = 11.5,
    sub_color: str | None = None,
    width: float = 0.56,
) -> None:
    """Title block on a dark plate so it stays legible over bright basemap.

    Satellite basemaps swing from near-black ocean to bright bare rock inside a
    single figure, so plain text over them is unreadable somewhere in every
    frame.
    """
    n_sub = subtitle.count("\n") + 1 if subtitle else 0
    height = 0.058 * (title_size / 21) + 0.040 * n_sub + 0.028
    ax.add_patch(
        __import__("matplotlib").patches.FancyBboxPatch(
            (0.018, y - height), width, height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            transform=ax.transAxes, facecolor="#07070fd9",
            edgecolor="#2e2e4a", linewidth=1.0, zorder=9,
        )
    )
    ax.text(
        0.034, y - 0.012, title, transform=ax.transAxes,
        fontsize=title_size, fontweight="bold", color=TEXT_COLOR,
        va="top", zorder=11,
    )
    if subtitle:
        ax.text(
            0.034, y - 0.018 - 0.048 * (title_size / 21), subtitle,
            transform=ax.transAxes, fontsize=sub_size,
            color=sub_color or SUBTITLE_COLOR, va="top", zorder=11,
            linespacing=1.6,
        )


def credit(ax, right: bool = True, text: str | None = None) -> None:
    ha = "right" if right else "left"
    x = 0.985 if right else 0.015
    ax.text(
        x, 0.014, text or CREDIT, transform=ax.transAxes, fontsize=7,
        color=CREDIT_COLOR, ha=ha, va="bottom", zorder=25,
    )


def map_furniture(
    ax,
    scale_km: float = 5.0,
    inset: bool = False,
    avoid: list[tuple[float, float, float, float]] | None = None,
) -> None:
    """Apply the standard set: place names, scale bar, north arrow."""
    ax.set_xticks([])
    ax.set_yticks([])
    # The locator inset owns the bottom-right corner when present, so the
    # scale bar and north arrow move to the bottom-left to avoid stacking.
    label_places(ax, avoid=avoid)
    scalebar(ax, scale_km, corner="left" if inset else "right")
    north_arrow(ax, corner="left" if inset else "right")
    if inset:
        locator_inset(ax)


def save(fig, name: str, dpi: int = 200) -> Path:
    SOCIAL.mkdir(parents=True, exist_ok=True)
    path = SOCIAL / name
    fig.savefig(path, dpi=dpi, facecolor=DARK_BG, bbox_inches="tight")
    return path
