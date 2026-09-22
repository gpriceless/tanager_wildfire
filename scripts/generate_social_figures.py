#!/usr/bin/env python3
"""Generate social-media-ready figures from FireSpec outputs.

V2: Story-first. Every figure starts with human context (place names,
emotional weight, the Tanager-1 backstory) and layers the science on top.
A random person scrolling LinkedIn should understand what they're looking
at and why it matters before seeing a single acronym.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import contextily as ctx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Transformer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUTS = Path("outputs")
SOCIAL = Path("figures/social")
SOCIAL.mkdir(parents=True, exist_ok=True)

CRS = "EPSG:32611"
BASEMAP = ctx.providers.Esri.WorldImagery

DARK_BG = "#0f0f1a"
TEXT_COLOR = "#f5f5f5"
SUBTITLE_COLOR = "#a0a0b8"
ACCENT = "#e94560"
CREDIT_COLOR = "#555570"

# UTM -> lat/lon for place labels
_transformer = Transformer.from_crs("EPSG:32611", "EPSG:4326", always_xy=True)

# Recognizable places within the burn scar (UTM 11N coords)
PLACES = {
    "Pacific Palisades": (348500, 3772500),
    "Topanga": (345500, 3770000),
    "Malibu": (344200, 3767200),
    "Santa Monica Mtns": (349000, 3769000),
}


def _load(name: str) -> xr.DataArray:
    da = rioxarray.open_rasterio(OUTPUTS / name).squeeze("band", drop=True)
    da = da.where(da != da.rio.nodata)
    return da


def _add_basemap(ax, alpha=0.5):
    try:
        ctx.add_basemap(ax, crs=CRS, source=BASEMAP, alpha=alpha, zorder=0)
    except Exception:
        pass


def _style_dark(fig, axes):
    fig.patch.set_facecolor(DARK_BG)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#222240")


def _label_places(ax, places=None, fontsize=9, color="#dddddd"):
    """Drop recognizable place-name pins on map axes (UTM coords)."""
    for name, (px, py) in (places or PLACES).items():
        ax.plot(px, py, "o", color=color, markersize=4, zorder=20,
                markeredgecolor="#00000066", markeredgewidth=0.5)
        ax.annotate(
            name, (px, py), textcoords="offset points",
            xytext=(6, 4), fontsize=fontsize, color=color,
            fontweight="bold", zorder=20,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#00000088",
                      edgecolor="none"),
        )


def _credit(ax, right=True):
    ha = "right" if right else "left"
    x = 0.98 if right else 0.02
    ax.text(x, 0.02,
            "Planet Tanager-1 · Analysis: github.com/gpriceless/tanager_wildfire",
            transform=ax.transAxes, fontsize=7.5, color=CREDIT_COLOR,
            ha=ha, va="bottom", zorder=10)


# --- Colormaps ---

_fire_colors = ["#000000", "#1a0a00", "#4d1a00", "#993300", "#e65c00",
                "#ff9933", "#ffcc66", "#ffffff"]
FIRE_CMAP = LinearSegmentedColormap.from_list("fire", _fire_colors, N=256)
FIRE_CMAP.set_bad(alpha=0)

_sev_colors = ["#2166ac", "#67a9cf", "#d1e5f0", "#fddbc7", "#ef8a62", "#b2182b"]
SEV_CMAP = LinearSegmentedColormap.from_list("sev", _sev_colors, N=256)
SEV_CMAP.set_bad(alpha=0)

_water_colors = ["#4a0000", "#8b0000", "#cd853f", "#f5deb3",
                 "#87ceeb", "#4682b4", "#00008b"]
WATER_CMAP = LinearSegmentedColormap.from_list("water", _water_colors, N=256)
WATER_CMAP.set_bad(alpha=0)

# ---------------------------------------------------------------------------
# FIGURE 1: The before/after — emotional anchor
# This is the first thing people should see. Green hills → burn scar.
# No acronyms. Just "before" and "after" with place names they recognize.
# ---------------------------------------------------------------------------

def fig1_before_after():
    """Before/after: green hills → burn scar with neighborhood labels."""
    print("  [1/5] Before & after with place names...")
    pre = _load("20241215_ndvi.tif")
    dnbr = _load("20241215_to_20250123swath2_dnbr.tif")

    xmin, xmax = 343000, 353100
    yhi, ylo = 3775000, 3765500
    pre_crop = pre.sel(x=slice(xmin, xmax), y=slice(yhi, ylo))
    dnbr_crop = dnbr.sel(x=slice(xmin, xmax), y=slice(yhi, ylo))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=200)
    _style_dark(fig, [ax1, ax2])

    # Left: pre-fire vegetation (green = healthy)
    green_cmap = plt.get_cmap("YlGn").copy()
    green_cmap.set_bad(alpha=0)
    ax1.imshow(
        np.ma.masked_invalid(pre_crop.values),
        extent=[xmin, xmax, ylo, yhi], origin="upper",
        cmap=green_cmap, vmin=0.0, vmax=0.7,
        interpolation="nearest", alpha=0.82, zorder=2,
    )
    _add_basemap(ax1, alpha=0.55)

    # Right: burn severity (warm = severe)
    ax2.imshow(
        np.ma.masked_invalid(dnbr_crop.values),
        extent=[xmin, xmax, ylo, yhi], origin="upper",
        cmap=SEV_CMAP, vmin=-0.1, vmax=1.3,
        interpolation="nearest", alpha=0.82, zorder=2,
    )
    _add_basemap(ax2, alpha=0.55)

    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
        _label_places(ax)

    # Headlines — not "NDVI" and "dNBR", just what people see
    ax1.text(0.04, 0.95, "December 15, 2024",
             transform=ax1.transAxes, fontsize=18, fontweight="bold",
             color=TEXT_COLOR, va="top", zorder=10)
    ax1.text(0.04, 0.88, "One month before the Palisades Fire",
             transform=ax1.transAxes, fontsize=11,
             color=SUBTITLE_COLOR, va="top", zorder=10)

    ax2.text(0.04, 0.95, "January 23, 2025",
             transform=ax2.transAxes, fontsize=18, fontweight="bold",
             color=TEXT_COLOR, va="top", zorder=10)
    ax2.text(0.04, 0.88, "23,000 acres burned · 5,000+ structures damaged",
             transform=ax2.transAxes, fontsize=11,
             color="#ff9999", va="top", zorder=10)

    fig.patches.append(plt.Rectangle(
        (0.498, 0.03), 0.004, 0.92,
        transform=fig.transFigure, facecolor=TEXT_COLOR, zorder=15,
    ))

    fig.text(0.5, 0.98,
             "The Palisades Fire — seen from a new kind of satellite",
             ha="center", fontsize=15, fontweight="bold",
             color=TEXT_COLOR, va="top")
    _credit(ax2)

    fig.tight_layout(pad=0.8, rect=[0, 0, 1, 0.96])
    fig.savefig(SOCIAL / "before_after.png", dpi=200,
                facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 2: What the satellite actually sees — the "zoom in" moment
# This is the payoff: not just "burned / not burned" but what's on the ground
# at the sub-pixel level. The analogy: X-ray vs photograph.
# ---------------------------------------------------------------------------

def fig2_char_fraction():
    """Char fraction: 'what's actually on the ground' with place context."""
    print("  [2/5] Char fraction — the reveal...")
    char = _load("20241215_frac_char.tif")

    xmin, xmax = 343500, 353000
    yhi, ylo = 3775000, 3766000
    char = char.sel(x=slice(xmin, xmax), y=slice(yhi, ylo))

    fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
    _style_dark(fig, ax)

    im = ax.imshow(
        np.ma.masked_invalid(char.values),
        extent=[xmin, xmax, ylo, yhi], origin="upper",
        cmap=FIRE_CMAP, vmin=0, vmax=1.0,
        interpolation="nearest", alpha=0.88, zorder=2,
    )
    _add_basemap(ax, alpha=0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    _label_places(ax)

    # Headline tells the story, not the method
    ax.text(0.03, 0.97,
            "A normal satellite says \"burned.\"",
            transform=ax.transAxes, fontsize=18, fontweight="bold",
            color=SUBTITLE_COLOR, va="top", zorder=10, style="italic")
    ax.text(0.03, 0.92,
            "This one says how much.",
            transform=ax.transAxes, fontsize=22, fontweight="bold",
            color=TEXT_COLOR, va="top", zorder=10)
    ax.text(0.03, 0.85,
            "Each 30 m pixel decomposed into char, vegetation, and soil fractions\n"
            "using 426 spectral bands — like an X-ray vs. a photograph",
            transform=ax.transAxes, fontsize=11,
            color=SUBTITLE_COLOR, va="top", zorder=10, linespacing=1.6)

    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.015, aspect=25)
    cbar.set_label("Char fraction per pixel", color=TEXT_COLOR, fontsize=11)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["None", "25%", "50%", "75%", "100%"])
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=9)

    _credit(ax)
    fig.tight_layout(pad=0.5)
    fig.savefig(SOCIAL / "char_fraction.png", dpi=200,
                facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 3: Hero severity — the data layer with full context
# For people who've engaged with posts 1-2 and want the full map.
# ---------------------------------------------------------------------------

def fig3_hero_severity():
    """Severity map with class labels and neighborhood context."""
    print("  [3/5] Severity map with context...")
    dnbr = _load("20241215_to_20250123swath2_dnbr.tif")
    dnbr = dnbr.sel(x=slice(343000, 353100), y=slice(3775000, 3765500))

    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    _style_dark(fig, ax)

    im = ax.imshow(
        np.ma.masked_invalid(dnbr.values),
        extent=[343000, 353100, 3765500, 3775000], origin="upper",
        cmap=SEV_CMAP, vmin=-0.1, vmax=1.3,
        interpolation="nearest", alpha=0.82, zorder=2,
    )
    _add_basemap(ax, alpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    _label_places(ax)

    ax.text(0.03, 0.96,
            "Mapping fire damage pixel by pixel",
            transform=ax.transAxes, fontsize=20, fontweight="bold",
            color=TEXT_COLOR, va="top", zorder=10)
    ax.text(0.03, 0.90,
            "Burn severity across the Palisades Fire footprint\n"
            "Tanager-1 hyperspectral · 426 bands · 30 m resolution",
            transform=ax.transAxes, fontsize=11,
            color=SUBTITLE_COLOR, va="top", zorder=10, linespacing=1.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.015)
    cbar.set_ticks([0.0, 0.27, 0.44, 0.66, 1.0])
    cbar.set_ticklabels(["Unburned", "Low", "Moderate", "Mod-High", "High"])
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    cbar.set_label("Burn Severity", color=TEXT_COLOR, fontsize=11)

    _credit(ax)
    fig.tight_layout(pad=0.5)
    fig.savefig(SOCIAL / "hero_severity.png", dpi=200,
                facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 4: Sensor comparison — "why this satellite matters"
# Reframed: not R² values, but what you can and can't see.
# Headline stat: "Sentinel-2 misses ⅔ of the burn detail."
# ---------------------------------------------------------------------------

def fig4_sensor_comparison():
    """Why 426 bands matter, told for a non-specialist."""
    print("  [4/5] Sensor comparison — why it matters...")

    sensors = [
        ("Tanager-1\n(426 bands)", 5, 1.0),
        ("EMIT\n(285 bands)", 7.4, 0.991),
        ("PRISMA\n(239 bands)", 12, 0.957),
        ("Sentinel-2\n(10 bands)", 60, 0.361),
    ]

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=200)
    _style_dark(fig, ax)

    xs = [s[1] for s in sensors]
    ys = [s[2] for s in sensors]

    ax.plot(xs, ys, "-", color=ACCENT, linewidth=2.5, zorder=4)
    ax.plot(xs, ys, "o", color=ACCENT, markersize=12, zorder=5)

    offsets = [(155, -25), (160, -155), (175, -75), (-50, 40)]
    for (name, res, r2), off in zip(sensors, offsets):
        ax.annotate(
            f"{name}\n{r2:.1%} of signal",
            (res, r2), textcoords="offset points",
            xytext=off, ha="center", fontsize=9.5,
            color=TEXT_COLOR, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#ffffff66", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f0f1add",
                      edgecolor="#ffffff22"),
            zorder=8,
        )

    ax.fill_between([48, 65], 0.15, 0.5, alpha=0.12, color=ACCENT, zorder=1)
    ax.text(56.5, 0.20,
            "⅔ of the burn\ndetail — gone",
            fontsize=12, color=ACCENT, ha="center", va="bottom",
            fontweight="bold", zorder=10)

    # Title at top of figure, not on the plot area
    fig.text(0.5, 0.97,
             "What happens when you have fewer spectral bands?",
             ha="center", fontsize=17, fontweight="bold",
             color=TEXT_COLOR, va="top")

    # Explanation in lower-left, away from the data points
    ax.text(0.03, 0.35,
            "Your phone camera sees 3 colors.\n"
            "Sentinel-2 sees 10. Tanager-1 sees 426.\n\n"
            "For basic indices the gap doesn't show —\n"
            "but for sub-pixel burn mapping, it's the\n"
            "difference between seeing and missing.",
            transform=ax.transAxes, fontsize=10,
            color=SUBTITLE_COLOR, va="top", linespacing=1.5)

    ax.set_xlabel("Spectral resolution (nm)", fontsize=11, color=SUBTITLE_COLOR)
    ax.set_ylabel("Burn-detail accuracy", fontsize=11, color=SUBTITLE_COLOR)
    ax.set_xlim(0, 65)
    ax.set_ylim(0.15, 1.08)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"])
    ax.grid(True, alpha=0.1, color="#555577")

    _credit(ax, right=False)
    fig.tight_layout(pad=1.0)
    fig.savefig(SOCIAL / "sensor_comparison.png", dpi=200,
                facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 5: Recovery — "watching the landscape heal from space"
# Emotional payoff: it's not just destruction, there's recovery.
# Water content map with a hopeful framing.
# ---------------------------------------------------------------------------

def fig5_recovery():
    """Water content: recovery and what satellites can monitor going forward."""
    print("  [5/5] Recovery — canopy water content...")
    sai = _load("20250407_SAI970.tif")

    fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
    _style_dark(fig, ax)

    im = ax.imshow(
        np.ma.masked_invalid(sai.values),
        extent=[float(sai.x.min()), float(sai.x.max()),
                float(sai.y.min()), float(sai.y.max())],
        origin="upper",
        cmap=WATER_CMAP, vmin=0.02, vmax=0.11,
        interpolation="nearest", alpha=0.88, zorder=2,
    )
    _add_basemap(ax, alpha=0.35)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(0.03, 0.97,
            "Three months later: watching recovery from space",
            transform=ax.transAxes, fontsize=19, fontweight="bold",
            color=TEXT_COLOR, va="top", zorder=10)
    ax.text(0.03, 0.91,
            "Canopy water content across Los Angeles County — April 7, 2025\n"
            "Blue = more water in vegetation (healthier). Brown = dry / burned.\n"
            "Measured through a water-absorption feature at 970 nm\n"
            "that only hyperspectral satellites can resolve.",
            transform=ax.transAxes, fontsize=11,
            color=SUBTITLE_COLOR, va="top", zorder=10, linespacing=1.6)

    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.015, aspect=25)
    cbar.set_label("Canopy water content", color=TEXT_COLOR, fontsize=11)
    cbar.set_ticks([0.03, 0.05, 0.07, 0.09, 0.11])
    cbar.set_ticklabels(["Dry", "", "Moderate", "", "Wet"])
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=9)

    _credit(ax)
    fig.tight_layout(pad=0.5)
    fig.savefig(SOCIAL / "water_content.png", dpi=200,
                facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    matplotlib.use("Agg")
    print("Generating social-media figures (v2 — story-first)...")
    print(f"Output directory: {SOCIAL.resolve()}\n")

    fig1_before_after()
    fig2_char_fraction()
    fig3_hero_severity()
    fig4_sensor_comparison()
    fig5_recovery()

    print(f"\nDone! {len(list(SOCIAL.glob('*.png')))} figures saved to {SOCIAL}/")
