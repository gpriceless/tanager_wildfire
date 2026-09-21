#!/usr/bin/env python3
"""Generate social-media-ready figures from existing FireSpec outputs.

Produces polished versions of the key visualizations with:
- Satellite basemap underlays via contextily
- Clean annotations with fire names and dates
- Dark backgrounds for contrast
- LinkedIn-optimized dimensions (1200x627 and 1080x1080)
- Minimal axis chrome — the imagery speaks
"""
from __future__ import annotations

import warnings
from pathlib import Path

import contextily as ctx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rioxarray  # noqa: F401  — registers .rio accessor
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore", category=FutureWarning)

OUTPUTS = Path("outputs")
SOCIAL = Path("figures/social")
SOCIAL.mkdir(parents=True, exist_ok=True)

CRS = "EPSG:32611"
BASEMAP = ctx.providers.Esri.WorldImagery

DARK_BG = "#1a1a2e"
ACCENT = "#e94560"
TEXT_COLOR = "#f0f0f0"
SUBTITLE_COLOR = "#b0b0b0"


def _load(name: str) -> xr.DataArray:
    da = rioxarray.open_rasterio(OUTPUTS / name).squeeze("band", drop=True)
    da = da.where(da != da.rio.nodata)
    return da


def _km_fmt(v, _):
    return f"{v / 1000:.0f}"


def _add_basemap(ax, alpha=0.5):
    try:
        ctx.add_basemap(ax, crs=CRS, source=BASEMAP, alpha=alpha, zorder=0)
    except Exception:
        pass


def _style_dark(fig, axes):
    """Apply dark theme to figure and axes."""
    fig.patch.set_facecolor(DARK_BG)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color("#333355")


# --- Custom colormaps for social flair ---

_fire_colors = ["#000000", "#1a0a00", "#4d1a00", "#993300", "#e65c00",
                 "#ff9933", "#ffcc66", "#ffffff"]
FIRE_CMAP = LinearSegmentedColormap.from_list("fire_social", _fire_colors, N=256)
FIRE_CMAP.set_bad(alpha=0)

_severity_colors = ["#2166ac", "#67a9cf", "#d1e5f0", "#fddbc7",
                     "#ef8a62", "#b2182b"]
SEV_CMAP = LinearSegmentedColormap.from_list("severity_social", _severity_colors, N=256)
SEV_CMAP.set_bad(alpha=0)

_water_colors = ["#4a0000", "#8b0000", "#cd853f", "#f5deb3",
                  "#87ceeb", "#4682b4", "#00008b"]
WATER_CMAP = LinearSegmentedColormap.from_list("water_social", _water_colors, N=256)
WATER_CMAP.set_bad(alpha=0)


def fig1_hero_severity():
    """Hero image: dNBR burn severity cropped to burn scar on satellite basemap."""
    print("  [1/5] Hero severity map...")
    dnbr = _load("20241215_to_20250123swath2_dnbr.tif")

    # Crop to Palisades burn scar where the action is
    dnbr = dnbr.sel(x=slice(343000, 353100), y=slice(3775000, 3765500))

    fig, ax = plt.subplots(figsize=(12, 6.27), dpi=200)
    _style_dark(fig, ax)

    im = ax.imshow(
        np.ma.masked_invalid(dnbr.values),
        extent=[
            float(dnbr.x.min()), float(dnbr.x.max()),
            float(dnbr.y.min()), float(dnbr.y.max()),
        ],
        origin="upper",
        cmap=SEV_CMAP, vmin=-0.1, vmax=1.3,
        interpolation="nearest", alpha=0.82, zorder=2,
    )
    _add_basemap(ax, alpha=0.6)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.text(
        0.03, 0.95,
        "Burn Severity — Palisades Fire, Los Angeles",
        transform=ax.transAxes, fontsize=20, fontweight="bold",
        color=TEXT_COLOR, va="top", zorder=10,
    )
    ax.text(
        0.03, 0.88,
        "dNBR from Tanager-1 hyperspectral (426 bands, 5 nm)\n"
        "Dec 15, 2024 → Jan 23, 2025",
        transform=ax.transAxes, fontsize=12,
        color=SUBTITLE_COLOR, va="top", zorder=10, linespacing=1.5,
    )

    # Severity class labels on colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("dNBR", color=TEXT_COLOR, fontsize=11)
    cbar.set_ticks([0.0, 0.27, 0.44, 0.66, 1.0])
    cbar.set_ticklabels(["Unburned", "Low", "Moderate-Low", "Moderate-High", "High"])
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=9)

    # Data credit
    ax.text(
        0.98, 0.02, "Data: Planet Tanager-1 · github.com/gpriceless/tanager_wildfire",
        transform=ax.transAxes, fontsize=8,
        color="#666688", ha="right", va="bottom", zorder=10,
    )

    fig.tight_layout(pad=0.5)
    fig.savefig(SOCIAL / "hero_severity.png", dpi=200,
                facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)


def fig2_char_fraction():
    """Char fraction with fire glow colormap on satellite."""
    print("  [2/5] Char fraction showcase...")
    char = _load("20241215_frac_char.tif")

    # Crop to the burn scar region for focus
    xmin, xmax = 344000, 353000
    ymin, ymax = 3766000, 3775000
    char = char.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))

    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=200)
    _style_dark(fig, ax)

    im = ax.imshow(
        np.ma.masked_invalid(char.values),
        extent=[
            float(char.x.min()), float(char.x.max()),
            float(char.y.min()), float(char.y.max()),
        ],
        origin="upper",
        cmap=FIRE_CMAP, vmin=0, vmax=1.0,
        interpolation="nearest", alpha=0.9, zorder=2,
    )
    _add_basemap(ax, alpha=0.4)

    ax.xaxis.set_major_formatter(FuncFormatter(_km_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_km_fmt))
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.text(
        0.03, 0.97,
        "What's Actually on the Ground",
        transform=ax.transAxes, fontsize=20, fontweight="bold",
        color=TEXT_COLOR, va="top", zorder=10,
    )
    ax.text(
        0.03, 0.92,
        "MESMA char fraction — sub-pixel burn mapping\n"
        "Tanager-1 · 426 bands · Palisades Fire, Jan 2025",
        transform=ax.transAxes, fontsize=12,
        color=SUBTITLE_COLOR, va="top", zorder=10, linespacing=1.5,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Char Fraction", color=TEXT_COLOR, fontsize=12)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=9)

    fig.tight_layout(pad=0.5)
    fig.savefig(SOCIAL / "char_fraction.png", dpi=200,
                facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)


def fig3_before_after():
    """Side-by-side: pre-fire NDVI (green vegetation) vs dNBR (burn scar)."""
    print("  [3/5] Before/after comparison...")
    pre_ndvi = _load("20241215_ndvi.tif")
    dnbr = _load("20241215_to_20250123swath2_dnbr.tif")

    # Crop both to the burn scar area
    xmin, xmax = 343000, 353100
    ymax, ymin = 3775000, 3765500
    pre_crop = pre_ndvi.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
    dnbr_crop = dnbr.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.27), dpi=200)
    _style_dark(fig, [ax1, ax2])

    # Pre-fire: NDVI showing green vegetation
    green_cmap = plt.get_cmap("YlGn").copy()
    green_cmap.set_bad(alpha=0)
    im1 = ax1.imshow(
        np.ma.masked_invalid(pre_crop.values),
        extent=[xmin, xmax, ymin, ymax], origin="upper",
        cmap=green_cmap, vmin=0.0, vmax=0.7,
        interpolation="nearest", alpha=0.85, zorder=2,
    )
    _add_basemap(ax1, alpha=0.5)

    # Post-fire: dNBR showing burn scar
    im2 = ax2.imshow(
        np.ma.masked_invalid(dnbr_crop.values),
        extent=[xmin, xmax, ymin, ymax], origin="upper",
        cmap=SEV_CMAP, vmin=-0.1, vmax=1.3,
        interpolation="nearest", alpha=0.85, zorder=2,
    )
    _add_basemap(ax2, alpha=0.5)

    for ax, label, date in [
        (ax1, "Before", "Dec 15, 2024 — Vegetation Health (NDVI)"),
        (ax2, "After", "Jan 23, 2025 — Burn Severity (dNBR)"),
    ]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.text(
            0.04, 0.95, label,
            transform=ax.transAxes, fontsize=22, fontweight="bold",
            color=TEXT_COLOR, va="top", zorder=10,
        )
        ax.text(
            0.04, 0.87, date,
            transform=ax.transAxes, fontsize=10,
            color=SUBTITLE_COLOR, va="top", zorder=10,
        )

    # Vertical divider
    fig.patches.append(plt.Rectangle(
        (0.498, 0.05), 0.004, 0.88,
        transform=fig.transFigure, facecolor=TEXT_COLOR, zorder=15,
    ))

    fig.suptitle(
        "Palisades Fire — Tanager-1 Hyperspectral",
        fontsize=16, fontweight="bold", color=TEXT_COLOR, y=0.98,
    )
    fig.tight_layout(pad=0.8, rect=[0, 0, 1, 0.95])
    fig.savefig(SOCIAL / "before_after.png", dpi=200,
                facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)


def fig4_sensor_comparison():
    """Polished dark-mode version of the information loss curve."""
    print("  [4/5] Sensor comparison (dark mode)...")

    sensors = {
        "Tanager-1": (5, 1.0),
        "EMIT": (7.4, 0.991),
        "PRISMA": (12, 0.957),
        "Sentinel-2": (60, 0.361),
    }

    fig, ax = plt.subplots(figsize=(12, 6.27), dpi=200)
    _style_dark(fig, ax)

    resolutions = [s[0] for s in sensors.values()]
    r2_values = [s[1] for s in sensors.values()]

    ax.plot(resolutions, r2_values, "o-", color=ACCENT, linewidth=2.5,
            markersize=10, zorder=5)

    label_offsets = {
        "Tanager-1": (-35, 18),
        "EMIT": (35, 12),
        "PRISMA": (40, -8),
        "Sentinel-2": (-15, -25),
    }
    for name, (res, r2) in sensors.items():
        offset = label_offsets[name]
        ax.annotate(
            f"{name}\nR² = {r2:.3f}",
            (res, r2), textcoords="offset points",
            xytext=offset, ha="center", fontsize=10,
            color=TEXT_COLOR, fontweight="bold",
        )

    ax.axhline(y=0.99, color="#444466", linestyle="--", alpha=0.5, zorder=1)
    ax.text(55, 0.995, "Broadband indices (NBR, NDVI)\nR² > 0.99 for ALL sensors",
            fontsize=9, color=SUBTITLE_COLOR, ha="right", va="bottom")

    ax.fill_between([50, 65], 0.2, 0.5, alpha=0.15, color=ACCENT, zorder=1)
    ax.text(57.5, 0.26, "⅔ of burn signal\nlost here",
            fontsize=10, color=ACCENT, ha="center", va="bottom", fontweight="bold")

    ax.set_xlabel("Spectral Resolution (nm, FWHM)", fontsize=12, color=TEXT_COLOR)
    ax.set_ylabel("MESMA Char Fraction R²\nvs. native Tanager-1", fontsize=12, color=TEXT_COLOR)
    ax.set_xlim(0, 65)
    ax.set_ylim(0.2, 1.05)
    ax.grid(True, alpha=0.15, color="#555577")

    ax.text(
        0.02, 0.04,
        "Why hyperspectral matters: narrowband products degrade sharply below 426 bands",
        transform=ax.transAxes, fontsize=10,
        color=SUBTITLE_COLOR, va="bottom",
    )

    fig.tight_layout(pad=1.0)
    fig.savefig(SOCIAL / "sensor_comparison.png", dpi=200,
                facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)


def fig5_water_content():
    """Water-content map with clean styling."""
    print("  [5/5] Water content map...")
    sai970 = _load("20250407_SAI970.tif")

    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=200)
    _style_dark(fig, ax)

    im = ax.imshow(
        np.ma.masked_invalid(sai970.values),
        extent=[
            float(sai970.x.min()), float(sai970.x.max()),
            float(sai970.y.min()), float(sai970.y.max()),
        ],
        origin="upper",
        cmap=WATER_CMAP, vmin=0.02, vmax=0.11,
        interpolation="nearest", alpha=0.9, zorder=2,
    )
    _add_basemap(ax, alpha=0.35)

    ax.xaxis.set_major_formatter(FuncFormatter(_km_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_km_fmt))
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.text(
        0.03, 0.97,
        "Canopy Water Content from Space",
        transform=ax.transAxes, fontsize=20, fontweight="bold",
        color=TEXT_COLOR, va="top", zorder=10,
    )
    ax.text(
        0.03, 0.92,
        "970 nm water-absorption depth — resolved by 426 hyperspectral bands\n"
        "Tanager-1 · April 7, 2025 · Los Angeles County",
        transform=ax.transAxes, fontsize=12,
        color=SUBTITLE_COLOR, va="top", zorder=10, linespacing=1.5,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("SAI 970 nm (water absorption depth)", color=TEXT_COLOR, fontsize=11)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=9)

    fig.tight_layout(pad=0.5)
    fig.savefig(SOCIAL / "water_content.png", dpi=200,
                facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    matplotlib.use("Agg")
    print("Generating social-media-ready figures...")
    print(f"Output directory: {SOCIAL.resolve()}\n")

    fig1_hero_severity()
    fig2_char_fraction()
    fig3_before_after()
    fig4_sensor_comparison()
    fig5_water_content()

    print(f"\nDone! {len(list(SOCIAL.glob('*.png')))} figures saved to {SOCIAL}/")
