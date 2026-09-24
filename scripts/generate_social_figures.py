#!/usr/bin/env python3
"""Generate social-media-ready figures from FireSpec outputs.

V3. Two changes over V2:

* **Correctness.** The severity rasters are now land-masked (open ocean carried
  dNBR up to ~0.3 and was being painted as "moderate burn" across most of the
  frame), and place labels are projected from real coordinates instead of
  hand-entered UTM values that were 7-11 km off.
* **New layers.** Adds the CAL FIRE damage-inspection points, a hyperspectral
  datacube graphic, measured spectral signatures with the Sentinel-2 bands
  overlaid, and a true/false-colour composite strip.

Run ``scripts/extract_spectral_cache.py`` first — the cube-derived figures read
its cached output.

Usage::

    python3 scripts/extract_spectral_cache.py
    python3 scripts/generate_social_figures.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import social_base as sb
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

CACHE = Path("outputs/social_spectral_cache.npz")

DNBR = "20241215_to_20250123swath2_dnbr.tif"

# Severity break points (dNBR), USGS/Key & Benson BARC convention.
SEV_BREAKS = [0.10, 0.27, 0.44, 0.66]
SEV_LABELS = ["Unburned", "Low", "Moderate", "Mod-High", "High"]

# Tighter frame for the MESMA char map, centred on the Franklin Fire scar in
# Malibu Canyon — the only burn the 2024-12-15 fraction rasters resolve.
FRANKLIN_FRAME = dict(xmin=337500, xmax=350500, ymin=3763500, ymax=3773500)


# ---------------------------------------------------------------------------
# 1. Before / after — the emotional anchor
# ---------------------------------------------------------------------------

def fig_before_after() -> None:
    print("  [1] Before & after ...")
    pre = sb.crop(sb.load("20241215_ndvi.tif"))
    dnbr = sb.crop(sb.load(DNBR, mask_water=True))
    ext = sb.extent_of()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.9), dpi=200)
    sb.style_dark(fig, [ax1, ax2])

    ax1.imshow(
        np.ma.masked_invalid(pre.values), extent=ext, origin="upper",
        cmap=sb.VEG_CMAP, vmin=0.05, vmax=0.65, interpolation="nearest",
        alpha=0.88, zorder=2,
    )
    sb.add_basemap(ax1, alpha=0.5)

    ax2.imshow(
        np.ma.masked_invalid(dnbr.values), extent=ext, origin="upper",
        cmap=sb.SEV_CMAP, vmin=-0.2, vmax=1.0, interpolation="nearest",
        alpha=0.88, zorder=2,
    )
    sb.add_basemap(ax2, alpha=0.5)

    for ax in (ax1, ax2):
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
    sb.overlay_perimeter(ax1, "Franklin", color="#8ab4ff", linewidth=1.5)
    sb.overlay_perimeter(ax2, "Palisades", color="#ffe9a8", linewidth=1.5)
    sb.map_furniture(ax1, scale_km=5, avoid=[(0.0, 0.84, 0.68, 1.0)])
    sb.map_furniture(ax2, scale_km=5, avoid=[(0.0, 0.84, 0.84, 1.0)])

    sb.headline(
        ax1, "15 December 2024",
        "Three weeks before the Palisades Fire.\nGreen is living chaparral.",
        title_size=19, width=0.62,
    )
    sb.headline(
        ax2, "23 January 2025",
        "Sixteen days after ignition.\nRed is where the chaparral stopped reflecting.",
        title_size=19, width=0.78, sub_color="#ffb3b3",
    )

    # The "before" image is not a clean baseline: the Franklin Fire (9-18 Dec
    # 2024, 4,086 acres, Malibu Canyon) had burned six days earlier and shows
    # as a low-NDVI scar north of Pepperdine. Calling it out is more honest
    # than letting a reader mistake it for part of the Palisades burn.
    fx, fy = sb.utm(-118.6804, 34.0557)
    ax1.annotate(
        "Franklin Fire scar\nburned 9–18 Dec, six days before this image",
        xy=(fx, fy), xytext=(fx + 600, fy - 4600), fontsize=9.5,
        color="#ffd08a", fontweight="bold", ha="center", va="top",
        zorder=21, linespacing=1.45,
        path_effects=[pe.withStroke(linewidth=2.6, foreground="#000000dd")],
        arrowprops=dict(arrowstyle="-", color="#ffd08a", lw=1.2, alpha=0.85),
    )

    fig.text(0.5, 0.995,
             "The western Palisades Fire, before and after — Santa Monica Mountains, California",
             ha="center", fontsize=15.5, fontweight="bold",
             color=sb.TEXT_COLOR, va="top")
    fig.patches.append(plt.Rectangle(
        (0.4985, 0.02), 0.003, 0.93, transform=fig.transFigure,
        facecolor="#ffffff", zorder=15,
    ))
    sb.credit(ax2)
    fig.tight_layout(pad=0.7, rect=[0, 0, 1, 0.955])
    print("      ->", sb.save(fig, "before_after.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Hero severity map, with the structures that burned
# ---------------------------------------------------------------------------

def fig_hero_severity() -> None:
    print("  [2] Hero severity + damaged structures ...")
    dnbr = sb.crop(sb.load(DNBR, mask_water=True))
    ext = sb.extent_of()

    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=200)
    sb.style_dark(fig, ax)

    im = ax.imshow(
        np.ma.masked_invalid(dnbr.values), extent=ext, origin="upper",
        cmap=sb.SEV_CMAP, vmin=-0.2, vmax=1.0, interpolation="nearest",
        alpha=0.85, zorder=2,
    )
    sb.add_basemap(ax, alpha=0.55)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])

    dins = sb.load_dins()
    inframe = dins.cx[ext[0]:ext[1], ext[2]:ext[3]]
    destroyed = inframe[inframe["DAMAGE"] == "Destroyed (>50%)"]
    ax.scatter(
        destroyed.geometry.x, destroyed.geometry.y, s=7, marker="s",
        facecolor="#ffffff", edgecolor="#000000", linewidth=0.25,
        zorder=12, alpha=0.95,
    )

    sb.overlay_perimeter(ax, "Palisades", color="#ffe9a8", linewidth=1.8)
    sb.map_furniture(ax, scale_km=5, inset=True,
                     avoid=[(0.0, 0.79, 0.55, 1.0)])

    sb.headline(
        ax, "Where the fire burned hottest",
        "Measured by Planet's Tanager-1 hyperspectral satellite. Inside the\n"
        "official fire perimeter the median dNBR is 0.588; outside it, 0.046.",
        title_size=22, width=0.50,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.012, aspect=24)
    cbar.set_ticks([-0.05, 0.185, 0.355, 0.55, 0.83])
    cbar.set_ticklabels(SEV_LABELS)
    cbar.ax.tick_params(colors=sb.TEXT_COLOR, labelsize=9.5)
    cbar.set_label("Burn severity (dNBR)", color=sb.TEXT_COLOR, fontsize=11)
    cbar.outline.set_edgecolor("#2a2a44")

    ax.legend(
        handles=[
            Line2D([0], [0], marker="s", color="none",
                   markerfacecolor="#ffffff", markeredgecolor="#000000",
                   markersize=7,
                   label=f"{len(destroyed):,} structures destroyed (CAL FIRE)"),
            Line2D([0], [0], color="#ffe9a8", lw=1.8,
                   label="Official Palisades Fire perimeter (NIFC)"),
        ],
        loc="upper left", frameon=True, facecolor="#07070fe6",
        edgecolor="#2e2e4a", fontsize=10, labelcolor=sb.TEXT_COLOR,
        bbox_to_anchor=(0.018, 0.775),  # just under the headline plate
    )

    sb.credit(ax)
    fig.tight_layout(pad=0.5)
    print("      ->", sb.save(fig, "hero_severity.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Char fraction — the sub-pixel reveal
# ---------------------------------------------------------------------------

def fig_palisades_char() -> None:
    """MESMA char fraction over the Palisades Fire, from an external library.

    Uses ``20250123swath2_frac_char.tif``, unmixed with the USGS splib07a
    library built by ``scripts/build_fire_endmember_library.py``. That library
    has no contact with the imagery, which is what makes the perimeter
    comparison out-of-sample: the older image-derived endmembers defined char
    as "the pixels that look burned", so their char fraction could not disagree
    with the threshold that produced it.

    Scored by ``scripts/validate_char_fractions.py``: inside the official
    Palisades perimeter the median is 0.389 against 0.000 outside every
    perimeter, AUC 0.795.
    """
    print("  [3a] Char fraction (Palisades, external library) ...")
    char = sb.crop(sb.load("20250123swath2_frac_char.tif", mask_water=True))
    ext = sb.extent_of()

    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=200)
    sb.style_dark(fig, ax)

    im = ax.imshow(
        np.ma.masked_invalid(char.values), extent=ext, origin="upper",
        cmap=sb.FIRE_CMAP, vmin=0, vmax=1.0, interpolation="nearest",
        alpha=0.9, zorder=2,
    )
    sb.add_basemap(ax, alpha=0.38)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    sb.overlay_perimeter(ax, "Palisades", color="#8ab4ff", linewidth=1.8)
    sb.map_furniture(ax, scale_km=5, avoid=[(0.0, 0.76, 0.58, 1.0)])

    sb.headline(
        ax, "How much of this pixel burned?",
        "Each 30 m pixel unmixed into char, plant, dry plant and soil against a\n"
        "USGS reference library that never saw this image. Inside the blue\n"
        "perimeter the median char fraction is 0.389; outside every known fire\n"
        "perimeter it is 0.000 (AUC 0.795).",
        title_size=23, width=0.56,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.012, aspect=24)
    cbar.set_label("Share of each pixel that is char", color=sb.TEXT_COLOR,
                   fontsize=11)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["None", "25%", "50%", "75%", "All"])
    cbar.ax.tick_params(colors=sb.TEXT_COLOR, labelsize=9.5)
    cbar.outline.set_edgecolor("#2a2a44")

    ax.text(0.022, 0.085,
            "Resolves wildland burn, not building damage: against CAL FIRE's\n"
            "structure survey the same map scores AUC 0.530, near chance.",
            transform=ax.transAxes, fontsize=9.5, color="#d8d8e8", va="bottom",
            zorder=10, style="italic",
            path_effects=[pe.withStroke(linewidth=2.6, foreground="#000000dd")])

    sb.credit(ax)
    fig.tight_layout(pad=0.5)
    print("      ->", sb.save(fig, "palisades_char.png"))
    plt.close(fig)


def fig_char_fraction() -> None:
    """MESMA char fraction over the Franklin Fire scar.

    Kept alongside the Palisades map because it is the same method on a second,
    independent fire: the 2024-12-15 scene predates the Palisades Fire, so the
    burn it resolves is the Franklin Fire of 9-18 December 2024.
    """
    print("  [3] Char fraction (Franklin Fire scar) ...")
    char = sb.crop(sb.load("20241215_frac_char.tif"), frame=FRANKLIN_FRAME)
    ext = sb.extent_of(FRANKLIN_FRAME)

    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=200)
    sb.style_dark(fig, ax)

    im = ax.imshow(
        np.ma.masked_invalid(char.values), extent=ext, origin="upper",
        cmap=sb.FIRE_CMAP, vmin=0, vmax=1.0, interpolation="nearest",
        alpha=0.9, zorder=2,
    )
    sb.add_basemap(ax, alpha=0.38)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    sb.overlay_perimeter(ax, "Franklin", color="#8ab4ff", linewidth=1.8)
    sb.map_furniture(ax, scale_km=3, avoid=[(0.0, 0.76, 0.58, 1.0)])

    sb.headline(
        ax, "This one says how much.",
        "A normal satellite says “burned.” Every 30 m pixel here is unmixed into\n"
        "char, living plant, dry plant and soil — the combination that best\n"
        "reproduces its measured 426-band spectrum.\n"
        "Franklin Fire scar, Malibu Canyon, imaged 15 December 2024. Inside the\n"
        "blue perimeter char averages 0.490; outside it, 0.047.",
        title_size=24, width=0.58,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.012, aspect=24)
    cbar.set_label("Share of each pixel that is char", color=sb.TEXT_COLOR,
                   fontsize=11)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["None", "25%", "50%", "75%", "All"])
    cbar.ax.tick_params(colors=sb.TEXT_COLOR, labelsize=9.5)
    cbar.outline.set_edgecolor("#2a2a44")

    ax.text(0.022, 0.085,
            "Endmembers derived from this scene \u2014 fractions are internally\n"
            "consistent, not calibrated against a spectral library.",
            transform=ax.transAxes, fontsize=9, color="#7a7a99", va="bottom",
            zorder=10, style="italic")

    sb.credit(ax)
    fig.tight_layout(pad=0.5)
    print("      ->", sb.save(fig, "char_fraction.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. The datacube — the visual that says "hyperspectral" at a glance
# ---------------------------------------------------------------------------

def fig_hypercube() -> None:
    print("  [4] Hyperspectral datacube ...")
    z = np.load(CACHE, allow_pickle=True)
    top = z["composite_true_color"]
    right = z["cube_right"]
    bottom = z["cube_bottom"]
    wl = z["wavelength"]

    # Downsample for a clean render at poster size.
    step = 3
    top = top[::step, ::step]
    right = right[:, ::step]
    bottom = bottom[:, ::step]

    n_rows, n_cols = top.shape[0], top.shape[1]
    depth = 0.55            # horizontal run per unit of image row (recession)
    tilt = 0.40             # vertical rise per unit of image row
    height = 0.50 * n_cols  # visual length of the spectral axis
    dx, dy = depth * n_rows, tilt * n_rows

    fig, ax = plt.subplots(figsize=(13, 9.5), dpi=200)
    sb.style_dark(fig, ax)
    ax.set_facecolor(sb.DARK_BG)

    def _tf(matrix):
        return Affine2D(np.array(matrix, dtype=float)) + ax.transData

    # Sheared images must be RGBA. Matplotlib resamples a transformed image
    # into its axis-aligned bounding box, and with a plain RGB array the
    # corners the parallelogram does not cover come out opaque black.
    def _rgba(arr: np.ndarray) -> np.ndarray:
        rgb = np.clip(np.nan_to_num(arr, nan=0.0), 0, 1)
        return np.dstack([rgb, np.ones(rgb.shape[:2], dtype=rgb.dtype)])

    def _shade(arr: np.ndarray) -> np.ndarray:
        norm = np.clip(np.nan_to_num(arr, nan=0.0) / 0.42, 0, 1)
        return plt.get_cmap("magma")(norm)

    # Top face. With origin="upper" the image's first row (north) sits at
    # y=n_rows, which the shear sends to the back edge of the parallelogram.
    ax.imshow(
        _rgba(top), extent=[0, n_cols, 0, n_rows], origin="upper",
        transform=_tf([[1, depth, 0], [0, tilt, 0], [0, 0, 1]]),
        interpolation="bilinear", zorder=6,
    )

    # Front face: spectral depth beneath the southern edge of the image.
    ax.imshow(
        _shade(bottom), extent=[0, n_cols, -height, 0], origin="upper",
        aspect="auto", interpolation="bilinear", zorder=5,
    )

    # Right face: spectral depth beneath the eastern edge. The column order is
    # flipped so image row 0 (north) lands at the back of the cube.
    ax.imshow(
        _shade(right[:, ::-1]), extent=[0, n_rows, -height, 0], origin="upper",
        aspect="auto", interpolation="bilinear",
        transform=_tf([[depth, 0, n_cols], [tilt, 1, 0], [0, 0, 1]]),
        zorder=4,
    )

    # The two dark stripes across the faces are real: atmospheric water vapour
    # absorbs almost everything there, so no surface reflectance is retrievable.
    band_y = lambda nm: -height * float(np.argmin(np.abs(wl - nm))) / len(wl)  # noqa: E731
    ax.annotate(
        "atmospheric water-vapour bands\n(no surface signal reaches orbit)",
        xy=(0.04 * n_cols, band_y(1400)),
        xytext=(-0.02 * n_cols, -height - 0.12 * n_cols),
        fontsize=10, color="#9a9ab5", ha="left", va="top", linespacing=1.45,
        zorder=12,
        arrowprops=dict(arrowstyle="-", color="#9a9ab5", lw=1.0, alpha=0.7),
    )

    corners = {
        "fl": (0.0, 0.0), "fr": (float(n_cols), 0.0),
        "bl": (dx, dy), "br": (n_cols + dx, dy),
    }
    edge = dict(color="#ffffff", lw=1.2, zorder=9, alpha=0.8)
    for a, b in [("fl", "fr"), ("fl", "bl"), ("fr", "br"), ("bl", "br")]:
        ax.plot(*zip(corners[a], corners[b]), **edge)
    for key in ("fl", "fr", "br"):
        cx, cy = corners[key]
        ax.plot([cx, cx], [cy, cy - height], **edge)
    ax.plot([0, n_cols], [-height, -height], **edge)
    ax.plot([n_cols, n_cols + dx], [-height, dy - height], **edge)

    arrow_x = n_cols + dx + 0.06 * n_cols
    ax.annotate(
        "", xy=(arrow_x, dy - height), xytext=(arrow_x, dy),
        arrowprops=dict(arrowstyle="-|>", color=sb.ACCENT, lw=2.2), zorder=10,
    )
    ax.text(
        arrow_x + 0.03 * n_cols, dy - height / 2,
        f"426 bands\n{wl.min():.0f}–{wl.max():.0f} nm",
        color=sb.ACCENT, fontsize=13, fontweight="bold", va="center",
        ha="left", linespacing=1.5, zorder=10,
    )
    ax.text(
        n_cols / 2, 0.06 * n_cols, "visible-light image",
        color="#ffffff", fontsize=10.5, ha="center", va="bottom", zorder=10,
        alpha=0.85,
    )

    ax.text(0.025, 0.975, "Every pixel is a spectrum",
            transform=ax.transAxes, fontsize=27, fontweight="bold",
            color=sb.TEXT_COLOR, va="top", zorder=10)
    ax.text(0.025, 0.912,
            "A photograph stores three numbers per pixel: red, green, blue.\n"
            "Tanager-1 stores 426 — a continuous curve from visible light\n"
            "through shortwave infrared. The image on top is what your eye\n"
            "would see. The glowing faces are the other 423 bands.",
            transform=ax.transAxes, fontsize=12.5, color=sb.SUBTITLE_COLOR,
            va="top", zorder=10, linespacing=1.65)
    ax.text(0.025, 0.045,
            "Santa Monica Mountains, Los Angeles · 23 January 2025 · "
            "surface reflectance",
            transform=ax.transAxes, fontsize=10.5, color=sb.SUBTITLE_COLOR,
            va="bottom", zorder=10)

    ax.set_xlim(-0.06 * n_cols, n_cols + dx + 0.36 * n_cols)
    ax.set_ylim(-height - 0.34 * n_cols, dy + 0.34 * n_cols)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)

    sb.credit(ax)
    fig.tight_layout(pad=0.4)
    print("      ->", sb.save(fig, "hypercube.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Spectral signatures — why 426 bands beats 10
# ---------------------------------------------------------------------------

def fig_spectral_signatures() -> None:
    print("  [5] Spectral signatures vs Sentinel-2 bands ...")
    from tanager.config import SENTINEL2_BANDS

    z = np.load(CACHE, allow_pickle=True)
    wl = z["wavelength"]
    good = z["good_wavelengths"].astype(bool)
    names = [str(n) for n in z["spectra_names"]]
    spectra = z["spectra"]
    counts = z["spectra_counts"]

    pretty = {
        "unburned_veg": ("Living chaparral", "#4ade80"),
        "burned": ("Burned — char and ash", "#ff5a5f"),
        "bare_soil": ("Bare soil", "#d9a441"),
    }
    order = ["unburned_veg", "burned", "bare_soil"]

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(13.5, 8.5), dpi=200, sharex=True,
        gridspec_kw={"height_ratios": [6.2, 1], "hspace": 0.06},
    )
    sb.style_dark(fig, [ax, axr])

    # Faint reminder of where Sentinel-2 samples, behind the curves.
    for b in SENTINEL2_BANDS.values():
        ax.axvspan(
            b["center_nm"] - b["fwhm_nm"] / 2,
            b["center_nm"] + b["fwhm_nm"] / 2,
            color="#60a5fa", alpha=0.10, zorder=1, linewidth=0,
        )

    # Bands where atmospheric water vapour makes the retrieval meaningless.
    for lo, hi in [(1340, 1465), (1790, 1955)]:
        for a in (ax, axr):
            a.axvspan(lo, hi, facecolor="#15151f", edgecolor="#33334d",
                      hatch="///", alpha=1.0, zorder=2, linewidth=0.6)
    ax.text(1402, 0.447, "atmospheric\nwater vapour", ha="center", va="top",
            fontsize=8.5, color="#7a7a99", zorder=6, linespacing=1.3)

    plotted = np.where(good, 1.0, np.nan)
    for key in order:
        if key not in names:
            continue
        s = spectra[names.index(key)]
        label, color = pretty[key]
        n = counts[names.index(key)]
        ax.plot(wl, s * plotted, color=color, lw=2.6, zorder=5,
                label=f"{label}   n = {n:,} pixels", solid_capstyle="round")

    # Named absorption features, annotated where they actually sit.
    features = [
        (672, 0.055, 0.021, "chlorophyll", "#4ade80"),
        (1120, 0.255, 0.345, "leaf water", "#60a5fa"),
        (2205, 0.263, 0.355, "char & clay", "#ff5a5f"),
    ]
    for x, y_data, y_text, txt, c in features:
        ax.annotate(
            txt, xy=(x, y_data), xytext=(x, y_text), ha="center",
            fontsize=10.5, color=c, fontweight="bold", zorder=8,
            arrowprops=dict(arrowstyle="-", color=c, lw=1.1, alpha=0.55,
                            shrinkA=3, shrinkB=3),
        )

    ax.set_ylabel("Surface reflectance", fontsize=12, color=sb.SUBTITLE_COLOR)
    ax.set_ylim(0, 0.46)
    ax.grid(True, alpha=0.09, color="#555577")
    leg = ax.legend(loc="upper left", frameon=True, facecolor="#07070fe6",
                    edgecolor="#2e2e4a", fontsize=11,
                    labelcolor=sb.TEXT_COLOR, bbox_to_anchor=(0.006, 0.99))
    leg.set_zorder(20)

    # --- Coverage ruler: one continuous bar vs ten chunks -------------------
    axr.add_patch(plt.Rectangle((380, 0.58), 2120, 0.30, facecolor="#ff5a5f",
                                edgecolor="none", zorder=4))
    for b in SENTINEL2_BANDS.values():
        axr.add_patch(plt.Rectangle(
            (b["center_nm"] - b["fwhm_nm"] / 2, 0.12), b["fwhm_nm"], 0.30,
            facecolor="#60a5fa", edgecolor="none", zorder=4,
        ))
    # Anchored left: the right-hand end of the ruler is where Sentinel-2's
    # widest band (B12, 180 nm) sits, and a label there lands on top of it.
    halo = [pe.withStroke(linewidth=2.6, foreground="#07070f")]
    axr.text(408, 0.73, "Tanager-1 · 426 contiguous bands", ha="left",
             va="center", fontsize=10.5, fontweight="bold", color="#ffffff",
             zorder=6, path_effects=halo)
    axr.text(408, 0.27, "Sentinel-2 · 10 land bands", ha="left", va="center",
             fontsize=10.5, fontweight="bold", color="#ffffff", zorder=6,
             path_effects=halo)
    axr.set_ylim(0, 1)
    axr.set_yticks([])
    axr.set_xlim(380, 2500)
    axr.set_xlabel("Wavelength (nm)", fontsize=12, color=sb.SUBTITLE_COLOR)
    axr.grid(False)
    for s in ("top", "left", "right"):
        axr.spines[s].set_visible(False)

    fig.text(0.5, 0.995,
             "What burned ground actually looks like, across 426 wavelengths",
             ha="center", fontsize=17, fontweight="bold",
             color=sb.TEXT_COLOR, va="top")
    fig.text(0.5, 0.947,
             "Three surfaces, measured by the same satellite on the same pass. "
             "The bar underneath is what each sensor gets to sample.",
             ha="center", fontsize=11.5, color=sb.SUBTITLE_COLOR, va="top")

    fig.text(0.985, 0.008,
             "Measured Tanager-1 surface reflectance, 23 Jan 2025 · class "
             "means · github.com/gpriceless/tanager_wildfire",
             ha="right", va="bottom", fontsize=7.5, color=sb.CREDIT_COLOR)
    fig.tight_layout(pad=1.0, rect=[0, 0.03, 1, 0.925])
    print("      ->", sb.save(fig, "spectral_signatures.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Composite strip — "6 of 426 bands"
# ---------------------------------------------------------------------------

def fig_composites() -> None:
    print("  [6] True / false colour composites ...")
    z = np.load(CACHE, allow_pickle=True)

    panels = [
        ("true_color", "What your eye would see",
         "Red · green · blue. Smoke haze and grey-brown hillsides."),
        ("nir_false_color", "Near-infrared",
         "Living plants glow red. The burn scar goes dark."),
        ("swir_burn", "Shortwave infrared",
         "Char and bare soil light up. The fire's edge becomes a hard line."),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 7), dpi=200)
    sb.style_dark(fig, axes)

    for ax, (key, title, sub) in zip(axes, panels):
        rgb = np.clip(np.nan_to_num(z[f"composite_{key}"], nan=0.0), 0, 1)
        bands = z[f"composite_bands_{key}"]
        ax.imshow(rgb, origin="upper", interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, -0.035, title, transform=ax.transAxes, ha="center",
                va="top", fontsize=14, fontweight="bold", color=sb.TEXT_COLOR)
        ax.text(0.5, -0.085, sub, transform=ax.transAxes, ha="center",
                va="top", fontsize=10, color=sb.SUBTITLE_COLOR,
                linespacing=1.4, wrap=True)
        ax.text(0.5, -0.155,
                " · ".join(f"{b:.0f} nm" for b in bands),
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                color=sb.ACCENT, fontfamily="monospace")

    fig.text(0.5, 0.99, "Three ways to look at the same moment",
             ha="center", fontsize=19, fontweight="bold",
             color=sb.TEXT_COLOR, va="top")
    fig.text(0.5, 0.935,
             "Each panel is built from 3 of Tanager-1's 426 bands. "
             "Same pixels, same second — only the wavelengths change.",
             ha="center", fontsize=11.5, color=sb.SUBTITLE_COLOR, va="top")
    fig.text(0.985, 0.012, sb.CREDIT, ha="right", va="bottom", fontsize=7.5,
             color=sb.CREDIT_COLOR)
    fig.tight_layout(pad=1.2, rect=[0, 0.075, 1, 0.91])
    print("      ->", sb.save(fig, "composites.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Structures — the human layer
# ---------------------------------------------------------------------------

def fig_structures() -> None:
    print("  [7] Structure damage vs measured severity ...")
    import geopandas as gpd  # noqa: F401
    import xarray as xr

    dnbr_full = sb.load(DNBR, mask_water=True)
    dins = sb.load_dins()
    ext = sb.extent_of()
    inframe = dins.cx[ext[0]:ext[1], ext[2]:ext[3]]

    vals = dnbr_full.sel(
        x=xr.DataArray(inframe.geometry.x.values, dims="p"),
        y=xr.DataArray(inframe.geometry.y.values, dims="p"),
        method="nearest",
    ).values

    groups = [
        ("No Damage", "#5aa9e6"),
        ("Affected (1-9%)", "#8ab4ff"),
        ("Minor (10-25%)", "#f0c419"),
        ("Major (26-50%)", "#f08a24"),
        ("Destroyed (>50%)", "#ff5a5f"),
    ]

    fig, (axm, axd) = plt.subplots(
        1, 2, figsize=(16, 8), dpi=200, gridspec_kw={"width_ratios": [1.45, 1]}
    )
    sb.style_dark(fig, [axm, axd])

    # --- Left: the map
    dnbr = sb.crop(dnbr_full)
    axm.imshow(
        np.ma.masked_invalid(dnbr.values), extent=ext, origin="upper",
        cmap=sb.SEV_CMAP, vmin=-0.2, vmax=1.0, interpolation="nearest",
        alpha=0.7, zorder=2,
    )
    sb.add_basemap(axm, alpha=0.5)
    axm.set_xlim(ext[0], ext[1])
    axm.set_ylim(ext[2], ext[3])

    for label, color in groups:
        g = inframe[inframe["DAMAGE"] == label]
        if len(g) == 0:
            continue
        axm.scatter(g.geometry.x, g.geometry.y, s=9, marker="o",
                    facecolor=color, edgecolor="#000000", linewidth=0.2,
                    zorder=12, alpha=0.9, label=f"{label}  ({len(g):,})")

    sb.overlay_perimeter(axm, "Palisades", color="#ffe9a8", linewidth=1.6)
    sb.map_furniture(axm, scale_km=5, avoid=[(0.0, 0.82, 0.72, 1.0)])
    axm.legend(loc="lower left", frameon=True, facecolor="#0b0b14ee",
               edgecolor="#2a2a44", fontsize=9, labelcolor=sb.TEXT_COLOR,
               title="CAL FIRE inspection", title_fontsize=9.5,
               bbox_to_anchor=(0.012, 0.06))
    axm.get_legend().get_title().set_color(sb.SUBTITLE_COLOR)
    sb.headline(
        axm, "Every dot is somebody's house",
        f"{len(inframe):,} structures inspected on foot by CAL FIRE inside this\n"
        "frame, drawn over the severity the satellite measured from orbit.",
        title_size=20, width=0.66,
    )

    # --- Right: do the two agree?
    data, colors, labels = [], [], []
    for label, color in groups:
        m = (inframe["DAMAGE"] == label).values & np.isfinite(vals)
        if m.sum() < 10:
            continue
        data.append(vals[m])
        colors.append(color)
        labels.append(f"{label}\nn={int(m.sum()):,}")

    # Raw points plus an interquartile bar. A smoothed violin on the n=12
    # "Major" group would draw a confident shape out of twelve observations.
    rng = np.random.default_rng(20260924)
    for i, (d, color) in enumerate(zip(data, colors), start=1):
        jitter = rng.uniform(-0.17, 0.17, size=d.size)
        axd.scatter(d, np.full(d.size, i) + jitter, s=5, color=color,
                    alpha=0.28, linewidth=0, zorder=3)
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        axd.plot([q1, q3], [i, i], color=color, lw=7, solid_capstyle="round",
                 alpha=0.95, zorder=5)
        axd.plot(med, i, "o", color="#ffffff", markersize=8,
                 markeredgecolor="#000000", markeredgewidth=0.8, zorder=6)

    axd.set_yticks(range(1, len(labels) + 1))
    axd.set_yticklabels(labels, fontsize=9.5, color=sb.TEXT_COLOR)
    axd.set_xlabel("Burn severity measured from orbit (dNBR)", fontsize=11,
                   color=sb.SUBTITLE_COLOR)
    axd.grid(True, axis="x", alpha=0.12, color="#555577")
    axd.set_xlim(-0.6, 1.2)
    axd.set_title(
        "Ground truth vs. orbit",
        fontsize=15, fontweight="bold", color=sb.TEXT_COLOR, pad=34,
    )
    axd.text(
        0.5, 1.035,
        "each dot one inspected structure · bar = interquartile range · "
        "white dot = median",
        transform=axd.transAxes, ha="center", va="bottom", fontsize=9,
        color=sb.SUBTITLE_COLOR,
    )
    med_dest = np.median(data[-1])
    med_none = np.median(data[0])
    axd.text(
        0.5, -0.145,
        f"Destroyed structures sit at dNBR {med_dest:.2f}; undamaged ones at "
        f"{med_none:.2f}.\nReal but overlapping — a 30 m pixel is wider "
        "than a house.",
        transform=axd.transAxes, ha="center", va="top", fontsize=10,
        color=sb.SUBTITLE_COLOR, linespacing=1.5,
    )

    sb.credit(axm)
    fig.tight_layout(pad=1.2, rect=[0, 0.04, 1, 1])
    print("      ->", sb.save(fig, "structures.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Sensor comparison
# ---------------------------------------------------------------------------

def fig_sensor_comparison() -> None:
    print("  [8] Sensor comparison ...")
    sensors = [
        ("Tanager-1", 426, 1.000, "#ff5a5f"),
        ("EMIT", 285, 0.991, "#f0a860"),
        ("PRISMA", 239, 0.957, "#f0c419"),
        ("Sentinel-2", 10, 0.361, "#5aa9e6"),
    ]

    fig, ax = plt.subplots(figsize=(13, 7), dpi=200)
    sb.style_dark(fig, ax)

    names = [s[0] for s in sensors]
    vals = [s[2] for s in sensors]
    bands = [s[1] for s in sensors]
    colors = [s[3] for s in sensors]
    ypos = np.arange(len(sensors))[::-1]

    ax.barh(ypos, vals, height=0.58, color=colors, alpha=0.9, zorder=3,
            edgecolor="none")
    for y, name, v, b, c in zip(ypos, names, vals, bands, colors):
        ax.text(v + 0.014, y, f"{v:.3f}", va="center", ha="left",
                fontsize=13, fontweight="bold", color=c, zorder=5)
        ax.text(-0.015, y + 0.20, name, va="center", ha="right", fontsize=13.5,
                fontweight="bold", color=sb.TEXT_COLOR, zorder=5)
        ax.text(-0.015, y - 0.20, f"{b} bands", va="center", ha="right",
                fontsize=10, color=sb.SUBTITLE_COLOR, zorder=5)

    ax.annotate(
        "", xy=(0.361, ypos[-1] - 0.42), xytext=(0.991, ypos[-1] - 0.42),
        arrowprops=dict(arrowstyle="<|-|>", color=sb.ACCENT, lw=1.6), zorder=6,
    )
    ax.text(0.676, ypos[-1] - 0.62,
            "the gap 416 extra bands buy you",
            ha="center", va="top", fontsize=11.5, color=sb.ACCENT,
            fontweight="bold", zorder=6)

    ax.set_xlim(0, 1.13)
    ax.set_ylim(ypos[-1] - 1.15, ypos[0] + 0.62)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"], fontsize=10)
    ax.set_xlabel(
        "Agreement with Tanager-1 char fraction  (R², same pixels, same day)",
        fontsize=11.5, color=sb.SUBTITLE_COLOR,
    )
    ax.grid(True, axis="x", alpha=0.12, color="#555577", zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    fig.text(0.5, 0.99,
             "Resample Tanager-1 down to each sensor, then try to map char again",
             ha="center", fontsize=17, fontweight="bold",
             color=sb.TEXT_COLOR, va="top")
    fig.text(0.5, 0.935,
             "Sentinel-2's 10 broad bands recover about a third of the sub-pixel "
             "burn signal. The other hyperspectral sensors recover nearly all of it.",
             ha="center", fontsize=11.5, color=sb.SUBTITLE_COLOR, va="top")

    sb.credit(ax, right=True)
    fig.tight_layout(pad=1.0, rect=[0, 0, 1, 0.90])
    print("      ->", sb.save(fig, "sensor_comparison.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9. Water content
# ---------------------------------------------------------------------------

def fig_water_content() -> None:
    print("  [9] Canopy water content ...")
    sai = sb.load("20250407_SAI970.tif")
    ext = [float(sai.x.min()), float(sai.x.max()),
           float(sai.y.min()), float(sai.y.max())]

    fig, ax = plt.subplots(figsize=(12, 8.5), dpi=200)
    sb.style_dark(fig, ax)

    im = ax.imshow(
        np.ma.masked_invalid(sai.values), extent=ext, origin="upper",
        cmap=sb.WATER_CMAP, vmin=0.02, vmax=0.11, interpolation="nearest",
        alpha=0.9, zorder=2,
    )
    sb.add_basemap(ax, alpha=0.35)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([])
    ax.set_yticks([])
    sb.scalebar(ax, 2.0)
    sb.north_arrow(ax)

    sb.headline(
        ax, "How thirsty is the landscape?",
        "Water held in the plant canopy, 7 April 2025 — north LA County.\n"
        "Measured from the depth of a 970 nm absorption dip that only a\n"
        "hyperspectral sensor can resolve. Dry fuel is what a fire runs on.",
        title_size=21, width=0.635,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.012, aspect=24)
    cbar.set_label("970 nm absorption depth (relative)", color=sb.TEXT_COLOR,
                   fontsize=11)
    cbar.set_ticks([0.03, 0.065, 0.10])
    cbar.set_ticklabels(["Drier", "", "Wetter"])
    cbar.ax.tick_params(colors=sb.TEXT_COLOR, labelsize=9.5)
    cbar.outline.set_edgecolor("#2a2a44")

    ax.text(0.03, 0.055,
            "Relative absorption depth — not calibrated to field fuel-moisture "
            "measurements.",
            transform=ax.transAxes, fontsize=9, color="#7a7a99", va="bottom",
            zorder=10, style="italic")

    sb.credit(ax)
    fig.tight_layout(pad=0.5)
    print("      ->", sb.save(fig, "water_content.png"))
    plt.close(fig)


FIGURES = [
    fig_before_after,
    fig_hero_severity,
    fig_palisades_char,
    fig_char_fraction,
    fig_hypercube,
    fig_spectral_signatures,
    fig_composites,
    fig_structures,
    fig_sensor_comparison,
    fig_water_content,
]


if __name__ == "__main__":
    matplotlib.use("Agg")
    if not CACHE.exists():
        raise SystemExit(
            f"Missing {CACHE}. Run: python3 scripts/extract_spectral_cache.py"
        )
    print("Generating social-media figures (v3)\n")
    for fn in FIGURES:
        fn()
    n = len(list(sb.SOCIAL.glob("*.png")))
    print(f"\nDone. {n} figures in {sb.SOCIAL}/")
