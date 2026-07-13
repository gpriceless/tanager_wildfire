"""Visualization utilities for Tanager-1 hyperspectral fire-analysis products.

This module provides map-making, diagnostic plotting, and interactive
visualization helpers for the Tanager product suite.  Heavy rendering
dependencies (matplotlib, contextily, geopandas, rioxarray) are imported
lazily inside each function so that the module remains importable in
headless or lightweight environments.

Public API:

* :func:`plot_map` — render a single-band or RGB raster as a map
* :func:`plot_before_after` — side-by-side pre/post fire comparison
* :func:`plot_temporal_trajectory` — time-series trajectory for a pixel or ROI
* :func:`plot_severity_summary` — histogram + map summary of burn severity
* :func:`plot_difference_map` — signed difference between two rasters
* :func:`interactive_map` — folium/ipyleaflet interactive map (notebook)
* :func:`show_product` — convenience wrapper to display a named product
* :func:`save_figure` — save a matplotlib figure to disk with sane defaults
* :func:`add_basemap` — overlay a web-tile basemap on an axes
* :func:`load_fire_perimeters` — load NIFC/GeoMAC fire perimeter polygons
* :func:`overlay_perimeters` — draw fire perimeter(s) on an axes
* :func:`add_scalebar` — add a distance scale-bar to a map axes
* :data:`PRODUCT_STYLES` — style configuration dict keyed by product name

Import direction:

* visualization.py MAY import from :mod:`tanager.config` and
  :mod:`tanager.io`.
* visualization.py MUST NOT import from :mod:`tanager.unmixing`,
  :mod:`tanager.severity`, :mod:`tanager.lfmc`, or
  :mod:`tanager.validation` to keep the dependency graph acyclic.

Heavy deps (matplotlib, contextily, geopandas, rioxarray) are lazy-imported
inside each function body so ``import tanager.visualization`` works in
environments that lack those packages.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import xarray as xr

if TYPE_CHECKING:  # pragma: no cover
    from datetime import datetime

    from leafmap import Map
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

__all__ = [
    "plot_map",
    "plot_before_after",
    "plot_temporal_trajectory",
    "plot_severity_summary",
    "plot_difference_map",
    "interactive_map",
    "show_product",
    "save_figure",
    "add_basemap",
    "load_fire_perimeters",
    "overlay_perimeters",
    "add_scalebar",
    "PRODUCT_STYLES",
]

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ProductStyle:
    """Colormap and scale configuration for a single Tanager product."""

    cmap: str
    vmin: float
    vmax: float
    label: str
    class_ticks: Optional[List[float]]


#: Mapping of product name → :class:`ProductStyle` presets used by plotting helpers.
PRODUCT_STYLES: Dict[str, ProductStyle] = {
    "nbr": ProductStyle(
        cmap="RdYlGn",
        vmin=-1.0,
        vmax=1.0,
        label="NBR",
        class_ticks=None,
    ),
    "ndvi": ProductStyle(
        cmap="RdYlGn",
        vmin=-1.0,
        vmax=1.0,
        label="NDVI",
        class_ticks=None,
    ),
    "ndwi": ProductStyle(
        cmap="RdYlBu",
        vmin=-1.0,
        vmax=1.0,
        label="NDWI",
        class_ticks=None,
    ),
    "dnbr": ProductStyle(
        cmap="RdYlGn_r",
        vmin=-0.5,
        vmax=1.3,
        label="dNBR (Burn Severity)",
        class_ticks=[0.1, 0.27, 0.44, 0.66],
    ),
    "cbi": ProductStyle(
        cmap="YlOrRd",
        vmin=0.0,
        vmax=3.0,
        label="CBI (Composite Burn Index)",
        class_ticks=[0.0, 1.0, 2.0, 3.0],
    ),
    "severity": ProductStyle(
        cmap="tab10",
        vmin=0,
        vmax=5,
        label="Severity Class",
        class_ticks=[0, 1, 2, 3, 4, 5],
    ),
    "char": ProductStyle(
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        label="Char Fraction",
        class_ticks=None,
    ),
    "pv": ProductStyle(
        cmap="Greens",
        vmin=0.0,
        vmax=1.0,
        label="Photosynthetic Vegetation",
        class_ticks=None,
    ),
    "npv": ProductStyle(
        cmap="YlOrBr",
        vmin=0.0,
        vmax=1.0,
        label="Non-Photosynthetic Vegetation",
        class_ticks=None,
    ),
    "soil": ProductStyle(
        cmap="copper",
        vmin=0.0,
        vmax=1.0,
        label="Soil Fraction",
        class_ticks=None,
    ),
    "lfmc": ProductStyle(
        cmap="RdYlGn",
        vmin=0.0,
        vmax=200.0,
        label="LFMC (%)",
        class_ticks=[30, 60, 90, 120],
    ),
}


# ---------------------------------------------------------------------------
# Public API stubs
# ---------------------------------------------------------------------------


def plot_map(
    da: xr.DataArray,
    title: str = "",
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    product_name: Optional[str] = None,
    publication: bool = False,
    figsize: Tuple[float, float] = (10, 8),
    basemap: bool = False,
    ax: Optional["Axes"] = None,
) -> "Figure":
    """Render a single-band raster as a georeferenced map with UTM axes.

    Parameters
    ----------
    da:
        2-D DataArray with ``x`` (easting) and ``y`` (northing) coordinates in
        metres (UTM).
    title:
        Figure title string.
    cmap:
        Colormap name.  When *None* and *product_name* is given, the value from
        :data:`PRODUCT_STYLES` is used.
    vmin, vmax:
        Colour scale limits.  When *None* and *product_name* is given, the
        values from :data:`PRODUCT_STYLES` are used.
    product_name:
        Key into :data:`PRODUCT_STYLES` (e.g. ``"nbr"``).  Provides default
        *cmap*, *vmin*, *vmax*, and colorbar label when explicit parameters are
        not supplied.
    publication:
        When ``True`` use DPI 300 and larger font sizes suitable for figures
        destined for publication.
    figsize:
        ``(width, height)`` in inches for the new figure.  Ignored when *ax*
        is provided.
    basemap:
        When ``True`` call :func:`add_basemap` after rendering the raster.  If
        :func:`add_basemap` raises :exc:`NotImplementedError` the error is
        silently swallowed (the basemap is still a stub).
    ax:
        Existing matplotlib ``Axes`` to draw into.  A new figure is created
        when *ax* is ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import copy as _copy

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # --- resolve style from PRODUCT_STYLES when product_name provided ----------
    style_label: Optional[str] = None
    if product_name is not None:
        style = PRODUCT_STYLES.get(product_name)
        if style is not None:
            if cmap is None:
                cmap = style.cmap
            if vmin is None:
                vmin = style.vmin
            if vmax is None:
                vmax = style.vmax
            style_label = style.label
        else:
            logger.warning(
                "product_name %r not found in PRODUCT_STYLES; ignoring", product_name
            )

    # --- figure / axes setup ---------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # --- publication font sizes ------------------------------------------------
    title_fontsize = 14
    label_fontsize = 12
    tick_labelsize = 10
    if publication:
        title_fontsize = 18
        label_fontsize = 14
        tick_labelsize = 12

    # --- extract coordinate bounds ---------------------------------------------
    # Support both 'x'/'y' and 'easting'/'northing' dimension names, falling
    # back gracefully to pixel-index axes when no recognised coordinate is found.
    x_coord: Optional[np.ndarray] = None
    y_coord: Optional[np.ndarray] = None
    for name in ("x", "easting", "lon", "longitude"):
        if name in da.coords:
            x_coord = da.coords[name].values
            break
    for name in ("y", "northing", "lat", "latitude"):
        if name in da.coords:
            y_coord = da.coords[name].values
            break

    if x_coord is not None and y_coord is not None:
        xmin = float(x_coord.min())
        xmax = float(x_coord.max())
        # Half-pixel expansion so the extent aligns with cell edges.
        dx = (xmax - xmin) / max(len(x_coord) - 1, 1) if len(x_coord) > 1 else 1.0
        ymin = float(y_coord.min())
        ymax = float(y_coord.max())
        dy = (ymax - ymin) / max(len(y_coord) - 1, 1) if len(y_coord) > 1 else 1.0
        extent: Optional[list] = [
            xmin - dx / 2,
            xmax + dx / 2,
            ymin - dy / 2,
            ymax + dy / 2,
        ]
        use_geo_axes = True
    else:
        extent = None
        use_geo_axes = False

    # --- handle NaN values via masked array ------------------------------------
    arr = np.ma.masked_invalid(da.values)

    # Build a copy of the colormap with NaN/masked pixels rendered transparent.
    if cmap is not None:
        cm_obj = _copy.copy(plt.get_cmap(cmap))
    else:
        cm_obj = _copy.copy(plt.get_cmap("viridis"))
    cm_obj.set_bad(color="white", alpha=0.0)

    # --- render raster ---------------------------------------------------------
    im = ax.imshow(
        arr,
        extent=extent,
        origin="lower",
        cmap=cm_obj,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # --- format axes -----------------------------------------------------------
    if use_geo_axes:
        km_formatter = FuncFormatter(lambda v, _: f"{v / 1000:.0f}")
        ax.xaxis.set_major_formatter(km_formatter)
        ax.yaxis.set_major_formatter(km_formatter)
        ax.set_xlabel("Easting (km)", fontsize=label_fontsize)
        ax.set_ylabel("Northing (km)", fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_labelsize)

    # --- title -----------------------------------------------------------------
    if title:
        ax.set_title(title, fontsize=title_fontsize)

    # --- colorbar --------------------------------------------------------------
    cb_label = style_label if style_label is not None else (product_name or "")
    cbar = fig.colorbar(im, ax=ax)
    if cb_label:
        cbar.set_label(cb_label, fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    # --- optional basemap overlay ----------------------------------------------
    if basemap:
        add_basemap(ax)

    # --- publication DPI -------------------------------------------------------
    if publication:
        fig.set_dpi(300)

    return fig


def plot_before_after(
    pre: xr.DataArray,
    post: xr.DataArray,
    product_name: str = "nbr",
    pre_label: Optional[str] = None,
    post_label: Optional[str] = None,
    fire_perimeters: Optional[Any] = None,
    basemap: bool = False,
    publication: bool = False,
    figsize: Tuple[float, float] = (16, 8),
) -> "Figure":
    """Render a side-by-side pre/post fire comparison figure.

    Each panel is rendered at its own spatial extent using the same colormap
    and value range drawn from :data:`PRODUCT_STYLES`.  A single shared
    colorbar spans both panels.

    Parameters
    ----------
    pre:
        Pre-fire raster DataArray with ``x`` (easting) and ``y`` (northing)
        coordinates in metres (UTM).
    post:
        Post-fire raster DataArray.  May have a different spatial extent than
        *pre*; each panel is rendered at its own extent.
    product_name:
        Key into :data:`PRODUCT_STYLES` (e.g. ``"nbr"``).  Provides the
        colormap (*cmap*), *vmin*, *vmax*, and colorbar label.
    pre_label:
        Title for the left (pre-fire) panel.  Defaults to ``"Pre-Fire"``.
    post_label:
        Title for the right (post-fire) panel.  Defaults to ``"Post-Fire"``.
    fire_perimeters:
        Optional GeoDataFrame of fire perimeter polygons.  When provided,
        :func:`overlay_perimeters` is called on both panels.
    basemap:
        When ``True`` call :func:`add_basemap` on both panels after rendering.
    publication:
        When ``True`` use DPI 300 and larger font sizes.
    figsize:
        ``(width, height)`` in inches for the new figure.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with two map panels and a shared colorbar.
    """
    import copy as _copy

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # --- resolve style --------------------------------------------------------
    style = PRODUCT_STYLES.get(product_name)
    if style is not None:
        cmap = style.cmap
        vmin = style.vmin
        vmax = style.vmax
        cb_label = style.label
    else:
        logger.warning(
            "plot_before_after: product_name %r not found in PRODUCT_STYLES; "
            "using viridis with data-range scaling",
            product_name,
        )
        cmap = "viridis"
        vmin = None
        vmax = None
        cb_label = product_name or ""

    # --- font sizes -----------------------------------------------------------
    title_fontsize = 14
    label_fontsize = 12
    tick_labelsize = 10
    if publication:
        title_fontsize = 18
        label_fontsize = 14
        tick_labelsize = 12

    # --- figure / axes --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax_pre, ax_post = axes

    # --- colormap with transparent NaN ----------------------------------------
    cm_obj = _copy.copy(plt.get_cmap(cmap))
    cm_obj.set_bad(color="white", alpha=0.0)

    # --- helper: compute extent from a DataArray ------------------------------
    def _extent(da: xr.DataArray) -> Optional[list]:
        x_coord: Optional["np.ndarray"] = None
        y_coord: Optional["np.ndarray"] = None
        for name in ("x", "easting", "lon", "longitude"):
            if name in da.coords:
                x_coord = da.coords[name].values
                break
        for name in ("y", "northing", "lat", "latitude"):
            if name in da.coords:
                y_coord = da.coords[name].values
                break

        if x_coord is None or y_coord is None:
            return None

        xmin = float(x_coord.min())
        xmax = float(x_coord.max())
        dx = (xmax - xmin) / max(len(x_coord) - 1, 1) if len(x_coord) > 1 else 1.0
        ymin = float(y_coord.min())
        ymax = float(y_coord.max())
        dy = (ymax - ymin) / max(len(y_coord) - 1, 1) if len(y_coord) > 1 else 1.0
        return [xmin - dx / 2, xmax + dx / 2, ymin - dy / 2, ymax + dy / 2]

    # --- helper: render one panel and return the AxesImage --------------------
    def _render_panel(ax: "Axes", da: xr.DataArray, panel_label: str) -> Any:
        arr = np.ma.masked_invalid(da.values)
        ext = _extent(da)
        use_geo = ext is not None

        im = ax.imshow(
            arr,
            extent=ext,
            origin="lower",
            cmap=cm_obj,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        if use_geo:
            km_fmt = FuncFormatter(lambda v, _: f"{v / 1000:.0f}")
            ax.xaxis.set_major_formatter(km_fmt)
            ax.yaxis.set_major_formatter(km_fmt)
            ax.set_xlabel("Easting (km)", fontsize=label_fontsize)
            ax.set_ylabel("Northing (km)", fontsize=label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_labelsize)

        ax.set_title(panel_label, fontsize=title_fontsize)
        return im

    # --- render both panels ---------------------------------------------------
    _render_panel(ax_pre, pre, pre_label or "Pre-Fire")
    im_post = _render_panel(ax_post, post, post_label or "Post-Fire")

    # --- shared colorbar spanning both panels ---------------------------------
    cbar = fig.colorbar(
        im_post,
        ax=axes.tolist(),
        orientation="horizontal",
        fraction=0.05,
        pad=0.08,
    )
    if cb_label:
        cbar.set_label(cb_label, fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    # --- optional overlays ----------------------------------------------------
    if fire_perimeters is not None:
        overlay_perimeters(ax_pre, fire_perimeters)
        overlay_perimeters(ax_post, fire_perimeters)

    if basemap:
        add_basemap(ax_pre)
        add_basemap(ax_post)

    # --- publication DPI ------------------------------------------------------
    if publication:
        fig.set_dpi(300)

    return fig


def plot_temporal_trajectory(
    dates: list,
    values: "list[float]",
    product_name: str = "NBR",
    fire_date: "str | datetime | None" = "2025-01-07",
    error_bands: "list[float] | None" = None,
    ax: "Optional[Axes]" = None,
    publication: bool = False,
    figsize: "Tuple[float, float]" = (12, 6),
) -> "Figure":
    """Plot a time-series trajectory (e.g. NBR) for a pixel or region of interest.

    Parameters
    ----------
    dates:
        Sequence of date values.  Each entry may be a :class:`str` (e.g.
        ``"2024-12-15"``), a :class:`datetime.datetime`, or any type accepted
        by :func:`pandas.to_datetime`.
    values:
        Corresponding spectral-index values (one per date).
    product_name:
        Label for the y-axis and the series legend entry (e.g. ``"NBR"``).
    fire_date:
        Date of the fire ignition event.  A red dashed vertical line and a
        legend entry are added when this argument is not ``None``.  Accepts
        the same types as *dates*.
    error_bands:
        Per-date ±uncertainty values.  When provided, a shaded band is drawn
        around the line using :meth:`~matplotlib.axes.Axes.fill_between`.
    ax:
        Existing :class:`~matplotlib.axes.Axes` to draw into.  A new figure
        is created when ``None``.
    publication:
        When ``True`` set DPI to 300 and increase font sizes for
        publication-quality output.
    figsize:
        ``(width, height)`` in inches for the new figure.  Ignored when
        *ax* is provided.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the time-series line chart.

    Examples
    --------
    >>> import matplotlib; matplotlib.use("Agg")
    >>> from tanager.visualization import plot_temporal_trajectory
    >>> dates = ["2024-12-15", "2024-12-25", "2025-01-07", "2025-01-15"]
    >>> values = [0.65, 0.62, 0.15, 0.20]
    >>> fig = plot_temporal_trajectory(dates, values, "NBR", fire_date="2025-01-07")
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd

    # --- parse dates -----------------------------------------------------------
    # pd.to_datetime handles strings, datetime objects, Timestamps, numpy
    # datetime64, and any other type pandas recognises.
    parsed_dates = pd.to_datetime(dates)

    # --- parse fire_date -------------------------------------------------------
    parsed_fire_date = None
    if fire_date is not None:
        parsed_fire_date = pd.to_datetime(fire_date)

    # --- font sizes -------------------------------------------------------
    label_fontsize = 12
    tick_labelsize = 10
    legend_fontsize = 10
    if publication:
        label_fontsize = 16
        tick_labelsize = 13
        legend_fontsize = 13

    # --- figure / axes setup --------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # --- main line plot -------------------------------------------------------
    ax.plot(
        parsed_dates,
        values,
        marker="o",
        linewidth=2,
        label=product_name,
        zorder=3,
    )

    # --- optional error-band shading ------------------------------------------
    if error_bands is not None:
        lower = [v - e for v, e in zip(values, error_bands)]
        upper = [v + e for v, e in zip(values, error_bands)]
        ax.fill_between(parsed_dates, lower, upper, alpha=0.2, label=f"{product_name} ±uncertainty")

    # --- fire date vertical line + phase labels --------------------------------
    if parsed_fire_date is not None:
        ax.axvline(
            parsed_fire_date,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Fire Ignition",
            zorder=4,
        )

        # Shaded background regions: pre-fire (light green), post-fire (light red)
        xmin_date = parsed_dates.min()
        xmax_date = parsed_dates.max()

        # Only shade pre-fire region if the fire date is not before the first date
        if parsed_fire_date > xmin_date:
            ax.axvspan(
                xmin_date,
                parsed_fire_date,
                alpha=0.06,
                color="green",
                zorder=0,
            )
            # "Pre-Fire" label: positioned in the left half of the pre-fire region
            pre_mid = xmin_date + (parsed_fire_date - xmin_date) / 2
            y_range = ax.get_ylim()
            pre_label_y = y_range[0] + (y_range[1] - y_range[0]) * 0.92
            ax.text(
                pre_mid,
                pre_label_y,
                "Pre-Fire",
                ha="center",
                va="top",
                fontsize=label_fontsize - 1,
                color="darkgreen",
                alpha=0.7,
            )

        # Post-fire / Recovery region
        if parsed_fire_date < xmax_date:
            ax.axvspan(
                parsed_fire_date,
                xmax_date,
                alpha=0.06,
                color="red",
                zorder=0,
            )
            post_mid = parsed_fire_date + (xmax_date - parsed_fire_date) / 2
            y_range = ax.get_ylim()
            post_label_y = y_range[0] + (y_range[1] - y_range[0]) * 0.92
            ax.text(
                post_mid,
                post_label_y,
                "Post-Fire / Recovery",
                ha="center",
                va="top",
                fontsize=label_fontsize - 1,
                color="darkred",
                alpha=0.7,
            )

    # --- axis labels and formatting -------------------------------------------
    ax.set_ylabel(product_name, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_labelsize)

    # Auto date formatting on the x-axis: monthly major ticks for multi-month
    # series, otherwise let matplotlib choose.
    date_range_days = (parsed_dates.max() - parsed_dates.min()).days
    if date_range_days > 60:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))

    fig.autofmt_xdate(rotation=30, ha="right")

    # --- legend ---------------------------------------------------------------
    ax.legend(fontsize=legend_fontsize)

    # --- publication DPI ------------------------------------------------------
    if publication:
        fig.set_dpi(300)

    return fig


def plot_severity_summary(
    fractions: xr.Dataset,
    cbi: xr.DataArray,
    severity_class: xr.DataArray,
    publication: bool = False,
    figsize: Tuple[float, float] = (18, 12),
) -> "Figure":
    """Render a 2\u00d73 multi-panel summary grid of spectral fractions and burn severity.

    Panels are arranged in two rows:

    * Top row (left to right): Char fraction, Photosynthetic Vegetation, Non-PV
    * Bottom row (left to right): Soil fraction, Composite Burn Index, Severity Class

    Each panel uses its own colormap and scale drawn from :data:`PRODUCT_STYLES`,
    has an individual colorbar, and shares the same UTM geographic formatting
    (Easting / Northing km labels).

    Parameters
    ----------
    fractions:
        Dataset containing 2-D DataArrays for the spectral fraction products.
        Required variables: ``"char"``, ``"pv"``, ``"npv"``, ``"soil"``.
        All arrays must share the same ``x`` (easting) and ``y`` (northing)
        coordinates in metres (UTM).
    cbi:
        2-D DataArray of the Composite Burn Index (CBI), sharing the same
        spatial coordinates as *fractions*.
    severity_class:
        2-D DataArray of integer burn-severity classes (e.g. 0–5), sharing
        the same spatial coordinates as *fractions*.
    publication:
        When ``True`` set DPI to 300 and use larger font sizes suitable for
        publication-quality output.
    figsize:
        ``(width, height)`` in inches for the figure.  Defaults to
        ``(18, 12)``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing 6 georeferenced panels (plus individual colorbars).
    """
    import copy as _copy

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # --- font sizes -------------------------------------------------------------
    title_fontsize = 13
    label_fontsize = 11
    tick_labelsize = 9
    if publication:
        title_fontsize = 17
        label_fontsize = 13
        tick_labelsize = 11

    # --- panel definitions: (product_name, data_array) -------------------------
    panels: List[Tuple[str, xr.DataArray]] = [
        ("char", fractions["char"]),
        ("pv",   fractions["pv"]),
        ("npv",  fractions["npv"]),
        ("soil", fractions["soil"]),
        ("cbi",  cbi),
        ("severity", severity_class),
    ]

    # --- helper: compute imshow extent from a DataArray -----------------------
    def _extent(da: xr.DataArray) -> Optional[list]:
        x_coord: Optional[np.ndarray] = None
        y_coord: Optional[np.ndarray] = None
        for name in ("x", "easting", "lon", "longitude"):
            if name in da.coords:
                x_coord = da.coords[name].values
                break
        for name in ("y", "northing", "lat", "latitude"):
            if name in da.coords:
                y_coord = da.coords[name].values
                break
        if x_coord is None or y_coord is None:
            return None
        xmin = float(x_coord.min())
        xmax = float(x_coord.max())
        dx = (xmax - xmin) / max(len(x_coord) - 1, 1) if len(x_coord) > 1 else 1.0
        ymin = float(y_coord.min())
        ymax = float(y_coord.max())
        dy = (ymax - ymin) / max(len(y_coord) - 1, 1) if len(y_coord) > 1 else 1.0
        return [xmin - dx / 2, xmax + dx / 2, ymin - dy / 2, ymax + dy / 2]

    # --- create 2x3 grid -------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    km_formatter = FuncFormatter(lambda v, _: f"{v / 1000:.0f}")

    for ax, (product_name, da) in zip(axes.flat, panels):
        style = PRODUCT_STYLES.get(product_name)
        if style is not None:
            cmap_name = style.cmap
            vmin = style.vmin
            vmax = style.vmax
            cb_label = style.label
        else:
            logger.warning(
                "plot_severity_summary: product %r not found in PRODUCT_STYLES; "
                "using viridis with data-range scaling",
                product_name,
            )
            cmap_name = "viridis"
            vmin = None
            vmax = None
            cb_label = product_name

        # --- NaN-masked array --------------------------------------------------
        arr = np.ma.masked_invalid(da.values)

        # --- colormap with transparent NaN pixels ------------------------------
        cm_obj = _copy.copy(plt.get_cmap(cmap_name))
        cm_obj.set_bad(color="white", alpha=0.0)

        # --- compute geographic extent -----------------------------------------
        ext = _extent(da)
        use_geo = ext is not None

        # --- render raster -----------------------------------------------------
        im = ax.imshow(
            arr,
            extent=ext,
            origin="lower",
            cmap=cm_obj,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        # --- UTM axis formatting -----------------------------------------------
        if use_geo:
            ax.xaxis.set_major_formatter(km_formatter)
            ax.yaxis.set_major_formatter(km_formatter)
            ax.set_xlabel("Easting (km)", fontsize=label_fontsize)
            ax.set_ylabel("Northing (km)", fontsize=label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_labelsize)

        ax.set_title(cb_label, fontsize=title_fontsize)

        # --- per-panel colorbar ------------------------------------------------
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cb_label, fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=tick_labelsize)

    # --- spacing ---------------------------------------------------------------
    fig.tight_layout()

    # --- publication DPI -------------------------------------------------------
    if publication:
        fig.set_dpi(300)

    return fig


def plot_difference_map(
    diff_da: xr.DataArray,
    product_name: str = "dnbr",
    class_boundaries: Optional[Dict[str, float]] = None,
    publication: bool = False,
    figsize: Tuple[float, float] = (10, 8),
) -> "Figure":
    """Render a styled dNBR (or other difference) raster with severity contour overlays.

    The base raster is rendered using :func:`plot_map` with the colormap and
    scale from :data:`PRODUCT_STYLES`.  When *class_boundaries* are provided
    (or when *product_name* is ``"dnbr"`` and the default USGS thresholds
    apply), contour lines are drawn at each severity boundary and labelled with
    the corresponding class name.

    Parameters
    ----------
    diff_da:
        2-D DataArray of the difference product (e.g. dNBR) with ``x``
        (easting) and ``y`` (northing) coordinates in metres (UTM).
    product_name:
        Key into :data:`PRODUCT_STYLES` used to select the colormap and scale
        (e.g. ``"dnbr"``).  Defaults to ``"dnbr"``.
    class_boundaries:
        Mapping of class name → threshold value used to draw severity contours.
        When *None* and *product_name* is ``"dnbr"``, the standard USGS
        thresholds are applied::

            {"Unburned": 0.1, "Low": 0.27, "Mod-Low": 0.44, "Mod-High": 0.66}

        For any other *product_name* with ``class_boundaries=None``, no
        contours are drawn.
    publication:
        When ``True`` set DPI to 300 and use larger font sizes.
    figsize:
        ``(width, height)`` in inches for the new figure.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the difference-map raster with labelled severity
        contours (when applicable).
    """
    # Default USGS dNBR severity thresholds (Miller & Thode 2007).
    _USGS_DNBR_THRESHOLDS: Dict[str, float] = {
        "Unburned": 0.1,
        "Low": 0.27,
        "Mod-Low": 0.44,
        "Mod-High": 0.66,
    }

    # Resolve which boundaries to use.
    if class_boundaries is None and product_name == "dnbr":
        class_boundaries = _USGS_DNBR_THRESHOLDS

    # Render the base raster using plot_map (reuses all style logic).
    fig = plot_map(diff_da, product_name=product_name, publication=publication, figsize=figsize)
    ax = fig.axes[0]

    # Overlay contours when boundaries are provided.
    if class_boundaries:
        # Extract 1-D coordinate arrays from the DataArray.
        x_coords: Optional[np.ndarray] = None
        y_coords: Optional[np.ndarray] = None
        for name in ("x", "easting", "lon", "longitude"):
            if name in diff_da.coords:
                x_coords = diff_da.coords[name].values
                break
        for name in ("y", "northing", "lat", "latitude"):
            if name in diff_da.coords:
                y_coords = diff_da.coords[name].values
                break

        data = diff_da.values  # contour handles NaN gracefully

        # Sort boundary levels so contour draws them in ascending order.
        sorted_levels = sorted(class_boundaries.values())

        if x_coords is not None and y_coords is not None:
            cs = ax.contour(
                x_coords,
                y_coords,
                data,
                levels=sorted_levels,
                colors="black",
                linewidths=1.0,
            )
        else:
            # Fall back to pixel-index coordinates when no geo coords are present.
            cs = ax.contour(
                data,
                levels=sorted_levels,
                colors="black",
                linewidths=1.0,
            )

        # Build a label dict: level → class name.  When two class names share
        # the same threshold value only one label is shown (last one wins).
        level_to_name: Dict[float, str] = {
            v: k for k, v in class_boundaries.items()
        }
        fmt = {level: level_to_name.get(level, f"{level:.2f}") for level in sorted_levels}

        ax.clabel(cs, fmt=fmt, fontsize=8)

    return fig


def interactive_map(
    layers: "list[tuple[xr.DataArray, str]] | None" = None,
    center: "tuple[float, float] | None" = None,
    zoom: int = 12,
    perimeters: "str | Path | Any | None" = None,
    basemap: str = "satellite",
) -> "Map":
    """Return a leafmap or folium interactive map for use in Jupyter notebooks.

    Parameters
    ----------
    layers:
        List of ``(DataArray, product_name)`` pairs.  Each DataArray is added
        as a raster overlay with the colormap from :data:`PRODUCT_STYLES`.
        Any CRS is accepted; rasters are reprojected to ``EPSG:4326`` as
        needed for web-tile display.  Pass ``None`` to create a map with only
        the basemap and any perimeter overlays.
    center:
        ``(latitude, longitude)`` for the initial map view.  Defaults to the
        centroid of the first raster layer's geographic extent when *None*.
    zoom:
        Initial zoom level (Leaflet/folium integer zoom).  Defaults to 12.
    perimeters:
        Fire perimeter polygons.  Accepts:

        * ``str`` or :class:`pathlib.Path` — path to a GeoJSON, Shapefile, or
          GeoPackage; loaded via :func:`load_fire_perimeters`.
        * ``geopandas.GeoDataFrame`` — used directly.
        * ``None`` — no perimeter overlay.
    basemap:
        Basemap style name.  Used only for the leafmap backend.
        ``"satellite"`` maps to ``"Esri.WorldImagery"``; ``"terrain"`` maps
        to ``"Esri.WorldShadedRelief"``; ``"osm"`` maps to
        ``"OpenStreetMap"``; any other string is passed through to leafmap
        directly.  The folium fallback always uses OpenStreetMap tiles.

    Returns
    -------
    leafmap.Map or folium.Map
        A displayable map widget (in Jupyter) or renderable folium Map.

    Raises
    ------
    ImportError
        When neither leafmap nor folium is installed.

    Examples
    --------
    >>> from unittest.mock import patch, MagicMock
    >>> with patch.dict("sys.modules", {"leafmap": MagicMock()}):
    ...     pass  # would return a mocked Map object
    """
    import tempfile

    # --- resolve perimeters -------------------------------------------------------
    # Do this early so we can fail fast before expensive raster I/O.
    perimeter_gdf = None
    if perimeters is not None:
        if isinstance(perimeters, (str, Path)):
            perimeter_gdf = load_fire_perimeters(perimeters)
        else:
            # Assume GeoDataFrame-like
            perimeter_gdf = perimeters

    # --- helper: reproject DataArray to EPSG:4326 for web display ---------------
    def _reproject_to_4326(da: xr.DataArray) -> xr.DataArray:
        """Reproject *da* to EPSG:4326 if not already in that CRS."""
        import rioxarray as _rxr  # noqa: F401 — registers .rio accessor

        if da.rio.crs is None:
            logger.warning(
                "interactive_map: DataArray has no CRS; assuming EPSG:4326"
            )
            da = da.rio.write_crs("EPSG:4326")
            return da

        if str(da.rio.crs).upper() != "EPSG:4326":
            da = da.rio.reproject("EPSG:4326")

        return da

    def _center_from_da(da: xr.DataArray) -> "tuple[float, float]":
        """Return ``(lat, lon)`` centroid of *da* (must be in EPSG:4326)."""
        lon_min = float(da.x.min())
        lon_max = float(da.x.max())
        lat_min = float(da.y.min())
        lat_max = float(da.y.max())
        return ((lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0)

    # --- reproject all layers to EPSG:4326 up-front ----------------------------
    layers_4326: "list[tuple[xr.DataArray, str]]" = []
    if layers:
        for da, product_name in layers:
            import rioxarray as _rxr  # noqa: F401
            da_4326 = _reproject_to_4326(da)
            layers_4326.append((da_4326, product_name))

    # --- determine map center --------------------------------------------------
    if center is None:
        if layers_4326:
            center = _center_from_da(layers_4326[0][0])
        else:
            # No layers: default to continental US
            center = (39.5, -98.35)

    # Basemap name mapping for leafmap
    _BASEMAP_MAP = {
        "satellite": "Esri.WorldImagery",
        "terrain": "Esri.WorldShadedRelief",
        "osm": "OpenStreetMap",
    }

    # --- attempt leafmap (primary) --------------------------------------------
    try:
        import leafmap  # lazy import

        leafmap_basemap = _BASEMAP_MAP.get(basemap, basemap)

        # leafmap.Map takes center as [lat, lon]
        m = leafmap.Map(
            center=list(center),
            zoom=zoom,
            basemap=leafmap_basemap,
        )

        # Track temp file paths so they remain on disk for the map session
        _tmp_files: List[str] = []

        for da_4326, product_name in layers_4326:
            style = PRODUCT_STYLES.get(product_name)

            cmap_name: Optional[str] = None
            vmin_val: Optional[float] = None
            vmax_val: Optional[float] = None
            if style is not None:
                cmap_name = style.cmap
                vmin_val = style.vmin
                vmax_val = style.vmax

            # Write to a named temp GeoTIFF — leafmap.add_raster needs a file path
            tmp_f = tempfile.NamedTemporaryFile(
                suffix=".tif", delete=False, prefix="tanager_"
            )
            tmp_path = tmp_f.name
            tmp_f.close()
            _tmp_files.append(tmp_path)

            da_4326.rio.to_raster(tmp_path)

            m.add_raster(
                tmp_path,
                colormap=cmap_name,
                vmin=vmin_val,
                vmax=vmax_val,
                layer_name=product_name,
                zoom_to_layer=(len(_tmp_files) == 1),  # zoom on first layer only
            )

        # --- perimeter overlay ------------------------------------------------
        if perimeter_gdf is not None and len(perimeter_gdf) > 0:
            peri_4326 = perimeter_gdf.to_crs("EPSG:4326")
            tmp_gj = tempfile.NamedTemporaryFile(
                suffix=".geojson", delete=False, prefix="tanager_peri_"
            )
            tmp_gj_path = tmp_gj.name
            tmp_gj.close()
            peri_4326.to_file(tmp_gj_path, driver="GeoJSON")
            _tmp_files.append(tmp_gj_path)

            m.add_geojson(
                tmp_gj_path,
                layer_name="Fire Perimeters",
                style={
                    "color": "red",
                    "weight": 2,
                    "fillOpacity": 0.05,
                },
            )

        m.add_layer_control()
        return m

    except ImportError:
        pass  # fall through to folium

    # --- attempt folium (fallback) --------------------------------------------
    try:
        import folium  # lazy import
        import matplotlib.colors as _mpl_colors
        import matplotlib.pyplot as _plt

        m_folium = folium.Map(location=list(center), zoom_start=zoom)

        for da_4326, product_name in layers_4326:
            style = PRODUCT_STYLES.get(product_name)

            cmap_name = "viridis"
            vmin_val = None
            vmax_val = None
            if style is not None:
                cmap_name = style.cmap
                vmin_val = style.vmin
                vmax_val = style.vmax

            cmap_obj = _plt.get_cmap(cmap_name)
            _vmin = vmin_val if vmin_val is not None else float(np.nanmin(da_4326.values))
            _vmax = vmax_val if vmax_val is not None else float(np.nanmax(da_4326.values))
            norm = _mpl_colors.Normalize(vmin=_vmin, vmax=_vmax)

            # Build a closure-safe colormap callable for folium
            def _make_colormap_fn(cm, nm):
                def _fn(x):
                    return cm(nm(x))
                return _fn

            colormap_fn = _make_colormap_fn(cmap_obj, norm)

            lat_min = float(da_4326.y.min())
            lat_max = float(da_4326.y.max())
            lon_min = float(da_4326.x.min())
            lon_max = float(da_4326.x.max())
            bounds = [[lat_min, lon_min], [lat_max, lon_max]]

            folium.raster_layers.ImageOverlay(
                image=da_4326.values,
                bounds=bounds,
                colormap=colormap_fn,
                name=product_name,
                opacity=0.8,
            ).add_to(m_folium)

        # --- perimeter overlay ------------------------------------------------
        if perimeter_gdf is not None and len(perimeter_gdf) > 0:
            peri_4326 = perimeter_gdf.to_crs("EPSG:4326")
            folium.GeoJson(
                peri_4326.__geo_interface__,
                name="Fire Perimeters",
                style_function=lambda _: {
                    "color": "red",
                    "weight": 2,
                    "fillOpacity": 0.05,
                },
            ).add_to(m_folium)

        folium.LayerControl().add_to(m_folium)
        return m_folium

    except ImportError:
        pass

    raise ImportError(
        "interactive_map requires either 'leafmap' or 'folium'. "
        "Install one with: pip install leafmap  or  pip install folium"
    )


def show_product(
    da: xr.DataArray,
    product_name: Optional[str] = None,
    scene_date: Optional[str] = None,
    interactive: bool = False,
) -> "Figure | Map":
    """Convenience helper to display a named Tanager product as a static or interactive map.

    Parameters
    ----------
    da:
        2-D DataArray to display.  When ``da.name`` is set and *product_name*
        is not provided, the name attribute is used as the product key.
    product_name:
        Key into :data:`PRODUCT_STYLES` (e.g. ``"nbr"``, ``"dnbr"``).  When
        *None*, the function attempts to read the name from ``da.name``.
    scene_date:
        Optional date string (e.g. ``"2025-01-23"``) appended to the figure
        title for static maps.  Ignored when *interactive* is ``True``.
    interactive:
        When ``True`` return an interactive :func:`interactive_map` widget
        (leafmap / folium).  When ``False`` (default) return a static
        matplotlib :class:`~matplotlib.figure.Figure` with a basemap overlay.

    Returns
    -------
    matplotlib.figure.Figure
        When *interactive* is ``False``.
    leafmap.Map or folium.Map
        When *interactive* is ``True``.

    Examples
    --------
    >>> import matplotlib; matplotlib.use("Agg")
    >>> import numpy as np, xarray as xr
    >>> import rioxarray  # noqa: F401
    >>> x = np.linspace(340000, 350000, 10)
    >>> y = np.linspace(3780000, 3790000, 10)
    >>> da = xr.DataArray(np.random.rand(10, 10), coords={"y": y, "x": x}, dims=["y", "x"])
    >>> da = da.rio.write_crs("EPSG:32611")
    >>> fig = show_product(da, "nbr", "2025-01-23")
    >>> hasattr(fig, "axes")
    True
    """
    # Auto-detect product_name from da.name when not explicitly supplied.
    resolved_name: Optional[str] = product_name
    if resolved_name is None and getattr(da, "name", None):
        resolved_name = da.name

    if interactive:
        return interactive_map([(da, resolved_name or "")])

    title = f"{(resolved_name or '').upper()} {scene_date or ''}".strip()
    return plot_map(da, product_name=resolved_name, title=title, basemap=True)


def save_figure(
    fig: "Figure",
    path: Union[str, Path],
    formats: List[str] = ["png"],
) -> List[Path]:
    """Save a matplotlib figure to disk in one or more formats.

    Parameters
    ----------
    fig:
        Figure to save.
    path:
        Base output path **without** extension.  Each entry in *formats* is
        appended as the file extension (e.g. "out/test" → "out/test.png").
    formats:
        List of format strings supported by matplotlib (e.g. ["png", "pdf",
        "svg"]).  Defaults to ["png"].

    Returns
    -------
    list[pathlib.Path]
        Resolved :class:`pathlib.Path` objects for each written file, in the
        same order as *formats*.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> paths = save_figure(fig, "outputs/my_figure", ["png", "pdf"])
    >>> # writes outputs/my_figure.png and outputs/my_figure.pdf
    """
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for fmt in formats:
        out = base.parent / f"{base.name}.{fmt}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        written.append(out)
        logger.debug("Saved figure to %s", out)
    return written


def add_basemap(
    ax: "Axes",
    source: str = "satellite",
    alpha: float = 0.3,
    crs: str = "EPSG:32611",
) -> "Axes":
    """Overlay a web-tile basemap on *ax* using contextily.

    The raster rendered onto *ax* should already be present (so the axes
    extent is set) before calling this function.  The basemap tiles are
    placed at ``zorder=0``, underneath any existing raster overlay.

    Parameters
    ----------
    ax:
        Axes that already contains a raster in a projected CRS.  The axes
        extent must be set (i.e. a raster has been rendered) before calling
        this function.
    source:
        Tile provider name.  One of:

        * ``"satellite"`` — Esri World Imagery (default)
        * ``"terrain"`` — Stadia StamenTerrain
        * ``"osm"`` — OpenStreetMap Mapnik

        Any other value falls back to ``"satellite"`` with a warning.
    alpha:
        Opacity of the basemap tile layer, in [0, 1].  Defaults to 0.3 so
        the raster data drawn on top remains legible.
    crs:
        EPSG string for the coordinate reference system of *ax*.  Defaults
        to ``"EPSG:32611"`` (UTM Zone 11N), the project standard CRS.

    Returns
    -------
    matplotlib.axes.Axes
        The same *ax* object, with basemap tiles added in-place (or
        unchanged if the network request failed).
    """
    import contextily as ctx  # lazy import — not required in headless environments

    # Map source string to a contextily provider.
    _SOURCE_MAP = {
        "satellite": ctx.providers.Esri.WorldImagery,
        "terrain": ctx.providers.Stadia.StamenTerrain,
        "osm": ctx.providers.OpenStreetMap.Mapnik,
    }
    provider = _SOURCE_MAP.get(source)
    if provider is None:
        logger.warning(
            "add_basemap: unknown source %r; falling back to satellite tiles", source
        )
        provider = ctx.providers.Esri.WorldImagery

    try:
        ctx.add_basemap(ax, crs=crs, source=provider, alpha=alpha, zorder=0)
    except Exception as exc:  # network errors, tile fetch failures, etc.
        logger.warning(
            "add_basemap: failed to fetch basemap tiles (source=%r, crs=%r): %s",
            source,
            crs,
            exc,
        )

    return ax

def load_fire_perimeters(
    path: Union[str, Path],
    *,
    crs: Optional[Any] = None,
) -> Any:
    """Load NIFC/GeoMAC fire perimeter polygons from a file.

    Parameters
    ----------
    path:
        Path to a GeoJSON, Shapefile, or GeoPackage containing perimeters.
    crs:
        Reproject to this CRS after loading.  No reprojection when ``None``.

    Returns
    -------
    geopandas.GeoDataFrame
    """
    import geopandas as gpd  # lazy import — not required in headless environments

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fire perimeter file not found: {path}")

    gdf = gpd.read_file(path)

    if "geometry" not in gdf.columns:
        raise ValueError(
            f"File {path!r} has no geometry column; found columns: {list(gdf.columns)}"
        )

    if crs is not None:
        gdf = gdf.to_crs(crs)

    return gdf


def overlay_perimeters(
    ax: "Axes",
    perimeters: Any,
    color: str = "red",
    linestyle: str = "--",
    linewidth: float = 2.0,
    label: bool = True,
) -> "Axes":
    """Draw fire perimeter polygon(s) on *ax* as vector boundary overlays.

    The perimeters GeoDataFrame is reprojected to EPSG:32611 (UTM Zone 11N)
    before plotting so that it aligns correctly with the raster data rendered
    on *ax*.  If *perimeters* is empty (zero features) the function returns
    *ax* unchanged.

    Parameters
    ----------
    ax:
        Target matplotlib Axes.  Must already have a projected extent (e.g.
        after :func:`plot_map` has been called).
    perimeters:
        GeoDataFrame of fire perimeter polygons.  Any CRS is accepted; the
        data is reprojected to ``EPSG:32611`` internally.
    color:
        Line (and label text) colour.  Defaults to ``"red"``.
    linestyle:
        Line style string accepted by matplotlib (e.g. ``"--"``, ``"-"``,
        ``":"``).  Defaults to ``"--"`` (dashed).
    linewidth:
        Outline width in points.  Defaults to ``2.0``.
    label:
        When ``True`` and the GeoDataFrame contains a ``"name"`` or
        ``"incident_name"`` column, text labels are placed at the centroid
        of each polygon.

    Returns
    -------
    matplotlib.axes.Axes
        The same *ax* object, with perimeter boundaries (and optional labels)
        added in-place.
    """

    # Guard: return immediately for empty GeoDataFrame
    if len(perimeters) == 0:
        return ax

    # Reproject to the project-standard UTM CRS so boundaries align with rasters.
    perimeters = perimeters.to_crs("EPSG:32611")

    # Draw boundary lines for each polygon.
    perimeters.boundary.plot(
        ax=ax,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )

    # Optionally add text labels at polygon centroids.
    if label:
        # Determine which column to use for the label text.
        name_col: Optional[str] = None
        if "name" in perimeters.columns:
            name_col = "name"
        elif "incident_name" in perimeters.columns:
            name_col = "incident_name"

        if name_col is not None:
            for _, row in perimeters.iterrows():
                centroid = row.geometry.centroid
                name_text = str(row[name_col])
                ax.text(
                    centroid.x,
                    centroid.y,
                    name_text,
                    fontsize=8,
                    color=color,
                    ha="center",
                )

    return ax


def add_scalebar(
    ax: "Axes",
    length_km: float = 5.0,
    location: str = "lower left",
) -> "Axes":
    """Add a distance scale-bar to a map axes.

    Draws a black-filled rectangle with a white outline and a text label in
    data coordinates.  The bar width equals ``length_km * 1000`` metres, which
    aligns exactly with UTM (metre-unit) projected axes.  No external packages
    beyond matplotlib are required.

    Parameters
    ----------
    ax:
        Target matplotlib Axes.  Should already have its limits set (e.g.
        after :func:`plot_map` has been called) so that the bar is positioned
        correctly within the data extent.
    length_km:
        Desired bar length in kilometres.  Defaults to ``5.0`` km.
    location:
        Corner of the axes in which to anchor the bar.  Recognised values:

        * ``"lower left"`` (default)
        * ``"lower right"``
        * ``"upper left"``
        * ``"upper right"``

        Unrecognised values fall back to ``"lower left"`` with a warning.

    Returns
    -------
    matplotlib.axes.Axes
        The same *ax* object with the scale bar rectangle and label added
        in-place.
    """
    from matplotlib.patches import Rectangle  # lazy import

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    x_range = xmax - xmin
    y_range = ymax - ymin

    # Bar dimensions in data coordinates (UTM metres).
    bar_width = length_km * 1000.0
    bar_height = y_range * 0.01  # ~1 % of the y-range for a thin bar

    # Margin offsets: 5 % of each axis range from the chosen corner.
    x_margin = x_range * 0.05
    y_margin = y_range * 0.05

    # Resolve location to bar anchor (lower-left corner of the rectangle).
    location_lower = location.lower().strip()
    if location_lower == "lower left":
        bar_x = xmin + x_margin
        bar_y = ymin + y_margin
    elif location_lower == "lower right":
        bar_x = xmax - x_margin - bar_width
        bar_y = ymin + y_margin
    elif location_lower == "upper left":
        bar_x = xmin + x_margin
        bar_y = ymax - y_range * 0.10 - bar_height
    elif location_lower == "upper right":
        bar_x = xmax - x_margin - bar_width
        bar_y = ymax - y_range * 0.10 - bar_height
    else:
        logger.warning(
            "add_scalebar: unrecognised location %r; falling back to 'lower left'",
            location,
        )
        bar_x = xmin + x_margin
        bar_y = ymin + y_margin

    # Draw the scale bar rectangle: black fill, white outline.
    rect = Rectangle(
        (bar_x, bar_y),
        bar_width,
        bar_height,
        linewidth=1,
        edgecolor="white",
        facecolor="black",
        zorder=10,
    )
    ax.add_patch(rect)

    # Format label: omit the decimal when length_km is a whole number.
    if length_km == int(length_km):
        label = f"{int(length_km)} km"
    else:
        label = f"{length_km} km"

    # Place text centred horizontally above the bar.
    text_x = bar_x + bar_width / 2.0
    text_y = bar_y + bar_height + y_range * 0.005  # small gap above bar
    ax.text(
        text_x,
        text_y,
        label,
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
        zorder=11,
    )

    return ax
