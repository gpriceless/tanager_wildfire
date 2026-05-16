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
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import xarray as xr

if TYPE_CHECKING:  # pragma: no cover
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
    before: xr.DataArray,
    after: xr.DataArray,
    *,
    titles: Tuple[str, str] = ("Before", "After"),
    **kwargs: Any,
) -> "Figure":
    """Render a side-by-side pre/post fire comparison figure.

    Parameters
    ----------
    before:
        Pre-fire raster DataArray.
    after:
        Post-fire raster DataArray.
    titles:
        Panel titles for the before and after axes.
    **kwargs:
        Forwarded to :func:`plot_map`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError


def plot_temporal_trajectory(
    data: xr.DataArray,
    *,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    roi: Optional[Any] = None,
    ax: Optional["Axes"] = None,
    **kwargs: Any,
) -> "Figure":
    """Plot a time-series trajectory for a pixel or region of interest.

    Parameters
    ----------
    data:
        DataArray with a ``time`` dimension.
    lat, lon:
        Coordinates of the target pixel (WGS-84).  Mutually exclusive with
        *roi*.
    roi:
        Polygon or bounding-box used to spatially average the signal.
    ax:
        Existing Axes to draw into.

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError


def plot_severity_summary(
    severity: xr.DataArray,
    *,
    ax: Optional["Axes"] = None,
    bins: int = 50,
    **kwargs: Any,
) -> "Figure":
    """Produce a histogram + map summary of burn severity.

    Parameters
    ----------
    severity:
        Burn-severity raster (e.g., dNBR or RBR).
    ax:
        Existing Axes for the histogram panel.
    bins:
        Number of histogram bins.

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError


def plot_difference_map(
    before: xr.DataArray,
    after: xr.DataArray,
    *,
    ax: Optional["Axes"] = None,
    cmap: str = "RdBu_r",
    **kwargs: Any,
) -> "Figure":
    """Render a signed difference map (after − before).

    Parameters
    ----------
    before:
        Pre-fire raster DataArray.
    after:
        Post-fire raster DataArray.
    ax:
        Existing Axes to draw into.
    cmap:
        Diverging colormap; defaults to ``"RdBu_r"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError


def interactive_map(
    data: xr.DataArray,
    *,
    zoom: int = 10,
    **kwargs: Any,
) -> Any:
    """Return a folium or ipyleaflet interactive map (notebook use).

    Parameters
    ----------
    data:
        Raster DataArray to overlay.
    zoom:
        Initial zoom level.

    Returns
    -------
    folium.Map or ipyleaflet.Map
    """
    raise NotImplementedError


def show_product(
    product_name: str,
    data: xr.DataArray,
    **kwargs: Any,
) -> "Figure":
    """Convenience wrapper that applies :data:`PRODUCT_STYLES` and calls :func:`plot_map`.

    Parameters
    ----------
    product_name:
        Key into :data:`PRODUCT_STYLES` (e.g. ``"severity"``, ``"lfmc"``).
    data:
        DataArray to render.
    **kwargs:
        Override any style key from :data:`PRODUCT_STYLES`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError


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
    import geopandas as gpd  # lazy import — not required in headless environments

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
    *,
    length_km: Optional[float] = None,
    location: str = "lower right",
    **kwargs: Any,
) -> None:
    """Add a distance scale-bar to a map axes.

    Parameters
    ----------
    ax:
        Target matplotlib Axes (must have a projected CRS).
    length_km:
        Desired bar length in kilometres.  Auto-selected when ``None``.
    location:
        Corner of the axes to place the bar in.
    """
    raise NotImplementedError
