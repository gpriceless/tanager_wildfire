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

import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
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

#: Mapping of product name → style kwargs passed to plotting functions.
#: Populated by downstream configuration; left empty here so the module
#: is importable with no side-effects.
PRODUCT_STYLES: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Public API stubs
# ---------------------------------------------------------------------------


def plot_map(
    data: xr.DataArray,
    *,
    ax: Optional["Axes"] = None,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    **kwargs: Any,
) -> "Figure":
    """Render a single-band or RGB raster as a georeferenced map.

    Parameters
    ----------
    data:
        2-D or 3-D (band, y, x) DataArray to render.
    ax:
        Existing matplotlib Axes to draw into.  A new figure is created when
        *ax* is ``None``.
    cmap:
        Colormap name passed to ``imshow``.
    vmin, vmax:
        Colour scale limits.
    title:
        Figure title.
    **kwargs:
        Additional keyword arguments forwarded to the underlying imshow call.

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError


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
    *,
    dpi: int = 150,
    bbox_inches: str = "tight",
    **kwargs: Any,
) -> Path:
    """Save a matplotlib figure to disk with sane defaults.

    Parameters
    ----------
    fig:
        Figure to save.
    path:
        Output file path.  The format is inferred from the extension.
    dpi:
        Resolution in dots per inch.
    bbox_inches:
        Passed to ``savefig``; ``"tight"`` clips excess whitespace.

    Returns
    -------
    pathlib.Path
        Resolved path of the saved file.
    """
    raise NotImplementedError


def add_basemap(
    ax: "Axes",
    *,
    source: Optional[str] = None,
    crs: Optional[Any] = None,
    **kwargs: Any,
) -> None:
    """Overlay a web-tile basemap on *ax* using contextily.

    Parameters
    ----------
    ax:
        Axes that already contains a raster in a projected CRS.
    source:
        Tile provider URL or contextily provider object.  Defaults to
        ``contextily.providers.OpenStreetMap.Mapnik``.
    crs:
        CRS of *ax*; auto-detected from the axes extent when ``None``.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def overlay_perimeters(
    ax: "Axes",
    perimeters: Any,
    *,
    edgecolor: str = "red",
    facecolor: str = "none",
    linewidth: float = 1.5,
    **kwargs: Any,
) -> None:
    """Draw fire perimeter polygon(s) on *ax*.

    Parameters
    ----------
    ax:
        Target matplotlib Axes.
    perimeters:
        GeoDataFrame of perimeter polygons.
    edgecolor:
        Outline colour.
    facecolor:
        Fill colour; ``"none"`` produces an outline-only style.
    linewidth:
        Outline width in points.
    """
    raise NotImplementedError


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
