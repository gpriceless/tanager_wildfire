"""Regression tests for raster plot orientation.

Guards against the north-south flip where ``imshow(origin="lower")`` mirrored
descending-y (north-up) Tanager rasters vertically — the axes were correct but
the pixel content was upside down.
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import xarray as xr

from tanager.visualization import _row_origin, plot_map


def _north_high_raster(descending: bool) -> xr.DataArray:
    """A raster whose northern half is high-valued, on a chosen y ordering."""
    ny, nx = 20, 10
    y = np.linspace(3_778_000, 3_774_000, ny)  # north -> south
    if not descending:
        y = y[::-1]
    x = np.linspace(348_000, 352_000, nx)
    data = np.zeros((ny, nx), dtype=float)
    # Fill the northern half (largest y) with the high value.
    north_mask = y >= np.median(y)
    data[north_mask, :] = 1.0
    return xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x})


def test_row_origin_matches_y_direction():
    assert _row_origin(_north_high_raster(descending=True)) == "upper"
    assert _row_origin(_north_high_raster(descending=False)) == "lower"


def test_no_y_coord_defaults_to_lower():
    da = xr.DataArray(np.zeros((4, 4)), dims=("row", "col"))
    assert _row_origin(da) == "lower"


def _north_is_at_top(da: xr.DataArray) -> bool:
    ax = plot_map(da, title="orientation")
    fig = ax.get_figure()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    h = buf.shape[0]
    return buf[: h // 3].mean() > buf[-h // 3 :].mean()


def test_north_renders_at_top_descending_grid():
    # The real Tanager layout: descending y, northern half high-valued.
    assert _north_is_at_top(_north_high_raster(descending=True))


def test_north_renders_at_top_ascending_grid():
    # Same geography, ascending y — must still render north at the top.
    assert _north_is_at_top(_north_high_raster(descending=False))
