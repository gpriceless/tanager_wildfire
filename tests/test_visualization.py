"""Integration tests for PRODUCT_STYLES, plot_map, and save_figure.

Focuses on CRS-aware (georeferenced) DataArray rendering — the path that
other unit test files do not cover.  Each test uses a synthetic DataArray
with EPSG:32611 CRS metadata written via ``da.rio.write_crs``.

Complements (but does not duplicate) the granular tests in:
- tests/test_product_styles.py
- tests/test_plot_map.py
- tests/test_save_figure.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import rioxarray  # noqa: F401 — registers the .rio accessor on xr.DataArray

from tanager.visualization import PRODUCT_STYLES, ProductStyle, plot_map, save_figure


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_georef_da(nx: int = 50, ny: int = 50) -> xr.DataArray:
    """Return a synthetic DataArray with EPSG:32611 CRS (UTM zone 11N).

    Coordinates are UTM easting/northing in metres so that ``plot_map``
    engages the geo-axes rendering path (Easting/Northing labels, km ticks).
    """
    x = np.linspace(340_000, 350_000, nx)
    y = np.linspace(3_780_000, 3_790_000, ny)
    rng = np.random.default_rng(0)
    data = rng.random((ny, nx)).astype(np.float32)
    da = xr.DataArray(data, coords={"y": y, "x": x}, dims=["y", "x"])
    da = da.rio.write_crs("EPSG:32611")
    return da


@pytest.fixture()
def georef_da() -> xr.DataArray:
    """50×50 DataArray with EPSG:32611 CRS — the primary fixture for this module."""
    return make_georef_da()


@pytest.fixture()
def georef_da_with_nan() -> xr.DataArray:
    """50×50 DataArray with EPSG:32611 CRS and a rectangular NaN patch."""
    da = make_georef_da()
    arr = da.values.copy()
    arr[15:35, 15:35] = np.nan
    da = da.copy(data=arr)
    return da.rio.write_crs("EPSG:32611")


@pytest.fixture()
def simple_fig():
    """Minimal matplotlib Figure for save_figure tests."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    yield fig
    plt.close(fig)


# ---------------------------------------------------------------------------
# PRODUCT_STYLES — structural checks
# ---------------------------------------------------------------------------


EXPECTED_PRODUCT_KEYS = frozenset(
    {"nbr", "ndvi", "ndwi", "dnbr", "cbi", "severity", "char", "pv", "npv", "soil", "lfmc"}
)


class TestProductStylesAllKeys:
    """PRODUCT_STYLES must contain exactly the 11 product keys."""

    def test_all_11_keys_present(self):
        assert set(PRODUCT_STYLES.keys()) == EXPECTED_PRODUCT_KEYS

    def test_has_no_extra_keys(self):
        unexpected = set(PRODUCT_STYLES.keys()) - EXPECTED_PRODUCT_KEYS
        assert not unexpected, f"Unexpected keys: {unexpected}"


class TestProductStylesCorrectTypes:
    """Each value in PRODUCT_STYLES must be a ProductStyle with valid fields."""

    @pytest.mark.parametrize("key", sorted(EXPECTED_PRODUCT_KEYS))
    def test_value_is_product_style(self, key):
        assert isinstance(PRODUCT_STYLES[key], ProductStyle)

    @pytest.mark.parametrize("key", sorted(EXPECTED_PRODUCT_KEYS))
    def test_cmap_is_non_empty_string(self, key):
        assert isinstance(PRODUCT_STYLES[key].cmap, str)
        assert PRODUCT_STYLES[key].cmap

    @pytest.mark.parametrize("key", sorted(EXPECTED_PRODUCT_KEYS))
    def test_vmin_less_than_vmax(self, key):
        style = PRODUCT_STYLES[key]
        assert style.vmin < style.vmax

    @pytest.mark.parametrize("key", sorted(EXPECTED_PRODUCT_KEYS))
    def test_label_is_non_empty_string(self, key):
        assert isinstance(PRODUCT_STYLES[key].label, str)
        assert PRODUCT_STYLES[key].label


# ---------------------------------------------------------------------------
# plot_map — CRS-aware integration tests
# ---------------------------------------------------------------------------


class TestPlotMapReturnsFigure:
    """plot_map must return a matplotlib Figure for georeferenced input."""

    def test_plot_map_returns_figure(self, georef_da):
        from matplotlib.figure import Figure

        fig = plot_map(georef_da)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)


class TestPlotMapAxesAreUTMNotPixels:
    """With UTM x/y coordinates, xlim must be in metres (> 100 000), not pixels."""

    def test_xlim_is_utm_metres(self, georef_da):
        fig = plot_map(georef_da)
        try:
            ax = fig.get_axes()[0]
            xlo, xhi = ax.get_xlim()
            # UTM easting for zone 11N is always > 100 000 m.
            # Pixel-space x would be in [0, 50], so this assertion distinguishes the paths.
            assert xlo > 100_000, (
                f"xlim lower bound {xlo:.1f} looks like pixel-space, expected UTM metres"
            )
            assert xhi > 100_000, (
                f"xlim upper bound {xhi:.1f} looks like pixel-space, expected UTM metres"
            )
        finally:
            plt.close(fig)


class TestPlotMapColorbarPresent:
    """A colorbar axis must be appended to the figure."""

    def test_colorbar_present(self, georef_da):
        fig = plot_map(georef_da, product_name="nbr")
        try:
            # imshow axis + colorbar axis = at least 2 axes.
            assert len(fig.get_axes()) > 1, "Expected colorbar axis, found only one axis"
        finally:
            plt.close(fig)


class TestPlotMapWithProductName:
    """product_name must look up PRODUCT_STYLES and apply its settings."""

    def test_product_name_uses_style_lookup(self, georef_da):
        from matplotlib.figure import Figure

        fig = plot_map(georef_da, product_name="dnbr")
        try:
            assert isinstance(fig, Figure)
            # Colorbar label should reflect the dnbr style label.
            cb_ax = fig.get_axes()[1]
            assert PRODUCT_STYLES["dnbr"].label in cb_ax.get_ylabel()
        finally:
            plt.close(fig)

    @pytest.mark.parametrize("product", sorted(EXPECTED_PRODUCT_KEYS))
    def test_all_product_styles_render_without_error(self, georef_da, product):
        fig = plot_map(georef_da, product_name=product)
        try:
            assert len(fig.get_axes()) >= 1
        finally:
            plt.close(fig)


class TestPlotMapNanHandling:
    """DataArrays with NaN values must render without raising errors."""

    def test_nan_patch_renders_without_error(self, georef_da_with_nan):
        from matplotlib.figure import Figure

        fig = plot_map(georef_da_with_nan, product_name="nbr")
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_all_nan_renders_without_error(self, georef_da):
        all_nan = georef_da.copy(data=np.full(georef_da.shape, np.nan))
        all_nan = all_nan.rio.write_crs("EPSG:32611")
        from matplotlib.figure import Figure

        fig = plot_map(all_nan, product_name="lfmc")
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)


class TestPlotMapWithCrsMetadata:
    """DataArray with .rio.crs set must work correctly through the full render path."""

    def test_crs_da_renders_utm_axes(self, georef_da):
        """Axes labels and xlim confirm geo rendering, not pixel rendering."""
        fig = plot_map(georef_da, product_name="ndvi")
        try:
            ax = fig.get_axes()[0]
            # UTM axes set Easting/Northing labels
            assert ax.get_xlabel() == "Easting (km)"
            assert ax.get_ylabel() == "Northing (km)"
        finally:
            plt.close(fig)

    def test_crs_da_xlim_reflects_coordinate_extent(self, georef_da):
        """xlim should span the DataArray's x coordinate range (in metres)."""
        fig = plot_map(georef_da)
        try:
            ax = fig.get_axes()[0]
            xlo, xhi = ax.get_xlim()
            # x goes from 340 000 to 350 000; half-pixel expansion stays nearby
            assert xlo < 340_000, f"xlim lower {xlo} should be just below 340 000"
            assert xhi > 350_000, f"xlim upper {xhi} should be just above 350 000"
        finally:
            plt.close(fig)

    def test_different_crs_same_utm_axes(self):
        """A DA with a different UTM CRS should still produce geo axes."""
        x = np.linspace(500_000, 510_000, 30)
        y = np.linspace(4_200_000, 4_210_000, 30)
        data = np.random.default_rng(7).random((30, 30))
        da = xr.DataArray(data, coords={"y": y, "x": x}, dims=["y", "x"])
        da = da.rio.write_crs("EPSG:32610")  # UTM zone 10N
        fig = plot_map(da)
        try:
            ax = fig.get_axes()[0]
            xlo, xhi = ax.get_xlim()
            assert xlo > 100_000  # definitely UTM metres, not pixels
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# save_figure — integration tests with real filesystem writes
# ---------------------------------------------------------------------------


class TestSaveFigureCreatesFiles:
    """save_figure must write actual files in the requested formats."""

    def test_creates_png_file(self, simple_fig):
        with tempfile.TemporaryDirectory() as td:
            paths = save_figure(simple_fig, Path(td) / "output", ["png"])
            assert paths[0].exists()
            assert paths[0].stat().st_size > 0

    def test_creates_pdf_file(self, simple_fig):
        with tempfile.TemporaryDirectory() as td:
            paths = save_figure(simple_fig, Path(td) / "output", ["pdf"])
            assert paths[0].exists()

    def test_creates_multiple_formats(self, simple_fig):
        with tempfile.TemporaryDirectory() as td:
            paths = save_figure(simple_fig, Path(td) / "fig", ["png", "pdf", "svg"])
            assert len(paths) == 3
            assert all(p.exists() for p in paths)

    def test_returns_path_objects(self, simple_fig):
        with tempfile.TemporaryDirectory() as td:
            paths = save_figure(simple_fig, Path(td) / "fig", ["png"])
            assert all(isinstance(p, Path) for p in paths)


class TestSaveFigureCreatesDirectories:
    """save_figure must create parent directories that do not yet exist."""

    def test_creates_nested_parent_dirs(self, simple_fig):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "level1" / "level2" / "level3" / "fig"
            paths = save_figure(simple_fig, target, ["png"])
            assert paths[0].exists()

    def test_works_when_parent_already_exists(self, simple_fig):
        with tempfile.TemporaryDirectory() as td:
            paths = save_figure(simple_fig, Path(td) / "fig", ["png"])
            assert paths[0].exists()

    def test_from_plot_map_georef_figure(self, georef_da):
        """End-to-end: render georef DA → save to disk."""
        fig = plot_map(georef_da, product_name="nbr")
        try:
            with tempfile.TemporaryDirectory() as td:
                paths = save_figure(fig, Path(td) / "nbr_map", ["png"])
                assert paths[0].exists()
                assert paths[0].stat().st_size > 0
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Integration tests — basemap, perimeters, and scalebar working together
# ---------------------------------------------------------------------------


class TestAddBasemapWithMockedContextily:
    """add_basemap must call contextily and return ax unchanged when mocked."""

    def test_mock_is_called(self, georef_da):
        """Verify contextily.add_basemap is invoked when add_basemap is called."""
        from unittest.mock import patch
        from tanager.visualization import add_basemap

        fig, ax = plt.subplots()
        ax.set_xlim(340_000, 350_000)
        ax.set_ylim(3_780_000, 3_790_000)
        try:
            with patch("contextily.add_basemap") as mock_ctx:
                result = add_basemap(ax)
            assert mock_ctx.called
        finally:
            plt.close(fig)

    def test_ax_returned_unchanged(self, georef_da):
        """add_basemap must return the exact same Axes object."""
        from unittest.mock import patch
        from tanager.visualization import add_basemap

        fig, ax = plt.subplots()
        ax.set_xlim(340_000, 350_000)
        ax.set_ylim(3_780_000, 3_790_000)
        try:
            with patch("contextily.add_basemap"):
                result = add_basemap(ax)
            assert result is ax
        finally:
            plt.close(fig)

    def test_xlim_ylim_preserved_after_call(self, georef_da):
        """Axes limits must not be altered by add_basemap."""
        from unittest.mock import patch
        from tanager.visualization import add_basemap

        fig, ax = plt.subplots()
        ax.set_xlim(340_000, 350_000)
        ax.set_ylim(3_780_000, 3_790_000)
        xlim_before = ax.get_xlim()
        ylim_before = ax.get_ylim()
        try:
            with patch("contextily.add_basemap"):
                add_basemap(ax)
            assert ax.get_xlim() == xlim_before
            assert ax.get_ylim() == ylim_before
        finally:
            plt.close(fig)


class TestAddBasemapOfflineGracefulDegradation:
    """add_basemap must not raise and must return ax when contextily fails."""

    def test_oserror_does_not_propagate(self):
        """OSError from contextily is swallowed; no exception escapes."""
        from unittest.mock import patch
        from tanager.visualization import add_basemap

        fig, ax = plt.subplots()
        ax.set_xlim(340_000, 350_000)
        ax.set_ylim(3_780_000, 3_790_000)
        try:
            with patch("contextily.add_basemap", side_effect=OSError("Network unreachable")):
                result = add_basemap(ax)
            assert result is ax
        finally:
            plt.close(fig)

    def test_ax_returned_after_network_failure(self):
        """ax is returned even when the tile fetch raises."""
        from unittest.mock import patch
        from tanager.visualization import add_basemap

        fig, ax = plt.subplots()
        ax.set_xlim(340_000, 350_000)
        ax.set_ylim(3_780_000, 3_790_000)
        try:
            with patch("contextily.add_basemap", side_effect=OSError("DNS failure")):
                result = add_basemap(ax)
            # ax is unchanged; no AttributeError, no None return
            assert result is ax
        finally:
            plt.close(fig)


class TestLoadFirePerimetersWithSyntheticGeoJSON:
    """load_fire_perimeters reads a small synthetic GeoJSON fixture correctly."""

    def test_returns_geodataframe_with_geometry(self):
        """A minimal GeoJSON file produces a GeoDataFrame with a geometry column."""
        import json
        import geopandas as gpd
        from tanager.visualization import load_fire_perimeters

        feature = {
            "type": "Feature",
            "properties": {"name": "Test Fire"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-118.5, 34.0],
                        [-118.4, 34.0],
                        [-118.4, 34.1],
                        [-118.5, 34.1],
                        [-118.5, 34.0],
                    ]
                ],
            },
        }
        geojson = {"type": "FeatureCollection", "features": [feature]}

        with tempfile.NamedTemporaryFile(
            suffix=".geojson", mode="w", delete=False, dir=tempfile.gettempdir()
        ) as f:
            json.dump(geojson, f)
            tmp_path = Path(f.name)

        try:
            gdf = load_fire_perimeters(tmp_path)
            assert isinstance(gdf, gpd.GeoDataFrame)
            assert "geometry" in gdf.columns
            assert len(gdf) == 1
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_feature_property_accessible(self):
        """Properties from the GeoJSON are available as GeoDataFrame columns."""
        import json
        from tanager.visualization import load_fire_perimeters

        feature = {
            "type": "Feature",
            "properties": {"name": "Integration Fire", "area_ha": 500},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-118.3, 34.0],
                        [-118.2, 34.0],
                        [-118.2, 34.1],
                        [-118.3, 34.1],
                        [-118.3, 34.0],
                    ]
                ],
            },
        }
        geojson = {"type": "FeatureCollection", "features": [feature]}

        with tempfile.NamedTemporaryFile(
            suffix=".geojson", mode="w", delete=False, dir=tempfile.gettempdir()
        ) as f:
            json.dump(geojson, f)
            tmp_path = Path(f.name)

        try:
            gdf = load_fire_perimeters(tmp_path)
            assert gdf.iloc[0]["name"] == "Integration Fire"
        finally:
            tmp_path.unlink(missing_ok=True)


class TestOverlayPerimetersDrawsOnAxes:
    """overlay_perimeters must add collections or lines to the axes."""

    def _make_synthetic_gdf(self):
        """Return a one-polygon GeoDataFrame in EPSG:4326."""
        import geopandas as gpd
        from shapely.geometry import Polygon

        poly = Polygon(
            [(-118.5, 34.0), (-118.4, 34.0), (-118.4, 34.1), (-118.5, 34.1)]
        )
        return gpd.GeoDataFrame(
            {"name": ["Synthetic Fire"]}, geometry=[poly], crs="EPSG:4326"
        )

    def test_collection_added_to_axes(self):
        """At least one collection or line must appear on ax after the call."""
        from tanager.visualization import overlay_perimeters

        fig, ax = plt.subplots()
        ax.set_xlim(340_000, 350_000)
        ax.set_ylim(3_780_000, 3_790_000)
        gdf = self._make_synthetic_gdf()
        try:
            overlay_perimeters(ax, gdf, label=False)
            assert len(ax.collections) > 0 or len(ax.lines) > 0
        finally:
            plt.close(fig)

    def test_returns_same_axes_object(self):
        """overlay_perimeters must return the identical Axes passed in."""
        from tanager.visualization import overlay_perimeters

        fig, ax = plt.subplots()
        ax.set_xlim(340_000, 350_000)
        ax.set_ylim(3_780_000, 3_790_000)
        gdf = self._make_synthetic_gdf()
        try:
            result = overlay_perimeters(ax, gdf, label=False)
            assert result is ax
        finally:
            plt.close(fig)

    def test_label_text_appears_when_requested(self):
        """When label=True, a text annotation is added to ax."""
        from tanager.visualization import overlay_perimeters

        fig, ax = plt.subplots()
        ax.set_xlim(340_000, 350_000)
        ax.set_ylim(3_780_000, 3_790_000)
        gdf = self._make_synthetic_gdf()
        try:
            overlay_perimeters(ax, gdf, label=True)
            assert len(ax.texts) > 0
        finally:
            plt.close(fig)


class TestAddScalebarAddsPatch:
    """add_scalebar must add a Rectangle patch with the correct width."""

    def _make_ax(self):
        """Return a Figure and Axes with UTM-scale limits."""
        fig, ax = plt.subplots()
        ax.set_xlim(340_000, 350_000)
        ax.set_ylim(3_780_000, 3_790_000)
        return fig, ax

    def test_patch_is_added(self):
        """A Rectangle patch must appear in ax.patches."""
        from tanager.visualization import add_scalebar

        fig, ax = self._make_ax()
        try:
            add_scalebar(ax, 5)
            assert len(ax.patches) >= 1
        finally:
            plt.close(fig)

    def test_patch_width_matches_requested_km(self):
        """Bar width in data coordinates must equal length_km * 1000 metres."""
        from tanager.visualization import add_scalebar

        fig, ax = self._make_ax()
        try:
            add_scalebar(ax, 5)
            rect = ax.patches[0]
            assert rect.get_width() == pytest.approx(5000.0)
        finally:
            plt.close(fig)

    def test_text_label_is_present(self):
        """A '5 km' label must be placed above the bar."""
        from tanager.visualization import add_scalebar

        fig, ax = self._make_ax()
        try:
            add_scalebar(ax, 5)
            labels = [t.get_text() for t in ax.texts]
            assert "5 km" in labels
        finally:
            plt.close(fig)

    def test_different_lengths_produce_correct_widths(self):
        """1 km and 10 km requests must produce 1 000 and 10 000 m widths."""
        from tanager.visualization import add_scalebar

        for km, expected_m in [(1, 1000.0), (10, 10_000.0)]:
            fig, ax = self._make_ax()
            try:
                add_scalebar(ax, km)
                assert ax.patches[0].get_width() == pytest.approx(expected_m)
            finally:
                plt.close(fig)


class TestEndToEndPlotMapBasemapPerimetersScalebar:
    """Full pipeline: plot_map → overlay_perimeters → add_scalebar, mocked basemap."""

    def test_all_elements_present_on_figure(self, georef_da):
        """After running the full pipeline the figure must have all expected elements."""
        import json
        import geopandas as gpd
        from shapely.geometry import Polygon
        from unittest.mock import patch
        from tanager.visualization import plot_map, overlay_perimeters, add_scalebar

        # Step 1: render the base map with a mocked basemap tile call.
        with patch("contextily.add_basemap"):
            fig = plot_map(georef_da, basemap=True, product_name="nbr")

        try:
            ax = fig.get_axes()[0]

            # Step 2: overlay a synthetic perimeter.
            poly = Polygon(
                [(-118.5, 34.0), (-118.4, 34.0), (-118.4, 34.1), (-118.5, 34.1)]
            )
            gdf = gpd.GeoDataFrame(
                {"name": ["Integration Fire"]}, geometry=[poly], crs="EPSG:4326"
            )
            overlay_perimeters(ax, gdf, label=True)

            # Step 3: add a scale bar.
            add_scalebar(ax, 5)

            # Assertions — figure has all expected elements.
            from matplotlib.figure import Figure
            assert isinstance(fig, Figure)

            # Colorbar axis present (from plot_map).
            assert len(fig.get_axes()) > 1

            # Perimeter boundary: collections or lines.
            assert len(ax.collections) > 0 or len(ax.lines) > 0

            # Scalebar patch present.
            assert len(ax.patches) >= 1

            # Scalebar label text present.
            scalebar_labels = [t.get_text() for t in ax.texts if "km" in t.get_text()]
            assert scalebar_labels, "Expected at least one '… km' label from add_scalebar"
        finally:
            plt.close(fig)

    def test_network_failure_does_not_abort_pipeline(self, georef_da):
        """Even when the basemap tile fetch fails, the rest of the pipeline works."""
        import geopandas as gpd
        from shapely.geometry import Polygon
        from unittest.mock import patch
        from tanager.visualization import plot_map, overlay_perimeters, add_scalebar
        from matplotlib.figure import Figure

        with patch("contextily.add_basemap", side_effect=OSError("no network")):
            fig = plot_map(georef_da, basemap=True)

        try:
            assert isinstance(fig, Figure)
            ax = fig.get_axes()[0]

            poly = Polygon(
                [(-118.5, 34.0), (-118.4, 34.0), (-118.4, 34.1), (-118.5, 34.1)]
            )
            gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
            overlay_perimeters(ax, gdf, label=False)
            add_scalebar(ax, 3)

            assert len(ax.patches) >= 1
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# plot_before_after — side-by-side pre/post comparison
# ---------------------------------------------------------------------------


def _make_pre_da() -> xr.DataArray:
    """Pre-fire DataArray (smaller spatial extent, 70x70)."""
    x = np.linspace(340_000, 347_000, 70)
    y = np.linspace(3_780_000, 3_787_000, 70)
    rng = np.random.default_rng(1)
    return xr.DataArray(rng.random((70, 70)) * 0.8, coords={"y": y, "x": x}, dims=["y", "x"])


def _make_post_da() -> xr.DataArray:
    """Post-fire DataArray (larger spatial extent, 100x100)."""
    x = np.linspace(339_000, 350_000, 100)
    y = np.linspace(3_779_000, 3_790_000, 100)
    rng = np.random.default_rng(2)
    return xr.DataArray(rng.random((100, 100)) * 0.3, coords={"y": y, "x": x}, dims=["y", "x"])


class TestPlotBeforeAfterReturnsFigure:
    """plot_before_after must return a matplotlib Figure."""

    def test_returns_figure(self):
        from matplotlib.figure import Figure
        from tanager.visualization import plot_before_after

        pre, post = _make_pre_da(), _make_post_da()
        fig = plot_before_after(pre, post, "nbr")
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)


class TestPlotBeforeAfterTwoPanels:
    """Figure must contain at least 2 map axes (plus colorbar)."""

    def test_has_at_least_three_axes(self):
        """2 map panels + 1 shared colorbar = at least 3 axes."""
        from tanager.visualization import plot_before_after

        pre, post = _make_pre_da(), _make_post_da()
        fig = plot_before_after(pre, post, "nbr")
        try:
            assert len(fig.axes) >= 3, (
                f"Expected >= 3 axes (2 map + colorbar), got {len(fig.axes)}"
            )
        finally:
            plt.close(fig)

    def test_panel_titles_match_labels(self):
        """Panel titles must use the supplied pre_label and post_label strings."""
        from tanager.visualization import plot_before_after

        pre, post = _make_pre_da(), _make_post_da()
        fig = plot_before_after(
            pre, post, "nbr",
            pre_label="Dec 15 Pre-Fire",
            post_label="Jan 23 Post-Fire",
        )
        try:
            titles = [ax.get_title() for ax in fig.axes[:2]]
            assert titles[0] == "Dec 15 Pre-Fire"
            assert titles[1] == "Jan 23 Post-Fire"
        finally:
            plt.close(fig)

    def test_default_panel_titles_when_no_labels(self):
        """Default titles must be 'Pre-Fire' and 'Post-Fire'."""
        from tanager.visualization import plot_before_after

        pre, post = _make_pre_da(), _make_post_da()
        fig = plot_before_after(pre, post, "nbr")
        try:
            titles = [ax.get_title() for ax in fig.axes[:2]]
            assert titles[0] == "Pre-Fire"
            assert titles[1] == "Post-Fire"
        finally:
            plt.close(fig)


class TestPlotBeforeAfterUTMAxes:
    """Both panels must render UTM metre-scale axes (not pixel indices)."""

    def test_pre_panel_xlim_is_utm(self):
        from tanager.visualization import plot_before_after

        pre, post = _make_pre_da(), _make_post_da()
        fig = plot_before_after(pre, post, "nbr")
        try:
            ax_pre = fig.axes[0]
            xlo, xhi = ax_pre.get_xlim()
            assert xlo > 100_000, f"Pre panel xlim lower {xlo} looks like pixels"
            assert xhi > 100_000, f"Pre panel xlim upper {xhi} looks like pixels"
        finally:
            plt.close(fig)

    def test_post_panel_xlim_is_utm(self):
        from tanager.visualization import plot_before_after

        pre, post = _make_pre_da(), _make_post_da()
        fig = plot_before_after(pre, post, "nbr")
        try:
            ax_post = fig.axes[1]
            xlo, xhi = ax_post.get_xlim()
            assert xlo > 100_000, f"Post panel xlim lower {xlo} looks like pixels"
            assert xhi > 100_000, f"Post panel xlim upper {xhi} looks like pixels"
        finally:
            plt.close(fig)

    def test_panels_have_different_extents_for_different_size_inputs(self):
        """Pre and post scenes with different extents should produce different xlims."""
        from tanager.visualization import plot_before_after

        pre, post = _make_pre_da(), _make_post_da()
        fig = plot_before_after(pre, post, "nbr")
        try:
            xlo_pre, xhi_pre = fig.axes[0].get_xlim()
            xlo_post, xhi_post = fig.axes[1].get_xlim()
            # Pre x runs 340–347k; post runs 339–350k — they differ
            assert xhi_post > xhi_pre, (
                f"Post xlim upper {xhi_post} should exceed pre {xhi_pre} for wider scene"
            )
        finally:
            plt.close(fig)


class TestPlotBeforeAfterNaNHandling:
    """NaN values in pre or post must render without error."""

    def test_pre_with_nan_renders(self):
        from matplotlib.figure import Figure
        from tanager.visualization import plot_before_after

        pre = _make_pre_da()
        arr = pre.values.copy()
        arr[10:30, 10:30] = np.nan
        pre_nan = pre.copy(data=arr)
        post = _make_post_da()

        fig = plot_before_after(pre_nan, post, "nbr")
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_post_all_nan_renders(self):
        from matplotlib.figure import Figure
        from tanager.visualization import plot_before_after

        pre = _make_pre_da()
        post = _make_post_da()
        post_nan = post.copy(data=np.full(post.shape, np.nan))

        fig = plot_before_after(pre, post_nan, "nbr")
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)


class TestPlotBeforeAfterAllProducts:
    """plot_before_after must render without error for every product in PRODUCT_STYLES."""

    @pytest.mark.parametrize("product", sorted({
        "nbr", "ndvi", "ndwi", "dnbr", "cbi", "severity", "char", "pv", "npv", "soil", "lfmc"
    }))
    def test_all_products_render(self, product):
        from tanager.visualization import plot_before_after

        pre, post = _make_pre_da(), _make_post_da()
        fig = plot_before_after(pre, post, product)
        try:
            assert len(fig.axes) >= 3
        finally:
            plt.close(fig)


class TestPlotBeforeAfterPublicationMode:
    """publication=True must set DPI to 300."""

    def test_publication_dpi(self):
        from tanager.visualization import plot_before_after

        pre, post = _make_pre_da(), _make_post_da()
        fig = plot_before_after(pre, post, "nbr", publication=True)
        try:
            assert fig.get_dpi() == 300
        finally:
            plt.close(fig)


class TestPlotBeforeAfterFirePerimeters:
    """When fire_perimeters is provided, both panels must get perimeter overlays."""

    def _make_perimeters(self):
        import geopandas as gpd
        from shapely.geometry import Polygon

        poly = Polygon(
            [(-118.5, 34.0), (-118.4, 34.0), (-118.4, 34.1), (-118.5, 34.1)]
        )
        return gpd.GeoDataFrame(
            {"name": ["Test Fire"]}, geometry=[poly], crs="EPSG:4326"
        )

    def test_perimeters_added_to_both_panels(self):
        """Each panel must gain at least one collection or line from the perimeter overlay."""
        from tanager.visualization import plot_before_after

        pre, post = _make_pre_da(), _make_post_da()
        perimeters = self._make_perimeters()

        fig = plot_before_after(pre, post, "nbr", fire_perimeters=perimeters)
        try:
            ax_pre = fig.axes[0]
            ax_post = fig.axes[1]
            pre_has_overlay = len(ax_pre.collections) > 0 or len(ax_pre.lines) > 0
            post_has_overlay = len(ax_post.collections) > 0 or len(ax_post.lines) > 0
            assert pre_has_overlay, "Pre-fire panel missing perimeter overlay"
            assert post_has_overlay, "Post-fire panel missing perimeter overlay"
        finally:
            plt.close(fig)


class TestPlotBeforeAfterBasemap:
    """When basemap=True, add_basemap must be called on both panels."""

    def test_basemap_called_on_both_panels(self):
        from unittest.mock import patch
        from tanager.visualization import plot_before_after

        pre, post = _make_pre_da(), _make_post_da()
        with patch("contextily.add_basemap") as mock_ctx:
            fig = plot_before_after(pre, post, "nbr", basemap=True)
        try:
            # add_basemap is called once per panel, so 2 calls total
            assert mock_ctx.call_count == 2, (
                f"Expected 2 contextily.add_basemap calls, got {mock_ctx.call_count}"
            )
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# plot_severity_summary — 2x3 multi-panel grid
# ---------------------------------------------------------------------------


def _make_fractions_ds(nx: int = 50, ny: int = 50) -> xr.Dataset:
    """Return a synthetic fractions Dataset with x/y UTM coordinates."""
    x = np.linspace(340_000, 350_000, nx)
    y = np.linspace(3_780_000, 3_790_000, ny)
    coords = {"y": y, "x": x}
    rng = np.random.default_rng(42)
    return xr.Dataset(
        {
            "char": xr.DataArray(rng.random((ny, nx)), coords=coords, dims=["y", "x"]),
            "pv": xr.DataArray(rng.random((ny, nx)), coords=coords, dims=["y", "x"]),
            "npv": xr.DataArray(rng.random((ny, nx)), coords=coords, dims=["y", "x"]),
            "soil": xr.DataArray(rng.random((ny, nx)), coords=coords, dims=["y", "x"]),
        }
    )


def _make_cbi_da(nx: int = 50, ny: int = 50) -> xr.DataArray:
    """Return a synthetic CBI DataArray (values 0–3)."""
    x = np.linspace(340_000, 350_000, nx)
    y = np.linspace(3_780_000, 3_790_000, ny)
    coords = {"y": y, "x": x}
    rng = np.random.default_rng(7)
    return xr.DataArray(rng.random((ny, nx)) * 3.0, coords=coords, dims=["y", "x"])


def _make_severity_da(nx: int = 50, ny: int = 50) -> xr.DataArray:
    """Return a synthetic severity-class DataArray (integer classes 0–4)."""
    x = np.linspace(340_000, 350_000, nx)
    y = np.linspace(3_780_000, 3_790_000, ny)
    coords = {"y": y, "x": x}
    rng = np.random.default_rng(13)
    return xr.DataArray(
        rng.integers(0, 5, (ny, nx)).astype(float), coords=coords, dims=["y", "x"]
    )


class TestPlotSeveritySummaryReturnsFigure:
    """plot_severity_summary must return a matplotlib Figure."""

    def test_returns_figure(self):
        from matplotlib.figure import Figure
        from tanager.visualization import plot_severity_summary

        fractions = _make_fractions_ds()
        cbi = _make_cbi_da()
        severity = _make_severity_da()

        fig = plot_severity_summary(fractions, cbi, severity)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)


class TestPlotSeveritySummarySixPanels:
    """Figure must contain 6 panel axes (plus 6 individual colorbars = 12 total)."""

    def test_at_least_six_axes(self):
        """Minimum 6 axes: one per panel (colorbars bring the total higher)."""
        from tanager.visualization import plot_severity_summary

        fractions = _make_fractions_ds()
        cbi = _make_cbi_da()
        severity = _make_severity_da()

        fig = plot_severity_summary(fractions, cbi, severity)
        try:
            assert len(fig.axes) >= 6, (
                f"Expected >=6 axes (6 panels + colorbars), got {len(fig.axes)}"
            )
        finally:
            plt.close(fig)

    def test_twelve_axes_with_colorbars(self):
        """Each panel gets its own colorbar, so 6 panels + 6 colorbars = 12."""
        from tanager.visualization import plot_severity_summary

        fractions = _make_fractions_ds()
        cbi = _make_cbi_da()
        severity = _make_severity_da()

        fig = plot_severity_summary(fractions, cbi, severity)
        try:
            assert len(fig.axes) == 12, (
                f"Expected 12 axes (6 panels + 6 colorbars), got {len(fig.axes)}"
            )
        finally:
            plt.close(fig)


class TestPlotSeveritySummaryUTMAxes:
    """All 6 panel axes must render in UTM (metre) coordinates, not pixel indices."""

    def test_all_panels_use_utm_xlim(self):
        """xlim for every panel must be in UTM metres (> 100 000)."""
        from tanager.visualization import plot_severity_summary

        fractions = _make_fractions_ds()
        cbi = _make_cbi_da()
        severity = _make_severity_da()

        fig = plot_severity_summary(fractions, cbi, severity)
        try:
            # The first 6 axes are the map panels; the remaining 6 are colorbars.
            # Colorbar axes have very small xlim, so we test only the 6 map panels.
            # We identify map panels by their xlim magnitude (UTM > 100 000).
            map_axes = [ax for ax in fig.axes if ax.get_xlim()[1] > 100_000]
            assert len(map_axes) == 6, (
                f"Expected 6 UTM-scale map panels, found {len(map_axes)}"
            )
        finally:
            plt.close(fig)

    def test_all_panels_have_easting_label(self):
        """All 6 map panels must carry 'Easting (km)' on the x-axis."""
        from tanager.visualization import plot_severity_summary

        fractions = _make_fractions_ds()
        cbi = _make_cbi_da()
        severity = _make_severity_da()

        fig = plot_severity_summary(fractions, cbi, severity)
        try:
            map_axes = [ax for ax in fig.axes if ax.get_xlim()[1] > 100_000]
            for ax in map_axes:
                assert ax.get_xlabel() == "Easting (km)", (
                    f"Panel xlabel {ax.get_xlabel()!r} should be 'Easting (km)'"
                )
        finally:
            plt.close(fig)


class TestPlotSeveritySummaryNaNHandling:
    """NaN values in any panel must render without error."""

    def test_nan_in_char_renders(self):
        """A fractions dataset with NaN in char must produce a valid figure."""
        from matplotlib.figure import Figure
        from tanager.visualization import plot_severity_summary

        fractions = _make_fractions_ds()
        char_arr = fractions["char"].values.copy()
        char_arr[10:30, 10:30] = np.nan
        fractions["char"] = fractions["char"].copy(data=char_arr)
        cbi = _make_cbi_da()
        severity = _make_severity_da()

        fig = plot_severity_summary(fractions, cbi, severity)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_all_nan_cbi_renders(self):
        """All-NaN CBI must produce a valid figure (masked, no crash)."""
        from matplotlib.figure import Figure
        from tanager.visualization import plot_severity_summary

        fractions = _make_fractions_ds()
        cbi = _make_cbi_da()
        cbi_nan = cbi.copy(data=np.full(cbi.shape, np.nan))
        severity = _make_severity_da()

        fig = plot_severity_summary(fractions, cbi_nan, severity)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)


class TestPlotSeveritySummaryPublicationMode:
    """publication=True must set figure DPI to 300."""

    def test_publication_dpi(self):
        from tanager.visualization import plot_severity_summary

        fractions = _make_fractions_ds()
        cbi = _make_cbi_da()
        severity = _make_severity_da()

        fig = plot_severity_summary(fractions, cbi, severity, publication=True)
        try:
            assert fig.get_dpi() == 300
        finally:
            plt.close(fig)

    def test_default_not_publication_dpi(self):
        """Without publication=True, DPI must not be forced to 300."""
        from tanager.visualization import plot_severity_summary

        fractions = _make_fractions_ds()
        cbi = _make_cbi_da()
        severity = _make_severity_da()

        fig = plot_severity_summary(fractions, cbi, severity, publication=False)
        try:
            # matplotlib default is 100 dpi — just check it's not 300
            assert fig.get_dpi() != 300
        finally:
            plt.close(fig)


class TestPlotSeveritySummaryFigsize:
    """Custom figsize must be honoured by the returned figure."""

    def test_custom_figsize(self):
        from tanager.visualization import plot_severity_summary

        fractions = _make_fractions_ds()
        cbi = _make_cbi_da()
        severity = _make_severity_da()
        custom_size = (12.0, 8.0)

        fig = plot_severity_summary(fractions, cbi, severity, figsize=custom_size)
        try:
            w, h = fig.get_size_inches()
            assert (w, h) == pytest.approx(custom_size)
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Integration tests — comparison panel functions with CRS-aware DataArrays
# ---------------------------------------------------------------------------


def _make_crs_da(nx: int = 50, ny: int = 50, seed: int = 0, scale: float = 1.0) -> xr.DataArray:
    """Return a synthetic DataArray with EPSG:32611 CRS written via rio.write_crs."""
    x = np.linspace(340_000, 350_000, nx)
    y = np.linspace(3_780_000, 3_790_000, ny)
    rng = np.random.default_rng(seed)
    data = rng.random((ny, nx)).astype(np.float32) * scale
    da = xr.DataArray(data, coords={"y": y, "x": x}, dims=["y", "x"])
    return da.rio.write_crs("EPSG:32611")


class TestPlotBeforeAfterWithCrsMetadata:
    """plot_before_after with CRS-tagged DataArrays must render 2 UTM panels and a shared colorbar."""

    def test_two_panels_and_shared_colorbar(self):
        """Figure has >= 3 axes: 2 map panels + shared colorbar."""
        from tanager.visualization import plot_before_after

        pre = _make_crs_da(seed=1)
        post = _make_crs_da(seed=2)
        fig = plot_before_after(pre, post, "nbr")
        try:
            assert len(fig.axes) >= 3, (
                f"Expected >= 3 axes (2 panels + colorbar), got {len(fig.axes)}"
            )
        finally:
            plt.close(fig)

    def test_both_panels_have_utm_xlim(self):
        """Both panels must show UTM metres (> 100 000) on the x-axis."""
        from tanager.visualization import plot_before_after

        pre = _make_crs_da(seed=3)
        post = _make_crs_da(seed=4)
        fig = plot_before_after(pre, post, "nbr")
        try:
            for idx, label in [(0, "pre"), (1, "post")]:
                xlo, xhi = fig.axes[idx].get_xlim()
                assert xlo > 100_000, (
                    f"{label} panel xlim lower {xlo:.0f} looks like pixel space"
                )
                assert xhi > 100_000, (
                    f"{label} panel xlim upper {xhi:.0f} looks like pixel space"
                )
        finally:
            plt.close(fig)

    def test_shared_colorbar_has_product_label(self):
        """The colorbar must carry the label from PRODUCT_STYLES for the requested product."""
        from tanager.visualization import plot_before_after

        pre = _make_crs_da(seed=5)
        post = _make_crs_da(seed=6)
        fig = plot_before_after(pre, post, "nbr")
        try:
            # The shared colorbar is rendered horizontally (orientation='horizontal'),
            # so its label is set on the x-axis of the colorbar Axes (ax[2] onward).
            # Check both x- and y-axis labels to be orientation-agnostic.
            cb_axes = fig.axes[2:]
            cb_labels = [ax.get_xlabel() + ax.get_ylabel() for ax in cb_axes]
            any_match = any(PRODUCT_STYLES["nbr"].label in lbl for lbl in cb_labels)
            assert any_match, (
                f"Expected '{PRODUCT_STYLES['nbr'].label}' in colorbar label, "
                f"got axes labels: {cb_labels}"
            )
        finally:
            plt.close(fig)


class TestPlotBeforeAfterWithPerimeters:
    """plot_before_after with fire_perimeters must draw overlays on both panels."""

    def _make_perimeter_gdf(self):
        """GeoDataFrame with a synthetic polygon in EPSG:4326."""
        import geopandas as gpd
        from shapely.geometry import Polygon

        poly = Polygon(
            [(-118.5, 34.0), (-118.4, 34.0), (-118.4, 34.1), (-118.5, 34.1)]
        )
        return gpd.GeoDataFrame({"name": ["CRS Test Fire"]}, geometry=[poly], crs="EPSG:4326")

    def test_perimeters_drawn_on_both_panels(self):
        """Both the pre and post axes must gain collections or lines from the overlay."""
        from tanager.visualization import plot_before_after

        pre = _make_crs_da(seed=10)
        post = _make_crs_da(seed=11)
        gdf = self._make_perimeter_gdf()
        fig = plot_before_after(pre, post, "nbr", fire_perimeters=gdf)
        try:
            ax_pre = fig.axes[0]
            ax_post = fig.axes[1]
            pre_has_overlay = len(ax_pre.collections) > 0 or len(ax_pre.lines) > 0
            post_has_overlay = len(ax_post.collections) > 0 or len(ax_post.lines) > 0
            assert pre_has_overlay, "Pre-fire panel has no perimeter overlay (collections/lines)"
            assert post_has_overlay, "Post-fire panel has no perimeter overlay (collections/lines)"
        finally:
            plt.close(fig)

    def test_figure_still_has_three_or_more_axes(self):
        """Adding perimeters must not remove the shared colorbar axis."""
        from tanager.visualization import plot_before_after

        pre = _make_crs_da(seed=12)
        post = _make_crs_da(seed=13)
        gdf = self._make_perimeter_gdf()
        fig = plot_before_after(pre, post, "nbr", fire_perimeters=gdf)
        try:
            assert len(fig.axes) >= 3
        finally:
            plt.close(fig)


class TestPlotDifferenceMapWithCrs:
    """plot_difference_map with a CRS-tagged DataArray must render contours."""

    def test_contours_present_for_crs_dnbr(self):
        """dNBR DataArray with rio CRS must still produce contour lines."""
        from tanager.visualization import plot_difference_map

        # Create a dNBR-like DataArray spanning the full severity range so all
        # four USGS thresholds fall within the data range.
        x = np.linspace(340_000, 350_000, 100)
        y = np.linspace(3_780_000, 3_790_000, 100)
        data = np.linspace(-0.2, 1.0, 100 * 100).reshape(100, 100).astype(np.float32)
        dnbr = xr.DataArray(data, coords={"y": y, "x": x}, dims=["y", "x"])
        dnbr = dnbr.rio.write_crs("EPSG:32611")

        fig = plot_difference_map(dnbr, "dnbr")
        ax = fig.axes[0]
        try:
            has_contours = any(
                hasattr(c, "get_paths") and len(c.get_paths()) > 0
                for c in ax.collections
            )
            assert has_contours, "Expected contour lines for CRS-tagged dNBR DataArray"
        finally:
            plt.close(fig)

    def test_utm_axes_present_for_crs_dnbr(self):
        """Map panel must use UTM-scale axes when CRS DataArray has UTM coordinates."""
        from tanager.visualization import plot_difference_map

        dnbr = _make_crs_da(seed=20, scale=1.3)  # values up to 1.3, spans dnbr range
        fig = plot_difference_map(dnbr, "dnbr")
        try:
            ax = fig.axes[0]
            xlo, xhi = ax.get_xlim()
            assert xlo > 100_000, f"xlim lower {xlo:.0f} looks like pixel-space, not UTM"
        finally:
            plt.close(fig)


class TestPlotSeveritySummaryWithCrs:
    """plot_severity_summary with CRS-tagged DataArrays must use UTM axes on all 6 panels."""

    def _make_crs_fractions_ds(self, nx: int = 50, ny: int = 50) -> xr.Dataset:
        """Fractions Dataset where each variable carries EPSG:32611 CRS."""
        x = np.linspace(340_000, 350_000, nx)
        y = np.linspace(3_780_000, 3_790_000, ny)
        coords = {"y": y, "x": x}
        rng = np.random.default_rng(99)

        def _var(seed_offset: int) -> xr.DataArray:
            da = xr.DataArray(
                rng.random((ny, nx)).astype(np.float32),
                coords=coords,
                dims=["y", "x"],
            )
            return da.rio.write_crs("EPSG:32611")

        return xr.Dataset(
            {
                "char": _var(0),
                "pv": _var(1),
                "npv": _var(2),
                "soil": _var(3),
            }
        )

    def test_six_utm_panels(self):
        """All 6 map panels must have UTM-scale x-axis limits (> 100 000 m)."""
        from tanager.visualization import plot_severity_summary

        fractions = self._make_crs_fractions_ds()
        cbi = _make_crs_da(seed=50, scale=3.0)
        severity = xr.DataArray(
            np.random.default_rng(55).integers(0, 5, (50, 50)).astype(float),
            coords={"y": np.linspace(3_780_000, 3_790_000, 50), "x": np.linspace(340_000, 350_000, 50)},
            dims=["y", "x"],
        ).rio.write_crs("EPSG:32611")

        fig = plot_severity_summary(fractions, cbi, severity)
        try:
            map_axes = [ax for ax in fig.axes if ax.get_xlim()[1] > 100_000]
            assert len(map_axes) == 6, (
                f"Expected 6 UTM-scale panels, found {len(map_axes)}"
            )
        finally:
            plt.close(fig)

    def test_all_six_panels_have_easting_label(self):
        """Every map panel must carry 'Easting (km)' on its x-axis."""
        from tanager.visualization import plot_severity_summary

        fractions = self._make_crs_fractions_ds()
        cbi = _make_crs_da(seed=60, scale=3.0)
        severity = xr.DataArray(
            np.random.default_rng(65).integers(0, 5, (50, 50)).astype(float),
            coords={"y": np.linspace(3_780_000, 3_790_000, 50), "x": np.linspace(340_000, 350_000, 50)},
            dims=["y", "x"],
        ).rio.write_crs("EPSG:32611")

        fig = plot_severity_summary(fractions, cbi, severity)
        try:
            map_axes = [ax for ax in fig.axes if ax.get_xlim()[1] > 100_000]
            for ax in map_axes:
                assert ax.get_xlabel() == "Easting (km)", (
                    f"Expected 'Easting (km)', got {ax.get_xlabel()!r}"
                )
        finally:
            plt.close(fig)


class TestComparisonFunctionsEndToEnd:
    """Simulate a mini fire-analysis pipeline calling all three comparison functions."""

    def test_all_three_functions_return_figures(self):
        """plot_before_after, plot_difference_map, and plot_severity_summary all return Figures."""
        from matplotlib.figure import Figure
        from tanager.visualization import (
            plot_before_after,
            plot_difference_map,
            plot_severity_summary,
        )

        # --- Arrange: synthetic pre- and post-fire NBR scenes -------------------
        x = np.linspace(340_000, 350_000, 60)
        y = np.linspace(3_780_000, 3_790_000, 60)
        coords = {"y": y, "x": x}
        rng = np.random.default_rng(2024)

        pre_nbr = xr.DataArray(
            rng.uniform(0.2, 0.8, (60, 60)).astype(np.float32),
            coords=coords,
            dims=["y", "x"],
        ).rio.write_crs("EPSG:32611")

        post_nbr = xr.DataArray(
            rng.uniform(-0.1, 0.5, (60, 60)).astype(np.float32),
            coords=coords,
            dims=["y", "x"],
        ).rio.write_crs("EPSG:32611")

        # Compute a dNBR-like difference (pre minus post × scale factor).
        dnbr_data = (pre_nbr.values - post_nbr.values) * 1.0
        dnbr = xr.DataArray(dnbr_data, coords=coords, dims=["y", "x"]).rio.write_crs("EPSG:32611")

        # Mock fraction Dataset from synthetic data.
        def _frac(s: int) -> xr.DataArray:
            arr = rng.random((60, 60)).astype(np.float32)
            return xr.DataArray(arr, coords=coords, dims=["y", "x"]).rio.write_crs("EPSG:32611")

        fractions = xr.Dataset({"char": _frac(0), "pv": _frac(1), "npv": _frac(2), "soil": _frac(3)})
        cbi = xr.DataArray(
            rng.uniform(0, 3, (60, 60)).astype(np.float32),
            coords=coords,
            dims=["y", "x"],
        ).rio.write_crs("EPSG:32611")
        severity = xr.DataArray(
            rng.integers(0, 5, (60, 60)).astype(float),
            coords=coords,
            dims=["y", "x"],
        ).rio.write_crs("EPSG:32611")

        # --- Act -----------------------------------------------------------------
        fig_ba = plot_before_after(pre_nbr, post_nbr, "nbr")
        fig_dm = plot_difference_map(dnbr, "dnbr")
        fig_ss = plot_severity_summary(fractions, cbi, severity)

        # --- Assert --------------------------------------------------------------
        try:
            assert isinstance(fig_ba, Figure), "plot_before_after did not return a Figure"
            assert isinstance(fig_dm, Figure), "plot_difference_map did not return a Figure"
            assert isinstance(fig_ss, Figure), "plot_severity_summary did not return a Figure"

            # Each function must have produced axes (non-trivial output).
            assert len(fig_ba.axes) >= 3, "plot_before_after: expected >= 3 axes"
            assert len(fig_dm.axes) >= 2, "plot_difference_map: expected >= 2 axes (map + colorbar)"
            assert len(fig_ss.axes) >= 6, "plot_severity_summary: expected >= 6 axes"
        finally:
            plt.close(fig_ba)
            plt.close(fig_dm)
            plt.close(fig_ss)

    def test_pipeline_figures_have_utm_scale_axes(self):
        """All map panels produced by the pipeline must render in UTM metres."""
        from tanager.visualization import (
            plot_before_after,
            plot_difference_map,
            plot_severity_summary,
        )

        x = np.linspace(340_000, 350_000, 40)
        y = np.linspace(3_780_000, 3_790_000, 40)
        coords = {"y": y, "x": x}
        rng = np.random.default_rng(777)

        def _da(scale: float = 1.0) -> xr.DataArray:
            return xr.DataArray(
                rng.random((40, 40)).astype(np.float32) * scale,
                coords=coords,
                dims=["y", "x"],
            ).rio.write_crs("EPSG:32611")

        pre = _da()
        post = _da()
        dnbr = _da(scale=1.3)
        fractions = xr.Dataset({"char": _da(), "pv": _da(), "npv": _da(), "soil": _da()})
        cbi = _da(scale=3.0)
        severity = xr.DataArray(
            rng.integers(0, 5, (40, 40)).astype(float),
            coords=coords,
            dims=["y", "x"],
        ).rio.write_crs("EPSG:32611")

        fig_ba = plot_before_after(pre, post, "nbr")
        fig_dm = plot_difference_map(dnbr, "dnbr")
        fig_ss = plot_severity_summary(fractions, cbi, severity)

        figs = [fig_ba, fig_dm, fig_ss]
        try:
            for fig in figs:
                utm_axes = [ax for ax in fig.axes if ax.get_xlim()[1] > 100_000]
                assert utm_axes, (
                    f"Figure {fig!r} produced no UTM-scale axes; all xlims were small"
                )
        finally:
            for fig in figs:
                plt.close(fig)


# ---------------------------------------------------------------------------
# Integration tests — plot_temporal_trajectory
# ---------------------------------------------------------------------------

# Realistic fire-trajectory dataset: NBR drops sharply at ignition then recovers.
_TRAJ_DATES = [
    "2024-12-15",
    "2024-12-25",
    "2025-01-01",
    "2025-01-07",
    "2025-01-15",
    "2025-01-23",
    "2025-02-01",
]
_TRAJ_VALUES = [0.65, 0.62, 0.60, 0.15, 0.20, 0.25, 0.30]  # NBR drops at fire
_TRAJ_FIRE_DATE = "2025-01-07"


class TestTemporalTrajectoryIntegration:
    """Integration-level tests for plot_temporal_trajectory.

    Each test exercises a scenario that combines multiple behaviours of the
    function and verifies the resulting matplotlib state at the axes level.
    These complement (but do not duplicate) the unit tests in
    tests/test_plot_temporal_trajectory.py.
    """

    # ------------------------------------------------------------------
    # 1. Line with markers present
    # ------------------------------------------------------------------

    def test_line_plotted_with_markers(self):
        """Data line must be present and must carry a visible marker.

        Verifies both that a line was drawn and that it uses an explicit
        marker symbol (not None / 'None' / empty string).
        """
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(
            _TRAJ_DATES, _TRAJ_VALUES, "NBR", fire_date=None
        )
        ax = fig.axes[0]
        try:
            # At least one line must be on the axes.
            assert len(ax.lines) >= 1, "Expected at least one Line2D on the axes"
            # The first line is the data series — it must have a marker.
            data_line = ax.lines[0]
            marker = data_line.get_marker()
            assert marker not in (None, "None", ""), (
                f"Data line has no marker; got marker={marker!r}"
            )
        finally:
            plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Fire-event axvline present
    # ------------------------------------------------------------------

    def test_fire_event_marker_present(self):
        """An axvline for fire_date must be added when fire_date is supplied.

        The vertical line is the second line on the axes (after the data
        series).  We confirm >= 2 lines are present, meaning the vline was
        created.
        """
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(
            _TRAJ_DATES, _TRAJ_VALUES, "NBR", fire_date=_TRAJ_FIRE_DATE
        )
        ax = fig.axes[0]
        try:
            assert len(ax.lines) >= 2, (
                f"Expected data line + fire vline (>= 2 lines), got {len(ax.lines)}"
            )
        finally:
            plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Error bands rendered
    # ------------------------------------------------------------------

    def test_error_bands_rendered(self):
        """A fill_between PolyCollection must appear in ax.collections when
        error_bands is provided.

        Uses realistic per-observation uncertainty values (±0.03–0.06 NBR).
        """
        from tanager.visualization import plot_temporal_trajectory

        error_bands = [0.04, 0.04, 0.05, 0.06, 0.05, 0.04, 0.03]
        fig = plot_temporal_trajectory(
            _TRAJ_DATES, _TRAJ_VALUES, "NBR",
            fire_date=_TRAJ_FIRE_DATE,
            error_bands=error_bands,
        )
        ax = fig.axes[0]
        try:
            assert len(ax.collections) >= 1, (
                "Expected at least one fill_between collection when error_bands given"
            )
        finally:
            plt.close(fig)

    # ------------------------------------------------------------------
    # 4. No fire_date → no vertical line
    # ------------------------------------------------------------------

    def test_no_fire_date_no_vline(self):
        """When fire_date=None, the axes must contain exactly one line (the
        data series) — no extra vertical line.
        """
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(
            _TRAJ_DATES, _TRAJ_VALUES, "NBR", fire_date=None
        )
        ax = fig.axes[0]
        try:
            assert len(ax.lines) == 1, (
                f"Expected exactly 1 line (data series) with fire_date=None, "
                f"got {len(ax.lines)}"
            )
        finally:
            plt.close(fig)

    # ------------------------------------------------------------------
    # 5. datetime objects accepted
    # ------------------------------------------------------------------

    def test_datetime_objects_work(self):
        """Actual datetime.date objects must be accepted for both dates and
        fire_date without raising an error.

        Mirrors the realistic NBR fire trajectory but passes native Python
        datetime objects instead of strings.
        """
        import datetime
        from tanager.visualization import plot_temporal_trajectory

        dt_dates = [
            datetime.date(2024, 12, 15),
            datetime.date(2024, 12, 25),
            datetime.date(2025, 1, 1),
            datetime.date(2025, 1, 7),
            datetime.date(2025, 1, 15),
            datetime.date(2025, 1, 23),
            datetime.date(2025, 2, 1),
        ]
        dt_fire = datetime.date(2025, 1, 7)

        fig = plot_temporal_trajectory(
            dt_dates, _TRAJ_VALUES, "NBR", fire_date=dt_fire
        )
        ax = fig.axes[0]
        try:
            # Both data line and fire vline must be present.
            assert len(ax.lines) >= 2, (
                "Expected data line + fire vline when datetime objects passed"
            )
        finally:
            plt.close(fig)

    # ------------------------------------------------------------------
    # 6. String dates accepted
    # ------------------------------------------------------------------

    def test_string_dates_work(self):
        """String dates in 'YYYY-MM-DD' format must be parsed correctly.

        Uses the canonical fire-trajectory strings defined at the top of this
        section and verifies the complete output: line present, vline present,
        and y-axis label matches the product name.
        """
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(
            _TRAJ_DATES, _TRAJ_VALUES, "NBR", fire_date=_TRAJ_FIRE_DATE
        )
        ax = fig.axes[0]
        try:
            # Data line present.
            assert len(ax.lines) >= 1, "No data line found for string date input"
            # Fire vline present.
            assert len(ax.lines) >= 2, "No fire vline found for string date input"
            # Y-axis label set correctly.
            assert ax.get_ylabel() == "NBR", (
                f"Expected ylabel='NBR', got {ax.get_ylabel()!r}"
            )
        finally:
            plt.close(fig)
