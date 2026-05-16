"""Tests for overlay_perimeters — vector boundary overlay on matplotlib Axes.

Covers:
- Returns the same Axes object (identity)
- Empty GeoDataFrame → returns ax unchanged, no collections added
- Boundary lines appear in ax.collections after call
- Reprojection: EPSG:4326 input is accepted and reprojected to EPSG:32611
- label=True: text labels drawn at centroids when "name" column exists
- label=True: text labels drawn at centroids when "incident_name" column exists
- label=True: no labels drawn when neither name column exists
- label=False: no text labels drawn even with name column
- color parameter is forwarded to boundary.plot
- Multiple polygons each get a label
- Already-projected (EPSG:32611) input works without error
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import

import matplotlib.pyplot as plt
import pytest

import geopandas as gpd
from shapely.geometry import Polygon

from tanager.visualization import overlay_perimeters


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_POLYGON_4326 = Polygon(
    [(-118.5, 34.0), (-118.4, 34.0), (-118.4, 34.1), (-118.5, 34.1)]
)

_POLYGON_4326_B = Polygon(
    [(-118.3, 34.0), (-118.2, 34.0), (-118.2, 34.1), (-118.3, 34.1)]
)


def _make_gdf(names=None, crs="EPSG:4326", polygons=None):
    """Return a GeoDataFrame with one or more polygons."""
    if polygons is None:
        polygons = [_POLYGON_4326]
    props = {}
    if names is not None:
        props["name"] = names
    return gpd.GeoDataFrame(props, geometry=polygons, crs=crs)


def _make_incident_gdf():
    """Return a GeoDataFrame using 'incident_name' instead of 'name'."""
    return gpd.GeoDataFrame(
        {"incident_name": ["Eaton Fire"]},
        geometry=[_POLYGON_4326],
        crs="EPSG:4326",
    )


def _make_empty_gdf():
    """Return a GeoDataFrame with zero features."""
    return gpd.GeoDataFrame(
        {"name": []},
        geometry=gpd.GeoSeries([], crs="EPSG:4326"),
        crs="EPSG:4326",
    )


@pytest.fixture()
def ax():
    """Matplotlib Axes with a UTM-like extent pre-set."""
    fig, _ax = plt.subplots()
    _ax.set_xlim(340_000, 350_000)
    _ax.set_ylim(3_780_000, 3_790_000)
    yield _ax
    plt.close(fig)


# ---------------------------------------------------------------------------
# Return value
# ---------------------------------------------------------------------------


class TestOverlayPerimetersReturnsAx:
    """overlay_perimeters must return the same Axes object it received."""

    def test_returns_ax_with_features(self, ax):
        gdf = _make_gdf(["Test Fire"])
        result = overlay_perimeters(ax, gdf)
        assert result is ax

    def test_returns_ax_when_empty(self, ax):
        result = overlay_perimeters(ax, _make_empty_gdf())
        assert result is ax

    def test_returns_ax_label_false(self, ax):
        gdf = _make_gdf(["Test Fire"])
        result = overlay_perimeters(ax, gdf, label=False)
        assert result is ax


# ---------------------------------------------------------------------------
# Empty GeoDataFrame
# ---------------------------------------------------------------------------


class TestOverlayPerimetersEmptyGeoDataFrame:
    """Empty GeoDataFrame must be a no-op — ax returned unchanged."""

    def test_no_collections_added_for_empty_gdf(self, ax):
        n_collections_before = len(ax.collections)
        overlay_perimeters(ax, _make_empty_gdf())
        assert len(ax.collections) == n_collections_before

    def test_no_texts_added_for_empty_gdf(self, ax):
        n_texts_before = len(ax.texts)
        overlay_perimeters(ax, _make_empty_gdf())
        assert len(ax.texts) == n_texts_before

    def test_xlim_unchanged_for_empty_gdf(self, ax):
        xlim_before = ax.get_xlim()
        overlay_perimeters(ax, _make_empty_gdf())
        assert ax.get_xlim() == xlim_before


# ---------------------------------------------------------------------------
# Boundary drawing — collections appear
# ---------------------------------------------------------------------------


class TestOverlayPerimetersBoundaryDrawn:
    """After a non-empty call, ax.collections must contain at least one item."""

    def test_collection_added(self, ax):
        gdf = _make_gdf(["Palisades Fire"])
        overlay_perimeters(ax, gdf, label=False)
        assert len(ax.collections) > 0

    def test_collection_or_lines_added(self, ax):
        """Either collections or lines must be non-empty after plotting."""
        gdf = _make_gdf(["Palisades Fire"])
        overlay_perimeters(ax, gdf, label=False)
        assert len(ax.collections) > 0 or len(ax.lines) > 0

    def test_two_polygons_produce_collections(self, ax):
        gdf = _make_gdf(["Fire A", "Fire B"], polygons=[_POLYGON_4326, _POLYGON_4326_B])
        overlay_perimeters(ax, gdf, label=False)
        assert len(ax.collections) > 0


# ---------------------------------------------------------------------------
# CRS reprojection
# ---------------------------------------------------------------------------


class TestOverlayPerimetersReprojection:
    """Input GeoDataFrames in any CRS should be accepted without error."""

    def test_wgs84_input_does_not_raise(self, ax):
        gdf = _make_gdf(["Test Fire"], crs="EPSG:4326")
        # Should not raise
        overlay_perimeters(ax, gdf, label=False)

    def test_already_utm_input_does_not_raise(self, ax):
        """EPSG:32611 input (already in the target CRS) works without error."""
        gdf_wgs = _make_gdf(["Test Fire"], crs="EPSG:4326")
        gdf_utm = gdf_wgs.to_crs("EPSG:32611")
        # Should not raise
        overlay_perimeters(ax, gdf_utm, label=False)

    def test_wgs84_still_adds_collection(self, ax):
        gdf = _make_gdf(["Test Fire"], crs="EPSG:4326")
        overlay_perimeters(ax, gdf, label=False)
        assert len(ax.collections) > 0


# ---------------------------------------------------------------------------
# Label drawing — name column
# ---------------------------------------------------------------------------


class TestOverlayPerimetersLabelFromNameColumn:
    """When label=True and 'name' column exists, text labels are drawn."""

    def test_label_added_for_name_column(self, ax):
        gdf = _make_gdf(["Test Fire"])
        overlay_perimeters(ax, gdf, label=True)
        assert len(ax.texts) > 0

    def test_label_text_matches_name(self, ax):
        gdf = _make_gdf(["Palisades Fire"])
        overlay_perimeters(ax, gdf, label=True)
        labels = [t.get_text() for t in ax.texts]
        assert "Palisades Fire" in labels

    def test_two_polygons_two_labels(self, ax):
        gdf = _make_gdf(
            ["Fire A", "Fire B"],
            polygons=[_POLYGON_4326, _POLYGON_4326_B],
        )
        overlay_perimeters(ax, gdf, label=True)
        assert len(ax.texts) == 2

    def test_label_uses_color_parameter(self, ax):
        gdf = _make_gdf(["Test Fire"])
        overlay_perimeters(ax, gdf, color="blue", label=True)
        assert any(t.get_color() == "blue" for t in ax.texts)


# ---------------------------------------------------------------------------
# Label drawing — incident_name column
# ---------------------------------------------------------------------------


class TestOverlayPerimetersLabelFromIncidentNameColumn:
    """When label=True and 'incident_name' column exists (but not 'name'), labels appear."""

    def test_label_added_for_incident_name_column(self, ax):
        gdf = _make_incident_gdf()
        overlay_perimeters(ax, gdf, label=True)
        assert len(ax.texts) > 0

    def test_label_text_matches_incident_name(self, ax):
        gdf = _make_incident_gdf()
        overlay_perimeters(ax, gdf, label=True)
        labels = [t.get_text() for t in ax.texts]
        assert "Eaton Fire" in labels


# ---------------------------------------------------------------------------
# Label drawing — no name column
# ---------------------------------------------------------------------------


class TestOverlayPerimetersNoNameColumn:
    """When neither name column exists, label=True produces no text elements."""

    def test_no_labels_when_no_name_column(self, ax):
        gdf = gpd.GeoDataFrame(
            {"area_ha": [1234.5]},
            geometry=[_POLYGON_4326],
            crs="EPSG:4326",
        )
        overlay_perimeters(ax, gdf, label=True)
        assert len(ax.texts) == 0


# ---------------------------------------------------------------------------
# label=False — no labels regardless of columns
# ---------------------------------------------------------------------------


class TestOverlayPerimetersLabelFalse:
    """When label=False, no text labels must appear even with a name column."""

    def test_label_false_no_texts(self, ax):
        gdf = _make_gdf(["Test Fire"])
        overlay_perimeters(ax, gdf, label=False)
        assert len(ax.texts) == 0

    def test_label_false_still_draws_boundary(self, ax):
        gdf = _make_gdf(["Test Fire"])
        overlay_perimeters(ax, gdf, label=False)
        assert len(ax.collections) > 0 or len(ax.lines) > 0


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------


class TestOverlayPerimetersDefaults:
    """Default color=red, linestyle=--, linewidth=2.0, label=True."""

    def test_defaults_do_not_raise(self, ax):
        gdf = _make_gdf(["Test Fire"])
        overlay_perimeters(ax, gdf)

    def test_default_label_true_adds_text(self, ax):
        gdf = _make_gdf(["Test Fire"])
        overlay_perimeters(ax, gdf)
        assert len(ax.texts) > 0
