"""Unit tests for add_scalebar — scale bar element in visualization.py."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import

import matplotlib.pyplot as plt
import pytest

from tanager.visualization import add_scalebar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTM_XLIM = (340_000, 350_000)  # 10 km span in UTM metres
_UTM_YLIM = (3_780_000, 3_790_000)  # 10 km span


def _make_ax(xlim=_UTM_XLIM, ylim=_UTM_YLIM):
    """Return a fresh Axes with UTM-scale limits."""
    fig, ax = plt.subplots()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return fig, ax


# ---------------------------------------------------------------------------
# Return value
# ---------------------------------------------------------------------------


class TestAddScalebarReturnsAx:
    """add_scalebar must return the same Axes object passed in."""

    def test_returns_ax(self):
        fig, ax = _make_ax()
        try:
            result = add_scalebar(ax, 5.0)
            assert result is ax
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Rectangle patch
# ---------------------------------------------------------------------------


class TestAddScalebarPatch:
    """A Rectangle patch must be added to the axes."""

    def test_adds_at_least_one_patch(self):
        fig, ax = _make_ax()
        try:
            add_scalebar(ax, 5.0)
            assert len(ax.patches) >= 1
        finally:
            plt.close(fig)

    def test_patch_width_equals_bar_metres(self):
        """Bar width in data coords must equal length_km * 1000 metres."""
        fig, ax = _make_ax()
        try:
            add_scalebar(ax, 5.0)
            rect = ax.patches[0]
            assert rect.get_width() == pytest.approx(5000.0)
        finally:
            plt.close(fig)

    def test_patch_width_scales_with_length_km(self):
        """2 km → 2000 m width; 10 km → 10 000 m width."""
        for km, expected_m in [(2.0, 2000.0), (10.0, 10_000.0)]:
            fig, ax = _make_ax()
            try:
                add_scalebar(ax, km)
                assert ax.patches[0].get_width() == pytest.approx(expected_m)
            finally:
                plt.close(fig)

    def test_patch_facecolor_is_black(self):
        """The scale bar fill must be black."""
        import matplotlib.colors as mcolors

        fig, ax = _make_ax()
        try:
            add_scalebar(ax, 5.0)
            fc = ax.patches[0].get_facecolor()
            # facecolor is returned as an RGBA tuple; compare to black
            assert mcolors.to_rgba("black") == pytest.approx(fc, abs=1e-6)
        finally:
            plt.close(fig)

    def test_patch_edgecolor_is_white(self):
        """The scale bar outline must be white (for contrast)."""
        import matplotlib.colors as mcolors

        fig, ax = _make_ax()
        try:
            add_scalebar(ax, 5.0)
            ec = ax.patches[0].get_edgecolor()
            assert mcolors.to_rgba("white") == pytest.approx(ec, abs=1e-6)
        finally:
            plt.close(fig)

    def test_patch_zorder_is_high(self):
        """The rectangle must sit above ordinary plot elements (zorder >= 10)."""
        fig, ax = _make_ax()
        try:
            add_scalebar(ax, 5.0)
            assert ax.patches[0].get_zorder() >= 10
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Text label
# ---------------------------------------------------------------------------


class TestAddScalebarLabel:
    """A text label must be added centred above the bar."""

    def test_adds_at_least_one_text(self):
        fig, ax = _make_ax()
        try:
            add_scalebar(ax, 5.0)
            assert len(ax.texts) >= 1
        finally:
            plt.close(fig)

    def test_label_text_whole_number(self):
        """Integer km values must format as '<n> km' without a decimal."""
        fig, ax = _make_ax()
        try:
            add_scalebar(ax, 5.0)
            labels = [t.get_text() for t in ax.texts]
            assert "5 km" in labels
        finally:
            plt.close(fig)

    def test_label_text_fractional_km(self):
        """Fractional km values must preserve the decimal in the label."""
        fig, ax = _make_ax()
        try:
            add_scalebar(ax, 2.5)
            labels = [t.get_text() for t in ax.texts]
            assert "2.5 km" in labels
        finally:
            plt.close(fig)

    def test_label_horizontal_alignment_center(self):
        """Label must be centred horizontally over the bar."""
        fig, ax = _make_ax()
        try:
            add_scalebar(ax, 5.0)
            assert ax.texts[0].get_ha() == "center"
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Location positioning
# ---------------------------------------------------------------------------


class TestAddScalebarLocation:
    """Bar anchor must move to the correct corner for each location string."""

    def _bar_x(self, ax):
        """Return x-position (left edge) of the first Rectangle patch."""
        return ax.patches[0].get_x()

    def _bar_y(self, ax):
        """Return y-position (bottom edge) of the first Rectangle patch."""
        return ax.patches[0].get_y()

    def test_lower_left_bar_near_left(self):
        """'lower left': bar left edge must be close to xmin."""
        fig, ax = _make_ax()
        try:
            add_scalebar(ax, 1.0, location="lower left")
            xmin, xmax = ax.get_xlim()
            # Should be within the left 20% of the axes
            assert self._bar_x(ax) < xmin + 0.20 * (xmax - xmin)
        finally:
            plt.close(fig)

    def test_lower_right_bar_near_right(self):
        """'lower right': bar right edge must be close to xmax."""
        fig, ax = _make_ax()
        try:
            add_scalebar(ax, 1.0, location="lower right")
            xmin, xmax = ax.get_xlim()
            bar_right = self._bar_x(ax) + 1000.0  # bar_width = 1 km
            # Right edge of bar should be within the right 20% of the axes
            assert bar_right > xmin + 0.80 * (xmax - xmin)
        finally:
            plt.close(fig)

    def test_upper_left_bar_near_top(self):
        """'upper left': bar y must be above the midpoint of the axes."""
        fig, ax = _make_ax()
        try:
            add_scalebar(ax, 1.0, location="upper left")
            ymin, ymax = ax.get_ylim()
            midpoint = (ymin + ymax) / 2.0
            assert self._bar_y(ax) > midpoint
        finally:
            plt.close(fig)

    def test_upper_right_bar_near_top_right(self):
        """'upper right': bar must be in the upper-right quadrant."""
        fig, ax = _make_ax()
        try:
            add_scalebar(ax, 1.0, location="upper right")
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            bar_right = self._bar_x(ax) + 1000.0
            assert bar_right > (xmin + xmax) / 2.0
            assert self._bar_y(ax) > (ymin + ymax) / 2.0
        finally:
            plt.close(fig)

    def test_unknown_location_falls_back_to_lower_left(self):
        """Unrecognised location strings must fall back to lower-left."""
        fig_ll, ax_ll = _make_ax()
        fig_unk, ax_unk = _make_ax()
        try:
            add_scalebar(ax_ll, 1.0, location="lower left")
            add_scalebar(ax_unk, 1.0, location="foobar")
            assert ax_ll.patches[0].get_x() == pytest.approx(ax_unk.patches[0].get_x())
            assert ax_ll.patches[0].get_y() == pytest.approx(ax_unk.patches[0].get_y())
        finally:
            plt.close(fig_ll)
            plt.close(fig_unk)


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------


class TestAddScalebarDefaults:
    """Default arguments must produce a 5 km bar in the lower-left."""

    def test_default_length_is_5km(self):
        fig, ax = _make_ax()
        try:
            add_scalebar(ax)
            assert ax.patches[0].get_width() == pytest.approx(5000.0)
        finally:
            plt.close(fig)

    def test_default_location_is_lower_left(self):
        """Calling with defaults and with location='lower left' must produce same position."""
        fig_def, ax_def = _make_ax()
        fig_ll, ax_ll = _make_ax()
        try:
            add_scalebar(ax_def)
            add_scalebar(ax_ll, 5.0, location="lower left")
            assert ax_def.patches[0].get_x() == pytest.approx(ax_ll.patches[0].get_x())
            assert ax_def.patches[0].get_y() == pytest.approx(ax_ll.patches[0].get_y())
        finally:
            plt.close(fig_def)
            plt.close(fig_ll)
