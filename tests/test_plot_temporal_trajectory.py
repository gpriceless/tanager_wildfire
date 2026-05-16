"""Tests for plot_temporal_trajectory time-series chart function."""

from __future__ import annotations

import datetime

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fig_and_ax():
    """Return a fresh (fig, ax) pair."""
    return plt.subplots(figsize=(8, 4))


def _close(*figs):
    for f in figs:
        if f is not None:
            plt.close(f)


STR_DATES = ["2024-12-15", "2024-12-25", "2025-01-07", "2025-01-15", "2025-01-23"]
FLOAT_VALUES = [0.65, 0.62, 0.15, 0.20, 0.25]
FIRE_DATE_STR = "2025-01-07"


# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------

class TestReturnsFigure:
    """plot_temporal_trajectory must return a matplotlib Figure."""

    def test_returns_figure(self):
        from matplotlib.figure import Figure
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(STR_DATES, FLOAT_VALUES, "NBR")
        try:
            assert isinstance(fig, Figure)
        finally:
            _close(fig)

    def test_figure_has_axes(self):
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(STR_DATES, FLOAT_VALUES, "NBR")
        try:
            assert len(fig.axes) >= 1
        finally:
            _close(fig)


# ---------------------------------------------------------------------------
# Line chart with markers
# ---------------------------------------------------------------------------

class TestLineChartWithMarkers:
    """The data line must appear as a Line2D with markers."""

    def test_has_at_least_one_line(self):
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(STR_DATES, FLOAT_VALUES, "NBR", fire_date=None)
        ax = fig.axes[0]
        try:
            assert len(ax.lines) >= 1
        finally:
            _close(fig)

    def test_data_line_has_marker(self):
        """The data series line must use a non-empty marker."""
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(STR_DATES, FLOAT_VALUES, "NBR", fire_date=None)
        ax = fig.axes[0]
        try:
            data_line = ax.lines[0]
            assert data_line.get_marker() not in (None, "None", "")
        finally:
            _close(fig)

    def test_data_line_has_correct_y_values(self):
        """The data line's y-data must match the supplied values."""
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(STR_DATES, FLOAT_VALUES, "NBR", fire_date=None)
        ax = fig.axes[0]
        try:
            y_data = list(ax.lines[0].get_ydata())
            assert y_data == pytest.approx(FLOAT_VALUES)
        finally:
            _close(fig)


# ---------------------------------------------------------------------------
# Y-axis label
# ---------------------------------------------------------------------------

class TestYAxisLabel:
    """Y-axis label must use product_name."""

    def test_ylabel_is_product_name(self):
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(STR_DATES, FLOAT_VALUES, "NBR", fire_date=None)
        try:
            assert fig.axes[0].get_ylabel() == "NBR"
        finally:
            _close(fig)

    def test_ylabel_is_not_empty(self):
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(STR_DATES, FLOAT_VALUES, "NDVI")
        try:
            assert fig.axes[0].get_ylabel() != ""
        finally:
            _close(fig)

    def test_ylabel_matches_custom_product_name(self):
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(STR_DATES, FLOAT_VALUES, "LFMC (%)")
        try:
            assert fig.axes[0].get_ylabel() == "LFMC (%)"
        finally:
            _close(fig)


# ---------------------------------------------------------------------------
# Fire date vertical line
# ---------------------------------------------------------------------------

class TestFireDateVerticalLine:
    """axvline for fire_date must be added when fire_date is supplied."""

    def test_fire_date_adds_vline(self):
        """With fire_date, ax.lines must have >= 2 lines (data + vline)."""
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(STR_DATES, FLOAT_VALUES, "NBR",
                                       fire_date=FIRE_DATE_STR)
        ax = fig.axes[0]
        try:
            assert len(ax.lines) >= 2
        finally:
            _close(fig)

    def test_no_fire_date_no_vline(self):
        """With fire_date=None, ax.lines should have exactly 1 line (just data)."""
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(STR_DATES, FLOAT_VALUES, "NBR",
                                       fire_date=None)
        ax = fig.axes[0]
        try:
            assert len(ax.lines) == 1
        finally:
            _close(fig)

    def test_fire_date_line_is_red(self):
        """The second line (the vline) must be red."""
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(STR_DATES, FLOAT_VALUES, "NBR",
                                       fire_date=FIRE_DATE_STR)
        ax = fig.axes[0]
        try:
            # The vline is the last line added
            vline = ax.lines[-1]
            import matplotlib.colors as mcolors
            rgba = mcolors.to_rgba(vline.get_color())
            assert rgba[0] == pytest.approx(1.0)  # R=1 for red
            assert rgba[1] == pytest.approx(0.0)  # G=0 for red
            assert rgba[2] == pytest.approx(0.0)  # B=0 for red
        finally:
            _close(fig)

    def test_fire_date_vline_dashed(self):
        """The fire date vline must use a dashed linestyle."""
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(STR_DATES, FLOAT_VALUES, "NBR",
                                       fire_date=FIRE_DATE_STR)
        ax = fig.axes[0]
        try:
            vline = ax.lines[-1]
            assert vline.get_linestyle() == "--"
        finally:
            _close(fig)


# ---------------------------------------------------------------------------
# Error bands
# ---------------------------------------------------------------------------

class TestErrorBands:
    """fill_between error bands must appear when error_bands is supplied."""

    def test_error_bands_create_collection(self):
        """fill_between adds a PolyCollection to ax.collections."""
        from tanager.visualization import plot_temporal_trajectory

        error_bands = [0.05, 0.04, 0.06, 0.05, 0.04]
        fig = plot_temporal_trajectory(
            STR_DATES, FLOAT_VALUES, "NBR",
            fire_date=None, error_bands=error_bands
        )
        ax = fig.axes[0]
        try:
            assert len(ax.collections) >= 1
        finally:
            _close(fig)

    def test_no_error_bands_no_fill_between(self):
        """Without error_bands, ax.collections must be empty (no spans either)."""
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(
            STR_DATES, FLOAT_VALUES, "NBR",
            fire_date=None, error_bands=None
        )
        ax = fig.axes[0]
        try:
            assert len(ax.collections) == 0
        finally:
            _close(fig)


# ---------------------------------------------------------------------------
# String vs datetime input dates
# ---------------------------------------------------------------------------

class TestDateInputTypes:
    """dates and fire_date may be strings or datetime objects."""

    def test_string_dates_work(self):
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(STR_DATES, FLOAT_VALUES, "NBR",
                                       fire_date=FIRE_DATE_STR)
        try:
            assert len(fig.axes[0].lines) >= 2
        finally:
            _close(fig)

    def test_datetime_dates_work(self):
        """datetime.date objects must be accepted without error."""
        from tanager.visualization import plot_temporal_trajectory

        dt_dates = [
            datetime.date(2024, 12, 15),
            datetime.date(2024, 12, 25),
            datetime.date(2025, 1, 7),
            datetime.date(2025, 1, 15),
            datetime.date(2025, 1, 23),
        ]
        dt_fire = datetime.date(2025, 1, 7)

        fig = plot_temporal_trajectory(dt_dates, FLOAT_VALUES, "NBR",
                                       fire_date=dt_fire)
        try:
            assert len(fig.axes[0].lines) >= 2
        finally:
            _close(fig)

    def test_mixed_string_and_datetime_values(self):
        """A mix of string and datetime.datetime objects must parse without error."""
        from tanager.visualization import plot_temporal_trajectory

        mixed_dates = [
            "2024-12-15",
            datetime.datetime(2024, 12, 25),
            "2025-01-07",
            datetime.datetime(2025, 1, 15),
            "2025-01-23",
        ]
        fig = plot_temporal_trajectory(mixed_dates, FLOAT_VALUES, "NBR",
                                       fire_date="2025-01-07")
        try:
            assert len(fig.axes[0].lines) >= 2
        finally:
            _close(fig)


# ---------------------------------------------------------------------------
# ax parameter (use-existing-axes path)
# ---------------------------------------------------------------------------

class TestExistingAxes:
    """When ax is provided, the function must draw into it and return its figure."""

    def test_uses_provided_axes(self):
        from tanager.visualization import plot_temporal_trajectory

        outer_fig, outer_ax = _make_fig_and_ax()
        try:
            result_fig = plot_temporal_trajectory(
                STR_DATES, FLOAT_VALUES, "NBR",
                fire_date=None, ax=outer_ax
            )
            assert result_fig is outer_fig
        finally:
            _close(outer_fig)

    def test_lines_appear_on_provided_axes(self):
        from tanager.visualization import plot_temporal_trajectory

        outer_fig, outer_ax = _make_fig_and_ax()
        try:
            plot_temporal_trajectory(
                STR_DATES, FLOAT_VALUES, "NBR",
                fire_date=None, ax=outer_ax
            )
            assert len(outer_ax.lines) >= 1
        finally:
            _close(outer_fig)


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

class TestLegend:
    """A legend must be present on the axes."""

    def test_legend_present(self):
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(STR_DATES, FLOAT_VALUES, "NBR",
                                       fire_date=FIRE_DATE_STR)
        ax = fig.axes[0]
        try:
            legend = ax.get_legend()
            assert legend is not None
        finally:
            _close(fig)

    def test_legend_contains_product_name(self):
        """The series label in the legend must match product_name."""
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(STR_DATES, FLOAT_VALUES, "NBR",
                                       fire_date=FIRE_DATE_STR)
        ax = fig.axes[0]
        try:
            legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
            assert any("NBR" in t for t in legend_texts)
        finally:
            _close(fig)

    def test_legend_contains_fire_ignition_label(self):
        """When fire_date is given, 'Fire Ignition' must appear in the legend."""
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(STR_DATES, FLOAT_VALUES, "NBR",
                                       fire_date=FIRE_DATE_STR)
        ax = fig.axes[0]
        try:
            legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
            assert any("Fire Ignition" in t for t in legend_texts)
        finally:
            _close(fig)


# ---------------------------------------------------------------------------
# publication mode
# ---------------------------------------------------------------------------

class TestPublicationMode:
    """publication=True must set DPI to 300."""

    def test_publication_sets_dpi_300(self):
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(STR_DATES, FLOAT_VALUES, "NBR",
                                       publication=True)
        try:
            assert fig.get_dpi() == 300
        finally:
            _close(fig)

    def test_default_mode_not_300_dpi(self):
        """Without publication=True, DPI must not be forced to 300."""
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(STR_DATES, FLOAT_VALUES, "NBR",
                                       publication=False)
        try:
            assert fig.get_dpi() != 300
        finally:
            _close(fig)


# ---------------------------------------------------------------------------
# figsize
# ---------------------------------------------------------------------------

class TestFigsize:
    """Custom figsize must be reflected in the returned figure."""

    def test_custom_figsize(self):
        from tanager.visualization import plot_temporal_trajectory

        fig = plot_temporal_trajectory(STR_DATES, FLOAT_VALUES, "NBR",
                                       figsize=(8, 4))
        try:
            w, h = fig.get_size_inches()
            assert (w, h) == pytest.approx((8.0, 4.0))
        finally:
            _close(fig)


# ---------------------------------------------------------------------------
# Acceptance criteria verification snippet (from task spec)
# ---------------------------------------------------------------------------

class TestAcceptanceCriteria:
    """Exact verification snippet from the task spec must pass."""

    def test_spec_verification_snippet(self):
        from tanager.visualization import plot_temporal_trajectory

        dates = ["2024-12-15", "2024-12-25", "2025-01-07", "2025-01-15", "2025-01-23"]
        values = [0.65, 0.62, 0.15, 0.20, 0.25]
        fig = plot_temporal_trajectory(dates, values, "NBR", fire_date="2025-01-07")
        ax = fig.axes[0]
        try:
            assert len(ax.lines) >= 2  # data line + fire date vline
            assert ax.get_ylabel() != ""
        finally:
            _close(fig)
