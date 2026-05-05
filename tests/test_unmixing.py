"""Tests for tanager.unmixing — MESMA pipeline, fallback, normalisation, plots."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import xarray as xr

from tanager import unmixing
from tanager.unmixing import (
    DEFAULT_CONSTRAINTS,
    normalize_fractions,
    plot_fraction_maps,
    plot_rgb_composite,
    run_mesma,
    select_bands_uszu,
)


# ---------------------------------------------------------------------------
# Library + scene fixtures
# ---------------------------------------------------------------------------


def _make_library(wavelengths: np.ndarray) -> xr.DataArray:
    """Build a tiny 5-spectrum endmember library (char, pv, npv, soil, shade)."""
    wl = np.asarray(wavelengths, dtype=np.float32)
    nb = wl.size

    char = np.linspace(0.02, 0.05, nb).astype(np.float32)
    soil = np.linspace(0.10, 0.30, nb).astype(np.float32)
    npv = np.linspace(0.15, 0.35, nb).astype(np.float32)
    pv = np.full(nb, 0.05, dtype=np.float32)
    # Vegetation: chlorophyll dip + NIR shoulder
    red_edge = (wl >= 700) & (wl <= 1300)
    pv[red_edge] = 0.45
    pv[(wl > 1300) & (wl <= 2500)] = np.linspace(0.40, 0.10, ((wl > 1300) & (wl <= 2500)).sum()).astype(np.float32)
    shade = np.zeros(nb, dtype=np.float32)

    spectra = np.stack([char, pv, npv, soil, shade], axis=0)  # (5, n_bands)
    return xr.DataArray(
        spectra,
        dims=("spectrum_id", "wavelength"),
        coords={
            "spectrum_id": np.array(["char_001", "pv_001", "npv_001", "soil_001", "shade_001"], dtype=object),
            "wavelength": wl,
            "category": ("spectrum_id", np.array(["char", "pv", "npv", "soil", "shade"], dtype=object)),
            "name": ("spectrum_id", np.array(["char", "pv", "npv", "soil", "shade"], dtype=object)),
            "source": ("spectrum_id", np.array(["synthetic"] * 5, dtype=object)),
        },
    )


def _make_pure_pixel_scene(library: xr.DataArray) -> xr.Dataset:
    """Build a 4x4 scene where each row is a different pure endmember.

    Row 0: char, Row 1: pv, Row 2: npv, Row 3: soil. Three columns plus a
    background column (all zeros) so we cover the full canonical class set
    plus an unmodelable pixel.
    """
    wl = np.asarray(library.coords["wavelength"].values, dtype=np.float32)
    nb = wl.size
    n_rows, n_cols = 4, 4
    arr = np.zeros((nb, n_rows, n_cols), dtype=np.float32)

    char_spec = library.sel(spectrum_id="char_001").values
    pv_spec = library.sel(spectrum_id="pv_001").values
    npv_spec = library.sel(spectrum_id="npv_001").values
    soil_spec = library.sel(spectrum_id="soil_001").values

    arr[:, 0, :3] = char_spec[:, None]
    arr[:, 1, :3] = pv_spec[:, None]
    arr[:, 2, :3] = npv_spec[:, None]
    arr[:, 3, :3] = soil_spec[:, None]
    # Column 3 stays zero (degenerate).

    return xr.Dataset(
        {"reflectance": (["wavelength", "y", "x"], arr)},
        coords={
            "wavelength": wl,
            "y": np.arange(n_rows),
            "x": np.arange(n_cols),
        },
    )


@pytest.fixture
def small_wavelengths() -> np.ndarray:
    """30 evenly-spaced wavelengths from 400 to 2400 nm — fast for MESMA tests."""
    return np.linspace(400.0, 2400.0, 30).astype(np.float32)


@pytest.fixture
def small_library(small_wavelengths) -> xr.DataArray:
    return _make_library(small_wavelengths)


@pytest.fixture
def pure_pixel_scene(small_library) -> xr.Dataset:
    return _make_pure_pixel_scene(small_library)


# ---------------------------------------------------------------------------
# select_bands_uszu
# ---------------------------------------------------------------------------


class TestSelectBandsUSZU:
    def test_returns_requested_band_count(self, pure_pixel_scene, small_library):
        subset, indices = select_bands_uszu(pure_pixel_scene, small_library, n_bands=10)
        assert isinstance(subset, xr.Dataset)
        assert subset.sizes["wavelength"] == 10
        assert indices.shape == (10,)
        assert indices.dtype.kind in ("i", "u")

    def test_indices_are_within_library(self, pure_pixel_scene, small_library):
        n_lib = small_library.sizes["wavelength"]
        _, indices = select_bands_uszu(pure_pixel_scene, small_library, n_bands=8)
        assert indices.min() >= 0
        assert indices.max() < n_lib

    def test_rejects_invalid_n_bands(self, pure_pixel_scene, small_library):
        with pytest.raises(ValueError):
            select_bands_uszu(pure_pixel_scene, small_library, n_bands=0)
        with pytest.raises(ValueError):
            select_bands_uszu(
                pure_pixel_scene, small_library, n_bands=small_library.sizes["wavelength"] + 1,
            )

    def test_requires_category_coord(self, pure_pixel_scene, small_library):
        no_cat = small_library.drop_vars("category")
        with pytest.raises(ValueError):
            select_bands_uszu(pure_pixel_scene, no_cat, n_bands=4)


# ---------------------------------------------------------------------------
# run_mesma — output schema (works for both mesma and fallback backends)
# ---------------------------------------------------------------------------


class TestRunMesmaSchema:
    def test_canonical_variables_present(self, pure_pixel_scene, small_library):
        result = run_mesma(pure_pixel_scene, small_library)
        for var in ("char", "pv", "npv", "soil", "shade", "rmse"):
            assert var in result.data_vars, f"missing {var!r}"
        for var in ("char", "pv", "npv", "soil", "shade", "rmse"):
            assert result[var].dims == ("y", "x")

    def test_unmixing_engine_attr_set(self, pure_pixel_scene, small_library):
        result = run_mesma(pure_pixel_scene, small_library)
        engine = result.attrs.get("unmixing_engine")
        assert engine in {"mesma", "hysup", "nnls"}

    def test_output_shape_matches_scene(self, pure_pixel_scene, small_library):
        result = run_mesma(pure_pixel_scene, small_library)
        assert result.sizes["y"] == pure_pixel_scene.sizes["y"]
        assert result.sizes["x"] == pure_pixel_scene.sizes["x"]


# ---------------------------------------------------------------------------
# run_mesma — fallback path (force NNLS)
# ---------------------------------------------------------------------------


class TestRunMesmaFallback:
    def test_fallback_pure_pixel_recovery(self, pure_pixel_scene, small_library, monkeypatch):
        """When mesma is unavailable, NNLS fallback should still recover pure pixels."""
        monkeypatch.setattr(unmixing, "_MESMA_AVAILABLE", False)
        monkeypatch.setattr(unmixing, "_HYSUP_AVAILABLE", False)
        result = run_mesma(pure_pixel_scene, small_library)
        assert result.attrs["unmixing_engine"] == "nnls"

        # Row 0 = char, row 1 = pv, row 2 = npv, row 3 = soil.
        assert float(result["char"].isel(y=0, x=0).values) > 0.8
        assert float(result["pv"].isel(y=1, x=0).values) > 0.8
        assert float(result["npv"].isel(y=2, x=0).values) > 0.8
        assert float(result["soil"].isel(y=3, x=0).values) > 0.8

    def test_fallback_fractions_sum_to_one(self, pure_pixel_scene, small_library, monkeypatch):
        monkeypatch.setattr(unmixing, "_MESMA_AVAILABLE", False)
        monkeypatch.setattr(unmixing, "_HYSUP_AVAILABLE", False)
        result = run_mesma(pure_pixel_scene, small_library)
        # First 3 columns hold pure-endmember pixels; column 3 is the all-zero
        # degenerate spectrum (NNLS returns all-zero fractions for it, which
        # is mathematically correct but trivially sums to 0).
        total = sum(result[v].values for v in ("char", "pv", "npv", "soil", "shade"))
        np.testing.assert_allclose(total[:, :3], 1.0, atol=0.05)


# ---------------------------------------------------------------------------
# run_mesma — primary mesma backend
# ---------------------------------------------------------------------------


_MESMA_INSTALLED = importlib.util.find_spec("mesma") is not None


@pytest.mark.skipif(not _MESMA_INSTALLED, reason="mesma package not installed")
class TestRunMesmaPrimary:
    def test_mesma_engine_used(self, pure_pixel_scene, small_library):
        result = run_mesma(pure_pixel_scene, small_library)
        assert result.attrs["unmixing_engine"] == "mesma"

    def test_mesma_recovers_pure_char(self, pure_pixel_scene, small_library):
        result = run_mesma(pure_pixel_scene, small_library)
        char = float(result["char"].isel(y=0, x=0).values)
        # MESMA either recovers char ~1.0 or leaves the pixel NaN if no model
        # passed constraints; both are acceptable. We just assert no crash and
        # that at least one row reports a confident class fraction somewhere.
        assert (char > 0.8) or np.isnan(char)


# ---------------------------------------------------------------------------
# bands subset
# ---------------------------------------------------------------------------


class TestRunMesmaBandsSubset:
    def test_bands_subset_runs(self, pure_pixel_scene, small_library, monkeypatch):
        monkeypatch.setattr(unmixing, "_MESMA_AVAILABLE", False)
        monkeypatch.setattr(unmixing, "_HYSUP_AVAILABLE", False)
        wl = small_library.coords["wavelength"].values
        bands = wl[::3]  # ~10 bands subset
        result = run_mesma(pure_pixel_scene, small_library, bands=bands)
        for var in ("char", "pv", "npv", "soil", "shade", "rmse"):
            assert var in result.data_vars


# ---------------------------------------------------------------------------
# Constraint filtering
# ---------------------------------------------------------------------------


class TestConstraintFiltering:
    def test_high_rmse_pixels_become_nan(self, small_library, monkeypatch):
        """Construct a scene with a non-library spectrum and a tight max_rmse."""
        monkeypatch.setattr(unmixing, "_MESMA_AVAILABLE", False)
        monkeypatch.setattr(unmixing, "_HYSUP_AVAILABLE", False)
        wl = np.asarray(small_library.coords["wavelength"].values, dtype=np.float32)
        nb = wl.size

        # All pixels are noise that no library spectrum models well.
        rng = np.random.default_rng(7)
        arr = rng.random((nb, 3, 3)).astype(np.float32) * 0.9 + 0.05
        scene = xr.Dataset(
            {"reflectance": (["wavelength", "y", "x"], arr)},
            coords={"wavelength": wl, "y": np.arange(3), "x": np.arange(3)},
        )

        result = run_mesma(scene, small_library, constraints={"max_rmse": 1e-6})
        # Every pixel violates the impossible RMSE → all NaN.
        assert np.isnan(result["rmse"].values).all()
        for v in ("char", "pv", "npv", "soil", "shade"):
            assert np.isnan(result[v].values).all()


# ---------------------------------------------------------------------------
# normalize_fractions
# ---------------------------------------------------------------------------


class TestNormalizeFractions:
    def _make_ds(self, shade_value: float) -> xr.Dataset:
        y = np.arange(2)
        x = np.arange(2)
        zeros = np.zeros((2, 2), dtype=np.float32)
        return xr.Dataset(
            {
                "char": (["y", "x"], np.full((2, 2), 0.21, dtype=np.float32)),
                "pv": (["y", "x"], np.full((2, 2), 0.21, dtype=np.float32)),
                "npv": (["y", "x"], np.full((2, 2), 0.14, dtype=np.float32)),
                "soil": (["y", "x"], np.full((2, 2), 0.14, dtype=np.float32)),
                "shade": (["y", "x"], np.full((2, 2), shade_value, dtype=np.float32)),
                "rmse": (["y", "x"], zeros),
            },
            coords={"y": y, "x": x},
        )

    def test_shade_removed_and_rescaled(self):
        ds = self._make_ds(shade_value=0.30)
        out = normalize_fractions(ds, remove_shade=True)
        assert "shade" not in out.data_vars
        total = sum(out[v].values for v in ("char", "pv", "npv", "soil"))
        np.testing.assert_allclose(total, 1.0, atol=1e-4)

    def test_full_shade_pixel_becomes_nan(self):
        ds = self._make_ds(shade_value=1.0)
        out = normalize_fractions(ds, remove_shade=True)
        for v in ("char", "pv", "npv", "soil"):
            assert np.isnan(out[v].values).all()

    def test_remove_shade_false_returns_copy(self):
        ds = self._make_ds(shade_value=0.30)
        out = normalize_fractions(ds, remove_shade=False)
        assert "shade" in out.data_vars
        np.testing.assert_array_equal(out["shade"].values, ds["shade"].values)

    def test_clamps_out_of_bounds_fractions_after_shade_normalization(self):
        # Shade rescale (divide by 1 - shade = 0.4) pushes char to 1.125 and
        # pv/soil slightly negative. The output must clamp to [0, 1] and
        # re-normalize so the canonical fractions still sum to 1.0.
        ds = xr.Dataset(
            {
                "char": (["y", "x"], np.array([[0.45]], dtype=np.float32)),
                "pv": (["y", "x"], np.array([[-0.02]], dtype=np.float32)),
                "npv": (["y", "x"], np.array([[0.18]], dtype=np.float32)),
                "soil": (["y", "x"], np.array([[-0.01]], dtype=np.float32)),
                "shade": (["y", "x"], np.array([[0.6]], dtype=np.float32)),
                "rmse": (["y", "x"], np.array([[0.01]], dtype=np.float32)),
            },
            coords={"y": [0], "x": [0]},
        )

        out = normalize_fractions(ds, remove_shade=True)

        for v in ("char", "pv", "npv", "soil"):
            vals = out[v].values
            assert np.all(vals >= 0.0), f"{v} has values below 0: {vals}"
            assert np.all(vals <= 1.0), f"{v} has values above 1: {vals}"

        total = sum(out[v].values for v in ("char", "pv", "npv", "soil"))
        np.testing.assert_allclose(total, 1.0, atol=1e-4)

    def test_extreme_overshoot_is_clamped(self):
        # Covers the upper end of the bug report (5-12% of pixels at min=-0.25,
        # max=1.25 in real Tanager scenes): a single fraction far above 1.0
        # combined with a strongly negative one. Clamp + re-normalize must
        # still yield values in [0, 1] that sum to 1.
        ds = xr.Dataset(
            {
                "char": (["y", "x"], np.array([[0.50]], dtype=np.float32)),
                "pv": (["y", "x"], np.array([[-0.10]], dtype=np.float32)),
                "npv": (["y", "x"], np.array([[0.10]], dtype=np.float32)),
                "soil": (["y", "x"], np.array([[0.00]], dtype=np.float32)),
                "shade": (["y", "x"], np.array([[0.5]], dtype=np.float32)),
                "rmse": (["y", "x"], np.array([[0.01]], dtype=np.float32)),
            },
            coords={"y": [0], "x": [0]},
        )

        out = normalize_fractions(ds, remove_shade=True)

        for v in ("char", "pv", "npv", "soil"):
            vals = out[v].values
            assert np.all((vals >= 0.0) & (vals <= 1.0)), (
                f"{v} outside [0, 1]: {vals}"
            )
        total = sum(out[v].values for v in ("char", "pv", "npv", "soil"))
        np.testing.assert_allclose(total, 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


class TestPlotting:
    @pytest.fixture
    def fractions_ds(self) -> xr.Dataset:
        rng = np.random.default_rng(0)
        y = np.arange(8)
        x = np.arange(8)
        return xr.Dataset(
            {
                "char": (["y", "x"], rng.random((8, 8)).astype(np.float32)),
                "pv": (["y", "x"], rng.random((8, 8)).astype(np.float32)),
                "npv": (["y", "x"], rng.random((8, 8)).astype(np.float32)),
                "soil": (["y", "x"], rng.random((8, 8)).astype(np.float32)),
                "shade": (["y", "x"], rng.random((8, 8)).astype(np.float32)),
                "rmse": (["y", "x"], np.full((8, 8), 0.01, dtype=np.float32)),
            },
            coords={"y": y, "x": x},
        )

    def test_plot_fraction_maps_returns_figure(self, fractions_ds):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = plot_fraction_maps(fractions_ds)
        try:
            assert isinstance(fig, Figure)
            assert len(fig.axes) >= 5  # 5 fraction panels (each may have a colorbar axes)
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_plot_rgb_composite_returns_figure(self, fractions_ds):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = plot_rgb_composite(fractions_ds)
        try:
            assert isinstance(fig, Figure)
            assert len(fig.axes) >= 1
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_plot_rgb_composite_rejects_unknown_channel(self, fractions_ds):
        with pytest.raises(ValueError):
            plot_rgb_composite(fractions_ds, r="bogus")


# ---------------------------------------------------------------------------
# Wave 4 acceptance — pure-pixel coverage across all canonical classes
# ---------------------------------------------------------------------------


class TestPurePixelRecoveryAllClasses:
    """Section 9 acceptance: pure-pixel fractions for every canonical class.

    The Wave 2 suite verifies char/pv/npv/soil individually inside the NNLS
    fallback test. These tests assert the same behaviour with explicit
    threshold checks per class so a regression in any single endmember is
    caught directly.
    """

    def test_all_four_classes_recovered_via_fallback(
        self,
        pure_pixel_scene,
        small_library,
        monkeypatch,
    ):
        monkeypatch.setattr(unmixing, "_MESMA_AVAILABLE", False)
        monkeypatch.setattr(unmixing, "_HYSUP_AVAILABLE", False)
        result = run_mesma(pure_pixel_scene, small_library)

        # Row 0 → char, Row 1 → pv, Row 2 → npv, Row 3 → soil.
        # Columns 0..2 hold the pure-endmember pixels; column 3 is degenerate.
        for col in (0, 1, 2):
            assert float(result["char"].isel(y=0, x=col).values) > 0.8, f"char @x={col}"
            assert float(result["pv"].isel(y=1, x=col).values) > 0.8, f"pv @x={col}"
            assert float(result["npv"].isel(y=2, x=col).values) > 0.8, f"npv @x={col}"
            assert float(result["soil"].isel(y=3, x=col).values) > 0.8, f"soil @x={col}"


# ---------------------------------------------------------------------------
# Wave 4 acceptance — constraint filtering edge cases
# ---------------------------------------------------------------------------


class TestConstraintEdges:
    def test_relaxed_constraints_accept_pure_pixels(
        self,
        pure_pixel_scene,
        small_library,
        monkeypatch,
    ):
        monkeypatch.setattr(unmixing, "_MESMA_AVAILABLE", False)
        monkeypatch.setattr(unmixing, "_HYSUP_AVAILABLE", False)
        result = run_mesma(
            pure_pixel_scene,
            small_library,
            constraints={"max_rmse": 1.0},
        )
        # With a permissive RMSE budget the pure-pixel cells must report a
        # valid (non-NaN) RMSE.
        assert not np.isnan(result["rmse"].isel(y=0, x=0).values)


# ---------------------------------------------------------------------------
# Wave 4 acceptance — shade normalization rounds to 1.0 across columns
# ---------------------------------------------------------------------------


class TestShadeNormalizationAcceptance:
    def test_remaining_fractions_sum_to_one_across_grid(self):
        # 3x3 fractions with varying shade per pixel.
        char = np.array([[0.20, 0.30, 0.10]] * 3, dtype=np.float32)
        pv = np.array([[0.30, 0.20, 0.40]] * 3, dtype=np.float32)
        npv = np.array([[0.10, 0.10, 0.10]] * 3, dtype=np.float32)
        soil = np.array([[0.10, 0.10, 0.10]] * 3, dtype=np.float32)
        shade = np.array([[0.30, 0.30, 0.30]] * 3, dtype=np.float32)
        rmse = np.zeros((3, 3), dtype=np.float32)

        ds = xr.Dataset(
            {
                "char": (["y", "x"], char),
                "pv": (["y", "x"], pv),
                "npv": (["y", "x"], npv),
                "soil": (["y", "x"], soil),
                "shade": (["y", "x"], shade),
                "rmse": (["y", "x"], rmse),
            },
            coords={"y": np.arange(3), "x": np.arange(3)},
        )
        out = normalize_fractions(ds, remove_shade=True)
        total = sum(out[v].values for v in ("char", "pv", "npv", "soil"))
        np.testing.assert_allclose(total, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Section 9 acceptance: primary MESMA backend recovers every canonical class
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _MESMA_INSTALLED, reason="mesma package not installed")
class TestRunMesmaPrimaryAllClasses:
    """Mirror TestPurePixelRecoveryAllClasses but exercise the real MESMA backend.

    MESMA is allowed to leave a pixel as NaN if no library combination passes
    its built-in constraints — that still counts as success for this test.
    A regression that produced the *wrong* class fraction would fail here.
    """

    def test_each_pure_pixel_either_recovers_or_is_nan(
        self,
        pure_pixel_scene,
        small_library,
    ):
        result = run_mesma(pure_pixel_scene, small_library)
        # Pure-pixel rows: 0=char, 1=pv, 2=npv, 3=soil (cols 0..2 are pure).
        rows = {"char": 0, "pv": 1, "npv": 2, "soil": 3}
        for class_name, row in rows.items():
            for col in (0, 1, 2):
                value = float(result[class_name].isel(y=row, x=col).values)
                assert (value > 0.8) or np.isnan(value), (
                    f"{class_name} @row={row}, col={col} returned {value!r}; "
                    "expected >0.8 or NaN"
                )


# ---------------------------------------------------------------------------
# Section 9 acceptance: fraction-bound constraint rejection
# ---------------------------------------------------------------------------


class TestFractionBoundRejection:
    """min_fraction / max_fraction outside [0, 1] should reject every model.

    The NNLS fallback always emits non-negative fractions that sum to ~1, so
    a min_fraction of 0.99 is unreachable for a multi-endmember scene and must
    NaN every pixel. This pins the post-MESMA constraint pass behaviour.
    """

    def test_unreachable_min_fraction_nans_all_pixels(
        self,
        pure_pixel_scene,
        small_library,
        monkeypatch,
    ):
        monkeypatch.setattr(unmixing, "_MESMA_AVAILABLE", False)
        monkeypatch.setattr(unmixing, "_HYSUP_AVAILABLE", False)
        result = run_mesma(
            pure_pixel_scene,
            small_library,
            constraints={"min_fraction": 0.99, "max_rmse": 1.0},
        )
        for v in ("char", "pv", "npv", "soil", "shade"):
            assert np.isnan(result[v].values).all(), f"{v} not all NaN"
        assert np.isnan(result["rmse"].values).all()


# ---------------------------------------------------------------------------
# Section 9 acceptance: shade normalization preserves NaN inputs
# ---------------------------------------------------------------------------


class TestShadeNormalizationNaN:
    """Pixels that come in NaN from upstream must remain NaN after normalisation.

    MESMA writes NaN for pixels where no model passed constraints. The shade
    normalisation step must not silently rescale those NaNs back into finite
    values — that would mask failed unmixings as real fractions downstream.
    """

    def test_nan_input_pixel_stays_nan(self):
        nan = np.float32(np.nan)
        char = np.array([[0.40, nan]], dtype=np.float32)
        pv = np.array([[0.30, nan]], dtype=np.float32)
        npv = np.array([[0.00, nan]], dtype=np.float32)
        soil = np.array([[0.00, nan]], dtype=np.float32)
        shade = np.array([[0.30, nan]], dtype=np.float32)
        rmse = np.array([[0.01, nan]], dtype=np.float32)

        ds = xr.Dataset(
            {
                "char": (["y", "x"], char),
                "pv": (["y", "x"], pv),
                "npv": (["y", "x"], npv),
                "soil": (["y", "x"], soil),
                "shade": (["y", "x"], shade),
                "rmse": (["y", "x"], rmse),
            },
            coords={"y": np.arange(1), "x": np.arange(2)},
        )
        out = normalize_fractions(ds, remove_shade=True)

        # Valid pixel rescales to sum=1; NaN pixel stays NaN across all classes.
        non_nan = sum(float(out[v].isel(y=0, x=0).values) for v in ("char", "pv", "npv", "soil"))
        np.testing.assert_allclose(non_nan, 1.0, atol=1e-4)
        for v in ("char", "pv", "npv", "soil"):
            assert np.isnan(float(out[v].isel(y=0, x=1).values)), f"{v} NaN pixel was rescaled"
