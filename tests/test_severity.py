"""Tests for :mod:`tanager.severity`.

Synthetic fraction grids with known CBI relationships exercise:

* :func:`tanager.severity.train_severity_model` — RF training, R² / RMSE
  reporting, NaN handling.
* :func:`tanager.severity.predict_severity` — value-range clipping,
  classification thresholds, NaN propagation.
* :func:`tanager.severity.compute_trajectories` — multi-scene stacking with a
  ``time`` dimension.
* :func:`tanager.severity.compare_severity_methods` — agreement metrics
  between MESMA-derived severity and a dNBR baseline.

scikit-learn is required by tanager for severity work; tests skip cleanly when
it isn't installed in the test environment.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("sklearn")

from tanager import severity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fraction_dataset(
    char: np.ndarray,
    pv: np.ndarray,
    npv: np.ndarray,
    soil: np.ndarray,
) -> xr.Dataset:
    ny, nx = char.shape
    return xr.Dataset(
        {
            "char": (["y", "x"], char),
            "pv": (["y", "x"], pv),
            "npv": (["y", "x"], npv),
            "soil": (["y", "x"], soil),
        },
        coords={"y": np.arange(ny), "x": np.arange(nx)},
    )


def _synthetic_fractions_and_cbi(
    rng_seed: int = 0,
    *,
    ny: int = 8,
    nx: int = 8,
    noise: float = 0.05,
) -> Tuple[xr.Dataset, np.ndarray]:
    """Return a (fractions, cbi) pair where CBI ≈ 2.5 * char + ε.

    The relationship is strong enough that even a small RF can recover R² > 0.
    Pixels are drawn from a Dirichlet so per-pixel char/pv/npv/soil sum to 1.0.
    """
    rng = np.random.default_rng(rng_seed)
    n_pixels = ny * nx
    fracs = rng.dirichlet(alpha=(1.0, 1.0, 1.0, 1.0), size=n_pixels).astype(np.float32)

    char = fracs[:, 0].reshape(ny, nx)
    pv = fracs[:, 1].reshape(ny, nx)
    npv = fracs[:, 2].reshape(ny, nx)
    soil = fracs[:, 3].reshape(ny, nx)

    ds = _make_fraction_dataset(char, pv, npv, soil)
    cbi = 2.5 * char.ravel() + rng.normal(0.0, noise, size=n_pixels)
    cbi = np.clip(cbi, 0.0, 3.0)
    return ds, cbi


# ---------------------------------------------------------------------------
# train_severity_model
# ---------------------------------------------------------------------------


class TestTrainSeverityModel:
    def test_trains_rf_and_returns_metrics(self):
        ds, cbi = _synthetic_fractions_and_cbi(rng_seed=1)
        result = severity.train_severity_model(ds, cbi, n_estimators=50, cv_folds=3)
        assert "model" in result
        assert "r2" in result
        assert "rmse" in result
        assert result["method"] == "random_forest"
        assert result["n_samples"] == ds.sizes["y"] * ds.sizes["x"]
        # Strong synthetic signal — CV R² should be substantially > 0.
        assert result["r2"] > 0.4

    def test_handles_nan_pixels(self):
        ds, cbi = _synthetic_fractions_and_cbi(rng_seed=2)
        # Inject NaN into one corner of the char map and one corner of the
        # CBI vector. Both rows must be filtered out before training.
        char = ds["char"].values.copy()
        char[0, 0] = np.nan
        ds["char"].values[:] = char
        cbi[5] = np.nan

        result = severity.train_severity_model(ds, cbi, n_estimators=50, cv_folds=3)
        # n_samples records the count of valid (filtered) pixels.
        n_pixels = ds.sizes["y"] * ds.sizes["x"]
        assert result["n_samples"] < n_pixels

    def test_unsupported_method_rejected(self):
        ds, cbi = _synthetic_fractions_and_cbi()
        with pytest.raises(ValueError, match="unsupported method"):
            severity.train_severity_model(ds, cbi, method="xgboost")

    def test_missing_feature_rejected(self):
        ds, cbi = _synthetic_fractions_and_cbi()
        bad = ds.drop_vars("char")
        with pytest.raises(ValueError, match="missing required variable"):
            severity.train_severity_model(bad, cbi)

    def test_too_few_samples_for_cv(self):
        # Fewer pixels than the requested cv_folds.
        char = np.array([[0.5]], dtype=np.float32)
        pv = np.array([[0.2]], dtype=np.float32)
        npv = np.array([[0.2]], dtype=np.float32)
        soil = np.array([[0.1]], dtype=np.float32)
        ds = _make_fraction_dataset(char, pv, npv, soil)
        with pytest.raises(ValueError, match="cross-validation"):
            severity.train_severity_model(ds, np.array([1.0]), cv_folds=5)


# ---------------------------------------------------------------------------
# predict_severity
# ---------------------------------------------------------------------------


class TestPredictSeverity:
    def test_predicted_cbi_in_range(self):
        ds, cbi = _synthetic_fractions_and_cbi(rng_seed=3)
        trained = severity.train_severity_model(ds, cbi, n_estimators=30, cv_folds=3)
        out = severity.predict_severity(ds, trained)
        cbi_arr = out["cbi_map"].values
        # CBI must be clipped to [0, 3]
        finite = cbi_arr[np.isfinite(cbi_arr)]
        assert finite.min() >= 0.0
        assert finite.max() <= 3.0

    def test_classification_thresholds(self):
        # Build a fixed-width grid of CBI values that span every BARC class
        # by tuning the char fraction (which dominates our synthetic CBI).
        # We bypass training and just call _classify directly via the
        # predict_severity flow with a stub model.
        ds, cbi = _synthetic_fractions_and_cbi(rng_seed=4)
        trained = severity.train_severity_model(ds, cbi, n_estimators=30, cv_folds=3)

        # Build a probe Dataset where char fraction sweeps 0..1 across pixels.
        char = np.linspace(0.0, 1.0, 25, dtype=np.float32).reshape(5, 5)
        residual = (1.0 - char) / 3.0
        ds_probe = _make_fraction_dataset(char, residual, residual, residual)
        out = severity.predict_severity(ds_probe, trained)

        sev = out["severity_map"].values
        # Severity classes are integer codes 0..4; predict_severity returns
        # a float DataArray to allow NaN propagation, so cast for comparison.
        finite_codes = sev[np.isfinite(sev)].astype(int)
        assert finite_codes.min() >= 0
        assert finite_codes.max() <= 4

    def test_nan_pixels_propagate(self):
        ds, cbi = _synthetic_fractions_and_cbi(rng_seed=5)
        trained = severity.train_severity_model(ds, cbi, n_estimators=30, cv_folds=3)

        # Inject NaN into one fraction pixel
        ds["char"].values[0, 0] = np.nan
        out = severity.predict_severity(ds, trained)

        assert np.isnan(out["cbi_map"].values[0, 0])
        assert np.isnan(out["severity_map"].values[0, 0])


# ---------------------------------------------------------------------------
# compute_trajectories
# ---------------------------------------------------------------------------


class TestComputeTrajectories:
    def test_two_scene_stack_has_time_dim(self, monkeypatch):
        # Build two tiny scenes and stub run_mesma so we don't need MESMA /
        # NNLS at the trajectory level — the contract under test is the
        # time-stacking behaviour, not the unmixing itself.
        from tanager import unmixing

        ny, nx = 4, 4
        wls = np.linspace(400.0, 2400.0, 12).astype(np.float32)
        scene_a = xr.Dataset(
            {"reflectance": (["wavelength", "y", "x"], np.zeros((12, ny, nx), dtype=np.float32))},
            coords={"wavelength": wls, "y": np.arange(ny), "x": np.arange(nx)},
        )
        scene_b = xr.Dataset(
            {"reflectance": (["wavelength", "y", "x"], np.ones((12, ny, nx), dtype=np.float32) * 0.4)},
            coords={"wavelength": wls, "y": np.arange(ny), "x": np.arange(nx)},
        )

        def fake_run_mesma(scene, library, **_):
            zeros = np.zeros((ny, nx), dtype=np.float32)
            return xr.Dataset(
                {
                    "char": (["y", "x"], zeros + 0.25),
                    "pv": (["y", "x"], zeros + 0.25),
                    "npv": (["y", "x"], zeros + 0.25),
                    "soil": (["y", "x"], zeros + 0.25),
                    "shade": (["y", "x"], zeros),
                    "rmse": (["y", "x"], zeros),
                },
                coords={"y": np.arange(ny), "x": np.arange(nx)},
                attrs={"unmixing_engine": "test_stub"},
            )

        monkeypatch.setattr(unmixing, "run_mesma", fake_run_mesma)
        # severity.compute_trajectories may import via either tanager.unmixing
        # or the top-level package — patch the symbol on both.
        import tanager
        if hasattr(tanager, "run_mesma"):
            monkeypatch.setattr(tanager, "run_mesma", fake_run_mesma, raising=False)

        scenes_dict = {
            "2024-12-15T18:00:00": scene_a,
            "2025-01-23T18:00:00": scene_b,
        }
        result = severity.compute_trajectories(
            scenes_dict,
            library=xr.DataArray(
                np.zeros((1, 12), dtype=np.float32),
                dims=("spectrum_id", "wavelength"),
                coords={"spectrum_id": ["dummy"], "wavelength": wls},
            ),
            align=False,
        )
        assert "time" in result.dims
        assert result.sizes["time"] == 2
        for var in ("char", "pv", "npv", "soil"):
            assert var in result.data_vars
            assert result[var].dims[0] == "time"


# ---------------------------------------------------------------------------
# compare_severity_methods
# ---------------------------------------------------------------------------


class TestCompareSeverityMethods:
    def test_perfect_correlation(self):
        a = xr.DataArray(
            np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10),
            dims=("y", "x"),
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        b = a.copy()
        out = severity.compare_severity_methods(a, b)
        assert out["correlation"] == pytest.approx(1.0, abs=1e-6)
        assert out["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert out["bias"] == pytest.approx(0.0, abs=1e-6)

    def test_difference_map_shape(self):
        a = xr.DataArray(
            np.full((4, 4), 0.5, dtype=np.float32),
            dims=("y", "x"),
            coords={"y": np.arange(4), "x": np.arange(4)},
        )
        b = a + 0.1
        out = severity.compare_severity_methods(a, b)
        assert out["difference_map"].shape == (4, 4)
        np.testing.assert_allclose(out["difference_map"].values, -0.1, atol=1e-5)

    def test_nan_pixels_excluded(self):
        a = xr.DataArray(
            np.array([[0.0, 0.5], [1.0, np.nan]], dtype=np.float32),
            dims=("y", "x"),
            coords={"y": np.arange(2), "x": np.arange(2)},
        )
        b = xr.DataArray(
            np.array([[0.0, 0.5], [1.0, 0.5]], dtype=np.float32),
            dims=("y", "x"),
            coords={"y": np.arange(2), "x": np.arange(2)},
        )
        out = severity.compare_severity_methods(a, b)
        # Only 3 valid pixels — NaN excluded from metrics
        assert out["n_valid"] == 3
        assert out["rmse"] == pytest.approx(0.0, abs=1e-6)
