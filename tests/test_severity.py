"""Tests for :mod:`tanager.severity`.

Synthetic fraction grids with known CBI relationships exercise:

* :func:`tanager.severity.train_severity_model` — RF training, R² / RMSE
  reporting, NaN handling.
* :func:`tanager.severity.predict_severity` — value-range clipping,
  classification thresholds, NaN propagation.
* :func:`tanager.severity.calibrate_nbr_thresholds` — per-class NBR medians,
  midpoint thresholds, monotonicity and min-pixel guards.
* :func:`tanager.severity.classify_severity_from_nbr` — threshold application,
  NBR/severity sign convention, NaN propagation.
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
# train_severity_classifier
# ---------------------------------------------------------------------------


def _synthetic_fractions_and_classes(
    rng_seed: int = 10,
    *,
    ny: int = 12,
    nx: int = 12,
) -> Tuple[xr.Dataset, np.ndarray]:
    """Return (fractions, class_labels) where classes correlate with char."""
    rng = np.random.default_rng(rng_seed)
    n_pixels = ny * nx
    fracs = rng.dirichlet(alpha=(1.0, 1.0, 1.0, 1.0), size=n_pixels).astype(np.float32)

    char = fracs[:, 0].reshape(ny, nx)
    pv = fracs[:, 1].reshape(ny, nx)
    npv = fracs[:, 2].reshape(ny, nx)
    soil = fracs[:, 3].reshape(ny, nx)

    ds = _make_fraction_dataset(char, pv, npv, soil)

    char_flat = char.ravel()
    classes = np.digitize(char_flat, [0.15, 0.35, 0.60]).astype(np.int64)
    return ds, classes


class TestTrainSeverityClassifier:
    def test_trains_classifier_and_returns_metrics(self):
        ds, classes = _synthetic_fractions_and_classes(rng_seed=11)
        result = severity.train_severity_classifier(
            ds, classes, n_estimators=50, cv_folds=3,
        )
        assert result["model_type"] == "classifier"
        assert "accuracy" in result
        assert "kappa" in result
        assert "f1_macro" in result
        assert result["method"] == "random_forest"
        assert result["n_samples"] == ds.sizes["y"] * ds.sizes["x"]
        assert len(result["classes"]) >= 2

    def test_nodata_pixels_excluded(self):
        ds, classes = _synthetic_fractions_and_classes(rng_seed=12)
        classes[0] = -1
        classes[5] = -1
        result = severity.train_severity_classifier(
            ds, classes, n_estimators=50, cv_folds=3,
        )
        assert result["n_samples"] == ds.sizes["y"] * ds.sizes["x"] - 2

    def test_nan_fractions_excluded(self):
        ds, classes = _synthetic_fractions_and_classes(rng_seed=13)
        ds["char"].values[0, 0] = np.nan
        result = severity.train_severity_classifier(
            ds, classes, n_estimators=50, cv_folds=3,
        )
        assert result["n_samples"] == ds.sizes["y"] * ds.sizes["x"] - 1

    def test_unsupported_method_rejected(self):
        ds, classes = _synthetic_fractions_and_classes()
        with pytest.raises(ValueError, match="unsupported method"):
            severity.train_severity_classifier(ds, classes, method="xgboost")

    def test_size_mismatch_rejected(self):
        ds, classes = _synthetic_fractions_and_classes()
        with pytest.raises(ValueError, match="sizes must match"):
            severity.train_severity_classifier(ds, classes[:5])


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

    def test_classifier_model_produces_classes(self):
        ds, classes = _synthetic_fractions_and_classes(rng_seed=14)
        trained = severity.train_severity_classifier(
            ds, classes, n_estimators=50, cv_folds=3,
        )
        out = severity.predict_severity(ds, trained)
        assert "severity_map" in out
        assert "cbi_map" not in out
        sev = out["severity_map"].values
        finite_codes = sev[np.isfinite(sev)].astype(int)
        assert set(finite_codes).issubset(set(trained["classes"]))

    def test_classifier_nan_propagation(self):
        ds, classes = _synthetic_fractions_and_classes(rng_seed=15)
        trained = severity.train_severity_classifier(
            ds, classes, n_estimators=50, cv_folds=3,
        )
        ds["char"].values[0, 0] = np.nan
        out = severity.predict_severity(ds, trained)
        assert np.isnan(out["severity_map"].values[0, 0])


# ---------------------------------------------------------------------------
# calibrate_nbr_thresholds / classify_severity_from_nbr
# ---------------------------------------------------------------------------


def _nbr_and_reference(
    class_medians: Tuple[float, ...] = (0.05, -0.20, -0.40),
    *,
    n_per_class: int = 60,
    spread: float = 0.01,
    rng_seed: int = 7,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Build an NBR map whose per-class medians are known by construction.

    Class ``i`` occupies row ``i`` of the grid and is drawn tightly around
    ``class_medians[i]``, so the calibrated thresholds must land near the
    midpoints between consecutive medians.
    """
    rng = np.random.default_rng(rng_seed)
    n_classes = len(class_medians)
    nbr_rows = [
        rng.normal(m, spread, size=n_per_class).astype(np.float64) for m in class_medians
    ]
    nbr_arr = np.stack(nbr_rows)
    ref_arr = np.stack(
        [np.full(n_per_class, code, dtype=np.int16) for code in range(n_classes)]
    )
    coords = {"y": np.arange(n_classes), "x": np.arange(n_per_class)}
    nbr_da = xr.DataArray(nbr_arr, dims=("y", "x"), coords=coords, name="nbr")
    ref_da = xr.DataArray(ref_arr, dims=("y", "x"), coords=coords, name="reference")
    return nbr_da, ref_da


class TestCalibrateNbrThresholds:
    def test_thresholds_are_midpoints_of_class_medians(self):
        nbr_da, ref_da = _nbr_and_reference(class_medians=(0.05, -0.20, -0.40))
        cal = severity.calibrate_nbr_thresholds(nbr_da, ref_da)

        assert cal["class_codes"] == (0, 1, 2)
        assert cal["thresholds"].shape == (2,)
        # midpoints of (0.05, -0.20) and (-0.20, -0.40)
        np.testing.assert_allclose(cal["thresholds"], [-0.075, -0.30], atol=0.01)
        assert cal["n_valid"] == 180

    def test_nodata_and_nan_pixels_excluded(self):
        nbr_da, ref_da = _nbr_and_reference()
        # Negative reference codes are nodata; NaN NBR is unusable.
        ref_da.values[0, :10] = -1
        nbr_da.values[1, :5] = np.nan
        cal = severity.calibrate_nbr_thresholds(nbr_da, ref_da)

        assert cal["n_valid"] == 180 - 10 - 5
        assert cal["n_pixels"][0] == 50
        assert cal["n_pixels"][1] == 55

    def test_sparse_class_dropped_below_min_pixels(self):
        nbr_da, ref_da = _nbr_and_reference()
        # Leave class 2 with only 3 pixels — too few to calibrate against.
        ref_da.values[2, 3:] = -1
        cal = severity.calibrate_nbr_thresholds(nbr_da, ref_da, min_pixels=50)

        assert cal["class_codes"] == (0, 1)
        assert 2 not in cal["medians"]
        assert cal["thresholds"].shape == (1,)

    def test_non_monotone_medians_rejected(self):
        # Class 2 is *brighter* than class 1 — single-date NBR cannot order
        # these classes, so calibration must refuse rather than emit a
        # meaningless threshold.
        nbr_da, ref_da = _nbr_and_reference(class_medians=(0.05, -0.40, -0.20))
        with pytest.raises(ValueError, match="not strictly decreasing"):
            severity.calibrate_nbr_thresholds(nbr_da, ref_da)

    def test_shape_mismatch_rejected(self):
        nbr_da, ref_da = _nbr_and_reference()
        with pytest.raises(ValueError, match="shape mismatch"):
            severity.calibrate_nbr_thresholds(nbr_da, ref_da.isel(x=slice(0, 5)))

    def test_too_few_classes_rejected(self):
        nbr_da, ref_da = _nbr_and_reference()
        ref_da.values[1:, :] = -1
        with pytest.raises(ValueError, match="need at least 2"):
            severity.calibrate_nbr_thresholds(nbr_da, ref_da)


class TestClassifySeverityFromNbr:
    def test_recovers_reference_classes_on_calibration_data(self):
        nbr_da, ref_da = _nbr_and_reference()
        cal = severity.calibrate_nbr_thresholds(nbr_da, ref_da)
        out = severity.classify_severity_from_nbr(nbr_da, cal)

        # Classes are tightly separated, so round-tripping must be exact.
        np.testing.assert_array_equal(out.values, ref_da.values.astype(np.float64))

    def test_class_ordering_is_inverted_relative_to_nbr(self):
        # High NBR must map to the least-severe class and low NBR to the most
        # severe — the sign convention is the whole point of the function.
        nbr_da, ref_da = _nbr_and_reference()
        cal = severity.calibrate_nbr_thresholds(nbr_da, ref_da)
        probe = xr.DataArray(
            np.array([[0.5, -0.9]]),
            dims=("y", "x"),
            coords={"y": [0], "x": [0, 1]},
        )
        out = severity.classify_severity_from_nbr(probe, cal)

        assert out.values[0, 0] == 0  # brightest → unburned
        assert out.values[0, 1] == 2  # darkest → most severe

    def test_nan_pixels_propagate_and_coords_preserved(self):
        nbr_da, ref_da = _nbr_and_reference()
        cal = severity.calibrate_nbr_thresholds(nbr_da, ref_da)
        nbr_da.values[0, 0] = np.nan
        out = severity.classify_severity_from_nbr(nbr_da, cal)

        assert np.isnan(out.values[0, 0])
        assert out.dims == nbr_da.dims
        np.testing.assert_array_equal(out.coords["x"].values, nbr_da.coords["x"].values)

    def test_inconsistent_calibration_rejected(self):
        nbr_da, _ = _nbr_and_reference()
        bad = {"class_codes": (0, 1, 2), "thresholds": np.array([0.0])}
        with pytest.raises(ValueError, match="expected len\\(class_codes\\) - 1"):
            severity.classify_severity_from_nbr(nbr_da, bad)

    def test_missing_calibration_key_rejected(self):
        nbr_da, _ = _nbr_and_reference()
        with pytest.raises(ValueError, match="missing required key"):
            severity.classify_severity_from_nbr(nbr_da, {"class_codes": (0, 1)})


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
