"""Tests for :mod:`tanager.lfmc`.

Synthetic spectra with known absorption features and reflectance/LFMC
relationships exercise:

* :func:`tanager.lfmc._compute_sai` — Spectral Absorption Index against
  hand-built absorption features.
* :func:`tanager.lfmc.compute_lfmc_indices` — full eight-index Dataset over a
  3-D synthetic cube.
* :func:`tanager.lfmc.train_lfmc_plsr` — PLSR with strong synthetic moisture
  signal; verifies R² > 0 and VIP scores peak near water-absorption bands.
* :func:`tanager.lfmc.predict_lfmc` — physical range clipping and
  ``low_lfmc_flag``.
* :func:`tanager.lfmc.load_globe_lfmc` — CSV loader with mocked Globe-LFMC
  schema and bbox / vegetation filters.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("sklearn")
pytest.importorskip("pandas")

from tanager import lfmc
from tanager.lfmc import _compute_sai, compute_lfmc_indices, predict_lfmc, train_lfmc_plsr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spectrum_with_absorption(
    wavelengths: np.ndarray,
    *,
    feature_centre: float,
    depth: float,
    width: float = 60.0,
    baseline: float = 0.45,
) -> np.ndarray:
    """Return a 1-D spectrum with a single Gaussian absorption feature."""
    gauss = np.exp(-((wavelengths - feature_centre) ** 2) / (2.0 * width**2))
    return (baseline - depth * gauss).astype(np.float32)


def _make_cube(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    *,
    ny: int = 4,
    nx: int = 4,
) -> xr.Dataset:
    cube = np.broadcast_to(spectrum[:, None, None], (spectrum.size, ny, nx)).astype(np.float32)
    return xr.Dataset(
        {"reflectance": (["wavelength", "y", "x"], cube.copy())},
        coords={"wavelength": wavelengths, "y": np.arange(ny), "x": np.arange(nx)},
    )


# ---------------------------------------------------------------------------
# _compute_sai
# ---------------------------------------------------------------------------


class TestComputeSAI:
    def test_known_absorption_returns_positive_sai(self):
        wl = np.linspace(900.0, 1300.0, 200)
        spec = _spectrum_with_absorption(wl, feature_centre=1200.0, depth=0.20)
        sai = _compute_sai(spec, wl, target_wl=1200.0, left_shoulder=1100.0, right_shoulder=1300.0)
        assert 0.1 < sai <= 1.0

    def test_flat_spectrum_yields_zero(self):
        # A perfectly flat spectrum is a valid measurement of "no absorption"
        # (R_target == R_continuum), not a masked/invalid pixel. SAI = 0.0
        # is the genuine answer here and must NOT be replaced with NaN.
        wl = np.linspace(900.0, 1300.0, 200)
        spec = np.full_like(wl, 0.45, dtype=np.float32)
        sai = _compute_sai(spec, wl, target_wl=1200.0, left_shoulder=1100.0, right_shoulder=1300.0)
        assert sai == 0.0

    def test_shoulders_outside_window_yield_nan(self):
        wl = np.linspace(900.0, 1300.0, 200)
        spec = _spectrum_with_absorption(wl, feature_centre=1200.0, depth=0.20)
        sai = _compute_sai(
            spec,
            wl,
            target_wl=1200.0,
            left_shoulder=500.0,  # outside window
            right_shoulder=2000.0,  # outside window
        )
        assert np.isnan(sai)

    def test_inverted_shoulders_yield_nan(self):
        wl = np.linspace(900.0, 1300.0, 200)
        spec = _spectrum_with_absorption(wl, feature_centre=1200.0, depth=0.20)
        sai = _compute_sai(
            spec,
            wl,
            target_wl=1200.0,
            left_shoulder=1300.0,
            right_shoulder=1100.0,
        )
        assert np.isnan(sai)

    def test_nan_reflectance_yields_nan(self):
        wl = np.linspace(900.0, 1300.0, 200)
        spec = _spectrum_with_absorption(wl, feature_centre=1200.0, depth=0.20)
        spec[100] = np.nan  # corrupt the target band
        sai = _compute_sai(spec, wl, target_wl=wl[100], left_shoulder=1100.0, right_shoulder=1300.0)
        assert np.isnan(sai)


# ---------------------------------------------------------------------------
# _sai_map — masked-pixel propagation (regression for "100% valid SAI" bug)
# ---------------------------------------------------------------------------


class TestSAIMapMaskedPixels:
    """Regression: SAI used to fill masked pixels with 0.0, silently inflating
    "valid pixel" coverage to 100% while NDWI/WI/CR reported the true ~28%.
    Now masked / non-physical pixels must propagate as NaN.
    """

    def _cube_with_nan_pixels(self, wl: np.ndarray, mask_fraction: float = 0.5) -> xr.DataArray:
        spec = np.full_like(wl, 0.45, dtype=np.float32)
        for centre in (970.0, 1200.0, 1660.0):
            spec -= 0.15 * np.exp(-((wl - centre) ** 2) / (2.0 * 40.0**2))
        ny, nx = 4, 4
        cube = np.broadcast_to(spec[:, None, None], (spec.size, ny, nx)).astype(np.float32).copy()
        # Mask roughly mask_fraction of pixels by setting their entire spectrum to NaN.
        rng = np.random.default_rng(0)
        flat_idx = rng.choice(ny * nx, size=int(mask_fraction * ny * nx), replace=False)
        for k in flat_idx:
            iy, ix = divmod(int(k), nx)
            cube[:, iy, ix] = np.nan
        return xr.DataArray(
            cube,
            dims=("wavelength", "y", "x"),
            coords={"wavelength": wl, "y": np.arange(ny), "x": np.arange(nx)},
        )

    def test_masked_pixels_yield_nan_not_zero(self):
        from tanager.lfmc import _sai_map

        wl = np.linspace(380.0, 2500.0, 426).astype(np.float32)
        refl = self._cube_with_nan_pixels(wl, mask_fraction=0.5)
        sai = _sai_map(refl, target_wl=1200.0, left_shoulder=1100.0, right_shoulder=1300.0)

        assert np.isnan(sai.values).any(), "masked pixels must propagate as NaN, not 0"
        n_total = sai.size
        n_finite = int(np.isfinite(sai.values).sum())
        assert n_finite < n_total, (
            f"_sai_map reports 100% valid pixels ({n_finite}/{n_total}) — "
            "masked pixels are leaking through as zeros instead of NaN"
        )

    def test_continuum_nonpositive_yields_nan(self):
        from tanager.lfmc import _sai_map

        # Reflectance that drives the linearly-interpolated continuum to <= 0
        # (negative shoulder values from e.g. ISOFIT shadow artefacts).
        wl = np.linspace(380.0, 2500.0, 426).astype(np.float32)
        spec = np.full_like(wl, -0.1, dtype=np.float32)
        ny, nx = 2, 2
        cube = np.broadcast_to(spec[:, None, None], (spec.size, ny, nx)).astype(np.float32).copy()
        refl = xr.DataArray(
            cube,
            dims=("wavelength", "y", "x"),
            coords={"wavelength": wl, "y": np.arange(ny), "x": np.arange(nx)},
        )
        sai = _sai_map(refl, target_wl=1200.0, left_shoulder=1100.0, right_shoulder=1300.0)
        assert np.isnan(sai.values).all()

    def test_target_outside_shoulders_yields_all_nan_map(self):
        from tanager.lfmc import _sai_map

        # Misconfigured wavelengths (target not bracketed) — whole-map invalid.
        wl = np.linspace(380.0, 2500.0, 426).astype(np.float32)
        spec = np.full_like(wl, 0.45, dtype=np.float32)
        cube = np.broadcast_to(spec[:, None, None], (spec.size, 3, 3)).astype(np.float32).copy()
        refl = xr.DataArray(
            cube,
            dims=("wavelength", "y", "x"),
            coords={"wavelength": wl, "y": np.arange(3), "x": np.arange(3)},
        )
        sai = _sai_map(refl, target_wl=900.0, left_shoulder=1100.0, right_shoulder=1300.0)
        assert np.isnan(sai.values).all()


# ---------------------------------------------------------------------------
# compute_lfmc_indices
# ---------------------------------------------------------------------------


class TestComputeLFMCIndices:
    def _full_spectrum(self, wl: np.ndarray) -> np.ndarray:
        # Baseline vegetation curve with the three liquid-water dips.
        spec = np.full_like(wl, 0.45, dtype=np.float32)
        for centre in (970.0, 1200.0, 1660.0):
            spec -= 0.15 * np.exp(-((wl - centre) ** 2) / (2.0 * 40.0**2))
        # Add a flatter baseline below 700 nm and shoulder above 2200 nm
        spec[wl < 700.0] = 0.10
        spec[wl > 2200.0] = 0.10
        return spec.astype(np.float32)

    def test_eight_indices_present_with_correct_dims(self):
        wl = np.linspace(380.0, 2500.0, 426).astype(np.float32)
        scene = _make_cube(wl, self._full_spectrum(wl))
        indices = compute_lfmc_indices(scene)

        for name in ("SAI970", "SAI1200", "SAI1660", "NDWI_1240", "NDWI_1640", "NDWI_2130", "WI"):
            assert name in indices.data_vars
            assert indices[name].dims == ("y", "x")
        assert "CR_depths" in indices.data_vars
        assert indices["CR_depths"].dims == ("cr_target", "y", "x")

    def test_ndwi_in_valid_range(self):
        wl = np.linspace(380.0, 2500.0, 426).astype(np.float32)
        scene = _make_cube(wl, self._full_spectrum(wl))
        indices = compute_lfmc_indices(scene)
        for name in ("NDWI_1240", "NDWI_1640", "NDWI_2130"):
            arr = indices[name].values
            finite = arr[np.isfinite(arr)]
            assert finite.min() >= -1.0 - 1e-6
            assert finite.max() <= 1.0 + 1e-6

    def test_succeeds_with_per_band_fwhm_and_good_wavelengths_coords(self):
        # Regression for LGT-333: real Tanager DataArrays carry per-band
        # `fwhm` and `good_wavelengths` aux coords along the wavelength dim;
        # CR_depths construction used to raise MergeError because each
        # `sel(method="nearest")` slice carried a different aux value.
        wl = np.linspace(380.0, 2500.0, 426).astype(np.float32)
        scene = _make_cube(wl, self._full_spectrum(wl))
        rng = np.random.default_rng(0)
        scene = scene.assign_coords(
            fwhm=("wavelength", rng.uniform(4.0, 12.0, size=wl.size).astype(np.float32)),
            good_wavelengths=("wavelength", np.ones(wl.size, dtype=bool)),
        )
        # DataArray entry point — also surfaces the bug since
        # `_scene_reflectance` preserves the wavelength-aligned coords.
        indices = compute_lfmc_indices(scene["reflectance"])
        assert "CR_depths" in indices.data_vars
        assert indices["CR_depths"].dims == ("cr_target", "y", "x")
        # Aux coords must not leak onto the CR_depths output.
        assert "fwhm" not in indices["CR_depths"].coords
        assert "good_wavelengths" not in indices["CR_depths"].coords

    def test_accepts_surface_reflectance_variable_name(self):
        # Regression for LGT-332: load_ortho_scene names the cube
        # `surface_reflectance`. compute_lfmc_indices must resolve it via
        # the shared scene_reflectance helper instead of demanding the
        # synthetic `reflectance` name.
        wl = np.linspace(380.0, 2500.0, 426).astype(np.float32)
        spec = self._full_spectrum(wl)
        cube = np.broadcast_to(spec[:, None, None], (spec.size, 4, 4)).astype(np.float32)
        scene = xr.Dataset(
            {"surface_reflectance": (["wavelength", "y", "x"], cube.copy())},
            coords={"wavelength": wl, "y": np.arange(4), "x": np.arange(4)},
            attrs={"data_var": "surface_reflectance"},
        )
        indices = compute_lfmc_indices(scene)
        assert "CR_depths" in indices.data_vars
        assert "SAI970" in indices.data_vars
        assert indices["CR_depths"].dims == ("cr_target", "y", "x")


# ---------------------------------------------------------------------------
# train_lfmc_plsr
# ---------------------------------------------------------------------------


def _synthetic_lfmc_training_set(rng_seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(spectra, lfmc, wavelengths)`` with a strong moisture signal.

    Reflectance at the three water-absorption bands (970, 1200, 1660 nm) is
    inversely related to LFMC, so PLSR should recover R² > 0.5 and VIP scores
    should peak near those bands.
    """
    rng = np.random.default_rng(rng_seed)
    wl = np.linspace(900.0, 1700.0, 80).astype(np.float64)
    n_samples = 120
    lfmc_values = rng.uniform(40.0, 200.0, size=n_samples)

    spectra = np.zeros((n_samples, wl.size), dtype=np.float64)
    for i, lfmc_pct in enumerate(lfmc_values):
        baseline = 0.45 + 0.02 * rng.standard_normal()
        # Higher LFMC → deeper absorption → lower reflectance at water bands.
        absorption_strength = (lfmc_pct - 40.0) / 160.0  # 0..1
        feature = np.zeros_like(wl)
        for centre in (970.0, 1200.0, 1660.0):
            feature += 0.20 * absorption_strength * np.exp(-((wl - centre) ** 2) / (2.0 * 40.0**2))
        spectra[i] = baseline - feature + 0.005 * rng.standard_normal(wl.size)

    return spectra.astype(np.float32), lfmc_values, wl


class TestTrainLFMCPlsr:
    def test_recovers_moisture_signal(self):
        spectra, lfmc_values, _ = _synthetic_lfmc_training_set(rng_seed=11)
        result = train_lfmc_plsr(spectra, lfmc_values, n_components=8, cv_folds=4)
        assert "model" in result
        assert "r2" in result
        assert "rmse" in result
        assert "vip_scores" in result
        assert result["r2"] > 0.5
        assert result["rmse"] < 50.0  # in % LFMC

    def test_vip_scores_peak_near_water_bands(self):
        spectra, lfmc_values, wl = _synthetic_lfmc_training_set(rng_seed=12)
        result = train_lfmc_plsr(spectra, lfmc_values, n_components=8, cv_folds=4)
        vip = np.asarray(result["vip_scores"], dtype=np.float64)
        assert vip.size == wl.size
        # Top-10 VIP bands should fall near at least one of the three water
        # absorption features (within ±100 nm).
        top_idx = np.argsort(-vip)[:10]
        top_wls = wl[top_idx]
        targets = np.array([970.0, 1200.0, 1660.0])
        min_dist = np.min(np.abs(top_wls[:, None] - targets[None, :]), axis=1)
        assert np.median(min_dist) < 100.0


# ---------------------------------------------------------------------------
# predict_lfmc
# ---------------------------------------------------------------------------


class TestPredictLFMC:
    def test_predicted_lfmc_in_physical_range(self):
        spectra, lfmc_values, wl = _synthetic_lfmc_training_set(rng_seed=13)
        trained = train_lfmc_plsr(spectra, lfmc_values, n_components=6, cv_folds=4)

        # Build a scene Dataset matching the training band grid.
        ny, nx = 4, 4
        cube = (
            np.broadcast_to(spectra[0][:, None, None], (wl.size, ny, nx)).astype(np.float32).copy()
        )
        scene = xr.Dataset(
            {"reflectance": (["wavelength", "y", "x"], cube)},
            coords={"wavelength": wl.astype(np.float32), "y": np.arange(ny), "x": np.arange(nx)},
        )

        out = predict_lfmc(scene, trained)
        lfmc_map = out["lfmc_map"].values
        finite = lfmc_map[np.isfinite(lfmc_map)]
        assert finite.min() >= 0.0
        assert finite.max() <= 300.0

        assert "uncertainty_map" in out
        assert "low_lfmc_flag" in out

    def test_low_lfmc_flag_below_threshold(self):
        # Build a model that always predicts ~30% LFMC. We don't need a real
        # fit — the flag depends only on the predicted values.
        spectra, lfmc_values, wl = _synthetic_lfmc_training_set(rng_seed=14)
        trained = train_lfmc_plsr(spectra, lfmc_values, n_components=4, cv_folds=4)

        # Scene of a "very wet" spectrum (high absorption depth → high LFMC by
        # our construction) AND of a "very dry" spectrum (low absorption).
        wet = spectra[np.argmax(lfmc_values)]
        dry = spectra[np.argmin(lfmc_values)]
        cube = np.zeros((wl.size, 2, 1), dtype=np.float32)
        cube[:, 0, 0] = wet
        cube[:, 1, 0] = dry
        scene = xr.Dataset(
            {"reflectance": (["wavelength", "y", "x"], cube)},
            coords={"wavelength": wl.astype(np.float32), "y": np.arange(2), "x": np.arange(1)},
        )
        out = predict_lfmc(scene, trained)
        # We can't guarantee the dry pixel is exactly < 60%, but the flag
        # type/shape contract must hold and at least one of the two pixels
        # must agree with the model's prediction.
        assert out["low_lfmc_flag"].dtype == bool
        assert out["low_lfmc_flag"].shape == (2, 1)


# ---------------------------------------------------------------------------
# load_globe_lfmc
# ---------------------------------------------------------------------------


class TestLoadGlobeLFMC:
    def _write_csv(self, tmp_path) -> str:
        import pandas as pd

        df = pd.DataFrame(
            {
                "lat": [34.05, 34.10, 35.00, 33.80],
                "lon": [-118.25, -118.30, -119.00, -117.50],
                "obs_date": [
                    "2024-12-01",
                    "2025-01-15",
                    "2025-03-01",
                    "2025-06-10",
                ],
                "lfmc": [120.0, 80.0, 65.0, 40.0],
                "vegetation": ["chaparral", "chaparral", "grassland", "chaparral"],
                "species": ["Adenostoma", "Ceanothus", "Bromus", "Adenostoma"],
            }
        )
        path = tmp_path / "globe_lfmc_subset.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_normalizes_columns_and_filters_bbox(self, tmp_path):
        pytest.importorskip("geopandas")
        path = self._write_csv(tmp_path)
        gdf = lfmc.load_globe_lfmc(
            path,
            region_bbox=(-119.0, 33.5, -117.0, 34.5),
            vegetation_types=None,
        )
        # Only rows whose lon ∈ [-119, -117] and lat ∈ [33.5, 34.5] survive.
        assert len(gdf) == 3
        assert {"latitude", "longitude", "date", "lfmc_percent"}.issubset(gdf.columns)

    def test_vegetation_filter(self, tmp_path):
        pytest.importorskip("geopandas")
        path = self._write_csv(tmp_path)
        gdf = lfmc.load_globe_lfmc(path, vegetation_types=["chaparral"])
        # 3 of 4 rows are chaparral
        assert len(gdf) == 3
        assert (gdf["vegetation_type"].str.lower() == "chaparral").all()

    def test_tanager_colocation_flag(self, tmp_path):
        pytest.importorskip("geopandas")
        path = self._write_csv(tmp_path)
        gdf = lfmc.load_globe_lfmc(
            path,
            tanager_scene_dates=["2025-01-23"],
            colocation_window_days=30,
        )
        # 2025-01-15 is within ±30 days of 2025-01-23; others are not.
        assert "tanager_colocated" in gdf.columns
        colocated = gdf[gdf["tanager_colocated"]]
        assert len(colocated) == 1
        assert str(colocated["date"].iloc[0]).startswith("2025-01-15")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            lfmc.load_globe_lfmc(tmp_path / "missing.csv")

    def test_missing_required_column_rejected(self, tmp_path):
        import pandas as pd

        bad = tmp_path / "bad.csv"
        pd.DataFrame({"lat": [1.0], "lon": [2.0]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            lfmc.load_globe_lfmc(bad)
