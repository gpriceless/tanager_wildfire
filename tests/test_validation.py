"""Tests for :mod:`tanager.validation`.

Synthetic GeoTIFFs and DataArrays exercise:

* :func:`tanager.validation.compute_accuracy` — continuous (R²/RMSE/MAE/bias)
  and classified (accuracy/Cohen's kappa/F1/confusion matrix) paths against
  hand-calculated values.
* :func:`tanager.validation.compare_sensors` — Tanager-vs-reference comparison
  format, improvement ratios, and pandas comparison table.
* :func:`tanager.validation.load_aviris3_reference` — GeoTIFF schema mapping
  and 3-m → 30-m spatial aggregation.
* :func:`tanager.validation.load_aviris3_reflectance` — AVIRIS-3 L2A
  reflectance NetCDF (284-band cube) loading and schema conversion.
* :func:`tanager.validation.cross_validate_aviris3` — cross-sensor MESMA
  validation pipeline.
* :func:`tanager.validation.load_barc_reference` — BARC code remapping,
  nodata handling, and target-grid alignment.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("sklearn")

from tanager import validation

# ---------------------------------------------------------------------------
# compute_accuracy — continuous
# ---------------------------------------------------------------------------


class TestComputeAccuracyContinuous:
    def test_perfect_prediction_yields_r2_one(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        obs = pred.copy()
        result = validation.compute_accuracy(pred, obs, "continuous")
        assert result["r2"] == pytest.approx(1.0)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["mae"] == pytest.approx(0.0)
        assert result["bias"] == pytest.approx(0.0)
        assert result["spearman"] == pytest.approx(1.0)
        assert result["n_valid"] == 4

    def test_constant_offset_bias_only(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        pred = obs + 0.5  # constant +0.5 bias
        result = validation.compute_accuracy(pred, obs, "continuous")
        assert result["bias"] == pytest.approx(0.5)
        assert result["mae"] == pytest.approx(0.5)
        assert result["rmse"] == pytest.approx(0.5)

    def test_nan_pairs_excluded(self):
        pred = np.array([1.0, 2.0, np.nan, 4.0])
        obs = np.array([1.0, np.nan, 3.0, 4.0])
        result = validation.compute_accuracy(pred, obs, "continuous")
        # Only the (1.0, 1.0) and (4.0, 4.0) pairs survive.
        assert result["n_valid"] == 2
        assert result["r2"] == pytest.approx(1.0)

    def test_xarray_input_accepted(self):
        pred = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=("idx",))
        obs = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=("idx",))
        result = validation.compute_accuracy(pred, obs, "continuous")
        assert result["r2"] == pytest.approx(1.0)

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            validation.compute_accuracy(
                np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), "continuous",
            )

    def test_all_nan_rejected(self):
        with pytest.raises(ValueError, match="No valid"):
            validation.compute_accuracy(
                np.array([np.nan, np.nan]),
                np.array([1.0, 2.0]),
                "continuous",
            )


# ---------------------------------------------------------------------------
# compute_accuracy — classified
# ---------------------------------------------------------------------------


class TestComputeAccuracyClassified:
    def test_perfect_classification(self):
        pred = np.array([0, 1, 2, 3, 4])
        obs = pred.copy()
        result = validation.compute_accuracy(pred, obs, "classified")
        assert result["accuracy"] == 1.0
        assert result["kappa"] == 1.0
        assert result["f1_macro"] == 1.0
        np.testing.assert_array_equal(result["confusion_matrix"], np.eye(5, dtype=np.int64))

    def test_completely_wrong_classification_low_kappa(self):
        # Two classes — predict every "0" as "1" and vice versa.
        pred = np.array([1, 0, 1, 0, 1, 0])
        obs = np.array([0, 1, 0, 1, 0, 1])
        result = validation.compute_accuracy(pred, obs, "classified")
        assert result["accuracy"] == 0.0
        assert result["kappa"] < 0.0  # systematic disagreement → negative kappa

    def test_nodata_excluded(self):
        pred = np.array([0, 1, 2, -1, 4])
        obs = np.array([0, 1, 2, 3, 4])
        result = validation.compute_accuracy(pred, obs, "classified", nodata=-1)
        assert result["n_valid"] == 4
        assert result["accuracy"] == 1.0

    def test_unknown_metric_type_rejected(self):
        with pytest.raises(ValueError, match="Unknown metric_type"):
            validation.compute_accuracy(np.array([1.0]), np.array([1.0]), "bogus")


# ---------------------------------------------------------------------------
# compare_sensors
# ---------------------------------------------------------------------------


class TestCompareSensors:
    def test_continuous_improvement_table(self):
        pd = pytest.importorskip("pandas")
        ground = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tanager = np.array([1.05, 2.0, 2.95, 4.05, 5.0])
        emit = np.array([1.5, 2.5, 2.5, 4.5, 5.5])

        result = validation.compare_sensors(
            tanager, emit, ground, sensor_name="emit", metric_type="continuous",
        )
        # Tanager should beat EMIT on every continuous metric here.
        improvements = result["improvement_ratios"]
        assert improvements["r2_improvement"] > 0
        assert improvements["rmse_reduction_pct"] > 0
        assert improvements["mae_reduction_pct"] > 0

        table = result["comparison_table"]
        assert isinstance(table, pd.DataFrame)
        # Columns: metric, tanager, emit, delta
        for col in ("metric", "tanager", "emit", "delta"):
            assert col in table.columns
        # Sanity: every "delta" row equals tanager - sensor
        np.testing.assert_allclose(
            table["delta"].values,
            (table["tanager"] - table["emit"]).values,
        )

    def test_classified_comparison(self):
        ground = np.array([0, 1, 2, 3, 4])
        tanager = np.array([0, 1, 2, 3, 4])  # perfect
        prisma = np.array([0, 1, 2, 2, 4])  # 1 mismatch

        result = validation.compare_sensors(
            tanager, prisma, ground, sensor_name="prisma", metric_type="classified",
        )
        assert result["improvement_ratios"]["accuracy_gain"] == pytest.approx(0.2)
        assert result["improvement_ratios"]["kappa_gain"] > 0


# ---------------------------------------------------------------------------
# simulate_sensor — spectral degradation onto reference sensor grids
# ---------------------------------------------------------------------------


class TestSimulateSensor:
    """Tests for :func:`tanager.validation.simulate_sensor`.

    Exercises EMIT/PRISMA/Sentinel-2 simulations on the synthetic 426-band
    Tanager fixture, plus DataArray vs Dataset handling, FWHM broadcasting,
    coordinate preservation, and attribute provenance.
    """

    @staticmethod
    def _emit_centers() -> np.ndarray:
        # EMIT_SENSOR: 285 bands, 381-2493 nm, ~7.4 nm spacing.
        return np.linspace(381.0, 2493.0, 285)

    @staticmethod
    def _prisma_centers() -> np.ndarray:
        # PRISMA_SENSOR: 239 bands, 400-2505 nm, ~12 nm spacing.
        return np.linspace(400.0, 2505.0, 239)

    @staticmethod
    def _sentinel2_centers() -> np.ndarray:
        # SENTINEL2_BANDS: 10 bands relevant to vegetation / burn analysis.
        return np.array(
            [490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190],
            dtype=np.float64,
        )

    @staticmethod
    def _sentinel2_fwhms() -> np.ndarray:
        return np.array(
            [65, 35, 30, 15, 15, 20, 115, 20, 90, 180], dtype=np.float64,
        )

    def test_output_dimensions_emit(self, synthetic_tanager_dataset):
        target = self._emit_centers()
        out = validation.simulate_sensor(
            synthetic_tanager_dataset["reflectance"], target, 8.5, "EMIT",
        )
        assert out.sizes["wavelength"] == 285
        assert out.sizes["y"] == synthetic_tanager_dataset.sizes["y"]
        assert out.sizes["x"] == synthetic_tanager_dataset.sizes["x"]

    def test_output_dimensions_prisma(self, synthetic_tanager_dataset):
        target = self._prisma_centers()
        out = validation.simulate_sensor(
            synthetic_tanager_dataset["reflectance"], target, 12.0, "PRISMA",
        )
        assert out.sizes["wavelength"] == 239

    def test_output_dimensions_sentinel2(self, synthetic_tanager_dataset):
        target = self._sentinel2_centers()
        fwhms = self._sentinel2_fwhms()
        out = validation.simulate_sensor(
            synthetic_tanager_dataset["reflectance"], target, fwhms, "Sentinel-2",
        )
        assert out.sizes["wavelength"] == 10

    def test_reflectance_bounds(self, synthetic_tanager_dataset):
        target = self._emit_centers()
        out = validation.simulate_sensor(
            synthetic_tanager_dataset["reflectance"], target, 8.5, "EMIT",
        )
        vals = out.values
        assert not np.isnan(vals).any()
        assert float(vals.min()) >= 0.0
        assert float(vals.max()) <= 1.0

    def test_fwhm_scalar_broadcast(self, synthetic_tanager_dataset):
        target = self._emit_centers()
        out = validation.simulate_sensor(
            synthetic_tanager_dataset["reflectance"], target, 8.5, "EMIT",
        )
        assert out.attrs["target_fwhm_nm"] == pytest.approx(8.5)

    def test_fwhm_array(self, synthetic_tanager_dataset):
        target = self._emit_centers()
        per_band = np.full(target.shape, 7.4)
        per_band[0] = 6.0
        per_band[-1] = 9.0
        out = validation.simulate_sensor(
            synthetic_tanager_dataset["reflectance"], target, per_band, "EMIT",
        )
        attr = out.attrs["target_fwhm_nm"]
        assert isinstance(attr, tuple)
        assert attr[0] == pytest.approx(6.0)
        assert attr[1] == pytest.approx(9.0)

    def test_spatial_coords_preserved(self, synthetic_tanager_dataset):
        target = self._sentinel2_centers()
        out = validation.simulate_sensor(
            synthetic_tanager_dataset["reflectance"],
            target,
            self._sentinel2_fwhms(),
            "Sentinel-2",
        )
        np.testing.assert_array_equal(
            out.coords["y"].values,
            synthetic_tanager_dataset.coords["y"].values,
        )
        np.testing.assert_array_equal(
            out.coords["x"].values,
            synthetic_tanager_dataset.coords["x"].values,
        )

    def test_wavelength_coord_replaced(self, synthetic_tanager_dataset):
        target = self._sentinel2_centers()
        out = validation.simulate_sensor(
            synthetic_tanager_dataset["reflectance"],
            target,
            self._sentinel2_fwhms(),
            "Sentinel-2",
        )
        np.testing.assert_allclose(
            out.coords["wavelength"].values,
            target.astype(np.float32),
        )

    def test_dataset_handling(self, synthetic_tanager_dataset):
        # Add a non-wavelength variable that should pass through unchanged.
        ds = synthetic_tanager_dataset.copy()
        ny = ds.sizes["y"]
        nx = ds.sizes["x"]
        ds["mask"] = (("y", "x"), np.ones((ny, nx), dtype=np.uint8))

        target = self._sentinel2_centers()
        out = validation.simulate_sensor(
            ds, target, self._sentinel2_fwhms(), "Sentinel-2",
        )

        assert isinstance(out, xr.Dataset)
        assert out.sizes["wavelength"] == 10
        assert out["reflectance"].sizes["wavelength"] == 10
        # mask has no wavelength dim and must be carried through unchanged.
        assert "wavelength" not in out["mask"].dims
        np.testing.assert_array_equal(out["mask"].values, ds["mask"].values)
        np.testing.assert_allclose(
            out.coords["wavelength"].values,
            target.astype(np.float32),
        )

    def test_attrs_set(self, synthetic_tanager_dataset):
        target = self._emit_centers()
        out = validation.simulate_sensor(
            synthetic_tanager_dataset["reflectance"], target, 8.5, "EMIT",
        )
        assert out.attrs["sensor_name"] == "EMIT"
        assert "target_fwhm_nm" in out.attrs

    def test_public_api_export(self):
        import tanager

        assert callable(tanager.simulate_sensor)
        assert tanager.simulate_sensor is validation.simulate_sensor

    def test_empty_target_centers_rejected(self, synthetic_tanager_dataset):
        with pytest.raises(ValueError, match="non-empty 1D array"):
            validation.simulate_sensor(
                synthetic_tanager_dataset["reflectance"],
                np.array([]),
                8.5,
                "EMIT",
            )

    def test_dataarray_without_wavelength_dim_rejected(self):
        bad = xr.DataArray(np.zeros((3, 3)), dims=("y", "x"))
        with pytest.raises(ValueError, match="'wavelength' dim"):
            validation.simulate_sensor(bad, np.array([700.0]), 5.0, "EMIT")


# ---------------------------------------------------------------------------
# load_aviris3_reference
# ---------------------------------------------------------------------------


class TestLoadAviris3Reference:
    def test_geotiff_with_descriptions_roundtrip(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_origin

        # 60×60 raster at 3 m resolution → coarsen factor 10 → 6×6 at 30 m.
        shape = (4, 60, 60)
        data = np.zeros(shape, dtype=np.float32)
        data[0] = 0.40  # char
        data[1] = 0.30  # pv
        data[2] = 0.20  # npv
        data[3] = 0.10  # soil

        path = tmp_path / "aviris3_fractions.tif"
        transform = from_origin(0.0, 1800.0, 3.0, 3.0)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=shape[1],
            width=shape[2],
            count=4,
            dtype="float32",
            crs="EPSG:32611",
            transform=transform,
        ) as dst:
            dst.write(data)
            dst.descriptions = ("char", "pv", "npv", "soil")

        ds = validation.load_aviris3_reference(path, target_resolution=30.0)
        # Aggregated to 6×6
        assert ds.sizes["y"] == 6
        assert ds.sizes["x"] == 6
        assert {"char", "pv", "npv", "soil"}.issubset(ds.data_vars)
        # Means preserve constant input
        assert float(ds["char"].mean()) == pytest.approx(0.40, rel=1e-5)
        assert float(ds["pv"].mean()) == pytest.approx(0.30, rel=1e-5)
        assert ds.attrs["source"] == "aviris3"
        assert ds.attrs["target_resolution"] == 30.0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validation.load_aviris3_reference(tmp_path / "missing.tif")

    def test_geotiff_without_descriptions_uses_default_order(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_origin

        shape = (4, 30, 30)
        data = np.full(shape, 0.25, dtype=np.float32)
        path = tmp_path / "aviris3_no_desc.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=shape[1],
            width=shape[2],
            count=4,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(0.0, 30.0, 1.0, 1.0),
        ) as dst:
            dst.write(data)

        ds = validation.load_aviris3_reference(path, target_resolution=1.0)
        # Default order is char, pv, npv, soil
        for var in ("char", "pv", "npv", "soil"):
            assert var in ds.data_vars


# ---------------------------------------------------------------------------
# load_barc_reference
# ---------------------------------------------------------------------------


class TestLoadBARCReference:
    def _write_barc_tif(self, path, codes: np.ndarray, nodata: int | None = None):
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_origin

        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=codes.shape[0],
            width=codes.shape[1],
            count=1,
            dtype="int16",
            crs="EPSG:32611",
            transform=from_origin(0.0, codes.shape[0] * 30.0, 30.0, 30.0),
            nodata=nodata,
        ) as dst:
            dst.write(codes.astype(np.int16), 1)

    def test_remaps_canonical_codes(self, tmp_path):
        codes = np.array(
            [[0, 1, 2], [3, 4, 5]],
            dtype=np.int16,
        )
        path = tmp_path / "barc.tif"
        self._write_barc_tif(path, codes)
        da = validation.load_barc_reference(path)
        # MTBS code 5 ("increased greenness") folds into class 4 (High)
        expected = np.array([[0, 1, 2], [3, 4, 4]], dtype=np.int16)
        np.testing.assert_array_equal(da.values, expected)
        assert da.attrs["source"] == "barc"

    def test_nodata_encoded_as_minus_one(self, tmp_path):
        codes = np.array([[0, 1], [99, 4]], dtype=np.int16)
        path = tmp_path / "barc_nd.tif"
        self._write_barc_tif(path, codes, nodata=99)
        da = validation.load_barc_reference(path)
        assert da.values[1, 0] == -1
        assert da.attrs["nodata"] == -1

    def test_custom_code_map_applied(self, tmp_path):
        codes = np.array([[10, 20]], dtype=np.int16)
        path = tmp_path / "barc_custom.tif"
        self._write_barc_tif(path, codes)
        da = validation.load_barc_reference(path, code_map={10: 0, 20: 4})
        np.testing.assert_array_equal(da.values, np.array([[0, 4]], dtype=np.int16))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validation.load_barc_reference(tmp_path / "missing.tif")

    def test_sbs_code_map_zeros_become_nodata(self, tmp_path):
        codes = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.int16)
        path = tmp_path / "sbs.tif"
        self._write_barc_tif(path, codes)
        da = validation.load_barc_reference(path, code_map=validation.SBS_CODE_MAP)
        expected = np.array([[-1, 1, 2], [3, 4, -1]], dtype=np.int16)
        np.testing.assert_array_equal(da.values, expected)

    def test_unmapped_codes_raise_by_default(self, tmp_path):
        codes = np.array([[1, 2], [3, 15]], dtype=np.int16)
        path = tmp_path / "sbs_unmapped.tif"
        self._write_barc_tif(path, codes)
        with pytest.raises(ValueError, match="codes not in code_map.*15"):
            validation.load_barc_reference(
                path, code_map=validation.SBS_CODE_MAP
            )

    def test_unmapped_codes_passthrough_when_not_strict(self, tmp_path):
        codes = np.array([[1, 2], [3, 15]], dtype=np.int16)
        path = tmp_path / "sbs_lax.tif"
        self._write_barc_tif(path, codes)
        da = validation.load_barc_reference(
            path, code_map=validation.SBS_CODE_MAP, strict=False
        )
        assert 15 in da.values

    def test_unmapped_codes_warn_message(self, tmp_path, caplog):
        codes = np.array([[1, 2], [3, 15]], dtype=np.int16)
        path = tmp_path / "sbs_warn.tif"
        self._write_barc_tif(path, codes)
        with caplog.at_level(logging.WARNING, logger="tanager.validation"):
            validation.load_barc_reference(
                path, code_map=validation.SBS_CODE_MAP, strict=False
            )
        assert "15" in caplog.text
        assert "codes not in code_map" in caplog.text

    def test_complete_code_map_no_error(self, tmp_path):
        codes = np.array([[0, 1], [4, 15]], dtype=np.int16)
        path = tmp_path / "sbs_complete.tif"
        self._write_barc_tif(path, codes)
        full_map = {**validation.SBS_CODE_MAP, 15: -1}
        da = validation.load_barc_reference(path, code_map=full_map)
        assert da.values[1, 1] == -1

    def test_target_grid_alignment_same_crs(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_origin

        codes = np.array([[1, 2], [3, 4]], dtype=np.int16)
        path = tmp_path / "barc_align.tif"
        self._write_barc_tif(path, codes)

        target = xr.DataArray(
            np.zeros((2, 2)),
            dims=("y", "x"),
            coords={"y": [45.0, 15.0], "x": [15.0, 45.0]},
            attrs={"crs": "EPSG:32611"},
        )
        da = validation.load_barc_reference(path, target_grid=target)
        assert da.shape == (2, 2)
        assert da.attrs["source"] == "barc"

    def test_target_grid_alignment_different_crs(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_origin

        codes = np.full((20, 20), 3, dtype=np.int16)
        path = tmp_path / "barc_reproject.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=20,
            width=20,
            count=1,
            dtype="int16",
            crs="EPSG:32611",
            transform=from_origin(370000.0, 3780000.0, 30.0, 30.0),
        ) as dst:
            dst.write(codes, 1)

        target = xr.DataArray(
            np.zeros((5, 5)),
            dims=("y", "x"),
            coords={
                "y": np.linspace(3779700.0, 3779400.0, 5),
                "x": np.linspace(370060.0, 370360.0, 5),
            },
            attrs={"crs": "EPSG:32611"},
        )
        da = validation.load_barc_reference(path, target_grid=target)
        assert da.shape == (5, 5)
        assert (da.values == 3).sum() > 0


# ---------------------------------------------------------------------------
# load_aviris3_reflectance
# ---------------------------------------------------------------------------


def _write_aviris3_netcdf(
    path,
    n_bands=20,
    n_rows=60,
    n_cols=80,
    *,
    wl_min=400.0,
    wl_max=2500.0,
    resolution_m=2.0,
    crs="EPSG:32611",
    rfl_var_name="Rfl",
    wl_var_name="wavelength",
):
    """Write a synthetic AVIRIS-3 L2A reflectance NetCDF for testing."""
    wavelengths = np.linspace(wl_min, wl_max, n_bands).astype(np.float32)
    rng = np.random.default_rng(99)
    rfl = rng.random((n_bands, n_rows, n_cols)).astype(np.float32) * 0.5

    y_coords = 3780000.0 - np.arange(n_rows) * resolution_m
    x_coords = 370000.0 + np.arange(n_cols) * resolution_m

    ds = xr.Dataset(
        {rfl_var_name: (("bands", "y", "x"), rfl)},
        coords={
            wl_var_name: ("bands", wavelengths),
            "y": y_coords,
            "x": x_coords,
        },
        attrs={"crs": crs},
    )
    ds.to_netcdf(path)
    return wavelengths, rfl, y_coords, x_coords


class TestLoadAviris3Reflectance:
    def test_basic_load(self, tmp_path):
        path = tmp_path / "aviris3_rfl.nc"
        wl, rfl, y, x = _write_aviris3_netcdf(path, n_bands=20, n_rows=30, n_cols=40)

        ds = validation.load_aviris3_reflectance(path)

        assert "reflectance" in ds.data_vars
        assert ds["reflectance"].dims == ("wavelength", "y", "x")
        assert ds.sizes["wavelength"] == 20
        assert ds.sizes["y"] == 30
        assert ds.sizes["x"] == 40
        assert ds.attrs["source"] == "aviris3_l2a"
        assert ds.attrs["crs"] == "EPSG:32611"

    def test_wavelength_range_subset(self, tmp_path):
        path = tmp_path / "aviris3_rfl_subset.nc"
        _write_aviris3_netcdf(path, n_bands=100, wl_min=400.0, wl_max=2500.0)

        ds = validation.load_aviris3_reflectance(
            path, wavelength_range=(800.0, 1200.0),
        )
        wl = ds.coords["wavelength"].values
        assert wl.min() >= 800.0
        assert wl.max() <= 1200.0
        assert ds.sizes["wavelength"] < 100

    def test_wavelength_range_empty_rejected(self, tmp_path):
        path = tmp_path / "aviris3_rfl_empty.nc"
        _write_aviris3_netcdf(path, n_bands=20, wl_min=400.0, wl_max=900.0)

        with pytest.raises(ValueError, match="selects no bands"):
            validation.load_aviris3_reflectance(
                path, wavelength_range=(2000.0, 2500.0),
            )

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validation.load_aviris3_reflectance(tmp_path / "missing.nc")

    def test_no_rfl_variable_raises(self, tmp_path):
        path = tmp_path / "aviris3_no_rfl.nc"
        ds = xr.Dataset({"temperature": (("z",), [1.0, 2.0])})
        ds.to_netcdf(path)

        with pytest.raises(ValueError, match="No reflectance variable"):
            validation.load_aviris3_reflectance(path)

    def test_fill_values_become_nan(self, tmp_path):
        path = tmp_path / "aviris3_fill.nc"
        n_bands, n_rows, n_cols = 10, 5, 5
        wavelengths = np.linspace(400, 2500, n_bands).astype(np.float32)
        rfl = np.full((n_bands, n_rows, n_cols), 0.3, dtype=np.float32)
        rfl[0, 0, 0] = -9999.0

        ds = xr.Dataset(
            {"Rfl": (("bands", "y", "x"), rfl)},
            coords={
                "wavelength": ("bands", wavelengths),
                "y": np.arange(n_rows, dtype=np.float64),
                "x": np.arange(n_cols, dtype=np.float64),
            },
        )
        ds.to_netcdf(path)

        loaded = validation.load_aviris3_reflectance(path)
        assert np.isnan(loaded["reflectance"].values[0, 0, 0])

    def test_micrometre_wavelengths_autoconverted(self, tmp_path):
        path = tmp_path / "aviris3_um.nc"
        n_bands = 10
        wl_um = np.linspace(0.4, 2.5, n_bands).astype(np.float32)
        rfl = np.random.default_rng(1).random((n_bands, 5, 5)).astype(np.float32)

        ds = xr.Dataset(
            {"Rfl": (("bands", "y", "x"), rfl)},
            coords={
                "wavelength": ("bands", wl_um),
                "y": np.arange(5, dtype=np.float64),
                "x": np.arange(5, dtype=np.float64),
            },
        )
        ds.to_netcdf(path)

        loaded = validation.load_aviris3_reflectance(path)
        wl_loaded = loaded.coords["wavelength"].values
        assert wl_loaded.min() >= 300.0
        assert wl_loaded.max() <= 3000.0

    def test_alternative_var_name(self, tmp_path):
        path = tmp_path / "aviris3_alt.nc"
        _write_aviris3_netcdf(
            path, n_bands=10, n_rows=5, n_cols=5,
            rfl_var_name="reflectance",
        )

        ds = validation.load_aviris3_reflectance(path)
        assert "reflectance" in ds.data_vars
        assert ds.sizes["wavelength"] == 10

    def test_public_api_export(self):
        import tanager
        assert callable(tanager.load_aviris3_reflectance)
        assert tanager.load_aviris3_reflectance is validation.load_aviris3_reflectance

    def test_cross_validate_export(self):
        import tanager
        assert callable(tanager.cross_validate_aviris3)
        assert tanager.cross_validate_aviris3 is validation.cross_validate_aviris3


# ---------------------------------------------------------------------------
# _aggregate_fractions_to_grid
# ---------------------------------------------------------------------------


class TestAggregateFractionsToGrid:
    def test_coarsen_preserves_mean(self):
        n_fine = 60
        n_coarse = 5
        y_fine = np.arange(n_fine, dtype=np.float64) * 2.0
        x_fine = np.arange(n_fine, dtype=np.float64) * 2.0
        data = np.full((n_fine, n_fine), 0.4, dtype=np.float32)

        fractions = xr.Dataset(
            {"char": (("y", "x"), data)},
            coords={"y": y_fine, "x": x_fine},
        )

        # Target coords within the coarsened grid range (after coarsen by
        # factor 10, coords span roughly 0..118 at 20m steps → centres at
        # 9..109). Place target points inside that range.
        target_y = np.arange(n_coarse, dtype=np.float64) * 20.0 + 10.0
        target_x = np.arange(n_coarse, dtype=np.float64) * 20.0 + 10.0

        result = validation._aggregate_fractions_to_grid(
            fractions,
            target_y=target_y,
            target_x=target_x,
            target_resolution=20.0,
        )

        assert result.sizes["y"] == n_coarse
        assert result.sizes["x"] == n_coarse
        np.testing.assert_allclose(
            result["char"].values,
            0.4,
            atol=1e-5,
        )

    def test_nan_pixels_excluded(self):
        n = 20
        y = np.arange(n, dtype=np.float64) * 2.0
        x = np.arange(n, dtype=np.float64) * 2.0
        data = np.full((n, n), 0.6, dtype=np.float32)
        data[:5, :5] = np.nan

        fractions = xr.Dataset(
            {"char": (("y", "x"), data)},
            coords={"y": y, "x": x},
        )

        target_y = np.array([5.0, 15.0, 25.0])
        target_x = np.array([5.0, 15.0, 25.0])

        result = validation._aggregate_fractions_to_grid(
            fractions,
            target_y=target_y,
            target_x=target_x,
            target_resolution=10.0,
        )

        # All cells should have valid data since coarsen with skipna=True
        # handles partial NaN coverage.
        assert result.sizes["y"] == 3
        assert result.sizes["x"] == 3


# ---------------------------------------------------------------------------
# cross_validate_aviris3
# ---------------------------------------------------------------------------


class TestCrossValidateAviris3:
    def test_perfect_agreement_yields_high_r2(self):
        """Cross-validation with identical synthetic data should show R² ≈ 1."""
        n_wl = 20
        n_y = 10
        n_x = 10
        wavelengths = np.linspace(400, 2500, n_wl).astype(np.float32)

        # Build a simple scene: char-like reflectance everywhere.
        rfl = np.zeros((n_wl, n_y, n_x), dtype=np.float32)
        for i, wl in enumerate(wavelengths):
            rfl[i] = 0.02 + (wl - 400) / (2500 - 400) * 0.03

        aviris3_scene = xr.Dataset(
            {"reflectance": (("wavelength", "y", "x"), rfl)},
            coords={
                "wavelength": wavelengths,
                "y": np.arange(n_y, dtype=np.float64),
                "x": np.arange(n_x, dtype=np.float64),
            },
            attrs={"source": "aviris3_l2a"},
        )

        # Build a minimal endmember library with the same wavelength grid.
        char_spec = np.zeros((1, n_wl), dtype=np.float32)
        for i, wl in enumerate(wavelengths):
            char_spec[0, i] = 0.02 + (wl - 400) / (2500 - 400) * 0.03
        soil_spec = np.full((1, n_wl), 0.20, dtype=np.float32)
        pv_spec = np.full((1, n_wl), 0.40, dtype=np.float32)
        npv_spec = np.full((1, n_wl), 0.15, dtype=np.float32)

        library = xr.DataArray(
            np.vstack([char_spec, soil_spec, pv_spec, npv_spec]),
            dims=("spectrum_id", "wavelength"),
            coords={
                "spectrum_id": ["char_0", "soil_0", "pv_0", "npv_0"],
                "wavelength": wavelengths,
                "category": (
                    "spectrum_id",
                    np.array(["char", "soil", "pv", "npv"], dtype=object),
                ),
                "name": (
                    "spectrum_id",
                    np.array(["char", "soil", "pv", "npv"], dtype=object),
                ),
                "source": (
                    "spectrum_id",
                    np.array(["synth", "synth", "synth", "synth"], dtype=object),
                ),
            },
        )

        # Run MESMA on the same scene to produce "Tanager" fractions.
        from tanager.unmixing import run_mesma

        tanager_fractions = run_mesma(aviris3_scene, library)

        # Cross-validate: should produce near-perfect agreement since the
        # same data and library go through the same MESMA implementation.
        result = validation.cross_validate_aviris3(
            aviris3_reflectance=aviris3_scene,
            tanager_fractions=tanager_fractions,
            library=library,
            target_resolution=1.0,
            fraction_variable="char",
        )

        assert result["method"] == "cross_sensor_mesma"
        assert result["fraction_variable"] == "char"
        assert result["overlap_area_pixels"] >= 10
        assert result["accuracy"]["r2"] > 0.9

    def test_missing_fraction_variable_rejected(self):
        aviris3 = xr.Dataset(
            {"reflectance": (("wavelength", "y", "x"), np.zeros((5, 3, 3)))},
            coords={
                "wavelength": np.linspace(400, 2500, 5).astype(np.float32),
                "y": np.arange(3.0),
                "x": np.arange(3.0),
            },
        )
        tanager = xr.Dataset(
            {"pv": (("y", "x"), np.ones((3, 3)))},
            coords={"y": np.arange(3.0), "x": np.arange(3.0)},
        )
        library = xr.DataArray(
            np.zeros((1, 5)),
            dims=("spectrum_id", "wavelength"),
            coords={
                "spectrum_id": ["test"],
                "wavelength": np.linspace(400, 2500, 5).astype(np.float32),
                "category": ("spectrum_id", np.array(["char"], dtype=object)),
            },
        )
        with pytest.raises(ValueError, match="missing.*char"):
            validation.cross_validate_aviris3(
                aviris3, tanager, library, fraction_variable="char",
            )


# ---------------------------------------------------------------------------
# DINS structure-damage cross-check (load_dins_reference / cross_check_dins)
# ---------------------------------------------------------------------------

# Five UTM 11N pixel-centre locations (30 m grid), one per DINS category.
_DINS_UTM_POINTS = [
    (350015.0, 3760135.0, "No Damage"),
    (350045.0, 3760105.0, "Affected (1-9%)"),
    (350075.0, 3760075.0, "Minor (10-25%)"),
    (350105.0, 3760045.0, "Major (26-50%)"),
    (350135.0, 3760015.0, "Destroyed (>50%)"),
]


def _write_synthetic_dins(path, categories=None, crs_name="urn:ogc:def:crs:OGC:1.3:CRS84"):
    """Write a 5-point DINS-style GeoJSON in WGS84 at known UTM 11N locations."""
    import json

    from pyproj import Transformer

    to_wgs84 = Transformer.from_crs("EPSG:32611", "EPSG:4326", always_xy=True)
    cats = categories or [c for _, _, c in _DINS_UTM_POINTS]
    features = []
    for (utm_x, utm_y, _), cat in zip(_DINS_UTM_POINTS, cats):
        lon, lat = to_wgs84.transform(utm_x, utm_y)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {"OBJECTID": len(features) + 1, "DAMAGE": cat},
            }
        )
    doc = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs_name}},
        "features": features,
    }
    path.write_text(json.dumps(doc))
    return path


def _synthetic_product_raster(values_at_points):
    """10x10 UTM 11N raster (30 m pixels) with given values at the 5 test points."""
    xs = 350015.0 + 30.0 * np.arange(10)
    ys = 3760285.0 - 30.0 * np.arange(10)  # descending, like a GeoTIFF
    data = np.zeros((10, 10), dtype=np.float64)
    for (utm_x, utm_y, _), value in zip(_DINS_UTM_POINTS, values_at_points):
        ix = int(np.argmin(np.abs(xs - utm_x)))
        iy = int(np.argmin(np.abs(ys - utm_y)))
        data[iy, ix] = value
    return xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": ys, "x": xs},
        attrs={"crs": "EPSG:32611"},
    )


class TestLoadDinsReference:
    def test_reprojects_to_utm(self, tmp_path):
        path = _write_synthetic_dins(tmp_path / "dins.geojson")
        gdf = validation.load_dins_reference(path)

        assert str(gdf.crs) == "EPSG:32611"
        assert len(gdf) == 5
        # Round-trip WGS84 -> UTM must land back on the source coordinates.
        for (utm_x, utm_y, cat), (_, row) in zip(_DINS_UTM_POINTS, gdf.iterrows()):
            assert abs(row.geometry.x - utm_x) < 0.01
            assert abs(row.geometry.y - utm_y) < 0.01
            assert row["DAMAGE"] == cat

    def test_damage_ordinal_column(self, tmp_path):
        path = _write_synthetic_dins(tmp_path / "dins.geojson")
        gdf = validation.load_dins_reference(path)
        assert list(gdf["damage_ordinal"]) == [0, 1, 2, 3, 4]

    def test_custom_target_crs(self, tmp_path):
        path = _write_synthetic_dins(tmp_path / "dins.geojson")
        gdf = validation.load_dins_reference(path, target_crs="EPSG:4326")
        assert str(gdf.crs) == "EPSG:4326"
        assert abs(float(gdf.geometry.x.iloc[0]) - (-118.6)) < 0.5

    def test_unknown_category_dropped(self, tmp_path):
        cats = ["No Damage", "Inaccessible", "Minor (10-25%)",
                "Major (26-50%)", "Destroyed (>50%)"]
        path = _write_synthetic_dins(tmp_path / "dins.geojson", categories=cats)
        gdf = validation.load_dins_reference(path)
        assert len(gdf) == 4
        assert "Inaccessible" not in set(gdf["DAMAGE"])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="DINS"):
            validation.load_dins_reference(tmp_path / "nope.geojson")

    def test_missing_damage_field_raises(self, tmp_path):
        path = _write_synthetic_dins(tmp_path / "dins.geojson")
        with pytest.raises(ValueError, match="WRONG_FIELD"):
            validation.load_dins_reference(path, damage_field="WRONG_FIELD")

    def test_public_api_export(self):
        import tanager

        assert tanager.load_dins_reference is validation.load_dins_reference
        assert tanager.cross_check_dins is validation.cross_check_dins
        assert tanager.DINS_DAMAGE_ORDINAL is validation.DINS_DAMAGE_ORDINAL


class TestCrossCheckDins:
    def test_perfectly_monotonic_product(self, tmp_path):
        path = _write_synthetic_dins(tmp_path / "dins.geojson")
        gdf = validation.load_dins_reference(path)
        raster = _synthetic_product_raster([0.0, 0.2, 0.3, 0.4, 0.5])

        result = validation.cross_check_dins(gdf, raster, product_name="dNBR")

        assert result["product_name"] == "dNBR"
        assert result["n_valid"] == 5
        assert result["n_outside"] == 0
        assert result["spearman_rho"] == pytest.approx(1.0)

        pc = result["per_category"]
        assert list(pc) == ["No Damage", "Affected (1-9%)", "Minor (10-25%)",
                            "Major (26-50%)", "Destroyed (>50%)"]
        assert pc["No Damage"]["mean"] == pytest.approx(0.0)
        assert pc["Destroyed (>50%)"]["mean"] == pytest.approx(0.5)
        assert pc["Destroyed (>50%)"]["count"] == 1

        b = result["binary"]
        # threshold=0.1: predicted damaged = [F,T,T,T,T]; observed = [F,T,T,T,T]
        assert (b["tp"], b["fp"], b["fn"], b["tn"]) == (4, 0, 0, 1)
        assert b["accuracy"] == pytest.approx(1.0)
        assert b["precision"] == pytest.approx(1.0)
        assert b["recall"] == pytest.approx(1.0)
        assert b["f1"] == pytest.approx(1.0)

    def test_threshold_configurable(self, tmp_path):
        path = _write_synthetic_dins(tmp_path / "dins.geojson")
        gdf = validation.load_dins_reference(path)
        raster = _synthetic_product_raster([0.0, 0.2, 0.3, 0.4, 0.5])

        result = validation.cross_check_dins(gdf, raster, threshold=0.35)
        b = result["binary"]
        # predicted damaged = [F,F,F,T,T]; observed = [F,T,T,T,T]
        assert (b["tp"], b["fp"], b["fn"], b["tn"]) == (2, 0, 2, 1)
        assert b["recall"] == pytest.approx(0.5)
        assert b["precision"] == pytest.approx(1.0)

    def test_anticorrelated_product(self, tmp_path):
        path = _write_synthetic_dins(tmp_path / "dins.geojson")
        gdf = validation.load_dins_reference(path)
        raster = _synthetic_product_raster([0.5, 0.4, 0.3, 0.2, 0.0])
        result = validation.cross_check_dins(gdf, raster)
        assert result["spearman_rho"] == pytest.approx(-1.0)

    def test_point_outside_raster_excluded(self, tmp_path):
        path = _write_synthetic_dins(tmp_path / "dins.geojson")
        gdf = validation.load_dins_reference(path)
        raster = _synthetic_product_raster([0.0, 0.2, 0.3, 0.4, 0.5])
        # Shrink the raster so the last point (350135, 3760015) falls outside.
        raster = raster.isel(y=slice(0, 9), x=slice(0, 4))

        result = validation.cross_check_dins(gdf, raster)
        assert result["n_valid"] == 4
        assert result["n_outside"] == 1
        assert "Destroyed (>50%)" not in result["per_category"]

    def test_nan_pixels_excluded(self, tmp_path):
        path = _write_synthetic_dins(tmp_path / "dins.geojson")
        gdf = validation.load_dins_reference(path)
        raster = _synthetic_product_raster([0.0, 0.2, 0.3, 0.4, np.nan])
        result = validation.cross_check_dins(gdf, raster)
        assert result["n_valid"] == 4
        assert result["n_outside"] == 1

    def test_crs_mismatch_reprojects_points(self, tmp_path):
        path = _write_synthetic_dins(tmp_path / "dins.geojson")
        # Points kept in WGS84; raster is UTM -> cross_check must reproject.
        gdf = validation.load_dins_reference(path, target_crs="EPSG:4326")
        raster = _synthetic_product_raster([0.0, 0.2, 0.3, 0.4, 0.5])
        result = validation.cross_check_dins(gdf, raster)
        assert result["n_valid"] == 5
        assert result["spearman_rho"] == pytest.approx(1.0)

    def test_geotiff_path_input(self, tmp_path):
        import rasterio
        from rasterio.transform import from_origin

        path = _write_synthetic_dins(tmp_path / "dins.geojson")
        gdf = validation.load_dins_reference(path)
        raster = _synthetic_product_raster([0.0, 0.2, 0.3, 0.4, 0.5])

        tif = tmp_path / "dnbr.tif"
        transform = from_origin(350000.0, 3760300.0, 30.0, 30.0)
        with rasterio.open(
            tif, "w", driver="GTiff", height=10, width=10, count=1,
            dtype="float64", crs="EPSG:32611", transform=transform,
        ) as dst:
            dst.write(raster.values, 1)

        result = validation.cross_check_dins(gdf, tif)
        assert result["n_valid"] == 5
        assert result["spearman_rho"] == pytest.approx(1.0)

    def test_no_overlap_raises(self, tmp_path):
        path = _write_synthetic_dins(tmp_path / "dins.geojson")
        gdf = validation.load_dins_reference(path)
        raster = _synthetic_product_raster([0.0, 0.2, 0.3, 0.4, 0.5])
        far_away = raster.assign_coords(x=raster.x + 1e6)
        with pytest.raises(ValueError, match="No DINS point"):
            validation.cross_check_dins(gdf, far_away)
