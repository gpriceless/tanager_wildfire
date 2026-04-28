"""Tests for :mod:`tanager.validation`.

Synthetic GeoTIFFs and DataArrays exercise:

* :func:`tanager.validation.compute_accuracy` — continuous (R²/RMSE/MAE/bias)
  and classified (accuracy/Cohen's kappa/F1/confusion matrix) paths against
  hand-calculated values.
* :func:`tanager.validation.compare_sensors` — Tanager-vs-reference comparison
  format, improvement ratios, and pandas comparison table.
* :func:`tanager.validation.load_aviris3_reference` — GeoTIFF schema mapping
  and 3-m → 30-m spatial aggregation.
* :func:`tanager.validation.load_barc_reference` — BARC code remapping,
  nodata handling, and target-grid alignment.
"""

from __future__ import annotations

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
