"""Tests for tanager.masks pixel-quality masking utilities."""

from __future__ import annotations

from functools import reduce
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from tanager.masks import apply_masks, cloud_mask, nodata_mask, water_mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_spectral_dataset(
    wavelengths: np.ndarray | None = None,
    shape: tuple[int, int] = (4, 4),
    fill: float = 0.5,
) -> xr.Dataset:
    """Minimal synthetic dataset for mask testing."""
    if wavelengths is None:
        wavelengths = np.linspace(400, 900, 5)
    n = len(wavelengths)
    y, x = shape
    data = np.full((n, y, x), fill, dtype=np.float32)
    return xr.Dataset(
        {"reflectance": (["wavelength", "y", "x"], data)},
        coords={"wavelength": wavelengths},
    )


def make_ndwi_dataset(
    ndwi_values: np.ndarray,
    wavelengths: np.ndarray | None = None,
) -> xr.Dataset:
    """Dataset whose ndwi() result matches ndwi_values exactly.

    Uses Green (~560 nm) and NIR (~860 nm) bands engineered to produce the
    requested NDWI = (Green - NIR) / (Green + NIR) values without calling the
    real ndwi() implementation.  Instead, tests that use this fixture patch
    tanager.spectral.ndwi directly.
    """
    if wavelengths is None:
        wavelengths = np.array([560.0, 860.0])
    y, x = ndwi_values.shape
    n = len(wavelengths)
    data = np.ones((n, y, x), dtype=np.float32)
    return xr.Dataset(
        {"reflectance": (["wavelength", "y", "x"], data)},
        coords={"wavelength": wavelengths},
    )


# ---------------------------------------------------------------------------
# nodata_mask — basic validity
# ---------------------------------------------------------------------------


class TestNodataMask:
    def test_all_valid_returns_all_true(self) -> None:
        ds = make_spectral_dataset()
        result = nodata_mask(ds)
        assert result.dtype == bool
        assert result.values.all()

    def test_single_nan_pixel_marked_invalid(self) -> None:
        ds = make_spectral_dataset()
        ds["reflectance"].values[0, 1, 1] = np.nan
        result = nodata_mask(ds)
        assert not result.values[1, 1], "Pixel with NaN should be False"
        # Other pixels should still be valid
        assert result.values[0, 0]

    def test_all_nan_dataset_returns_all_false(self) -> None:
        wavelengths = np.linspace(400, 900, 3)
        n = len(wavelengths)
        data = np.full((n, 4, 4), np.nan, dtype=np.float32)
        ds = xr.Dataset(
            {"reflectance": (["wavelength", "y", "x"], data)},
            coords={"wavelength": wavelengths},
        )
        result = nodata_mask(ds)
        assert not result.values.any()

    def test_inf_pixel_marked_invalid(self) -> None:
        ds = make_spectral_dataset()
        ds["reflectance"].values[0, 2, 3] = np.inf
        result = nodata_mask(ds)
        assert not result.values[2, 3]

    def test_neg_inf_pixel_marked_invalid(self) -> None:
        ds = make_spectral_dataset()
        ds["reflectance"].values[0, 0, 0] = -np.inf
        result = nodata_mask(ds)
        assert not result.values[0, 0]

    def test_returns_dataarray(self) -> None:
        ds = make_spectral_dataset()
        result = nodata_mask(ds)
        assert isinstance(result, xr.DataArray)

    def test_output_has_spatial_dims(self) -> None:
        ds = make_spectral_dataset(shape=(6, 8))
        result = nodata_mask(ds)
        assert result.sizes == {"y": 6, "x": 8}

    def test_does_not_modify_input(self) -> None:
        ds = make_spectral_dataset()
        original = ds["reflectance"].values.copy()
        nodata_mask(ds)
        np.testing.assert_array_equal(ds["reflectance"].values, original)


class TestNodataMaskFillValue:
    def test_fill_value_pixel_marked_invalid(self) -> None:
        ds = make_spectral_dataset()
        ds["reflectance"].values[:, 2, 2] = -9999.0
        result = nodata_mask(ds, fill_value=-9999.0)
        assert not result.values[2, 2]

    def test_fill_value_partial_band_still_invalid(self) -> None:
        # Only one band set to fill_value — the pixel should be invalid
        ds = make_spectral_dataset()
        ds["reflectance"].values[0, 1, 1] = -9999.0
        result = nodata_mask(ds, fill_value=-9999.0)
        assert not result.values[1, 1]

    def test_non_fill_pixels_valid_with_fill_check(self) -> None:
        ds = make_spectral_dataset()
        ds["reflectance"].values[:, 3, 3] = -9999.0
        result = nodata_mask(ds, fill_value=-9999.0)
        # Pixels not equal to fill_value and not NaN should be True
        assert result.values[0, 0]
        assert result.values[1, 2]

    def test_nan_also_invalid_when_fill_value_given(self) -> None:
        ds = make_spectral_dataset()
        ds["reflectance"].values[0, 0, 0] = np.nan
        result = nodata_mask(ds, fill_value=-9999.0)
        assert not result.values[0, 0]

    def test_fill_value_none_ignores_sentinel(self) -> None:
        # -9999 is NOT NaN so should NOT be masked when fill_value=None
        ds = make_spectral_dataset()
        ds["reflectance"].values[:, 2, 2] = -9999.0
        result = nodata_mask(ds, fill_value=None)
        assert result.values[2, 2]


# ---------------------------------------------------------------------------
# cloud_mask — data variable branch
# ---------------------------------------------------------------------------


class TestCloudMaskDataVariable:
    def test_uses_existing_data_variable(self) -> None:
        ds = make_spectral_dataset()
        flag = np.zeros((4, 4), dtype=np.int8)
        ds["beta_cirrus_mask"] = xr.DataArray(flag, dims=["y", "x"])
        result = cloud_mask(ds)
        assert result.values.all(), "All zeros → all clear → all True"

    def test_cloudy_pixels_are_false(self) -> None:
        ds = make_spectral_dataset()
        flag = np.zeros((4, 4), dtype=np.int8)
        flag[1, 1] = 1
        ds["beta_cirrus_mask"] = xr.DataArray(flag, dims=["y", "x"])
        result = cloud_mask(ds)
        assert not result.values[1, 1]
        assert result.values[0, 0]

    def test_returns_dataarray(self) -> None:
        ds = make_spectral_dataset()
        ds["beta_cirrus_mask"] = xr.DataArray(np.zeros((4, 4), dtype=np.int8), dims=["y", "x"])
        result = cloud_mask(ds)
        assert isinstance(result, xr.DataArray)


# ---------------------------------------------------------------------------
# cloud_mask — HDF5 branch
# ---------------------------------------------------------------------------


def _patch_field_reader(fields: dict[str, np.ndarray | None]):
    """Patch _read_mask_field_from_hdf5 to return per-field arrays from a dict."""
    def fake_reader(filepath, field_name):
        return fields.get(field_name)
    return patch("tanager.masks._read_mask_field_from_hdf5", side_effect=fake_reader)


class TestCloudMaskHDF5:
    def test_reads_from_hdf5_via_filepath_arg(self) -> None:
        ds = make_spectral_dataset(shape=(3, 3))
        cirrus = np.zeros((3, 3), dtype=np.int8)
        cirrus[0, 0] = 1

        with _patch_field_reader({"beta_cirrus_mask": cirrus, "beta_cloud_mask": None}):
            result = cloud_mask(ds, filepath="/fake/file.h5")

        assert not result.values[0, 0]
        assert result.values[1, 1]

    def test_reads_from_encoding_source(self) -> None:
        ds = make_spectral_dataset(shape=(3, 3))
        ds.encoding["source"] = "/encoded/file.h5"
        flag = np.zeros((3, 3), dtype=np.int8)

        with _patch_field_reader({"beta_cirrus_mask": flag, "beta_cloud_mask": None}):
            result = cloud_mask(ds)

        assert result.values.all()

    def test_shape_mismatch_falls_through_to_allTrue(self) -> None:
        ds = make_spectral_dataset(shape=(3, 3))
        # Return a mask with wrong shape — should be ignored
        wrong_shape_flag = np.zeros((5, 5), dtype=np.int8)

        with _patch_field_reader({"beta_cirrus_mask": wrong_shape_flag, "beta_cloud_mask": None}):
            result = cloud_mask(ds, filepath="/fake/file.h5")

        # Shape mismatch on every field: fall through to all-True
        assert result.values.all()

    def test_or_combines_cirrus_and_cloud(self) -> None:
        """Both fields contribute — pixel is cloudy if either flags it (LGT-297)."""
        ds = make_spectral_dataset(shape=(3, 3))
        cirrus = np.zeros((3, 3), dtype=np.int8)
        cirrus[0, 0] = 1  # cirrus flags (0,0)
        cloud = np.zeros((3, 3), dtype=np.int8)
        cloud[1, 1] = 1  # cloud flags (1,1)

        with _patch_field_reader({"beta_cirrus_mask": cirrus, "beta_cloud_mask": cloud}):
            result = cloud_mask(ds, filepath="/fake/file.h5")

        # Both flagged pixels should be False (cloudy); others True (clear)
        assert not result.values[0, 0]
        assert not result.values[1, 1]
        assert result.values[0, 1]
        assert result.values[2, 2]

    def test_only_beta_cloud_mask_present(self) -> None:
        """If cirrus is missing but cloud is present, mask still works (LGT-297)."""
        ds = make_spectral_dataset(shape=(3, 3))
        cloud = np.zeros((3, 3), dtype=np.int8)
        cloud[2, 2] = 1

        with _patch_field_reader({"beta_cirrus_mask": None, "beta_cloud_mask": cloud}):
            result = cloud_mask(ds, filepath="/fake/file.h5")

        assert not result.values[2, 2]
        assert result.values[0, 0]


class TestCandidateMaskPaths:
    """The path-search helper covers SWATHS and GRIDS layouts (LGT-297)."""

    def test_grids_path_present_for_cirrus(self) -> None:
        from tanager.masks import _candidate_mask_paths

        paths = _candidate_mask_paths("beta_cirrus_mask")
        assert "/HDFEOS/GRIDS/HYP/Data Fields/beta_cirrus_mask" in paths

    def test_grids_path_present_for_cloud(self) -> None:
        from tanager.masks import _candidate_mask_paths

        paths = _candidate_mask_paths("beta_cloud_mask")
        assert "/HDFEOS/GRIDS/HYP/Data Fields/beta_cloud_mask" in paths

    def test_swaths_path_still_searched(self) -> None:
        from tanager.masks import _candidate_mask_paths

        paths = _candidate_mask_paths("beta_cirrus_mask")
        assert "/HDFEOS/SWATHS/HYP/Data Fields/beta_cirrus_mask" in paths
        assert "/HDFEOS/SWATHS/HYP/Metadata/beta_cirrus_mask" in paths

    def test_grids_path_searched_before_swaths(self) -> None:
        """Ortho SR is the production layout — search GRIDS first."""
        from tanager.masks import _candidate_mask_paths

        paths = _candidate_mask_paths("beta_cirrus_mask")
        grids_idx = paths.index("/HDFEOS/GRIDS/HYP/Data Fields/beta_cirrus_mask")
        swaths_idx = paths.index("/HDFEOS/SWATHS/HYP/Data Fields/beta_cirrus_mask")
        assert grids_idx < swaths_idx


class TestReadMaskFieldFromHDF5:
    """End-to-end of the HDF5 reader against synthetic SWATHS and GRIDS files."""

    def _write_fixture(self, path, hdf5_path: str, data: np.ndarray) -> None:
        import h5py

        with h5py.File(path, "w") as f:
            grp = f
            parts = hdf5_path.strip("/").split("/")
            for part in parts[:-1]:
                grp = grp.require_group(part)
            grp.create_dataset(parts[-1], data=data)

    def test_reads_grids_layout(self, tmp_path) -> None:
        from tanager.masks import _read_mask_field_from_hdf5

        path = tmp_path / "ortho.h5"
        arr = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        self._write_fixture(path, "/HDFEOS/GRIDS/HYP/Data Fields/beta_cirrus_mask", arr)

        result = _read_mask_field_from_hdf5(str(path), "beta_cirrus_mask")
        assert result is not None
        np.testing.assert_array_equal(result, arr)

    def test_reads_swaths_layout(self, tmp_path) -> None:
        from tanager.masks import _read_mask_field_from_hdf5

        path = tmp_path / "swath.h5"
        arr = np.array([[0, 0], [0, 1]], dtype=np.uint8)
        self._write_fixture(path, "/HDFEOS/SWATHS/HYP/Data Fields/beta_cirrus_mask", arr)

        result = _read_mask_field_from_hdf5(str(path), "beta_cirrus_mask")
        assert result is not None
        np.testing.assert_array_equal(result, arr)

    def test_reads_beta_cloud_mask_from_grids(self, tmp_path) -> None:
        from tanager.masks import _read_mask_field_from_hdf5

        path = tmp_path / "ortho_cloud.h5"
        arr = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        self._write_fixture(path, "/HDFEOS/GRIDS/HYP/Data Fields/beta_cloud_mask", arr)

        result = _read_mask_field_from_hdf5(str(path), "beta_cloud_mask")
        assert result is not None
        np.testing.assert_array_equal(result, arr)

    def test_returns_none_when_absent(self, tmp_path) -> None:
        from tanager.masks import _read_mask_field_from_hdf5

        path = tmp_path / "empty.h5"
        # Stash an unrelated dataset
        self._write_fixture(path, "/HDFEOS/GRIDS/HYP/Data Fields/some_other_field",
                            np.zeros((2, 2), dtype=np.uint8))

        result = _read_mask_field_from_hdf5(str(path), "beta_cirrus_mask")
        assert result is None


class TestCloudMaskDataVariableOrCombine:
    """Data-variable branch also OR-combines cirrus + cloud (LGT-297)."""

    def test_or_combines_when_both_present(self) -> None:
        ds = make_spectral_dataset(shape=(3, 3))
        cirrus = np.zeros((3, 3), dtype=np.int8)
        cirrus[0, 0] = 1
        cloud = np.zeros((3, 3), dtype=np.int8)
        cloud[2, 2] = 1
        ds["beta_cirrus_mask"] = xr.DataArray(cirrus, dims=["y", "x"])
        ds["beta_cloud_mask"] = xr.DataArray(cloud, dims=["y", "x"])

        result = cloud_mask(ds)

        assert not result.values[0, 0]
        assert not result.values[2, 2]
        assert result.values[1, 1]

    def test_only_cloud_variable_present(self) -> None:
        ds = make_spectral_dataset(shape=(3, 3))
        cloud = np.zeros((3, 3), dtype=np.int8)
        cloud[1, 2] = 1
        ds["beta_cloud_mask"] = xr.DataArray(cloud, dims=["y", "x"])

        result = cloud_mask(ds)

        assert not result.values[1, 2]
        assert result.values[0, 0]


# ---------------------------------------------------------------------------
# cloud_mask — fallback branch
# ---------------------------------------------------------------------------


class TestCloudMaskFallback:
    def test_fallback_returns_all_true(self) -> None:
        ds = make_spectral_dataset(shape=(5, 6))
        result = cloud_mask(ds)
        assert result.values.all()

    def test_fallback_has_correct_shape(self) -> None:
        ds = make_spectral_dataset(shape=(5, 6))
        result = cloud_mask(ds)
        assert result.sizes == {"y": 5, "x": 6}

    def test_fallback_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        ds = make_spectral_dataset()
        with caplog.at_level(logging.WARNING, logger="tanager.masks"):
            cloud_mask(ds)
        assert any("beta_cirrus_mask" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# water_mask
# ---------------------------------------------------------------------------


class TestWaterMask:
    def _make_ndwi_da(self, values: np.ndarray) -> xr.DataArray:
        return xr.DataArray(values, dims=["y", "x"])

    def test_land_pixels_are_true(self) -> None:
        ndwi_values = np.array([[-0.5, -0.2], [0.1, 0.29]])
        ds = make_ndwi_dataset(ndwi_values)

        with patch("tanager.spectral.ndwi", return_value=self._make_ndwi_da(ndwi_values), create=True):
            result = water_mask(ds, threshold=0.3)

        assert result.values.all(), "All NDWI <= 0.3 → all land → all True"

    def test_water_pixels_are_false(self) -> None:
        ndwi_values = np.array([[0.5, 0.8], [0.31, 0.9]])
        ds = make_ndwi_dataset(ndwi_values)

        with patch("tanager.spectral.ndwi", return_value=self._make_ndwi_da(ndwi_values), create=True):
            result = water_mask(ds, threshold=0.3)

        assert not result.values.any(), "All NDWI > 0.3 → all water → all False"

    def test_threshold_boundary_is_land(self) -> None:
        ndwi_values = np.array([[0.3]])
        ds = make_ndwi_dataset(ndwi_values)

        with patch("tanager.spectral.ndwi", return_value=self._make_ndwi_da(ndwi_values), create=True):
            result = water_mask(ds, threshold=0.3)

        assert result.values[0, 0], "NDWI == threshold → land (<=)"

    def test_mixed_land_and_water(self) -> None:
        ndwi_values = np.array([[0.1, 0.5], [-0.2, 0.4]])
        ds = make_ndwi_dataset(ndwi_values)

        with patch("tanager.spectral.ndwi", return_value=self._make_ndwi_da(ndwi_values), create=True):
            result = water_mask(ds, threshold=0.3)

        assert result.values[0, 0]       # 0.1 <= 0.3 → land
        assert not result.values[0, 1]   # 0.5 > 0.3 → water
        assert result.values[1, 0]       # -0.2 <= 0.3 → land
        assert not result.values[1, 1]   # 0.4 > 0.3 → water

    def test_custom_threshold(self) -> None:
        ndwi_values = np.array([[0.1, 0.5]])
        ds = make_ndwi_dataset(ndwi_values)

        with patch("tanager.spectral.ndwi", return_value=self._make_ndwi_da(ndwi_values), create=True):
            result = water_mask(ds, threshold=0.2)

        assert result.values[0, 0]       # 0.1 <= 0.2 → land
        assert not result.values[0, 1]   # 0.5 > 0.2 → water

    def test_returns_dataarray(self) -> None:
        ndwi_values = np.zeros((3, 3))
        ds = make_ndwi_dataset(ndwi_values)

        with patch("tanager.spectral.ndwi", return_value=self._make_ndwi_da(ndwi_values), create=True):
            result = water_mask(ds)

        assert isinstance(result, xr.DataArray)


# ---------------------------------------------------------------------------
# apply_masks
# ---------------------------------------------------------------------------


class TestApplyMasks:
    def _make_mask(self, shape: tuple[int, int], value: bool) -> xr.DataArray:
        return xr.DataArray(np.full(shape, value, dtype=bool), dims=["y", "x"])

    def test_single_all_true_mask_retains_data(self) -> None:
        ds = make_spectral_dataset(shape=(3, 3), fill=1.0)
        mask = self._make_mask((3, 3), True)
        result = apply_masks(ds, [mask])
        # All pixels valid — values unchanged
        np.testing.assert_array_equal(
            result["reflectance"].values[0],
            ds["reflectance"].values[0],
        )

    def test_single_all_false_mask_sets_all_nan(self) -> None:
        ds = make_spectral_dataset(shape=(3, 3), fill=1.0)
        mask = self._make_mask((3, 3), False)
        result = apply_masks(ds, [mask])
        assert np.isnan(result["reflectance"].values).all()

    def test_masked_pixels_are_nan(self) -> None:
        ds = make_spectral_dataset(shape=(2, 2), fill=5.0)
        mask = xr.DataArray([[True, False], [True, True]], dims=["y", "x"])
        result = apply_masks(ds, [mask])
        assert not np.isnan(result["reflectance"].values[:, 0, 0]).any()
        assert np.isnan(result["reflectance"].values[:, 0, 1]).all()

    def test_unmasked_pixels_retain_original_values(self) -> None:
        ds = make_spectral_dataset(shape=(3, 4), fill=7.5)
        mask = self._make_mask((3, 4), True)
        result = apply_masks(ds, [mask])
        np.testing.assert_allclose(
            result["reflectance"].values,
            ds["reflectance"].values,
        )

    def test_logical_and_of_multiple_masks(self) -> None:
        ds = make_spectral_dataset(shape=(2, 2), fill=1.0)
        m1 = xr.DataArray([[True, True], [True, False]], dims=["y", "x"])
        m2 = xr.DataArray([[True, False], [True, True]], dims=["y", "x"])
        result = apply_masks(ds, [m1, m2])
        # (0,0): T&T=True → valid
        assert not np.isnan(result["reflectance"].values[:, 0, 0]).any()
        # (0,1): T&F=False → NaN
        assert np.isnan(result["reflectance"].values[:, 0, 1]).all()
        # (1,0): T&T=True → valid
        assert not np.isnan(result["reflectance"].values[:, 1, 0]).any()
        # (1,1): F&T=False → NaN
        assert np.isnan(result["reflectance"].values[:, 1, 1]).all()

    def test_three_masks_combined(self) -> None:
        ds = make_spectral_dataset(shape=(2, 2), fill=2.0)
        m1 = xr.DataArray([[True, True], [False, True]], dims=["y", "x"])
        m2 = xr.DataArray([[True, True], [True, True]], dims=["y", "x"])
        m3 = xr.DataArray([[True, False], [True, True]], dims=["y", "x"])
        result = apply_masks(ds, [m1, m2, m3])
        assert not np.isnan(result["reflectance"].values[:, 0, 0]).any()  # T&T&T
        assert np.isnan(result["reflectance"].values[:, 0, 1]).all()      # T&T&F
        assert np.isnan(result["reflectance"].values[:, 1, 0]).all()      # F&T&T
        assert not np.isnan(result["reflectance"].values[:, 1, 1]).any()  # T&T&T

    def test_empty_mask_list_raises(self) -> None:
        ds = make_spectral_dataset()
        with pytest.raises(ValueError, match="at least one mask"):
            apply_masks(ds, [])

    def test_returns_dataset(self) -> None:
        ds = make_spectral_dataset(shape=(3, 3))
        mask = self._make_mask((3, 3), True)
        result = apply_masks(ds, [mask])
        assert isinstance(result, xr.Dataset)

    def test_does_not_modify_input(self) -> None:
        ds = make_spectral_dataset(shape=(3, 3), fill=3.0)
        mask = self._make_mask((3, 3), False)
        original = ds["reflectance"].values.copy()
        apply_masks(ds, [mask])
        np.testing.assert_array_equal(ds["reflectance"].values, original)


# ---------------------------------------------------------------------------
# Task 5: End-to-end combined mask verification
# ---------------------------------------------------------------------------


class TestTask5CombinedMaskVerification:
    """End-to-end verification: combined mask produces correct NaN/value pattern."""

    def test_combined_mask_pixels_are_nan_or_original(self) -> None:
        """Masked pixels become NaN; unmasked pixels retain their original values."""
        rng = np.random.default_rng(42)
        y_size, x_size = 6, 8
        n_bands = 5
        wavelengths = np.linspace(400, 900, n_bands)
        data = rng.random((n_bands, y_size, x_size)).astype(np.float32) + 0.1

        ds = xr.Dataset(
            {"reflectance": (["wavelength", "y", "x"], data)},
            coords={"wavelength": wavelengths},
        )

        # Build synthetic masks
        nd_mask = nodata_mask(ds)  # should be all-True (no NaN in data)

        # cloud: mark row 0 as cloudy
        cloud_arr = np.ones((y_size, x_size), dtype=bool)
        cloud_arr[0, :] = False
        synthetic_cloud_mask = xr.DataArray(cloud_arr, dims=["y", "x"])

        # water: mark column 0 as water
        water_arr = np.ones((y_size, x_size), dtype=bool)
        water_arr[:, 0] = False
        synthetic_water_mask = xr.DataArray(water_arr, dims=["y", "x"])

        result = apply_masks(ds, [nd_mask, synthetic_cloud_mask, synthetic_water_mask])

        combined = nd_mask & synthetic_cloud_mask & synthetic_water_mask

        # Verify masked pixels
        for yi in range(y_size):
            for xi in range(x_size):
                if not combined.values[yi, xi]:
                    assert np.isnan(result["reflectance"].values[:, yi, xi]).all(), (
                        f"Expected NaN at ({yi},{xi}) but got values"
                    )
                else:
                    np.testing.assert_allclose(
                        result["reflectance"].values[:, yi, xi],
                        data[:, yi, xi],
                        err_msg=f"Pixel ({yi},{xi}) should retain original values",
                    )

    def test_nodata_mask_integration_with_apply(self) -> None:
        """NaN pixels are masked out by apply_masks."""
        wavelengths = np.linspace(400, 900, 4)
        data = np.ones((4, 3, 3), dtype=np.float32)
        data[:, 1, 1] = np.nan

        ds = xr.Dataset(
            {"reflectance": (["wavelength", "y", "x"], data)},
            coords={"wavelength": wavelengths},
        )

        nd = nodata_mask(ds)
        result = apply_masks(ds, [nd])

        # The NaN pixel stays NaN
        assert np.isnan(result["reflectance"].values[:, 1, 1]).all()
        # Valid pixels are unchanged
        np.testing.assert_allclose(result["reflectance"].values[:, 0, 0], 1.0)


# ---------------------------------------------------------------------------
# LGT-297 — real ortho HDF5 integration test
# ---------------------------------------------------------------------------


_REAL_FIRE_DIR = "data/raw/fire"
_REAL_FIRE_SCENES = [
    "20241215_185916_33_4001_ortho_sr_hdf5.h5",
    "20250123_185507_64_4001_ortho_sr_hdf5.h5",
    "20250407_192235_24_4001_ortho_sr_hdf5.h5",
]


def _real_fire_scene_paths() -> list[str]:
    import os

    base = os.path.join(os.getcwd(), _REAL_FIRE_DIR)
    available = [os.path.join(base, name) for name in _REAL_FIRE_SCENES
                 if os.path.exists(os.path.join(base, name))]
    return available


@pytest.mark.skipif(
    not _real_fire_scene_paths(),
    reason="Real ortho SR HDF5 scenes not present in data/raw/fire/",
)
class TestCloudMaskRealOrthoHDF5:
    """LGT-297: cloud_mask must read masks from real ortho SR HDF5 files.

    These scenes use the GRIDS layout, not SWATHS. Before the fix,
    _read_beta_cirrus_from_hdf5 silently returned None and cloud_mask
    fell back to all-True. After the fix, the mask is loaded and at
    least one of the three real scenes contains non-trivial cloud pixels.
    """

    def test_reads_masks_from_all_real_scenes(self) -> None:
        import h5py

        for path in _real_fire_scene_paths():
            with h5py.File(path, "r") as f:
                cirrus_path = "/HDFEOS/GRIDS/HYP/Data Fields/beta_cirrus_mask"
                cloud_path = "/HDFEOS/GRIDS/HYP/Data Fields/beta_cloud_mask"
                assert cirrus_path in f or cloud_path in f, (
                    f"Real scene {path} has neither cirrus nor cloud mask"
                )
                # Build a minimal dataset shaped to match the mask
                ref = f[cirrus_path] if cirrus_path in f else f[cloud_path]
                y_size, x_size = ref.shape

            ds = xr.Dataset(
                {"reflectance": (["wavelength", "y", "x"],
                                 np.zeros((1, y_size, x_size), dtype=np.float32))},
                coords={"wavelength": np.array([500.0])},
            )

            mask = cloud_mask(ds, filepath=path)
            assert mask.shape == (y_size, x_size)
            assert mask.dtype == bool

    def test_real_scene_produces_non_trivial_mask(self) -> None:
        """At least one real scene must produce a non-all-True (non-trivial) mask.

        The pre-fix bug returned all-True silently. This test guards against
        regression: if the loader breaks again, every scene becomes all-True.
        """
        import h5py

        non_trivial_seen = False
        for path in _real_fire_scene_paths():
            with h5py.File(path, "r") as f:
                cirrus_path = "/HDFEOS/GRIDS/HYP/Data Fields/beta_cirrus_mask"
                cloud_path = "/HDFEOS/GRIDS/HYP/Data Fields/beta_cloud_mask"
                ref = f[cirrus_path] if cirrus_path in f else f[cloud_path]
                y_size, x_size = ref.shape

            ds = xr.Dataset(
                {"reflectance": (["wavelength", "y", "x"],
                                 np.zeros((1, y_size, x_size), dtype=np.float32))},
                coords={"wavelength": np.array([500.0])},
            )

            mask = cloud_mask(ds, filepath=path)
            cloudy_count = int((~mask).sum().item())
            print(
                f"[LGT-297] {path.split('/')[-1]}: shape={mask.shape}, "
                f"cloudy_pixels={cloudy_count}/{mask.size}"
            )
            if cloudy_count > 0:
                non_trivial_seen = True

        assert non_trivial_seen, (
            "All real scenes returned all-True masks — cloud masking is "
            "silently disabled (LGT-297 regression)."
        )
