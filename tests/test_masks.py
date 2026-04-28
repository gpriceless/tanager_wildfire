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


class TestCloudMaskHDF5:
    def test_reads_from_hdf5_via_filepath_arg(self) -> None:
        ds = make_spectral_dataset(shape=(3, 3))
        flag = np.zeros((3, 3), dtype=np.int8)
        flag[0, 0] = 1

        with patch("tanager.masks._read_beta_cirrus_from_hdf5", return_value=flag) as mock_read:
            result = cloud_mask(ds, filepath="/fake/file.h5")

        mock_read.assert_called_once_with("/fake/file.h5")
        assert not result.values[0, 0]
        assert result.values[1, 1]

    def test_reads_from_encoding_source(self) -> None:
        ds = make_spectral_dataset(shape=(3, 3))
        ds.encoding["source"] = "/encoded/file.h5"
        flag = np.zeros((3, 3), dtype=np.int8)

        with patch("tanager.masks._read_beta_cirrus_from_hdf5", return_value=flag):
            result = cloud_mask(ds)

        assert result.values.all()

    def test_shape_mismatch_falls_through_to_allTrue(self) -> None:
        ds = make_spectral_dataset(shape=(3, 3))
        # Return a mask with wrong shape
        wrong_shape_flag = np.zeros((5, 5), dtype=np.int8)

        with patch("tanager.masks._read_beta_cirrus_from_hdf5", return_value=wrong_shape_flag):
            result = cloud_mask(ds, filepath="/fake/file.h5")

        # Shape mismatch: fall through to all-True
        assert result.values.all()


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
