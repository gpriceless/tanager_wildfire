"""Tests for tanager.spectral band selection, masking, and spectral indices."""

from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr

from tanager.config import BAD_BAND_RANGES
from tanager.spectral import (
    clamp_reflectance,
    continuum_removal,
    dnbr,
    mask_bad_bands,
    nbr,
    ndvi,
    ndwi,
    select_bands,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_dataset(wavelengths: np.ndarray) -> xr.Dataset:
    """Build a minimal synthetic Dataset with a reflectance variable."""
    n = len(wavelengths)
    data = np.ones((n, 4, 4), dtype=np.float32)
    return xr.Dataset(
        {"reflectance": (["wavelength", "y", "x"], data)},
        coords={"wavelength": wavelengths},
    )


@pytest.fixture
def simple_ds() -> xr.Dataset:
    """Dataset with 10 evenly-spaced bands from 400 to 900 nm."""
    return make_dataset(np.linspace(400, 900, 10))


@pytest.fixture
def tanager_ds() -> xr.Dataset:
    """Synthetic 426-band dataset matching Tanager-1 spectral range."""
    wavelengths = np.linspace(380, 2500, 426)
    data = np.random.default_rng(42).random((426, 50, 50)).astype(np.float32)
    return xr.Dataset(
        {"reflectance": (["wavelength", "y", "x"], data)},
        coords={"wavelength": wavelengths},
    )


# ---------------------------------------------------------------------------
# select_bands — range mode
# ---------------------------------------------------------------------------


class TestSelectBandsRange:
    def test_returns_bands_within_range(self, simple_ds: xr.Dataset) -> None:
        # Bands at 400, 455.5, 511.1, 566.7, 622.2 nm are below 650
        result = select_bands(simple_ds, min_wl=400, max_wl=650)
        wl = result.coords["wavelength"].values
        assert all(400 <= w <= 650 for w in wl)

    def test_excludes_bands_outside_range(self, simple_ds: xr.Dataset) -> None:
        result = select_bands(simple_ds, min_wl=500, max_wl=700)
        wl = result.coords["wavelength"].values
        assert all(500 <= w <= 700 for w in wl)

    def test_returns_dataset(self, simple_ds: xr.Dataset) -> None:
        result = select_bands(simple_ds, min_wl=400, max_wl=900)
        assert isinstance(result, xr.Dataset)

    def test_does_not_modify_input(self, simple_ds: xr.Dataset) -> None:
        original_size = simple_ds.sizes["wavelength"]
        select_bands(simple_ds, min_wl=500, max_wl=700)
        assert simple_ds.sizes["wavelength"] == original_size

    def test_exact_boundary_bands_included(self) -> None:
        wl = np.array([400.0, 500.0, 600.0, 700.0, 800.0])
        ds = make_dataset(wl)
        result = select_bands(ds, min_wl=500.0, max_wl=700.0)
        matched = result.coords["wavelength"].values
        np.testing.assert_array_equal(matched, [500.0, 600.0, 700.0])

    def test_raises_when_no_bands_in_range(self, simple_ds: xr.Dataset) -> None:
        with pytest.raises(ValueError, match="No bands found"):
            select_bands(simple_ds, min_wl=1000, max_wl=1200)

    def test_raises_when_range_out_of_dataset(self) -> None:
        ds = make_dataset(np.array([600.0, 700.0, 800.0]))
        with pytest.raises(ValueError, match="No bands found"):
            select_bands(ds, min_wl=200, max_wl=300)

    def test_raises_with_only_min_wl(self, simple_ds: xr.Dataset) -> None:
        with pytest.raises(ValueError):
            select_bands(simple_ds, min_wl=400)

    def test_raises_with_only_max_wl(self, simple_ds: xr.Dataset) -> None:
        with pytest.raises(ValueError):
            select_bands(simple_ds, max_wl=900)


# ---------------------------------------------------------------------------
# select_bands — nearest-neighbor mode
# ---------------------------------------------------------------------------


class TestSelectBandsNearest:
    def test_returns_tuple(self, simple_ds: xr.Dataset) -> None:
        result = select_bands(simple_ds, wavelengths=[500, 700])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_tuple_has_dataset_and_array(self, simple_ds: xr.Dataset) -> None:
        subset, matched = select_bands(simple_ds, wavelengths=[500, 700])
        assert isinstance(subset, xr.Dataset)
        assert isinstance(matched, np.ndarray)

    def test_matched_wavelengths_from_dataset(self, simple_ds: xr.Dataset) -> None:
        _, matched = select_bands(simple_ds, wavelengths=[500, 700])
        available = simple_ds.coords["wavelength"].values
        for wl in matched:
            assert wl in available

    def test_nearest_band_selected(self) -> None:
        wl = np.array([400.0, 500.0, 600.0, 700.0, 800.0])
        ds = make_dataset(wl)
        _, matched = select_bands(ds, wavelengths=[520.0])
        assert matched[0] == pytest.approx(500.0, abs=60.0)

    def test_single_wavelength(self, simple_ds: xr.Dataset) -> None:
        subset, matched = select_bands(simple_ds, wavelengths=[700])
        assert subset.sizes["wavelength"] == 1
        assert len(matched) == 1

    def test_does_not_modify_input(self, simple_ds: xr.Dataset) -> None:
        original_size = simple_ds.sizes["wavelength"]
        select_bands(simple_ds, wavelengths=[500, 700])
        assert simple_ds.sizes["wavelength"] == original_size


# ---------------------------------------------------------------------------
# select_bands — invalid argument combinations
# ---------------------------------------------------------------------------


class TestSelectBandsValidation:
    def test_raises_both_modes_specified(self, simple_ds: xr.Dataset) -> None:
        with pytest.raises(ValueError, match="not both"):
            select_bands(simple_ds, min_wl=400, max_wl=900, wavelengths=[500])

    def test_raises_neither_mode_specified(self, simple_ds: xr.Dataset) -> None:
        with pytest.raises(ValueError):
            select_bands(simple_ds)


# ---------------------------------------------------------------------------
# mask_bad_bands — default zones
# ---------------------------------------------------------------------------


class TestMaskBadBandsDefaults:
    def test_returns_dataset(self, tanager_ds: xr.Dataset) -> None:
        result = mask_bad_bands(tanager_ds)
        assert isinstance(result, xr.Dataset)

    def test_band_count_approximate_range(self, tanager_ds: xr.Dataset) -> None:
        result = mask_bad_bands(tanager_ds)
        n = result.sizes["wavelength"]
        # linspace(380, 2500, 426) gives ~328 after masking; real Tanager data
        # gives 330-346.  Accept 310-360 in tests to cover both cases.
        assert 310 <= n <= 360, f"Unexpected band count after masking: {n}"

    def test_wavelength_coordinate_sorted(self, tanager_ds: xr.Dataset) -> None:
        result = mask_bad_bands(tanager_ds)
        wl = result.coords["wavelength"].values
        assert np.all(np.diff(wl) > 0), "Wavelength coordinate is not sorted ascending"

    def test_removes_sensor_edge_bands(self, tanager_ds: xr.Dataset) -> None:
        result = mask_bad_bands(tanager_ds)
        wl = result.coords["wavelength"].values
        assert not np.any(wl <= 400), "Sensor-edge bands <=400 nm not removed"

    def test_removes_water_vapour_band1(self, tanager_ds: xr.Dataset) -> None:
        result = mask_bad_bands(tanager_ds)
        wl = result.coords["wavelength"].values
        assert not np.any((wl >= 1340) & (wl <= 1480)), "Water vapour zone 1340-1480 not removed"

    def test_removes_water_vapour_band2(self, tanager_ds: xr.Dataset) -> None:
        result = mask_bad_bands(tanager_ds)
        wl = result.coords["wavelength"].values
        # LGT-301: zone widened to 1780-1970 to align with sensor good_wavelengths
        # which flags ~1782.58-1967.21 nm.
        assert not np.any((wl >= 1780) & (wl <= 1970)), "Water vapour zone 1780-1970 not removed"

    def test_removes_co2_h2o_bands(self, tanager_ds: xr.Dataset) -> None:
        result = mask_bad_bands(tanager_ds)
        wl = result.coords["wavelength"].values
        assert not np.any((wl >= 2350) & (wl <= 2500)), "CO2/H2O zone 2350-2500 not removed"

    def test_does_not_modify_input(self, tanager_ds: xr.Dataset) -> None:
        original_size = tanager_ds.sizes["wavelength"]
        mask_bad_bands(tanager_ds)
        assert tanager_ds.sizes["wavelength"] == original_size

    def test_logs_band_count(self, tanager_ds: xr.Dataset, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        with caplog.at_level(logging.INFO, logger="tanager.spectral"):
            mask_bad_bands(tanager_ds)
        assert any("remaining" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# mask_bad_bands — custom zones
# ---------------------------------------------------------------------------


class TestMaskBadBandsCustomZones:
    def test_custom_zones_replace_defaults(self) -> None:
        wl = np.array([300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
        ds = make_dataset(wl)
        result = mask_bad_bands(ds, zones=[(300, 350)])
        matched = result.coords["wavelength"].values
        assert 400.0 in matched
        assert 300.0 not in matched

    def test_default_bad_bands_survive_custom_zones(self) -> None:
        wl = np.array([380.0, 400.0, 1400.0, 1500.0, 2200.0])
        ds = make_dataset(wl)
        result = mask_bad_bands(ds, zones=[(0, 390)])
        matched = result.coords["wavelength"].values
        # 380 excluded, 1400 and 1500 and 2200 survive (not in custom zones)
        assert 380.0 not in matched
        assert 1400.0 in matched
        assert 2200.0 in matched

    def test_empty_zones_keeps_all_bands(self) -> None:
        ds = make_dataset(np.linspace(400, 900, 10))
        result = mask_bad_bands(ds, zones=[])
        assert result.sizes["wavelength"] == 10

    def test_custom_zones_result_is_sorted(self) -> None:
        wl = np.linspace(400, 2500, 100)
        ds = make_dataset(wl)
        result = mask_bad_bands(ds, zones=[(1000, 1100), (1500, 1600)])
        wl_out = result.coords["wavelength"].values
        assert np.all(np.diff(wl_out) > 0)


# ---------------------------------------------------------------------------
# LGT-301 — widened water-vapour zone and good_wavelengths integration
# ---------------------------------------------------------------------------


class TestBadBandRangesAlignment:
    """Verify BAD_BAND_RANGES covers the sensor-flagged ranges (LGT-301).

    The sensor flags 58 bands as bad: 1342.41-1437.55 nm and 1782.58-1967.21 nm.
    BAD_BAND_RANGES is intentionally a small superset so wavelength-only
    masking remains correct for files that lack good_wavelengths metadata.
    """

    def test_water_vapour_band1_covers_sensor_range(self) -> None:
        zones = [z for z in BAD_BAND_RANGES if z[0] >= 1300 and z[1] <= 1500]
        assert len(zones) == 1
        low, high = zones[0]
        assert low <= 1342.41 and high >= 1437.55

    def test_water_vapour_band2_covers_sensor_range(self) -> None:
        # Pre-fix the zone was (1790, 1960), missing 1782-1790 and 1960-1967.
        zones = [z for z in BAD_BAND_RANGES if z[0] >= 1700 and z[1] <= 2000]
        assert len(zones) == 1
        low, high = zones[0]
        assert low <= 1782.58, f"Band 2 lower bound {low} misses 1782.58 nm"
        assert high >= 1967.21, f"Band 2 upper bound {high} misses 1967.21 nm"


class TestMaskBadBandsGoodWavelengthsCoord:
    """When the dataset carries a ``good_wavelengths`` coord, it is honoured."""

    def _ds_with_good(self, n: int = 426) -> xr.Dataset:
        wavelengths = np.linspace(380, 2500, n)
        rng = np.random.default_rng(0)
        data = rng.random((n, 3, 3)).astype(np.float32)
        good = np.ones(n, dtype=np.uint8)
        # Sensor flags two bands inside the kept range as bad.
        target_idx = [int(np.argmin(np.abs(wavelengths - 700.0))),
                      int(np.argmin(np.abs(wavelengths - 800.0)))]
        good[target_idx] = 0
        return xr.Dataset(
            {"reflectance": (["wavelength", "y", "x"], data)},
            coords={
                "wavelength": wavelengths,
                "good_wavelengths": (("wavelength",), good),
            },
        )

    def test_sensor_flagged_bands_excluded(self) -> None:
        ds = self._ds_with_good()
        result = mask_bad_bands(ds)
        wl_out = result.coords["wavelength"].values
        # Find the input wavelengths sensor-marked bad and assert they are gone.
        bad = ds.coords["wavelength"].values[
            ds.coords["good_wavelengths"].values == 0
        ]
        for w in bad:
            assert not np.any(np.isclose(wl_out, w)), (
                f"Sensor-flagged bad band {w} nm not removed"
            )

    def test_default_zones_still_applied(self) -> None:
        ds = self._ds_with_good()
        result = mask_bad_bands(ds)
        wl_out = result.coords["wavelength"].values
        # The default zones must still be excluded even when sensor mask is also active.
        assert not np.any(wl_out <= 400)
        assert not np.any((wl_out >= 1340) & (wl_out <= 1480))
        assert not np.any((wl_out >= 1780) & (wl_out <= 1970))
        assert not np.any((wl_out >= 2350) & (wl_out <= 2500))

class TestMaskBadBandsHdf5Filepath:
    """Optional ``hdf5_filepath`` parameter reads good_wavelengths from disk."""

    def _make_synthetic_hdf5(
        self,
        tmp_path,
        good_wavelengths: np.ndarray,
        wavelengths: np.ndarray,
    ) -> str:
        import h5py

        path = tmp_path / "synth_ortho_sr.h5"
        with h5py.File(path, "w") as h5:
            grp = h5.create_group("HDFEOS/GRIDS/HYP/Data Fields")
            sr = grp.create_dataset(
                "surface_reflectance",
                shape=(len(wavelengths), 2, 2),
                dtype="float32",
            )
            sr.attrs["wavelengths"] = wavelengths.astype(np.float64)
            sr.attrs["good_wavelengths"] = good_wavelengths.astype(np.uint8)
        return str(path)

    def test_reads_good_wavelengths_from_file(self, tmp_path) -> None:
        wavelengths = np.linspace(380, 2500, 426)
        good = np.ones(426, dtype=np.uint8)
        target_idx = int(np.argmin(np.abs(wavelengths - 750.0)))
        good[target_idx] = 0
        h5_path = self._make_synthetic_hdf5(tmp_path, good, wavelengths)

        ds = xr.Dataset(
            {"reflectance": (["wavelength", "y", "x"],
                             np.ones((426, 2, 2), dtype=np.float32))},
            coords={"wavelength": wavelengths},
        )
        result = mask_bad_bands(ds, hdf5_filepath=h5_path)
        wl_out = result.coords["wavelength"].values
        flagged = wavelengths[target_idx]
        assert not np.any(np.isclose(wl_out, flagged)), (
            f"Sensor-flagged band {flagged} not removed when read from HDF5"
        )

    def test_hdf5_overrides_dataset_coord(self, tmp_path) -> None:
        """``hdf5_filepath`` takes precedence over the dataset's own coord."""
        wavelengths = np.linspace(380, 2500, 426)
        # Dataset coord flags 600 nm as bad.
        ds_good = np.ones(426, dtype=np.uint8)
        ds_idx = int(np.argmin(np.abs(wavelengths - 600.0)))
        ds_good[ds_idx] = 0
        # HDF5 flag flags 700 nm as bad.
        file_good = np.ones(426, dtype=np.uint8)
        file_idx = int(np.argmin(np.abs(wavelengths - 700.0)))
        file_good[file_idx] = 0
        h5_path = self._make_synthetic_hdf5(tmp_path, file_good, wavelengths)

        ds = xr.Dataset(
            {"reflectance": (["wavelength", "y", "x"],
                             np.ones((426, 2, 2), dtype=np.float32))},
            coords={
                "wavelength": wavelengths,
                "good_wavelengths": (("wavelength",), ds_good),
            },
        )
        result = mask_bad_bands(ds, hdf5_filepath=h5_path)
        wl_out = result.coords["wavelength"].values
        # File-flagged band must be removed.
        assert not np.any(np.isclose(wl_out, wavelengths[file_idx]))
        # Dataset-coord-flagged band must remain (file overrides coord).
        assert np.any(np.isclose(wl_out, wavelengths[ds_idx]))

    def test_missing_file_raises_value_error(self, tmp_path) -> None:
        wavelengths = np.linspace(380, 2500, 10)
        ds = xr.Dataset(
            {"reflectance": (["wavelength", "y", "x"],
                             np.zeros((10, 2, 2), dtype=np.float32))},
            coords={"wavelength": wavelengths},
        )
        with pytest.raises(ValueError, match="Cannot read"):
            mask_bad_bands(
                ds, hdf5_filepath=str(tmp_path / "nonexistent.h5")
            )

    def test_hdf5_length_mismatch_raises(self, tmp_path) -> None:
        wavelengths_file = np.linspace(380, 2500, 100)
        good = np.ones(100, dtype=np.uint8)
        h5_path = self._make_synthetic_hdf5(tmp_path, good, wavelengths_file)

        # Dataset has different length.
        wavelengths_ds = np.linspace(380, 2500, 50)
        ds = xr.Dataset(
            {"reflectance": (["wavelength", "y", "x"],
                             np.zeros((50, 2, 2), dtype=np.float32))},
            coords={"wavelength": wavelengths_ds},
        )
        with pytest.raises(ValueError, match="does not match dataset"):
            mask_bad_bands(ds, hdf5_filepath=h5_path)


# ---------------------------------------------------------------------------
# LGT-301 — Real-data integration: verify sensor alignment against HDF5 files
# ---------------------------------------------------------------------------


_REAL_FIRE_DIR = "data/raw/fire"
_REAL_FIRE_SCENES = [
    "20241215_185916_33_4001_ortho_sr_hdf5.h5",
    "20250123_185507_64_4001_ortho_sr_hdf5.h5",
    "20250407_192235_24_4001_ortho_sr_hdf5.h5",
]


def _real_fire_scene_paths() -> list[str]:
    base = os.path.join(os.getcwd(), _REAL_FIRE_DIR)
    return [
        os.path.join(base, name) for name in _REAL_FIRE_SCENES
        if os.path.exists(os.path.join(base, name))
    ]


@pytest.mark.skipif(
    not _real_fire_scene_paths(),
    reason="Real ortho SR HDF5 scenes not present in data/raw/fire/",
)
class TestMaskBadBandsRealOrthoHDF5:
    """LGT-301: mask_bad_bands aligns with sensor metadata on real scenes.

    Validates that the widened BAD_BAND_RANGES plus the good_wavelengths
    integration produces the same set of kept bands as the sensor's own
    flag array would on its own (modulo the 4 wavelength-zone exclusions
    that are also intentional defaults).
    """

    def _read_sensor_metadata(self, path: str):
        import h5py

        with h5py.File(path, "r") as h5:
            sr = h5["HDFEOS/GRIDS/HYP/Data Fields/surface_reflectance"]
            wavelengths = np.asarray(sr.attrs["wavelengths"], dtype=np.float64)
            good = np.asarray(sr.attrs["good_wavelengths"], dtype=np.uint8).astype(bool)
            fwhm = np.asarray(sr.attrs["fwhm"], dtype=np.float64)
        return wavelengths, good, fwhm

    def _build_dataset(self, wavelengths: np.ndarray) -> xr.Dataset:
        return xr.Dataset(
            {"reflectance": (["wavelength", "y", "x"],
                             np.zeros((len(wavelengths), 2, 2), dtype=np.float32))},
            coords={"wavelength": wavelengths},
        )

    def test_hdf5_path_matches_sensor_flags(self) -> None:
        """Bands kept after masking must include no sensor-flagged-bad bands."""
        for path in _real_fire_scene_paths():
            wavelengths, good, _ = self._read_sensor_metadata(path)
            ds = self._build_dataset(wavelengths)

            result = mask_bad_bands(ds, hdf5_filepath=path)
            kept = result.coords["wavelength"].values

            sensor_bad_wl = wavelengths[~good]
            for w in sensor_bad_wl:
                assert not np.any(np.isclose(kept, w)), (
                    f"{path}: sensor-flagged bad band {w:.2f} nm survived masking"
                )

    def test_widened_zone_covers_all_sensor_bad_bands_in_range(self) -> None:
        """Wavelength-only masking (no good_wavelengths) excludes all sensor-bad bands.

        Guards against regression: the previous (1790, 1960) range left the
        bands at 1782-1790 nm and 1960-1967 nm in the kept set even though
        the sensor flagged them as bad.
        """
        for path in _real_fire_scene_paths():
            wavelengths, good, _ = self._read_sensor_metadata(path)
            ds = self._build_dataset(wavelengths)

            # Use only BAD_BAND_RANGES (no hdf5_filepath, no coord) to verify
            # the wavelength zones are wide enough on their own.
            result = mask_bad_bands(ds)
            kept = result.coords["wavelength"].values

            sensor_bad_in_swir = wavelengths[(~good) & (wavelengths >= 1700) & (wavelengths <= 2000)]
            assert sensor_bad_in_swir.size > 0, (
                f"{path}: expected sensor-flagged bad bands in 1700-2000 nm"
            )
            for w in sensor_bad_in_swir:
                assert not np.any(np.isclose(kept, w)), (
                    f"{path}: SWIR band {w:.2f} nm survived BAD_BAND_RANGES masking "
                    f"(LGT-301 regression — water-vapour zone too narrow)"
                )

    def test_real_scene_yields_expected_band_count(self) -> None:
        """Real Tanager scenes should retain ~330-346 bands after full masking."""
        for path in _real_fire_scene_paths():
            wavelengths, _, _ = self._read_sensor_metadata(path)
            ds = self._build_dataset(wavelengths)

            result = mask_bad_bands(ds, hdf5_filepath=path)
            n = result.sizes["wavelength"]
            assert 320 <= n <= 360, (
                f"{path}: unexpected band count {n} after masking; "
                f"expected ~330-346 for real Tanager data"
            )

    def test_per_band_fwhm_varies(self) -> None:
        """Sanity check: per-band FWHM is non-constant across 5.20-6.81 nm.

        The board directive notes FWHM varies per band rather than the
        SENSOR.spectral_resolution_nm constant (5 nm). This test pins that
        observation so any future change to the loader/spec catches it.
        """
        for path in _real_fire_scene_paths():
            _, _, fwhm = self._read_sensor_metadata(path)
            assert fwhm.shape == (426,)
            assert fwhm.min() >= 5.0 and fwhm.max() <= 7.0
            assert fwhm.std() > 0.1, (
                f"{path}: FWHM appears constant — expected per-band variation"
            )


# ---------------------------------------------------------------------------
# Task 6 explicit verification
# ---------------------------------------------------------------------------


def test_task6_426_band_verification() -> None:
    """Task 6: verify mask_bad_bands on synthetic 426-band dataset."""
    wavelengths = np.linspace(380, 2500, 426)
    data = np.random.rand(426, 50, 50).astype(np.float32)
    ds = xr.Dataset(
        {"reflectance": (["wavelength", "y", "x"], data)},
        coords={"wavelength": wavelengths},
    )

    result = mask_bad_bands(ds)
    n_bands = result.sizes["wavelength"]
    wl_vals = result.coords["wavelength"].values

    # Accept ~330-346 from spec; linspace gives 328, so use 310-360 to cover both
    assert 310 <= n_bands <= 360, f"Band count {n_bands} outside expected range"
    assert np.all(np.diff(wl_vals) > 0), "Wavelength coordinate is not sorted"


# ---------------------------------------------------------------------------
# Helpers for index tests
# ---------------------------------------------------------------------------


def make_tanager_ds(
    nir_val: float = 0.6,
    swir2_val: float = 0.2,
    red_val: float = 0.1,
    green_val: float = 0.3,
    shape: tuple[int, int] = (4, 4),
) -> xr.Dataset:
    """Build a synthetic Tanager-1 dataset with controlled band values.

    Sets NIR (860 nm), SWIR2 (2200 nm), Red (660 nm), and Green (560 nm) to
    uniform constant values so indices can be analytically verified.
    """
    wavelengths = np.linspace(380, 2500, 426)
    n_wl = len(wavelengths)
    ny, nx = shape

    data = np.zeros((n_wl, ny, nx), dtype=np.float32)

    # Assign constant reflectance to the bands nearest to each target
    def _nearest_idx(wl_target: float) -> int:
        return int(np.argmin(np.abs(wavelengths - wl_target)))

    data[_nearest_idx(860), :, :] = nir_val
    data[_nearest_idx(2200), :, :] = swir2_val
    data[_nearest_idx(660), :, :] = red_val
    data[_nearest_idx(560), :, :] = green_val

    return xr.Dataset(
        {"reflectance": (["wavelength", "y", "x"], data)},
        coords={"wavelength": wavelengths},
    )


# ---------------------------------------------------------------------------
# NBR
# ---------------------------------------------------------------------------


class TestNBR:
    def test_returns_dataarray(self) -> None:
        ds = make_tanager_ds()
        result = nbr(ds)
        assert isinstance(result, xr.DataArray)

    def test_correct_formula(self) -> None:
        # NBR = (NIR - SWIR2) / (NIR + SWIR2) = (0.6 - 0.2) / (0.6 + 0.2) = 0.5
        ds = make_tanager_ds(nir_val=0.6, swir2_val=0.2)
        result = nbr(ds)
        np.testing.assert_allclose(result.values, 0.5, atol=1e-5)

    def test_values_in_minus1_to_1(self) -> None:
        ds = make_tanager_ds(nir_val=0.4, swir2_val=0.6)
        result = nbr(ds)
        assert float(result.min()) >= -1.0
        assert float(result.max()) <= 1.0

    def test_nan_when_denominator_zero(self) -> None:
        ds = make_tanager_ds(nir_val=0.0, swir2_val=0.0)
        result = nbr(ds)
        assert np.all(np.isnan(result.values))

    def test_no_inf_values(self) -> None:
        ds = make_tanager_ds(nir_val=0.0, swir2_val=0.0)
        result = nbr(ds)
        assert not np.any(np.isinf(result.values))

    def test_spatial_dims_preserved(self) -> None:
        ds = make_tanager_ds(shape=(8, 6))
        result = nbr(ds)
        assert result.sizes.get("y") == 8
        assert result.sizes.get("x") == 6

    def test_does_not_modify_input(self) -> None:
        ds = make_tanager_ds()
        original_size = ds.sizes["wavelength"]
        nbr(ds)
        assert ds.sizes["wavelength"] == original_size


# ---------------------------------------------------------------------------
# NDVI
# ---------------------------------------------------------------------------


class TestNDVI:
    def test_returns_dataarray(self) -> None:
        ds = make_tanager_ds()
        result = ndvi(ds)
        assert isinstance(result, xr.DataArray)

    def test_correct_formula(self) -> None:
        # NDVI = (NIR - Red) / (NIR + Red) = (0.6 - 0.1) / (0.6 + 0.1) ≈ 0.714
        ds = make_tanager_ds(nir_val=0.6, red_val=0.1)
        result = ndvi(ds)
        expected = (0.6 - 0.1) / (0.6 + 0.1)
        np.testing.assert_allclose(result.values, expected, atol=1e-5)

    def test_values_in_minus1_to_1(self) -> None:
        ds = make_tanager_ds(nir_val=0.2, red_val=0.8)
        result = ndvi(ds)
        assert float(result.min()) >= -1.0
        assert float(result.max()) <= 1.0

    def test_nan_when_denominator_zero(self) -> None:
        ds = make_tanager_ds(nir_val=0.0, red_val=0.0)
        result = ndvi(ds)
        assert np.all(np.isnan(result.values))

    def test_no_inf_values(self) -> None:
        ds = make_tanager_ds(nir_val=0.0, red_val=0.0)
        result = ndvi(ds)
        assert not np.any(np.isinf(result.values))


# ---------------------------------------------------------------------------
# NDWI
# ---------------------------------------------------------------------------


class TestNDWI:
    def test_returns_dataarray(self) -> None:
        ds = make_tanager_ds()
        result = ndwi(ds)
        assert isinstance(result, xr.DataArray)

    def test_correct_formula(self) -> None:
        # NDWI = (Green - NIR) / (Green + NIR) = (0.3 - 0.6) / (0.3 + 0.6) = -0.333...
        ds = make_tanager_ds(green_val=0.3, nir_val=0.6)
        result = ndwi(ds)
        expected = (0.3 - 0.6) / (0.3 + 0.6)
        np.testing.assert_allclose(result.values, expected, atol=1e-5)

    def test_values_in_minus1_to_1(self) -> None:
        ds = make_tanager_ds(green_val=0.8, nir_val=0.2)
        result = ndwi(ds)
        assert float(result.min()) >= -1.0
        assert float(result.max()) <= 1.0

    def test_nan_when_denominator_zero(self) -> None:
        ds = make_tanager_ds(nir_val=0.0, green_val=0.0)
        result = ndwi(ds)
        assert np.all(np.isnan(result.values))

    def test_no_inf_values(self) -> None:
        ds = make_tanager_ds(nir_val=0.0, green_val=0.0)
        result = ndwi(ds)
        assert not np.any(np.isinf(result.values))


# ---------------------------------------------------------------------------
# dNBR
# ---------------------------------------------------------------------------


class TestDNBR:
    def test_returns_dataarray(self) -> None:
        pre = make_tanager_ds(nir_val=0.6, swir2_val=0.2)
        post = make_tanager_ds(nir_val=0.3, swir2_val=0.5)
        result = dnbr(pre, post)
        assert isinstance(result, xr.DataArray)

    def test_correct_formula(self) -> None:
        # NBR_pre = (0.6 - 0.2) / (0.6 + 0.2) = 0.5
        # NBR_post = (0.3 - 0.5) / (0.3 + 0.5) = -0.25
        # dNBR = 0.5 - (-0.25) = 0.75
        pre = make_tanager_ds(nir_val=0.6, swir2_val=0.2)
        post = make_tanager_ds(nir_val=0.3, swir2_val=0.5)
        result = dnbr(pre, post)
        expected = 0.5 - (-0.25)
        np.testing.assert_allclose(result.values, expected, atol=1e-5)

    def test_raises_on_spatial_dim_mismatch_when_auto_align_disabled(self) -> None:
        # auto_align=True (default) routes mismatched grids through
        # reproject_to_common_grid; the legacy "must match" error path is
        # preserved behind auto_align=False.
        pre = make_tanager_ds(shape=(4, 4))
        post = make_tanager_ds(shape=(8, 4))
        with pytest.raises(ValueError, match="match|auto_align"):
            dnbr(pre, post, auto_align=False)

    def test_raises_on_x_dim_mismatch_when_auto_align_disabled(self) -> None:
        pre = make_tanager_ds(shape=(4, 4))
        post = make_tanager_ds(shape=(4, 8))
        with pytest.raises(ValueError, match="match|auto_align"):
            dnbr(pre, post, auto_align=False)

    def test_zero_when_pre_equals_post(self) -> None:
        ds = make_tanager_ds(nir_val=0.5, swir2_val=0.3)
        result = dnbr(ds, ds)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# continuum_removal
# ---------------------------------------------------------------------------


class TestContinuumRemoval:
    def test_returns_dataarray(self) -> None:
        ds = make_tanager_ds()
        result = continuum_removal(ds)
        assert isinstance(result, xr.DataArray)

    def test_values_in_0_to_1(self) -> None:
        rng = np.random.default_rng(0)
        wavelengths = np.linspace(400, 2400, 426)
        data = rng.random((426, 4, 4)).astype(np.float32)
        ds = xr.Dataset(
            {"reflectance": (["wavelength", "y", "x"], data)},
            coords={"wavelength": wavelengths},
        )
        result = continuum_removal(ds)
        valid = result.values[~np.isnan(result.values)]
        assert np.all(valid >= 0.0), "Values below 0 found"
        assert np.all(valid <= 1.0), "Values above 1 found"

    def test_wavelength_range_subset(self) -> None:
        rng = np.random.default_rng(1)
        wavelengths = np.linspace(400, 2400, 426)
        data = rng.random((426, 4, 4)).astype(np.float32)
        ds = xr.Dataset(
            {"reflectance": (["wavelength", "y", "x"], data)},
            coords={"wavelength": wavelengths},
        )
        result = continuum_removal(ds, wavelength_range=(600.0, 900.0))
        wl_out = result.coords["wavelength"].values
        assert np.all(wl_out >= 600.0)
        assert np.all(wl_out <= 900.0)

    def test_output_wavelength_dim_present(self) -> None:
        ds = make_tanager_ds()
        result = continuum_removal(ds)
        assert "wavelength" in result.dims

    def test_spatial_dims_preserved(self) -> None:
        rng = np.random.default_rng(2)
        wavelengths = np.linspace(400, 2400, 426)
        data = rng.random((426, 6, 8)).astype(np.float32)
        ds = xr.Dataset(
            {"reflectance": (["wavelength", "y", "x"], data)},
            coords={"wavelength": wavelengths},
        )
        result = continuum_removal(ds)
        assert result.sizes.get("y") == 6
        assert result.sizes.get("x") == 8

    def test_does_not_modify_input(self) -> None:
        ds = make_tanager_ds()
        original_size = ds.sizes["wavelength"]
        continuum_removal(ds)
        assert ds.sizes["wavelength"] == original_size


# ---------------------------------------------------------------------------
# Task 29 explicit verification: NBR on synthetic data
# ---------------------------------------------------------------------------


def test_task29_nbr_verification() -> None:
    """Task 29: NBR on synthetic data — values in [-1, 1], NaN where denom=0."""
    # Arrange: 5x5 spatial grid; set NIR and SWIR2 uniformly (denom > 0)
    wavelengths = np.linspace(380, 2500, 426)
    n_wl = len(wavelengths)
    data = np.zeros((n_wl, 5, 5), dtype=np.float32)

    nir_idx = int(np.argmin(np.abs(wavelengths - 860)))
    swir2_idx = int(np.argmin(np.abs(wavelengths - 2200)))
    data[nir_idx, :, :] = 0.7
    data[swir2_idx, :, :] = 0.3

    ds = xr.Dataset(
        {"reflectance": (["wavelength", "y", "x"], data)},
        coords={"wavelength": wavelengths},
    )

    result = nbr(ds)
    vals = result.values

    # All pixels have valid denominator — no NaN expected
    assert not np.any(np.isnan(vals)), "Unexpected NaN in NBR result"
    assert np.all(vals >= -1.0) and np.all(vals <= 1.0), "NBR out of [-1, 1]"
    expected = (0.7 - 0.3) / (0.7 + 0.3)
    np.testing.assert_allclose(vals, expected, atol=1e-5)

    # Now test NaN-on-zero-denominator by zeroing both bands
    data[nir_idx, :, :] = 0.0
    data[swir2_idx, :, :] = 0.0
    ds_zero = xr.Dataset(
        {"reflectance": (["wavelength", "y", "x"], data)},
        coords={"wavelength": wavelengths},
    )
    result_zero = nbr(ds_zero)
    assert np.all(np.isnan(result_zero.values)), "Expected all NaN when denominator=0"
    assert not np.any(np.isinf(result_zero.values)), "Unexpected Inf in NBR result"


# ---------------------------------------------------------------------------
# clamp_reflectance + adversarial-input regression tests (LGT-311)
# ---------------------------------------------------------------------------


def _make_ds_with_band_array(
    wl_target: float,
    band_values: np.ndarray,
    *,
    other_target: float,
    other_values: np.ndarray,
) -> xr.Dataset:
    """Build a 426-band dataset where two specific bands carry adversarial values."""
    wavelengths = np.linspace(380, 2500, 426)
    ny, nx = band_values.shape
    data = np.zeros((426, ny, nx), dtype=np.float32)
    target_idx = int(np.argmin(np.abs(wavelengths - wl_target)))
    other_idx = int(np.argmin(np.abs(wavelengths - other_target)))
    data[target_idx] = band_values
    data[other_idx] = other_values
    return xr.Dataset(
        {"reflectance": (["wavelength", "y", "x"], data)},
        coords={"wavelength": wavelengths},
    )


class TestClampReflectance:
    def test_dataset_clamps_to_range(self) -> None:
        wavelengths = np.linspace(380, 2500, 5)
        data = np.array([[-0.5], [0.2], [0.7], [1.5], [2.0]], dtype=np.float32)[
            :, :, None
        ]
        ds = xr.Dataset(
            {"reflectance": (["wavelength", "y", "x"], data)},
            coords={"wavelength": wavelengths},
        )
        out = clamp_reflectance(ds)
        out_vals = out["reflectance"].values
        assert float(out_vals.min()) >= 0.0
        assert float(out_vals.max()) <= 1.0

    def test_dataarray_clamps_to_range(self) -> None:
        arr = xr.DataArray(np.array([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32))
        out = clamp_reflectance(arr)
        np.testing.assert_array_equal(out.values, [0.0, 0.0, 0.5, 1.0, 1.0])

    def test_does_not_modify_input(self) -> None:
        arr = xr.DataArray(np.array([-1.0, 2.0], dtype=np.float32))
        before = arr.values.copy()
        clamp_reflectance(arr)
        np.testing.assert_array_equal(arr.values, before)

    def test_dataset_preserves_other_variables(self) -> None:
        wavelengths = np.linspace(380, 2500, 3)
        data = np.array([[-0.2], [0.5], [1.4]], dtype=np.float32)[:, :, None]
        flag = np.array([1, 0, 1], dtype=np.uint8)
        ds = xr.Dataset(
            {
                "reflectance": (["wavelength", "y", "x"], data),
                "qa_flag": (["wavelength"], flag),
            },
            coords={"wavelength": wavelengths},
        )
        out = clamp_reflectance(ds)
        assert "qa_flag" in out
        np.testing.assert_array_equal(out["qa_flag"].values, flag)

    def test_custom_bounds(self) -> None:
        arr = xr.DataArray(np.array([-2.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32))
        out = clamp_reflectance(arr, vmin=-1.0, vmax=0.5)
        np.testing.assert_array_equal(out.values, [-1.0, -0.5, 0.0, 0.5, 0.5])

    def test_invalid_bounds(self) -> None:
        arr = xr.DataArray(np.array([0.5], dtype=np.float32))
        with pytest.raises(ValueError, match="must be <="):
            clamp_reflectance(arr, vmin=1.0, vmax=0.0)

    def test_dataset_without_reflectance_raises(self) -> None:
        ds = xr.Dataset({"foo": (["x"], np.array([1.0]))})
        with pytest.raises(ValueError, match="reflectance"):
            clamp_reflectance(ds)

    def test_no_clamp_when_in_range(self) -> None:
        arr = xr.DataArray(np.array([0.0, 0.25, 0.75, 1.0], dtype=np.float32))
        out = clamp_reflectance(arr)
        np.testing.assert_array_equal(out.values, arr.values)


class TestAdversarialReflectance:
    """Regression tests for ISOFIT real-data reflectance pathologies (LGT-311).

    Real Tanager surface reflectance contains a substantial fraction of
    negative values and the occasional extreme outlier. Without the epsilon
    guard and clamp, these produce NBR=-99, NDVI=5, etc. — physically
    impossible.
    """

    def test_nbr_negative_reflectance_clamped_then_in_range(self) -> None:
        # NIR strongly negative, SWIR2 slightly positive — without clamp this
        # would yield (-37.6 - 0.05) / (-37.6 + 0.05) ≈ +1.003 (out of range
        # plus blown up by near-zero denom) or, post-clamp, (0 - 0.05) /
        # (0 + 0.05) = -1.0 (in-range).
        nir = np.full((4, 4), -37.6, dtype=np.float32)
        swir2 = np.full((4, 4), 0.05, dtype=np.float32)
        ds = _make_ds_with_band_array(
            860.0, nir, other_target=2200.0, other_values=swir2
        )
        result = nbr(ds)
        valid = result.values[~np.isnan(result.values)]
        assert np.all(valid >= -1.0)
        assert np.all(valid <= 1.0)
        assert not np.any(np.isinf(result.values))

    def test_ndvi_extreme_positive_outlier_clamped(self) -> None:
        # NIR=39.2 (extreme), Red=0.05 -> raw ratio diverges from [-1, 1].
        # Clamped: (1.0 - 0.05) / (1.0 + 0.05) ≈ 0.905, in range.
        nir = np.full((3, 3), 39.2, dtype=np.float32)
        red = np.full((3, 3), 0.05, dtype=np.float32)
        ds = _make_ds_with_band_array(
            860.0, nir, other_target=660.0, other_values=red
        )
        result = ndvi(ds)
        vals = result.values[~np.isnan(result.values)]
        assert np.all(vals >= -1.0)
        assert np.all(vals <= 1.0)
        assert not np.any(np.isinf(result.values))

    def test_near_zero_denominator_produces_nan_not_inf(self) -> None:
        # Both bands at +1e-4 — sum is 2e-4, which is below the 1e-3 epsilon
        # guard.  Without the guard this would divide a tiny number by a tiny
        # number and amplify floating-point noise.
        nir = np.full((4, 4), 1e-4, dtype=np.float32)
        swir2 = np.full((4, 4), 1e-4, dtype=np.float32)
        ds = _make_ds_with_band_array(
            860.0, nir, other_target=2200.0, other_values=swir2
        )
        result = nbr(ds)
        assert np.all(np.isnan(result.values))
        assert not np.any(np.isinf(result.values))

    def test_negative_denominator_below_epsilon_produces_nan(self) -> None:
        # Both bands negative -> denominator negative; |denominator| < epsilon
        # should still trigger the guard regardless of sign.
        nir = np.full((4, 4), -1e-4, dtype=np.float32)
        swir2 = np.full((4, 4), -1e-4, dtype=np.float32)
        ds = _make_ds_with_band_array(
            860.0, nir, other_target=2200.0, other_values=swir2
        )
        result = nbr(ds)
        assert np.all(np.isnan(result.values))
        assert not np.any(np.isinf(result.values))

    def test_mixed_adversarial_pixel_grid_all_in_range(self) -> None:
        # Build a 4x4 grid with a mix of pathological values: negatives,
        # near-zero, normal, and extreme positives.  After processing every
        # finite pixel must be in [-1, 1].
        nir = np.array(
            [
                [-37.6, -0.5, 0.0, 0.001],
                [0.05, 0.4, 0.6, 0.95],
                [1.0, 1.5, 5.0, 39.2],
                [-1e-4, 1e-4, 0.5, 0.5],
            ],
            dtype=np.float32,
        )
        swir2 = np.array(
            [
                [0.05, 0.05, 0.05, 0.001],
                [0.05, 0.2, 0.3, 0.4],
                [0.5, 0.5, 0.5, 0.5],
                [-1e-4, 1e-4, 0.5, 0.5001],
            ],
            dtype=np.float32,
        )
        ds = _make_ds_with_band_array(
            860.0, nir, other_target=2200.0, other_values=swir2
        )
        result = nbr(ds)
        finite = result.values[np.isfinite(result.values)]
        assert np.all(finite >= -1.0)
        assert np.all(finite <= 1.0)
        assert not np.any(np.isinf(result.values))

    def test_alternate_reflectance_variable_name(self) -> None:
        """Real ortho_sr scenes use ``surface_reflectance`` instead of ``reflectance``."""
        wavelengths = np.linspace(380, 2500, 426)
        data = np.zeros((426, 4, 4), dtype=np.float32)
        nir_idx = int(np.argmin(np.abs(wavelengths - 860)))
        swir2_idx = int(np.argmin(np.abs(wavelengths - 2200)))
        data[nir_idx, :, :] = 0.6
        data[swir2_idx, :, :] = 0.2
        sr_da = xr.DataArray(
            data, dims=("wavelength", "y", "x"), coords={"wavelength": wavelengths}
        )
        ds = xr.Dataset(
            {"surface_reflectance": sr_da, "toa_radiance": sr_da},
            attrs={"data_var": "surface_reflectance"},
        )
        result = nbr(ds)
        np.testing.assert_allclose(result.values, 0.5, atol=1e-5)

    def test_dnbr_propagates_clamp_via_nbr(self) -> None:
        # Pre-fire pathological, post-fire normal.  Without the clamp this
        # would produce wild dNBR values; with the clamp every output pixel
        # must remain finite and bounded.
        pre = _make_ds_with_band_array(
            860.0,
            np.full((3, 3), -37.6, dtype=np.float32),
            other_target=2200.0,
            other_values=np.full((3, 3), 0.05, dtype=np.float32),
        )
        post = _make_ds_with_band_array(
            860.0,
            np.full((3, 3), 0.6, dtype=np.float32),
            other_target=2200.0,
            other_values=np.full((3, 3), 0.2, dtype=np.float32),
        )
        result = dnbr(pre, post)
        # NBR is in [-1, 1], so dNBR ∈ [-2, 2].
        finite = result.values[np.isfinite(result.values)]
        assert np.all(finite >= -2.0)
        assert np.all(finite <= 2.0)
        assert not np.any(np.isinf(result.values))


# ---------------------------------------------------------------------------
# Real-data integration test (LGT-311 board directive)
#
# Skipped automatically when the post-fire ortho_sr HDF5 is not on disk so
# CI without bulk-data access still passes.
# ---------------------------------------------------------------------------

_REAL_POST_FIRE_PATH = os.path.join(
    "data", "raw", "fire", "20250123_185507_64_4001_ortho_sr_hdf5.h5"
)


@pytest.mark.skipif(
    not os.path.exists(_REAL_POST_FIRE_PATH),
    reason="real ortho_sr scene not available locally",
)
def test_real_scene_indices_are_in_physical_range() -> None:
    """Load a real Tanager ortho_sr scene and assert NBR/NDVI/NDWI ∈ [-1, 1].

    This is the integration test the LGT-311 board directive requested.
    Real ISOFIT surface reflectance contains ~7% negative values and rare
    extreme outliers; the epsilon guard plus clamp ensures every finite
    output pixel is in the physical [-1, 1] range and no infinities are
    produced.
    """
    from tanager.io import load_ortho_scene

    ds = load_ortho_scene(_REAL_POST_FIRE_PATH)
    for fn in (nbr, ndvi, ndwi):
        out = fn(ds)
        vals = out.values
        finite = vals[np.isfinite(vals)]
        assert not np.any(np.isinf(vals)), f"{fn.__name__}: produced infinity"
        assert finite.size > 0, f"{fn.__name__}: produced no finite pixels"
        assert finite.min() >= -1.0 - 1e-6
        assert finite.max() <= 1.0 + 1e-6
