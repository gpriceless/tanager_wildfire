"""Tests for spatial co-registration (LGT-313 / LGT-299).

Covers:
    * ``tanager.io.reproject_to_common_grid`` — alignment of multi-temporal
      Tanager scenes onto a single shared UTM grid.
    * ``tanager.spectral.dnbr`` — auto-alignment path when pre/post scenes
      have differing grids.

Synthetic ortho_sr HDF5 files are produced via the helper imported from
``tests.test_io`` so the tests exercise the same h5py code path
``load_ortho_scene`` uses on real Tanager products. Real-data integration is
gated behind ``os.path.exists`` checks so the suite still passes on machines
without the bulk fire data.

Test naming: <function>_<scenario>_<expected_outcome>
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tests.test_io import _write_synthetic_ortho_h5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_overlapping_ortho_pair(
    tmp_path: Path,
) -> tuple[Path, Path]:
    """Write two synthetic ortho_sr scenes with overlapping but different grids.

    Scene A: 5x4 pixel grid, UTM bounds (300_000, 4_000_000)–(304_500, 4_003_600)
    Scene B: 6x5 pixel grid, UTM bounds (302_000, 4_001_000)–(307_400, 4_005_500)

    The two rectangles overlap on a roughly 2_500 x 2_600 m patch, well above
    the 10% minimum-overlap threshold.
    """
    path_a = tmp_path / "scene_a.h5"
    path_b = tmp_path / "scene_b.h5"
    _write_synthetic_ortho_h5(
        path_a,
        n_bands=8,
        y_dim=4,
        x_dim=5,
        upper_left=(300_000.0, 4_003_600.0),
        lower_right=(304_500.0, 4_000_000.0),
    )
    _write_synthetic_ortho_h5(
        path_b,
        n_bands=8,
        y_dim=5,
        x_dim=6,
        upper_left=(302_000.0, 4_005_500.0),
        lower_right=(307_400.0, 4_001_000.0),
    )
    return path_a, path_b


def _write_disjoint_ortho_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Write two synthetic ortho_sr scenes with non-overlapping bounds."""
    path_a = tmp_path / "scene_disjoint_a.h5"
    path_b = tmp_path / "scene_disjoint_b.h5"
    _write_synthetic_ortho_h5(
        path_a,
        n_bands=8,
        y_dim=4,
        x_dim=5,
        upper_left=(300_000.0, 4_003_600.0),
        lower_right=(303_000.0, 4_000_000.0),
    )
    _write_synthetic_ortho_h5(
        path_b,
        n_bands=8,
        y_dim=4,
        x_dim=5,
        upper_left=(400_000.0, 5_003_600.0),
        lower_right=(403_000.0, 5_000_000.0),
    )
    return path_a, path_b


def _write_mismatched_crs_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Same bounds but different EPSG codes (different UTM zones)."""
    path_a = tmp_path / "scene_utm11.h5"
    path_b = tmp_path / "scene_utm12.h5"
    _write_synthetic_ortho_h5(
        path_a,
        n_bands=8,
        y_dim=4,
        x_dim=5,
        upper_left=(300_000.0, 4_003_600.0),
        lower_right=(303_000.0, 4_000_000.0),
        epsg_code=32611,
        zone_code=11,
    )
    _write_synthetic_ortho_h5(
        path_b,
        n_bands=8,
        y_dim=4,
        x_dim=5,
        upper_left=(300_000.0, 4_003_600.0),
        lower_right=(303_000.0, 4_000_000.0),
        epsg_code=32612,
        zone_code=12,
    )
    return path_a, path_b


# ---------------------------------------------------------------------------
# reproject_to_common_grid — happy path
# ---------------------------------------------------------------------------


class TestReprojectToCommonGrid:
    def test_overlapping_scenes_produce_identical_xy_coords(self, tmp_path):
        from tanager.io import load_ortho_scene, reproject_to_common_grid

        path_a, path_b = _write_overlapping_ortho_pair(tmp_path)
        ds_a = load_ortho_scene(path_a)
        ds_b = load_ortho_scene(path_b)

        aligned = reproject_to_common_grid([ds_a, ds_b], target_resolution=900.0)

        assert len(aligned) == 2
        np.testing.assert_array_equal(
            aligned[0].coords["x"].values, aligned[1].coords["x"].values
        )
        np.testing.assert_array_equal(
            aligned[0].coords["y"].values, aligned[1].coords["y"].values
        )
        assert aligned[0].sizes["y"] == aligned[1].sizes["y"]
        assert aligned[0].sizes["x"] == aligned[1].sizes["x"]

    def test_aligned_dataset_preserves_wavelength_and_aux_coords(self, tmp_path):
        from tanager.io import load_ortho_scene, reproject_to_common_grid

        path_a, path_b = _write_overlapping_ortho_pair(tmp_path)
        ds_a = load_ortho_scene(path_a)
        ds_b = load_ortho_scene(path_b)

        aligned = reproject_to_common_grid([ds_a, ds_b], target_resolution=900.0)

        np.testing.assert_array_equal(
            aligned[0].coords["wavelength"].values, ds_a.coords["wavelength"].values
        )
        # fwhm and good_wavelengths must survive the warp.
        assert "fwhm" in aligned[0].coords
        assert "good_wavelengths" in aligned[0].coords
        assert aligned[0].coords["fwhm"].sizes == {"wavelength": ds_a.sizes["wavelength"]}

    def test_aligned_dataset_records_target_grid_in_attrs(self, tmp_path):
        from tanager.io import load_ortho_scene, reproject_to_common_grid

        path_a, path_b = _write_overlapping_ortho_pair(tmp_path)
        ds_a = load_ortho_scene(path_a)
        ds_b = load_ortho_scene(path_b)

        aligned = reproject_to_common_grid([ds_a, ds_b], target_resolution=900.0)

        assert "aligned_to" in aligned[0].attrs
        assert aligned[0].attrs["crs"] == ds_a.attrs["crs"]
        meta = aligned[0].attrs["aligned_to"]
        assert meta["resolution"] == pytest.approx(900.0)
        assert isinstance(meta["bounds"], tuple)
        assert isinstance(meta["shape"], tuple)

    def test_target_bounds_override_uses_caller_extent(self, tmp_path):
        from tanager.io import load_ortho_scene, reproject_to_common_grid

        path_a, path_b = _write_overlapping_ortho_pair(tmp_path)
        ds_a = load_ortho_scene(path_a)
        ds_b = load_ortho_scene(path_b)

        target = (302_500.0, 4_001_500.0, 304_000.0, 4_003_000.0)
        aligned = reproject_to_common_grid(
            [ds_a, ds_b], target_bounds=target, target_resolution=300.0
        )

        # Width/height = (1500/300, 1500/300) = (5, 5)
        assert aligned[0].sizes["x"] == 5
        assert aligned[0].sizes["y"] == 5

    def test_warped_data_var_preserves_aliases(self, tmp_path):
        """surface_reflectance and toa_radiance both remain on the aligned Dataset."""
        from tanager.io import load_ortho_scene, reproject_to_common_grid

        path_a, path_b = _write_overlapping_ortho_pair(tmp_path)
        ds_a = load_ortho_scene(path_a)
        ds_b = load_ortho_scene(path_b)

        aligned = reproject_to_common_grid([ds_a, ds_b], target_resolution=900.0)

        for ds in aligned:
            assert "surface_reflectance" in ds.data_vars
            assert "toa_radiance" in ds.data_vars

    def test_three_scenes_share_grid(self, tmp_path):
        """Three-way alignment — common case for pre/post/recovery analyses."""
        from tanager.io import load_ortho_scene, reproject_to_common_grid

        path_a, path_b = _write_overlapping_ortho_pair(tmp_path)
        path_c = tmp_path / "scene_c.h5"
        _write_synthetic_ortho_h5(
            path_c,
            n_bands=8,
            y_dim=4,
            x_dim=5,
            upper_left=(301_500.0, 4_004_000.0),
            lower_right=(305_000.0, 4_001_500.0),
        )
        scenes = [
            load_ortho_scene(path_a),
            load_ortho_scene(path_b),
            load_ortho_scene(path_c),
        ]

        aligned = reproject_to_common_grid(scenes, target_resolution=900.0)
        ref_x = aligned[0].coords["x"].values
        ref_y = aligned[0].coords["y"].values
        for other in aligned[1:]:
            np.testing.assert_array_equal(other.coords["x"].values, ref_x)
            np.testing.assert_array_equal(other.coords["y"].values, ref_y)


# ---------------------------------------------------------------------------
# reproject_to_common_grid — error paths
# ---------------------------------------------------------------------------


class TestReprojectToCommonGridErrors:
    def test_disjoint_scenes_raise_value_error(self, tmp_path):
        from tanager.io import load_ortho_scene, reproject_to_common_grid

        path_a, path_b = _write_disjoint_ortho_pair(tmp_path)
        ds_a = load_ortho_scene(path_a)
        ds_b = load_ortho_scene(path_b)

        with pytest.raises(ValueError, match="overlap|do not overlap"):
            reproject_to_common_grid([ds_a, ds_b])

    def test_mismatched_crs_raises_value_error(self, tmp_path):
        from tanager.io import load_ortho_scene, reproject_to_common_grid

        path_a, path_b = _write_mismatched_crs_pair(tmp_path)
        ds_a = load_ortho_scene(path_a)
        ds_b = load_ortho_scene(path_b)

        with pytest.raises(ValueError, match="same CRS"):
            reproject_to_common_grid([ds_a, ds_b])

    def test_single_dataset_raises_value_error(self, tmp_path):
        from tanager.io import load_ortho_scene, reproject_to_common_grid

        path_a, _ = _write_overlapping_ortho_pair(tmp_path)
        ds_a = load_ortho_scene(path_a)

        with pytest.raises(ValueError, match="at least 2"):
            reproject_to_common_grid([ds_a])

    def test_missing_crs_raises_value_error(self, synthetic_tanager_dataset):
        from tanager.io import reproject_to_common_grid

        # synthetic_tanager_dataset has no crs/epsg in attrs.
        with pytest.raises(ValueError, match="CRS"):
            reproject_to_common_grid([synthetic_tanager_dataset, synthetic_tanager_dataset])

    def test_overlap_below_threshold_raises(self, tmp_path):
        """A barely-touching pair below 10% should be rejected by default."""
        from tanager.io import load_ortho_scene, reproject_to_common_grid

        # Scene A is 3000 x 3600 m (10.8e6 m²). Scene B has only a thin
        # 200 x 300 m corner overlap (60_000 m²) — 0.55% of A.
        path_a = tmp_path / "small_overlap_a.h5"
        path_b = tmp_path / "small_overlap_b.h5"
        _write_synthetic_ortho_h5(
            path_a,
            n_bands=4,
            y_dim=4,
            x_dim=4,
            upper_left=(300_000.0, 4_003_600.0),
            lower_right=(303_000.0, 4_000_000.0),
        )
        _write_synthetic_ortho_h5(
            path_b,
            n_bands=4,
            y_dim=4,
            x_dim=4,
            upper_left=(302_800.0, 4_000_300.0),
            lower_right=(305_800.0, 3_996_700.0),
        )
        ds_a = load_ortho_scene(path_a)
        ds_b = load_ortho_scene(path_b)

        with pytest.raises(ValueError, match=r"\d+\.\d%.*threshold|threshold"):
            reproject_to_common_grid([ds_a, ds_b])

    def test_overlap_below_threshold_can_be_overridden_by_target_bounds(self, tmp_path):
        """When the caller supplies target_bounds the overlap check is skipped."""
        from tanager.io import load_ortho_scene, reproject_to_common_grid

        path_a, path_b = _write_overlapping_ortho_pair(tmp_path)
        ds_a = load_ortho_scene(path_a)
        ds_b = load_ortho_scene(path_b)

        aligned = reproject_to_common_grid(
            [ds_a, ds_b],
            target_bounds=(302_000.0, 4_001_000.0, 304_500.0, 4_003_600.0),
            target_resolution=500.0,
            min_overlap_fraction=0.99,  # would normally fail
        )
        assert aligned[0].sizes["x"] == aligned[1].sizes["x"]

    def test_invalid_resampling_raises(self, tmp_path):
        from tanager.io import load_ortho_scene, reproject_to_common_grid

        path_a, path_b = _write_overlapping_ortho_pair(tmp_path)
        ds_a = load_ortho_scene(path_a)
        ds_b = load_ortho_scene(path_b)

        with pytest.raises(ValueError, match="resampling"):
            reproject_to_common_grid(
                [ds_a, ds_b], target_resolution=900.0, resampling="bogus"
            )


# ---------------------------------------------------------------------------
# dnbr auto-alignment
# ---------------------------------------------------------------------------


def _build_aligned_synthetic_pair() -> tuple[xr.Dataset, xr.Dataset]:
    """Two tiny synthetic Datasets with matching pixel grids (no CRS needed)."""
    wavelengths = np.array([660.0, 860.0, 2200.0], dtype=np.float32)
    pre_data = np.full((3, 4, 4), 0.5, dtype=np.float32)
    post_data = np.full((3, 4, 4), 0.2, dtype=np.float32)
    coords = {
        "wavelength": wavelengths,
        "y": np.arange(4),
        "x": np.arange(4),
    }
    pre = xr.Dataset(
        {"reflectance": (["wavelength", "y", "x"], pre_data)}, coords=coords
    )
    post = xr.Dataset(
        {"reflectance": (["wavelength", "y", "x"], post_data)}, coords=coords
    )
    return pre, post


class TestDnbrAutoAlign:
    def test_dnbr_on_aligned_scenes_skips_reproject(self):
        from tanager.spectral import dnbr

        pre, post = _build_aligned_synthetic_pair()
        out = dnbr(pre, post)
        assert tuple(out.dims) == ("y", "x")
        assert out.shape == (4, 4)
        assert np.all(np.isfinite(out.values))

    def test_dnbr_with_auto_align_false_raises_on_grid_mismatch(self, tmp_path):
        from tanager.io import load_ortho_scene
        from tanager.spectral import dnbr

        path_a, path_b = _write_overlapping_ortho_pair(tmp_path)
        ds_a = load_ortho_scene(path_a)
        ds_b = load_ortho_scene(path_b)

        # rename surface_reflectance -> reflectance so dnbr's reflectance lookup
        # finds the cube; both datasets carry the same wavelength grid here so
        # the only real difference is spatial.
        ds_a = ds_a.rename({"surface_reflectance": "reflectance"}).drop_vars(
            "toa_radiance"
        )
        ds_b = ds_b.rename({"surface_reflectance": "reflectance"}).drop_vars(
            "toa_radiance"
        )

        with pytest.raises(ValueError, match="match|auto_align"):
            dnbr(ds_a, ds_b, auto_align=False)

    def test_dnbr_with_auto_align_true_warps_misaligned_scenes(self, tmp_path, caplog):
        import logging

        from tanager.io import load_ortho_scene
        from tanager.spectral import dnbr

        path_a, path_b = _write_overlapping_ortho_pair(tmp_path)
        ds_a = load_ortho_scene(path_a)
        ds_b = load_ortho_scene(path_b)

        with caplog.at_level(logging.WARNING, logger="tanager.spectral"):
            out = dnbr(ds_a, ds_b)

        assert tuple(out.dims) == ("y", "x")
        # Output grid covers the intersection at 30 m resolution.
        assert out.sizes["y"] >= 1
        assert out.sizes["x"] >= 1
        assert any("auto-aligning" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Real-data integration (LGT-313 board directive)
#
# Skipped automatically when the bulk fire HDF5s are not on disk so CI without
# bulk-data access still passes.
# ---------------------------------------------------------------------------

_REAL_FIRE_DIR = os.path.join("data", "raw", "fire")
_REAL_PRE_FIRE = os.path.join(_REAL_FIRE_DIR, "20241215_185916_33_4001_ortho_sr_hdf5.h5")
_REAL_POST_FIRE = os.path.join(_REAL_FIRE_DIR, "20250123_185507_64_4001_ortho_sr_hdf5.h5")
_REAL_RECOVERY = os.path.join(_REAL_FIRE_DIR, "20250407_192235_24_4001_ortho_sr_hdf5.h5")


def _have_real_pair(*paths: str) -> bool:
    return all(os.path.exists(p) for p in paths)


@pytest.mark.integration
@pytest.mark.skipif(
    not _have_real_pair(_REAL_POST_FIRE, _REAL_RECOVERY),
    reason="real ortho_sr post-fire/recovery scenes not available locally",
)
def test_real_post_fire_and_recovery_align_and_dnbr_in_range():
    """Post-fire (Jan 23) and recovery (Apr 7) scenes share the LA fire footprint.

    Validates the full pipeline: load_ortho_scene → reproject_to_common_grid →
    dnbr produces values within the USGS [-2, 2] dNBR convention.
    """
    from tanager.io import load_ortho_scene, reproject_to_common_grid
    from tanager.spectral import dnbr

    post = load_ortho_scene(_REAL_POST_FIRE)
    recovery = load_ortho_scene(_REAL_RECOVERY)

    aligned_post, aligned_recovery = reproject_to_common_grid([post, recovery])

    # Aligned scenes must share their UTM grid exactly.
    np.testing.assert_array_equal(
        aligned_post.coords["x"].values, aligned_recovery.coords["x"].values
    )
    np.testing.assert_array_equal(
        aligned_post.coords["y"].values, aligned_recovery.coords["y"].values
    )
    assert aligned_post.attrs["crs"] == post.attrs["crs"]

    # dNBR convention: positive => burn (NBR fell). On post→recovery we expect
    # negative dNBR over recovering pixels. We just assert physical range and
    # that finite pixels exist — sign analysis is a separate Phase 3 product.
    out = dnbr(aligned_post, aligned_recovery)
    finite = out.values[np.isfinite(out.values)]
    assert finite.size > 0, "dnbr produced no finite pixels on real scene pair"
    assert finite.min() >= -2.0 - 1e-6
    assert finite.max() <= 2.0 + 1e-6


@pytest.mark.integration
@pytest.mark.skipif(
    not _have_real_pair(_REAL_PRE_FIRE, _REAL_POST_FIRE),
    reason="real pre-fire/post-fire ortho_sr scenes not available locally",
)
def test_real_pre_and_post_fire_have_minimal_or_no_overlap():
    """Pre-fire (Dec 15) covers Malibu coast; post-fire (Jan 23) covers Eaton.

    Per the LGT-299 issue these scenes are geographically distant — the Y
    ranges are disjoint (~3.75M m vs ~3.81M m UTM north). The function
    should refuse to align them rather than silently produce a near-empty
    common grid.
    """
    from tanager.io import load_ortho_scene, reproject_to_common_grid

    pre = load_ortho_scene(_REAL_PRE_FIRE)
    post = load_ortho_scene(_REAL_POST_FIRE)

    with pytest.raises(ValueError, match="overlap|threshold"):
        reproject_to_common_grid([pre, post])
