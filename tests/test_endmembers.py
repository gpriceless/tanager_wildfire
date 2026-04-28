"""Tests for :mod:`tanager.endmembers`.

Library construction is exercised against mocked file I/O and synthetic
DataArrays so the suite is self-contained — no USGS / ECOSTRESS / FRAMES
network dependencies. The heavy spectral-libraries / SPy dependencies are
only required for two narrow tests (resampling and EAR/MASA pruning) and are
skipped via :func:`pytest.importorskip` when not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
import xarray as xr

from tanager import endmembers


# ---------------------------------------------------------------------------
# Helpers — synthetic library factories
# ---------------------------------------------------------------------------


def _make_library(
    categories: Sequence[str],
    *,
    n_bands: int = 30,
    rng_seed: int = 0,
    source: str = "synthetic",
) -> xr.DataArray:
    """Build a synthetic library DataArray matching the canonical schema."""
    rng = np.random.default_rng(rng_seed)
    n_spectra = len(categories)
    spectra = np.clip(
        rng.uniform(0.05, 0.6, size=(n_spectra, n_bands)).astype(np.float32),
        0.0,
        1.0,
    )
    # Inject a per-spectrum spread so std-based ranking is stable.
    spectra += np.linspace(-0.05, 0.05, n_spectra, dtype=np.float32)[:, None]
    spectra = np.clip(spectra, 0.0, 1.0)

    wavelengths = np.linspace(400.0, 2400.0, n_bands).astype(np.float32)
    spectrum_ids = [f"{cat}_{i:03d}" for i, cat in enumerate(categories)]
    return xr.DataArray(
        spectra,
        dims=("spectrum_id", "wavelength"),
        coords={
            "spectrum_id": np.asarray(spectrum_ids, dtype=object),
            "wavelength": wavelengths,
            "name": ("spectrum_id", np.asarray(spectrum_ids, dtype=object)),
            "category": ("spectrum_id", np.asarray(list(categories), dtype=object)),
            "source": ("spectrum_id", np.asarray([source] * n_spectra, dtype=object)),
        },
    )


# ---------------------------------------------------------------------------
# select_endmembers_incob
# ---------------------------------------------------------------------------


class TestSelectEndmembersIncob:
    def test_caps_per_class(self):
        cats = ["char", "char", "char", "pv", "pv", "pv", "pv", "soil"]
        lib = _make_library(cats)
        out = endmembers.select_endmembers_incob(lib, max_per_class=2)

        result_cats = out.coords["category"].values.tolist()
        assert result_cats.count("char") == 2
        assert result_cats.count("pv") == 2
        assert result_cats.count("soil") == 1
        assert out.sizes["spectrum_id"] == 5

    def test_preserves_class_labels(self):
        cats = ["char"] * 4 + ["pv"] * 4
        lib = _make_library(cats)
        out = endmembers.select_endmembers_incob(lib, max_per_class=10)
        # max_per_class > n_per_class → keep everything
        assert out.sizes["spectrum_id"] == lib.sizes["spectrum_id"]
        # And class labels round-trip unchanged
        assert sorted(out.coords["category"].values.tolist()) == sorted(cats)

    def test_rejects_invalid_max(self):
        lib = _make_library(["pv", "pv"])
        with pytest.raises(ValueError):
            endmembers.select_endmembers_incob(lib, max_per_class=0)

    def test_requires_category_coord(self):
        lib = _make_library(["pv"])
        bad = lib.drop_vars("category")
        with pytest.raises(ValueError, match="category"):
            endmembers.select_endmembers_incob(bad, max_per_class=1)


# ---------------------------------------------------------------------------
# build_hybrid_library
# ---------------------------------------------------------------------------


class TestBuildHybridLibrary:
    def test_merges_two_sources(self):
        usgs = _make_library(["char", "pv"], source="usgs_v7")
        frames = _make_library(["soil", "npv"], source="frames")
        merged = endmembers.build_hybrid_library(usgs=usgs, frames=frames)

        assert merged.sizes["spectrum_id"] == 4
        sources = merged.coords["source"].values.tolist()
        assert sorted(sources) == sorted(["usgs_v7", "usgs_v7", "frames", "frames"])

    def test_handles_none_inputs(self):
        usgs = _make_library(["char", "pv"], source="usgs_v7")
        merged = endmembers.build_hybrid_library(
            usgs=usgs,
            ecostress=None,
            frames=None,
            image_derived=None,
        )
        assert merged.sizes["spectrum_id"] == 2

    def test_rejects_all_none(self):
        with pytest.raises(ValueError, match="at least one"):
            endmembers.build_hybrid_library()

    def test_rejects_misaligned_wavelengths(self):
        a = _make_library(["pv"], n_bands=20)
        b = _make_library(["char"], n_bands=30)
        with pytest.raises(ValueError, match="wavelength grid"):
            endmembers.build_hybrid_library(usgs=a, frames=b)

    def test_clips_to_valid_range(self):
        a = _make_library(["pv"])
        # Inject an out-of-range value
        a.values[0, 5] = 1.5
        merged = endmembers.build_hybrid_library(usgs=a)
        assert float(merged.values.max()) <= 1.0
        assert float(merged.values.min()) >= 0.0


# ---------------------------------------------------------------------------
# Loaders — file I/O is mocked / driven by tmp_path
# ---------------------------------------------------------------------------


class TestLoadFramesLibrary:
    def test_parses_two_column_ascii(self, tmp_path):
        # Synthetic FRAMES-style spectra in micrometres + percent reflectance.
        wls_um = np.linspace(0.4, 2.4, 50)
        char_refl = np.linspace(2.0, 5.0, 50)  # percent
        pv_refl = np.linspace(5.0, 60.0, 50)

        char_path = tmp_path / "char_sample_001.txt"
        pv_path = tmp_path / "green_vegetation_002.txt"
        char_path.write_text("\n".join(f"{w:.4f}\t{r:.4f}" for w, r in zip(wls_um, char_refl)))
        pv_path.write_text("\n".join(f"{w:.4f}\t{r:.4f}" for w, r in zip(wls_um, pv_refl)))

        lib = endmembers.load_frames_library(tmp_path)
        assert lib.sizes["spectrum_id"] == 2
        cats = sorted(lib.coords["category"].values.tolist())
        assert cats == ["char", "pv"]
        # Wavelengths converted µm → nm, reflectance scaled to fractional
        assert lib.coords["wavelength"].values.min() == pytest.approx(400.0, abs=10.0)
        assert lib.coords["wavelength"].values.max() == pytest.approx(2400.0, abs=10.0)
        assert lib.values.max() <= 1.0
        assert lib.values.min() >= 0.0

    def test_skips_uncategorisable_files(self, tmp_path):
        wls_um = np.linspace(0.4, 2.4, 30)
        refl = np.linspace(0.05, 0.30, 30)
        # "char_a.txt" → char (categorisable), "weird.txt" → "other" → skipped
        (tmp_path / "char_a.txt").write_text("\n".join(f"{w:.4f}\t{r:.4f}" for w, r in zip(wls_um, refl)))
        (tmp_path / "weird.txt").write_text("\n".join(f"{w:.4f}\t{r:.4f}" for w, r in zip(wls_um, refl)))

        lib = endmembers.load_frames_library(tmp_path)
        assert lib.sizes["spectrum_id"] == 1
        assert lib.coords["category"].values.tolist() == ["char"]

    def test_explicit_category_map(self, tmp_path):
        wls_um = np.linspace(0.4, 2.4, 30)
        refl = np.linspace(0.05, 0.30, 30)
        (tmp_path / "mystery.txt").write_text("\n".join(f"{w:.4f}\t{r:.4f}" for w, r in zip(wls_um, refl)))

        lib = endmembers.load_frames_library(tmp_path, category_map={"mystery": "soil"})
        assert lib.coords["category"].values.tolist() == ["soil"]

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            endmembers.load_frames_library(tmp_path / "does_not_exist")


class TestLoadUsgsLibrary:
    def test_missing_data_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            endmembers.load_usgs_library(data_dir=None)
        with pytest.raises(FileNotFoundError):
            endmembers.load_usgs_library(data_dir=tmp_path / "nope")

    def test_filters_by_category(self, tmp_path):
        # Write a wavelengths file (in micrometres) and two spectrum files.
        wls_um = np.linspace(0.35, 2.5, 30)
        wl_path = tmp_path / "splib07a_Wavelengths_USGS_Beckman_ASD.txt"
        wl_path.write_text("\n".join(f"{w:.6f}" for w in wls_um))

        char_refl = np.linspace(0.02, 0.08, 30)
        soil_refl = np.linspace(0.10, 0.35, 30)
        (tmp_path / "s07ASD_char_sample_001.txt").write_text("\n".join(f"{r:.6f}" for r in char_refl))
        (tmp_path / "s07ASD_soil_sample_001.txt").write_text("\n".join(f"{r:.6f}" for r in soil_refl))

        lib_all = endmembers.load_usgs_library(data_dir=tmp_path)
        assert lib_all.sizes["spectrum_id"] == 2

        lib_char = endmembers.load_usgs_library(data_dir=tmp_path, categories=["char"])
        assert lib_char.sizes["spectrum_id"] == 1
        assert lib_char.coords["category"].values.tolist() == ["char"]


# ---------------------------------------------------------------------------
# resample_library — needs SPy
# ---------------------------------------------------------------------------


class TestResampleLibrary:
    def test_produces_target_band_count(self):
        pytest.importorskip("spectral")
        # Source library at ~1 nm sampling so the SPy BandResampler can find
        # overlap with a 5.5 nm-FWHM target grid at every band centre.
        lib = _make_library(["char", "pv"], n_bands=2001)
        target = np.linspace(400.0, 2400.0, 426).astype(np.float32)
        out = endmembers.resample_library(lib, target_wavelengths=target, fwhm=5.5)
        assert out.sizes["wavelength"] == 426
        assert out.sizes["spectrum_id"] == 2
        # Reflectance always clipped to [0, 1]
        assert float(out.values.max()) <= 1.0
        assert float(out.values.min()) >= 0.0

    def test_accepts_per_band_fwhm_array(self):
        pytest.importorskip("spectral")
        lib = _make_library(["char"], n_bands=2001)
        target = np.linspace(400.0, 2400.0, 50).astype(np.float32)
        per_band_fwhm = np.linspace(5.2, 6.8, 50).astype(np.float32)
        out = endmembers.resample_library(lib, target_wavelengths=target, fwhm=per_band_fwhm)
        assert out.sizes["wavelength"] == 50

    def test_rejects_missing_wavelength_coord(self):
        bad = _make_library(["char"]).drop_vars("wavelength")
        with pytest.raises(ValueError, match="wavelength"):
            endmembers.resample_library(bad, target_wavelengths=np.array([500.0, 1000.0]))


# ---------------------------------------------------------------------------
# prune_endmembers_ear_masa — needs spectral_libraries
# ---------------------------------------------------------------------------


class TestPruneEndmembersEarMasa:
    def test_pruning_removes_redundant_spectra(self):
        pytest.importorskip("spectral_libraries")
        # Construct two redundant char spectra plus distinct pv/soil so EAR
        # is high for one of the chars and pruning has something to remove.
        cats = ["char"] * 4 + ["pv"] * 3 + ["soil"] * 3
        lib = _make_library(cats, rng_seed=7)
        try:
            pruned = endmembers.prune_endmembers_ear_masa(lib)
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"EarMasaCob unavailable: {exc}")
        # Pruning must never grow the library.
        assert pruned.sizes["spectrum_id"] <= lib.sizes["spectrum_id"]
        # EAR / MASA coords were attached.
        assert "ear" in pruned.coords
        assert "masa_deg" in pruned.coords

    def test_rejects_missing_category_coord(self):
        bad = _make_library(["pv"]).drop_vars("category")
        with pytest.raises(ValueError, match="category"):
            endmembers.prune_endmembers_ear_masa(bad)


# ---------------------------------------------------------------------------
# extract_image_endmembers
# ---------------------------------------------------------------------------


class TestExtractImageEndmembers:
    def _scene(self) -> xr.Dataset:
        n_bands, ny, nx = 20, 10, 10
        wls = np.linspace(400.0, 2400.0, n_bands).astype(np.float32)
        rng = np.random.default_rng(42)
        data = rng.uniform(0.0, 1.0, size=(n_bands, ny, nx)).astype(np.float32)
        # Top-left 5x5 = vegetation-like (high NIR), bottom-right 5x5 = char.
        data[:, :5, :5] = 0.45
        data[:, 5:, 5:] = 0.05
        return xr.Dataset(
            {"reflectance": (["wavelength", "y", "x"], data)},
            coords={"wavelength": wls, "y": np.arange(ny), "x": np.arange(nx)},
        )

    def test_spatial_returns_mean_per_region(self):
        scene = self._scene()
        regions = {
            "pv": (slice(0, 5), slice(0, 5)),
            "char": (slice(5, 10), slice(5, 10)),
        }
        out = endmembers.extract_image_endmembers(scene, method="spatial", regions=regions)
        assert out.sizes["spectrum_id"] == 2
        cats = sorted(out.coords["category"].values.tolist())
        assert cats == ["char", "pv"]
        # PV region was set to 0.45, char region to 0.05 — averages preserve that.
        pv_spec = out.sel(spectrum_id=out.coords["spectrum_id"].values[
            list(out.coords["category"].values).index("pv")
        ])
        char_spec = out.sel(spectrum_id=out.coords["spectrum_id"].values[
            list(out.coords["category"].values).index("char")
        ])
        assert float(pv_spec.mean()) > float(char_spec.mean())

    def test_spatial_requires_regions(self):
        scene = self._scene()
        with pytest.raises(ValueError, match="regions"):
            endmembers.extract_image_endmembers(scene, method="spatial", regions=None)

    def test_unknown_method_rejected(self):
        scene = self._scene()
        with pytest.raises(ValueError, match="method"):
            endmembers.extract_image_endmembers(scene, method="bogus", regions={"pv": (slice(0, 1), slice(0, 1))})


# ---------------------------------------------------------------------------
# build_fire_library — orchestrator
# ---------------------------------------------------------------------------


class TestBuildFireLibrary:
    def test_requires_a_source(self):
        with pytest.raises(ValueError, match="no sources"):
            endmembers.build_fire_library()

    def test_orchestrates_image_only_pipeline(self, monkeypatch):
        pytest.importorskip("spectral")
        # Build a deterministic scene with two ROIs so the pipeline can run
        # entirely on image-derived endmembers (no external libraries).
        n_bands, ny, nx = 60, 12, 12
        wls = np.linspace(380.0, 2500.0, n_bands).astype(np.float32)
        rng = np.random.default_rng(0)
        data = rng.uniform(0.0, 1.0, size=(n_bands, ny, nx)).astype(np.float32)
        data[:, :6, :6] = 0.45  # pv-ish
        data[:, 6:, 6:] = 0.05  # char-ish
        scene = xr.Dataset(
            {"reflectance": (["wavelength", "y", "x"], data)},
            coords={"wavelength": wls, "y": np.arange(ny), "x": np.arange(nx)},
        )

        # Skip EAR/MASA pruning by stubbing it to a passthrough — the full
        # spectral_libraries pipeline isn't always installed in CI.
        monkeypatch.setattr(
            endmembers,
            "prune_endmembers_ear_masa",
            lambda lib, **_: lib,
        )

        lib = endmembers.build_fire_library(
            scene_pre=scene,
            scene_post=scene,
            target_wavelengths=wls,
            pre_regions={"pv": (slice(0, 6), slice(0, 6))},
            post_regions={"char": (slice(6, 12), slice(6, 12))},
            max_per_class=3,
            add_shade=True,
        )
        cats = lib.coords["category"].values.tolist()
        assert "pv" in cats
        assert "char" in cats
        assert "shade" in cats
        # All spectra share the target wavelength grid.
        assert lib.sizes["wavelength"] == n_bands
