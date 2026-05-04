"""Tests for :mod:`scripts.run_pipeline` stages.

Exercises stages from the FireSpec end-to-end runner against synthetic Tanager
fixtures. Real HDF5 scenes are tested via the (slower) ``integration`` marker
in ``test_io.py`` — these tests stay quick by hitting only synthetic data.
"""

from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("spectral")
pytest.importorskip("sklearn")

# scripts/ is not a package, so add it to sys.path before importing run_pipeline.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

run_pipeline = importlib.import_module("run_pipeline")


# ---------------------------------------------------------------------------
# stage_sensor_comparison — happy path on synthetic fixture
# ---------------------------------------------------------------------------


class TestStageSensorComparison:
    def test_returns_ok_status_and_writes_csv(
        self, synthetic_tanager_dataset, tmp_path: Path,
    ):
        result = run_pipeline.stage_sensor_comparison(
            synthetic_tanager_dataset, "synth_001", tmp_path,
        )
        assert result.status == "ok", f"expected ok status, got {result.status}: {result.detail}"
        assert result.name == "sensor_comparison"
        assert result.artifacts, "stage should emit a CSV artifact"

        csv_path = tmp_path / "synth_001_sensor_comparison.csv"
        assert csv_path.exists()
        assert csv_path in result.artifacts

    def test_csv_has_expected_columns_and_three_sensor_rows(
        self, synthetic_tanager_dataset, tmp_path: Path,
    ):
        result = run_pipeline.stage_sensor_comparison(
            synthetic_tanager_dataset, "synth_002", tmp_path,
        )
        assert result.status == "ok"

        csv_path = tmp_path / "synth_002_sensor_comparison.csv"
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == [
                "sensor_name", "tanager_r2", "reference_r2", "rmse_reduction_pct",
            ]
            rows = list(reader)

        assert len(rows) == 3
        sensor_names = {r["sensor_name"] for r in rows}
        assert sensor_names == {"EMIT", "PRISMA", "Sentinel-2"}

        for row in rows:
            # tanager_r2 against itself is trivially 1.0; reference_r2 is finite.
            assert float(row["tanager_r2"]) == pytest.approx(1.0, abs=1e-6)
            assert np.isfinite(float(row["reference_r2"]))
            assert np.isfinite(float(row["rmse_reduction_pct"]))

    def test_detail_string_lists_each_sensor(
        self, synthetic_tanager_dataset, tmp_path: Path,
    ):
        result = run_pipeline.stage_sensor_comparison(
            synthetic_tanager_dataset, "synth_003", tmp_path,
        )
        assert result.status == "ok"
        for sensor_name in ("EMIT", "PRISMA", "Sentinel-2"):
            assert sensor_name in result.detail

    def test_handles_signature_fixture(
        self, synthetic_tanager_dataset_with_signatures, tmp_path: Path,
    ):
        ds = synthetic_tanager_dataset_with_signatures()
        result = run_pipeline.stage_sensor_comparison(ds, "synth_sig", tmp_path)
        assert result.status == "ok"

        csv_path = tmp_path / "synth_sig_sensor_comparison.csv"
        assert csv_path.exists()


# ---------------------------------------------------------------------------
# stage_sensor_comparison — error path via the @_stage decorator
# ---------------------------------------------------------------------------


class TestStageSensorComparisonErrorHandling:
    def test_returns_error_status_when_simulate_sensor_raises(
        self, synthetic_tanager_dataset, tmp_path: Path, monkeypatch,
    ):
        def _raise(*_args, **_kwargs):  # noqa: D401 — match simulate_sensor signature
            raise RuntimeError("simulate failure")

        monkeypatch.setattr(run_pipeline, "simulate_sensor", _raise)
        result = run_pipeline.stage_sensor_comparison(
            synthetic_tanager_dataset, "synth_err", tmp_path,
        )

        assert result.status == "error"
        assert "simulate failure" in result.detail
        assert result.artifacts == []
        # CSV is not written when the stage errors out before the writer.
        assert not (tmp_path / "synth_err_sensor_comparison.csv").exists()

    def test_returns_error_status_for_dataset_without_wavelength(
        self, tmp_path: Path,
    ):
        # nbr() raises on a Dataset with no reflectance variable; @_stage catches it.
        empty_ds = xr.Dataset({"unrelated": (("y", "x"), np.zeros((2, 2)))})
        result = run_pipeline.stage_sensor_comparison(empty_ds, "synth_bad", tmp_path)
        assert result.status == "error"


# ---------------------------------------------------------------------------
# Sensor spec helpers
# ---------------------------------------------------------------------------


class TestBuildSensorSpecs:
    def test_returns_three_sensors_with_correct_band_counts(self):
        specs = run_pipeline._build_sensor_specs()
        assert [s[0] for s in specs] == ["EMIT", "PRISMA", "Sentinel-2"]

        # EMIT: 285 bands per config
        assert specs[0][1].shape == (285,)
        # PRISMA: 239 bands per config
        assert specs[1][1].shape == (239,)
        # Sentinel-2: 10 bands from SENTINEL2_BANDS dict
        assert specs[2][1].shape == (10,)

    def test_emit_and_prisma_centers_span_expected_range(self):
        specs = run_pipeline._build_sensor_specs()
        emit_centers = specs[0][1]
        prisma_centers = specs[1][1]

        assert emit_centers[0] == pytest.approx(381.0)
        assert emit_centers[-1] == pytest.approx(2493.0)
        assert prisma_centers[0] == pytest.approx(400.0)
        assert prisma_centers[-1] == pytest.approx(2505.0)

    def test_sentinel2_uses_per_band_fwhm(self):
        specs = run_pipeline._build_sensor_specs()
        s2_fwhm = specs[2][2]
        assert isinstance(s2_fwhm, np.ndarray)
        assert s2_fwhm.shape == (10,)
        # B12 (2190 nm) has the widest FWHM — 180 nm.
        assert s2_fwhm.max() == pytest.approx(180.0)
