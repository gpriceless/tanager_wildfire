"""End-to-end execution tests for the competition deliverable notebooks.

These tests run each of the five deliverable notebooks
(``01-data-discovery`` … ``05-sensor-comparison``) through
``jupyter nbconvert --execute`` and assert a clean exit. They exist because
the notebooks had *no* automated coverage: a stress test found that
``04-temporal-recovery`` OOM-killed the machine twice (LGT-1012), and a
notebook-execution test is the only thing that would have caught that
regression before it reached a human.

Cost and gating
---------------
Executing a notebook is minutes-long and requires the real Tanager HDF5
scenes under :data:`tanager.config.DATA_DIR`, so every test here is:

* marked ``@pytest.mark.slow`` — skipped by default, opt in with ``-m slow``;
* skipped when ``jupyter``/``nbconvert`` is not installed;
* skipped when no ``*.h5`` scenes are present (CI without the data assets).

Each notebook runs with ``TANAGER_MAX_JOBS=2`` so the joblib worker cap keeps
memory bounded (see :func:`tanager.config.parallel_jobs`); an unbounded run is
what caused the LGT-1012 OOMs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("nbconvert")
pytest.importorskip("nbformat")

from tanager.config import DATA_DIR  # noqa: E402  (after importorskip)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_NOTEBOOK_DIR = _REPO_ROOT / "notebooks"

# The five competition deliverables, in narrative order.
_NOTEBOOKS = [
    "01-data-discovery.ipynb",
    "02-burn-severity.ipynb",
    "03-fuel-moisture.ipynb",
    "04-temporal-recovery.ipynb",
    "05-sensor-comparison.ipynb",
]

# Generous ceiling: a full 426-band scene run is minutes-long; anything past
# this indicates a hang, not slow-but-healthy progress.
_EXECUTE_TIMEOUT_S = 1800


def _has_scene_data() -> bool:
    """True when at least one raw HDF5 scene is available to the notebooks."""
    return DATA_DIR.is_dir() and any(DATA_DIR.glob("*.h5"))


@pytest.mark.slow
@pytest.mark.parametrize("notebook_name", _NOTEBOOKS)
def test_notebook_executes_cleanly(notebook_name: str, tmp_path: Path) -> None:
    """Each deliverable notebook runs end-to-end without raising.

    Executes the notebook in place (writing the rendered copy to a throwaway
    location so the repo copy is untouched) with the worker cap pinned low,
    and asserts a zero exit status. ``nbconvert --execute`` returns non-zero
    on the first uncaught cell exception, so a clean exit means every cell ran.
    """
    notebook_path = _NOTEBOOK_DIR / notebook_name
    assert notebook_path.is_file(), f"missing deliverable notebook: {notebook_path}"

    if not _has_scene_data():
        pytest.skip(
            f"no *.h5 scenes under {DATA_DIR}; notebook execution needs real data"
        )

    output_path = tmp_path / notebook_name

    env = dict(os.environ)
    # Bound joblib workers so parallel stages stay memory-safe (LGT-1012).
    env["TANAGER_MAX_JOBS"] = "2"
    # Keep matplotlib headless inside the kernel subprocess.
    env.setdefault("MPLBACKEND", "Agg")

    cmd = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=1500",
        "--output",
        str(output_path),
        str(notebook_path),
    ]

    result = subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=_EXECUTE_TIMEOUT_S,
    )

    assert result.returncode == 0, (
        f"{notebook_name} failed to execute cleanly "
        f"(exit {result.returncode}).\n"
        f"--- stderr (tail) ---\n{result.stderr[-4000:]}"
    )
