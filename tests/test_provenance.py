"""Provenance gate: no claim ships without a provenance line.

These tests enforce structural integrity between the project's external
data artifacts, the claims derived from them, and the code that produces
those claims.  They are designed to catch the failure modes that have
recurred in this project:

    CBI = 3 × char_fraction  → R² = 0.998  (circular, caught late)
    inject_water_absorption() → cv_r² = 0.904  (circular, still live)
    LFMC map mean 37.7% on April scene  (3× off reality, never caught)

The gate's value is measured by one thing: would it have caught those
before a human did?
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "provenance.json"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


@pytest.fixture(scope="module")
def manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Manifest schema: the file exists and is well-formed
# ---------------------------------------------------------------------------


class TestManifestSchema:
    def test_manifest_exists(self):
        assert MANIFEST_PATH.exists(), (
            "provenance.json missing from project root — "
            "every external artifact needs a provenance entry"
        )

    def test_manifest_has_required_sections(self, manifest):
        for key in ("version", "artifacts", "claims", "climatology"):
            assert key in manifest, f"Manifest missing required section: {key}"

    def test_every_artifact_has_required_fields(self, manifest):
        required = {"path", "description", "provenance_status"}
        for key, entry in manifest["artifacts"].items():
            missing = required - set(entry.keys())
            assert not missing, (
                f"Artifact {key!r} missing required fields: {missing}"
            )

    def test_every_claim_has_required_fields(self, manifest):
        required = {"notebook", "description", "backing_artifact", "synthetic"}
        for key, entry in manifest["claims"].items():
            missing = required - set(entry.keys())
            assert not missing, (
                f"Claim {key!r} missing required fields: {missing}"
            )


# ---------------------------------------------------------------------------
# 2. Bundled artifact checksums
# ---------------------------------------------------------------------------


class TestArtifactIntegrity:
    """Every bundled artifact must match its manifest checksum.

    If someone replaces a reference file (e.g. re-downloads a BAER
    raster with different class coding), the checksum mismatch catches
    it before the accuracy metrics silently change.
    """

    @staticmethod
    def _md5(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def test_bundled_artifacts_match_checksums(self, manifest):
        for key, entry in manifest["artifacts"].items():
            if not entry.get("bundled", False):
                continue
            expected_md5 = entry.get("checksum_md5")
            if not expected_md5:
                pytest.fail(
                    f"Bundled artifact {key!r} has no checksum — "
                    "every bundled file needs a checksum"
                )
            path = PROJECT_ROOT / entry["path"]
            assert path.exists(), (
                f"Bundled artifact {key!r} not found at {path}"
            )
            actual_md5 = self._md5(path)
            assert actual_md5 == expected_md5, (
                f"Checksum mismatch for {key!r} ({path.name}): "
                f"manifest says {expected_md5}, file is {actual_md5}. "
                "If the file was intentionally replaced, update "
                "provenance.json with the new checksum and review "
                "all claims that depend on this artifact."
            )


# ---------------------------------------------------------------------------
# 3. Claims audit: every claim has a backing artifact
# ---------------------------------------------------------------------------


class TestClaimsIntegrity:
    """Every headline number must trace to an external artifact or be
    explicitly labelled synthetic.  This is the structural check that
    would have caught CBI = 3 × char (no backing artifact, not marked
    synthetic → test fails).
    """

    def test_claims_have_backing_artifacts(self, manifest):
        artifacts = manifest["artifacts"]
        for key, claim in manifest["claims"].items():
            backing = claim.get("backing_artifact")
            if backing is not None:
                assert backing in artifacts, (
                    f"Claim {key!r} references artifact {backing!r} "
                    "which does not exist in the manifest"
                )
            elif not claim.get("synthetic", False):
                pytest.fail(
                    f"Claim {key!r} has no backing artifact and is "
                    "not marked synthetic — it must be one or the other. "
                    "If the targets are generated in-repo, set "
                    '"synthetic": true and label any metric as '
                    "demonstrating methodology, never accuracy."
                )

    def test_no_circular_claim_without_synthetic_label(self, manifest):
        """A claim with independence=none must be marked synthetic."""
        for key, claim in manifest["claims"].items():
            if claim.get("independence") == "none":
                assert claim.get("synthetic", False), (
                    f"Claim {key!r} has independence=none (training "
                    "targets generated in-repo) but is not marked "
                    "synthetic=true. This is the CBI=3×char failure "
                    "mode: a metric that measures your own formula."
                )

    def test_synthetic_claims_have_no_accuracy_language(self, manifest):
        """Synthetic claims must not use language that implies real accuracy.

        Negations like "not accuracy" or "never accuracy" are fine — the
        check looks for affirmative phrasing that frames a synthetic
        metric as a real measurement.
        """
        affirmative_patterns = [
            r"(?<!not\s)(?<!never\s)(?<!no\s)ground.truth\s+accuracy",
            r"(?<!not\s)(?<!never\s)(?<!no\s)true\s+accuracy",
            r"(?<!not\s)(?<!never\s)(?<!no\s)real\s+accuracy",
            r"(?<!not\s)(?<!never\s)(?<!demonstrates\s)accuracy\s+of",
            r"validates?\s+accuracy",
        ]
        import re as _re
        for key, claim in manifest["claims"].items():
            if claim.get("synthetic", False):
                desc = (claim.get("description", "") + " " +
                        claim.get("independence_note", "")).lower()
                for pattern in affirmative_patterns:
                    match = _re.search(pattern, desc)
                    assert match is None, (
                        f"Synthetic claim {key!r} uses accuracy language "
                        f"({match.group()!r}) in its description or note. "
                        "Synthetic metrics demonstrate methodology, "
                        "never accuracy."
                    )


# ---------------------------------------------------------------------------
# 4. Globe-LFMC colocation guardrail
# ---------------------------------------------------------------------------


class TestColocationGuardrail:
    """The colocation check already exists in load_globe_lfmc() but is
    bypassed because the default does not require scene dates.  This test
    verifies that the guardrail actually fires: passing the real Tanager
    scene dates must return 0 colocated (the SoCal Globe-LFMC record
    ends 2023-01-30; the scenes are Dec 2024 – Apr 2025).

    When a notebook trains on Globe-LFMC field data, any PLSR result
    claiming real-data accuracy must acknowledge this temporal gap.
    """

    GLOBE_LFMC_PATHS = [
        PROJECT_ROOT / "data" / "reference" / "globe_lfmc" / "globe_lfmc.csv",
        PROJECT_ROOT / "data" / "reference" / "globe_lfmc" / "Globe-LFMC-2.0.xlsx",
    ]

    TANAGER_SCENE_DATES = [
        "2024-12-15",
        "2025-01-23",
        "2025-04-07",
    ]

    @pytest.fixture
    def globe_lfmc_path(self):
        for p in self.GLOBE_LFMC_PATHS:
            if p.exists():
                return p
        pytest.skip("Globe-LFMC data not available locally")

    def test_zero_colocated_with_tanager_scenes(self, globe_lfmc_path):
        from tanager.lfmc import load_globe_lfmc

        gdf = load_globe_lfmc(
            globe_lfmc_path,
            region_bbox=(-119.5, 33.5, -117.0, 35.0),
            vegetation_types=["shrub"],
            tanager_scene_dates=self.TANAGER_SCENE_DATES,
            colocation_window_days=30,
        )
        n_colocated = int(gdf["tanager_colocated"].sum())
        assert n_colocated == 0, (
            f"Expected 0 colocated Globe-LFMC observations with Tanager "
            f"scenes (SoCal record ends 2023-01), got {n_colocated}. "
            "If new Globe-LFMC data has been published covering 2024-2025, "
            "update this test and the manifest's colocation note."
        )


# ---------------------------------------------------------------------------
# 5. LFMC climatological range check
# ---------------------------------------------------------------------------


class TestLFMCRangeCheck:
    """Any LFMC map produced for a known scene month must fall within the
    Globe-LFMC climatological envelope.

    This is the check that would have caught the 3×-off LFMC map before
    a human did: mean 37.7% on an April scene, when the Globe-LFMC SoCal
    shrub April median is 119%.

    The test does not run the full pipeline (that requires loading a
    ~1.3 GB scene); it validates the range-check function itself and
    confirms it would reject the known-bad output.
    """

    def test_april_37_percent_is_implausible(self, manifest):
        """The actual failure case: mean 37.7% in April."""
        from tanager.provenance import check_lfmc_range

        result = check_lfmc_range(37.7, scene_month=4)
        assert not result["plausible"], (
            "LFMC mean 37.7% in April passed the range check — "
            "this is the known-bad value (Globe-LFMC April median "
            "is 119%) and should have been caught"
        )

    def test_april_119_percent_is_plausible(self, manifest):
        """The expected value for April should pass."""
        from tanager.provenance import check_lfmc_range

        result = check_lfmc_range(119.0, scene_month=4)
        assert result["plausible"], (
            f"LFMC mean 119% in April failed the range check: "
            f"{result['detail']}"
        )

    @pytest.mark.parametrize("month,median", [
        (1, 76), (2, 94), (3, 113), (4, 119), (5, 105), (6, 86),
        (7, 73), (8, 65), (9, 61), (10, 60), (11, 63), (12, 70),
    ])
    def test_climatological_median_passes(self, manifest, month, median):
        """Each month's own median should pass the range check."""
        from tanager.provenance import check_lfmc_range

        result = check_lfmc_range(float(median), scene_month=month)
        assert result["plausible"], (
            f"Month {month} median {median}% failed its own range check: "
            f"{result['detail']}"
        )

    @pytest.mark.parametrize("month,median", [
        (1, 76), (4, 119), (9, 61),
    ])
    def test_3x_off_is_implausible(self, manifest, month, median):
        """A value 3× below the median should always fail."""
        from tanager.provenance import check_lfmc_range

        too_low = median / 3.0
        result = check_lfmc_range(too_low, scene_month=month)
        assert not result["plausible"], (
            f"LFMC {too_low:.1f}% (3× below month-{month} median "
            f"{median}%) passed — the gate is too permissive"
        )


# ---------------------------------------------------------------------------
# 6. Circular-dependency source scan
# ---------------------------------------------------------------------------


class TestCircularDependencyScan:
    """Grep the notebook source for patterns that generate training
    targets in-repo.  If found, the corresponding claim in the manifest
    must exist and be marked synthetic.

    This catches:
    - inject_water_absorption() → synthetic LFMC targets
    - CBI = N * fraction → synthetic CBI targets (if it ever returns)
    - rng.uniform for training targets
    """

    @staticmethod
    def _notebook_source(path: Path) -> str:
        with open(path) as f:
            nb = json.load(f)
        parts = []
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                parts.append("".join(cell.get("source", [])))
        return "\n".join(parts)

    CIRCULAR_PATTERNS = [
        (
            r"inject_water_absorption",
            "nb03_lfmc_plsr_synthetic",
            "inject_water_absorption() generates synthetic LFMC targets",
        ),
        (
            r"(?:cbi|CBI)\s*=\s*\d+\s*\*\s*(?:char|fractions?\[)",
            None,
            "CBI computed as a linear function of the fraction being predicted",
        ),
        (
            r"rng\.uniform\(\s*\d+.*?,\s*\d+.*?\)\s*$",
            None,
            "rng.uniform() used to generate training targets",
        ),
    ]

    def test_inject_water_absorption_is_flagged(self, manifest):
        """The synthetic LFMC path was removed from NB03. This test is
        the guard that keeps it out: if inject_water_absorption() (or any
        synthetic LFMC training target) ever returns to the notebook, a
        matching synthetic-labelled claim must exist in the manifest.

        NB03 now ships spectral water indices only; the calibrated LFMC
        regression is gated behind a data-availability assessment rather
        than trained on a self-generated target."""
        nb03 = NOTEBOOKS_DIR / "03-fuel-moisture.ipynb"
        if not nb03.exists():
            pytest.skip("Notebook 03 not found")

        source = self._notebook_source(nb03)
        if "inject_water_absorption" not in source:
            # Expected steady state: synthetic path is gone, and no
            # synthetic PLSR claim should linger in the manifest.
            assert "nb03_lfmc_plsr_synthetic" not in manifest["claims"], (
                "inject_water_absorption() is no longer in NB03 but a "
                "nb03_lfmc_plsr_synthetic claim still exists — remove the "
                "stale synthetic claim from the manifest"
            )
            return

        # Regression guard: if the synthetic path ever comes back, it
        # must be declared as a synthetic claim.
        claim = manifest["claims"].get("nb03_lfmc_plsr_synthetic")
        assert claim is not None and claim.get("synthetic", False), (
            "inject_water_absorption() found in NB03 but no synthetic "
            "nb03_lfmc_plsr_synthetic claim in the manifest — every "
            "synthetic training path must have a synthetic claim entry"
        )

    def test_no_cbi_formula_in_notebooks(self, manifest):
        """CBI = N × char was removed; this catches it returning."""
        for nb_path in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
            source = self._notebook_source(nb_path)
            match = re.search(
                r"(?:cbi|CBI)\s*=\s*\d+\.?\d*\s*\*\s*(?:char|fractions?\[)",
                source,
            )
            if match:
                claim_key = f"{nb_path.stem}_cbi_formula"
                claim = manifest["claims"].get(claim_key)
                if claim is None or not claim.get("synthetic", False):
                    pytest.fail(
                        f"CBI = N × fraction found in {nb_path.name} "
                        f"({match.group()!r}) but no synthetic claim "
                        "in the manifest. This is a circular dependency: "
                        "the model's target is derived from its own "
                        "prediction feature."
                    )


# ---------------------------------------------------------------------------
# 7. Implausible score check
# ---------------------------------------------------------------------------


class TestImplausibleScores:
    """R² > 0.95 for a physical remote-sensing retrieval is a bug
    report, not a result.  This test scans the manifest's claims for
    any non-synthetic metric above the threshold.

    In this project the genuinely-referenced kappa is 0.527. The
    implausible ones were 0.998 and 0.904 — both circular.
    """

    IMPLAUSIBLE_R2 = 0.95

    def test_no_implausible_nonsynthetic_r2(self, manifest):
        for key, claim in manifest["claims"].items():
            if claim.get("synthetic", False):
                continue
            metric = (claim.get("metric") or "").lower()
            value = claim.get("value")
            if value is None:
                continue
            if metric in ("r2", "r²", "cv_r2", "cv_r²"):
                assert value < self.IMPLAUSIBLE_R2, (
                    f"Claim {key!r} reports {metric}={value} — "
                    f"above the {self.IMPLAUSIBLE_R2} implausibility "
                    "threshold for a physical retrieval. Either the "
                    "metric is circular (mark it synthetic) or the "
                    "threshold needs justification."
                )


# ---------------------------------------------------------------------------
# 8. Claims table renders without error
# ---------------------------------------------------------------------------


class TestClaimsTable:
    def test_render_claims_table(self):
        from tanager.provenance import render_claims_table

        table = render_claims_table()
        assert "| Claim |" in table
        assert "synthetic" in table.lower() or "Synthetic" in table
        lines = table.strip().split("\n")
        assert len(lines) >= 3, "Claims table has fewer than 3 lines (header + separator + 1 claim)"
