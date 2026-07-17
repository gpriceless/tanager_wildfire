"""Provenance manifest and claim-integrity gate.

Loads ``provenance.json`` from the project root (the file next to
``pyproject.toml``) and exposes helpers that answer three questions:

1. **Do the bundled artifacts match their recorded checksums?**
   :func:`verify_artifact` computes the MD5 of a file on disk and
   compares it against the manifest entry.

2. **Is a predicted LFMC value physically plausible?**
   :func:`check_lfmc_range` compares a scene-mean LFMC against the
   Globe-LFMC 2.0 SoCal shrub monthly climatology stored in the
   manifest. A mean outside ``[median / tol, median * tol]`` for the
   scene month is flagged as implausible.

3. **Does every headline claim have a backing artifact?**
   :func:`audit_claims` returns a list of claims whose backing artifact
   is missing, whose artifact has unverifiable provenance, or whose
   metric is derived from synthetic (in-repo-generated) targets without
   being labelled as such.

The manifest is deliberately conservative: artifacts whose provenance
cannot be established are declared ``"provenance_status": "unverifiable"``
rather than passing silently, and claims derived from synthetic targets
are marked ``"synthetic": true`` so downstream consumers can distinguish
methodology demonstrations from accuracy measurements.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MANIFEST_PATH = _PROJECT_ROOT / "provenance.json"

_manifest_cache: Optional[dict[str, Any]] = None


def _load_manifest() -> dict[str, Any]:
    global _manifest_cache
    if _manifest_cache is not None:
        return _manifest_cache
    if not _MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Provenance manifest not found at {_MANIFEST_PATH}. "
            "Run from the project root or set the path explicitly."
        )
    with open(_MANIFEST_PATH) as f:
        _manifest_cache = json.load(f)
    return _manifest_cache


def manifest() -> dict[str, Any]:
    """Return the parsed provenance manifest (cached after first load)."""
    return _load_manifest()


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_artifact(key: str) -> dict[str, Any]:
    """Verify a single artifact's checksum against the manifest.

    Returns a dict with ``key``, ``status`` (``"ok"``, ``"missing"``,
    ``"mismatch"``, ``"not_bundled"``, ``"no_checksum"``), and
    ``detail``.
    """
    m = _load_manifest()
    entry = m["artifacts"].get(key)
    if entry is None:
        return {"key": key, "status": "unknown", "detail": f"No manifest entry for {key!r}"}

    if not entry.get("bundled", False):
        return {"key": key, "status": "not_bundled", "detail": "Artifact not bundled in repo"}

    expected_md5 = entry.get("checksum_md5")
    if not expected_md5:
        return {"key": key, "status": "no_checksum", "detail": "No checksum recorded"}

    path = _PROJECT_ROOT / entry["path"]
    if not path.exists():
        return {"key": key, "status": "missing", "detail": f"File not found: {path}"}

    actual = _md5(path)
    if actual != expected_md5:
        return {
            "key": key,
            "status": "mismatch",
            "detail": f"Expected MD5 {expected_md5}, got {actual}",
        }
    return {"key": key, "status": "ok", "detail": "Checksum verified"}


def verify_all_bundled() -> list[dict[str, Any]]:
    """Verify checksums for every artifact marked ``bundled: true``."""
    m = _load_manifest()
    results = []
    for key, entry in m["artifacts"].items():
        if entry.get("bundled", False):
            results.append(verify_artifact(key))
    return results


def check_lfmc_range(
    mean_lfmc_percent: float,
    scene_month: int,
) -> dict[str, Any]:
    """Check whether a predicted LFMC mean is plausible for a given month.

    Uses the Globe-LFMC 2.0 SoCal shrub monthly climatology from the
    manifest. Returns a dict with ``plausible`` (bool), ``expected_median``,
    ``tolerance_factor``, ``low_bound``, ``high_bound``, and ``detail``.

    A mean outside ``[median / tol, median * tol]`` is implausible.
    This would have caught the 3×-off LFMC map (mean 37.7% on an April
    scene when the Globe-LFMC median is 119%).
    """
    m = _load_manifest()
    clim = m["climatology"]["globe_lfmc_socal_shrub_monthly_median"]
    month_key = str(scene_month)
    if month_key not in clim["values"]:
        return {
            "plausible": False,
            "detail": f"No climatology for month {scene_month}",
        }

    median = float(clim["values"][month_key])
    tol = float(clim["tolerance_factor"])
    low = median / tol
    high = median * tol

    plausible = low <= mean_lfmc_percent <= high
    return {
        "plausible": plausible,
        "mean_lfmc": mean_lfmc_percent,
        "expected_median": median,
        "tolerance_factor": tol,
        "low_bound": low,
        "high_bound": high,
        "detail": (
            f"LFMC mean {mean_lfmc_percent:.1f}% is "
            + ("within" if plausible else "OUTSIDE")
            + f" [{low:.0f}, {high:.0f}]% "
            + f"(month {scene_month} median {median:.0f}%, tol {tol}×)"
        ),
    }


def audit_claims() -> list[dict[str, Any]]:
    """Audit every claim in the manifest for integrity.

    Returns a list of findings. Each finding has ``claim_key``,
    ``severity`` (``"error"`` or ``"warning"``), and ``detail``.
    An empty list means all claims pass.
    """
    m = _load_manifest()
    artifacts = m.get("artifacts", {})
    claims = m.get("claims", {})
    findings: list[dict[str, Any]] = []

    for claim_key, claim in claims.items():
        backing = claim.get("backing_artifact")

        if backing is not None and backing not in artifacts:
            findings.append({
                "claim_key": claim_key,
                "severity": "error",
                "detail": f"Backing artifact {backing!r} not found in manifest",
            })

        if backing is not None and backing in artifacts:
            art = artifacts[backing]
            if art.get("provenance_status") == "unverifiable":
                findings.append({
                    "claim_key": claim_key,
                    "severity": "warning",
                    "detail": (
                        f"Backing artifact {backing!r} has unverifiable provenance: "
                        + (art.get("provenance_note") or "no note")
                    ),
                })

        if backing is None and not claim.get("synthetic", False):
            findings.append({
                "claim_key": claim_key,
                "severity": "error",
                "detail": "Claim has no backing artifact and is not marked synthetic",
            })

        if claim.get("independence") == "none" and not claim.get("synthetic", False):
            findings.append({
                "claim_key": claim_key,
                "severity": "error",
                "detail": (
                    "Claim has independence=none (targets generated in-repo) "
                    "but is not marked synthetic"
                ),
            })

    return findings


def render_claims_table() -> str:
    """Render the claims table as a markdown table.

    This is the machine-generated claims table the issue asks for —
    generated from the manifest, not hand-maintained, so it cannot
    drift from the code.
    """
    m = _load_manifest()
    claims = m.get("claims", {})
    artifacts = m.get("artifacts", {})

    lines = [
        "| Claim | Metric | Value | Backing artifact | Independence | Synthetic |",
        "|-------|--------|-------|-----------------|--------------|-----------|",
    ]
    for key, c in claims.items():
        value = c.get("value")
        value_str = f"{value}" if value is not None else "—"
        backing = c.get("backing_artifact", "—") or "NONE"
        if backing in artifacts:
            art = artifacts[backing]
            prov = art.get("provenance_status", "unknown")
            if prov == "unverifiable":
                backing += " ⚠"
        independence = c.get("independence", "—")
        synthetic = "yes" if c.get("synthetic") else "no"
        lines.append(
            f"| {c.get('description', key)} "
            f"| {c.get('metric', '—')} "
            f"| {value_str} "
            f"| {backing} "
            f"| {independence} "
            f"| {synthetic} |"
        )
    return "\n".join(lines)
