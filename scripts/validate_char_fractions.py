"""Validate a MESMA char-fraction raster against external burn references.

Two comparisons, both against sources outside the imagery:

* inside vs outside the official interagency fire perimeter;
* CAL FIRE DINS ``Destroyed (>50%)`` vs ``No Damage`` structure points.

Reported per comparison: n, median, mean, and the rank-based probability that a
burned pixel scores above an unburned one (AUC — 0.5 is no discrimination,
1.0 is perfect). AUC is the number to read: a difference of medians can look
large while the distributions still overlap almost completely.

Usage::

    python scripts/validate_char_fractions.py outputs/20250123swath2_frac_char.tif \
        --incident PALISADES --dins
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from palisades_common import point_indices, polygon_masks  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("validate")


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """P(positive > negative), ties counted as half. Equivalent to Mann-Whitney U / (n*m)."""
    from scipy.stats import rankdata

    if positive.size == 0 or negative.size == 0:
        return float("nan")
    combined = np.concatenate([positive, negative])
    ranks = rankdata(combined)
    rank_sum = float(ranks[: positive.size].sum())
    n, m = positive.size, negative.size
    return (rank_sum - n * (n + 1) / 2.0) / (n * m)


def describe(values: np.ndarray) -> str:
    return (
        f"n={values.size:>7d} median={np.median(values):.3f} mean={np.mean(values):.3f} "
        f"p90={np.percentile(values, 90):.3f}"
    )


def report(name: str, positive: np.ndarray, negative: np.ndarray,
           pos_label: str, neg_label: str) -> dict:
    print(f"\n--- {name} ---")
    print(f"  {pos_label:<22s} {describe(positive)}")
    print(f"  {neg_label:<22s} {describe(negative)}")
    score = auc(positive, negative)
    print(f"  separation             median diff = {np.median(positive) - np.median(negative):+.3f}"
          f"   AUC = {score:.3f}")
    return {
        "comparison": name,
        "n_positive": int(positive.size),
        "n_negative": int(negative.size),
        "median_positive": float(np.median(positive)),
        "median_negative": float(np.median(negative)),
        "auc": float(score),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raster", type=Path)
    parser.add_argument("--incident", default="PALISADES")
    parser.add_argument("--dins", action="store_true",
                        help="also compare DINS destroyed vs no-damage points")
    args = parser.parse_args()

    import rioxarray  # noqa: F401
    import xarray as xr

    data = xr.open_dataarray(args.raster, engine="rasterio").squeeze("band", drop=True)
    values = np.asarray(data.values, dtype=np.float64)
    finite = np.isfinite(values)
    print(f"{args.raster.name}: {finite.sum()} / {finite.size} pixels modeled "
          f"({100.0 * finite.mean():.1f}%)")

    inside, outside = polygon_masks(data, args.incident)
    results = [
        report(
            f"{args.incident} perimeter (official interagency)",
            values[inside & finite],
            values[outside & finite],
            "inside perimeter",
            "outside all perimeters",
        )
    ]

    if args.dins:
        for label, damage in (("destroyed", "Destroyed (>50%)"), ("no_damage", "No Damage")):
            row, col = point_indices(data, damage)
            log.info("DINS %s: %d points land on the raster grid", label, row.size)
        row_d, col_d = point_indices(data, "Destroyed (>50%)")
        row_n, col_n = point_indices(data, "No Damage")
        destroyed = values[row_d, col_d]
        no_damage = values[row_n, col_n]
        results.append(
            report(
                "CAL FIRE DINS structure damage",
                destroyed[np.isfinite(destroyed)],
                no_damage[np.isfinite(no_damage)],
                "destroyed (>50%)",
                "no damage",
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
