#!/usr/bin/env python3
"""Declare nodata and stamp units / long_name on product rasters already on disk.

The pipeline writes these tags at save time (``tanager.io.write_product_raster``),
but products generated before that save path existed carry ``nodata=None`` beside
large NaN regions and no units, and their ``long_name`` says
``surface_reflectance`` — the array the index was derived from rather than the
index itself. A reader cannot tell a fraction raster from a percent raster from
such a file, and tools that honour the nodata flag treat NaN cells as data.

This is a metadata-only backfill. Pixel values are never recomputed, clipped, or
filled: each file's band data is read before and after the update and the script
fails loudly if a single value or its NaN mask changed. Anything that looks wrong
in the numbers is a finding for the range check, not something to fix here.

Usage:
    python scripts/tag_product_rasters.py [PATH ...] [--dry-run]

With no PATH, every ``.tif`` under ``outputs/`` is processed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tanager.io import product_metadata, stamp_raster_metadata  # noqa: E402


def _read_bands(path: Path) -> tuple[np.ndarray, float | None, dict, tuple]:
    import rasterio

    with rasterio.open(path) as src:
        return src.read(), src.nodata, dict(src.tags()), tuple(src.descriptions)


def tag_file(path: Path, dry_run: bool = False) -> str:
    """Stamp one raster and verify its pixels survived unchanged.

    Returns a one-line status for the caller to print.

    Raises:
        KeyError: If the product is not registered in ``PRODUCT_METADATA``.
        RuntimeError: If any pixel value or NaN position changed.
    """
    meta = product_metadata(path.stem)
    if meta is None:
        raise KeyError(
            f"No product metadata registered for {path.stem!r}; add it to "
            f"tanager.io.PRODUCT_METADATA"
        )

    before, nodata_before, tags_before, desc_before = _read_bands(path)
    if dry_run:
        return (
            f"DRY-RUN {path.name}: nodata {nodata_before!r} -> nan, "
            f"units {tags_before.get('units')!r} -> {meta['units']!r}, "
            f"long_name {desc_before[0]!r} -> {meta['long_name']!r}"
        )

    stamp_raster_metadata(path, meta=meta)

    after, nodata_after, tags_after, _ = _read_bands(path)
    if before.shape != after.shape:
        raise RuntimeError(f"{path}: shape changed {before.shape} -> {after.shape}")
    if not np.array_equal(np.isnan(before), np.isnan(after)):
        raise RuntimeError(f"{path}: NaN mask changed")
    if not np.array_equal(before, after, equal_nan=True):
        raise RuntimeError(f"{path}: pixel values changed")

    for key in ("units", "long_name"):
        if tags_after.get(key) != meta[key]:
            raise RuntimeError(
                f"{path}: tag {key!r} read back as {tags_after.get(key)!r}, "
                f"expected {meta[key]!r}"
            )
    if nodata_after is None or not np.isnan(nodata_after):
        raise RuntimeError(f"{path}: nodata read back as {nodata_after!r}, expected nan")

    return (
        f"OK {path.name}: nodata {nodata_before!r} -> nan, "
        f"units={meta['units']!r}, long_name={meta['long_name']!r} "
        f"({before.size} pixels verified unchanged)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without writing",
    )
    args = parser.parse_args(argv)

    paths = args.paths or sorted((REPO_ROOT / "outputs").glob("*.tif"))
    if not paths:
        print("no rasters found", file=sys.stderr)
        return 1

    failures = 0
    for path in paths:
        try:
            print(tag_file(path, dry_run=args.dry_run))
        except (KeyError, RuntimeError) as exc:
            failures += 1
            print(f"FAIL {path.name}: {exc}", file=sys.stderr)

    print(f"\n{len(paths) - failures}/{len(paths)} rasters stamped, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
