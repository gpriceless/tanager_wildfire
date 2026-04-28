"""Tanager sensor configuration, constants, and data directory resolution.

This module is stdlib-only (os, pathlib, types) and must remain importable
without any optional heavy dependencies such as rasterio, hypercoast, or xarray.

Module-level constants
----------------------
SENSOR          : SimpleNamespace   Tanager-1 sensor parameters (dot-accessible)
BAD_BAND_RANGES : list[tuple]       Wavelength ranges (nm) to exclude from analysis
FIRE_SCENES     : dict[str, dict]   Hardcoded catalog of known fire collection scenes
BAND_ALIASES    : dict[str, int]    Common band name → center wavelength (nm)
DATA_DIR        : pathlib.Path      Root directory for raw fire scene HDF5 files
"""

import os
from pathlib import Path
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Sensor parameters
# ---------------------------------------------------------------------------

SENSOR = SimpleNamespace(
    name="Tanager-1",
    # Number of spectral bands (confirmed from HDF5 metadata and research)
    n_bands=426,
    # Spectral range reported in Planet documentation (nm)
    wavelength_min_nm=380,
    wavelength_max_nm=2500,
    # Nominal spectral resolution; actual FWHM per band ~5.4 nm (see research)
    spectral_resolution_nm=5,
    # Ground sampling distance (metres)
    spatial_resolution_m=30,
    # Swath width (km).  Planet documentation cites 18 km; confirm via STAC bbox
    # when the catalog module queries live metadata.
    swath_width_km=18,
)

# ---------------------------------------------------------------------------
# Bad band ranges
# Each tuple is (low_nm, high_nm) — inclusive on both ends.
# Bands whose centre wavelength falls within any range should be excluded
# before spectral analysis to avoid atmospheric absorption artefacts and
# sensor edge effects.
# ---------------------------------------------------------------------------

BAD_BAND_RANGES: list[tuple[int, int]] = [
    (0, 400),       # sensor edge / UV — below reliable detector response
    (1340, 1480),   # water vapour absorption band 1
    (1790, 1960),   # water vapour absorption band 2
    (2350, 2500),   # CO2 / H2O absorption at long-wave sensor edge
]

# ---------------------------------------------------------------------------
# Fire scene catalog
#
# Source: research/tanager-data-access-evaluation.md, Section 4.
#
# NOTE: The proposal assumed 12 scenes; Tobler's research confirmed 11 in the
# static STAC catalog as of 2026-04-27.  A possible 12th scene may appear
# when the live catalog is queried at runtime.  Use catalog.list_fire_scenes()
# as the source of truth for the authoritative scene count and metadata.
#
# bbox values are set to None here because accurate bounding boxes require a
# live STAC query (pystac reads item.bbox from catalog.json on the fly).
# The catalog module populates bbox when scenes are fetched at runtime.
#
# Phases follow the competition analysis timeline relative to the
# Palisades/Eaton fires (ignition: 2025-01-07, Los Angeles area):
#   pre-fire        — before ignition
#   post-fire       — days 0–30 post-ignition
#   early-recovery  — days 31–120
#   mid-recovery    — days 121–270
#   late-recovery   — days 271–365
#   other           — outside the primary fire footprint
# ---------------------------------------------------------------------------

FIRE_SCENES: dict[str, dict] = {
    "20241215_185916_33_4001": {
        "datetime": "2024-12-15T18:59:16Z",
        "phase": "pre-fire",
        "days_relative_to_ignition": -23,
        "notes": "Pre-fire baseline; ~23 days before Palisades/Eaton ignition",
        "bbox": None,  # Populated at runtime via catalog.list_fire_scenes()
    },
    "20250123_185507_64_4001": {
        "datetime": "2025-01-23T18:55:07Z",
        "phase": "post-fire",
        "days_relative_to_ignition": 16,
        "notes": "Immediate post-fire; primary swath over burn scar",
        "bbox": None,
    },
    "20250123_185518_92_4001": {
        "datetime": "2025-01-23T18:55:18Z",
        "phase": "post-fire",
        "days_relative_to_ignition": 16,
        "notes": "Immediate post-fire; adjacent swath",
        "bbox": None,
    },
    "20250407_192235_24_4001": {
        "datetime": "2025-04-07T19:22:35Z",
        "phase": "early-recovery",
        "days_relative_to_ignition": 90,
        "notes": "Early recovery (~90 days post-ignition); primary swath",
        "bbox": None,
    },
    "20250407_192229_16_4001": {
        "datetime": "2025-04-07T19:22:29Z",
        "phase": "early-recovery",
        "days_relative_to_ignition": 90,
        "notes": "Early recovery (~90 days post-ignition); adjacent swath",
        "bbox": None,
    },
    "20250724_190927_83_4001": {
        "datetime": "2025-07-24T19:09:27Z",
        "phase": "mid-recovery",
        "days_relative_to_ignition": 198,
        "notes": "Mid-recovery (~200 days post-ignition)",
        "bbox": None,
    },
    "20250726_192343_21_4001": {
        "datetime": "2025-07-26T19:23:43Z",
        "phase": "mid-recovery",
        "days_relative_to_ignition": 200,
        "notes": "Mid-recovery (~200 days post-ignition)",
        "bbox": None,
    },
    "20250726_192422_87_4001": {
        "datetime": "2025-07-26T19:24:22Z",
        "phase": "mid-recovery",
        "days_relative_to_ignition": 200,
        "notes": "Mid-recovery (~200 days post-ignition); adjacent swath",
        "bbox": None,
    },
    "20250902_190116_02_4001": {
        "datetime": "2025-09-02T19:01:16Z",
        "phase": "late-recovery",
        "days_relative_to_ignition": 238,
        "notes": "Late recovery (~240 days post-ignition)",
        "bbox": None,
    },
    "20250902_190121_86_4001": {
        "datetime": "2025-09-02T19:01:21Z",
        "phase": "late-recovery",
        "days_relative_to_ignition": 238,
        "notes": "Late recovery (~240 days post-ignition); adjacent swath",
        "bbox": None,
    },
    "20250920_193207_61_4001": {
        "datetime": "2025-09-20T19:32:07Z",
        "phase": "other",
        "days_relative_to_ignition": None,
        "notes": "Likely Northern Arizona; outside primary Palisades/Eaton footprint",
        "bbox": None,
    },
}

# ---------------------------------------------------------------------------
# Band aliases
# Maps common spectral band names to their nominal centre wavelength (nm).
# Use with select_bands() in spectral.py to identify band indices by name.
# ---------------------------------------------------------------------------

BAND_ALIASES: dict[str, int] = {
    "BLUE": 470,
    "GREEN": 560,
    "RED": 660,
    "RED_EDGE": 705,
    "NIR": 860,
    "SWIR1": 1610,
    "SWIR2": 2200,
}

# ---------------------------------------------------------------------------
# Data directory
#
# Default resolves to  <project_root>/data/raw/fire/  where project_root is
# three levels above this file: src/tanager/config.py → src/tanager → src
#   → project_root.
#
# Override by setting the TANAGER_DATA_DIR environment variable, e.g.:
#   export TANAGER_DATA_DIR=/mnt/fast_storage/tanager/fire
#
# The directory is not required to exist at import time; consumers should
# call DATA_DIR.mkdir(parents=True, exist_ok=True) before writing files.
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR: Path = (
    Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "fire"
)
DATA_DIR: Path = Path(os.environ.get("TANAGER_DATA_DIR", _DEFAULT_DATA_DIR))
