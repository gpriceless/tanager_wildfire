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
# Runtime / performance tuning
# ---------------------------------------------------------------------------
#
# Default joblib worker cap for the parallel stages (continuum removal in
# spectral.py, RandomForest training in severity.py).
#
# Each continuum-removal loky worker peaks near ~1.3 GB on a 426-band scene,
# so an unbounded ``n_jobs=-1`` (one worker per core) can exceed available RAM.
# Bounding n_jobs makes memory predictable: 4 workers ≈ 5 GB per notebook.
# Heavy one-off runs can raise it via TANAGER_MAX_JOBS.

DEFAULT_MAX_JOBS = 4


def parallel_jobs(default: int = DEFAULT_MAX_JOBS) -> int:
    """Return the joblib worker cap, honouring the ``TANAGER_MAX_JOBS`` env var.

    ``TANAGER_MAX_JOBS`` accepts a positive integer (explicit worker count) or a
    negative value to mean "all cores" (the legacy unbounded behaviour — opt-in
    only, since unbounded parallelism can OOM). ``0``, empty, or invalid
    values fall back to ``default``.
    """
    raw = os.environ.get("TANAGER_MAX_JOBS")
    if not raw or not raw.strip():
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    if val < 0:
        return -1  # explicit opt-in to all-cores
    return val or default

# ---------------------------------------------------------------------------
# Reference sensor specifications (for spectral degradation simulations)
#
# Used by validation.simulate_sensor() to convolve Tanager-1 native 426-band
# spectra into the lower-resolution channels of EMIT, PRISMA, and Sentinel-2
# for the +5 competition tie-breaker (Tanager vs reference sensor).
#
# Sources: research/sensor-comparison-research.md section 5.1.
# ---------------------------------------------------------------------------

EMIT_SENSOR = SimpleNamespace(
    name="EMIT",
    n_bands=285,
    wavelength_min_nm=381,
    wavelength_max_nm=2493,
    spectral_resolution_nm=7.4,
    fwhm_nm=8.5,
    spatial_resolution_m=60,
)

PRISMA_SENSOR = SimpleNamespace(
    name="PRISMA",
    n_bands=239,
    wavelength_min_nm=400,
    wavelength_max_nm=2505,
    spectral_resolution_nm=12,
    fwhm_nm=12,
    spatial_resolution_m=30,
)

# Sentinel-2 MSI (MultiSpectral Instrument) — 10 bands relevant to vegetation /
# burn analysis (visible, red-edge, NIR, SWIR). FWHM is the published
# bandwidth; gsd_m records the native ground sample distance per band.
SENTINEL2_BANDS: dict[str, dict[str, float]] = {
    "B2":  {"center_nm": 490,  "fwhm_nm": 65,  "gsd_m": 10},
    "B3":  {"center_nm": 560,  "fwhm_nm": 35,  "gsd_m": 10},
    "B4":  {"center_nm": 665,  "fwhm_nm": 30,  "gsd_m": 10},
    "B5":  {"center_nm": 705,  "fwhm_nm": 15,  "gsd_m": 20},
    "B6":  {"center_nm": 740,  "fwhm_nm": 15,  "gsd_m": 20},
    "B7":  {"center_nm": 783,  "fwhm_nm": 20,  "gsd_m": 20},
    "B8":  {"center_nm": 842,  "fwhm_nm": 115, "gsd_m": 10},
    "B8A": {"center_nm": 865,  "fwhm_nm": 20,  "gsd_m": 20},
    "B11": {"center_nm": 1610, "fwhm_nm": 90,  "gsd_m": 20},
    "B12": {"center_nm": 2190, "fwhm_nm": 180, "gsd_m": 20},
}

# ---------------------------------------------------------------------------
# Bad band ranges
# Each tuple is (low_nm, high_nm) — inclusive on both ends.
# Bands whose centre wavelength falls within any range should be excluded
# before spectral analysis to avoid atmospheric absorption artefacts and
# sensor edge effects.
# ---------------------------------------------------------------------------

BAD_BAND_RANGES: list[tuple[int, int]] = [
    (0, 400),       # sensor edge / UV — below reliable detector response
    (1340, 1480),   # water vapour absorption band 1 (sensor flags 1342.41–1437.55 nm)
    (1780, 1970),   # water vapour absorption band 2 (sensor flags 1782.58–1967.21 nm)
    (2350, 2500),   # CO2 / H2O absorption at long-wave sensor edge
]

# ---------------------------------------------------------------------------
# Fire complex ignition dates
#
# The LA-area Tanager scenes span two distinct fire complexes separated by
# ~40 km.  days_relative_to_ignition in FIRE_SCENES is computed against
# the ignition date of the fire_complex each scene covers.
# ---------------------------------------------------------------------------

FIRE_IGNITION_DATES: dict[str, str] = {
    "palisades": "2025-01-07",   # Palisades Fire (& nearby Eaton Fire, same day)
    "hughes":    "2025-01-22",   # Hughes Fire, Castaic — 15 days later
}

# ---------------------------------------------------------------------------
# Fire scene catalog
#
# Source: research/tanager-data-access-evaluation.md
#
# NOTE: The initial estimate was 12 scenes; verification confirmed 11 in the
# static STAC catalog as of 2026-04-27.  A possible 12th scene may appear
# when the live catalog is queried at runtime.  Use catalog.list_fire_scenes()
# as the source of truth for the authoritative scene count and metadata.
#
# bbox values are set to None here because accurate bounding boxes require a
# live STAC query (pystac reads item.bbox from catalog.json on the fly).
# The catalog module populates bbox when scenes are fetched at runtime.
#
# fire_complex identifies which fire each scene covers; this determines the
# ignition date used to compute days_relative_to_ignition.  Scenes verified
# against HDF5 grid coordinates on 2026-07-13.
#
# Phases follow the competition analysis timeline relative to each scene's
# fire_complex ignition date:
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
        "fire_complex": "palisades",
        "days_relative_to_ignition": -23,
        "notes": "Pre-fire baseline (Palisades coast, ~34.0°N)",
        "bbox": None,  # Populated at runtime via catalog.list_fire_scenes()
    },
    "20250123_185507_64_4001": {
        "datetime": "2025-01-23T18:55:07Z",
        "phase": "post-fire",
        "fire_complex": "hughes",
        "days_relative_to_ignition": 1,
        "notes": "Hughes Fire area (Castaic, ~34.5°N); 1 day post-Hughes ignition",
        "bbox": None,
    },
    "20250123_185518_92_4001": {
        "datetime": "2025-01-23T18:55:18Z",
        "phase": "post-fire",
        "fire_complex": "palisades",
        "days_relative_to_ignition": 16,
        "notes": "Palisades Fire area (~34.0°N); overlaps Dec 15 pre-fire footprint",
        "bbox": None,
    },
    "20250407_192235_24_4001": {
        "datetime": "2025-04-07T19:22:35Z",
        "phase": "early-recovery",
        "fire_complex": "hughes",
        "days_relative_to_ignition": 75,
        "notes": "Hughes Fire area (Castaic, ~34.5°N); 75 days post-Hughes ignition",
        "bbox": None,
    },
    "20250407_192229_16_4001": {
        "datetime": "2025-04-07T19:22:29Z",
        "phase": "early-recovery",
        "fire_complex": None,
        "days_relative_to_ignition": 90,
        "notes": "Adjacent swath; fire_complex unverified (no local HDF5)",
        "bbox": None,
    },
    "20250724_190927_83_4001": {
        "datetime": "2025-07-24T19:09:27Z",
        "phase": "other",
        "fire_complex": None,
        "days_relative_to_ignition": None,
        "notes": "Utah (38.5°N, -112°W); outside LA fire footprint",
        "bbox": None,
    },
    "20250726_192343_21_4001": {
        "datetime": "2025-07-26T19:23:43Z",
        "phase": "mid-recovery",
        "fire_complex": None,
        "days_relative_to_ignition": 200,
        "notes": "Mid-recovery; fire_complex unverified (no local HDF5)",
        "bbox": None,
    },
    "20250726_192422_87_4001": {
        "datetime": "2025-07-26T19:24:22Z",
        "phase": "mid-recovery",
        "fire_complex": None,
        "days_relative_to_ignition": 200,
        "notes": "Mid-recovery; adjacent swath; fire_complex unverified",
        "bbox": None,
    },
    "20250902_190116_02_4001": {
        "datetime": "2025-09-02T19:01:16Z",
        "phase": "other",
        "fire_complex": None,
        "days_relative_to_ignition": None,
        "notes": "Utah (38.5°N, -112°W); outside LA fire footprint",
        "bbox": None,
    },
    "20250902_190121_86_4001": {
        "datetime": "2025-09-02T19:01:21Z",
        "phase": "other",
        "fire_complex": None,
        "days_relative_to_ignition": None,
        "notes": "Utah (38.5°N, -112°W); outside LA fire footprint",
        "bbox": None,
    },
    "20250920_193207_61_4001": {
        "datetime": "2025-09-20T19:32:07Z",
        "phase": "late-recovery",
        "fire_complex": None,
        "days_relative_to_ignition": 256,
        "notes": "Late recovery; LA area (33.9°N, -118.5°W); fire_complex unverified",
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
