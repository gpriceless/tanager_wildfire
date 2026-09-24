"""Build the fire endmember library used to unmix the Palisades post-fire scene.

Every spectrum comes from the USGS Spectral Library Version 7 (splib07a,
``ASCIIdata_splib07a.zip``, DOI:10.5066/F7RR1WDJ). Nothing here is derived from
the Tanager imagery, so the library carries no knowledge of where the fire
burned and the perimeter/DINS comparisons downstream stay independent.

The selection is written out by hand rather than discovered by a filename
heuristic: the heuristic maps "Cheatgrass" to photosynthetic vegetation and
"Grass_Golden_Dry" to the same class, which is the sort of silent mislabel that
only shows up as a bad fraction map. See ``data/reference/endmembers/PROVENANCE.md``
for why each spectrum is in or out.

Usage::

    python scripts/build_fire_endmember_library.py [--scene SCENE_H5]

Writes ``data/reference/endmembers/palisades_fire_library.csv`` (resampled to
the Tanager band centres of ``--scene``) and prints a per-class summary.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tanager.endmembers import load_usgs_library, resample_library  # noqa: E402
from tanager.io import load_ortho_scene  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_endmembers")

# ScienceBase file record for ASCIIdata_splib07a.zip under item 586e8c88e4b0f5ce109fccae
# ("Spectra of materials in ASCII format"), the child item of the splib07 data
# release 5807a2a2e4b0841e59e3a18d.
SPLIB07A_URL = (
    "https://www.sciencebase.gov/catalog/file/get/586e8c88e4b0f5ce109fccae"
    "?f=__disk__a7%2F4f%2F91%2Fa74f913e0b7d1b8123ad059e52506a02b75a2832"
)
SPLIB07A_MD5 = "bfe74068d85811e52e5e07d017720a17"  # published on the ScienceBase record

DEFAULT_SCENE = REPO_ROOT / "data" / "raw" / "fire" / "20250123_185518_92_4001_ortho_sr_hdf5.h5"
CACHE_DIR = REPO_ROOT / "data" / "downloads"
OUT_CSV = REPO_ROOT / "data" / "reference" / "endmembers" / "palisades_fire_library.csv"

# The library, spelled out. Keys are splib07a file stems; values are the fire
# class each spectrum stands for. Rationale per class is in PROVENANCE.md.
SELECTION: dict[str, str] = {
    # --- char: the only measured burned-surface spectra in splib07 ---------
    "splib07a_BurnArea_Traverse_WRF00-01_ASDFRb_AREF": "char",
    "splib07a_BurnArea_TopSurface_WRF00-02_ASDFRb_AREF": "char",
    # --- pv: the CA01 field campaign, i.e. California chaparral -----------
    "splib07a_Chamise_CA01-ADFA-1_bush_1_ASDFRa_AREF": "pv",
    "splib07a_Chamise_CA01-ADFA-2_bush_2_ASDFRa_AREF": "pv",
    "splib07a_Buckbrush_CA01-CECU-1_bush_1_ASDFRa_AREF": "pv",
    "splib07a_Buckbrush_CA01-CECU-2_bush_2_ASDFRa_AREF": "pv",
    "splib07a_Buckbrush_CA01-CECU-3_bush_3_ASDFRa_AREF": "pv",
    "splib07a_Manzanita_CA01-ARVI-1_bush_1_ASDFRa_AREF": "pv",
    "splib07a_Manzanita_CA01-ARVI-4_bush_4_ASDFRa_AREF": "pv",
    "splib07a_Manzanita_CA01-ARVI-6_bush_6_ASDFRa_AREF": "pv",
    "splib07a_Oak_QUDU_CA01-QUDU-1_bush_1_ASDFRa_AREF": "pv",
    "splib07a_Oak_QUDU_CA01-QUDU-3_bush_3_ASDFRa_AREF": "pv",
    "splib07a_Toyon_CA01-HEAR-1_bush_ASDFRa_AREF": "pv",
    "splib07a_Yerba_Santa_CA01-ERCA-1_bush_ASDFRa_AREF": "pv",
    "splib07a_Gray-Pine_CA01-PISA-1_branch_ASDFRa_AREF": "pv",
    # --- npv: cured California grass and litter ---------------------------
    "splib07a_Grass_AETR70_CA01-AETR-2_NPV_ASDFRa_AREF": "npv",
    "splib07a_Grass_AETR95_CA01-AETR-1_NPV_ASDFRa_AREF": "npv",
    "splib07a_Grass_CA01-TACA-1_meadow_NPV_ASDFRa_AREF": "npv",
    "splib07a_Grass_Golden_Dry_GDS480_ASDFRa_AREF": "npv",
    # --- soil: natural surfaces, none of them Californian (see PROVENANCE) -
    "splib07a_Sand_GrndIsle1_no_oil_ASDFRa_AREF": "soil",
    "splib07a_Sand_GrndIsle2_no_visibl_oil_ASDFRa_AREF": "soil",
    "splib07a_Stonewall_Playa_Dry_Mud_2001_ASDFRa_AREF": "soil",
    "splib07a_Pyroxene_Basalt_CU01-20A_ASDFRa_AREF": "soil",
}

# Added by --with-urban. The four-class library has nothing that can stand for a
# residential block, so every built surface in it has to be absorbed by "soil";
# these two classes exist to test whether that mis-assignment is what suppresses
# char at destroyed structures. ``debris`` is USGS's World Trade Center
# collapse-debris set — pulverised concrete, gypsum and glass fibre. It is the
# nearest measured analogue to a destroyed-structure surface in any public
# library, and it is a *collapse* product, not a combustion product: it is a
# diagnostic stand-in, not a wildfire ash measurement.
URBAN_SELECTION: dict[str, str] = {
    "splib07a_Concrete_GDS375_Lt_Gry_Road_ASDFRa_AREF": "urban",
    "splib07a_Asphalt_GDS376_Blck_Road_old_ASDFRa_AREF": "urban",
    "splib07a_Asphalt_Shingle_GDS367_DkGry_ASDFRa_AREF": "urban",
    "splib07a_Asphalt_Shingle_GDS368_Lgray_ASDFRa_AREF": "urban",
    "splib07a_Brick_GDS350_Dk_Red_Building_ASDFRa_AREF": "urban",
    "splib07a_Sheet_Metal_GDS352_crg_Galvn_ASDFRa_AREF": "urban",
    "splib07a_WTC_Dust_Debris_WTC01-2_ASDFRa_AREF": "debris",
    "splib07a_WTC_Dust_Debris_WTC01-28_ASDFRa_AREF": "debris",
    "splib07a_Concrete_WTC01-37A_ASDFRa_AREF": "debris",
}

# USGS ASD field spectrometer resolution. splib07a spectra are resampled to a
# 1 nm grid, so 1 nm is the source FWHM to deconvolve against Tanager's ~5.5 nm.
ASD_SOURCE_FWHM_NM = 1.0


def fetch_splib07a(cache_dir: Path) -> Path:
    """Download and extract ASCIIdata_splib07a.zip, returning the data root.

    The archive is verified against the MD5 published on the ScienceBase record
    before it is unpacked; a library that silently changed under us would
    invalidate every fraction downstream.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    root = cache_dir / "splib07a" / "ASCIIdata_splib07a"
    if root.is_dir():
        log.info("splib07a already extracted at %s", root)
        return root

    zip_path = cache_dir / "ASCIIdata_splib07a.zip"
    if not zip_path.exists():
        log.info("downloading ASCIIdata_splib07a.zip (~21.8 MB) from ScienceBase")
        urllib.request.urlretrieve(SPLIB07A_URL, zip_path)

    digest = hashlib.md5(zip_path.read_bytes()).hexdigest()
    if digest != SPLIB07A_MD5:
        raise RuntimeError(
            f"ASCIIdata_splib07a.zip MD5 {digest} != published {SPLIB07A_MD5}; refusing to use it"
        )

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir / "splib07a")
    if not root.is_dir():
        raise RuntimeError(f"expected {root} after extraction")
    return root


def build_library(splib_root: Path, target_wavelengths: np.ndarray,
                  target_fwhm: np.ndarray, *, with_urban: bool = False) -> xr.DataArray:
    """Select the curated spectra and resample them onto the Tanager grid."""
    selection = dict(SELECTION)
    if with_urban:
        selection.update(URBAN_SELECTION)

    full = load_usgs_library(data_dir=splib_root, sensor_hint="ASD")
    names = np.asarray(full.coords["name"].values, dtype=str)

    missing = sorted(set(selection) - set(names.tolist()))
    if missing:
        raise RuntimeError(f"{len(missing)} selected spectra not found in splib07a: {missing}")

    order = [int(np.where(names == stem)[0][0]) for stem in selection]
    subset = full.isel(spectrum_id=order)
    categories = [selection[stem] for stem in selection]
    subset = subset.assign_coords(
        category=("spectrum_id", categories),
        spectrum_id=[f"usgs_{cat}_{i:02d}" for i, cat in enumerate(categories)],
    )

    resampled = resample_library(
        subset,
        target_wavelengths,
        fwhm=target_fwhm,
        source_fwhm=ASD_SOURCE_FWHM_NM,
    )
    resampled.attrs.update(
        library_source="usgs_splib07a",
        library_doi="10.5066/F7RR1WDJ",
        library_citation=(
            "Kokaly, R.F., et al., 2017, USGS Spectral Library Version 7: "
            "U.S. Geological Survey Data Series 1035, 61 p."
        ),
        image_derived="false",
    )
    return resampled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE,
                        help="Tanager scene whose band centres/FWHM the library is resampled to")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--with-urban", action="store_true",
                        help="add the built-material and building-debris classes")
    args = parser.parse_args()
    if args.out is None:
        args.out = OUT_CSV.with_name(
            "palisades_fire_library_urban.csv" if args.with_urban else OUT_CSV.name
        )

    scene = load_ortho_scene(args.scene, wavelength_range=(370.0, 2510.0))
    wavelengths = np.asarray(scene.coords["wavelength"].values, dtype=np.float64)
    fwhm = np.asarray(scene.coords["fwhm"].values, dtype=np.float64)
    del scene

    splib_root = fetch_splib07a(CACHE_DIR)
    library = build_library(splib_root, wavelengths, fwhm, with_urban=args.with_urban)

    frame = pd.DataFrame(
        library.values,
        index=pd.Index(np.asarray(library.coords["spectrum_id"].values, dtype=str), name="spectrum_id"),
        columns=np.round(wavelengths, 3),
    )
    frame.insert(0, "category", np.asarray(library.coords["category"].values, dtype=str))
    frame.insert(1, "usgs_name", np.asarray(library.coords["name"].values, dtype=str))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, float_format="%.6f")

    cats = np.asarray(library.coords["category"].values, dtype=str)
    log.info("wrote %s", args.out.relative_to(REPO_ROOT))
    for cat in sorted(set(cats.tolist())):
        log.info("  %-7s %d spectra", cat, int((cats == cat).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
