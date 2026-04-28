"""Tanager — hyperspectral wildfire analysis pipeline for Planet Tanager-1."""

__version__ = "0.1.0"

# Public API: when one of these names is accessed, the corresponding
# submodule is imported on demand (PEP 562 lazy module __getattr__).
#
# Heavy dependencies (rasterio, geopandas, hypercoast) take 1-3 s to import.
# Deferring them means `import tanager` is instant whether you are in a
# Jupyter kernel, a CLI tool, or a unit test that only needs config constants.
#
# Submodules that do not yet exist will raise ModuleNotFoundError when accessed —
# that is expected during incremental development.

_LAZY_EXPORTS: dict[str, str] = {
    # config ---------------------------------------------------------------
    "SENSOR": "config",
    "BAD_BAND_RANGES": "config",
    "FIRE_SCENES": "config",
    "BAND_ALIASES": "config",
    "DATA_DIR": "config",
    # catalog --------------------------------------------------------------
    "list_fire_scenes": "catalog",
    "download_scene": "catalog",
    "get_scene_metadata": "catalog",
    # io -------------------------------------------------------------------
    "load_scene": "io",
    "load_ortho_scene": "io",
    "get_spatial_info": "io",
    # spectral -------------------------------------------------------------
    "select_bands": "spectral",
    "mask_bad_bands": "spectral",
    "clamp_reflectance": "spectral",
    "nbr": "spectral",
    "ndvi": "spectral",
    "ndwi": "spectral",
    "dnbr": "spectral",
    "continuum_removal": "spectral",
    # masks ----------------------------------------------------------------
    "nodata_mask": "masks",
    "cloud_mask": "masks",
    "water_mask": "masks",
    "apply_masks": "masks",
    # endmembers -----------------------------------------------------------
    "load_usgs_library": "endmembers",
    "load_ecostress_library": "endmembers",
    "load_frames_library": "endmembers",
    "resample_library": "endmembers",
    "build_hybrid_library": "endmembers",
    "select_endmembers_incob": "endmembers",
    "prune_endmembers_ear_masa": "endmembers",
    "extract_image_endmembers": "endmembers",
    "build_fire_library": "endmembers",
    # unmixing -------------------------------------------------------------
    "run_mesma": "unmixing",
    "select_bands_uszu": "unmixing",
    "normalize_fractions": "unmixing",
    "plot_fraction_maps": "unmixing",
    "plot_rgb_composite": "unmixing",
    # severity -------------------------------------------------------------
    "train_severity_model": "severity",
    "predict_severity": "severity",
    "compute_trajectories": "severity",
    "compare_severity_methods": "severity",
    # lfmc -----------------------------------------------------------------
    "compute_lfmc_indices": "lfmc",
    "train_lfmc_plsr": "lfmc",
    "predict_lfmc": "lfmc",
    "load_globe_lfmc": "lfmc",
    # validation -----------------------------------------------------------
    "compute_accuracy": "validation",
    "compare_sensors": "validation",
    "load_aviris3_reference": "validation",
    "load_barc_reference": "validation",
}


def __getattr__(name: str) -> object:
    """Lazy-import public API symbols from their home submodules.

    Args:
        name: Attribute name requested on the ``tanager`` package.

    Returns:
        The attribute from the relevant submodule.

    Raises:
        AttributeError: If ``name`` is not part of the public API.
        ModuleNotFoundError: If the submodule has not been created yet.
    """
    if name in _LAZY_EXPORTS:
        import importlib

        module = importlib.import_module(f".{_LAZY_EXPORTS[name]}", __name__)
        attr = getattr(module, name)
        # Cache in module globals so subsequent accesses skip __getattr__.
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy exports in dir() output so IDEs can autocomplete the API."""
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))


__all__ = ["__version__", *_LAZY_EXPORTS.keys()]
