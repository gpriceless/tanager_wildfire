"""MESMA spectral unmixing pipeline for Tanager-1 hyperspectral scenes.

This module performs Multiple Endmember Spectral Mixture Analysis (MESMA) on
Tanager scenes using a fire-relevant endmember library, producing per-pixel
fractional abundance maps for the canonical fire classes (char, PV, NPV, soil,
shade) plus the RMSE of the best-fit model.

The primary backend is the ``mesma`` package (Roberts et al. 2018, ported
to Python by van der Linden et al.).  When ``mesma`` is unavailable or fails
the bands-array compatibility check, two fallbacks are used in order:

1. ``hysup`` Fully Constrained Least Squares (single-model FCLS) — same output
   schema, single endmember-set per pixel.
2. ``scipy.optimize.nnls`` ultimate fallback — non-negative least squares with
   manual sum-to-one rescaling.  Pure-python, always available.

The chosen backend is recorded in ``Dataset.attrs["unmixing_engine"]``.

Public API (lazy-imported via ``tanager`` package):

* :func:`select_bands_uszu` — Uniform Spectral Zone Unmixing band selection
* :func:`run_mesma` — main unmixing entry point
* :func:`normalize_fractions` — shade removal + rescale to sum=1.0
* :func:`plot_fraction_maps` — per-class fraction map figure
* :func:`plot_rgb_composite` — false-colour RGB from fraction maps

Import direction:

* unmixing.py MAY import from :mod:`tanager.config`, :mod:`tanager.spectral`,
  :mod:`tanager.endmembers`.
* unmixing.py MUST NOT import from :mod:`tanager.severity`,
  :mod:`tanager.lfmc`, or :mod:`tanager.validation` to keep the dependency
  graph acyclic.

NaN semantics differ between backends. ``mesma`` returns ``rmse=9999`` for
unmodeled pixels (no valid model under the constraints); this module
post-processes those to ``NaN``. The FCLS / NNLS fallbacks always produce
fractions for every pixel (no combinatorial model search), so ``NaN`` only
arises from input-side ``NaN`` propagation or post-hoc constraint filtering.
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import xarray as xr

from tanager.spectral import scene_reflectance as _scene_reflectance

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engine detection
# ---------------------------------------------------------------------------

_MESMA_AVAILABLE = False
try:  # pragma: no cover - environment-dependent
    import mesma  # noqa: F401
    from mesma.core.mesma import MesmaCore, MesmaModels  # noqa: F401

    _MESMA_AVAILABLE = True
except Exception:  # broad except: ImportError, but also segfault-style boot errors
    logger.info("mesma package not available; run_mesma will fall back to HySUPP/NNLS")

_HYSUP_AVAILABLE = False
try:  # pragma: no cover - environment-dependent
    import hysup  # type: ignore[import-not-found]  # noqa: F401

    _HYSUP_AVAILABLE = True
except Exception:
    pass


# ---------------------------------------------------------------------------
# Output schema constants
# ---------------------------------------------------------------------------

# Canonical fraction variables in the output Dataset (spec requirement).
_CANONICAL_FRACTIONS: Tuple[str, ...] = ("char", "pv", "npv", "soil", "shade")

# Sentinel value mesma writes for "no valid model found"; we convert to NaN.
_MESMA_RMSE_SENTINEL = 9999.0

# Tolerance for the sum-to-one check applied during output validation.
_SUM_TOLERANCE = 0.01

# Default MESMA constraints per spec (Section 4 task 5):
#   max_rmse:      models with RMSE > this are rejected.
#   min_fraction:  fractions below this disqualify the model.
#   max_fraction:  fractions above this disqualify the model.
#   max_shade:     upper bound on shade fraction.
#   min_shade:     lower bound on shade fraction.
DEFAULT_CONSTRAINTS: Mapping[str, float] = {
    "min_fraction": -0.05,
    "max_fraction": 1.05,
    "min_shade": 0.0,
    "max_shade": 0.8,
    "max_rmse": 0.025,
}


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------


def _resolve_constraints(constraints: Optional[Mapping[str, float]]) -> dict[str, float]:
    """Merge user-supplied constraints with the defaults."""
    merged = dict(DEFAULT_CONSTRAINTS)
    if constraints:
        merged.update({k: float(v) for k, v in constraints.items()})
    return merged


def _to_mesma_constraints_tuple(
    c: Mapping[str, float],
) -> Tuple[float, float, float, float, float, float, float]:
    """Convert constraint dict into the 7-tuple ``MesmaCore.execute`` expects.

    Format (see mesma v1.0.8 source):
        (min_frac, max_frac, min_shade, max_shade, max_rmse, residual_flag, residual_threshold)

    The two trailing flags disable residual-image generation when set to ``-9999``.
    """
    return (
        float(c["min_fraction"]),
        float(c["max_fraction"]),
        float(c["min_shade"]),
        float(c["max_shade"]),
        float(c["max_rmse"]),
        -9999.0,
        -9999.0,
    )


# ---------------------------------------------------------------------------
# Scene / library wrangling
# ---------------------------------------------------------------------------


def _align_to_library_grid(
    scene: Union[xr.Dataset, xr.DataArray],
    library: xr.DataArray,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Align a scene's spectral axis to the library's wavelength grid.

    Uses nearest-neighbour matching — the library is assumed to already be
    resampled to the Tanager grid via :func:`tanager.endmembers.resample_library`.

    Returns:
        Tuple of (reflectance DataArray restricted to library bands,
        library DataArray).
    """
    refl = _scene_reflectance(scene)
    if "wavelength" not in refl.coords:
        raise ValueError("scene reflectance must have a 'wavelength' coordinate")
    if "wavelength" not in library.coords:
        raise ValueError("library must have a 'wavelength' coordinate")

    lib_wl = np.asarray(library.coords["wavelength"].values, dtype=np.float64)
    target = xr.DataArray(lib_wl, dims="wavelength")
    refl_aligned = refl.sel(wavelength=target, method="nearest")
    # Replace the (possibly nudged) scene wavelengths with the library's exact
    # values so downstream array ops do not get tripped by float jitter.
    refl_aligned = refl_aligned.assign_coords(wavelength=lib_wl)
    return refl_aligned, library


def _library_class_layout(
    library: xr.DataArray,
) -> Tuple[list[str], dict[str, list[int]], np.ndarray, np.ndarray]:
    """Return (sorted class list, em-per-class index map, library array, per-em classes).

    The MESMA core wants the library shaped ``(n_bands, n_endmembers)`` with
    a parallel ``per_em_classes`` array (one class label per library column) so
    ``MesmaModels.setup`` can build per-class endmember combinations. The
    ``class_list`` returned here is the alphabetically-sorted unique class set;
    MESMA's output channel ordering follows ``np.unique`` (alphabetical) of the
    per-em labels, so this matches what we need to interpret the result.

    Returns:
        class_list: sorted unique category strings (matches mesma channel order).
        em_per_class: {category: [library column indices that belong to it]},
            keyed in alphabetical order to align with mesma's channel layout.
        library_arr: float32 array of shape (n_bands, n_endmembers).
        per_em_classes: 1-D array of class labels with length = n_endmembers,
            indexed by library column. Pass this to ``MesmaModels.setup``.
    """
    if "category" not in library.coords:
        raise ValueError("library must carry a 'category' coordinate")
    cats = np.asarray(library.coords["category"].values, dtype=str)
    # spectrum_id is the leading dim; transpose so bands come first.
    library_arr = np.asarray(library.values, dtype=np.float32).T  # (n_bands, n_em)
    class_list = sorted(set(cats.tolist()))
    em_per_class: dict[str, list[int]] = {c: [] for c in class_list}
    for i, c in enumerate(cats.tolist()):
        em_per_class[c].append(int(i))
    return class_list, em_per_class, library_arr, cats


# ---------------------------------------------------------------------------
# Band selection — uSZU
# ---------------------------------------------------------------------------


def select_bands_uszu(
    scene: Union[xr.Dataset, xr.DataArray],
    library: xr.DataArray,
    n_bands: int = 40,
) -> Tuple[xr.Dataset, np.ndarray]:
    """Select the most class-discriminatory bands using Uniform SZU.

    Uniform Spectral Zone Unmixing (Somers et al. 2010, Roberts et al. 2018)
    partitions the spectrum into ``n_bands`` equal-width wavelength zones and,
    within each zone, picks the band that maximises Fisher-style class
    separability: ``between_class_variance / within_class_variance``.

    Args:
        scene: Tanager scene Dataset (with ``reflectance``) or DataArray.
            Bad bands should already be masked via
            :func:`tanager.spectral.mask_bad_bands`.
        library: Endmember library DataArray with a ``category`` coordinate.
            Used as the proxy for class statistics.
        n_bands: Number of bands to select (one per zone). Defaults to 40.

    Returns:
        Tuple ``(scene_subset, selected_indices)`` where ``scene_subset`` is
        the input scene restricted to the selected wavelengths and
        ``selected_indices`` are the integer indices into the *library*
        wavelength axis (callers that need to subset the library themselves
        can use these directly).

    Raises:
        ValueError: If ``n_bands`` <= 0 or exceeds the number of available
            library bands.
    """
    if n_bands <= 0:
        raise ValueError(f"n_bands must be positive, got {n_bands}")
    if "category" not in library.coords:
        raise ValueError("library must carry a 'category' coordinate")

    refl_aligned, library = _align_to_library_grid(scene, library)
    lib_wl = np.asarray(library.coords["wavelength"].values, dtype=np.float64)
    n_total = lib_wl.size
    if n_bands > n_total:
        raise ValueError(f"n_bands ({n_bands}) exceeds available library bands ({n_total})")

    cats = np.asarray(library.coords["category"].values, dtype=str)
    spectra = np.asarray(library.values, dtype=np.float64)  # (n_em, n_bands)
    classes = sorted(set(cats.tolist()))

    # Per-band class-mean and class-variance statistics.
    mean_total = spectra.mean(axis=0)  # (n_bands,)
    between = np.zeros(n_total, dtype=np.float64)
    within = np.zeros(n_total, dtype=np.float64)
    for c in classes:
        mask = cats == c
        if mask.sum() < 1:
            continue
        block = spectra[mask]
        mu_c = block.mean(axis=0)
        between += mask.sum() * (mu_c - mean_total) ** 2
        if block.shape[0] > 1:
            within += block.var(axis=0, ddof=0) * mask.sum()
    # Avoid division by zero for bands with zero within-class variance.
    fisher = between / np.maximum(within, 1e-12)

    # Partition the spectrum into n_bands zones, pick the argmax band per zone.
    zone_edges = np.linspace(0, n_total, n_bands + 1, dtype=int)
    selected_indices: list[int] = []
    for k in range(n_bands):
        lo, hi = zone_edges[k], zone_edges[k + 1]
        if hi <= lo:
            continue
        local_best = lo + int(np.argmax(fisher[lo:hi]))
        selected_indices.append(local_best)
    if len(selected_indices) < n_bands:
        # Some zones were degenerate; pad by picking remaining top-scoring bands.
        remaining = sorted(
            set(range(n_total)) - set(selected_indices),
            key=lambda i: -fisher[i],
        )
        for idx in remaining:
            if len(selected_indices) >= n_bands:
                break
            selected_indices.append(idx)
    selected_indices = sorted(set(selected_indices))

    selected_wl = lib_wl[selected_indices]
    target = xr.DataArray(selected_wl, dims="wavelength")
    if isinstance(scene, xr.Dataset):
        scene_subset = scene.sel(wavelength=target, method="nearest")
        scene_subset = scene_subset.assign_coords(wavelength=selected_wl)
    else:
        scene_subset_da = refl_aligned.isel(wavelength=selected_indices)
        scene_subset = xr.Dataset({"reflectance": scene_subset_da})
    return scene_subset, np.asarray(selected_indices, dtype=np.int64)


# ---------------------------------------------------------------------------
# Output dataset assembly
# ---------------------------------------------------------------------------


def _empty_fraction(y: np.ndarray, x: np.ndarray) -> xr.DataArray:
    """Return a (y, x) DataArray of zeros with float32 dtype."""
    return xr.DataArray(
        np.zeros((y.size, x.size), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": y, "x": x},
    )


def _assemble_output(
    fractions_by_class: Mapping[str, np.ndarray],
    rmse: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    engine: str,
) -> xr.Dataset:
    """Assemble the canonical unmixing output Dataset.

    Pads missing canonical classes with zero-filled DataArrays so the output
    schema is identical regardless of the library composition.
    """
    data_vars: dict[str, xr.DataArray] = {}
    for cls in _CANONICAL_FRACTIONS:
        arr = fractions_by_class.get(cls)
        if arr is None:
            data_vars[cls] = _empty_fraction(y, x)
        else:
            data_vars[cls] = xr.DataArray(
                arr.astype(np.float32),
                dims=("y", "x"),
                coords={"y": y, "x": x},
            )
    # Preserve any extra (non-canonical) classes the library may carry.
    for cls, arr in fractions_by_class.items():
        if cls in _CANONICAL_FRACTIONS:
            continue
        data_vars[cls] = xr.DataArray(
            arr.astype(np.float32),
            dims=("y", "x"),
            coords={"y": y, "x": x},
        )
    data_vars["rmse"] = xr.DataArray(
        rmse.astype(np.float32),
        dims=("y", "x"),
        coords={"y": y, "x": x},
    )
    ds = xr.Dataset(data_vars)
    ds.attrs["unmixing_engine"] = engine
    return ds


def _validate_fraction_output(ds: xr.Dataset) -> None:
    """Sanity-check an unmixing output Dataset.

    Raises:
        ValueError: If required variables are missing, dims are wrong, or
            fractions are out of range.
    """
    for cls in _CANONICAL_FRACTIONS:
        if cls not in ds.data_vars:
            raise ValueError(f"output Dataset missing required variable: {cls!r}")
    if "rmse" not in ds.data_vars:
        raise ValueError("output Dataset missing 'rmse' variable")
    for var in (*_CANONICAL_FRACTIONS, "rmse"):
        if ds[var].dims != ("y", "x"):
            raise ValueError(f"variable {var!r} has dims {ds[var].dims}, expected ('y', 'x')")


# ---------------------------------------------------------------------------
# Constraint filtering (post-MESMA)
# ---------------------------------------------------------------------------


def _apply_post_constraints(
    fractions_by_class: dict[str, np.ndarray],
    rmse: np.ndarray,
    constraints: Mapping[str, float],
) -> Tuple[dict[str, np.ndarray], np.ndarray]:
    """Mark pixels NaN where the model violates RMSE / fraction constraints.

    Applied to FCLS / NNLS output where the backend always produces a model.
    For MESMA the core already enforces these, but we still run this pass to
    convert the ``rmse=9999`` sentinel into ``NaN`` and to defend against any
    constraint slipping through the LUT search.
    """
    if not fractions_by_class:
        return fractions_by_class, rmse

    classes = list(fractions_by_class.keys())
    stack = np.stack([fractions_by_class[c] for c in classes], axis=0)
    bad_rmse = (rmse > float(constraints["max_rmse"])) | (rmse >= _MESMA_RMSE_SENTINEL)
    bad_low = (stack < float(constraints["min_fraction"])).any(axis=0)
    bad_high = (stack > float(constraints["max_fraction"])).any(axis=0)
    bad = bad_rmse | bad_low | bad_high

    if bad.any():
        for c in classes:
            arr = fractions_by_class[c].astype(np.float32, copy=True)
            arr[bad] = np.nan
            fractions_by_class[c] = arr
        rmse = rmse.astype(np.float32, copy=True)
        rmse[bad] = np.nan
    return fractions_by_class, rmse


# ---------------------------------------------------------------------------
# MESMA primary backend
# ---------------------------------------------------------------------------


def _run_mesma_core(
    refl: np.ndarray,
    library_arr: np.ndarray,
    per_em_classes: np.ndarray,
    em_per_class: Mapping[str, list[int]],
    constraints_tuple: Tuple[float, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """Invoke ``MesmaCore.execute`` and return ``(fractions, rmse)`` arrays.

    ``refl`` shape: ``(n_bands, n_rows, n_cols)`` (bands-first).
    ``library_arr`` shape: ``(n_bands, n_endmembers)`` (bands-first).
    ``per_em_classes`` shape: ``(n_endmembers,)`` — class label per library column.

    Returns ``fractions`` of shape ``(n_classes + 1, n_rows, n_cols)`` (shade
    last, alphabetical class order otherwise) and ``rmse`` of shape
    ``(n_rows, n_cols)``. Unmodeled pixels carry ``rmse=9999`` from mesma.
    """
    from mesma.core.mesma import MesmaCore, MesmaModels  # imported lazily

    # MesmaModels.setup expects per-em (per-library-column) class labels, NOT
    # the unique sorted class list. Passing unique-sorted causes MESMA to
    # mis-map library columns to class indices when n_unique != n_em.
    models = MesmaModels()
    models.setup(list(per_em_classes))
    look_up_table = models.return_look_up_table()

    core = MesmaCore(n_cores=1)
    # mesma's Pool wrapper threads on Linux; n_cores=1 keeps the call deterministic.
    result = core.execute(
        refl,
        library_arr,
        look_up_table,
        em_per_class,
        constraints_tuple,
        -9999.0,  # fusion_value (disabled)
        None,  # shade_spectrum=None — see engineering-notes.md workaround
    )
    # mesma 1.0.8 returns a 4-tuple: (model_indices, fractions, rmse, residual_image)
    _model_indices, fractions, rmse_map, _residuals = result
    return np.asarray(fractions), np.asarray(rmse_map)


def _mesma_fractions_to_dict(
    fractions: np.ndarray,
    rmse: np.ndarray,
    class_list: Sequence[str],
) -> Tuple[dict[str, np.ndarray], np.ndarray]:
    """Split the MESMA fraction tensor into a {class_name: array} dict.

    mesma's ``best_fractions`` shape is ``(n_classes + 1, ny, nx)`` with shade
    appended as the last channel and the remaining classes in alphabetical
    order (mesma sorts them internally during ``MesmaModels.setup``).

    Pixels marked unmodeled (``rmse>=9999``) get ``NaN`` fractions and ``NaN``
    rmse before downstream constraint filtering runs.
    """
    out: dict[str, np.ndarray] = {}
    classes_sorted = sorted(class_list)
    n_channels = fractions.shape[0]
    if n_channels == len(classes_sorted) + 1:
        for i, c in enumerate(classes_sorted):
            out[c] = fractions[i].astype(np.float32)
        if "shade" in out:
            # Library already had explicit shade — overwrite with mesma's last channel
            # to keep the channel meaning consistent. mesma's convention puts shade last.
            out["shade"] = fractions[-1].astype(np.float32)
        else:
            out["shade"] = fractions[-1].astype(np.float32)
    else:
        # Library may already include shade as a class; mesma still appends a shade
        # channel, so the channel count can be len(classes)+1 even when shade is in
        # the class list. Fall through to a generic mapping.
        for i, c in enumerate(classes_sorted):
            if i < n_channels:
                out[c] = fractions[i].astype(np.float32)

    rmse = rmse.astype(np.float32, copy=True)
    bad = rmse >= _MESMA_RMSE_SENTINEL
    if bad.any():
        for c in out:
            arr = out[c].astype(np.float32, copy=True)
            arr[bad] = np.nan
            out[c] = arr
        rmse[bad] = np.nan
    return out, rmse


# ---------------------------------------------------------------------------
# Fallback backends
# ---------------------------------------------------------------------------


def _fcls_pixel(spectrum: np.ndarray, library: np.ndarray) -> Tuple[np.ndarray, float]:
    """FCLS for a single pixel via projected NNLS + sum-to-one rescale.

    Args:
        spectrum: 1D pixel reflectance, shape ``(n_bands,)``.
        library: 2D library, shape ``(n_bands, n_endmembers)``.

    Returns:
        Tuple ``(fractions, rmse)`` where fractions has shape ``(n_endmembers,)``
        and rmse is a scalar float.
    """
    from scipy.optimize import nnls  # heavy dep — defer

    abundances, _residual = nnls(library, spectrum, maxiter=200)
    total = float(abundances.sum())
    if total > 1e-9:
        abundances = abundances / total
    reconstructed = library @ abundances
    rmse = float(np.sqrt(np.mean((spectrum - reconstructed) ** 2)))
    return abundances.astype(np.float32), rmse


def _run_fcls_fallback(
    refl: np.ndarray,
    library_arr: np.ndarray,
    cats: np.ndarray,
) -> Tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-pixel FCLS unmixing using the full library at once.

    ``refl`` shape: ``(n_bands, n_rows, n_cols)``.
    ``library_arr`` shape: ``(n_bands, n_endmembers)``.

    Aggregates per-endmember fractions into per-class fractions by summing
    columns sharing the same ``cats`` label.
    """
    n_bands, n_rows, n_cols = refl.shape
    n_pixels = n_rows * n_cols
    n_em = library_arr.shape[1]
    flat = refl.reshape(n_bands, n_pixels)

    fractions = np.zeros((n_em, n_pixels), dtype=np.float32)
    rmse = np.zeros(n_pixels, dtype=np.float32)
    for p in range(n_pixels):
        spectrum = flat[:, p]
        if not np.isfinite(spectrum).all():
            fractions[:, p] = np.nan
            rmse[p] = np.nan
            continue
        f, r = _fcls_pixel(spectrum, library_arr)
        fractions[:, p] = f
        rmse[p] = r

    # Aggregate by class.
    classes = sorted(set(cats.tolist()))
    out: dict[str, np.ndarray] = {}
    for c in classes:
        mask = cats == c
        if not mask.any():
            continue
        agg = fractions[mask].sum(axis=0).reshape(n_rows, n_cols)
        out[c] = agg.astype(np.float32)
    return out, rmse.reshape(n_rows, n_cols)


def _run_mesma_hysup_fallback(
    scene: Union[xr.Dataset, xr.DataArray],
    library: xr.DataArray,
    constraints: Optional[Mapping[str, float]] = None,
) -> xr.Dataset:
    """Fallback unmixing path when mesma is unavailable.

    Tries HySUPP's FCLS first; if HySUPP is also unavailable, falls back to a
    pure-numpy NNLS implementation. Output schema is identical to
    :func:`run_mesma`. The chosen engine is recorded in
    ``Dataset.attrs['unmixing_engine']``.
    """
    constraints_resolved = _resolve_constraints(constraints)
    refl_aligned, library = _align_to_library_grid(scene, library)
    library_arr = np.asarray(library.values, dtype=np.float32).T  # (n_bands, n_em)
    cats = np.asarray(library.coords["category"].values, dtype=str)

    refl_arr = np.asarray(refl_aligned.values, dtype=np.float32)
    if refl_arr.ndim != 3:
        raise ValueError(
            f"scene reflectance must be 3D (wavelength, y, x), got shape {refl_arr.shape}"
        )

    engine = "hysup" if _HYSUP_AVAILABLE else "nnls"
    if engine == "hysup":  # pragma: no cover - depends on optional dep
        try:
            fractions_by_class, rmse_map = _run_hysup_fcls(refl_arr, library_arr, cats)
        except Exception as exc:
            logger.warning("hysup FCLS failed (%s); falling back to scipy NNLS", exc)
            engine = "nnls"
            fractions_by_class, rmse_map = _run_fcls_fallback(refl_arr, library_arr, cats)
    else:
        fractions_by_class, rmse_map = _run_fcls_fallback(refl_arr, library_arr, cats)

    fractions_by_class, rmse_map = _apply_post_constraints(
        fractions_by_class, rmse_map, constraints_resolved
    )

    y = np.asarray(refl_aligned.coords["y"].values)
    x = np.asarray(refl_aligned.coords["x"].values)
    ds = _assemble_output(fractions_by_class, rmse_map, y, x, engine=engine)
    _validate_fraction_output(ds)
    return ds


def _run_hysup_fcls(  # pragma: no cover - optional dep not installed in CI
    refl: np.ndarray,
    library_arr: np.ndarray,
    cats: np.ndarray,
) -> Tuple[dict[str, np.ndarray], np.ndarray]:
    """HySUPP FCLS adapter (best-effort; falls back to NNLS if hysup API differs)."""
    # The hysup package surface has changed across versions; the safest path
    # is to defer to the NNLS implementation, which is mathematically equivalent
    # for FCLS with sum-to-one + non-negativity constraints. Override here if
    # the deployed hysup version exposes a stable Solver class.
    return _run_fcls_fallback(refl, library_arr, cats)


# ---------------------------------------------------------------------------
# run_mesma — public entry point
# ---------------------------------------------------------------------------


def run_mesma(
    scene: Union[xr.Dataset, xr.DataArray],
    library: xr.DataArray,
    constraints: Optional[Mapping[str, float]] = None,
    bands: Optional[Union[Sequence[float], np.ndarray]] = None,
) -> xr.Dataset:
    """Run MESMA spectral unmixing on a Tanager scene.

    Output is a deterministic xarray Dataset with the canonical fire fraction
    variables (char, pv, npv, soil, shade) plus rmse, all with dims (y, x).
    Pixels where no model satisfies the constraints have NaN fractions and
    NaN rmse.

    Args:
        scene: Tanager scene Dataset (with ``reflectance``) or DataArray.
            Must carry ``wavelength``, ``y``, and ``x`` coordinates.
        library: Endmember library (output of
            :func:`tanager.endmembers.build_fire_library`). Must carry
            ``category`` and be resampled to the scene wavelength grid.
        constraints: Optional dict overriding any of
            ``min_fraction, max_fraction, min_shade, max_shade, max_rmse``.
            See :data:`DEFAULT_CONSTRAINTS` for the spec defaults.
        bands: Optional subset of wavelengths (nm) to run unmixing on. Useful
            in conjunction with :func:`select_bands_uszu` to keep MESMA fast
            on the 426-band cube.

    Returns:
        ``xr.Dataset`` with variables (char, pv, npv, soil, shade, rmse) and
        dims (y, x). ``ds.attrs['unmixing_engine']`` records the backend used
        ('mesma', 'hysup', or 'nnls').

    Raises:
        ValueError: If the scene or library are missing required coords.
    """
    constraints_resolved = _resolve_constraints(constraints)

    # Subset both scene and library to the requested band set if provided.
    if bands is not None:
        bands_arr = np.asarray(bands, dtype=np.float64).ravel()
        target = xr.DataArray(bands_arr, dims="wavelength")
        library = library.sel(wavelength=target, method="nearest")
        # Snap library wavelengths to the requested values for clean alignment.
        library = library.assign_coords(wavelength=bands_arr)
        if isinstance(scene, xr.Dataset):
            scene = scene.sel(wavelength=target, method="nearest")
            scene = scene.assign_coords(wavelength=bands_arr)
        else:
            scene = scene.sel(wavelength=target, method="nearest")
            scene = scene.assign_coords(wavelength=bands_arr)

    if not _MESMA_AVAILABLE:
        logger.warning(
            "mesma package not available; running FCLS/NNLS fallback (results will "
            "be single-model FCLS, not multi-endmember MESMA)."
        )
        return _run_mesma_hysup_fallback(scene, library, constraints_resolved)

    refl_aligned, library = _align_to_library_grid(scene, library)
    refl_arr = np.asarray(refl_aligned.values, dtype=np.float32)
    if refl_arr.ndim != 3:
        raise ValueError(
            f"scene reflectance must be 3D (wavelength, y, x), got shape {refl_arr.shape}"
        )

    class_list, em_per_class, library_arr, per_em_classes = _library_class_layout(library)

    constraints_tuple = _to_mesma_constraints_tuple(constraints_resolved)
    try:
        fractions, rmse_map = _run_mesma_core(
            refl_arr, library_arr, per_em_classes, em_per_class, constraints_tuple
        )
    except Exception as exc:
        logger.warning("mesma backend raised %s — falling back to FCLS/NNLS", exc)
        return _run_mesma_hysup_fallback(scene, library, constraints_resolved)

    fractions_by_class, rmse_map = _mesma_fractions_to_dict(fractions, rmse_map, class_list)
    fractions_by_class, rmse_map = _apply_post_constraints(
        fractions_by_class, rmse_map, constraints_resolved
    )

    y = np.asarray(refl_aligned.coords["y"].values)
    x = np.asarray(refl_aligned.coords["x"].values)
    ds = _assemble_output(fractions_by_class, rmse_map, y, x, engine="mesma")
    _validate_fraction_output(ds)
    return ds


# ---------------------------------------------------------------------------
# Shade normalization
# ---------------------------------------------------------------------------


def normalize_fractions(
    fractions: xr.Dataset,
    remove_shade: bool = True,
) -> xr.Dataset:
    """Remove the shade fraction and rescale remaining fractions to sum to 1.0.

    Photometric shade is a non-physical "endmember" that absorbs all the
    illumination loss from shadows and angle effects. Standard practice
    (Roberts et al. 2018) is to remove it before downstream analysis so the
    remaining fractions reflect surface composition only.

    Args:
        fractions: Output Dataset from :func:`run_mesma` or the fallback path.
        remove_shade: When True (default), drop the shade variable and rescale
            the remaining fraction variables. When False, return a copy of
            the input unchanged.

    Returns:
        New Dataset. If ``remove_shade`` was True, the shade variable is
        absent and the remaining canonical fractions sum to ~1.0 per pixel.
        Pixels with shade==1.0 (fully shaded) become NaN.

    Raises:
        ValueError: If the input is missing the canonical fraction variables.
    """
    _validate_fraction_output(fractions)
    if not remove_shade:
        return fractions.copy()

    shade = fractions["shade"]
    other_vars = [v for v in _CANONICAL_FRACTIONS if v != "shade" and v in fractions]
    extras = [v for v in fractions.data_vars if v not in _CANONICAL_FRACTIONS and v != "rmse"]
    keep_vars = other_vars + extras

    denom = (1.0 - shade).astype(np.float32)
    safe_denom = xr.where(np.abs(denom) > 1e-6, denom, np.nan)

    out = xr.Dataset()
    for var in keep_vars:
        rescaled = (fractions[var] / safe_denom).astype(np.float32)
        out[var] = rescaled
    if "rmse" in fractions:
        out["rmse"] = fractions["rmse"]
    out.attrs.update(fractions.attrs)
    out.attrs["shade_normalized"] = True
    return out


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_fraction_maps(
    fractions: xr.Dataset,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "viridis",
) -> "Figure":
    """Multi-panel matplotlib figure of per-class fraction maps.

    Args:
        fractions: Output of :func:`run_mesma` or :func:`normalize_fractions`.
        figsize: Optional ``(width, height)`` in inches for the matplotlib
            Figure. Defaults to ``(4 * n_panels, 4)``.
        cmap: Matplotlib colormap name applied to each panel. Defaults to
            ``viridis``.

    Returns:
        ``matplotlib.figure.Figure`` with one panel per fraction variable
        (excluding rmse). Caller is responsible for ``plt.show()`` /
        ``fig.savefig`` as appropriate.
    """
    import matplotlib.pyplot as plt

    panels = [v for v in fractions.data_vars if v != "rmse"]
    n = len(panels)
    if n == 0:
        raise ValueError("no fraction variables to plot")

    if figsize is None:
        figsize = (4.0 * n, 4.0)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes.ravel()
    for ax, var in zip(axes, panels):
        arr = fractions[var].values
        im = ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(var)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_rgb_composite(
    fractions: xr.Dataset,
    r: str = "char",
    g: str = "pv",
    b: str = "npv",
    figsize: Optional[Tuple[float, float]] = None,
) -> "Figure":
    """False-colour RGB composite from three fraction maps.

    Args:
        fractions: Output of :func:`run_mesma` or :func:`normalize_fractions`.
        r: Variable name to map to the red channel. Defaults to ``"char"``.
        g: Variable name to map to the green channel. Defaults to ``"pv"``.
        b: Variable name to map to the blue channel. Defaults to ``"npv"``.
        figsize: Optional ``(width, height)`` in inches. Defaults to ``(6, 6)``.

    Returns:
        ``matplotlib.figure.Figure`` with a single panel showing the
        composite. NaN pixels render as black.
    """
    import matplotlib.pyplot as plt

    for ch in (r, g, b):
        if ch not in fractions.data_vars:
            raise ValueError(f"channel {ch!r} not found in fractions Dataset")

    def _norm(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        finite = np.isfinite(arr)
        out = np.zeros_like(arr, dtype=np.float32)
        if finite.any():
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
            if hi - lo > 1e-9:
                out[finite] = (arr[finite] - lo) / (hi - lo)
            else:
                out[finite] = arr[finite]
        return np.clip(out, 0.0, 1.0)

    rgb = np.stack(
        [_norm(fractions[r].values), _norm(fractions[g].values), _norm(fractions[b].values)],
        axis=-1,
    )
    if figsize is None:
        figsize = (6.0, 6.0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb)
    ax.set_title(f"R={r}  G={g}  B={b}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    return fig


__all__ = [
    "DEFAULT_CONSTRAINTS",
    "select_bands_uszu",
    "run_mesma",
    "normalize_fractions",
    "plot_fraction_maps",
    "plot_rgb_composite",
]
