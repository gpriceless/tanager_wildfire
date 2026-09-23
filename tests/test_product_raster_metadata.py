"""Unit tests for the product-raster save path and its metadata backfill.

Covers ``tanager.io.product_metadata`` / ``write_product_raster`` /
``stamp_raster_metadata`` and ``scripts/tag_product_rasters.tag_file``.

These rasters are written to a tmp_path with synthetic pixel values. The
synthetic values exercise the write/read code path only — no number produced
here is a measurement of anything, and none of these tests asserts a physical
result. What is asserted is metadata: that a saved product declares its nodata
value, its units, and a long_name naming the product rather than the array it
was derived from.

Test naming: <function>_<scenario>_<expected_outcome>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("rioxarray")

from tanager.io import (  # noqa: E402
    PRODUCT_METADATA,
    product_metadata,
    stamp_raster_metadata,
    write_product_raster,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from tag_product_rasters import tag_file  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_product(n_rows: int = 4, n_cols: int = 5, n_nan: int = 3) -> xr.DataArray:
    """A 2-D product array on a UTM grid with a few NaN cells.

    Named ``surface_reflectance`` on purpose: that is the name the real
    pipeline arrays carry, and it is the wrong long_name for a derived index.
    """
    values = np.arange(n_rows * n_cols, dtype=np.float32).reshape(n_rows, n_cols)
    values = values / values.max()
    flat = values.ravel()
    flat[:n_nan] = np.nan
    return xr.DataArray(
        values,
        dims=("y", "x"),
        coords={
            "y": np.linspace(3_800_000, 3_800_000 - 30 * (n_rows - 1), n_rows),
            "x": np.linspace(360_000, 360_000 + 30 * (n_cols - 1), n_cols),
        },
        name="surface_reflectance",
    )


# ---------------------------------------------------------------------------
# product_metadata
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "stem,expected_units,expected_long_name",
    [
        ("20241215_ndvi", "dimensionless", "Normalized Difference Vegetation Index"),
        ("20241215_frac_char", "fraction", "MESMA char fraction after shade normalization"),
        ("20241215_mesma_rmse", "reflectance", "MESMA best-model RMSE"),
        ("20250123_CR_depths_1200nm", "dimensionless", "Continuum-removal absorption depth at 1200 nm"),
        ("20250407_SAI970", "dimensionless", "Spectral Absorption Index at 970 nm"),
        ("20250123_NDWI_1640", "dimensionless", "Normalized Difference Water Index, R860 vs R1640"),
    ],
)
def test_product_metadata_known_stems_returns_units_and_long_name(
    stem, expected_units, expected_long_name
):
    meta = product_metadata(stem)

    assert meta is not None
    assert meta["units"] == expected_units
    assert meta["long_name"] == expected_long_name


def test_product_metadata_multidate_dnbr_stem_resolves_to_dnbr():
    """The dNBR filename carries two scene ids and a ``to`` token before the key."""
    meta = product_metadata("20241215_to_20250123swath2_dnbr")

    assert meta is not None
    assert meta["units"] == "dimensionless"
    assert "Differenced Normalized Burn Ratio" in meta["long_name"]


def test_product_metadata_prefers_longest_token_run():
    """``frac_char`` must win over a bare ``char`` suffix match."""
    assert product_metadata("20241215_frac_char")["units"] == "fraction"


def test_product_metadata_accepts_full_filename():
    assert product_metadata("outputs/20241215_ndvi.tif") == product_metadata("ndvi")


def test_product_metadata_unregistered_product_returns_none():
    assert product_metadata("20241215_not_a_real_product") is None


def test_product_metadata_returns_a_copy_so_callers_cannot_mutate_registry():
    meta = product_metadata("ndvi")
    meta["units"] = "tampered"

    assert PRODUCT_METADATA["ndvi"]["units"] == "dimensionless"


# ---------------------------------------------------------------------------
# write_product_raster
# ---------------------------------------------------------------------------

def test_write_product_raster_declares_nan_nodata(tmp_path):
    path = tmp_path / "20241215_ndvi.tif"

    write_product_raster(_make_product(), path, crs="EPSG:32611")

    with rasterio.open(path) as src:
        assert src.nodata is not None
        assert np.isnan(src.nodata)


def test_write_product_raster_writes_units_and_long_name_tags(tmp_path):
    path = tmp_path / "20241215_ndvi.tif"

    write_product_raster(_make_product(), path, crs="EPSG:32611")

    with rasterio.open(path) as src:
        assert src.tags()["units"] == "dimensionless"
        assert src.tags()["long_name"] == "Normalized Difference Vegetation Index"


def test_write_product_raster_band_description_names_product_not_source_array(tmp_path):
    """Regression: rioxarray derives the band description from the array name,
    which left NDVI rasters labelled ``surface_reflectance``."""
    path = tmp_path / "20241215_ndvi.tif"

    write_product_raster(_make_product(), path, crs="EPSG:32611")

    with rasterio.open(path) as src:
        assert src.descriptions[0] == "Normalized Difference Vegetation Index"
        assert src.descriptions[0] != "surface_reflectance"


def test_write_product_raster_preserves_pixel_values(tmp_path):
    da = _make_product()
    path = tmp_path / "20241215_ndvi.tif"

    write_product_raster(da, path, crs="EPSG:32611")

    with rasterio.open(path) as src:
        written = src.read(1)
    assert np.array_equal(written, da.values.astype(np.float32), equal_nan=True)


def test_write_product_raster_caller_attrs_override_registry(tmp_path):
    da = _make_product().assign_attrs(long_name="MESMA char fraction, variant B")
    path = tmp_path / "20241215_frac_char.tif"

    write_product_raster(da, path, crs="EPSG:32611")

    with rasterio.open(path) as src:
        assert src.tags()["long_name"] == "MESMA char fraction, variant B"
        assert src.tags()["units"] == "fraction"


def test_write_product_raster_unregistered_product_raises(tmp_path):
    """Tags are never guessed — an unknown product is a registry gap, not a default."""
    path = tmp_path / "20241215_mystery_index.tif"

    with pytest.raises(KeyError, match="PRODUCT_METADATA"):
        write_product_raster(_make_product(), path, crs="EPSG:32611")


def test_write_product_raster_explicit_product_overrides_path_stem(tmp_path):
    path = tmp_path / "scratch_copy.tif"

    write_product_raster(_make_product(), path, crs="EPSG:32611", product="nbr")

    with rasterio.open(path) as src:
        assert src.tags()["long_name"] == "Normalized Burn Ratio"


# ---------------------------------------------------------------------------
# stamp_raster_metadata
# ---------------------------------------------------------------------------

def _write_untagged(path: Path, da: xr.DataArray) -> Path:
    """Write a raster the way the pipeline did before this save path existed."""
    import rioxarray  # noqa: F401

    rio_da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    rio_da = rio_da.rio.write_crs("EPSG:32611", inplace=False)
    rio_da.rio.to_raster(str(path), compress="DEFLATE", dtype="float32")
    return path


def test_stamp_raster_metadata_declares_nodata_on_untagged_float_raster(tmp_path):
    path = _write_untagged(tmp_path / "20241215_ndvi.tif", _make_product())
    with rasterio.open(path) as src:
        assert src.nodata is None  # positive control: the defect is present

    stamp_raster_metadata(path)

    with rasterio.open(path) as src:
        assert np.isnan(src.nodata)
        assert src.tags()["units"] == "dimensionless"


def test_stamp_raster_metadata_leaves_integer_raster_nodata_alone(tmp_path):
    """NaN is not representable in an int band; guessing a sentinel would
    silently turn a class code into nodata."""
    codes = xr.DataArray(
        np.array([[0, 1], [2, 3]], dtype=np.int16),
        dims=("y", "x"),
        coords={"y": [3_800_000.0, 3_799_970.0], "x": [360_000.0, 360_030.0]},
        name="barc_severity",
    )
    path = tmp_path / "20250123_barc_severity.tif"
    import rioxarray  # noqa: F401

    rio_da = codes.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    rio_da = rio_da.rio.write_crs("EPSG:32611", inplace=False)
    rio_da.rio.to_raster(str(path), compress="DEFLATE", dtype="int16")

    stamp_raster_metadata(path)

    with rasterio.open(path) as src:
        assert src.nodata is None
        assert src.tags()["units"] == "class"


def test_stamp_raster_metadata_does_not_change_pixels(tmp_path):
    da = _make_product()
    path = _write_untagged(tmp_path / "20241215_ndvi.tif", da)
    with rasterio.open(path) as src:
        before = src.read()

    stamp_raster_metadata(path)

    with rasterio.open(path) as src:
        after = src.read()
    assert np.array_equal(before, after, equal_nan=True)


def test_stamp_raster_metadata_unregistered_product_raises(tmp_path):
    path = _write_untagged(tmp_path / "20241215_mystery.tif", _make_product())

    with pytest.raises(KeyError, match="PRODUCT_METADATA"):
        stamp_raster_metadata(path)


# ---------------------------------------------------------------------------
# tag_product_rasters.tag_file
# ---------------------------------------------------------------------------

def test_tag_file_reports_the_metadata_it_wrote(tmp_path):
    path = _write_untagged(tmp_path / "20241215_ndvi.tif", _make_product())

    status = tag_file(path)

    assert status.startswith("OK")
    assert "dimensionless" in status
    with rasterio.open(path) as src:
        assert np.isnan(src.nodata)


def test_tag_file_dry_run_leaves_file_untouched(tmp_path):
    path = _write_untagged(tmp_path / "20241215_ndvi.tif", _make_product())

    status = tag_file(path, dry_run=True)

    assert status.startswith("DRY-RUN")
    with rasterio.open(path) as src:
        assert src.nodata is None
        assert "units" not in src.tags()


def test_tag_file_raises_when_a_pixel_changes(tmp_path, monkeypatch):
    """Positive control for the backfill's own safety check.

    The pixels-unchanged assertion is the only thing standing between a
    metadata backfill and a silent data edit, so prove it can fail: patch the
    stamper to also write a pixel and require tag_file to notice.
    """
    path = _write_untagged(tmp_path / "20241215_ndvi.tif", _make_product())
    real_stamp = stamp_raster_metadata

    def _stamp_and_corrupt(target, meta=None, declare_nodata=True):
        result = real_stamp(target, meta=meta, declare_nodata=declare_nodata)
        with rasterio.open(target, "r+") as dst:
            band = dst.read(1)
            band[-1, -1] = -999.0
            dst.write(band, 1)
        return result

    monkeypatch.setattr("tag_product_rasters.stamp_raster_metadata", _stamp_and_corrupt)

    with pytest.raises(RuntimeError, match="pixel values changed"):
        tag_file(path)


def test_tag_file_raises_when_the_nan_mask_changes(tmp_path, monkeypatch):
    """Second positive control: filling NaN cells must be caught, not accepted."""
    path = _write_untagged(tmp_path / "20241215_ndvi.tif", _make_product())
    real_stamp = stamp_raster_metadata

    def _stamp_and_fill(target, meta=None, declare_nodata=True):
        result = real_stamp(target, meta=meta, declare_nodata=declare_nodata)
        with rasterio.open(target, "r+") as dst:
            band = dst.read(1)
            band[np.isnan(band)] = 0.0
            dst.write(band, 1)
        return result

    monkeypatch.setattr("tag_product_rasters.stamp_raster_metadata", _stamp_and_fill)

    with pytest.raises(RuntimeError, match="NaN mask changed"):
        tag_file(path)


def test_tag_file_unregistered_product_raises(tmp_path):
    path = _write_untagged(tmp_path / "20241215_mystery.tif", _make_product())

    with pytest.raises(KeyError, match="PRODUCT_METADATA"):
        tag_file(path)
