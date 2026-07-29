"""Tests for PRODUCT_STYLES dictionary and ProductStyle dataclass.

Verifies all 11 product type entries have correct colormap presets per spec.
"""

import dataclasses

import pytest

from tanager.visualization import PRODUCT_STYLES, ProductStyle

# ---------------------------------------------------------------------------
# ProductStyle dataclass
# ---------------------------------------------------------------------------


class TestProductStyleDataclass:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(ProductStyle)

    def test_has_required_fields(self):
        field_names = {f.name for f in dataclasses.fields(ProductStyle)}
        assert field_names == {"cmap", "vmin", "vmax", "label", "class_ticks"}

    def test_instantiation_with_all_fields(self):
        style = ProductStyle(cmap="viridis", vmin=0.0, vmax=1.0, label="Test", class_ticks=[0.0, 0.5, 1.0])
        assert style.cmap == "viridis"
        assert style.vmin == 0.0
        assert style.vmax == 1.0
        assert style.label == "Test"
        assert style.class_ticks == [0.0, 0.5, 1.0]

    def test_class_ticks_accepts_none(self):
        style = ProductStyle(cmap="viridis", vmin=0.0, vmax=1.0, label="Test", class_ticks=None)
        assert style.class_ticks is None


# ---------------------------------------------------------------------------
# PRODUCT_STYLES dictionary — presence and type
# ---------------------------------------------------------------------------


EXPECTED_PRODUCTS = {
    "nbr", "ndvi", "ndwi", "dnbr", "cbi", "severity", "char", "pv", "npv", "soil", "lfmc",
    "sai970", "sai1200", "sai1660",
    "ndwi_1240", "ndwi_1640", "ndwi_2130",
    "wi",
    "cr_depth_970", "cr_depth_1200", "cr_depth_1700", "cr_depth_2100",
    "delta_nbr", "delta_ndvi",
}


class TestProductStylesKeys:
    def test_contains_all_11_products(self):
        assert set(PRODUCT_STYLES.keys()) == EXPECTED_PRODUCTS

    def test_has_exactly_24_entries(self):
        assert len(PRODUCT_STYLES) == 24

    @pytest.mark.parametrize("product", sorted(EXPECTED_PRODUCTS))
    def test_each_value_is_product_style_instance(self, product):
        assert isinstance(PRODUCT_STYLES[product], ProductStyle)

    @pytest.mark.parametrize("product", sorted(EXPECTED_PRODUCTS))
    def test_each_entry_has_non_empty_label(self, product):
        assert PRODUCT_STYLES[product].label


# ---------------------------------------------------------------------------
# PRODUCT_STYLES dictionary — spec table values
# ---------------------------------------------------------------------------


class TestProductStylesSpecValues:
    """Verify cmap/vmin/vmax match the specification table."""

    def test_nbr_cmap(self):
        assert PRODUCT_STYLES["nbr"].cmap == "RdYlGn"

    def test_nbr_vmin(self):
        assert PRODUCT_STYLES["nbr"].vmin == -1.0

    def test_nbr_vmax(self):
        assert PRODUCT_STYLES["nbr"].vmax == 1.0

    def test_dnbr_cmap(self):
        assert PRODUCT_STYLES["dnbr"].cmap == "RdYlGn_r"

    def test_dnbr_vmin(self):
        assert PRODUCT_STYLES["dnbr"].vmin == -0.5

    def test_dnbr_vmax(self):
        assert PRODUCT_STYLES["dnbr"].vmax == 1.3

    def test_cbi_cmap(self):
        assert PRODUCT_STYLES["cbi"].cmap == "YlOrRd"

    def test_cbi_vmin(self):
        assert PRODUCT_STYLES["cbi"].vmin == 0.0

    def test_cbi_vmax(self):
        assert PRODUCT_STYLES["cbi"].vmax == 3.0

    def test_cbi_class_ticks(self):
        assert PRODUCT_STYLES["cbi"].class_ticks == [0.0, 1.0, 2.0, 3.0]

    def test_lfmc_cmap(self):
        assert PRODUCT_STYLES["lfmc"].cmap == "RdYlGn"

    def test_lfmc_vmin(self):
        assert PRODUCT_STYLES["lfmc"].vmin == 0.0

    def test_lfmc_vmax(self):
        assert PRODUCT_STYLES["lfmc"].vmax == 200.0

    def test_ndvi_cmap(self):
        assert PRODUCT_STYLES["ndvi"].cmap == "RdYlGn"

    def test_ndwi_vmin(self):
        assert PRODUCT_STYLES["ndwi"].vmin == -1.0

    def test_ndwi_vmax(self):
        assert PRODUCT_STYLES["ndwi"].vmax == 1.0

    def test_char_vmin_zero(self):
        assert PRODUCT_STYLES["char"].vmin == 0.0

    def test_char_vmax_one(self):
        assert PRODUCT_STYLES["char"].vmax == 1.0

    def test_pv_vmin_zero(self):
        assert PRODUCT_STYLES["pv"].vmin == 0.0

    def test_pv_vmax_one(self):
        assert PRODUCT_STYLES["pv"].vmax == 1.0

    def test_npv_vmin_zero(self):
        assert PRODUCT_STYLES["npv"].vmin == 0.0

    def test_npv_vmax_one(self):
        assert PRODUCT_STYLES["npv"].vmax == 1.0

    def test_soil_vmin_zero(self):
        assert PRODUCT_STYLES["soil"].vmin == 0.0

    def test_soil_vmax_one(self):
        assert PRODUCT_STYLES["soil"].vmax == 1.0

    def test_severity_vmin_zero(self):
        assert PRODUCT_STYLES["severity"].vmin == 0

    def test_severity_vmax_four(self):
        assert PRODUCT_STYLES["severity"].vmax == 4

    def test_severity_class_ticks_match_barc_classes(self):
        assert PRODUCT_STYLES["severity"].class_ticks == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# PRODUCT_STYLES dictionary — range sanity
# ---------------------------------------------------------------------------


class TestProductStylesRangeSanity:
    @pytest.mark.parametrize("product", sorted(EXPECTED_PRODUCTS))
    def test_vmin_less_than_vmax(self, product):
        style = PRODUCT_STYLES[product]
        assert style.vmin < style.vmax, f"{product}: vmin ({style.vmin}) >= vmax ({style.vmax})"
