
from src.utils.formatters import (
    format_ci,
    format_effect,
    format_effect_pct,
    format_p_value,
)


class TestFormatEffect:
    def test_positive(self):
        assert format_effect(1.23) == "+1.23"

    def test_negative(self):
        assert format_effect(-0.45) == "-0.45"

    def test_zero(self):
        assert format_effect(0.0) == "+0.00"

    def test_large_number(self):
        assert format_effect(1234567.89) == "+1234567.89"

    def test_small_number(self):
        assert format_effect(0.001) == "+0.00"


class TestFormatEffectPct:
    def test_positive(self):
        assert format_effect_pct(12.3) == "+12.3%"

    def test_negative(self):
        assert format_effect_pct(-5.7) == "-5.7%"

    def test_zero(self):
        assert format_effect_pct(0.0) == "+0.0%"

    def test_large_percentage(self):
        assert format_effect_pct(100.0) == "+100.0%"


class TestFormatPValue:
    def test_default_decimals(self):
        assert format_p_value(0.05) == "0.0500"

    def test_custom_decimals(self):
        assert format_p_value(0.05, decimals=2) == "0.05"

    def test_very_small(self):
        assert format_p_value(0.0001) == "0.0001"

    def test_one(self):
        assert format_p_value(1.0) == "1.0000"

    def test_zero(self):
        assert format_p_value(0.0) == "0.0000"


class TestFormatCi:
    def test_basic(self):
        assert format_ci(1.23, 4.56) == "[1.23, 4.56]"

    def test_negative_values(self):
        assert format_ci(-3.0, -1.0) == "[-3.00, -1.00]"

    def test_zero_crossing(self):
        assert format_ci(-1.0, 1.0) == "[-1.00, 1.00]"

    def test_same_values(self):
        assert format_ci(5.0, 5.0) == "[5.00, 5.00]"
