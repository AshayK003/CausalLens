"""Unit tests for pure helper functions in app.py."""

from app import (
    _format_arima_order,
    _format_optional_float,
    _format_residuals_status,
    _generate_narrative,
)


class TestFormatOptionalFloat:
    def test_none_returns_na(self):
        assert _format_optional_float(None) == "N/A"

    def test_formats_float(self):
        assert _format_optional_float(3.14159) == "3.1"

    def test_custom_format(self):
        assert _format_optional_float(0.123456, ".4f") == "0.1235"


class TestFormatArimaOrder:
    def test_none_returns_na(self):
        assert _format_arima_order(None) == "N/A"

    def test_formats_tuple(self):
        assert _format_arima_order((1, 1, 1)) == "(1, 1, 1)"


class TestFormatResidualsStatus:
    def test_none_returns_na(self):
        assert _format_residuals_status(None) == "N/A"

    def test_ok_returns_white_noise(self):
        assert _format_residuals_status(True) == "White noise"

    def test_not_ok_returns_check_model(self):
        assert _format_residuals_status(False) == "Check model"


class TestGenerateNarrative:
    def test_significant_increase(self):
        result = {
            "effect": 5.0,
            "effect_pct": 10.0,
            "p_value": 0.01,
            "significant": True,
            "direction": "increase",
            "n_pre": 100,
            "n_post": 50,
            "ci_lower": 2.0,
            "ci_upper": 8.0,
            "dates": ["2020-01-01"] * 150,
            "intervention_idx": 100,
        }
        narrative = _generate_narrative(result, "revenue")
        assert "increased by 10.0%" in narrative
        assert "p-value: 0.0100" in narrative
        assert "significant" in narrative

    def test_not_significant(self):
        result = {
            "effect": 0.5,
            "effect_pct": 1.0,
            "p_value": 0.3,
            "significant": False,
            "direction": "increase",
            "n_pre": 100,
            "n_post": 50,
            "ci_lower": -1.0,
            "ci_upper": 2.0,
            "dates": ["2020-01-01"] * 150,
            "intervention_idx": 100,
        }
        narrative = _generate_narrative(result, "revenue")
        assert "not statistically significant" in narrative
        assert "p=0.3000" in narrative

    def test_residuals_warning(self):
        result = {
            "effect": 5.0,
            "effect_pct": 10.0,
            "p_value": 0.01,
            "significant": True,
            "direction": "increase",
            "n_pre": 100,
            "n_post": 50,
            "ci_lower": 2.0,
            "ci_upper": 8.0,
            "dates": ["2020-01-01"] * 150,
            "intervention_idx": 100,
            "residuals_ok": False,
        }
        narrative = _generate_narrative(result, "revenue")
        assert "Residual diagnostics" in narrative
