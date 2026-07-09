"""Unit tests for subgroup analysis module."""

import numpy as np
import pandas as pd
import pytest

from src.core.engine import Method
from src.core.subgroup import MIN_SEGMENT_SIZE, _create_segments, run_subgroup_analysis


@pytest.fixture
def panel_df():
    """Single daily series long enough for ARIMA ITS per quarter.

    Unique dates only (duplicate dates collapse in the engine and can leave
    too few pre/post points inside a quarter segment).
    """
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    np.random.seed(42)
    # Mild level shift after mid-year so effects are estimable
    base = np.random.normal(100, 5, 365)
    base[180:] += 8.0
    return pd.DataFrame({
        "date": dates,
        "value": base,
    })


class TestCreateSegments:
    def test_quarter(self, panel_df):
        result = _create_segments(panel_df, "date", "value", "quarter")
        assert "_segment" in result.columns
        assert result["_segment"].nunique() > 0

    def test_month(self, panel_df):
        result = _create_segments(panel_df, "date", "value", "month")
        assert result["_segment"].nunique() > 0

    def test_weekday(self, panel_df):
        result = _create_segments(panel_df, "date", "value", "weekday")
        assert result["_segment"].nunique() == 7

    def test_value_bin(self, panel_df):
        result = _create_segments(panel_df, "date", "value", "value_bin")
        assert result["_segment"].nunique() == 4

    def test_unknown_raises(self, panel_df):
        with pytest.raises(ValueError, match="Unknown segment_by"):
            _create_segments(panel_df, "date", "value", "unknown")


class TestRunSubgroupAnalysis:
    def test_returns_results(self, panel_df):
        # Intervention mid-series so each quarter still has pre/post mass
        # when segmented by weekday (more points per segment than quarter).
        results = run_subgroup_analysis(
            panel_df,
            "date",
            "value",
            "2020-07-01",
            method=Method.ARIMA,
            segment_by="weekday",
        )
        assert len(results) > 0
        assert all(hasattr(r, "effect") for r in results)

    def test_skips_small_segments(self):
        dates = pd.date_range("2020-01-01", periods=60, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "group": ["A"] * 30 + ["B"] * 30,
            "value": np.random.normal(100, 10, 60),
        })
        results = run_subgroup_analysis(
            df,
            "date",
            "value",
            "2020-03-01",
            method=Method.ARIMA,
            segment_by="value_bin",
        )
        for r in results:
            assert r.n_points >= MIN_SEGMENT_SIZE

    def test_results_sorted_by_effect(self, panel_df):
        results = run_subgroup_analysis(
            panel_df,
            "date",
            "value",
            "2020-07-01",
            method=Method.ARIMA,
            segment_by="weekday",
        )
        if len(results) > 1:
            effects = [abs(r.effect) for r in results]
            assert effects == sorted(effects, reverse=True)
