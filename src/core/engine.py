from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from ..utils.constants import SIGNIFICANCE_LEVEL
from ..utils.validators import (
    validate_dataframe,
    validate_intervention_date,
    validate_series_length,
)
from .arima_its import run_arima_its

__all__ = ["causal_effect", "Method", "CausalResult"]

logger = logging.getLogger(__name__)


def _make_result(
    method: str,
    src,
    dates: list[str],
    n_pre: int,
    n_post: int,
    _counterfactual: str = "counterfactual",
    _fitted: str | None = "fitted_values",
    _observed: str = "observed",
    **extra,
) -> CausalResult:
    """Build a CausalResult from any method-specific result object."""
    cf = getattr(src, _counterfactual)

    # Resolve significant — prefer src attribute, else compute from p-value
    if hasattr(src, "significant"):
        significant = src.significant
    else:
        significant = src.p_value < SIGNIFICANCE_LEVEL

    # Resolve direction — prefer src attribute, else infer from effect sign
    if hasattr(src, "direction"):
        direction = src.direction
    else:
        direction = "increase" if src.effect > 0 else "decrease"

    return CausalResult(
        method=method,
        effect=src.effect,
        effect_pct=src.effect_pct,
        ci_lower=src.ci_lower,
        ci_upper=src.ci_upper,
        p_value=src.p_value,
        significant=significant,
        direction=direction,
        counterfactual=cf,
        fitted_values=getattr(src, _fitted) if _fitted else cf,
        observed=getattr(src, _observed),
        intervention_idx=src.intervention_idx,
        dates=dates,
        n_pre=n_pre,
        n_post=n_post,
        **extra,
    )


class Method(str, Enum):
    ARIMA = "arima"
    BSTS = "bsts"
    SARIMAX = "sarimax"
    DID = "did"
    SYNTHETIC_CONTROL = "synthetic_control"


@dataclass
class CausalResult:
    method: str
    effect: float
    effect_pct: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool
    direction: str
    counterfactual: np.ndarray
    fitted_values: np.ndarray
    observed: np.ndarray
    intervention_idx: int
    dates: list[str]
    n_pre: int
    n_post: int
    arima_order: tuple[int, int, int] | None = None
    aic: float | None = None
    ljung_box_pvalue: float | None = None
    residuals_ok: bool | None = None
    seasonal_order: tuple[int, int, int, int] | None = None


def causal_effect(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    intervention_date: str,
    method: Method = Method.ARIMA,
    group_col: str | None = None,
    treatment_unit: str | None = None,
) -> CausalResult:
    if date_col not in df.columns:
        # fall back to auto-detection if passed column doesn't exist
        detected_date, detected_metric = validate_dataframe(df)
        date_col = detected_date
        metric_col = detected_metric

    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(
            df[date_col].astype(str), errors="coerce", format="mixed"
        )
        n_failed = df[date_col].isna().sum()
        if n_failed > 0:
            df = df.dropna(subset=[date_col])

    if df.empty:
        raise ValueError("No valid data after date parsing.")

    df = df.sort_values(date_col).reset_index(drop=True)
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=[metric_col])

    if df.empty:
        raise ValueError("No valid data after numeric conversion.")

    dates = df[date_col].values
    y = df[metric_col].values

    dates_pd = pd.to_datetime(dates)

    is_panel = method in (Method.DID, Method.SYNTHETIC_CONTROL)

    if not is_panel and not dates_pd.is_unique:
        logger.warning("Duplicate dates detected. Aggregating data by date.")
        df = df.groupby(date_col)[metric_col].mean().reset_index()
        dates_pd = pd.DatetimeIndex(df[date_col]).sort_values()
        dates = df[date_col].values
        y = df[metric_col].values

    if is_panel:
        dates_pd = pd.DatetimeIndex(dates).sort_values()
        intervention_idx = validate_intervention_date(dates_pd, intervention_date)
        # Validate panel has enough time points per group
        n_time_points = dates_pd.nunique()
        if n_time_points < 30:
            raise ValueError(
                f"Panel data has only {n_time_points} unique time points, "
                f"but at least 30 are needed. "
                f"Upload data with longer time series."
            )
    else:
        intervention_idx = validate_intervention_date(dates_pd, intervention_date)
        validate_series_length(y)

    if method == Method.ARIMA:
        arima_result = run_arima_its(y, intervention_idx)
        result = _make_result(
            "arima", arima_result,
            dates=[str(d)[:10] for d in dates],
            n_pre=intervention_idx,
            n_post=len(y) - intervention_idx,
            arima_order=arima_result.arima_order,
            aic=arima_result.aic,
            ljung_box_pvalue=arima_result.ljung_box_pvalue,
            residuals_ok=arima_result.residuals_ok,
        )
    elif method == Method.BSTS:
        from .bsts import run_bsts
        try:
            bsts_result = run_bsts(y, intervention_idx)
        except RuntimeError as e:
            raise ValueError(f"BSTS analysis failed: {e}")
        result = _make_result(
            "bsts", bsts_result,
            dates=[str(d)[:10] for d in dates],
            n_pre=intervention_idx,
            n_post=len(y) - intervention_idx,
        )
    elif method == Method.SARIMAX:
        sarimax_result = run_arima_its(y, intervention_idx, seasonal=True)
        result = _make_result(
            "sarimax", sarimax_result,
            dates=[str(d)[:10] for d in dates],
            n_pre=intervention_idx,
            n_post=len(y) - intervention_idx,
            arima_order=sarimax_result.arima_order,
            aic=sarimax_result.aic,
            ljung_box_pvalue=sarimax_result.ljung_box_pvalue,
            residuals_ok=sarimax_result.residuals_ok,
            seasonal_order=sarimax_result.seasonal_order,
        )
    elif method == Method.DID:
        if group_col is None or treatment_unit is None:
            raise ValueError(
                "DiD method requires group_col and treatment_unit parameters."
            )
        from .did import run_did
        did_result = run_did(
            df=df,
            time_col=date_col,
            outcome_col=metric_col,
            group_col=group_col,
            treatment_unit=treatment_unit,
            intervention_date=intervention_date,
        )
        result = _make_result(
            "did", did_result,
            dates=did_result.dates,
            n_pre=did_result.intervention_idx,
            n_post=len(did_result.dates) - did_result.intervention_idx,
            _fitted=None,
        )
    elif method == Method.SYNTHETIC_CONTROL:
        if group_col is None or treatment_unit is None:
            raise ValueError(
                "Synthetic Control method requires group_col and treatment_unit parameters."
            )
        from .synthetic_control import run_synthetic_control
        sc_result = run_synthetic_control(
            df=df,
            time_col=date_col,
            outcome_col=metric_col,
            unit_col=group_col,
            treated_unit=treatment_unit,
            intervention_date=intervention_date,
        )
        result = _make_result(
            "synthetic_control", sc_result,
            dates=sc_result.dates,
            n_pre=sc_result.intervention_idx,
            n_post=len(sc_result.dates) - sc_result.intervention_idx,
            _counterfactual="synth_outcome",
            _fitted="synth_outcome",
            _observed="treated_outcome",
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(
        f"Analysis complete: method={result.method}, effect={result.effect:.4f}, "
        f"p={result.p_value:.4f}, significant={result.significant}"
    )

    return result
