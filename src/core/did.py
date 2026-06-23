from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

__all__ = ["DiDResult", "run_did"]

logger = logging.getLogger(__name__)


@dataclass
class DiDResult:
    effect: float
    effect_pct: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool
    direction: str
    treated_pre_mean: float
    treated_post_mean: float
    control_pre_mean: float
    control_post_mean: float
    parallel_trends_p: float
    n_treated: int
    n_control: int
    dates: list[str]
    observed: np.ndarray
    counterfactual: np.ndarray
    intervention_idx: int


def run_did(
    df: pd.DataFrame,
    time_col: str,
    outcome_col: str,
    group_col: str,
    treatment_unit: str,
    intervention_date: str,
) -> DiDResult:
    if group_col not in df.columns:
        raise ValueError(
            f"Group column '{group_col}' not found in data. "
            f"Available columns: {list(df.columns)}"
        )

    if treatment_unit not in df[group_col].unique():
        raise ValueError(
            f"Treatment unit '{treatment_unit}' not found in column '{group_col}'. "
            f"Available units: {sorted(df[group_col].unique())}"
        )

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    df["post"] = (df[time_col] >= pd.Timestamp(intervention_date)).astype(int)
    df["treated"] = (df[group_col] == treatment_unit).astype(int)
    df["treated_x_post"] = df["treated"] * df["post"]

    treated_df = df[df[group_col] == treatment_unit].copy()
    control_df = df[df[group_col] != treatment_unit].copy()

    dates = sorted(df[time_col].unique())
    intervention_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(intervention_date))

    treated_pre = treated_df[treated_df["post"] == 0][outcome_col]
    treated_post = treated_df[treated_df["post"] == 1][outcome_col]
    control_pre = control_df[control_df["post"] == 0][outcome_col]
    control_post = control_df[control_df["post"] == 1][outcome_col]

    treated_pre_mean = float(treated_pre.mean())
    treated_post_mean = float(treated_post.mean())
    control_pre_mean = float(control_pre.mean())
    control_post_mean = float(control_post.mean())

    # ── Parallel trends test (pre-intervention only) ──────────────
    # Regress outcome on time trend + treated + time×treated.
    # A significant interaction means the pre-trends differ — violation.
    pre_df = df[df["post"] == 0].copy()
    pre_df = pre_df.sort_values(time_col).reset_index(drop=True)
    pre_df["time_trend"] = pre_df.groupby(group_col).cumcount()
    pre_X = pre_df[["time_trend", "treated"]].copy()
    pre_X["time_trend_x_treated"] = pre_X["time_trend"] * pre_X["treated"]
    pre_X = sm.add_constant(pre_X)
    pre_y = pre_df[outcome_col]

    n_pre_periods = pre_df[group_col].value_counts().min()
    if pre_X["time_trend"].nunique() < 2 or pre_X["treated"].nunique() < 2 or n_pre_periods < 4:
        parallel_trends_p = float("nan")
        logger.warning(
            "Parallel trends test skipped: insufficient pre-intervention data "
            "(need ≥2 time periods with both treatment and control groups, "
            f"got {n_pre_periods} min periods per group)"
        )
    else:
        try:
            pre_model = sm.OLS(pre_y, pre_X).fit()
            parallel_trends_p = float(pre_model.pvalues.get("time_trend_x_treated", 1.0))
        except Exception as e:
            parallel_trends_p = float("nan")
            logger.warning(f"Parallel trends test failed: {e}")

    if parallel_trends_p is not None and not (np.isnan(parallel_trends_p) or parallel_trends_p > 0.05):
        logger.warning(
            f"Parallel trends assumption may be violated "
            f"(p={parallel_trends_p:.4f} < 0.05). "
            f"DiD results may be unreliable."
        )

    X = df[["treated", "post", "treated_x_post"]]
    X = sm.add_constant(X)
    y = df[outcome_col]

    model = sm.OLS(y, X).fit()

    did_effect = float(model.params["treated_x_post"])
    ci = model.conf_int().loc["treated_x_post"]
    ci_lower = float(ci[0])
    ci_upper = float(ci[1])
    p_value = float(model.pvalues["treated_x_post"])

    mean_level = abs(treated_pre_mean) + 1e-10
    effect_pct = did_effect / mean_level * 100

    date_strs = [str(d)[:10] for d in dates]

    observed_all = np.zeros(len(dates))
    counterfactual_all = np.zeros(len(dates))

    for i, d in enumerate(dates):
        day_data = df[df[time_col] == d]
        treated_val = day_data[day_data[group_col] == treatment_unit][outcome_col].values
        control_vals = day_data[day_data[group_col] != treatment_unit][outcome_col].values

        if len(treated_val) > 0:
            observed_all[i] = treated_val[0]
        if len(control_vals) > 0:
            control_mean = np.mean(control_vals)
            if i < intervention_idx:
                counterfactual_all[i] = treated_val[0] if len(treated_val) > 0 else control_mean
            else:
                counterfactual_all[i] = control_mean + (treated_pre_mean - control_pre_mean)

    direction = "increase" if did_effect > 0 else "decrease"
    significant = p_value < 0.05

    return DiDResult(
        effect=did_effect,
        effect_pct=effect_pct,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        significant=significant,
        direction=direction,
        treated_pre_mean=treated_pre_mean,
        treated_post_mean=treated_post_mean,
        control_pre_mean=control_pre_mean,
        control_post_mean=control_post_mean,
        parallel_trends_p=parallel_trends_p,
        n_treated=len(treated_df),
        n_control=len(control_df),
        dates=date_strs,
        observed=observed_all,
        counterfactual=counterfactual_all,
        intervention_idx=intervention_idx,
    )
