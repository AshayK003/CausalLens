from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["run_bsts", "BSTSResult"]


@dataclass
class BSTSResult:
    effect: float
    effect_pct: float
    ci_lower: float
    ci_upper: float
    p_value: float
    counterfactual: np.ndarray
    fitted_values: np.ndarray
    observed: np.ndarray
    intervention_idx: int


def run_bsts(
    y: np.ndarray,
    intervention_idx: int,
) -> BSTSResult:
    try:
        from causalimpact import CausalImpact
        ci = CausalImpact(
            pd.DataFrame({"y": y}).assign(index=pd.RangeIndex(len(y))),
            [0, intervention_idx - 1],
            [intervention_idx, len(y) - 1],
            model_args={"niter": 2000},
        )
    except ImportError:
        raise RuntimeError(
            "causalimpact package is not installed. "
            "Install it with: pip install causalimpact"
        )
    except Exception as e:
        raise RuntimeError(f"BSTS model fitting failed: {e}")

    n = len(y)
    post_actual = y[intervention_idx:]

    # Detect broken causalimpact (empty model dict or None inferences)
    model_broken = (
        not isinstance(ci.model, dict)
        or not ci.model
        or ci.inferences is None
    )

    if model_broken:
        raise RuntimeError(
            "BSTS model failed to fit properly on this system. "
            "The causalimpact package may be incompatible with your Python/numpy version. "
            "Try using the ARIMA method instead."
        )

    try:
        post_predicted = np.asarray(ci.model.predicted_mean).flatten()
        if len(post_predicted) > len(post_actual):
            post_predicted = post_predicted[: len(post_actual)]
        elif len(post_predicted) < len(post_actual):
            post_predicted = np.pad(
                post_predicted, (0, len(post_actual) - len(post_predicted))
            )
    except Exception as e:
        raise RuntimeError(f"BSTS prediction extraction failed: {e}")

    effect = float(np.mean(post_actual - post_predicted))

    # effect_pct = (mean_effect / mean_abs_predicted) * 100
    mean_abs_predicted = float(np.mean(np.abs(post_predicted)))
    if abs(mean_abs_predicted) > 1e-10:
        effect_pct = float(effect / mean_abs_predicted * 100)
    else:
        pre_mean = float(np.mean(y[:intervention_idx])) if intervention_idx > 0 else 1.0
        effect_pct = float(effect / (abs(pre_mean) + 1e-10) * 100)

    ci_lower = np.nan
    ci_upper = np.nan
    p_value = np.nan

    try:
        inferences = ci.inferences
        if inferences is not None and hasattr(inferences, "columns"):
            # Try multiple possible column naming conventions across causalimpact versions
            col_map = {
                "p_value": ["posterior_tail_area", "p_value", "p-val"],
                "ci_lower": [
                    "abs_effect_lower", "point_pred_lower",
                    "pred_abs_effect_lower", "avg_abs_effect_lower",
                ],
                "ci_upper": [
                    "abs_effect_upper", "point_pred_upper",
                    "pred_abs_effect_upper", "avg_abs_effect_upper",
                ],
            }

            for key, candidates in col_map.items():
                found = [c for c in candidates if c in inferences.columns]
                if found and key == "p_value":
                    p_value = float(inferences[found[0]].iloc[0])
                    # p-value should be in [0, 1]; clamp if outside
                    if not (0 <= p_value <= 1):
                        p_value = np.nan
                elif found and key in ("ci_lower", "ci_upper"):
                    val = float(inferences[found[0]].mean())
                    if key == "ci_lower":
                        ci_lower = val
                    else:
                        ci_upper = val

            # Validate CI ordering: if lower > upper, swap
            if not np.isnan(ci_lower) and not np.isnan(ci_upper) and ci_lower > ci_upper:
                ci_lower, ci_upper = ci_upper, ci_lower
    except Exception as e:
        logger.warning(f"Could not extract BSTS inferences: {e}")

    se = float(np.std(post_actual - post_predicted) / np.sqrt(len(post_actual)))

    if np.isnan(ci_lower) or np.isnan(ci_upper):
        ci_lower = effect - 1.96 * se
        ci_upper = effect + 1.96 * se

    if np.isnan(p_value):
        if se > 0:
            from scipy import stats
            t_stat = effect / se
            p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=len(post_actual) - 1)))
        else:
            p_value = 1.0

    counterfactual_full = np.zeros(n)
    try:
        fitted = np.asarray(ci.model.predicted_mean).flatten()
        counterfactual_full[: len(fitted)] = fitted
    except Exception:
        pass

    return BSTSResult(
        effect=effect,
        effect_pct=effect_pct,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        counterfactual=counterfactual_full,
        fitted_values=counterfactual_full,
        observed=y,
        intervention_idx=intervention_idx,
    )
