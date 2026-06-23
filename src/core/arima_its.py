from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ..utils.constants import DEFAULT_CI_ALPHA

__all__ = ["run_arima_its", "ITSResult"]

logger = logging.getLogger(__name__)


@dataclass
class ITSResult:
    effect: float
    effect_pct: float
    ci_lower: float
    ci_upper: float
    p_value: float
    counterfactual: np.ndarray
    fitted_values: np.ndarray
    observed: np.ndarray
    intervention_idx: int
    arima_order: tuple[int, int, int] | tuple[int, int, int, int]
    aic: float
    ljung_box_pvalue: float
    residuals_ok: bool
    seasonal_order: tuple[int, int, int, int] | None = None


def _select_arima_order(
    y: np.ndarray,
    max_p: int = 3,
    max_d: int = 1,
    max_q: int = 3,
) -> tuple[int, int, int]:
    best_aic = np.inf
    best_order = (1, 1, 1)

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(y, order=(p, d, q))
                        fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

    return best_order


def _select_arima_order_fast(
    y: np.ndarray,
) -> tuple[int, int, int]:
    if len(y) > 1000:
        return _select_arima_order(y, max_p=1, max_d=1, max_q=1)
    if len(y) > 300:
        return _select_arima_order(y, max_p=2, max_d=1, max_q=2)
    return _select_arima_order(y)


def _infer_seasonal_period(n: int) -> int:
    if n > 10000:
        return 0
    if n > 3000:
        return 7
    if n > 200:
        return 12
    if n > 50:
        return 4
    return 0


def _select_seasonal_order(
    y: np.ndarray,
    s: int,
) -> tuple[int, int, int, int]:
    n = len(y)

    if n > 5000:
        return (0, 1, 1, s)

    best_aic = np.inf
    best_order = (0, 1, 1, s)

    for P in range(2):
        for Q in range(2):
            if P == 0 and Q == 0:
                continue
            for D in range(2):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = SARIMAX(
                            y,
                            order=(1, 1, 1),
                            seasonal_order=(P, D, Q, s),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        fitted = model.fit(disp=False, maxiter=30)
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (P, D, Q, s)
                except Exception:
                    continue

    return best_order


def _check_residuals(residuals: np.ndarray) -> tuple[float, bool]:
    n_lags = min(10, len(residuals) // 5)
    if n_lags < 1:
        return 1.0, True
    lb_result = acorr_ljungbox(residuals, lags=[n_lags], return_df=True)
    lb_pvalue = float(lb_result["lb_pvalue"].iloc[0])
    return lb_pvalue, lb_pvalue > 0.05


def run_arima_its(
    y: np.ndarray,
    intervention_idx: int,
    order: tuple[int, int, int] | None = None,
    seasonal: bool = False,
    seasonal_order: tuple[int, int, int, int] | None = None,
) -> ITSResult:
    y = np.asarray(y, dtype=float)
    n = len(y)

    pre = y[:intervention_idx]
    post = y[intervention_idx:]

    if order is None:
        if n > 5000:
            selected_order = _select_arima_order(pre, max_p=2, max_d=1, max_q=2)
        elif len(pre) > 1000:
            selected_order = _select_arima_order_fast(pre)
        else:
            selected_order = _select_arima_order(pre)
        logger.info(f"Auto-ARIMA selected order {selected_order}")
    else:
        selected_order = order

    selected_seasonal = None
    use_sarimax = False

    if seasonal:
        s = seasonal_order[3] if seasonal_order else _infer_seasonal_period(n)
        if s > 0:
            if seasonal_order is None:
                selected_seasonal = _select_seasonal_order(pre, s)
            else:
                selected_seasonal = seasonal_order
            use_sarimax = True
            logger.info(
                f"SARIMAX seasonal order {selected_seasonal} (period s={s})"
            )

    if use_sarimax:
        model = SARIMAX(
            pre,
            order=selected_order,
            seasonal_order=selected_seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
    else:
        model = ARIMA(pre, order=selected_order)

    try:
        fitted = model.fit(disp=False) if use_sarimax else model.fit()
    except ValueError as e:
        raise ValueError(
            f"{'SARIMAX' if use_sarimax else 'ARIMA'} model fitting failed. "
            f"The data may be non-stationary or too short. "
            f"Try a different intervention date or method. Details: {e}"
        )
    except Exception as e:
        raise ValueError(
            f"{'SARIMAX' if use_sarimax else 'ARIMA'} model fitting failed: {e}"
        )

    forecast = fitted.get_forecast(steps=len(post))
    predicted_mean = np.asarray(forecast.predicted_mean).flatten()
    raw_ci = forecast.conf_int(alpha=DEFAULT_CI_ALPHA)
    forecast_ci = np.asarray(raw_ci)

    if forecast_ci.ndim == 1:
        forecast_ci = forecast_ci.reshape(-1, 2)

    counterfactual_full = np.concatenate([
        np.asarray(fitted.fittedvalues).flatten(),
        predicted_mean,
    ])

    post_actual = y[intervention_idx:]
    post_predicted = predicted_mean

    effect = float(np.mean(post_actual - post_predicted))

    # effect_pct = (mean_effect / mean_abs_predicted) * 100
    # Avoids per-point ratio distortion when predictions are near zero.
    mean_abs_predicted = float(np.mean(np.abs(post_predicted)))
    if abs(mean_abs_predicted) > 1e-10:
        effect_pct = float(effect / mean_abs_predicted * 100)
    else:
        pre_mean = float(np.mean(y[:intervention_idx])) if intervention_idx > 0 else 1.0
        effect_pct = float(effect / (abs(pre_mean) + 1e-10) * 100)

    ci_lower = float(np.mean(forecast_ci[:, 0]))
    ci_upper = float(np.mean(forecast_ci[:, 1]))

    se = (ci_upper - ci_lower) / (2 * stats.norm.ppf(0.975))
    if se > 0:
        t_stat = effect / se
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=len(post) - 1)))
    else:
        p_value = 1.0

    fitted_all = np.zeros(n)
    fitted_values_arr = np.asarray(fitted.fittedvalues).flatten()
    fitted_all[: len(fitted_values_arr)] = fitted_values_arr
    fitted_all[intervention_idx:] = post_predicted

    ljung_box_pvalue, residuals_ok = _check_residuals(fitted.resid)

    return ITSResult(
        effect=effect,
        effect_pct=effect_pct,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        counterfactual=counterfactual_full,
        fitted_values=fitted_all,
        observed=y,
        intervention_idx=intervention_idx,
        arima_order=selected_order,
        aic=float(fitted.aic),
        ljung_box_pvalue=ljung_box_pvalue,
        residuals_ok=residuals_ok,
        seasonal_order=selected_seasonal,
    )
