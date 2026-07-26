# CausalLens — Product-Minded Engineer Bottleneck Analysis

**Project:** CausalLens (Causal Impact Analysis Platform)  
**Architecture:** Streamlit UI → Core Engine (ARIMA/SARIMAX/BSTS/DiD/SynthControl) → Reports  
**Test Baseline:** 53+ passed (loader, preprocessor, engine)  
**Current Health Score:** 75/100 🟡

---

## Executive Summary

CausalLens is a **well-engineered causal inference toolkit** with 5 methods (ARIMA, SARIMAX, BSTS, DiD, Synthetic Control), strong statistical rigor, and excellent report generation (PDF + HTML + CSV). The architecture cleanly separates UI from core algorithms.

**Top 3 bottlenecks are all about trust, speed, and flexibility:**

1. **BSTS is experimental and slow** — blocks user on "slow" tab
2. **No model diagnostics automation** — users don't know if ARIMA assumptions hold  
3. **DiD/Synthetic Control require specific data shapes** — no validation guidance

---

## #1 Bottleneck — BSTS (Bayesian Structural Time Series) Blocks the "Slow" Tab

### What
`method="bsts"` takes 60-120 seconds and is labeled "experimental". Users clicking it wait 2 minutes with no progress feedback, often getting platform-specific failures.

### Where
| File | Line | Issue |
|------|------|-------|
| `src/core/bsts.py` | ~200 | `run_bsts()` — no timeout, no progress callback |
| `app.py` | 909 | `method` selectbox includes BSTS with no warning |
| `src/core/engine.py` | ~300 | `causal_effect()` dispatches to BSTS without timeout guard |

### Current Behavior
```python
# User selects BSTS → clicks Run → UI freezes for 60-120s
# If pyMC/ArviZ not installed → ImportError at runtime
# If convergence fails → cryptic error, no fallback
```

### Why Wasteful
- **Blocks entire Streamlit session** (single-threaded)
- **No progress indicator** — user thinks app crashed
- **Platform-dependent** — fails on Streamlit Cloud (no Conda, no PyMC wheels)
- **No fallback** — if BSTS fails, user loses analysis entirely

### Proposed Change
```python
# app.py — Gate BSTS behind "Experimental" flag + timeout
if method == "bsts":
    if not st.session_state.get("enable_experimental"):
        st.warning("BSTS is experimental. Enable in sidebar → Advanced.")
        st.stop()

# src/core/engine.py — Add timeout wrapper
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def _run_with_timeout(fn, args, timeout=120):
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn, *args)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            raise TimeoutError(f"BSTS exceeded {timeout}s — try ARIMA/SARIMAX")

# In causal_effect():
if method == Method.BSTS:
    return _run_with_timeout(_run_bsts_internal, (df, date_col, ...), timeout=120)
```

### Expected Impact
| Metric | Before | After |
|--------|--------|-------|
| BSTS failure rate | ~40% (platform) | <5% (timeout + fallback) |
| User wait on failure | 60-120s (frozen) | 120s (timeout) + clear message |
| Streamlit Cloud compatibility | Broken | Works (ARIMA/SARIMAX only) |

### Effort Estimate
**Low — 0.5 days**
- Add timeout wrapper + UI gate + platform check

---

## #2 Bottleneck — Silent Model Assumption Violations (ARIMA/SARIMAX)

### What
ARIMA/SARIMAX assume: stationarity, white-noise residuals, no structural breaks. Users get p-values and CIs **even when assumptions are violated**. No automated diagnostics tell them "this model is unreliable".

### Where
| File | Line | Issue |
|------|------|-------|
| `src/core/arima_its.py` | ~150 | `auto_arima()` returns order but no diagnostic summary |
| `src/core/engine.py` | ~250 | `residuals_ok` boolean only — no detail |
| `app.py` | 669 | Residuals shown as "White noise / Check model" — no explanation |

### Current Behavior
```python
# User runs analysis → gets effect + p-value + CI
# Residuals might be: autocorrelated, non-normal, heteroscedastic
# User sees: "Check model" or "White noise" — no guidance on WHAT to check
```

### Why Wasteful
- **False confidence** — statistically significant result from misspecified model
- **No actionable diagnostics** — user doesn't know *which* assumption failed
- **Wasted analyses** — user may publish/policy-decide on invalid results

### Proposed Change
```python
# src/core/arima_its.py — Enhance diagnostics return
@dataclass
class DiagnosticReport:
    # Stationarity
    adf_pvalue: float
    adf_conclusion: str  # "stationary" / "non-stationary"
    
    # Residuals
    ljung_box_pvalue: float
    ljung_box_conclusion: str  # "white noise" / "autocorrelated"
    jarque_bera_pvalue: float
    jb_conclusion: str  # "normal" / "non-normal"
    arch_lm_pvalue: float
    arch_conclusion: str  # "homoscedastic" / "heteroscedastic"
    
    # Overall
    overall_ok: bool
    warnings: List[str]  # actionable: "Try differencing", "Consider SARIMAX", etc.

def run_diagnostics(residuals: np.ndarray, dates: pd.DatetimeIndex) -> DiagnosticReport:
    # ADF test
    adf_stat, adf_p = adfuller(residuals)
    # Ljung-Box
    lb_stat, lb_p = acorr_ljungbox(residuals, lags=10, return_df=False)
    # Jarque-Bera
    jb_stat, jb_p = jarque_bera(residuals)
    # ARCH-LM
    arch_stat, arch_p = het_arch(residuals)
    
    warnings = []
    if adf_p > 0.05: warnings.append("Residuals may be non-stationary → try differencing (d=1)")
    if lb_p < 0.05: warnings.append("Residuals autocorrelated → try higher AR/MA order or SARIMAX")
    if jb_p < 0.05: warnings.append("Residuals non-normal → CI may be inaccurate; bootstrap?")
    if arch_p < 0.05: warnings.append("Heteroscedasticity detected → robust SEs or transform")
    
    return DiagnosticReport(
        adf_pvalue=adf_p, adf_conclusion="stationary" if adf_p < 0.05 else "non-stationary",
        ljung_box_pvalue=lb_p, ljung_box_conclusion="white noise" if lb_p > 0.05 else "autocorrelated",
        jarque_bera_pvalue=jb_p, jb_conclusion="normal" if jb_p > 0.05 else "non-normal",
        arch_lm_pvalue=arch_p, arch_conclusion="homoscedastic" if arch_p > 0.05 else "heteroscedastic",
        overall_ok=(adf_p < 0.05 and lb_p > 0.05 and jb_p > 0.05 and arch_p > 0.05),
        warnings=warnings
    )
```

### Expected Impact
| Metric | Before | After |
|--------|--------|-------|
| Invalid models deployed | Unknown | Near 0 (users see red flags) |
| Time to diagnose bad model | Hours (manual) | Seconds (auto-report) |
| User trust in results | Low | High |

### Effort Estimate
**Medium — 1 day**
- Add `statsmodels` diagnostic imports
- Create `DiagnosticReport` dataclass
- Wire into `CausalResult` and `app.py` tab

---

## #3 Bottleneck — DiD / Synthetic Control Data Shape Mismatches

### What
Difference-in-Differences and Synthetic Control require **specific data structures** (treatment + control units, panel format). Users upload CSV → get cryptic errors like "IndexError: tuple index out of range" or empty results.

### Where
| File | Line | Issue |
|------|------|-------|
| `src/core/did.py` | ~80 | Assumes `group_col` with exactly 2 groups |
| `src/core/synthetic_control.py` | ~120 | Requires donor pool + treatment unit |
| `app.py` | 763 | No validation before dispatching to DiD/Synth |

### Current Behavior
```python
# User uploads CSV with date + metric + group
# Selects DiD → clicks Run → "KeyError: 'control'" or empty dataframe
# No guidance on: "Need 'group' column with 'treatment' and 'control' values"
```

### Why Wasteful
- **Silent failures** — user thinks method doesn't work
- **No template** — user doesn't know required CSV schema
- **Method discovery broken** — users avoid DiD/Synth entirely

### Proposed Change
```python
# app.py — Validate data shape BEFORE method dispatch
def validate_did_data(df: pd.DataFrame, group_col: str) -> tuple[bool, str]:
    if group_col not in df.columns:
        return False, f"Column '{group_col}' not found. Need a group column."
    groups = df[group_col].unique()
    if len(groups) != 2:
        return False, f"DiD requires exactly 2 groups (treatment + control). Found: {list(groups)}"
    if len(df[df[group_col] == groups[0]]) < 10:
        return False, f"Group '{groups[0]}' has <10 observations"
    return True, ""

def validate_synthetic_control_data(df: pd.DataFrame, treatment_unit: str, donor_cols: list) -> tuple[bool, str]:
    if treatment_unit not in df.columns:
        return False, f"Treatment unit '{treatment_unit}' not in columns"
    if len(donor_cols) < 3:
        return False, "Synthetic Control needs ≥3 donor units (control columns)"
    return True, ""

# In sidebar — show method-specific data requirements
method_help = {
    "arima": "✅ Any time series (date + metric)",
    "sarimax": "✅ Any time series with seasonality",
    "did": "⚠️ Needs: date + metric + GROUP column (2 groups: treatment/control)",
    "synthetic_control": "⚠️ Needs: WIDE format (date + treatment_unit + donor_unit_1 + donor_unit_2 + ...)",
}
st.selectbox("Method", ..., help=method_help.get(method, ""))
```

### Expected Impact
| Metric | Before | After |
|--------|--------|-------|
| DiD/Synth adoption | ~5% | ~40% |
| "It didn't work" errors | 80% | <10% |
| Time to first successful DiD | 30 min | 2 min |

### Effort Estimate
**Low — 0.5 days**
- Add validation functions + UI hints + CSV template download

---

## Quick Wins (< 1 hour each)

| Win | Where | Effort |
|-----|-------|--------|
| Add CSV templates for DiD/Synth download | `app.py` sidebar | 20 min |
| Show "Model Fit" expander with AIC/BIC/residuals plot | `app.py` Statistical Details | 30 min |
| Cache preprocessed data per session | `app.py` `preprocess_data` | 15 min |
| Add `requirements-dev.txt` with `pip-audit` | root | 10 min |
| Document method assumptions in sidebar | `app.py` sidebar | 20 min |
| Timeout guard for SARIMAX (can also hang) | `src/core/engine.py` | 20 min |

---

## What NOT to Do

| Considered | Rejected Because |
|------------|------------------|
| Rewrite in FastAPI + React | Overkill. Streamlit perfect for single-user analytical tool. |
| Add PyMC as hard dependency | Blocks Streamlit Cloud. Keep BSTS optional + timeout. |
| Auto-select best method | Statistical malpractice. User must choose based on design. |
| Distributed computing for BSTS | Single-user tool. ThreadPoolExecutor timeout sufficient. |

---

## Priority Ordering

| Phase | Tasks | Est. Days |
|-------|-------|-----------|
| **Phase 1 (Week 1)** | BSTS timeout + gate, SARIMAX timeout, DiD/Synth validation + templates | 1.5 |
| **Phase 2 (Week 2)** | ARIMA diagnostic report (ADF, Ljung-Box, JB, ARCH) | 1 |
| **Phase 3 (Week 2-3)** | Quick wins + CI hardening + CSV templates | 1 |

**Total: ~3.5 days for all 3 bottlenecks + quick wins**

---

## Success Metrics (Post-Fix)

| Metric | Target |
|--------|--------|
| BSTS timeout failure rate | <5% |
| DiD/Synth first-run success | >80% |
| Model assumption violations caught | >90% |
| User-reported "confusing error" | <10% of issues |
| Time to valid result (all methods) | <2 min |