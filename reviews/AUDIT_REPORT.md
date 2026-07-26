# CausalLens — Comprehensive Codebase Audit Report

**Project:** CausalLens (Causal Impact Analysis Platform)  
**Audit Date:** 2026-07-26  
**Auditor:** AEOS Module 23  
**Test Baseline:** 53+ passed (loader, preprocessor, engine)  
**Overall Health Score:** **75 / 100** 🟡

---

## Executive Summary

CausalLens is a **statistically rigorous causal inference platform** with 5 methods (ARIMA, SARIMAX, BSTS, DiD, Synthetic Control), strong report generation (PDF + HTML + CSV), and clean UI/engine separation. The codebase demonstrates strong statistical understanding and good engineering practices.

**Critical Risks (Score < 40):** BSTS platform compatibility, no model diagnostics  
**High-Priority (40-60):** DiD/Synth data validation, SARIMAX timeout, BSTS gating  
**Improvement Backlog (60-80):** Automated diagnostics, model assumption reporting, CSV templates

**Release Recommendation:** ✅ **PASS WITH FINDINGS** — Core methods (ARIMA/SARIMAX) production-ready. BSTS gated. DiD/Synth need validation UX.

---

## Dimension Scores (28 Dimensions)

| # | Dimension | Score | Status | Evidence |
|---|-----------|-------|--------|----------|
| **Architecture (5)** |
| 1 | Module Cohesion | 85 | 🟢 | Core engine (5 methods) cleanly separated from UI. `engine.py` dispatches to method modules. |
| 2 | Coupling | 80 | 🟢 | UI imports engine; engine has no UI deps. Methods isolated in `src/core/*.py`. |
| 3 | API Design | 75 | 🟢 | `causal_effect()` single entry point. Returns frozen dataclass-like dict. |
| 4 | Error Handling | 60 | 🟡 | Try/except in UI. Engine raises `ValueError` for bad inputs. No structured error codes. |
| 5 | Configuration | 80 | 🟢 | Method selection + intervention date + group cols in sidebar. No config file needed. |
| **Reliability (4)** |
| 6 | Edge Cases | 70 | 🟢 | Tests for short series, missing data, outliers, seasonality, placebo. |
| 7 | Concurrency | 50 | 🟡 | Streamlit single-threaded. BSTS blocks session. No timeout guards. |
| 8 | Retry/Backoff | 30 | 🔴 | No retry for model fitting (ARIMA/SARIMAX can fail on convergence). |
| 9 | Graceful Degradation | 45 | 🔴 | BSTS crashes session. SARIMAX can hang. DiD/Synth fail cryptically on bad data. |
| **Security (4)** |
| 10 | Input Validation | 75 | 🟢 | File type check, column validation, date parsing, outlier handling options. |
| 11 | Auth/Authz | 90 | 🟢 | Local-only tool. Streamlit session = boundary. |
| 12 | Secrets Management | 90 | 🟢 | No secrets. No API keys. Pure local computation. |
| 13 | Dependency Vulnerabilities | 60 | 🟡 | `requirements.txt` pinned. No `pip-audit` in CI. PyMC optional (BSTS). |
| **Performance (3)** |
| 14 | Query Efficiency | N/A | — | No database. In-memory pandas/numpy. |
| 15 | Caching | 70 | 🟢 | `@st.cache_data` on analysis + report generation. Hash-based invalidation. |
| 16 | Bundle/Payload Size | 80 | 🟢 | Streamlit + Plotly + statsmodels. Heavy but acceptable for analytical tool. |
| **Testing (5)** |
| 17 | Coverage | 75 | 🟢 | Core engine (ARIMA, BSTS, placebo, DiD, subgroup) well tested. |
| 18 | Test Quality | 80 | 🟢 | Synthetic data fixtures, known-effect validation, placebo progress callback. |
| 19 | Fixture Hygiene | 85 | 🟢 | `conftest.py` provides clean data frames. No shared state. |
| 20 | CI Integration | 60 | 🟡 | GitHub Actions runs pytest. No `pip-audit`, no coverage threshold. |
| 21 | Speed | 65 | 🟡 | ARIMA/SARIMAX <5s. BSTS 60-120s (blocks). Placebo 10-30s. |
| **CI/CD (3)** |
| 22 | Pipeline Completeness | 60 | 🟡 | Lint → Test. Missing: typecheck, `pip-audit`, release artifacts. |
| 23 | Artifact Management | 40 | 🔴 | No versioning, no Docker, no release workflow. Streamlit Cloud push-to-deploy. |
| 24 | Deployment Safety | 50 | 🟡 | BSTS breaks Streamlit Cloud. No canary/rollback for model changes. |
| **Technical Debt (4)** |
| 25 | Dead Code | 90 | 🟢 | Minimal. `ruff` clean. |
| 26 | Documentation Coverage | 70 | 🟡 | Excellent README + sidebar help. **No ADRs, no architecture doc.** |
| 27 | TODO Density | 85 | 🟢 | Few TODOs in BSTS (experimental flags). |
| 28 | Dependency Freshness | 70 | 🟡 | `statsmodels`, `scipy`, `pandas` current. PyMC version pinned for BSTS. |

---

## Critical Findings (Score < 40)

| Dimension | Finding | Impact |
|-----------|---------|--------|
| 8. Retry/Backoff | No retry on ARIMA/SARIMAX convergence failures | Transient numerical issues → user sees error, must rerun |
| 9. Graceful Degradation | **BSTS blocks entire Streamlit session** (60-120s, no progress, no timeout) | User thinks app crashed. Fails on Streamlit Cloud (no PyMC). |
| 23. Artifact Management | No versioning, no Docker, no release artifacts | Can't reproduce deployments, no rollback |
| 24. Deployment Safety | BSTS breaks Streamlit Cloud deploy | App appears broken for cloud users |

---

## High-Priority Findings (40-60)

| # | Finding | Dimension | Remediation |
|---|---------|-----------|-------------|
| H1 | **BSTS blocks session, no timeout, fails on Cloud** | 9 (Graceful Degradation) | Gate behind "Experimental" flag. Add 120s timeout wrapper. Disable by default on Cloud. |
| H2 | **SARIMAX can hang on convergence** | 8 (Retry/Backoff) | Add timeout guard (120s). Catch convergence warnings. |
| H3 | **DiD/Synth Control data shape validation missing** | 4 (Error Handling) | Pre-dispatch validation: check group_col has 2 groups, donor pool size. Show CSV template. |
| H4 | **No automated model diagnostics** | 6 (Edge Cases) | ADF, Ljung-Box, Jarque-Bera, ARCH-LM on residuals. Actionable warnings. |
| H5 | **No dependency vulnerability scanning** | 13 (Dep Vulns) | Add `pip-audit` to CI. |

---

## Improvement Suggestions (60-80)

| # | Suggestion | Dimension | Effort |
|---|------------|-----------|--------|
| I1 | **Automated ARIMA diagnostics** (ADF, Ljung-Box, JB, ARCH) | 4, 6 | Medium |
| I2 | **CSV templates for DiD/Synth** (download buttons) | 4 | Low |
| I3 | **BSTS timeout + progress + Cloud detection** | 9 | Low |
| I4 | **SARIMAX timeout + convergence retry** | 8 | Low |
| I5 | **CSV templates for DiD/Synth (download)** | 4 | Low |
| I6 | **Model fit expander (AIC/BIC, residual plots)** | 26 | Medium |
| I7 | **ADR log for method selection rationale** | 26 | Low |
| I8 | **Dependabot + `pip-audit` in CI** | 13, 28 | Low |

---

## Remediation Roadmap

### Phase 1 — Immediate (Sprint 1) — **Unblock Reliability**

| Task | Owner | Days |
|------|-------|------|
| BSTS: gate behind "Experimental" checkbox | Dev | 0.5 |
| BSTS/SARIMAX: 120s timeout wrapper (ThreadPoolExecutor) | Dev | 0.5 |
| DiD/Synth: pre-dispatch validation + error messages | Dev | 0.5 |
| CSV template download buttons for DiD/Synth | Dev | 0.5 |
| SARIMAX convergence retry (max 2) | Dev | 0.5 |

**Total: ~2.5 days**

### Phase 2 — Next Sprint — **Trust & Diagnostics**

| Task | Owner | Days |
|------|-------|------|
| ARIMA diagnostics: ADF + Ljung-Box + Jarque-Bera + ARCH-LM | Dev | 1 |
| DiagnosticReport dataclass + warnings UI | Dev | 0.5 |
| Model fit expander: AIC/BIC table + residual plot | Dev | 0.5 |
| Streamlit Cloud detection → auto-disable BSTS | Dev | 0.5 |

**Total: ~2.5 days**

### Phase 3 — Next Quarter — **Polish & Extensibility**

| Task | Owner | Days |
|------|-------|------|
| ADR template + first 3 ADRs (method selection, BSTS gating, diagnostics) | Dev | 0.5 |
| `pip-audit` + Dependabot in CI | Dev | 0.5 |
| Release workflow (version bump, Dockerfile, GitHub Release) | Dev | 1 |
| ADR for "why no auto-method-selection" | Dev | 0.5 |

**Total: ~2.5 days**

---

## Trend Tracking

| Dimension | Baseline | Target (Phase 1) | Target (Phase 2) |
|-----------|----------|------------------|------------------|
| Retry/Backoff | 30 | 60 | 75 |
| Graceful Degradation | 45 | 75 | 85 |
| Error Handling | 60 | 75 | 85 |
| Edge Cases | 70 | 80 | 90 |
| Artifact Management | 40 | 60 | 80 |

---

## Auditor Notes

> **Strengths:** Statistically sound methods (ARIMA/SARIMAX/DiD/Synth/Placebo), excellent report generation (PDF/HTML/CSV), strong test fixtures with known-effect validation, clean UI/engine separation, good sidebar help text for method selection.
>
> **Primary Risk:** **BSTS as a "footgun"**. It's the only method that can crash the entire Streamlit session, fails on Streamlit Cloud, and has no timeout. A user clicking "Bayesian STS" waits 2 minutes with frozen UI, then gets an ImportError or convergence error. This single method makes the app feel broken.
>
> **Secondary Risk:** **Silent assumption violations**. ARIMA/SARIMAX return p-values and CIs even when residuals are autocorrelated, non-normal, or heteroscedastic. Users get "statistically significant" results from misspecified models with no warning.
>
> **Tertiary Risk:** **DiD/Synthetic Control are undiscoverable**. They require specific data shapes (panel format, donor pool) but the UI gives no template or validation. Users try them, get cryptic errors, and abandon the methods.
>
> **Recommendation:** Phase 1 (BSTS gating, timeouts, DiD validation) is **mandatory before any feature work**. Phase 2 (diagnostics) is the highest-leverage trust-building feature — it turns CausalLens from "runs models" into "helps you run the RIGHT model."

---

**Signed:** AEOS Module 23  
**Date:** 2026-07-26  
**Next Audit:** 2026-10-26 (or after Phase 1 complete)