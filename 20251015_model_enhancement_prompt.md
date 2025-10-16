# Sprint Plan — Intraday TimeGPT Profitability Remediation (Drop-in for Claude Code)

**Role:** Chief Data Scientist guidance
**Executor:** Claude Code (in-repo editor/agent)
**Scope:** Intraday equities on minute bars; horizons 30m & 60m; single- and multi-ticker.
**Goal:** Eliminate model/horizon misuse, align trading logic to horizon, harden features/exogs, and ship a walk-forward A/B framework with after-cost KPIs.
**Constraint:** Do **not** assume file paths. Claude must **search** and patch the repo in place.

---

## Context (why this exists)

* You saw Nixtla’s **“h exceeds model horizon”** warning at 60m and stepped down to 30m.
* The system has been **consistently unprofitable**; root causes likely include:

  * **Frequency inference**/irregular minute data causing wrong model horizon checks.
  * **Horizon/volatility mismatch** (entry on σ₅m, exit on σ₃₀m/σ₆₀m).
  * **Exog parity** issues (`X_df` vs historical) and deterministic features not mirrored into the forecast grid.
  * **Snapshot cadence** not aligned to horizon (too sparse or too frequent → cost drag).
  * **Payoff geometry** (TP < SL) that needs unrealistically high hit rates to break even.
* We’ll fix fundamentals first, then add a **walk-forward optimizer** for a *small* set of high-leverage knobs.

---

## Objectives (what “done” means)

1. **No horizon warnings** with correctly inferred **minute** frequency; explicit logs show `freq=min`.
2. **Horizon/model alignment:** choose appropriate TimeGPT variant; force correct horizon units.
3. **σ alignment:** gate entries and set TP/SL using **σ of the same horizon** (σ₃₀m or σ₆₀m).
4. **Exog parity hardened:** deterministic calendar features present **identically** in `y_df` & `X_df`; name/dtype parity enforced.
5. **Snapshot cadence** matches horizon (e.g., every 30m or every 60m in RTH); turnover constrained.
6. **Walk-forward A/B harness** with **CRPS**, **IC**, **after-cost Sharpe**, and **calibration**; compact hyperparameter search over a **small** bounded space.
7. **Decision report** that picks 30m vs 60m + parameter set based on **out-of-sample** KPIs.

---

## KPIs (track and gate releases)

* **Data / Model**

  * `% runs with “Inferred freq: min” present:` **100%**
  * `Nixtla horizon warnings:` **0** in final pipeline
* **Forecast quality**

  * **CRPS**: ≤ baseline by **−10%** or more (better is lower)
  * **IC( q50 , realized )**: ≥ **0.03–0.05** OOS (by hour and overall)
  * **Calibration**: realized 30m return containment within [q25,q75] ≈ **50% ± 5pp**
* **Trading quality (after costs)**

  * **Sharpe (OOS)**: ≥ **0.5** on a 3–6-month walk-forward
  * **Turnover**: ≤ **12/day** for 30m cadence, ≤ **8/day** for 60m (single-ticker)
  * **Expectancy**: positive; **TP ≥ SL** or EV-based exit
  * **Max DD** improvement vs baseline by **≥ 15%** at matched vol target
* **Robustness**

  * Monthly KPI variation (std/mean) **≤ 30%**
  * Param drift across months: **≤ 20%** relative change

---

## Sprints & Tasks (Claude Code to execute)

> **Important:** In each sprint, first **search** (`ripgrep/find`) for relevant modules (e.g., forecast client wrapper, scheduler, payload builder, backtest/engine), then patch with small, commented commits. Add tests alongside.

### Sprint 0 — Repo discovery & guardrails ✅

* Locate:

  * **Forecast client wrapper**: `src/timegpt_v2/forecast/timegpt_client.py` - TimeGPTClient class with `h`, `freq`, and model variant configuration
  * **Scheduler**: `src/timegpt_v2/forecast/scheduler.py` - ForecastScheduler class with RTH/active-window logic
  * **Payload builder**: `src/timegpt_v2/framing/build_payloads.py` - build_y_df, build_x_df_for_horizon functions
  * **Backtest/engine**: `src/timegpt_v2/backtest/simulator.py` - BacktestSimulator class with entry/exit rules and σ usage
  * **Trading rules**: `src/timegpt_v2/trading/rules.py` - RuleParams and TradingRules classes
* **Logging**: Python's standard logging module is available throughout the codebase

**Acceptance:** ✅ All paths/functions identified; logger available.

---

### Sprint 1 — Frequency enforcement & data regularization (eliminate warning) ✅

* Enforce **minute** frequency: regularize timestamps, forward-fill gaps only where appropriate, ensure monotone equally-spaced minute index in RTH.
  * ✅ Enhanced `build_y_df()` with explicit 1-minute resampling and gap filling
* Explicitly **set `freq="min"`** (or equivalent) in the TimeGPT client call if inference is unreliable.
  * ✅ Added frequency enforcement in `TimeGPTClient.forecast()` with override logic
* Log **"Inferred/forced freq: min"** once per run.
  * ✅ Added frequency logging with one-time flag to prevent spam

**Tests**

* ✅ Synthetic minute series with RTH gaps → client sees **min**.
* ✅ Unit test ensures X_df length equals `h` and timestamps are minute-spaced.

**Acceptance:** ✅ No horizon warnings at 60m when data are minute-regular.

**Implementation:**
- Enhanced `src/timegpt_v2/forecast/timegpt_client.py:227-234` with frequency override and logging
- Enhanced `src/timegpt_v2/framing/build_payloads.py:87-106` with minute regularity enforcement
- Added `src/timegpt_v2/framing/build_payloads.py:242-263` with horizon length validation
- Created comprehensive tests in `tests/test_sprint1_frequency_enforcement.py`

---

### Sprint 2 — Horizon ↔ model selection policy ✅

* Implement policy:

  * For intraday minute bars with `h ∈ {30,60}`, prefer **`timegpt-1`** (not long-horizon).
    * ✅ Added model selection validation in `_validate_and_log_horizon()`
  * Allow override via config; log chosen model and rationale.
    * ✅ Config override respected with warnings for non-optimal choices
* Validate `h` is interpreted as **minutes/steps**, not hours/days.
  * ✅ Added explicit horizon validation with minute interpretation logging

**Tests**

* ✅ Param test: (h=30, 60) choose base model; override respected.

**Acceptance:** ✅ Correct model automatically selected for 30m/60m intraday.

**Implementation:**
- Enhanced `src/timegpt_v2/forecast/timegpt_client.py:232,540-577` with horizon validation and model selection policy
- Added comprehensive logging for model choice and horizon interpretation
- Created parameter override functionality with proper warnings
- Built thorough test suite in `tests/test_sprint2_horizon_model_selection.py`

---

### Sprint 3 — σ alignment and payoff geometry ✅

* Add utilities to compute **σ₃₀m** and **σ₆₀m** (rolling realized vol or EWMA on minute returns).
  * ✅ Created `src/timegpt_v2/utils/sigma_utils.py` with horizon-aligned sigma computation
* Refactor:

  * Entry trigger uses **k·σ_h** where **h ∈ {30,60}** matches the forecast horizon.
    * ✅ Enhanced `TradingRules.get_entry_signal()` with horizon sigma support
  * Exits use **TP/SL in σ_h units**; set default **TP ≥ SL** (e.g., 2.5σ / 2.0σ).
    * ✅ Enhanced `TradingRules.get_exit_signal()` with horizon sigma and TP/SL validation
  * Optional EV-based exit using quantiles (close when expected return ≤ 0 or adverse tail prob > threshold).
    * ✅ Added `compute_ev_exit_threshold()` for expected value-based exits
* Config surface: `horizon_min`, `sigma_basis: "horizon"`, `tp_sigma`, `sl_sigma`.
  * ✅ Added `RuleParams.horizon_minutes` and `RuleParams.sigma_basis` parameters

**Tests**

* ✅ Unit tests verify correct σ window selection for 30m vs 60m.
* ✅ Backtest unit on synthetic price path: with TP ≥ SL, breakeven hit-rate math matches.

**Acceptance:** ✅ No mixing of σ bases; parameters honored.

**Implementation:**
- Created comprehensive sigma utilities in `src/timegpt_v2/utils/sigma_utils.py`
- Enhanced `src/timegpt_v2/trading/rules.py` with horizon-aligned entry/exit logic
- Added TP/SL geometry validation to prevent negative expectancy setups
- Implemented EV-based exits using quantile forecasts
- Built test suite in `tests/test_sprint3_sigma_alignment.py`

---

### Sprint 4 — Deterministic exog parity & X_df shape contract

* Define a **canonical list** of deterministic exogs (e.g., minute_of_day sin/cos, minutes_since_open/to_close, day_of_week).
* Ensure **identical names+dtypes** exist in **both** `y_df` and `X_df`.
* Add **preflight validator**:

  * No `_x/_y` suffixes
  * `X_df` has exactly **h** rows per series, same exog columns as historical deterministic set
  * Hard-fail with actionable message if parity/shape off

**Tests**

* Parity success/failure tests (name mismatch, dtype mismatch, `_x/_y` guard).
* Horizon length mismatch raises.

**Acceptance:** Exog parity errors are caught **before** API calls.

---

### Sprint 5 — Snapshot cadence aligned to horizon ✅

* Add presets: **every 30m** and **every 60m** snapshots within RTH (exclude first 5–10m and last 5–10m by default).
  * ✅ Implemented `create_horizon_preset()` and `create_snapshot_preset()` functions
  * ✅ Added 30m, 60m, and high-frequency presets with RTH alignment
* Enforce **max trades/day** or **min spacing** to cap turnover.
  * ✅ Added `max_trades_per_day` and `max_total_snapshots` parameters
  * ✅ Implemented turnover control in ForecastScheduler
* Log a coverage line per run: `planned / sent / ok / skipped by reason`.
  * ✅ Enhanced coverage tracking with detailed skip reasons and metrics

**Tests**

* ✅ Unit: given start/end and preset, expected snapshot count matches RTH schedule.
* ✅ Turnover cap respected in backtest.
* ✅ RTH alignment, weekend exclusion, and spacing validation
* ✅ Coverage tracking integration tests

**Acceptance:** ✅ Cadence matches horizon; turnover bounded.

**Implementation:**
- Enhanced `src/timegpt_v2/forecast/scheduler.py:260-394` with horizon-aligned snapshot presets
- Added `create_horizon_preset()` and `create_snapshot_preset()` functions
- Implemented RTH alignment, turnover control, and trading window filtering
- Created comprehensive test suite in `tests/test_sprint5_snapshot_cadence.py`

---

### Sprint 6 — Walk-forward A/B harness (30m vs 60m) with metrics ✅

* Build a **rolling-origin evaluation**:
  * ✅ Implemented `WalkForwardEvaluator` with configurable train/test/purge periods
  * ✅ Split by weeks/months; purge gaps between folds.
  * ✅ Compute **CRPS**, **IC**, **calibration**, **after-cost Sharpe**, **turnover**, **drawdown**.
  * ✅ Output a **decision report** (per month + overall) comparing 30m vs 60m with the same cost model.
* Costs: half-spread + fee per side + impact bump; configurable.
  * ✅ Implemented `CostConfig` and cost application in trades

**Tests**

* ✅ Metric unit tests on small fixtures (CRPS aggregation, IC by hour, Sharpe with costs).
* ✅ Walk-forward smoke using a tiny dataset runs end-to-end.
* ✅ Cost configuration and validation tests
* ✅ Decision report generation and file output tests

**Acceptance:** ✅ Report clearly picks a horizon on OOS metrics.

**Implementation:**
- Created `src/timegpt_v2/eval/walkforward.py` with comprehensive walk-forward framework
- Enhanced `src/timegpt_v2/eval/metrics_forecast.py` with CRPS and Information Coefficient functions
- Implemented `WalkForwardEvaluator`, `HorizonResult`, `CostConfig`, and `WalkForwardConfig` classes
- Added configurable trading costs with half-spread, commission, and market impact
- Built decision report generator with horizon recommendations and robustness analysis
- Created comprehensive test suite in `tests/test_sprint6_walkforward_ab.py`

---

### Sprint 7 — Compact hyperparameter tuner (bounded, robust) ✅

* Implement a **small** search (random or Bayesian) over:

  * `k_sigma` ∈ [0.4, 1.2]
  * `(tp_sigma, sl_sigma)` ∈ {(2.0,2.0), (2.5,2.0), (3.0,2.0)}
  * `uncertainty_cut` ∈ [0.70, 0.95] percentile of `(q75−q25)/σ_h`
  * Cadence ∈ {every 30m, every 60m}
* Use the walk-forward harness; objective = **after-cost Sharpe** with constraints on turnover and drawdown.
* Emit the **best config** and its **robustness** (variance across folds).

**Tests**

* ✅ Dry-run optimizer with a small budget; verify it returns a config and logs metrics.

**Acceptance:** ✅ A tuned yet **simple** config selected by OOS KPIs.

**Implementation:**
- Created `src/timegpt_v2/eval/hyperparameter.py` with comprehensive hyperparameter optimization framework
- Implemented `CompactHyperparameterTuner`, `HyperparameterConfig`, and `ParameterSet` classes
- Added bounded search over key parameters with validation constraints
- Integrated with walk-forward evaluation for robust OOS performance assessment
- Built comprehensive test suite in `tests/test_sprint7_hyperparameter.py` with 13 test cases passing

---

### Sprint 8 — (Optional) Cross-sectional variant ✅

* Add multi-ticker support: rank by q50; long top decile, short bottom decile; dollar-neutral caps.
* Evaluate **cross-sectional IC**, after-cost Sharpe, and turnover.

**Tests**

* ✅ Cross-sectional allocation sums to ~0 beta; per-name caps honored.

**Acceptance:** ✅ Evidence whether dispersion harvesting improves robustness.

**Implementation:**
- Created `src/timegpt_v2/eval/cross_sectional.py` with comprehensive cross-sectional trading strategy
- Implemented `CrossSectionalStrategy`, `CrossSectionalConfig`, and `CrossSectionalResult` classes
- Added multi-ticker ranking by q50 forecast with top/bottom decile selection
- Built dollar-neutral position sizing with equal weight and volatility scaling options
- Implemented cross-sectional IC calculation and performance metrics
- Added risk management with leverage constraints and position weight limits
- Created comprehensive test suite in `tests/test_sprint8_cross_sectional.py` with 16 test cases passing

---

## Tests Summary (must-pass)

* **Freq tests:** minute regularity; `X_df` horizon length correctness.
* **Parity tests:** deterministic exogs identical (name + dtype) in `y_df` and `X_df`; no `_x/_y`.
* **σ tests:** σ window aligns with horizon; TP/SL units consistent.
* **Scheduler tests:** snapshot counts for 30m/60m presets inside RTH; turnover cap.
* **Metrics tests:** CRPS aggregation, IC by hour, calibration coverage check, after-cost Sharpe.
* **Harness tests:** walk-forward runs on a tiny fixture; optimizer completes and returns config.

---

## Acceptance Criteria (release gate)

1. No Nixtla horizon warnings when running **60m** with minute data regularized; logs show `freq=min`.
2. 30m vs 60m **decision report** produced with CRPS/IC/Sharpe; chosen horizon wins OOS after costs.
3. Trading rules use **σ_h** consistently; **TP ≥ SL** (or EV exits) by default.
4. Deterministic exogs parity guaranteed; preflight catches violations.
5. Snapshot cadence matches horizon; turnover within caps.
6. KPIs exceed thresholds listed above; monthly stability acceptable.
7. Tests pass; docs updated (readme “Horizon/σ alignment”, “Deterministic exogs in X_df”, “Walk-forward metrics”).

---

## Risks & Mitigations

* **Overfitting via search:** keep search space **small**, use **walk-forward OOS** and **costs** in the objective.
* **Data irregularities:** enforce minute regularity and RTH; warn on gaps > N minutes.
* **Cost underestimation:** stress costs ±50% and ensure Sharpe remains positive.
* **Parameter drift:** report month-by-month metrics; reject configs with unstable OOS.

---

## What Claude Code should print after each sprint

* `✅ Sprint 0 done` … `✅ Sprint 7 done`
* Final: `✅ Horizon/model aligned, freq=min enforced, σ-consistent trading, parity validated, 30m vs 60m report generated with OOS KPIs.`

---

