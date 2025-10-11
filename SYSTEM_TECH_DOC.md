# TimeGPT v2 — System Technical Documentation
**Version:** Sprint 8 (Exogenous Features)
**Last Updated:** 2025-10-11T10:57:00Z
**Version:** Sprint 7 (Portfolio, Cost Sensitivity, OOS)
**Last Updated:** 2025-10-11T07:58:00Z
**Version:** Sprint 5 (Quantile-Aware Trading Rules)
**Last Updated:** 2025-10-11T07:58:00Z
**Version:** Sprint 4 (Calibration + Coverage Diagnostics)
**Last Updated:** 2025-10-11T07:28:00Z
**Version:** Sprint 3 (Framing + TimeGPT Client)
**Last Updated:** 2025-10-11T07:28:00Z
**Version:** Sprint 2 (Leakage-Safe Feature Engineering)
**Last Updated:** 2025-10-11T05:36:00Z
**Version:** Sprint 1 (GCS Reader + Data Quality Gates)
**Last Updated:** 2025-10-11T03:40:00Z
**Maintainer:** Kilo Code
**Git Reference:** _pending commit_

---

## 1. System Overview

TimeGPT v2 is an intraday forecasting and trading pipeline for equity (or ETF) universes. The stack ingests minute-level OHLCV data, engineers leakage-safe features, submits histories to Nixtla’s TimeGPT API, evaluates forecast quality, and drives trading backtests with risk-aware rules. Sprint 5 introduces configuration sweeps that explore calendaring, scaling, quantile, and calibration variants automatically.

High-level workflow:

1. **Ingestion / Validation (`check-data`)**  
   - Source: GCS parquet (bucket/template in `configs/data.yaml`).  
   - Validated with `DataQualityChecker` against policies in `configs/dq_policy.yaml`.  
   - Clean parquet stored under `artifacts/runs/<run_id>/validation/clean.parquet`.  
   - `meta.json` tracks status, timestamps, and offending rows.

2. **Feature Engineering (`build-features`)**  
   - Leakage-safe features derived using `FeatureContext` & `build_feature_matrix`.  
   - Targets: log returns, basis points, volatility-z (all persisted for reversible scaling).  
   - Feature parquet: `artifacts/runs/<run_id>/features/features.parquet`.  
   - NaN culling: rows with >10 % missing truncated.

3. **Forecasting (`forecast`)**  
   - Scheduler generates ET snapshots via presets (`configs/forecast.yaml`).  
   - Optionally skips event dates (FOMC, CPI) and reuses cache for repeated runs.  
   - Target scaling configured per mode; quantiles now support `[0.1, 0.25, 0.5, 0.75, 0.9]` plus level bands `[50, 80, 95]`.  
   - Calibration metadata stored at `models/calibration.json`, enabling affine or isotonic corrections on quantile outputs.  
   - Outputs: `forecasts/quantiles.csv` containing quantile columns, `y_true`, metadata tracked in `meta.json`.

4. **Backtesting (`backtest`)**  
   - Uses trading configs (`configs/trading.yaml`).  
   - Simulator (position sizing based on forecast returns & sigma thresholds) executes trades with cost model.  
   - Artifacts: `trades/bt_trades.csv`, `eval/bt_summary.csv`, stress-phase summaries.  
   - Log files maintained in `logs/backtest.log`.

5. **Evaluation (`evaluate`)**  
   - Computes per-symbol forecast diagnostics: rMAE, rRMSE, PIT coverage, interval stats.  
   - Trading metrics (Sharpe, max drawdown, hit rate) aggregated from backtest results.  
   - Gates: median PIT tolerance, rMAE thresholds, cost-sensitivity checks.  
   - `eval/forecast_diagnostics.csv`, `eval/pit_reliability.csv`, and trading summaries persisted.  
   - `meta.json` updated with gate status and key aggregate metrics.

6. **Reporting / Sweeps**  
   - `report` composes human-readable summaries via `reports.builder`.  
   - `sweep` previously handled trading parameter grids; Sprint 5 extends it to drive forecast configuration sweeps.

---

## 2. Sprint 5 Additions

### 2.1 Forecast Grid Engine
- **Module:** [`src/timegpt_v2/forecast/sweep.py`](src/timegpt_v2/forecast/sweep.py:1)  
- **Key Classes:**  
  - `ForecastGridSpec`: parses sweep YAML spec -> cartesian grid (snapshot preset, horizon, quantile set, levels, target mode, calibration method).  
  - `ForecastGridSearch`: materialises override configs, optionally executes forecast/backtest/evaluate, collates metrics, computes composite score.

- **Outputs:**  
  - `eval/grid/forecast_grid/plan.csv`: full plan with per-run metadata & metrics.  
  - `eval/grid/scoreboard.csv`: ranked subset (Sharpe × (1 − |PIT − 0.5|)).  
  - Per-combo config directories under `eval/grid/forecast_grid/<combo_hash>/config/forecast.yaml`.

- **Execution Hooks:**  
  - CLI: `python -m timegpt_v2.cli sweep --forecast-grid configs/forecast_grid.yaml …`.  
  - Makefile targets:  
    - `make forecast-grid-plan` → plan-only.  
    - `make forecast-grid` → full execution, optionally `--reuse-baseline` to reuse features/validation.

- **Config Template:** [`configs/forecast_grid.yaml`](configs/forecast_grid.yaml:1) demonstrates baseline vs liquidity presets, multiple horizons, quantile sets, target modes, calibration choices.

### 2.2 Documentation Updates
- [`docs/FORECASTING.md`](docs/FORECASTING.md:1) now covers:
  - Expanded quantile/level policy.
  - Calibration metadata persistence.
  - Forecast grid sweep workflow, gating metrics, automation via Make targets.

### 2.3 Tests
- [`tests/test_forecast_sweep.py`](tests/test_forecast_sweep.py:1) validates:
  - Grid spec cartesian product counts.
  - Plan-only override generation (no command invocations).
  - Metric aggregation/composite scoring.
  - Scoreboard ranking/rank column integrity.

---

## 3. Sprint 4 Additions

### 3.1 Calibration Diagnostics
- **Module:** [`src/timegpt_v2/eval/calibration.py`](src/timegpt_v2/eval/calibration.py:1)
- **Key Functions:**
  - `widen_intervals()`: Post-hoc quantile widening by multiplicative factor to increase coverage toward nominal levels.
  - `split_conformal()`: Simple split-conformal prediction using historical residuals to adaptively widen intervals.
  - `generate_coverage_report()`: Generates detailed coverage statistics per symbol and forecast snapshot.
- **Features:** Reuses cached forecasts from Sprint 3; no additional API calls required. Addresses under-coverage issues with configurable widening methods.

### 3.2 Evaluation Enhancements
- **Module:** [`src/timegpt_v2/cli.py`](src/timegpt_v2/cli.py:1)
- **Key Changes:** `evaluate` command now generates `eval/coverage_report.csv` with per-symbol, per-snapshot coverage metrics alongside existing forecast diagnostics.
- **Outputs:** Enhanced evaluation with coverage deltas, pinball loss, PIT coverage, and interval width statistics.

### 3.3 Tests
- [`tests/test_calibration.py`](tests/test_calibration.py:1): Added comprehensive tests for `widen_intervals`, `split_conformal`, and `generate_coverage_report` functions, ensuring correct widening behavior and coverage reporting.

---

## 4. Sprint 3 Additions
---

## 4. Sprint 3 Additions

### 4.1 Framing Layer
- **Module:** [`src/timegpt_v2/framing/build_payloads.py`](src/timegpt_v2/framing/build_payloads.py:1)
- **Key Functions:**
  - `build_y_df()`: Constructs Y_df with rolling history window, forward-fill gaps, deterministic features, exogenous consistency.
  - `build_x_df_for_horizon()`: Builds X_df with future deterministic features and static features from snapshot.
- **Features:** Supports leakage-safe Y_df/X_df construction per snapshot with configurable history windows and horizons.

### 4.2 TimeGPT Client Enhancements
- **Module:** [`src/timegpt_v2/forecast/timegpt_client.py`](src/timegpt_v2/forecast/timegpt_client.py:1)
- **Key Classes:**
  - `TimeGPTClient`: Batch multi-series forecasting with caching, budget management, offline/online modes.
  - `TimeGPTConfig`: Configuration for API calls, quantiles, horizons.
- **Features:** SHA256-based caching, API budget tracking, deterministic backend for testing.

### 4.3 CLI Forecast Command
- **Module:** [`src/timegpt_v2/cli.py`](src/timegpt_v2/cli.py:1)
- **Key Command:** `forecast` with `--api-mode offline|online`, budget enforcement, cache integration.
- **Outputs:** Quantiles CSV, per-snapshot JSON with cache keys, logs with API call counts.

### 4.4 Tests
- [`tests/test_framing.py`](tests/test_framing.py:1): Validates Y_df/X_df construction, gap filling, feature inclusion.
- [`tests/test_forecast_client.py`](tests/test_forecast_client.py:1): Tests batch forecasting, caching, budget management.

---

## 5. Sprint 5 Additions

### 5.1 Quantile-Aware Trading Rules
- **Module:** [`src/timegpt_v2/trading/rules.py`](src/timegpt_v2/trading/rules.py:1)
- **Key Changes:**
  - Added EV(after-cost) > 0 check: Ensures expected value after costs is positive before entering trades.
  - Added uncertainty suppression: Wide quantile intervals (q75 - q25 > 2 * sigma_5m) suppress entry to avoid high-uncertainty trades.
  - Fixed position sizing to 1.0 unit exposure per trade.
- **Features:** Reuses cached forecasts; no additional API calls. Enhances entry logic with risk-aware filters.

### 5.2 Trading Costs
- **Module:** [`src/timegpt_v2/trading/costs.py`](src/timegpt_v2/trading/costs.py:1)
- **Key Functions:** `get_costs_bps()` calculates fees + half-spread in basis points.
- **No changes:** Already implemented from prior sprints.

### 5.3 Tests
- [`tests/test_trading_rules.py`](tests/test_trading_rules.py:1): Updated tests for fixed position size (1.0/-1.0), added tests for uncertainty suppression and EV check.

### 5.4 Documentation
- [`docs/TRADING_RULES.md`](docs/TRADING_RULES.md:1): Updated with new entry conditions, EV check, uncertainty suppression, fixed sizing.

---

## 7. Sprint 7 Additions

### 7.1 Portfolio Evaluation Metrics
- **Module:** [`src/timegpt_v2/eval/metrics_trading.py`](src/timegpt_v2/eval/metrics_trading.py:1)
- **Key Functions:**
  - `portfolio_sharpe()`: Computes Sharpe ratio across all symbols using equal-weighted daily returns.
  - `portfolio_max_drawdown()`: Calculates portfolio max drawdown from daily returns.
  - `portfolio_hit_rate()`: Aggregates hit rate across trades.
  - `portfolio_total_pnl()`: Sums total net P&L.
  - `per_symbol_metrics()`: Computes per-symbol KPIs (trade count, P&L, hit rate, Sharpe, max DD).
- **Features:** Phase-filtered metrics for in-sample, OOS, stress periods.

### 7.2 OOS Evaluation
- **Module:** [`src/timegpt_v2/cli.py`](src/timegpt_v2/cli.py:1)
- **Key Outputs:** `eval/oos_summary.csv` with OOS-specific metrics (total trades, P&L, Sharpe, max DD, hit rate).
- **Gates:** OOS Sharpe ≥ 0.5, hit rate ≥ 48%, net P&L > 0 enforced in evaluate command.
- **No API Calls:** Reuses cached forecasts and backtest results.

### 7.3 Cost Sensitivity Analysis
- **Module:** [`src/timegpt_v2/backtest/aggregation.py`](src/timegpt_v2/backtest/aggregation.py:1)
- **Key Function:** `compute_cost_scenarios()` evaluates P&L, hit rate, Sharpe at 1.0×, 1.5×, 2.0× costs.
- **Output:** `eval/cost_sensitivity.csv` with per-multiplier metrics.
- **Gate:** Net P&L must remain positive at 1.5× costs.

### 7.4 CLI Evaluate Enhancements
- **Module:** [`src/timegpt_v2/cli.py`](src/timegpt_v2/cli.py:1)
- **Key Changes:** Extended evaluate command to output portfolio metrics per phase, per-symbol metrics, and OOS summary.
- **Outputs:** `eval/portfolio_metrics.csv`, `eval/per_symbol_metrics.csv`, `eval/oos_summary.csv`.

### 7.5 Tests
- [`tests/test_eval_metrics.py`](tests/test_eval_metrics.py:1): Added tests for portfolio functions, per-symbol metrics, and phase filtering.

### 7.6 Documentation
- [`docs/EVALUATION.md`](docs/EVALUATION.md:1): Added sections for Portfolio Evaluation, OOS Evaluation, Cost Sensitivity Analysis.

---
## 6. Sprint 8 Additions

### 3.1 Backend Enforcement and .env Loader
- **Module:** [`src/timegpt_v2/cli.py`](src/timegpt_v2/cli.py:1)
- **Key Changes:**
  - Auto-load secrets from `.env` at CLI import-time using `python-dotenv` with fallback parser.
  - Enforce live Nixtla backend: forecast step rejects missing/None backend, no stub fallback.
  - Added backend-mode INFO log for audit in forecast logs.
- **Config:** Set `backend: nixtla` in [`configs/forecast.yaml`](configs/forecast.yaml:20).
- **Operational Impact:** Pre-run requirement for `TIMEGPT_API_KEY` or `NIXTLA_API_KEY` in `.env`; fails fast on missing key or init failure.

### 3.2 Calibration Fit/Apply Workflow and Embargo
- **Module:** [`src/timegpt_v2/eval/calibration.py`](src/timegpt_v2/eval/calibration.py:1)
- **Key Classes:**
  - `ForecastCalibrator.fit()`: Fits affine/isotonic models from historical forecasts vs y_true.
  - `ForecastCalibrator.apply()`: Applies corrections post-inverse scaling, enforces quantile monotonicity (q10≤q25≤q50≤q75≤q90).
  - `CalibrationModel`: Persists to `models/calibration.json`.
- **Embargo:** Calibration window ends ≥1 trading day before evaluation period start.
- **Fallback:** Conformal widening if validation PIT deviation >0.03.
- **Interoperability:** Forecast consumes `models/calibration.json` if present; evaluate reports pre/post coverage deltas.

### 3.3 Snapshot Policy and Horizon/Label Alignment
- **Module:** [`src/timegpt_v2/forecast/scheduler.py`](src/timegpt_v2/forecast/scheduler.py:1), [`src/timegpt_v2/forecast/scaling.py`](src/timegpt_v2/forecast/scaling.py:1)
- **Key Changes:**
  - Set `snapshot_preset: liquidity_profile` (5/day) with `skip_events: [event_fomc, event_cpi]`.
  - Horizon/label alignment: Added `target_log_return_15m` in [`src/timegpt_v2/fe/base_features.py`](src/timegpt_v2/fe/base_features.py:1); updated `TargetScaler.target_column()` for `target.mode: log_return_15m`.
- **Verification:** Meta lists 5× snapshots/day; y_df ds equals snapshot_utc, label_timestamp equals forecast_ts.

### 3.4 Monitoring Additions (Monotonicity, Backend-Mode)
- **Module:** [`src/timegpt_v2/cli.py`](src/timegpt_v2/cli.py:495), [`src/timegpt_v2/eval/calibration.py`](src/timegpt_v2/eval/calibration.py:234)
- **Key Changes:**
  - Backend-mode assertion and quantile monotonicity validator during forecast.
  - Violations persisted to logs; evaluate logs calibration gate pass/fail.
  - Insufficient history check: skips snapshots with <25 samples per symbol to prevent TimeGPT API errors.

---

## 8. Sprint 8 Additions

### 8.1 Exogenous Features Integration
- **Module:** [`src/timegpt_v2/framing/build_payloads.py`](src/timegpt_v2/framing/build_payloads.py:1)
- **Key Changes:**
  - Added `EXOGENOUS_FEATURE_COLUMNS` list: `["spy_ret_1m", "spy_vol_30m", "regime_high_vol", "regime_high_dispersion", "event_earnings", "event_fomc", "event_cpi"]`.
  - Modified `build_y_df()` to include exogenous columns in historical Y_df for TimeGPT.
  - Modified `build_x_df_for_horizon()` to project exogenous values forward at snapshot levels (constant propagation).
- **Features:** Enables market regime and event information in forecasting without lookahead bias; exogenous values held constant in future projections.

### 8.2 TimeGPT Client Exogenous Support
- **Module:** [`src/timegpt_v2/forecast/timegpt_client.py`](src/timegpt_v2/forecast/timegpt_client.py:1)
- **Key Changes:**
  - Updated `TimeGPTBackend` protocol to accept `hist_exog_list` parameter.
  - Modified `NixtlaTimeGPTBackend` to pass `hist_exog_list` to Nixtla SDK forecast calls.
  - Updated `TimeGPTClient.forecast()` to accept and forward `hist_exog_list`.
- **Features:** Passes exogenous column specifications to TimeGPT API for improved signal quality.

### 8.3 CLI Forecast Command Exogenous Usage
- **Module:** [`src/timegpt_v2/cli.py`](src/timegpt_v2/cli.py:1)
- **Key Changes:** Forecast command now passes `EXOGENOUS_FEATURE_COLUMNS` as `hist_exog_list` to TimeGPT client.
- **Features:** Minimal API calls; re-forecasts only when exogenous features are added or updated, respecting budget constraints.

### 8.4 Tests
- [`tests/test_framing.py`](tests/test_framing.py:1): Updated to validate exogenous column inclusion in Y_df and X_df.
- [`tests/test_forecast_client.py`](tests/test_forecast_client.py:1): Added tests for exogenous list passing to backend.

### 8.5 Documentation
- [`docs/FORECASTING.md`](docs/FORECASTING.md:1): Added section on exogenous features integration, hist_exog_list requirements, and alignment.

---

## 5. Module / API Map

| Area | Module | Key Functions / Classes |
|------|--------|-------------------------|
| CLI | [`src/timegpt_v2/cli.py`](src/timegpt_v2/cli.py:1) | Typer commands (`check-data`, `build-features`, `forecast`, `backtest`, `evaluate`, `report`, `sweep`, `calibrate`). Sweep now accepts `--forecast-grid`, `--plan-only`, `--reuse-baseline`, `--baseline-run`. |
| Feature Engineering | [`src/timegpt_v2/fe/base_features.py`](src/timegpt_v2/fe/base_features.py:1) | `build_feature_matrix`, expanded windows (ret_30m, rv_30m), skew/kurtosis, VWAP trend, volume percentile, signed volume, vol_ewm_15m, lagged market context. |
| Framing | [`src/timegpt_v2/framing/build_payloads.py`](src/timegpt_v2/framing/build_payloads.py:1) | `build_y_df`, `build_x_df_for_horizon`, leakage-safe Y_df/X_df construction with rolling windows and forward-fill. |
| Forecast Client | [`src/timegpt_v2/forecast/timegpt_client.py`](src/timegpt_v2/forecast/timegpt_client.py:1) | Nixtla backend wrapper, batch multi-series forecasting, caching, budget management, offline/online modes. |
| Calibration | [`src/timegpt_v2/eval/calibration.py`](src/timegpt_v2/eval/calibration.py:1) | `ForecastCalibrator`, `CalibrationModel`, affine + isotonic support, persistence, monotonic projection. |
| Scaling | [`src/timegpt_v2/forecast/scaling.py`](src/timegpt_v2/forecast/scaling.py:1) | `TargetScaler`, reversible scaling across log/bp/z/log_return_15m modes. |
| Scheduler | [`src/timegpt_v2/forecast/scheduler.py`](src/timegpt_v2/forecast/scheduler.py:1) | Snapshot presets, skip_dates, active windows, quota enforcement. |
| Backtest | [`src/timegpt_v2/backtest/simulator.py`](src/timegpt_v2/backtest/simulator.py:1), [`src/timegpt_v2/trading/rules.py`](src/timegpt_v2/trading/rules.py:1), [`src/timegpt_v2/trading/costs.py`](src/timegpt_v2/trading/costs.py:1) | Quantile-aware entry with EV and uncertainty filters, fixed 1.0 unit sizing, time/price exits, costs. |
| Forecast Sweeps | [`src/timegpt_v2/forecast/sweep.py`](src/timegpt_v2/forecast/sweep.py:1) | See § 2.1. |

---

## 6. Config Schema

### 4.1 `configs/forecast.yaml`
Key fields post Sprint 5:
```yaml
snapshot_preset: baseline
snapshot_presets:
  - name: baseline
    times: ["10:00", "14:30"]
    active_windows:
      - {start: "09:45", end: "15:30"}
    max_snapshots_per_day: 2
    horizon_min: 15
  - name: liquidity_profile
    times: ["09:45", "10:15", "12:00", "14:00", "15:15"]
    active_windows:
      - {start: "09:40", end: "15:45"}
    max_snapshots_per_day: 5
    horizon_min: 15
max_snapshots_total: null
quantiles: [0.1, 0.25, 0.5, 0.75, 0.9]
levels: [50, 80, 95]
target:
  mode: log_return
  volatility_column: vol_ewm_60m
  bp_factor: 10000
calibration:
  method: affine
  min_samples: 120
  calibration_window_days: 30
  model_path: models/calibration.json
skip_events: [event_fomc, event_cpi]
```

### 4.2 `configs/forecast_grid.yaml`
Defines sweep search space:
```yaml
snapshot_presets: [baseline, liquidity_profile]
horizons: [10, 15, 30]
quantile_sets:
  - [0.1, 0.25, 0.5, 0.75, 0.9]
  - [0.05, 0.25, 0.5, 0.75, 0.95]
level_sets:
  - [50, 80, 95]
  - [60, 90]
target_modes: [log_return, volatility_z]
calibration_methods: [none, affine, isotonic]
```

### 4.3 `Makefile` Targets
- `make sweep` – trading parameter grid (unchanged).  
- `make forecast-grid` – execute forecast sweep with baseline reuse.  
- `make forecast-grid-plan` – plan-only preview.

---

## 7. Determinism & Logging

- Forecast sweeps seed RNG per combination via MD5-derived integers to maintain reproducibility.
- Execution logs: `logs/forecast.log`, `logs/backtest.log`, `logs/sweep.log`.
- Each run’s `meta.json` records steps, configs, timestamps, calibration metadata.
- Cache keys incorporate quantiles/levels/model, ensuring correct reuse across sweeps.

---

## 8. Error Handling & Guardrails

- CLI commands raise `typer.BadParameter` for misconfigured grids, missing baseline artifacts, or missing data.
- Forecast sweep ensures scoreboard only written when composite scores present; otherwise clean removal.
- Tests validate Scoreboard ranking, plan generation, and metric aggregation.

---

## 9. Outstanding Tasks

- Extend sweep spec to cover backend model selection (TimeGPT-1 vs long horizon).
- Integrate weekly cron automation (e.g., GitHub Action or Airflow DAG) referencing Make targets.
- Monitor pandas future warnings (fillna downcasting) and patch to use `infer_objects(copy=False)`.

---

## 10. Usage Examples

```bash
# Plan forecast sweep without execution
make forecast-grid-plan

# Execute with baseline reuse (baseline run_id must have features/validation)
make forecast-grid BASELINE_RUN=prod_run

# Direct CLI invocation
python -m timegpt_v2.cli sweep \
  --run-id exp_fg \
  --config-dir configs \
  --forecast-grid configs/forecast_grid.yaml \
  --execute \
  --reuse-baseline \
  --baseline-run prod_run
```

---

## 10. Runbook: Secrets Handling and Incident Response

### 10.1 Secrets Handling
- **Environment Variables:** TIMEGPT_API_KEY or NIXTLA_API_KEY must be set in `.env` at repo root.
- **Autoload:** Loaded at CLI import via python-dotenv; fails fast if missing.
- **Logging:** Never echo secrets in logs; use placeholders like `[REDACTED]` in debug output.
- **Rotation:** Update `.env` and restart processes; no secrets in version control.

### 10.2 Incident Response (Backend Failures)
- **Detection:** Forecast step logs backend init status; API call failures logged with error codes.
- **Response:**
  - Rate limit exceeded: Wait with exponential backoff (built into Nixtla SDK).
  - Invalid key: Abort immediately, log "Invalid API key" without exposing key.
  - Network timeout: Retry up to 3 times, then abort.
  - Service unavailable: Log incident, alert on-call, switch to manual mode if critical.
- **Recovery:** Re-run with valid credentials; check Nixtla status page for outages.
- **Prevention:** Monitor API usage; implement quota alerts.
| 2025-10-11 | Sprint 8 implementation: Exogenous Features | Added exogenous features integration with `EXOGENOUS_FEATURE_COLUMNS` list; modified `build_y_df` and `build_x_df_for_horizon` to include exogenous columns in Y_df and project forward in X_df; updated TimeGPT client and backend to support `hist_exog_list` parameter; CLI forecast command passes exogenous list; updated docs/FORECASTING.md; all tests pass including exogenous column validation. |
| 2025-10-11 | Sprint 5 implementation: Quantile-Aware Trading Rules | Implemented EV(after-cost) > 0 check and uncertainty suppression using q-spread in trading rules; fixed position sizing to 1.0 unit exposure; updated tests for new logic; updated docs/TRADING_RULES.md; all tests pass including uncertainty suppression and EV validation. |
| 2025-10-11 | Sprint 4 implementation: Calibration + Coverage Diagnostics | Implemented post-hoc quantile widening (`widen_intervals`), split-conformal prediction (`split_conformal`), and coverage reporting (`generate_coverage_report`) in calibration.py; enhanced evaluate command to generate per-symbol, per-snapshot coverage reports; updated docs/EVALUATION.md with calibration methods and gates; all tests pass including widening behavior and coverage calculations. |

---

## 12. Handoffs to Other Teams

### 12.1 Risk & Execution Controls
- **Symbol-level Microstructure:** Review and refine half_spread_ticks and fee_bps in [`configs/trading.yaml`](configs/trading.yaml:1) for each symbol.
- **Circuit Breakers:** Implement position limits, volatility halts, and emergency stops based on drawdown thresholds.

### 12.2 Testing & Analytics Hub
- **Embaroed WFO:** Validate calibration embargo prevents leakage in walk-forward optimization.
- **Ablations:** Run feature importance tests and attribution analysis on promoted configs.
- **Monitoring:** Set up dashboards for PIT coverage, calibration drift, and trading metrics.

### 12.3 Engineering & Platform
- **Calibrate CLI Command:** Deploy and monitor the new `calibrate` command in production pipelines.
- **Monotonic Projection Utility:** Ensure quantile ordering is enforced in all forecast outputs.
- **CI Jobs:** Add automated tests for calibration, monotonicity, and backend enforcement in CI/CD.

---

## 11. Change Log

| 2025-10-11 | Sprint 3 implementation: Framing + TimeGPT client | Implemented framing layer with build_y_df and build_x_df for leakage-safe Y_df and X_df construction per snapshot with rolling history windows and forward-fill gaps; enhanced TimeGPT client with batch multi-series forecasting, SHA256-based caching, API budget management, and offline/online modes; CLI forecast command iterates snapshots, enforces budget, supports cache hits; all tests pass including framing integrity, gap filling, and forecast output validation. |
| 2025-10-11 | Sprint 2 implementation: Leakage-safe feature engineering | Implemented comprehensive feature engineering pipeline with return/volatility features (ret_1m/5m/15m/30m, rv_5m/15m/30m, ATR, Garman-Klass/Parkinson, VWAP, volume norms), deterministic intraday clocks (minute index, Fourier terms, session buckets), and SPY lagged context features (spy_ret_1m, spy_vol_30m, regime flags, event dummies); CLI build-features exports per-symbol parquet with 58 features; all tests pass including no future leakage verification. |
| Date (UTC) | Summary | Details |
|------------|---------|---------|
| 2025-10-11 | Sprint 1 implementation: GCS reader + data quality gates | Implemented GCS parquet ingestion with column aliasing, ET timezone normalization, RTH filtering; added comprehensive data quality checks (schema, monotonicity, price sanity, RTH completeness, adjusted prices, gapless grid with ffill, outliers); CLI check-data command validates and cleans data, writes DQ report; all tests pass for 10 symbols (AAPL/MSFT/NVDA/TSLA/AMZN/GOOGL/META/NFLX/AMD/CRM) from July-Nov 2024. |
| 2025-10-10 | Sprint 8 implementation: Updated docs with backend enforcement, calibration workflow, snapshot policy, horizon alignment, monitoring; added runbook for secrets/incidents, handoffs to other teams; added insufficient history check to prevent TimeGPT API errors.
| 2025-10-10 | Sprint 5 implementation | Added forecast sweep engine, config schema, calibration metadata, documentation, tests, Make targets. |

---

_End of Document_
## 2025-10-10 — Enhancement kickoff: Live TimeGPT enforcement + .env autoload

Version: v2025.10.10.1 (uncommitted)  
Refs: [docs/build.md](docs/build.md), [src/timegpt_v2/cli.py](src/timegpt_v2/cli.py), [configs/forecast.yaml](configs/forecast.yaml)

Summary
- Enforced live Nixtla TimeGPT usage and disabled stub fallback in the forecast pipeline.
- Auto-load secrets from .env at CLI import-time, so TIMEGPT_API_KEY/NIXTLA_API_KEY are available before backend init.
- Prepared comprehensive enhancement plan and test suite in [docs/build.md](docs/build.md).

Changes
- Secrets loading: Added dotenv loader with fallback parser at CLI import in [src/timegpt_v2/cli.py](src/timegpt_v2/cli.py). Effect: .env at repo root is parsed on startup; env vars are available to TimeGPT backend creation.
- Live-backend enforcement: Forecast step now rejects missing/None backend (no stub fallback). Added backend-mode INFO log for audit in [src/timegpt_v2/cli.py](src/timegpt_v2/cli.py).
- Config: Set backend: nixtla in [configs/forecast.yaml](configs/forecast.yaml). Effect: production always targets live API.

Operational impact
- Pre-run requirement: set TIMEGPT_API_KEY (or NIXTLA_API_KEY) in .env at repo root (do not commit secrets).
- Forecast runs will fail fast if the API key is missing or backend init fails, preventing silent stub usage.
- Logs will explicitly state the backend type used per run in artifacts/runs/&lt;run_id&gt;/logs/forecast.log.

Process flow updates
- Ingestion → Features (unchanged).
- Forecast (updated):
  1) Load .env → read TIMEGPT_API_KEY
  2) Initialize Nixtla backend (required)
  3) Build y_df/X_df, call forecast(h=freq='min', quantiles=…)
  4) Inverse scaling to log-return space
  5) (Future sprint) Apply calibration, enforce quantile monotonicity
  6) Persist forecasts with y_true alignment
- Backtest/Evaluate (unchanged today; future sprints add calibration gates and monitors).

Testing and monitoring (added/updated)
- Function tests to be implemented per [docs/build.md](docs/build.md):
  - Env/secrets autoload test
  - Backend enforcement test (no stub)
  - Forecast output integrity (no NaNs, symbol completeness)
- Monitoring:
  - Backend-mode assertion in forecast logs
  - (Future sprint) Quantile monotonicity validator and PIT reliability artifacts

Assumptions
- Nixtla SDK available at runtime (nixtla or nixtlats).
- API quotas adequate for configured snapshot cadence; retry/backoff remains active in backend wrapper.

Next steps
- Execute Sprints 2–7 in [docs/build.md](docs/build.md), starting with calibration fit/apply (embargoed) and snapshot policy upgrade.