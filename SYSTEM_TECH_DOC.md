# TimeGPT v2 — System Technical Documentation
**Version:** Sprint 9 (Reporting Enhancements)
**Last Updated:** 2025-10-11T13:29:00Z
**Git Reference:** 96af4e6
**Maintainer:** Kilo Code

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
    - Target scaling configured per mode; quantiles support `[0.1, 0.25, 0.5, 0.75, 0.9]` plus level bands `[50, 80, 95]`.
    - Calibration metadata stored at `models/calibration.json`, enabling affine or isotonic corrections on quantile outputs.
    - API batching: Dynamic symbol batching and payload partitioning to prevent "payload too large" errors.
    - Exogenous features: Passes market regime and event data (SPY returns, regimes, events) to TimeGPT API.
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

## 2. Sprint 1 Additions

### 2.1 GCS Reader + Data Quality Gates
- **Module:** [`src/timegpt_v2/io/gcs_reader.py`](src/timegpt_v2/io/gcs_reader.py:1), [`src/timegpt_v2/quality/checks.py`](src/timegpt_v2/quality/checks.py:1)
- **Key Functions:**
  - `read_parquet_from_gcs()`: Reads partitioned parquet files with column aliasing and ET timezone normalization.
  - `DataQualityChecker`: Validates schema, monotonicity, price sanity, RTH completeness, adjusted prices, gapless grid, outliers.
- **Features:** Ingests minute-level OHLCV data from GCS bronze layer; enforces data quality gates before feature engineering.
- **Outputs:** Clean parquet stored under `artifacts/runs/<run_id>/validation/clean.parquet`; `meta.json` with status and offending rows.

### 2.2 CLI Check-Data Command
- **Module:** [`src/timegpt_v2/cli.py`](src/timegpt_v2/cli.py:1)
- **Key Command:** `check-data` validates and cleans data, writes DQ report.
- **Features:** Filters to regular session bars between 09:30–16:00 ET.

### 2.3 Tests
- [`tests/test_io_gcs_reader.py`](tests/test_io_gcs_reader.py:1): Validates GCS reading, column aliasing, timezone handling.
- [`tests/test_quality_checks.py`](tests/test_quality_checks.py:1): Tests DQ checks for 10 symbols (AAPL/MSFT/NVDA/TSLA/AMZN/GOOGL/META/NFLX/AMD/CRM) from July-Nov 2024.

---

## 3. Sprint 2 Additions

### 3.1 Leakage-Safe Feature Engineering
- **Module:** [`src/timegpt_v2/fe/base_features.py`](src/timegpt_v2/fe/base_features.py:1), [`src/timegpt_v2/fe/context.py`](src/timegpt_v2/fe/context.py:1)
- **Key Functions:**
  - `build_feature_matrix()`: Computes return/volatility features (ret_1m/5m/15m/30m, rv_5m/15m/30m, ATR, Garman-Klass/Parkinson), deterministic intraday clocks (minute index, Fourier terms, session buckets), SPY lagged context (spy_ret_1m, spy_vol_30m, regime flags, event dummies).
- **Features:** 58 features total; no future leakage via FeatureContext ensuring targets are not used in predictors.
- **Outputs:** Per-symbol parquet under `artifacts/runs/<run_id>/features/features.parquet`; NaN culling for rows >10% missing.

### 3.2 CLI Build-Features Command
- **Module:** [`src/timegpt_v2/cli.py`](src/timegpt_v2/cli.py:1)
- **Key Command:** `build-features` exports features with leakage-safe construction.

### 3.3 Tests
- [`tests/test_fe.py`](tests/test_fe.py:1): Validates no future leakage, feature completeness, NaN handling.

---

## 4. Sprint 3 Additions

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

## 5. Sprint 4 Additions

### 5.1 Calibration Diagnostics
- **Module:** [`src/timegpt_v2/eval/calibration.py`](src/timegpt_v2/eval/calibration.py:1)
- **Key Functions:**
  - `widen_intervals()`: Post-hoc quantile widening by multiplicative factor to increase coverage toward nominal levels.
  - `split_conformal()`: Simple split-conformal prediction using historical residuals to adaptively widen intervals.
  - `generate_coverage_report()`: Generates detailed coverage statistics per symbol and forecast snapshot.
- **Features:** Reuses cached forecasts; no additional API calls. Addresses under-coverage with configurable widening.

### 5.2 Evaluation Enhancements
- **Module:** [`src/timegpt_v2/cli.py`](src/timegpt_v2/cli.py:1)
- **Key Changes:** `evaluate` generates `eval/coverage_report.csv` with per-symbol coverage metrics.
- **Outputs:** Coverage deltas, pinball loss, PIT coverage, interval width.

### 5.3 Tests
- [`tests/test_calibration.py`](tests/test_calibration.py:1): Tests widening, conformal, coverage reporting.

---

## 6. Sprint 5 Additions

### 6.1 Forecast Grid Engine
- **Module:** [`src/timegpt_v2/forecast/sweep.py`](src/timegpt_v2/forecast/sweep.py:1)
- **Key Classes:**
  - `ForecastGridSpec`: Parses sweep YAML → cartesian grid.
  - `ForecastGridSearch`: Executes sweeps, collates metrics, computes scores.
- **Outputs:** Plan CSV, scoreboard CSV, per-combo configs.

### 6.2 Quantile-Aware Trading Rules
- **Module:** [`src/timegpt_v2/trading/rules.py`](src/timegpt_v2/trading/rules.py:1)
- **Key Changes:** EV(after-cost) > 0 check, uncertainty suppression, fixed 1.0 sizing.

### 6.3 Tests & Docs
- [`tests/test_forecast_sweep.py`](tests/test_forecast_sweep.py:1), [`tests/test_trading_rules.py`](tests/test_trading_rules.py:1).
- [`docs/FORECASTING.md`](docs/FORECASTING.md:1), [`docs/TRADING_RULES.md`](docs/TRADING_RULES.md:1).

---

## 7. Sprint 6 Additions

### 7.1 Backtester + Param Sweep
- **Module:** [`src/timegpt_v2/backtest/simulator.py`](src/timegpt_v2/backtest/simulator.py:1), [`src/timegpt_v2/backtest/grid.py`](src/timegpt_v2/backtest/grid.py:1)
- **Key Features:** Position sizing, cost model, aggregation; trading parameter grids (k_sigma, s_stop, s_take).
- **Outputs:** Trades CSV, summary CSV, logs.

### 7.2 Tests
- [`tests/test_backtest_simulator.py`](tests/test_backtest_simulator.py:1), [`tests/test_backtest_grid.py`](tests/test_backtest_grid.py:1).

---

## 8. Sprint 7 Additions

### 8.1 Portfolio Evaluation Metrics
- **Module:** [`src/timegpt_v2/eval/metrics_trading.py`](src/timegpt_v2/eval/metrics_trading.py:1)
- **Key Functions:** `portfolio_sharpe()`, `portfolio_max_drawdown()`, etc.; phase-filtered for IS/OOS/stress.

### 8.2 OOS Evaluation & Cost Sensitivity
- **Outputs:** `eval/oos_summary.csv`, `eval/cost_sensitivity.csv`.
- **Gates:** OOS Sharpe ≥ 0.5, positive P&L at 1.5× costs.

### 8.3 Tests & Docs
- [`tests/test_eval_metrics.py`](tests/test_eval_metrics.py:1).
- [`docs/EVALUATION.md`](docs/EVALUATION.md:1).

---

## 9. Sprint 8 Additions

### 9.1 Backend Enforcement & Calibration Workflow
- **Module:** [`src/timegpt_v2/cli.py`](src/timegpt_v2/cli.py:1), [`src/timegpt_v2/eval/calibration.py`](src/timegpt_v2/eval/calibration.py:1)
- **Key Features:** .env autoload, live Nixtla enforcement, calibration fit/apply with embargo, monotonicity enforcement.

### 9.2 Exogenous Features
- **Modules:**
    - [`src/timegpt_v2/config/model.py`](src/timegpt_v2/config/model.py:1)
    - [`src/timegpt_v2/config/loader.py`](src/timegpt_v2/config/loader.py:1)
    - [`src/timegpt_v2/forecast/exogenous.py`](src/timegpt_v2/forecast/exogenous.py:1)
    - [`src/timegpt_v2/forecast/timegpt_client.py`](src/timegpt_v2/forecast/timegpt_client.py:1)
- **Key Changes:**
    - Implemented a configurable exogenous feature pipeline controlled by the `exog` section in `configs/forecast.yaml`.
    - The `ForecastExogConfig` dataclass in `src/timegpt_v2/config/model.py` defines the schema for the exogenous feature configuration.
    - The `load_forecast_exog_config` function in `src/timegpt_v2/config/loader.py` loads and validates the configuration.
    - The `exogenous.py` module contains utility functions for normalizing names, selecting available features, and merging them into the history and future dataframes.
    - The `TimeGPTClient` now uses these new utilities to handle exogenous features before calling the Nixtla API.
- **Configuration:**
    - `use_exogs`: Enable/disable the pipeline.
    - `strict_exog`: Fail on missing features if `True`.
    - `hist_exog_list`, `futr_exog_list`: Declare historical and future exogenous features.
    - `exog_name_map`: Map declared names to actual feature column names.
    - `impute_strategy`: Impute missing values using `ffill`, `bfill`, or `zero`.
- **CLI Overrides:**
    - `--use-exogs/--no-exogs`
    - `--strict-exog/--no-strict-exog`
    - `--exog-name-map`
- **Common Errors & Fixes:**
    - **`ValueError: Missing historical/future exogenous features`**: This error occurs in strict mode when a declared exogenous feature is not found in the feature matrix.
        - **Fix:** Ensure that the feature matrix contains all the declared exogenous features, or run in permissive mode (`--no-strict-exog`).
    - **`KeyError: 'feature_name'`**: This can happen if the feature name is misspelled in the configuration or if the feature is not generated by the feature engineering pipeline.
        - **Fix:** Check the spelling of the feature name in `configs/forecast.yaml` and make sure it is present in the output of the `build-features` command.

### 9.3 Snapshot Policy & Monitoring
- Snapshot preset: liquidity_profile (5/day), skip events.
- Monitoring: Backend-mode logs, monotonicity checks, insufficient history skips.

### 9.4 Tests & Docs
- Updated tests for exogenous, calibration.
- [`docs/FORECASTING.md`](docs/FORECASTING.md:1).

---

## 10. Sprint 9 Additions

### 10.1 Comprehensive Reporting
- **Module:** [`src/timegpt_v2/reports/builder.py`](src/timegpt_v2/reports/builder.py:1)
- **Key Function:** `build_report()` generates markdown reports with configuration, forecast KPIs, portfolio metrics, cost sensitivity, grid search results.
- **Features:** Pulls from evaluation CSVs, formats tables, highlights gates (forecast/calibration pass/fail).

### 10.2 CLI Report Command
- **Module:** [`src/timegpt_v2/cli.py`](src/timegpt_v2/cli.py:1)
- **Key Command:** `report` composes human-readable summaries.

### 10.3 README Quickstart & Makefile Demo
- **README.md:** Added quickstart guide, API batching notes.
- **Makefile:** Added `demo` target for full pipeline run.

### 10.4 API Batching & Payload Controls
- **Module:** [`src/timegpt_v2/forecast/timegpt_client.py`](src/timegpt_v2/forecast/timegpt_client.py:1)
- **Key Features:** Dynamic symbol batching, num_partitions for large requests to prevent "payload too large" errors.
---

## 5. Module / API Map

| Area | Module | Key Functions / Classes |
|------|--------|-------------------------|
| Loader | [`src/timegpt_v2/loader/gcs_loader.py`](src/timegpt_v2/loader/gcs_loader.py:1) | `load_history`, `enumerate_month_uris` |
| CLI | [`src/timegpt_v2/cli.py`](src/timegpt_v2/cli.py:1) | Typer commands (`check-data`, `build-features`, `forecast`, `backtest`, `evaluate`, `report`, `sweep`). Sweep accepts `--forecast-grid`, `--plan-only`, `--reuse-baseline`, `--baseline-run`. |
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
| 2025-10-11 | Sprint 9 implementation: Reporting Enhancements | Implemented comprehensive reporting in `reports/builder.py` with `build_report()` generating markdown summaries including configuration, forecast KPIs, portfolio metrics, cost sensitivity, and grid search results; added CLI `report` command; updated README with quickstart guide and API batching notes; added Makefile `demo` target for full pipeline run; all tests pass. |
| 2025-10-11 | API Batching & Payload Controls | Added dynamic symbol batching and payload partitioning in `timegpt_client.py` to prevent "payload too large" errors from TimeGPT API; supports `num_partitions` for large requests. |
| 2025-10-11 | Sprint 8 implementation: Exogenous Features | Added exogenous features integration with `EXOGENOUS_FEATURE_COLUMNS` list; modified `build_y_df` and `build_x_df_for_horizon` to include exogenous columns in Y_df and project forward in X_df; updated TimeGPT client and backend to support `hist_exog_list` parameter; CLI forecast command passes exogenous list; updated docs/FORECASTING.md; all tests pass including exogenous column validation. |
| 2025-10-11 | Sprint 5 implementation: Quantile-Aware Trading Rules | Implemented EV(after-cost) > 0 check and uncertainty suppression using q-spread in trading rules; fixed position sizing to 1.0 unit exposure; updated tests for new logic; updated docs/TRADING_RULES.md; all tests pass including uncertainty suppression and EV validation. |
| 2025-10-11 | Sprint 4 implementation: Calibration + Coverage Diagnostics | Implemented post-hoc quantile widening (`widen_intervals`), split-conformal prediction (`split_conformal`), and coverage reporting (`generate_coverage_report`) in calibration.py; enhanced evaluate command to generate per-symbol, per-snapshot coverage reports; updated docs/EVALUATION.md with calibration methods and gates; all tests pass including widening behavior and coverage calculations. |

--- 

## 11. Sprint 10 Additions

### 11.1 Multi-month GCS Loader
- **Module:** [`src/timegpt_v2/loader/gcs_loader.py`](src/timegpt_v2/loader/gcs_loader.py:1)
- **Key Functions:**
  - `load_history()`: Loads all months intersecting the history window snapshot_ts - rolling_history_days → snapshot_ts. This prevents underfilled history for early snapshots.
- **Features:** Ingests minute-level OHLCV data from GCS bronze layer; automatically pulls all months intersecting the history window.
- **Tests:** [`tests/test_loader_multimonth.py`](tests/test_loader_multimonth.py:1): Validates the multi-month loading logic.

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

## 13. Change Log

| 2025-10-11 | Sprint 3 implementation: Framing + TimeGPT client | Implemented framing layer with build_y_df and build_x_df for leakage-safe Y_df and X_df construction per snapshot with rolling history windows and forward-fill gaps; enhanced TimeGPT client with batch multi-series forecasting, SHA256-based caching, API budget management, and offline/online modes; CLI forecast command iterates snapshots, enforces budget, supports cache hits; all tests pass including framing integrity, gap filling, and forecast output validation. |
| 2025-10-11 | Sprint 2 implementation: Leakage-safe feature engineering | Implemented comprehensive feature engineering pipeline with return/volatility features (ret_1m/5m/15m/30m, rv_5m/15m/30m, ATR, Garman-Klass/Parkinson, VWAP, volume norms), deterministic intraday clocks (minute index, Fourier terms, session buckets), and SPY lagged context features (spy_ret_1m, spy_vol_30m, regime flags, event dummies); CLI build-features exports per-symbol parquet with 58 features; all tests pass including no future leakage verification. |
| Date (UTC) | Summary | Details |
|------------|---------|---------|
| 2025-10-11 | Sprint 1 implementation: GCS reader + data quality gates | Implemented GCS parquet ingestion with column aliasing, ET timezone normalization, RTH filtering; added comprehensive data quality checks (schema, monotonicity, price sanity, RTH completeness, adjusted prices, gapless grid with ffill, outliers); CLI check-data command validates and cleans data, writes DQ report; all tests pass for 10 symbols (AAPL/MSFT/NVDA/TSLA/AMZN/GOOGL/META/NFLX/AMD/CRM) from July-Nov 2024. |
| 2025-10-10 | Sprint 8 implementation: Updated docs with backend enforcement, calibration workflow, snapshot policy, horizon alignment, monitoring; added runbook for secrets/incidents, handoffs to other teams; added insufficient history check to prevent TimeGPT API errors.
| 2025-10-10 | Sprint 5 implementation | Added forecast sweep engine, config schema, calibration metadata, documentation, tests, Make targets. |

---

_End of Document_

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