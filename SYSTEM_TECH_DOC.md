# TimeGPT v2 — System Technical Documentation  
**Version:** Sprint 5 (Forecast Grid & Operational Automation)  
**Last Updated:** 2025-10-10T04:58:25Z  
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

## 3. Module / API Map

| Area | Module | Key Functions / Classes |
|------|--------|-------------------------|
| CLI | [`src/timegpt_v2/cli.py`](src/timegpt_v2/cli.py:1) | Typer commands (`check-data`, `build-features`, `forecast`, `backtest`, `evaluate`, `report`, `sweep`). Sweep now accepts `--forecast-grid`, `--plan-only`, `--reuse-baseline`, `--baseline-run`. |
| Feature Engineering | [`src/timegpt_v2/fe/base_features.py`](src/timegpt_v2/fe/base_features.py:1) | `build_feature_matrix`, expanded windows (ret_30m, rv_30m), skew/kurtosis, VWAP trend, volume percentile, signed volume. |
| Forecast Client | [`src/timegpt_v2/forecast/timegpt_client.py`](src/timegpt_v2/forecast/timegpt_client.py:1) | Nixtla backend wrapper, caching, config (freq, horizon, quantiles, levels, model). |
| Calibration | [`src/timegpt_v2/eval/calibration.py`](src/timegpt_v2/eval/calibration.py:1) | `ForecastCalibrator`, `CalibrationModel`, affine + isotonic support, persistence. |
| Scaling | [`src/timegpt_v2/forecast/scaling.py`](src/timegpt_v2/forecast/scaling.py:1) | `TargetScaler`, reversible scaling across log/bp/z modes. |
| Scheduler | [`src/timegpt_v2/forecast/scheduler.py`](src/timegpt_v2/forecast/scheduler.py:1) | Snapshot presets, skip_dates, active windows, quota enforcement. |
| Backtest | [`src/timegpt_v2/backtest/simulator.py`](src/timegpt_v2/backtest/simulator.py:1), [`src/timegpt_v2/trading/rules.py`](src/timegpt_v2/trading/rules.py:1) | Position sizing on forecast z-scores, EOD stop/time stops, costs. |
| Forecast Sweeps | [`src/timegpt_v2/forecast/sweep.py`](src/timegpt_v2/forecast/sweep.py:1) | See § 2.1. |

---

## 4. Config Schema

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

## 5. Determinism & Logging

- Forecast sweeps seed RNG per combination via MD5-derived integers to maintain reproducibility.
- Execution logs: `logs/forecast.log`, `logs/backtest.log`, `logs/sweep.log`.
- Each run’s `meta.json` records steps, configs, timestamps, calibration metadata.
- Cache keys incorporate quantiles/levels/model, ensuring correct reuse across sweeps.

---

## 6. Error Handling & Guardrails

- CLI commands raise `typer.BadParameter` for misconfigured grids, missing baseline artifacts, or missing data.
- Forecast sweep ensures scoreboard only written when composite scores present; otherwise clean removal.
- Tests validate Scoreboard ranking, plan generation, and metric aggregation.

---

## 7. Outstanding Tasks

- Extend sweep spec to cover backend model selection (TimeGPT-1 vs long horizon).
- Integrate weekly cron automation (e.g., GitHub Action or Airflow DAG) referencing Make targets.
- Monitor pandas future warnings (fillna downcasting) and patch to use `infer_objects(copy=False)`.

---

## 8. Usage Examples

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

## 9. Change Log

| Date (UTC) | Summary | Details |
|------------|---------|---------|
| 2025-10-10 | Sprint 5 implementation | Added forecast sweep engine, config schema, calibration metadata, documentation, tests, Make targets. |

---

_End of Document_