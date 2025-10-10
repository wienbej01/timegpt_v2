# TimeGPT Intraday — **v2** Sprint & Task Plan

*(Drop this file into the root of your new repo `timegpt_v2` as `SPRINTS.md` and execute sprint-by-sprint.)*

> Goal: rebuild a **GCS-first, Data-Quality–led, quantile-aware, short-horizon** intraday system.
> Guarantees: robust data contracts, leakage-safe features, real TimeGPT quantiles, **working parameter sweeps** (fix for k/s bug), and measurable KPIs.
> Every sprint includes: tasks → tests → docs → git.

---

## Repo Skeleton (create now)

```text
timegpt_v2/
  README.md
  SPRINTS.md
  pyproject.toml
  Makefile
  .gitignore
  .editorconfig
  .pre-commit-config.yaml

  configs/
    universe.yaml
    data.yaml
    dq_policy.yaml
    forecast.yaml
    trading.yaml
    backtest.yaml

  src/timegpt_v2/
    __init__.py
    cli.py

    io/gcs_reader.py
    quality/contracts.py
    quality/checks.py

    fe/base_features.py
    fe/deterministic.py
    fe/context.py

    framing/build_payloads.py

    forecast/timegpt_client.py
    forecast/scheduler.py

    trading/rules.py
    trading/costs.py

    backtest/simulator.py
    backtest/grid.py

    eval/metrics_forecast.py
    eval/metrics_trading.py
    eval/calibration.py

    reports/builder.py

    utils/dt.py
    utils/log.py
    utils/cache.py
    utils/events.py
    utils/synthetic.py

  tests/
    conftest.py
    test_io_gcs_reader.py
    test_quality_checks.py
    test_fe.py
    test_framing.py
    test_forecast_client.py
    test_scheduler.py
    test_trading_rules.py
    test_backtest_simulator.py
    test_backtest_grid.py
    test_eval_metrics.py
    test_reports.py

  docs/
    ARCHITECTURE.md
    DATA_QUALITY.md
    FORECASTING.md
    TRADING_RULES.md
    EVALUATION.md

  artifacts/                # gitignored; created per run_id
```

---

## Tooling & Conventions (put in Sprint 0)

* Python **3.10+**, full **type hints**, `from __future__ import annotations`.
* Lint/Format: **ruff**, **black**, **isort**. Types: **mypy** (strict optional later).
* Tests: **pytest**, **pytest-cov**, **hypothesis** (for property tests).
* CLI: **typer** (commands below).
* Logs: `logging` or `structlog` JSON; all logs → `artifacts/runs/<run_id>/logs/`.
* Pre-commit: run ruff/black/isort/mypy on staged changes.

**Makefile (minimum):**

```make
.PHONY: install lint fmt test test-cov run report sweep

install:  ## install package + dev
\tpip install -e .[dev]

lint:
\truff check .
\tblack --check .
\tisort --check-only .
\tmypy src/timegpt_v2

fmt:
\tblack .
\tisort .

test:
\tpytest -q

test-cov:
\tpytest --cov=src/timegpt_v2 --cov-report=term-missing

run:
\tpython -m timegpt_v2.cli run --run-id dev --config-dir configs

report:
\tpython -m timegpt_v2.cli report --run-id dev

sweep:
\tpython -m timegpt_v2.cli sweep --run-id dev --grid-config configs/trading.yaml
```

**pyproject.toml (deps to include):**

* runtime: `pandas`, `numpy`, `pyarrow`, `gcsfs`, `typer`, `pydantic`, `pyyaml`, `tqdm`, `structlog` (or std logging), `scipy`
* nixtla client: `nixtla` (or `nixtlats` depending on SDK)
* dev: `pytest`, `pytest-cov`, `hypothesis`, `ruff`, `black`, `isort`, `mypy`

**.gitignore (key lines):**

```
.venv/
artifacts/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
```

---

# Sprints

Each sprint ends with:

* ✅ All tasks implemented
* 🧪 Tests passing (`make test-cov`)
* 📝 Docs updated
* ⬆️ Commit & push

---

## **Sprint 0 — Scaffold & Guardrails**

**Tasks**

* Create the repo skeleton exactly as above.
* Fill `pyproject.toml`, `.pre-commit-config.yaml`, `Makefile`, `.editorconfig`, `.gitignore`.
* Minimal `src/timegpt_v2/cli.py` with Typer app and commands stubs:

  * `check-data`, `build-features`, `forecast`, `backtest`, `evaluate`, `report`, `sweep`
* `README.md`: environment setup, `make install`, `make test`.

**Tests**

* `pytest -q` runs; no failures.
* `make lint` clean.

**Docs**

* `docs/ARCHITECTURE.md` with one-paragraph responsibility per module.

**Git**

* Commit: `[S0] scaffold + tooling`
* Push: `main`

---

## **Sprint 1 — GCS Reader & Data Quality Gate**

**Objective**: Ingest minute Parquet from GCS; enforce contracts before any FE/forecasting.

**Tasks**

1. `io/gcs_reader.py`

   * Read GCS using `gcsfs`, expand template: `stocks/{ticker}/{yyyy}/{ticker}_{yyyy_mm}.parquet`
   * Map aliases → canonical: time(`t|timestamp|ts`), price(`c|close`) etc.
   * Localize to `America/New_York`, filter RTH `[09:30,16:00)`.

2. `quality/contracts.py` & `quality/checks.py`

   * Schema/dtype check (no missing required cols).
   * Monotonic timestamp; no duplicates per (symbol, ts).
   * Price sanity: `low ≤ min(open,close) ≤ high ≤ max(open,close)`; volume ≥ 0.
   * RTH completeness ≥ policy (default 95% of 390 bars/day).
   * Corporate action: require adjusted prices (config toggle).
   * Outlier scan (robust z on returns) → mark, don’t drop.
   * Build **gapless grid**; set `ffill_flag`; **drop** days with sustained ffill if policy says so.

3. CLI `check-data`

   * Write `validation/dq_report.json` + `logs/loader.log`.
   * **Exit non-zero** if hard policies fail (schema, monotonic, RTH completeness).

**Configs**

```yaml
# configs/data.yaml
gcs:
  bucket: jwss_data_store
  template: "stocks/{ticker}/{yyyy}/{ticker}_{yyyy_mm}.parquet"

# configs/dq_policy.yaml
rth_min_pct: 0.95
drop_days_with_sustained_ffill: true
hard_fail_on_schema: true
```

**Tests**

* `tests/test_io_gcs_reader.py`: reads synthetic GCS-like paths; alias mapping correct.
* `tests/test_quality_checks.py`: fails on bad schema; passes on good data; flags low RTH completeness.

**Docs**

* `docs/DATA_QUALITY.md` covering each check & policy.

**Git**

* Commit: `[S1] GCS reader + DQ gate`
* Push

---

## **Sprint 2 — Feature Engineering (Leakage-Safe)**

**Objective**: Stationary targets and lean, robust features.

**Tasks**

* `fe/base_features.py`:

  * Default **return-space** targets (log-return).
  * `ret_1m/5m/15m`, `rv_5m/15m`, `ATR_5m`; **Parkinson/Garman–Klass** vol.
  * `VWAP_30m`, `z_close_vwap_30m`, `vol_5m_norm` (5m vs 20-day median).
* `fe/deterministic.py`: minute index, Fourier sin/cos, session buckets (opening/lunch/power hour).
* `fe/context.py`: optional SPY features (lagged), regime flags (vol/dispersion), event dummies (earnings/FOMC/CPI).
* CLI `build-features`: write parquet to `artifacts/runs/<id>/features/`.

**Tests**

* `tests/test_fe.py`:

  * No future leakage (feature ts ≤ label ts).
  * NaN rate < 1% post-policy; else rows dropped.
  * Deterministic features are reproducible.

**Docs**

* `docs/FORECASTING.md` section “Feature policy & leakage”.

**Git**

* Commit: `[S2] Leakage-safe FE (returns default)`
* Push

---

## **Sprint 3 — Framing & TimeGPT Client (Quantiles)**

**Objective**: Build payloads; integrate TimeGPT quantiles with multi-series batching.

**Tasks**

* `framing/build_payloads.py`:

  * `build_y_df()` → `unique_id, ds, y` up to snapshot.
  * `build_x_df_for_horizon()` → future deterministic features available for forecast horizon (minute-by-minute to h).
* `forecast/timegpt_client.py`:

  * Batch **multi-series**; `freq="min"`; `h=10–20` by default; quantiles `[0.25,0.5,0.75]`.
  * Caching by `(symbol, date, snapshot, h, quantiles)`; persistence fallback logs warning.
  * Log per-symbol success row: “Forecasted <uid> h=<h> with q=[…]”.
* CLI `forecast`:

  * Persist `forecasts/quantiles.csv` with `ts(UTC), symbol, q25,q50,q75`.
  * Fail if any snapshot missing quantiles.

**Configs**

```yaml
# configs/forecast.yaml
snapshots_et: ["10:00", "14:30"]
horizon_min: 15
freq: "min"
quantiles: [0.25, 0.5, 0.75]
batch_multi_series: true
retry: { max: 3, backoff_sec: 2 }
cache: true
tz: America/New_York
```

**Tests**

* `tests/test_framing.py`: payload shapes; ds monotonic; last ds == snapshot.
* `tests/test_forecast_client.py`:

  * On stubbed SDK → quantiles present; not all equal; logs contain success lines.
  * Cache hit on rerun.

**Docs**

* `docs/FORECASTING.md` quantiles & batching.

**Git**

* Commit: `[S3] Framing + TimeGPT quantiles (batched)`
* Push

---

## **Sprint 4 — Calibration & Richer TimeGPT Options** ✅ **COMPLETED**

**Objective**: Tune the API call and correct residual bias with per-symbol affine/isotonic calibration.

**Tasks**

* Extended quantiles & metadata: `forecast.yaml` supports presets (baseline, liquidity_profile) with custom snapshot times, horizons, and active windows. Forecast runs record the active preset, horizon, and skipped event dates in `meta.json`.
* Event-aware scheduling: `skip_events` lists feature columns (e.g., `event_fomc`, `event_cpi`). If they fire, the scheduler drops that trading day. Skipped dates are tracked per run.
* Target scaling: `TargetScaler` introduced; log/basis-point/z-scored targets round-tripped back to raw log returns before evaluation. Helper columns (target_bp_ret_1m, target_z_ret_1m, vol_ewm_60m) remain leakage-safe.
* Forecast diagnostics: Evaluation writes `eval/forecast_diagnostics.csv`, capturing interval width stats alongside rMAE/rRMSE/PIT.
* Feature enrichments: Additional OHLCV-derived signals (skew/kurtosis, VWAP trend, volume percentile, range %, signed volume) feed both the feature matrix and the future payload, giving TimeGPT more context.
* Calibration layer: Per-symbol affine scaling and isotonic regression based on recent residuals; persisted in `models/calibration.json`.
* Richer TimeGPT config: Expanded quantile sets `[0.1, 0.25, 0.5, 0.75, 0.9]`, optional levels `[50, 80, 95]`, model selection (`timegpt-1`, `timegpt-1-long-horizon`).

**Tests**

* `tests/test_scheduler.py`: Scheduler respects skip dates and event-aware exclusions.
* `tests/test_target_scaling.py`: Scaling utilities ensure forward/inverse transform consistency.
* `tests/test_calibration.py`: Calibration tests for invertibility and monotonicity.

**Docs**

* `docs/FORECASTING.md`: Updated with presets, scaling, diagnostics, and calibration process.
* `docs/forecast_quality_plan.md`: Sprint 4 tasks marked complete.

**Git**

* Commit: `[S4] Calibration layer + richer TimeGPT options`
* Push

---

## **Sprint 5 — Grid Search & Operationalisation** ✅ **COMPLETED**

**Objective**: Automate configuration exploration and harden operations with forecast grid sweeps.

**Tasks**

* Extended grid runner to iterate over snapshot sets, horizons, target scaling, quantile configurations, and calibration toggles.
* Built composite scoring (`score = Sharpe × (1 - PIT deviation)`) to rank runs.
* Added weekly automation hooks (Make targets) to retrain/fine-tune with latest data and update calibration.
* Forecast grid engine: `src/timegpt_v2/forecast/sweep.py` with `ForecastGridSpec` and `ForecastGridSearch` for cross-product sweeps.
* Configuration schema: `configs/forecast_grid.yaml` defines sweep search space.
* CLI extensions: `sweep` now accepts `--forecast-grid`, `--plan-only`, `--reuse-baseline`, `--baseline-run`.
* Make targets: `make forecast-grid` and `make forecast-grid-plan` for execution and planning.
* Documentation: Updated `docs/FORECASTING.md` with sweep workflow, gating metrics, automation via Make targets.
* Technical docs: Created `SYSTEM_TECH_DOC.md` with system overview, module map, config schema, and change log.

**Tests**

* `tests/test_forecast_sweep.py`: Validates grid spec cross-product, plan-only configs, metric aggregation, and scoreboard ranking.

**Docs**

* `docs/FORECASTING.md`: Added forecast configuration sweeps section.
* `SYSTEM_TECH_DOC.md`: Comprehensive technical documentation.

**Git**

* Commit: `[S5] Forecast grid sweeps + operational automation`
* Push

---

## **Sprint 6 — Backtester & Grid Sweep (Fix k/s Bug)**

**Objective**: Reliable simulation, **sweep parameters** and ensure **different params lead to different outcomes**.

**Tasks**

* `backtest/simulator.py`:

  * Event-driven loop per symbol, per snapshot, then minute stepping to exits.
  * Outputs: `trades/bt_trades.csv` and `eval/bt_summary.csv`.
* **Grid runner** `backtest/grid.py`:

  * Iterate parameter grid (`k_sigma × s_stop × s_take`) **without reusing cached signals/P&L**.
  * For each combo: seed random components (if any) deterministically by param hash.
  * Save **separate** outputs under `eval/grid/<combo_hash>/bt_summary.csv`.
* CLI `sweep`: run grid, compile ranking table.

**Bug-proofing for “identical stats” issue**

* Ensure `rules.py` **reads k/s from function args** (not global state).
* Simulator must **instantiate rule engine per run** with the specific params.
* Ensure **no accidental reuse** of prior trades/DataFrame via in-place ops.
* **Test** that two distinct grids (e.g., k=0.5 vs 1.0) produce **different trade counts or P&L** on synthetic series.

**Tests**

* `tests/test_backtest_simulator.py`:

  * Cashflow accounting equals sum of trades; no NaNs.
* `tests/test_backtest_grid.py`:

  * On `utils/synthetic.py` price path, (k=0.5,s=1.0) vs (k=1.0,s=2.0) → **different** metrics.
  * Grid saves distinct `bt_summary.csv` under unique hashes.

**Docs**

* `docs/EVALUATION.md` section “Parameter sweeps & reproducibility”.

**Git**

* Commit: `[S6] Backtester + grid sweep (paramization fixed)`
* Push

---

## **Sprint 7 — Forecast & Trading Evaluation (Gates)**

**Objective**: Compute forecast & trading KPIs; fail fast if gates not met.

**Tasks**

* `eval/metrics_forecast.py`: MAE/RMSE vs persistence; **rMAE/rRMSE**; **pinball loss**; PIT coverage.
* `eval/metrics_trading.py`: hit rate, avg & total P&L (bp), Sharpe (daily agg), max DD (bp).
* `eval/calibration.py`: reliability plots & coverage checks (±2% tolerance).
* CLI `evaluate`: produce `eval/forecast_metrics.csv`, `eval/bt_summary.csv` and **exit non-zero** if gates fail:

  * Forecast per symbol: median `rMAE < 0.95` and `rRMSE < 0.97` (IS).
  * Calibration: coverage within ±2%.

**Tests**

* `tests/test_eval_metrics.py`: pinball loss decreases when quantiles tightened around truth (synthetic).
* Coverage test: nominal 0.25/0.75 near 25/75% hit on synthetic.

**Docs**

* `docs/EVALUATION.md` add “Gates & failure policy”.

**Git**

* Commit: `[S7] Eval gates (forecast+calibration)`
* Push

---

## **Sprint 8 — Portfolio P&L, Costs Sensitivity & OOS**

**Objective**: Prove viability beyond one month/one symbol; sensitivity to costs.

**Tasks**

* Extend configs to **OOS month** and optional **stress month**.
* CLI `backtest` can run multiple months; aggregator composes **portfolio** results across tickers.
* CLI `evaluate` supports **cost multipliers** (1.0×, 1.5×, 2.0×) producing a small table.
* Produce `reports/robustness_report.md` with OOS & cost sensitivity.

**KPI gates**

* Portfolio OOS: **Sharpe ≥ 0.5**, **net P&L > 0**, **hit-rate ≥ 48%**; at **1.5× costs** → **≥ 0**.

**Tests**

* `tests/test_reports.py`: report builds and contains required sections; cost table present.

**Docs**

* `docs/EVALUATION.md` finalize OOS and sensitivity.

**Git**

* Commit: `[S8] OOS + cost sensitivity + portfolio`
* Push

---

## **Sprint 9 — Reporting & README polish**

**Objective**: One-shot run instructions; clean repo state.

**Tasks**

* `reports/builder.py`: assemble `report.md` (universe, configs, KPIs, pass/fail, top grid).
* `README.md`: quickstart (5 min): install → check-data → build-features → forecast → backtest → evaluate → report.
* Example command set with sample configs.

**Tests**

* Docs spellcheck/links; `make report` generates file.

**Git**

* Commit: `[S9] Finalized reporting + README quickstart`
* Push

---

## Example Configs (drop-in defaults)

```yaml
# configs/universe.yaml
tickers: [AAPL, MSFT, NVDA]
dates: { start: "2024-07-01", end: "2024-08-31" }
tz: America/New_York
rth: { open: "09:30", close: "16:00" }
```

```yaml
# configs/forecast.yaml
snapshots_et: ["10:00", "14:30"]
horizon_min: 15
freq: "min"
quantiles: [0.25, 0.5, 0.75]
batch_multi_series: true
retry: { max: 3, backoff_sec: 2 }
cache: true
tz: America/New_York
```

```yaml
# configs/trading.yaml
k_sigma: [0.5, 0.75, 1.0]
s_stop: [1.0, 1.5, 2.0]
s_take: [1.0]
time_stop_et: "15:55"
fees_bps: 0.5
half_spread_ticks: { AAPL: 1, MSFT: 1, NVDA: 1 }
max_open_per_symbol: 1
daily_trade_cap: 3
no_trade: { earnings: true, fomc: true, cpi: true, buffer_min: 10 }
```

```yaml
# configs/backtest.yaml
cost_multipliers: [1.0, 1.5, 2.0]
portfolio_aggregation: equal_weight
```

---

## CLI Contract (LLM coder must implement)

* `check-data --config-dir configs --run-id <id>`
* `build-features --config-dir configs --run-id <id>`
* `forecast --config-dir configs --run-id <id>`
* `backtest --config-dir configs --run-id <id>`
* `sweep --config-dir configs --run-id <id> [--grid-config configs/trading.yaml]`
* `evaluate --config-dir configs --run-id <id>`
* `report --config-dir configs --run-id <id>`

Each command:

* creates `artifacts/runs/<id>/...`
* writes a JSON `meta.json` with configs, start/end time, versions
* logs to `logs/*.log`
* returns **non-zero** exit code on gate failures

---

## Acceptance KPIs (project-level)

* **Forecast**: median `rMAE < 0.95` AND `rRMSE < 0.97` vs persistence on IS; calibrated (±2%).
* **Trading**: OOS portfolio **Sharpe ≥ 0.5**, **net P&L > 0**, **hit ≥ 48%**; ≥ 0 P&L at 1.5× costs.
* **Grid**: different (k,s) combos produce **different** outcomes on synthetic and real data (no “identical stats” bug).
* **Ops**: deterministic runs, artifact-complete, quota-safe; CI lint+tests green.

---

## Git Discipline (each sprint)

1. `make fmt && make lint && make test`
2. Commit with prefix: `[S<N>] …` (concise, imperative)
3. Push to `main` (or feature branch + PR if you prefer reviews)
4. Tag releases: `v2-sprint<N>`

---

### Notes to the LLM Coder

* Treat configuration as **single source of truth**; no hidden globals.
* Ensure **rules & simulator** read parameters from function arguments **per run** (not module state) → this **fixes the k/s identical stats bug**.
* Never mutate shared DataFrames in place across runs; copy as needed.
* Cache **only** TimeGPT responses keyed by `(symbol, date, snapshot, horizon, quantiles)`; **do not** cache trading outcomes between grid points.
* All timestamps are ET internally; export to UTC in files (add `ts_utc` column).
* Tests must run fast (use `utils/synthetic.py` for fixtures).

---

**You can now paste this entire file into `timegpt_v2/SPRINTS.md` and start with Sprint 0.**
