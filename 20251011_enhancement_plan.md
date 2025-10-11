# TimeGPT v2 — API-Budget-Aware Sprint Plan (LLM Implementation Prompt)

> Paste this **entire file** into your LLM coder. It is self-contained and prescriptive.  
> Goal: build an intraday equity forecasting & trading system around Nixtla TimeGPT with **strict API-budget limits**, **rolling history windows**, **short intraday horizons**, and **artifact-first reproducibility**.

---

## Context & Objectives

- You are implementing a **new repo** `timegpt_v2` (Python 3.10+).  
- Data source: **GCS Parquet minute bars** (AAPL/MSFT/NVDA to start).  
- Modeling: **TimeGPT quantile forecasts** at minute frequency with **short horizons (5–30m)** and **rolling history windows (10–120 trading days)**.  
- Trading: **quantile-aware entries**, volatility-scaled stops, cost model (fee + half-spread), one position per symbol.  
- Evaluation: **rMAE/rRMSE vs persistence**, **pinball loss**, **coverage (PIT)**, after-cost P&L, Sharpe, cost sensitivity, OOS checks.  
- **Hard constraint:** API calls are limited. You **must** implement a **Budget Manager**, **cache**, **dry-run/simulate mode**, and **batching** to keep calls low.

**Non-goals**: live execution, order routing, alpha blending with other models.

---

## Global Rules (apply to all sprints)

- **Artifacts-first**: every command writes under `artifacts/runs/<run_id>/…` (CSV, JSON logs). Never depend on in-memory state between commands.  
- **Strict determinism**: set seeds; avoid in-place mutations that leak across runs.  
- **Config-driven**: no magic constants; all knobs in YAML.  
- **Budget hard-stop**: Abort forecasting if **per-run** or **daily** budget would be exceeded.  
- **Batching**: **one multi-series TimeGPT call per snapshot** for all whitelisted symbols.  
- **Caching**: hash key = `(symbol_set, start_ts, end_ts, snapshot_ts, freq, horizon_min, quantiles, features_hash, model_version)`. Cache hits **skip API calls**.  
- **Gating**: Trade only symbols that pass **forecast gates** (rMAE/rRMSE < 1, calibrated).  
- **Docs + Push**: Every sprint must update docs and **git push**.

---

## New/Updated Config Knobs (add to repo configs)

Add/extend these keys (top-level file suggestions in parentheses):

- **API budget** (`configs/forecast.yaml`):
  - `api_budget: { per_run: 25, per_day: 100, cooldown_sec: 1 }`
  - `api_mode: "online" | "offline"` (offline = no TimeGPT calls; use cached/simulated)
  - `cache_dir: "artifacts/forecast_cache"`
  - `batch_multi_series: true`
- **Snapshots & horizons** (`configs/forecast.yaml`):
  - `snapshots_et: ["09:45","10:30","13:30","14:45"]`
  - `horizons_min: [5,10,15,20]`  *(choose one at runtime or via sweep)*
  - `history_days: [10,20,40,60,120]`  *(rolling window options)*
- **Symbol gating** (`configs/universe.yaml`):
  - `whitelist: [AAPL]` *(expand as symbols pass gates)*
- **Trading gates & costs** (`configs/trading.yaml`):
  - `k_sigma: [0.5,0.75,1.0]`, `s_stop: [1.0,1.5,2.0]`, `s_take: [1.0]`
  - `fees_bps: 0.5`, `half_spread_ticks: { AAPL: 1, MSFT: 1, NVDA: 1 }`
  - `daily_trade_cap: 3`, `time_stop_et: "15:55"`
- **Evaluation gates** (`configs/backtest.yaml`):
  - `forecast_gates: { rmae_lt: 0.95, rrmse_lt: 0.97, coverage_tol: 0.02 }`
  - `trading_gates: { oos_sharpe_gte: 0.5, hit_rate_gte: 0.48, pnl_at_1p5x_costs_gte: 0 }`

---

## Sprint Plan (API-Budget–Aware)

> Follow these sprints **in order**. Each sprint ends with:
> 1) Unit/integration tests green, 2) Docs updated, 3) Commit + push.  
> Use **synthetic/offline modes** during development to limit API usage.

---

### **Sprint 0 — Scaffold, CLI, and Budget Manager (no API calls)** ✅ COMPLETED

**Goal**: Create repo skeleton, CLI with subcommands, and **Budget Manager** + **Forecast Cache** (no TimeGPT yet).

**Deliverables**
- Repo skeleton (as previously agreed), CLI commands:
  `check-data`, `build-features`, `forecast`, `backtest`, `evaluate`, `report`, `sweep`
- `utils/cache.py`: file-based cache (read/write with hash key).
- `utils/api_budget.py`: tracks **per_run**/**per_day** calls, enforces hard stops; JSON ledger at `artifacts/api_budget.json`.
- `forecast/scheduler.py`: generates snapshot timestamps for given date range & RTH; no API calls.
- `utils/synthetic.py`: generate deterministic price series for tests.

**Tests**
- Cache write/read roundtrip; hash key stable.
- Budget ledger increments; refuses calls beyond limit.
- Scheduler yields expected timestamps for a sample week.

**Docs**: `ARCHITECTURE.md` (modules), `EVALUATION.md` (KPI gates overview).

**Git**: `[S0] scaffold + budget manager + cache` → push.

---

### **Sprint 1 — GCS Reader + Data Quality Gate (no API calls)**

**Goal**: Ingest GCS parquet, enforce contracts, **gapless RTH grid**, ffill flags, DQ report.

**Deliverables**
- `io/gcs_reader.py`: read monthly parquet from `gcs://…`, alias mapping, ET tz, RTH filter.  
- `quality/contracts.py` & `quality/checks.py`: schema/dtype, monotonic, RTH completeness ≥ 95%, price sanity, outlier flags, **ffill gating**.  
- CLI `check-data`: writes `validation/dq_report.json`, fails on hard violations.

**Tests**
- Bad schema → non-zero exit.  
- Low RTH completeness → fail if below threshold.  
- Ffill day dropped when sustained holes detected.

**Docs**: `DATA_QUALITY.md`.

**Git**: `[S1] GCS + DQ gate` → push.

---

### **Sprint 2 — Leakage-Safe Feature Engineering (no API calls)**

**Goal**: Return-space target; base vol features; deterministic intraday clocks; optional SPY lagged context.

**Deliverables**
- `fe/base_features.py`: log returns; `ret_1m/5m/15m`, `rv_5m/15m`, ATR, Garman–Klass/Parkinson; VWAP(30m), z-close-VWAP, volume norms.  
- `fe/deterministic.py`: minute index, Fourier terms, session buckets.  
- `fe/context.py`: optional SPY lagged features, event dummies.  
- CLI `build-features`: export per-symbol parquet in `artifacts/runs/<id>/features/`.

**Tests**
- No future leakage (ts alignment).  
- NaN rate policy enforced (drop rows if needed).  
- Reproducibility on synthetic data.

**Docs**: `FORECASTING.md` (feature policy).

**Git**: `[S2] leakage-safe FE` → push.

---

### **Sprint 3 — Framing + TimeGPT Client (API minimal, cache on)**

**Goal**: Build `Y_df` and `X_df` per snapshot; **single batched multi-series TimeGPT call** per snapshot; **cache everything**.

**API minimization**
- Start with **whitelist: [AAPL]**.  
- **snapshots_et: ["10:30","14:30"]**, **horizon_min: 10** only.  
- `api_budget.per_run = 6` (example); batch all whitelisted symbols in **one** call per snapshot/day.  
- If cache hit for `(snapshot,h)`, **no call**.

**Deliverables**
- `framing/build_payloads.py`:  
  - `build_y_df(symbol, t_snapshot)` → `unique_id, ds, y` (history window from config; rolling).  
  - `build_x_df_for_horizon(symbol, t_snapshot, h)` → future deterministic features.  
- `forecast/timegpt_client.py`:  
  - `forecast_batch(unique_ids, Y_df, X_df, freq="min", h, quantiles)`  
  - Logs per snapshot: series count, h, quantiles, **calls_used**.  
  - Cache write: `forecasts/quantiles.csv` (append) and per-snapshot JSON with the hash key.  
- CLI `forecast`: iterates snapshots; enforces **Budget Manager**; supports `api_mode=offline|online`.

**Tests**
- Offline mode uses cache only; zero calls.  
- Online mode: ledger increments by **1 per snapshot**; abort when exceeding per_run.  
- Quantiles present and not all equal; shapes correct.

**Docs**: `FORECASTING.md` (scheduler, horizons, window).

**Git**: `[S3] framing + minimal batched TimeGPT + cache` → push.

---

### **Sprint 4 — Calibration + Coverage Diagnostics (reuse cache; low API)**

**Goal**: Fix under-coverage. Add **post-hoc quantile widening** and simple **split-conformal** option; add **pinball loss**, PIT coverage.

**API minimization**
- **Reuse cached forecasts** from Sprint 3; **no new calls**.  
- Calibration works on quantiles & realized returns.

**Deliverables**
- `eval/calibration.py`:  
  - `widen_intervals(q25,q50,q75, alpha)`;  
  - `split_conformal(residuals, quantiles)`;  
  - Coverage report per symbol & snapshot.  
- `eval/metrics_forecast.py`: rMAE, rRMSE vs persistence; pinball loss; coverage deltas.  
- CLI `evaluate`: computes metrics and fails if `forecast_gates` unmet.

**Tests**
- Synthetic series: widening increases coverage toward nominal.  
- Pinball loss decreases when forecasts approach truth.  
- Gate failure returns non-zero exit.

**Docs**: `EVALUATION.md` (calibration + gates).

**Git**: `[S4] calibration + coverage diagnostics` → push.

---

### **Sprint 5 — Quantile-Aware Trading Rules (no extra API)**

**Goal**: Translate calibrated quantiles to trades; **EV(after-cost) filter**; inventory and cooldown.

**Deliverables**
- `trading/rules.py`:  
  - Long if `q25 > last + cost_bp` & `|q50−last| ≥ k·σ_5m`; short if mirror condition.  
  - Variance stop `s·σ_5m`, take-profit `s_take·σ_5m`, time stop 15:55 ET.  
  - Max 1 position/symbol; `daily_trade_cap`; cooldown bars.  
  - **EV(after-cost) > 0** check using q-spread and uncertainty.  
- `trading/costs.py`: fees + half-spread.

**Tests**
- Inventory ∈ {−1,0,+1}; stops/TP trigger correctly on synthetic paths.  
- With wide intervals (uncertain), entries suppressed.

**Docs**: `TRADING_RULES.md`.

**Git**: `[S5] quantile-aware trading rules` → push.

---

### **Sprint 6 — Backtester + Grid Sweep (no extra API; fix “identical stats” bug)**

**Goal**: Event-driven backtester; **parameter grid** across `(history_days, horizon_min, k_sigma, s_stop, s_take)` using **cached forecasts** only. Guarantee different grids → different results.

**Deliverables**
- `backtest/simulator.py`: snapshot loop → minute stepping until exits; outputs `trades/bt_trades.csv`, `eval/bt_summary.csv`.  
- `backtest/grid.py`:  
  - Iterate grid; **do not call API**; use cached forecasts & features.  
  - Hash each combo; write under `eval/grid/<hash>/bt_summary.csv`.  
  - Per-combo seed set from hash (if any randomness).  
- CLI `sweep`: runs grid; writes ranking table.

**Bug-proofing**
- All rule params **passed as args** to rule engine; no module-level globals.  
- Simulator **rebuilds** state per grid combo; **no in-place reuse**.

**Tests**
- Two distinct combos produce **different** trade counts/P&L on synthetic fixtures.  
- Grid outputs saved under different hash dirs.

**Docs**: `EVALUATION.md` (grid methodology).

**Git**: `[S6] backtester + param sweep (no duplicate stats)` → push.

---

### **Sprint 7 — Portfolio, Cost Sensitivity, OOS (no extra API)**

**Goal**: Evaluate portfolio P&L; test **cost multipliers** and **OOS month**; symbol gating for live set.

**Deliverables**
- `eval/metrics_trading.py`: per-symbol & portfolio KPIs: hit-rate, P&L, Sharpe, max DD.  
- Cost sensitivity curve: 1.0×, 1.5×, 2.0×.  
- OOS evaluation month(s).  
- CLI `evaluate` extended: portfolio & cost sensitivity outputs.

**Tests**
- OOS and cost tables created; portfolio sums equal symbol aggregates.  
- Gates applied: portfolio must meet `trading_gates` or command fails.

**Docs**: `EVALUATION.md` (portfolio, OOS, costs).

**Git**: `[S7] portfolio + OOS + cost sensitivity` → push.

---

### **Sprint 8 — Exogenous Features (X_df) with Minimal Calls**

**Goal**: Add `X_df` (market/sector/seasonality/event flags) to improve signal. Keep API usage low via **precompute once** and **cache**.

**API minimization**
- Re-forecast **only** for the chosen `(history_days, horizon_min)` and **whitelisted symbols**.  
- Batch multi-series; respect `api_budget`.

**Deliverables**
- Extend `framing/build_payloads.py` to assemble `X_df_future` and `hist_exog_list`.  
- Re-run `forecast` in **online mode** with X_df to refresh cache.  
- Re-run `evaluate` & `sweep` using cached outputs.

**Tests**
- If `api_mode=offline`, forecasts read from cache; zero calls.  
- With X_df online once, cache grows; subsequent runs offline.

**Docs**: `FORECASTING.md` (X_df requirements & alignment).

**Git**: `[S8] exogenous features + minimal online refresh` → push.

---

### **Sprint 9 — Reporting & Final Gates**

**Goal**: One-click reports; fail-fast gates; README quickstart.

**Deliverables**
- `reports/builder.py`: `report.md` + `robustness_report.md` (config dump, KPIs, grid top-k, cost curves, gates pass/fail).  
- README: quickstart (install → check-data → build-features → forecast [offline/online] → backtest → evaluate → report).  
- All commands wired in `Makefile`.

**Tests**
- Report files created; contain required sections.  
- `make test-cov` passes.

**Git**: `[S9] reporting + README` → push.

---

## Acceptance & API-Budget Checks (each sprint)

- **Budget ledger** exists; API calls used ≤ `api_budget.per_run`.  
- **Offline mode** runs end-to-end with **zero calls**.  
- **Cache hit-ratio** reported; increasing over runs.  
- **Batching** verified: **≤ 1 call per snapshot** for all whitelisted symbols.

---

## Coder Runbook (end of each sprint)

1) `make fmt && make lint && make test && make test-cov`  
2) Update docs under `docs/` and README as needed  
3) Commit with `[S<N>] …` and push  
4) Save a demo run:  
   - `python -m timegpt_v2.cli check-data --config-dir configs --run-id demo`  
   - `python -m timegpt_v2.cli build-features --config-dir configs --run-id demo`  
   - `python -m timegpt_v2.cli forecast --config-dir configs --run-id demo --api-mode offline` *(online only when instructed, and within budget)*  
   - `python -m timegpt_v2.cli backtest --config-dir configs --run-id demo`  
   - `python -m timegpt_v2.cli evaluate --config-dir configs --run-id demo`  
   - `python -m timegpt_v2.cli report --config-dir configs --run-id demo`

---

## Final Notes

- Start with **AAPL only** until forecast gates pass; expand the whitelist gradually.  
- Prefer **h = 10–15m** (AAPL/MSFT) and **5–10m** (NVDA) initially; history window **40–60 trading days**.  
- Keep **API usage minimal**: cache everything, batch once per snapshot, offline mode for all sweeps, only refresh cache online when validating a new setting that matters.
