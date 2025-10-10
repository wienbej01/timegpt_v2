# Forecasting Foundations

## Feature Policy & Leakage

The intraday feature stack is anchored on return-space targets to avoid scale drift and to keep the prediction target stationary. Minute close-to-close log returns define the default label (`target_log_return_1m`), while predictor columns are strictly sourced from information available at or before the feature timestamp. Rolling windows never peek forward: momentum (`ret_1m/5m/15m`), realised variance (`rv_5m/15m`), ATR, and Parkinson / Garman–Klass volatility are all computed with past bars only. Volume-sensitive signals (`VWAP_30m`, `z_close_vwap_30m`, `vol_5m_norm`) normalise intraday liquidity by recent history, ensuring each row represents the state just prior to the forecast horizon. Deterministic components expose an explicit `minute_index`, Fourier harmonics, and session buckets—fully reproducible from the timestamp and trading calendar—while contextual enrichments keep SPY factors lagged by one minute and layer on regime flags and event dummies (earnings/FOMC/CPI). After the feature pass we drop any row where the NaN share exceeds 1%, guaranteeing downstream models see leakage-safe and numerically stable matrices stored under `artifacts/runs/<run_id>/features/`.

## Quantiles & Batching

Sprint 3 wires the framing layer directly into the TimeGPT client. For each trading day and configured ET snapshot we build a leakage-safe target frame (`unique_id, ds, y`) and project the deterministic feature block forward minute-by-minute to the forecast horizon. The client batches all series in one call with `freq="min"`, requesting `[0.1, 0.25, 0.5, 0.75, 0.9]` quantiles (optionally paired with probability levels `[50, 80, 95]`) and defaulting to a 15-minute horizon (configurable via `configs/forecast.yaml`). Results are cached per `(symbol, trade_date, snapshot, horizon, quantiles)` so re-forecasts hit disk-backed storage when possible, logging cache hits to keep runs auditable. Aggregated quantiles flow into `artifacts/runs/<run_id>/forecasts/quantiles.csv`—one row per symbol, stamped in UTC—ensuring downstream evaluation receives consistent multi-series forecasts without duplicate API calls. Calibration metadata (method, window, samples) is versioned under `models/calibration.json` so post-processing layers remain reproducible.

Sprint 1 enhances the pipeline with configurable target scaling. The `target` block in `configs/forecast.yaml` lets you upload raw log returns, basis-point returns, or volatility-standardised z-scores; whichever mode you pick, the CLI posts scaled history to TimeGPT but always writes quantiles and `y_true` back in raw log-return space. The base feature matrix now includes helper columns (`target_bp_ret_1m`, `target_z_ret_1m`, `vol_ewm_60m`) computed without forward leakage to support these transforms. Every evaluation run emits `eval/forecast_diagnostics.csv`, capturing per-symbol rMAE, rRMSE, PIT deviation, and interval-width summaries so calibration drift is immediately visible.

## Snapshot policy

The forecasting pipeline is driven by a scheduler that generates a series of snapshot timestamps for each trading day. Sprint 3 introduces **snapshot presets**, allowing you to define multiple intraday cadences and switch between them without editing code. In `configs/forecast.yaml`, set `snapshot_preset` to one of the entries under `snapshot_presets`. Each preset can specify its own `times`, `active_windows`, `max_snapshots_per_day`, and even override `horizon_min`, making it easy to compare, for example, a two-snapshot baseline against a five-snapshot liquidity profile. The scheduler still honours weekends and holidays, but now also accepts `skip_events`—a list of event feature columns (e.g., `event_fomc`, `event_cpi`) that trigger a full-day skip when present in the feature matrix. The CLI logs skipped dates and records the active preset, horizon, and event filters in run metadata for downstream analysis.

If no preset is defined, the legacy `snapshots_et` + `active_windows` configuration remains valid. Regardless of configuration style, `max_snapshots_per_day`, `max_snapshots_total`, and `max_batch_size` act as guardrails to keep API usage predictable while still covering the entire universe deterministically.

## Forecast configuration sweeps (Sprint 5)

Sprint 5 operationalises the forecasting workflow by introducing a configuration sweep runner. The CLI accepts an optional `--forecast-grid` YAML (see [`configs/forecast_grid.yaml`](configs/forecast_grid.yaml:1)) that defines cross-products of snapshot presets, horizons, quantile/level sets, target scaling modes, and calibration methods. For each combination, `timegpt_v2.cli sweep --forecast-grid ...`:

1. Materialises an isolated config directory with overrides for the selected preset, horizon, quantiles, calibration mode, and target scaling.
2. Executes `forecast`, `backtest`, and `evaluate` commands (or `--plan-only` to emit configs without execution).
3. Collects diagnostics (median PIT deviation, rMAE) and trading metrics (Sharpe) into `eval/grid/forecast_grid/plan.csv`.
4. Ranks completed runs via `composite_score = Sharpe × (1 - |PIT-0.5|)` and persists an ordered `scoreboard.csv`.

Plan-only mode is useful for review or asynchronous execution, while `--reuse-baseline --baseline-run <run_id>` reuses feature and validation artifacts to avoid redundant regeneration. The Makefile exposes convenience targets:

- `make forecast-grid-plan` → dry-run plan.
- `make forecast-grid` → execute the sweep using `configs/forecast_grid.yaml`.

Production presets should satisfy gating thresholds (median PIT ≤ 0.03 deviation, median rMAE ≤ 0.7, Sharpe ≥ 0.5) before promotion. Update the grid spec to explore additional cadences or calibration strategies as new market regimes emerge.
