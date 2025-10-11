# Evaluation

This document describes the evaluation methods used in the TimeGPT Intraday v2 project.

## Parameter sweeps & reproducibility

Sprint 6 introduced a full intraday backtester and sweep workflow:

- `timegpt_v2.backtest.simulator.BacktestSimulator` executes an event-driven loop. Entries are
  evaluated at each forecast snapshot per symbol, and exits are walked forward minute-by-minute
  until a variance, take-profit, or 15:55 ET time stop triggers. Each run emits a trade blotter at
  `artifacts/runs/<run_id>/trades/bt_trades.csv` and an aggregate summary in
  `artifacts/runs/<run_id>/eval/bt_summary.csv`. Net P&L reconciles exactly to the trade ledger.
- `timegpt_v2.backtest.grid.GridSearch` iterates the cross-product of
  `k_sigma × s_stop × s_take`. Every combination is hashed to `eval/grid/<combo_hash>/` where the
  corresponding `bt_summary.csv` is written. The hash is also used to seed stochastic components,
  guaranteeing deterministic outputs for each grid point.
- The CLI `timegpt_v2.cli sweep` command loads forecasts, features, and validated price bars, then
  generates a ranked summary table while persisting the per-grid folders. Distinct (k, s) settings
  therefore produce distinct statistics, fixing the historical "identical stats" bug.

Sprint 5 added forecast configuration sweeps:

- `timegpt_v2.forecast.sweep.ForecastGridSearch` explores snapshot presets, horizons, quantile sets,
  target scaling modes, and calibration methods. Each combo materialises an isolated config directory
  and optionally executes forecast/backtest/evaluate pipelines.
- Composite scoring ranks runs via `Sharpe × (1 − |PIT − 0.5|)` and writes `eval/grid/forecast_grid/scoreboard.csv`.
- Make targets `forecast-grid-plan` and `forecast-grid` provide convenient automation; `--reuse-baseline`
  reuses feature/validation artefacts to avoid redundant regeneration.

This layout enables reproducible parameter studies while keeping single-run artifacts lightweight
and auditable.

## Portfolio Evaluation

Sprint 7 introduces comprehensive portfolio-level evaluation metrics:

- **Portfolio KPIs:** Aggregate performance across all symbols using equal-weighted daily returns.
  Metrics include Sharpe ratio, maximum drawdown, hit rate, and total net P&L.
- **Per-Symbol Metrics:** Individual symbol performance tracking with trade count, net P&L,
  hit rate, Sharpe, and max drawdown.
- **Phase-Based Analysis:** Metrics computed separately for in-sample, out-of-sample (OOS),
  and stress test periods to assess robustness across regimes.
- Outputs: `eval/portfolio_metrics.csv`, `eval/per_symbol_metrics.csv`, `eval/oos_summary.csv`.

## OOS Evaluation

Out-of-sample evaluation ensures the strategy performs on unseen data:

- **OOS Months:** Configured in `configs/backtest.yaml` as `oos_months` list (e.g., ["2024-09", "2024-10"]).
- **OOS Gates:** Portfolio must achieve Sharpe ≥ 0.5, hit rate ≥ 48%, net P&L > 0 in OOS period.
- **OOS Summary:** Dedicated output file `eval/oos_summary.csv` with OOS-specific metrics.
- No fitting or tuning on OOS data; evaluation only.

## Cost Sensitivity Analysis

Transaction cost robustness testing:

- **Cost Multipliers:** Evaluate performance at 1.0×, 1.5×, and 2.0× base costs.
- **Cost Components:** Fees (bps) + half-spread (ticks) per symbol.
- **Sensitivity Curve:** `eval/cost_sensitivity.csv` shows how Sharpe, hit rate, and P&L
  degrade with increasing costs.
- **Gate:** Net P&L must remain positive at 1.5× costs.
- Implementation: Recomputes net P&L by scaling costs in trade blotter.

## Gates & failure policy

The evaluation pipeline includes a set of gates to ensure that the performance of the model and the trading strategy meet a minimum set of criteria. If any of these gates fail, the `evaluate` command will exit with a non-zero exit code.

### Forecast Gates

*   **rMAE:** The median relative mean absolute error (rMAE) must be less than 0.95.
*   **rRMSE:** The median relative root mean squared error (rRMSE) must be less than 0.97.

### Calibration Gates

## Sprint 4: Calibration + Coverage Diagnostics

Sprint 4 introduces post-hoc quantile calibration methods to address under-coverage issues:

- **Post-hoc Quantile Widening (`widen_intervals`):** Applies a multiplicative factor to interval half-widths, increasing coverage toward nominal levels. Used when PIT coverage is consistently below target.
- **Split-Conformal Prediction (`split_conformal`):** Implements a simple conformal prediction approach using historical residuals to widen intervals adaptively. Computes the conformal width as the quantile of absolute residuals and applies it symmetrically around the median forecast.
- **Coverage Report per Symbol & Snapshot (`generate_coverage_report`):** Generates detailed coverage statistics broken down by symbol and forecast snapshot, enabling fine-grained analysis of calibration performance across time and assets.

These methods work on cached forecasts from Sprint 3, ensuring no additional API calls are required. The `evaluate` command now produces `eval/coverage_report.csv` alongside existing metrics, providing comprehensive calibration diagnostics.

### Calibration Gates

*   **PIT Coverage:** The probability integral transform (PIT) coverage for the 25th and 75th percentiles must be within ±2% of the nominal coverage (i.e., between 48% and 52% for a 50% confidence interval).
*   **PIT Coverage:** The probability integral transform (PIT) coverage for the 25th and 75th percentiles must be within ±2% of the nominal coverage (i.e., between 48% and 52% for a 50% confidence interval).
