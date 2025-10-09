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
  therefore produce distinct statistics, fixing the historical “identical stats” bug.

This layout enables reproducible parameter studies while keeping single-run artifacts lightweight
and auditable.

## OOS portfolio & cost sensitivity

Sprint 8 extends the evaluation stack beyond a single-sample, single-symbol view:

- `timegpt_v2.cli backtest` now annotates every trade with its calendar month and regime
  (in-sample, out-of-sample, stress) using the configuration in `configs/backtest.yaml`. Portfolio
  summaries aggregate net P&L, hit-rate, and Sharpe across tickers with equal-weight daily returns
  and persist to `eval/portfolio_summary.csv`.
- `timegpt_v2.cli evaluate` reads those summaries, recomputes performance under configured cost
  multipliers, and enforces the Sprint 8 KPI gates (OOS Sharpe ≥ 0.5, net P&L > 0, hit-rate ≥ 48%,
  and non-negative net P&L at 1.5× costs). Results are written to `eval/cost_sensitivity.csv`.
- `timegpt_v2.reports.builder.build_report` assembles `reports/robustness_report.md`, highlighting
  the OOS portfolio metrics alongside the cost-sensitivity table for review.

These hooks document how portfolio viability and transaction-cost robustness are tracked as the
system moves beyond the initial calibration window.

## Gates & failure policy

The evaluation pipeline includes a set of gates to ensure that the performance of the model and the trading strategy meet a minimum set of criteria. If any of these gates fail, the `evaluate` command will exit with a non-zero exit code.

### Forecast Gates

*   **rMAE:** The median relative mean absolute error (rMAE) must be less than 0.95.
*   **rRMSE:** The median relative root mean squared error (rRMSE) must be less than 0.97.

### Calibration Gates

*   **PIT Coverage:** The probability integral transform (PIT) coverage for the 25th and 75th percentiles must be within ±2% of the nominal coverage (i.e., between 48% and 52% for a 50% confidence interval).
