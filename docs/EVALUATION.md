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

## Gates & failure policy

The evaluation pipeline includes a set of gates to ensure that the performance of the model and the trading strategy meet a minimum set of criteria. If any of these gates fail, the `evaluate` command will exit with a non-zero exit code.

### Forecast Gates

*   **rMAE:** The median relative mean absolute error (rMAE) must be less than 0.95.
*   **rRMSE:** The median relative root mean squared error (rRMSE) must be less than 0.97.

### Calibration Gates

*   **PIT Coverage:** The probability integral transform (PIT) coverage for the 25th and 75th percentiles must be within ±2% of the nominal coverage (i.e., between 48% and 52% for a 50% confidence interval).
