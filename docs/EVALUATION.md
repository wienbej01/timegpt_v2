# Evaluation

This document describes the evaluation methods used in the TimeGPT Intraday v2 project.

## Parameter sweeps & reproducibility

The project includes a grid search capability to sweep over different trading parameters and evaluate their performance. The grid search is implemented in the `backtest/grid.py` module and can be run from the CLI using the `sweep` command.

The grid search iterates over a grid of parameters defined in the `configs/trading.yaml` file. For each combination of parameters, it runs a backtest and saves the results to a separate file under the `eval/grid/<combo_hash>` directory, where `<combo_hash>` is a hash of the parameter combination. This ensures that the results of each run are reproducible and can be easily compared.

To ensure that different parameter settings lead to different outcomes, the backtest simulator is instantiated with a new set of trading rules for each run. This prevents the simulator from reusing cached signals or P&L from previous runs.

## Gates & failure policy

The evaluation pipeline includes a set of gates to ensure that the performance of the model and the trading strategy meet a minimum set of criteria. If any of these gates fail, the `evaluate` command will exit with a non-zero exit code.

### Forecast Gates

*   **rMAE:** The median relative mean absolute error (rMAE) must be less than 0.95.
*   **rRMSE:** The median relative root mean squared error (rRMSE) must be less than 0.97.

### Calibration Gates

*   **PIT Coverage:** The probability integral transform (PIT) coverage for the 25th and 75th percentiles must be within ±2% of the nominal coverage (i.e., between 48% and 52% for a 50% confidence interval).
