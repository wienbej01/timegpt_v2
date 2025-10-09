# Evaluation

This document describes the evaluation methods used in the TimeGPT Intraday v2 project.

## Parameter sweeps & reproducibility

The project includes a grid search capability to sweep over different trading parameters and evaluate their performance. The grid search is implemented in the `backtest/grid.py` module and can be run from the CLI using the `sweep` command.

The grid search iterates over a grid of parameters defined in the `configs/trading.yaml` file. For each combination of parameters, it runs a backtest and saves the results to a separate file under the `eval/grid/<combo_hash>` directory, where `<combo_hash>` is a hash of the parameter combination. This ensures that the results of each run are reproducible and can be easily compared.

To ensure that different parameter settings lead to different outcomes, the backtest simulator is instantiated with a new set of trading rules for each run. This prevents the simulator from reusing cached signals or P&L from previous runs.
