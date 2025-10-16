# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TimeGPT Intraday v2 is a financial forecasting and trading platform that:
- Loads market data from GCS buckets
- Builds feature matrices for intraday stock prediction
- Generates quantile forecasts using TimeGPT API
- Backtests trading strategies with risk management
- Evaluates performance with comprehensive metrics

## Development Commands

### Setup
```bash
make install          # Install package + dev dependencies and pre-commit hooks
```

### Code Quality
```bash
make lint             # Run ruff, black, isort, mypy checks
make fmt              # Format code with black and isort
make test             # Run pytest
make test-cov         # Run tests with coverage report
```

### Pipeline Commands

#### CLI Parameter Overrides (NEW)
The system now supports CLI parameter overrides for tickers and dates, eliminating the need for custom universe configuration files:

```bash
# Full pipeline with CLI parameter overrides
RUN_ID=custom_test
python -m timegpt_v2.cli check-data \
  --config-dir configs --config-name forecast.yaml --universe-name universe.yaml \
  --tickers "TSLA,AAPL,MSFT" --start-date "2025-02-01" --end-date "2025-03-31" --run-id "$RUN_ID"

python -m timegpt_v2.cli build-features \
  --config-dir configs --config-name forecast.yaml --universe-name universe.yaml \
  --tickers "TSLA,AAPL,MSFT" --start-date "2025-02-01" --end-date "2025-03-31" --run-id "$RUN_ID"

python -m timegpt_v2.cli forecast \
  --config-dir configs --config-name forecast.yaml --run-id "$RUN_ID" --api-mode offline

python -m timegpt_v2.cli backtest \
  --config-dir configs --config-name forecast.yaml --universe-name universe.yaml \
  --tickers "TSLA,AAPL,MSFT" --start-date "2025-02-01" --end-date "2025-03-31" --run-id "$RUN_ID"

python -m timegpt_v2.cli evaluate --config-dir configs --run-id "$RUN_ID"
python -m timegpt_v2.cli report --config-dir configs --run-id "$RUN_ID"
```

**New CLI Parameters:**
- `--tickers`: Comma-separated list of ticker symbols (overrides universe config)
- `--start-date`: Trading window start date in YYYY-MM-DD format (overrides universe config)
- `--end-date`: Trading window end date in YYYY-MM-DD format (overrides universe config)

#### Standard Pipeline Commands
```bash
# Full demo pipeline (≈5 minutes)
RUN_ID=dev
python -m timegpt_v2.cli check-data --config-dir configs --run-id "$RUN_ID"
python -m timegpt_v2.cli build-features --config-dir configs --run-id "$RUN_ID"
python -m timegpt_v2.cli forecast --config-dir configs --run-id "$RUN_ID" --api-mode offline
python -m timegpt_v2.cli backtest --config-dir configs --run-id "$RUN_ID"
python -m timegpt_v2.cli evaluate --config-dir configs --run-id "$RUN_ID"
python -m timegpt_v2.cli report --config-dir configs --run-id "$RUN_ID"

# Individual commands
python -m timegpt_v2.cli check-data --config-dir configs --run-id <run_id>
python -m timegpt_v2.cli build-features --config-dir configs --run-id <run_id>
python -m timegpt_v2.cli forecast --config-dir configs --run-id <run_id> [--api-mode offline|online]
python -m timegpt_v2.cli backtest --config-dir configs --run-id <run_id>
python -m timegpt_v2.cli evaluate --config-dir configs --run-id <run_id>
python -m timegpt_v2.cli report --config-dir configs --run-id <run_id>

# Parameter sweeps
python -m timegpt_v2.cli sweep --config-dir configs --run-id <run_id> --api-mode offline
make forecast-grid-plan  # Dry-run forecast grid plan
make forecast-grid       # Execute forecast grid with baseline reuse

# Calibration
python -m timegpt_v2.cli calibrate --config-dir configs --run-id <run_id> [--baseline-run <baseline_id>]
```

### Quick Smoke Test
```bash
RUN_ID=smoke_aapl_2024 python smoke.py
# Or manual steps:
python -m timegpt_v2.cli check-data --config-name forecast_smoke.yaml --universe-name universe_smoke.yaml --run-id <run_id>
python -m timegpt_v2.cli build-features --config-name forecast_smoke.yaml --universe-name universe_smoke.yaml --run-id <run_id>
python -m timegpt_v2.cli forecast --config-name forecast_smoke.yaml --universe-name universe_smoke.yaml --run-id <run_id>
python -m timegpt_v2.cli backtest --config-name forecast_smoke.yaml --run-id <run_id>

# Quick smoke test with CLI parameter overrides
RUN_ID=smoke_cli
python -m timegpt_v2.cli check-data --tickers "AAPL" --start-date "2024-07-01" --end-date "2024-07-15" --run-id "$RUN_ID"
python -m timegpt_v2.cli build-features --tickers "AAPL" --start-date "2024-07-01" --end-date "2024-07-15" --run-id "$RUN_ID"
python -m timegpt_v2.cli forecast --run-id "$RUN_ID" --api-mode offline
python -m timegpt_v2.cli backtest --tickers "AAPL" --start-date "2024-07-01" --end-date "2024-07-15" --run-id "$RUN_ID"
```

### 30-Minute Optimization Commands

The system has been optimized for 30-minute forecast horizons. Use these commands for the optimized setup:

```bash
# TSLA February 2024 test with 30-minute optimization
export TIMEGPT_API_KEY="your_api_key_here"
RUN_ID=tsla_feb2024_30m

python -m timegpt_v2.cli check-data \
  --config-name forecast_tsla_online.yaml \
  --universe-name universe_tsla_feb2024.yaml \
  --run-id "$RUN_ID"

python -m timegpt_v2.cli build-features \
  --config-name forecast_tsla_online.yaml \
  --universe-name universe_tsla_feb2024.yaml \
  --run-id "$RUN_ID"

python -m timegpt_v2.cli forecast \
  --config-name forecast_tsla_online.yaml \
  --run-id "$RUN_ID" \
  --api-mode online

python -m timegpt_v2.cli backtest \
  --config-name trading_tsla_30m.yaml \
  --universe-name universe_tsla_feb2024.yaml \
  --run-id "$RUN_ID"
```

### Key Configuration Files

**Standard Configurations**:
- `configs/data.yaml`: GCS bucket paths and data loading settings
- `configs/universe.yaml`: Stock universe and date ranges
- `configs/forecast.yaml`: TimeGPT API settings, quantiles, snapshots, exogenous features
- `configs/trading.yaml`: Trading rules, risk parameters, cost settings
- `configs/backtest.yaml`: Phase definitions and evaluation settings
- `configs/dq_policy.yaml`: Data quality validation rules
- `configs/forecast_grid.yaml`: Forecast parameter sweep specifications

**30-Minute Optimized Configurations**:
- `configs/trading_30m_optimized.yaml`: Fine-tuned for 30-minute forecasts
- `configs/forecast_tsla_online.yaml`: TSLA-specific online configuration
- `configs/trading_tsla_30m.yaml`: TSLA 30-minute optimized trading
- `configs/universe_tsla_feb2024.yaml`: TSLA test universe (Feb 1-15, 2024)

## Architecture

### Core Modules

**CLI (`src/timegpt_v2/cli.py`)**: Main entry point with Typer commands defining the pipeline workflow

**Data Pipeline**:
- `src/timegpt_v2/io/gcs_reader.py`: GCS bucket reading with bronze layer data access
- `src/timegpt_v2/loader/gcs_loader.py`: Multi-month history loading
- `src/timegpt_v2/quality/`: Data quality validation and contracts
- `src/timegpt_v2/fe/`: Feature engineering (deterministic, context-based, base features)

**Forecasting**:
- `src/timegpt_v2/forecast/timegpt_client.py`: TimeGPT API client with backends
- `src/timegpt_v2/forecast/scheduler.py`: Snapshot scheduling with trading holidays
- `src/timegpt_v2/forecast/scaling.py`: Target scaling transformations
- `src/timegpt_v2/forecast/exogenous.py`: Exogenous feature handling
- `src/timegpt_v2/forecast/batcher.py`: Request batching and size controls
- `src/timegpt_v2/forecast/sweep.py`: Forecast grid search functionality

**Trading & Backtesting**:
- `src/timegpt_v2/backtest/simulator.py`: Trading simulation engine
- `src/timegpt_v2/backtest/aggregation.py`: Phase-based performance aggregation
- `src/timegpt_v2/backtest/grid.py`: Trading parameter grid search
- `src/timegpt_v2/trading/rules.py`: Trading rule engine with risk management
- `src/timegpt_v2/trading/costs.py`: Trading cost modeling

**Evaluation**:
- `src/timegpt_v2/eval/metrics_forecast.py`: Forecast accuracy metrics (MAE, RMSE, PIT coverage)
- `src/timegpt_v2/eval/metrics_trading.py`: Trading performance metrics (Sharpe, drawdown)
- `src/timegpt_v2/eval/calibration.py`: Quantile calibration and conformal widening

**Configuration & Utilities**:
- `src/timegpt_v2/config/`: YAML-based configuration loading
- `src/timegpt_v2/utils/cache.py`: Forecast caching
- `src/timegpt_v2/utils/api_budget.py`: API usage budgeting
- `src/timegpt_v2/reports/builder.py`: Final report generation

### Pipeline Flow

1. **Data Validation**: `check-data` loads raw market data from GCS, validates with quality policies
2. **Feature Engineering**: `build-features` creates feature matrix with technical indicators and context
3. **Forecasting**: `forecast` generates TimeGPT quantile forecasts at scheduled snapshots
4. **Backtesting**: `backtest` simulates trading strategies using forecasts with risk controls
5. **Evaluation**: `evaluate` computes forecast accuracy and trading performance metrics
6. **Reporting**: `report` assembles comprehensive analysis and robustness checks

### Key Configuration Files

- `configs/data.yaml`: GCS bucket paths and data loading settings
- `configs/universe.yaml`: Stock universe and date ranges
- `configs/forecast.yaml`: TimeGPT API settings, quantiles, snapshots, exogenous features
- `configs/trading.yaml`: Trading rules, risk parameters, cost settings
- `configs/backtest.yaml`: Phase definitions and evaluation settings
- `configs/dq_policy.yaml`: Data quality validation rules
- `configs/forecast_grid.yaml`: Forecast parameter sweep specifications

### Data Flow

All artifacts are stored under `artifacts/runs/<run_id>/`:
- `validation/`: Data quality reports and cleaned price data
- `features/`: Feature matrix parquet files
- `forecasts/`: Quantile forecasts (CSV format with cache)
- `trades/`: Backtest trade execution logs
- `eval/`: Performance metrics and evaluation results
- `logs/`: Detailed execution logs per command

### API Modes

- **Online**: Hits TimeGPT API for fresh forecasts
- **Offline**: Uses cached forecasts only (for sweeps/backtests)
- Use `--api-mode offline` for parameter sweeps to avoid API costs

### Exogenous Features

Configure in `configs/forecast.yaml` under `exog` section:
- Enable/disable pipeline with `use_exogs`
- Control validation strictness with `strict_exog`
- Declare historical vs future features
- Override with CLI flags: `--use-exogs/--no-exogs`, `--strict-exog/--no-strict-exog`

### Trading Parameters

Key parameters in `configs/trading.yaml`:
- `k_sigma`: Entry signal threshold (standard deviations)
- `s_stop`: Stop-loss distance
- `s_take`: Take-profit distance
- `uncertainty_k`: Uncertainty multiplier for position sizing
- `daily_trade_cap`: Maximum trades per day
- `max_open_per_symbol`: Max concurrent positions per symbol

### Testing

Run single test: `pytest tests/test_specific_module.py -q`
Tests use hypothesis for property-based testing and include uncertainty handling validation.

### Common Issues

- **Missing GCS mount**: Ensure `~/gcs-mount/` is mapped to production bucket
- **API key errors**: Set `TIMEGPT_API_KEY` or `NIXTLA_API_KEY` in environment
- **Payload size limits**: Adjust `max_bytes_per_call` and `num_partitions` in forecast config
- **Missing exogenous features**: Use `--no-strict-exog` or check feature generation
- **CLI parameter overrides not working**: Ensure the full pipeline runs sequentially. Backtest requires forecasts to exist from previous steps
- **No trades generated with CLI parameters**: Check that the trading strategy parameters in config files are appropriate for the selected symbols and time period