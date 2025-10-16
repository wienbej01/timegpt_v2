# CLI Parameter Override Guide

This guide explains how to use the new CLI parameter override functionality that allows you to specify tickers and trading dates as command-line parameters instead of creating custom configuration files.

## Overview

The CLI parameter override system enables flexible, ad-hoc testing without the need to create or modify universe configuration files. You can now specify:

- **Tickers**: Comma-separated list of stock symbols
- **Trading Window**: Start and end dates for analysis
- **Dynamic Configuration**: Override any universe settings at runtime

## Available CLI Parameters

### `--tickers`
- **Description**: Comma-separated list of ticker symbols to analyze
- **Format**: `"TSLA,AAPL,MSFT"` (no spaces, comma-separated)
- **Override**: Replaces the `tickers` field in universe configuration
- **Required**: No (defaults to universe config if not specified)

### `--start-date`
- **Description**: Trading window start date
- **Format**: `"YYYY-MM-DD"` (ISO 8601 date format)
- **Override**: Replaces the `trading_window.start` and `dates.start` fields
- **Required**: No (defaults to universe config if not specified)

### `--end-date`
- **Description**: Trading window end date
- **Format**: `"YYYY-MM-DD"` (ISO 8601 date format)
- **Override**: Replaces the `trading_window.end` and `dates.end` fields
- **Required**: No (defaults to universe config if not specified)

## Supported Commands

The CLI parameters are supported by the following commands:

1. **`check-data`** - Data validation and loading
2. **`build-features`** - Feature engineering
3. **`forecast`** - TimeGPT forecast generation (inherited from previous steps)
4. **`backtest`** - Trading simulation and backtesting

## Usage Examples

### Basic Multi-Symbol Analysis
```bash
RUN_ID=multi_symbol_test
TICKERS="TSLA,AAPL,MSFT"
START_DATE="2025-02-01"
END_DATE="2025-03-31"

# Complete pipeline
python -m timegpt_v2.cli check-data \
  --config-dir configs --config-name forecast.yaml --universe-name universe.yaml \
  --tickers "$TICKERS" --start-date "$START_DATE" --end-date "$END_DATE" --run-id "$RUN_ID"

python -m timegpt_v2.cli build-features \
  --config-dir configs --config-name forecast.yaml --universe-name universe.yaml \
  --tickers "$TICKERS" --start-date "$START_DATE" --end-date "$END_DATE" --run-id "$RUN_ID"

python -m timegpt_v2.cli forecast \
  --config-dir configs --config-name forecast.yaml --run-id "$RUN_ID" --api-mode offline

python -m timegpt_v2.cli backtest \
  --config-dir configs --config-name forecast.yaml --universe-name universe.yaml \
  --tickers "$TICKERS" --start-date "$START_DATE" --end-date "$END_DATE" --run-id "$RUN_ID"

python -m timegpt_v2.cli evaluate --config-dir configs --run-id "$RUN_ID"
```

### Single Symbol Quick Test
```bash
RUN_ID=tsla_quick
python -m timegpt_v2.cli check-data \
  --tickers "TSLA" --start-date "2025-01-01" --end-date "2025-01-31" --run-id "$RUN_ID"

python -m timegpt_v2.cli build-features \
  --tickers "TSLA" --start-date "2025-01-01" --end-date "2025-01-31" --run-id "$RUN_ID"

python -m timegpt_v2.cli forecast --run-id "$RUN_ID" --api-mode offline

python -m timegpt_v2.cli backtest \
  --tickers "TSLA" --start-date "2025-01-01" --end-date "2025-01-31" --run-id "$RUN_ID"
```

### Date Range Only Override
```bash
# Use default tickers from universe config but override dates
RUN_ID=date_override_test
python -m timegpt_v2.cli check-data \
  --start-date "2024-12-01" --end-date "2024-12-31" --run-id "$RUN_ID"

python -m timegpt_v2.cli build-features \
  --start-date "2024-12-01" --end-date "2024-12-31" --run-id "$RUN_ID"

python -m timegpt_v2.cli forecast --run-id "$RUN_ID" --api-mode offline

python -m timegpt_v2.cli backtest \
  --start-date "2024-12-01" --end-date "2024-12-31" --run-id "$RUN_ID"
```

### Ticker Only Override
```bash
# Use default date range from universe config but override tickers
RUN_ID=ticker_override_test
python -m timegpt_v2.cli check-data \
  --tickers "NVDA,AMD" --run-id "$RUN_ID"

python -m timegpt_v2.cli build-features \
  --tickers "NVDA,AMD" --run-id "$RUN_ID"

python -m timegpt_v2.cli forecast --run-id "$RUN_ID" --api-mode offline

python -m timegpt_v2.cli backtest \
  --tickers "NVDA,AMD" --run-id "$RUN_ID"
```

## Configuration Override Behavior

### Universe Configuration Structure
The CLI parameters override the following fields in universe configuration:

```yaml
tickers: [TICKER_1, TICKER_2, ...]  # Overridden by --tickers
trading_window:
  start: "YYYY-MM-DD"              # Overridden by --start-date
  end: "YYYY-MM-DD"                # Overridden by --end-date
dates:
  start: "YYYY-MM-DD"              # Overridden by --start-date (backward compatibility)
  end: "YYYY-MM-DD"                # Overridden by --end-date (backward compatibility)
```

### Inheritance Rules
1. **CLI Parameters Take Priority**: Any CLI parameter will override the corresponding universe config field
2. **Partial Override**: You can override just tickers, just dates, or both
3. **Backward Compatibility**: If no CLI parameters are provided, the system uses universe config as before
4. **Configuration Persistence**: Original universe config files remain unchanged

## Output and Results

### Results Location
All results are stored under `artifacts/runs/<run_id>/` with the standard directory structure:
- `validation/`: Data quality reports
- `features/`: Feature matrices
- `forecasts/`: TimeGPT quantile forecasts
- `trades/`: Backtest trade execution logs
- `eval/`: Performance metrics and analysis

### Expected Trade Distribution
When analyzing multiple symbols, trade distribution will vary based on:
- **Symbol Volatility**: Higher volatility symbols typically generate more trading signals
- **Forecast Confidence**: Strategy filters based on forecast certainty
- **Risk Management**: Position sizing and risk controls affect trade frequency

Example distribution (TSLA, AAPL, MSFT, Feb-Mar 2025):
- TSLA: 71 trades (84.5%)
- AAPL: 12 trades (14.3%)
- MSFT: 1 trade (1.2%)

This variation is **normal behavior** and reflects the strategy's signal generation logic.

## Troubleshooting

### Common Issues

#### 1. "Forecasts not found" Error
**Cause**: Running backtest without completing previous pipeline steps
**Solution**: Ensure full pipeline runs: check-data → build-features → forecast → backtest

#### 2. No Trades Generated
**Cause**: Trading strategy parameters may not be suitable for selected symbols/timeframe
**Solution**:
- Check trading configuration in `configs/trading.yaml`
- Verify symbol data quality and availability
- Consider adjusting `k_sigma` (signal threshold) or other trading parameters

#### 3. Date Format Errors
**Cause**: Invalid date format
**Solution**: Use strict `YYYY-MM-DD` format (e.g., "2025-02-01")

#### 4. Invalid Ticker Symbols
**Cause**: Ticker not found in data source
**Solution**: Verify ticker symbols are valid and available in your GCS data bucket

### Debugging Commands

#### Check Data Loading
```bash
# Run check-data with verbose output to verify data loading
python -m timegpt_v2.cli check-data --tickers "TSLA" --start-date "2025-02-01" --end-date "2025-02-28" --run-id debug_data
```

#### Verify Forecasts
```bash
# Check if forecasts were generated for all symbols
grep -c "TSLA," artifacts/runs/debug_data/forecasts/quantiles.csv
grep -c "AAPL," artifacts/runs/debug_data/forecasts/quantiles.csv
```

#### Analyze Trade Distribution
```bash
# Count trades by symbol
head -n 1 artifacts/runs/debug_data/trades/bt_trades.csv && tail -n +2 artifacts/runs/debug_data/trades/bt_trades.csv | cut -d',' -f1 | sort | uniq -c
```

## Performance Considerations

### Execution Time
- **Multi-symbol analysis**: Longer execution time due to increased data processing
- **Date range**: Longer periods require more data loading and processing
- **Feature engineering**: Scales linearly with number of symbols and time period

### Resource Usage
- **Memory**: Scales with number of symbols and date range
- **Storage**: Each symbol generates separate forecast files and trade logs
- **API Usage**: Online mode consumes API quota proportional to symbols and snapshots

## Best Practices

### 1. Use Meaningful Run IDs
```bash
# Good: descriptive and timestamped
RUN_ID="tsla_feb2025_test_$(date +%Y%m%d_%H%M%S)"

# Bad: generic and uninformative
RUN_ID="test"
```

### 2. Validate with Small Tests First
```bash
# Start with short date range
RUN_ID="validation_test"
python -m timegpt_v2.cli check-data \
  --tickers "TSLA" --start-date "2025-02-01" --end-date "2025-02-07" --run-id "$RUN_ID"
```

### 3. Use Offline Mode for Development
```bash
# Save API costs during development
python -m timegpt_v2.cli forecast --run-id "$RUN_ID" --api-mode offline
```

### 4. Monitor Results
```bash
# Quick results check
python -c "
import pandas as pd
df = pd.read_csv('artifacts/runs/$RUN_ID/trades/bt_trades.csv')
print(f'Total trades: {len(df)}')
print(f'Symbols: {df[\"symbol\"].unique()}')
print(f'Date range: {df[\"entry_ts\"].min()} to {df[\"entry_ts\"].max()}')
"
```

## Migration from Universe Config Files

### Before (Custom Universe Config)
```bash
# 1. Create custom universe file: universe_custom.yaml
# 2. Run pipeline with custom config
python -m timegpt_v2.cli check-data --universe-name universe_custom.yaml --run-id test
# 3. Repeat for all pipeline steps
```

### After (CLI Parameters)
```bash
# 1. Run pipeline directly with CLI parameters
RUN_ID=test
python -m timegpt_v2.cli check-data \
  --tickers "TSLA,AAPL" --start-date "2025-02-01" --end-date "2025-03-31" --run-id "$RUN_ID"
# 2. Repeat for all pipeline steps with same parameters
```

**Benefits:**
- No need to create custom config files
- Easier to parameterize and automate
- Cleaner git history (no config file changes)
- Better for ad-hoc analysis and experimentation