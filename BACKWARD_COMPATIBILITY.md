# Trading Window Feature - Backward Compatibility Guide

## Overview

The new trading window enforcement feature is designed to be fully backward compatible with existing TimeGPT v2 configurations. This guide explains how existing configurations will be handled and how to migrate to the new format when ready.

## Backward Compatibility Guarantees

### 1. Legacy Configuration Support
Existing configurations using the old `dates` format will continue to work without any changes:

```yaml
# Existing format (continues to work)
universe:
  tickers: ["AAPL", "MSFT"]
  dates:
    start: "2024-01-01"
    end: "2024-12-31"
```

**Behavior**: The system will automatically migrate this to the new trading window format internally, with `enforce_trading_window` set to `False` (permissive mode) to maintain existing behavior.

### 2. Automatic Migration Logic
When legacy configurations are detected:

1. **Trading Window Creation**: Automatically creates `TradingWindowConfig` from legacy `dates`
2. **Permissive Mode**: Sets `enforce_trading_window: false` to maintain current behavior
3. **Warning Messages**: Logs clear warnings about permissive mode usage
4. **Graceful Fallback**: Falls back to legacy behavior if parsing fails

### 3. API Compatibility
All existing CLI commands and Python APIs continue to work unchanged:

```bash
# All existing commands continue to work
python -m timegpt_v2.cli check-data --config-dir configs --run-id legacy_run
python -m timegpt_v2.cli forecast --config-dir configs --run-id legacy_run
python -m timegpt_v2.cli backtest --config-dir configs --run-id legacy_run
```

## Migration Path

### Step 1: Enable Enforcement (Optional)
When ready to enforce trading windows, add the new configuration:

```yaml
# New format (recommended)
universe:
  tickers: ["AAPL", "MSFT"]
  trading_window:
    start: "2024-01-01"
    end: "2024-12-31"
    history_backfill_days: 90
    enforce_trading_window: true  # Enable enforcement
```

### Step 2: Configure History Backfill (Optional)
Fine-tune the historical data loading separate from trading window:

```yaml
universe:
  tickers: ["AAPL", "MSFT"]
  trading_window:
    start: "2024-01-01"
    end: "2024-12-31"
    history_backfill_days: 180  # Load 6 months of history
    enforce_trading_window: true
```

### Step 3: Verify Compliance
Use the new `doctor` command to verify trading window compliance:

```bash
python -m timegpt_v2.cli doctor --run-id my_run --save-report
```

## Configuration Mapping

### Legacy → New Format

| Legacy Field | New Field | Behavior |
|-------------|-----------|----------|
| `universe.dates.start` | `universe.trading_window.start` | Trading window start date |
| `universe.dates.end` | `universe.trading_window.end` | Trading window end date |
| `forecast.rolling_history_days` | `universe.trading_window.history_backfill_days` | History backfill period |
| N/A | `universe.trading_window.enforce_trading_window` | Enforcement flag (defaults to `false`) |

### Mixed Configuration Support
Both legacy and new formats can be mixed - the system will merge configurations intelligently:

```yaml
# This works - will be merged intelligently
universe:
  tickers: ["AAPL", "MSFT"]
  dates:
    start: "2024-01-01"  # Legacy format
  trading_window:
    enforce_trading_window: true  # New format

forecast:
  rolling_history_days: 90  # Will be overridden by trading_window.history_backfill_days if set
```

## Behavior Changes

### Without Migration (Permissive Mode)
- ✅ All existing functionality preserved
- ✅ Trading window violations logged but not enforced
- ✅ Historical data loading behavior unchanged
- ⚠️ Warning messages about permissive mode

### With Migration (Enforcement Mode)
- ✅ Trading windows strictly enforced
- ✅ Historical data loading separated from trading window
- ✅ Better control over data ranges and trading periods
- ✅ Comprehensive diagnostics and reporting

## Troubleshooting

### Issue: "Permissive mode" warnings
**Cause**: Using legacy configuration format
**Solution**: Either migrate to new format or explicitly set `enforce_trading_window: false`

### Issue: Missing data after migration
**Cause**: Trading window too restrictive or insufficient `history_backfill_days`
**Solution**: Increase `history_backfill_days` or expand trading window dates

### Issue: Fewer trades than expected
**Cause**: Trading window enforcement removing out-of-window trades
**Solution**: Use `doctor` command to analyze compliance, adjust trading window if needed

## Testing Migration

### 1. Test in Permissive Mode First
```bash
# Test with existing config (should work identically)
RUN_ID=test_migration python -m timegpt_v2.cli forecast --config-dir configs

# Compare results with baseline
```

### 2. Enable Enforcement Gradually
```yaml
# Start with generous trading window
trading_window:
  start: "2023-12-01"  # Start earlier
  end: "2025-01-31"    # End later
  enforce_trading_window: true
```

### 3. Use Doctor Command for Validation
```bash
# Analyze compliance
python -m timegpt_v2.cli doctor --run-id test_migration --output-format json --save-report
```

## Rollback Plan

If issues arise after migration:

1. **Immediate Rollback**: Set `enforce_trading_window: false`
2. **Configuration Rollback**: Revert to legacy `dates` format
3. **Data Verification**: Use `doctor` command to compare behavior

```yaml
# Rollback configuration
universe:
  tickers: ["AAPL", "MSFT"]
  trading_window:
    enforce_trading_window: false  # Disable enforcement
```

## Support

For migration issues:
1. Check logs for permissive mode warnings
2. Use `doctor` command for diagnostic analysis
3. Compare pre/post migration results
4. Review trading window configuration carefully

The migration is designed to be safe and reversible, with clear diagnostic feedback at every step.