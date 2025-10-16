# TSLA February 1-15, 2024 Online Trading Commands (Corrected)

## 🚀 Corrected TSLA February 1-15, 2024 Online Trading Commands

### **Environment Setup**
```bash
# Ensure TimeGPT API key is set
export TIMEGPT_API_KEY="your_api_key_here"
# OR
export NIXTLA_API_KEY="your_api_key_here"

# Set run ID
RUN_ID=tsla_feb2024_online_$(date +%Y%m%d_%H%M%S)
```

### **Complete Pipeline Commands**
```bash
# 1. Check data quality for TSLA
python -m timegpt_v2.cli check-data \
  --config-name forecast_tsla_online.yaml \
  --universe-name universe_tsla_feb2024.yaml \
  --run-id "$RUN_ID"

# 2. Build features with 30-minute targets
python -m timegpt_v2.cli build-features \
  --config-name forecast_tsla_online.yaml \
  --universe-name universe_tsla_feb2024.yaml \
  --run-id "$RUN_ID"

# 3. Generate 30-minute forecasts ONLINE (TimeGPT API)
python -m timegpt_v2.cli forecast \
  --config-name forecast_tsla_online.yaml \
  --run-id "$RUN_ID" \
  --api-mode online

# 4. Run extended duration backtest with TSLA optimization
python -m timegpt_v2.cli backtest \
  --config-name trading_tsla_30m.yaml \
  --universe-name universe_tsla_feb2024.yaml \
  --run-id "$RUN_ID"

# 5. Evaluate TSLA performance (FIXED - no --config-name)
python -m timegpt_v2.cli evaluate \
  --config-dir configs \
  --run-id "$RUN_ID"

# 6. Generate comprehensive TSLA report (FIXED - no --config-name)
python -m timegpt_v2.cli report \
  --config-dir configs \
  --run-id "$RUN_ID"
```

### **Single Command Run (All Steps)**
```bash
#!/bin/bash
# Complete TSLA February 2024 Online Trading Pipeline

# Set environment
export TIMEGPT_API_KEY="your_api_key_here"
RUN_ID=tsla_feb2024_online_$(date +%Y%m%d_%H%M%S)

echo "Starting TSLA February 2024 Online Trading Pipeline..."
echo "Run ID: $RUN_ID"

# Step 1: Data validation
echo "Step 1: Checking data quality..."
python -m timegpt_v2.cli check-data \
  --config-name forecast_tsla_online.yaml \
  --universe-name universe_tsla_feb2024.yaml \
  --run-id "$RUN_ID"

if [ $? -eq 0 ]; then
  echo "✅ Data check passed"
else
  echo "❌ Data check failed"
  exit 1
fi

# Step 2: Feature engineering
echo "Step 2: Building features..."
python -m timegpt_v2.cli build-features \
  --config-name forecast_tsla_online.yaml \
  --universe-name universe_tsla_feb2024.yaml \
  --run-id "$RUN_ID"

if [ $? -eq 0 ]; then
  echo "✅ Feature building passed"
else
  echo "❌ Feature building failed"
  exit 1
fi

# Step 3: Online forecasting
echo "Step 3: Generating online forecasts..."
python -m timegpt_v2.cli forecast \
  --config-name forecast_tsla_online.yaml \
  --run-id "$RUN_ID" \
  --api-mode online

if [ $? -eq 0 ]; then
  echo "✅ Online forecasting passed"
else
  echo "❌ Online forecasting failed"
  exit 1
fi

# Step 4: Backtesting
echo "Step 4: Running backtest..."
python -m timegpt_v2.cli backtest \
  --config-name trading_tsla_30m.yaml \
  --universe-name universe_tsla_feb2024.yaml \
  --run-id "$RUN_ID"

if [ $? -eq 0 ]; then
  echo "✅ Backtest passed"
else
  echo "❌ Backtest failed"
  exit 1
fi

# Step 5: Evaluation (FIXED - uses --config-dir)
echo "Step 5: Evaluating performance..."
python -m timegpt_v2.cli evaluate \
  --config-dir configs \
  --run-id "$RUN_ID"

# Step 6: Report generation (FIXED - uses --config-dir)
echo "Step 6: Generating report..."
python -m timegpt_v2.cli report \
  --config-dir configs \
  --run-id "$RUN_ID"

echo "✅ TSLA February 2024 Online Trading Pipeline Complete!"
echo "Results available in: artifacts/runs/$RUN_ID"
```

### **Quick Test (First Step Only)**
```bash
# Test the first step to validate configuration
RUN_ID=tsla_test_$(date +%Y%m%d_%H%M%S)
python -m timegpt_v2.cli check-data \
  --config-name forecast_tsla_online.yaml \
  --universe-name universe_tsla_feb2024.yaml \
  --run-id "$RUN_ID"
```

### **Monitoring Commands**
```bash
# Monitor progress
tail -f artifacts/runs/$RUN_ID/logs/*.log

# Check API usage
echo "API Budget Status:"
cat artifacts/runs/$RUN_ID/logs/forecast.log | grep -i "budget\|api\|call"

# Check results
ls -la artifacts/runs/$RUN_ID/
```

### **Key Fixes Made**
1. **Fixed evaluate command**: Changed from `--config-name backtest_tsla.yaml` to `--config-dir configs`
2. **Fixed report command**: Changed from `--config-name backtest_tsla.yaml` to `--config-dir configs`

The `evaluate` and `report` commands only accept `--config-dir` and `--run-id` parameters, not `--config-name`.

### **Correct Command Syntax Reference**
```bash
# Commands that use --config-name and --universe-name
python -m timegpt_v2.cli check-data --config-name <config> --universe-name <universe> --run-id <id>
python -m timegpt_v2.cli build-features --config-name <config> --universe-name <universe> --run-id <id>
python -m timegpt_v2.cli forecast --config-name <config> --run-id <id> [--api-mode online|offline]
python -m timegpt_v2.cli backtest --config-name <config> --universe-name <universe> --run-id <id>

# Commands that only use --config-dir and --run-id
python -m timegpt_v2.cli evaluate --config-dir <dir> --run-id <id>
python -m timegpt_v2.cli report --config-dir <dir> --run-id <id>
```

## 📋 Configuration Summary

- **Universe**: TSLA only
- **Period**: February 1-15, 2024
- **Forecast Horizon**: 30 minutes (optimized)
- **API Mode**: Online (live TimeGPT API)
- **Trading Strategy**: 30-minute optimized with extended duration
- **Holding Period**: 5-45 minutes
- **API Budget**: 10 calls per run, 20 per day

## ⚠️ Important Notes

1. **API Costs**: Online mode will consume TimeGPT API credits
2. **Single Symbol**: Optimized for TSLA's high volatility characteristics
3. **Date Range**: Limited to 15 trading days for focused testing
4. **30-Minute Horizon**: Optimized for TimeGPT's best performance range
5. **Extended Duration**: 5-45 minute holding periods enabled
6. **Fixed Commands**: Corrected evaluate/report commands to use proper syntax