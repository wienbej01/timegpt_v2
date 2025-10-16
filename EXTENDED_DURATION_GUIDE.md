# Extended Duration Trading Guide (10-90 Minute Holding Periods)

## 🎯 Problem Solved

The original system was taking very short-term positions due to:
1. **Tight exit thresholds**: 1-2x 5-minute volatility caused premature exits
2. **Short-term volatility scaling**: Using `sigma_5m` made exits hypersensitive to noise
3. **Early time stop**: Forced exit at 15:55 cut trading day short
4. **No minimum holding period**: Immediate exits allowed on minor price movements

## 🚀 Solution Overview

I've implemented a comprehensive extended duration trading system with:

### **Key Features**
- **10-90 minute holding periods** with enforced minimum/maximum limits
- **Dynamic volatility scaling** that relaxes exit thresholds over time
- **Longer-term volatility measures** (15m, 30m, 60m, 90m) instead of 5-minute
- **Intelligent exit logic** that balances risk with holding period goals
- **Enhanced monitoring** with detailed duration analytics

## 📁 New Files Created

### **1. Extended Trading Configuration**
- `configs/trading_extended.yaml` - Optimized for longer holds
- Relaxed stop-loss (3-5x volatility vs 1-2x)
- Extended take-profit (3-4x volatility vs 1x)
- Time stop moved to 16:00 (market close)

### **2. Extended Trading Rules**
- `src/timegpt_v2/trading/extended_rules.py` - New duration-aware logic
- `ExtendedRuleParams` with min/max holding periods
- `ExtendedTradingRules` with dynamic exit scaling
- Comprehensive exit reason tracking

### **3. Extended Backtest Simulator**
- `ExtendedBacktestSimulator` in `src/timegpt_v2/backtest/simulator.py`
- Enhanced volatility window selection
- Minimum holding period enforcement
- Detailed duration analytics

### **4. Enhanced Feature Engineering**
- Added `vol_ewm_30m` and `vol_ewm_90m` volatility measures
- Updated column schema to support extended features
- Backward compatibility maintained

## 🔧 How It Works

### **1. Dynamic Volatility Scaling**
```python
# Scales volatility based on intended holding period
scaling_factor = min(holding_minutes / 15.0, 3.0)  # Cap at 3x
scaled_volatility = sigma_5m * scaling_factor
```

### **2. Time-Relaxed Exit Thresholds**
```python
# Exit thresholds relax over longer holds
time_factor = min(holding_minutes / 30.0, 2.0)
effective_s_stop = base_s_stop * time_factor
effective_s_take = base_s_take * time_factor
```

### **3. Minimum Holding Period Enforcement**
- Trades forced to hold minimum 10 minutes
- Violations automatically extended to meet requirement
- Prevents whipsaw exits on short-term noise

### **4. Volatility Window Selection**
```python
volatility_col_map = {
    15: "vol_ewm_30m",  # 15-min window uses 30-min volatility
    30: "vol_ewm_60m",  # 30-min window uses 60-min volatility
    60: "vol_ewm_60m",  # 60-min window uses 60-min volatility
    90: "vol_ewm_90m",  # 90-min window uses 90-min volatility
}
```

## 🎮 Usage Instructions

### **Option 1: Use Extended Configuration**
```bash
# Use the extended trading configuration
RUN_ID=extended_duration_test

python -m timegpt_v2.cli check-data --config-dir configs --run-id "$RUN_ID" --universe-name universe_feb2025.yaml
python -m timegpt_v2.cli build-features --config-dir configs --run-id "$RUN_ID" --universe-name universe_feb2025.yaml
python -m timegpt_v2.cli forecast --config-dir configs --run-id "$RUN_ID" --universe-name universe_feb2025.yaml --api-mode offline
# Use extended trading config
python -m timegpt_v2.cli backtest --config-name trading_extended.yaml --universe-name universe_feb2025.yaml --run-id "$RUN_ID"
python -m timegpt_v2.cli evaluate --config-dir configs --run-id "$RUN_ID" --universe-name universe_feb2025.yaml
python -m timegpt_v2.cli report --config-dir configs --run-id "$RUN_ID" --universe-name universe_feb2025.yaml
```

### **Option 2: Modify Existing Config**
Edit `configs/trading.yaml`:
```yaml
# Extended duration parameters
s_stop: [3.0, 4.0, 5.0]  # Wider stop-loss
s_take: [3.0, 4.0]  # Wider take-profit
time_stop_et: "16:00"  # Trade until market close
daily_trade_cap: 2  # Fewer, longer trades
```

### **Option 3: Code Integration**
```python
from timegpt_v2.trading.extended_rules import ExtendedRuleParams, ExtendedTradingRules
from timegpt_v2.backtest.simulator import ExtendedBacktestSimulator

# Create extended parameters
params = ExtendedRuleParams(
    k_sigma=1.0,
    s_stop=4.0,  # 4x volatility stop-loss
    s_take=3.0,  # 3x volatility take-profit
    uncertainty_k=5.0,
    min_holding_minutes=10,
    max_holding_minutes=90,
    volatility_window=15,  # Use 15-minute volatility
    exit_relaxation_factor=2.0
)

# Create extended rules
rules = ExtendedTradingRules(
    costs=trading_costs,
    time_stop=time(16, 0),
    daily_trade_cap=2,
    max_open_per_symbol=1
)

# Run extended backtest
simulator = ExtendedBacktestSimulator(rules=params, params=params)
trades_df, summary_df = simulator.run(forecasts, features, prices)
```

## 📊 Enhanced Analytics

### **New Duration Metrics**
- `avg_hold_minutes` - Average holding period
- `median_hold_minutes` - Median holding period
- `min_hold_minutes` - Shortest trade
- `max_hold_minutes` - Longest trade
- `trades_under_10min` - Trades < 10 minutes (should be 0)
- `trades_10_30min` - Trades in 10-30 minute range
- `trades_30_60min` - Trades in 30-60 minute range
- `trades_over_60min` - Trades > 60 minutes

### **Exit Reason Tracking**
- `take_profit_{minutes}min` - Take profit exit
- `stop_loss_{minutes}min` - Stop loss exit
- `min_hold_period` - Enforced minimum hold
- `max_hold_period` - Maximum holding period reached
- `time_stop` - Market close exit
- `extended_to_min_hold` - Extended to meet minimum

## ⚙️ Configuration Tuning

### **Conservative (Safer)**
```yaml
min_holding_minutes: 15
max_holding_minutes: 60
s_stop: [4.0, 5.0]  # Tighter stops
s_take: [2.5, 3.0]  # Earlier profits
volatility_window: 30  # Use longer volatility
```

### **Aggressive (Higher Risk/Return)**
```yaml
min_holding_minutes: 10
max_holding_minutes: 90
s_stop: [5.0, 6.0]  # Wider stops
s_take: [4.0, 5.0]  # Later profits
volatility_window: 15  # Use shorter volatility
```

### **Balanced (Recommended Start)**
```yaml
min_holding_minutes: 10
max_holding_minutes: 90
s_stop: [3.0, 4.0, 5.0]
s_take: [3.0, 4.0]
volatility_window: 15
uncertainty_k: [5.0, 6.0]
```

## 🎯 Expected Benefits

1. **Reduced Noise Trading**: Longer volatility windows reduce sensitivity to short-term price fluctuations
2. **Better Forecast Edge**: 60-minute forecasts have more time to materialize
3. **Lower Transaction Costs**: Fewer trades with longer duration
4. **Improved Risk Management**: Dynamic exits balance risk with holding goals
5. **Enhanced Analytics**: Detailed duration tracking for optimization

## ⚠️ Important Notes

- **Feature Rebuilding**: Need to rebuild features after adding new volatility columns
- **Backtesting**: Extended duration system requires `ExtendedBacktestSimulator`
- **Parameter Tuning**: Start conservative and adjust based on results
- **Market Hours**: Extended time stop allows full trading day utilization
- **Risk Management**: Longer holds increase exposure to market movements

## 🔍 Validation

Run the extended system and compare:
1. **Holding Period Distribution**: Check trades are in 10-90 minute range
2. **Hit Rate**: Should improve with better forecast edge utilization
3. **Average PnL per Trade**: Should increase with longer duration
4. **Number of Trades**: Should decrease (higher quality trades)
5. **Volatility Scaling**: Verify exits are less sensitive to short-term noise

This extended duration system addresses the core issue of short-term trading by implementing sophisticated time-aware exit logic that allows your 60-minute forecasts to fully materialize while managing risk appropriately.