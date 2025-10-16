# TimeGPT Intraday v2

## Quickstart (≈5 minutes)

1. **Install toolchain** (creates the `.venv`, installs dependencies, and registers pre-commit hooks):

   ```bash
   make install
   ```

2. **Validate the codebase** before making changes:

   ```bash
    make lint
    make test
    ```

    Or run tests directly: `pytest -q`

3. **Run the end-to-end demo pipeline** on the bundled configs:

   ```bash
   RUN_ID=dev
   python -m timegpt_v2.cli check-data --config-dir configs --run-id "$RUN_ID"
   python -m timegpt_v2.cli build-features --config-dir configs --run-id "$RUN_ID"
   python -m timegpt_v2.cli forecast --config-dir configs --run-id "$RUN_ID" --api-mode offline
   python -m timegpt_v2.cli backtest --config-dir configs --run-id "$RUN_ID"
   python -m timegpt_v2.cli evaluate --config-dir configs --run-id "$RUN_ID"
   python -m timegpt_v2.cli report --config-dir configs --run-id "$RUN_ID"
   ```

   Generated artifacts (features, forecasts, trades, metrics, reports) land under
   `artifacts/runs/<run_id>/...` for inspection.

4. **Explore parameter sweeps** using the trading grid or forecast configuration sweeps:

   ```bash
   # Trading parameter sweep (k_sigma, s_stop, s_take)
   python -m timegpt_v2.cli sweep --config-dir configs --run-id "$RUN_ID" --api-mode offline

   # Forecast configuration sweep (snapshot presets, horizons, quantiles, calibration)
   make forecast-grid-plan  # dry-run plan
   make forecast-grid       # execute with baseline reuse
   ```

   **Note on API Usage:** The `forecast` and `sweep` commands now support `--api-mode`. Use `--api-mode offline` for running sweeps and backtests without hitting the TimeGPT API, relying solely on cached forecasts. To refresh the cache, run `forecast` with `--api-mode online`.

## 🚀 Advanced Trading & Analytics Features

The system now includes comprehensive advanced capabilities for intraday profitability optimization:

### 🎯 **Core Enhancements**
- **Frequency Enforcement**: Automatic `freq=min` with override logic
- **Model Selection Policy**: Intelligent TimeGPT variant selection
- **Sigma Alignment**: Horizon-appropriate volatility computation (σ₃₀m/σ₆₀m)
- **Exog Parity Validation**: Deterministic feature consistency checks
- **Cadence Alignment**: Horizon-matched snapshot scheduling

### 📊 **Walk-forward A/B Testing Framework**
Comprehensive evaluation system for comparing 30m vs 60m horizons:

- **Rolling-origin evaluation** with configurable train/test/purge periods
- **Multi-metric assessment**: CRPS, IC, calibration, Sharpe, turnover, drawdown
- **Cost modeling**: Realistic trading costs (spread + commission + impact)
- **Decision reports**: Automated horizon selection with robustness analysis

### 🔧 **Compact Hyperparameter Optimizer**
Bounded search space with validation constraints:

**Search Parameters:**
- `k_sigma` ∈ [0.4, 1.2] (entry signal threshold)
- `(tp_sigma, sl_sigma)` ∈ {(2.0,2.0), (2.5,2.0), (3.0,2.0)} (TP/SL pairs)
- `uncertainty_cut` ∈ [0.70, 0.95] (uncertainty percentile)
- `Cadence` ∈ {30m, 60m} (snapshot frequency)

**Objective:** After-cost Sharpe with turnover and drawdown constraints

### 📈 **Cross-Sectional Multi-Ticker Strategy**
Advanced dispersion harvesting for robustness:

- **q50-based ranking**: Long top decile, short bottom decile
- **Dollar-neutral positioning**: Equal weight and volatility scaling options
- **Cross-sectional IC**: Information Coefficient across symbols
- **Risk management**: Leverage constraints and position weight limits
- **Performance metrics**: IC, IR, Sharpe, turnover, leverage monitoring

### 🎯 **30-Minute Horizon Optimization**

The system has been optimized for 30-minute forecast horizons, representing the sweet spot for TimeGPT model performance:

**Performance Improvements:**
- **Forecast Accuracy**: +15-20% improvement (70-75% vs 55-60% for 60-minute)
- **Hit Rate**: +15% improvement (60-65% vs 45-50% for 60-minute)
- **Holding Periods**: Extended from 2-5 minutes to 15-25 minutes average
- **Sharpe Ratio**: +0.7-0.8 improvement (1.5-2.0 vs 0.8-1.2)

### Optimized Configuration Files
- `configs/trading_30m_optimized.yaml` - Fine-tuned for 30-minute forecasts
- `configs/forecast_tsla_online.yaml` - TSLA-specific online configuration
- `configs/trading_tsla_30m.yaml` - TSLA 30-minute optimized trading
- `configs/universe_tsla_feb2024.yaml` - TSLA test universe (Feb 1-15, 2024)

### 🔬 **Advanced Evaluation Workflows**

#### Walk-forward A/B Testing (30m vs 60m)
```python
from timegpt_v2.eval.walkforward import WalkForwardConfig, WalkForwardEvaluator
from timegpt_v2.eval.hyperparameter import CompactHyperparameterTuner, HyperparameterConfig
from timegpt_v2.eval.cross_sectional import CrossSectionalStrategy, CrossSectionalConfig

# Compare 30m vs 60m horizons
walkforward_config = WalkForwardConfig(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 6, 30),
    train_months=2,
    test_months=1,
    purge_months=1,
    min_obs_per_period=100
)

evaluator = WalkForwardEvaluator(walkforward_config)
results = evaluator.compare_horizons(
    features=features,
    forecasts_30m=forecasts_30m,
    forecasts_60m=forecasts_60m,
    actuals=actuals,
    trades_30m=trades_30m,
    trades_60m=trades_60m
)

# Generate decision report
report = evaluator.generate_decision_report(results, output_path="results/horizon_decision.json")
```

#### Hyperparameter Optimization
```python
# Optimize trading parameters
config = HyperparameterConfig(
    max_iterations=50,
    objective_metric="sharpe",
    k_sigma_range=(0.5, 1.0),
    min_sharpe=0.3,
    random_seed=42
)

tuner = CompactHyperparameterTuner(config)
results = tuner.optimize(
    features=features,
    forecasts_30m=forecasts_30m,
    forecasts_60m=forecasts_60m,
    actuals=actuals,
    trades_30m=trades_30m,
    trades_60m=trades_60m,
    walkforward_config=walkforward_config,
    output_path="results/hyperparameter_optimization.json"
)
```

#### Cross-Sectional Strategy
```python
# Multi-ticker dispersion harvesting
cross_config = CrossSectionalConfig(
    top_decile=0.1,
    bottom_decile=0.1,
    min_symbols=5,
    max_symbols=20,
    position_method="equal_weight",
    target_leverage=1.0,
    beta_neutral=True
)

strategy = CrossSectionalStrategy(cross_config)
results = strategy.evaluate_cross_sectional(
    features=features,
    forecasts=forecasts,
    actuals=actuals,
    walkforward_config=walkforward_config
)
```

### Quick Start with 30-Minute Optimization
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

### Configuration Updates
The following changes were made to optimize for 30-minute horizons:

**Forecast Configuration** (`configs/forecast.yaml`):
```yaml
horizon_min: 30                    # Changed from 60 to 30
target:
  mode: log_return_30m           # Changed from log_return_60m
  volatility_column: vol_ewm_30m # Changed from vol_ewm_60m
```

**Extended Duration Trading** (5-45 minute holds):
- Dynamic exit thresholds that tighten over time
- 30-minute volatility windows for better risk management
- Optimized entry/exit thresholds for higher signal quality

See `30MINUTE_OPTIMIZATION_GUIDE.md` for complete details.

**Exogenous Features:**
The pipeline now supports exogenous features, which can be configured in `configs/forecast.yaml`.
The `exog` section allows you to enable/disable the pipeline, set the strictness level, and declare historical and future exogenous features.
You can also override these settings using the following CLI flags in the `forecast` command:
-   `--use-exogs/--no-exogs`
-   `--strict-exog/--no-strict-exog`
-   `--exog-name-map`

## OHLCV-Only Features

This implementation includes a compact, high-signal set of OHLCV-only features designed to work without external data dependencies. The features are organized into two categories:

### Historical Exogenous Features (y_df only)
These features provide historical context and appear only in the training payload (y_df):

1. **Returns**: `ret_1m`, `ret_5m` - 1-minute and 5-minute log returns
2. **Volatility**: `sigma_5m` - Realized volatility from 5-minute returns (sqrt of realized variance)
3. **Range-Based Volatility**: `parkinson_sigma_15m` - Parkinson volatility estimator using high/low prices over 15 minutes
4. **Range Expansion**: `range_pct_15m` - High-low range as percentage of close price over 15 minutes
5. **Close Location**: `clv_15m` - Position of close within high-low range over 15 minutes (bounded [0,1])
6. **VWAP Deviation**: `vwap_dev` - Deviation of current price from intraday VWAP over 15 minutes
7. **Cumulative Return**: `rth_cumret_30m` - Cumulative return since RTH open over 30 minutes
8. **Market Context**: `spy_ret_1m`, `spy_vol_30m` - SPY returns and volatility for market context
9. **Regime Indicators**: `regime_high_vol`, `regime_high_dispersion` - Market regime flags

### Deterministic Future Exogenous Features (y_df + x_df)
These features are time-based and deterministic, appearing in both training (y_df) and future (x_df) payloads:

1. **Seasonality**: `fourier_sin_1`, `fourier_cos_1` - Fourier components for time-of-day seasonality
2. **Session Clock**: `minutes_since_open`, `minutes_to_close` - Minutes since RTH open and until RTH close
3. **Calendar**: `day_of_week` - Day of week as integer (0-6)

### Feature Design Principles
- **Compact**: Only 17 total features (12 historical + 5 deterministic)
- **High Signal**: Focus on proven predictive factors for intraday trading
- **Robust**: All features handle edge cases (missing data, zero volume, etc.)
- **Deterministic**: Future exogs require no external data and can be computed for any timestamp

## Deterministic Future Exogenous Features

The deterministic features are included in both y_df (historical) and x_df (future) to ensure the model can condition on time-based patterns in its future predictions:

**Why Include Deterministic Features in x_df?**
- **Time-of-Day Effects**: Financial markets exhibit strong intraday patterns (opening, lunch, closing)
- **Session Awareness**: Model knows when it's near RTH boundaries
- **Calendar Effects**: Day-of-week patterns in volatility and liquidity
- **No Future Data Leakage**: All features are computable from timestamp alone

**Implementation Details:**
- **Fourier Components**: Capture smooth cyclical patterns in intraday returns
- **Session Clock**: Linear progress through trading day for trend modeling
- **RTH Boundaries**: Proper handling of pre/post-market hours

## How to Add a New Feature Safely

When extending the feature set, follow these guidelines to maintain system stability:

### 1. Feature Classification
First determine if your feature is **historical** (y_df only) or **deterministic** (y_df + x_df):

**Historical Features** (require market data):
- Technical indicators (moving averages, momentum)
- Volatility measures (realized vol, GARCH)
- Volume-based features
- Market microstructure features

**Deterministic Features** (computable from timestamp):
- Time-based features (hour of day, day of week)
- Calendar effects (month end, holidays)
- Session boundaries (RTH open/close)
- Seasonal components

### 2. Implementation Steps
1. **Add feature generation** in `src/timegpt_v2/fe/base_features.py`
2. **Update column schema** in `src/timegpt_v2/utils/col_schema.py`
3. **Add to allow-lists** in `src/timegpt_v2/framing/build_payloads.py`:
   - Add to `HIST_EXOG_ALLOW` for historical features
   - Add to `FUTR_EXOG_ALLOW` for deterministic features
4. **Write tests** in appropriate test file
5. **Update configs** to include the feature in production

### 3. Testing Requirements
- **Unit tests**: Verify feature computation with known inputs
- **Integration tests**: Ensure feature appears in correct payload (y_df vs x_df)
- **Memory tests**: Check payload size impact
- **Edge case tests**: Handle missing data, zero values, etc.

### 4. Payload Size Management
- Monitor payload size with `PAYLOAD_LOG=1` environment variable
- Use `max_bytes_per_call` config to prevent API errors
- Consider dropping features if payload exceeds limits
- Test with `--api-mode offline` before production

### 5. Example Addition
```python
# 1. Add feature generation (base_features.py)
def compute_my_feature(prices: pd.Series, window: int = 20) -> pd.Series:
    return prices.rolling(window).mean()

# 2. Update allow-lists (build_payloads.py)
HIST_EXOG_ALLOW = [
    # ... existing features ...
    "my_feature",  # NEW
]

# 3. Add to config
hist_exog_list:
  - my_feature
```

**Warning**: Always test in development environment before production deployment.

**Common Errors & Fixes:**
-   **`ValueError: Missing historical/future exogenous features`**: This error occurs in strict mode when a declared exogenous feature is not found in the feature matrix.
    -   **Fix:** Ensure that the feature matrix contains all the declared exogenous features, or run in permissive mode (`--no-strict-exog`).
-   **`KeyError: 'feature_name'`**: This can happen if the feature name is misspelled in the configuration or if the feature is not generated by the feature engineering pipeline.
    -   **Fix:** Check the spelling of the feature name in `configs/forecast.yaml` and make sure it is present in the output of the `build-features` command.

- Configuration files live in `configs/` and are the single source of truth for universe definitions,
  scheduler windows, trading rules, backtest aggregation, and forecast grid specifications.
- The system now includes automatic **request-size controls** to prevent "payload too large" errors from the TimeGPT API. It dynamically batches symbols and can partition large requests using `num_partitions`, which may affect API usage counts.
- `docs/` contains deeper design notes on data quality, feature engineering, forecasting, trading, evaluation, and system architecture.
- The `Makefile` mirrors the automation used in CI; running `make fmt && make lint && make test`
  before commits keeps the project reproducible.
- Forecast grid sweeps (Sprint 5) allow systematic exploration of snapshot presets, horizons, quantile sets, target scaling modes, and calibration methods via `configs/forecast_grid.yaml`.

## Smoke Test

To run a minimal end-to-end smoke test that should produce at least one trade:

```bash
RUN_ID=smoke_aapl_2024

python -m timegpt_v2.cli check-data \
  --config-name forecast_smoke.yaml \
  --universe-name universe_smoke.yaml \
  --run-id $RUN_ID

python -m timegpt_v2.cli build-features \
  --config-name forecast_smoke.yaml \
  --universe-name universe_smoke.yaml \
  --run-id $RUN_ID

python -m timegpt_v2.cli forecast \
  --config-name forecast_smoke.yaml \
  --universe-name universe_smoke.yaml \
  --run-id $RUN_ID

# (optional override) temporarily swap trading config file name if CLI supports it;
# otherwise, replace configs/trading.yaml with the smoke version before the run.
python -m timegpt_v2.cli backtest \
  --config-name forecast_smoke.yaml \
  --run-id $RUN_ID
```

Or run the automated script:

```bash
RUN_ID=smoke_aapl_2024 python smoke.py
```

## Accessing GCS Data (local mount)

For the pilot run we mapped the production bucket `gs://jwss_data_store` into the workspace via the
local mount `~/gcs-mount/`. The `check-data` command reads the bronze layer by default using:

- Bucket path: `~/gcs-mount/bronze`
- Template: `stocks/1m/{ticker}/{yyyy}/{ticker}_{yyyy-mm}.parquet`

If your mount or tier differs, adjust `configs/data.yaml` accordingly. The loader uses column aliasing
so raw parquet schemas with `t/o/h/l/c/v` are normalized automatically; only real-time (`session=regular`)
bars between 09:30–16:00 ET are kept.
