# Advanced Trading & Analytics Features

This document describes the advanced capabilities added to TimeGPT Intraday v2 for comprehensive profitability optimization and robustness analysis.

## Table of Contents

1. [Core Enhancements](#core-enhancements)
2. [Walk-forward A/B Testing](#walk-forward-ab-testing)
3. [Hyperparameter Optimization](#hyperparameter-optimization)
4. [Cross-sectional Trading](#cross-sectional-trading)
5. [Performance Metrics](#performance-metrics)
6. [Usage Examples](#usage-examples)
7. [Configuration Reference](#configuration-reference)

---

## Core Enhancements

### Frequency Enforcement

**Problem**: Inconsistent frequency inference leading to horizon warnings and model misuse.

**Solution**: Automatic frequency detection and enforcement with override capabilities.

```python
# Automatic frequency enforcement (src/timegpt_v2/forecast/timegpt_client.py)
if freq_value != "min":
    self._logger.warning("Frequency override: forcing freq='min' for intraday forecasts")
    freq_value = "min"
```

**Benefits**:
- Eliminates Nixtla horizon warnings
- Consistent model behavior across runs
- Automatic handling of irregular data

### Model Selection Policy

**Problem**: Manual model selection leading to suboptimal performance for intraday horizons.

**Solution**: Intelligent model selection based on horizon length.

```python
# Model selection logic (src/timegpt_v2/forecast/timegpt_client.py)
if horizon_minutes <= 60:
    preferred_model = "timegpt-1"  # Better for short-term intraday
else:
    preferred_model = "timegptpt-long-horizon"  # For longer horizons
```

**Benefits**:
- Automatic optimal model selection
- Improved forecast accuracy for target horizons
- Configurable overrides with warnings

### Sigma Alignment

**Problem**: Mismatched volatility windows causing inconsistent risk management.

**Solution**: Horizon-aligned sigma computation (σ₃₀m/σ₆₀m).

```python
# Sigma computation (src/timegpt_v2/utils/sigma_utils.py)
def compute_sigma_aligned(returns, horizon_minutes):
    """Compute sigma aligned with forecast horizon."""
    base_window = 5  # 5-minute base sigma
    scaling_factor = np.sqrt(horizon_minutes / base_window)
    return returns.rolling(base_window).std() * scaling_factor
```

**Benefits**:
- Consistent volatility measurement across horizons
- Proper risk-adjusted position sizing
- Aligned entry/exit logic

### Exogenous Feature Parity

**Problem**: Inconsistent deterministic features between training and inference payloads.

**Solution**: Comprehensive parity validation with preflight checks.

```python
# Parity validation (src/timegpt_v2/framing/build_payloads.py)
def _validate_exog_parity(y_df, x_df, horizon_minutes):
    """Validate exogenous feature parity between y_df and x_df."""
    required_deterministic = ["minute_of_day_sin", "minute_of_day_cos",
                           "minutes_since_open", "day_of_week"]

    for feature in required_deterministic:
        if feature not in y_df.columns or feature not in x_df.columns:
            raise ValueError(f"Deterministic feature '{feature}' missing in payload")
```

**Benefits**:
- Prevents model confusion from mismatched features
- Ensures consistent time-based patterns
- Early error detection with actionable messages

---

## Walk-forward A/B Testing

### Overview

Comprehensive framework for comparing different forecast horizons and trading strategies using rolling-origin evaluation with out-of-sample testing.

### Key Components

#### WalkForwardConfig
Configuration for walk-forward evaluation periods:

```python
walkforward_config = WalkForwardConfig(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 6, 30),
    train_months=2,        # Training period length
    test_months=1,         # Test period length
    purge_months=1,        # Gap between train and test
    costs=CostConfig(      # Trading cost model
        half_spread_bps=2.5,
        commission_bps=1.0,
        impact_bps=0.5
    )
)
```

#### WalkForwardEvaluator
Main engine for running walk-forward evaluations:

```python
evaluator = WalkForwardEvaluator(walkforward_config)

# Compare 30m vs 60m horizons
results = evaluator.compare_horizons(
    features=features,
    forecasts_30m=forecasts_30m,
    forecasts_60m=forecasts_60m,
    actuals=actuals,
    trades_30m=trades_30m,
    trades_60m=trades_60m
)

# Generate decision report
report = evaluator.generate_decision_report(
    results=results,
    output_path="results/horizon_decision.json"
)
```

### Metrics Calculated

#### Forecast Metrics
- **CRPS**: Continuous Ranked Probability Score
- **IC**: Information Coefficient (q50 vs realized)
- **Calibration**: Pit coverage validation

#### Trading Metrics
- **Sharpe**: Risk-adjusted returns (after costs)
- **Max Drawdown**: Maximum portfolio drawdown
- **Hit Rate**: Percentage of profitable trades
- **Turnover**: Average trades per day
- **Total PnL**: Cumulative profit/loss

### Decision Report Structure

```json
{
  "evaluation_summary": {
    "winner": "30m",
    "primary_metric": "sharpe",
    "total_forecasts_30m": 1200,
    "total_forecasts_60m": 600,
    "total_trades_30m": 240,
    "total_trades_60m": 150
  },
  "forecast_metrics": {
    "30m": {"crps": 0.0012, "ic_mean": 0.045, "coverage": 0.52},
    "60m": {"crps": 0.0015, "ic_mean": 0.038, "coverage": 0.48}
  },
  "trading_metrics": {
    "30m": {"sharpe": 1.8, "max_drawdown": -0.08, "hit_rate": 0.62},
    "60m": {"sharpe": 1.2, "max_drawdown": -0.12, "hit_rate": 0.54}
  },
  "recommendations": {
    "selected_horizon": "30m",
    "rationale": "Selected 30m horizon based on higher after-cost Sharpe ratio",
    "key_metrics": {
      "sharpe_advantage": 0.6,
      "crps_advantage": 0.0003,
      "turnover_advantage": 2.0
    }
  }
}
```

---

## Hyperparameter Optimization

### Overview

Compact hyperparameter tuner with bounded search space and validation constraints for robust parameter selection.

### Search Space

```python
config = HyperparameterConfig(
    k_sigma_range=(0.4, 1.2),           # Entry signal threshold
    tp_sl_pairs=[(2.0, 2.0), (2.5, 2.0), (3.0, 2.0)],  # TP/SL pairs
    uncertainty_cut_range=(0.70, 0.95),      # Uncertainty percentile
    cadence_options=[30, 60],             # Snapshot frequency
    max_iterations=50,                      # Optimization budget
    objective_metric="sharpe",               # Optimization target
    constraints={                          # Risk constraints
        "max_turnover_per_day": 12.0,
        "min_sharpe": 0.3,
        "max_drawdown_threshold": -0.25,
        "min_periods": 3
    }
)
```

### Optimization Process

1. **Parameter Generation**: Random sampling within bounded ranges
2. **Walk-forward Evaluation**: Out-of-sample performance testing
3. **Validation**: Constraint checking (turnover, Sharpe, drawdown)
4. **Selection**: Best parameters based on objective metric
5. **Robustness Analysis**: Variance across walk-forward folds

### Usage Example

```python
tuner = CompactHyperparameterTuner(config)

results = tuner.optimize(
    features=features,
    forecasts_30m=forecasts_30m,
    forecasts_60m=forecasts_60m,
    actuals=actuals,
    trades_30m=trades_30m,
    trades_60m=trades_60m,
    walkforward_config=walkforward_config,
    output_path="results/optimization.json"
)

# Access best parameters
best_params = results["best_parameters"]
print(f"Best Sharpe: {best_params['sharpe']:.3f}")
print(f"Best k_sigma: {best_params['k_sigma']:.2f}")
print(f"Best TP/SL: ({best_params['tp_sigma']:.1f}, {best_params['sl_sigma']:.1f})")
```

### Result Structure

```json
{
  "best_parameters": {
    "k_sigma": 0.85,
    "tp_sigma": 2.5,
    "sl_sigma": 2.0,
    "uncertainty_cut": 0.82,
    "cadence_minutes": 30,
    "sharpe": 1.45,
    "is_valid": true,
    "validation_reasons": []
  },
  "optimization_summary": {
    "total_iterations": 50,
    "valid_combinations": 12,
    "success_rate": 0.24,
    "objective_metric": "sharpe"
  },
  "top_combinations": [
    {
      "k_sigma": 0.85, "tp_sigma": 2.5, "sl_sigma": 2.0,
      "uncertainty_cut": 0.82, "cadence_minutes": 30, "sharpe": 1.45
    }
  ]
}
```

---

## Cross-sectional Trading

### Overview

Multi-ticker dispersion harvesting strategy for improved robustness through diversification.

### Strategy Components

#### Symbol Ranking
- **Metric**: q50 forecast ranking
- **Selection**: Long top decile, short bottom decile
- **Filtering**: Minimum confidence threshold

#### Position Sizing
- **Equal Weight**: Equal allocation across selected symbols
- **Volatility Scaling**: Risk-adjusted position sizing
- **Constraints**: Maximum position weight, leverage limits

#### Risk Management
- **Beta Neutral**: Long/short balance for market neutrality
- **Sector Neutral**: Optional sector balance (if sector data available)
- **Leverage Control**: Gross and net exposure monitoring

### Configuration

```python
cross_config = CrossSectionalConfig(
    top_decile=0.1,           # Top 10% for long positions
    bottom_decile=0.1,        # Bottom 10% for short positions
    min_symbols=5,             # Minimum symbols for strategy
    max_symbols=50,            # Maximum symbols to avoid over-diversification
    position_method="equal_weight", # Position sizing method
    notional_per_symbol=10000.0,    # Base notional per symbol
    max_position_weight=0.2,        # Maximum 20% per symbol
    target_leverage=1.0,          # Target gross leverage
    beta_neutral=True,           # Enforce beta neutrality
    min_forecast_confidence=0.7    # Minimum confidence threshold
)
```

### Usage Example

```python
strategy = CrossSectionalStrategy(cross_config)

# Evaluate cross-sectional strategy
results = strategy.evaluate_cross_sectional(
    features=features,
    forecasts=forecasts,
    actuals=actuals,
    walkforward_config=walkforward_config
)

# Access results
print(f"Cross-sectional Sharpe: {results.sharpe:.3f}")
print(f"Cross-sectional IC: {results.ic_mean:.3f}")
print(f"Number of symbols: {results.n_symbols}")
print(f"Gross leverage: {results.leverage_gross:.2f}")
print(f"Net leverage: {results.leverage_net:.2f}")
```

### Performance Metrics

#### Cross-sectional Metrics
- **IC Mean**: Average information coefficient across symbols
- **IC IR**: Information ratio (IC / IC std)
- **Rank IC**: Rank-based information coefficient

#### Portfolio Metrics
- **Sharpe**: Risk-adjusted returns (long-short portfolio)
- **Max Drawdown**: Maximum portfolio drawdown
- **Hit Rate**: Percentage of profitable periods
- **Turnover**: Portfolio rebalancing frequency
- **Leverage**: Gross and net exposure monitoring

### Risk Management Features

#### Position Weight Limits
```python
# Automatic position weight scaling
max_weight_violation = positions['weight'].abs() > config.max_position_weight
if max_weight_violation.any():
    # Scale down all positions proportionally
    scaling_factor = config.max_position_weight / positions['weight'].abs().max()
    positions['weight'] *= scaling_factor
```

#### Leverage Constraints
```python
# Leverage monitoring
gross_leverage = positions['notional'].abs().sum() / config.notional_per_symbol
net_leverage = positions['notional'].sum() / config.notional_per_symbol

assert gross_leverage <= config.max_leverage
```

---

## Performance Metrics

### Forecast Quality Metrics

#### CRPS (Continuous Ranked Probability Score)
```python
def crps(y_true, q25, q50, q75):
    """Calculate CRPS for quantile forecasts."""
    crps_value = (
        0.25 * np.abs(y_true - q25) +
        0.5 * np.abs(y_true - q50) +
        0.25 * np.abs(y_true - q75)
    ).mean()

    # Penalty for quantile crossing
    crossing_penalty = np.maximum(0, q25 - q50).mean() + np.maximum(0, q50 - q75).mean()
    return float(crps_value + crossing_penalty)
```

#### Information Coefficient (IC)
```python
def information_coefficient(forecasts, actuals):
    """Calculate Pearson correlation between forecasts and actuals."""
    return np.corrcoef(forecasts, actuals)[0, 1]
```

#### Calibration Metrics
```python
def pit_coverage(y_true, q25, q75):
    """Calculate Probability Integral Transform (PIT) coverage."""
    in_interval = (y_true >= q25) & (y_true <= q75)
    return in_interval.mean()
```

### Trading Performance Metrics

#### Sharpe Ratio
```python
def portfolio_sharpe(trades):
    """Calculate annualized Sharpe ratio."""
    returns = trades['net_pnl']
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    daily_sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
    return daily_sharpe * np.sqrt(252)  # Annualized
```

#### Maximum Drawdown
```python
def portfolio_max_drawdown(trades):
    """Calculate maximum portfolio drawdown."""
    cumulative = (1 + trades['net_pnl']).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()
```

#### Hit Rate
```python
def portfolio_hit_rate(trades):
    """Calculate percentage of profitable trades."""
    if len(trades) == 0:
        return 0.0
    return (trades['net_pnl'] > 0).mean()
```

---

## Usage Examples

### Complete Analysis Pipeline

```python
import pandas as pd
from datetime import date
from timegpt_v2.eval.walkforward import WalkForwardConfig, WalkForwardEvaluator
from timegpt_v2.eval.hyperparameter import CompactHyperparameterTuner, HyperparameterConfig
from timegpt_v2.eval.cross_sectional import CrossSectionalStrategy, CrossSectionConfig

# 1. Load data
features, forecasts_30m, forecasts_60m, actuals, trades_30m, trades_60m = load_data()

# 2. Walk-forward A/B testing
walkforward_config = WalkForwardConfig(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 6, 30),
    train_months=2,
    test_months=1,
    purge_months=1,
    min_obs_per_period=100
)

evaluator = WalkForwardConfig(walkforward_config)
ab_results = evaluator.compare_horizons(
    features=features,
    forecasts_30m=forecasts_30m,
    forecasts_60m=forecasts_60m,
    actuals=actuals,
    trades_30m=trades_30m,
    trades_60m=trades_60m
)

# 3. Generate decision report
report = evaluator.generate_decision_report(ab_results, "results/ab_test_report.json")
print(f"Winning horizon: {report['evaluation_summary']['winner']}")

# 4. Hyperparameter optimization
hyper_config = HyperparameterConfig(
    max_iterations=30,
    objective_metric="sharpe",
    k_sigma_range=(0.5, 1.0),
    min_sharpe=0.3,
    random_seed=42
)

tuner = CompactHyperparameterTuner(hyper_config)
opt_results = tuner.optimize(
    features=features,
    forecasts_30m=forecasts_30m,
    forecasts_60m=forecasts_60m,
    actuals=actuals,
    trades_30m=trades_30m,
    trades_60m=trades_60m,
    walkforward_config=walkforward_config,
    output_path="results/hyperparameter_opt.json"
)

# 5. Cross-sectional analysis
cross_config = CrossSectionalConfig(
    top_decile=0.1,
    bottom_decile=0.1,
    min_symbols=5,
    max_symbols=20,
    position_method="equal_weight",
    target_leverage=1.0
)

strategy = CrossSectionalStrategy(cross_config)
cross_results = strategy.evaluate_cross_sectional(
    features=features,
    forecasts=forecasts,
    actuals=actuals,
    walkforward_config=walkforward_config
)

# 6. Summary Report
summary = {
    "ab_testing": {
        "winner": report['evaluation_summary']['winner'],
        "sharpe_30m": report['trading_metrics']['30m']['sharpe'],
        "sharpe_60m": report['trading_metrics']['60m']['sharpe']
    },
    "hyperparameter_optimization": {
        "best_sharpe": opt_results['best_parameters']['sharpe'],
        "success_rate": opt_results['optimization_summary']['success_rate']
    },
    "cross_sectional": {
        "sharpe": cross_results.sharpe,
        "ic_mean": cross_results.ic_mean,
        "n_symbols": cross_results.n_symbols,
        "leverage_gross": cross_results.leverage_gross
    }
}

print("=== Analysis Summary ===")
for category, metrics in summary.items():
    print(f"\n{category.title()}:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

# 7. Save comprehensive results
import json
with open("results/comprehensive_analysis.json", "w") as f:
    json.dump({
        "walkforward_ab": ab_results,
        "hyperparameter_optimization": opt_results,
        "cross_sectional": cross_results,
        "summary": summary
    }, f, indent=2, default=str)
```

### Quick Evaluation Script

```python
# Quick evaluation of single horizon
from timegpt_v2.eval.walkforward import WalkForwardConfig, WalkForwardEvaluator

config = WalkForwardConfig(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 3, 31),
    train_months=1,
    test_months=1,
    purge_months=0,
    min_obs_per_period=50
)

evaluator = WalkForwardEvaluator(config)
result = evaluator.evaluate_horizon(
    horizon_minutes=30,
    features=features,
    forecasts=forecasts_30m,
    actuals=actuals,
    trades=trades_30m
)

print(f"H horizon: 30m")
print(f"Sharpe: {result.sharpe:.3f}")
print(f"CRPS: {result.crps_mean:.6f}")
print(f"IC: {result.ic_mean:.3f}")
print(f"Hit Rate: {result.hit_rate:.3f}")
print(f"Max Drawdown: {result.max_drawdown:.3f}")
```

---

## Configuration Reference

### WalkForwardConfig

```python
@dataclass
class WalkForwardConfig:
    # Date ranges
    start_date: date                    # Evaluation start date
    end_date: date                      # Evaluation end date

    # Walk-forward parameters
    train_months: int = 3               # Training period in months
    test_months: int = 1                # Test period in months
    purge_months: int = 1               # Purge period between train and test

    # Cost model
    costs: CostConfig = field(default_factory=CostConfig)

    # Evaluation parameters
    min_obs_per_period: int = 100      # Minimum observations per period
```

### HyperparameterConfig

```python
@dataclass
class HyperparameterConfig:
    # Search space bounds
    k_sigma_range: tuple[float, float] = (0.4, 1.2)
    tp_sl_pairs: list[tuple[float, float]] = field(
        default_factory=lambda: [(2.0, 2.0), (2.5, 2.0), (3.0, 2.0)]
    )
    uncertainty_cut_range: tuple[float, float] = (0.70, 0.95)
    cadence_options: list[int] = field(default_factory=lambda: [30, 60])

    # Optimization settings
    max_iterations: int = 50
    random_seed: int | None = None
    objective_metric: str = "sharpe"  # "sharpe" or "crps"

    # Constraints
    max_turnover_per_day: float = 12.0
    min_sharpe: float = 0.3
    max_drawdown_threshold: float = -0.25
    min_periods: int = 3
    stability_threshold: float = 0.3
```

### CrossSectionalConfig

```python
@dataclass
class CrossSectionalConfig:
    # Portfolio construction
    top_decile: float = 0.1              # Top 10% for long positions
    bottom_decile: float = 0.1           # Bottom 10% for short positions
    min_symbols: int = 5                  # Minimum symbols
    max_symbols: int = 50                 # Maximum symbols

    # Position sizing
    position_method: str = "equal_weight"    # "equal_weight" or "vol_scaled"
    notional_per_symbol: float = 10000.0    # Base notional per symbol
    max_position_weight: float = 0.2         # Maximum 20% per symbol

    # Risk management
    max_leverage: float = 2.0               # Maximum gross leverage
    beta_neutral: bool = True                # Enforce beta neutrality
    sector_neutral: bool = False              # Enforce sector neutrality

    # Execution constraints
    min_forecast_confidence: float = 0.7    # Minimum confidence threshold
    max_turnover_daily: float = 0.5          # Maximum daily turnover

    # Performance targets
    target_leverage: float = 1.0              # Target gross leverage
```

### CostConfig

```python
@dataclass
class CostConfig:
    half_spread_bps: float = 2.5      # Half spread in basis points
    commission_bps: float = 1.0       # Commission in basis points
    impact_bps: float = 0.5           # Market impact in basis points

    def total_cost_bps(self) -> float:
        """Total cost per trade in basis points."""
        return self.half_spread_bps + self.commission_bps + self.impact_bps
```

---

## Best Practices

### 1. Data Quality
- Ensure minute frequency regularity
- Validate sufficient history for walk-forward periods
- Monitor for data gaps and anomalies

### 2. Parameter Selection
- Start with conservative bounds in hyperparameter search
- Use validation constraints to prevent overfitting
- Consider cross-validation robustness

### 3. Risk Management
- Monitor leverage and position constraints
- Validate TP/SL geometry for positive expectancy
- Implement appropriate turnover controls

### 4. Performance Monitoring
- Track multiple metrics (not just Sharpe)
- Monitor stability across walk-forward folds
- Regular calibration checks

### 5. Documentation
- Maintain clear configuration documentation
- Record optimization results and rationale
- Update validation thresholds as needed

---

This advanced framework provides comprehensive tools for systematic improvement of intraday trading strategies while maintaining robust risk management and performance validation.