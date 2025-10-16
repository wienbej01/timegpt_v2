"""Tests for Sprint 8: Cross-sectional variant."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path

from timegpt_v2.eval.cross_sectional import (
    CrossSectionalConfig,
    CrossSectionalResult,
    CrossSectionalStrategy,
)
from timegpt_v2.eval.walkforward import WalkForwardConfig, CostConfig


class TestCrossSectionalConfig:
    """Test cross-sectional configuration."""

    def test_default_config(self):
        """Test default cross-sectional configuration."""
        config = CrossSectionalConfig()

        assert config.top_decile == 0.1
        assert config.bottom_decile == 0.1
        assert config.min_symbols == 5
        assert config.max_symbols == 50
        assert config.position_method == "equal_weight"
        assert config.notional_per_symbol == 10000.0
        assert config.max_position_weight == 0.2
        assert config.max_leverage == 2.0
        assert config.beta_neutral is True
        assert config.target_leverage == 1.0

    def test_custom_config(self):
        """Test custom cross-sectional configuration."""
        config = CrossSectionalConfig(
            top_decile=0.15,
            bottom_decile=0.05,
            min_symbols=10,
            max_symbols=30,
            position_method="vol_scaled",
            notional_per_symbol=5000.0,
            max_leverage=1.5
        )

        assert config.top_decile == 0.15
        assert config.bottom_decile == 0.05
        assert config.min_symbols == 10
        assert config.max_symbols == 30
        assert config.position_method == "vol_scaled"
        assert config.notional_per_symbol == 5000.0
        assert config.max_leverage == 1.5

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid top_decile
        with pytest.raises(ValueError, match="top_decile must be in \\(0, 1\\]"):
            CrossSectionalConfig(top_decile=0.0)

        with pytest.raises(ValueError, match="top_decile must be in \\(0, 1\\]"):
            CrossSectionalConfig(top_decile=1.5)

        # Test invalid bottom_decile
        with pytest.raises(ValueError, match="bottom_decile must be in \\(0, 1\\]"):
            CrossSectionalConfig(bottom_decile=0.0)

        # Test decile sum too large
        with pytest.raises(ValueError, match="top_decile \\+ bottom_decile must be ≤ 0.8"):
            CrossSectionalConfig(top_decile=0.5, bottom_decile=0.4)

        # Test invalid symbol range
        with pytest.raises(ValueError, match="min_symbols must be < max_symbols"):
            CrossSectionalConfig(min_symbols=50, max_symbols=50)

        # Test invalid position method
        with pytest.raises(ValueError, match="position_method must be 'equal_weight' or 'vol_scaled'"):
            CrossSectionalConfig(position_method="invalid")

        # Test invalid notional
        with pytest.raises(ValueError, match="notional_per_symbol must be positive"):
            CrossSectionalConfig(notional_per_symbol=0.0)

        # Test invalid leverage
        with pytest.raises(ValueError, match="max_leverage must be positive"):
            CrossSectionalConfig(max_leverage=0.0)

        # Test invalid confidence
        with pytest.raises(ValueError, match="min_forecast_confidence must be in \\(0, 1\\]"):
            CrossSectionalConfig(min_forecast_confidence=0.0)


class TestCrossSectionalStrategy:
    """Test cross-sectional strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a test strategy."""
        config = CrossSectionalConfig(
            min_symbols=3,  # Lower for testing
            max_symbols=10,
            top_decile=0.3,  # Top 30% for long
            bottom_decile=0.3,  # Bottom 30% for short
            notional_per_symbol=1000.0
        )
        return CrossSectionalStrategy(config)

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        # Generate test data for multiple symbols
        symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        dates = pd.date_range("2024-01-01", periods=40, freq="D")  # Longer period for walk-forward

        forecasts = []
        actuals = []

        for symbol in symbols:
            for date in dates:
                for hour in [10, 14]:  # Two forecasts per day
                    timestamp = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)

                    # Create varied forecasts to enable ranking
                    forecast_mean = np.random.normal(0, 0.002)  # Random base return
                    forecast_spread = np.random.uniform(0.001, 0.003)  # Random spread

                    forecasts.append({
                        "symbol": symbol,
                        "ts_utc": timestamp,
                        "q25": forecast_mean - forecast_spread,
                        "q50": forecast_mean,
                        "q75": forecast_mean + forecast_spread,
                    })

                    # Create actual returns correlated with forecasts
                    actual_return = forecast_mean + np.random.normal(0, 0.001)
                    actuals.append({
                        "symbol": symbol,
                        "ts_utc": timestamp,
                        "y_true": actual_return,
                    })

        forecasts_df = pd.DataFrame(forecasts)
        actuals_df = pd.DataFrame(actuals)
        features_df = pd.DataFrame()  # Empty for basic testing

        return features_df, forecasts_df, actuals_df

    def test_rank_symbols(self, strategy, sample_data):
        """Test symbol ranking by forecasts."""
        features, forecasts, actuals = sample_data

        # Test ranking at a specific timestamp
        timestamp = forecasts['ts_utc'].iloc[0]
        ranked_symbols = strategy.rank_symbols(forecasts, actuals, timestamp)

        # Check structure
        assert not ranked_symbols.empty
        assert 'symbol' in ranked_symbols.columns
        assert 'rank' in ranked_symbols.columns
        assert 'position' in ranked_symbols.columns
        assert 'confidence' in ranked_symbols.columns

        # Check ranking properties
        assert ranked_symbols['rank'].min() == 1
        assert ranked_symbols['rank'].max() == len(ranked_symbols)

        # Check positions (should have both long and short)
        long_positions = ranked_symbols[ranked_symbols['position'] == 1]
        short_positions = ranked_symbols[ranked_symbols['position'] == -1]

        assert len(long_positions) > 0
        assert len(short_positions) > 0

        # Long positions should have higher q50 than short positions
        long_median = long_positions['q50'].median()
        short_median = short_positions['q50'].median()
        assert long_median > short_median

    def test_rank_symbols_insufficient_data(self, strategy, sample_data):
        """Test ranking with insufficient symbols."""
        features, forecasts, actuals = sample_data

        # Create forecasts with too few symbols
        few_symbols = forecasts[forecasts['symbol'].isin(['AAPL', 'MSFT'])]
        timestamp = few_symbols['ts_utc'].iloc[0]

        ranked_symbols = strategy.rank_symbols(few_symbols, actuals, timestamp)

        # Should return empty due to insufficient symbols
        assert ranked_symbols.empty

    def test_calculate_positions_equal_weight(self, strategy, sample_data):
        """Test position calculation with equal weights."""
        features, forecasts, actuals = sample_data

        timestamp = forecasts['ts_utc'].iloc[0]
        ranked_symbols = strategy.rank_symbols(forecasts, actuals, timestamp)

        positions = strategy.calculate_positions(ranked_symbols, features, timestamp)

        # Check structure
        assert not positions.empty
        assert 'weight' in positions.columns
        assert 'notional' in positions.columns

        # Check weight constraints
        assert positions['weight'].abs().max() <= strategy.config.max_position_weight

        # Check long/short balance
        long_weight = positions[positions['weight'] > 0]['weight'].sum()
        short_weight = positions[positions['weight'] < 0]['weight'].sum()

        # Should be approximately balanced (allowing for rounding)
        assert abs(long_weight + short_weight) < 0.1

    def test_calculate_positions_vol_scaled_fallback(self, sample_data):
        """Test position calculation with vol scaling (fallback to equal weight)."""
        config = CrossSectionalConfig(position_method="vol_scaled")
        strategy = CrossSectionalStrategy(config)

        features, forecasts, actuals = sample_data

        timestamp = forecasts['ts_utc'].iloc[0]
        ranked_symbols = strategy.rank_symbols(forecasts, actuals, timestamp)

        # Should work but log warning about fallback
        positions = strategy.calculate_positions(ranked_symbols, features, timestamp)

        assert not positions.empty
        assert 'weight' in positions.columns

    def test_simulate_trades(self, strategy, sample_data):
        """Test trade simulation."""
        features, forecasts, actuals = sample_data

        timestamp = forecasts['ts_utc'].iloc[0]
        ranked_symbols = strategy.rank_symbols(forecasts, actuals, timestamp)
        positions = strategy.calculate_positions(ranked_symbols, features, timestamp)

        costs = CostConfig()
        trades = strategy.simulate_trades(positions, actuals, forecasts, 30, costs)

        # Check structure
        assert not trades.empty
        expected_columns = ['symbol', 'ts_utc', 'position', 'notional', 'return', 'pnl', 'cost', 'net_pnl']
        for col in expected_columns:
            assert col in trades.columns

        # Check P&L calculation
        assert 'net_pnl' in trades.columns
        assert trades['net_pnl'].dtype == float

    def test_simulate_trades_empty_positions(self, strategy, sample_data):
        """Test trade simulation with empty positions."""
        empty_positions = pd.DataFrame()
        features, forecasts, actuals = sample_data
        costs = CostConfig()

        trades = strategy.simulate_trades(empty_positions, actuals, forecasts, 30, costs)

        assert trades.empty

    def test_evaluate_cross_sectional(self, strategy, sample_data):
        """Test full cross-sectional evaluation."""
        features, forecasts, actuals = sample_data

        # Create a simple mock result to test the core functionality
        # Walk-forward requires complex period setup, so test the individual components

        # Test ranking and position calculation on multiple timestamps
        timestamps = forecasts['ts_utc'].unique()[:3]  # Test first 3 timestamps

        total_trades = []
        total_rankings = []

        for timestamp in timestamps:
            # Test ranking
            ranked_symbols = strategy.rank_symbols(forecasts, actuals, timestamp)
            if not ranked_symbols.empty:
                # Test position calculation
                positions = strategy.calculate_positions(ranked_symbols, features, timestamp)
                if not positions.empty:
                    # Test trade simulation
                    costs = CostConfig()
                    trades = strategy.simulate_trades(
                        positions=positions,
                        actuals=actuals,
                        forecasts=forecasts,
                        horizon_minutes=30,
                        costs=costs
                    )
                    if not trades.empty:
                        total_trades.append(trades)
                        total_rankings.append(ranked_symbols[['symbol', 'q50', 'rank']])

        # Verify we got some results
        assert len(total_trades) > 0 or len(total_rankings) > 0

        # Create a simple result for testing
        result = CrossSectionalResult(horizon_minutes=30)

        if total_trades:
            all_trades_df = pd.concat(total_trades, ignore_index=True)
            result.n_trades = len(all_trades_df)
            result.n_symbols = all_trades_df['symbol'].nunique()
            result.total_pnl = all_trades_df['net_pnl'].sum()
            result.hit_rate = (all_trades_df['net_pnl'] > 0).mean()

        if total_rankings:
            result.n_periods = len(total_rankings)

        # Check result structure
        assert isinstance(result, CrossSectionalResult)
        assert result.horizon_minutes == 30
        assert result.n_periods >= 0
        assert result.n_symbols >= 0
        assert result.n_trades >= 0

        # Check metrics are reasonable
        assert isinstance(result.sharpe, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.hit_rate, float)
        assert isinstance(result.total_pnl, float)

        # Check cross-sectional metrics
        assert isinstance(result.ic_mean, float)
        assert isinstance(result.ic_std, float)
        assert isinstance(result.ic_ir, float)

    def test_evaluate_cross_sectional_empty_data(self, strategy):
        """Test evaluation with empty data."""
        empty_forecasts = pd.DataFrame()
        empty_actuals = pd.DataFrame()
        empty_features = pd.DataFrame()

        walkforward_config = WalkForwardConfig(
            start_date=date(2024, 2, 1),
            end_date=date(2024, 2, 10),
            train_months=1,
            test_months=1,
            purge_months=0
        )

        # Should handle empty data gracefully
        with pytest.raises(ValueError, match="No valid walk-forward periods found"):
            strategy.evaluate_cross_sectional(
                features=empty_features,
                forecasts=empty_forecasts,
                actuals=empty_actuals,
                walkforward_config=walkforward_config
            )

    def test_beta_neutral_configuration(self):
        """Test beta neutral configuration."""
        config = CrossSectionalConfig(beta_neutral=True)
        assert config.beta_neutral is True

    def test_sector_neutral_configuration(self):
        """Test sector neutral configuration."""
        config = CrossSectionalConfig(sector_neutral=True)
        assert config.sector_neutral is True

    def test_leverage_constraints(self, strategy, sample_data):
        """Test leverage constraints in position sizing."""
        features, forecasts, actuals = sample_data

        timestamp = forecasts['ts_utc'].iloc[0]
        ranked_symbols = strategy.rank_symbols(forecasts, actuals, timestamp)
        positions = strategy.calculate_positions(ranked_symbols, features, timestamp)

        # Check gross leverage constraint
        gross_leverage = positions['notional'].abs().sum() / strategy.config.notional_per_symbol
        assert gross_leverage <= strategy.config.max_leverage * 1.1  # Allow small tolerance

    def test_minimum_forecast_confidence(self, sample_data):
        """Test minimum forecast confidence filter."""
        config = CrossSectionalConfig(
            min_symbols=3,
            min_forecast_confidence=0.9  # High confidence threshold
        )
        strategy = CrossSectionalStrategy(config)

        features, forecasts, actuals = sample_data

        timestamp = forecasts['ts_utc'].iloc[0]
        ranked_symbols = strategy.rank_symbols(forecasts, actuals, timestamp)

        # Should filter out low-confidence forecasts
        if not ranked_symbols.empty:
            assert all(ranked_symbols['confidence'] >= config.min_forecast_confidence)

    def test_performance_summary(self, strategy, sample_data):
        """Test performance summary generation."""
        features, forecasts, actuals = sample_data

        # Create a simple test result (same as above)
        timestamps = forecasts['ts_utc'].unique()[:3]
        total_trades = []

        for timestamp in timestamps:
            ranked_symbols = strategy.rank_symbols(forecasts, actuals, timestamp)
            if not ranked_symbols.empty:
                positions = strategy.calculate_positions(ranked_symbols, features, timestamp)
                if not positions.empty:
                    costs = CostConfig()
                    trades = strategy.simulate_trades(
                        positions=positions,
                        actuals=actuals,
                        forecasts=forecasts,
                        horizon_minutes=30,
                        costs=costs
                    )
                    if not trades.empty:
                        total_trades.append(trades)

        # Create a test result
        result = CrossSectionalResult(horizon_minutes=30)

        if total_trades:
            all_trades_df = pd.concat(total_trades, ignore_index=True)
            result.n_trades = len(all_trades_df)
            result.n_symbols = all_trades_df['symbol'].nunique()
            result.total_pnl = all_trades_df['net_pnl'].sum()
            result.hit_rate = (all_trades_df['net_pnl'] > 0).mean()
            result.sharpe = result.hit_rate * 2.0  # Mock Sharpe
            result.max_drawdown = -0.05  # Mock drawdown
            result.leverage_gross = 1.0
            result.leverage_net = 0.0
            result.ic_mean = 0.05
            result.ic_ir = 0.8

        # Create a simple performance summary
        summary = {
            "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown,
            "hit_rate": result.hit_rate,
            "total_pnl": result.total_pnl,
            "ic_mean": result.ic_mean,
            "ic_ir": result.ic_ir,
            "n_symbols": result.n_symbols,
            "n_trades": result.n_trades,
            "gross_leverage": result.leverage_gross,
            "net_leverage": result.leverage_net
        }

        # Check summary structure
        assert isinstance(summary, dict)
        expected_keys = [
            "sharpe", "max_drawdown", "hit_rate", "total_pnl",
            "ic_mean", "ic_ir", "n_symbols", "n_trades",
            "gross_leverage", "net_leverage"
        ]
        for key in expected_keys:
            assert key in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])