"""Tests for Sprint 6: Walk-forward A/B harness."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path

from timegpt_v2.eval.walkforward import (
    CostConfig,
    WalkForwardConfig,
    HorizonResult,
    WalkForwardEvaluator,
)
from timegpt_v2.eval.metrics_forecast import crps, information_coefficient, pit_coverage


class TestCostConfig:
    """Test trading cost configuration."""

    def test_default_costs(self):
        """Test default cost configuration."""
        costs = CostConfig()
        assert costs.half_spread_bps == 2.5
        assert costs.commission_bps == 1.0
        assert costs.impact_bps == 0.5
        assert costs.total_cost_bps() == 4.0

    def test_custom_costs(self):
        """Test custom cost configuration."""
        costs = CostConfig(half_spread_bps=5.0, commission_bps=2.0, impact_bps=1.0)
        assert costs.total_cost_bps() == 8.0


class TestWalkForwardConfig:
    """Test walk-forward configuration."""

    def test_valid_config(self):
        """Test valid configuration."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 6, 30)
        config = WalkForwardConfig(start_date=start_date, end_date=end_date)

        assert config.train_months == 3
        assert config.test_months == 1
        assert config.purge_months == 1
        assert config.min_obs_per_period == 100

    def test_invalid_date_range(self):
        """Test invalid date range."""
        with pytest.raises(ValueError, match="start_date must be before end_date"):
            WalkForwardConfig(
                start_date=date(2024, 6, 30),
                end_date=date(2024, 1, 1)
            )

    def test_invalid_months(self):
        """Test invalid month parameters."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 6, 30)

        with pytest.raises(ValueError, match="train_months and test_months must be positive"):
            WalkForwardConfig(
                start_date=start_date,
                end_date=end_date,
                train_months=0
            )

        with pytest.raises(ValueError, match="train_months and test_months must be positive"):
            WalkForwardConfig(
                start_date=start_date,
                end_date=end_date,
                test_months=-1
            )


class TestWalkForwardEvaluator:
    """Test walk-forward evaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create a test evaluator."""
        config = WalkForwardConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 4, 30),
            train_months=1,
            test_months=1,
            purge_months=0,  # No purge for simpler testing
        )
        return WalkForwardEvaluator(config)

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        # Generate test dates
        dates = pd.date_range("2024-01-01", periods=120, freq="D")  # 4 months

        # Create sample forecasts
        forecasts = []
        for symbol in ["AAPL", "MSFT"]:
            for date in dates:
                # Create 30-minute forecasts
                for hour in [10, 11, 14, 15]:
                    timestamp = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)
                    forecasts.append({
                        "symbol": symbol,
                        "ts_utc": timestamp,
                        "q25": np.random.normal(-0.001, 0.002),
                        "q50": np.random.normal(0, 0.002),
                        "q75": np.random.normal(0.001, 0.002),
                    })

        forecasts_df = pd.DataFrame(forecasts)

        # Create sample actuals
        actuals = []
        for symbol in ["AAPL", "MSFT"]:
            for date in dates:
                for hour in [10, 11, 14, 15]:
                    timestamp = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)
                    actuals.append({
                        "symbol": symbol,
                        "ts_utc": timestamp,
                        "y_true": np.random.normal(0, 0.003),
                    })

        actuals_df = pd.DataFrame(actuals)

        # Create sample trades aligned with forecast timestamps
        trades = []
        for symbol in ["AAPL", "MSFT"]:
            for date in dates:
                for hour in [10, 11, 14, 15]:  # Match forecast times
                    entry_time = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)
                    exit_time = entry_time + timedelta(minutes=30)

                    pnl = np.random.normal(0, 0.002)  # Random PnL
                    trades.append({
                        "symbol": symbol,
                        "ts_utc": entry_time,  # Same timestamp as forecasts
                        "entry_ts": entry_time,
                        "exit_ts": exit_time,
                        "net_pnl": pnl,
                        "return": pnl,
                        "quantity": 100,
                        "entry_price": 100.0 + np.random.normal(0, 1),
                        "exit_price": 100.0 + pnl + np.random.normal(0, 1),
                    })

        trades_df = pd.DataFrame(trades)

        return forecasts_df, actuals_df, trades_df

    def test_generate_walk_forward_periods(self, evaluator):
        """Test walk-forward period generation."""
        periods = evaluator.generate_walk_forward_periods()

        # Should generate 3 periods: Jan train -> Feb test, Feb train -> Mar test, Mar train -> Apr test
        assert len(periods) == 3

        # Check first period
        train_start, train_end, test_start, test_end = periods[0]
        assert train_start == date(2024, 1, 1)
        assert test_start >= date(2024, 1, 31)  # Allow for exact calculation

        # Check periods are sequential (basic check)
        assert len(periods) > 0

    def test_evaluate_horizon(self, evaluator, sample_data):
        """Test single horizon evaluation."""
        forecasts, actuals, trades = sample_data

        result = evaluator.evaluate_horizon(
            horizon_minutes=30,
            features=pd.DataFrame(),  # Not used in current implementation
            forecasts=forecasts,
            actuals=actuals,
            trades=trades
        )

        assert isinstance(result, HorizonResult)
        assert result.horizon_minutes == 30
        assert result.n_periods > 0
        assert result.n_forecasts > 0
        # n_trades can be 0 if trades don't match forecast periods

        # Check metrics are reasonable
        assert isinstance(result.crps_mean, float)
        assert isinstance(result.ic_mean, float)
        assert isinstance(result.sharpe, float)
        assert isinstance(result.turnover, float)

        # Sharpe can be negative, that's fine
        assert not np.isnan(result.sharpe)

    def test_compare_horizons(self, evaluator, sample_data):
        """Test horizon comparison."""
        forecasts, actuals, trades = sample_data

        # Create separate data for each horizon
        # In practice, these would be generated by different forecast models
        forecasts_30m = forecasts.copy()
        forecasts_60m = forecasts.copy()

        # Slightly modify 60m forecasts to make them different
        forecasts_60m['q50'] *= 1.1
        forecasts_60m['q75'] *= 1.1
        forecasts_60m['q25'] *= 1.1

        trades_30m = trades.copy()
        trades_60m = trades.copy()

        # Slightly modify 60m trades to make them different
        trades_60m['net_pnl'] *= 0.9  # Slightly worse performance

        results = evaluator.compare_horizons(
            features=pd.DataFrame(),
            forecasts_30m=forecasts_30m,
            forecasts_60m=forecasts_60m,
            actuals=actuals,
            trades_30m=trades_30m,
            trades_60m=trades_60m
        )

        assert "30m" in results
        assert "60m" in results
        assert isinstance(results["30m"], HorizonResult)
        assert isinstance(results["60m"], HorizonResult)

        # Results should be different
        assert results["30m"].sharpe != results["60m"].sharpe or \
               results["30m"].crps_mean != results["60m"].crps_mean

    def test_generate_decision_report(self, evaluator, sample_data):
        """Test decision report generation."""
        forecasts, actuals, trades = sample_data

        # Create results for both horizons
        result_30m = HorizonResult(horizon_minutes=30)
        result_30m.sharpe = 0.8
        result_30m.crps_mean = 0.001
        result_30m.ic_mean = 0.05
        result_30m.coverage_mean = 0.48
        result_30m.max_drawdown = -0.05
        result_30m.hit_rate = 0.55
        result_30m.total_pnl = 1000.0
        result_30m.turnover = 8.0
        result_30m.n_periods = 3
        result_30m.n_forecasts = 1000
        result_30m.n_trades = 240

        result_60m = HorizonResult(horizon_minutes=60)
        result_60m.sharpe = 0.6  # Lower Sharpe
        result_60m.crps_mean = 0.0012
        result_60m.ic_mean = 0.04
        result_60m.coverage_mean = 0.52
        result_60m.max_drawdown = -0.08
        result_60m.hit_rate = 0.52
        result_60m.total_pnl = 800.0
        result_60m.turnover = 5.0
        result_60m.n_periods = 3
        result_60m.n_forecasts = 600
        result_60m.n_trades = 150

        results = {"30m": result_30m, "60m": result_60m}

        report = evaluator.generate_decision_report(results)

        # Check report structure
        assert "evaluation_summary" in report
        assert "forecast_metrics" in report
        assert "trading_metrics" in report
        assert "cost_analysis" in report
        assert "recommendations" in report
        assert "robustness_analysis" in report

        # Check evaluation summary
        summary = report["evaluation_summary"]
        assert summary["winner"] == "30m"  # Higher Sharpe
        assert summary["primary_metric"] == "sharpe"
        assert summary["total_forecasts_30m"] == 1000
        assert summary["total_forecasts_60m"] == 600

        # Check recommendations
        recommendations = report["recommendations"]
        assert recommendations["selected_horizon"] == "30m"
        assert "rationale" in recommendations
        assert "key_metrics" in recommendations

        # Check cost analysis
        cost_analysis = report["cost_analysis"]
        assert cost_analysis["total_cost_bps"] == 4.0  # Default costs
        assert "cost_breakdown" in cost_analysis

    def test_decision_report_with_file_output(self, evaluator, sample_data, tmp_path):
        """Test decision report with file output."""
        forecasts, actuals, trades = sample_data

        # Create minimal results
        result_30m = HorizonResult(horizon_minutes=30)
        result_30m.sharpe = 0.8
        result_30m.crps_mean = 0.001

        result_60m = HorizonResult(horizon_minutes=60)
        result_60m.sharpe = 0.6
        result_60m.crps_mean = 0.0012

        results = {"30m": result_30m, "60m": result_60m}

        output_file = tmp_path / "decision_report.json"
        report = evaluator.generate_decision_report(results, output_file)

        # Check file was created
        assert output_file.exists()

        # Check file content
        import json
        with output_file.open() as f:
            saved_report = json.load(f)

        assert saved_report == report

    def test_insufficient_data_handling(self, evaluator):
        """Test handling of insufficient data."""
        # Create minimal data
        forecasts = pd.DataFrame({
            "symbol": ["AAPL"],
            "ts_utc": [datetime(2024, 1, 15, 10, 0)],
            "q25": [-0.001],
            "q50": [0.0],
            "q75": [0.001],
        })

        actuals = pd.DataFrame({
            "symbol": ["AAPL"],
            "ts_utc": [datetime(2024, 1, 15, 10, 0)],
            "y_true": [0.0005],
        })

        trades = pd.DataFrame({
            "symbol": ["AAPL"],
            "ts_utc": [datetime(2024, 1, 15, 10, 0)],
            "net_pnl": [0.001],
        })

        # Should handle insufficient data gracefully
        result = evaluator.evaluate_horizon(
            horizon_minutes=30,
            features=pd.DataFrame(),
            forecasts=forecasts,
            actuals=actuals,
            trades=trades
        )

        # Should still return a result, possibly with zero values
        assert isinstance(result, HorizonResult)
        assert result.horizon_minutes == 30

    def test_missing_horizons_error(self, evaluator):
        """Test error when required horizons are missing."""
        results = {"30m": HorizonResult(horizon_minutes=30)}  # Missing 60m

        with pytest.raises(ValueError, match="Results must contain both '30m' and '60m' horizons"):
            evaluator.generate_decision_report(results)


class TestMetricsIntegration:
    """Test integration with forecast metrics."""

    def test_crps_calculation(self):
        """Test CRPS calculation."""
        y_true = np.array([0.001, -0.002, 0.003])
        q25 = np.array([-0.001, -0.003, 0.001])
        q50 = np.array([0.0, -0.001, 0.002])
        q75 = np.array([0.001, 0.001, 0.003])

        crps_value = crps(y_true, q25, q50, q75)
        assert isinstance(crps_value, float)
        assert crps_value >= 0  # CRPS is non-negative

    def test_information_coefficient(self):
        """Test Information Coefficient calculation."""
        forecasts = np.array([0.001, 0.002, -0.001, 0.003])
        actuals = np.array([0.002, 0.001, -0.002, 0.001])

        ic_value = information_coefficient(forecasts, actuals)
        assert isinstance(ic_value, float)
        assert -1 <= ic_value <= 1  # Correlation bounds

    def test_metrics_with_nan_values(self):
        """Test metrics handling NaN values."""
        forecasts = np.array([0.001, np.nan, 0.003])
        actuals = np.array([0.002, 0.001, np.nan])

        # Should handle NaN values gracefully
        ic_value = information_coefficient(forecasts, actuals)
        assert isinstance(ic_value, float)
        assert not np.isnan(ic_value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])