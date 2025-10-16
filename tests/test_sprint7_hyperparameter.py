"""Tests for Sprint 7: Compact hyperparameter tuner."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path

from timegpt_v2.eval.hyperparameter import (
    HyperparameterConfig,
    ParameterSet,
    CompactHyperparameterTuner,
)
from timegpt_v2.eval.walkforward import WalkForwardConfig, CostConfig


class TestHyperparameterConfig:
    """Test hyperparameter configuration."""

    def test_default_config(self):
        """Test default hyperparameter configuration."""
        config = HyperparameterConfig()

        # Check search space bounds
        assert config.k_sigma_range == (0.4, 1.2)
        assert config.tp_sl_pairs == [(2.0, 2.0), (2.5, 2.0), (3.0, 2.0)]
        assert config.uncertainty_cut_range == (0.70, 0.95)
        assert config.cadence_options == [30, 60]

        # Check optimization settings
        assert config.max_iterations == 50
        assert config.objective_metric == "sharpe"

        # Check constraints
        assert config.max_turnover_per_day == 12.0
        assert config.min_sharpe == 0.3
        assert config.max_drawdown_threshold == -0.25

    def test_custom_config(self):
        """Test custom hyperparameter configuration."""
        config = HyperparameterConfig(
            max_iterations=20,
            objective_metric="crps",
            min_sharpe=0.5,
            random_seed=42
        )

        assert config.max_iterations == 20
        assert config.objective_metric == "crps"
        assert config.min_sharpe == 0.5
        assert config.random_seed == 42

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid objective metric
        with pytest.raises(ValueError, match="objective_metric must be 'sharpe' or 'crps'"):
            HyperparameterConfig(objective_metric="invalid_metric")

        # Test invalid TP/SL pair
        with pytest.raises(ValueError, match="All TP/SL pairs must have TP >= SL"):
            HyperparameterConfig(tp_sl_pairs=[(1.0, 2.0)])  # TP < SL

        # Test invalid max iterations
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            HyperparameterConfig(max_iterations=0)


class TestParameterSet:
    """Test parameter set class."""

    def test_parameter_set_creation(self):
        """Test parameter set creation."""
        params = ParameterSet(
            k_sigma=0.8,
            tp_sigma=2.5,
            sl_sigma=2.0,
            uncertainty_cut=0.8,
            cadence_minutes=30
        )

        assert params.k_sigma == 0.8
        assert params.tp_sigma == 2.5
        assert params.sl_sigma == 2.0
        assert params.uncertainty_cut == 0.8
        assert params.cadence_minutes == 30

    def test_parameter_set_to_dict(self):
        """Test parameter set conversion to dictionary."""
        params = ParameterSet(
            k_sigma=0.7,
            tp_sigma=3.0,
            sl_sigma=2.0,
            uncertainty_cut=0.85,
            cadence_minutes=60
        )
        params.sharpe = 1.2
        params.crps = 0.001
        params.is_valid = True
        params.validation_reasons = []

        param_dict = params.to_dict()

        assert isinstance(param_dict, dict)
        assert param_dict["k_sigma"] == 0.7
        assert param_dict["sharpe"] == 1.2
        assert param_dict["is_valid"] is True
        assert param_dict["validation_reasons"] == []


class TestCompactHyperparameterTuner:
    """Test compact hyperparameter tuner."""

    @pytest.fixture
    def tuner(self):
        """Create a test tuner."""
        config = HyperparameterConfig(
            max_iterations=10,  # Small number for testing
            random_seed=42,
            k_sigma_range=(0.5, 1.0),
            min_sharpe=0.1  # Lower threshold for testing
        )
        return CompactHyperparameterTuner(config)

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        # Generate test data for Feb 1-15, 2024 (as requested)
        dates = pd.date_range("2024-02-01", periods=15, freq="D")

        # Create sample forecasts
        forecasts_30m = []
        forecasts_60m = []

        for symbol in ["AAPL"]:
            for date in dates:
                for hour in [10, 11, 14, 15]:
                    timestamp = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)

                    # 30m forecasts
                    forecasts_30m.append({
                        "symbol": symbol,
                        "ts_utc": timestamp,
                        "q25": np.random.normal(-0.001, 0.002),
                        "q50": np.random.normal(0, 0.002),
                        "q75": np.random.normal(0.001, 0.002),
                    })

                    # 60m forecasts (different times)
                    if hour in [11, 15]:  # Fewer 60m forecasts
                        forecasts_60m.append({
                            "symbol": symbol,
                            "ts_utc": timestamp,
                            "q25": np.random.normal(-0.0015, 0.0025),
                            "q50": np.random.normal(0, 0.0025),
                            "q75": np.random.normal(0.0015, 0.0025),
                        })

        forecasts_30m_df = pd.DataFrame(forecasts_30m)
        forecasts_60m_df = pd.DataFrame(forecasts_60m)

        # Create sample actuals
        actuals = []
        for symbol in ["AAPL"]:
            for date in dates:
                for hour in [10, 11, 14, 15]:
                    timestamp = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)
                    actuals.append({
                        "symbol": symbol,
                        "ts_utc": timestamp,
                        "y_true": np.random.normal(0.0001, 0.003),
                    })

        actuals_df = pd.DataFrame(actuals)

        # Create sample trades aligned with forecasts
        trades_30m = []
        trades_60m = []

        for symbol in ["AAPL"]:
            for date in dates:
                for hour in [10, 11, 14, 15]:
                    timestamp = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)

                    # 30m trades
                    pnl_30m = np.random.normal(0, 0.002)
                    trades_30m.append({
                        "symbol": symbol,
                        "ts_utc": timestamp,
                        "entry_ts": timestamp,
                        "exit_ts": timestamp + timedelta(minutes=30),
                        "net_pnl": pnl_30m,
                        "return": pnl_30m,
                        "quantity": 100,
                        "entry_price": 150.0 + np.random.normal(0, 5),
                        "exit_price": 150.0 + pnl_30m + np.random.normal(0, 5),
                    })

                    # 60m trades (different times)
                    if hour in [11, 15]:
                        pnl_60m = np.random.normal(0.0005, 0.003)
                        trades_60m.append({
                            "symbol": symbol,
                            "ts_utc": timestamp,
                            "entry_ts": timestamp,
                            "exit_ts": timestamp + timedelta(minutes=60),
                            "net_pnl": pnl_60m,
                            "return": pnl_60m,
                            "quantity": 100,
                            "entry_price": 150.0 + np.random.normal(0, 5),
                            "exit_price": 150.0 + pnl_60m + np.random.normal(0, 5),
                        })

        trades_30m_df = pd.DataFrame(trades_30m)
        trades_60m_df = pd.DataFrame(trades_60m)

        features_df = pd.DataFrame()  # Empty as not used in current implementation

        return features_df, forecasts_30m_df, forecasts_60m_df, actuals_df, trades_30m_df, trades_60m_df

    def test_parameter_generation(self, tuner):
        """Test random parameter generation."""
        params = tuner.generate_parameter_set()

        # Check parameter bounds
        assert tuner.config.k_sigma_range[0] <= params.k_sigma <= tuner.config.k_sigma_range[1]
        assert tuner.config.uncertainty_cut_range[0] <= params.uncertainty_cut <= tuner.config.uncertainty_cut_range[1]
        assert params.cadence_minutes in tuner.config.cadence_options

        # Check TP/SL pair
        assert (params.tp_sigma, params.sl_sigma) in tuner.config.tp_sl_pairs
        assert params.tp_sigma >= params.sl_sigma  # All pairs should satisfy this

    def test_parameter_validation_success(self, tuner):
        """Test parameter validation with valid parameters."""
        from timegpt_v2.eval.walkforward import HorizonResult

        params = ParameterSet(
            k_sigma=0.8,
            tp_sigma=2.5,
            sl_sigma=2.0,  # TP >= SL
            uncertainty_cut=0.8,
            cadence_minutes=30
        )

        # Create a mock result that passes validation
        result = HorizonResult(horizon_minutes=30)
        result.sharpe = 1.5  # Above min_sharpe
        result.turnover = 8.0  # Below max_turnover
        result.max_drawdown = -0.1  # Above threshold (less negative)
        result.n_periods = 4  # Above min_periods

        validated_params = tuner.validate_parameters(params, result)

        assert validated_params.is_valid is True
        assert len(validated_params.validation_reasons) == 0

    def test_parameter_validation_failure(self, tuner):
        """Test parameter validation with invalid parameters."""
        from timegpt_v2.eval.walkforward import HorizonResult

        params = ParameterSet(
            k_sigma=0.8,
            tp_sigma=1.5,  # TP < SL - should fail
            sl_sigma=2.0,
            uncertainty_cut=0.8,
            cadence_minutes=30
        )

        # Create a mock result that fails validation
        result = HorizonResult(horizon_minutes=30)
        result.sharpe = 0.1  # Below min_sharpe
        result.turnover = 20.0  # Above max_turnover
        result.max_drawdown = -0.4  # Below threshold (more negative)
        result.n_periods = 2  # Below min_periods

        validated_params = tuner.validate_parameters(params, result)

        assert validated_params.is_valid is False
        assert len(validated_params.validation_reasons) > 0
        assert any("TP sigma should be >= SL sigma" in reason for reason in validated_params.validation_reasons)

    def test_trading_filters(self, tuner):
        """Test trading rule filtering."""
        params = ParameterSet(
            k_sigma=0.8,
            tp_sigma=2.5,
            sl_sigma=2.0,
            uncertainty_cut=0.8,
            cadence_minutes=30
        )

        # Create sample trades
        trades = pd.DataFrame({
            "symbol": ["AAPL"] * 100,
            "ts_utc": pd.date_range("2024-02-01", periods=100, freq="h"),
            "return": np.random.normal(0, 0.002, 100),
            "quantity": [100] * 100,
        })

        filtered_trades = tuner._apply_trading_filters(trades, params)

        # Should filter some trades based on k_sigma and uncertainty cut
        assert len(filtered_trades) <= len(trades)

        # Check quantity adjustment based on TP/SL ratio
        if not filtered_trades.empty and "quantity" in filtered_trades.columns:
            assert all(q >= 100 for q in filtered_trades["quantity"])

    def test_optimization_small_budget(self, tuner, sample_data):
        """Test optimization with small budget."""
        features, forecasts_30m, forecasts_60m, actuals, trades_30m, trades_60m = sample_data

        walkforward_config = WalkForwardConfig(
            start_date=date(2024, 2, 1),
            end_date=date(2024, 2, 15),
            train_months=1,
            test_months=1,
            purge_months=0,
            min_obs_per_period=10  # Low threshold for testing
        )

        results = tuner.optimize(
            features=features,
            forecasts_30m=forecasts_30m,
            forecasts_60m=forecasts_60m,
            actuals=actuals,
            trades_30m=trades_30m,
            trades_60m=trades_60m,
            walkforward_config=walkforward_config
        )

        # Check results structure
        assert "best_parameters" in results
        assert "optimization_summary" in results
        assert "top_combinations" in results
        assert "optimization_config" in results

        # Check best parameters
        best_params = results["best_parameters"]
        assert "k_sigma" in best_params
        assert "tp_sigma" in best_params
        assert "sl_sigma" in best_params
        assert "uncertainty_cut" in best_params
        assert "cadence_minutes" in best_params

        # Check optimization summary
        summary = results["optimization_summary"]
        assert summary["total_iterations"] == tuner.config.max_iterations
        assert 0 <= summary["success_rate"] <= 1.0

    def test_optimization_with_file_output(self, tuner, sample_data, tmp_path):
        """Test optimization with file output."""
        features, forecasts_30m, forecasts_60m, actuals, trades_30m, trades_60m = sample_data

        walkforward_config = WalkForwardConfig(
            start_date=date(2024, 2, 1),
            end_date=date(2024, 2, 15),
            train_months=1,
            test_months=1,
            purge_months=0,
            min_obs_per_period=10
        )

        output_file = tmp_path / "optimization_results.json"
        results = tuner.optimize(
            features=features,
            forecasts_30m=forecasts_30m,
            forecasts_60m=forecasts_60m,
            actuals=actuals,
            trades_30m=trades_30m,
            trades_60m=trades_60m,
            walkforward_config=walkforward_config,
            output_path=output_file
        )

        # Check file was created
        assert output_file.exists()

        # Check file content
        import json
        with output_file.open() as f:
            saved_results = json.load(f)

        # Compare key fields separately (JSON converts tuples to lists)
        assert saved_results["best_parameters"] == results["best_parameters"]
        assert saved_results["optimization_summary"] == results["optimization_summary"]
        assert saved_results["top_combinations"] == results["top_combinations"]
        # Convert ranges back to tuples for comparison
        saved_config = saved_results["optimization_config"].copy()
        saved_config["k_sigma_range"] = tuple(saved_config["k_sigma_range"])
        saved_config["uncertainty_cut_range"] = tuple(saved_config["uncertainty_cut_range"])
        saved_config["tp_sl_pairs"] = [tuple(pair) for pair in saved_config["tp_sl_pairs"]]
        assert saved_config == results["optimization_config"]

    def test_objective_metric_selection(self, sample_data):
        """Test optimization with different objective metrics."""

        # Test Sharpe optimization
        config_sharpe = HyperparameterConfig(
            max_iterations=5,
            objective_metric="sharpe",
            random_seed=42
        )
        tuner_sharpe = CompactHyperparameterTuner(config_sharpe)

        # Test CRPS optimization
        config_crps = HyperparameterConfig(
            max_iterations=5,
            objective_metric="crps",
            random_seed=42
        )
        tuner_crps = CompactHyperparameterTuner(config_crps)

        features, forecasts_30m, forecasts_60m, actuals, trades_30m, trades_60m = sample_data

        walkforward_config = WalkForwardConfig(
            start_date=date(2024, 2, 1),
            end_date=date(2024, 2, 15),
            train_months=1,
            test_months=1,
            purge_months=0,
            min_obs_per_period=10
        )

        results_sharpe = tuner_sharpe.optimize(
            features=features,
            forecasts_30m=forecasts_30m,
            forecasts_60m=forecasts_60m,
            actuals=actuals,
            trades_30m=trades_30m,
            trades_60m=trades_60m,
            walkforward_config=walkforward_config
        )

        results_crps = tuner_crps.optimize(
            features=features,
            forecasts_30m=forecasts_30m,
            forecasts_60m=forecasts_60m,
            actuals=actuals,
            trades_30m=trades_30m,
            trades_60m=trades_60m,
            walkforward_config=walkforward_config
        )

        # Check that objective metric is reflected in results
        assert results_sharpe["optimization_config"]["objective_metric"] == "sharpe"
        assert results_crps["optimization_config"]["objective_metric"] == "crps"

    def test_empty_data_handling(self, tuner):
        """Test handling of empty or insufficient data."""
        empty_features = pd.DataFrame()
        empty_forecasts = pd.DataFrame()
        empty_actuals = pd.DataFrame()
        empty_trades = pd.DataFrame()

        walkforward_config = WalkForwardConfig(
            start_date=date(2024, 2, 1),
            end_date=date(2024, 2, 15),
            train_months=1,
            test_months=1,
            purge_months=0
        )

        results = tuner.optimize(
            features=empty_features,
            forecasts_30m=empty_forecasts,
            forecasts_60m=empty_forecasts,
            actuals=empty_actuals,
            trades_30m=empty_trades,
            trades_60m=empty_trades,
            walkforward_config=walkforward_config
        )

        # Should still return results structure
        assert "best_parameters" in results
        assert "optimization_summary" in results

        # Best parameters should be default values since no evaluation occurred
        best_params = results["best_parameters"]
        assert best_params["is_valid"] is False  # Should be invalid due to no data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])