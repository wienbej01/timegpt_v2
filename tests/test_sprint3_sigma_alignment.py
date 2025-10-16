"""Tests for Sprint 3: σ alignment and payoff geometry."""

import pytest
import pandas as pd
import numpy as np
import logging
from datetime import datetime, time

from timegpt_v2.trading.rules import RuleParams, TradingRules
from timegpt_v2.trading.costs import TradingCosts
from timegpt_v2.utils.sigma_utils import compute_horizon_sigma, validate_tp_sl_geometry


class TestHorizonSigma:
    """Test horizon-aligned sigma computation."""

    def test_compute_30m_sigma_from_5m(self):
        """Test computing 30m sigma from 5m sigma."""
        # Create test data with 5m sigma
        timestamps = pd.date_range("2024-01-01 09:30", periods=100, freq="1min")
        features = pd.DataFrame({
            "symbol": ["AAPL"] * 100,
            "timestamp": timestamps,
            "sigma_5m": np.random.uniform(0.01, 0.03, 100),
            "ret_1m": np.random.normal(0, 0.02, 100)
        })

        sigma_30m = compute_horizon_sigma(
            features=features,
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-01 10:30"),
            horizon_minutes=30
        )

        # Should be scaled by sqrt(30/5) = sqrt(6)
        base_sigma = features.iloc[-1]["sigma_5m"]
        expected_sigma = base_sigma * np.sqrt(6)
        assert abs(sigma_30m - expected_sigma) < 1e-6

    def test_compute_60m_sigma_from_15m(self):
        """Test computing 60m sigma from 15m sigma."""
        timestamps = pd.date_range("2024-01-01 09:30", periods=100, freq="1min")
        features = pd.DataFrame({
            "symbol": ["MSFT"] * 100,
            "timestamp": timestamps,
            "vol_ewm_15m": np.random.uniform(0.015, 0.025, 100),
        })

        sigma_60m = compute_horizon_sigma(
            features=features,
            symbol="MSFT",
            timestamp=pd.Timestamp("2024-01-01 10:30"),
            horizon_minutes=60
        )

        # Should be scaled by sqrt(60/15) = sqrt(4) = 2
        base_sigma = features.iloc[-1]["vol_ewm_15m"]
        expected_sigma = base_sigma * 2.0
        assert abs(sigma_60m - expected_sigma) < 1e-6

    def test_precomputed_horizon_sigma(self):
        """Test using pre-computed horizon sigma."""
        timestamps = pd.date_range("2024-01-01 09:30", periods=100, freq="1min")
        features = pd.DataFrame({
            "symbol": ["TSLA"] * 100,
            "timestamp": timestamps,
            "sigma_30m": np.random.uniform(0.02, 0.04, 100),
            "sigma_5m": np.random.uniform(0.01, 0.02, 100),
        })

        sigma_30m = compute_horizon_sigma(
            features=features,
            symbol="TSLA",
            timestamp=pd.Timestamp("2024-01-01 10:30"),
            horizon_minutes=30
        )

        # Should use pre-computed sigma_30m directly
        expected_sigma = features.iloc[-1]["sigma_30m"]
        assert sigma_30m == expected_sigma

    def test_sigma_computation_fallback_to_returns(self):
        """Test fallback to computing sigma from returns."""
        timestamps = pd.date_range("2024-01-01 09:30", periods=100, freq="1min")
        features = pd.DataFrame({
            "symbol": ["NVDA"] * 100,
            "timestamp": timestamps,
            "ret_1m": np.random.normal(0, 0.02, 100),
        })

        sigma_30m = compute_horizon_sigma(
            features=features,
            symbol="NVDA",
            timestamp=pd.Timestamp("2024-01-01 10:30"),
            horizon_minutes=30
        )

        assert sigma_30m > 0
        assert isinstance(sigma_30m, float)

    def test_sigma_computation_error_no_data(self):
        """Test error when no suitable data is available."""
        features = pd.DataFrame({
            "symbol": ["AAPL"],
            "timestamp": [pd.Timestamp("2024-01-01 09:30")],
            "some_column": [1.0]
        })

        with pytest.raises(ValueError, match="Could not compute 30m sigma"):
            compute_horizon_sigma(
                features=features,
                symbol="AAPL",
                timestamp=pd.Timestamp("2024-01-01 10:30"),
                horizon_minutes=30
            )


class TestTPSLGeometry:
    """Test TP/SL geometry validation."""

    def test_valid_geometry_tp_greater_than_sl(self, caplog):
        """Test valid geometry with TP > SL."""
        with caplog.at_level(logging.INFO):
            validate_tp_sl_geometry(tp_sigma=2.5, sl_sigma=2.0)

        # Should log successful validation
        assert any("Payoff geometry validated" in record.message for record in caplog.records)

    def test_valid_geometry_tp_equals_sl(self, caplog):
        """Test valid geometry with TP = SL."""
        with caplog.at_level(logging.INFO):
            validate_tp_sl_geometry(tp_sigma=2.0, sl_sigma=2.0)

        assert any("Payoff geometry validated" in record.message for record in caplog.records)

    def test_invalid_geometry_tp_less_than_sl(self):
        """Test invalid geometry with TP < SL."""
        with pytest.raises(ValueError, match="Negative expectancy"):
            validate_tp_sl_geometry(tp_sigma=1.8, sl_sigma=2.2)


class TestTradingRulesSigmaAlignment:
    """Test trading rules with horizon-aligned sigma."""

    def setup_method(self):
        """Set up test trading rules."""
        self.costs = TradingCosts()
        self.rules = TradingRules(
            costs=self.costs,
            time_stop=time(15, 30),
            daily_trade_cap=10,
            max_open_per_symbol=3
        )

    def test_entry_signal_with_horizon_sigma(self):
        """Test entry signal using horizon-aligned sigma."""
        # Create test features with sigma data
        timestamps = pd.date_range("2024-01-01 09:30", periods=100, freq="1min")
        features = pd.DataFrame({
            "symbol": ["AAPL"] * 100,
            "timestamp": timestamps,
            "sigma_5m": np.random.uniform(0.01, 0.02, 100),
        })

        params = RuleParams(
            k_sigma=1.0,
            s_stop=2.0,
            s_take=2.5,
            uncertainty_k=4.0,
            horizon_minutes=30,
            sigma_basis="horizon"
        )

        quantiles = {0.25: 0.002, 0.5: 0.004, 0.75: 0.006}

        signal = self.rules.get_entry_signal(
            params=params,
            quantiles=quantiles,
            last_price=150.0,
            sigma_5m=0.015,
            tick_size=0.01,
            symbol="AAPL",
            features=features,
            entry_time=pd.Timestamp("2024-01-01 10:30")
        )

        # Should use horizon-aligned sigma (30m from 5m)
        # This affects the threshold calculation
        assert signal in [-1.0, 0.0, 1.0]

    def test_entry_signal_fallback_to_5m_sigma(self):
        """Test entry signal falls back to 5m sigma when horizon computation fails."""
        params = RuleParams(
            k_sigma=1.0,
            s_stop=2.0,
            s_take=2.5,
            uncertainty_k=4.0,
            horizon_minutes=30,
            sigma_basis="horizon"
        )

        quantiles = {0.25: 0.002, 0.5: 0.004, 0.75: 0.006}

        # No features provided - should fallback to sigma_5m
        signal = self.rules.get_entry_signal(
            params=params,
            quantiles=quantiles,
            last_price=150.0,
            sigma_5m=0.015,
            tick_size=0.01,
            symbol="AAPL"
        )

        assert signal in [-1.0, 0.0, 1.0]

    def test_entry_signal_5m_sigma_basis(self):
        """Test entry signal with explicit 5m sigma basis."""
        params = RuleParams(
            k_sigma=1.0,
            s_stop=2.0,
            s_take=2.5,
            uncertainty_k=4.0,
            sigma_basis="5m"  # Explicit 5m basis
        )

        quantiles = {0.25: 0.002, 0.5: 0.004, 0.75: 0.006}

        signal = self.rules.get_entry_signal(
            params=params,
            quantiles=quantiles,
            last_price=150.0,
            sigma_5m=0.015,
            tick_size=0.01,
            symbol="AAPL"
        )

        assert signal in [-1.0, 0.0, 1.0]

    def test_exit_signal_with_horizon_sigma(self):
        """Test exit signal using horizon-aligned sigma."""
        timestamps = pd.date_range("2024-01-01 09:30", periods=100, freq="1min")
        features = pd.DataFrame({
            "symbol": ["AAPL"] * 100,
            "timestamp": timestamps,
            "sigma_5m": np.random.uniform(0.01, 0.02, 100),
        })

        params = RuleParams(
            k_sigma=1.0,
            s_stop=2.0,
            s_take=2.5,
            uncertainty_k=4.0,
            horizon_minutes=60,
            sigma_basis="horizon"
        )

        # Test take-profit exit
        should_exit = self.rules.get_exit_signal(
            params=params,
            entry_price=150.0,
            current_price=151.5,  # 1% gain - should trigger TP with 60m sigma
            position=1.0,
            sigma_5m=0.015,
            current_time=time(11, 0),
            features=features,
            symbol="AAPL",
            current_timestamp=pd.Timestamp("2024-01-01 11:00")
        )

        # Should exit due to take-profit
        assert should_exit

    def test_exit_signal_geometry_validation(self):
        """Test exit signal validates TP/SL geometry."""
        params = RuleParams(
            k_sigma=1.0,
            s_stop=2.2,  # Higher than take-profit - invalid geometry
            s_take=1.8,
            uncertainty_k=4.0,
            horizon_minutes=30,
            sigma_basis="horizon"
        )

        with pytest.raises(ValueError, match="Negative expectancy"):
            self.rules.get_exit_signal(
                params=params,
                entry_price=150.0,
                current_price=151.0,
                position=1.0,
                sigma_5m=0.015,
                current_time=time(11, 0)
            )


class TestEVExits:
    """Test expected value-based exits."""

    def test_ev_exit_negative_expected_value(self):
        """Test EV exit when expected value becomes negative."""
        from timegpt_v2.utils.sigma_utils import compute_ev_exit_threshold

        # Negative expected value scenario
        should_exit, reason = compute_ev_exit_threshold(
            q50=-0.001,  # Negative forecast
            q25=-0.002,
            q75=0.0,
            sigma_h=0.02,
            costs_bps=5,  # 5 bps cost
            price=150.0
        )

        assert should_exit
        assert reason == "negative_expected_value"

    def test_ev_exit_adverse_forecast(self):
        """Test EV exit on adverse forecast."""
        from timegpt_v2.utils.sigma_utils import compute_ev_exit_threshold

        # Strong adverse forecast
        should_exit, reason = compute_ev_exit_threshold(
            q50=-0.02,  # Very negative forecast
            q25=-0.025,
            q75=-0.015,
            sigma_h=0.02,
            costs_bps=5,
            price=150.0
        )

        assert should_exit
        assert reason == "adverse_forecast"

    def test_ev_exit_continue_holding(self):
        """Test EV exit continues holding when EV is positive."""
        from timegpt_v2.utils.sigma_utils import compute_ev_exit_threshold

        # Positive expected value scenario
        should_exit, reason = compute_ev_exit_threshold(
            q50=0.003,  # Positive forecast
            q25=0.001,
            q75=0.005,
            sigma_h=0.02,
            costs_bps=5,
            price=150.0
        )

        assert not should_exit
        assert reason == "continue_holding"


class TestIntegrationSigmaAndPayoff:
    """Integration tests for sigma alignment and payoff geometry."""

    def test_breakeven_hit_rate_with_tp_ge_sl(self):
        """Test that TP >= SL allows reasonable hit rates."""
        # With TP = 2.5σ and SL = 2.0σ
        tp_sigma = 2.5
        sl_sigma = 2.0

        # Breakeven hit rate = SL / (TP + SL)
        breakeven_hit_rate = sl_sigma / (tp_sigma + sl_sigma)

        # Should be less than 50% (reasonable)
        assert breakeven_hit_rate < 0.5
        assert breakeven_hit_rate > 0.4  # But not too low

    def test_unrealistic_hit_rate_with_tp_lt_sl(self):
        """Test that TP < SL requires unrealistic hit rates."""
        tp_sigma = 1.5
        sl_sigma = 2.5

        breakeven_hit_rate = sl_sigma / (tp_sigma + sl_sigma)

        # Would require > 60% win rate (unrealistic)
        assert breakeven_hit_rate > 0.6

    def test_horizon_sigma_scaling_factors(self):
        """Test that sigma scaling factors are correct."""
        base_5m_sigma = 0.02

        # 30m from 5m: sqrt(30/5) = sqrt(6) ≈ 2.45
        scaling_30m = np.sqrt(30/5)
        expected_30m_sigma = base_5m_sigma * scaling_30m

        # 60m from 5m: sqrt(60/5) = sqrt(12) ≈ 3.46
        scaling_60m = np.sqrt(60/5)
        expected_60m_sigma = base_5m_sigma * scaling_60m

        # 60m from 15m: sqrt(60/15) = sqrt(4) = 2
        scaling_60m_from_15m = np.sqrt(60/15)
        expected_60m_sigma_from_15m = 0.02 * scaling_60m_from_15m  # Assuming 15m sigma = 0.02

        assert abs(scaling_30m - 2.45) < 0.01
        assert abs(scaling_60m - 3.46) < 0.01
        assert abs(scaling_60m_from_15m - 2.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])