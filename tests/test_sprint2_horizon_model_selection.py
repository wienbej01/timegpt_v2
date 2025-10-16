"""Tests for Sprint 2: Horizon ↔ model selection policy."""

import pytest
import pandas as pd
import numpy as np
import logging
from unittest.mock import patch

from timegpt_v2.forecast.timegpt_client import TimeGPTClient, TimeGPTConfig


class TestHorizonValidation:
    """Test horizon validation as minutes."""

    def test_positive_horizon_passes(self):
        """Test that positive horizons pass validation."""
        config = TimeGPTConfig(model="timegpt-1")
        client = TimeGPTClient(config=config)

        # Should not raise exception
        client._validate_and_log_horizon(30)
        client._validate_and_log_horizon(60)
        client._validate_and_log_horizon(120)

    def test_zero_horizon_fails(self):
        """Test that zero horizon fails validation."""
        config = TimeGPTConfig()
        client = TimeGPTClient(config=config)

        with pytest.raises(ValueError, match="Horizon must be positive"):
            client._validate_and_log_horizon(0)

    def test_negative_horizon_fails(self):
        """Test that negative horizon fails validation."""
        config = TimeGPTConfig()
        client = TimeGPTClient(config=config)

        with pytest.raises(ValueError, match="Horizon must be positive"):
            client._validate_and_log_horizon(-30)

    def test_excessive_horizon_fails(self):
        """Test that horizon exceeding daily limit fails."""
        config = TimeGPTConfig()
        client = TimeGPTClient(config=config)

        with pytest.raises(ValueError, match="exceeds daily limit"):
            client._validate_and_log_horizon(1500)  # > 1440 minutes (1 day)

    def test_horizon_logged_as_minutes(self, caplog):
        """Test that horizon is logged as minutes."""
        config = TimeGPTConfig(model="timegpt-1")
        client = TimeGPTClient(config=config)

        with caplog.at_level(logging.INFO):
            client._validate_and_log_horizon(30)

        # Should log that horizon is interpreted as minutes
        assert any("interpreted as minutes" in record.message for record in caplog.records)
        assert any("h=30 interpreted as minutes" in record.message for record in caplog.records)


class TestModelSelectionPolicy:
    """Test model selection policy for intraday horizons."""

    def test_timegpt1_for_30min_horizon(self, caplog):
        """Test that timegpt-1 is used for 30-minute horizon."""
        config = TimeGPTConfig(model="timegpt-1")
        client = TimeGPTClient(config=config)

        with caplog.at_level(logging.INFO):
            client._validate_and_log_horizon(30)

        # Should log optimal model choice
        info_messages = [r.message for r in caplog.records if r.levelname == "INFO"]
        assert any("timegpt-1" in msg and "h=30" in msg for msg in info_messages)

    def test_timegpt1_for_60min_horizon(self, caplog):
        """Test that timegpt-1 is used for 60-minute horizon."""
        config = TimeGPTConfig(model="timegpt-1")
        client = TimeGPTClient(config=config)

        with caplog.at_level(logging.INFO):
            client._validate_and_log_horizon(60)

        # Should log optimal model choice
        info_messages = [r.message for r in caplog.records if r.levelname == "INFO"]
        assert any("timegpt-1" in msg and "h=60" in msg for msg in info_messages)

    def test_non_timegpt1_warning_for_30min(self, caplog):
        """Test that warning is issued for non-timegpt-1 model at 30min horizon."""
        config = TimeGPTConfig(model="timegpt-pro")  # Different model
        client = TimeGPTClient(config=config)

        with caplog.at_level(logging.WARNING):
            client._validate_and_log_horizon(30)

        # Should log warning about model choice
        warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("prefer 'timegpt-1'" in msg and "h=30" in msg for msg in warning_messages)

    def test_non_timegpt1_warning_for_60min(self, caplog):
        """Test that warning is issued for non-timegpt-1 model at 60min horizon."""
        config = TimeGPTConfig(model="timegpt-pro")  # Different model
        client = TimeGPTClient(config=config)

        with caplog.at_level(logging.WARNING):
            client._validate_and_log_horizon(60)

        # Should log warning about model choice
        warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("prefer 'timegpt-1'" in msg and "h=60" in msg for msg in warning_messages)

    def test_other_horizon_model_logging(self, caplog):
        """Test model logging for horizons outside 30-60min range."""
        config = TimeGPTConfig(model="timegpt-1")
        client = TimeGPTClient(config=config)

        with caplog.at_level(logging.INFO):
            client._validate_and_log_horizon(15)  # Outside optimal range

        # Should log that horizon is outside optimal range
        info_messages = [r.message for r in caplog.records if r.levelname == "INFO"]
        assert any("outside 30-60min optimal range" in msg for msg in info_messages)

    def test_once_per_run_logging(self, caplog):
        """Test that certain logs only appear once per run."""
        config = TimeGPTConfig(model="timegpt-1")
        client = TimeGPTClient(config=config)

        with caplog.at_level(logging.INFO):
            client._validate_and_log_horizon(30)
            client._validate_and_log_horizon(60)  # Second call

        # "Horizon validation" message should only appear once
        horizon_validation_logs = [
            r.message for r in caplog.records
            if "Horizon validation: h=" in r.message and "interpreted as minutes" in r.message
        ]
        assert len(horizon_validation_logs) == 1

        # Both horizons should be logged individually
        assert any("30min" in r.message for r in caplog.records)
        assert any("60min" in r.message for r in caplog.records)


class TestIntegrationHorizonAndForecast:
    """Integration tests for horizon validation in forecast calls."""

    def test_forecall_calls_horizon_validation(self, caplog):
        """Test that forecast method calls horizon validation."""
        config = TimeGPTConfig(model="timegpt-1")
        client = TimeGPTClient(config=config)

        # Create dummy data
        y_df = pd.DataFrame({
            "unique_id": ["AAPL"],
            "ds": pd.date_range("2024-01-01 09:30", periods=100, freq="1min"),
            "y": np.random.normal(0, 0.01, 100)
        })

        with caplog.at_level(logging.INFO):
            try:
                client.forecast(
                    y_df=y_df,
                    x_df=None,
                    features=pd.DataFrame(),
                    snapshot_ts=pd.Timestamp("2024-01-01 10:00"),
                    run_id="test",
                    horizon=30
                )
            except Exception:
                pass  # Expected to fail due to no backend

        # Should have horizon validation logs
        assert any("interpreted as minutes" in record.message for record in caplog.records)
        assert any("timegpt-1" in record.message and "30min" in record.message for record in caplog.records)

    def test_invalid_horizon_in_forecast(self):
        """Test that invalid horizon in forecast call raises error."""
        config = TimeGPTConfig(model="timegpt-1")
        client = TimeGPTClient(config=config)

        y_df = pd.DataFrame({
            "unique_id": ["AAPL"],
            "ds": pd.date_range("2024-01-01 09:30", periods=100, freq="1min"),
            "y": np.random.normal(0, 0.01, 100)
        })

        with pytest.raises(ValueError, match="Horizon must be positive"):
            client.forecast(
                y_df=y_df,
                x_df=None,
                features=pd.DataFrame(),
                snapshot_ts=pd.Timestamp("2024-01-01 10:00"),
                run_id="test",
                horizon=0
            )


class TestParameterOverride:
    """Test parameter override functionality."""

    def test_horizon_override_in_config(self):
        """Test that horizon can be overridden in config."""
        config = TimeGPTConfig(model="timegpt-1", horizon=45)  # Non-standard horizon
        client = TimeGPTClient(config=config)

        # Should use the overridden horizon
        client._validate_and_log_horizon(45)

        # Config should preserve the overridden value
        assert config.horizon == 45

    def test_model_override_with_warning(self, caplog):
        """Test model override with warning."""
        config = TimeGPTConfig(model="timegpt-pro")  # Override default
        client = TimeGPTClient(config=config)

        with caplog.at_level(logging.WARNING):
            client._validate_and_log_horizon(30)

        # Should warn about non-optimal model choice
        warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("prefer 'timegpt-1'" in msg for msg in warning_messages)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])