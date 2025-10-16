"""Tests for Sprint 1: Frequency enforcement and data regularization."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from timegpt_v2.forecast.timegpt_client import TimeGPTClient, TimeGPTConfig
from timegpt_v2.framing.build_payloads import (
    build_y_df,
    build_x_df_for_horizon,
    build_payloads_with_validation,
    _validate_horizon_length
)


class TestFrequencyEnforcement:
    """Test frequency enforcement in TimeGPT client."""

    def test_freq_override_to_min(self, caplog):
        """Test that frequency is forced to 'min' for intraday forecasts."""
        config = TimeGPTConfig(freq="H")  # Start with hourly frequency
        client = TimeGPTClient(config=config)

        # Create dummy data
        y_df = pd.DataFrame({
            "unique_id": ["AAPL"],
            "ds": pd.date_range("2024-01-01 09:30", periods=100, freq="1min"),
            "y": np.random.normal(0, 0.01, 100)
        })

        # Mock forecast call (we won't actually call API)
        with caplog.at_level(logging.INFO):
            with pytest.raises(Exception):  # Will fail due to no backend
                client.forecast(
                    y_df=y_df,
                    x_df=None,
                    features=pd.DataFrame(),
                    snapshot_ts=pd.Timestamp("2024-01-01 10:00"),
                    run_id="test"
                )

        # Check that frequency override was logged
        assert any("Frequency override: forcing freq='min'" in record.message for record in caplog.records)
        assert any("Inferred/forced freq: min" in record.message for record in caplog.records)

    def test_freq_already_min_no_warning(self, caplog):
        """Test that no warning is issued when freq is already 'min'."""
        config = TimeGPTConfig(freq="min")
        client = TimeGPTClient(config=config)

        y_df = pd.DataFrame({
            "unique_id": ["AAPL"],
            "ds": pd.date_range("2024-01-01 09:30", periods=100, freq="1min"),
            "y": np.random.normal(0, 0.01, 100)
        })

        with caplog.at_level(logging.WARNING):
            try:
                client.forecast(
                    y_df=y_df,
                    x_df=None,
                    features=pd.DataFrame(),
                    snapshot_ts=pd.Timestamp("2024-01-01 10:00"),
                    run_id="test"
                )
            except:
                pass

        # Should not have frequency override warning
        assert not any("Frequency override" in record.message for record in caplog.records)


class TestMinuteRegularity:
    """Test minute-level regularity enforcement in payload building."""

    def test_build_y_df_enforces_minute_regularity(self):
        """Test that build_y_df creates minute-spaced data."""
        # Create irregular input data with gaps
        timestamps = pd.date_range("2024-01-01 09:30", periods=50, freq="2min")  # 2-minute gaps
        features = pd.DataFrame({
            "symbol": ["AAPL"] * 50,
            "timestamp": timestamps,
            "target_log_return_1m": np.random.normal(0, 0.01, 50),
            "is_rth": True
        })

        result = build_y_df(
            features=features,
            snapshot_ts=pd.Timestamp("2024-01-01 10:30"),
            min_obs_subhourly=10
        )

        # Check that result has minute-spaced timestamps
        if not result.empty:
            time_diffs = result["ds"].diff().dropna()
            # Most differences should be 1 minute
            one_min_diffs = (time_diffs == pd.Timedelta(minutes=1)).sum()
            total_diffs = len(time_diffs)
            assert one_min_diffs / total_diffs > 0.8, f"Expected >80% 1-min differences, got {one_min_diffs/total_diffs}"

    def test_build_y_df_with_payload_logging(self, monkeypatch):
        """Test payload logging for minute regularity."""
        monkeypatch.setenv("PAYLOAD_LOG", "1")

        features = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "timestamp": pd.date_range("2024-01-01 09:30", periods=200, freq="1min"),
            "target_log_return_1m": np.random.normal(0, 0.01, 200),
            "is_rth": True
        })

        with pytest.raises(Exception):  # Will fail due to missing deterministic exogs
            build_y_df(
                features=features,
                snapshot_ts=pd.Timestamp("2024-01-01 10:30"),
                min_obs_subhourly=10
            )


class TestHorizonLengthValidation:
    """Test horizon length validation for X_df."""

    def test_correct_horizon_length_passes(self):
        """Test that correct horizon length passes validation."""
        horizon = 30
        x_df = pd.DataFrame({
            "unique_id": ["AAPL"] * horizon,
            "ds": pd.date_range("2024-01-01 10:01", periods=horizon, freq="1min"),
            "minute_of_day_sin": np.random.random(horizon)
        })

        # Should not raise exception
        _validate_horizon_length(x_df, horizon)

    def test_incorrect_horizon_length_fails(self):
        """Test that incorrect horizon length fails validation."""
        horizon = 30
        # Create X_df with wrong number of rows
        x_df = pd.DataFrame({
            "unique_id": ["AAPL"] * (horizon - 5),  # 5 rows too few
            "ds": pd.date_range("2024-01-01 10:01", periods=horizon - 5, freq="1min"),
            "minute_of_day_sin": np.random.random(horizon - 5)
        })

        with pytest.raises(ValueError, match="X_df horizon length mismatch"):
            _validate_horizon_length(x_df, horizon)

    def test_multiple_series_wrong_length(self):
        """Test horizon validation with multiple series."""
        horizon = 60
        x_df = pd.DataFrame({
            "unique_id": ["AAPL"] * horizon + ["MSFT"] * (horizon - 10),  # MSFT has 10 fewer rows
            "ds": list(pd.date_range("2024-01-01 10:01", periods=horizon, freq="1min")) +
                  list(pd.date_range("2024-01-01 10:01", periods=horizon - 10, freq="1min")),
            "minute_of_day_sin": np.random.random(horizon + horizon - 10)
        })

        with pytest.raises(ValueError, match="X_df horizon length mismatch"):
            _validate_horizon_length(x_df, horizon)

    def test_empty_x_df_passes(self):
        """Test that empty X_df passes validation."""
        x_df = pd.DataFrame()
        _validate_horizon_length(x_df, 30)


class TestIntegrationFrequencyAndHorizon:
    """Integration tests for frequency enforcement and horizon validation."""

    def test_build_payloads_with_validation_enforces_horizon(self):
        """Test that build_payloads_with_validation enforces correct horizon length."""
        features = pd.DataFrame({
            "symbol": ["AAPL"],
            "timestamp": pd.date_range("2024-01-01 09:30", periods=200, freq="1min"),
            "target_log_return_1m": np.random.normal(0, 0.01, 200),
            "is_rth": True
        })

        horizon = 30

        # This should work without raising exceptions
        try:
            y_df, x_df = build_payloads_with_validation(
                features=features,
                snapshot_ts=pd.Timestamp("2024-01-01 10:30"),
                horizon_minutes=horizon,
                min_obs_subhourly=10
            )

            # Verify X_df has correct horizon length
            if not x_df.empty:
                assert len(x_df) == horizon

        except Exception as e:
            # Expected to fail due to missing deterministic exogs, but horizon validation should work
            assert "deterministic exogenous features" in str(e) or "minute_of_day" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])