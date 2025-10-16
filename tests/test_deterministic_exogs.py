"""Tests for deterministic future exogenous features."""

from __future__ import annotations

import pandas as pd
import pytest

from timegpt_v2.framing.build_payloads import build_x_df_for_horizon, build_y_df


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Create sample feature data with deterministic columns."""
    timestamps = pd.date_range("2024-07-01 09:30:00", periods=100, freq="1min", tz="UTC")

    data = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": "AAPL",
        "target_log_return_1m": [0.001] * 100,
        "fourier_sin_1": [0.1] * 100,
        "fourier_cos_1": [0.9] * 100,
        "day_of_week": [0] * 100,  # Monday
        "minute_of_day": list(range(100)),
        "minutes_since_open": list(range(100)),  # Mock values
        "minutes_to_close": [390 - i for i in range(100)],  # Mock values
        "spy_ret_1m": [0.0005] * 100,
    })

    return data


def test_deterministic_features_in_y_df(sample_features: pd.DataFrame) -> None:
    """Test that deterministic features appear in y_df."""
    snapshot_ts = pd.Timestamp("2024-07-01 10:00:00", tz="UTC")

    y_df = build_y_df(
        sample_features,
        snapshot_ts,
        target_column="target_log_return_1m",
        rolling_window_days=1,
        min_obs_subhourly=1,
        symbols=["AAPL"]
    )

    # Check that deterministic features appear in y_df
    deterministic_features = [
        "fourier_sin_1",  # minute_of_day_sin equivalent
        "fourier_cos_1",  # minute_of_day_cos equivalent
        "day_of_week",
        "minutes_since_open",
        "minutes_to_close",
    ]

    for feat in deterministic_features:
        assert feat in y_df.columns, f"Deterministic feature {feat} missing from y_df"

    # Check that values are reasonable
    assert (y_df["fourier_sin_1"].abs() <= 1).all(), "fourier_sin_1 should be bounded [-1,1]"
    assert (y_df["fourier_cos_1"].abs() <= 1).all(), "fourier_cos_1 should be bounded [-1,1]"
    assert (y_df["day_of_week"] >= 0).all(), "day_of_week should be non-negative"
    assert (y_df["day_of_week"] <= 6).all(), "day_of_week should be <= 6"
    assert (y_df["minutes_since_open"] >= 0).all(), "minutes_since_open should be non-negative"
    assert (y_df["minutes_to_close"] >= 0).all(), "minutes_to_close should be non-negative"


def test_deterministic_features_in_x_df(sample_features: pd.DataFrame) -> None:
    """Test that deterministic features appear in x_df and are computed for future timestamps."""
    snapshot_ts = pd.Timestamp("2024-07-01 10:00:00", tz="UTC")
    horizon_minutes = 15

    x_df = build_x_df_for_horizon(
        sample_features,
        snapshot_ts,
        horizon_minutes,
        symbols=["AAPL"]
    )

    # Check that x_df has the expected shape
    assert len(x_df) == horizon_minutes, f"x_df should have {horizon_minutes} rows"
    assert x_df["unique_id"].nunique() == 1, "Should have one unique symbol"

    # Check that deterministic features appear in x_df
    deterministic_features = [
        "fourier_sin_1",  # minute_of_day_sin equivalent
        "fourier_cos_1",  # minute_of_day_cos equivalent
        "day_of_week",
        "minutes_since_open",
        "minutes_to_close",
    ]

    for feat in deterministic_features:
        assert feat in x_df.columns, f"Deterministic feature {feat} missing from x_df"
        assert x_df[feat].notna().all(), f"Deterministic feature {feat} should have no NaNs"

    # Check that values are reasonable for future timestamps
    assert (x_df["fourier_sin_1"].abs() <= 1).all(), "fourier_sin_1 should be bounded [-1,1]"
    assert (x_df["fourier_cos_1"].abs() <= 1).all(), "fourier_cos_1 should be bounded [-1,1]"
    assert (x_df["day_of_week"] >= 0).all(), "day_of_week should be non-negative"
    assert (x_df["day_of_week"] <= 6).all(), "day_of_week should be <= 6"
    assert (x_df["minutes_since_open"] >= 0).all(), "minutes_since_open should be non-negative"
    assert (x_df["minutes_to_close"] >= 0).all(), "minutes_to_close should be non-negative"


def test_time_progression_in_x_df(sample_features: pd.DataFrame) -> None:
    """Test that deterministic features progress correctly over future timestamps."""
    # Use 2:00 PM ET which is 18:00 UTC (during RTH)
    snapshot_ts = pd.Timestamp("2024-07-01 18:00:00", tz="UTC")
    horizon_minutes = 10

    x_df = build_x_df_for_horizon(
        sample_features,
        snapshot_ts,
        horizon_minutes,
        symbols=["AAPL"]
    )

    # Convert timestamps to ET for verification
    timestamps_et = pd.to_datetime(x_df["ds"], utc=True).dt.tz_convert("America/New_York")

    # Check that minutes_since_open increases over time
    minutes_since_open = x_df["minutes_since_open"].values
    assert (minutes_since_open[1:] >= minutes_since_open[:-1]).all(), "minutes_since_open should be non-decreasing"

    # Check that minutes_to_close decreases over time (approaching close)
    minutes_to_close = x_df["minutes_to_close"].values
    assert (minutes_to_close[1:] <= minutes_to_close[:-1]).all(), "minutes_to_close should be non-increasing"

    # Verify specific values for known times
    # First future minute after 18:00 UTC (2:00 PM ET) is 18:01 UTC (2:01 PM ET)
    # 2:01 PM ET should be 4 hours 31 minutes after 9:30 AM open = 271 minutes
    expected_since_open = 14 * 60 + 1 - (9 * 60 + 30)  # 2:01 PM = 14:01 - 9:30 = 271 minutes
    assert minutes_since_open[0] == expected_since_open, f"First minute should be {expected_since_open} minutes after open, got {minutes_since_open[0]}"

    # 2:01 PM ET should be 1 hour 59 minutes before 4:00 PM close = 119 minutes
    expected_to_close = (16 * 60 + 0) - (14 * 60 + 1)  # 4:00 PM - 2:01 PM = 119 minutes
    assert minutes_to_close[0] == expected_to_close, f"First minute should be {expected_to_close} minutes before close, got {minutes_to_close[0]}"


def test_seasonal_features(sample_features: pd.DataFrame) -> None:
    """Test that seasonal features (fourier components) are computed correctly."""
    snapshot_ts = pd.Timestamp("2024-07-01 10:00:00", tz="UTC")
    horizon_minutes = 5

    x_df = build_x_df_for_horizon(
        sample_features,
        snapshot_ts,
        horizon_minutes,
        symbols=["AAPL"]
    )

    # Check that sin/cos components maintain proper relationship
    sin_values = x_df["fourier_sin_1"].values
    cos_values = x_df["fourier_cos_1"].values

    # For each time point, sin^2 + cos^2 should equal 1 (unit circle)
    for i in range(len(sin_values)):
        magnitude_squared = sin_values[i]**2 + cos_values[i]**2
        assert abs(magnitude_squared - 1.0) < 1e-10, f"sin^2 + cos^2 should equal 1, got {magnitude_squared}"

    # Check that day_of_week is consistent across the horizon
    day_values = x_df["day_of_week"].values
    assert (day_values == day_values[0]).all(), "day_of_week should be constant within a short horizon"


def test_rth_boundaries() -> None:
    """Test deterministic features at RTH boundaries."""

    # Test data at RTH open (9:30 AM ET = 13:30 UTC)
    open_timestamp = pd.Timestamp("2024-07-01 13:30:00", tz="UTC")
    test_data = pd.DataFrame({
        "timestamp": [open_timestamp],
        "symbol": "AAPL",
        "target_log_return_1m": [0.001],
    })

    # Test x_df at RTH open - first future minute is 9:31 AM ET
    x_df_open = build_x_df_for_horizon(
        test_data,
        open_timestamp,
        horizon_minutes=1,
        symbols=["AAPL"]
    )

    # At 9:31 AM ET, minutes_since_open should be 1 (minute after 9:30)
    assert x_df_open["minutes_since_open"].iloc[0] == 1, "At 9:31 AM ET, minutes_since_open should be 1"

    # Test data near RTH close (3:58 PM ET = 19:58 UTC)
    close_timestamp = pd.Timestamp("2024-07-01 19:58:00", tz="UTC")
    test_data_close = pd.DataFrame({
        "timestamp": [close_timestamp],
        "symbol": "AAPL",
        "target_log_return_1m": [0.001],
    })

    # Test x_df near RTH close - first future minute is 3:59 PM ET
    x_df_close = build_x_df_for_horizon(
        test_data_close,
        close_timestamp,
        horizon_minutes=1,
        symbols=["AAPL"]
    )

    # At 3:59 PM ET, minutes_to_close should be 1 (1 minute before 4:00 PM)
    assert x_df_close["minutes_to_close"].iloc[0] == 1, "At 3:59 PM ET, minutes_to_close should be 1"


def test_payload_logging_with_deterministic_features(sample_features: pd.DataFrame, monkeypatch) -> None:
    """Test that payload logging includes deterministic features."""
    import os
    import logging

    # Enable payload logging
    monkeypatch.setenv("PAYLOAD_LOG", "1")

    # Capture logs
    logger = logging.getLogger("timegpt_v2.build_payloads")

    # Create a handler to capture log messages
    import io
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        snapshot_ts = pd.Timestamp("2024-07-01 10:00:00", tz="UTC")

        # Build x_df to trigger payload logging
        x_df = build_x_df_for_horizon(
            sample_features,
            snapshot_ts,
            horizon_minutes=5,
            symbols=["AAPL"]
        )

        # Check that log contains deterministic features
        log_output = log_capture.getvalue()
        assert "PAYLOAD x_df cols=" in log_output, "Payload logging should trigger"
        assert "fourier_sin_1" in log_output, "Payload log should include fourier_sin_1"
        assert "fourier_cos_1" in log_output, "Payload log should include fourier_cos_1"
        assert "day_of_week" in log_output, "Payload log should include day_of_week"
        assert "minutes_since_open" in log_output, "Payload log should include minutes_since_open"
        assert "minutes_to_close" in log_output, "Payload log should include minutes_to_close"

    finally:
        logger.removeHandler(handler)


def test_deterministic_features_only_in_x_df(sample_features: pd.DataFrame) -> None:
    """Test that x_df contains ONLY deterministic features, not historical exogenous features."""
    snapshot_ts = pd.Timestamp("2024-07-01 10:00:00", tz="UTC")

    x_df = build_x_df_for_horizon(
        sample_features,
        snapshot_ts,
        horizon_minutes=5,
        symbols=["AAPL"]
    )

    # x_df should contain only deterministic features, not historical ones
    deterministic_allowed = [
        "unique_id", "ds",
        "fourier_sin_1", "fourier_cos_1", "fourier_sin_2", "fourier_cos_2", "fourier_sin_3", "fourier_cos_3",
        "day_of_week", "is_month_end",
        "minutes_since_open", "minutes_to_close",
        "minute_of_day", "minute_index", "minute_progress",
        "session_open", "session_lunch", "session_power"
    ]

    # Check that all columns are deterministic
    for col in x_df.columns:
        assert col in deterministic_allowed, f"Column {col} should not be in x_df (only deterministic features allowed)"

    # Check that no historical features are present
    historical_forbidden = ["spy_ret_1m", "target_log_return_1m", "sigma_5m", "parkinson_sigma_15m"]
    for col in historical_forbidden:
        assert col not in x_df.columns, f"Historical feature {col} should not be in x_df"


if __name__ == "__main__":
    pytest.main([__file__])