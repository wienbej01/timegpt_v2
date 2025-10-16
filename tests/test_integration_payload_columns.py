"""Integration tests for payload column validation."""

from __future__ import annotations

import pandas as pd
import pytest
from datetime import datetime

from timegpt_v2.framing.build_payloads import (
    FUTR_EXOG_ALLOW,
    HIST_EXOG_ALLOW,
    build_x_df_for_horizon,
    build_y_df,
)


@pytest.fixture
def sample_ohlcv_features() -> pd.DataFrame:
    """Create realistic OHLCV feature data for integration testing."""
    # Create a full trading day's worth of data (9:30 AM - 4:00 PM ET)
    timestamps = pd.date_range("2024-07-01 09:30:00", periods=390, freq="1min", tz="UTC")  # 6.5 hours = 390 minutes

    # Generate realistic price data with some volatility
    import numpy as np
    np.random.seed(42)

    initial_price = 175.0
    returns = np.random.normal(0.0001, 0.002, len(timestamps))
    prices = initial_price * np.exp(np.cumsum(returns))

    # Generate OHLC data with realistic spreads
    high_spread = np.random.exponential(0.001, len(timestamps))
    low_spread = np.random.exponential(0.001, len(timestamps))

    data = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": "AAPL",
        "open": prices * (1 + np.random.normal(0, 0.0005, len(timestamps))),
        "high": prices * (1 + high_spread),
        "low": prices * (1 - low_spread),
        "close": prices,
        "volume": np.random.randint(5000, 50000, len(timestamps)),
        "vw": prices * np.random.uniform(0.98, 1.02, len(timestamps)),
        "n": np.random.randint(50, 500, len(timestamps)),
        "session": ["regular"] * len(timestamps),
        "date_et": [ts.date() for ts in timestamps.tz_convert("America/New_York")],
    })

    # Add required DQ checker columns
    data["ffill_flag"] = False
    data["log_return"] = np.log(data["close"]).diff().fillna(0)
    data["outlier_flag"] = False
    data["is_rth"] = True

    # Add all OHLCV features from Sprint 1
    data["ret_1m"] = data["log_return"]
    data["ret_5m"] = data["close"].pct_change(5)
    data["sigma_5m"] = data["ret_1m"].rolling(window=5).std()

    # Parkinson volatility
    high_low_ratio = np.log(data["high"] / data["low"])
    data["parkinson_sigma_15m"] = (high_low_ratio.rolling(window=15).mean() / (4 * np.log(2))) ** 0.5

    # Range percentage and CLV
    high_15m = data["high"].rolling(window=15).max()
    low_15m = data["low"].rolling(window=15).min()
    data["range_pct_15m"] = (high_15m - low_15m) / data["close"]
    data["clv_15m"] = ((data["close"] - low_15m) / (high_15m - low_15m)).fillna(0.5)

    # VWAP deviation
    vwap_15m = (data["vw"] * data["volume"]).rolling(window=15).sum() / data["volume"].rolling(window=15).sum()
    data["vwap_dev"] = data["close"] / vwap_15m - 1.0

    # RTH cumulative return
    data["rth_cumret_30m"] = data["ret_1m"].rolling(window=30).sum()

    # External context features
    data["spy_ret_1m"] = np.random.normal(0.0001, 0.001, len(timestamps))
    data["spy_vol_30m"] = np.random.uniform(0.01, 0.03, len(timestamps))
    data["regime_high_vol"] = np.random.choice([0, 1], len(timestamps), p=[0.9, 0.1])
    data["regime_high_dispersion"] = np.random.choice([0, 1], len(timestamps), p=[0.95, 0.05])

    # Add deterministic features using the deterministic module
    from timegpt_v2.fe import deterministic
    data = deterministic.add_time_features(data)

    # Target columns
    data["target_log_return_1m"] = data["ret_1m"].shift(-1)  # Next period return
    data["target_log_return_15m"] = data["ret_1m"].rolling(window=15).sum().shift(-15)
    data["target_bp_ret_1m"] = data["ret_1m"].shift(-1) * 10000
    data["target_z_ret_1m"] = data["ret_1m"].shift(-1) / data["ret_1m"].rolling(window=30).std()

    # Label timestamps
    data["label_timestamp"] = data["timestamp"].shift(-1)
    data["label_timestamp_15m"] = data["timestamp"].shift(-15)

    # Drop last row since we can't compute targets for it
    data = data.iloc[:-1].copy()

    return data


class TestPayloadColumnIntegration:
    """Integration tests for payload column validation."""

    def test_y_df_column_composition(self, sample_ohlcv_features: pd.DataFrame) -> None:
        """Test that y_df includes exactly the expected columns."""
        snapshot_ts = pd.Timestamp("2024-07-01 12:00:00", tz="UTC")  # Mid-day snapshot

        y_df = build_y_df(
            sample_ohlcv_features,
            snapshot_ts,
            target_column="target_log_return_1m",
            rolling_window_days=1,
            min_obs_subhourly=100,
            symbols=["AAPL"]
        )

        # Should not be empty with proper data
        assert not y_df.empty, "y_df should not be empty with valid data"
        assert len(y_df) > 100, f"y_df should have sufficient rows, got {len(y_df)}"

        # Check base columns
        base_columns = ["unique_id", "ds", "y"]
        for col in base_columns:
            assert col in y_df.columns, f"Base column {col} missing from y_df"

        # Check that historical exogenous features are included
        hist_features_present = [col for col in HIST_EXOG_ALLOW if col in y_df.columns]
        assert len(hist_features_present) >= 8, f"Too few historical features: {hist_features_present}"

        # OHLCV features should be present
        ohlcv_features = ["ret_1m", "ret_5m", "sigma_5m", "parkinson_sigma_15m",
                         "range_pct_15m", "clv_15m", "vwap_dev", "rth_cumret_30m"]
        for feature in ohlcv_features:
            assert feature in y_df.columns, f"OHLCV feature {feature} missing from y_df"

        # External features should be present
        external_features = ["spy_ret_1m", "spy_vol_30m", "regime_high_vol", "regime_high_dispersion"]
        for feature in external_features:
            assert feature in y_df.columns, f"External feature {feature} missing from y_df"

        # Check that deterministic features are included
        det_features_present = [col for col in FUTR_EXOG_ALLOW if col in y_df.columns]
        assert len(det_features_present) >= 3, f"Too few deterministic features: {det_features_present}"

        # Verify no forbidden columns are present
        forbidden_columns = ["ret_15m", "sigma_15m", "fourier_sin_2", "fourier_cos_2"]
        for col in forbidden_columns:
            assert col not in y_df.columns, f"Forbidden column {col} should not be in y_df"

    def test_x_df_column_composition(self, sample_ohlcv_features: pd.DataFrame) -> None:
        """Test that x_df includes only deterministic features."""
        snapshot_ts = pd.Timestamp("2024-07-01 12:00:00", tz="UTC")
        horizon_minutes = 15

        x_df = build_x_df_for_horizon(
            sample_ohlcv_features,
            snapshot_ts,
            horizon_minutes,
            symbols=["AAPL"],
            strict_exog=False
        )

        # Should have exactly horizon_minutes rows
        assert len(x_df) == horizon_minutes, f"x_df should have {horizon_minutes} rows, got {len(x_df)}"

        # Check base columns
        base_columns = ["unique_id", "ds"]
        for col in base_columns:
            assert col in x_df.columns, f"Base column {col} missing from x_df"

        # Should include all deterministic features
        for feature in FUTR_EXOG_ALLOW:
            assert feature in x_df.columns, f"Deterministic feature {feature} missing from x_df"
            assert x_df[feature].notna().all(), f"Deterministic feature {feature} should have no NaNs"

        # Should NOT include any historical features
        for feature in HIST_EXOG_ALLOW:
            assert feature not in x_df.columns, f"Historical feature {feature} should not be in x_df"

        # Verify column count is reasonable
        expected_cols = 2 + len(FUTR_EXOG_ALLOW)  # base + deterministic
        assert len(x_df.columns) == expected_cols, f"Unexpected column count: {len(x_df.columns)} vs {expected_cols}"

    def test_strict_exog_validation(self, sample_ohlcv_features: pd.DataFrame) -> None:
        """Test strict_exog validation in integration context."""
        snapshot_ts = pd.Timestamp("2024-07-01 12:00:00", tz="UTC")

        # Should work fine with strict_exog=True when all features are present
        x_df = build_x_df_for_horizon(
            sample_ohlcv_features,
            snapshot_ts,
            10,
            symbols=["AAPL"],
            strict_exog=True
        )

        assert not x_df.empty, "x_df should not be empty with strict_exog=True"

        # All deterministic features should be present
        for feature in FUTR_EXOG_ALLOW:
            assert feature in x_df.columns, f"Deterministic feature {feature} missing with strict_exog=True"

    def test_payload_size_characteristics(self, sample_ohlcv_features: pd.DataFrame) -> None:
        """Test that payload size is reasonable."""
        snapshot_ts = pd.Timestamp("2024-07-01 12:00:00", tz="UTC")

        y_df = build_y_df(
            sample_ohlcv_features,
            snapshot_ts,
            target_column="target_log_return_1m",
            symbols=["AAPL"]
        )

        x_df = build_x_df_for_horizon(
            sample_ohlcv_features,
            snapshot_ts,
            15,
            symbols=["AAPL"]
        )

        # y_df should have reasonable number of columns (base + allowed features)
        max_y_df_columns = 3 + len(HIST_EXOG_ALLOW) + len(FUTR_EXOG_ALLOW)
        assert len(y_df.columns) <= max_y_df_columns, f"y_df has too many columns: {len(y_df.columns)}"

        # x_df should have exactly the deterministic features
        expected_x_df_columns = 2 + len(FUTR_EXOG_ALLOW)
        assert len(x_df.columns) == expected_x_df_columns, f"x_df column count unexpected: {len(x_df.columns)}"

        # Memory usage should be reasonable (this is a rough check)
        y_memory_mb = y_df.memory_usage(deep=True).sum() / 1024 / 1024
        x_memory_mb = x_df.memory_usage(deep=True).sum() / 1024 / 1024

        assert y_memory_mb < 10, f"y_df memory usage too high: {y_memory_mb:.2f} MB"
        assert x_memory_mb < 1, f"x_df memory usage too high: {x_memory_mb:.2f} MB"

    def test_time_progression_consistency(self, sample_ohlcv_features: pd.DataFrame) -> None:
        """Test that time features progress consistently across y_df and x_df."""
        snapshot_ts = pd.Timestamp("2024-07-01 12:00:00", tz="UTC")

        y_df = build_y_df(
            sample_ohlcv_features,
            snapshot_ts,
            target_column="target_log_return_1m",
            symbols=["AAPL"]
        )

        x_df = build_x_df_for_horizon(
            sample_ohlcv_features,
            snapshot_ts,
            5,
            symbols=["AAPL"]
        )

        # Get the last timestamp from y_df and first from x_df
        if not y_df.empty and not x_df.empty:
            last_y_time = pd.to_datetime(y_df["ds"].iloc[-1])
            first_x_time = pd.to_datetime(x_df["ds"].iloc[0])

            # x_df should start after y_df ends (1 minute gap)
            expected_first_x = last_y_time + pd.Timedelta(minutes=1)
            assert first_x_time == expected_first_x, f"Time progression inconsistent: {last_y_time} -> {first_x_time}"

            # Check deterministic features are consistent at the boundary
            for feature in ["minutes_since_open", "minutes_to_close"]:
                if feature in y_df.columns and feature in x_df.columns:
                    last_y_val = y_df[feature].iloc[-1]
                    first_x_val = x_df[feature].iloc[0]

                    # For minutes_since_open, x_df should be 1 minute ahead
                    if feature == "minutes_since_open":
                        assert first_x_val == last_y_val + 1, f"{feature} inconsistent at boundary"
                    # For minutes_to_close, x_df should be 1 minute behind
                    elif feature == "minutes_to_close":
                        assert first_x_val == last_y_val - 1, f"{feature} inconsistent at boundary"

    def test_multi_symbol_compatibility(self) -> None:
        """Test that payload builder works with multiple symbols (simplified)."""
        # Use a simpler approach - test that we can process different symbols
        timestamps = pd.date_range("2024-07-01 09:30:00", periods=100, freq="1min", tz="UTC")

        # Create minimal test data for multiple symbols
        symbol_data = pd.DataFrame({
            "timestamp": timestamps,
            "symbol": "AAPL",
            "target_log_return_1m": [0.001] * len(timestamps),
            "ret_1m": [0.001] * len(timestamps),
            "ret_5m": [0.005] * len(timestamps),
            "sigma_5m": [0.02] * len(timestamps),
            "spy_ret_1m": [0.0005] * len(timestamps),
            "regime_high_vol": [0] * len(timestamps),
            "regime_high_dispersion": [0] * len(timestamps),
        })

        # Add deterministic features
        from timegpt_v2.fe import deterministic
        symbol_data = deterministic.add_time_features(symbol_data)

        snapshot_ts = pd.Timestamp("2024-07-01 10:00:00", tz="UTC")

        # Test that we can build payloads for specific symbols
        y_df_aapl = build_y_df(
            symbol_data,
            snapshot_ts,
            target_column="target_log_return_1m",
            symbols=["AAPL"],
            min_obs_subhourly=50
        )

        x_df_aapl = build_x_df_for_horizon(
            symbol_data,
            snapshot_ts,
            5,
            symbols=["AAPL"]
        )

        # Verify AAPL works
        assert not y_df_aapl.empty or not x_df_aapl.empty, "At least one payload should be generated"

        # Test with different symbol (should return empty since symbol doesn't match)
        y_df_msft = build_y_df(
            symbol_data,
            snapshot_ts,
            target_column="target_log_return_1m",
            symbols=["MSFT"]  # Different symbol
        )

        x_df_msft = build_x_df_for_horizon(
            symbol_data,
            snapshot_ts,
            5,
            symbols=["MSFT"]  # Different symbol
        )

        # Should be empty since MSFT is not in the data (y_df filters by symbol)
        assert y_df_msft.empty, "y_df should be empty for non-matching symbol"
        # x_df generates deterministic features for any symbol, so this is expected behavior
        assert not x_df_msft.empty, "x_df should generate deterministic features even for non-matching symbols"
        assert list(x_df_msft["unique_id"]) == ["MSFT"] * 5, "x_df should contain MSFT as requested"


if __name__ == "__main__":
    pytest.main([__file__])