"""Tests for Sprint 4: Deterministic exog parity & X_df shape contract."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from timegpt_v2.framing.build_payloads import (
    build_payloads_with_validation,
    build_y_df,
    build_x_df_for_horizon,
    _validate_horizon_length,
    _validate_exog_parity
)
from timegpt_v2.fe.deterministic import get_deterministic_exog_names


class TestCanonicalExogList:
    """Test canonical deterministic exogenous feature list."""

    def test_deterministic_exog_names(self):
        """Test that deterministic exog names are well-defined."""
        det_exog_names = get_deterministic_exog_names()

        # Should be a list of strings
        assert isinstance(det_exog_names, list)
        assert all(isinstance(name, str) for name in det_exog_names)

        # Should have expected features
        expected_features = ["fourier_sin_1", "fourier_cos_1", "minutes_since_open", "minutes_to_close", "day_of_week"]
        for feature in expected_features:
            assert feature in det_exog_names

        # Should not have duplicates
        assert len(det_exog_names) == len(set(det_exog_names))

    def test_exog_names_immutability(self):
        """Test that exog names list is a copy, not reference."""
        names1 = get_deterministic_exog_names()
        names2 = get_deterministic_exog_names()

        # Modifying one shouldn't affect the other
        names1.append("test_feature")
        assert "test_feature" not in names2


class TestHorizonLengthValidation:
    """Test X_df horizon length validation."""

    def test_correct_horizon_length(self):
        """Test validation passes with correct horizon length."""
        horizon = 30
        symbols = ["AAPL", "MSFT"]

        x_df = pd.DataFrame({
            "unique_id": symbols * horizon,
            "ds": pd.date_range("2024-01-01 10:01", periods=horizon, freq="1min").repeat(len(symbols)),
            "fourier_sin_1": np.random.random(horizon * len(symbols)),
            "fourier_cos_1": np.random.random(horizon * len(symbols)),
            "minutes_since_open": np.arange(horizon).repeat(len(symbols)),
            "minutes_to_close": (horizon - 1 - np.arange(horizon)).repeat(len(symbols)),
            "day_of_week": [0] * (horizon * len(symbols))
        })

        # Should not raise exception
        _validate_horizon_length(x_df, horizon)

    def test_incorrect_horizon_length_fails(self):
        """Test validation fails with incorrect horizon length."""
        horizon = 60

        x_df = pd.DataFrame({
            "unique_id": ["AAPL"] * (horizon - 10),  # 10 rows too few
            "ds": pd.date_range("2024-01-01 10:01", periods=horizon - 10, freq="1min"),
            "fourier_sin_1": np.random.random(horizon - 10),
            "fourier_cos_1": np.random.random(horizon - 10),
            "minutes_since_open": np.arange(horizon - 10),
            "minutes_to_close": (horizon - 1 - np.arange(horizon - 10)),
            "day_of_week": [0] * (horizon - 10)
        })

        with pytest.raises(ValueError, match="X_df horizon length mismatch"):
            _validate_horizon_length(x_df, horizon)

    def test_empty_x_df_passes(self):
        """Test that empty X_df passes validation."""
        x_df = pd.DataFrame()
        _validate_horizon_length(x_df, 30)

    def test_multiple_symbols_different_lengths(self):
        """Test validation catches mixed-length symbols."""
        horizon = 30

        x_df = pd.DataFrame({
            "unique_id": ["AAPL"] * horizon + ["MSFT"] * (horizon - 5),
            "ds": list(pd.date_range("2024-01-01 10:01", periods=horizon, freq="1min")) +
                  list(pd.date_range("2024-01-01 10:01", periods=horizon - 5, freq="1min")),
            "fourier_sin_1": np.random.random(horizon + horizon - 5),
            "fourier_cos_1": np.random.random(horizon + horizon - 5),
            "minutes_since_open": np.arange(horizon + horizon - 5),
            "minutes_to_close": (horizon - 1 - np.arange(horizon + horizon - 5)),
            "day_of_week": [0] * (horizon + horizon - 5)
        })

        with pytest.raises(ValueError, match="X_df horizon length mismatch"):
            _validate_horizon_length(x_df, horizon)


class TestExogParityValidation:
    """Test exogenous feature parity validation."""

    def test_perfect_parity_passes(self):
        """Test validation passes with perfect exog parity."""
        det_exog_names = get_deterministic_exog_names()

        y_df = pd.DataFrame({
            "unique_id": ["AAPL"] * 10,
            "ds": pd.date_range("2024-01-01 09:30", periods=10, freq="1min"),
            "y": np.random.normal(0, 0.01, 10),
            **{col: np.random.random(10) for col in det_exog_names}
        })

        x_df = pd.DataFrame({
            "unique_id": ["AAPL"] * 5,
            "ds": pd.date_range("2024-01-01 10:01", periods=5, freq="1min"),
            **{col: np.random.random(5) for col in det_exog_names}
        })

        # Should not raise exception
        _validate_exog_parity(y_df, x_df)

    def test_forbidden_suffixes_fail(self):
        """Test validation fails with forbidden suffixes."""
        det_exog_names = get_deterministic_exog_names()

        y_df = pd.DataFrame({
            "unique_id": ["AAPL"] * 10,
            "ds": pd.date_range("2024-01-01 09:30", periods=10, freq="1min"),
            "y": np.random.normal(0, 0.01, 10),
            **{col: np.random.random(10) for col in det_exog_names}
        })

        # Add forbidden suffix
        x_df = pd.DataFrame({
            "unique_id": ["AAPL"] * 5,
            "ds": pd.date_range("2024-01-01 10:01", periods=5, freq="1min"),
            "fourier_sin_1_x": np.random.random(5),  # Forbidden _x suffix
            **{col: np.random.random(5) for col in det_exog_names if col != "fourier_sin_1"}
        })

        with pytest.raises(ValueError, match="Forbidden suffixes"):
            _validate_exog_parity(y_df, x_df)

    def test_name_mismatch_fails(self):
        """Test validation fails with deterministic exog name mismatch."""
        det_exog_names = get_deterministic_exog_names()

        y_df = pd.DataFrame({
            "unique_id": ["AAPL"] * 10,
            "ds": pd.date_range("2024-01-01 09:30", periods=10, freq="1min"),
            "y": np.random.normal(0, 0.01, 10),
            **{col: np.random.random(10) for col in det_exog_names}
        })

        # Missing one deterministic feature
        x_df = pd.DataFrame({
            "unique_id": ["AAPL"] * 5,
            "ds": pd.date_range("2024-01-01 10:01", periods=5, freq="1min"),
            **{col: np.random.random(5) for col in det_exog_names if col != "day_of_week"}
        })

        with pytest.raises(ValueError, match="Deterministic exog name mismatch"):
            _validate_exog_parity(y_df, x_df)

    def test_dtype_mismatch_fails(self):
        """Test validation fails with dtype mismatches."""
        det_exog_names = get_deterministic_exog_names()

        y_df = pd.DataFrame({
            "unique_id": ["AAPL"] * 10,
            "ds": pd.date_range("2024-01-01 09:30", periods=10, freq="1min"),
            "y": np.random.normal(0, 0.01, 10),
            **{col: np.random.random(10).astype(np.float32) for col in det_exog_names}
        })

        # Use different dtype for x_df
        x_df = pd.DataFrame({
            "unique_id": ["AAPL"] * 5,
            "ds": pd.date_range("2024-01-01 10:01", periods=5, freq="1min"),
            **{col: np.random.random(5).astype(np.int16) for col in det_exog_names}  # Different dtype
        })

        with pytest.raises(TypeError, match="Deterministic exog dtype mismatch"):
            _validate_exog_parity(y_df, x_df)


class TestIntegratedPayloadValidation:
    """Test integrated payload building with validation."""

    def test_build_payloads_with_validation_success(self):
        """Test successful payload building with validation."""
        # Create test features with sufficient history (need > min_obs_subhourly = 1008)
        # Create multiple days of data
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        all_timestamps = []
        for date in dates:
            # Create RTH hours (9:30-16:00 = 390 minutes)
            day_start = pd.Timestamp(f"{date.date()} 09:30", tz="America/New_York")
            day_timestamps = pd.date_range(day_start, periods=390, freq="1min")
            all_timestamps.extend(day_timestamps)

        features = pd.DataFrame({
            "symbol": ["AAPL"] * len(all_timestamps),
            "timestamp": all_timestamps,
            "target_log_return_1m": np.random.normal(0, 0.02, len(all_timestamps)),
            "ret_1m": np.random.normal(0, 0.02, len(all_timestamps)),
            "sigma_5m": np.random.uniform(0.01, 0.03, len(all_timestamps)),
            "is_rth": True
        })

        y_df, x_df = build_payloads_with_validation(
            features=features,
            snapshot_ts=pd.Timestamp("2024-01-01 15:00", tz="America/New_York"),
            horizon_minutes=30,
            min_obs_subhourly=200,  # Lower threshold for test
            strict_exog=True
        )

        # Verify shapes and content
        assert not y_df.empty
        assert not x_df.empty

        # X_df should have exactly horizon rows per symbol
        length_check = x_df.groupby("unique_id").size()
        assert all(length_check == 30)

        # Both should have deterministic exogs
        det_exog_names = get_deterministic_exog_names()
        for col in det_exog_names:
            assert col in y_df.columns
            assert col in x_df.columns

        # No forbidden suffixes
        all_cols = list(y_df.columns) + list(x_df.columns)
        assert not any(col.endswith("_x") or col.endswith("_y") for col in all_cols)

    def test_build_payloads_strict_exog_fails(self):
        """Test payload building fails when strict exog validation fails."""
        # Create minimal features
        features = pd.DataFrame({
            "symbol": ["AAPL"] * 10,
            "timestamp": pd.date_range("2024-01-01 09:30", periods=10, freq="1min"),
            "target_log_return_1m": np.random.normal(0, 0.02, 10),
            "is_rth": True
        })

        # Should fail with strict_exog=True due to missing volatility features
        with pytest.raises(ValueError, match="strict_exog=True"):
            build_payloads_with_validation(
                features=features,
                snapshot_ts=pd.Timestamp("2024-01-01 09:40"),
                horizon_minutes=5,
                strict_exog=True
            )

    def test_build_payloads_non_strict_exog_succeeds(self):
        """Test payload building succeeds with non-strict exog validation."""
        # Create minimal features
        features = pd.DataFrame({
            "symbol": ["AAPL"] * 10,
            "timestamp": pd.date_range("2024-01-01 09:30", periods=10, freq="1min"),
            "target_log_return_1m": np.random.normal(0, 0.02, 10),
            "is_rth": True
        })

        # Should succeed with strict_exog=False
        y_df, x_df = build_payloads_with_validation(
            features=features,
            snapshot_ts=pd.Timestamp("2024-01-01 09:40"),
            horizon_minutes=5,
            strict_exog=False
        )

        assert not y_df.empty
        assert not x_df.empty


class TestDeterministicExogGeneration:
    """Test deterministic exogenous feature generation."""

    def test_deterministic_features_consistency(self):
        """Test that deterministic features are consistent between history and future."""
        # Test data
        timestamps = pd.date_range("2024-01-01 09:30", periods=50, freq="1min")
        features = pd.DataFrame({
            "symbol": ["AAPL"] * 50,
            "timestamp": timestamps,
            "target_log_return_1m": np.random.normal(0, 0.02, 50),
            "is_rth": True
        })

        snapshot_ts = pd.Timestamp("2024-01-01 10:00")
        horizon_minutes = 20

        y_df, x_df = build_payloads_with_validation(
            features=features,
            snapshot_ts=snapshot_ts,
            horizon_minutes=horizon_minutes
        )

        # Check that deterministic exogs have same dtypes
        det_exog_names = get_deterministic_exog_names()
        for col in det_exog_names:
            if col in y_df.columns and col in x_df.columns:
                assert y_df[col].dtype == x_df[col].dtype
                # Check reasonable ranges
                if "fourier" in col:
                    assert y_df[col].between(-1.0, 1.0).all()
                    assert x_df[col].between(-1.0, 1.0).all()
                elif "minutes" in col:
                    assert y_df[col].between(0, 390).all()
                    assert x_df[col].between(0, 390).all()
                elif col == "day_of_week":
                    assert y_df[col].between(0, 6).all()
                    assert x_df[col].between(0, 6).all()

    def test_multiple_symbols_deterministic_parity(self):
        """Test deterministic exog parity across multiple symbols."""
        timestamps = pd.date_range("2024-01-01 09:30", periods=100, freq="1min")
        features = pd.DataFrame({
            "symbol": ["AAPL"] * 50 + ["MSFT"] * 50,
            "timestamp": list(timestamps) * 2,
            "target_log_return_1m": np.random.normal(0, 0.02, 100),
            "is_rth": True
        })

        y_df, x_df = build_payloads_with_validation(
            features=features,
            snapshot_ts=pd.Timestamp("2024-01-01 10:00"),
            horizon_minutes=30
        )

        # Both symbols should have same deterministic features
        aapl_y = y_df[y_df["unique_id"] == "AAPL"]
        msft_y = y_df[y_df["unique_id"] == "MSFT"]
        aapl_x = x_df[x_df["unique_id"] == "AAPL"]
        msft_x = x_df[x_df["unique_id"] == "MSFT"]

        # Check column parity
        assert set(aapl_y.columns) == set(msft_y.columns)
        assert set(aapl_x.columns) == set(msft_x.columns)

        # Check deterministic feature parity
        det_exog_names = get_deterministic_exog_names()
        for col in det_exog_names:
            assert col in aapl_y.columns
            assert col in msft_y.columns
            assert col in aapl_x.columns
            assert col in msft_x.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])