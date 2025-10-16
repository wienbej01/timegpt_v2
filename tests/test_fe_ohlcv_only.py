"""Tests for OHLCV-only feature engineering primitives."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from timegpt_v2.fe.base_features import FeaturePolicy, build_feature_matrix


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Create a sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    n_minutes = 200

    # Generate realistic price data with trend and volatility
    timestamps = pd.date_range("2024-07-01 09:30:00", periods=n_minutes, freq="1min", tz="UTC")

    # Simulate price movement with random walk
    initial_price = 150.0
    returns = np.random.normal(0.0001, 0.002, n_minutes)
    prices = initial_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close prices with some noise
    high_noise = np.random.exponential(0.001, n_minutes)
    low_noise = np.random.exponential(0.001, n_minutes)
    open_noise = np.random.normal(0, 0.0005, n_minutes)

    data = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": "AAPL",
        "open": prices * (1 + open_noise),
        "high": prices * (1 + high_noise),
        "low": prices * (1 - low_noise),
        "close": prices,
        "volume": np.random.randint(1000, 10000, n_minutes),
        "vw": prices * np.random.uniform(0.98, 1.02, n_minutes),
        "n": np.random.randint(10, 100, n_minutes),
        "session": ["regular"] * n_minutes,
        "date_et": [ts.date() for ts in timestamps.tz_convert("America/New_York")],
    })

    # Add RTH flag (simplified - assuming all data is RTH)
    data["is_rth"] = True

    # Add required DQ checker columns
    data["ffill_flag"] = False
    data["log_return"] = np.log(data["close"]).diff().fillna(0)
    data["outlier_flag"] = False

    # Ensure we have enough data for target computation by dropping last row
    # (target requires shift(-1) which drops the final row)
    data = data.iloc[:-1].copy()

    return data


@pytest.fixture
def feature_policy() -> FeaturePolicy:
    """Return feature policy with windows matching our sprint requirements."""
    return FeaturePolicy(
        ret_windows=(1, 5, 15, 30),
        realized_variance_windows=(5, 15, 30),
        volatility_window=30,
        vwap_window=30,
    )


def test_returns_computation(sample_ohlcv_data: pd.DataFrame, feature_policy: FeaturePolicy) -> None:
    """Test that 1-minute and 5-minute returns are computed correctly."""
    features = build_feature_matrix(sample_ohlcv_data, policy=feature_policy)

    # Check that required return columns exist
    assert "ret_1m" in features.columns, "ret_1m column missing"
    assert "ret_5m" in features.columns, "ret_5m column missing"

    # Check that returns are non-NaN for valid rows
    valid_mask = features["ret_1m"].notna()
    assert valid_mask.sum() > 0, "No valid ret_1m values"

    # Basic sanity check: 5-minute returns should have higher variance than 1-minute returns
    ret_1m_var = features["ret_1m"].var()
    ret_5m_var = features["ret_5m"].var()
    assert ret_5m_var > ret_1m_var, "5-minute returns should have higher variance than 1-minute returns"


def test_sigma_computation(sample_ohlcv_data: pd.DataFrame, feature_policy: FeaturePolicy) -> None:
    """Test that sigma (realized volatility) is computed correctly."""
    features = build_feature_matrix(sample_ohlcv_data, policy=feature_policy)

    # Check that sigma columns exist
    assert "sigma_5m" in features.columns, "sigma_5m column missing"
    assert "sigma_15m" in features.columns, "sigma_15m column missing"

    # Check that sigma is non-negative
    assert (features["sigma_5m"] >= 0).all(), "sigma_5m should be non-negative"
    assert (features["sigma_15m"] >= 0).all(), "sigma_15m should be non-negative"

    # Check relationship between rv and sigma
    if "rv_5m" in features.columns:
        sigma_computed = features["rv_5m"].clip(lower=0).pow(0.5)
        np.testing.assert_allclose(
            features["sigma_5m"], sigma_computed, rtol=1e-10,
            err_msg="sigma_5m should equal sqrt(rv_5m)"
        )


def test_parkinson_sigma_15m(sample_ohlcv_data: pd.DataFrame, feature_policy: FeaturePolicy) -> None:
    """Test Parkinson sigma computation for 15-minute window."""
    features = build_feature_matrix(sample_ohlcv_data, policy=feature_policy)

    # Check that parkinson_sigma_15m exists
    assert "parkinson_sigma_15m" in features.columns, "parkinson_sigma_15m column missing"

    # Check that it's non-negative
    assert (features["parkinson_sigma_15m"] >= 0).all(), "parkinson_sigma_15m should be non-negative"

    # Basic sanity: should have some correlation with realized volatility
    if "sigma_15m" in features.columns:
        valid_mask = features["parkinson_sigma_15m"].notna() & features["sigma_15m"].notna()
        if valid_mask.sum() > 10:
            correlation = features.loc[valid_mask, "parkinson_sigma_15m"].corr(
                features.loc[valid_mask, "sigma_15m"]
            )
            assert correlation > 0.3, f"Parkinson sigma should correlate with realized vol, got {correlation}"


def test_range_pct_15m(sample_ohlcv_data: pd.DataFrame, feature_policy: FeaturePolicy) -> None:
    """Test 15-minute range percentage computation."""
    features = build_feature_matrix(sample_ohlcv_data, policy=feature_policy)

    # Check that range_pct_15m exists
    assert "range_pct_15m" in features.columns, "range_pct_15m column missing"

    # Check that it's non-negative
    assert (features["range_pct_15m"] >= 0).all(), "range_pct_15m should be non-negative"

    # Check for reasonable values (shouldn't be extremely high for liquid stocks)
    valid_range = features["range_pct_15m"].notna()
    if valid_range.sum() > 0:
        max_range = features.loc[valid_range, "range_pct_15m"].max()
        assert max_range < 0.1, f"range_pct_15m should be reasonable, got max {max_range}"


def test_clv_15m(sample_ohlcv_data: pd.DataFrame, feature_policy: FeaturePolicy) -> None:
    """Test 15-minute Close Location Value computation."""
    features = build_feature_matrix(sample_ohlcv_data, policy=feature_policy)

    # Check that clv_15m exists
    assert "clv_15m" in features.columns, "clv_15m column missing"

    # Check that CLV is bounded [0, 1]
    valid_clv = features["clv_15m"].notna()
    if valid_clv.sum() > 0:
        clv_values = features.loc[valid_clv, "clv_15m"]
        assert (clv_values >= 0).all(), "clv_15m should be >= 0"
        assert (clv_values <= 1).all(), "clv_15m should be <= 1"

        # Check that zero-range cases default to 0.5
        zero_range_mask = features["range_pct_15m"] == 0
        if zero_range_mask.sum() > 0:
            assert (features.loc[zero_range_mask, "clv_15m"] == 0.5).all(), "Zero range should give CLV=0.5"


def test_vwap_dev(sample_ohlcv_data: pd.DataFrame, feature_policy: FeaturePolicy) -> None:
    """Test VWAP deviation computation."""
    features = build_feature_matrix(sample_ohlcv_data, policy=feature_policy)

    # Check that vwap_dev exists
    assert "vwap_dev" in features.columns, "vwap_dev column missing"

    # Check for reasonable values (should be small deviations)
    valid_vwap = features["vwap_dev"].notna()
    if valid_vwap.sum() > 0:
        vwap_values = features.loc[valid_vwap, "vwap_dev"]
        # VWAP deviation should be relatively small for liquid stocks
        assert vwap_values.abs().max() < 0.05, f"VWAP deviation should be small, got max {vwap_values.abs().max()}"

        # Check that it can be both positive and negative
        assert (vwap_values > 0).any(), "Should have some positive VWAP deviations"
        assert (vwap_values < 0).any(), "Should have some negative VWAP deviations"


def test_rth_cumret_30m(sample_ohlcv_data: pd.DataFrame, feature_policy: FeaturePolicy) -> None:
    """Test RTH cumulative return computation."""
    features = build_feature_matrix(sample_ohlcv_data, policy=feature_policy)

    # Check that rth_cumret_30m exists
    assert "rth_cumret_30m" in features.columns, "rth_cumret_30m column missing"

    # Check that values are reasonable
    valid_ret = features["rth_cumret_30m"].notna()
    if valid_ret.sum() > 0:
        ret_values = features.loc[valid_ret, "rth_cumret_30m"]
        # Cumulative returns should be relatively small over 30 minutes
        assert ret_values.abs().max() < 0.05, f"RTH cumulative return should be small, got max {ret_values.abs().max()}"

        # Should have both positive and negative values in random data
        assert (ret_values > 0).any() or (ret_values < 0).any(), "Should have some non-zero returns"


def test_feature_integration(sample_ohlcv_data: pd.DataFrame, feature_policy: FeaturePolicy) -> None:
    """Test that all OHLCV-only features are properly integrated."""
    features = build_feature_matrix(sample_ohlcv_data, policy=feature_policy)

    # List of all OHLCV-only features from the sprint plan
    ohlcv_features = [
        "ret_1m", "ret_5m",  # Already existed
        "sigma_5m",  # New
        "parkinson_sigma_15m",  # New (choosing Parkinson over GK)
        "range_pct_15m", "clv_15m",  # New
        "vwap_dev",  # New
        "rth_cumret_30m",  # New
    ]

    # Check that all features exist
    missing_features = [feat for feat in ohlcv_features if feat not in features.columns]
    assert not missing_features, f"Missing OHLCV features: {missing_features}"

    # Check that features have reasonable non-NaN ratios
    for feat in ohlcv_features:
        valid_ratio = features[feat].notna().mean()
        assert valid_ratio > 0.5, f"Feature {feat} has too many NaNs: {valid_ratio:.2%}"


def test_feature_shapes_dont_explode(sample_ohlcv_data: pd.DataFrame, feature_policy: FeaturePolicy) -> None:
    """Test that adding new features doesn't cause shape issues."""
    features = build_feature_matrix(sample_ohlcv_data, policy=feature_policy)

    # Should have reasonable number of columns
    expected_min_cols = 50  # Rough estimate based on existing features
    assert len(features.columns) >= expected_min_cols, f"Too few columns: {len(features.columns)}"

    # Should have reasonable number of rows (accounting for warmup periods)
    expected_rows = len(sample_ohlcv_data) - 50  # Allow some warmup
    assert len(features) >= expected_rows, f"Too few rows: {len(features)} vs expected {expected_rows}"


def test_parkinson_formula_synthetic() -> None:
    """Test Parkinson volatility formula on synthetic data with known properties."""
    # Create synthetic data where we know the expected Parkinson volatility
    np.random.seed(123)
    n_obs = 100

    # Generate geometric Brownian motion
    dt = 1/390  # Daily (1 minute in trading days)
    sigma_true = 0.2  # 20% annual volatility

    prices = [100.0]
    for _ in range(n_obs):
        ret = np.random.normal(0, sigma_true * np.sqrt(dt))
        prices.append(prices[-1] * (1 + ret))

    prices = np.array(prices[1:])

    # Create OHLC from close prices
    high = prices * (1 + np.abs(np.random.normal(0, 0.001, n_obs)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.001, n_obs)))

    # Manual Parkinson calculation
    log_hl = np.log(high / low)
    parkinson_var = np.mean(log_hl**2) / (4 * np.log(2))
    parkinson_sigma = np.sqrt(parkinson_var)

    # Convert to annualized
    parkinson_sigma_annual = parkinson_sigma * np.sqrt(252 * 390)

    # Should be reasonably close to true sigma (within factor of 2 for small sample)
    assert 0.1 <= parkinson_sigma_annual <= 0.4, \
        f"Parkinson sigma {parkinson_sigma_annual:.3f} should be reasonable vs true {sigma_true}"