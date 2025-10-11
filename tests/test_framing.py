from __future__ import annotations

import pandas as pd

from timegpt_v2.framing.build_payloads import build_x_df_for_horizon, build_y_df


def _sample_features() -> pd.DataFrame:
    base = pd.Timestamp("2024-01-02 14:15", tz="UTC")
    total_minutes = 59040  # ensure coverage through +15 minute label horizon
    rows = []
    symbols = ["AAPL", "MSFT"]
    for minute in range(total_minutes):
        ts = base + pd.Timedelta(minutes=minute)
        for idx, symbol in enumerate(symbols):
            target_1m = 0.001 * (minute + idx)
            future_index = minute + 15
            if future_index < total_minutes:
                target_15m = 0.015 + 0.001 * (future_index + idx)
                label_15m = ts + pd.Timedelta(minutes=15)
            else:
                target_15m = float("nan")
                label_15m = pd.NaT
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "target_log_return_1m": target_1m,
                    "target_log_return_15m": target_15m,
                    "label_timestamp": ts + pd.Timedelta(minutes=1),
                    "label_timestamp_15m": label_15m,
                    "ret_1m": 0.001,
                    "ret_5m": 0.002,
                    "ret_15m": 0.003,
                    "ret_30m": 0.004,
                    "rv_5m": 0.0001,
                    "rv_15m": 0.0002,
                    "rv_30m": 0.0003,
                    "ret_skew_15m": 0.1,
                    "ret_kurt_15m": 3.0,
                    "vol_parkinson_30m": 0.02,
                    "vol_garman_klass_30m": 0.018,
                    "vwap_30m": 100.0 + idx,
                    "vwap_trend_5m": 0.5,
                    "vol_5m_norm": 1.2,
                    "volume_percentile_20d": 0.8,
                    "range_pct": 0.01,
                    "signed_volume_5m": 10.0,
                }
            )
    return pd.DataFrame(rows)


def test_build_y_df_shapes_and_monotonicity() -> None:
    features = _sample_features()
    snapshot = pd.Timestamp("2024-01-02 14:30", tz="UTC")
    y_df = build_y_df(features, snapshot, min_obs_subhourly=0)

    # Check that required columns are present
    required_cols = ["unique_id", "ds", "y"]
    for col in required_cols:
        assert col in y_df.columns

    for symbol in ["AAPL", "MSFT"]:
        symbol_slice = y_df[y_df["unique_id"] == symbol]
        assert not symbol_slice.empty
        assert symbol_slice["ds"].is_monotonic_increasing


def test_build_y_df_log_return_15m() -> None:
    features = _sample_features()
    snapshot = pd.Timestamp("2024-01-02 14:30", tz="UTC")
    y_df = build_y_df(features, snapshot, target_column="target_log_return_15m", min_obs_subhourly=0)

    assert not y_df.empty
    for symbol in ["AAPL", "MSFT"]:
        symbol_slice = y_df[y_df["unique_id"] == symbol]
        assert symbol_slice["y"].notna().all()


def test_build_x_df_future_minutes() -> None:
    features = _sample_features()
    snapshot = pd.Timestamp("2024-01-02 14:30", tz="UTC")
    horizon = 3

    x_df = build_x_df_for_horizon(features, snapshot, horizon)

    assert len(x_df) == horizon * 2
    for symbol in ["AAPL", "MSFT"]:
        symbol_slice = x_df[x_df["unique_id"] == symbol]
        assert symbol_slice["ds"].iloc[-1] == snapshot + pd.Timedelta(minutes=horizon)
        assert symbol_slice["ds"].is_monotonic_increasing


def test_build_y_df_fills_gaps() -> None:
    # Create features with gaps (skip some minutes)
    base = pd.Timestamp("2024-01-02 14:15", tz="UTC")
    rows = []
    # Only include minutes 0, 2, 4, ..., 20
    for minute in range(0, 21, 2):
        ts = base + pd.Timedelta(minutes=minute)
        rows.append(
            {
                "timestamp": ts,
                "symbol": "AAPL",
                "target_log_return_1m": 0.001 * minute,
                "ret_1m": 0.001,
            }
        )
    features = pd.DataFrame(rows)
    snapshot = pd.Timestamp("2024-01-02 14:35", tz="UTC")  # 20 minutes later
    y_df = build_y_df(features, snapshot, min_obs_subhourly=0)

    # Check that gaps are filled
    aapl_slice = y_df[y_df["unique_id"] == "AAPL"]
    expected_minutes = 21  # from 0 to 20 inclusive
    assert len(aapl_slice) == expected_minutes
    assert aapl_slice["ds"].is_monotonic_increasing
    # Check that ds are consecutive minutes
    ds_diff = aapl_slice["ds"].diff().dropna()
    assert (ds_diff == pd.Timedelta(minutes=1)).all()
    # Check that gaps are forward-filled (no NaNs in y for TimeGPT compatibility)
    assert aapl_slice["y"].isna().sum() == 0  # all gaps filled
    assert aapl_slice["y"].notna().sum() == expected_minutes  # all minutes have data
