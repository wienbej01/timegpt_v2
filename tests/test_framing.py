from __future__ import annotations

import pandas as pd

from timegpt_v2.framing.build_payloads import build_x_df_for_horizon, build_y_df


def _sample_features() -> pd.DataFrame:
    base = pd.Timestamp("2024-01-02 14:15", tz="UTC")
    rows = []
    for minute in range(16):  # include 14:30 snapshot
        ts = base + pd.Timedelta(minutes=minute)
        for idx, symbol in enumerate(["AAPL", "MSFT"]):
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "target_log_return_1m": 0.001 * (minute + idx),
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
    y_df = build_y_df(features, snapshot)

    assert list(y_df.columns) == ["unique_id", "ds", "y"]
    for symbol in ["AAPL", "MSFT"]:
        symbol_slice = y_df[y_df["unique_id"] == symbol]
        assert not symbol_slice.empty
        assert symbol_slice["ds"].is_monotonic_increasing
        assert symbol_slice["ds"].iloc[-1] == snapshot


def test_build_x_df_future_minutes() -> None:
    features = _sample_features()
    snapshot = pd.Timestamp("2024-01-02 14:30", tz="UTC")
    horizon = 3

    x_df = build_x_df_for_horizon(features, snapshot, horizon)

    assert len(x_df) == horizon * 2
    for symbol in ["AAPL", "MSFT"]:
        symbol_slice = x_df[x_df["unique_id"] == symbol]
        assert list(symbol_slice["minute_ahead"]) == [1, 2, 3]
        assert symbol_slice["ds"].iloc[-1] == snapshot + pd.Timedelta(minutes=horizon)
        assert symbol_slice["ds"].is_monotonic_increasing
        for column in ["minute_index", "fourier_sin_1", "session_open"]:
            assert column in symbol_slice.columns
        for column in [
            "ret_1m",
            "ret_5m",
            "ret_15m",
            "ret_30m",
            "rv_5m",
            "rv_15m",
            "rv_30m",
            "volume_percentile_20d",
        ]:
            assert column in symbol_slice.columns
            assert symbol_slice[column].notna().all()
