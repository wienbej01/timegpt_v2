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
