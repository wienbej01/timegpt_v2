"""Payload construction utilities for TimeGPT requests."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import timedelta

import pandas as pd

from timegpt_v2.fe import deterministic


def build_y_df(
    features: pd.DataFrame,
    snapshot_ts: pd.Timestamp,
    *,
    target_column: str = "target_log_return_1m",
) -> pd.DataFrame:
    """Return the ``y`` frame for TimeGPT up to *snapshot_ts* (inclusive)."""

    if target_column not in features.columns:
        raise KeyError(f"Target column '{target_column}' missing from features")

    working = features.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    cutoff = pd.to_datetime(snapshot_ts, utc=True)

    filtered = working.loc[working["timestamp"] <= cutoff].copy()
    filtered = filtered.loc[filtered[target_column].notna()]
    if filtered.empty:
        return pd.DataFrame(columns=["unique_id", "ds", "y"])

    filtered.sort_values(["symbol", "timestamp"], inplace=True)
    renamed = filtered.rename(
        columns={"symbol": "unique_id", "timestamp": "ds", target_column: "y"}
    )
    return renamed[["unique_id", "ds", "y"]].reset_index(drop=True)


def build_x_df_for_horizon(
    features: pd.DataFrame,
    snapshot_ts: pd.Timestamp,
    horizon_minutes: int,
    *,
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Construct deterministic projections for minutes after *snapshot_ts*."""

    if horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be positive")

    base_symbols = list(symbols) if symbols is not None else []
    if not base_symbols:
        if "symbol" not in features.columns:
            raise KeyError("features must include a 'symbol' column when symbols not provided")
        base_symbols = sorted(str(sym) for sym in features["symbol"].unique())

    snapshot = pd.to_datetime(snapshot_ts, utc=True)
    future_index = pd.date_range(
        snapshot + timedelta(minutes=1),
        periods=horizon_minutes,
        freq="1min",
    )

    future_frames = []
    for symbol in base_symbols:
        template = pd.DataFrame({"timestamp": future_index, "symbol": symbol})
        det = deterministic.add_time_features(template)
        det["unique_id"] = symbol
        det.rename(columns={"timestamp": "ds"}, inplace=True)
        det["minute_ahead"] = ((det["ds"] - snapshot).dt.total_seconds() // 60).astype(int)
        future_frames.append(det)

    if not future_frames:
        return pd.DataFrame()

    combined = pd.concat(future_frames, ignore_index=True)
    ordered_cols = ["unique_id", "ds", "minute_ahead"] + [
        col for col in combined.columns if col not in {"unique_id", "ds", "minute_ahead"}
    ]
    combined.sort_values(["unique_id", "ds"], inplace=True)
    return combined[ordered_cols].reset_index(drop=True)


__all__ = ["build_y_df", "build_x_df_for_horizon"]
