"""Payload construction utilities for TimeGPT requests."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import timedelta

import pandas as pd

from timegpt_v2.fe import deterministic

EXOGENOUS_FEATURE_COLUMNS = [
    "spy_ret_1m",
    "spy_vol_30m",
    "regime_high_vol",
    "regime_high_dispersion",
    "event_earnings",
    "event_fomc",
    "event_cpi",
]


def build_y_df(
    features: pd.DataFrame,
    snapshot_ts: pd.Timestamp,
    *,
    target_column: str = "target_log_return_1m",
    rolling_window_days: int = 40,
    min_obs_subhourly: int = 1008,
) -> pd.DataFrame:
    """Return the ``y`` frame for TimeGPT up to *snapshot_ts* (inclusive) with rolling window."""
    if target_column not in features.columns:
        raise KeyError(f"Target column '{target_column}' missing from features")

    working = features.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    cutoff = pd.to_datetime(snapshot_ts, utc=True)

    window_start = cutoff - pd.Timedelta(days=rolling_window_days)
    filtered = working.loc[working["timestamp"] <= cutoff].copy()
    filtered = filtered.loc[filtered["timestamp"] >= window_start].copy()
    filtered = filtered.loc[filtered[target_column].notna()]
    if filtered.empty:
        return pd.DataFrame(columns=["unique_id", "ds", "y"])

    # Enforce min_obs_subhourly for minute data
    if min_obs_subhourly > 0:
        counts = filtered.groupby("symbol").size()
        valid_symbols = counts[counts >= min_obs_subhourly].index
        filtered = filtered[filtered["symbol"].isin(valid_symbols)]

    if filtered.empty:
        return pd.DataFrame(columns=["unique_id", "ds", "y"])

    filtered.sort_values(["symbol", "timestamp"], inplace=True)
    renamed = filtered.rename(
        columns={"symbol": "unique_id", "timestamp": "ds", target_column: "y"}
    )

    # Fill gaps to make continuous time series for TimeGPT
    renamed = (
        renamed.set_index("ds")
        .groupby("unique_id")
        .resample("1min", include_groups=False)
        .asfreq()
        .reset_index()
    )
    # Forward fill y for non-trading hours to avoid NaN in target
    renamed["y"] = renamed.groupby("unique_id")["y"].ffill()

    # Minimal schema for the API
    return renamed[["unique_id", "ds", "y"]].reset_index(drop=True)


def build_x_df_for_horizon(
    features: pd.DataFrame,
    snapshot_ts: pd.Timestamp,
    horizon_minutes: int,
    *,
    symbols: Iterable[str] | None = None,
    hist_exog_list: list[str] | None = None,
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
        future_frames.append(det)

    if not future_frames:
        return pd.DataFrame()

    combined = pd.concat(future_frames, ignore_index=True)

    exog_cols_to_include = hist_exog_list or []
    if exog_cols_to_include:
        working = features.copy()
        working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
        snapshot_rows = (
            working.loc[
                (working["timestamp"] == snapshot) & working["symbol"].isin(base_symbols),
                ["symbol", *exog_cols_to_include],
            ]
            .drop_duplicates(subset="symbol", keep="last")
            .rename(columns={"symbol": "unique_id"})
        )
        combined = combined.merge(snapshot_rows, on="unique_id", how="left")

    ordered_cols = ["unique_id", "ds"] + exog_cols_to_include
    combined.sort_values(["unique_id", "ds"], inplace=True)
    return combined[ordered_cols].reset_index(drop=True)


__all__ = ["build_y_df", "build_x_df_for_horizon"]
