"""Payload construction utilities for TimeGPT requests."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import timedelta

import pandas as pd

from timegpt_v2.fe import deterministic

STATIC_FEATURE_COLUMNS = [
    "ret_1m",
    "ret_5m",
    "ret_15m",
    "ret_30m",
    "rv_5m",
    "rv_15m",
    "rv_30m",
    "ret_skew_15m",
    "ret_kurt_15m",
    "vol_parkinson_30m",
    "vol_garman_klass_30m",
    "vwap_30m",
    "vwap_trend_5m",
    "vol_5m_norm",
    "volume_percentile_20d",
    "range_pct",
    "signed_volume_5m",
]
DETERMINISTIC_FEATURE_COLUMNS = [
    "minute_of_day",
    "minute_index",
    "minute_progress",
    "fourier_sin_1",
    "fourier_cos_1",
    "fourier_sin_2",
    "fourier_cos_2",
    "fourier_sin_3",
    "fourier_cos_3",
    "session_open",
    "session_lunch",
    "session_power",
    "day_of_week",
    "is_month_end",
]
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
    rolling_window_days: int = 90,
) -> pd.DataFrame:
    """Return the ``y`` frame for TimeGPT up to *snapshot_ts* (inclusive) with rolling window.

    Args:
        features: Feature dataframe with timestamp, symbol, and target columns
        snapshot_ts: Snapshot timestamp for the forecast
        target_column: Name of the target column to use as y
        rolling_window_days: Number of days of historical data to include (default: 90)
    """

    if target_column not in features.columns:
        raise KeyError(f"Target column '{target_column}' missing from features")

    working = features.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    cutoff = pd.to_datetime(snapshot_ts, utc=True)

    # Apply rolling window to reduce payload size
    window_start = cutoff - pd.Timedelta(days=rolling_window_days)
    filtered = working.loc[working["timestamp"] <= cutoff].copy()
    filtered = filtered.loc[filtered["timestamp"] >= window_start].copy()
    filtered = filtered.loc[filtered[target_column].notna()]
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

    # Add deterministic features for exogenous consistency
    renamed = deterministic.add_time_features(renamed.rename(columns={"ds": "timestamp"}))
    renamed.rename(columns={"timestamp": "ds"}, inplace=True)

    # Add exogenous features present in X_df for API consistency
    renamed["symbol"] = renamed["unique_id"]

    renamed["minute_ahead"] = 0  # Historical data has no future minutes
    # Include static features in y_df for exogenous consistency
    static_columns = [col for col in STATIC_FEATURE_COLUMNS if col in renamed.columns]
    # Include deterministic features in y_df for exogenous consistency
    deterministic_columns = [col for col in DETERMINISTIC_FEATURE_COLUMNS if col in renamed.columns]
    # Include exogenous features
    exogenous_columns = [col for col in EXOGENOUS_FEATURE_COLUMNS if col in renamed.columns]
    # Include additional exogenous features
    additional_exog = ["minute_ahead"]
    columns = ["unique_id", "ds", "y"] + static_columns + deterministic_columns + exogenous_columns + additional_exog
    return renamed[columns].reset_index(drop=True)


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

    static_columns = [col for col in STATIC_FEATURE_COLUMNS if col in features.columns]
    exogenous_columns = [col for col in EXOGENOUS_FEATURE_COLUMNS if col in features.columns]
    merge_columns = static_columns + exogenous_columns
    if merge_columns:
        working = features.copy()
        working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
        snapshot_rows = (
            working.loc[
                (working["timestamp"] == snapshot) & working["symbol"].isin(base_symbols),
                ["symbol", *merge_columns],
            ]
            .drop_duplicates(subset="symbol", keep="last")
            .rename(columns={"symbol": "unique_id"})
        )
        combined = combined.merge(snapshot_rows, on="unique_id", how="left")

    ordered_cols = ["unique_id", "ds", "minute_ahead"] + [
        col for col in combined.columns if col not in {"unique_id", "ds", "minute_ahead", "symbol"}
    ]
    combined.sort_values(["unique_id", "ds"], inplace=True)
    return combined[ordered_cols].reset_index(drop=True)


__all__ = ["build_y_df", "build_x_df_for_horizon"]
