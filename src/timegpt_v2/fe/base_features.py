"""Leakage-safe base feature engineering primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from timegpt_v2.fe import deterministic
from timegpt_v2.fe.context import FeatureContext, apply_context_features
from timegpt_v2.utils.col_schema import FEATURE_MATRIX_COLS, RAW_DATA_COLS

_LOG_EPS = 1e-12


@dataclass(frozen=True)
class FeaturePolicy:
    """Rolling windows and thresholds used across feature calculations."""

    ret_windows: tuple[int, ...] = (1, 5, 15, 30)
    realized_variance_windows: tuple[int, ...] = (5, 15, 30)
    atr_window: int = 5
    volatility_window: int = 30
    vwap_window: int = 30
    vwap_trend_window: int = 5
    volume_window: int = 5
    volume_median_days: int = 20
    volume_percentile_minutes: int = 390 * 20
    signed_volume_window: int = 5
    beta_window: int = 60
    liquidity_percentile_window: int = 390 * 5


_DEFAULT_POLICY = FeaturePolicy()


def build_feature_matrix(
    bars: pd.DataFrame,
    *,
    context: FeatureContext | None = None,
    policy: FeaturePolicy | None = None,
) -> pd.DataFrame:
    """Return a leakage-safe feature matrix with targets and metadata.

    Parameters
    ----------
    bars:
        Input intraday bars containing at minimum ``timestamp``, ``symbol``, ``open``,
        ``high``, ``low``, ``close``, and ``volume``.
    context:
        Optional feature context for market-wide and event dummies.
    policy:
        Rolling window configuration; defaults to :data:`_DEFAULT_POLICY`.
    """

    if bars.empty:
        return bars.copy()

    working = (
        bars.copy()
        .sort_values(["symbol", "timestamp"])  # deterministic order
        .reset_index(drop=True)
    )
    required = RAW_DATA_COLS
    missing = required - set(working.columns)
    if missing:
        raise ValueError(f"Missing required columns for feature generation: {sorted(missing)}")

    policy = policy or _DEFAULT_POLICY

    features = _compute_symbol_features(working, policy)
    features = deterministic.add_time_features(features)

    if context is None:
        context = FeatureContext(
            symbols=tuple(sorted(working["symbol"].unique())),
            market_data=bars,
        )
    elif context.market_data is None:
        context = replace(context, market_data=bars)
    features = apply_context_features(features, context)

    features["timestamp_et"] = features["timestamp"].dt.tz_convert("America/New_York")

    features = _drop_sparse_rows(features)

    # Final check of the schema
    missing_cols = FEATURE_MATRIX_COLS - set(features.columns)
    if missing_cols:
        raise KeyError(f"Missing columns in the feature matrix: {sorted(missing_cols)}")

    return features.reset_index(drop=True)


def _compute_symbol_features(frame: pd.DataFrame, policy: FeaturePolicy) -> pd.DataFrame:
    grouped = []
    for _, group in frame.groupby("symbol", sort=False):
        enriched = _compute_single_symbol_features(group, policy)
        grouped.append(enriched)
    combined = pd.concat(grouped, ignore_index=True)
    return combined.sort_values(["symbol", "timestamp"])  # ensure stable ordering


def _compute_single_symbol_features(group: pd.DataFrame, policy: FeaturePolicy) -> pd.DataFrame:
    local = group.copy()
    local["timestamp"] = pd.to_datetime(local["timestamp"], utc=True)
    local.set_index("timestamp", inplace=True)

    log_close = np.log(local["close"].clip(lower=_LOG_EPS))
    log_open = np.log(local["open"].clip(lower=_LOG_EPS))

    local["target_log_return_1m"] = log_close.shift(-1) - log_close
    local["target_log_return_15m"] = log_close.shift(-15) - log_close
    local["target_log_return_30m"] = log_close.shift(-30) - log_close  # 30-minute target
    local["target_log_return_60m"] = log_close.shift(-60) - log_close
    local["target_bp_ret_1m"] = local["target_log_return_1m"] * 10_000.0

    for window in policy.ret_windows:
        local[f"ret_{window}m"] = log_close.diff(window)

    minute_returns = log_close.diff()
    vol_ewm = minute_returns.pow(2).ewm(span=60, adjust=False, min_periods=1).mean().pow(0.5)
    local["vol_ewm_60m"] = vol_ewm
    vol_ewm_15 = minute_returns.pow(2).ewm(span=15, adjust=False, min_periods=1).mean().pow(0.5)
    local["vol_ewm_15m"] = vol_ewm_15

    # Extended volatility measures for longer position durations
    vol_ewm_30 = minute_returns.pow(2).ewm(span=30, adjust=False, min_periods=1).mean().pow(0.5)
    local["vol_ewm_30m"] = vol_ewm_30
    vol_ewm_90 = minute_returns.pow(2).ewm(span=90, adjust=False, min_periods=1).mean().pow(0.5)
    local["vol_ewm_90m"] = vol_ewm_90
    scale = vol_ewm.replace(0.0, np.nan)
    scale_safe = scale.ffill().bfill().fillna(1.0)
    local["target_z_ret_1m"] = local["target_log_return_1m"] / (scale_safe + _LOG_EPS)
    for window in policy.realized_variance_windows:
        local[f"rv_{window}m"] = minute_returns.pow(2).rolling(window=window, min_periods=1).sum()
        local[f"sigma_{window}m"] = local[f"rv_{window}m"].clip(lower=0).pow(0.5)

    local["ret_skew_15m"] = minute_returns.rolling(window=15, min_periods=5).skew().fillna(0.0)
    local["ret_kurt_15m"] = minute_returns.rolling(window=15, min_periods=5).kurt().fillna(3.0)

    true_range = _true_range(local["high"], local["low"], log_close)
    local[f"atr_{policy.atr_window}m"] = true_range.rolling(
        window=policy.atr_window, min_periods=1
    ).mean()

    high_clip = local["high"].clip(lower=_LOG_EPS)
    low_clip = local["low"].clip(lower=_LOG_EPS)
    log_hl = np.log(high_clip) - np.log(low_clip)
    parkinson = log_hl.pow(2) / (4.0 * math.log(2.0))
    local["vol_parkinson_30m"] = (
        parkinson.rolling(window=policy.volatility_window, min_periods=5).mean().pow(0.5)
    )
    # Parkinson volatility for 15m window
    local["parkinson_sigma_15m"] = parkinson.rolling(window=15, min_periods=1).mean().pow(0.5)

    log_co = log_close - log_open
    gk = 0.5 * log_hl.pow(2) - (2.0 * math.log(2.0) - 1.0) * log_co.pow(2)
    local["vol_garman_klass_30m"] = (
        gk.rolling(window=policy.volatility_window, min_periods=5).mean().clip(lower=0).pow(0.5)
    )

    weighted_price = local["close"] * local["volume"].astype(float)
    rolling_vol = local["volume"].rolling(window=policy.vwap_window, min_periods=1).sum()
    rolling_weighted_price = weighted_price.rolling(window=policy.vwap_window, min_periods=1).sum()
    local["vwap_30m"] = np.where(
        rolling_vol == 0, local["close"], rolling_weighted_price / rolling_vol
    )
    rolling_std = local["close"].rolling(window=policy.vwap_window, min_periods=5).std()
    local["z_close_vwap_30m"] = (local["close"] - local["vwap_30m"]) / (rolling_std + _LOG_EPS)

    # VWAP deviation
    close_safe = local["close"].replace(0, np.nan)
    vwap_safe = local["vwap_30m"].replace(0, np.nan)
    local["vwap_dev"] = (close_safe / vwap_safe - 1.0).fillna(0.0)

    trend_window = max(policy.vwap_trend_window, 1)
    local[f"vwap_trend_{trend_window}m"] = local["vwap_30m"] - local["vwap_30m"].shift(trend_window)

    volume_5m = local["volume"].rolling(window=policy.volume_window, min_periods=1).sum()
    days_window = policy.volume_median_days * 390 // policy.volume_window
    median_volume = volume_5m.rolling(
        window=max(days_window, policy.volume_window), min_periods=5
    ).median()
    local["vol_5m_norm"] = volume_5m / (median_volume.replace(0, np.nan) + _LOG_EPS)

    vol_percentile_window = max(policy.volume_percentile_minutes, policy.volume_window)
    rolling_vol_max = (
        local["volume"]
        .rolling(window=vol_percentile_window, min_periods=1)
        .max()
        .replace(0, np.nan)
    )
    local["volume_percentile_20d"] = (
        (local["volume"] / (rolling_vol_max + _LOG_EPS)).clip(upper=1.0).fillna(0.0)
    )

    close_safe_range = local["close"].replace(0, np.nan)
    local["range_pct"] = ((local["high"] - local["low"]) / (close_safe_range + _LOG_EPS)).fillna(0.0)

    # 15-minute range percentage
    high_15m = local["high"].rolling(window=15, min_periods=1).max()
    low_15m = local["low"].rolling(window=15, min_periods=1).min()
    local["range_pct_15m"] = ((high_15m - low_15m) / (close_safe_range + _LOG_EPS)).fillna(0.0)

    # 15-minute Close Location Value (CLV)
    range_15m = high_15m - low_15m
    range_safe = range_15m.replace(0, np.nan)
    local["clv_15m"] = ((close_safe_range - low_15m) / range_safe).fillna(0.5)

    price_diff = local["close"].diff().fillna(0.0)
    direction = price_diff.apply(np.sign)
    signed_volume = direction * local["volume"].astype(float)
    local[f"signed_volume_{policy.signed_volume_window}m"] = signed_volume.rolling(
        window=policy.signed_volume_window, min_periods=1
    ).sum()

    # RTH cumulative return (last 30 minutes)
    # Check if we have ET time information available
    if "is_rth" in local.columns:
        # Find RTH open price for each day and compute cumulative return
        local_et = local.index.tz_convert("America/New_York")
        local["date_et"] = local_et.date
        local["time_et"] = local_et.time

        # Mark RTH hours (9:30 AM - 4:00 PM ET)
        rth_start = pd.Timestamp("09:30").time()
        rth_end = pd.Timestamp("16:00").time()
        local["is_rth_hour"] = ((local["time_et"] >= rth_start) & (local["time_et"] <= rth_end))

        # For each day, get the first RTH price as baseline
        local["rth_open_price"] = local.where(local["is_rth_hour"], np.nan)["close"].groupby(local["date_et"]).transform("first")

        # Compute cumulative return since RTH open (last 30 minutes only)
        minutes_since_rth_open = ((local_et - local_et.normalize()).total_seconds() / 60).astype(int)
        is_last_30m = minutes_since_rth_open >= 270  # 4.5 hours = 270 minutes after 9:30 AM

        local["rth_cumret_30m"] = np.where(
            is_last_30m & local["is_rth_hour"],
            (close_safe_range / local["rth_open_price"] - 1.0).fillna(0.0),
            0.0
        )
    else:
        # Fallback: simple 30-minute cumulative return
        local["rth_cumret_30m"] = log_close.diff(30).fillna(0.0)

    local.reset_index(inplace=True)
    local["label_timestamp"] = local["timestamp"].shift(-1)
    local["label_timestamp_15m"] = local["timestamp"].shift(-15)
    local["label_timestamp_30m"] = local["timestamp"].shift(-30)  # 30-minute label
    local["label_timestamp_60m"] = local["timestamp"].shift(-60)
    feature_cols = [
        "timestamp",
        "symbol",
        "target_log_return_1m",
        "target_log_return_15m",
        "target_log_return_30m",  # 30-minute target
        "target_log_return_60m",
        "target_bp_ret_1m",
        "target_z_ret_1m",
        "label_timestamp",
        "label_timestamp_15m",
        "label_timestamp_30m",  # 30-minute label
        "label_timestamp_60m",
    ]
    feature_cols.extend(
        col
        for col in local.columns
        if col not in feature_cols
    )

    # Drop rows with undefined target to avoid leakage downstream.
    valid = local["target_log_return_1m"].notna()
    result = local.loc[valid, feature_cols]
    return result


def _true_range(high: pd.Series, low: pd.Series, log_close: pd.Series) -> pd.Series:
    prev_close = np.exp(log_close.shift(1))
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return tr_components.max(axis=1)


def _drop_sparse_rows(features: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        col
        for col in features.columns
        if col
        not in {
            "timestamp",
            "symbol",
            "label_timestamp",
            "label_timestamp_15m",
            "target_log_return_1m",
            "target_log_return_15m",
            "target_bp_ret_1m",
            "target_z_ret_1m",
        }
    ]
    if not feature_columns:
        return features
    data = features.copy()
    row_nan_share = data[feature_columns].isna().mean(axis=1)
    filtered = data.loc[row_nan_share <= 0.1].copy()
    if filtered.empty:
        return filtered

    filtered.sort_values(["symbol", "timestamp"], inplace=True)
    grouped = filtered.groupby("symbol", sort=False)[feature_columns]
    filtered.loc[:, feature_columns] = grouped.ffill()

    remaining_na = filtered[feature_columns].isna().any(axis=1)
    if remaining_na.any():
        filtered = filtered.loc[~remaining_na].copy()
    return filtered


__all__ = ["FeaturePolicy", "build_feature_matrix"]
