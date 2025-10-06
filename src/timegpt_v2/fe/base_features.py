"""Leakage-safe base feature engineering primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from timegpt_v2.fe import deterministic
from timegpt_v2.fe.context import FeatureContext, apply_context_features

_LOG_EPS = 1e-12


@dataclass(frozen=True)
class FeaturePolicy:
    """Rolling windows and thresholds used across feature calculations."""

    ret_windows: tuple[int, ...] = (1, 5, 15)
    realized_variance_windows: tuple[int, ...] = (5, 15)
    atr_window: int = 5
    volatility_window: int = 30
    vwap_window: int = 30
    volume_window: int = 5
    volume_median_days: int = 20


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
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
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

    features = _drop_sparse_rows(features)
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

    for window in policy.ret_windows:
        local[f"ret_{window}m"] = log_close.diff(window)

    minute_returns = log_close.diff()
    for window in policy.realized_variance_windows:
        local[f"rv_{window}m"] = minute_returns.pow(2).rolling(window=window, min_periods=1).sum()

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

    volume_5m = local["volume"].rolling(window=policy.volume_window, min_periods=1).sum()
    days_window = policy.volume_median_days * 390 // policy.volume_window
    median_volume = volume_5m.rolling(
        window=max(days_window, policy.volume_window), min_periods=5
    ).median()
    local["vol_5m_norm"] = volume_5m / (median_volume.replace(0, np.nan) + _LOG_EPS)

    local.reset_index(inplace=True)
    local["label_timestamp"] = local["timestamp"].shift(-1)
    feature_cols = [
        "timestamp",
        "symbol",
        "target_log_return_1m",
        "label_timestamp",
    ]
    feature_cols.extend(
        col
        for col in local.columns
        if col not in feature_cols and col not in {"open", "high", "low", "close", "volume"}
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
            "target_log_return_1m",
        }
    ]
    if not feature_columns:
        return features
    data = features.copy()
    row_nan_share = data[feature_columns].isna().mean(axis=1)
    filtered = data.loc[row_nan_share <= 0.01].copy()
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
