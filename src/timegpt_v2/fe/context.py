"""Contextual features blending market regimes and event dummies."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FeatureContext:
    symbols: Iterable[str]
    market_symbol: str = "SPY"
    market_data: pd.DataFrame | None = None
    events: pd.DataFrame | None = None
    dispersion_threshold: float = 2.0
    volatility_threshold: float = 2.0
    dispersion_window: int = 120

    _event_types: tuple[str, ...] = field(default=("earnings", "fomc", "cpi"), init=False)


def apply_context_features(features: pd.DataFrame, context: FeatureContext) -> pd.DataFrame:
    enriched = features.copy()

    enriched = _apply_market_features(enriched, context)
    enriched = _apply_regime_flags(enriched, context)
    enriched = _apply_event_dummies(enriched, context)

    return enriched


def _apply_market_features(features: pd.DataFrame, context: FeatureContext) -> pd.DataFrame:
    if context.market_data is not None:
        market_df = context.market_data.copy()
        if "symbol" in market_df.columns:
            market_df = market_df.loc[market_df["symbol"] == context.market_symbol].copy()
    else:
        market_df = features.loc[features["symbol"] == context.market_symbol, :].copy()
        if {"close", "timestamp"}.issubset(market_df.columns):
            source = market_df
        else:
            base = features.merge(
                features[["timestamp", "symbol", "close"]],
                on=["timestamp", "symbol"],
                how="left",
                suffixes=(None, "_base"),
            )
            source = base.loc[base["symbol"] == context.market_symbol, ["timestamp", "close_base"]]
            source.rename(columns={"close_base": "close"}, inplace=True)
        market_df = source.copy()

    if market_df.empty or "close" not in market_df.columns:
        for col in ("spy_ret_1m", "spy_vol_30m"):
            features[col] = 0.0
        return features

    market = market_df.sort_values("timestamp")
    if "close" not in market.columns:
        raise ValueError("market data must include a 'close' column")

    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    market.set_index("timestamp", inplace=True)

    close_series = market["close"].clip(lower=1e-12)
    log_close = np.log(close_series)
    raw_ret = log_close.diff()
    lagged_ret = raw_ret.shift(1).fillna(0.0)
    lagged_vol = raw_ret.rolling(window=30, min_periods=5).std().shift(1).fillna(0.0)

    market["spy_ret_1m"] = lagged_ret
    market["spy_vol_30m"] = lagged_vol
    market.reset_index(inplace=True)

    broadcast = market[["timestamp", "spy_ret_1m", "spy_vol_30m"]]
    merged = features.merge(broadcast, on="timestamp", how="left")
    merged["spy_ret_1m"] = merged["spy_ret_1m"].fillna(0.0)
    merged["spy_vol_30m"] = merged["spy_vol_30m"].ffill()
    merged["spy_vol_30m"] = merged["spy_vol_30m"].fillna(0.0)
    return merged


def _apply_regime_flags(features: pd.DataFrame, context: FeatureContext) -> pd.DataFrame:
    data = features.copy()
    returns = data[["timestamp", "symbol", "ret_1m"]].dropna()
    if returns.empty:
        data["regime_high_vol"] = False
        data["regime_high_dispersion"] = False
        return data

    dispersion_series = (
        returns.groupby("timestamp")["ret_1m"]
        .apply(lambda x: float(np.nanstd(x)))
        .rename("cs_dispersion")
    )
    dispersion = dispersion_series.to_frame().reset_index()
    dispersion = dispersion.sort_values("timestamp")
    rolling_median = (
        dispersion["cs_dispersion"]
        .rolling(window=context.dispersion_window, min_periods=20)
        .median()
    )
    mad = (
        (dispersion["cs_dispersion"] - rolling_median)
        .abs()
        .rolling(window=context.dispersion_window, min_periods=20)
        .median()
    )
    z_dispersion = np.zeros(len(dispersion))
    valid = mad > 1e-8
    z_dispersion[valid] = 0.6745 * (
        (dispersion.loc[valid, "cs_dispersion"] - rolling_median.loc[valid]) / mad.loc[valid]
    )
    dispersion["dispersion_z"] = z_dispersion

    merged = data.merge(dispersion[["timestamp", "dispersion_z"]], on="timestamp", how="left")
    merged["dispersion_z"] = merged["dispersion_z"].fillna(0.0)
    merged["regime_high_dispersion"] = merged["dispersion_z"].abs() >= context.dispersion_threshold

    merged["regime_high_vol"] = merged["spy_vol_30m"].abs() >= context.volatility_threshold
    return merged


def _apply_event_dummies(features: pd.DataFrame, context: FeatureContext) -> pd.DataFrame:
    if context.events is None or context.events.empty:
        for event in context._event_types:
            features[f"event_{event}"] = False
        return features

    events = context.events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True).dt.floor("min")
    event_pivot = (
        events.assign(value=True)
        .pivot_table(index="timestamp", columns="event_type", values="value", aggfunc="any")
        .fillna(False)
    )
    event_pivot = event_pivot.astype(bool)
    event_pivot = event_pivot.reindex(columns=context._event_types, fill_value=False)
    event_pivot.columns = [f"event_{name}" for name in event_pivot.columns]
    event_pivot.reset_index(inplace=True)

    merged = features.merge(event_pivot, on="timestamp", how="left")
    for column in [col for col in merged.columns if col.startswith("event_")]:
        merged[column] = merged[column].fillna(False).astype(bool)
    return merged


__all__ = ["FeatureContext", "apply_context_features"]
