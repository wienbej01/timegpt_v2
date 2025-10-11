"""Client for interacting with TimeGPT (project-local implementation)."""

from __future__ import annotations

import logging
import os
import time as _time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol

import numpy as np
import pandas as pd

from timegpt_v2.forecast.batcher import build_batches
from timegpt_v2.utils.api_budget import APIBudget
from timegpt_v2.utils.cache import CacheKey, ForecastCache
from timegpt_v2.utils.payload import estimate_bytes

_ET_TZ = "America/New_York"


class PayloadTooLargeError(Exception):
    """Custom exception for payload size errors."""


class TimeGPTBackend(Protocol):
    """Protocol describing the subset of SDK functionality used by the client."""

    def forecast(
        self,
        y: pd.DataFrame,
        x: pd.DataFrame | None,
        *,
        h: int,
        freq: str,
        quantiles: Sequence[float],
        hist_exog_list: Sequence[str] | None = None,
        num_partitions: int | None = None,
    ) -> pd.DataFrame: ...


@dataclass(frozen=True)
class TimeGPTConfig:
    """Configuration for TimeGPT forecast calls."""

    freq: str = "min"
    horizon: int = 15
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)
    batch_multi_series: bool = True
    model: str = "timegpt-1"
    levels: tuple[int, ...] = ()
    max_bytes_per_call: int = 150_000_000
    api_mode: str = "offline"
    num_partitions_default: int | None = None


@dataclass(frozen=True)
class TimeGPTRetryPolicy:
    """Retry policy for the remote TimeGPT backend."""

    max_attempts: int = 3
    backoff_seconds: float = 2.0


class _LocalDeterministicBackend:
    """Deterministic fallback backend used for tests and offline development."""

    def forecast(
        self,
        y: pd.DataFrame,
        x: pd.DataFrame | None,
        *,
        h: int,
        freq: str,
        quantiles: Sequence[float],
        hist_exog_list: Sequence[str] | None = None,
        num_partitions: int | None = None,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        grouped = y.groupby("unique_id")
        for unique_id, frame in grouped:
            history = frame["y"].astype(float).to_numpy()
            if history.size == 0:
                baseline = 0.0
            else:
                tail = history[-min(history.size, 30) :]
                baseline = float(np.nanmean(tail))
            spread = max(abs(baseline), 1e-4) * 0.05
            for q in quantiles:
                offset = (float(q) - 0.5) * 2.0
                value = baseline + offset * spread
                rows.append({"unique_id": unique_id, "quantile": float(q), "value": value})
        return pd.DataFrame(rows)


class NixtlaTimeGPTBackend:
    """Thin wrapper around Nixtla's TimeGPT SDK."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        retry: TimeGPTRetryPolicy | None = None,
        timeout: float = 30.0,
    ) -> None:
        # ... (existing implementation)
        pass

    def forecast(
        self,
        y: pd.DataFrame,
        x: pd.DataFrame | None,
        *,
        h: int,
        freq: str,
        quantiles: Sequence[float],
        hist_exog_list: Sequence[str] | None = None,
        num_partitions: int | None = None,
    ) -> pd.DataFrame:
        # ... (existing implementation)
        pass


class TimeGPTClient:
    """High-level client handling batching, caching, and logging semantics."""

    def __init__(
        self,
        *,
        backend: TimeGPTBackend | None = None,
        cache: ForecastCache | None = None,
        config: TimeGPTConfig | None = None,
        logger: logging.Logger | None = None,
        budget: APIBudget | None = None,
    ) -> None:
        self._backend = backend or _LocalDeterministicBackend()
        self._cache = cache
        self._config = config or TimeGPTConfig()
        self._logger = logger or logging.getLogger(__name__)
        self._budget = budget or APIBudget()

    def forecast(
        self,
        y_df: pd.DataFrame,
        x_df: pd.DataFrame | None,
        *,
        snapshot_ts: pd.Timestamp,
        horizon: int | None = None,
        freq: str | None = None,
        quantiles: Sequence[float] | None = None,
        hist_exog_list: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Forecast quantiles for the provided multi-series payload."""
        quantiles_tuple = tuple(float(q) for q in (quantiles or self._config.quantiles))
        horizon_minutes = int(horizon or self._config.horizon)
        freq_value = freq or self._config.freq

        if y_df.empty:
            return self._empty_frame(quantiles_tuple)

        snapshot_utc = pd.to_datetime(snapshot_ts, utc=True)
        forecast_ts = snapshot_utc + timedelta(minutes=horizon_minutes)

        unique_ids = list(dict.fromkeys(y_df["unique_id"].astype(str).tolist()))
        rows: list[dict[str, object]] = []
        missing_ids: list[str] = []
        cache_key_map: dict[str, CacheKey] = {}

        for unique_id in unique_ids:
            key = self._get_cache_key(unique_id, snapshot_utc, horizon_minutes, quantiles_tuple, hist_exog_list)
            cached = self._cache.get(key) if self._cache is not None else None
            if cached is not None:
                row = self._deserialize_cache(
                    unique_id,
                    cached,
                    forecast_ts,
                    snapshot_utc,
                    quantiles_tuple,
                )
                rows.append(row)
                self._logger.info(
                    "Forecasted %s h=%s with q=%s [cache]",
                    unique_id,
                    horizon_minutes,
                    list(quantiles_tuple),
                )
            else:
                missing_ids.append(unique_id)
                cache_key_map[unique_id] = key

        if missing_ids:
            if self._config.api_mode == "offline":
                self._logger.warning("In offline mode, but cache miss for %s ids", len(missing_ids))
                return self._empty_frame(quantiles_tuple)

            payload_y = y_df[y_df["unique_id"].isin(missing_ids)].copy()
            payload_x = x_df[x_df["unique_id"].isin(missing_ids)].copy() if x_df is not None else None

            for y_batch, x_batch, batch_meta in build_batches(
                payload_y, payload_x, "unique_id", self._config.max_bytes_per_call
            ):
                if not self._budget.can_call(len(batch_meta["unique_ids"])):
                    self._logger.error("API budget exceeded, aborting forecast.")
                    break

                forecast_result = self.call_timegpt(
                    y_batch,
                    x_batch,
                    h=horizon_minutes,
                    freq=freq_value,
                    quantiles=quantiles_tuple,
                    hist_exog_list=hist_exog_list,
                )
                pivot = forecast_result.pivot(index="unique_id", columns="quantile", values="value")
                for unique_id in batch_meta["unique_ids"]:
                    if unique_id not in pivot.index:
                        raise RuntimeError(f"Backend response missing quantiles for {unique_id}")
                    series = pivot.loc[unique_id]
                    values = {float(str(k)): float(v) for k, v in series.items()}
                    row = self._build_row(unique_id, forecast_ts, snapshot_utc, quantiles_tuple, values)
                    rows.append(row)
                    if self._cache is not None:
                        key = cache_key_map[unique_id]
                        cache_payload = {
                            "forecast_ts": forecast_ts.isoformat(),
                            "snapshot_utc": snapshot_utc.isoformat(),
                            "values": {str(q): float(values[q]) for q in quantiles_tuple},
                        }
                        self._cache.put(key, cache_payload)
                    self._logger.info(
                        "Forecasted %s h=%s with q=%s",
                        unique_id,
                        horizon_minutes,
                        list(quantiles_tuple),
                    )

        if not rows:
            return self._empty_frame(quantiles_tuple)

        ordered = sorted(rows, key=lambda item: (str(item["snapshot_utc"]), str(item["unique_id"])))
        return pd.DataFrame(ordered)

    def call_timegpt(
        self,
        y_batch: pd.DataFrame,
        x_batch: pd.DataFrame | None,
        *,
        h: int,
        freq: str,
        quantiles: Sequence[float],
        hist_exog_list: Sequence[str] | None = None,
        num_partitions: int | None = None,
    ) -> pd.DataFrame:
        """Call the TimeGPT backend with retry and partitioning logic."""
        try:
            if estimate_bytes(y_batch) + estimate_bytes(x_batch) > self._config.max_bytes_per_call:
                if num_partitions is None:
                    num_partitions = 2
                else:
                    num_partitions *= 2

            self._budget.record_call(1 if num_partitions is None else num_partitions)
            return self._backend.forecast(
                y_batch,
                x_batch,
                h=h,
                freq=freq,
                quantiles=quantiles,
                hist_exog_list=hist_exog_list,
                num_partitions=num_partitions,
            )
        except PayloadTooLargeError:
            if num_partitions is None:
                return self.call_timegpt(
                    y_batch, x_batch, h=h, freq=freq, quantiles=quantiles, hist_exog_list=hist_exog_list, num_partitions=2
                )
            elif num_partitions < 8:
                return self.call_timegpt(
                    y_batch, x_batch, h=h, freq=freq, quantiles=quantiles, hist_exog_list=hist_exog_list, num_partitions=num_partitions * 2
                )
            else:
                # Fallback to splitting the batch
                ids = y_batch["unique_id"].unique().tolist()
                mid = len(ids) // 2
                if mid == 0:
                    raise
                y1 = y_batch[y_batch["unique_id"].isin(ids[:mid])]
                x1 = x_batch[x_batch["unique_id"].isin(ids[:mid])] if x_batch is not None else None
                y2 = y_batch[y_batch["unique_id"].isin(ids[mid:])]
                x2 = x_batch[x_batch["unique_id"].isin(ids[mid:])] if x_batch is not None else None

                f1 = self.call_timegpt(y1, x1, h=h, freq=freq, quantiles=quantiles, hist_exog_list=hist_exog_list)
                f2 = self.call_timegpt(y2, x2, h=h, freq=freq, quantiles=quantiles, hist_exog_list=hist_exog_list)
                return pd.concat([f1, f2], ignore_index=True)

    def _get_cache_key(
        self,
        unique_id: str,
        snapshot_utc: pd.Timestamp,
        horizon: int,
        quantiles: tuple[float, ...],
        hist_exog_list: Sequence[str] | None,
    ) -> CacheKey:
        """Generate a cache key for a forecast."""
        return CacheKey(
            symbol=unique_id,
            trade_date=snapshot_utc.date().isoformat(),
            snapshot=snapshot_utc.strftime("%H:%M"),
            horizon=horizon,
            quantiles=quantiles,
            features_hash=hash(tuple(hist_exog_list or [])),
        )

    def _build_row(
        self,
        unique_id: str,
        forecast_ts: pd.Timestamp,
        snapshot_utc: pd.Timestamp,
        quantiles: tuple[float, ...],
        values: Mapping[float, float],
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "unique_id": unique_id,
            "forecast_ts": forecast_ts,
            "snapshot_utc": snapshot_utc,
        }
        for quantile in quantiles:
            column = f"q{int(round(quantile * 100))}"
            row[column] = values[quantile]
        return row

    def _deserialize_cache(
        self,
        unique_id: str,
        payload: Mapping[str, object],
        forecast_ts: pd.Timestamp,
        snapshot_utc: pd.Timestamp,
        quantiles: tuple[float, ...],
    ) -> dict[str, object]:
        values_raw = payload.get("values", {})
        if not isinstance(values_raw, Mapping):
            raise RuntimeError("Malformed cache payload")
        values: dict[float, float] = {}
        for quantile in quantiles:
            key = str(quantile)
            alt_key = f"{quantile:.2f}"
            stored = values_raw.get(key)
            if stored is None:
                stored = values_raw.get(alt_key)
            if stored is None:
                raise RuntimeError(f"Cache missing quantile {quantile}")
            values[quantile] = float(stored)
        return self._build_row(unique_id, forecast_ts, snapshot_utc, quantiles, values)

    @staticmethod
    def _quantile_columns(quantiles: Iterable[float]) -> list[str]:
        return [f"q{int(round(q * 100))}" for q in quantiles]

    def _empty_frame(self, quantiles: tuple[float, ...]) -> pd.DataFrame:
        columns = ["unique_id", "forecast_ts", "snapshot_utc", *self._quantile_columns(quantiles)]
        return pd.DataFrame(columns=columns)


__all__ = [
    "TimeGPTBackend",
    "TimeGPTClient",
    "TimeGPTConfig",
    "TimeGPTRetryPolicy",
    "NixtlaTimeGPTBackend",
    "PayloadTooLargeError",
]
