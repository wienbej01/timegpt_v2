"""Client for interacting with TimeGPT (project-local implementation)."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol

import numpy as np
import pandas as pd

from timegpt_v2.utils.cache import CacheKey, ForecastCache

_ET_TZ = "America/New_York"


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
    ) -> pd.DataFrame: ...


@dataclass(frozen=True)
class TimeGPTConfig:
    """Configuration for TimeGPT forecast calls."""

    freq: str = "min"
    horizon: int = 15
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)
    batch_multi_series: bool = True


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


class TimeGPTClient:
    """High-level client handling batching, caching, and logging semantics."""

    def __init__(
        self,
        *,
        backend: TimeGPTBackend | None = None,
        cache: ForecastCache | None = None,
        config: TimeGPTConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._backend = backend or _LocalDeterministicBackend()
        self._cache = cache
        self._config = config or TimeGPTConfig()
        self._logger = logger or logging.getLogger(__name__)

    def forecast(
        self,
        y_df: pd.DataFrame,
        x_df: pd.DataFrame | None,
        *,
        snapshot_ts: pd.Timestamp,
        horizon: int | None = None,
        freq: str | None = None,
        quantiles: Sequence[float] | None = None,
    ) -> pd.DataFrame:
        """Forecast quantiles for the provided multi-series payload."""

        quantiles_tuple = tuple(float(q) for q in (quantiles or self._config.quantiles))
        horizon_minutes = int(horizon or self._config.horizon)
        freq_value = freq or self._config.freq

        if y_df.empty:
            return self._empty_frame(quantiles_tuple)

        snapshot_utc = pd.to_datetime(snapshot_ts, utc=True)
        forecast_ts = snapshot_utc + timedelta(minutes=horizon_minutes)
        snapshot_et = snapshot_utc.tz_convert(_ET_TZ)
        trade_date = snapshot_et.date().isoformat()
        snapshot_label = snapshot_et.strftime("%H:%M")

        unique_ids = list(dict.fromkeys(y_df["unique_id"].astype(str).tolist()))
        rows: list[dict[str, object]] = []
        missing_ids: list[str] = []
        cache_keys: list[CacheKey] = []

        for unique_id in unique_ids:
            key = CacheKey(
                symbol=unique_id,
                trade_date=trade_date,
                snapshot=snapshot_label,
                horizon=horizon_minutes,
                quantiles=quantiles_tuple,
            )
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
                cache_keys.append(key)

        if missing_ids:
            payload_y = y_df[y_df["unique_id"].isin(missing_ids)].copy()
            payload_x = (
                x_df[x_df["unique_id"].isin(missing_ids)].copy() if x_df is not None else None
            )
            forecast_result = self._backend.forecast(
                payload_y,
                payload_x,
                h=horizon_minutes,
                freq=freq_value,
                quantiles=quantiles_tuple,
            )
            pivot = forecast_result.pivot(index="unique_id", columns="quantile", values="value")
            for unique_id, key in zip(missing_ids, cache_keys, strict=True):
                if unique_id not in pivot.index:
                    raise RuntimeError(f"Backend response missing quantiles for {unique_id}")
                series = pivot.loc[unique_id]
                values = {
                    float(str(index_key)): float(series[index_key]) for index_key in series.index
                }
                row = self._build_row(unique_id, forecast_ts, snapshot_utc, quantiles_tuple, values)
                rows.append(row)
                if self._cache is not None:
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


__all__ = ["TimeGPTClient", "TimeGPTConfig", "TimeGPTBackend"]
