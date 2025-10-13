"""Client for interacting with TimeGPT (project-local implementation)."""

from __future__ import annotations

import json
import logging
import os
import time as _time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from timegpt_v2.config.model import ForecastExogConfig
from timegpt_v2.forecast.batcher import build_batches
from timegpt_v2.forecast.exogenous import (
    estimate_payload_bytes,
    merge_future_exogs,
    merge_history_exogs,
    normalize_names,
    preflight_log,
    select_available,
)
from timegpt_v2.utils.api_budget import APIBudget
from timegpt_v2.utils.cache import CacheKey, ForecastCache
from timegpt_v2.utils.payload import estimate_bytes
from timegpt_v2.utils.col_schema import ALL_EXOG_COLS

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
    exog: ForecastExogConfig = field(default_factory=ForecastExogConfig)


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
        if not api_key:
            raise ValueError("TimeGPT API key not provided")
        try:
            from nixtla import NixtlaClient
        except ImportError as exc:
            raise ImportError("nixtla package not installed") from exc

        self._client = NixtlaClient(
            api_key=api_key,
            base_url=base_url,
            max_retries=retry.max_attempts if retry else 3,
            timeout=timeout,
        )
        self._retry_policy = retry or TimeGPTRetryPolicy()

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
        """Call TimeGPT API and transform response to expected format."""
        # Prepare exogenous features
        hist_exog_cols = list(hist_exog_list) if hist_exog_list else []
        X_df = None
        if x is not None and not x.empty:
            # X_df should include ds, unique_id, and exogenous columns
            X_df = x.copy()
            # Ensure ds column exists (assume same as y)
            if "ds" not in X_df.columns:
                # Merge ds from y
                X_df = X_df.merge(y[["unique_id", "ds"]], on="unique_id", how="left")

        try:
            # Call Nixtla API
            result = self._client.forecast(
                df=y,
                h=h,
                freq=freq,
                quantiles=list(quantiles),
                X_df=X_df,
                hist_exog_list=hist_exog_cols if hist_exog_cols else None,
                num_partitions=num_partitions,
            )
        except Exception as exc:
            error_msg = str(exc).lower()
            if "payload" in error_msg and "large" in error_msg:
                raise PayloadTooLargeError(f"Payload too large: {exc}") from exc
            # Re-raise other exceptions
            raise

        # Transform result to expected format: unique_id, quantile, value
        rows: list[dict[str, object]] = []
        for unique_id, group in result.groupby("unique_id"):
            row = group.iloc[-1]
            for q in quantiles:
                col_name = f"TimeGPT-q-{int(round(q * 100))}"
                if col_name in result.columns:
                    value = float(row[col_name])
                    rows.append({"unique_id": unique_id, "quantile": q, "value": value})

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
        features: pd.DataFrame,
        snapshot_ts: pd.Timestamp,
        run_id: str,
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

        unique_ids = list(dict.fromkeys(y_df["unique_id"].astype(str).tolist()))
        rows: list[dict[str, object]] = []
        missing_ids: list[str] = []
        cache_key_map: dict[str, CacheKey] = {}

        for unique_id in unique_ids:
            key = self._get_cache_key(unique_id, snapshot_utc, horizon_minutes, quantiles_tuple)
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
                    features=features,
                    run_id=run_id,
                    h=horizon_minutes,
                    freq=freq_value,
                    quantiles=quantiles_tuple,
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
        features: pd.DataFrame,
        run_id: str,
        h: int,
        freq: str,
        quantiles: Sequence[float],
        hist_exog_list: Sequence[str] | None = None,
        num_partitions: int | None = None,
    ) -> pd.DataFrame:
        """Call the TimeGPT backend with retry and partitioning logic."""
        if not self._config.exog.use_exogs:
            y_batch_exog = y_batch
            x_batch_exog = x_batch
        else:
            # Normalize names
            hist_exog_declared = normalize_names(
                self._config.exog.hist_exog_list_raw, self._config.exog.exog_name_map
            )
            futr_exog_declared = normalize_names(
                self._config.exog.futr_exog_list_raw, self._config.exog.exog_name_map
            )

            # Select available
            hist_exog_present, hist_exog_missing = select_available(
                hist_exog_declared, features.columns
            )
            futr_exog_present, futr_exog_missing = select_available(
                futr_exog_declared, features.columns
            )

            y_shape_before = y_batch.shape
            x_shape_before = x_batch.shape if x_batch is not None else (0, 0)

            # Merge history exogs
            y_batch_exog = merge_history_exogs(
                y_batch,
                features,
                hist_exog_declared,
                self._config.exog.strict_exog,
                self._config.exog.impute_strategy,
                self._logger,
            )

            # Build future frame
            if futr_exog_declared:
                x_batch_exog = merge_future_exogs(
                    x_df=x_batch,
                    features_df=features,
                    futr_exogs=futr_exog_declared,
                    strict=True,  # ensure event flags present; default False if missing
                )
            else:
                x_batch_exog = x_batch

            y_shape_after = y_batch_exog.shape
            x_shape_after = x_batch_exog.shape if x_batch_exog is not None else (0, 0)

            # Log preflight info
            preflight_log(
                self._logger,
                hist_exog_declared,
                hist_exog_present,
                hist_exog_missing,
                futr_exog_declared,
                futr_exog_present,
                futr_exog_missing,
                y_shape_before,
                y_shape_after,
                x_shape_before,
                x_shape_after,
            )

            # Create metadata dictionary
            meta_exogs = {
                "declared_hist": hist_exog_declared,
                "present_hist": hist_exog_present,
                "dropped_hist": hist_exog_missing,
                "declared_futr": futr_exog_declared,
                "present_futr": futr_exog_present,
                "dropped_futr": futr_exog_missing,
                "y_shape_before": list(y_shape_before),
                "y_shape_after": list(y_shape_after),
                "x_shape_before": list(x_shape_before),
                "x_shape_after": list(x_shape_after),
                "estimated_y_bytes": int(estimate_payload_bytes(y_batch_exog)),
                "estimated_x_bytes": int(estimate_payload_bytes(x_batch_exog)),
            }

            # Final check of the schema
            missing_y_cols = set(hist_exog_declared) - set(y_batch_exog.columns)
            if missing_y_cols:
                raise KeyError(f"Missing columns in the y_batch_exog: {sorted(missing_y_cols)}")

            if x_batch_exog is not None:
                missing_x_cols = set(futr_exog_declared) - set(x_batch_exog.columns)
                if missing_x_cols:
                    raise KeyError(f"Missing columns in the x_batch_exog: {sorted(missing_x_cols)}")

            # Write metadata to file
            run_dir = Path("artifacts") / "runs" / run_id
            meta_exogs_path = run_dir / "meta_exogs.json"
            run_dir.mkdir(parents=True, exist_ok=True)
            with meta_exogs_path.open("w", encoding="utf-8") as f:
                json.dump(meta_exogs, f, indent=2)

        hist_exog_list_final = hist_exog_present

        try:
            if estimate_payload_bytes(y_batch_exog) + estimate_payload_bytes(x_batch_exog) > self._config.max_bytes_per_call:
                if num_partitions is None:
                    num_partitions = 2
                else:
                    num_partitions *= 2

            self._budget.record_call(1 if num_partitions is None else num_partitions)
            return self._backend.forecast(
                y_batch_exog,
                x_batch_exog,
                h=h,
                freq=freq,
                quantiles=quantiles,
                hist_exog_list=hist_exog_list_final,
                num_partitions=num_partitions,
            )
        except PayloadTooLargeError:
            if num_partitions is None:
                return self.call_timegpt(
                    y_batch, x_batch, features=features, run_id=run_id, h=h, freq=freq, quantiles=quantiles, hist_exog_list=hist_exog_list, num_partitions=2
                )
            elif num_partitions < 2:
                return self.call_timegpt(
                    y_batch, x_batch, features=features, run_id=run_id, h=h, freq=freq, quantiles=quantiles, hist_exog_list=hist_exog_list, num_partitions=num_partitions * 2
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

                f1 = self.call_timegpt(y1, x1, features=features, run_id=run_id, h=h, freq=freq, quantiles=quantiles, hist_exog_list=hist_exog_list)
                f2 = self.call_timegpt(y2, x2, features=features, run_id=run_id, h=h, freq=freq, quantiles=quantiles, hist_exog_list=hist_exog_list)
                return pd.concat([f1, f2], ignore_index=True)

    def _get_cache_key(
        self,
        unique_id: str,
        snapshot_utc: pd.Timestamp,
        horizon: int,
        quantiles: tuple[float, ...],
    ) -> CacheKey:
        """Generate a cache key for a forecast."""
        exog_features = tuple(sorted(self._config.exog.hist_exog_list) + sorted(self._config.exog.futr_exog_list))
        return CacheKey(
            symbol=unique_id,
            trade_date=snapshot_utc.date().isoformat(),
            snapshot=snapshot_utc.strftime("%H:%M"),
            horizon=horizon,
            quantiles=quantiles,
            features_hash=hash(exog_features),
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
