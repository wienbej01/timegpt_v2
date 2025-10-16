from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional
from pathlib import Path

import pandas as pd

from timegpt_v2.config.model import TradingWindowConfig
from timegpt_v2.utils.trading_window import compute_load_ranges


def enumerate_month_uris(
    template: str, start_dt: date, end_dt: date, symbol: str
) -> Iterable[str]:
    """Yield one URI per month in [start_dt, end_dt] inclusive."""
    current_dt = start_dt
    while current_dt <= end_dt:
        yyyy = current_dt.year
        mm = current_dt.month
        yyyy_mm = f"{yyyy}-{mm:02d}"
        uri = template.format(ticker=symbol, yyyy=yyyy, yyyy_mm=yyyy_mm)
        yield uri
        # Move to the next month
        if current_dt.month == 12:
            current_dt = date(current_dt.year + 1, 1, 1)
        else:
            current_dt = date(current_dt.year, current_dt.month + 1, 1)


def load_history(
    symbol: str,
    start: date,
    end: date,
    rolling_history_days: int,
    gcs_config: Dict,
    logger: logging.Logger,
    trading_window: Optional[TradingWindowConfig] = None,
) -> pd.DataFrame:
    """
    Load multi-month history for a single symbol.

    Args:
        symbol: Symbol to load
        start: Historical start date (legacy interface for backward compatibility)
        end: Historical end date (legacy interface for backward compatibility)
        rolling_history_days: Days of history to load (legacy interface)
        gcs_config: GCS configuration
        logger: Logger instance
        trading_window: Optional trading window configuration for new behavior
    """
    # Use new trading window logic if available, otherwise fall back to legacy behavior
    if trading_window is not None:
        load_start, load_end, _, _ = compute_load_ranges(trading_window)
        hist_start = load_start
        hist_end = load_end
    else:
        # Legacy behavior: always extend back by rolling_history_days
        hist_start = start - timedelta(days=rolling_history_days)
        hist_end = end

    bucket = gcs_config.get("bucket", "")
    template = gcs_config.get("template", "")

    raw_uris = enumerate_month_uris(template, hist_start, hist_end, symbol)
    all_dfs: List[pd.DataFrame] = []

    is_local = Path(bucket).exists()

    for raw_uri in raw_uris:
        if is_local:
            uri = Path(bucket) / raw_uri
        else:
            uri = f"gs://{bucket}/{raw_uri}"
        try:
            df = pd.read_parquet(uri)
            logger.info(f"Loaded {len(df)} rows from {uri}")
            all_dfs.append(df)
        except FileNotFoundError:
            logger.warning(f"File not found, skipping: {uri}")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.drop_duplicates(inplace=True)

    if "ts" in combined.columns and pd.api.types.is_datetime64_any_dtype(combined["ts"]):
        combined = combined.rename(columns={"ts": "timestamp"})
    elif "t" in combined.columns and pd.api.types.is_numeric_dtype(combined["t"]):
        combined["timestamp"] = pd.to_datetime(combined["t"], unit="ms", utc=True)
        combined = combined.drop(columns=["t"])

    if "timestamp" not in combined.columns:
        return combined

    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined = combined[
        (combined["timestamp"].dt.date >= hist_start) & (combined["timestamp"].dt.date <= hist_end)
    ]

    return combined