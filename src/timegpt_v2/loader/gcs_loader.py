from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, Iterable, List
from pathlib import Path

import pandas as pd


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
) -> pd.DataFrame:
    """Load multi-month history for a single symbol."""
    hist_start = start - timedelta(days=rolling_history_days)
    bucket = gcs_config.get("bucket", "")
    template = gcs_config.get("template", "")

    raw_uris = enumerate_month_uris(template, hist_start, end, symbol)
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
        (combined["timestamp"].dt.date >= hist_start) & (combined["timestamp"].dt.date <= end)
    ]

    return combined