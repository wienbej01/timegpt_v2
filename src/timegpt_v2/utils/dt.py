"""Datetime helpers for the TimeGPT intraday system."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from pandas import DatetimeIndex

ET_ZONE = ZoneInfo("America/New_York")
UTC_ZONE = ZoneInfo("UTC")


def to_utc(ts: datetime) -> datetime:
    """Convert a naive or timezone-aware timestamp to UTC."""
    localized = ts.replace(tzinfo=ET_ZONE) if ts.tzinfo is None else ts.astimezone(ET_ZONE)
    return localized.astimezone(UTC_ZONE)


def ensure_et_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame index is timezone-aware ET without mutating the input."""
    index = frame.index
    if not isinstance(index, DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")
    if index.tz is None:
        localized = frame.copy()
        localized.index = index.tz_localize(ET_ZONE)
        return localized
    converted = frame.copy()
    converted.index = index.tz_convert(ET_ZONE)
    return converted
