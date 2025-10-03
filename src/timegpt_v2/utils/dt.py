"""Datetime helpers."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytz


ET_ZONE = pytz.timezone("America/New_York")


def to_utc(ts: datetime) -> datetime:
    """Convert naive or ET timestamp to UTC."""
    localized = ET_ZONE.localize(ts) if ts.tzinfo is None else ts.astimezone(ET_ZONE)
    return localized.astimezone(pytz.UTC)


def ensure_et_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame index is timezone-aware ET."""
    if frame.index.tz is None:
        frame = frame.tz_localize(ET_ZONE)
    else:
        frame = frame.tz_convert(ET_ZONE)
    return frame
