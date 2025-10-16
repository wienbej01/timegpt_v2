"""Deterministic time-of-day and calendar features."""

from __future__ import annotations

import numpy as np
import pandas as pd

_RTH_MINUTES = 390

# Single source of truth for deterministic exogenous feature names
DETERMINISTIC_EXOG = [
    "fourier_sin_1",
    "fourier_cos_1",
    "minutes_since_open",
    "minutes_to_close",
    "day_of_week"
]


def get_deterministic_exog_names() -> list[str]:
    """Return a copy of the deterministic exogenous feature names.

    This is the single source of truth for deterministic exog names used
    throughout the pipeline to ensure parity between historical and future
    exogenous features.

    Returns:
        List of deterministic exogenous feature names.
    """
    return DETERMINISTIC_EXOG.copy()


_SESSION_BUCKETS = {
    "session_open": (0, 60),  # first hour
    "session_lunch": (150, 240),
    "session_power": (300, _RTH_MINUTES),
}


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Append deterministic seasonal features without mutating *frame*."""
    if frame.empty:
        return frame.copy()

    data = frame.copy()
    timestamps = pd.to_datetime(data["timestamp"], utc=True).dt.tz_convert("America/New_York")

    minute_of_day = _minute_of_day(timestamps)
    minute_progress = minute_of_day.astype(float) / float(_RTH_MINUTES - 1)

    data["minute_of_day"] = minute_of_day
    data["minute_index"] = minute_of_day
    data["minute_progress"] = minute_progress

    for harmonic in (1, 2, 3):
        angle = 2.0 * np.pi * float(harmonic) * minute_progress
        data[f"fourier_sin_{harmonic}"] = np.sin(angle)
        data[f"fourier_cos_{harmonic}"] = np.cos(angle)

    for bucket, (start, end) in _SESSION_BUCKETS.items():
        data[bucket] = ((minute_of_day >= start) & (minute_of_day < end)).astype(int)

    data["day_of_week"] = timestamps.dt.dayofweek
    data["is_month_end"] = timestamps.dt.is_month_end.astype(int)

    # Session clock features
    rth_open_hour = 9
    rth_open_minute = 30
    rth_close_hour = 16
    rth_close_minute = 0

    # Minutes since RTH open (9:30 AM ET)
    # Convert ET time to minutes since midnight
    et_minutes = timestamps.dt.hour.astype(int) * 60 + timestamps.dt.minute.astype(int)
    rth_open_minutes = rth_open_hour * 60 + rth_open_minute  # 9:30 = 570

    minutes_since_open = et_minutes - rth_open_minutes
    minutes_since_open = minutes_since_open.clip(lower=0, upper=_RTH_MINUTES - 1)
    data["minutes_since_open"] = minutes_since_open.astype(int)

    # Minutes to RTH close (4:00 PM ET)
    rth_close_minutes = rth_close_hour * 60 + rth_close_minute  # 16:00 = 960
    minutes_to_close = rth_close_minutes - et_minutes
    minutes_to_close = minutes_to_close.clip(lower=0, upper=_RTH_MINUTES - 1)
    data["minutes_to_close"] = minutes_to_close.astype(int)

    return data


def _minute_of_day(timestamps: pd.Series) -> np.ndarray:
    minutes = timestamps.dt.hour.astype(int) * 60 + timestamps.dt.minute.astype(int) - (9 * 60 + 30)
    minutes = minutes.clip(lower=0, upper=_RTH_MINUTES - 1)
    return minutes.to_numpy()


def build_det_exogs_for_history(
    df_bars: pd.DataFrame, *,
    tz: str,
    rth_open: str,
    rth_close: str
) -> pd.DataFrame:
    """Build deterministic exogenous features aligned to historical bar data.

    Computes deterministic exogenous features for historical data with exact
    same column names and dtypes as will be used for future data to ensure
    parity with the future exogenous features.

    Args:
        df_bars: Historical OHLCV bar data with 'timestamp' column
        tz: Timezone string (e.g., "America/New_York")
        rth_open: Regular trading hours open time (e.g., "09:30")
        rth_close: Regular trading hours close time (e.g., "16:00")

    Returns:
        DataFrame with deterministic exogenous features only, indexed same as df_bars.
        Columns match DETERMINISTIC_EXOG exactly with enforced dtypes:
        - day_of_week: int8
        - minutes_since_open, minutes_to_close: int32
        - fourier_sin_1, fourier_cos_1: float32
    """
    if df_bars.empty:
        return pd.DataFrame(index=df_bars.index, columns=get_deterministic_exog_names())

    # Convert timestamps to specified timezone
    timestamps = pd.to_datetime(df_bars["timestamp"], utc=True).dt.tz_convert(tz)

    # Extract components
    minute_of_day = _minute_of_day(timestamps)
    minute_progress = minute_of_day.astype(float) / float(_RTH_MINUTES - 1)

    # Create deterministic features with enforced dtypes
    result = pd.DataFrame(index=df_bars.index)

    # Fourier features (float32)
    result["fourier_sin_1"] = np.sin(2.0 * np.pi * minute_progress).astype(np.float32)
    result["fourier_cos_1"] = np.cos(2.0 * np.pi * minute_progress).astype(np.float32)

    # Time-based features (int32)
    result["minutes_since_open"] = minute_of_day.astype(np.int32)
    result["minutes_to_close"] = (_RTH_MINUTES - 1 - minute_of_day).astype(np.int32)

    # Day of week (int8)
    result["day_of_week"] = timestamps.dt.dayofweek.astype(np.int8)

    # Ensure column order matches DETERMINISTIC_EXOG
    result = result[get_deterministic_exog_names()]

    return result


def build_det_exogs_for_future(
    future_index: pd.DataFrame, *,
    tz: str,
    rth_open: str,
    rth_close: str
) -> pd.DataFrame:
    """Build deterministic exogenous features for future timestamps.

    Computes deterministic exogenous features for future timestamps with exact
    same column names and dtypes as historical data to ensure parity.

    Args:
        future_index: DataFrame with future timestamps (must have 'timestamp' column or be DatetimeIndex)
        tz: Timezone string (e.g., "America/New_York")
        rth_open: Regular trading hours open time (e.g., "09:30")
        rth_close: Regular trading hours close time (e.g., "16:00")

    Returns:
        DataFrame with deterministic exogenous features only, indexed same as future_index.
        Columns match DETERMINISTIC_EXOG exactly with enforced dtypes:
        - day_of_week: int8
        - minutes_since_open, minutes_to_close: int32
        - fourier_sin_1, fourier_cos_1: float32
    """
    if future_index.empty:
        return pd.DataFrame(index=future_index.index, columns=get_deterministic_exog_names())

    # Handle both DataFrame with timestamp column and DatetimeIndex
    if isinstance(future_index, pd.DataFrame):
        if 'timestamp' in future_index.columns:
            timestamps = pd.to_datetime(future_index["timestamp"], utc=True).dt.tz_convert(tz)
        else:
            timestamps = future_index.index.to_timestamp().tz_convert(tz) if hasattr(future_index.index, 'to_timestamp') else future_index.index
    else:
        timestamps = pd.to_datetime(future_index, utc=True).tz_convert(tz)

    # Extract components
    minute_of_day = _minute_of_day(timestamps)
    minute_progress = minute_of_day.astype(float) / float(_RTH_MINUTES - 1)

    # Create deterministic features with enforced dtypes
    result = pd.DataFrame(index=future_index.index)

    # Fourier features (float32)
    result["fourier_sin_1"] = np.sin(2.0 * np.pi * minute_progress).astype(np.float32)
    result["fourier_cos_1"] = np.cos(2.0 * np.pi * minute_progress).astype(np.float32)

    # Time-based features (int32)
    result["minutes_since_open"] = minute_of_day.astype(np.int32)
    result["minutes_to_close"] = (_RTH_MINUTES - 1 - minute_of_day).astype(np.int32)

    # Day of week (int8)
    result["day_of_week"] = timestamps.dt.dayofweek.astype(np.int8)

    # Ensure column order matches DETERMINISTIC_EXOG
    result = result[get_deterministic_exog_names()]

    return result


__all__ = [
    "add_time_features",
    "get_deterministic_exog_names",
    "build_det_exogs_for_history",
    "build_det_exogs_for_future"
]
