"""GCS reader utilities for intraday bar data."""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from timegpt_v2.loader.gcs_loader import load_history
from timegpt_v2.config.model import TradingWindowConfig

try:  # pragma: no cover - optional import for runtime
    import gcsfs  # type: ignore
except ImportError:  # pragma: no cover - handled in __post_init__
    gcsfs = None  # type: ignore

import fsspec

ET_ZONE = ZoneInfo("America/New_York")

_COLUMN_ALIASES = {
    "timestamp": {"ts", "timestamp", "time", "t"},  # Prioritize 'ts' (datetime) over 't' (numeric)
    "open": {"open", "o"},
    "high": {"high", "h"},
    "low": {"low", "l"},
    "close": {"close", "c"},
    "volume": {"volume", "v"},
    "symbol": {"symbol", "ticker"},
    "adj_open": {"adj_open", "open_adj"},
    "adj_high": {"adj_high", "high_adj"},
    "adj_low": {"adj_low", "low_adj"},
    "adj_close": {"adj_close", "close_adj", "adjusted_close"},
}

_RTH_START = time(9, 30)
_RTH_END = time(16, 0)


@dataclass
class ReaderConfig:
    bucket: str
    template: str
    skip_timestamp_normalization: bool = False


class GCSReader:
    """Read intraday bars stored as parquet files on GCS or local filesystem."""

    def __init__(self, config: ReaderConfig, fs: fsspec.AbstractFileSystem | None = None) -> None:
        self._config = config
        self._fs = fs or self._build_filesystem(config.bucket)
        self._base_path = self._derive_base_path(config.bucket)

    def read_universe(
        self,
        tickers: Iterable[str],
        start: date,
        end: date,
        rolling_history_days: int,
        trading_window: TradingWindowConfig | None = None,
    ) -> pd.DataFrame:
        """Read multiple tickers across date range into a single DataFrame."""
        logger = logging.getLogger(__name__)
        print("\n=== FORENSIC AUDIT: GCS READER - read_universe ===")
        print(f"Reading tickers: {list(tickers)}")
        print(f"Date range: {start} to {end}")
        if trading_window:
            print(f"Trading window: {trading_window.start} to {trading_window.end}")
            print(f"History backfill days: {trading_window.history_backfill_days}")

        frames = []
        gcs_config = {"bucket": self._config.bucket, "template": self._config.template}
        for ticker in tickers:
            frame = load_history(
                symbol=ticker,
                start=start,
                end=end,
                rolling_history_days=rolling_history_days,
                gcs_config=gcs_config,
                logger=logger,
                trading_window=trading_window,
            )
            if not frame.empty:
                frame = self._normalise_dataframe(frame, ticker)
                print(f"\nTicker {ticker}: {len(frame)} rows")
                # Check for duplicates within this ticker
                dup_count = frame.duplicated(subset=["symbol", "timestamp"]).sum()
                if dup_count > 0:
                    print(f"  !!! WARNING: {dup_count} duplicates found in {ticker} data !!!")
                    dup_mask = frame.duplicated(subset=["symbol", "timestamp"], keep=False)
                    print("  First few duplicates:")
                    print(
                        frame[dup_mask]
                        .sort_values("timestamp")
                        .head(10)[["symbol", "timestamp", "open", "close"]]
                    )
                frames.append(frame)

        if not frames:
            return pd.DataFrame()

        print(f"\nCombining {len(frames)} ticker frames...")
        combined = pd.concat(frames, ignore_index=True)
        print(f"Combined frame: {len(combined)} rows")

        # Check for duplicates after concatenation
        dup_after_concat = combined.duplicated(subset=["symbol", "timestamp"]).sum()
        print(f"Duplicates after concatenation: {dup_after_concat}")

        result = combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        # Final check
        dup_final = result.duplicated(subset=["symbol", "timestamp"]).sum()
        print(f"Duplicates in final result: {dup_final}")

        if dup_final > 0:
            print("\n!!! DUPLICATES FOUND IN read_universe OUTPUT !!!")
            dup_mask = result.duplicated(subset=["symbol", "timestamp"], keep=False)
            print("First 20 duplicate rows:")
            print(
                result[dup_mask]
                .sort_values(["symbol", "timestamp"])
                .head(20)[["symbol", "timestamp", "open", "close"]]
            )

        return result

    def _join_path(self, relative: str) -> str:
        if self._base_path is None:
            return relative
        if isinstance(self._base_path, Path):
            return str(self._base_path / relative)
        return f"{self._base_path}/{relative}".replace("//", "/")

    @staticmethod
    def _build_filesystem(bucket: str) -> fsspec.AbstractFileSystem:
        path = Path(bucket)
        if path.exists():
            return fsspec.filesystem("file")
        if gcsfs is None:
            raise RuntimeError("gcsfs is required to access remote GCS buckets")
        return gcsfs.GCSFileSystem()

    @staticmethod
    def _derive_base_path(bucket: str) -> Path | str | None:
        path = Path(bucket)
        if path.exists():
            return path
        return bucket.strip("/") or None

    def _normalise_dataframe(self, frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if frame.empty:
            return frame

        # Handle timestamp column creation and drop source columns before renaming
        if "ts" in frame.columns and pd.api.types.is_datetime64_any_dtype(frame["ts"]):
            frame["timestamp"] = frame["ts"]
            # Drop both ts and t to avoid duplicate renaming
            cols_to_drop = ["ts"]
            if "t" in frame.columns:
                cols_to_drop.append("t")
            frame = frame.drop(columns=cols_to_drop)
        elif "t" in frame.columns and pd.api.types.is_numeric_dtype(frame["t"]):
            frame["timestamp"] = pd.to_datetime(frame["t"], unit="ms", utc=True)
            frame = frame.drop(columns=["t"])

        renamed = _rename_columns(frame)
        ticker_str = str(ticker)
        if "symbol" in renamed.columns:
            renamed["symbol"] = renamed["symbol"].fillna(ticker_str).astype(str)
        else:
            renamed["symbol"] = ticker_str

        timestamps = renamed["timestamp"]
        if self._config.skip_timestamp_normalization and pd.api.types.is_datetime64_any_dtype(
            timestamps
        ):
            if hasattr(timestamps, "dt") and timestamps.dt.tz is not None:
                renamed["timestamp"] = timestamps.dt.tz_convert(ET_ZONE)
            else:
                renamed["timestamp"] = timestamps.dt.tz_localize(ET_ZONE)
        elif pd.api.types.is_numeric_dtype(timestamps):
            converted = pd.to_datetime(timestamps, unit="ms", utc=True)
            renamed["timestamp"] = converted.dt.tz_convert(ET_ZONE)
        elif pd.api.types.is_datetime64_any_dtype(timestamps):
            if hasattr(timestamps, "dt") and timestamps.dt.tz is not None:
                renamed["timestamp"] = timestamps.dt.tz_convert(ET_ZONE)
            else:
                renamed["timestamp"] = timestamps.dt.tz_localize(ET_ZONE)
        else:
            try:
                converted = pd.to_datetime(timestamps, utc=False, errors="raise")
                if converted.dt.tz is None:
                    converted = converted.dt.tz_localize(ET_ZONE)
                else:
                    converted = converted.dt.tz_convert(ET_ZONE)
                renamed["timestamp"] = converted
            except ValueError as e:
                if "duplicate keys" in str(e) and isinstance(timestamps, pd.DataFrame):
                    first_col = timestamps.iloc[:, 0]
                    converted = pd.to_datetime(first_col, utc=False, errors="raise")
                    if converted.dt.tz is None:
                        converted = converted.dt.tz_localize(ET_ZONE)
                    else:
                        converted = converted.dt.tz_convert(ET_ZONE)
                    renamed["timestamp"] = converted
                else:
                    raise

        renamed = _filter_rth(renamed)
        renamed = renamed.sort_values("timestamp").reset_index(drop=True)
        return renamed


def _rename_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}

    # Special handling for timestamp column to avoid conflicts
    # Prefer 'ts' (datetime) over 't' (numeric) when both exist
    timestamp_aliases = _COLUMN_ALIASES["timestamp"]
    timestamp_source = None

    # Check for datetime timestamp first (highest priority)
    if "ts" in frame.columns and pd.api.types.is_datetime64_any_dtype(frame["ts"]):
        timestamp_source = "ts"
    # Then check for numeric timestamp
    elif "t" in frame.columns:
        timestamp_source = "t"
    # Fall back to other aliases
    else:
        for alias in timestamp_aliases:
            if alias in frame.columns and alias not in {"ts", "t"}:
                timestamp_source = alias
                break

    if timestamp_source:
        rename_map[timestamp_source] = "timestamp"

    # Handle all other columns
    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical == "timestamp":
            continue  # Already handled above
        for alias in aliases:
            if alias in frame.columns:
                rename_map[alias] = canonical
                break  # Only map the first matching alias to avoid duplicates

    renamed = frame.rename(columns=rename_map).copy()
    return renamed


def _filter_rth(frame: pd.DataFrame) -> pd.DataFrame:
    timestamps = frame["timestamp"].dt.tz_convert(ET_ZONE)
    mask = (timestamps.dt.time >= _RTH_START) & (timestamps.dt.time < _RTH_END)
    return frame.loc[mask].copy()


def _iter_months(start: date, end: date) -> Iterable[tuple[int, int]]:
    if start > end:
        return []
    cursor = date(start.year, start.month, 1)
    end_marker = date(end.year, end.month, 1)
    months: list[tuple[int, int]] = []
    while cursor <= end_marker:
        months.append((cursor.year, cursor.month))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return months