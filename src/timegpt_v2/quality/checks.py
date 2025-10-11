"""Data quality check implementations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from timegpt_v2.quality.contracts import DEFAULT_CONTRACT, DataContract, DataQualityPolicy

ET_ZONE = ZoneInfo("America/New_York")
_EXPECTED_RTH_BARS = 390
_FFORWARD_COLUMNS = ["open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close"]


@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: str
    details: dict[str, Any]


@dataclass
class DataQualityReport:
    passed: bool
    checks: tuple[CheckResult, ...]
    policy: DataQualityPolicy
    rows_before: int
    rows_after: int
    dropped_days: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "policy": self.policy.__dict__,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "dropped_days": list(self.dropped_days),
            "checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "severity": check.severity,
                    "details": check.details,
                }
                for check in self.checks
            ],
        }


class DataQualityChecker:
    """Run a suite of deterministic data quality checks."""

    def __init__(
        self,
        contract: DataContract | None = None,
        policy: DataQualityPolicy | None = None,
    ) -> None:
        self._contract = contract or DEFAULT_CONTRACT
        self._policy = policy or DataQualityPolicy()

    @property
    def policy(self) -> DataQualityPolicy:
        return self._policy

    def validate(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
        working = frame.copy()
        print("\n=== FORENSIC AUDIT: BEFORE ANY CHECKS ===")
        print(working.info())
        print(working.head())

        # Check for duplicates BEFORE any processing
        dup_before = working.duplicated(subset=["symbol", "timestamp"]).sum()
        print(f"\nDuplicates BEFORE _prepare_columns: {dup_before}")
        if dup_before > 0:
            print("\n!!! DUPLICATES FOUND BEFORE ANY PROCESSING !!!")
            print("This means duplicates exist in the data passed from GCS reader.")
            # Show some examples of duplicates
            dup_mask = working.duplicated(subset=["symbol", "timestamp"], keep=False)
            dup_examples = working[dup_mask].sort_values(["symbol", "timestamp"]).head(20)
            print("\nFirst 20 duplicate rows:")
            print(dup_examples[["symbol", "timestamp", "open", "close"]])

            # Check which symbols have duplicates
            dup_by_symbol = working[dup_mask].groupby("symbol").size()
            print("\nDuplicates by symbol:")
            print(dup_by_symbol)

        rows_before = len(working)
        checks: list[CheckResult] = []
        all_passed = True

        schema_passed, schema_details = self._check_schema(working)
        checks.append(CheckResult("schema", schema_passed, "hard", schema_details))
        if self._policy.hard_fail_on_schema and not schema_passed:
            report = DataQualityReport(
                passed=False,
                checks=tuple(checks),
                policy=self._policy,
                rows_before=rows_before,
                rows_after=rows_before,
                dropped_days=(),
            )
            return frame.copy(), report
        if not schema_passed:
            all_passed = False

        self._prepare_columns(working)
        print("\n=== FORENSIC AUDIT: AFTER _prepare_columns ===")
        print(working.info())
        print(working.head())

        dup_after = working.duplicated(subset=["symbol", "timestamp"]).sum()
        print(f"\nDuplicates AFTER _prepare_columns: {dup_after}")
        if dup_after > dup_before:
            print(f"\n!!! _prepare_columns CREATED {dup_after - dup_before} NEW DUPLICATES !!!")
            # Show the newly created duplicates
            dup_mask = working.duplicated(subset=["symbol", "timestamp"], keep=False)
            dup_examples = working[dup_mask].sort_values(["symbol", "timestamp"]).head(20)
            print("\nFirst 20 duplicate rows after _prepare_columns:")
            print(dup_examples[["symbol", "timestamp", "open", "close"]])

        monotonic_passed, monotonic_details = self._check_monotonicity(working)
        checks.append(CheckResult("monotonic", monotonic_passed, "hard", monotonic_details))
        if not monotonic_passed:
            all_passed = False
            # Drop duplicates to allow downstream checks to run
            working.drop_duplicates(subset=["symbol", "timestamp"], keep="first", inplace=True)
            working.reset_index(drop=True, inplace=True)

        price_passed, price_details = self._check_price_sanity(working)
        checks.append(CheckResult("price_sanity", price_passed, "hard", price_details))
        if not price_passed:
            all_passed = False

        completeness_passed, completeness_details = self._check_rth_completeness(working)
        checks.append(
            CheckResult("rth_completeness", completeness_passed, "hard", completeness_details)
        )
        if not completeness_passed:
            all_passed = False

        adjusted_passed, adjusted_details = self._check_adjusted_prices(working)
        checks.append(CheckResult("adjusted_price", adjusted_passed, "hard", adjusted_details))
        if not adjusted_passed:
            all_passed = False

        working, gapless_details = self._build_gapless_grid(working)
        checks.append(
            CheckResult("gapless_grid", gapless_details.pop("passed"), "warn", gapless_details)
        )

        working, outlier_details = self._flag_outliers(working)
        checks.append(CheckResult("outliers", True, "warn", outlier_details))

        rows_after = len(working)
        dropped_days = tuple(gapless_details.get("dropped_days", []))
        report = DataQualityReport(
            passed=all_passed,
            checks=tuple(checks),
            policy=self._policy,
            rows_before=rows_before,
            rows_after=rows_after,
            dropped_days=dropped_days,
        )
        return working, report

    def _check_schema(self, frame: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
        required = set(self._contract.required_columns)
        if not self._policy.require_adjusted:
            required -= {"adj_open", "adj_high", "adj_low", "adj_close"}
        present = set(frame.columns)
        missing = sorted(required - present)
        return (not missing, {"missing": missing})

    def _prepare_columns(self, frame: pd.DataFrame) -> None:
        if "timestamp" in frame.columns:
            timestamps = frame["timestamp"]
            # Only process timestamps if they're not already properly timezone-aware
            if not pd.api.types.is_datetime64_any_dtype(timestamps):
                timestamps = pd.to_datetime(timestamps, utc=False, errors="coerce")
                frame["timestamp"] = (
                    timestamps.dt.tz_localize(ET_ZONE)
                    if timestamps.dt.tz is None
                    else timestamps.dt.tz_convert(ET_ZONE)
                )
            elif hasattr(timestamps, "dt") and timestamps.dt.tz is None:
                # Timezone-naive datetime, localize it
                frame["timestamp"] = timestamps.dt.tz_localize(ET_ZONE)
            elif hasattr(timestamps, "dt") and timestamps.dt.tz != ET_ZONE:
                # Already timezone-aware but wrong timezone, convert it
                frame["timestamp"] = timestamps.dt.tz_convert(ET_ZONE)
            # If already in ET_ZONE, leave it as-is
        if "symbol" in frame.columns:
            frame["symbol"] = frame["symbol"].astype(str)
        frame.sort_values(["symbol", "timestamp"], inplace=True)
        frame.reset_index(drop=True, inplace=True)

    def _check_monotonicity(self, frame: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
        duplicates = int(frame.duplicated(subset=["symbol", "timestamp"]).sum())
        non_monotonic: list[str] = []
        for symbol, group in frame.groupby("symbol"):
            if not group["timestamp"].is_monotonic_increasing:
                non_monotonic.append(str(symbol))
        passed = duplicates == 0 and not non_monotonic
        return passed, {"duplicates": duplicates, "non_monotonic_symbols": non_monotonic}

    def _check_price_sanity(self, frame: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
        if not {"open", "high", "low", "close", "volume"}.issubset(frame.columns):
            return False, {"reason": "missing price columns"}
        min_price = frame[["open", "close"]].min(axis=1)
        max_price = frame[["open", "close"]].max(axis=1)
        sanity_mask = (
            (frame["low"] <= min_price) & (frame["high"] >= max_price) & (frame["volume"] >= 0)
        )
        invalid_rows = int((~sanity_mask).sum())
        return invalid_rows == 0, {"invalid_rows": invalid_rows}

    def _check_rth_completeness(self, frame: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
        if frame.empty:
            return True, {"details": []}
        frame["session_date"] = frame["timestamp"].dt.tz_convert(ET_ZONE).dt.normalize()
        counts = (
            frame.groupby(["symbol", "session_date"])["timestamp"].count().reset_index(name="rows")
        )

        def get_expected_bars(session_date):
            if session_date.month == 11 and session_date.day == 29:
                return 210  # Half-day for Black Friday
            return _EXPECTED_RTH_BARS

        counts["expected_rows"] = counts["session_date"].apply(get_expected_bars)
        counts["ratio"] = counts["rows"] / counts["expected_rows"]
        failing = counts[counts["ratio"] < self._policy.rth_min_pct]
        details = counts.to_dict(orient="records")
        frame.drop(columns="session_date", inplace=True)
        return failing.empty, {"counts": details, "failing": failing.to_dict(orient="records")}

    def _check_adjusted_prices(self, frame: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
        if not self._policy.require_adjusted:
            return True, {"checked": False}
        required_adjusted = {"adj_open", "adj_high", "adj_low", "adj_close"}
        missing = sorted(required_adjusted - set(frame.columns))
        return not missing, {"missing": missing, "checked": True}

    def _build_gapless_grid(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        if frame.empty:
            return frame, {"passed": True, "dropped_days": [], "ffill_rows": 0}
        gapless_frames: list[pd.DataFrame] = []
        dropped_days: list[str] = []
        forward_columns = [col for col in _FFORWARD_COLUMNS if col in frame.columns]
        for symbol, group in frame.groupby("symbol", sort=False):
            group = group.copy().set_index("timestamp")
            index = pd.DatetimeIndex(group.index)
            normalized_index = index.normalize()
            for session_start, session in group.groupby(normalized_index):
                session_day = session_start.date()
                expected_index = _expected_session_index(session_day)
                reindexed = session.reindex(expected_index)
                ffilled = reindexed.copy()
                if forward_columns:
                    ffilled[forward_columns] = ffilled[forward_columns].ffill()
                ffilled["symbol"] = symbol
                ffill_flag = (
                    reindexed[forward_columns].isna().any(axis=1)
                    if forward_columns
                    else pd.Series(False, index=reindexed.index)
                )
                ffilled["ffill_flag"] = ffill_flag
                if self._policy.drop_days_with_sustained_ffill:
                    longest = _longest_true_run(ffill_flag.to_numpy())
                    if longest >= self._policy.sustained_ffill_run:
                        dropped_days.append(f"{str(symbol)}:{session_day.isoformat()}")
                        continue
                gapless_frames.append(ffilled.reset_index().rename(columns={"index": "timestamp"}))
        if not gapless_frames:
            return pd.DataFrame(columns=list(frame.columns) + ["ffill_flag"]), {
                "passed": False,
                "dropped_days": dropped_days,
                "ffill_rows": 0,
            }
        combined = pd.concat(gapless_frames, ignore_index=True)
        combined.sort_values(["symbol", "timestamp"], inplace=True)
        combined.reset_index(drop=True, inplace=True)
        combined["ffill_flag"] = combined["ffill_flag"].fillna(False)
        return combined, {
            "passed": True,
            "dropped_days": dropped_days,
            "ffill_rows": int(combined["ffill_flag"].sum()),
        }

    def _flag_outliers(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        if frame.empty:
            frame["outlier_flag"] = []
            return frame, {"flags": 0}
        frame = frame.copy()
        frame["log_return"] = frame.groupby("symbol")["close"].transform(
            lambda x: np.log(x / x.shift(1))
        )
        returns = frame["log_return"].dropna()
        if returns.empty:
            frame["outlier_flag"] = False
            return frame, {"flags": 0}
        median = returns.median()
        mad = np.median(np.abs(returns - median))
        if mad == 0:
            z_scores = np.zeros(len(frame))
        else:
            z_scores = np.abs(0.6745 * (frame["log_return"] - median) / mad)
        frame["outlier_flag"] = z_scores > 6
        flags = int(frame["outlier_flag"].sum())
        return frame, {"flags": flags}


def _expected_session_index(day: date) -> pd.DatetimeIndex:
    start = datetime.combine(day, time(9, 30), tzinfo=ET_ZONE)
    end_exclusive = datetime.combine(day, time(16, 0), tzinfo=ET_ZONE)
    return pd.date_range(start=start, end=end_exclusive - pd.Timedelta(minutes=1), freq="1min")


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for flag in values:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)
