"""Utilities for forecast coverage tracking and skip reason enumeration."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Final

from timegpt_v2.utils.col_schema import ALL_EXOG_COLS


class SkipReason(str, Enum):
    """Enumeration of reasons why forecast snapshots might be skipped."""

    # Gate reasons - expected operational skips
    GATE_MIN_OBS = "GATE_MIN_OBS"  # Not enough historical observations
    GATE_RTH = "GATE_RTH"          # Outside regular trading hours window
    GATE_WARMUP = "GATE_WARMUP"    # Warmup period not satisfied

    # Trading window enforcement
    SKIP_BEFORE_TRADE_WINDOW = "SKIP_BEFORE_TRADE_WINDOW"  # Before configured trading window start
    SKIP_AFTER_TRADE_WINDOW = "SKIP_AFTER_TRADE_WINDOW"    # After configured trading window end

    # Configuration limits - potentially problematic caps
    BUDGET = "BUDGET"                          # API budget exceeded
    MAX_SNAPSHOTS_TOTAL = "MAX_SNAPSHOTS_TOTAL"  # Global snapshot cap reached
    MAX_SNAPSHOTS_DAY = "MAX_SNAPSHOTS_DAY"      # Daily snapshot cap reached

    # Data and filtering issues - bugs to investigate
    FILTER_DATE = "FILTER_DATE"        # Date window filtering (potential bug)
    FILTER_PHASE = "FILTER_PHASE"      # Phase filtering in scheduler (misuse)
    CACHE_SKIP = "CACHE_SKIP"          # Cache indicated run complete (potential bug)
    PAYLOAD_EMPTY = "PAYLOAD_EMPTY"    # No target rows or payload underflow

    # System errors
    ERROR_API = "ERROR_API"            # TimeGPT API call failed
    ERROR_DATA = "ERROR_DATA"          # Data processing error
    ERROR_UNKNOWN = "ERROR_UNKNOWN"    # Unexpected error


class CoverageTracker:
    """Tracks forecast execution coverage and skip reasons."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

        # Counters for tracking execution
        self.counters: dict[str, int] = {
            "planned": 0,
            "sent": 0,
            "skipped": 0,
            "ok": 0,
            "fail": 0,
        }

        # Detailed skip reason tracking
        self.skip_reasons: dict[SkipReason, int] = {reason: 0 for reason in SkipReason}

        # Track snapshot timestamps for debugging
        self.planned_snapshots: list[str] = []
        self.sent_snapshots: list[str] = []
        self.skipped_snapshots: dict[SkipReason, list[str]] = {reason: [] for reason in SkipReason}

    def add_planned(self, snapshot_ts: str) -> None:
        """Record a planned snapshot."""
        self.counters["planned"] += 1
        self.planned_snapshots.append(snapshot_ts)

    def add_sent(self, snapshot_ts: str) -> None:
        """Record a successfully sent forecast request."""
        self.counters["sent"] += 1
        self.counters["ok"] += 1
        self.sent_snapshots.append(snapshot_ts)

    def add_skipped(self, snapshot_ts: str, reason: SkipReason, details: str | None = None) -> None:
        """Record a skipped snapshot with reason."""
        self.counters["skipped"] += 1
        self.skip_reasons[reason] += 1
        self.skipped_snapshots[reason].append(snapshot_ts)

        log_msg = f"SKIP snapshot {snapshot_ts} reason={reason}"
        if details:
            log_msg += f" details={details}"
        self.logger.info(log_msg)

    def add_failed(self, snapshot_ts: str, reason: SkipReason, details: str | None = None) -> None:
        """Record a failed forecast attempt."""
        self.counters["fail"] += 1
        self.counters["skipped"] += 1  # Failed snapshots are also skipped
        self.skip_reasons[reason] += 1
        self.skipped_snapshots[reason].append(snapshot_ts)

        log_msg = f"FAIL snapshot {snapshot_ts} reason={reason}"
        if details:
            log_msg += f" details={details}"
        self.logger.error(log_msg)

    def get_coverage_summary(self) -> dict[str, object]:
        """Get a comprehensive coverage summary."""
        return {
            "counters": self.counters.copy(),
            "skip_reasons": {reason.value: count for reason, count in self.skip_reasons.items()},
            "coverage_pct": round(self.counters["ok"] / max(self.counters["planned"], 1) * 100, 2),
            "planned_count": len(self.planned_snapshots),
            "sent_count": len(self.sent_snapshots),
        }

    def log_coverage_summary(self) -> None:
        """Log the coverage summary in a structured format."""
        summary = self.get_coverage_summary()

        # Build the skip reason details
        active_skip_reasons = {reason: count for reason, count in summary["skip_reasons"].items() if count > 0}

        self.logger.info(
            "COVERAGE planned=%s sent=%s ok=%s fail=%s skipped=%s coverage_pct=%s%% by_reason=%s",
            summary["counters"]["planned"],
            summary["counters"]["sent"],
            summary["counters"]["ok"],
            summary["counters"]["fail"],
            summary["counters"]["skipped"],
            summary["coverage_pct"],
            active_skip_reasons
        )

    def get_skip_details(self, reason: SkipReason) -> list[str]:
        """Get list of skipped snapshots for a specific reason."""
        return self.skipped_snapshots[reason].copy()

    def reset(self) -> None:
        """Reset all counters and tracking."""
        self.counters = {"planned": 0, "sent": 0, "skipped": 0, "ok": 0, "fail": 0}
        self.skip_reasons = {reason: 0 for reason in SkipReason}
        self.planned_snapshots.clear()
        self.sent_snapshots.clear()
        for reason in SkipReason:
            self.skipped_snapshots[reason].clear()


# Global constants for easy access
SKIP_REASONS: Final = SkipReason
ALL_SKIP_REASONS: Final = list(SkipReason)


def format_skip_report(tracker: CoverageTracker) -> str:
    """Format a detailed skip report for debugging."""
    lines = ["=== FORECAST COVERAGE REPORT ==="]
    summary = tracker.get_coverage_summary()

    # Basic stats
    lines.append(f"Planned snapshots: {summary['counters']['planned']}")
    lines.append(f"Sent requests: {summary['counters']['sent']}")
    lines.append(f"Successful: {summary['counters']['ok']}")
    lines.append(f"Failed: {summary['counters']['fail']}")
    lines.append(f"Coverage: {summary['coverage_pct']}%")
    lines.append("")

    # Skip reasons
    lines.append("SKIP REASONS:")
    for reason, count in summary["skip_reasons"].items():
        if count > 0:
            lines.append(f"  {reason}: {count}")

    lines.append("")

    # Detailed snapshots for each skip reason (limit to first 10 for readability)
    lines.append("SKIP DETAILS (first 10 per reason):")
    for reason in SkipReason:
        snapshots = tracker.get_skip_details(reason)
        if snapshots:
            lines.append(f"  {reason}:")
            for ts in snapshots[:10]:
                lines.append(f"    {ts}")
            if len(snapshots) > 10:
                lines.append(f"    ... and {len(snapshots) - 10} more")

    return "\n".join(lines)


__all__ = [
    "SkipReason",
    "CoverageTracker",
    "SKIP_REASONS",
    "ALL_SKIP_REASONS",
    "format_skip_report",
]