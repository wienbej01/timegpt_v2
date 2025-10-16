"""
Trading window diagnostics and coverage analysis utilities.

This module provides tools for diagnosing trading window violations,
coverage gaps, and other date-related issues in the trading pipeline.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Tuple

import pandas as pd

from timegpt_v2.config.model import TradingWindowConfig
from timegpt_v2.utils.coverage import CoverageTracker, SkipReason


@dataclass
class TradingWindowDiagnostic:
    """Diagnostic information about trading window compliance."""

    total_snapshots_planned: int
    snapshots_in_trading_window: int
    snapshots_outside_window: int
    window_violations: int
    coverage_gaps: List[Tuple[date, date]]
    skip_reasons: Dict[str, int]
    per_symbol_stats: Dict[str, Dict[str, int]]
    compliance_rate: float
    first_snapshot: datetime
    last_snapshot: datetime


class TradingWindowDoctor:
    """Diagnostic tool for trading window analysis and coverage reporting."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def analyze_trading_window_compliance(
        self,
        trading_window: TradingWindowConfig,
        coverage_tracker: CoverageTracker,
        snapshots_planned: List[datetime],
        features_df: pd.DataFrame,
        forecasts_df: pd.DataFrame | None = None,
    ) -> TradingWindowDiagnostic:
        """
        Analyze trading window compliance and generate diagnostic report.

        Args:
            trading_window: Trading window configuration
            coverage_tracker: Coverage tracker with skip reasons
            snapshots_planned: List of planned snapshot timestamps
            features_df: Feature matrix for reference
            forecasts_df: Optional forecast results

        Returns:
            TradingWindowDiagnostic with comprehensive analysis
        """
        self.logger.info("Starting trading window compliance analysis...")

        # Basic counts
        total_snapshots = len(snapshots_planned)
        snapshots_in_window = 0
        snapshots_outside_window = 0

        # Analyze each snapshot
        snapshot_dates = [ts.date() for ts in snapshots_planned]
        skip_reasons = defaultdict(int)
        per_symbol_stats = defaultdict(lambda: defaultdict(int))

        # Get coverage summary
        coverage_summary = coverage_tracker.get_coverage_summary()
        skip_reasons.update(coverage_summary["counters"])

        # Count window compliance
        if trading_window.start and trading_window.end:
            for snapshot_date in snapshot_dates:
                if trading_window.start <= snapshot_date <= trading_window.end:
                    snapshots_in_window += 1
                else:
                    snapshots_outside_window += 1

        # Calculate compliance rate
        compliance_rate = (snapshots_in_window / total_snapshots * 100) if total_snapshots > 0 else 0

        # Identify coverage gaps
        coverage_gaps = self._identify_coverage_gaps(snapshot_dates, features_df)

        # Per-symbol analysis
        symbols = features_df["symbol"].unique() if "symbol" in features_df.columns else []
        for symbol in symbols:
            symbol_stats = self._analyze_symbol_coverage(
                symbol, trading_window, coverage_tracker, snapshots_planned, features_df
            )
            per_symbol_stats[symbol] = symbol_stats

        # Get first/last snapshots
        first_snapshot = snapshots_planned[0] if snapshots_planned else datetime.min
        last_snapshot = snapshots_planned[-1] if snapshots_planned else datetime.max

        # Count window violations (from coverage tracker)
        window_violations = (
            skip_reasons.get("SKIP_BEFORE_TRADE_WINDOW", 0) +
            skip_reasons.get("SKIP_AFTER_TRADE_WINDOW", 0)
        )

        diagnostic = TradingWindowDiagnostic(
            total_snapshots_planned=total_snapshots,
            snapshots_in_trading_window=snapshots_in_window,
            snapshots_outside_window=snapshots_outside_window,
            window_violations=window_violations,
            coverage_gaps=coverage_gaps,
            skip_reasons=dict(skip_reasons),
            per_symbol_stats=dict(per_symbol_stats),
            compliance_rate=compliance_rate,
            first_snapshot=first_snapshot,
            last_snapshot=last_snapshot,
        )

        self.logger.info(
            f"Trading window analysis complete: {snapshots_in_window}/{total_snapshots} "
            f"snapshots in window ({compliance_rate:.1f}% compliance)"
        )

        return diagnostic

    def _identify_coverage_gaps(
        self,
        snapshot_dates: List[date],
        features_df: pd.DataFrame
    ) -> List[Tuple[date, date]]:
        """Identify gaps in data coverage."""
        gaps = []

        if "timestamp" not in features_df.columns:
            return gaps

        # Get actual data dates
        features_df["timestamp"] = pd.to_datetime(features_df["timestamp"])
        actual_dates = sorted(features_df["timestamp"].dt.date.unique())

        if not actual_dates:
            return gaps

        # Find gaps between consecutive dates
        for i in range(len(actual_dates) - 1):
            current_date = actual_dates[i]
            next_date = actual_dates[i + 1]

            # Check if there's a weekend gap (normal)
            current_weekday = current_date.weekday()
            next_weekday = next_date.weekday()

            # If gap is more than 1 day (or crosses weekend), it's a coverage gap
            date_diff = (next_date - current_date).days
            if date_diff > 1:
                # Adjust for weekends
                if current_weekday == 4 and next_weekday == 0:  # Fri to Mon
                    continue  # Normal weekend gap
                elif date_diff > 3:  # More than weekend
                    gaps.append((current_date, next_date))

        return gaps

    def _analyze_symbol_coverage(
        self,
        symbol: str,
        trading_window: TradingWindowConfig,
        coverage_tracker: CoverageTracker,
        snapshots_planned: List[datetime],
        features_df: pd.DataFrame,
    ) -> Dict[str, int]:
        """Analyze coverage for a specific symbol."""
        stats = {
            "snapshots_planned": 0,
            "snapshots_sent": 0,
            "snapshots_skipped": 0,
            "data_available": 0,
            "window_violations": 0,
        }

        # Count planned snapshots for this symbol
        stats["snapshots_planned"] = len(snapshots_planned)

        # Get symbol-specific data
        symbol_features = features_df[features_df["symbol"] == symbol] if "symbol" in features_df.columns else pd.DataFrame()

        if not symbol_features.empty and "timestamp" in symbol_features.columns:
            symbol_features["timestamp"] = pd.to_datetime(symbol_features["timestamp"])
            unique_dates = symbol_features["timestamp"].dt.date.nunique()
            stats["data_available"] = unique_dates

        # Count sent/skipped snapshots from coverage tracker
        coverage_summary = coverage_tracker.get_coverage_summary()
        stats["snapshots_sent"] = coverage_summary["counters"].get("ok", 0)

        # Count skipped snapshots
        for reason, count in coverage_summary["counters"].items():
            if reason.startswith("SKIP_") or reason.startswith("ERROR_"):
                stats["snapshots_skipped"] += count

        # Count window violations specifically
        stats["window_violations"] = (
            coverage_summary["counters"].get("SKIP_BEFORE_TRADE_WINDOW", 0) +
            coverage_summary["counters"].get("SKIP_AFTER_TRADE_WINDOW", 0)
        )

        return stats

    def generate_diagnostic_report(
        self,
        diagnostic: TradingWindowDiagnostic,
        trading_window: TradingWindowConfig,
        output_format: str = "text"
    ) -> str:
        """
        Generate a comprehensive diagnostic report.

        Args:
            diagnostic: Trading window diagnostic results
            trading_window: Trading window configuration
            output_format: Output format ('text' or 'json')

        Returns:
            Formatted diagnostic report
        """
        if output_format == "json":
            return self._generate_json_report(diagnostic, trading_window)
        else:
            return self._generate_text_report(diagnostic, trading_window)

    def _generate_text_report(
        self,
        diagnostic: TradingWindowDiagnostic,
        trading_window: TradingWindowConfig
    ) -> str:
        """Generate text-based diagnostic report."""
        lines = []
        lines.append("=" * 60)
        lines.append("TRADING WINDOW DIAGNOSTIC REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Trading window configuration
        lines.append("TRADING WINDOW CONFIGURATION:")
        if trading_window.start and trading_window.end:
            lines.append(f"  Trading Period: {trading_window.start} to {trading_window.end}")
            lines.append(f"  History Backfill: {trading_window.history_backfill_days} days")
            lines.append(f"  Enforcement: {'ENABLED' if trading_window.enforce_trading_window else 'DISABLED (permissive)'}")
        else:
            lines.append("  Trading Window: Not configured (using legacy mode)")
        lines.append("")

        # Summary statistics
        lines.append("SUMMARY STATISTICS:")
        lines.append(f"  Total Snapshots Planned: {diagnostic.total_snapshots_planned}")
        lines.append(f"  Snapshots in Trading Window: {diagnostic.snapshots_in_trading_window}")
        lines.append(f"  Snapshots Outside Window: {diagnostic.snapshots_outside_window}")
        lines.append(f"  Trading Window Compliance: {diagnostic.compliance_rate:.1f}%")
        lines.append(f"  Window Violations: {diagnostic.window_violations}")
        lines.append(f"  Date Range: {diagnostic.first_snapshot.date()} to {diagnostic.last_snapshot.date()}")
        lines.append("")

        # Skip reasons
        lines.append("SKIP REASONS:")
        if diagnostic.skip_reasons:
            for reason, count in sorted(diagnostic.skip_reasons.items()):
                lines.append(f"  {reason}: {count}")
        else:
            lines.append("  No skips recorded")
        lines.append("")

        # Coverage gaps
        if diagnostic.coverage_gaps:
            lines.append("COVERAGE GAPS:")
            for gap_start, gap_end in diagnostic.coverage_gaps:
                lines.append(f"  {gap_start} to {gap_end}")
            lines.append("")

        # Per-symbol breakdown
        if diagnostic.per_symbol_stats:
            lines.append("PER-SYMBOL BREAKDOWN:")
            for symbol, stats in sorted(diagnostic.per_symbol_stats.items()):
                lines.append(f"  {symbol}:")
                for metric, value in stats.items():
                    lines.append(f"    {metric}: {value}")
                lines.append("")

        # Recommendations
        lines.append("RECOMMENDATIONS:")
        recommendations = self._generate_recommendations(diagnostic, trading_window)
        if recommendations:
            for rec in recommendations:
                lines.append(f"  • {rec}")
        else:
            lines.append("  No issues detected - trading window compliance looks good!")

        return "\n".join(lines)

    def _generate_json_report(
        self,
        diagnostic: TradingWindowDiagnostic,
        trading_window: TradingWindowConfig
    ) -> str:
        """Generate JSON-based diagnostic report."""
        import json

        report = {
            "trading_window": {
                "start": trading_window.start.isoformat() if trading_window.start else None,
                "end": trading_window.end.isoformat() if trading_window.end else None,
                "history_backfill_days": trading_window.history_backfill_days,
                "enforce_trading_window": trading_window.enforce_trading_window,
            },
            "summary": {
                "total_snapshots_planned": diagnostic.total_snapshots_planned,
                "snapshots_in_trading_window": diagnostic.snapshots_in_trading_window,
                "snapshots_outside_window": diagnostic.snapshots_outside_window,
                "trading_window_compliance_percent": diagnostic.compliance_rate,
                "window_violations": diagnostic.window_violations,
                "date_range": {
                    "start": diagnostic.first_snapshot.isoformat(),
                    "end": diagnostic.last_snapshot.isoformat(),
                }
            },
            "skip_reasons": diagnostic.skip_reasons,
            "coverage_gaps": [
                {"start": start.isoformat(), "end": end.isoformat()}
                for start, end in diagnostic.coverage_gaps
            ],
            "per_symbol_stats": diagnostic.per_symbol_stats,
            "recommendations": self._generate_recommendations(diagnostic, trading_window),
        }

        return json.dumps(report, indent=2)

    def _generate_recommendations(
        self,
        diagnostic: TradingWindowDiagnostic,
        trading_window: TradingWindowConfig
    ) -> List[str]:
        """Generate actionable recommendations based on diagnostic results."""
        recommendations = []

        # Low compliance rate
        if diagnostic.compliance_rate < 80:
            recommendations.append(
                f"Low trading window compliance ({diagnostic.compliance_rate:.1f}%). "
                "Consider adjusting snapshot times or trading window dates."
            )

        # Window violations
        if diagnostic.window_violations > 0:
            recommendations.append(
                f"Found {diagnostic.window_violations} trading window violations. "
                "Check snapshot scheduling and trading window configuration."
            )

        # Coverage gaps
        if diagnostic.coverage_gaps:
            recommendations.append(
                f"Found {len(diagnostic.coverage_gaps)} coverage gaps in data. "
                "Investigate data availability for missing dates."
            )

        # Permissive mode warning
        if not trading_window.enforce_trading_window:
            recommendations.append(
                "Trading window enforcement is disabled. "
                "Enable enforcement to prevent trades outside configured dates."
            )

        # High skip rate
        total_skips = sum(
            count for reason, count in diagnostic.skip_reasons.items()
            if reason.startswith("SKIP_") or reason.startswith("ERROR_")
        )
        if diagnostic.total_snapshots_planned > 0:
            skip_rate = (total_skips / diagnostic.total_snapshots_planned) * 100
            if skip_rate > 50:
                recommendations.append(
                    f"High skip rate ({skip_rate:.1f}%). "
                    "Review data quality and configuration settings."
                )

        return recommendations