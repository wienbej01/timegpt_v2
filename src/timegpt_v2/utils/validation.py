"""Validation utilities for forecast configuration limits and budgets."""

from __future__ import annotations

import logging
from datetime import date, datetime, time
from typing import Final

from timegpt_v2.forecast.scheduler import ForecastScheduler

logger = logging.getLogger(__name__)

# Global constants for validation
DEFAULT_MAX_SNAPSHOTS_TOTAL_WARNING: Final = 50
DEFAULT_MAX_SNAPSHOTS_DAY_WARNING: Final = 24
MIN_REASONABLE_DAILY_SNAPSHOTS: Final = 1
MAX_REASONABLE_DAILY_SNAPSHOTS: Final = 24  # One per hour max


def validate_snapshot_limits(
    scheduler: ForecastScheduler,
    *,
    min_reasonable_daily: int = MIN_REASONABLE_DAILY_SNAPSHOTS,
    max_reasonable_daily: int = MAX_REASONABLE_DAILY_SNAPSHOTS,
    warn_total_threshold: int = DEFAULT_MAX_SNAPSHOTS_TOTAL_WARNING,
) -> dict[str, object]:
    """
    Validate snapshot limits and warn about potential issues.

    Args:
        scheduler: Configured ForecastScheduler instance
        min_reasonable_daily: Minimum reasonable snapshots per trading day
        max_reasonable_daily: Maximum reasonable snapshots per trading day
        warn_total_threshold: Threshold for warning about total snapshot limits

    Returns:
        Dictionary with validation results and recommendations
    """
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "recommendations": [],
        "planned_count": 0,
        "limiting_factor": None,
    }

    # Calculate theoretical maximum snapshots without limits
    theoretical_snapshots = []
    for dt in scheduler.dates:
        if scheduler._is_trading_day(dt):
            for snapshot_time in scheduler.snapshots:
                if scheduler._within_active_window(snapshot_time):
                    theoretical_snapshots.append(datetime.combine(dt, snapshot_time, tzinfo=scheduler.tz))

    theoretical_count = len(theoretical_snapshots)

    # Get actual planned snapshots (with limits applied)
    planned_snapshots = scheduler.generate_snapshots()
    planned_count = len(planned_snapshots)
    validation_results["planned_count"] = planned_count

    # Check daily limits
    if scheduler.max_snapshots_per_day is not None:
        max_daily = scheduler.max_snapshots_per_day
        if max_daily < min_reasonable_daily:
            validation_results["errors"].append(
                f"max_snapshots_per_day={max_daily} is too low (minimum {min_reasonable_daily})"
            )
            validation_results["valid"] = False
        elif max_daily > max_reasonable_daily:
            validation_results["warnings"].append(
                f"max_snapshots_per_day={max_daily} is very high (> {max_reasonable_daily})"
            )

        # Check if daily limit would truncate many snapshots
        from collections import defaultdict
        daily_counts = defaultdict(int)
        for snapshot in theoretical_snapshots:
            daily_counts[snapshot.date()] += 1

        truncated_days = [
            (date, count, max_daily)
            for date, count in daily_counts.items()
            if count > max_daily
        ]

        if truncated_days:
            total_truncated = sum(count - max_daily for _, count, max_daily in truncated_days)
            validation_results["warnings"].append(
                f"Daily limit will truncate {total_truncated} snapshots across {len(truncated_days)} days"
            )
            validation_results["limiting_factor"] = "max_snapshots_per_day"

    # Check total limits
    if scheduler.max_total_snapshots is not None:
        max_total = scheduler.max_total_snapshots

        if max_total < theoretical_count:
            truncated_total = theoretical_count - max_total
            validation_results["warnings"].append(
                f"Total limit will truncate {truncated_total} snapshots "
                f"({theoretical_count} theoretical → {max_total} allowed)"
            )
            validation_results["limiting_factor"] = "max_total_snapshots"

        if max_total < warn_total_threshold:
            validation_results["warnings"].append(
                f"max_total_snapshots={max_total} is very low "
                f"(threshold {warn_total_threshold})"
            )

        if max_total < 10:
            validation_results["warnings"].append(
                "max_total_snapshots < 10 may be too restrictive for production runs"
            )

    # Check if no limits are set (could be dangerous for production)
    if scheduler.max_total_snapshots is None and scheduler.max_snapshots_per_day is None:
        validation_results["recommendations"].append(
            "Consider setting max_total_snapshots for production safety"
        )

    # Check snapshot frequency vs daily limits
    if len(scheduler.snapshots) > 0:
        snapshots_per_day = len(scheduler.snapshots)
        if scheduler.max_snapshots_per_day is not None:
            if snapshots_per_day > scheduler.max_snapshots_per_day:
                validation_results["errors"].append(
                    f"Configured {snapshots_per_day} snapshots per day "
                    f"but max_snapshots_per_day={scheduler.max_snapshots_per_day}"
                )
                validation_results["valid"] = False

    return validation_results


def log_validation_results(results: dict[str, object], logger: logging.Logger) -> None:
    """Log validation results in a structured format."""
    if results["errors"]:
        logger.error("VALIDATION ERRORS: %s", "; ".join(results["errors"]))

    if results["warnings"]:
        logger.warning("VALIDATION WARNINGS: %s", "; ".join(results["warnings"]))

    if results["recommendations"]:
        logger.info("VALIDATION RECOMMENDATIONS: %s", "; ".join(results["recommendations"]))

    # Log limiting factor if present
    if results["limiting_factor"]:
        logger.warning(
            "LIMITING FACTOR: %s may restrict forecast execution",
            results["limiting_factor"]
        )

    logger.info(
        "SNAPSHOT VALIDATION: planned=%d valid=%s limiting_factor=%s",
        results["planned_count"],
        results["valid"],
        results["limiting_factor"] or "none"
    )


def get_production_safety_recommendations(results: dict[str, object]) -> list[str]:
    """Get production safety recommendations based on validation results."""
    recommendations = []

    if results["limiting_factor"] == "max_total_snapshots":
        recommendations.append(
            "Consider increasing max_total_snapshots or removing it for full coverage"
        )

    if results["limiting_factor"] == "max_snapshots_per_day":
        recommendations.append(
            "Consider increasing max_snapshots_per_day or reviewing snapshot frequency"
        )

    if results["planned_count"] < 20:
        recommendations.append(
            "Low snapshot count may be insufficient for robust evaluation"
        )

    if results["warnings"]:
        recommendations.append(
            "Review warnings and consider adjusting configuration for production safety"
        )

    return recommendations


__all__ = [
    "validate_snapshot_limits",
    "log_validation_results",
    "get_production_safety_recommendations",
    "DEFAULT_MAX_SNAPSHOTS_TOTAL_WARNING",
    "DEFAULT_MAX_SNAPSHOTS_DAY_WARNING",
]