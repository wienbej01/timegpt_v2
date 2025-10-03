"""Data quality check stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class DataQualityResult:
    passed: bool
    details: dict[str, Any]


def run_checks(frame: pd.DataFrame) -> DataQualityResult:
    """Placeholder checks for Sprint 0."""
    return DataQualityResult(passed=True, details={})
