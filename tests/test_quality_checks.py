from __future__ import annotations

import pandas as pd

from timegpt_v2.quality.checks import DataQualityChecker
from timegpt_v2.utils.synthetic import generate_bars


def test_schema_failure_detected():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-07-01", periods=2, freq="1min"),
            "symbol": ["AAPL", "AAPL"],
        }
    )
    checker = DataQualityChecker()
    _, report = checker.validate(frame)

    schema_check = next(check for check in report.checks if check.name == "schema")
    assert not schema_check.passed
    assert not report.passed


def test_quality_passes_on_synthetic_full_day():
    frame = generate_bars()
    checker = DataQualityChecker()
    clean, report = checker.validate(frame)

    assert report.passed
    assert "ffill_flag" in clean.columns
    assert clean["ffill_flag"].sum() == 0


def test_rth_completeness_failure_flags_day():
    frame = generate_bars()
    # remove a block of bars to violate 95% completeness
    frame = frame.drop(index=frame.index[10:60]).reset_index(drop=True)

    checker = DataQualityChecker()
    _, report = checker.validate(frame)

    completeness = next(check for check in report.checks if check.name == "rth_completeness")
    assert not completeness.passed
    assert not report.passed
