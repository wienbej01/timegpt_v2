"""Tests for forecast calibration functionality."""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from timegpt_v2.eval.calibration import (
    AffineCalibration,
    CalibrationConfig,
    CalibrationModel,
    ForecastCalibrator,
    apply_conformal_widening,
    compute_embargo_cutoff,
    enforce_quantile_monotonicity,
    filter_calibration_window,
)


class TestEnforceQuantileMonotonicity:
    """Test quantile monotonicity enforcement."""

    def test_enforce_monotonicity_crossing(self) -> None:
        """Crossing quantiles should be projected to monotonic order."""
        data = pd.DataFrame(
            {
                "q10": [0.1, 0.2],
                "q25": [0.3, 0.1],
                "q50": [0.2, 0.3],
                "q75": [0.4, 0.4],
                "q90": [0.5, 0.5],
            }
        )

        result = enforce_quantile_monotonicity(data)

        assert (
            result.loc[0, "q10"]
            <= result.loc[0, "q25"]
            <= result.loc[0, "q50"]
            <= result.loc[0, "q75"]
            <= result.loc[0, "q90"]
        )
        assert (
            result.loc[1, "q10"]
            <= result.loc[1, "q25"]
            <= result.loc[1, "q50"]
            <= result.loc[1, "q75"]
            <= result.loc[1, "q90"]
        )

    def test_enforce_monotonicity_already_monotonic(self) -> None:
        """Already monotonic quantiles should be unchanged."""
        data = pd.DataFrame(
            {
                "q10": [0.1, 0.2],
                "q25": [0.2, 0.3],
                "q50": [0.3, 0.4],
                "q75": [0.4, 0.5],
                "q90": [0.5, 0.6],
            }
        )

        result = enforce_quantile_monotonicity(data)

        pd.testing.assert_frame_equal(result, data)

    def test_enforce_monotonicity_empty(self) -> None:
        """Empty frames should remain empty."""
        data = pd.DataFrame()

        result = enforce_quantile_monotonicity(data)

        assert result.empty


class TestApplyConformalWidening:
    """Test conformal widening fallback logic."""

    def test_conformal_widening_high_deviation(self) -> None:
        """Intervals widen when PIT deviation exceeds threshold."""
        forecasts = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "ts_utc": ["2024-01-01T10:00:00Z", "2024-01-02T10:00:00Z"],
                "q25": [0.1, 0.2],
                "q50": [0.2, 0.3],
                "q75": [0.3, 0.4],
            }
        )

        actuals = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "ts_utc": ["2024-01-01T10:00:00Z", "2024-01-02T10:00:00Z"],
                "y_true": [0.5, 0.6],
            }
        )

        result = apply_conformal_widening(
            forecasts,
            actuals,
            pit_deviation_threshold=0.0,
            window=10,
        )

        assert result.loc[0, "q25"] < forecasts.loc[0, "q25"]
        assert result.loc[0, "q75"] > forecasts.loc[0, "q75"]

    def test_conformal_widening_low_deviation(self) -> None:
        """Intervals remain unchanged when deviation within tolerance."""
        forecasts = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "ts_utc": ["2024-01-01T10:00:00Z", "2024-01-02T10:00:00Z"],
                "q25": [0.1, 0.1],
                "q50": [0.2, 0.2],
                "q75": [0.3, 0.3],
            }
        )

        actuals = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "ts_utc": ["2024-01-01T10:00:00Z", "2024-01-02T10:00:00Z"],
                "y_true": [0.2, 0.4],
            }
        )

        result = apply_conformal_widening(
            forecasts,
            actuals,
            pit_deviation_threshold=0.1,
            window=5,
        )

        pd.testing.assert_frame_equal(result, forecasts)

    def test_conformal_widening_respects_window(self) -> None:
        """Rolling window quantile controls widening magnitude."""
        forecasts = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 4,
                "ts_utc": [
                    "2024-01-01T10:00:00Z",
                    "2024-01-02T10:00:00Z",
                    "2024-01-03T10:00:00Z",
                    "2024-01-04T10:00:00Z",
                ],
                "q25": [0.1, 0.1, 0.1, 0.1],
                "q50": [0.1, 0.1, 0.1, 0.1],
                "q75": [0.1, 0.1, 0.1, 0.1],
            }
        )

        actuals = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 4,
                "ts_utc": [
                    "2024-01-01T10:00:00Z",
                    "2024-01-02T10:00:00Z",
                    "2024-01-03T10:00:00Z",
                    "2024-01-04T10:00:00Z",
                ],
                "y_true": [0.4, 0.3, 0.2, 0.15],
            }
        )

        result = apply_conformal_widening(
            forecasts,
            actuals,
            pit_deviation_threshold=0.0,
            window=2,
        )

        deltas = result["q75"] - result["q50"]
        assert deltas.iloc[0] == pytest.approx(0.3)
        assert deltas.iloc[1] == pytest.approx(0.225)
        assert deltas.iloc[2] <= deltas.iloc[1]
        assert deltas.iloc[3] <= deltas.iloc[2]

    def test_conformal_widening_skips_symbols_without_actuals(self) -> None:
        """Only symbols with actuals should be widened."""
        forecasts = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
                "ts_utc": ["2024-01-01T10:00:00Z", "2024-01-01T10:00:00Z"],
                "q25": [0.1, 0.2],
                "q50": [0.2, 0.3],
                "q75": [0.3, 0.4],
            }
        )

        actuals = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "ts_utc": ["2024-01-01T10:00:00Z"],
                "y_true": [0.5],
            }
        )

        result = apply_conformal_widening(
            forecasts,
            actuals,
            pit_deviation_threshold=0.0,
            window=5,
        )

        assert result.loc[0, "q25"] < forecasts.loc[0, "q25"]
        assert result.loc[0, "q75"] > forecasts.loc[0, "q75"]
        pd.testing.assert_series_equal(result.loc[1], forecasts.loc[1])


class TestCalibrationConfig:
    """Test calibration configuration parsing."""

    def test_from_mapping_default(self) -> None:
        config = CalibrationConfig.from_mapping(None)
        assert config.method == "affine"
        assert config.min_samples == 50
        assert config.calibration_window_days == 30
        assert config.model_path == "models/calibration.json"
        assert config.conformal_fallback is False
        assert config.pit_deviation_threshold == pytest.approx(0.03)
        assert config.conformal_window == 50

    def test_from_mapping_custom(self) -> None:
        payload = {
            "method": "isotonic",
            "min_samples": 100,
            "calibration_window_days": 60,
            "model_path": "custom/path.json",
            "conformal_fallback": True,
            "pit_deviation_threshold": 0.05,
            "conformal_window": 25,
        }
        config = CalibrationConfig.from_mapping(payload)
        assert config.method == "isotonic"
        assert config.min_samples == 100
        assert config.calibration_window_days == 60
        assert config.model_path == "custom/path.json"
        assert config.conformal_fallback is True
        assert config.pit_deviation_threshold == pytest.approx(0.05)
        assert config.conformal_window == 25

    def test_from_mapping_invalid_method(self) -> None:
        with pytest.raises(ValueError, match="calibration.method must be one of"):
            CalibrationConfig.from_mapping({"method": "invalid"})


class TestAffineCalibration:
    """Test affine calibration transforms."""

    def test_apply(self) -> None:
        cal = AffineCalibration(slope=2.0, intercept=1.0)
        values = np.array([1.0, 2.0, 3.0])
        result = cal.apply(values)
        np.testing.assert_array_equal(result, np.array([3.0, 5.0, 7.0]))

    def test_inverse(self) -> None:
        cal = AffineCalibration(slope=2.0, intercept=1.0)
        values = np.array([3.0, 5.0, 7.0])
        result = cal.inverse(values)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_inverse_zero_slope(self) -> None:
        cal = AffineCalibration(slope=0.0, intercept=1.0)
        with pytest.raises(ValueError, match="Cannot invert"):
            cal.inverse(np.array([1.0]))


class TestCalibrationModel:
    """Test calibration model persistence."""

    def test_save_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.json"
            config = CalibrationConfig(
                method="affine",
                model_path=str(path),
                conformal_fallback=True,
                pit_deviation_threshold=0.02,
                conformal_window=15,
            )
            model = CalibrationModel(config=config)
            model.affine_models["AAPL"] = {
                "q50": AffineCalibration(
                    slope=1.5, intercept=0.1, n_samples=100, last_updated="2024-01-01"
                )
            }

            model.save(path)
            loaded = CalibrationModel.load(path)

            assert loaded.config.method == "affine"
            assert loaded.config.conformal_fallback is True
            assert loaded.config.pit_deviation_threshold == pytest.approx(0.02)
            assert loaded.config.conformal_window == 15
            assert "AAPL" in loaded.affine_models
            assert "q50" in loaded.affine_models["AAPL"]
            cal = loaded.affine_models["AAPL"]["q50"]
            assert cal.slope == 1.5
            assert cal.intercept == 0.1
            assert cal.n_samples == 100

    def test_load_nonexistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing.json"
            model = CalibrationModel.load(path)
            assert model.affine_models == {}
            assert model.isotonic_models == {}


class TestForecastCalibrator:
    """Test forecast calibrator fitting and application."""

    def test_fit_affine(self) -> None:
        config = CalibrationConfig(
            method="affine",
            min_samples=2,
            conformal_fallback=True,
            pit_deviation_threshold=0.1,
            conformal_window=10,
        )
        calibrator = ForecastCalibrator(config)
        forecasts = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "ts_utc": ["2024-01-01T10:00:00Z", "2024-01-02T10:00:00Z"],
                "q50": [0.1, 0.2],
            }
        )
        actuals = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "ts_utc": ["2024-01-01T10:00:00Z", "2024-01-02T10:00:00Z"],
                "y_true": [0.15, 0.25],
            }
        )

        calibrator.fit(forecasts, actuals)
        stats = calibrator.get_calibration_stats()
        assert len(stats) == 1
        assert stats.iloc[0]["symbol"] == "AAPL"
        assert stats.iloc[0]["quantile"] == "q50"
        assert stats.iloc[0]["method"] == "affine"

    def test_apply_affine(self) -> None:
        config = CalibrationConfig(method="affine")
        calibrator = ForecastCalibrator(config)
        calibrator._model.affine_models["AAPL"] = {
            "q50": AffineCalibration(slope=2.0, intercept=0.1)
        }
        forecasts = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
                "ts_utc": ["2024-01-01T10:00:00Z", "2024-01-01T10:00:00Z"],
                "q50": [0.1, 0.2],
            }
        )

        result = calibrator.apply(forecasts)
        assert result.loc[0, "q50"] == pytest.approx(0.3)
        assert result.loc[1, "q50"] == pytest.approx(0.2)

    def test_apply_none_method(self) -> None:
        config = CalibrationConfig(method="none")
        calibrator = ForecastCalibrator(config)
        forecasts = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "ts_utc": ["2024-01-01T10:00:00Z"],
                "q50": [0.1],
            }
        )

        result = calibrator.apply(forecasts)
        pd.testing.assert_frame_equal(result, forecasts)

    def test_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.json"
            config = CalibrationConfig(
                model_path=str(path),
                conformal_fallback=True,
                pit_deviation_threshold=0.02,
                conformal_window=20,
            )
            calibrator = ForecastCalibrator(config)
            forecasts = pd.DataFrame(
                {
                    "symbol": ["AAPL"],
                    "ts_utc": ["2024-01-01T10:00:00Z"],
                    "q50": [0.1],
                }
            )
            actuals = pd.DataFrame(
                {
                    "symbol": ["AAPL"],
                    "ts_utc": ["2024-01-01T10:00:00Z"],
                    "y_true": [0.15],
                }
            )
            calibrator.fit(forecasts, actuals)
            calibrator.save()

            new_calibrator = ForecastCalibrator(config)
            assert new_calibrator.load()
            result = new_calibrator.apply(forecasts)
            assert result.loc[0, "q50"] != pytest.approx(0.1)

    def test_get_calibration_stats(self) -> None:
        config = CalibrationConfig()
        calibrator = ForecastCalibrator(config)
        calibrator._model.affine_models["AAPL"] = {
            "q50": AffineCalibration(
                slope=1.5, intercept=0.1, n_samples=100, last_updated="2024-01-01"
            )
        }
        calibrator._model.isotonic_models["MSFT"] = {"q75": None}

        stats = calibrator.get_calibration_stats()
        assert len(stats) == 2
        affine_row = stats[(stats["symbol"] == "AAPL") & (stats["quantile"] == "q50")]
        assert len(affine_row) == 1
        assert affine_row.iloc[0]["method"] == "affine"
        assert affine_row.iloc[0]["slope"] == pytest.approx(1.5)
        isotonic_row = stats[(stats["symbol"] == "MSFT") & (stats["quantile"] == "q75")]
        assert len(isotonic_row) == 1
        assert isotonic_row.iloc[0]["method"] == "isotonic"


class TestCalibrationEmbargoUtilities:
    """Ensure embargo helpers enforce temporal integrity."""

    def test_compute_embargo_cutoff_skips_weekends_and_holidays(self) -> None:
        eval_start = date(2024, 1, 10)  # Wednesday
        holidays = {date(2024, 1, 8)}  # Monday
        cutoff = compute_embargo_cutoff(eval_start, embargo_days=2, holidays=holidays)
        # Expected path: 2024-01-10 -> 9 (Tue) -> 8 (Mon, holiday) -> 7 (Sun) -> 6 (Sat) -> 5 (Fri)
        assert cutoff == date(2024, 1, 5)

    def test_filter_calibration_window_bounds(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC")
        forecasts = pd.DataFrame(
            {
                "symbol": ["AAPL"] * len(timestamps),
                "ts_utc": timestamps,
                "q50": np.linspace(0.0, 0.09, len(timestamps)),
                "y_true": np.linspace(0.1, 0.19, len(timestamps)),
            }
        )
        embargo_cutoff = date(2024, 1, 8)
        filtered, window_start = filter_calibration_window(
            forecasts,
            embargo_cutoff=embargo_cutoff,
            window_days=3,
        )
        assert window_start == date(2024, 1, 5)
        assert not filtered.empty
        assert filtered["ts_utc"].dt.date.max() <= embargo_cutoff


from timegpt_v2.eval.calibration import generate_coverage_report, split_conformal, widen_intervals


class TestWidenIntervals:
    """Test post-hoc quantile widening."""

    def test_widen_intervals_basic(self) -> None:
        """Basic widening functionality."""
        q25 = np.array([0.1, 0.2])
        q50 = np.array([0.2, 0.3])
        q75 = np.array([0.3, 0.4])
        alpha = 0.5

        new_q25, new_q50, new_q75 = widen_intervals(q25, q50, q75, alpha)

        # Check that q50 remains unchanged
        np.testing.assert_array_equal(new_q50, q50)

        # Check that intervals are widened
        assert new_q25[0] < q25[0]
        assert new_q75[0] > q75[0]
        assert new_q25[1] < q25[1]
        assert new_q75[1] > q75[1]

    def test_widen_intervals_alpha_zero(self) -> None:
        """Zero alpha should leave intervals unchanged."""
        q25 = np.array([0.1])
        q50 = np.array([0.2])
        q75 = np.array([0.3])
        alpha = 0.0

        new_q25, new_q50, new_q75 = widen_intervals(q25, q50, q75, alpha)

        assert np.allclose(new_q25, q25)
        assert np.allclose(new_q50, q50)
        assert np.allclose(new_q75, q75)


class TestSplitConformal:
    """Test split-conformal prediction."""

    def test_split_conformal_basic(self) -> None:
        """Basic conformal prediction functionality."""
        residuals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        quantiles = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])

        adjusted_quantiles, widths = split_conformal(residuals, quantiles, alpha=0.1)

        # Check shape
        assert adjusted_quantiles.shape == (2, 3)
        assert len(widths) == 2

        # Check that q50 is unchanged
        np.testing.assert_array_equal(adjusted_quantiles[:, 1], quantiles[:, 1])

        # Check that intervals are widened
        assert adjusted_quantiles[0, 0] < quantiles[0, 0]
        assert adjusted_quantiles[0, 2] > quantiles[0, 2]

    def test_split_conformal_insufficient_data(self) -> None:
        """Insufficient data should return original quantiles."""
        residuals = np.array([0.1])
        quantiles = np.array([[0.1, 0.2, 0.3]])

        adjusted_quantiles, widths = split_conformal(residuals, quantiles, alpha=0.1)

        np.testing.assert_array_equal(adjusted_quantiles, quantiles)
        np.testing.assert_array_equal(widths, quantiles)


class TestGenerateCoverageReport:
    """Test coverage report generation."""

    def test_generate_coverage_report_basic(self) -> None:
        """Basic coverage report functionality."""
        forecasts = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "MSFT"],
                "ts_utc": ["2024-01-01T10:00:00Z", "2024-01-01T11:00:00Z", "2024-01-01T10:00:00Z"],
                "snapshot_utc": [
                    "2024-01-01T10:00:00Z",
                    "2024-01-01T11:00:00Z",
                    "2024-01-01T10:00:00Z",
                ],
                "q25": [0.1, 0.2, 0.15],
                "q50": [0.2, 0.3, 0.25],
                "q75": [0.3, 0.4, 0.35],
            }
        )

        actuals = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "MSFT"],
                "ts_utc": ["2024-01-01T10:00:00Z", "2024-01-01T11:00:00Z", "2024-01-01T10:00:00Z"],
                "y_true": [0.25, 0.35, 0.3],
            }
        )

        report = generate_coverage_report(forecasts, actuals)

        assert len(report) == 3  # One row per forecast row
        assert "symbol" in report.columns
        assert "snapshot_utc" in report.columns
        assert "coverage" in report.columns
        assert "count" in report.columns

        # Check coverage calculations
        assert report.loc[0, "coverage"] == 1.0  # 0.25 is within [0.1, 0.3]
        assert report.loc[1, "coverage"] == 1.0  # 0.35 is within [0.2, 0.4]
        assert report.loc[2, "coverage"] == 1.0  # 0.3 is within [0.15, 0.35]

    def test_generate_coverage_report_empty(self) -> None:
        """Empty inputs should return empty report."""
        report = generate_coverage_report(pd.DataFrame(), pd.DataFrame())
        assert report.empty
