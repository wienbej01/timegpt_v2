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

        assert result.loc[0, "q10"] <= result.loc[0, "q25"] <= result.loc[0, "q50"] <= result.loc[0, "q75"] <= result.loc[0, "q90"]
        assert result.loc[1, "q10"] <= result.loc[1, "q25"] <= result.loc[1, "q50"] <= result.loc[1, "q75"] <= result.loc[1, "q90"]

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
                "symbol": ["AAPL"],
                "ts_utc": ["2024-01-01T10:00:00Z"],
                "q25": [0.1],
                "q50": [0.2],
                "q75": [0.3],
            }
        )

        actuals = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "ts_utc": ["2024-01-01T10:00:00Z"],
                "y_true": [0.2],
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
        assert deltas.iloc[1] == pytest.approx(0.25)
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
                "q50": AffineCalibration(slope=1.5, intercept=0.1, n_samples=100, last_updated="2024-01-01")
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
        config = CalibrationConfig(method="affine", min_samples=2, conformal_fallback=True, pit_deviation_threshold=0.1, conformal_window=10)
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
            "q50": AffineCalibration(slope=1.5, intercept=0.1, n_samples=100, last_updated="2024-01-01")
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
        assert filtered["ts_utc"].dt.date.min() >= window_start