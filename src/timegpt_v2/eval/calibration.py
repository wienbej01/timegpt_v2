from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the reliability curve."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1])

    bin_sums = np.bincount(bin_ids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(bin_ids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(bin_ids, minlength=len(bins))

    # Avoid division by zero
    nonzero = bin_total > 0
    prob_true = np.full(len(bins), 0.0)
    prob_pred = np.full(len(bins), 0.0)
    prob_true[nonzero] = bin_true[nonzero] / bin_total[nonzero]
    prob_pred[nonzero] = bin_sums[nonzero] / bin_total[nonzero]

    return prob_pred, prob_true


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for forecast calibration."""

    method: str = "affine"  # "affine", "isotonic", or "none"
    min_samples: int = 50
    calibration_window_days: int = 30
    model_path: str = "models/calibration.json"
    conformal_fallback: bool = False
    pit_deviation_threshold: float = 0.03
    conformal_window: int = 50

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object] | None) -> CalibrationConfig:
        if payload is None:
            return cls()

        method = str(payload.get("method", cls.method)).lower()
        if method not in {"affine", "isotonic", "none"}:
            raise ValueError("calibration.method must be one of: affine, isotonic, none")

        min_samples = int(payload.get("min_samples", cls.min_samples))
        calibration_window_days = int(
            payload.get("calibration_window_days", cls.calibration_window_days)
        )
        model_path = str(payload.get("model_path", cls.model_path))
        conformal_fallback = bool(payload.get("conformal_fallback", cls.conformal_fallback))
        pit_deviation_threshold = float(
            payload.get("pit_deviation_threshold", cls.pit_deviation_threshold)
        )
        conformal_window = int(payload.get("conformal_window", cls.conformal_window))

        return cls(
            method=method,
            min_samples=min_samples,
            calibration_window_days=calibration_window_days,
            model_path=model_path,
            conformal_fallback=conformal_fallback,
            pit_deviation_threshold=pit_deviation_threshold,
            conformal_window=conformal_window,
        )


@dataclass
class AffineCalibration:
    """Per-symbol affine calibration parameters."""

    slope: float = 1.0
    intercept: float = 0.0
    n_samples: int = 0
    last_updated: str = ""

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Apply affine transformation."""
        return self.slope * values + self.intercept

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Inverse affine transformation."""
        if abs(self.slope) < 1e-10:
            raise ValueError("Cannot invert affine calibration with near-zero slope")
        return (values - self.intercept) / self.slope


@dataclass
class CalibrationModel:
    """Container for calibration models per symbol and quantile."""

    affine_models: dict[str, dict[str, AffineCalibration]] = field(default_factory=dict)
    isotonic_models: dict[str, dict[str, IsotonicRegression]] = field(default_factory=dict)
    config: CalibrationConfig = field(default_factory=CalibrationConfig)

    def save(self, path: str | Path) -> None:
        """Save calibration parameters to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Only save affine models (isotonic models are not easily serializable)
        serializable_data = {
            "config": {
                "method": self.config.method,
                "min_samples": self.config.min_samples,
                "calibration_window_days": self.config.calibration_window_days,
                "model_path": self.config.model_path,
                "conformal_fallback": self.config.conformal_fallback,
                "pit_deviation_threshold": self.config.pit_deviation_threshold,
                "conformal_window": self.config.conformal_window,
            },
            "affine_models": {},
        }

        for symbol, quantile_models in self.affine_models.items():
            serializable_data["affine_models"][symbol] = {}
            for quantile, model in quantile_models.items():
                serializable_data["affine_models"][symbol][quantile] = {
                    "slope": model.slope,
                    "intercept": model.intercept,
                    "n_samples": model.n_samples,
                    "last_updated": model.last_updated,
                }

        with path.open("w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> CalibrationModel:
        """Load calibration parameters from disk."""
        path = Path(path)
        if not path.exists():
            return cls()

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        config_payload = data.get("config", {})
        config = CalibrationConfig(
            method=str(config_payload.get("method", "affine")),
            min_samples=int(config_payload.get("min_samples", 50)),
            calibration_window_days=int(config_payload.get("calibration_window_days", 30)),
            model_path=str(config_payload.get("model_path", "models/calibration.json")),
            conformal_fallback=bool(config_payload.get("conformal_fallback", False)),
            pit_deviation_threshold=float(config_payload.get("pit_deviation_threshold", 0.03)),
            conformal_window=int(config_payload.get("conformal_window", 50)),
        )

        model = cls(config=config)

        for symbol, quantile_models in data.get("affine_models", {}).items():
            model.affine_models[symbol] = {}
            for quantile, model_data in quantile_models.items():
                model.affine_models[symbol][quantile] = AffineCalibration(
                    slope=model_data["slope"],
                    intercept=model_data["intercept"],
                    n_samples=model_data["n_samples"],
                    last_updated=model_data["last_updated"],
                )

        return model


def compute_embargo_cutoff(
    eval_start: date,
    embargo_days: int,
    holidays: Iterable[date] | None = None,
) -> date:
    """Derive the last permissible calibration date given an embargo."""
    if embargo_days <= 0:
        return eval_start
    holidays_set = set(holidays or ())
    cutoff = eval_start
    trading_days_used = 0
    while trading_days_used < embargo_days:
        cutoff -= timedelta(days=1)
        if cutoff.weekday() < 5 and cutoff not in holidays_set:
            trading_days_used += 1
    return cutoff


def filter_calibration_window(
    forecasts: pd.DataFrame,
    *,
    embargo_cutoff: date,
    window_days: int | None,
) -> tuple[pd.DataFrame, date | None]:
    """Filter forecasts to the embargo-safe calibration window."""
    if "ts_utc" not in forecasts.columns:
        raise KeyError("Forecasts must include 'ts_utc' column for calibration window selection")
    working = forecasts.copy()
    working["ts_utc"] = pd.to_datetime(working["ts_utc"], utc=True, errors="coerce")
    working = working.dropna(subset=["ts_utc"])
    working = working.loc[working["ts_utc"].dt.date <= embargo_cutoff]
    if window_days is None or window_days <= 0:
        return working, None
    window_start = embargo_cutoff - timedelta(days=window_days)
    working = working.loc[working["ts_utc"].dt.date >= window_start]
    return working, window_start


class ForecastCalibrator:
    """Calibrates forecast quantiles using per-symbol affine or isotonic regression."""

    def __init__(self, config: CalibrationConfig) -> None:
        self._config = config
        # Initialize empty model; loading deferred to load()
        self._model = CalibrationModel(config=config)

    def fit(self, forecasts: pd.DataFrame, actuals: pd.DataFrame) -> None:
        """Fit calibration models using historical forecasts and actuals."""
        if self._config.method == "none":
            return

        # Ensure forecasts and actuals are aligned
        symbol_col = "symbol" if "symbol" in forecasts.columns else "unique_id"
        ts_col = "ts_utc" if "ts_utc" in forecasts.columns else "forecast_ts"
        merge_cols = [symbol_col, ts_col]
        merged = forecasts.merge(
            actuals, left_on=merge_cols, right_on=merge_cols, how="inner", suffixes=("", "_actual")
        )

        if merged.empty:
            raise ValueError("No overlapping data between forecasts and actuals")

        # Get quantile columns
        quantile_cols = [col for col in forecasts.columns if col.startswith("q")]

        for _symbol, group in merged.groupby("symbol"):
            if self._config.method != "affine" and len(group) < self._config.min_samples:
                continue

            y_true = group["y_true"].values

            for quantile_col in quantile_cols:
                y_pred = group[quantile_col].values

                # Remove NaN values
                valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                if self._config.method != "affine" and valid_mask.sum() < self._config.min_samples:
                    continue

                y_true_valid = y_true[valid_mask]
                y_pred_valid = y_pred[valid_mask]

                # Affine calibration requires at least 1 sample
                if self._config.method == "affine" and len(y_true_valid) < 1:
                    continue

                if self._config.method == "affine":
                    self._fit_affine(_symbol, quantile_col, y_true_valid, y_pred_valid)
                elif self._config.method == "isotonic":
                    self._fit_isotonic(symbol, quantile_col, y_true_valid, y_pred_valid)

    def _fit_affine(
        self, symbol: str, quantile_col: str, y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """Fit affine calibration for a specific symbol and quantile."""
        # Simple linear regression (y_true = slope * y_pred + intercept)
        lr = LinearRegression()
        lr.fit(y_pred.reshape(-1, 1), y_true)

        slope = float(lr.coef_[0])
        intercept = float(lr.intercept_)

        # Store the calibration parameters
        if symbol not in self._model.affine_models:
            self._model.affine_models[symbol] = {}

        self._model.affine_models[symbol][quantile_col] = AffineCalibration(
            slope=slope,
            intercept=intercept,
            n_samples=len(y_true),
            last_updated=pd.Timestamp.now().isoformat(),
        )

    def _fit_isotonic(
        self, symbol: str, quantile_col: str, y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """Fit isotonic regression calibration for a specific symbol and quantile."""
        # Isotonic regression to ensure monotonicity
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(y_pred, y_true)

        # Store the calibration model
        if symbol not in self._model.isotonic_models:
            self._model.isotonic_models[symbol] = {}

        self._model.isotonic_models[symbol][quantile_col] = ir

    def apply(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        """Apply calibration to forecasts."""
        if self._config.method == "none":
            return forecasts.copy()

        calibrated = forecasts.copy()
        quantile_cols = [col for col in forecasts.columns if col.startswith("q")]

        # Handle both "symbol" and "unique_id" column names
        symbol_col = "symbol" if "symbol" in calibrated.columns else "unique_id"
        for _symbol, group in calibrated.groupby(symbol_col):
            for quantile_col in quantile_cols:
                if self._config.method == "affine":
                    if (
                        _symbol in self._model.affine_models
                        and quantile_col in self._model.affine_models[_symbol]
                    ):
                        model = self._model.affine_models[_symbol][quantile_col]
                        mask = calibrated[symbol_col] == _symbol
                        vals = model.apply(calibrated.loc[mask, quantile_col].values)
                        vals = np.round(vals, 10)
                        calibrated.loc[mask, quantile_col] = vals

                elif self._config.method == "isotonic":
                    if (
                        symbol in self._model.isotonic_models
                        and quantile_col in self._model.isotonic_models[symbol]
                    ):
                        model = self._model.isotonic_models[symbol][quantile_col]
                        mask = calibrated["symbol"] == symbol
                        calibrated.loc[mask, quantile_col] = model.predict(
                            calibrated.loc[mask, quantile_col].values
                        )

        return calibrated

    def save(self) -> None:
        """Save calibration models to disk."""
        self._model.save(self._config.model_path)

    def load(self) -> bool:
        """Load calibration models from disk. Returns True if models existed."""
        model = CalibrationModel.load(self._config.model_path)
        loaded = Path(self._config.model_path).exists()
        self._model = model
        return loaded

    def get_calibration_stats(self) -> pd.DataFrame:
        """Get statistics about fitted calibration models."""
        rows = []

        for symbol, quantile_models in self._model.affine_models.items():
            for quantile, model in quantile_models.items():
                rows.append(
                    {
                        "symbol": symbol,
                        "quantile": quantile,
                        "method": "affine",
                        "slope": model.slope,
                        "intercept": model.intercept,
                        "n_samples": model.n_samples,
                        "last_updated": model.last_updated,
                    }
                )

        for symbol, quantile_models in self._model.isotonic_models.items():
            for quantile in quantile_models.keys():
                rows.append(
                    {
                        "symbol": symbol,
                        "quantile": quantile,
                        "method": "isotonic",
                        "slope": np.nan,
                        "intercept": np.nan,
                        "n_samples": np.nan,
                        "last_updated": np.nan,
                    }
                )

        return pd.DataFrame(rows)


def enforce_quantile_monotonicity(forecasts: pd.DataFrame, logger=None) -> pd.DataFrame:
    """Enforce quantile monotonicity using isotonic regression."""
    import logging

    if logger is None:
        logger = logging.getLogger(__name__)

    if forecasts.empty:
        return forecasts.copy()

    quantile_cols = [col for col in forecasts.columns if col.startswith("q")]
    if not quantile_cols:
        return forecasts.copy()

    result = forecasts.copy()
    quantiles = [float(col[1:]) / 100 for col in quantile_cols]  # q10 -> 0.1, etc.

    violations_detected = 0
    total_rows = len(result)

    for idx, row in result.iterrows():
        values = [row[col] for col in quantile_cols]

        # Check for monotonicity violations before correction
        for i in range(1, len(values)):
            if values[i] < values[i - 1]:
                violations_detected += 1
                if violations_detected <= 5:  # Log first 5 violations for debugging
                    logger.warning(
                        "Quantile monotonicity violation detected at row %s: %s",
                        idx,
                        {col: val for col, val in zip(quantile_cols, values, strict=False)},
                    )
                break

        # Use isotonic regression to enforce monotonicity
        from sklearn.isotonic import IsotonicRegression

        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(quantiles, values)
        monotonic_values = ir.predict(quantiles)
        for col, val in zip(quantile_cols, monotonic_values, strict=False):
            result.loc[idx, col] = val

    if violations_detected > 0:
        logger.info(
            "Quantile monotonicity: detected %d violations out of %d rows (%.1f%%)",
            violations_detected,
            total_rows,
            100.0 * violations_detected / total_rows,
        )

    return result


def apply_conformal_widening(
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    pit_deviation_threshold: float = 0.03,
    window: int = 50,
) -> pd.DataFrame:
    """Apply conformal widening per symbol when PIT deviation exceeds tolerance."""
    if forecasts.empty or actuals.empty:
        return forecasts.copy()

    if window <= 0:
        window = 1

    required_quantiles = {"q25", "q50", "q75"}
    if not required_quantiles.issubset(forecasts.columns):
        return forecasts.copy()

    symbol_col = "symbol" if "symbol" in forecasts.columns else "unique_id"
    if symbol_col not in forecasts.columns:
        return forecasts.copy()

    ts_col = "ts_utc" if "ts_utc" in forecasts.columns else "forecast_ts"
    if ts_col not in forecasts.columns:
        return forecasts.copy()

    actual_symbol_col = symbol_col if symbol_col in actuals.columns else "symbol"
    if actual_symbol_col not in actuals.columns:
        return forecasts.copy()

    actual_ts_col_candidates = [ts_col, "ts_utc", "forecast_ts"]
    actual_ts_col = next((col for col in actual_ts_col_candidates if col in actuals.columns), None)
    if actual_ts_col is None or "y_true" not in actuals.columns:
        return forecasts.copy()

    forecasts_std = forecasts.copy()
    forecasts_std["_symbol"] = forecasts_std[symbol_col]
    forecasts_std["_ts"] = pd.to_datetime(forecasts_std[ts_col], utc=True, errors="coerce")
    forecasts_std["_row"] = forecasts_std.index

    actuals_std = actuals[[actual_symbol_col, actual_ts_col, "y_true"]].copy()
    actuals_std["_symbol"] = actuals_std[actual_symbol_col]
    actuals_std["_ts"] = pd.to_datetime(actuals_std[actual_ts_col], utc=True, errors="coerce")
    actuals_std = actuals_std.dropna(subset=["_ts", "y_true"])

    if forecasts_std["_ts"].isna().all() or actuals_std.empty:
        return forecasts.copy()

    merged = forecasts_std[["_row", "_symbol", "_ts", "q25", "q50", "q75"]].merge(
        actuals_std[["_symbol", "_ts", "y_true"]],
        on=["_symbol", "_ts"],
        how="inner",
    )

    if merged.empty:
        return forecasts.copy()

    merged = merged.sort_values(["_symbol", "_ts"])
    merged["pit_covered"] = (merged["q25"] <= merged["y_true"]) & (
        merged["y_true"] <= merged["q75"]
    )

    adjustments: list[pd.Series] = []
    for _symbol, group in merged.groupby("_symbol", sort=False):
        coverage = float(group["pit_covered"].mean())
        pit_deviation = abs(coverage - 0.5)
        if pit_deviation <= pit_deviation_threshold:
            continue

        errors = np.abs(group["y_true"].to_numpy() - group["q50"].to_numpy())
        if errors.size == 0:
            continue

        window_size = max(1, min(window, errors.size))
        delta_series = (
            pd.Series(errors, index=group["_row"])
            .rolling(window=window_size, min_periods=1)
            .quantile(0.25)
            .dropna()
        )
        if delta_series.empty:
            continue
        adjustments.append(delta_series)

    if not adjustments:
        return forecasts.copy()

    combined_delta = pd.concat(adjustments).groupby(level=0).last()
    if combined_delta.empty:
        return forecasts.copy()

    result = forecasts.copy()
    result.loc[combined_delta.index, "q25"] = (
        result.loc[combined_delta.index, "q50"] - combined_delta
    )
    result.loc[combined_delta.index, "q75"] = (
        result.loc[combined_delta.index, "q50"] + combined_delta
    )

    return result


def widen_intervals(
    q25: np.ndarray, q50: np.ndarray, q75: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply post-hoc quantile widening by factor alpha."""
    # Widen intervals by multiplying the half-width by (1 + alpha)
    half_width = (q75 - q25) / 2
    widened_half_width = half_width * (1 + alpha)
    new_q25 = q50 - widened_half_width
    new_q75 = q50 + widened_half_width
    return new_q25, q50, new_q75


def split_conformal(
    residuals: np.ndarray, quantiles: np.ndarray, alpha: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Apply simple split-conformal prediction for quantiles."""
    if len(residuals) < 2:
        # Not enough data, return original quantiles
        return quantiles, quantiles

    # Sort residuals
    sorted_residuals = np.sort(np.abs(residuals))

    # For alpha coverage, find the (1-alpha) quantile of absolute residuals
    n = len(sorted_residuals)
    k = int(np.ceil((1 - alpha) * n))
    if k >= n:
        k = n - 1
    if k < 0:
        k = 0
    conformal_width = sorted_residuals[k]

    # Apply to quantiles: widen by conformal_width
    q25_adjusted = (
        quantiles[:, 0] - conformal_width
    )  # Assuming quantiles is (n, 3) for q25, q50, q75
    q75_adjusted = quantiles[:, 2] + conformal_width
    q50_adjusted = quantiles[:, 1]  # q50 unchanged

    return np.column_stack([q25_adjusted, q50_adjusted, q75_adjusted]), np.full(
        len(quantiles), conformal_width
    )


def generate_coverage_report(forecasts: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Generate coverage report per symbol and snapshot."""
    if forecasts.empty or actuals.empty:
        return pd.DataFrame()

    symbol_col = "symbol" if "symbol" in forecasts.columns else "unique_id"
    ts_col = "ts_utc" if "ts_utc" in forecasts.columns else "forecast_ts"

    merged = forecasts.merge(
        actuals, left_on=[symbol_col, ts_col], right_on=[symbol_col, ts_col], how="inner"
    )

    if merged.empty:
        return pd.DataFrame()

    # Group by symbol and snapshot
    report_rows = []
    for (symbol, snapshot), group in merged.groupby([symbol_col, "snapshot_utc"]):
        if len(group) == 0:
            continue

        y_true = group["y_true"].values
        q25 = group["q25"].values
        q75 = group["q75"].values

        coverage = np.mean((y_true >= q25) & (y_true <= q75))
        count = len(group)

        report_rows.append(
            {
                "symbol": symbol,
                "snapshot_utc": snapshot,
                "coverage": coverage,
                "count": count,
            }
        )

    return pd.DataFrame(report_rows)


__all__ = [
    "reliability_curve",
    "CalibrationConfig",
    "AffineCalibration",
    "CalibrationModel",
    "ForecastCalibrator",
    "compute_embargo_cutoff",
    "filter_calibration_window",
    "enforce_quantile_monotonicity",
    "apply_conformal_widening",
]
