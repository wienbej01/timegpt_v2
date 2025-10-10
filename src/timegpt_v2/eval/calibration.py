from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
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
    
    @classmethod
    def from_mapping(cls, payload: Mapping[str, object] | None) -> CalibrationConfig:
        if payload is None:
            return cls()
        
        method = str(payload.get("method", cls.method)).lower()
        if method not in {"affine", "isotonic", "none"}:
            raise ValueError("calibration.method must be one of: affine, isotonic, none")
        
        min_samples = int(payload.get("min_samples", cls.min_samples))
        calibration_window_days = int(payload.get("calibration_window_days", cls.calibration_window_days))
        model_path = str(payload.get("model_path", cls.model_path))
        
        return cls(
            method=method,
            min_samples=min_samples,
            calibration_window_days=calibration_window_days,
            model_path=model_path,
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
            },
            "affine_models": {}
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
        
        config = CalibrationConfig(
            method=data["config"]["method"],
            min_samples=data["config"]["min_samples"],
            calibration_window_days=data["config"]["calibration_window_days"],
            model_path=data["config"]["model_path"],
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


class ForecastCalibrator:
    """Calibrates forecast quantiles using per-symbol affine or isotonic regression."""
    
    def __init__(self, config: CalibrationConfig) -> None:
        self._config = config
        self._model = CalibrationModel.load(config.model_path)
    
    def fit(self, forecasts: pd.DataFrame, actuals: pd.DataFrame) -> None:
        """Fit calibration models using historical forecasts and actuals."""
        if self._config.method == "none":
            return
        
        # Ensure forecasts and actuals are aligned
        merged = forecasts.merge(
            actuals, 
            on=["symbol", "ts_utc"], 
            how="inner",
            suffixes=("", "_actual")
        )
        
        if merged.empty:
            raise ValueError("No overlapping data between forecasts and actuals")
        
        # Get quantile columns
        quantile_cols = [col for col in forecasts.columns if col.startswith("q")]
        
        for symbol, group in merged.groupby("symbol"):
            if len(group) < self._config.min_samples:
                continue
            
            y_true = group["y_true"].values
            
            for quantile_col in quantile_cols:
                y_pred = group[quantile_col].values
                
                # Remove NaN values
                valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                if valid_mask.sum() < self._config.min_samples:
                    continue
                
                y_true_valid = y_true[valid_mask]
                y_pred_valid = y_pred[valid_mask]
                
                if self._config.method == "affine":
                    self._fit_affine(symbol, quantile_col, y_true_valid, y_pred_valid)
                elif self._config.method == "isotonic":
                    self._fit_isotonic(symbol, quantile_col, y_true_valid, y_pred_valid)
    
    def _fit_affine(self, symbol: str, quantile_col: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
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
    
    def _fit_isotonic(self, symbol: str, quantile_col: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Fit isotonic regression calibration for a specific symbol and quantile."""
        # Isotonic regression to ensure monotonicity
        ir = IsotonicRegression(out_of_bounds='clip')
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
        
        for symbol, group in calibrated.groupby("symbol"):
            for quantile_col in quantile_cols:
                if self._config.method == "affine":
                    if (symbol in self._model.affine_models and 
                        quantile_col in self._model.affine_models[symbol]):
                        model = self._model.affine_models[symbol][quantile_col]
                        mask = calibrated["symbol"] == symbol
                        calibrated.loc[mask, quantile_col] = model.apply(
                            calibrated.loc[mask, quantile_col].values
                        )
                
                elif self._config.method == "isotonic":
                    if (symbol in self._model.isotonic_models and 
                        quantile_col in self._model.isotonic_models[symbol]):
                        model = self._model.isotonic_models[symbol][quantile_col]
                        mask = calibrated["symbol"] == symbol
                        calibrated.loc[mask, quantile_col] = model.predict(
                            calibrated.loc[mask, quantile_col].values
                        )
        
        return calibrated
    
    def save(self) -> None:
        """Save calibration models to disk."""
        self._model.save(self._config.model_path)
    
    def get_calibration_stats(self) -> pd.DataFrame:
        """Get statistics about fitted calibration models."""
        rows = []
        
        for symbol, quantile_models in self._model.affine_models.items():
            for quantile, model in quantile_models.items():
                rows.append({
                    "symbol": symbol,
                    "quantile": quantile,
                    "method": "affine",
                    "slope": model.slope,
                    "intercept": model.intercept,
                    "n_samples": model.n_samples,
                    "last_updated": model.last_updated,
                })
        
        for symbol, quantile_models in self._model.isotonic_models.items():
            for quantile in quantile_models.keys():
                rows.append({
                    "symbol": symbol,
                    "quantile": quantile,
                    "method": "isotonic",
                    "slope": np.nan,
                    "intercept": np.nan,
                    "n_samples": np.nan,
                    "last_updated": np.nan,
                })
        
        return pd.DataFrame(rows)


__all__ = [
    "reliability_curve",
    "CalibrationConfig",
    "AffineCalibration",
    "CalibrationModel",
    "ForecastCalibrator",
]