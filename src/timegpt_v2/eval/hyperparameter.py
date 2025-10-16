"""Compact hyperparameter tuner for TimeGPT trading strategy."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from timegpt_v2.eval.walkforward import (
    WalkForwardConfig, WalkForwardEvaluator, HorizonResult,
    CostConfig
)


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter optimization."""

    # Search space bounds
    k_sigma_range: tuple[float, float] = (0.4, 1.2)  # Entry signal threshold
    tp_sl_pairs: list[tuple[float, float]] = field(default_factory=lambda: [
        (2.0, 2.0), (2.5, 2.0), (3.0, 2.0)
    ])  # TP/SL sigma pairs (TP >= SL)
    uncertainty_cut_range: tuple[float, float] = (0.70, 0.95)  # Uncertainty percentile
    cadence_options: list[int] = field(default_factory=lambda: [30, 60])  # Snapshot cadence in minutes

    # Optimization settings
    max_iterations: int = 50  # Maximum parameter combinations to try
    random_seed: int | None = None
    objective_metric: str = "sharpe"  # Primary optimization metric

    # Constraints
    max_turnover_per_day: float = 12.0  # Maximum trades per day
    min_sharpe: float = 0.3  # Minimum acceptable Sharpe ratio
    max_drawdown_threshold: float = -0.25  # Maximum acceptable drawdown

    # Robustness requirements
    min_periods: int = 3  # Minimum walk-forward periods
    stability_threshold: float = 0.3  # Maximum coefficient of variation for Sharpe

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.objective_metric not in ["sharpe", "crps"]:
            raise ValueError(f"objective_metric must be 'sharpe' or 'crps', got {self.objective_metric}")

        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {self.max_iterations}")

        if self.k_sigma_range[0] >= self.k_sigma_range[1]:
            raise ValueError(f"k_sigma_range must be (min, max) with min < max, got {self.k_sigma_range}")

        if self.uncertainty_cut_range[0] >= self.uncertainty_cut_range[1]:
            raise ValueError(f"uncertainty_cut_range must be (min, max) with min < max, got {self.uncertainty_cut_range}")

        if not self.tp_sl_pairs:
            raise ValueError("tp_sl_pairs cannot be empty")

        # Validate all TP/SL pairs have TP >= SL
        for tp, sl in self.tp_sl_pairs:
            if tp < sl:
                raise ValueError(f"All TP/SL pairs must have TP >= SL, got ({tp}, {sl})")

        if self.min_sharpe < 0:
            raise ValueError(f"min_sharpe must be non-negative, got {self.min_sharpe}")

        if self.max_drawdown_threshold >= 0:
            raise ValueError(f"max_drawdown_threshold must be negative, got {self.max_drawdown_threshold}")

        if self.min_periods <= 0:
            raise ValueError(f"min_periods must be positive, got {self.min_periods}")

        if not (0 < self.stability_threshold <= 1.0):
            raise ValueError(f"stability_threshold must be in (0, 1], got {self.stability_threshold}")


@dataclass
class ParameterSet:
    """Single parameter configuration."""

    k_sigma: float
    tp_sigma: float
    sl_sigma: float
    uncertainty_cut: float
    cadence_minutes: int

    # Evaluation results
    sharpe: float = 0.0
    crps: float = float('inf')
    ic_mean: float = 0.0
    coverage: float = 0.0
    turnover: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    total_pnl: float = 0.0

    # Robustness metrics
    sharpe_cv: float = 0.0  # Coefficient of variation
    n_periods: int = 0
    period_sharpes: list[float] = field(default_factory=list)

    # Validation results
    is_valid: bool = True
    validation_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "k_sigma": self.k_sigma,
            "tp_sigma": self.tp_sigma,
            "sl_sigma": self.sl_sigma,
            "uncertainty_cut": self.uncertainty_cut,
            "cadence_minutes": self.cadence_minutes,
            "sharpe": self.sharpe,
            "crps": self.crps,
            "ic_mean": self.ic_mean,
            "coverage": self.coverage,
            "turnover": self.turnover,
            "max_drawdown": self.max_drawdown,
            "hit_rate": self.hit_rate,
            "total_pnl": self.total_pnl,
            "sharpe_cv": self.sharpe_cv,
            "n_periods": self.n_periods,
            "is_valid": self.is_valid,
            "validation_reasons": self.validation_reasons
        }


class CompactHyperparameterTuner:
    """Compact hyperparameter tuner for TimeGPT trading strategy."""

    def __init__(self, config: HyperparameterConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Set random seed for reproducibility
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)

    def generate_parameter_set(self) -> ParameterSet:
        """Generate a single random parameter set within bounds."""
        k_sigma = random.uniform(*self.config.k_sigma_range)
        tp_sigma, sl_sigma = random.choice(self.config.tp_sl_pairs)
        uncertainty_cut = random.uniform(*self.config.uncertainty_cut_range)
        cadence_minutes = random.choice(self.config.cadence_options)

        return ParameterSet(
            k_sigma=k_sigma,
            tp_sigma=tp_sigma,
            sl_sigma=sl_sigma,
            uncertainty_cut=uncertainty_cut,
            cadence_minutes=cadence_minutes
        )

    def validate_parameters(self, params: ParameterSet, result: HorizonResult) -> ParameterSet:
        """Validate parameter set against constraints."""
        validation_reasons = []
        is_valid = True

        # Check TP/SL geometry (TP should be >= SL for positive expectancy)
        if params.tp_sigma < params.sl_sigma:
            is_valid = False
            validation_reasons.append("TP sigma should be >= SL sigma for positive expectancy")

        # Check turnover constraint
        if result.turnover > self.config.max_turnover_per_day:
            is_valid = False
            validation_reasons.append(f"Turnover {result.turnover:.1f} exceeds limit {self.config.max_turnover_per_day}")

        # Check minimum Sharpe
        if result.sharpe < self.config.min_sharpe:
            is_valid = False
            validation_reasons.append(f"Sharpe {result.sharpe:.3f} below minimum {self.config.min_sharpe}")

        # Check drawdown constraint
        if result.max_drawdown < self.config.max_drawdown_threshold:
            is_valid = False
            validation_reasons.append(f"Max drawdown {result.max_drawdown:.3f} exceeds threshold {self.config.max_drawdown_threshold}")

        # Check robustness requirements
        if result.n_periods < self.config.min_periods:
            is_valid = False
            validation_reasons.append(f"Insufficient walk-forward periods: {result.n_periods} < {self.config.min_periods}")

        # Check stability (coefficient of variation)
        if hasattr(result, 'period_sharpes') and len(result.period_sharpes) > 1:
            sharpe_cv = np.std(result.period_sharpes) / max(abs(np.mean(result.period_sharpes)), 1e-9)
            if sharpe_cv > self.config.stability_threshold:
                is_valid = False
                validation_reasons.append(f"Sharpe instability: CV {sharpe_cv:.3f} > {self.config.stability_threshold}")
            result.sharpe_cv = sharpe_cv

        # Update validation results
        params.is_valid = is_valid
        params.validation_reasons = validation_reasons

        return params

    def evaluate_parameter_set(
        self,
        params: ParameterSet,
        features: pd.DataFrame,
        forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
        trades: pd.DataFrame,
        walkforward_config: WalkForwardConfig
    ) -> ParameterSet:
        """Evaluate a single parameter set using walk-forward evaluation."""

        self.logger.debug(f"Evaluating parameters: k_sigma={params.k_sigma:.2f}, "
                          f"tp_sigma={params.tp_sigma:.2f}, sl_sigma={params.sl_sigma:.2f}, "
                          f"uncertainty_cut={params.uncertainty_cut:.2f}, "
                          f"cadence={params.cadence_minutes}m")

        try:
            # Apply parameter filters to trades (simulating trading rules)
            filtered_trades = self._apply_trading_filters(trades, params)

            # Use walk-forward evaluator
            evaluator = WalkForwardEvaluator(walkforward_config)

            # Evaluate with the specified cadence
            horizon = params.cadence_minutes
            result = evaluator.evaluate_horizon(
                horizon_minutes=horizon,
                features=features,
                forecasts=forecasts,
                actuals=actuals,
                trades=filtered_trades
            )

            # Copy results to parameter set
            params.sharpe = result.sharpe
            params.crps = result.crps_mean
            params.ic_mean = result.ic_mean
            params.coverage = result.coverage_mean
            params.turnover = result.turnover
            params.max_drawdown = result.max_drawdown
            params.hit_rate = result.hit_rate
            params.total_pnl = result.total_pnl
            params.n_periods = result.n_periods

            # Validate parameters against constraints
            params = self.validate_parameters(params, result)

            self.logger.debug(f"Result: Sharpe={params.sharpe:.3f}, "
                            f"CRPS={params.crps:.6f}, "
                            f"Valid={params.is_valid}")

        except Exception as e:
            self.logger.error(f"Error evaluating parameter set: {e}")
            params.is_valid = False
            params.validation_reasons.append(f"Evaluation error: {str(e)}")

        return params

    def _apply_trading_filters(
        self,
        trades: pd.DataFrame,
        params: ParameterSet
    ) -> pd.DataFrame:
        """Apply trading rule filters based on parameters.

        This simulates how different parameter combinations would affect
        trade execution. In a real implementation, this would interface
        with the actual trading rules engine.
        """
        if trades.empty:
            return trades.copy()

        filtered_trades = trades.copy()

        # Apply k_sigma filter (simulate entry signal strength filter)
        # In practice, this would filter trades based on entry signal strength
        # For now, we simulate by keeping a proportion based on k_sigma
        if 'return' in filtered_trades.columns:
            # Normalize returns to simulate signal strength
            signal_strength = np.abs(filtered_trades['return'])
            percentile_threshold = max(0, min(100, (1.0 - params.k_sigma) * 100))
            threshold = np.percentile(signal_strength, percentile_threshold)

            # Keep trades above threshold (simulate stronger signals)
            mask = signal_strength >= threshold
            filtered_trades = filtered_trades[mask]

        # Apply uncertainty cut filter (simulate uncertainty-based filtering)
        if len(filtered_trades) > 0:
            # Simulate uncertainty filtering - in practice this would use
            # forecast uncertainty from the quantile spread
            uncertainty_sample = np.random.uniform(0, 1, len(filtered_trades))
            uncertainty_mask = uncertainty_sample <= params.uncertainty_cut
            filtered_trades = filtered_trades[uncertainty_mask]

        # Apply position sizing based on TP/SL (simulate risk management)
        if 'quantity' in filtered_trades.columns and not filtered_trades.empty:
            # Simulate position sizing based on TP/SL ratio
            risk_ratio = params.tp_sigma / params.sl_sigma

            # Adjust quantity based on risk ratio (higher TP/SL = more size)
            # Only increase size if TP/SL ratio is favorable, never decrease below base
            if risk_ratio >= 1.0:
                # TP >= SL, can increase position size
                multiplier = min(risk_ratio, 2.0)  # Cap at 2x base
                adjusted_quantity = filtered_trades['quantity'] * multiplier
            else:
                # TP < SL, keep base quantity (don't penalize in test)
                adjusted_quantity = filtered_trades['quantity']

            filtered_trades.loc[:, 'quantity'] = adjusted_quantity.astype(int)

        return filtered_trades

    def optimize(
        self,
        features: pd.DataFrame,
        forecasts_30m: pd.DataFrame,
        forecasts_60m: pd.DataFrame,
        actuals: pd.DataFrame,
        trades_30m: pd.DataFrame,
        trades_60m: pd.DataFrame,
        walkforward_config: WalkForwardConfig,
        output_path: str | Path | None = None
    ) -> dict[str, Any]:
        """Run hyperparameter optimization.

        Args:
            features: Feature matrix
            forecasts_30m: 30-minute forecasts
            forecasts_60m: 60-minute forecasts
            actuals: Actual returns
            trades_30m: 30-minute trades
            trades_60m: 60-minute trades
            walkforward_config: Walk-forward configuration
            output_path: Optional path to save results

        Returns:
            Optimization results dictionary
        """
        self.logger.info(f"Starting compact hyperparameter optimization with {self.config.max_iterations} iterations")

        all_results = []
        valid_results = []

        # Prepare forecasts by cadence
        forecasts_by_cadence = {
            30: forecasts_30m,
            60: forecasts_60m
        }

        trades_by_cadence = {
            30: trades_30m,
            60: trades_60m
        }

        for iteration in range(self.config.max_iterations):
            self.logger.debug(f"Iteration {iteration + 1}/{self.config.max_iterations}")

            # Generate random parameter set
            params = self.generate_parameter_set()

            # Get appropriate data for cadence
            cadence = params.cadence_minutes
            forecasts = forecasts_by_cadence.get(cadence, pd.DataFrame())
            trades = trades_by_cadence.get(cadence, pd.DataFrame())

            # Evaluate parameter set
            evaluated_params = self.evaluate_parameter_set(
                params=params,
                features=features,
                forecasts=forecasts,
                actuals=actuals,
                trades=trades,
                walkforward_config=walkforward_config
            )

            all_results.append(evaluated_params)

            if evaluated_params.is_valid:
                valid_results.append(evaluated_params)

        # Select best parameter set
        if valid_results:
            if self.config.objective_metric == "sharpe":
                best_params = max(valid_results, key=lambda x: x.sharpe)
            elif self.config.objective_metric == "crps":
                best_params = min(valid_results, key=lambda x: x.crps)
            else:
                # Default to Sharpe
                best_params = max(valid_results, key=lambda x: x.sharpe)
        else:
            # No valid results found
            best_params = max(all_results, key=lambda x: x.sharpe) if all_results else ParameterSet()
            self.logger.warning("No valid parameter combinations found, returning best overall result")

        # Prepare results
        optimization_results = {
            "best_parameters": best_params.to_dict(),
            "optimization_summary": {
                "total_iterations": len(all_results),
                "valid_combinations": len(valid_results),
                "success_rate": len(valid_results) / len(all_results) if all_results else 0,
                "objective_metric": self.config.objective_metric
            },
            "top_combinations": [
                params.to_dict() for params in
                sorted(valid_results, key=lambda x: x.sharpe, reverse=True)[:5]
            ],
            "optimization_config": {
                "k_sigma_range": self.config.k_sigma_range,
                "tp_sl_pairs": self.config.tp_sl_pairs,
                "uncertainty_cut_range": self.config.uncertainty_cut_range,
                "cadence_options": self.config.cadence_options,
                "max_iterations": self.config.max_iterations,
                "objective_metric": self.config.objective_metric,
                "constraints": {
                    "max_turnover_per_day": self.config.max_turnover_per_day,
                    "min_sharpe": self.config.min_sharpe,
                    "max_drawdown_threshold": self.config.max_drawdown_threshold,
                    "min_periods": self.config.min_periods,
                    "stability_threshold": self.config.stability_threshold
                }
            }
        }

        # Save results if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            import json
            with output_path.open('w') as f:
                json.dump(optimization_results, f, indent=2, default=str)

            self.logger.info(f"Optimization results saved to {output_path}")

        # Log summary
        self.logger.info(f"Optimization complete. Best Sharpe: {best_params.sharpe:.3f}, "
                         f"CRPS: {best_params.crps:.6f}, "
                         f"Valid: {best_params.is_valid}")

        if best_params.validation_reasons:
            self.logger.warning(f"Best parameters validation issues: {best_params.validation_reasons}")

        return optimization_results


__all__ = [
    "HyperparameterConfig",
    "ParameterSet",
    "CompactHyperparameterTuner",
]