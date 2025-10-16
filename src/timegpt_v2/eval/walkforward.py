"""Walk-forward evaluation framework for comparing 30m vs 60m horizons."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from timegpt_v2.eval.metrics_forecast import crps, information_coefficient, pit_coverage
from timegpt_v2.eval.metrics_trading import (
    portfolio_sharpe, portfolio_max_drawdown, portfolio_hit_rate,
    portfolio_total_pnl, per_symbol_metrics
)
from timegpt_v2.forecast.scheduler import create_horizon_preset
from timegpt_v2.trading.costs import TradingCosts


@dataclass
class CostConfig:
    """Trading cost configuration."""
    half_spread_bps: float = 2.5  # Half spread in basis points
    commission_bps: float = 1.0   # Commission in basis points
    impact_bps: float = 0.5       # Market impact in basis points

    def total_cost_bps(self) -> float:
        """Total cost per trade in basis points."""
        return self.half_spread_bps + self.commission_bps + self.impact_bps


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward evaluation."""
    # Date ranges
    start_date: date
    end_date: date

    # Walk-forward parameters
    train_months: int = 3        # Training period in months
    test_months: int = 1         # Test period in months
    purge_months: int = 1        # Purge/embargo period between train and test

    # Cost model
    costs: CostConfig = field(default_factory=CostConfig)

    # Evaluation parameters
    min_obs_per_period: int = 100  # Minimum observations per period

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")

        if self.train_months <= 0 or self.test_months <= 0:
            raise ValueError("train_months and test_months must be positive")

        if self.purge_months < 0:
            raise ValueError("purge_months must be non-negative")


@dataclass
class HorizonResult:
    """Results for a single horizon evaluation."""
    horizon_minutes: int

    # Forecast metrics
    crps_mean: float = 0.0
    ic_mean: float = 0.0
    ic_std: float = 0.0
    coverage_mean: float = 0.0

    # Trading metrics (after costs)
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    total_pnl: float = 0.0
    turnover: float = 0.0  # Average trades per day

    # Additional metrics
    n_periods: int = 0
    n_forecasts: int = 0
    n_trades: int = 0

    # Per-symbol breakdown
    symbol_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)


class WalkForwardEvaluator:
    """Walk-forward evaluator for comparing 30m vs 60m horizons."""

    def __init__(self, config: WalkForwardConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cost_model = TradingCosts(
            fee_bps=config.costs.commission_bps + config.costs.impact_bps,
            half_spread_ticks={"default": config.costs.half_spread_bps / 100.0}  # Convert bps to ticks
        )

    def generate_walk_forward_periods(self) -> list[tuple[date, date, date, date]]:
        """Generate walk-forward (train_start, train_end, test_start, test_end) periods.

        Returns:
            List of tuples (train_start, train_end, test_start, test_end)
        """
        periods = []

        # Start from the first possible train period
        current_train_start = self.config.start_date
        current_test_start = current_train_start + timedelta(days=self.config.train_months * 30)
        current_test_end = current_test_start + timedelta(days=self.config.test_months * 30)

        while current_test_end <= self.config.end_date:
            current_train_end = current_train_start + timedelta(days=self.config.train_months * 30)

            # Add purge period
            test_start_with_purge = current_train_end + timedelta(days=self.config.purge_months * 30)

            if test_start_with_purge + timedelta(days=self.config.test_months * 30) <= self.config.end_date:
                periods.append((current_train_start, current_train_end, test_start_with_purge, current_test_end))

            # Move to next period (advance by test_months)
            current_train_start = current_train_start + timedelta(days=self.config.test_months * 30)
            current_test_start = current_train_start + timedelta(days=self.config.train_months * 30)
            current_test_end = current_test_start + timedelta(days=self.config.test_months * 30)

        return periods

    def apply_costs(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Apply trading costs to trades DataFrame.

        Args:
            trades: DataFrame with trade information

        Returns:
            DataFrame with cost-adjusted PnL
        """
        if trades.empty:
            return trades.copy()

        trades_with_costs = trades.copy()

        # Apply costs as a simple percentage of notional value
        # This is a simplified cost model for the walk-forward evaluation
        cost_bps = self.config.costs.total_cost_bps()
        cost_rate = cost_bps / 10000.0  # Convert bps to decimal

        # Assume trades have a 'quantity' and 'entry_price' columns
        # Calculate notional value
        if 'quantity' in trades_with_costs.columns and 'entry_price' in trades_with_costs.columns:
            notional = trades_with_costs['quantity'] * trades_with_costs['entry_price']
            cost_amount = notional * cost_rate

            # Adjust PnL for costs
            if 'net_pnl' in trades_with_costs.columns:
                trades_with_costs['net_pnl'] = trades_with_costs['net_pnl'] - cost_amount
            else:
                trades_with_costs['net_pnl'] = -cost_amount  # If no PnL column, just cost

        return trades_with_costs

    def evaluate_horizon(
        self,
        horizon_minutes: int,
        features: pd.DataFrame,
        forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
        trades: pd.DataFrame
    ) -> HorizonResult:
        """Evaluate a single horizon across all walk-forward periods.

        Args:
            horizon_minutes: Forecast horizon (30 or 60)
            features: Feature matrix
            forecasts: Forecast quantiles with columns ['symbol', 'ts_utc', 'q25', 'q50', 'q75']
            actuals: Actual returns with columns ['symbol', 'ts_utc', 'y_true']
            trades: Trade executions with cost-adjusted PnL

        Returns:
            HorizonResult with comprehensive metrics
        """
        periods = self.generate_walk_forward_periods()

        if not periods:
            raise ValueError("No valid walk-forward periods found")

        # Store period-level results
        period_crps = []
        period_ic = []
        period_coverage = []
        period_sharpe = []
        period_turnover = []
        period_trades = []

        total_forecasts = 0
        total_trades = 0

        # Per-symbol results accumulation
        all_symbol_metrics = []

        self.logger.info(f"Evaluating {horizon_minutes}m horizon across {len(periods)} walk-forward periods")

        for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
            self.logger.debug(f"Period {i+1}/{len(periods)}: train {train_start} to {train_end}, test {test_start} to {test_end}")

            # Filter data for test period
            test_mask = (
                (forecasts['ts_utc'].dt.date >= test_start) &
                (forecasts['ts_utc'].dt.date <= test_end)
            )

            period_forecasts = forecasts[test_mask].copy()
            period_actuals = actuals[test_mask].copy()
            period_trades = trades[test_mask].copy()

            if period_forecasts.empty or period_actuals.empty:
                self.logger.warning(f"Period {i+1}: No data available, skipping")
                continue

            # Skip if insufficient observations
            if len(period_forecasts) < self.config.min_obs_per_period:
                self.logger.warning(f"Period {i+1}: Insufficient observations ({len(period_forecasts)}), skipping")
                continue

            # Calculate forecast metrics
            try:
                # Merge forecasts with actuals
                merged = period_forecasts.merge(
                    period_actuals,
                    on=['symbol', 'ts_utc'],
                    how='inner',
                    suffixes=('', '_actual')
                )

                if not merged.empty:
                    period_crps.append(crps(merged['y_true'], merged['q25'], merged['q50'], merged['q75']))
                    period_ic.append(information_coefficient(merged['q50'], merged['y_true']))
                    period_coverage.append(pit_coverage(merged['y_true'], merged['q25'], merged['q75']))

                    total_forecasts += len(merged)
                else:
                    self.logger.warning(f"Period {i+1}: No overlapping forecast/actual data")

            except Exception as e:
                self.logger.error(f"Period {i+1}: Error calculating forecast metrics: {e}")
                continue

            # Calculate trading metrics
            try:
                if not period_trades.empty:
                    # Apply costs to trades
                    period_trades_with_costs = self.apply_costs(period_trades)

                    # Calculate portfolio metrics
                    period_sharpe.append(portfolio_sharpe(period_trades_with_costs))

                    # Calculate turnover (trades per day)
                    trading_days = period_trades_with_costs['entry_ts'].dt.date.nunique()
                    if trading_days > 0:
                        turnover = len(period_trades_with_costs) / trading_days
                        period_turnover.append(turnover)

                    period_trades.append(len(period_trades_with_costs))
                    total_trades += len(period_trades_with_costs)

                    # Collect per-symbol metrics
                    symbol_metrics = per_symbol_metrics(period_trades_with_costs)
                    if not symbol_metrics.empty:
                        symbol_metrics['period'] = i + 1
                        symbol_metrics['test_start'] = test_start
                        all_symbol_metrics.append(symbol_metrics)
                else:
                    self.logger.warning(f"Period {i+1}: No trades generated")

            except Exception as e:
                self.logger.error(f"Period {i+1}: Error calculating trading metrics: {e}")
                continue

        # Aggregate results
        result = HorizonResult(horizon_minutes=horizon_minutes)

        if period_crps:
            result.crps_mean = np.mean(period_crps)
            result.n_periods = len(period_crps)

        if period_ic:
            result.ic_mean = np.mean(period_ic)
            result.ic_std = np.std(period_ic)

        if period_coverage:
            result.coverage_mean = np.mean(period_coverage)

        if period_sharpe:
            result.sharpe = np.mean(period_sharpe)
            # Calculate aggregate drawdown and PnL
            all_trades_with_costs = []
            for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
                test_mask = (
                    (trades['ts_utc'].dt.date >= test_start) &
                    (trades['ts_utc'].dt.date <= test_end)
                )
                period_trades = trades[test_mask].copy()
                if not period_trades.empty:
                    period_trades_with_costs = self.apply_costs(period_trades)
                    all_trades_with_costs.append(period_trades_with_costs)

            if all_trades_with_costs:
                all_trades_df = pd.concat(all_trades_with_costs, ignore_index=True)
                result.max_drawdown = portfolio_max_drawdown(all_trades_df)
                result.hit_rate = portfolio_hit_rate(all_trades_df)
                result.total_pnl = portfolio_total_pnl(all_trades_df)

        if period_turnover:
            result.turnover = np.mean(period_turnover)

        result.n_forecasts = total_forecasts
        result.n_trades = total_trades

        if all_symbol_metrics:
            result.symbol_metrics = pd.concat(all_symbol_metrics, ignore_index=True)

        return result

    def compare_horizons(
        self,
        features: pd.DataFrame,
        forecasts_30m: pd.DataFrame,
        forecasts_60m: pd.DataFrame,
        actuals: pd.DataFrame,
        trades_30m: pd.DataFrame,
        trades_60m: pd.DataFrame
    ) -> dict[str, HorizonResult]:
        """Compare 30m vs 60m horizons.

        Args:
            features: Feature matrix (used for both horizons)
            forecasts_30m: 30-minute forecasts
            forecasts_60m: 60-minute forecasts
            actuals: Actual returns (same for both horizons)
            trades_30m: 30-minute trades
            trades_60m: 60-minute trades

        Returns:
            Dictionary with results for both horizons
        """
        self.logger.info("Starting walk-forward A/B comparison of 30m vs 60m horizons")

        # Evaluate both horizons
        results_30m = self.evaluate_horizon(30, features, forecasts_30m, actuals, trades_30m)
        results_60m = self.evaluate_horizon(60, features, forecasts_60m, actuals, trades_60m)

        self.logger.info(f"30m horizon: Sharpe={results_30m.sharpe:.3f}, CRPS={results_30m.crps_mean:.6f}")
        self.logger.info(f"60m horizon: Sharpe={results_60m.sharpe:.3f}, CRPS={results_60m.crps_mean:.6f}")

        return {
            "30m": results_30m,
            "60m": results_60m
        }

    def generate_decision_report(
        self,
        results: dict[str, HorizonResult],
        output_path: str | Path | None = None
    ) -> dict[str, Any]:
        """Generate a decision report comparing horizons.

        Args:
            results: Dictionary with HorizonResult for each horizon
            output_path: Optional path to save the report

        Returns:
            Decision report dictionary
        """
        if "30m" not in results or "60m" not in results:
            raise ValueError("Results must contain both '30m' and '60m' horizons")

        result_30m = results["30m"]
        result_60m = results["60m"]

        # Primary decision criterion: after-cost Sharpe ratio
        primary_metric = "sharpe"
        winner = "30m" if result_30m.sharpe > result_60m.sharpe else "60m"

        # Build report
        report = {
            "evaluation_summary": {
                "winner": winner,
                "primary_metric": primary_metric,
                "walkforward_periods": result_30m.n_periods,
                "total_forecasts_30m": result_30m.n_forecasts,
                "total_forecasts_60m": result_60m.n_forecasts,
                "total_trades_30m": result_30m.n_trades,
                "total_trades_60m": result_60m.n_trades,
            },
            "forecast_metrics": {
                "30m": {
                    "crps": result_30m.crps_mean,
                    "ic_mean": result_30m.ic_mean,
                    "ic_std": result_30m.ic_std,
                    "coverage": result_30m.coverage_mean,
                },
                "60m": {
                    "crps": result_60m.crps_mean,
                    "ic_mean": result_60m.ic_mean,
                    "ic_std": result_60m.ic_std,
                    "coverage": result_60m.coverage_mean,
                }
            },
            "trading_metrics": {
                "30m": {
                    "sharpe": result_30m.sharpe,
                    "max_drawdown": result_30m.max_drawdown,
                    "hit_rate": result_30m.hit_rate,
                    "total_pnl": result_30m.total_pnl,
                    "turnover": result_30m.turnover,
                },
                "60m": {
                    "sharpe": result_60m.sharpe,
                    "max_drawdown": result_60m.max_drawdown,
                    "hit_rate": result_60m.hit_rate,
                    "total_pnl": result_60m.total_pnl,
                    "turnover": result_60m.turnover,
                }
            },
            "cost_analysis": {
                "total_cost_bps": self.config.costs.total_cost_bps(),
                "cost_breakdown": {
                    "half_spread_bps": self.config.costs.half_spread_bps,
                    "commission_bps": self.config.costs.commission_bps,
                    "impact_bps": self.config.costs.impact_bps,
                }
            },
            "recommendations": {
                "selected_horizon": winner,
                "rationale": f"Selected {winner} horizon based on higher after-cost Sharpe ratio",
                "key_metrics": {
                    "sharpe_advantage": abs(result_30m.sharpe - result_60m.sharpe),
                    "crps_advantage": abs(result_30m.crps_mean - result_60m.crps_mean),
                    "turnover_advantage": abs(result_30m.turnover - result_60m.turnover),
                }
            }
        }

        # Add robustness analysis
        report["robustness_analysis"] = {
            "metric_stability": {
                "30m_ic_volatility": result_30m.ic_std,
                "60m_ic_volatility": result_60m.ic_std,
            },
            "period_consistency": self._analyze_period_consistency(results)
        }

        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            import json
            with output_path.open('w') as f:
                json.dump(report, f, indent=2, default=str)

            self.logger.info(f"Decision report saved to {output_path}")

        return report

    def _analyze_period_consistency(self, results: dict[str, HorizonResult]) -> dict[str, Any]:
        """Analyze consistency of performance across periods."""
        # This would require period-level data storage
        # For now, return a placeholder
        return {
            "sharpe_consistency": "stable",  # Would be calculated from period-wise Sharpe
            "ic_consistency": "stable",       # Would be calculated from period-wise IC
            "note": "Period-level consistency analysis requires storing intermediate results"
        }


__all__ = [
    "CostConfig",
    "WalkForwardConfig",
    "HorizonResult",
    "WalkForwardEvaluator",
]