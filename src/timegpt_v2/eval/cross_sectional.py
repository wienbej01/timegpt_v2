"""Cross-sectional trading strategy for multi-ticker dispersion harvesting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from timegpt_v2.eval.walkforward import (
    WalkForwardConfig, WalkForwardEvaluator, HorizonResult,
    CostConfig
)
from timegpt_v2.eval.metrics_forecast import information_coefficient
from timegpt_v2.eval.metrics_trading import (
    portfolio_sharpe, portfolio_max_drawdown, portfolio_hit_rate,
    portfolio_total_pnl, per_symbol_metrics
)


@dataclass
class CrossSectionalConfig:
    """Configuration for cross-sectional trading strategy."""

    # Portfolio construction
    top_decile: float = 0.1  # Top 10% for long positions
    bottom_decile: float = 0.1  # Bottom 10% for short positions
    min_symbols: int = 5  # Minimum symbols for cross-sectional strategy
    max_symbols: int = 50  # Maximum symbols to avoid over-diversification

    # Position sizing
    position_method: str = "equal_weight"  # "equal_weight" or "vol_scaled"
    notional_per_symbol: float = 10000.0  # Base notional per symbol
    max_position_weight: float = 0.2  # Maximum weight per symbol (20%)

    # Risk management
    max_leverage: float = 2.0  # Maximum gross leverage
    beta_neutral: bool = True  # Enforce beta neutrality
    sector_neutral: bool = False  # Enforce sector neutrality (if sector data available)

    # Execution constraints
    min_forecast_confidence: float = 0.7  # Minimum forecast confidence to include
    max_turnover_daily: float = 0.5  # Maximum daily turnover (50% of portfolio)

    # Performance targets
    target_leverage: float = 1.0  # Target gross leverage

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not (0 < self.top_decile <= 1.0):
            raise ValueError(f"top_decile must be in (0, 1], got {self.top_decile}")

        if not (0 < self.bottom_decile <= 1.0):
            raise ValueError(f"bottom_decile must be in (0, 1], got {self.bottom_decile}")

        if self.top_decile + self.bottom_decile > 0.8:
            raise ValueError(f"top_decile + bottom_decile must be ≤ 0.8, got {self.top_decile + self.bottom_decile}")

        if self.min_symbols >= self.max_symbols:
            raise ValueError(f"min_symbols must be < max_symbols, got {self.min_symbols} ≥ {self.max_symbols}")

        if self.position_method not in ["equal_weight", "vol_scaled"]:
            raise ValueError(f"position_method must be 'equal_weight' or 'vol_scaled', got {self.position_method}")

        if self.notional_per_symbol <= 0:
            raise ValueError(f"notional_per_symbol must be positive, got {self.notional_per_symbol}")

        if not (0 < self.max_position_weight <= 1.0):
            raise ValueError(f"max_position_weight must be in (0, 1], got {self.max_position_weight}")

        if self.max_leverage <= 0:
            raise ValueError(f"max_leverage must be positive, got {self.max_leverage}")

        if not (0 < self.min_forecast_confidence <= 1.0):
            raise ValueError(f"min_forecast_confidence must be in (0, 1], got {self.min_forecast_confidence}")


@dataclass
class CrossSectionalResult:
    """Results for cross-sectional strategy evaluation."""

    # Basic metrics
    horizon_minutes: int
    n_periods: int = 0
    n_symbols: int = 0
    n_forecasts: int = 0
    n_trades: int = 0

    # Cross-sectional metrics
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0  # Information Coefficient (IC) / std(IC)
    rank_ic: float = 0.0  # Rank IC

    # Portfolio metrics (long-short)
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    total_pnl: float = 0.0
    turnover: float = 0.0

    # Risk metrics
    beta: float = 0.0  # Portfolio beta
    leverage_gross: float = 0.0  # Gross leverage
    leverage_net: float = 0.0  # Net exposure

    # Per-symbol breakdown
    symbol_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Sector breakdown (if available)
    sector_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)


class CrossSectionalStrategy:
    """Cross-sectional trading strategy for multi-ticker dispersion harvesting."""

    def __init__(self, config: CrossSectionalConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

    def rank_symbols(
        self,
        forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
        timestamp: datetime
    ) -> pd.DataFrame:
        """Rank symbols by forecast returns for a given timestamp.

        Args:
            forecasts: DataFrame with forecast quantiles
            actuals: DataFrame with actual returns (for confidence calculation)
            timestamp: Timestamp for ranking

        Returns:
            DataFrame with ranked symbols and signals
        """
        # Filter forecasts for the timestamp
        mask = forecasts['ts_utc'] == timestamp
        if mask.sum() == 0:
            return pd.DataFrame()

        forecast_data = forecasts[mask].copy()

        # Ensure we have enough symbols
        if len(forecast_data) < self.config.min_symbols:
            self.logger.warning(f"Insufficient symbols at {timestamp}: {len(forecast_data)} < {self.config.min_symbols}")
            return pd.DataFrame()

        # Calculate forecast confidence as quantile spread
        forecast_data['confidence'] = (
            forecast_data['q75'] - forecast_data['q25']
        ) / abs(forecast_data['q50']).clip(lower=1e-6)

        # Filter by minimum confidence
        forecast_data = forecast_data[
            forecast_data['confidence'] >= self.config.min_forecast_confidence
        ]

        if len(forecast_data) < self.config.min_symbols:
            self.logger.warning(f"Insufficient confident symbols at {timestamp}: {len(forecast_data)} < {self.config.min_symbols}")
            return pd.DataFrame()

        # Rank by q50 forecast
        forecast_data['rank'] = forecast_data['q50'].rank(ascending=False, method='min')

        # Determine position direction (long top decile, short bottom decile)
        n_symbols = len(forecast_data)
        top_n = max(1, int(n_symbols * self.config.top_decile))
        bottom_n = max(1, int(n_symbols * self.config.bottom_decile))

        forecast_data['position'] = 0
        forecast_data.loc[forecast_data['rank'] <= top_n, 'position'] = 1  # Long
        forecast_data.loc[forecast_data['rank'] > (n_symbols - bottom_n), 'position'] = -1  # Short

        # Cap number of symbols
        selected_symbols = forecast_data[
            (forecast_data['position'] == 1) | (forecast_data['position'] == -1)
        ].copy()

        if len(selected_symbols) > self.config.max_symbols:
            # Keep top-ranked long and short symbols
            long_symbols = selected_symbols[selected_symbols['position'] == 1].nlargest(
                self.config.max_symbols // 2, 'q50'
            )
            short_symbols = selected_symbols[selected_symbols['position'] == -1].nsmallest(
                self.config.max_symbols // 2, 'q50'
            )
            selected_symbols = pd.concat([long_symbols, short_symbols], ignore_index=True)

        # Re-rank the selected symbols only (for test expectations)
        if not selected_symbols.empty:
            selected_symbols['rank'] = selected_symbols['q50'].rank(ascending=False, method='min')

        return selected_symbols

    def calculate_positions(
        self,
        ranked_symbols: pd.DataFrame,
        features: pd.DataFrame,
        timestamp: datetime
    ) -> pd.DataFrame:
        """Calculate position sizes based on ranking and volatility.

        Args:
            ranked_symbols: DataFrame with ranked symbols and signals
            features: Feature matrix with volatility data
            timestamp: Timestamp for position calculation

        Returns:
            DataFrame with position sizes
        """
        if ranked_symbols.empty:
            return pd.DataFrame()

        positions = ranked_symbols.copy()

        if self.config.position_method == "equal_weight":
            # Equal weight positions
            long_symbols = positions[positions['position'] == 1]
            short_symbols = positions[positions['position'] == -1]

            if not long_symbols.empty:
                long_weight = min(
                    self.config.max_position_weight,
                    1.0 / len(long_symbols)
                )
                positions.loc[positions['position'] == 1, 'weight'] = long_weight

            if not short_symbols.empty:
                short_weight = min(
                    self.config.max_position_weight,
                    1.0 / len(short_symbols)
                )
                positions.loc[positions['position'] == -1, 'weight'] = -short_weight

        elif self.config.position_method == "vol_scaled":
            # Volatility-scaled positions
            # This would require volatility data from features
            # For now, use equal weight as fallback
            self.logger.warning("Volatility scaling not implemented, using equal weight")
            # Recursively call with equal_weight method to avoid infinite recursion
            temp_config = CrossSectionalConfig(
                position_method="equal_weight",
                notional_per_symbol=self.config.notional_per_symbol,
                max_position_weight=self.config.max_position_weight,
                target_leverage=self.config.target_leverage
            )
            temp_strategy = CrossSectionalStrategy(temp_config)
            return temp_strategy.calculate_positions(ranked_symbols, features, timestamp)

        # Calculate notional positions
        positions['notional'] = positions['weight'] * self.config.notional_per_symbol

        # Ensure target leverage while respecting position weight limits
        gross_exposure = positions['notional'].abs().sum()
        if gross_exposure > 0:
            target_gross = self.config.target_leverage * self.config.notional_per_symbol
            leverage_factor = target_gross / gross_exposure

            # Apply leverage factor but don't exceed max_position_weight
            positions['notional'] *= leverage_factor
            positions['weight'] *= leverage_factor

            # Ensure no position exceeds max_position_weight
            max_weight_violation = positions['weight'].abs() > self.config.max_position_weight
            if max_weight_violation.any():
                # Scale down all weights proportionally
                max_current_weight = positions['weight'].abs().max()
                scaling_factor = self.config.max_position_weight / max_current_weight
                positions['weight'] *= scaling_factor
                positions['notional'] *= scaling_factor

        return positions

    def simulate_trades(
        self,
        positions: pd.DataFrame,
        actuals: pd.DataFrame,
        forecasts: pd.DataFrame,
        horizon_minutes: int,
        costs: CostConfig
    ) -> pd.DataFrame:
        """Simulate trade execution and P&L for cross-sectional strategy.

        Args:
            positions: DataFrame with position sizes
            actuals: DataFrame with actual returns
            forecasts: DataFrame with forecasts
            horizon_minutes: Forecast horizon in minutes
            costs: Trading cost configuration

        Returns:
            DataFrame with trade results
        """
        if positions.empty:
            return pd.DataFrame()

        trades = []

        for _, position in positions.iterrows():
            symbol = position['symbol']
            timestamp = position['ts_utc']
            weight = position['weight']
            notional = position['notional']

            # Find corresponding actual return
            actual_mask = (actuals['symbol'] == symbol) & (actuals['ts_utc'] == timestamp)
            if actual_mask.sum() == 0:
                continue

            actual_return = actuals[actual_mask]['y_true'].iloc[0]

            # Calculate P&L
            pnl = weight * actual_return

            # Apply costs (simplified)
            cost_rate = costs.total_cost_bps() / 10000.0
            cost_amount = abs(weight) * cost_rate
            net_pnl = pnl - cost_amount

            trades.append({
                'symbol': symbol,
                'ts_utc': timestamp,
                'position': weight,
                'notional': notional,
                'return': actual_return,
                'pnl': pnl,
                'cost': cost_amount,
                'net_pnl': net_pnl,
                'horizon_minutes': horizon_minutes
            })

        return pd.DataFrame(trades)

    def evaluate_cross_sectional(
        self,
        features: pd.DataFrame,
        forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
        walkforward_config: WalkForwardConfig
    ) -> CrossSectionalResult:
        """Evaluate cross-sectional strategy using walk-forward.

        Args:
            features: Feature matrix
            forecasts: Forecast quantiles
            actuals: Actual returns
            walkforward_config: Walk-forward configuration

        Returns:
            CrossSectionalResult with comprehensive metrics
        """
        self.logger.info("Starting cross-sectional strategy evaluation")

        # Use walk-forward evaluator for period management
        evaluator = WalkForwardEvaluator(walkforward_config)
        periods = evaluator.generate_walk_forward_periods()

        if not periods:
            raise ValueError("No valid walk-forward periods found")

        # Store period-level results
        period_results = []
        all_trades = []
        all_rankings = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
            self.logger.debug(f"Period {i+1}/{len(periods)}: test {test_start} to {test_end}")

            # Filter data for test period
            test_mask = (
                (forecasts['ts_utc'].dt.date >= test_start) &
                (forecasts['ts_utc'].dt.date <= test_end)
            )

            period_forecasts = forecasts[test_mask].copy()
            period_actuals = actuals[test_mask].copy()

            if period_forecasts.empty or period_actuals.empty:
                self.logger.warning(f"Period {i+1}: No data available, skipping")
                continue

            # Get unique timestamps for rebalancing
            timestamps = sorted(period_forecasts['ts_utc'].unique())

            period_trades = []
            period_rankings = []

            for timestamp in timestamps:
                # Rank symbols
                ranked_symbols = self.rank_symbols(period_forecasts, period_actuals, timestamp)
                if ranked_symbols.empty:
                    continue

                # Calculate positions
                positions = self.calculate_positions(ranked_symbols, features, timestamp)
                if positions.empty:
                    continue

                # Simulate trades
                trades = self.simulate_trades(
                    positions=positions,
                    actuals=period_actuals,
                    forecasts=period_forecasts,
                    horizon_minutes=30,  # Default horizon
                    costs=walkforward_config.costs
                )

                if not trades.empty:
                    period_trades.append(trades)

                # Store rankings for IC calculation
                period_rankings.append(ranked_symbols[['symbol', 'q50', 'rank']])

            if period_trades:
                period_trades_df = pd.concat(period_trades, ignore_index=True)
                all_trades.append(period_trades_df)

            if period_rankings:
                period_rankings_df = pd.concat(period_rankings, ignore_index=True)
                all_rankings.append(period_rankings_df)

        # Aggregate results
        result = CrossSectionalResult(horizon_minutes=30)  # Default horizon

        if all_trades:
            all_trades_df = pd.concat(all_trades, ignore_index=True)

            # Portfolio metrics
            result.sharpe = portfolio_sharpe(all_trades_df)
            result.max_drawdown = portfolio_max_drawdown(all_trades_df)
            result.hit_rate = portfolio_hit_rate(all_trades_df)
            result.total_pnl = portfolio_total_pnl(all_trades_df)

            # Turnover and leverage
            result.turnover = len(all_trades_df) / len(periods) if periods else 0
            result.leverage_gross = all_trades_df['notional'].abs().sum() / self.config.notional_per_symbol
            result.leverage_net = all_trades_df['notional'].sum() / self.config.notional_per_symbol

            # Per-symbol metrics
            result.symbol_metrics = per_symbol_metrics(all_trades_df)
            result.n_trades = len(all_trades_df)

        if all_rankings:
            all_rankings_df = pd.concat(all_rankings, ignore_index=True)

            # Calculate cross-sectional IC
            merged_ic = []
            for timestamp in all_rankings_df['ts_utc'].unique():
                timestamp_rankings = all_rankings_df[all_rankings_df['ts_utc'] == timestamp]
                timestamp_actuals = actuals[actuals['ts_utc'] == timestamp]

                if not timestamp_actuals.empty:
                    merged = timestamp_rankings.merge(
                        timestamp_actuals[['symbol', 'y_true']],
                        on='symbol',
                        how='inner'
                    )
                    if len(merged) >= 3:  # Need at least 3 symbols for meaningful correlation
                        ic = information_coefficient(merged['q50'], merged['y_true'])
                        merged_ic.append(ic)

            if merged_ic:
                result.ic_mean = np.mean(merged_ic)
                result.ic_std = np.std(merged_ic)
                result.ic_ir = result.ic_mean / result.ic_std if result.ic_std > 0 else 0

        result.n_periods = len([t for t in all_trades if t is not None])
        result.n_symbols = len(all_trades_df['symbol'].unique()) if all_trades else 0

        return result


__all__ = [
    "CrossSectionalConfig",
    "CrossSectionalResult",
    "CrossSectionalStrategy",
]