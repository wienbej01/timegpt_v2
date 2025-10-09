from __future__ import annotations

import hashlib
import itertools
import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from timegpt_v2.backtest.simulator import BacktestSimulator
from timegpt_v2.trading.costs import TradingCosts
from timegpt_v2.trading.rules import TradingRules


@dataclass
class GridPoint:
    """A point in the parameter grid."""

    k_sigma: float
    s_stop: float
    s_take: float

    def to_hash(self) -> str:
        """Return a hash of the grid point."""
        return hashlib.md5(str(self).encode()).hexdigest()  # noqa: S324


class GridSearch:
    """Performs a grid search over trading parameters."""

    def __init__(self, trading_cfg: Mapping[str, Any], logger: logging.Logger) -> None:
        self.trading_cfg = trading_cfg
        self.logger = logger

    def run(
        self, forecasts: pd.DataFrame, features: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Run the grid search."""
        grid_points = self._get_grid_points()
        results = []

        for point in grid_points:
            self.logger.info(f"Running grid point: {point}")
            trading_costs = TradingCosts(
                fee_bps=self.trading_cfg["fees_bps"],
                half_spread_ticks=self.trading_cfg["half_spread_ticks"],
            )
            time_stop = datetime.strptime(self.trading_cfg["time_stop_et"], "%H:%M").time()
            trading_rules = TradingRules(
                costs=trading_costs,
                k_sigma=point.k_sigma,
                s_stop=point.s_stop,
                s_take=point.s_take,
                time_stop=time_stop,
                daily_trade_cap=self.trading_cfg["daily_trade_cap"],
                max_open_per_symbol=self.trading_cfg["max_open_per_symbol"],
            )
            simulator = BacktestSimulator(rules=trading_rules, logger=self.logger)
            _, summary = simulator.run(forecasts, features, prices)
            summary["grid_point"] = json.dumps(asdict(point))
            results.append(summary)

        return pd.concat(results, ignore_index=True)

    def _get_grid_points(self) -> list[GridPoint]:
        """Get the list of grid points to evaluate."""
        param_lists = {
            "k_sigma": self.trading_cfg["k_sigma"],
            "s_stop": self.trading_cfg["s_stop"],
            "s_take": self.trading_cfg["s_take"],
        }
        keys, values = zip(*param_lists.items(), strict=True)
        return [GridPoint(**dict(zip(keys, v, strict=True))) for v in itertools.product(*values)]
