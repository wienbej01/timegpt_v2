from __future__ import annotations

import hashlib
import itertools
import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from timegpt_v2.backtest.simulator import BacktestSimulator
from timegpt_v2.trading.costs import TradingCosts
from timegpt_v2.trading.rules import RuleParams, TradingRules


@dataclass(frozen=True)
class GridPoint:
    """Parameter combination for the sweep."""

    k_sigma: float
    s_stop: float
    s_take: float

    def to_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.md5(payload).hexdigest()  # noqa: S324


class GridSearch:
    """Perform parameter sweeps and persist per-point summaries."""

    def __init__(
        self,
        trading_cfg: Mapping[str, Any],
        logger: logging.Logger,
        output_root: Path,
        tick_size: float = 0.01,
    ) -> None:
        self.trading_cfg = trading_cfg
        self.logger = logger
        self.output_root = output_root
        self.tick_size = tick_size

    def run(
        self,
        forecasts: pd.DataFrame,
        features: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        results: list[dict[str, Any]] = []
        grid_points = self._get_grid_points()
        self.output_root.mkdir(parents=True, exist_ok=True)

        for point in grid_points:
            combo_hash = point.to_hash()
            seed = int(combo_hash[:16], 16) & 0x7FFFFFFF
            self.logger.info("sweep_run", extra={"grid_point": asdict(point), "hash": combo_hash})

            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            _ = rng.standard_normal(1)  # Ensure deterministic generator usage.

            trading_costs = TradingCosts(
                fee_bps=float(self.trading_cfg["fees_bps"]),
                half_spread_ticks=self.trading_cfg["half_spread_ticks"],
            )
            time_stop = datetime.strptime(str(self.trading_cfg["time_stop_et"]), "%H:%M").time()
            rules = TradingRules(
                costs=trading_costs,
                time_stop=time_stop,
                daily_trade_cap=int(self.trading_cfg["daily_trade_cap"]),
                max_open_per_symbol=int(self.trading_cfg["max_open_per_symbol"]),
            )
            params = RuleParams(
                k_sigma=float(point.k_sigma),
                s_stop=float(point.s_stop),
                s_take=float(point.s_take),
            )
            simulator = BacktestSimulator(
                rules=rules,
                params=params,
                logger=self.logger,
                tick_size=self.tick_size,
            )

            _trades, summary = simulator.run(
                forecasts.copy(deep=True),
                features.copy(deep=True),
                prices.copy(deep=True),
            )

            combo_dir = self.output_root / combo_hash
            combo_dir.mkdir(parents=True, exist_ok=True)
            summary_path = combo_dir / "bt_summary.csv"
            summary.to_csv(summary_path, index=False)

            row = summary.iloc[0].to_dict()
            row.update(
                {
                    "grid_point": json.dumps(asdict(point)),
                    "combo_hash": combo_hash,
                    "seed": seed,
                    "trade_count": int(summary.iloc[0]["trade_count"]),
                    "total_net_pnl": float(summary.iloc[0]["total_net_pnl"]),
                    "total_gross_pnl": float(summary.iloc[0]["total_gross_pnl"]),
                    "hit_rate": float(summary.iloc[0]["hit_rate"]),
                }
            )
            row["k_sigma"] = point.k_sigma
            row["s_stop"] = point.s_stop
            row["s_take"] = point.s_take
            results.append(row)

        if not results:
            return pd.DataFrame(
                columns=[
                    "grid_point",
                    "combo_hash",
                    "seed",
                    "trade_count",
                    "total_net_pnl",
                    "total_gross_pnl",
                    "hit_rate",
                ]
            )

        frame = pd.DataFrame(results)
        frame.sort_values("total_net_pnl", ascending=False, inplace=True)
        frame.reset_index(drop=True, inplace=True)
        frame["rank"] = frame.index + 1
        return frame

    def _get_grid_points(self) -> list[GridPoint]:
        param_lists = {
            "k_sigma": self.trading_cfg["k_sigma"],
            "s_stop": self.trading_cfg["s_stop"],
            "s_take": self.trading_cfg["s_take"],
        }
        keys, values = zip(*param_lists.items(), strict=True)
        return [
            GridPoint(**dict(zip(keys, combo, strict=True))) for combo in itertools.product(*values)
        ]
