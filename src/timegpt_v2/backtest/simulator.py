from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from timegpt_v2.trading.rules import TradingRules


@dataclass
class Trade:
    """Represents a single trade."""

    symbol: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    position: int


class BacktestSimulator:
    """Simulates a trading strategy based on forecasts and rules."""

    def __init__(self, rules: TradingRules, logger: logging.Logger) -> None:
        self.rules = rules
        self.logger = logger

    def run(
        self, forecasts: pd.DataFrame, features: pd.DataFrame, prices: pd.DataFrame
    ) -> tuple[list[Trade], pd.DataFrame]:
        """Run the backtest simulation."""
        trades: list[Trade] = []
        inventory: dict[str, int] = {}
        daily_trade_counts: dict[str, int] = {}

        for snapshot_ts_str in forecasts["snapshot_utc"].unique():
            snapshot_ts = pd.to_datetime(snapshot_ts_str, utc=True)
            forecast_group = forecasts[forecasts["snapshot_utc"] == snapshot_ts_str]
            for _, forecast in forecast_group.iterrows():
                symbol = forecast["symbol"]
                if inventory.get(symbol, 0) != 0:
                    continue

                trade_date = snapshot_ts.date()
                daily_trade_key = f"{symbol}_{trade_date}"
                if daily_trade_counts.get(daily_trade_key, 0) >= self.rules.daily_trade_cap:
                    continue

                if snapshot_ts not in prices.index:
                    continue

                price_at_snapshot = prices.loc[snapshot_ts, symbol]
                features_at_snapshot = features.loc[snapshot_ts, "rv_5m"]

                signal = self.rules.get_entry_signal(
                    q25=forecast["q25"],
                    q50=forecast["q50"],
                    q75=forecast["q75"],
                    last_price=price_at_snapshot,
                    sigma_5m=features_at_snapshot,
                    tick_size=0.01,  # TODO: Get from config
                    symbol=symbol,
                )

                if signal != 0:
                    entry_price = price_at_snapshot
                    inventory[symbol] = signal
                    daily_trade_counts[daily_trade_key] = (
                        daily_trade_counts.get(daily_trade_key, 0) + 1
                    )

                    exit_ts, exit_price = self._find_exit(
                        entry_price,
                        signal,
                        snapshot_ts,
                        prices[symbol],
                        features["rv_5m"],
                    )

                    trades.append(
                        Trade(
                            symbol=symbol,
                            entry_ts=snapshot_ts,
                            exit_ts=exit_ts,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            position=signal,
                        )
                    )
                    inventory[symbol] = 0

        summary = self._summarize_trades(trades)
        return trades, summary

    def _find_exit(
        self,
        entry_price: float,
        position: int,
        entry_ts: pd.Timestamp,
        price_series: pd.Series,
        feature_series: pd.Series,
    ) -> tuple[pd.Timestamp, float]:
        """Find the exit timestamp and price for a trade."""
        exit_ts = entry_ts + pd.Timedelta(minutes=1)
        while exit_ts < price_series.index[-1]:
            current_price = price_series.loc[exit_ts]
            if self.rules.get_exit_signal(
                entry_price=entry_price,
                current_price=current_price,
                position=position,
                sigma_5m=feature_series.loc[exit_ts],
                current_time=exit_ts.time(),
            ):
                return exit_ts, current_price
            exit_ts += pd.Timedelta(minutes=1)
        return exit_ts, price_series.loc[exit_ts]

    def _summarize_trades(self, trades: list[Trade]) -> pd.DataFrame:
        """Summarize the performance of the trades."""
        if not trades:
            return pd.DataFrame()

        trade_data = {
            "symbol": [t.symbol for t in trades],
            "pnl": [(t.exit_price - t.entry_price) * t.position for t in trades],
        }
        df = pd.DataFrame(trade_data)
        summary = {
            "total_pnl": df["pnl"].sum(),
            "hit_rate": (df["pnl"] > 0).mean(),
            "trade_count": len(df),
        }
        return pd.DataFrame([summary])
