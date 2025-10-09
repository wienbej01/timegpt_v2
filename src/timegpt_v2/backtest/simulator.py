from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import NamedTuple, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from timegpt_v2.trading.rules import RuleParams, TradingRules

ET_ZONE = ZoneInfo("America/New_York")

_TRADE_COLUMNS = [
    "symbol",
    "entry_ts",
    "exit_ts",
    "entry_price",
    "exit_price",
    "position",
    "gross_pnl",
    "net_pnl",
    "holding_minutes",
    "costs_bp",
]


@dataclass(slots=True)
class Trade:
    """Represents a single executed trade."""

    symbol: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    position: int
    costs_bp: float

    def gross_pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.position

    def net_pnl(self) -> float:
        cost_per_side = (self.costs_bp / 10_000.0) * self.entry_price * abs(self.position)
        return self.gross_pnl() - 2.0 * cost_per_side

    def holding_minutes(self) -> float:
        delta = self.exit_ts - self.entry_ts
        return delta.total_seconds() / 60.0


class ForecastRow(NamedTuple):
    snapshot_ts: pd.Timestamp
    symbol: str
    q25: float
    q50: float
    q75: float


class BacktestSimulator:
    """Event-driven simulator that produces trade blotters and summaries."""

    def __init__(
        self,
        *,
        rules: TradingRules,
        params: RuleParams,
        logger: logging.Logger | None = None,
        tick_size: float = 0.01,
    ) -> None:
        self.rules = rules
        self.params = params
        self.tick_size = tick_size
        self.logger = logger or logging.getLogger(__name__)

    def run(
        self,
        forecasts: pd.DataFrame,
        features: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the backtest simulation."""
        forecast_rows = self._prepare_forecasts(forecasts)
        sigma_map = self._prepare_sigma(features)
        price_map = self._prepare_prices(prices)

        trades: list[Trade] = []
        open_positions: dict[str, list[pd.Timestamp]] = {}
        daily_trade_counts: dict[tuple[str, date], int] = {}

        for snapshot_ts_obj, snapshot_group in forecast_rows.groupby("snapshot_ts", sort=True):
            snapshot_ts = cast(pd.Timestamp, snapshot_ts_obj)

            for sym, exits in list(open_positions.items()):
                remaining = [ts for ts in exits if snapshot_ts < ts]
                if remaining:
                    open_positions[sym] = remaining
                else:
                    open_positions.pop(sym, None)

            for row_tuple in snapshot_group.itertuples(index=False, name="ForecastRow"):
                row = cast(ForecastRow, row_tuple)
                symbol = row.symbol
                if len(open_positions.get(symbol, [])) >= self.rules.max_open_per_symbol:
                    continue

                trade_date_key = (symbol, snapshot_ts.date())
                if daily_trade_counts.get(trade_date_key, 0) >= self.rules.daily_trade_cap:
                    continue

                price_series = price_map.get(symbol)
                if price_series is None or snapshot_ts not in price_series.index:
                    continue

                sigma_snapshot = sigma_map.get((symbol, snapshot_ts))
                if sigma_snapshot is None or np.isnan(sigma_snapshot) or sigma_snapshot <= 0:
                    continue

                last_price = float(price_series.loc[snapshot_ts])
                signal = self.rules.get_entry_signal(
                    self.params,
                    q25=row.q25,
                    q50=row.q50,
                    q75=row.q75,
                    last_price=last_price,
                    sigma_5m=sigma_snapshot,
                    tick_size=self.tick_size,
                    symbol=symbol,
                )
                if signal == 0:
                    continue

                exit_ts, exit_price = self._find_exit(
                    symbol=symbol,
                    entry_ts=snapshot_ts,
                    entry_price=last_price,
                    position=signal,
                    price_series=price_series,
                    sigma_map=sigma_map,
                )

                costs_bp = self.rules.costs.get_costs_bps(
                    symbol=symbol,
                    tick_size=self.tick_size,
                    price=last_price,
                )
                trade = Trade(
                    symbol=symbol,
                    entry_ts=snapshot_ts,
                    exit_ts=exit_ts,
                    entry_price=last_price,
                    exit_price=exit_price,
                    position=signal,
                    costs_bp=costs_bp,
                )
                trades.append(trade)
                open_positions.setdefault(symbol, []).append(exit_ts)
                daily_trade_counts[trade_date_key] = daily_trade_counts.get(trade_date_key, 0) + 1

        trades_df = self._trades_to_frame(trades)
        summary_df = self._summarize_trades(trades_df)
        return trades_df, summary_df

    def _prepare_forecasts(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        working = forecasts.copy()
        if "ts_utc" in working.columns and "snapshot_utc" not in working.columns:
            working = working.rename(columns={"ts_utc": "snapshot_utc"})
        working["snapshot_ts"] = pd.to_datetime(working["snapshot_utc"], utc=True)
        working["symbol"] = working["symbol"].astype(str)
        expected_cols = {"snapshot_ts", "symbol", "q25", "q50", "q75"}
        missing = expected_cols - set(working.columns)
        if missing:
            raise ValueError(f"Forecasts missing required columns: {sorted(missing)}")
        return working[["snapshot_ts", "symbol", "q25", "q50", "q75"]].sort_values(
            ["snapshot_ts", "symbol"]
        )

    def _prepare_sigma(self, features: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], float]:
        if features.empty:
            return {}
        working = features.copy()
        if "timestamp" not in working.columns:
            raise ValueError("Features must include a 'timestamp' column")
        working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
        working["symbol"] = working["symbol"].astype(str)
        if "rv_5m" not in working.columns:
            raise ValueError("Features must include 'rv_5m'")
        sigma = np.sqrt(working["rv_5m"].clip(lower=0))
        working = working.assign(sigma_5m=sigma)
        sigma_map: dict[tuple[str, pd.Timestamp], float] = {}
        symbols = working["symbol"].tolist()
        timestamps = working["timestamp"].tolist()
        sigma_vals = working["sigma_5m"].tolist()
        for symbol, ts, value in zip(symbols, timestamps, sigma_vals, strict=True):
            symbol_str = cast(str, symbol)
            ts_ts = cast(pd.Timestamp, ts)
            sigma_map[(symbol_str, ts_ts)] = float(value)
        return sigma_map

    def _prepare_prices(self, prices: pd.DataFrame) -> dict[str, pd.Series]:
        if {"timestamp", "symbol", "close"} - set(prices.columns):
            raise ValueError("Prices must include 'timestamp', 'symbol', and 'close'")
        working = prices.copy()
        working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
        working["symbol"] = working["symbol"].astype(str)
        price_map: dict[str, pd.Series] = {}
        for symbol, group in working.groupby("symbol", sort=False):
            symbol_str = cast(str, symbol)
            series = group.set_index("timestamp").sort_index()["close"].astype(float)
            price_map[symbol_str] = cast(pd.Series, series)
        return price_map

    def _find_exit(
        self,
        *,
        symbol: str,
        entry_ts: pd.Timestamp,
        entry_price: float,
        position: int,
        price_series: pd.Series,
        sigma_map: dict[tuple[str, pd.Timestamp], float],
    ) -> tuple[pd.Timestamp, float]:
        if entry_ts not in price_series.index:
            raise ValueError("Entry timestamp missing from price series")

        mask = price_series.index > entry_ts
        candidate_times = cast(pd.DatetimeIndex, price_series.index[mask])
        if candidate_times.empty:
            return entry_ts, float(price_series.loc[entry_ts])

        fallback_ts = cast(pd.Timestamp, candidate_times[-1])
        fallback_price = float(price_series.loc[fallback_ts])

        for ts_obj in candidate_times:
            ts = cast(pd.Timestamp, ts_obj)
            sigma = sigma_map.get((symbol, ts))
            if sigma is None or np.isnan(sigma):
                continue

            current_price = float(price_series.loc[ts])
            current_time_et = ts.tz_convert(ET_ZONE).time()
            if self.rules.get_exit_signal(
                self.params,
                entry_price=entry_price,
                current_price=current_price,
                position=position,
                sigma_5m=float(sigma),
                current_time=current_time_et,
            ):
                return ts, current_price

        return fallback_ts, fallback_price

    def _trades_to_frame(self, trades: Iterable[Trade]) -> pd.DataFrame:
        rows = []
        for trade in trades:
            rows.append(
                {
                    "symbol": trade.symbol,
                    "entry_ts": trade.entry_ts,
                    "exit_ts": trade.exit_ts,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "position": trade.position,
                    "gross_pnl": trade.gross_pnl(),
                    "net_pnl": trade.net_pnl(),
                    "holding_minutes": trade.holding_minutes(),
                    "costs_bp": trade.costs_bp,
                }
            )
        if not rows:
            return pd.DataFrame(columns=_TRADE_COLUMNS)
        frame = pd.DataFrame(rows)
        frame.sort_values(["entry_ts", "symbol"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        return frame

    def _summarize_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            payload = {
                "trade_count": 0,
                "total_gross_pnl": 0.0,
                "total_net_pnl": 0.0,
                "hit_rate": 0.0,
                "avg_hold_minutes": 0.0,
                "pnl_stdev": 0.0,
            }
            return pd.DataFrame([payload])

        net_pnl = trades["net_pnl"]
        gross_pnl = trades["gross_pnl"]
        payload = {
            "trade_count": int(len(trades)),
            "total_gross_pnl": float(gross_pnl.sum()),
            "total_net_pnl": float(net_pnl.sum()),
            "hit_rate": float((net_pnl > 0).mean()),
            "avg_hold_minutes": float(trades["holding_minutes"].mean()),
            "pnl_stdev": float(net_pnl.std(ddof=0)) if len(trades) > 1 else 0.0,
        }
        return pd.DataFrame([payload])
