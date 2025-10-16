from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import NamedTuple, cast, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from timegpt_v2.trading.rules import RuleParams, TradingRules
from timegpt_v2.trading.extended_rules import ExtendedRuleParams, ExtendedTradingRules

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
    position: float
    costs_bp: float

    def gross_pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.position

    def net_pnl(self) -> float:
        cost_per_side = (self.costs_bp / 10_000.0) * self.entry_price * abs(self.position)
        return self.gross_pnl() - 2.0 * cost_per_side

    def holding_minutes(self) -> float:
        delta = self.exit_ts - self.entry_ts
        return delta.total_seconds() / 60.0





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
        trading_window: Optional[object] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the backtest simulation."""
        forecast_rows = self._prepare_forecasts(forecasts)
        sigma_map = self._prepare_sigma(features)
        price_map = self._prepare_prices(prices)

        trades: list[Trade] = []
        open_positions: dict[str, list[pd.Timestamp]] = {}
        daily_trade_counts: dict[tuple[str, date], int] = {}
        c_total_rows = 0
        c_price_miss = 0
        c_sigma_miss = 0
        c_signal_zero = 0
        c_cap_limits = 0
        c_trades = 0
        c_window_violations = 0
        last_logged_date: date | None = None

        for snapshot_ts_obj, snapshot_group in forecast_rows.groupby("snapshot_ts", sort=True):
            snapshot_ts = cast(pd.Timestamp, snapshot_ts_obj)
            current_date = snapshot_ts.date()
            if current_date != last_logged_date:
                self.logger.info("SNAPSHOT %s (ET) — symbols=%d", snapshot_ts.tz_convert(ET_ZONE), len(snapshot_group))
                last_logged_date = current_date

            for sym, exits in list(open_positions.items()):
                remaining = [ts for ts in exits if snapshot_ts < ts]
                if remaining:
                    open_positions[sym] = remaining
                else:
                    open_positions.pop(sym, None)

            for row in snapshot_group.itertuples(index=False):
                c_total_rows += 1
                symbol = row.symbol

                # Apply trading window clamp if provided
                if trading_window is not None and hasattr(trading_window, 'start') and hasattr(trading_window, 'end'):
                    from timegpt_v2.utils.trading_window import is_date_in_trading_window
                    if not is_date_in_trading_window(current_date, trading_window):
                        c_window_violations += 1
                        continue

                if len(open_positions.get(symbol, [])) >= self.rules.max_open_per_symbol:
                    c_cap_limits += 1
                    continue

                trade_date_key = (symbol, snapshot_ts.date())
                if daily_trade_counts.get(trade_date_key, 0) >= self.rules.daily_trade_cap:
                    c_cap_limits += 1
                    continue

                price_series = price_map.get(symbol)
                if price_series is None or snapshot_ts not in price_series.index:
                    c_price_miss += 1
                    continue

                sigma_snapshot = sigma_map.get((symbol, snapshot_ts))
                if sigma_snapshot is None or np.isnan(sigma_snapshot) or sigma_snapshot <= 0:
                    c_sigma_miss += 1
                    continue

                last_price = float(price_series.loc[snapshot_ts])

                quantiles = {float(q.replace("q", ""))/100: getattr(row, q) for q in row._fields if q.startswith("q") and q[1:].isdigit()}
                signal = self.rules.get_entry_signal(
                    self.params,
                    quantiles=quantiles,
                    last_price=last_price,
                    sigma_5m=sigma_snapshot,
                    tick_size=self.tick_size,
                    symbol=symbol,
                )
                if signal == 0:
                    c_signal_zero += 1
                    continue

                exit_ts, exit_price = self._find_exit(
                    symbol=symbol,
                    entry_ts=snapshot_ts,
                    entry_price=last_price,
                    position=signal,
                    price_series=price_series,
                    sigma_map=sigma_map,
                )

                # Apply trading window clamp to exit as well
                if trading_window is not None and hasattr(trading_window, 'start') and hasattr(trading_window, 'end'):
                    from timegpt_v2.utils.trading_window import is_date_in_trading_window
                    if not is_date_in_trading_window(exit_ts.date(), trading_window):
                        # Extend exit to end of trading window if exit falls outside
                        if exit_ts.date() < trading_window.start:
                            exit_ts = pd.Timestamp(f"{trading_window.start.isoformat()}T{exit_ts.time():%H:%M:%S}").tz_convert(exit_ts.tzinfo)
                            exit_price = float(price_series.loc[exit_ts]) if exit_ts in price_series.index else exit_price
                        else:  # exit after trading window end
                            exit_ts = pd.Timestamp(f"{trading_window.end.isoformat()}T{exit_ts.time():%H:%M:%S}").tz_convert(exit_ts.tzinfo)
                            exit_price = float(price_series.loc[exit_ts]) if exit_ts in price_series.index else exit_price

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
                c_trades += 1
                open_positions.setdefault(symbol, []).append(exit_ts)
                daily_trade_counts[trade_date_key] = daily_trade_counts.get(trade_date_key, 0) + 1

        # Add window violations to logging if trading window is enforced
        if trading_window is not None and hasattr(trading_window, 'enforce_trading_window') and trading_window.enforce_trading_window:
            self.logger.info(
                "FUNNEL total=%d price_miss=%d sigma_miss=%d cap_limits=%d signal_zero=%d window_violations=%d trades=%d",
                c_total_rows, c_price_miss, c_sigma_miss, c_cap_limits, c_signal_zero, c_window_violations, c_trades
            )
        else:
            self.logger.info(
                "FUNNEL total=%d price_miss=%d sigma_miss=%d cap_limits=%d signal_zero=%d trades=%d",
                c_total_rows, c_price_miss, c_sigma_miss, c_cap_limits, c_signal_zero, c_trades
            )

        trades_df = self._trades_to_frame(trades)
        summary_df = self._summarize_trades(trades_df)
        return trades_df, summary_df

    def _prepare_forecasts(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        working = forecasts.copy()
        if "ts_utc" in working.columns and "snapshot_utc" not in working.columns:
            working = working.rename(columns={"ts_utc": "snapshot_utc"})
        working["snapshot_ts"] = pd.to_datetime(working["snapshot_utc"], utc=True)
        working["symbol"] = working["symbol"].astype(str)
        
        quantile_cols = [col for col in working.columns if col.startswith("q")]
        if not quantile_cols:
            raise ValueError("No quantile columns found in forecasts")

        expected_cols = {"snapshot_ts", "symbol"} | set(quantile_cols)
        missing = expected_cols - set(working.columns)
        if missing:
            raise ValueError(f"Forecasts missing required columns: {sorted(missing)}")
        return working[["snapshot_ts", "symbol", *quantile_cols]].sort_values(
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
        # Prefer realized variance if available
        if "rv_5m" in working.columns:
            sigma = np.sqrt(working["rv_5m"].clip(lower=0))
        elif "vol_ewm_15m" in working.columns:
            # Use ewm volatility (already a sigma proxy)
            sigma = working["vol_ewm_15m"].clip(lower=0)
            self.logger.info("Using vol_ewm_15m as sigma fallback")
        else:
            # Last-resort: derive from 1m returns (rolling std over 5 bars)
            if "ret_1m" not in working.columns:
                raise ValueError("Features must include 'rv_5m' or 'vol_ewm_15m' or 'ret_1m'")
            rolling = (
                working
                .sort_values(["symbol","timestamp"])
                .groupby("symbol")["ret_1m"].rolling(5, min_periods=3).std().reset_index(level=0, drop=True)
            )
            sigma = rolling.fillna(method="bfill").fillna(0.0).clip(lower=0)
            self.logger.info("Using rolling std of ret_1m as sigma fallback")
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
            series = (
                group.sort_values("timestamp")
                .groupby("timestamp", sort=True)["close"]
                .mean()
                .astype(float)
            )
            price_map[symbol_str] = cast(pd.Series, series)
        return price_map

    def _find_exit(
        self,
        *,
        symbol: str,
        entry_ts: pd.Timestamp,
        entry_price: float,
        position: float,
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


class ExtendedBacktestSimulator(BacktestSimulator):
    """Extended simulator with support for longer position durations."""

    def __init__(
        self,
        *,
        rules: ExtendedTradingRules,
        params: ExtendedRuleParams,
        logger: logging.Logger | None = None,
        tick_size: float = 0.01,
    ) -> None:
        # Initialize parent with compatible rules
        compat_rules = TradingRules(
            costs=rules.costs,
            time_stop=rules.time_stop,
            daily_trade_cap=rules.daily_trade_cap,
            max_open_per_symbol=rules.max_open_per_symbol,
        )
        compat_params = RuleParams(
            k_sigma=params.k_sigma,
            s_stop=params.s_stop,
            s_take=params.s_take,
            uncertainty_k=params.uncertainty_k,
        )
        super().__init__(rules=compat_rules, params=compat_params, logger=logger, tick_size=tick_size)
        self.extended_rules = rules
        self.extended_params = params

    def run(
        self,
        forecasts: pd.DataFrame,
        features: pd.DataFrame,
        prices: pd.DataFrame,
        trading_window: Optional[object] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the extended backtest simulation."""
        forecast_rows = self._prepare_forecasts(forecasts)
        sigma_map = self._prepare_extended_sigma(features, self.extended_params.volatility_window)
        price_map = self._prepare_prices(prices)

        trades: list[Trade] = []
        open_positions: dict[str, list[pd.Timestamp]] = {}
        daily_trade_counts: dict[tuple[str, date], int] = {}
        c_total_rows = 0
        c_price_miss = 0
        c_sigma_miss = 0
        c_signal_zero = 0
        c_cap_limits = 0
        c_trades = 0
        c_window_violations = 0
        c_min_hold_violations = 0
        last_logged_date: date | None = None

        for snapshot_ts_obj, snapshot_group in forecast_rows.groupby("snapshot_ts", sort=True):
            snapshot_ts = cast(pd.Timestamp, snapshot_ts_obj)
            current_date = snapshot_ts.date()
            if current_date != last_logged_date:
                self.logger.info("SNAPSHOT %s (ET) — symbols=%d", snapshot_ts.tz_convert(ET_ZONE), len(snapshot_group))
                last_logged_date = current_date

            # Check open positions for exits
            for sym, exits in list(open_positions.items()):
                remaining = [ts for ts in exits if snapshot_ts < ts]
                if remaining:
                    open_positions[sym] = remaining
                else:
                    open_positions.pop(sym, None)

            for row in snapshot_group.itertuples(index=False):
                c_total_rows += 1
                symbol = row.symbol

                # Apply trading window clamp if provided
                if trading_window is not None and hasattr(trading_window, 'start') and hasattr(trading_window, 'end'):
                    from timegpt_v2.utils.trading_window import is_date_in_trading_window
                    if not is_date_in_trading_window(current_date, trading_window):
                        c_window_violations += 1
                        continue

                if len(open_positions.get(symbol, [])) >= self.rules.max_open_per_symbol:
                    c_cap_limits += 1
                    continue

                trade_date_key = (symbol, snapshot_ts.date())
                if daily_trade_counts.get(trade_date_key, 0) >= self.rules.daily_trade_cap:
                    c_cap_limits += 1
                    continue

                price_series = price_map.get(symbol)
                if price_series is None or snapshot_ts not in price_series.index:
                    c_price_miss += 1
                    continue

                sigma_snapshot = sigma_map.get((symbol, snapshot_ts))
                if sigma_snapshot is None or np.isnan(sigma_snapshot) or sigma_snapshot <= 0:
                    c_sigma_miss += 1
                    continue

                last_price = float(price_series.loc[snapshot_ts])

                quantiles = {float(q.replace("q", ""))/100: getattr(row, q) for q in row._fields if q.startswith("q") and q[1:].isdigit()}
                signal = self.extended_rules.get_entry_signal(
                    self.extended_params,
                    quantiles=quantiles,
                    last_price=last_price,
                    sigma_5m=sigma_snapshot,
                    tick_size=self.tick_size,
                    symbol=symbol,
                )
                if signal == 0:
                    c_signal_zero += 1
                    continue

                exit_ts, exit_price, exit_reason = self._find_extended_exit(
                    symbol=symbol,
                    entry_ts=snapshot_ts,
                    entry_price=last_price,
                    position=signal,
                    price_series=price_series,
                    sigma_map=sigma_map,
                )

                # Check for minimum holding period violations
                holding_minutes = (exit_ts - snapshot_ts).total_seconds() / 60.0
                if holding_minutes < self.extended_params.min_holding_minutes:
                    c_min_hold_violations += 1
                    # Extend exit to minimum holding period
                    min_exit_ts = snapshot_ts + pd.Timedelta(minutes=self.extended_params.min_holding_minutes)
                    if min_exit_ts in price_series.index:
                        exit_ts = min_exit_ts
                        exit_price = float(price_series.loc[min_exit_ts])
                        holding_minutes = self.extended_params.min_holding_minutes
                        exit_reason = "extended_to_min_hold"

                # Apply trading window clamp to exit as well
                if trading_window is not None and hasattr(trading_window, 'start') and hasattr(trading_window, 'end'):
                    from timegpt_v2.utils.trading_window import is_date_in_trading_window
                    if not is_date_in_trading_window(exit_ts.date(), trading_window):
                        # Extend exit to end of trading window if exit falls outside
                        if exit_ts.date() < trading_window.start:
                            exit_ts = pd.Timestamp(f"{trading_window.start.isoformat()}T{exit_ts.time():%H:%M:%S}").tz_convert(exit_ts.tzinfo)
                            exit_price = float(price_series.loc[exit_ts]) if exit_ts in price_series.index else exit_price
                        else:  # exit after trading window end
                            exit_ts = pd.Timestamp(f"{trading_window.end.isoformat()}T{exit_ts.time():%H:%M:%S}").tz_convert(exit_ts.tzinfo)
                            exit_price = float(price_series.loc[exit_ts]) if exit_ts in price_series.index else exit_price

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
                c_trades += 1
                open_positions.setdefault(symbol, []).append(exit_ts)
                daily_trade_counts[trade_date_key] = daily_trade_counts.get(trade_date_key, 0) + 1

                # Log extended trade info
                self.logger.info(
                    "TRADE %s %s %s->%s hold=%.0fmin reason=%s pnl=%.4f",
                    symbol, snapshot_ts.tz_convert(ET_ZONE).strftime("%H:%M"),
                    "LONG" if signal > 0 else "SHORT",
                    exit_ts.tz_convert(ET_ZONE).strftime("%H:%M"),
                    holding_minutes, exit_reason, trade.net_pnl()
                )

        # Enhanced logging
        self.logger.info(
            "EXTENDED FUNNEL total=%d price_miss=%d sigma_miss=%d cap_limits=%d signal_zero=%d window_violations=%d min_hold_violations=%d trades=%d",
            c_total_rows, c_price_miss, c_sigma_miss, c_cap_limits, c_signal_zero, c_window_violations, c_min_hold_violations, c_trades
        )

        trades_df = self._trades_to_frame(trades)
        summary_df = self._summarize_extended_trades(trades_df)
        return trades_df, summary_df

    def _prepare_extended_sigma(self, features: pd.DataFrame, volatility_window: int) -> dict[tuple[str, pd.Timestamp], float]:
        """Prepare sigma using the specified volatility window."""
        if features.empty:
            return {}
        working = features.copy()
        if "timestamp" not in working.columns:
            raise ValueError("Features must include a 'timestamp' column")
        working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
        working["symbol"] = working["symbol"].astype(str)

        # Use the requested volatility window
        volatility_col_map = {
            5: "vol_ewm_15m",
            15: "vol_ewm_30m",
            30: "vol_ewm_60m",
            60: "vol_ewm_60m",
            90: "vol_ewm_90m"
        }

        volatility_col = volatility_col_map.get(volatility_window, "vol_ewm_30m")
        if volatility_col in working.columns:
            sigma = working[volatility_col].clip(lower=0)
            self.logger.info(f"Using {volatility_col} as sigma for {volatility_window}m window")
        elif "rv_5m" in working.columns:
            sigma = np.sqrt(working["rv_5m"].clip(lower=0))
            self.logger.info(f"Using rv_5m as sigma fallback for {volatility_window}m window")
        else:
            # Last-resort: derive from 1m returns
            if "ret_1m" not in working.columns:
                raise ValueError("Features must include volatility measures or ret_1m")
            rolling = (
                working
                .sort_values(["symbol","timestamp"])
                .groupby("symbol")["ret_1m"].rolling(volatility_window, min_periods=max(3, volatility_window//3)).std().reset_index(level=0, drop=True)
            )
            sigma = rolling.fillna(method="bfill").fillna(0.0).clip(lower=0)
            self.logger.info(f"Using rolling std of ret_1m with window={volatility_window} as sigma fallback")

        working = working.assign(sigma=sigma)
        sigma_map: dict[tuple[str, pd.Timestamp], float] = {}
        symbols = working["symbol"].tolist()
        timestamps = working["timestamp"].tolist()
        sigma_vals = working["sigma"].tolist()
        for symbol, ts, value in zip(symbols, timestamps, sigma_vals, strict=True):
            symbol_str = cast(str, symbol)
            ts_ts = cast(pd.Timestamp, ts)
            sigma_map[(symbol_str, ts_ts)] = float(value)
        return sigma_map

    def _find_extended_exit(
        self,
        *,
        symbol: str,
        entry_ts: pd.Timestamp,
        entry_price: float,
        position: float,
        price_series: pd.Series,
        sigma_map: dict[tuple[str, pd.Timestamp], float],
    ) -> tuple[pd.Timestamp, float, str]:
        """Find exit using extended duration logic."""
        if entry_ts not in price_series.index:
            raise ValueError("Entry timestamp missing from price series")

        mask = price_series.index > entry_ts
        candidate_times = cast(pd.DatetimeIndex, price_series.index[mask])
        if candidate_times.empty:
            return entry_ts, float(price_series.loc[entry_ts]), "no_candidates"

        entry_time_et = entry_ts.tz_convert(ET_ZONE).time()
        fallback_ts = cast(pd.Timestamp, candidate_times[-1])
        fallback_price = float(price_series.loc[fallback_ts])

        for ts_obj in candidate_times:
            ts = cast(pd.Timestamp, ts_obj)
            sigma = sigma_map.get((symbol, ts))
            if sigma is None or np.isnan(sigma):
                continue

            current_price = float(price_series.loc[ts])
            current_time_et = ts.tz_convert(ET_ZONE).time()
            holding_minutes = (ts - entry_ts).total_seconds() / 60.0

            should_exit, exit_reason = self.extended_rules.get_exit_signal(
                self.extended_params,
                entry_price=entry_price,
                current_price=current_price,
                position=position,
                sigma_5m=float(sigma),
                current_time=current_time_et,
                entry_time=entry_time_et,
                holding_minutes=holding_minutes,
            )
            if should_exit:
                return ts, current_price, exit_reason

        return fallback_ts, fallback_price, "max_horizon_reached"

    def _summarize_extended_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Summarize trades with extended duration metrics."""
        if trades.empty:
            payload = {
                "trade_count": 0,
                "total_gross_pnl": 0.0,
                "total_net_pnl": 0.0,
                "hit_rate": 0.0,
                "avg_hold_minutes": 0.0,
                "pnl_stdev": 0.0,
                "median_hold_minutes": 0.0,
                "min_hold_minutes": 0.0,
                "max_hold_minutes": 0.0,
                "trades_under_10min": 0,
                "trades_10_30min": 0,
                "trades_30_60min": 0,
                "trades_over_60min": 0,
            }
            return pd.DataFrame([payload])

        net_pnl = trades["net_pnl"]
        gross_pnl = trades["gross_pnl"]
        hold_minutes = trades["holding_minutes"]

        # Categorize trades by duration
        trades_under_10 = (hold_minutes < 10).sum()
        trades_10_30 = ((hold_minutes >= 10) & (hold_minutes < 30)).sum()
        trades_30_60 = ((hold_minutes >= 30) & (hold_minutes < 60)).sum()
        trades_over_60 = (hold_minutes >= 60).sum()

        payload = {
            "trade_count": int(len(trades)),
            "total_gross_pnl": float(gross_pnl.sum()),
            "total_net_pnl": float(net_pnl.sum()),
            "hit_rate": float((net_pnl > 0).mean()),
            "avg_hold_minutes": float(hold_minutes.mean()),
            "median_hold_minutes": float(hold_minutes.median()),
            "min_hold_minutes": float(hold_minutes.min()),
            "max_hold_minutes": float(hold_minutes.max()),
            "pnl_stdev": float(net_pnl.std(ddof=0)) if len(trades) > 1 else 0.0,
            "trades_under_10min": int(trades_under_10),
            "trades_10_30min": int(trades_10_30),
            "trades_30_60min": int(trades_30_60),
            "trades_over_60min": int(trades_over_60),
        }
        return pd.DataFrame([payload])
