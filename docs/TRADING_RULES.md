# Trading Rules

This document describes the trading rules used in the TimeGPT Intraday v2 project.

## Entry Rules

The entry rules are designed to identify trading opportunities based on the quantile forecasts provided by the TimeGPT model.

### Long Entry

A long position is entered if the following conditions are met:

*   The 25th percentile of the forecast (`q25`) is greater than the trading costs in basis points (return space).
*   The absolute value of the 50th percentile of the forecast (`q50`) is greater than or equal to a configurable multiple (`k_sigma`) of the 5-minute volatility (`sigma_5m`).
*   The quantile spread (`q75 - q25`) is less than or equal to 2 times the 5-minute volatility (uncertainty suppression).
*   The expected value after costs (`q50 - cost_return`) is greater than 0.

When all checks pass, the position size is 1.0 (full unit exposure).

### Short Entry

A short position is entered if the following conditions are met:

*   The 75th percentile of the forecast (`q75`) is less than minus the trading costs in basis points (return space).
*   The absolute value of the 50th percentile of the forecast (`q50`) is greater than or equal to a configurable multiple (`k_sigma`) of the 5-minute volatility (`sigma_5m`).
*   The quantile spread (`q75 - q25`) is less than or equal to 2 times the 5-minute volatility (uncertainty suppression).
*   The expected value after costs (`q50 + cost_return`) is less than 0.

When all checks pass, the position size is -1.0 (full unit exposure).

## Exit Rules

The exit rules are designed to close open positions based on a set of predefined conditions.

### Take Profit

A position is closed to take profit if the current price moves in a favorable direction by a configurable multiple (`s_take`) of the 5-minute volatility (`sigma_5m`).

*   **Long position:** `current_price >= entry_price + s_take * sigma_5m`
*   **Short position:** `current_price <= entry_price - s_take * sigma_5m`

### Stop Loss

A position is closed to limit losses if the current price moves in an unfavorable direction by a configurable multiple (`s_stop`) of the 5-minute volatility (`sigma_5m`).

*   **Long position:** `current_price <= entry_price - s_stop * sigma_5m`
*   **Short position:** `current_price >= entry_price + s_stop * sigma_5m`

### Time Stop

All open positions are closed at a configurable time of day (`time_stop`), which is typically set to a few minutes before the market closes.

## Sizing

The sizing of positions is determined by the following rules:

*   **Fixed size:** Each trade is sized at 1.0 unit of exposure (full position).
*   **Max open per symbol:** The simulator instantiates one rule engine per parameter set and never
    opens more than `max_open_per_symbol` positions per symbol simultaneously.
*   **Daily trade cap:** `daily_trade_cap` enforces a per-symbol limit on new entries each trading
    day to keep turnover bounded.
*   **Cooldown:** Positions cannot be re-entered immediately after exit; overlapping positions are prevented.
