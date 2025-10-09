# Trading Rules

This document describes the trading rules used in the TimeGPT Intraday v2 project.

## Entry Rules

The entry rules are designed to identify trading opportunities based on the quantile forecasts provided by the TimeGPT model.

### Long Entry

A long position is entered if the following conditions are met:

*   The 25th percentile of the forecast (`q25`) is greater than the last price plus the trading costs in basis points.
*   The absolute difference between the 50th percentile of the forecast (`q50`) and the last price is greater than or equal to a configurable multiple (`k_sigma`) of the 5-minute volatility (`sigma_5m`).

```
q25 > last_price + costs_bp
AND
|q50 - last_price| >= k_sigma * sigma_5m
```

### Short Entry

A short position is entered if the following conditions are met:

*   The 75th percentile of the forecast (`q75`) is less than the last price minus the trading costs in basis points.
*   The absolute difference between the 50th percentile of the forecast (`q50`) and the last price is greater than or equal to a configurable multiple (`k_sigma`) of the 5-minute volatility (`sigma_5m`).

```
q75 < last_price - costs_bp
AND
|q50 - last_price| >= k_sigma * sigma_5m
```

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

*   **Uncertainty-scaled:** The size of a position can be scaled based on the uncertainty of the forecast (e.g., the difference between `q75` and `q25`).
*   **Max open per symbol:** A maximum of one open position is allowed per symbol at any given time.
*   **Daily trade cap:** A maximum number of trades are allowed per day.
