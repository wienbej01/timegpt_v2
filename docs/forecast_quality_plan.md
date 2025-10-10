# Forecast Quality Enhancement Sprints

## Sprint 1 — Diagnostics & Target Scaling
**Goal:** Understand current failures and stabilise the prediction target.

- **Tasks**
  - Add CLI/utility to dump per-symbol metrics (`rMAE`, `rRMSE`, PIT deviation, interval width) into `artifacts/runs/<id>/eval/forecast_diagnostics.csv`.
  - Plot histograms of `y_true` and interval widths (Matplotlib notebook or script) to confirm dispersion.
  - Implement volatility-standardised (`z_ret`) and basis-point (`bp_ret`) targets; update forecast writer to store raw `y_true` with scaling metadata.
- **KPIs**
  - Diagnostics file generated with ≥1 entry per symbol. **(Complete)**
  - Scaled target distribution has mean ~0 and std ~1 (reported in diagnostics). **(In progress – instrumentation ready)**
- **Tests**
  - Unit test for scaling utilities (forward/inverse transform consistency). **(Complete)**
  - CLI test ensuring diagnostics CSV columns exist when forecast data is present. **(Pending manual verification)**

## Sprint 2 — Feature Augmentation (Ticker Only)
**Goal:** Provide richer context from OHLCV-derived signals.

- **Tasks**
  - Add volatility (1m/5m/15m/30m RV, Garman–Klass, Parkinson) and momentum (VWAP slope, cumulative VWAP deviation, ATR gradients) features.
  - Introduce microstructure proxies: volume percentiles, bar overlap ratio, true range percentage.
  - Ensure all new features are lagged appropriately and included both in the feature matrix and the `X_df` horizon payload.
- **KPIs**
  - Feature coverage ≥ 95% after NaN filtering.
  - Forecast interval median width increases by ≥ 30% relative to baseline (proxy for avoiding collapsed bands).
- **Tests**
  - Feature unit tests verifying no future leakage (timestamps only from past data).
  - Regression test confirming `build_x_df_for_horizon` carries new feature columns.

## Sprint 3 — Snapshot & Horizon Optimisation
**Goal:** Align forecasting cadence with market regimes.

- **Tasks**
  - Parameterise snapshot configurations (e.g., `{09:45, 10:15, 12:00, 14:00, 15:15}`) and horizon options (5/15/30 minutes).
  - Update scheduler config to support multiple presets and event-aware exclusions (skip FOMC, CPI windows).
  - Instrument metrics to compare outcomes across snapshot/horizon sets.
- **KPIs**
  - At least one alternative snapshot set achieves PIT deviation ≤ 0.08 on hold-out data.
  - Horizon sensitivity report summarising rMAE and interval width per horizon.
- **Tests**
  - Scheduler tests covering new configuration parsing and quota enforcement.
  - Smoke test running the pipeline with new snapshot sets to ensure artifacts exist.

## Sprint 4 — TimeGPT Configuration & Calibration
**Goal:** Tune the API call and correct residual bias.

- **Tasks**
  - Expand quantile set to include `[0.1, 0.25, 0.5, 0.75, 0.9]` and optionally `level=[50, 80, 95]`.
  - Evaluate backend variants (`timegpt-1`, `timegpt-1-long-horizon`), adjust timeouts/retries.
  - Build calibration layer: per-symbol affine scaling and/or isotonic regression based on recent residuals.
  - Persist calibration parameters (`models/calibration.json`) and apply them prior to evaluation.
- **KPIs**
  - Post-calibration PIT deviation ≤ 0.05 on validation month.
  - Median rMAE improvement of ≥ 10% versus uncalibrated baseline.
- **Tests**
  - Calibration unit tests ensuring invertibility and monotonicity.
  - Integration test verifying calibrated forecasts still produce monotonically ordered quantiles.

## Sprint 5 — Grid Search & Operationalisation
**Goal:** Automate configuration exploration and harden operations.

- **Tasks**
  - Extend grid runner to iterate over snapshot sets, horizons, target scaling, quantile configurations, and calibration toggles.
  - Build composite scoring (e.g., `score = Sharpe × (1 - PIT deviation)`) to rank runs.
  - Add weekly automation hooks (cron or make target) to retrain/fine-tune with latest data and update calibration.
  - Document new pipeline, feature additions, and calibration process in `docs/FORECASTING.md`.
- **KPIs**
  - Grid search produces ranked summary with at least 10 configurations tested.
  - Production preset meeting target gates: median PIT deviation ≤ 0.03, rMAE ≤ 0.7, trading Sharpe ≥ 0.5 on OOS.
- **Tests**
  - Grid runner tests ensuring parameter combinations generate distinct artifact directories.
  - Documentation lint/check (spellcheck or markdown link validation).
