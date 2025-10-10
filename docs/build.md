# Enhancement Build Plan: TimeGPT v2 Intraday Forecasting

Purpose of the codebase
- This repository implements an intraday forecasting-to-trading pipeline that ingests cleaned minute bars, builds leakage-safe features, produces probabilistic forecasts (quantile bands), and executes a rules-based backtest with realistic costs/sensitivity gates.
- Non-negotiables: temporal integrity, explicit costs/slippage, drawdown limits, parsimony, explainability, reproducibility, gating and promotion via unbiased KPIs.

Purpose of this enhancement build
- Replace all stub forecasting with live TimeGPT and add robust calibration, improve snapshot cadence, align labels with horizons, refine features/rules, and harden evaluation gates to materially improve PIT coverage, calibration, and trading PnL/Sharpe.
- Target outcome: median PIT deviation ≤ 0.03, rMAE ≤ 0.7 vs persistence, OOS Sharpe ≥ 0.5 under 1.5× cost sensitivity; uphold failure gates.

Scope boundary
- Live inference only (no stub fallback).
- 10-symbol large-cap tech basket (current configs) but methods generalize to larger universes.
- Minute frequency only; no changes to data loader contracts.

Reference implementation anchors (clickable)
- CLI pipeline: [src/timegpt_v2/cli.py](src/timegpt_v2/cli.py:1)
- TimeGPT client/backends: [src/timegpt_v2/forecast/timegpt_client.py](src/timegpt_v2/forecast/timegpt_client.py:1)
- Snapshot scheduler: [src/timegpt_v2/forecast/scheduler.py](src/timegpt_v2/forecast/scheduler.py:1)
- Framing/payloads: [src/timegpt_v2/framing/build_payloads.py](src/timegpt_v2/framing/build_payloads.py:1)
- Target scaling: [src/timegpt_v2/forecast/scaling.py](src/timegpt_v2/forecast/scaling.py:1)
- Calibration module: [src/timegpt_v2/eval/calibration.py](src/timegpt_v2/eval/calibration.py:1)
- Forecast config: [configs/forecast.yaml](configs/forecast.yaml:1)
- Trading rules: [src/timegpt_v2/trading/rules.py](src/timegpt_v2/trading/rules.py:1)
- Evaluation metrics/gates: [src/timegpt_v2/cli.py](src/timegpt_v2/cli.py:763)

Completed preflight (Sprint 0)
- Load .env automatically and surface TIMEGPT_API_KEY/NIXTLA_API_KEY to the process at module import in [src/timegpt_v2/cli.py](src/timegpt_v2/cli.py:15).
- Enforce live backend (no stub fallback). The CLI raises if backend is not initialized and sets backend: nixtla in [configs/forecast.yaml](configs/forecast.yaml:20).
- Logging now states the resolved backend mode at forecast start in [src/timegpt_v2/cli.py](src/timegpt_v2/cli.py:495).

Sprint 1 — Wire live TimeGPT and secrets (COMPLETED)
Objective
- Guarantee live Nixtla backend via API key from .env and disable deterministic stub in all production runs.
Tasks
- Confirm .env is parsed on import (via python-dotenv or fallback). Verify environment contains TIMEGPT_API_KEY or NIXTLA_API_KEY.
- Require forecast.yaml backend: nixtla. Remove any "auto" usage in production configs.
- Ensure CLI fails fast when backend init fails and prints actionable error.
Verification
- Run: RUN_ID=smoke python -m timegpt_v2.cli forecast --config-dir configs --run-id smoke
- Expect logger line “Using forecast backend=NixtlaTimeGPTBackend …” and no “[cache]” only runs; see [artifacts/runs/<run>/logs/forecast.log](artifacts/runs/).
- Gate: fail if backend not nixtla.
Tests
- test_env_loading: simulate .env and assert key present in os.environ before CLI subcommands.
- test_backend_enforced: patch Nixtla init to raise; assert CLI raises typer.BadParameter.
- test_no_stub_path: ensure client is not instantiated with _LocalDeterministicBackend anywhere when backend: nixtla.

Sprint 2 — Quantile calibration (fit/apply, embargo, monotonicity)
Objective
- Achieve nominal coverage and reduce PIT deviation using affine/isotonic calibration with embargo to prevent leakage. Provide conformal fallback.
Tasks
- Implement a new CLI command calibrate to fit models from historical forecasts vs y_true (prior runs), using [ForecastCalibrator.fit()](src/timegpt_v2/eval/calibration.py:161) with CalibrationConfig from [configs/forecast.yaml](configs/forecast.yaml:36).
- Enforce embargo: calibration window must end at least 1 trading day before the evaluation period start.
- Persist to models/calibration.json via [CalibrationModel.save()](src/timegpt_v2/eval/calibration.py:93).
- Apply during forecast: after inverse scaling, load model and [ForecastCalibrator.apply()](src/timegpt_v2/eval/calibration.py:234); then enforce quantile monotonicity with an isotonic projection q10≤q25≤q50≤q75≤q90.
- Add optional split-conformal widening fallback when validation PIT deviation > 0.03: set half-width δ = empirical quantile_0.25(|y−q50|) on rolling window; assign q25/q75 = q50 ± δ.
Interoperability
- Forecast step consumes models/calibration.json if present; calibrate step writes it; evaluate step reports pre/post coverage deltas.
Tests
- test_calibrate_fit_persist: feed synthetic forecasts/actuals, fit affine and isotonic, assert model files/keys exist.
- test_apply_monotone: craft crossing quantiles, assert projection enforces order.
- test_conformal_widening: with undercoverage synthetic set, assert PIT rises toward nominal and widths bounded.
- test_embargo: ensure calibrate refuses overlapping periods with evaluation by ≥ 1 trading day.

Sprint 3 — Snapshot policy and cadence
Objective
- Increase sample efficiency and robustness by moving to liquidity_profile preset (5/day) with event-day skips.
Tasks
- Set snapshot_preset: liquidity_profile in [configs/forecast.yaml](configs/forecast.yaml:1).
- Keep skip_events: [event_fomc, event_cpi]; ensure skip-dates populated at [src/timegpt_v2/cli.py](src/timegpt_v2/cli.py:366).
- Optional: add a late-day snapshot (e.g., 15:15) if time_stop allows execution.
Verification
- Meta lists 5× snapshots per day in [artifacts/runs/<run>/meta.json](artifacts/runs/aug_oct_2024_prod/meta.json:23).
Tests
- test_scheduler_active_windows: verify snapshots fall inside active window bounds at [src/timegpt_v2/forecast/scheduler.py](src/timegpt_v2/forecast/scheduler.py:44).
- test_skip_events: assert all skip_dates removed from snapshots when feature flags present.

Sprint 4 — Horizon/label alignment
Objective
- Align economics: either set horizon_min: 1 for 1-minute label or create a 15-minute aggregated label for horizon=15.
Option A (preferred): add target_log_return_15m
- In [src/timegpt_v2/fe/base_features.py](src/timegpt_v2/fe/base_features.py:1) compute target_log_return_15m = log Close(t+15m) − log Close(t) with label_timestamp = t+15m; ensure no peeking.
- Update [TargetScaler.target_column()](src/timegpt_v2/forecast/scaling.py:55) to route when target.mode: log_return_15m.
- Use this target in [build_y_df()](src/timegpt_v2/framing/build_payloads.py:33) via target_column selection.
Option B (minimal change)
- Set horizon_min: 1 in [configs/forecast.yaml](configs/forecast.yaml:19) and sweep horizons ∈ {1, 5, 15} using [src/timegpt_v2/forecast/sweep.py](src/timegpt_v2/forecast/sweep.py:46).
Tests
- test_label_alignment: assert y_df ds equals snapshot_utc and label_timestamp equals forecast_ts for chosen horizon.
- test_no_leakage_labels: backshift windows; confirm future bars are not used.

Sprint 5 — Feature refinements (parsimonious, leakage-safe)
Objective
- Improve scale-awareness and cross-sectional conditioning without overfitting.
Tasks
- Add vol_ewm_15m alongside existing vol_ewm_60m; ensure used only as history before snapshot.
- Add lagged market context (SPY/QQQ/VIX t−1m) with symbol betas and volatility dispersion features.
- Add snapshot “liquidity bucket” categorical from vol_5m_norm/range_pct rolling percentiles.
- Ensure deterministic projection [build_x_df_for_horizon()](src/timegpt_v2/framing/build_payloads.py:60) carries minute_ahead and time features; consider projecting bucket/regime flags.
Ablations
- Run ablations removing context/vol/liquidity; promote only if composite_score or PIT improves.
Tests
- test_feature_leakage: verify all new features are computed from t’≤ snapshot.
- test_feature_presence_xdf: assert projected X_df includes required deterministic columns.

Sprint 6 — Trading rulebook and risk controls
Objective
- Make entry/size/exit consistent with calibrated quantiles and volatility.
Entries
- Long if q25 > 0 and q50 > k_sigma × σ_60m; Short if q75 < 0 and q50 < −k_sigma × σ_60m; k_sigma ∈ {0.25, 0.5, 0.75}.
Position sizing
- Size ∝ |q50|/σ_60m capped by cap_per_trade and max_open_per_symbol.
Exits
- Time-based: exit at forecast_ts; hard stop at time_stop.
- Price-based: s_stop = c_stop × σ_60m; s_take = c_take × σ_60m; c_stop ∈ {0.5, 1.0}, c_take ∈ {1.0, 1.5}.
Filters
- Skip event days and abnormal liquidity (vol_5m_norm < 10th pct or range_pct > 99th pct).
Costs & slippage
- Maintain symbol-specific half_spread_ticks and fee_bps in [configs/trading.yaml](configs/trading.yaml:1); review with Risk & Execution Controls.
Tests
- test_rule_consistency: simulate quantile sets and σ_60m to assert entry direction and sizing monotonicity.
- test_cost_sensitivity: verify 1.5× costs scenario still positive PnL for promoted configs.

Sprint 7 — Sweeps, promotion, and monitoring
Objective
- Systematically search presets/horizons/calibration/targets and promote under strict gates.
Tasks
- Update [configs/forecast_grid.yaml](configs/forecast_grid.yaml:1) with:
  - snapshot_presets: [baseline, liquidity_profile]
  - horizons: [1, 5, 15]
  - quantile_sets: [[0.1,0.25,0.5,0.75,0.9]]
  - calibration_methods: [none, affine, isotonic, conformal]
  - target_modes: [log_return, (optional) log_return_15m]
- Execute: make forecast-grid or timegpt_v2.cli sweep --forecast-grid configs/forecast_grid.yaml --execute
- Promote top runs where: PIT dev ≤ 0.03, median rMAE ≤ 0.7, OOS Sharpe ≥ 0.5, positive at 1.5× costs; rank by composite_score at [src/timegpt_v2/forecast/sweep.py](src/timegpt_v2/forecast/sweep.py:344).
Monitoring
- Add backend-mode assertion and quantile monotonicity validator during forecast; persist violations to logs.
Tests
- test_grid_plan_integrity: ensure plan.csv and scoreboard.csv present and well-formed.
- test_gates_enforced: assert evaluate exits non-zero when gates violated at [src/timegpt_v2/cli.py](src/timegpt_v2/cli.py:908).

Sprint 8 — Documentation, ops, and handoffs
Objective
- Keep architecture and ops docs synchronized; define handoffs.
Tasks
- Update SYSTEM_TECH_DOC.md with:
  - Backend enforcement and .env loader
  - Calibration fit/apply workflow and embargo
  - Snapshot policy and horizon/label alignment
  - Monitoring additions (monotonicity, backend-mode)
- Add runbook section for secrets handling and incident response (backend failures).
Handoffs
- Risk & Execution Controls: symbol-level microstructure parameters; circuit breakers.
- Testing & Analytics Hub: embargoed WFO, ablations, attribution.
- Engineering & Platform: calibrate CLI command, monotonic projection utility, CI jobs.

Function and interoperability tests (consolidated checklist)
- Env/secrets: .env autoload, backend init, failure path
- Framing: y_df inclusive to snapshot, X_df deterministic projection, symbol completeness per snapshot
- Calibration: fit/apply, embargo, monotonicity, conformal fallback
- Scheduler: active windows, event skips, holiday handling
- Forecast output: no NaNs; quantile order; y_true join correctness
- Backtest: rules consistency; cost sensitivity; OOS gate checks; logging completeness
- Grid: plan/scoreboard integrity; composite_score ordering

Run commands (reference)
- Data/Features: RUN_ID=<id> python -m timegpt_v2.cli check-data --config-dir configs --run-id <id>
- Build features: RUN_ID=<id> python -m timegpt_v2.cli build-features --config-dir configs --run-id <id>
- Forecast (live): RUN_ID=<id> python -m timegpt_v2.cli forecast --config-dir configs --run-id <id>
- Backtest: RUN_ID=<id> python -m timegpt_v2.cli backtest --config-dir configs --run-id <id>
- Evaluate (gates): RUN_ID=<id> python -m timegpt_v2.cli evaluate --config-dir configs --run-id <id>
- Sweep: RUN_ID=<id> python -m timegpt_v2.cli sweep --config-dir configs --run-id <id> --forecast-grid configs/forecast_grid.yaml

External references (TimeGPT docs via Context7)
- Quantile forecasts: https://github.com/nixtla/nixtla/blob/main/timegpt-docs/forecasting/probabilistic/quantiles.mdx
- Prediction intervals (levels): https://github.com/nixtla/nixtla/blob/main/nbs/docs/tutorials/11_uncertainty_quantification_with_prediction_intervals.ipynb
- Cross-validation for probabilistic forecasts: https://github.com/nixtla/nixtla/blob/main/timegpt-docs/forecasting/evaluation/cross_validation.mdx
- Exogenous X_df usage: https://github.com/nixtla/nixtla/blob/main/nbs/docs/tutorials/22_how_to_improve_forecast_accuracy.ipynb

Assumptions and constraints
- All signals evaluated at bar_close + 1 bar; no forward peeking; purged/embargoed splits for calibration and WFO.
- Costs/slippage modeled via fee_bps and half_spread_ticks; cost sensitivity required to pass.
- API rate limits respected via retry policy; disk caching remains enabled for idempotent re-runs (no duplicates).

Success criteria and promotion
- Median PIT deviation ≤ 0.03; median rMAE ≤ 0.7; OOS Sharpe ≥ 0.5; positive PnL at 1.5× costs.
- No quantile crossing; interval widths scale with realized volatility deciles.

Constitutional Fitness Score (CFS)
- Economic rationale: strong (vol-scaling, quantile-consistent entries). 
- Market realism: strong (costs, event skips, cadence, embargo).
- Parsimony: moderate (features add minimal, justified variables).
- Overfit risk: controlled (embargo, sweeps, gates).
- CFS: 8.6/10.

Implementation readiness
- Sprint 0 changes are applied: .env autoload + backend enforcement. 
- Ready to proceed with Sprint 2 (calibration) and subsequent sprints per this plan.

Appendix — File edits to anticipate
- New command: calibrate in [src/timegpt_v2/cli.py](src/timegpt_v2/cli.py:1)
- Monotone projection util in [src/timegpt_v2/eval/calibration.py](src/timegpt_v2/eval/calibration.py:234)
- Feature additions in [src/timegpt_v2/fe/base_features.py](src/timegpt_v2/fe/base_features.py:1)
- Target selection support for 15m label in [src/timegpt_v2/forecast/scaling.py](src/timegpt_v2/forecast/scaling.py:55)
- Grid config updates in [configs/forecast_grid.yaml](configs/forecast_grid.yaml:1)
- Tests in tests/: test_env.py, test_backend.py, test_calibration.py, test_scheduler.py, test_framing.py, test_forecast_output.py, test_rules.py, test_grid.py

Interoperability surfaces
- Forecast consumes models/calibration.json when present; evaluate logs calibration gate pass/fail; sweep aggregates metrics into scoreboard for ranking/promotion.

Execution notes
- Export TIMEGPT_API_KEY in .env; do not echo secrets in logs.
- For any backend failure, abort early, log reason, avoid stub fallback.

End of plan.