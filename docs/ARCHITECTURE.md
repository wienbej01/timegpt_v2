# Architecture Overview

## System Flow

```
Raw Data → Quality Gates → Features → Framing → Forecast → Trading → Backtest → Evaluation → Reports
```

## Module Overview

### `io` - Data Ingestion
- `gcs_reader`: Fetches and normalizes raw intraday bars from GCS Parquet files
- Handles timezone conversion, symbol mapping, and basic data cleaning

### `quality` - Data Quality Gates
- `contracts`: Defines strict data schemas and validation rules
- `checks`: Implements DQ validation routines that gate all downstream processing
- Enforces completeness, monotonicity, price sanity, and outlier detection

### `fe` - Feature Engineering
- `base_features`: Return-space targets, volatility measures, VWAP calculations
- `deterministic`: Intraday clocks, Fourier terms, session buckets
- `context`: Optional SPY lagged features and event dummies

### `framing` - Payload Construction
- `build_payloads`: Translates features into TimeGPT Y_df and X_df formats
- Handles rolling history windows and future deterministic features

### `forecast` - TimeGPT Integration
- `scheduler`: Generates trading day snapshot timestamps with holiday/weekend filtering
- `timegpt_client`: Batched multi-series forecasting with caching and budget management
- `scaling`: Target transformation and inverse scaling for forecasts

### `trading` - Portfolio Logic
- `rules`: Quantile-aware entry/exit rules with EV filtering
- `costs`: Transaction cost modeling (fees + spread)

### `backtest` - Simulation Engine
- `simulator`: Event-driven backtester with position management
- `grid`: Parameter sweep orchestration with cache reuse

### `eval` - Performance Metrics
- `metrics_forecast`: rMAE/rRMSE vs persistence, pinball loss, PIT coverage
- `metrics_trading`: Hit rate, P&L, Sharpe, drawdown analysis
- `calibration`: Post-hoc quantile adjustment and conformal methods

### `reports` - Output Generation
- `builder`: Assembles markdown reports and robustness summaries

### `utils` - Shared Infrastructure
- `cache`: File-based forecast cache with SHA256 hashing
- `api_budget`: API call tracking with hard stops and JSON ledger
- `synthetic`: Deterministic price series generation for testing
- `dt`: Datetime utilities and timezone handling
- `log`: Structured logging configuration
- `events`: Market event detection and filtering

## Key Design Principles

- **Artifacts-first**: All outputs written to `artifacts/runs/<run_id>/` for reproducibility
- **Config-driven**: No magic constants; all parameters in YAML files
- **Budget-aware**: API calls limited with caching and dry-run modes
- **Leakage-safe**: Strict temporal separation between training and prediction data
- **Deterministic**: Seeded random operations for reproducible results
