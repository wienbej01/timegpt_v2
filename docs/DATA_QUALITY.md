# Data Quality Policies

The ingestion pipeline enforces a contract that every bar includes canonical timestamps, prices, volumes, and adjusted fields. Each dataset is evaluated against the following checks before it can advance downstream:

- **Schema** – required columns (`timestamp`, `symbol`, OHLCV, adjusted OHLC) must be present; failure stops the run when `hard_fail_on_schema` is enabled.
- **Monotonicity** – timestamps must be strictly increasing per symbol with no duplicates.
- **Price Sanity** – validates that lows bound opens/closes, highs cap them, and volume remains non-negative.
- **RTH Completeness** – verifies that each trading day has at least the configured percentage of the 390 regular-hours minutes.
- **Adjusted Prices** – when enabled, asserts availability of adjusted OHLC columns to prevent corporate-action distortions.
- **Gapless Grid** – reindexes each session to the full RTH minute grid, forward filling short gaps, tagging synthetic bars with `ffill_flag`, and optionally dropping days with sustained gaps.
- **Outlier Scan** – applies a robust z-score to intraday log returns to flag anomalous moves without removing them.

Policy thresholds are defined in `configs/dq_policy.yaml` and serialized into `artifacts/runs/<run_id>/validation/dq_report.json` for downstream visibility.
