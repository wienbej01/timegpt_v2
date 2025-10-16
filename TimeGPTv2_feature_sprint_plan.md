
# TimeGPT v2 — OHLCV-Only Feature Enablement & Forecast Payload Plan
**Target repo:** `wienbej01/timegpt_v2`  
**Executor:** Claude Code (editor/agent)  
**Owner:** Jacob — Chief Data Scientist  
**Scope:** Implement a compact, high-signal **OHLCV-only** feature set and wire it through the **forecast payload builder** (`y_df` & `x_df`) with tests, diagnostics, and smoke runs.

---

## Guiding Principles
- **Deterministic future exogs only** in `x_df` (time-of-day, session clock), so the model can condition on them in the horizon.  
- **Historical context** in `y_df` includes short-horizon returns/volatility, range/expansion, VWAP deviation, session progress.  
- Keep payload **small and stable**. Prefer **one** range-volatility estimator (Parkinson _or_ Garman–Klass).  
- Tests first; **instrument** the payload builder to log which columns are actually sent per snapshot.

---

## Files (expected)
- `src/timegpt_v2/framing/build_payloads.py`  ← payload assembly for `y_df` & `x_df`
- `src/timegpt_v2/fe/`                       ← feature engineering (add new OHLCV-only features)
- `src/timegpt_v2/utils/col_schema.py`       ← column names/types registry
- `src/timegpt_v2/cli.py`                    ← CLI wiring
- `configs/forecast.yaml`, `configs/*.yaml`  ← configuration
- `tests/`                                   ← unit/integration tests (new)

> If paths differ, update this plan accordingly.

---

## Feature Set (OHLCV-only)
### Historical exogs (`y_df` only)
- `ret_1m`, `ret_5m`  
- `sigma_5m` = `sqrt(rv_5m)` (realized vol over last 5 minutes)  
- **Choose one** range σ (Parkinson **or** Garman–Klass) over last 15 minutes: `parkinson_sigma_15m` **or** `gk_sigma_15m`  
- `range_pct_15m` = `(high_15m - low_15m) / close_t`  
- `clv_15m` = `(close - low)/(high - low)` on a 15m lookback (bounded [0,1], define 0.5 on zero-range)  
- `vwap_dev` = `(close / vwap_intraday - 1)`  
- `rth_cumret_30m` = cumulative return since RTH open (last 30m)  

### Deterministic exogs (appear in **both** `y_df` and `x_df`)
- `minute_of_day_sin`, `minute_of_day_cos` (seasonality)  
- `minutes_since_open`, `minutes_to_close` (session clock)  
- (Optional tiny categorical) `day_of_week` as int (0..4)

---

## Sprint 0 — Environment & Safety (0.5 day)
**Goals**
- Create a working branch, enable tests, add a dry-run flag for payload logging.

**Tasks**
1. Create branch `feat/ohlcv-exogs`.
2. Add `poetry`/`requirements.txt` test deps if missing: `pytest`, `pandas`, `numpy`, `pyyaml`.
3. Add env toggle `PAYLOAD_LOG=1` for verbose payload column logging in `build_payloads.py` (emit once per snapshot).

**Acceptance**
- `pytest -q` runs (even if zero tests initially).
- A no-op forecast run logs a single line like:  
  `PAYLOAD y_df cols=['unique_id','ds','y','ret_1m',...], x_df cols=['unique_id','ds','minute_of_day_sin',...]`

---

## Sprint 1 — Feature Engineering Primitives (1 day)
**Goals**
- Implement the missing OHLCV-only computations inside `src/timegpt_v2/fe` with small, composable helpers.

**Tasks**
1. Add functions in `src/timegpt_v2/fe/base_features.py` or a new `fe/ohlcv_only.py`:
   - `compute_ret_1m`, `compute_ret_5m` (log or simple, consistent with target)
   - `compute_rv_5m` and `sigma_5m = sqrt(rv_5m.clip(lower=0))`
   - `compute_parkinson_sigma_15m(bars)` **or** `compute_gk_sigma_15m(bars)` using rolling 15m windows
   - `compute_range_pct_15m`
   - `compute_clv_15m` (handle zero-range → set to 0.5)
   - `compute_vwap_dev_intraday` (requires intraday VWAP; if not present, compute cumulative VWAP from OHLCV)
   - `compute_rth_cumret_30m` (needs session boundaries; reuse existing RTH logic)
2. Extend `col_schema.py` with new column names and dtypes.
3. Wire these functions into your current feature pipeline so they are materialized before forecasting.

**Tests**
- `tests/test_fe_ohlcv_only.py`
  - Construct a small minute-level DataFrame and assert expected shapes & no NaNs for normal periods.
  - Parkinson/GK: assert non-negativity and correlation with realized vol on a synthetic path.
  - VWAP deviation: verify 0 near start and sign with price > VWAP.

**Acceptance**
- `pytest -q` green on the added tests.
- Running feature build produces all listed columns for a sample day (AAPL).

---

## Sprint 2 — Deterministic Future Exogs (0.5–1 day)
**Goals**
- Ensure deterministic exogs are **present in y_df and x_df**.

**Tasks**
1. Implement in `fe/deterministic.py` (or similar):
   - `minute_of_day_sin/cos` from timestamp (ET-tz aware).
   - `minutes_since_open` & `minutes_to_close` using RTH window from config.
   - `day_of_week` as int 0–4.
2. Ensure these columns are produced historically (features stage) and **recomputed for future minutes** inside `build_x_df_for_horizon`.

**Tests**
- `tests/test_deterministic_exogs.py`
  - For a given ET timestamp, assert expected sin/cos position and correct minutes-to/from RTH boundaries.
  - Ensure future grid (`x_df`) columns exist and are non-null across the horizon minutes.

**Acceptance**
- `x_df` contains the deterministic exogs; `y_df` contains the same names (plus history-only exogs).

---

## Sprint 3 — Payload Builder Wiring (1 day)
**Goals**
- Replace the small hard-coded allow-list with a compact **explicit allow-list** that includes the features above, and ensure `x_df` carries **only deterministic** columns.

**Tasks**
1. In `framing/build_payloads.py`:
   - Replace `EXOGENOUS_FEATURE_COLUMNS` with two groups:
     ```python
     HIST_EXOG_ALLOW = [
       "ret_1m","ret_5m","sigma_5m","parkinson_sigma_15m", # or gk_sigma_15m
       "range_pct_15m","clv_15m","vwap_dev","rth_cumret_30m",
     ]
     FUTR_EXOG_ALLOW = [
       "minute_of_day_sin","minute_of_day_cos",
       "minutes_since_open","minutes_to_close","day_of_week",
     ]
     ```
   - `y_df`: select intersection with `HIST_EXOG_ALLOW + FUTR_EXOG_ALLOW` (so deterministic appears in both).
   - `x_df`: **only** intersection with `FUTR_EXOG_ALLOW`.
2. Add a guard: if `strict_exog: true` (config), raise if any requested **deterministic** exog is missing.
3. Log payload column lists when `PAYLOAD_LOG=1`.

**Tests**
- `tests/test_payload_builder.py`
  - Given a small features frame, assert `y_df` includes hist+deterministic exogs; `x_df` includes only deterministic.
  - If `strict_exog: true` and a deterministic exog missing → raise.

**Acceptance**
- Deterministic columns are mirrored in `y_df` & `x_df`; historical columns appear only in `y_df`.

---

## Sprint 4 — Config & CLI Plumbing (0.5 day)
**Goals**
- Make the new exogs first-class in config; keep byte size under control.

**Tasks**
1. Update `configs/forecast.yaml`:
   - `exog.use_exogs: true`
   - `exog.strict_exog: true` (optional but recommended once stable)
   - Document the two allow-lists in comments.
2. `cli.py`: no API change, but add a `--payload-log` flag to set `PAYLOAD_LOG=1` for ad-hoc runs (or read from config).

**Tests**
- Minimal: `pytest -q` still green; dry-run forecast prints payload cols with the new deterministic set.

**Acceptance**
- One snapshot run shows the intended columns in logs.

---

## Sprint 5 — Integration Test & Smoke Run (1 day)
**Goals**
- Validate end-to-end on a narrow universe and day, verify payload size and that trades still occur.

**Tasks**
1. Add `configs/forecast_smoke.yaml` + `configs/universe_smoke.yaml` (AAPL, single day, min history relaxed).
2. Run the standard pipeline:
   ```bash
   RUN_ID=ohlcv_smoke
   python -m timegpt_v2.cli build-features --config-name forecast_smoke.yaml --universe-name universe_smoke.yaml --run-id $RUN_ID
   python -m timegpt_v2.cli forecast       --config-name forecast_smoke.yaml --universe-name universe_smoke.yaml --run-id $RUN_ID --payload-log
   python -m timegpt_v2.cli backtest       --config-name forecast_smoke.yaml --run-id $RUN_ID
   ```
3. Inspect `artifacts/runs/$RUN_ID/logs`:
   - Confirm payload columns as expected.
   - Confirm `bt_trades.csv` non-empty (at least 1–2 trades).

**Tests**
- `tests/test_integration_payload_columns.py` loads last run payload metadata (or a captured fixture) and asserts column presence/absence rules.

**Acceptance**
- Smoke passes; payload size reasonable (no partitions explosion); ≥1 trade generated in backtest on the sample day.

---

## Sprint 6 — Diagnostics & Docs (0.5 day)
**Goals**
- Make feature usage transparent, document how to extend safely.

**Tasks**
1. Add a single-line **funnel** log in backtester (if not present): counts of rows → signal-pass → trade.
2. Add a `README.md` section:
   - “OHLCV-only features” (list & definitions)
   - “Deterministic future exogs and why”
   - “How to add a new feature safely” (must appear in `y_df` and, if deterministic, also in `x_df`; watch byte ceiling)

**Acceptance**
- Docs merged; a sample log line printed per run summarizing payload cols and funnel counts.

---

## Risk & Rollback
- **Payload bloat** → monitor `max_bytes_per_call`; if exceeded, drop Tier-B columns first.  
- **Time-zone drift** → deterministic features must use ET; add an assertion in tests.  
- **Sparse days** → VWAP/unit tests should handle zero-volume minutes (ffill guarded).  
- **Strict mode** too brittle → keep `strict_exog: false` while iterating.

---

## Done Definition (global)
- All new features materialized in features parquet with correct dtypes.
- `y_df` includes **historical** + **deterministic** exogs; `x_df` includes **deterministic** exogs only.
- Tests are green; smoke run yields trades; logs show intended columns.
- README updated; byte and API budgets respected.

---

## Appendix — Minimal Feature Formulas (reference)

- `ret_1m(t) = close(t)/close(t-1) - 1`  *(or log)*  
- `ret_5m(t) = close(t)/close(t-5) - 1`
- `rv_5m(t) = Σ_{i=t-4..t} (ret_1m(i))^2` ; `sigma_5m = sqrt(rv_5m)`
- **Parkinson σ (window W):** `σ²_P = (1/(4 ln 2)) * mean[(ln(H/L))^2]` over W; σ_P = sqrt(σ²_P)  
- **Garman–Klass σ (W):** `σ²_GK = 0.5*(ln(H/L))^2 - (2ln2 -1)*(ln(C/O))^2` (window mean), √ for σ  
- `range_pct_15m = (high_{15m} - low_{15m}) / close_t`
- `clv_15m = (close_t - low_{15m}) / max(high_{15m} - low_{15m}, ε)`; if denom=0 → 0.5
- `vwap_dev = close_t / VWAP_intraday_t - 1`
- `minute_of_day_sin = sin(2π * minute_idx / total_rth_minutes)` ; cos analog
- `minutes_since_open`, `minutes_to_close` from RTH window
- `day_of_week = int(dt.weekday())`

