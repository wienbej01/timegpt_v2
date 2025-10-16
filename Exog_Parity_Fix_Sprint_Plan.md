
# Permanent Exogenous Parity Fix — Sprint Plan (Claude Code)

**Repo:** `wienbej01/timegpt_v2`  
**Goal:** Permanently eliminate Nixtla API `ValueError/TypeError` due to **exogenous parity mismatches** between `y_df` (history) and `X_df` (future), and prevent `_x/_y`-suffix regressions.  
**Executor:** Claude Code (editor/agent); small, audited patches; tests; friendly errors.

---

## Non‑Negotiables

- **No `_x`/`_y` column suffixes** in payloads.
- **Exact same deterministic exog names & dtypes** in both `y_df` and `X_df`.
- **Single source of truth** for deterministic exog names.
- **Preflight validation** before API call (fail fast with actionable error).
- **Unit tests** to prevent regressions permanently.

---

## SPRINT 1 — Centralize Deterministic Exogs (Names + Builders)

**Objective:** One canonical list of deterministic exog names; two builders to compute them for **history** and **future** without merges that create suffixes.

**Edits**
- File: `src/timegpt_v2/fe/deterministic.py`
  - Add:
    ```python
    DETERMINISTIC_EXOG = [
        "minute_of_day_sin", "minute_of_day_cos",
        "minutes_since_open", "minutes_to_close",
        "day_of_week"
    ]

    def get_deterministic_exog_names() -> list[str]:
        return DETERMINISTIC_EXOG.copy()
    ```
  - Implement:
    ```python
    def build_det_exogs_for_history(df_bars: pd.DataFrame, *, tz: str, rth_open: str, rth_close: str) -> pd.DataFrame: ...
    def build_det_exogs_for_future(future_index: pd.DataFrame, *, tz: str, rth_open: str, rth_close: str) -> pd.DataFrame: ...
    ```
  - Enforce dtypes:
    - `day_of_week`: `int8`
    - `minutes_since_open`, `minutes_to_close`: `int32`
    - sin/cos: `float32`
  - **Do not** use merges that can collide; compute columns directly and `pd.concat` by index.

**Acceptance**
- `get_deterministic_exog_names()` returns the list.
- Both builders output frames with **exactly** those columns, correct dtypes, aligned row‑for‑row to inputs.

---

## SPRINT 2 — Payload Builder Parity & Ban Suffixes

**Objective:** Ensure `y_df` includes **historical** + **deterministic** exogs; `X_df` includes **only deterministic** exogs; ban `_x/_y` and validate parity pre‑API.

**Edits**
- File: `src/timegpt_v2/framing/build_payloads.py`
  - Import:
    ```python
    from timegpt_v2.fe.deterministic import (
        get_deterministic_exog_names,
        build_det_exogs_for_history,
        build_det_exogs_for_future,
    )
    ```
  - Define allow‑lists:
    ```python
    HIST_EXOG_ALLOW = [
        "ret_1m","ret_5m","sigma_5m",
        "parkinson_sigma_15m",  # or gk_sigma_15m
        "range_pct_15m","clv_15m","vwap_dev","rth_cumret_30m",
    ]
    FUTR_EXOG_ALLOW = get_deterministic_exog_names()
    ```
  - In `build_y_df(...)`:
    - Compute deterministic exogs via `build_det_exogs_for_history(...)` and `concat` to the working frame.
    - Select columns: `["unique_id","ds","y"] + (present ∩ HIST_EXOG_ALLOW) + FUTR_EXOG_ALLOW`.
  - In `build_x_df_for_horizon(...)`:
    - Create grid `["unique_id","ds"]`, compute deterministic with `build_det_exogs_for_future(...)`, and `concat` column‑wise.
    - Select columns: `["unique_id","ds"] + FUTR_EXOG_ALLOW`.

  - Add validator:
    ```python
    def _validate_exog_parity(y_df: pd.DataFrame, x_df: pd.DataFrame):
        forbidden = [c for c in [*y_df.columns, *x_df.columns] if c.endswith("_x") or c.endswith("_y")]
        if forbidden:
            raise ValueError(f"Forbidden suffixes in payload columns: {sorted(set(forbidden))}")

        det = set(get_deterministic_exog_names())
        y_det = det & set(y_df.columns)
        x_det = det & set(x_df.columns)
        if y_det != x_det:
            raise ValueError(
                f"Deterministic exog name mismatch: in_y_only={sorted(y_det - x_det)}, in_x_only={sorted(x_det - y_det)}"
            )

        mismatches = []
        for col in sorted(det):
            if col in y_df.columns and col in x_df.columns:
                if str(y_df[col].dtype) != str(x_df[col].dtype):
                    mismatches.append((col, str(y_df[col].dtype), str(x_df[col].dtype)))
        if mismatches:
            raise TypeError(f"Deterministic exog dtype mismatch: {mismatches}")
    ```
  - Call `_validate_exog_parity(y_df, x_df)` right before returning payloads to the caller.

**Acceptance**
- Any `_x/_y` column triggers **immediate** error before API.
- Names & dtypes of deterministic exogs are identical across `y_df`/`X_df`.

---

## SPRINT 3 — Strict Mode & Friendly Error

**Objective:** Expose strict mode in config; fail fast with actionable message; optional payload logging.

**Edits**
- `configs/forecast.yaml`:
  ```yaml
  exog:
    use_exogs: true
    strict_exog: true
  ```
- In the forecast CLI path (e.g., `src/timegpt_v2/cli.py`), plumb `strict_exog` to the payload builder.
- Add an optional `--payload-log` or env `PAYLOAD_LOG=1` to print, once per run:
  - `PAYLOAD y_df=[...]`
  - `PAYLOAD x_df=[...]`

**Acceptance**
- With strict mode on, parity errors never reach the API layer; they stop in preflight with a single clear line.

---

## SPRINT 4 — Tests (Lock It In)

**Objective:** Regression‑proof the parity guarantees.

**New file:** `tests/test_payload_parity.py`

**Tests**
1. **test_parity_success**: tiny synthetic sample → build `y_df` & `x_df` → assert:
   - deterministic names present in **both**,
   - **no** `_x/_y` suffixes,
   - dtypes match.
2. **test_suffix_guard_raises**: inject a fake `minute_of_day_sin_x` → expect `ValueError` with “Forbidden suffixes”.
3. **test_mismatch_name_raises**: drop one deterministic column from `x_df` → expect name mismatch error.
4. **test_dtype_mismatch_raises**: cast `day_of_week` in `x_df` to float → expect `TypeError` listing the tuple.

**Acceptance**
- `pytest -q` green.
- Removing any deterministic exog or changing a dtype causes tests to fail (prevents regressions).

---

## SPRINT 5 — Smoke Run & Commit

**Objective:** Verify on smallest universe & day; confirm no API errors; then commit.

**Steps**
```bash
RUN_ID=exog_parity_smoke

python -m timegpt_v2.cli build-features --config-name forecast_smoke.yaml --universe-name universe_smoke.yaml --run-id $RUN_ID
python -m timegpt_v2.cli forecast       --config-name forecast_smoke.yaml --universe-name universe_smoke.yaml --run-id $RUN_ID --payload-log
python -m timegpt_v2.cli backtest       --config-name forecast_smoke.yaml --run-id $RUN_ID
```

**Checks**
- Logs show identical deterministic exog sets for `y_df` and `x_df` (names & dtypes).
- No `_x/_y` in any payload.
- No Nixtla `ValueError/TypeError` thrown.

**Commit**
- Message: `fix(payload): enforce deterministic exog parity; ban _x/_y; add tests`

---

## Rollback & Risks

- If payload size warnings appear, keep deterministic set minimal (5 cols).  
- TZ/clock math must use ET; add assertions in builders (future timestamps must be ET-convertible).  
- Keep strict mode on in CI to catch regressions early.

---

## Final Deliverable (what Claude should print when done)

“✅ Exog parity is enforced. No `_x/_y` allowed. Historical/future deterministic exogs match exactly (names & dtypes). Tests added and green.”
