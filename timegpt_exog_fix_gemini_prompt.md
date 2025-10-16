# TimeGPT Exogenous Merge Fix — Prompt for Gemini CLI

You are a senior Python engineer tasked with fixing an exogenous-features merge bug in my TimeGPT integration so forecasts stop failing with a KeyError on exog column names after a merge that creates suffixes.

## Repo
- Working dir: `~/timegpt_v2`  
- Key files to modify:
  - `src/timegpt_v2/forecast/exogenous.py`
  - `src/timegpt_v2/forecast/timegpt_client.py`
  - Add tests in `tests/` if missing.

## Context (Observed)
- History dataframe (`y_batch`) already contains history exogs: `['spy_ret_1m','spy_vol_30m','regime_high_vol','regime_high_dispersion']`.
- `merge_history_exogs(...)` merges features again, causing pandas to create `_x`/`_y` suffixed duplicates.
- Then code calls `dropna(subset=[<unsuffixed exogs>])` and crashes with `KeyError` because the bare names no longer exist.
- Config: `strict_exog=True`, `impute_strategy='none'`.
- We also declare future exogs: `['event_earnings','event_fomc','event_cpi']`, but the future payload (`x_batch`) often only has `unique_id, ds`. We need to ensure future exogs are present at forecast timestamps.

## Goals
1) Make `merge_history_exogs` robust: no KeyError when duplicates exist; coalesce and keep the canonical bare names.
2) Ensure `x_batch` includes the declared future exogs at the forecast horizon timestamps.
3) Preserve current behavior: if `strict_exog=True` and `impute_strategy='none'`, drop rows that truly have NaNs in required exogs. If configured `impute_strategy='zero'`, fill numerics with 0 and booleans with False before the strict check.
4) Add unit tests that reproduce the bug and prove the fix.
5) Provide a minimal integration smoke run command that executes a forecast without KeyError.

## Make these code changes

### A) Patch `merge_history_exogs` to coalesce suffixed columns back to bare names
Open `src/timegpt_v2/forecast/exogenous.py` and **replace the existing `merge_history_exogs` implementation** with the following robust version that:
- Merges with controlled suffixes
- Coalesces `base` and `_feat` into the bare name
- Optionally imputes (`zero`) then enforces strict dropna on the **bare** names

```python
# src/timegpt_v2/forecast/exogenous.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional
import logging

logger = logging.getLogger("timegpt_v2.forecast")

def merge_history_exogs(
    y_df: pd.DataFrame,
    features_df: pd.DataFrame,
    exogs: List[str],
    strict: bool = True,
    impute_strategy: str = "none",
    log: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Merge historical exogenous variables into y_df and return a dataframe
    that contains the bare exog names without pandas suffixes.
    - If y_df already contains some of exogs, we coalesce with features_df values.
    - If impute_strategy == 'zero', impute numerics with 0 and booleans with False.
    - If strict is True, drop rows with NaNs in required exogs (after coalesce/impute).
    """
    lg = log or logger

    # Sanity: keys must exist
    for col in ("unique_id", "ds"):
        if col not in y_df.columns:
            raise KeyError(f"y_df missing required column '{col}'")
        if col not in features_df.columns:
            raise KeyError(f"features_df missing required column '{col}'")

    # Select only the columns we need from features to avoid accidental suffix explosion
    needed_cols = ["unique_id", "ds"] + [c for c in exogs if c in features_df.columns]
    feats = features_df[needed_cols].copy()

    # Merge with controlled suffixes to avoid _x/_y ambiguity
    merged = y_df.merge(
        feats,
        on=["unique_id", "ds"],
        how="left",
        suffixes=("", "_feat"),
    )

    # Coalesce any existing columns in y_df with the _feat columns from features_df
    for c in exogs:
        base = c
        feat = f"{c}_feat"

        if base in merged.columns and feat in merged.columns:
            # y_df value wins unless NaN, then take features
            merged[base] = merged[base].where(~merged[base].isna(), merged[feat])
            merged.drop(columns=[feat], inplace=True)
        elif feat in merged.columns and base not in merged.columns:
            # Only features had it; rename to bare name
            merged.rename(columns={feat: base}, inplace=True)
        elif base not in merged.columns:
            # Neither side had it; create it as NaN to allow downstream handling
            merged[base] = np.nan

    # Optional imputation
    if impute_strategy == "zero":
        for c in exogs:
            if c not in merged.columns:
                merged[c] = np.nan
            # Boolean or numeric fill
            if pd.api.types.is_bool_dtype(merged[c]):
                merged[c] = merged[c].fillna(False)
            else:
                merged[c] = merged[c].fillna(0.0)

    # Strict enforcement: drop rows with NaNs in bare exog names
    if strict:
        missing = [c for c in exogs if c not in merged.columns]
        if missing:
            raise KeyError(f"Missing required exogs after merge: {missing}")
        rows_before = len(merged)
        merged.dropna(subset=exogs, inplace=True)
        rows_after = len(merged)
        if rows_after < rows_before:
            lg.warning(f"Dropped {rows_before - rows_after} rows with NaNs in hist exogs {exogs}")

    return merged
```

### B) Ensure future exogs are present in `x_batch`
If there is a helper for future exogs, make it analogous. Otherwise, add this to `exogenous.py`:

```python
def merge_future_exogs(
    x_df: pd.DataFrame,
    features_df: pd.DataFrame,
    futr_exogs: List[str],
    strict: bool = True,
) -> pd.DataFrame:
    """
    Left-join future exogs onto the future frame (x_df) on ['unique_id','ds'].
    For boolean event flags, default missing to False when strict=True.
    """
    if not futr_exogs:
        return x_df

    needed_cols = ["unique_id", "ds"] + [c for c in futr_exogs if c in features_df.columns]
    feats = features_df[needed_cols].copy()
    out = x_df.merge(feats, on=["unique_id", "ds"], how="left")

    if strict:
        for c in futr_exogs:
            if c not in out.columns:
                # Create and default to False for event flags
                out[c] = False
            elif pd.api.types.is_bool_dtype(out[c]):
                out[c] = out[c].fillna(False)

    return out
```

### C) Wire the calls in `timegpt_client.py`
In `src/timegpt_v2/forecast/timegpt_client.py`:

1) Wherever we currently call `merge_history_exogs(...)`, keep the same call but now it is robust and will not KeyError on suffixes.

2) Right before sending to the API, if `futr_exog_declared` is non-empty and `x_batch` doesn’t yet have those columns, join them from `features`:

```python
# Pseudocode inside call_timegpt(...) after x_batch/y_batch are assembled and before API call:
from .exogenous import merge_future_exogs

if futr_exog_declared:
    x_batch = merge_future_exogs(
        x_df=x_batch,
        features_df=features,
        futr_exogs=futr_exog_declared,
        strict=True,   # ensure event flags present; default False if missing
    )
```

3) Optional defensive tweak to avoid duplicates right up front:
```python
# If history already contains exogs, let the merge function coalesce.
# But we can drop to reduce noise (not strictly required now).
if hist_exog_declared:
    y_batch = y_batch.drop(columns=[c for c in hist_exog_declared if c in y_batch.columns], errors="ignore")
```

### D) Unit tests
Add `tests/test_exogenous_merge.py`:

```python
import pandas as pd
from src/timegpt_v2.forecast.exogenous import merge_history_exogs, merge_future_exogs

def test_merge_history_exogs_coalesce_suffixes():
    y = pd.DataFrame({
        "unique_id":["AAPL","AAPL"],
        "ds": pd.to_datetime(["2024-06-11 14:00","2024-06-11 14:01"]),
        "y":[0.1, -0.2],
        "spy_ret_1m":[0.0, 0.0],  # already present
    })
    feats = pd.DataFrame({
        "unique_id":["AAPL","AAPL"],
        "ds": pd.to_datetime(["2024-06-11 14:00","2024-06-11 14:01"]),
        "spy_ret_1m":[0.05, None],
        "spy_vol_30m":[1.2, 1.3],
        "regime_high_vol":[False, False],
        "regime_high_dispersion":[False, False],
    })
    exogs = ["spy_ret_1m","spy_vol_30m","regime_high_vol","regime_high_dispersion"]
    out = merge_history_exogs(y, feats, exogs, strict=False, impute_strategy="none")
    # Bare names must exist, no _feat suffixes
    assert set(exogs).issubset(out.columns)
    # Coalesce y value if not NaN, else take features; here y had 0.0 so it stays
    assert out.loc[0,"spy_ret_1m"] == 0.0

def test_merge_future_exogs_adds_events():
    x = pd.DataFrame({
        "unique_id":["AAPL"],
        "ds": pd.to_datetime(["2024-06-13 18:31"]),
    })
    feats = pd.DataFrame({
        "unique_id":["AAPL"],
        "ds": pd.to_datetime(["2024-06-13 18:31"]),
        "event_earnings":[False],
        "event_fomc":[False],
        "event_cpi":[False],
    })
    out = merge_future_exogs(x, feats, ["event_earnings","event_fomc","event_cpi"], strict=True)
    assert all(col in out.columns for col in ["event_earnings","event_fomc","event_cpi"])
```

### E) Config escape hatch (optional)
If needed, add support for `impute_strategy: zero` in your yaml to be kinder to boolean flags. Don’t change defaults here, just ensure the code honors it.

## Acceptance criteria
- Running the same smoke forecast no longer raises `KeyError` in `dropna(subset=...)`.
- `y_df_exog` contains bare columns `['spy_ret_1m','spy_vol_30m','regime_high_vol','regime_high_dispersion']` with no `_x`/`_y` or `*_feat` leftovers.
- When `strict_exog=True & impute_strategy='none'`, rows with actual NaNs get dropped, but not due to column-name mismatches.
- `x_batch` includes `['event_earnings','event_fomc','event_cpi']` at forecast horizon rows.

## Commands to run locally after you generate patches
```bash
# From repo root
pytest -q

# Example smoke run (adjust to your config names)
python -m src.timegpt_v2.cli forecast   --config configs/forecast_smoke.yaml   --universe configs/universe_smoke.yaml   --run-id smoke_aapl_2024-07
```
