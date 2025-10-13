import pandas as pd
from timegpt_v2.forecast.exogenous import merge_history_exogs, merge_future_exogs

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
