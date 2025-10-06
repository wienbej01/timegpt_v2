from __future__ import annotations

import logging

import pandas as pd
import pytest

from timegpt_v2.forecast.timegpt_client import TimeGPTBackend, TimeGPTClient, TimeGPTConfig
from timegpt_v2.framing.build_payloads import build_x_df_for_horizon
from timegpt_v2.utils.cache import ForecastCache


class RecordingBackend(TimeGPTBackend):
    def __init__(self) -> None:
        self.calls = 0

    def forecast(
        self,
        y: pd.DataFrame,
        x: pd.DataFrame | None,
        *,
        h: int,
        freq: str,
        quantiles: tuple[float, ...],
    ) -> pd.DataFrame:
        self.calls += 1
        rows = []
        for unique_id, frame in y.groupby("unique_id"):
            values = frame["y"].astype(float).to_numpy()
            base = float(values.mean()) if values.size else 0.0
            for idx, quantile in enumerate(quantiles):
                rows.append(
                    {
                        "unique_id": unique_id,
                        "quantile": float(quantile),
                        "value": base + (idx - 1) * 0.01,
                    }
                )
        return pd.DataFrame(rows)


def _sample_payload() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    base = pd.Timestamp("2024-01-02 14:25", tz="UTC")
    rows = []
    for minute in range(6):
        ts = base + pd.Timedelta(minutes=minute)
        for idx, symbol in enumerate(["AAPL", "MSFT"]):
            rows.append(
                {
                    "unique_id": symbol,
                    "ds": ts,
                    "y": 0.002 * (minute + idx),
                }
            )
    y_df = pd.DataFrame(rows)
    snapshot = base + pd.Timedelta(minutes=5)
    features = pd.DataFrame(
        {
            "timestamp": y_df["ds"],
            "symbol": y_df["unique_id"],
            "target_log_return_1m": y_df["y"],
        }
    )
    x_df = build_x_df_for_horizon(features, snapshot, horizon_minutes=2)
    return y_df, x_df, snapshot


def test_client_logs_and_quantiles(caplog: pytest.LogCaptureFixture) -> None:
    backend = RecordingBackend()
    cache = ForecastCache(root=None)
    logger = logging.getLogger("timegpt_v2.test_client")
    logger.setLevel(logging.INFO)

    y_df, x_df, snapshot = _sample_payload()
    client = TimeGPTClient(
        backend=backend,
        cache=cache,
        config=TimeGPTConfig(horizon=15),
        logger=logger,
    )

    with caplog.at_level(logging.INFO, logger=logger.name):
        result = client.forecast(
            y_df,
            x_df,
            snapshot_ts=snapshot,
            horizon=15,
            quantiles=(0.25, 0.5, 0.75),
        )

    assert backend.calls == 1
    expected_columns = {"q25", "q50", "q75", "forecast_ts", "snapshot_utc", "unique_id"}
    assert expected_columns.issubset(result.columns)

    for symbol in ["AAPL", "MSFT"]:
        row = result[result["unique_id"] == symbol].iloc[0]
        values = [row["q25"], row["q50"], row["q75"]]
        assert len(set(values)) > 1

    log_lines = "\n".join(caplog.messages)
    assert "Forecasted AAPL h=15 with q=[0.25, 0.5, 0.75]" in log_lines

    with caplog.at_level(logging.INFO, logger=logger.name):
        second = client.forecast(
            y_df,
            x_df,
            snapshot_ts=snapshot,
            horizon=15,
            quantiles=(0.25, 0.5, 0.75),
        )

    assert backend.calls == 1
    pd.testing.assert_frame_equal(result, second)
    assert any("[cache]" in message for message in caplog.messages)
