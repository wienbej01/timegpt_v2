"""Target scaling utilities for TimeGPT forecasting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TargetScalingConfig:
    """Configuration describing how forecast targets are scaled."""

    mode: str = "log_return"
    bp_factor: float = 10_000.0
    volatility_column: str = "vol_ewm_60m"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object] | None) -> TargetScalingConfig:
        if payload is None:
            return cls()
        mode_raw = payload.get("mode", cls.mode)
        mode = str(mode_raw).lower()

        bp_raw = payload.get("bp_factor", cls.bp_factor)
        if isinstance(bp_raw, str):
            bp_factor = float(bp_raw.strip())
        elif isinstance(bp_raw, (int, float)):
            bp_factor = float(bp_raw)
        else:
            raise TypeError("bp_factor must be a number or numeric string")

        vol_raw = payload.get("volatility_column", cls.volatility_column)
        volatility_column = str(vol_raw)
        allowed_modes = {"log_return", "basis_point", "volatility_z", "log_return_15m"}
        if mode not in allowed_modes:
            allowed_str = ", ".join(sorted(allowed_modes))
            raise ValueError(f"target.mode must be one of: {allowed_str}")
        return cls(mode=mode, bp_factor=bp_factor, volatility_column=volatility_column)

    def to_metadata(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "bp_factor": self.bp_factor,
            "volatility_column": self.volatility_column,
        }


class TargetScaler:
    """Apply scaling transformations to forecast targets."""

    def __init__(self, config: TargetScalingConfig) -> None:
        self._config = config

    @property
    def target_column(self) -> str:
        mapping = {
            "log_return": "target_log_return_1m",
            "log_return_15m": "target_log_return_15m",
            "basis_point": "target_bp_ret_1m",
            "volatility_z": "target_z_ret_1m",
        }
        return mapping[self._config.mode]

    @property
    def metadata(self) -> dict[str, object]:
        return self._config.to_metadata()

    def inverse_quantiles(
        self,
        forecasts: pd.DataFrame,
        *,
        features: pd.DataFrame,
        quantile_columns: Sequence[str],
    ) -> pd.DataFrame:
        """Convert quantile forecasts back to log-return space."""
        if self._config.mode in {"log_return", "log_return_15m"}:
            return forecasts

        updated = forecasts.copy()
        if self._config.mode == "basis_point":
            factor = self._config.bp_factor
            for column in quantile_columns:
                updated[column] = updated[column] / factor
            return updated

        volatility_column = self._config.volatility_column
        if volatility_column not in features.columns:
            raise ValueError(
                f"Volatility column '{volatility_column}' missing from feature matrix."
            )
        label_column = self.label_timestamp_column
        if label_column not in features.columns:
            raise ValueError(
                f"Label timestamp column '{label_column}' missing from feature matrix."
            )
        scale_frame = (
            features[[label_column, "symbol", volatility_column]]
            .rename(
                columns={
                    label_column: "forecast_ts",
                    "symbol": "unique_id",
                    volatility_column: "scale",
                }
            )
            .dropna(subset=["forecast_ts"])
        )
        scale_frame["forecast_ts"] = pd.to_datetime(scale_frame["forecast_ts"], utc=True)
        scale_frame["unique_id"] = scale_frame["unique_id"].astype(str)

        updated["forecast_ts"] = pd.to_datetime(updated["forecast_ts"], utc=True)
        updated["unique_id"] = updated["unique_id"].astype(str)
        updated = updated.merge(scale_frame, on=["forecast_ts", "unique_id"], how="left")
        if updated["scale"].isna().any():
            missing_rows = updated.loc[updated["scale"].isna(), ["forecast_ts", "unique_id"]]
            raise ValueError(
                "Missing volatility scale values for some forecasts. "
                f"Examples:\n{missing_rows.head()}"
            )
        for column in quantile_columns:
            updated[column] = updated[column] * updated["scale"]
        updated.drop(columns="scale", inplace=True)
        return updated

    @property
    def label_timestamp_column(self) -> str:
        mapping = {
            "log_return": "label_timestamp",
            "log_return_15m": "label_timestamp_15m",
            "basis_point": "label_timestamp",
            "volatility_z": "label_timestamp",
        }
        return mapping[self._config.mode]


__all__ = ["TargetScaler", "TargetScalingConfig"]
