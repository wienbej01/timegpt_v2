"""Data contracts and policy configuration for data quality checks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnContract:
    """Defines the expected column schema."""

    name: str
    dtype: str
    required: bool = True


@dataclass(frozen=True)
class DataContract:
    """Collection of expected columns for the dataset."""

    columns: tuple[ColumnContract, ...]

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns if column.required)


DEFAULT_CONTRACT = DataContract(
    columns=(
        ColumnContract("timestamp", "datetime64[ns, America/New_York]"),
        ColumnContract("symbol", "string"),
        ColumnContract("open", "float"),
        ColumnContract("high", "float"),
        ColumnContract("low", "float"),
        ColumnContract("close", "float"),
        ColumnContract("volume", "int"),
        ColumnContract("adj_open", "float"),
        ColumnContract("adj_high", "float"),
        ColumnContract("adj_low", "float"),
        ColumnContract("adj_close", "float"),
    ),
)


@dataclass(frozen=True)
class DataQualityPolicy:
    """Operational guardrails for data validation."""

    rth_min_pct: float = 0.95
    drop_days_with_sustained_ffill: bool = True
    hard_fail_on_schema: bool = True
    require_adjusted: bool = True
    sustained_ffill_run: int = 5

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> DataQualityPolicy:
        payload = payload or {}
        allowed_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in payload.items() if k in allowed_keys}
        return cls(**filtered)  # type: ignore[arg-type]


def merge_contract(base: DataContract, extras: Iterable[ColumnContract]) -> DataContract:
    columns = tuple(base.columns) + tuple(extras)
    return DataContract(columns=columns)
