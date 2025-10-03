"""Data contracts definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ColumnContract:
    name: str
    dtype: str
    required: bool = True


@dataclass(frozen=True)
class DataContract:
    columns: Sequence[ColumnContract]
