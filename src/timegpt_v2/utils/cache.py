"""Cache helpers (stub)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheKey:
    symbol: str
    snapshot: str
    horizon: int
    quantiles: tuple[float, ...]


class ForecastCache:
    def get(self, key: CacheKey) -> Mapping[str, Any] | None:
        return None

    def put(self, key: CacheKey, value: Mapping[str, Any]) -> None:
        raise NotImplementedError
