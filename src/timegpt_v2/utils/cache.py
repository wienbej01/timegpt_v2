"""Simple forecast cache with optional on-disk persistence."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CacheKey:
    """Key describing a forecast entry."""

    symbol: str
    trade_date: str
    snapshot: str
    horizon: int
    quantiles: tuple[float, ...]
    model: str = "timegpt-1"
    levels: tuple[int, ...] = ()
    features_hash: int = 0


class ForecastCache:
    """Forecast cache storing entries in memory and optionally on disk."""

    def __init__(self, root: Path | None = None, *, logger: logging.Logger | None = None) -> None:
        self._root = root
        self._logger = logger or logging.getLogger(__name__)
        self._memory: dict[CacheKey, Mapping[str, Any]] = {}
        if self._root is not None:
            self._root.mkdir(parents=True, exist_ok=True)

    def get(self, key: CacheKey) -> Mapping[str, Any] | None:
        """Return cached payload for *key* if present."""

        if key in self._memory:
            return self._memory[key]
        if self._root is None:
            return None
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - logged and ignored
            self._logger.warning("Cache read failed for %s: %s", path, exc)
            return None
        self._memory[key] = loaded
        return loaded

    def put(self, key: CacheKey, value: Mapping[str, Any]) -> None:
        """Persist *value* for *key* in memory and disk (best effort)."""

        self._memory[key] = value
        if self._root is None:
            return
        path = self._path_for(key)
        try:
            path.write_text(json.dumps(value), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - logged and tolerated
            self._logger.warning("Cache persistence failed for %s: %s", path, exc)

    def _path_for(self, key: CacheKey) -> Path:
        quantile_signature = ",".join(map(str, key.quantiles))
        level_signature = ",".join(map(str, key.levels)) if key.levels else ""
        payload = (
            f"{key.symbol}|{key.trade_date}|{key.snapshot}|{key.horizon}|"
            f"{quantile_signature}|{key.model}|{level_signature}"
        )
        digest = sha256(payload.encode("utf-8")).hexdigest()
        filename = f"{key.symbol}_{digest}.json"
        if self._root is None:  # pragma: no cover - guarded by caller
            raise RuntimeError("Cache root not configured")
        return self._root / filename


__all__ = ["CacheKey", "ForecastCache"]
