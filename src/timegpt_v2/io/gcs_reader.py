"""GCS reader utilities for intraday bar data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ReaderConfig:
    bucket: str
    template: str


class GCSReader:
    """Placeholder implementation until Sprint 1."""

    def __init__(self, config: ReaderConfig, *, client: Any | None = None) -> None:
        self._config = config
        self._client = client

    def read_symbol_month(self, path: Path) -> pd.DataFrame:
        """Read a symbol/month parquet file. Stub for Sprint 0."""
        raise NotImplementedError
