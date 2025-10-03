"""Trading rules stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TradingSignal:
    signal: Any
