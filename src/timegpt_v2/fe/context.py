"""Context managers and helpers for feature pipelines."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class FeatureContext:
    symbols: Iterable[str]
