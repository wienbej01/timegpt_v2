"""Event handling utilities (stub)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Event:
    name: str
    timestamp: datetime
