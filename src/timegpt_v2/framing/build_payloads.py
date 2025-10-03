"""Payload construction utilities for TimeGPT requests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def build_payload(frame: pd.DataFrame) -> Mapping[str, Any]:
    """Stub payload builder."""
    return {"rows": len(frame)}
