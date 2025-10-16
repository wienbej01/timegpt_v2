from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class TradingWindowConfig:
    """Configuration for trading window enforcement and history backfill."""

    # Trading window: the only period when snapshots/trades can occur
    start: Optional[date] = None
    end: Optional[date] = None

    # History backfill: optional warmup loaded before trading_window.start for model context
    history_backfill_days: int = 0

    # Enforcement policy
    enforce_trading_window: bool = True


@dataclass
class ForecastExogConfig:
    """Configuration for exogenous features in the forecast."""

    use_exogs: bool = False
    hist_exog_list_raw: List[str] = Field(default_factory=list)
    futr_exog_list_raw: List[str] = Field(default_factory=list)
    hist_exog_list: List[str] = Field(default_factory=list)
    futr_exog_list: List[str] = Field(default_factory=list)
    strict_exog: bool = False
    exog_name_map: Dict[str, str] = Field(default_factory=dict)
    impute_strategy: str = "none"
