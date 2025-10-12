from __future__ import annotations

from typing import Dict, List

from pydantic import Field
from pydantic.dataclasses import dataclass


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
