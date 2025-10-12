from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .model import ForecastExogConfig


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if isinstance(loaded, dict):
        return loaded
    raise TypeError(f"{path} must contain a mapping")


def load_forecast_exog_config(config_path: Path) -> ForecastExogConfig:
    """Load the exogenous feature configuration from a YAML file."""
    config = _load_yaml(config_path)
    exog_config_raw = config.get("exog", {})

    exog_name_map = exog_config_raw.get("exog_name_map", {})
    hist_exog_list_raw = exog_config_raw.get("hist_exog_list", [])
    futr_exog_list_raw = exog_config_raw.get("futr_exog_list", [])

    hist_exog_list = [exog_name_map.get(name, name) for name in hist_exog_list_raw]
    futr_exog_list = [exog_name_map.get(name, name) for name in futr_exog_list_raw]

    return ForecastExogConfig(
        use_exogs=exog_config_raw.get("use_exogs", False),
        hist_exog_list_raw=hist_exog_list_raw,
        futr_exog_list_raw=futr_exog_list_raw,
        hist_exog_list=hist_exog_list,
        futr_exog_list=futr_exog_list,
        strict_exog=exog_config_raw.get("strict_exog", False),
        exog_name_map=exog_name_map,
        impute_strategy=exog_config_raw.get("impute_strategy", "none"),
    )
