"""Logging utilities."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs" / "app.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=log_path, filemode="a")
    return logging.getLogger("timegpt_v2")
