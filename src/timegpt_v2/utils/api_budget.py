"""API budget manager with hard stops and JSON ledger."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BudgetConfig:
    """Configuration for API budget limits."""

    per_run: int
    per_day: int
    cooldown_sec: int


@dataclass
class BudgetState:
    """Current budget state."""

    calls_today: int
    calls_this_run: int
    last_call: datetime | None
    today: date

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls_today": self.calls_today,
            "calls_this_run": self.calls_this_run,
            "last_call": self.last_call.isoformat() + "Z" if self.last_call else None,
            "today": self.today.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgetState:
        last_call = None
        if data.get("last_call"):
            last_call_str = str(data["last_call"]).rstrip("Z")
            last_call = datetime.fromisoformat(last_call_str)
        return cls(
            calls_today=int(data["calls_today"]),
            calls_this_run=int(data["calls_this_run"]),
            last_call=last_call,
            today=date.fromisoformat(str(data["today"])),
        )


class BudgetManager:
    """Manages API call budget with hard stops."""

    def __init__(
        self,
        config: BudgetConfig,
        ledger_path: Path,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.ledger_path = ledger_path
        self.logger = logger or logging.getLogger(__name__)
        self._state = self._load_state()

    def _load_state(self) -> BudgetState:
        """Load budget state from ledger file."""
        if not self.ledger_path.exists():
            return BudgetState(calls_today=0, calls_this_run=0, last_call=None, today=date.today())
        try:
            data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            state = BudgetState.from_dict(data)
            # Reset if day changed
            if state.today != date.today():
                state = BudgetState(calls_today=0, calls_this_run=0, last_call=None, today=date.today())
            return state
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            self.logger.warning("Failed to load budget state: %s", exc)
            return BudgetState(calls_today=0, calls_this_run=0, last_call=None, today=date.today())

    def _save_state(self) -> None:
        """Save budget state to ledger file."""
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            self.ledger_path.write_text(json.dumps(self._state.to_dict(), indent=2), encoding="utf-8")
        except OSError as exc:
            self.logger.warning("Failed to save budget state: %s", exc)

    def can_make_call(self, num_calls: int = 1) -> bool:
        """Check if a call can be made without exceeding budget."""
        if self._state.calls_this_run + num_calls > self.config.per_run:
            return False
        if self._state.calls_today + num_calls > self.config.per_day:
            return False
        return True

    def record_call(self, num_calls: int = 1) -> None:
        """Record a successful API call."""
        now = datetime.utcnow()
        self._state.calls_this_run += num_calls
        self._state.calls_today += num_calls
        self._state.last_call = now
        self._save_state()

    def reset_run(self) -> None:
        """Reset per-run counter (for new runs)."""
        self._state.calls_this_run = 0
        self._save_state()

    @property
    def calls_this_run(self) -> int:
        return self._state.calls_this_run

    @property
    def calls_today(self) -> int:
        return self._state.calls_today

    @property
    def remaining_per_run(self) -> int:
        return max(0, self.config.per_run - self._state.calls_this_run)

    @property
    def remaining_today(self) -> int:
        return max(0, self.config.per_day - self._state.calls_today)