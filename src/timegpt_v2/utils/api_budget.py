"""API budget manager for TimeGPT."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class APIBudget:
    """Tracks API usage against a budget."""

    per_run: int = 25
    per_day: int = 100
    cooldown_sec: int = 1
    _run_calls: int = field(init=False, default=0)
    _day_calls: int = field(init=False, default=0)
    _last_call_ts: float = field(init=False, default=0.0)

    def can_call(self, num_calls: int = 1) -> bool:
        """Check if a call is within budget."""
        if self._run_calls + num_calls > self.per_run:
            return False
        if self._day_calls + num_calls > self.per_day:
            return False
        return True

    def record_call(self, num_calls: int = 1) -> None:
        """Record an API call."""
        self._run_calls += num_calls
        self._day_calls += num_calls
        self._last_call_ts = time.time()

    def cooldown(self) -> None:
        """Wait for the cooldown period to elapse."""
        if self.cooldown_sec > 0:
            elapsed = time.time() - self._last_call_ts
            if elapsed < self.cooldown_sec:
                time.sleep(self.cooldown_sec - elapsed)