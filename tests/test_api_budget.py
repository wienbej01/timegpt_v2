"""Tests for API budget manager."""

from __future__ import annotations

import tempfile
from datetime import date, datetime
from pathlib import Path

import pytest

from timegpt_v2.utils.api_budget import BudgetConfig, BudgetManager, BudgetState


@pytest.fixture
def temp_ledger_path() -> Path:
    """Create a temporary file for budget ledger testing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        yield Path(f.name)
    f.close()


def test_budget_config() -> None:
    """Test budget configuration."""
    config = BudgetConfig(per_run=10, per_day=50, cooldown_sec=5)
    assert config.per_run == 10
    assert config.per_day == 50
    assert config.cooldown_sec == 5


def test_budget_state_serialization() -> None:
    """Test budget state serialization."""
    now = datetime.utcnow()
    state = BudgetState(
        calls_today=5,
        calls_this_run=2,
        last_call=now,
        today=date.today(),
    )

    data = state.to_dict()
    restored = BudgetState.from_dict(data)

    assert restored.calls_today == state.calls_today
    assert restored.calls_this_run == state.calls_this_run
    assert restored.today == state.today
    assert restored.last_call == state.last_call


def test_budget_manager_initialization(temp_ledger_path: Path) -> None:
    """Test budget manager initialization."""
    config = BudgetConfig(per_run=10, per_day=50, cooldown_sec=5)
    manager = BudgetManager(config, temp_ledger_path)

    assert manager.calls_this_run == 0
    assert manager.calls_today == 0
    assert manager.remaining_per_run == 10
    assert manager.remaining_today == 50


def test_budget_manager_can_make_call(temp_ledger_path: Path) -> None:
    """Test call permission checking."""
    config = BudgetConfig(per_run=2, per_day=5, cooldown_sec=5)
    manager = BudgetManager(config, temp_ledger_path)

    # Should allow calls initially
    assert manager.can_make_call()
    assert manager.can_make_call(2)

    # Record calls
    manager.record_call(1)
    assert manager.calls_this_run == 1
    assert manager.remaining_per_run == 1

    # Should still allow one more
    assert manager.can_make_call()

    # Record another call
    manager.record_call(1)
    assert manager.calls_this_run == 2
    assert not manager.can_make_call()  # At limit


def test_budget_manager_enforces_limits(temp_ledger_path: Path) -> None:
    """Test that budget limits are enforced."""
    config = BudgetConfig(per_run=1, per_day=2, cooldown_sec=5)
    manager = BudgetManager(config, temp_ledger_path)

    # First call allowed
    assert manager.can_make_call()
    manager.record_call()

    # Second call should be blocked (per_run limit)
    assert not manager.can_make_call()

    # Reset run counter
    manager.reset_run()
    assert manager.calls_this_run == 0
    assert manager.calls_today == 1  # Daily counter persists

    # Should allow one more call
    assert manager.can_make_call()
    manager.record_call()

    # Now at daily limit
    assert not manager.can_make_call()


def test_budget_manager_persistence(temp_ledger_path: Path) -> None:
    """Test budget state persistence."""
    config = BudgetConfig(per_run=10, per_day=50, cooldown_sec=5)

    # Create manager and record calls
    manager1 = BudgetManager(config, temp_ledger_path)
    manager1.record_call(3)
    manager1.record_call(2)

    # Create new manager instance - should load persisted state
    manager2 = BudgetManager(config, temp_ledger_path)
    assert manager2.calls_this_run == 5
    assert manager2.calls_today == 5


def test_budget_manager_corrupt_ledger_handling(temp_ledger_path: Path) -> None:
    """Test handling of corrupt ledger files."""
    config = BudgetConfig(per_run=10, per_day=50, cooldown_sec=5)

    # Write invalid JSON
    temp_ledger_path.write_text("invalid json", encoding="utf-8")

    # Should initialize with defaults without crashing
    manager = BudgetManager(config, temp_ledger_path)
    assert manager.calls_this_run == 0
    assert manager.calls_today == 0
