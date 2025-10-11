"""Tests for API budget."""

from timegpt_v2.utils.api_budget import APIBudget


def test_api_budget():
    """Test API budget."""
    budget = APIBudget(per_run=10, per_day=20)

    assert budget.can_call(1)
    budget.record_call(1)
    assert budget.can_call(10) is False

    budget = APIBudget(per_run=100, per_day=10)
    assert budget.can_call(1)
    budget.record_call(1)
    assert budget.can_call(10) is False

    budget = APIBudget(per_run=5, per_day=5)
    assert budget.can_call(6) is False