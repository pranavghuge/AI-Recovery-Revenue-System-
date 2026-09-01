"""Tests for Recoverly's policy-isolated retry decision engine."""

import ast
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from backend.policy_belief_table import estimate_success_probability
from backend.retry_policy import RETRY_COST_INR, decide_action


DECISION_AT = datetime(2026, 1, 30, 10, 0)


def _context(**overrides: object) -> dict[str, object]:
    context: dict[str, object] = {
        "amount": 1_000.0,
        "attempt_number": 1,
        "decision_at": DECISION_AT,
        "scheduled_retry_ats": [],
    }
    context.update(overrides)
    return context


def test_baseline_retries_after_exactly_four_hours_without_reasoning() -> None:
    decision = decide_action("issuer_decline", 0.99, _context(), policy="baseline")

    assert decision == {
        "action": "retry_scheduled",
        "retry_at": DECISION_AT + timedelta(hours=4),
        "expected_value": 400.0,
    }


def test_smart_policy_selects_salary_window_for_insufficient_funds() -> None:
    decision = decide_action("insufficient_funds", 0.99, _context(), policy="smart")

    assert decision["action"] == "retry_scheduled"
    assert decision["retry_at"] == datetime(2026, 2, 1, 9, 0)
    assert decision["expected_value"] == 820.0


def test_smart_policy_expected_value_uses_its_belief_table_and_retry_cost() -> None:
    amount = 100.0
    decision = decide_action("bank_timeout", 0.99, _context(amount=amount), policy="smart")
    expected_probability = estimate_success_probability("bank_timeout", "retry_now", 1)

    assert decision["action"] == "retry_now"
    assert decision["expected_value"] == round(expected_probability * amount, 2)
    assert decision["expected_value"] - RETRY_COST_INR > 0


@pytest.mark.parametrize(
    ("reason", "context", "expected_action"),
    [
        ("expired_card", _context(), "notify_update_card"),
        ("expired_card", _context(attempt_number=2), "no_action"),
        ("bank_timeout", _context(attempt_number=2), "no_action"),
        ("bank_timeout", _context(amount=1), "no_action"),
        (
            "bank_timeout",
            _context(
                scheduled_retry_ats=[
                    DECISION_AT,
                    DECISION_AT + timedelta(hours=4),
                    DECISION_AT + timedelta(hours=24),
                    datetime(2026, 2, 1, 9, 0),
                ]
            ),
            "no_action",
        ),
    ],
)
def test_policy_safety_boundaries(reason: str, context: dict[str, object], expected_action: str) -> None:
    decision = decide_action(reason, 0.99, context, policy="smart")

    assert decision["action"] == expected_action
    if reason == "expired_card":
        assert decision["action"] not in {"retry_now", "retry_scheduled"}


def test_policy_is_pure_and_uses_passed_context_for_duplicate_prevention() -> None:
    context = _context()

    assert decide_action("gateway_error", 0.9, context) == decide_action("gateway_error", 0.9, context)
    duplicate_context = _context(scheduled_retry_ats=[DECISION_AT + timedelta(hours=4)])
    assert decide_action("gateway_error", 0.9, duplicate_context)["retry_at"] != DECISION_AT + timedelta(hours=4)


def test_retry_policy_does_not_import_outcome_model() -> None:
    policy_path = Path(__file__).parents[1] / "backend" / "retry_policy.py"
    tree = ast.parse(policy_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    )

    assert not any("outcome_model" in module for module in imported_modules)
