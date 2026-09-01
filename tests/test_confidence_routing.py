"""Tests for conservative low-confidence recovery routing."""

from datetime import datetime

import pytest

from backend.retry_policy import CONFIDENCE_MANUAL_REVIEW_THRESHOLD, decide_action


def _context() -> dict[str, object]:
    return {
        "amount": 20_000.0,
        "attempt_number": 1,
        "decision_at": datetime(2026, 1, 30, 10, 0),
        "scheduled_retry_ats": [],
    }


@pytest.mark.parametrize("reason", ["insufficient_funds", "expired_card"])
def test_low_confidence_forces_manual_review_regardless_of_reason(reason: str) -> None:
    decision = decide_action(reason, 0.54, _context(), policy="smart")

    assert decision == {
        "action": "escalate_manual_review",
        "retry_at": None,
        "expected_value": 0.0,
    }


def test_exact_threshold_does_not_force_manual_review() -> None:
    decision = decide_action("expired_card", CONFIDENCE_MANUAL_REVIEW_THRESHOLD, _context(), policy="smart")

    assert decision["action"] == "notify_update_card"


def test_high_confidence_preserves_reason_based_action() -> None:
    decision = decide_action("insufficient_funds", 0.55 + 0.01, _context(), policy="smart")

    assert decision["action"] == "retry_scheduled"


@pytest.mark.parametrize("confidence", [None, -0.01, 1.01])
def test_invalid_confidence_is_rejected(confidence: object) -> None:
    with pytest.raises(ValueError, match="confidence"):
        decide_action("bank_timeout", confidence, _context(), policy="smart")
