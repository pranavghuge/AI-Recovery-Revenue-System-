"""Tests for the standalone, policy-blind retry outcome model."""

from datetime import datetime

import pytest

from backend.data.outcome_model import (
    OutcomeSimulationError,
    retry_success_probability,
    simulate_retry_outcome,
)


def _transaction(**overrides: object) -> dict[str, object]:
    transaction: dict[str, object] = {
        "txn_id": "txn-001",
        "failure_reason": "insufficient_funds",
        "attempt_number": 1,
    }
    transaction.update(overrides)
    return transaction


def test_probability_is_policy_blind() -> None:
    retry_at = datetime(2026, 2, 3, 9, 0)
    baseline_transaction = _transaction(policy="baseline")
    smart_transaction = _transaction(policy="smart")

    assert retry_success_probability(baseline_transaction, retry_at) == retry_success_probability(
        smart_transaction, retry_at
    )


def test_probability_decays_with_attempt_number() -> None:
    retry_at = datetime(2026, 2, 10, 9, 0)

    first_attempt = retry_success_probability(_transaction(attempt_number=1), retry_at)
    second_attempt = retry_success_probability(_transaction(attempt_number=2), retry_at)

    assert second_attempt < first_attempt


def test_insufficient_funds_has_salary_window_uplift() -> None:
    transaction = _transaction()

    salary_window_probability = retry_success_probability(transaction, datetime(2026, 2, 3, 9, 0))
    non_salary_probability = retry_success_probability(transaction, datetime(2026, 2, 10, 9, 0))

    assert salary_window_probability > non_salary_probability


def test_expired_card_has_near_zero_probability_without_card_update() -> None:
    retry_at = datetime(2026, 2, 10, 9, 0)
    expired_card = _transaction(failure_reason="expired_card")
    updated_card = _transaction(failure_reason="expired_card", card_updated=True)

    assert retry_success_probability(expired_card, retry_at) <= 0.01
    assert retry_success_probability(updated_card, retry_at) > retry_success_probability(expired_card, retry_at)


def test_outcome_sampling_is_deterministic_for_same_inputs() -> None:
    transaction = _transaction(txn_id="txn-deterministic", attempt_number=2)
    retry_at = datetime(2026, 2, 3, 9, 0)

    assert simulate_retry_outcome(transaction, retry_at) == simulate_retry_outcome(transaction, retry_at)


def test_unsimulatable_transaction_logs_and_raises(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(OutcomeSimulationError, match="missing required field"):
        retry_success_probability({"failure_reason": "bank_timeout", "attempt_number": 1}, datetime(2026, 2, 3))

    assert "Outcome cannot be simulated" in caplog.text
