"""Policy-blind retry outcome simulation for Recoverly evaluation.

Policies choose an action and retry time. This module alone maps that chosen
time to a success probability and deterministic simulated outcome. It has no
policy parameter and must never be imported by the retry-policy module.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import hashlib
import logging
from typing import Any


LOGGER = logging.getLogger(__name__)
OUTCOME_RANDOM_SEED = 42

BASE_RETRY_SUCCESS_PROBABILITIES = {
    "insufficient_funds": 0.35,
    "bank_timeout": 0.55,
    "expired_card": 0.01,
    "gateway_error": 0.50,
    "issuer_decline": 0.28,
    "threeds_dropoff": 0.15,
}
ATTEMPT_DECAY_FACTOR = 0.65
INSUFFICIENT_FUNDS_SALARY_MULTIPLIER = 2.0
EXPIRED_CARD_UPDATED_PROBABILITY = 0.60
MAX_SUCCESS_PROBABILITY = 0.95


class OutcomeSimulationError(ValueError):
    """Raised when a transaction cannot be evaluated by the outcome model."""


def retry_success_probability(transaction: Mapping[str, Any], retry_at: datetime) -> float:
    """Return a policy-independent retry success probability.

    The model uses only transaction properties and the proposed retry time. A
    caller that cannot supply the required fields receives an explicit error so
    evaluation can log and exclude the row instead of silently zero-filling it.
    """

    reason, attempt_number = _validate_inputs(transaction, retry_at)

    if reason == "expired_card" and transaction.get("card_updated", False):
        base_probability = EXPIRED_CARD_UPDATED_PROBABILITY
    else:
        base_probability = BASE_RETRY_SUCCESS_PROBABILITIES[reason]

    probability = base_probability * (ATTEMPT_DECAY_FACTOR ** (attempt_number - 1))
    if reason == "insufficient_funds" and _is_salary_window(retry_at):
        probability *= INSUFFICIENT_FUNDS_SALARY_MULTIPLIER

    return min(probability, MAX_SUCCESS_PROBABILITY)


def simulate_retry_outcome(transaction: Mapping[str, Any], retry_at: datetime) -> bool:
    """Return a deterministic sampled recovery result for a transaction/time pair.

    The pseudo-random draw is derived from the fixed seed plus ``txn_id``,
    ``retry_at``, and ``attempt_number``. Re-running evaluation therefore gives
    the same result for the same candidate retry, without a global RNG whose
    ordering could affect results.
    """

    _, attempt_number = _validate_inputs(transaction, retry_at)
    probability = retry_success_probability(transaction, retry_at)
    return _stable_uniform(str(transaction["txn_id"]), retry_at, attempt_number) < probability


def _validate_inputs(transaction: Mapping[str, Any], retry_at: datetime) -> tuple[str, int]:
    if not isinstance(retry_at, datetime):
        _raise_unsimulatable("retry_at must be a datetime")

    missing_fields = [
        field
        for field in ("txn_id", "failure_reason", "attempt_number")
        if transaction.get(field) is None
    ]
    if missing_fields:
        _raise_unsimulatable(f"missing required field(s): {', '.join(missing_fields)}")

    reason = str(transaction["failure_reason"])
    if reason not in BASE_RETRY_SUCCESS_PROBABILITIES:
        _raise_unsimulatable(f"unsupported failure_reason: {reason}")

    attempt_number = transaction["attempt_number"]
    if isinstance(attempt_number, bool) or not isinstance(attempt_number, int) or attempt_number < 1:
        _raise_unsimulatable("attempt_number must be an integer greater than or equal to 1")

    return reason, attempt_number


def _raise_unsimulatable(message: str) -> None:
    LOGGER.warning("Outcome cannot be simulated: %s", message)
    raise OutcomeSimulationError(message)


def _is_salary_window(retry_at: datetime) -> bool:
    return 1 <= retry_at.day <= 5


def _stable_uniform(txn_id: str, retry_at: datetime, attempt_number: int) -> float:
    payload = f"{OUTCOME_RANDOM_SEED}|{txn_id}|{retry_at.isoformat()}|{attempt_number}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") / 2**64
