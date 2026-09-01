"""Stateless baseline and smart recovery-policy decisions.

This module relies exclusively on policy_belief_table.py for smart-policy
estimates. Outcome simulation is deliberately isolated in the data layer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from typing import Any

from backend.policy_belief_table import estimate_success_probability


RETRY_COST_INR = 5.0
MAX_RETRY_ATTEMPTS = 2
BASELINE_RETRY_DELAY = timedelta(hours=4)
BASELINE_SUCCESS_ESTIMATE = 0.40
SALARY_RETRY_HOUR = 9
VALID_POLICIES = frozenset({"baseline", "smart"})
VALID_ACTIONS = frozenset(
    {
        "retry_now",
        "retry_scheduled",
        "notify_update_card",
        "escalate_manual_review",
        "no_action",
    }
)


def decide_action(reason: str, confidence: float, txn_context: dict, policy: str = "smart") -> dict:
    """Choose a compliant recovery action without mutating or storing state.

    ``confidence`` belongs to the fixed module contract. T07 applies its
    confidence-routing threshold; T06 intentionally leaves it unchanged here.
    The result's ``expected_value`` is the gross expected recovered ₹ amount
    (probability × amount). A ₹5 retry cost is used only to compare a retry
    against the ``no_action`` utility of ₹0.
    """

    del confidence
    if policy not in VALID_POLICIES:
        raise ValueError(f"Unsupported policy: {policy}")

    decision_at = _decision_at(txn_context)
    amount = _amount(txn_context)
    attempt_number = _attempt_number(txn_context)

    if attempt_number >= MAX_RETRY_ATTEMPTS:
        return _decision("no_action", None, 0.0)
    if reason == "expired_card":
        return _decision("notify_update_card", None, 0.0)

    if policy == "baseline":
        retry_at = decision_at + BASELINE_RETRY_DELAY
        expected_value = BASELINE_SUCCESS_ESTIMATE * amount
        return _retry_or_no_action("retry_scheduled", retry_at, expected_value, txn_context)

    candidates = _smart_candidates(reason, attempt_number, amount, decision_at)
    viable_candidates = [
        candidate for candidate in candidates if not _is_duplicate_retry_at(candidate["retry_at"], txn_context)
    ]
    if not viable_candidates:
        return _decision("no_action", None, 0.0)

    best_candidate = max(viable_candidates, key=lambda candidate: candidate["utility"])
    return _retry_or_no_action(
        best_candidate["action"], best_candidate["retry_at"], best_candidate["expected_value"], txn_context
    )


def _smart_candidates(
    reason: str, attempt_number: int, amount: float, decision_at: datetime
) -> list[dict[str, Any]]:
    candidate_windows = (
        ("retry_now", "retry_now", decision_at),
        ("retry_4h", "retry_scheduled", decision_at + timedelta(hours=4)),
        ("retry_24h", "retry_scheduled", decision_at + timedelta(hours=24)),
        ("salary_window", "retry_scheduled", _next_salary_window(decision_at)),
    )
    candidates = []
    for belief_window, action, retry_at in candidate_windows:
        probability = estimate_success_probability(reason, belief_window, attempt_number)
        expected_value = probability * amount
        candidates.append(
            {
                "action": action,
                "retry_at": retry_at,
                "expected_value": expected_value,
                "utility": expected_value - RETRY_COST_INR,
            }
        )
    return candidates


def _retry_or_no_action(
    action: str, retry_at: datetime, expected_value: float, txn_context: Mapping[str, Any]) -> dict:
    if expected_value - RETRY_COST_INR <= 0 or _is_duplicate_retry_at(retry_at, txn_context):
        return _decision("no_action", None, 0.0)
    return _decision(action, retry_at, expected_value)


def _decision(action: str, retry_at: datetime | None, expected_value: float) -> dict:
    return {
        "action": action,
        "retry_at": retry_at,
        "expected_value": round(expected_value, 2),
    }


def _decision_at(txn_context: Mapping[str, Any]) -> datetime:
    value = txn_context.get("decision_at", txn_context.get("timestamp"))
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError("decision_at must be an ISO datetime") from error
    raise ValueError("txn_context requires a decision_at datetime")


def _amount(txn_context: Mapping[str, Any]) -> float:
    try:
        amount = float(txn_context["amount"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("txn_context requires a numeric amount") from error
    if amount < 0:
        raise ValueError("amount must be non-negative")
    return amount


def _attempt_number(txn_context: Mapping[str, Any]) -> int:
    value = txn_context.get("attempt_number", 1)
    try:
        attempt_number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("attempt_number must be an integer") from error
    if isinstance(value, bool) or attempt_number < 1:
        raise ValueError("attempt_number must be an integer greater than or equal to 1")
    return attempt_number


def _next_salary_window(decision_at: datetime) -> datetime:
    same_day_candidate = decision_at.replace(hour=SALARY_RETRY_HOUR, minute=0, second=0, microsecond=0)
    if 1 <= decision_at.day <= 5 and decision_at < same_day_candidate:
        return same_day_candidate
    if decision_at.month == 12:
        return decision_at.replace(year=decision_at.year + 1, month=1, day=1, hour=SALARY_RETRY_HOUR, minute=0, second=0, microsecond=0)
    return decision_at.replace(month=decision_at.month + 1, day=1, hour=SALARY_RETRY_HOUR, minute=0, second=0, microsecond=0)


def _is_duplicate_retry_at(retry_at: datetime, txn_context: Mapping[str, Any]) -> bool:
    scheduled_times = txn_context.get("scheduled_retry_ats", ())
    if isinstance(scheduled_times, (str, datetime)):
        scheduled_times = (scheduled_times,)
    if not isinstance(scheduled_times, Iterable):
        raise ValueError("scheduled_retry_ats must be iterable when provided")
    return any(_as_datetime(value) == retry_at for value in scheduled_times)


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None
