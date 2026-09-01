"""Reproducible synthetic failed-payment data for Recoverly."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import uuid

import numpy as np


RANDOM_SEED = 42
DEFAULT_TRANSACTION_COUNT = 20_000
CUSTOMER_COUNT = 5_000
SIMULATION_START = datetime(2026, 1, 1)
SIMULATION_DAYS = 90
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("transactions.csv")

FAILURE_REASONS = (
    "insufficient_funds",
    "bank_timeout",
    "expired_card",
    "gateway_error",
    "issuer_decline",
    "threeds_dropoff",
)
BASE_REASON_WEIGHTS = np.array((0.35, 0.20, 0.15, 0.15, 0.10, 0.05))
CARD_TYPES = ("UPI", "debit", "credit", "netbanking")
CARD_TYPE_WEIGHTS = (0.45, 0.25, 0.20, 0.10)
GATEWAY_OUTAGE_DAYS = (8, 18, 24)
GATEWAY_OUTAGE_HOURS = (3, 4)

FIELDNAMES = (
    "txn_id",
    "amount",
    "timestamp",
    "card_type",
    "card_last4",
    "is_recurring",
    "customer_id",
    "failure_reason",
    "attempt_number",
    "success",
    "card_updated",
)


def generate_transactions(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    transaction_count: int = DEFAULT_TRANSACTION_COUNT,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Generate failed-payment records and return a printed realism summary.

    ``card_updated`` is an auxiliary event flag required to model the documented
    reset of recurring expired-card failures. All generated rows are failed
    attempts; retry recovery is simulated later by the policy-blind outcome
    model and stored in ``retry_outcomes.csv``.
    """

    if transaction_count <= 0:
        raise ValueError("transaction_count must be greater than zero")

    np.random.seed(seed)
    events = _build_events(transaction_count, seed)
    rows = _assign_failure_reasons(events)
    _write_rows(Path(output_path), rows)

    summary = salary_correlation_summary(rows)
    print(
        "insufficient_funds rate: "
        f"pre-month-end={summary['pre_month_end_rate']:.3f}, "
        f"salary-window={summary['salary_window_rate']:.3f}, "
        f"ratio={summary['pre_to_salary_ratio']:.2f}x"
    )
    return summary


def salary_correlation_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Measure the engineered insufficient-funds salary-date correlation."""

    pre_month_end_rows = [row for row in rows if _is_pre_month_end(_parse_timestamp(row["timestamp"]))]
    salary_window_rows = [row for row in rows if _is_salary_window(_parse_timestamp(row["timestamp"]))]

    pre_month_end_rate = _reason_rate(pre_month_end_rows, "insufficient_funds")
    salary_window_rate = _reason_rate(salary_window_rows, "insufficient_funds")
    ratio = pre_month_end_rate / salary_window_rate if salary_window_rate else float("inf")
    return {
        "pre_month_end_rate": pre_month_end_rate,
        "salary_window_rate": salary_window_rate,
        "pre_to_salary_ratio": ratio,
    }


def _build_events(transaction_count: int, seed: int) -> list[dict[str, Any]]:
    customer_ids = [f"cust_{index:05d}" for index in range(CUSTOMER_COUNT)]
    customer_weights = 1 / np.power(np.arange(1, CUSTOMER_COUNT + 1), 1.1)
    customer_weights /= customer_weights.sum()
    customer_cards = {
        customer_id: f"{int(np.random.randint(0, 10_000)):04d}" for customer_id in customer_ids
    }
    events: list[dict[str, Any]] = []
    for index in range(transaction_count):
        failure_reason = str(np.random.choice(FAILURE_REASONS, p=BASE_REASON_WEIGHTS))
        timestamp = _sample_timestamp_for_reason(failure_reason)
        customer_id = (
            customer_ids[index]
            if index < CUSTOMER_COUNT
            else str(np.random.choice(customer_ids, p=customer_weights))
        )
        card_type = _sample_card_type(failure_reason)
        is_recurring = _sample_is_recurring(failure_reason)
        amount = _sample_amount(failure_reason)
        attempt_number = int(np.random.choice((1, 2), p=(0.82, 0.18)))
        events.append(
            {
                "txn_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"recoverly-{seed}-{index}")),
                "amount": round(amount, 2),
                "timestamp": timestamp,
                "card_type": card_type,
                "card_last4": customer_cards[customer_id],
                "is_recurring": is_recurring,
                "customer_id": customer_id,
                "attempt_number": attempt_number,
                "failure_reason": failure_reason,
            }
        )

    return sorted(events, key=lambda event: event["timestamp"])


def _assign_failure_reasons(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active_expired_cards_until: dict[str, datetime] = {}
    rows: list[dict[str, Any]] = []

    for event in events:
        timestamp = event["timestamp"]
        card_last4 = event["card_last4"]
        reason = str(event["failure_reason"])
        card_updated = False
        active_until = active_expired_cards_until.get(card_last4)

        if event["is_recurring"] and active_until and timestamp <= active_until:
            if np.random.random() < 0.12:
                card_updated = True
                del active_expired_cards_until[card_last4]
            else:
                reason = "expired_card"

        if reason == "expired_card" and event["is_recurring"]:
            active_expired_cards_until[card_last4] = timestamp + timedelta(days=30)

        rows.append(
            {
                "txn_id": event["txn_id"],
                "amount": f"{event['amount']:.2f}",
                "timestamp": timestamp.isoformat(),
                "card_type": event["card_type"],
                "card_last4": card_last4,
                "is_recurring": str(event["is_recurring"]),
                "customer_id": event["customer_id"],
                "failure_reason": reason,
                "attempt_number": event["attempt_number"],
                "success": "False",
                "card_updated": str(card_updated),
            }
        )

    return rows


def _sample_timestamp_for_reason(reason: str) -> datetime:
    if reason == "insufficient_funds" and np.random.random() < 0.90:
        return _sample_timestamp_from_days(_pre_month_end_days())
    if reason == "insufficient_funds" and np.random.random() < 0.80:
        return _sample_timestamp_from_days(_salary_window_days())
    if reason == "bank_timeout" and np.random.random() < 0.95:
        return _sample_timestamp(hour_choices=(11, 12, 13, 19, 20, 21))
    if reason == "gateway_error" and np.random.random() < 0.95:
        return _sample_timestamp_from_days(GATEWAY_OUTAGE_DAYS, hour_choices=GATEWAY_OUTAGE_HOURS)
    return _sample_timestamp()


def _sample_timestamp_from_days(
    day_choices: tuple[int, ...], hour_choices: tuple[int, ...] | None = None
) -> datetime:
    candidate_days = [
        SIMULATION_START + timedelta(days=offset)
        for offset in range(SIMULATION_DAYS)
        if (SIMULATION_START + timedelta(days=offset)).day in day_choices
    ]
    selected_day = candidate_days[int(np.random.randint(0, len(candidate_days)))]
    return _sample_timestamp(base_day=selected_day, hour_choices=hour_choices)


def _sample_timestamp(base_day: datetime | None = None, hour_choices: tuple[int, ...] | None = None) -> datetime:
    day = base_day or (SIMULATION_START + timedelta(days=int(np.random.randint(0, SIMULATION_DAYS))))
    if hour_choices:
        hour = int(np.random.choice(hour_choices))
    else:
        hour = _sample_traffic_hour()
    return day.replace(
        hour=hour,
        minute=int(np.random.randint(0, 60)),
        second=int(np.random.randint(0, 60)),
    )


def _sample_traffic_hour() -> int:
    hour_weights = np.ones(24)
    hour_weights[11:14] = 4
    hour_weights[19:22] = 4
    hour_weights /= hour_weights.sum()
    return int(np.random.choice(np.arange(24), p=hour_weights))


def _sample_card_type(reason: str) -> str:
    reason_weights = {
        "insufficient_funds": (0.99, 0.005, 0.003, 0.002),
        "bank_timeout": (0.05, 0.85, 0.05, 0.05),
        "expired_card": (0.05, 0.10, 0.75, 0.10),
        "gateway_error": (0.15, 0.10, 0.15, 0.60),
        "issuer_decline": (0.05, 0.10, 0.80, 0.05),
        "threeds_dropoff": (0.95, 0.02, 0.02, 0.01),
    }
    return str(np.random.choice(CARD_TYPES, p=reason_weights[reason]))


def _sample_is_recurring(reason: str) -> bool:
    recurring_probability = {
        "insufficient_funds": 0.25,
        "bank_timeout": 0.15,
        "expired_card": 1.0,
        "gateway_error": 0.10,
        "issuer_decline": 0.10,
        "threeds_dropoff": 0.0,
    }[reason]
    return bool(np.random.random() < recurring_probability)


def _sample_amount(reason: str) -> float:
    if reason == "issuer_decline":
        return float(np.clip(np.random.lognormal(mean=10.0, sigma=0.35), 10_000, 50_000))
    return float(np.clip(np.random.lognormal(mean=7.6, sigma=0.75), 200, 12_000))


def _pre_month_end_days() -> tuple[int, ...]:
    return tuple(
        day
        for day in range(1, 32)
        if any(
            _is_pre_month_end(SIMULATION_START + timedelta(days=offset))
            and (SIMULATION_START + timedelta(days=offset)).day == day
            for offset in range(SIMULATION_DAYS)
        )
    )


def _salary_window_days() -> tuple[int, ...]:
    return (1, 2, 3, 4, 5)


def _is_peak_traffic(timestamp: datetime) -> bool:
    return 11 <= timestamp.hour <= 13 or 19 <= timestamp.hour <= 21


def _is_salary_window(timestamp: datetime) -> bool:
    return 1 <= timestamp.day <= 5


def _is_pre_month_end(timestamp: datetime) -> bool:
    return (timestamp + timedelta(days=3)).month != timestamp.month


def _reason_rate(rows: list[dict[str, Any]], reason: str) -> float:
    if not rows:
        return 0.0
    return sum(row["failure_reason"] == reason for row in rows) / len(rows)


def _parse_timestamp(value: Any) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))


def _write_rows(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    generate_transactions()
