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
    outage_starts = _gateway_outage_starts()

    events: list[dict[str, Any]] = []
    for index in range(transaction_count):
        timestamp = _sample_timestamp()
        customer_id = (
            customer_ids[index]
            if index < CUSTOMER_COUNT
            else str(np.random.choice(customer_ids, p=customer_weights))
        )
        card_type = str(np.random.choice(CARD_TYPES, p=CARD_TYPE_WEIGHTS))
        is_recurring = bool(np.random.random() < 0.30)
        amount = float(np.clip(np.random.lognormal(mean=7.7, sigma=1.0), 200, 50_000))
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
                "in_gateway_outage": _is_in_outage(timestamp, outage_starts),
            }
        )

    return sorted(events, key=lambda event: event["timestamp"])


def _assign_failure_reasons(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active_expired_cards_until: dict[str, datetime] = {}
    rows: list[dict[str, Any]] = []

    for event in events:
        timestamp = event["timestamp"]
        card_last4 = event["card_last4"]
        card_updated = False
        active_until = active_expired_cards_until.get(card_last4)

        if event["is_recurring"] and active_until and timestamp <= active_until:
            if np.random.random() < 0.12:
                card_updated = True
                del active_expired_cards_until[card_last4]
                reason = _choose_failure_reason(event)
            else:
                reason = "expired_card"
        else:
            reason = _choose_failure_reason(event)

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


def _choose_failure_reason(event: dict[str, Any]) -> str:
    weights = BASE_REASON_WEIGHTS.astype(float).copy()
    timestamp = event["timestamp"]

    if _is_pre_month_end(timestamp):
        weights[0] *= 5
    elif _is_salary_window(timestamp):
        weights[0] *= 0.20

    if _is_peak_traffic(timestamp):
        weights[1] *= 2.5
    if event["in_gateway_outage"]:
        weights[3] *= 5
    if event["amount"] >= 10_000:
        weights[4] *= 2
    if event["card_type"] == "UPI":
        weights[5] *= 4

    weights[2] *= 1.5 if event["is_recurring"] else 0.20
    weights /= weights.sum()
    return str(np.random.choice(FAILURE_REASONS, p=weights))


def _sample_timestamp() -> datetime:
    day_offset = int(np.random.randint(0, SIMULATION_DAYS))
    hour_weights = np.ones(24)
    hour_weights[11:14] = 4
    hour_weights[19:22] = 4
    hour_weights /= hour_weights.sum()
    hour = int(np.random.choice(np.arange(24), p=hour_weights))
    return SIMULATION_START + timedelta(
        days=day_offset,
        hours=hour,
        minutes=int(np.random.randint(0, 60)),
        seconds=int(np.random.randint(0, 60)),
    )


def _gateway_outage_starts() -> list[datetime]:
    return [
        SIMULATION_START + timedelta(days=int(np.random.randint(0, SIMULATION_DAYS)), hours=int(np.random.randint(0, 22)))
        for _ in range(12)
    ]


def _is_in_outage(timestamp: datetime, outage_starts: list[datetime]) -> bool:
    return any(start <= timestamp < start + timedelta(hours=2) for start in outage_starts)


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
