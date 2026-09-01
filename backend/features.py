"""Leakage-safe feature preparation and frozen customer-disjoint data splits."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import random
from statistics import median
from typing import Any


LOGGER = logging.getLogger(__name__)
DATA_DIRECTORY = Path(__file__).parent / "data"
SOURCE_TRANSACTIONS_PATH = DATA_DIRECTORY / "transactions.csv"
TRAIN_SET_PATH = DATA_DIRECTORY / "train_set_v1.csv"
DEV_SET_PATH = DATA_DIRECTORY / "dev_set_v1.csv"
TEST_SET_PATH = DATA_DIRECTORY / "test_set_v1.csv"
SPLIT_SEED = 42

MODEL_FEATURE_COLUMNS = (
    "amount",
    "hour_of_day",
    "day_of_month",
    "card_type",
    "is_recurring",
    "customer_past_failure_count",
    "customer_past_failure_reasons_distribution",
    "attempt_number",
)
NUMERIC_FEATURE_COLUMNS = (
    "amount",
    "hour_of_day",
    "day_of_month",
    "customer_past_failure_count",
    "attempt_number",
)
CATEGORICAL_FEATURE_COLUMNS = ("card_type", "is_recurring", "customer_past_failure_reasons_distribution")
FORBIDDEN_MODEL_FEATURE_COLUMNS = frozenset({"failure_reason", "success", "retry_outcomes"})

SPLIT_FIELDNAMES = (
    "txn_id",
    "customer_id",
    "amount",
    "timestamp",
    "card_type",
    "card_last4",
    "is_recurring",
    "attempt_number",
    "card_updated",
    "hour_of_day",
    "day_of_month",
    "customer_past_failure_count",
    "customer_past_failure_reasons_distribution",
    "failure_reason",
)

# Set when T04 freezes the generated held-out test artifact.
TEST_SET_V1_SHA256 = "3af19f793f00a237a595a8c548bb98b2e99c012841a49a0757d2d81a9bfc9bb3"


def prepare_dataset_splits(
    source_path: Path | str = SOURCE_TRANSACTIONS_PATH,
    train_path: Path | str = TRAIN_SET_PATH,
    dev_path: Path | str = DEV_SET_PATH,
    test_path: Path | str = TEST_SET_PATH,
) -> dict[str, Any]:
    """Create deterministic customer-disjoint 70/15/15 dataset splits."""

    assert_leakage_safe_feature_columns(MODEL_FEATURE_COLUMNS)
    source_rows = _read_csv(Path(source_path))
    feature_rows = _derive_feature_rows(source_rows)
    splits = _split_by_customer(feature_rows)
    medians = _numeric_medians(splits["train"])

    for split_name, rows in splits.items():
        splits[split_name] = impute_feature_records(rows, medians)

    paths = {"train": Path(train_path), "dev": Path(dev_path), "test": Path(test_path)}
    for split_name, path in paths.items():
        _write_csv(path, splits[split_name])

    test_checksum = calculate_file_sha256(paths["test"])
    return {
        "counts": {split_name: len(rows) for split_name, rows in splits.items()},
        "numeric_medians": medians,
        "test_set_sha256": test_checksum,
    }


def assert_leakage_safe_feature_columns(feature_columns: Iterable[str]) -> None:
    """Reject classifier inputs that expose labels or post-policy outcomes."""

    columns = set(feature_columns)
    forbidden = columns & FORBIDDEN_MODEL_FEATURE_COLUMNS
    if forbidden:
        raise ValueError(f"Forbidden leakage feature(s): {', '.join(sorted(forbidden))}")
    if columns != set(MODEL_FEATURE_COLUMNS):
        raise ValueError("Classifier features must match the Evaluation Contract allowlist exactly")


def impute_feature_records(
    records: Iterable[Mapping[str, Any]], numeric_medians: Mapping[str, float]
) -> list[dict[str, Any]]:
    """Impute missing feature values with documented, visible defaults."""

    imputed_rows: list[dict[str, Any]] = []
    for source_record in records:
        record = dict(source_record)
        for column in NUMERIC_FEATURE_COLUMNS:
            value = _as_float(record.get(column))
            if value is None:
                value = numeric_medians[column]
                LOGGER.warning("Imputing missing %s with training median", column)
            record[column] = value

        for column in CATEGORICAL_FEATURE_COLUMNS:
            if record.get(column) in (None, ""):
                record[column] = "unknown"
                LOGGER.warning("Imputing missing %s with unknown category", column)
        imputed_rows.append(record)
    return imputed_rows


def calculate_file_sha256(path: Path | str) -> str:
    """Return the SHA-256 digest used to protect the frozen held-out set."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen_test_set(test_path: Path | str = TEST_SET_PATH) -> None:
    """Raise when the versioned held-out test data differs from its frozen hash."""

    if not TEST_SET_V1_SHA256:
        raise RuntimeError("TEST_SET_V1_SHA256 has not been frozen")
    actual_checksum = calculate_file_sha256(test_path)
    if actual_checksum != TEST_SET_V1_SHA256:
        raise ValueError("test_set_v1.csv checksum does not match its frozen value")


def _derive_feature_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_customer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        customer_id = str(row.get("customer_id") or "unknown")
        rows_by_customer[customer_id].append(row)

    feature_rows: list[dict[str, Any]] = []
    for customer_rows in rows_by_customer.values():
        reason_counts: dict[str, int] = defaultdict(int)
        for source_row in sorted(customer_rows, key=_timestamp_sort_key):
            timestamp = _parse_timestamp(source_row.get("timestamp"))
            history_count = sum(reason_counts.values())
            feature_rows.append(
                {
                    "txn_id": source_row.get("txn_id", ""),
                    "customer_id": source_row.get("customer_id") or "unknown",
                    "amount": source_row.get("amount"),
                    "timestamp": source_row.get("timestamp", ""),
                    "card_type": source_row.get("card_type"),
                    "card_last4": source_row.get("card_last4", ""),
                    "is_recurring": _normalize_bool(source_row.get("is_recurring")),
                    "attempt_number": source_row.get("attempt_number"),
                    "card_updated": _normalize_bool(source_row.get("card_updated")),
                    "hour_of_day": timestamp.hour if timestamp else None,
                    "day_of_month": timestamp.day if timestamp else None,
                    "customer_past_failure_count": history_count,
                    "customer_past_failure_reasons_distribution": _reason_distribution(reason_counts, history_count),
                    "failure_reason": source_row.get("failure_reason", ""),
                }
            )
            reason = source_row.get("failure_reason")
            if reason:
                reason_counts[str(reason)] += 1
    return feature_rows


def _split_by_customer(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows_by_customer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_customer[str(row["customer_id"])].append(row)

    total_rows = len(rows)
    targets = {"train": total_rows * 0.70, "dev": total_rows * 0.15, "test": total_rows * 0.15}
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    split_counts = {"train": 0, "dev": 0, "test": 0}
    shuffle = random.Random(SPLIT_SEED)
    customers = list(rows_by_customer.items())
    shuffle.shuffle(customers)
    customers.sort(key=lambda item: len(item[1]), reverse=True)

    for _, customer_rows in customers:
        split_name = max(
            split_counts,
            key=lambda name: (targets[name] - split_counts[name], -split_counts[name]),
        )
        split_rows[split_name].extend(customer_rows)
        split_counts[split_name] += len(customer_rows)

    return split_rows


def _numeric_medians(rows: list[dict[str, Any]]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for column in NUMERIC_FEATURE_COLUMNS:
        values = [_as_float(row.get(column)) for row in rows]
        present_values = [value for value in values if value is not None]
        if not present_values:
            raise ValueError(f"Cannot calculate median for {column} without a present value")
        medians[column] = float(median(present_values))
    return medians


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SPLIT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        LOGGER.warning("Unable to parse timestamp for feature derivation")
        return None


def _timestamp_sort_key(row: Mapping[str, Any]) -> datetime:
    return _parse_timestamp(row.get("timestamp")) or datetime.min


def _normalize_bool(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if str(value).lower() in {"true", "false"}:
        return str(value).lower()
    return "unknown" if value in (None, "") else str(value)


def _reason_distribution(reason_counts: Mapping[str, int], history_count: int) -> str:
    if not history_count:
        return "{}"
    distribution = {reason: round(count / history_count, 6) for reason, count in sorted(reason_counts.items())}
    return json.dumps(distribution, sort_keys=True, separators=(",", ":"))


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    summary = prepare_dataset_splits()
    print(f"split counts: {summary['counts']}")
    print(f"test_set_v1.csv sha256: {summary['test_set_sha256']}")
