"""Tests for Recoverly's reproducible synthetic data generator."""

import csv
from collections import Counter
from pathlib import Path

from backend.data.generator import FIELDNAMES, FAILURE_REASONS, generate_transactions, salary_correlation_summary


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def test_generator_is_deterministic(tmp_path: Path) -> None:
    first_output = tmp_path / "first.csv"
    second_output = tmp_path / "second.csv"

    generate_transactions(first_output, transaction_count=1_000)
    generate_transactions(second_output, transaction_count=1_000)

    assert first_output.read_bytes() == second_output.read_bytes()


def test_generator_writes_required_schema_and_all_failure_reasons(tmp_path: Path) -> None:
    output_path = tmp_path / "transactions.csv"
    generate_transactions(output_path, transaction_count=2_000)
    rows = _read_rows(output_path)

    assert tuple(rows[0]) == FIELDNAMES
    assert {row["failure_reason"] for row in rows} == set(FAILURE_REASONS)
    assert all(200 <= float(row["amount"]) <= 50_000 for row in rows)
    assert all(row["success"] == "False" for row in rows)


def test_generator_produces_pre_month_end_insufficient_funds_spike(tmp_path: Path) -> None:
    output_path = tmp_path / "transactions.csv"
    generate_transactions(output_path, transaction_count=8_000)
    summary = salary_correlation_summary(_read_rows(output_path))

    assert summary["pre_month_end_rate"] > summary["salary_window_rate"]
    assert summary["pre_to_salary_ratio"] >= 2.0


def test_generator_includes_approximately_five_thousand_power_law_customers(tmp_path: Path) -> None:
    output_path = tmp_path / "transactions.csv"
    generate_transactions(output_path, transaction_count=6_000)
    rows = _read_rows(output_path)
    customer_counts = Counter(row["customer_id"] for row in rows)

    assert len(customer_counts) == 5_000
    assert max(customer_counts.values()) > 10 * min(customer_counts.values())
