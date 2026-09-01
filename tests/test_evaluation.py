"""Tests for fair, deterministic held-out policy evaluation."""

import csv
from pathlib import Path

import pytest

from backend.evaluation import RETRY_OUTCOMES_PATH, compute_metrics, evaluate_held_out_policies
from backend.features import TEST_SET_PATH


TEST_OUTPUT_A = Path("work") / "t08-retry-outcomes-a.csv"
TEST_OUTPUT_B = Path("work") / "t08-retry-outcomes-b.csv"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def test_evaluation_uses_same_held_out_transaction_ids_for_both_policies() -> None:
    result = evaluate_held_out_policies(output_path=TEST_OUTPUT_A)
    rows = _read_rows(TEST_OUTPUT_A)
    baseline_ids = {row["txn_id"] for row in rows if row["policy"] == "baseline"}
    smart_ids = {row["txn_id"] for row in rows if row["policy"] == "smart"}

    assert baseline_ids == smart_ids == set(result["eligible_txn_ids"])
    assert not result["excluded_txn_ids"]


def test_metric_formulas_match_the_outcome_rows_exactly() -> None:
    result = evaluate_held_out_policies(output_path=TEST_OUTPUT_A)
    rows = _read_rows(TEST_OUTPUT_A)
    total_failed_attempts = len(result["eligible_txn_ids"])

    for policy in ("baseline", "smart"):
        policy_rows = [row for row in rows if row["policy"] == policy]
        recovered_count = sum(row["recovered"] == "True" for row in policy_rows)
        recovered_amount = sum(float(row["recovered_amount"]) for row in policy_rows)
        assert result["metrics"][policy]["recovery_rate"] == recovered_count / total_failed_attempts
        assert result["metrics"][policy]["rupees_recovered"] == round(recovered_amount, 2)

    assert result["metrics"]["comparison"]["incremental_uplift"] == round(
        result["metrics"]["smart"]["rupees_recovered"]
        - result["metrics"]["baseline"]["rupees_recovered"],
        2,
    )


def test_repeated_evaluation_produces_byte_identical_outcomes() -> None:
    evaluate_held_out_policies(output_path=TEST_OUTPUT_A)
    evaluate_held_out_policies(output_path=TEST_OUTPUT_B)

    assert TEST_OUTPUT_A.read_bytes() == TEST_OUTPUT_B.read_bytes()


def test_reported_evaluation_rejects_non_frozen_data_source() -> None:
    with pytest.raises(ValueError, match="test_set_v1"):
        evaluate_held_out_policies(test_path=TEST_SET_PATH.with_name("dev_set_v1.csv"), output_path=TEST_OUTPUT_A)


def test_metric_function_requires_nonempty_matched_set() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        compute_metrics([], 0)


def test_default_output_location_is_the_versioned_retry_outcomes_artifact() -> None:
    assert RETRY_OUTCOMES_PATH.name == "retry_outcomes.csv"
