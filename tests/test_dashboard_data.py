"""Tests for held-out-only dashboard data preparation."""

import csv
from pathlib import Path

import pytest

from backend.evaluation import RETRY_OUTCOMES_PATH, compute_metrics
from backend.features import TEST_SET_PATH
from dashboard.app import (
    _read_outcome_rows,
    load_dashboard_metrics,
    load_demo_cases,
    load_feature_importances,
    load_opportunity_matrix,
    load_revenue_at_risk,
    run_demo_case,
)


def test_dashboard_uses_shared_metric_formulas_on_frozen_test_data() -> None:
    dashboard_metrics = load_dashboard_metrics()
    outcome_rows = _read_outcome_rows(RETRY_OUTCOMES_PATH)
    total_failed_attempts = sum(1 for _ in TEST_SET_PATH.open(encoding="utf-8")) - 1

    assert dashboard_metrics == compute_metrics(outcome_rows, total_failed_attempts)


def test_dashboard_rejects_non_test_metric_source() -> None:
    with pytest.raises(ValueError, match="test_set_v1"):
        load_dashboard_metrics(test_path=TEST_SET_PATH.with_name("dev_set_v1.csv"))


def test_dashboard_rejects_outcomes_that_do_not_cover_the_frozen_test_set() -> None:
    with pytest.raises(ValueError, match="frozen held-out"):
        load_dashboard_metrics(outcomes_path=Path("backend/data/demo_cases.csv"))


def test_demo_cases_are_explicitly_separate_from_reported_metrics() -> None:
    demo_cases = load_demo_cases()
    with TEST_SET_PATH.open(encoding="utf-8") as test_file:
        test_ids = {line.split(",", 1)[0] for line in test_file.readlines()[1:]}

    assert demo_cases
    assert {case["source_split"] for case in demo_cases} <= {"train", "dev"}
    assert {case["txn_id"] for case in demo_cases}.isdisjoint(test_ids)


def test_demo_action_view_uses_the_api_handler() -> None:
    response = run_demo_case(load_demo_cases()[0])

    assert response["action"] == "notify_update_card"
    assert response["explanation"]
    assert response["explanation_source"] == "template_fallback"


def test_dashboard_loads_classifier_feature_importances() -> None:
    importances = load_feature_importances()

    assert importances
    assert all(isinstance(row["feature"], str) and row["feature"] for row in importances)
    assert all(isinstance(row["importance"], float) for row in importances)


def test_opportunity_matrix_uses_only_frozen_evaluation_inputs() -> None:
    opportunities = load_opportunity_matrix()

    assert len(opportunities) == sum(1 for _ in TEST_SET_PATH.open(encoding="utf-8")) - 1
    assert set(opportunities[0]) == {
        "Failure reason",
        "Policy belief recovery probability",
        "Transaction amount (₹)",
        "Expected Recovery Value (₹)",
    }
    assert all(0.0 <= row["Policy belief recovery probability"] <= 1.0 for row in opportunities)
    with pytest.raises(ValueError, match="test_set_v1"):
        load_opportunity_matrix(test_path=TEST_SET_PATH.with_name("dev_set_v1.csv"))


def test_revenue_at_risk_groups_frozen_failed_amounts_by_reason() -> None:
    revenue_at_risk = load_revenue_at_risk()
    test_rows = list(csv.DictReader(TEST_SET_PATH.open(encoding="utf-8")))

    assert {row["Failure reason"] for row in revenue_at_risk} == {
        row["failure_reason"] for row in test_rows
    }
    assert sum(row["Revenue at risk (₹)"] for row in revenue_at_risk) == pytest.approx(
        sum(float(row["amount"]) for row in test_rows)
    )
