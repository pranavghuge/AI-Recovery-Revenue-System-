"""Tests for held-out-only dashboard data preparation."""

from pathlib import Path

import pytest

from backend.evaluation import RETRY_OUTCOMES_PATH, compute_metrics
from backend.features import TEST_SET_PATH
from dashboard.app import _read_outcome_rows, load_dashboard_metrics, load_demo_cases, run_demo_case


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
