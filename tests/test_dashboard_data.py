"""Tests for held-out-only dashboard data preparation."""

import csv
import json
from pathlib import Path

import pytest

import dashboard.app as dashboard_app
from backend.evaluation import RETRY_OUTCOMES_PATH, compute_metrics
from backend.features import TEST_SET_PATH
from dashboard.app import (
    _read_outcome_rows,
    load_dashboard_metrics,
    load_demo_cases,
    load_feature_importances,
    load_opportunity_matrix,
    load_revenue_at_risk,
    parse_history_distribution,
    run_demo_case,
    run_live_simulation,
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


def test_live_simulator_posts_leakage_safe_payload_to_the_api(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "amount": 1_000.0,
        "hour_of_day": 9,
        "day_of_month": 1,
        "card_type": "debit",
        "is_recurring": False,
        "customer_past_failure_count": 0,
        "customer_past_failure_reasons_distribution": {},
        "attempt_number": 1,
        "decision_at": "2026-02-01T09:00:00",
        "scheduled_retry_ats": [],
    }
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"action":"retry_now"}'

    def fake_urlopen(request, timeout: int):
        captured["url"] = request.full_url
        captured["body"] = request.data
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(dashboard_app, "urlopen", fake_urlopen)

    assert run_live_simulation(payload, api_url="http://api.test/predict-recovery-action") == {
        "action": "retry_now"
    }
    assert captured["url"] == "http://api.test/predict-recovery-action"
    assert json.loads(captured["body"]) == payload
    assert captured["timeout"] == 10


def test_live_simulator_validates_history_distribution_json() -> None:
    assert parse_history_distribution('{"bank_timeout": 0.75}') == {"bank_timeout": 0.75}
    with pytest.raises(ValueError, match="valid JSON"):
        parse_history_distribution("not-json")
