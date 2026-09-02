"""Integration tests for the Recoverly recovery-action API."""

import csv
import json

from fastapi.testclient import TestClient

import backend.api.main as api_main
from backend.api.main import app
from backend.features import TEST_SET_PATH


client = TestClient(app)


def _demo_payload() -> dict[str, object]:
    with (TEST_SET_PATH.parent / "demo_cases.csv").open(newline="", encoding="utf-8") as csv_file:
        row = next(csv.DictReader(csv_file))
    return {
        "txn_id": row["txn_id"],
        "amount": float(row["amount"]),
        "hour_of_day": int(row["hour_of_day"]),
        "day_of_month": int(row["day_of_month"]),
        "card_type": row["card_type"],
        "is_recurring": row["is_recurring"] == "true",
        "customer_past_failure_count": int(row["customer_past_failure_count"]),
        "customer_past_failure_reasons_distribution": json.loads(row["customer_past_failure_reasons_distribution"]),
        "attempt_number": int(row["attempt_number"]),
        "decision_at": row["decision_at"],
        "scheduled_retry_ats": [],
    }


def test_api_returns_the_full_recovery_action_contract() -> None:
    response = client.post("/predict-recovery-action", json=_demo_payload())

    assert response.status_code == 200
    assert set(response.json()) == {
        "reason",
        "confidence",
        "action",
        "retry_at",
        "expected_value",
        "amount",
        "explanation",
        "explanation_source",
    }


def test_api_uses_cached_explanation_for_a_demo_case(monkeypatch) -> None:
    monkeypatch.setattr(
        api_main,
        "explain_decision_with_source",
        lambda _: (_ for _ in ()).throw(AssertionError("live path used")),
    )

    response = client.post("/predict-recovery-action", json=_demo_payload())

    assert response.status_code == 200
    assert "Request a card update" in response.json()["explanation"]
    assert response.json()["explanation_source"] == "template_fallback"


def test_api_uses_live_explanation_path_for_a_non_cached_example(monkeypatch) -> None:
    monkeypatch.setattr(
        api_main,
        "explain_decision_with_source",
        lambda decision: (f"fresh {decision['action']}", "ai_generated"),
    )
    payload = _demo_payload()
    payload["txn_id"] = "fresh-non-cached-example"

    response = client.post("/predict-recovery-action", json=payload)

    assert response.status_code == 200
    assert response.json()["explanation"].startswith("fresh ")
    assert response.json()["explanation_source"] == "ai_generated"


def test_api_uses_explanation_fallback_for_a_non_cached_example_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    payload = _demo_payload()
    payload["txn_id"] = "fresh-fallback-example"

    response = client.post("/predict-recovery-action", json=payload)

    assert response.status_code == 200
    assert response.json()["explanation"].startswith("Failure reason ")
    assert response.json()["explanation_source"] == "template_fallback"


def test_api_enforces_confidence_routing_and_expired_card_safety(monkeypatch) -> None:
    payload = _demo_payload()
    monkeypatch.setattr(api_main, "predict_failure_reason", lambda _: {"reason": "bank_timeout", "confidence": 0.54})
    monkeypatch.setattr(api_main, "explain_decision_with_source", lambda _: ("fallback", "template_fallback"))
    payload["txn_id"] = "low-confidence"

    low_confidence_response = client.post("/predict-recovery-action", json=payload)

    assert low_confidence_response.status_code == 200
    assert low_confidence_response.json()["action"] == "escalate_manual_review"

    monkeypatch.setattr(api_main, "predict_failure_reason", lambda _: {"reason": "expired_card", "confidence": 0.99})
    payload["txn_id"] = "expired-card"
    expired_card_response = client.post("/predict-recovery-action", json=payload)

    assert expired_card_response.status_code == 200
    assert expired_card_response.json()["action"] == "notify_update_card"


def test_demo_cases_are_disjoint_from_the_frozen_test_set() -> None:
    with (TEST_SET_PATH.parent / "demo_cases.csv").open(newline="", encoding="utf-8") as csv_file:
        demo_ids = {row["txn_id"] for row in csv.DictReader(csv_file)}
    with TEST_SET_PATH.open(newline="", encoding="utf-8") as csv_file:
        test_ids = {row["txn_id"] for row in csv.DictReader(csv_file)}

    assert len(demo_ids) == 5
    assert demo_ids.isdisjoint(test_ids)


def test_api_rejects_forbidden_label_and_outcome_fields() -> None:
    payload = _demo_payload()
    payload["failure_reason"] = "expired_card"

    response = client.post("/predict-recovery-action", json=payload)

    assert response.status_code == 422
