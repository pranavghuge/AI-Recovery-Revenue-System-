"""Tests for grounded Gemini narration and deterministic explanation fallback."""

from datetime import datetime

import pytest

import backend.explain as explain


def _decision(action: str = "retry_scheduled") -> dict[str, object]:
    return {
        "reason": "insufficient_funds",
        "confidence": 0.82,
        "action": action,
        "retry_at": datetime(2026, 2, 1, 9, 0),
        "expected_value": 820.0,
        "amount": 1_000.0,
        "ignored_internal_value": 999_999,
    }


@pytest.mark.parametrize("action", sorted(explain.FALLBACK_TEMPLATES))
def test_every_action_has_a_grounded_template_fallback(monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    explanation = explain.explain_decision(_decision(action))

    assert explanation
    assert explain._passes_grounding_check(explanation, explain._allowed_decision(_decision(action)))
    assert "999999" not in explanation


def test_provider_error_uses_template_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(explain, "_generate_from_gemini", lambda *_: (_ for _ in ()).throw(TimeoutError()))

    explanation = explain.explain_decision(_decision("retry_now"))

    assert explanation == explain._fallback_explanation(explain._allowed_decision(_decision("retry_now")))


def test_adversarial_invented_number_is_rejected_then_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    responses = iter(["Recover 999999 immediately.", "Recover 777777 immediately."])
    monkeypatch.setattr(explain, "_generate_from_gemini", lambda *_: next(responses))

    explanation = explain.explain_decision(_decision())

    assert "999999" not in explanation
    assert "777777" not in explanation
    assert explanation == explain._fallback_explanation(explain._allowed_decision(_decision()))


def test_grounded_live_response_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    grounded_response = "Reason insufficient_funds, confidence 0.82, amount 1000.0, expected value 820.0."
    monkeypatch.setattr(explain, "_generate_from_gemini", lambda *_: grounded_response)

    assert explain.explain_decision(_decision()) == grounded_response


def test_gemini_model_uses_current_default_and_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert explain._gemini_model_name() == "gemini-2.5-flash"

    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom-model")
    assert explain._gemini_model_name() == "gemini-custom-model"


def test_explanation_source_reports_live_or_template_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    fallback_explanation, fallback_source = explain.explain_decision_with_source(_decision())

    assert fallback_source == "template_fallback"
    assert explain.is_template_fallback(_decision(), fallback_explanation)

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        explain,
        "_generate_from_gemini",
        lambda *_: "Reason insufficient_funds, confidence 0.82, amount 1000.0, expected value 820.0.",
    )
    _, live_source = explain.explain_decision_with_source(_decision())

    assert live_source == "ai_generated"


def test_grounding_retry_is_limited_to_one_and_stays_within_total_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls: list[float] = []
    clock_values = iter((0.0, 0.0, 2.4))
    monkeypatch.setattr(explain.time, "monotonic", lambda: next(clock_values))

    def adversarial_response(_: str, __: str, timeout_seconds: float) -> str:
        calls.append(timeout_seconds)
        return "Invented value 999999."

    monkeypatch.setattr(explain, "_generate_from_gemini", adversarial_response)
    explanation = explain.explain_decision(_decision())

    assert len(calls) == 2
    assert calls[0] == 2.0
    assert calls[1] == pytest.approx(0.6)
    assert sum(calls) <= explain.LIVE_PATH_LATENCY_BUDGET_SECONDS
    assert explanation == explain._fallback_explanation(explain._allowed_decision(_decision()))
