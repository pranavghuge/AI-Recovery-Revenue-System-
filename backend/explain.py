"""Grounded plain-English narration for deterministic recovery decisions."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import json
import os
import re
import time
from typing import Any


ALLOWED_DECISION_KEYS = frozenset(
    {"reason", "confidence", "action", "retry_at", "expected_value", "amount"}
)
LIVE_CALL_TIMEOUT_SECONDS = 2.0
LIVE_PATH_LATENCY_BUDGET_SECONDS = 3.0
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?")


def explain_decision(decision: dict) -> str:
    """Return a grounded narration or deterministic action-specific fallback."""

    return explain_decision_with_source(decision)[0]


def explain_decision_with_source(decision: dict) -> tuple[str, str]:
    """Return a grounded narration or deterministic action-specific fallback.

    Gemini receives only the six permitted values. A live response gets one
    retry only when it fails numeric grounding; provider errors and timeouts go
    directly to the fallback. The retry uses the time left from one 3-second
    total budget and can never exceed that budget.
    """

    grounded_decision = _allowed_decision(decision)
    action = str(grounded_decision.get("action", ""))
    if action not in FALLBACK_TEMPLATES:
        raise ValueError(f"Unsupported action for explanation: {action}")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return _fallback_explanation(grounded_decision), "template_fallback"

    prompt = _build_prompt(grounded_decision)
    deadline = time.monotonic() + LIVE_PATH_LATENCY_BUDGET_SECONDS
    first_timeout = _remaining_timeout(deadline)
    if first_timeout <= 0:
        return _fallback_explanation(grounded_decision), "template_fallback"

    try:
        explanation = _generate_from_gemini(prompt, api_key, first_timeout)
    except Exception:
        return _fallback_explanation(grounded_decision), "template_fallback"
    if _passes_grounding_check(explanation, grounded_decision):
        return explanation, "ai_generated"

    retry_timeout = _remaining_timeout(deadline)
    if retry_timeout <= 0:
        return _fallback_explanation(grounded_decision), "template_fallback"
    try:
        retried_explanation = _generate_from_gemini(prompt, api_key, retry_timeout)
    except Exception:
        return _fallback_explanation(grounded_decision), "template_fallback"
    if _passes_grounding_check(retried_explanation, grounded_decision):
        return retried_explanation, "ai_generated"
    return _fallback_explanation(grounded_decision), "template_fallback"


def is_template_fallback(decision: dict, explanation: str) -> bool:
    """Return whether text is exactly the deterministic fallback for a decision."""

    return explanation == _fallback_explanation(_allowed_decision(decision))


def _generate_from_gemini(prompt: str, api_key: str, timeout_seconds: float) -> str:
    """Make one synchronous Gemini call with the provided remaining timeout."""

    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(timeout_seconds * 1_000)),
    )
    response = client.models.generate_content(
        model=_gemini_model_name(), contents=prompt
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty explanation")
    return response.text.strip()


def _gemini_model_name() -> str:
    """Return the deployment-configured Gemini model without storing secrets."""

    return os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def _allowed_decision(decision: dict) -> dict[str, Any]:
    if not isinstance(decision, dict):
        raise ValueError("decision must be a dictionary")
    return {key: decision[key] for key in ALLOWED_DECISION_KEYS if key in decision}


def _build_prompt(decision: dict[str, Any]) -> str:
    values = json.dumps(decision, default=_json_value, sort_keys=True)
    return (
        "Write one concise, plain-English payment recovery explanation. "
        "You may state only the values supplied below. Do not calculate, round, "
        "convert, infer, or introduce any number or factual claim. Preserve any "
        "numbers exactly as supplied.\n"
        f"Decision values: {values}"
    )


def _passes_grounding_check(explanation: str, decision: dict[str, Any]) -> bool:
    """Accept output only when every numeric token is represented in the input."""

    allowed_numbers = {
        _normalized_number(number)
        for value in decision.values()
        for number in NUMBER_PATTERN.findall(_json_value(value))
    }
    return all(_normalized_number(number) in allowed_numbers for number in NUMBER_PATTERN.findall(explanation))


def _normalized_number(value: str) -> Decimal | str:
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation:
        return value


def _remaining_timeout(deadline: float) -> float:
    return max(0.0, min(LIVE_CALL_TIMEOUT_SECONDS, deadline - time.monotonic()))


def _fallback_explanation(decision: dict[str, Any]) -> str:
    return FALLBACK_TEMPLATES[str(decision["action"])].format(**_template_values(decision))


def _template_values(decision: dict[str, Any]) -> dict[str, str]:
    return {
        key: _json_value(decision.get(key, "not provided"))
        for key in ALLOWED_DECISION_KEYS
    }


def _json_value(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None:
        return "none"
    return str(value)


FALLBACK_TEMPLATES = {
    "retry_now": (
        "Failure reason {reason} at confidence {confidence}. Retry now for amount {amount}; "
        "the modeled expected recovered amount is {expected_value}."
    ),
    "retry_scheduled": (
        "Failure reason {reason} at confidence {confidence}. Schedule the retry at {retry_at} "
        "for amount {amount}; the modeled expected recovered amount is {expected_value}."
    ),
    "notify_update_card": (
        "Failure reason {reason} at confidence {confidence}. Request a card update for amount {amount}; "
        "the modeled expected recovered amount is {expected_value}."
    ),
    "escalate_manual_review": (
        "Failure reason {reason} at confidence {confidence}. Escalate this amount {amount} for manual review; "
        "the modeled expected recovered amount is {expected_value}."
    ),
    "no_action": (
        "Failure reason {reason} at confidence {confidence}. Take no action for amount {amount}; "
        "the modeled expected recovered amount is {expected_value}."
    ),
}
