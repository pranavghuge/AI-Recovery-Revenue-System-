"""FastAPI entry point for one Recoverly recovery decision."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI

from backend.api.schemas import RecoveryActionRequest, RecoveryActionResponse
from backend.classifier import predict_failure_reason
from backend.explain import explain_decision
from backend.retry_policy import decide_action


EXPLANATION_CACHE_PATH = Path(__file__).parents[1] / "data" / "explanation_cache.json"
app = FastAPI(title="Recoverly API", version="0.1.0")


@app.post("/predict-recovery-action", response_model=RecoveryActionResponse)
def predict_recovery_action(request: RecoveryActionRequest) -> RecoveryActionResponse:
    """Classify, decide, and explain one failed-payment recovery action."""

    classifier_features = {
        "amount": request.amount,
        "hour_of_day": request.hour_of_day,
        "day_of_month": request.day_of_month,
        "card_type": request.card_type,
        "is_recurring": str(request.is_recurring).lower(),
        "customer_past_failure_count": request.customer_past_failure_count,
        "customer_past_failure_reasons_distribution": json.dumps(
            request.customer_past_failure_reasons_distribution, sort_keys=True, separators=(",", ":")
        ),
        "attempt_number": request.attempt_number,
    }
    prediction = predict_failure_reason(classifier_features)
    decision = decide_action(
        prediction["reason"],
        prediction["confidence"],
        {
            "amount": request.amount,
            "attempt_number": request.attempt_number,
            "decision_at": request.decision_at,
            "scheduled_retry_ats": request.scheduled_retry_ats,
        },
    )
    explanation_decision = {
        **prediction,
        **decision,
        "amount": request.amount,
    }
    explanation = _cached_explanation(request.txn_id) or explain_decision(explanation_decision)
    return RecoveryActionResponse(**prediction, **decision, amount=request.amount, explanation=explanation)


def _cached_explanation(txn_id: str | None) -> str | None:
    if not txn_id or not EXPLANATION_CACHE_PATH.exists():
        return None
    with EXPLANATION_CACHE_PATH.open(encoding="utf-8") as cache_file:
        cache = json.load(cache_file)
    entry = cache.get("explanations", {}).get(txn_id)
    return entry.get("explanation") if isinstance(entry, dict) else None
