"""Leakage-safe request and response schemas for the recovery-action API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RecoveryActionRequest(BaseModel):
    """Classifier-safe features plus policy scheduling context.

    ``decision_at`` and ``scheduled_retry_ats`` are policy context only; they
    are never sent to the classifier. Labels, success, and retry outcomes are
    explicitly rejected by the strict schema.
    """

    model_config = ConfigDict(extra="forbid")

    txn_id: str | None = None
    amount: float = Field(ge=0)
    hour_of_day: int = Field(ge=0, le=23)
    day_of_month: int = Field(ge=1, le=31)
    card_type: str = Field(min_length=1)
    is_recurring: bool
    customer_past_failure_count: int = Field(ge=0)
    customer_past_failure_reasons_distribution: dict[str, float] = Field(default_factory=dict)
    attempt_number: int = Field(ge=1)
    decision_at: datetime
    scheduled_retry_ats: list[datetime] = Field(default_factory=list)


class RecoveryActionCandidate(BaseModel):
    """One smart-policy option evaluated before selecting the final action."""

    action: str
    retry_at: datetime
    expected_value: float


class RecoveryActionResponse(BaseModel):
    reason: str
    confidence: float
    action: str
    retry_at: datetime | None
    expected_value: float
    candidates: list[RecoveryActionCandidate] | None = None
    amount: float
    explanation: str
    explanation_source: Literal["ai_generated", "template_fallback"]
