"""Documented policy beliefs, intentionally separate from evaluation ground truth.

These values are human-authored planning estimates used only by the smart
policy. They must not be derived from or synchronized with outcome_model.py.
"""

from __future__ import annotations


POLICY_ATTEMPT_DECAY_FACTOR = 0.65

POLICY_BELIEF_TABLE: dict[str, dict[str, float]] = {
    "insufficient_funds": {
        # Immediate retries are unlikely before funds are replenished.
        "retry_now": 0.08,
        # Four hours may help only when a balance changes during the day.
        "retry_4h": 0.14,
        # A one-day delay gives a modest chance of a balance refresh.
        "retry_24h": 0.20,
        # Salary-window timing is estimated highest from typical income-crediting patterns.
        "salary_window": 0.82,
    },
    "bank_timeout": {
        # A fresh retry may succeed when a transient bank timeout clears.
        "retry_now": 0.72,
        # A short delay can avoid a brief infrastructure blip.
        "retry_4h": 0.58,
        # A day later remains plausible but loses transient-error value.
        "retry_24h": 0.42,
        # Salary timing has no special relevance to a bank timeout.
        "salary_window": 0.35,
    },
    "expired_card": {
        # An expired card needs an update rather than an immediate retry.
        "retry_now": 0.01,
        # Delaying does not solve an expired credential without intervention.
        "retry_4h": 0.01,
        # A one-day delay still has near-zero recovery without a card update.
        "retry_24h": 0.01,
        # Salary timing cannot repair an expired card.
        "salary_window": 0.01,
    },
    "gateway_error": {
        # An immediate retry can work after a transient gateway error clears.
        "retry_now": 0.68,
        # A short delay is estimated strongest for a short-lived outage.
        "retry_4h": 0.74,
        # A day later remains viable but unnecessarily delays recovery.
        "retry_24h": 0.56,
        # Salary timing has no causal advantage for gateway availability.
        "salary_window": 0.40,
    },
    "issuer_decline": {
        # An immediate repeat is unlikely to change an issuer decision.
        "retry_now": 0.10,
        # A short delay may help if the issuer limit or state refreshes.
        "retry_4h": 0.20,
        # A day later is the strongest modeled non-intervention opportunity.
        "retry_24h": 0.32,
        # Salary timing offers a small potential improvement for account balance changes.
        "salary_window": 0.28,
    },
    "threeds_dropoff": {
        # Retrying immediately gives the customer the quickest chance to complete 3DS.
        "retry_now": 0.38,
        # Four hours retains some recovery opportunity but loses customer attention.
        "retry_4h": 0.26,
        # A day later is less likely once the authentication flow was abandoned.
        "retry_24h": 0.16,
        # Salary timing does not address authentication abandonment.
        "salary_window": 0.12,
    },
}


def estimate_success_probability(reason: str, candidate_window: str, attempt_number: int) -> float:
    """Return the policy's own estimate for a candidate retry window."""

    if reason not in POLICY_BELIEF_TABLE:
        raise ValueError(f"Unsupported failure reason: {reason}")
    if candidate_window not in POLICY_BELIEF_TABLE[reason]:
        raise ValueError(f"Unsupported candidate window: {candidate_window}")
    if isinstance(attempt_number, bool) or not isinstance(attempt_number, int) or attempt_number < 1:
        raise ValueError("attempt_number must be an integer greater than or equal to 1")
    return POLICY_BELIEF_TABLE[reason][candidate_window] * (POLICY_ATTEMPT_DECAY_FACTOR ** (attempt_number - 1))
