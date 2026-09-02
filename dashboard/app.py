"""Streamlit dashboard for held-out Recoverly recovery results."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.api.main import predict_recovery_action
from backend.api.schemas import RecoveryActionRequest
from backend.classifier import _load_artifact, _records_to_matrix
from backend.evaluation import POLICIES, RETRY_OUTCOMES_PATH, compute_metrics
from backend.features import TEST_SET_PATH, impute_feature_records, verify_frozen_test_set
from backend.retry_policy import decide_action


DEMO_CASES_PATH = Path(__file__).parents[1] / "backend" / "data" / "demo_cases.csv"
FEATURE_IMPORTANCES_PATH = (
    Path(__file__).parents[1] / "backend" / "models" / "feature_importances.csv"
)
ACTION_STATUS_BADGES = {
    "escalate_manual_review": "🟡 REVIEW",
    "no_action": "🔴 BLOCKED",
    "retry_now": "🟢 AUTO",
    "retry_scheduled": "🟢 AUTO",
    "notify_update_card": "🟢 AUTO",
}
LIVE_SIMULATION_API_URL = os.environ.get(
    "RECOVERLY_API_URL", "http://127.0.0.1:8000/predict-recovery-action"
)


def load_dashboard_metrics(
    test_path: Path | str = TEST_SET_PATH,
    outcomes_path: Path | str = RETRY_OUTCOMES_PATH,
) -> dict[str, dict[str, float]]:
    """Return reportable metrics only when sourced from the frozen test split."""

    resolved_test_path = Path(test_path).resolve()
    if resolved_test_path != TEST_SET_PATH.resolve():
        raise ValueError("Dashboard metrics may only use test_set_v1.csv")
    verify_frozen_test_set(resolved_test_path)

    test_rows = _read_csv(resolved_test_path)
    outcome_rows = _read_outcome_rows(Path(outcomes_path))
    _assert_outcomes_match_frozen_test_set(outcome_rows, test_rows)
    return compute_metrics(outcome_rows, total_failed_attempts=len(test_rows))


def load_demo_cases(demo_cases_path: Path | str = DEMO_CASES_PATH) -> list[dict[str, str]]:
    """Load illustrative cases that remain separate from reported metrics."""

    return _read_csv(Path(demo_cases_path))


def load_feature_importances(
    feature_importances_path: Path | str = FEATURE_IMPORTANCES_PATH,
) -> list[dict[str, float | str]]:
    """Load persisted classifier importances for dashboard transparency."""

    rows = _read_csv(Path(feature_importances_path))
    if any({"feature", "importance"} - row.keys() for row in rows):
        raise ValueError("Feature importances do not match the classifier artifact schema")
    return [{"feature": row["feature"], "importance": float(row["importance"])} for row in rows]


def load_opportunity_matrix(
    test_path: Path | str = TEST_SET_PATH,
    outcomes_path: Path | str = RETRY_OUTCOMES_PATH,
) -> list[dict[str, float | str]]:
    """Build predicted smart-policy opportunities from frozen evaluation inputs."""

    test_rows = _load_frozen_test_rows(test_path)
    outcome_rows = _read_outcome_rows(Path(outcomes_path))
    _assert_outcomes_match_frozen_test_set(outcome_rows, test_rows)
    predictions = _predict_opportunity_rows(test_rows)

    opportunities: list[dict[str, float | str]] = []
    for row, prediction in zip(test_rows, predictions, strict=True):
        amount = float(row["amount"])
        decision = decide_action(
            prediction["reason"],
            prediction["confidence"],
            {
                "amount": amount,
                "attempt_number": int(float(row["attempt_number"])),
                "decision_at": datetime.fromisoformat(row["timestamp"]),
                "scheduled_retry_ats": [],
            },
        )
        expected_value = float(decision["expected_value"])
        opportunities.append(
            {
                "Failure reason": prediction["reason"],
                "Policy belief recovery probability": expected_value / amount if amount else 0.0,
                "Transaction amount (₹)": amount,
                "Expected Recovery Value (₹)": expected_value,
            }
        )
    return opportunities


def load_revenue_at_risk(test_path: Path | str = TEST_SET_PATH) -> list[dict[str, float | str]]:
    """Group frozen failed-transaction value by the observed failure reason."""

    revenue_by_reason: dict[str, float] = {}
    for row in _load_frozen_test_rows(test_path):
        reason = row["failure_reason"]
        revenue_by_reason[reason] = revenue_by_reason.get(reason, 0.0) + float(row["amount"])
    return [
        {"Failure reason": reason, "Revenue at risk (₹)": round(amount, 2)}
        for reason, amount in sorted(revenue_by_reason.items(), key=lambda item: item[1], reverse=True)
    ]


def run_demo_case(case: dict[str, str]) -> dict[str, Any]:
    """Invoke the API handler with one demo case's leakage-safe request body."""

    request = RecoveryActionRequest(
        txn_id=case["txn_id"],
        amount=float(case["amount"]),
        hour_of_day=int(case["hour_of_day"]),
        day_of_month=int(case["day_of_month"]),
        card_type=case["card_type"],
        is_recurring=case["is_recurring"].lower() == "true",
        customer_past_failure_count=int(case["customer_past_failure_count"]),
        customer_past_failure_reasons_distribution=json.loads(
            case["customer_past_failure_reasons_distribution"]
        ),
        attempt_number=int(case["attempt_number"]),
        decision_at=case["decision_at"],
        scheduled_retry_ats=[],
    )
    return predict_recovery_action(request).model_dump(mode="json")


def run_live_simulation(
    payload: dict[str, Any], api_url: str = LIVE_SIMULATION_API_URL
) -> dict[str, Any]:
    """Send a user-entered, leakage-safe scenario to the running FastAPI endpoint."""

    request = Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise ValueError(f"Recovery API rejected the simulation: {details}") from error
    except URLError as error:
        raise ConnectionError(
            "Could not reach the Recovery API. Start it at the configured RECOVERLY_API_URL first."
        ) from error


def render_dashboard() -> None:
    """Render the Streamlit dashboard when launched with streamlit run."""

    import streamlit as st

    metrics = load_dashboard_metrics()
    demo_cases = load_demo_cases()
    feature_importances = load_feature_importances()
    opportunity_matrix = load_opportunity_matrix()
    revenue_at_risk = load_revenue_at_risk()
    baseline = metrics["baseline"]
    smart = metrics["smart"]
    uplift = metrics["comparison"]["incremental_uplift"]

    st.set_page_config(page_title="Recoverly", layout="wide")
    st.title("Recoverly: recovery decisions with evidence")
    st.caption(
        "Actual figures use only the frozen held-out test set and the policy-blind outcome model. "
        "Predicted values are policy belief-table estimates. Demo cases below are illustrative "
        "train/dev examples and are not included in the actual metrics."
    )

    baseline_column, smart_column, uplift_column = st.columns(3)
    baseline_column.metric("Actual baseline recovered", _format_inr(baseline["rupees_recovered"]))
    baseline_column.caption(f"Actual recovery rate: {baseline['recovery_rate']:.1%}")
    smart_column.metric("Actual agent recovered", _format_inr(smart["rupees_recovered"]))
    smart_column.caption(f"Actual recovery rate: {smart['recovery_rate']:.1%}")
    uplift_column.metric("Actual incremental uplift", _format_inr(uplift))
    uplift_column.caption("Actual smart ₹ recovered − actual baseline ₹ recovered")

    st.subheader("Actual held-out policy comparison")
    st.bar_chart(
        {
            "Actual baseline recovery rate": baseline["recovery_rate"],
            "Actual agent recovery rate": smart["recovery_rate"],
        }
    )
    st.dataframe(
        [
            {
                "Policy": "Baseline",
                "Actual recovery rate": f"{baseline['recovery_rate']:.1%}",
                "Actual ₹ recovered": _format_inr(baseline["rupees_recovered"]),
            },
            {
                "Policy": "Agent",
                "Actual recovery rate": f"{smart['recovery_rate']:.1%}",
                "Actual ₹ recovered": _format_inr(smart["rupees_recovered"]),
            },
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Model transparency")
    st.caption("Feature importance from the trained failure-reason classifier.")
    st.bar_chart({row["feature"]: row["importance"] for row in feature_importances})

    st.subheader("Recovery opportunity matrix")
    st.caption(
        "Predicted smart-policy opportunities from the frozen held-out test set. "
        "The x-axis is the policy-belief recovery probability for the selected action, "
        "calculated as Expected Recovery Value ÷ transaction amount; it is not an actual outcome."
    )
    st.scatter_chart(
        opportunity_matrix,
        x="Policy belief recovery probability",
        y="Transaction amount (₹)",
        size="Expected Recovery Value (₹)",
        color="Failure reason",
    )

    st.subheader("Revenue at risk by failure reason")
    st.caption("Failed transaction value grouped by observed failure reason in the frozen held-out test set.")
    st.bar_chart(revenue_at_risk, x="Failure reason", y="Revenue at risk (₹)")

    st.subheader("Illustrative recovery action")
    selected_case = st.selectbox(
        "Choose a demo case",
        demo_cases,
        format_func=lambda case: f"{case['expected_action']} — {case['txn_id'][:8]}",
    )
    st.caption(
        f"Source split: {selected_case['source_split']}. "
        f"Expected illustrative action: {selected_case['expected_action']}."
    )
    if st.button("Run recovery action", type="primary"):
        _render_decision_response(st, run_demo_case(selected_case))

    st.subheader("Live recovery simulator")
    st.caption(
        "Live simulation — not part of held-out or demo metrics. "
        "This form sends your scenario to the running FastAPI recovery-action endpoint."
    )
    with st.form("live-recovery-simulator"):
        amount = st.number_input("Amount (₹)", min_value=0.0, value=1_000.0, step=100.0)
        hour_of_day = st.slider("Hour of day", min_value=0, max_value=23, value=9)
        day_of_month = st.slider("Day of month", min_value=1, max_value=31, value=1)
        card_type = st.selectbox("Card type", ("UPI", "debit", "credit", "netbanking"))
        is_recurring = st.checkbox("Recurring payment", value=False)
        past_failure_count = st.number_input(
            "Customer past failure count", min_value=0, value=0, step=1
        )
        history_distribution = st.text_area(
            "Past failure reason distribution (JSON)", value="{}"
        )
        attempt_number = st.number_input("Attempt number", min_value=1, value=1, step=1)
        decision_date = st.date_input("Decision date")
        decision_time = st.time_input("Decision time")
        submitted = st.form_submit_button("Simulate recovery action", type="primary")

    if submitted:
        try:
            simulation_payload = {
                "amount": float(amount),
                "hour_of_day": hour_of_day,
                "day_of_month": day_of_month,
                "card_type": card_type,
                "is_recurring": is_recurring,
                "customer_past_failure_count": int(past_failure_count),
                "customer_past_failure_reasons_distribution": parse_history_distribution(
                    history_distribution
                ),
                "attempt_number": int(attempt_number),
                "decision_at": datetime.combine(decision_date, decision_time).isoformat(),
                "scheduled_retry_ats": [],
            }
            _render_decision_response(st, run_live_simulation(simulation_payload))
        except (ConnectionError, ValueError) as error:
            st.error(str(error))


def _render_decision_response(st: Any, response: dict[str, Any]) -> None:
    """Render a previously made API decision without changing it."""

    st.write(f"Reason: {response['reason']} ({response['confidence']:.2%} confidence)")
    st.write(f"Action: {response['action']}")
    st.markdown(f"### Status: {ACTION_STATUS_BADGES[response['action']]}")
    if response["retry_at"]:
        st.write(f"Retry at: {response['retry_at']}")
    st.write(f"Expected Recovery Value (Predicted): {_format_inr(response['expected_value'])}")

    candidates = response.get("candidates")
    if candidates:
        st.caption("Smart-policy candidate comparison (predicted Expected Recovery Value).")
        st.dataframe(
            [
                {
                    "Action": candidate["action"],
                    "Retry at": candidate["retry_at"],
                    "Expected Recovery Value (₹)": _format_inr(candidate["expected_value"]),
                }
                for candidate in candidates
            ],
            hide_index=True,
            use_container_width=True,
        )

    explanation_source = {
        "ai_generated": "AI generated",
        "template_fallback": "Template fallback",
    }[response["explanation_source"]]
    st.caption(f"Explanation source: {explanation_source}")
    st.info(response["explanation"])


def parse_history_distribution(value: str) -> dict[str, float]:
    """Validate the API's optional historical-failure feature from form JSON."""

    try:
        distribution = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("Past failure reason distribution must be valid JSON.") from error
    if not isinstance(distribution, dict):
        raise ValueError("Past failure reason distribution must be a JSON object.")
    try:
        return {str(reason): float(weight) for reason, weight in distribution.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("Past failure reason distribution values must be numeric.") from error


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _load_frozen_test_rows(test_path: Path | str) -> list[dict[str, str]]:
    resolved_test_path = Path(test_path).resolve()
    if resolved_test_path != TEST_SET_PATH.resolve():
        raise ValueError("Dashboard charts may only use test_set_v1.csv")
    verify_frozen_test_set(resolved_test_path)
    return _read_csv(resolved_test_path)


def _classifier_features(row: dict[str, str]) -> dict[str, str]:
    return {
        "amount": row["amount"],
        "hour_of_day": row["hour_of_day"],
        "day_of_month": row["day_of_month"],
        "card_type": row["card_type"],
        "is_recurring": row["is_recurring"],
        "customer_past_failure_count": row["customer_past_failure_count"],
        "customer_past_failure_reasons_distribution": row[
            "customer_past_failure_reasons_distribution"
        ],
        "attempt_number": row["attempt_number"],
    }


def _predict_opportunity_rows(rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    """Batch existing classifier inference so chart rendering stays responsive."""

    artifact = _load_artifact()
    features = [_classifier_features(row) for row in rows]
    prepared_features = impute_feature_records(features, artifact["numeric_medians"])
    matrix, _ = _records_to_matrix(prepared_features)
    probabilities = artifact["model"].predict_proba(matrix)
    return [
        {
            "reason": str(artifact["model"].classes_[max(range(len(row)), key=row.__getitem__)]),
            "confidence": float(max(row)),
        }
        for row in probabilities
    ]


def _read_outcome_rows(path: Path) -> list[dict[str, Any]]:
    raw_rows = _read_csv(path)
    required_columns = {"txn_id", "policy", "retry_at", "recovered", "recovered_amount"}
    if raw_rows and not required_columns.issubset(raw_rows[0]):
        raise ValueError("Outcome data does not match the frozen held-out evaluation schema")
    return [
        {
            "txn_id": row["txn_id"],
            "policy": row["policy"],
            "retry_at": row["retry_at"],
            "recovered": row["recovered"].lower() == "true",
            "recovered_amount": float(row["recovered_amount"]),
        }
        for row in raw_rows
    ]


def _assert_outcomes_match_frozen_test_set(
    outcome_rows: list[dict[str, Any]], test_rows: list[dict[str, str]]
) -> None:
    frozen_ids = {row["txn_id"] for row in test_rows}
    for policy in POLICIES:
        outcome_ids = {row["txn_id"] for row in outcome_rows if row["policy"] == policy}
        if outcome_ids != frozen_ids:
            raise ValueError(f"{policy} outcomes do not match the frozen held-out test set")


def _format_inr(value: float) -> str:
    return f"₹{value:,.2f}"


if __name__ == "__main__":
    render_dashboard()
