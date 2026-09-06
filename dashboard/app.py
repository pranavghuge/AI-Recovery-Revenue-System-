"""Streamlit dashboard for held-out Recoverly recovery results."""

from __future__ import annotations

import csv
import json
import os
import socket
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import altair as alt
import pandas as pd

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
    "escalate_manual_review": ("REVIEW", "review"),
    "no_action": ("BLOCKED", "blocked"),
    "retry_now": ("AUTO", "auto"),
    "retry_scheduled": ("AUTO", "auto"),
    "notify_update_card": ("AUTO", "auto"),
}
LIVE_SIMULATION_API_URL = os.environ.get(
    "RECOVERLY_API_URL", "http://127.0.0.1:8000/predict-recovery-action"
)
LIVE_SIMULATION_TIMEOUT_SECONDS = 15
STYLES_PATH = Path(__file__).with_name("styles.css")
THEME = {
    "bg_base": "#0E1116",
    "bg_panel": "#13181E",
    "ink_primary": "#E6E8EB",
    "ink_secondary": "#8A929E",
    "interactive": "#5D6673",
    "signal": "#3DDC84",
    "predicted": "#5B8DEF",
    "warn": "#E8A33D",
    "block": "#D9534F",
}


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
        with urlopen(request, timeout=LIVE_SIMULATION_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except (TimeoutError, socket.timeout) as error:
        raise ConnectionError(
            "The Recovery API did not respond within 15 seconds. "
            "Check that the backend is running, then try the simulation again."
        ) from error
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
    baseline_recovery_rate = f"{baseline['recovery_rate']:.1%}"
    smart_recovery_rate = f"{smart['recovery_rate']:.1%}"

    st.set_page_config(page_title="Recoverly", page_icon="💳", layout="wide")
    _load_styles(st)
    hero_section = st.container(key="hero-metrics", gap=0)
    hero_section.markdown(
        '<h1 class="dashboard-title">Recoverly: recovery decisions with evidence</h1>',
        unsafe_allow_html=True,
    )
    hero_section.caption(
        "Actual figures use only the frozen held-out test set and the policy-blind outcome model. "
        "Predicted values are policy belief-table estimates. Demo cases below are illustrative "
        "train/dev examples and are not included in the actual metrics."
    )

    baseline_column, smart_column, uplift_column = hero_section.columns(3)
    baseline_column.metric("Actual baseline recovered", _format_inr(baseline["rupees_recovered"]))
    baseline_column.markdown(
        "Actual recovery rate: "
        f"{_actual_number(baseline_recovery_rate)}",
        unsafe_allow_html=True,
    )
    smart_column.metric("Actual agent recovered", _format_inr(smart["rupees_recovered"]))
    smart_column.markdown(
        "Actual recovery rate: "
        f"{_actual_number(smart_recovery_rate)}",
        unsafe_allow_html=True,
    )
    uplift_column.metric("Actual incremental uplift", _format_inr(uplift))
    uplift_column.caption("Actual smart ₹ recovered − actual baseline ₹ recovered")

    comparison_section = st.container(key="policy-comparison", gap=0)
    comparison_section.subheader("Actual held-out policy comparison")
    comparison_section.bar_chart(
        {
            "Actual baseline recovery rate": baseline["recovery_rate"],
            "Actual agent recovery rate": smart["recovery_rate"],
        },
        color=THEME["signal"],
    )
    actual_comparison = [
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
    ]
    comparison_section.dataframe(
        _style_semantic_columns(
            actual_comparison,
            ("Actual recovery rate", "Actual ₹ recovered"),
            "var(--signal)",
        ),
        hide_index=True,
        width="stretch",
    )

    transparency_section = st.container(key="model-transparency", gap=0)
    transparency_section.subheader("Model transparency")
    transparency_section.caption("Feature importance from the trained failure-reason classifier.")
    transparency_section.altair_chart(
        _feature_importance_chart(feature_importances), width="stretch"
    )

    opportunity_section = st.container(key="opportunity-matrix", gap=0)
    opportunity_section.subheader("Recovery opportunity matrix")
    opportunity_section.caption(
        "Predicted smart-policy opportunities from the frozen held-out test set. "
        "The x-axis is the policy-belief recovery probability for the selected action, "
        "calculated as Expected Recovery Value ÷ transaction amount; it is not an actual outcome."
    )
    opportunity_section.altair_chart(
        _opportunity_matrix_chart(opportunity_matrix), width="stretch"
    )

    risk_section = st.container(key="revenue-at-risk", gap=0)
    risk_section.subheader("Revenue at risk by failure reason")
    risk_section.caption(
        "Failed transaction value grouped by observed failure reason in the frozen held-out test set."
    )
    risk_section.bar_chart(
        revenue_at_risk,
        x="Failure reason",
        y="Revenue at risk (₹)",
        color=THEME["signal"],
    )

    demo_section = st.container(key="illustrative-action", gap=0)
    demo_section.subheader("Illustrative recovery action")
    selected_case = demo_section.selectbox(
        "Choose a demo case",
        demo_cases,
        format_func=lambda case: f"{case['expected_action']} — {case['txn_id'][:8]}",
    )
    demo_section.caption(
        f"Source split: {selected_case['source_split']}. "
        f"Expected illustrative action: {selected_case['expected_action']}."
    )
    if demo_section.button("Run recovery action", type="primary"):
        _render_decision_response(st, run_demo_case(selected_case))

    live_section = st.container(key="live-simulator", gap=0)
    live_section.subheader("Live recovery simulator")
    live_section.markdown(
        '<p class="live-simulator-disclosure">'
        "Live simulation — not part of held-out or demo metrics. "
        "This form sends your scenario to the running FastAPI recovery-action endpoint."
        "</p>",
        unsafe_allow_html=True,
    )
    with live_section.form("live-recovery-simulator"):
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
        feedback = live_section.empty()
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
            feedback.markdown(_simulator_loading_feedback(), unsafe_allow_html=True)
            response = run_live_simulation(simulation_payload)
            feedback.empty()
            _render_decision_response(st, response)
        except (ConnectionError, ValueError) as error:
            feedback.markdown(_simulator_error_feedback(str(error)), unsafe_allow_html=True)


def _render_decision_response(st: Any, response: dict[str, Any]) -> None:
    """Render one API response as a read-only classification-to-explanation trace."""

    explanation_source = {
        "ai_generated": "AI generated",
        "template_fallback": "Template fallback",
    }[response["explanation_source"]]
    confidence = f"{response['confidence']:.2%}"

    trace_section = st.container(key="decision-trace", gap=0)
    trace_section.subheader("Decision trace")
    trace_section.caption("Actual outputs from this single recovery-action API response.")
    classify_column, decide_column, explain_column = trace_section.columns(3)
    with classify_column:
        st.markdown(
            _trace_panel(
                1,
                "Classify",
                _trace_row("Reason", escape(str(response["reason"])))
                + _trace_row("Confidence", _diagnostic_number(confidence)),
            ),
            unsafe_allow_html=True,
        )
    with decide_column:
        decision_rows = _trace_row("Action", escape(str(response["action"])))
        decision_rows += _status_badge(str(response["action"]))
        if response["retry_at"]:
            decision_rows += _trace_row("Retry at", escape(str(response["retry_at"])))
        st.markdown(
            _trace_panel(
                2,
                "Decide",
                decision_rows
                + _trace_row(
                    "Expected Recovery Value",
                    _predicted_number(_format_inr(response["expected_value"])),
                ),
            ),
            unsafe_allow_html=True,
        )
    with explain_column:
        st.markdown(
            _trace_panel(
                3,
                "Explain",
                _trace_row("Source", escape(explanation_source))
                + f'<div class="trace-explanation">{escape(str(response["explanation"]))}</div>',
            ),
            unsafe_allow_html=True,
        )

    candidates = response.get("candidates")
    if candidates:
        trace_section.caption("Scored policy candidates (predicted Expected Recovery Value).")
        candidate_rows = [
            {
                "Action": candidate["action"],
                "Retry at": candidate["retry_at"],
                "Expected Recovery Value (₹)": _format_inr(candidate["expected_value"]),
            }
            for candidate in candidates
        ]
        trace_section.dataframe(
            _style_candidate_rows(candidate_rows),
            hide_index=True,
            width="stretch",
        )


def _load_styles(st: Any) -> None:
    """Load the local presentation stylesheet without altering dashboard behavior."""

    st.markdown(
        f"<style>{STYLES_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def _actual_number(value: str) -> str:
    """Return a display-only wrapper for a verified held-out number."""

    return f'<span class="actual-number">{value}</span>'


def _predicted_number(value: str) -> str:
    """Return a display-only wrapper for a model-estimated number."""

    return f'<span class="predicted-number">{value}</span>'


def _diagnostic_number(value: str) -> str:
    """Return a display-only wrapper for a model diagnostic, not an estimate."""

    return f'<span class="diagnostic-number">{value}</span>'


def _status_badge(action: str) -> str:
    """Return the existing action status as a presentation-only semantic pill."""

    label, tone = ACTION_STATUS_BADGES[action]
    return (
        f'<span class="action-badge action-badge--{tone}">'
        '<span class="action-badge-dot"></span>'
        f"{label}</span>"
    )


def _trace_panel(step: int, title: str, content: str) -> str:
    """Return one display-only panel in the existing decision sequence."""

    return (
        '<section class="decision-trace-panel">'
        '<div class="decision-trace-header">'
        f'<span class="decision-trace-marker">{step}</span>'
        f'<h4 class="decision-trace-title">{escape(title)}</h4>'
        "</div>"
        f"{content}"
        "</section>"
    )


def _trace_row(label: str, value: str) -> str:
    """Return one display-only label/value row within a decision-trace panel."""

    return (
        '<div class="decision-trace-row">'
        f'<span class="decision-trace-label">{escape(label)}</span>'
        f'<span class="decision-trace-value">{value}</span>'
        "</div>"
    )


def _simulator_loading_feedback() -> str:
    """Return the display-only loading state for an in-flight simulator request."""

    return (
        '<div class="simulator-feedback simulator-feedback--loading">'
        '<span class="simulator-loading-indicator" aria-hidden="true"></span>'
        "<span>Running recovery decision…</span>"
        "</div>"
    )


def _simulator_error_feedback(message: str) -> str:
    """Return an escaped, display-only error state without a raw traceback."""

    return (
        '<div class="simulator-feedback simulator-feedback--error">'
        "<strong>Recovery simulation unavailable.</strong>"
        f"<span>{escape(message)}</span>"
        "</div>"
    )


def _style_semantic_columns(
    rows: list[dict[str, Any]], columns: tuple[str, ...], color: str
) -> Any:
    """Style existing read-only table numbers without changing their values."""

    return pd.DataFrame(rows).style.set_properties(
        subset=list(columns),
        **{
            "color": color,
            "font-variant-numeric": "tabular-nums",
            "font-weight": "600",
        },
    )


def _style_candidate_rows(rows: list[dict[str, Any]]) -> Any:
    """Keep ranked policy candidates readable without adding row emphasis."""

    dataframe = pd.DataFrame(rows)
    styles = pd.DataFrame(
        # Streamlit renders Styler cell rules in an isolated dataframe surface,
        # so use the established theme values directly rather than CSS variables.
        "color: #E6E8EB;",
        index=dataframe.index,
        columns=dataframe.columns,
    )
    return dataframe.style.apply(lambda _: styles, axis=None).set_properties(
        **{"font-variant-numeric": "tabular-nums"}
    )


def _feature_importance_chart(rows: list[dict[str, float | str]]) -> alt.Chart:
    """Render non-negative classifier importances with a zero-based axis."""

    return (
        alt.Chart(pd.DataFrame(rows))
        .mark_bar(color=THEME["ink_secondary"])
        .encode(
            x=alt.X("feature:N", sort=None, title="Feature"),
            y=alt.Y(
                "importance:Q",
                scale=alt.Scale(domainMin=0),
                title="Feature importance",
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("importance:Q", title="Importance", format=".8f"),
            ],
        )
        .configure(background=THEME["bg_panel"])
        .configure_axis(
            domainColor=THEME["ink_secondary"],
            gridColor=THEME["ink_secondary"],
            labelColor=THEME["ink_primary"],
            titleColor=THEME["ink_primary"],
        )
        .configure_view(stroke=THEME["ink_secondary"])
    )


def _opportunity_matrix_chart(rows: list[dict[str, float | str]]) -> alt.Chart:
    """Return the existing predicted-opportunity chart with explicit theme colors."""

    return (
        alt.Chart(pd.DataFrame(rows))
        .mark_circle(color=THEME["predicted"], opacity=0.8)
        .encode(
            x=alt.X(
                "Policy belief recovery probability:Q",
                title="Policy belief recovery probability",
            ),
            y=alt.Y("Transaction amount (₹):Q", title="Transaction amount (₹)"),
            size=alt.Size("Expected Recovery Value (₹):Q"),
            tooltip=[
                alt.Tooltip("Failure reason:N", title="Failure reason"),
                alt.Tooltip(
                    "Policy belief recovery probability:Q",
                    title="Policy belief recovery probability",
                ),
                alt.Tooltip("Transaction amount (₹):Q", title="Transaction amount (₹)"),
                alt.Tooltip(
                    "Expected Recovery Value (₹):Q", title="Expected Recovery Value (₹)"
                ),
            ],
        )
        .configure(background=THEME["bg_panel"])
        .configure_axis(
            domainColor=THEME["ink_secondary"],
            gridColor=THEME["ink_secondary"],
            labelColor=THEME["ink_primary"],
            titleColor=THEME["ink_primary"],
        )
        .configure_view(stroke=THEME["ink_secondary"])
    )


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
