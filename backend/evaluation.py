"""Held-out baseline-versus-agent evaluation for Recoverly.

Only ``test_set_v1.csv`` can supply reportable metrics. Policies produce an
action and timing; this module delegates the resulting recovery simulation to
the separate policy-blind outcome model.
"""

from __future__ import annotations

import csv
from datetime import datetime
import logging
from pathlib import Path
from typing import Any

from backend.classifier import _load_artifact, _records_to_matrix
from backend.data.outcome_model import OutcomeSimulationError, retry_success_probability, simulate_retry_outcome
from backend.features import MODEL_FEATURE_COLUMNS, TEST_SET_PATH, impute_feature_records, verify_frozen_test_set
from backend.retry_policy import decide_action


LOGGER = logging.getLogger(__name__)
RETRY_OUTCOMES_PATH = Path(__file__).parent / "data" / "retry_outcomes.csv"
POLICIES = ("baseline", "smart")
RETRY_ACTIONS = frozenset({"retry_now", "retry_scheduled"})
OUTCOME_FIELDNAMES = ("txn_id", "policy", "retry_at", "recovered", "recovered_amount")


def evaluate_held_out_policies(
    test_path: Path | str = TEST_SET_PATH,
    output_path: Path | str = RETRY_OUTCOMES_PATH,
) -> dict[str, Any]:
    """Evaluate baseline and smart policies on the one frozen held-out split."""

    resolved_test_path = Path(test_path).resolve()
    if resolved_test_path != TEST_SET_PATH.resolve():
        raise ValueError("Reported evaluation may only use test_set_v1.csv")
    verify_frozen_test_set(resolved_test_path)

    raw_rows = _read_csv(resolved_test_path)
    eligible_rows, excluded_txn_ids = _eligible_rows(raw_rows)
    predictions = _predict_rows(eligible_rows)
    outcome_rows = _evaluate_predictions(eligible_rows, predictions)
    _assert_same_transaction_set(outcome_rows)
    _write_outcomes(Path(output_path), outcome_rows)

    metrics = compute_metrics(outcome_rows, len(eligible_rows))
    return {
        "metrics": metrics,
        "eligible_txn_ids": [str(row["txn_id"]) for row in eligible_rows],
        "excluded_txn_ids": excluded_txn_ids,
    }


def compute_metrics(outcome_rows: list[dict[str, Any]], total_failed_attempts: int) -> dict[str, dict[str, float]]:
    """Apply the exact dashboard formulas from Evaluation Contract 0.12."""

    if total_failed_attempts <= 0:
        raise ValueError("total_failed_attempts must be greater than zero")

    policy_metrics: dict[str, dict[str, float]] = {}
    for policy in POLICIES:
        rows = [row for row in outcome_rows if row["policy"] == policy]
        recovered_count = sum(bool(row["recovered"]) for row in rows)
        recovered_amount = sum(float(row["recovered_amount"]) for row in rows)
        policy_metrics[policy] = {
            "recovery_rate": recovered_count / total_failed_attempts,
            "rupees_recovered": round(recovered_amount, 2),
        }

    policy_metrics["comparison"] = {
        "incremental_uplift": round(
            policy_metrics["smart"]["rupees_recovered"] - policy_metrics["baseline"]["rupees_recovered"],
            2,
        )
    }
    return policy_metrics


def _eligible_rows(raw_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str]]:
    eligible_rows: list[dict[str, Any]] = []
    excluded_txn_ids: list[str] = []
    for row in raw_rows:
        transaction = _outcome_transaction(row)
        try:
            # Validation resides in the outcome model so invalid rows are excluded
            # before either policy is invoked, preserving a matched evaluation set.
            retry_success_probability(transaction, _decision_at(row))
        except OutcomeSimulationError as error:
            txn_id = str(row.get("txn_id", "unknown"))
            LOGGER.warning("Excluding unsimulatable held-out transaction %s: %s", txn_id, error)
            excluded_txn_ids.append(txn_id)
            continue
        eligible_rows.append(dict(row))
    return eligible_rows, excluded_txn_ids


def _predict_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    artifact = _load_artifact()
    prepared_rows = impute_feature_records(rows, artifact["numeric_medians"])
    matrix, _ = _records_to_matrix(prepared_rows)
    probabilities = artifact["model"].predict_proba(matrix)
    class_indices = probabilities.argmax(axis=1)
    return [
        {
            "reason": str(artifact["model"].classes_[class_index]),
            "confidence": float(probability[class_index]),
        }
        for probability, class_index in zip(probabilities, class_indices, strict=True)
    ]


def _evaluate_predictions(
    rows: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    outcome_rows: list[dict[str, Any]] = []
    for row, prediction in zip(rows, predictions, strict=True):
        transaction = _outcome_transaction(row)
        context = {
            "amount": float(row["amount"]),
            "attempt_number": int(float(row["attempt_number"])),
            "decision_at": _decision_at(row),
            "scheduled_retry_ats": [],
        }
        for policy in POLICIES:
            decision = decide_action(prediction["reason"], prediction["confidence"], context, policy=policy)
            recovered = False
            recovered_amount = 0.0
            retry_at = decision["retry_at"]
            if decision["action"] in RETRY_ACTIONS:
                recovered = simulate_retry_outcome(transaction, retry_at)
                recovered_amount = float(row["amount"]) if recovered else 0.0
            outcome_rows.append(
                {
                    "txn_id": str(row["txn_id"]),
                    "policy": policy,
                    "retry_at": retry_at,
                    "recovered": recovered,
                    "recovered_amount": recovered_amount,
                }
            )
    return outcome_rows


def _assert_same_transaction_set(outcome_rows: list[dict[str, Any]]) -> None:
    transaction_ids_by_policy = {
        policy: {str(row["txn_id"]) for row in outcome_rows if row["policy"] == policy}
        for policy in POLICIES
    }
    if transaction_ids_by_policy["baseline"] != transaction_ids_by_policy["smart"]:
        raise RuntimeError("Policies were not evaluated on the same held-out transactions")


def _outcome_transaction(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "txn_id": str(row.get("txn_id", "")),
        "failure_reason": row.get("failure_reason"),
        "attempt_number": int(float(row["attempt_number"])),
        "card_updated": str(row.get("card_updated", "")).lower() == "true",
    }


def _decision_at(row: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(str(row["timestamp"]))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _write_outcomes(path: Path, outcome_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTCOME_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            {
                "txn_id": row["txn_id"],
                "policy": row["policy"],
                "retry_at": row["retry_at"].isoformat() if row["retry_at"] else "",
                "recovered": str(row["recovered"]),
                "recovered_amount": f"{float(row['recovered_amount']):.2f}",
            }
            for row in outcome_rows
        )


if __name__ == "__main__":
    result = evaluate_held_out_policies()
    baseline = result["metrics"]["baseline"]
    smart = result["metrics"]["smart"]
    uplift = result["metrics"]["comparison"]["incremental_uplift"]
    print(
        "held-out evaluation: "
        f"baseline={baseline['recovery_rate']:.1%} / INR {baseline['rupees_recovered']:.2f}; "
        f"smart={smart['recovery_rate']:.1%} / INR {smart['rupees_recovered']:.2f}; "
        f"uplift=INR {uplift:.2f}"
    )
