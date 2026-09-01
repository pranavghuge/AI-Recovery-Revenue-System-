"""Failure-reason classifier trained only on leakage-safe Recoverly features."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

from backend.features import (
    MODEL_FEATURE_COLUMNS,
    TEST_SET_PATH,
    TRAIN_SET_PATH,
    DEV_SET_PATH,
    assert_leakage_safe_feature_columns,
    impute_feature_records,
    verify_frozen_test_set,
)


MODELS_DIRECTORY = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIRECTORY / "failure_reason_classifier.joblib"
FEATURE_IMPORTANCES_PATH = MODELS_DIRECTORY / "feature_importances.csv"
FAILURE_REASONS = (
    "insufficient_funds",
    "bank_timeout",
    "expired_card",
    "gateway_error",
    "issuer_decline",
    "threeds_dropoff",
)
CARD_TYPES = ("UPI", "debit", "credit", "netbanking", "unknown")
RECURRING_VALUES = ("true", "false", "unknown")


def predict_failure_reason(features: dict) -> dict:
    """Return the predicted failure reason and its maximum class probability."""

    artifact = _load_artifact()
    assert_leakage_safe_feature_columns(MODEL_FEATURE_COLUMNS)
    prepared_features = impute_feature_records([features], artifact["numeric_medians"])
    matrix, _ = _records_to_matrix(prepared_features)
    probabilities = artifact["model"].predict_proba(matrix)[0]
    class_index = int(np.argmax(probabilities))
    return {
        "reason": str(artifact["model"].classes_[class_index]),
        "confidence": float(probabilities[class_index]),
    }


def train_classifier(
    train_path: Path | str = TRAIN_SET_PATH,
    dev_path: Path | str = DEV_SET_PATH,
    test_path: Path | str = TEST_SET_PATH,
    model_path: Path | str = MODEL_PATH,
    feature_importances_path: Path | str = FEATURE_IMPORTANCES_PATH,
) -> dict[str, float]:
    """Train on train/dev records, then evaluate the persisted model on frozen test data."""

    assert_leakage_safe_feature_columns(MODEL_FEATURE_COLUMNS)
    verify_frozen_test_set(test_path)
    train_records = _read_records(Path(train_path))
    dev_records = _read_records(Path(dev_path))
    test_records = _read_records(Path(test_path))
    training_records = train_records + dev_records

    numeric_medians = _training_numeric_medians(training_records)
    prepared_training_records = impute_feature_records(training_records, numeric_medians)
    prepared_test_records = impute_feature_records(test_records, numeric_medians)
    training_matrix, feature_names = _records_to_matrix(prepared_training_records)
    test_matrix, _ = _records_to_matrix(prepared_test_records)
    training_labels = np.array([record["failure_reason"] for record in prepared_training_records])
    test_labels = np.array([record["failure_reason"] for record in prepared_test_records])

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=4,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(training_matrix, training_labels)

    artifact = {
        "model": model,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "numeric_medians": numeric_medians,
        "encoded_feature_names": feature_names,
    }
    _write_artifact(Path(model_path), artifact)
    _write_feature_importances(Path(feature_importances_path), feature_names, model.feature_importances_)

    predictions = model.predict(test_matrix)
    return {
        "macro_f1": float(f1_score(test_labels, predictions, average="macro")),
        "accuracy": float(accuracy_score(test_labels, predictions)),
    }


def evaluate_frozen_test_set(
    test_path: Path | str = TEST_SET_PATH,
    model_path: Path | str = MODEL_PATH,
) -> dict[str, float]:
    """Evaluate the already-trained artifact once against the protected test split."""

    verify_frozen_test_set(test_path)
    artifact = _load_artifact(Path(model_path))
    records = impute_feature_records(_read_records(Path(test_path)), artifact["numeric_medians"])
    matrix, _ = _records_to_matrix(records)
    labels = np.array([record["failure_reason"] for record in records])
    predictions = artifact["model"].predict(matrix)
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def _records_to_matrix(records: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    feature_names = [
        "amount",
        "hour_of_day",
        "day_of_month",
        "customer_past_failure_count",
        "attempt_number",
        *(f"card_type={value}" for value in CARD_TYPES),
        *(f"is_recurring={value}" for value in RECURRING_VALUES),
        *(f"history_reason={reason}" for reason in FAILURE_REASONS),
    ]
    matrix = []
    for record in records:
        card_type = str(record["card_type"])
        recurring = str(record["is_recurring"])
        history_distribution = _parse_history_distribution(record["customer_past_failure_reasons_distribution"])
        matrix.append(
            [
                float(record["amount"]),
                float(record["hour_of_day"]),
                float(record["day_of_month"]),
                float(record["customer_past_failure_count"]),
                float(record["attempt_number"]),
                *(float(card_type == value) for value in CARD_TYPES),
                *(float(recurring == value) for value in RECURRING_VALUES),
                *(history_distribution[reason] for reason in FAILURE_REASONS),
            ]
        )
    return np.asarray(matrix, dtype=float), feature_names


def _parse_history_distribution(value: Any) -> dict[str, float]:
    if value in (None, "", "unknown"):
        return {reason: 0.0 for reason in FAILURE_REASONS}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {reason: 0.0 for reason in FAILURE_REASONS}
    return {
        reason: float(parsed.get(reason, 0.0)) if isinstance(parsed, dict) else 0.0
        for reason in FAILURE_REASONS
    }


def _training_numeric_medians(records: list[dict[str, Any]]) -> dict[str, float]:
    numeric_columns = (
        "amount",
        "hour_of_day",
        "day_of_month",
        "customer_past_failure_count",
        "attempt_number",
    )
    return {
        column: float(np.median([float(record[column]) for record in records]))
        for column in numeric_columns
    }


def _read_records(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def _write_feature_importances(path: Path, feature_names: list[str], importances: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(zip(feature_names, importances, strict=True), key=lambda item: item[1], reverse=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=("feature", "importance"))
        writer.writeheader()
        writer.writerows(
            {"feature": feature_name, "importance": f"{importance:.8f}"}
            for feature_name, importance in ranked
        )


def _load_artifact(path: Path = MODEL_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError("Failure-reason classifier has not been trained")
    return joblib.load(path)


if __name__ == "__main__":
    metrics = train_classifier()
    print(f"frozen test macro-F1: {metrics['macro_f1']:.3f}")
    print(f"frozen test accuracy: {metrics['accuracy']:.3f}")
