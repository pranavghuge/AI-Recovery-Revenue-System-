"""Tests for the Recoverly failure-reason classifier."""

import csv
from pathlib import Path

from backend.classifier import (
    FAILURE_REASONS,
    FEATURE_IMPORTANCES_PATH,
    MODEL_PATH,
    evaluate_frozen_test_set,
    predict_failure_reason,
)
from backend.features import MODEL_FEATURE_COLUMNS, TEST_SET_PATH, assert_leakage_safe_feature_columns


def _first_test_feature_row() -> dict[str, str]:
    with TEST_SET_PATH.open(newline="", encoding="utf-8") as csv_file:
        row = next(csv.DictReader(csv_file))
    return {column: row[column] for column in MODEL_FEATURE_COLUMNS}


def test_prediction_matches_fixed_contract() -> None:
    prediction = predict_failure_reason(_first_test_feature_row())

    assert set(prediction) == {"reason", "confidence"}
    assert prediction["reason"] in FAILURE_REASONS
    assert 0.0 <= prediction["confidence"] <= 1.0


def test_frozen_test_macro_f1_meets_target() -> None:
    metrics = evaluate_frozen_test_set()

    assert metrics["macro_f1"] >= 0.70
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_classifier_uses_only_allowlisted_features() -> None:
    assert_leakage_safe_feature_columns(MODEL_FEATURE_COLUMNS)
    assert "failure_reason" not in MODEL_FEATURE_COLUMNS
    assert "success" not in MODEL_FEATURE_COLUMNS


def test_feature_importances_are_persisted() -> None:
    with FEATURE_IMPORTANCES_PATH.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert MODEL_PATH.exists()
    assert rows
    assert set(rows[0]) == {"feature", "importance"}
    assert sum(float(row["importance"]) for row in rows) > 0.99
