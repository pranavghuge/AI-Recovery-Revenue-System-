"""Tests for leakage-safe feature creation and frozen dataset splits."""

import csv
from pathlib import Path

import pytest

from backend.features import (
    FORBIDDEN_MODEL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    SOURCE_TRANSACTIONS_PATH,
    TEST_SET_PATH,
    assert_leakage_safe_feature_columns,
    impute_feature_records,
    verify_frozen_test_set,
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def test_model_feature_contract_is_leakage_safe() -> None:
    assert_leakage_safe_feature_columns(MODEL_FEATURE_COLUMNS)
    assert set(MODEL_FEATURE_COLUMNS).isdisjoint(FORBIDDEN_MODEL_FEATURE_COLUMNS)

    with pytest.raises(ValueError, match="Forbidden leakage"):
        assert_leakage_safe_feature_columns((*MODEL_FEATURE_COLUMNS, "failure_reason"))


def test_frozen_splits_are_customer_disjoint_and_near_target_ratios() -> None:
    train_rows = _read_rows(TEST_SET_PATH.with_name("train_set_v1.csv"))
    dev_rows = _read_rows(TEST_SET_PATH.with_name("dev_set_v1.csv"))
    test_rows = _read_rows(TEST_SET_PATH)
    total_rows = len(train_rows) + len(dev_rows) + len(test_rows)

    train_customers = {row["customer_id"] for row in train_rows}
    dev_customers = {row["customer_id"] for row in dev_rows}
    test_customers = {row["customer_id"] for row in test_rows}

    assert train_customers.isdisjoint(dev_customers)
    assert train_customers.isdisjoint(test_customers)
    assert dev_customers.isdisjoint(test_customers)
    assert len(_read_rows(SOURCE_TRANSACTIONS_PATH)) == total_rows
    assert abs(len(train_rows) / total_rows - 0.70) < 0.02
    assert abs(len(dev_rows) / total_rows - 0.15) < 0.02
    assert abs(len(test_rows) / total_rows - 0.15) < 0.02


def test_missing_features_are_imputed_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    record = {
        "amount": None,
        "hour_of_day": "",
        "day_of_month": None,
        "customer_past_failure_count": "",
        "attempt_number": None,
        "card_type": None,
        "is_recurring": "",
        "customer_past_failure_reasons_distribution": None,
    }
    medians = {
        "amount": 100.0,
        "hour_of_day": 12.0,
        "day_of_month": 15.0,
        "customer_past_failure_count": 2.0,
        "attempt_number": 1.0,
    }

    imputed = impute_feature_records([record], medians)[0]

    assert imputed["amount"] == 100.0
    assert imputed["card_type"] == "unknown"
    assert imputed["is_recurring"] == "unknown"
    assert "Imputing missing" in caplog.text


def test_frozen_test_set_checksum_rejects_changes(tmp_path: Path) -> None:
    frozen_copy = tmp_path / "test_set_v1.csv"
    frozen_copy.write_bytes(TEST_SET_PATH.read_bytes())
    verify_frozen_test_set(frozen_copy)

    frozen_copy.write_bytes(frozen_copy.read_bytes() + b"\nchanged")
    with pytest.raises(ValueError, match="checksum"):
        verify_frozen_test_set(frozen_copy)
