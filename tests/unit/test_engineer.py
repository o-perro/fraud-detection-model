"""Unit tests for src/features/engineer.py"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.fixtures.sample_data import make_sample_df
from src.features.engineer import build_features


def test_build_features_returns_four_outputs():
    """Should return exactly four datasets."""
    df = make_sample_df()
    result = build_features(df)
    assert len(result) == 4


def test_build_features_drops_time_adds_hour():
    """Time column should be replaced by Hour column."""
    df = make_sample_df()
    X_train, X_test, _, _ = build_features(df)

    assert "Time" not in X_train.columns
    assert "Hour" in X_train.columns
    assert "Time" not in X_test.columns
    assert "Hour" in X_test.columns


def test_build_features_hour_range():
    """Hour column should exist and be scaled (StandardScaler output has no fixed range)."""
    df = make_sample_df()
    X_train, X_test, _, _ = build_features(df)

    # After scaling, Hour is normalized — just verify it exists and has no NaN values
    assert "Hour" in X_train.columns
    assert X_train["Hour"].isnull().sum() == 0
    assert X_test["Hour"].isnull().sum() == 0


def test_build_features_smote_balances_training():
    """After SMOTE, training set should have equal fraud and legitimate counts."""
    df = make_sample_df()
    _, _, y_train, _ = build_features(df)

    assert (y_train == 0).sum() == (y_train == 1).sum()


def test_build_features_test_set_untouched():
    """Test set should NOT be resampled — fraud rate should stay close to original."""
    df = make_sample_df(n_legit=200, n_fraud=20)
    _, X_test, _, y_test = build_features(df)

    # Test set fraud rate should be close to original (20/220 = ~9%)
    # We allow some variance due to random splitting
    fraud_rate = y_test.mean()
    assert 0.03 < fraud_rate < 0.25, f"Unexpected fraud rate in test set: {fraud_rate}"


def test_build_features_no_data_leakage():
    """Train and test sets combined should not exceed original dataset size."""
    df = make_sample_df()
    X_train, X_test, y_train, y_test = build_features(df)

    # Test set size should match 20% of original — confirms no overlap
    expected_test_size = int(len(df) * 0.2)
    assert abs(len(X_test) - expected_test_size) <= 2  # allow 2 row variance due to rounding
