"""Unit tests for src/models/train.py"""

import pytest
import joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.fixtures.sample_data import make_sample_df
from src.features.engineer import build_features
from src.models.train import train_model, tune_threshold, save_model


@pytest.fixture
def processed_data():
    """Shared fixture — runs feature engineering once for all model tests."""
    df = make_sample_df()
    return build_features(df)


def test_train_model_returns_fitted_model(processed_data):
    """train_model() should return a fitted XGBClassifier."""
    from xgboost import XGBClassifier
    X_train, _, y_train, _, _ = processed_data
    model = train_model(X_train, y_train)
    assert isinstance(model, XGBClassifier)
    assert hasattr(model, "feature_importances_")


def test_tune_threshold_returns_valid_threshold(processed_data):
    """Threshold should be a float between 0 and 1."""
    X_train, X_test, y_train, y_test, _ = processed_data
    model = train_model(X_train, y_train)
    threshold, metrics = tune_threshold(model, X_test, y_test)

    assert 0 < threshold < 1
    assert "pr_auc" in metrics
    assert "best_f1" in metrics
    assert "threshold" in metrics


def test_save_model_creates_file(processed_data, tmp_path):
    """save_model() should write a .pkl file to the specified path."""
    X_train, X_test, y_train, y_test, scaler = processed_data
    model = train_model(X_train, y_train)
    threshold, metrics = tune_threshold(model, X_test, y_test)

    save_path = tmp_path / "test_model.pkl"
    save_model(model, scaler, threshold, metrics, save_path)

    assert save_path.exists()


def test_save_model_payload_contains_expected_keys(processed_data, tmp_path):
    """Loaded model payload should contain model, scaler, threshold, and metrics."""
    X_train, X_test, y_train, y_test, scaler = processed_data
    model = train_model(X_train, y_train)
    threshold, metrics = tune_threshold(model, X_test, y_test)

    save_path = tmp_path / "test_model.pkl"
    save_model(model, scaler, threshold, metrics, save_path)

    payload = joblib.load(save_path)
    assert "model" in payload
    assert "scaler" in payload      # critical — needed for correct prediction on new data
    assert "threshold" in payload
    assert "pr_auc" in payload
    assert "best_f1" in payload
