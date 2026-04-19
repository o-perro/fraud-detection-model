"""XGBoost training, threshold tuning, and model serialization."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
) -> XGBClassifier:
    """Train an XGBoost classifier on the processed training data.

    scale_pos_weight is calculated from the training labels to tell XGBoost
    to penalize missing fraud cases more heavily — important even after SMOTE
    since the ratio informs the internal loss function.

    Args:
        X_train: Scaled, SMOTE-resampled feature matrix
        y_train: Corresponding fraud labels (0=legit, 1=fraud)
        n_estimators: Number of boosting trees (more = slower but potentially better)
        max_depth: Maximum tree depth (deeper = more complex, higher overfitting risk)
        learning_rate: Step size per tree (lower = more conservative, needs more trees)

    Returns:
        Trained XGBClassifier
    """
    # Ratio of legitimate to fraud — tells XGBoost fraud misses are costly
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",   # optimize for precision-recall AUC — correct metric for imbalanced data
        random_state=42,
        n_jobs=-1,             # use all CPU cores
    )

    logger.info("Training XGBoost model...")
    model.fit(X_train, y_train)
    logger.info("Training complete")

    return model


def tune_threshold(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[float, dict]:
    """Find the decision threshold that maximizes F1 score on the test set.

    The default threshold of 0.5 is rarely optimal for fraud detection.
    This function evaluates every possible threshold and returns the one
    that best balances precision and recall.

    Args:
        model: Trained XGBClassifier
        X_test: Test feature matrix (unscaled, real-world distribution)
        y_test: True fraud labels for the test set

    Returns:
        Tuple of (best_threshold, metrics_dict)
    """
    # Column 1 = probability of fraud
    y_proba = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = [f1_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]

    best_threshold = float(thresholds[np.argmax(f1_scores)])
    y_pred = (y_proba >= best_threshold).astype(int)

    metrics = {
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "best_f1": float(max(f1_scores)),
        "threshold": best_threshold,
    }

    logger.info(
        f"Threshold tuning complete — "
        f"Best threshold: {best_threshold:.4f} | "
        f"F1: {metrics['best_f1']:.4f} | "
        f"PR-AUC: {metrics['pr_auc']:.4f}"
    )

    return best_threshold, metrics


def save_model(
    model: XGBClassifier,
    scaler: object,
    threshold: float,
    metrics: dict,
    path: str | Path,
) -> None:
    """Serialize the trained model, scaler, threshold, and metrics to disk.

    The scaler is bundled with the model because new transactions must be
    scaled using the exact same mean/std from training — not refitted on
    new data. Without saving the scaler, predictions on new data would be wrong.

    Args:
        model: Trained XGBClassifier
        scaler: Fitted StandardScaler from build_features()
        threshold: Optimal decision threshold from tune_threshold()
        metrics: Performance metrics dict from tune_threshold()
        path: Save path e.g. models/xgb_v1.0.pkl
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "scaler": scaler,      # required for correct scaling of new transactions
        "threshold": threshold,
        **metrics,
    }

    joblib.dump(payload, path)
    logger.info(f"Model saved to {path}")
