"""Load a saved model and score new credit card transactions."""

import logging
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def load_model(path: str | Path) -> dict:
    """Load a saved model payload from disk.

    Args:
        path: Path to the .pkl file e.g. models/xgb_v1.0.pkl

    Returns:
        Dict containing model, scaler, threshold, and metrics
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")

    payload = joblib.load(path)

    # Validate the payload has everything we need for prediction
    required_keys = {"model", "scaler", "threshold"}
    missing = required_keys - set(payload.keys())
    if missing:
        raise ValueError(
            f"Model payload is missing required keys: {missing}. "
            "This model may have been saved without the scaler — retrain using the updated pipeline."
        )

    logger.info(f"Loaded model from {path} | Threshold: {payload['threshold']:.4f}")
    return payload


def preprocess_transactions(
    df: pd.DataFrame,
    scaler: object,
) -> pd.DataFrame:
    """Apply the same preprocessing used during training to new transactions.

    New data must go through identical transformations as training data —
    same Time→Hour conversion, same scaler with training mean/std.
    Any deviation causes training-serving skew and incorrect predictions.

    Args:
        df: Raw transaction dataframe with same columns as creditcard.csv
        scaler: Fitted StandardScaler saved during training

    Returns:
        Preprocessed DataFrame ready for model.predict_proba()
    """
    df = df.copy()

    # Convert Time to Hour using same formula as training
    df["Hour"] = (df["Time"] % 86400) / 3600
    df = df.drop(columns=["Time"], errors="ignore")

    # Drop Class column if present — won't exist on real unseen transactions
    df = df.drop(columns=["Class"], errors="ignore")

    # Apply training scaler — uses training mean/std, not refitted on new data
    df[["Amount", "Hour"]] = scaler.transform(df[["Amount", "Hour"]])

    return df


def predict(
    df: pd.DataFrame,
    payload: dict,
) -> pd.DataFrame:
    """Score new transactions and return fraud predictions with probability scores.

    Args:
        df: Raw transaction dataframe (same structure as creditcard.csv)
        payload: Loaded model payload from load_model()

    Returns:
        DataFrame with original columns plus:
        - fraud_probability: model's confidence this transaction is fraud (0-1)
        - is_fraud: 1 if fraud_probability >= threshold, 0 otherwise
    """
    model = payload["model"]
    scaler = payload["scaler"]
    threshold = payload["threshold"]

    # Preprocess using training scaler — critical for correct predictions
    X = preprocess_transactions(df, scaler)

    # Get fraud probability for each transaction
    fraud_proba = model.predict_proba(X)[:, 1]

    results = df.copy()
    results["fraud_probability"] = fraud_proba
    results["is_fraud"] = (fraud_proba >= threshold).astype(int)

    n_flagged = results["is_fraud"].sum()
    logger.info(
        f"Scored {len(results):,} transactions | "
        f"Flagged as fraud: {n_flagged} ({n_flagged/len(results)*100:.2f}%)"
    )

    return results
