"""One-off script to train and save the fraud detection model using the src/ pipeline.

Run from the project root:
    python scripts/train_model.py

Saves model artifact to models/xgb_v1.1.pkl — includes scaler, threshold, and metrics.
v1.1 corrects v1.0 which was missing the scaler in the saved payload.
"""

import logging
from pathlib import Path

from src.data.loader import load_raw_data
from src.features.engineer import build_features
from src.models.train import train_model, tune_threshold, save_model

# Basic logging setup so we see progress in the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the full training pipeline and save the model."""

    # Paths relative to project root
    data_path = Path("data/raw/creditcard.csv")
    model_path = Path("models/xgb_v1.1.pkl")

    logger.info("Starting training pipeline")

    # Step 1: Load and validate raw data
    df = load_raw_data(data_path)

    # Step 2: Feature engineering — returns scaler alongside processed data
    X_train, X_test, y_train, y_test, scaler = build_features(df)

    # Step 3: Train XGBoost
    model = train_model(X_train, y_train)

    # Step 4: Find optimal threshold on test set
    threshold, metrics = tune_threshold(model, X_test, y_test)

    # Step 5: Save model + scaler + threshold + metrics together
    save_model(model, scaler, threshold, metrics, model_path)

    logger.info(
        f"Training complete — "
        f"Model saved to {model_path} | "
        f"F1: {metrics['best_f1']:.4f} | "
        f"PR-AUC: {metrics['pr_auc']:.4f} | "
        f"Threshold: {threshold:.4f}"
    )


if __name__ == "__main__":
    main()
