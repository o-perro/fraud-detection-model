"""Feature engineering pipeline for the credit card fraud dataset."""

import logging

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def build_features(
    df: pd.DataFrame,
    test_size: float = 0.2,
    smote_random_state: int = 42,
    split_random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Run the full feature engineering pipeline on the raw dataset.

    Steps:
    1. Convert Time (raw seconds) to Hour of day (0-23)
    2. Stratified train/test split — preserves 0.17% fraud ratio
    3. Fit StandardScaler on train only, apply to both sets (prevents data leakage)
    4. Apply SMOTE to training set only — test set stays real-world distribution

    Args:
        df: Raw dataframe from load_raw_data()
        test_size: Fraction of data held out for testing (default 0.2 = 80/20 split)
        smote_random_state: Random seed for SMOTE reproducibility
        split_random_state: Random seed for train/test split reproducibility

    Returns:
        Tuple of (X_train_resampled, X_test, y_train_resampled, y_test, scaler)
        The scaler must be saved alongside the model for correct prediction on new data.
    """
    df = df.copy()  # never modify the original dataframe

    # Step 1: Convert Time to Hour of day
    # Raw seconds since first transaction is arbitrary — hour of day is meaningful
    df["Hour"] = (df["Time"] % 86400) / 3600
    df = df.drop(columns=["Time"])
    logger.info("Converted Time to Hour of day")

    # Step 2: Stratified train/test split
    # stratify=y ensures both sets maintain the 0.17% fraud ratio
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split_random_state, stratify=y
    )
    logger.info(
        f"Train/test split complete — "
        f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows"
    )

    # Step 3: Scale Amount and Hour
    # Fit on training data only — fitting on test data would be data leakage
    # Scaling is required before SMOTE so all features contribute equally to
    # distance calculations when generating synthetic fraud samples
    scaler = StandardScaler()
    X_train[["Amount", "Hour"]] = scaler.fit_transform(X_train[["Amount", "Hour"]])
    X_test[["Amount", "Hour"]] = scaler.transform(X_test[["Amount", "Hour"]])
    logger.info("StandardScaler applied to Amount and Hour")

    # Step 4: SMOTE on training data only
    # Generates synthetic fraud samples to balance the training set
    # Never apply to test set — it must reflect real-world distribution
    smote = SMOTE(random_state=smote_random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logger.info(
        f"SMOTE complete — "
        f"Legitimate: {(y_train_resampled==0).sum():,} | "
        f"Fraud: {(y_train_resampled==1).sum():,}"
    )

    # Return scaler alongside data — it must be saved with the model so new
    # transactions can be scaled using the exact same mean/std from training
    return X_train_resampled, X_test, y_train_resampled, y_test, scaler
