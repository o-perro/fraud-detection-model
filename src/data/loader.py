"""Functions for loading and validating the raw credit card fraud dataset."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columns we expect in the raw dataset — validates nothing got dropped or renamed
EXPECTED_COLUMNS = (
    ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
)


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the raw creditcard.csv dataset.

    Checks for expected columns and missing values before returning.
    Raises ValueError if validation fails so problems surface early
    rather than silently propagating bad data downstream.

    Args:
        path: Path to creditcard.csv — typically data/raw/creditcard.csv

    Returns:
        Validated DataFrame with 31 columns (Time, V1-V28, Amount, Class)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download creditcard.csv from Kaggle and place it in data/raw/."
        )

    logger.info(f"Loading dataset from {path}")
    df = pd.read_csv(path)

    # Validate expected columns are present
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    # Validate no missing values — SMOTE and StandardScaler will error on NaN
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        raise ValueError(
            f"Dataset contains missing values — handle before preprocessing:\n"
            f"{cols_with_nulls}"
        )

    logger.info(
        f"Loaded {len(df):,} rows | "
        f"Fraud rate: {df['Class'].mean()*100:.3f}%"
    )

    return df
