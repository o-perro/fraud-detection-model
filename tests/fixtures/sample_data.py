"""Shared test fixtures — small synthetic datasets that mimic the real data structure."""

import pandas as pd
import numpy as np


def make_sample_df(n_legit: int = 200, n_fraud: int = 20, random_state: int = 42) -> pd.DataFrame:
    """Create a small synthetic dataframe that mirrors creditcard.csv structure.

    Uses random data so tests run in milliseconds without needing the real dataset.
    n_fraud is set higher than real-world ratio so tests have enough fraud cases to work with.

    Args:
        n_legit: Number of legitimate transaction rows
        n_fraud: Number of fraud transaction rows
        random_state: Seed for reproducibility

    Returns:
        DataFrame with same columns as creditcard.csv
    """
    rng = np.random.default_rng(random_state)
    n_total = n_legit + n_fraud

    data = {
        "Time": rng.uniform(0, 172792, n_total),
        "Amount": rng.uniform(0, 5000, n_total),
        "Class": [0] * n_legit + [1] * n_fraud,
    }

    # Add V1-V28 PCA features — random normal values mimic the real distribution
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, n_total)

    return pd.DataFrame(data)
