"""Unit tests for src/data/loader.py"""

import pytest
import pandas as pd
from pathlib import Path

from src.data.loader import load_raw_data, EXPECTED_COLUMNS


def test_load_raw_data_file_not_found(tmp_path):
    """Should raise FileNotFoundError when the CSV doesn't exist."""
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_raw_data(tmp_path / "nonexistent.csv")


def test_load_raw_data_missing_columns(tmp_path):
    """Should raise ValueError when expected columns are missing."""
    # Create a CSV with wrong columns
    bad_df = pd.DataFrame({"wrong_col": [1, 2, 3]})
    bad_path = tmp_path / "bad.csv"
    bad_df.to_csv(bad_path, index=False)

    with pytest.raises(ValueError, match="missing expected columns"):
        load_raw_data(bad_path)


def test_load_raw_data_missing_values(tmp_path):
    """Should raise ValueError when dataset contains NaN values."""
    # Build a valid-structure df but inject a NaN
    data = {col: [1.0, 2.0] for col in EXPECTED_COLUMNS}
    data["Amount"] = [None, 2.0]  # inject missing value
    df_with_nulls = pd.DataFrame(data)
    null_path = tmp_path / "nulls.csv"
    df_with_nulls.to_csv(null_path, index=False)

    with pytest.raises(ValueError, match="missing values"):
        load_raw_data(null_path)


def test_load_raw_data_returns_dataframe(tmp_path):
    """Should return a DataFrame with correct columns when file is valid."""
    # Build a minimal valid CSV
    data = {col: [1.0, 2.0] for col in EXPECTED_COLUMNS}
    data["Class"] = [0, 1]
    valid_df = pd.DataFrame(data)
    valid_path = tmp_path / "valid.csv"
    valid_df.to_csv(valid_path, index=False)

    result = load_raw_data(valid_path)

    assert isinstance(result, pd.DataFrame)
    assert set(EXPECTED_COLUMNS).issubset(set(result.columns))
