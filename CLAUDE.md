# Fraud Detection Model — Project Context

## What This Project Does
Binary classification model to detect fraudulent credit card transactions. The goal is to flag fraud with high recall (catch as many fraudulent transactions as possible) while keeping false positives low enough to be operationally useful.

## Dataset
- **Source:** Kaggle ULB Credit Card Fraud Detection dataset
- **File:** `data/raw/creditcard.csv` — never modify this file
- **Size:** ~284,807 transactions, 492 fraud cases (~0.17% positive class)
- **Features:** `Time`, `Amount`, `Class` (0=legit, 1=fraud), and V1–V28 (PCA-transformed — do not attempt to reverse-engineer or rename these)

## Modeling Approach
- **Primary model:** XGBoost (handles imbalance well, interpretable via feature importance)
- **Class imbalance strategy:** SMOTE oversampling via `imbalanced-learn` on training data only — never apply SMOTE to the test set
- **Evaluation metrics:** Precision-Recall AUC and F1 score — do NOT use accuracy, it is misleading on this dataset
- **Threshold tuning:** the default 0.5 threshold is rarely optimal here; tune based on the business cost of false negatives (missed fraud) vs false positives (blocked legitimate transactions)

## Project Structure Reminders
- `src/data/` — data loading and validation logic
- `src/features/` — feature engineering (scaling `Amount`, dropping or encoding `Time`)
- `src/models/` — training, evaluation, and saving model artifacts
- `src/utils/` — shared helpers (logging, config loading)
- `notebooks/` — exploration only; graduate any reusable logic to `src/`
- `data/processed/` — train/test splits and resampled data go here
- `models/` — serialized artifacts, versioned (e.g. `xgb_v1.0.pkl`)

## Key Decisions Made So Far
- Using `pyproject.toml` + `uv` for dependency management
- `imbalanced-learn` included for SMOTE
- `.gitignore` excludes `data/` and `models/` artifacts — these are not committed to git
