# Fraud Detection Model

Binary classification model to detect fraudulent credit card transactions using the Kaggle ULB dataset. Built with XGBoost and SMOTE resampling to handle severe class imbalance (~0.17% fraud rate).

---

## Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions, 492 fraud cases
- **Features:** `Time`, `Amount`, `Class` (0=legit, 1=fraud), V1–V28 (PCA-transformed)
- **License:** Open Database License (ODbL)

> Download `creditcard.csv` from Kaggle and place it in `data/raw/`. It is excluded from git.

---

## Project Structure

```
fraud-detection-model/
├── src/
│   ├── data/         # Data loading and validation
│   ├── features/     # Feature engineering
│   ├── models/       # Training, evaluation, serialization
│   └── utils/        # Shared helpers
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory data analysis
│   ├── 02_features.ipynb   # Feature engineering experiments
│   └── 03_modeling.ipynb   # Model prototyping
├── data/
│   ├── raw/          # Original source data — never modified
│   ├── processed/    # Train/test splits, resampled data
│   └── outputs/      # Predictions, evaluation reports
├── models/           # Serialized model artifacts (e.g. xgb_v1.0.pkl)
├── tests/
│   └── unit/
├── .env.example
└── pyproject.toml
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/o-perro/fraud-detection-model.git
cd fraud-detection-model
```

### 2. Install dependencies
```bash
# Install uv if you don't have it
pip install uv

# Create virtual environment and install all dependencies
uv venv
source .venv/bin/activate       # Mac/Linux
uv pip install -e ".[dev]"
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and set RAW_DATA_PATH if your data lives outside data/raw/
```

### 4. Add the dataset
Download `creditcard.csv` from Kaggle and place it at:
```
data/raw/creditcard.csv
```

---

## Resuming Development

Every time you return to this project:

```bash
# 1. Navigate to the project
cd /Users/gabrielolson/projects/fraud-detection-model

# 2. Activate the virtual environment — you'll see (.venv) in your prompt when active
source .venv/bin/activate

# 3. Launch Jupyter to work through notebooks
jupyter lab
```

> The virtual environment only needs to be created once (`uv venv` + `uv pip install`).
> After that, just activate it each session with `source .venv/bin/activate`.

---

## Running the Project

```bash
# Start Jupyter to work through notebooks
jupyter lab

# Run tests
pytest tests/unit/

# Lint and format
ruff check src/
ruff format src/
```

---

## Results

*Model: XGBoost v1.0 | Optimal threshold: 0.9843*

| Metric | Score |
|--------|-------|
| F1 Score | 0.8571 |
| Recall (fraud) | 79.6% (78/98 fraud cases caught) |
| False Positives | 6 legitimate transactions incorrectly flagged |
| Optimal Threshold | 0.9843 |

---

## Key Design Decisions

- **XGBoost** chosen for its strong performance on tabular data and interpretability via feature importance
- **SMOTE** applied to training data only to address class imbalance — never applied to test data
- **Precision-Recall AUC** used as primary metric — accuracy is misleading at 0.17% fraud rate
- **Threshold tuning** applied post-training to balance cost of missed fraud vs false alarms
