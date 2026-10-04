# AI-Driven Water Quality Monitoring & Potability Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning and Explainable AI (XAI) system for water potability assessment, real-time sensor anomaly detection, and predictive quality monitoring.

---

## 🚀 Quick Start & Environment Setup

This project uses a locked, reproducible Python environment. Python **3.10, 3.11, or 3.12** is required.

### 1. Clone the Repository
```bash
git clone https://github.com/ajstarks117/Water-Quality-Monitoring.git
cd Water-Quality-Monitoring
```

### 2. Environment Bootstrap

#### Windows (PowerShell):
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```

#### macOS / Linux / Git Bash:
```bash
chmod +x setup.sh
./setup.sh
```

#### Using Make:
```bash
make setup
```

### 3. Verify Installation
Run the import check to ensure all core packages are properly installed without version conflicts:
```bash
python -c "import pandas, sklearn, xgboost, shap, streamlit; print('All core packages successfully imported!')"
```

Or run the test suite:
```bash
pytest tests/ -v
```

---

## 📦 Pinned Core Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pandas` | `2.2.3` | Tabular data manipulation & cleaning |
| `numpy` | `1.26.4` | Numerical computation & array operations |
| `scikit-learn` | `1.5.2` | Preprocessing, baseline ML models, evaluation metrics |
| `xgboost` | `2.1.2` | Gradient boosting classifier & regressor |
| `shap` | `0.46.0` | Model explainability & SHAP value interpretation |
| `imbalanced-learn` | `0.12.4` | Class imbalance handling (SMOTE / ADASYN) |
| `joblib` | `1.4.2` | Pipeline serialization & model persistence |
| `streamlit` | `1.40.0` | Interactive web dashboard & inference UI |
| `matplotlib` | `3.9.2` | Static data visualization & reporting |
| `seaborn` | `0.13.2` | Statistical data plotting |
| `plotly` | `5.24.1` | Interactive charts & geospatial monitoring |
| `pyyaml` | `6.0.2` | Configuration file management |
| `pytest` | `8.3.3` | Unit, integration & regression test suite |

---

## 📁 Project Structure

```
├── data/
│   ├── raw/                # Immutable raw datasets (gitignored)
│   ├── interim/            # Intermediate transformed datasets (gitignored)
│   ├── processed/          # Final cleaned training & evaluation datasets
│   └── README.md           # Dataset sources and ingestion instructions
├── docs/                   # Architecture diagrams, specifications, presentations
├── models/                 # Model persistence directory (only frozen best_model.pkl tracked)
├── notebooks/              # Exploratory data analysis & prototyping notebooks
├── src/                    # Production source code modules
│   ├── __init__.py
│   ├── data/               # Data loading, cleaning, preprocessing pipelines
│   ├── features/           # Feature engineering & transformation
│   ├── models/             # Model training, evaluation, tuning & inference
│   ├── visualization/      # Plotting & dashboard rendering utilities
│   └── app.py              # Streamlit web application entrypoint
├── tests/                  # Pytest test suite
│   ├── __init__.py
│   └── test_environment.py # Environment & dependency validation tests
├── .gitignore              # Ignored files (venv, secrets, data cache, binaries)
├── LICENSE                 # MIT License
├── Makefile                # Automation commands (setup, test, clean)
├── README.md               # Project documentation stub
├── requirements.txt        # Pinned dependency requirements
├── setup.ps1               # Windows PowerShell bootstrap script
└── setup.sh                # macOS/Linux bootstrap script
```

---

## 🛡️ Collaboration & Branching Rules

- **Default Branch**: `main` (Protected)
- **Feature Branches**: `feat/<milestone-id>-<description>`, `fix/<issue-id>`, `chore/<task>`
- Pull requests require review before merging to `main`.
- Never commit secrets (`.env`), raw bulk data files (`data/raw/`), virtual environments (`venv/`), or untracked intermediate model checkpoints.
