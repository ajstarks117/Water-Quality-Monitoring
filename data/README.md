# Data Directory & Ingestion Guidelines

This directory houses dataset artifacts at different stages of the ML lifecycle. Raw data files are excluded from Git history to maintain repository hygiene and avoid storing large immutable binaries.

---

## 📁 Subdirectory Layout

- `data/raw/`: Original, immutable raw water quality datasets (e.g. `water_potability.csv`). Files in this directory should never be edited directly.
- `data/interim/`: Intermediate transformed datasets undergoing cleaning, missing value imputation, outlier handling, or exploratory transformations.
- `data/processed/`: Canonical, split datasets (`train.csv`, `test.csv`, `val.csv`) ready for baseline model training and evaluation.

---

## 📥 How to Obtain & Place Datasets Locally

### Primary Dataset: Water Quality / Potability Dataset
1. **Source**: Download the Water Potability dataset from Kaggle or the course repository:
   - Kaggle: [Water Quality and Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
2. **Placement**:
   - Save the raw CSV file as:
     ```
     data/raw/water_potability.csv
     ```
3. **Expected Schema**:
   - Columns: `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`, `Potability`
   - Target Column: `Potability` (Binary: `0` = Not Potable, `1` = Potable)

---

## ⚠️ Data Policy
- **Do not commit raw CSVs, Excel files, or interim data** to Git (`data/raw/*` and `data/interim/*` are ignored in `.gitignore`).
- Only `.gitkeep` markers and this `README.md` are version-controlled.
- All pipeline outputs must be generated via reproducible scripts in `src/data/` and configured via `config/config.yaml`.
