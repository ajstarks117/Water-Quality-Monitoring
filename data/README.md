# Data Directory & Ingestion Guidelines

This directory houses dataset artifacts at different stages of the ML lifecycle. Raw and interim data files are excluded from Git history to maintain repository hygiene and avoid storing large datasets in source control.

---

## 📁 Subdirectory Layout

- `data/raw/`: Original, immutable raw CPCB / NWDP water quality monitoring CSV files for Maharashtra and Uttar Pradesh (Physical, Biological, Chemical).
- `data/interim/`: Intermediate processed datasets:
  - `data/interim/cpcb_joined.csv`: Exact-timestamp joined pre-label dataset.
  - `data/interim/cpcb_labeled.csv`: Ground-truth WQI scored and class-labeled dataset (3,287 records, 70 columns).
  - `data/interim/quarantine.csv`: Records quarantined due to insufficient active parameter coverage (<3 parameters).
- `data/processed/`: Canonical split datasets (`train.csv`, `test.csv`, `val.csv`) ready for baseline model training and evaluation.

---

## 📥 Dataset Sourcing & Local Pipeline Execution

### Sourcing Official CPCB Datasets
Official water quality monitoring records are obtained from the **National Water Data Portal (NWDP)** (`nwdp.nwic.gov.in`) / Central Pollution Control Board (CPCB) for **Maharashtra** and **Uttar Pradesh** (2021–2025).

Place the 6 official raw CSV files in `data/raw/`:
1. `*maharashtra*physical*.csv`
2. `*maharashtra*biological*.csv`
3. `*maharashtra*chemical*.csv`
4. `*uttar*pradesh*physical*.csv`
5. `*uttar*pradesh*biological*.csv`
6. `*uttar*pradesh*chemical*.csv`

### Reproducing the Interim Pipeline
Run the data joining and WQI labeling engines:
```bash
# Step 1: Join physical, biological, and chemical records on exact timestamps
python src/data/join_cpcb.py

# Step 2: Compute Weighted Arithmetic WQI and assign ground-truth class labels
python src/data/compute_wqi.py
```

---

## 🔒 Frozen Schema Contract (Milestone 1.4)
- **Target Column**: `wqi_class` (Multi-Class: Excellent, Good, Poor, Very Poor, Unsuitable)
- **Secondary Target**: `Potability` (Binary: 0 = Non-Potable, 1 = Potable)
- **Candidate Feature Columns**:
  - `Potential of Hydrogen (pH)`
  - `Dissolved oxygen (mg/L)`
  - `Biochemical Oxygen Demand (mg/L)`
  - `Fecal Coliform (MPN/100mL)`
- **Data Leakage Guard**: `wqi_score` is strictly excluded from all feature sets.
- Full documentation: [`Docs/data_dictionary.md`](file:///d:/Coding/College/Data%20science/CP/Docs/data_dictionary.md) and [`Docs/interface_contracts.md`](file:///d:/Coding/College/Data%20science/CP/Docs/interface_contracts.md).
