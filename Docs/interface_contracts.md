# Cross-Track Interface Contracts

**Document ID**: `CONTRACT-INTER-TRACK-2026-V2`  
**Milestone**: Milestone 1.4 (Data Dictionary & Schema Freeze)  
**Status**: **FROZEN**  
**Last Updated**: 2026-10-05  

This document specifies the two hard interface contracts binding **Track A (Data)**, **Track B (ML)**, **Track C (XAI/Decision Support)**, and **Track D (Dashboard/Integration)**.

---

## 1. Cleaned Data Contract

- **Owner**: Track A (Data Pipeline)
- **Freeze Milestone**: **M1.4** (Data Dictionary & Schema Freeze)
- **Consumer Tracks**: Track B (ML Models), Track C (Explainability), Track D (Dashboard)
- **Location**: `data/interim/cpcb_labeled.csv` (and downstream `data/processed/train.csv`, `data/processed/test.csv`)
- **Central Configuration**: `config/config.yaml` (`feature_columns`, `target_column`, `secondary_target_column`)

### Frozen Schema Specification (Candidate Feature Set & Targets)

| Column Name | Data Type | Units | Plausible Range | Role | Description |
|:---|:---|:---|:---|:---|:---|
| `Potential of Hydrogen (pH)` | `float64` | pH scale ($0-14$) | $[0.0, 14.0]$ | **Candidate Feature** | Acidity/alkalinity balance. CPCB limit: $6.5 - 8.5$. |
| `Dissolved oxygen (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 30.0]$ | **Candidate Feature** | Dissolved oxygen concentration. CPCB limit: $\ge 5.0\text{ mg/L}$. |
| `Biochemical Oxygen Demand (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 200.0]$ | **Candidate Feature** | 3-day BOD at 27ºC. CPCB limit: $\le 3.0\text{ mg/L}$. |
| `Fecal Coliform (MPN/100mL)` | `float64` | $\text{MPN/100mL}$ | $[0.0, 10^8]$ | **Candidate Feature** | Pathogenic bacterial indicator. CPCB limit: $\le 250\text{ MPN/100mL}$. |
| `wqi_class` | `object` / `string` | Categorical Label | 5 Classes | **Primary Target** | Multi-class label: `Excellent`, `Good`, `Poor`, `Very Poor`, `Unsuitable`. |
| `Potability` | `int64` | Binary Flag | `[0, 1]` | **Secondary Target** | Binary potability: `1` (Potable: Excellent/Good), `0` (Non-Potable). |

### Mandatory Data Leakage Prevention Guard

> [!CAUTION]
> **STRICT TARGET LEAKAGE GUARD**:
> `wqi_score` is the exact continuous mathematical number from which `wqi_class` and `Potability` were computed.
> `wqi_score` **MUST NEVER** appear in `feature_columns` in `config/config.yaml` or in any feature matrix passed to ML models (`X_train`, `X_test`).
> Including `wqi_score` would cause 100% artificial accuracy and total test leakage.

### Excluded Metadata & High-Missingness Columns
- **Administrative / Spatial Metadata (24 columns)**: `SlNo`, `Station`, `Agency`, `State LGD Code`, `State`, `District LGD Code`, `District`, `Tehsil`, `Block`, `Village`, `River`, `Basin`, `Tributary`, `Subtributary`, `SubSubtributary`, `Local River`, `Latitude`, `Longitude`, `Data Acquisition Time`, `has_physical_record`, `year`, `month`, `day`, `sampling_date`. (Retained for grouping, filtering, and dashboard map rendering).
- **High-Missingness Parameters (40 columns)**: All parameters with $>85\%$ missingness are excluded from the baseline candidate feature set.
- Complete 70-column enumeration is documented in [`Docs/data_dictionary.md`](file:///d:/Coding/College/Data%20science/CP/Docs/data_dictionary.md).

### Imputation & Preprocessing Guarantees
- Cleaned train/test split files: `data/processed/train.csv` and `data/processed/test.csv` (generated in Phase 2 / 4).
- Candidate feature column ordering is strictly preserved in `config/config.yaml` (`feature_columns`).
- **Modification Rule**: Any schema change after M1.4 requires an explicit exception in `#water-quality-core` and updates to `config/config.yaml`.

### 1.5 Final Extended Feature List (Milestone 4.1 Feature Engineering)

Per Milestone 4.1, the raw candidate feature set frozen in M1.4 is preserved intact, with domain-justified, leakage-checked engineered additions:

| Feature Name | Type | Input Provenance | Domain Justification | Prediction-Time Availability | Max $|r|$ vs Raw |
|:---|:---|:---|:---|:---|---:|
| `Potential of Hydrogen (pH)` | Raw (M1.4) | Raw User Sensor Input | Core chemical equilibrium metric | Always Available | — |
| `Dissolved oxygen (mg/L)` | Raw (M1.4) | Raw User Sensor Input | Primary aquatic oxygen health metric | Always Available | — |
| `Biochemical Oxygen Demand (mg/L)` | Raw (M1.4) | Raw User Sensor Input | Organic pollution load metric | Always Available | — |
| `Fecal Coliform (MPN/100mL)` | Raw (M1.4) | Raw User Sensor Input | Microbial pathogenic indicator | Always Available | — |
| `Log10 Fecal Coliform` | **Engineered (M4.1)** | `log10(max(FC, 1.0))` | Linearizes 7 orders of magnitude bacterial skewness ($+33.98$) | Computed from raw FC input | $0.484$ (Non-redundant) |

> **Leakage & Provenance Guarantee**: `Log10 Fecal Coliform` depends strictly on `Fecal Coliform (MPN/100mL)` entered by the dashboard user at inference time. It has zero dependence on `wqi_class`, `wqi_score`, `Potability`, or global/future aggregates.

### 1.6 M4.2 Contract Freeze: Stratified Train/Test Split & Preprocessing Pipeline

- **Owner**: Track B (ML Lead)
- **Freeze Milestone**: **M4.2** (Train/Test Split & Leakage Audit - CONTRACT FREEZE)
- **Status**: **FROZEN**
- **Cross-Track Binding**: Documented and frozen interface binding Track D (Dashboard & Inference API). Available for consumption when Milestone 9.2 is implemented in Phase 9.
- **Milestone 4.2 Status**: **FROZEN & FORMALLY CLOSED** (Technical verification complete; all 4 tests pass; 8-item leakage audit clean).

#### Frozen Feature Contract (Exact Names & Column Order)
The feature matrix $X$ passed into the preprocessing pipeline or model training consists strictly of the following 5 columns in exact order:
1. `Potential of Hydrogen (pH)` — `float64` (Raw M1.4)
2. `Dissolved oxygen (mg/L)` — `float64` (Raw M1.4)
3. `Biochemical Oxygen Demand (mg/L)` — `float64` (Raw M1.4)
4. `Fecal Coliform (MPN/100mL)` — `float64` (Raw M1.4)
5. `Log10 Fecal Coliform` — `float64` (Engineered M4.1)

#### Target Variable
- **Primary Target**: `wqi_class` (Categorical label with 5 classes: `Unsuitable`, `Very Poor`, `Poor`, `Good`, `Excellent`)
- **Strict Leakage Guard**: `wqi_score`, `wqi_class`, and `Potability` are **STRICTLY FORBIDDEN** from ever appearing in the feature matrix $X$.

#### Split Configuration & Stratification
- **Split Ratio**: 80% Train, 20% Test (Stratified on `wqi_class`).
- **Random Seed**: `42` (configured via `config/config.yaml`).
- **Cleaned Dataset Row Count**: 3,287 rows (after M2.1 deduplication and M2.2 missing-value handling).
- **Split Counts**: Train = 2,629 rows (80.0%), Test = 658 rows (20.0%).
- **Class Distribution**:
  - `Unsuitable`: Full 1,795 (54.61%) | Train 1,436 (54.62%) | Test 359 (54.56%)
  - `Very Poor`: Full 921 (28.02%) | Train 737 (28.03%) | Test 184 (27.96%)
  - `Poor`: Full 454 (13.81%) | Train 363 (13.81%) | Test 91 (13.83%)
  - `Good`: Full 110 (3.35%) | Train 88 (3.35%) | Test 22 (3.34%)
  - `Excellent`: Full 7 (0.21%) | Train 6 (0.23%) | Test 1 (0.15%)
- **Rare-Class Exception (`Excellent`, $n=7$)**:
  With only 7 samples in the entire dataset, a pure integer split yields 6 train (85.7%) and 1 test (14.3%). This integer rounding difference (+0.02 pp in train, -0.06 pp in test) is an unavoidable mathematical property of integer splitting on extremely rare classes. The classes are NOT merged; `Excellent` is preserved as a distinct class with representation in both train and test sets.
- **Physical Feature-Value Coincidences**:
  Cross-split duplicate checking identified 22 pairs of rows sharing identical 5-feature numeric readings between train and test. Verification confirmed these represent distinct physical monitoring records (different sampling stations or dates) rather than duplicated data pipeline rows. Because original row indices are strictly disjoint (0 cross-split index overlap), this represents natural physical sensor value repetition and passed the Section 19 Leakage Audit (S19-05).

#### Preprocessing Pipeline Guarantee
- The preprocessing pipeline (`StandardScaler` fitted on numeric columns) is fitted **STRICTLY on $X_{\text{train}}$**.
- It is NEVER fitted on $X_{\text{test}}$ or the full un-split dataset.
- $X_{\text{test}}$ is transformed solely using the statistics (means and variances) learned from $X_{\text{train}}$.

#### Saved Artifact Locations (Gitignored)
- Preprocessing Pipeline: `models/preprocessing_pipeline_v1.pkl`
- Feature Matrices: `data/processed/X_train_v1.csv`, `data/processed/X_test_v1.csv`
- Target Series: `data/processed/y_train_v1.csv`, `data/processed/y_test_v1.csv`
- Audit Reports: `data/processed/leakage_audit_v1.json`, `data/processed/class_distribution_v1.json`

#### Regeneration Command
```bash
python -m src.models.train
```
Or programmatically:
```python
from src.models.train import perform_split
split_result = perform_split()
```

#### Specification Section 19 Leakage Audit Checklist
All 8 verification items passed:
1. `S19-01`: Target not in features (`wqi_class`, `wqi_score`, `Potability` absent from $X$). [PASS]
2. `S19-02`: Row-independent transformation ($\log_{10}$ is strictly vectorized element-wise). [PASS]
3. `S19-03`: Zero test-set influence during fit (fitted solely on $X_{\text{train}}$). [PASS]
4. `S19-04`: Reproducible split (`random_seed=42`, stratified). [PASS]
5. `S19-05`: No cross-split source-row duplication (train and test index sets strictly disjoint). [PASS]
6. `S19-06`: Target distribution preserved within tolerance (with documented rare-class rounding). [PASS]
7. `S19-07`: Artifacts persistable without runtime computation. [PASS]
8. `S19-08`: Forward-compatible pipeline output shape matches expected model input. [PASS]

---

## 2. Model Artifact Contract

- **Owner**: Track B (ML Models)
- **Freeze Milestone**: **M6.3** (Best Model Freezing & Packaging)
- **Consumer Tracks**: Track C (Explainability), Track D (Dashboard & Inference API)
- **Location**: `models/best_model.pkl` (referenced via `config/config.yaml`)

### Artifact Interface Guarantees
The serialized object loaded via `joblib.load("models/best_model.pkl")` must be a scikit-learn compatible `Pipeline` or estimator exposing:

1. **Multi-Class & Binary Prediction**:
   ```python
   y_pred = model.predict(X)
   # Returns: np.ndarray of shape (n_samples,) with class labels or binary integers
   ```
2. **Probability Estimation**:
   ```python
   y_prob = model.predict_proba(X)
   # Returns: np.ndarray of shape (n_samples, n_classes) with normalized probabilities
   ```
3. **Input Format**:
   - `X`: `pandas.DataFrame` or `numpy.ndarray` containing candidate feature columns matching the exact names and dtypes defined in the Cleaned Data Contract.
   - Built-in preprocessing: The pipeline must encapsulate any required scaling (e.g. `StandardScaler`) and imputation internally so raw inputs can be passed directly to `.predict()`.

---

## 3. Parallel Development & Mocking Strategy

To prevent Tracks C and D from being blocked:
- **Interim Dataset Ready**: Tracks B, C, and D can directly consume `data/interim/cpcb_labeled.csv`.
- **Mock Model**: Tracks C and D can use a baseline model (e.g. `RandomForestClassifier` or `LogisticRegression`) serialized at `models/mock_model.pkl` until M6.3 is completed.
