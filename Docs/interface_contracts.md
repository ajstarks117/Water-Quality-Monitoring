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
- Cleaned train/test split files: `data/processed/train.csv` and `data/processed/test.csv` (generated in Phase 2).
- Candidate feature column ordering is strictly preserved in `config/config.yaml` (`feature_columns`).
- **Modification Rule**: Any schema change after M1.4 requires an explicit exception in `#water-quality-core` and updates to `config/config.yaml`.

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
