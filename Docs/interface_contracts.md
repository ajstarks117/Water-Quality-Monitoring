# Cross-Track Interface Contracts

This document specifies the two hard interface contracts binding Track A (Data), Track B (ML), Track C (XAI/Decision Support), and Track D (Dashboard/Integration).

---

## 1. Cleaned Data Contract

- **Owner**: Track A (Data Pipeline)
- **Freeze Milestone**: **M1.2** (Cleaned Data Pipeline & Schema Lock)
- **Consumer Tracks**: Track B (ML Models), Track C (Explainability), Track D (Dashboard)
- **Location**: `data/processed/` (referenced via `config/config.yaml`)

### Schema Specification
| Column Name | Data Type | Units / Range | Role |
|---|---|---|---|
| `ph` | `float64` | 0.0 - 14.0 | Feature |
| `Hardness` | `float64` | mg/L (approx. 47 - 323) | Feature |
| `Solids` | `float64` | ppm (approx. 320 - 61227) | Feature |
| `Chloramines` | `float64` | ppm (approx. 0.35 - 13.1) | Feature |
| `Sulfate` | `float64` | mg/L (approx. 129 - 481) | Feature |
| `Conductivity` | `float64` | μS/cm (approx. 181 - 753) | Feature |
| `Organic_carbon` | `float64` | ppm (approx. 2.2 - 28.3) | Feature |
| `Trihalomethanes` | `float64` | μg/L (approx. 0.7 - 124.0) | Feature |
| `Turbidity` | `float64` | NTU (approx. 1.4 - 6.7) | Feature |
| `Potability` | `int64` | `0` (Not Potable), `1` (Potable) | **Target** |

### Imputation & Preprocessing Guarantees
- No `NaN`, `null`, or `inf` values in `data/processed/`.
- Cleaned train/test split files: `data/processed/train.csv` and `data/processed/test.csv`.
- Feature column ordering is preserved in `config/config.yaml` (`feature_columns`).
- **Modification Rule**: Any schema change after M1.2 requires written consent in the team channel and updates to `config/config.yaml`.

---

## 2. Model Artifact Contract

- **Owner**: Track B (ML Models)
- **Freeze Milestone**: **M6.3** (Best Model Freezing & Packaging)
- **Consumer Tracks**: Track C (Explainability), Track D (Dashboard & Inference API)
- **Location**: `models/best_model.pkl` (referenced via `config/config.yaml`)

### Artifact Interface Guarantees
The serialized object loaded via `joblib.load("models/best_model.pkl")` must be a scikit-learn compatible `Pipeline` or estimator exposing:

1. **Binary Prediction**:
   ```python
   y_pred = model.predict(X)
   # Returns: np.ndarray of shape (n_samples,) with integer values in {0, 1}
   ```
2. **Probability Estimation**:
   ```python
   y_prob = model.predict_proba(X)
   # Returns: np.ndarray of shape (n_samples, 2) where y_prob[:, 1] is P(Potable)
   ```
3. **Input Format**:
   - `X`: `pandas.DataFrame` or `numpy.ndarray` containing all 9 feature columns matching the exact names and dtypes defined in the Cleaned Data Contract.
   - Built-in preprocessing: The pipeline must encapsulate any required scaling (e.g. `StandardScaler`) internally so raw user inputs can be passed directly to `.predict()`.

---

## 3. Parallel Development & Mocking Strategy

To prevent Tracks C and D from being blocked prior to M1.2 and M6.3:
- **Mock Data**: Tracks B, C, and D can generate synthetic samples matching the schema above using `src/data/mock_data.py`.
- **Mock Model**: Tracks C and D can use a dummy baseline model (e.g. `DummyClassifier(strategy="stratified")` or `LogisticRegression()`) serialized at `models/mock_model.pkl` until M6.3 is completed.
