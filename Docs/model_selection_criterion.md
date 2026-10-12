# Model Selection Criterion & Evaluation Protocol

**Document ID**: `CRITERION-MODEL-SELECT-2026-V1`  
**Milestone**: Milestone 5.2 (Baseline Evaluation & Comparison Table)  
**Status**: **PRE-DECLARED & FROZEN PRIOR TO EVALUATION**  
**Specification Reference**: Sections 22, 23, 26, 27  
**Owner**: ML Lead (Person B)  

---

## 1. Purpose & Anti-Cherry-Picking Mandate

Per **Specification Section 23**, the model-selection criterion must be declared in writing **prior to inspecting baseline or tuned model performance**. Declaring the primary evaluation metric after reviewing results introduces severe confirmation bias and risk of cherry-picking metrics to favour a specific model.

This document establishes the hard mathematical criteria that will govern model comparison across Phase 5 (Baseline ML) and Phase 6 (Model Improvement & Final Selection).

---

## 2. Metric Hierarchy

```
Primary Selection Metric:
└── Macro-Averaged F1-Score (Macro F1)
    │
    ├── Secondary Metric 1: Recall on Severe Pollution Classes ('Unsuitable' & 'Very Poor')
    ├── Secondary Metric 2: Weighted F1 & Balanced Accuracy
    └── Operational Tie-Breakers:
        ├── Inference Latency & Model Footprint
        └── Explainability (SHAP tree/linear explainer compatibility)
```

---

## 3. Primary Metric: Macro F1

### Definition
$$\text{Macro } F_1 = \frac{1}{K} \sum_{k=1}^{K} F_1^{(k)}$$
where $K = 5$ classes (`Excellent`, `Good`, `Poor`, `Very Poor`, `Unsuitable`), and $F_1^{(k)}$ is the unweighted harmonic mean of precision and recall for class $k$:
$$F_1^{(k)} = 2 \cdot \frac{\text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$

### Domain & Statistical Justification
1. **Severe Class Imbalance**: The CPCB labeled dataset exhibits extreme class imbalance:
   - `Unsuitable`: $54.6\%$ ($1,795$ rows)
   - `Very Poor`: $28.0\%$ ($921$ rows)
   - `Poor`: $13.8\%$ ($454$ rows)
   - `Good`: $3.4\%$ ($110$ rows)
   - `Excellent`: $0.2\%$ ($7$ rows)
2. **Failure Mode of Standard Accuracy**: A trivial majority-class classifier predicting `Unsuitable` for every sample achieves **$54.6\%$ accuracy**, despite completely failing to identify clean potable water or moderate contamination.
3. **Equal Accountability Across Classes**: Macro F1 computes metrics per class independently and averages them with equal weight ($0.20$ each), penalizing models that sacrifice rare classes (`Excellent`, `Good`) to maximize majority accuracy.

---

## 4. Secondary Metric: Critical Pollution Recall

### Definition
$$\text{Recall}_{\text{Unsuitable}} = \frac{TP_{\text{Unsuitable}}}{TP_{\text{Unsuitable}} + FN_{\text{Unsuitable}}}$$
$$\text{Recall}_{\text{Very Poor}} = \frac{TP_{\text{Very Poor}}}{TP_{\text{Very Poor}} + FN_{\text{Very Poor}}}$$

### Domain Justification
In water quality management and public health monitoring, **Type II errors (False Negatives)** on hazardous water conditions carry catastrophic consequences (consumption of pathogenic bacterial loads or hypoxic water). A model that misclassifies `Unsuitable` water as `Good` is dangerous. Therefore, when Macro F1 scores are comparable (within $0.02$), higher recall on contaminated water will break the tie.

---

## 5. Prohibited Selection Rules

> [!CAUTION]
> **DISALLOWED SELECTION CRITERION**:
> - **Accuracy alone MUST NEVER be used as the selection criterion.**
> - A model with higher overall accuracy but inferior Macro F1 will be rejected in favor of the model with superior Macro F1.
> - High accuracy accompanied by low Macro F1 is a documented indicator of minority-class collapse under imbalance.

---

## 6. Full Evaluation Metric Set (Reporting Contract)

Per **Specification Section 26**, every candidate model must report the complete metric matrix across the untouched test split ($X_{\text{test}}$, $n=658$):

| Metric | Scikit-Learn Formulation | Purpose |
|:---|:---|:---|
| `Accuracy` | `accuracy_score(y_true, y_pred)` | Overall correct prediction rate |
| `Precision (Macro)` | `precision_score(y_true, y_pred, average='macro', zero_division=0)` | Unweighted mean precision across classes |
| `Recall (Macro)` | `recall_score(y_true, y_pred, average='macro', zero_division=0)` | Unweighted mean recall across classes |
| `Macro F1` | `f1_score(y_true, y_pred, average='macro', zero_division=0)` | **PRIMARY SELECTION CRITERION** |
| `Weighted F1` | `f1_score(y_true, y_pred, average='weighted', zero_division=0)` | Sample-weighted F1 balance |
| `Training Time (s)` | CPU wall-clock elapsed time in seconds | Computational feasibility |
| `Confusion Matrix` | `confusion_matrix(y_true, y_pred, labels=...)` | Multi-class confusion error structure |

---

## 7. Operational Tie-Breaking Order

If two candidate models achieve Macro F1 within $\pm 0.015$:
1. **Critical Pollution Recall**: Higher average recall across `Unsuitable` and `Very Poor`.
2. **Explainability**: Support for exact/fast TreeSHAP (e.g. Random Forest, XGBoost) over kernel approximations.
3. **Inference Latency**: Lower per-sample prediction time on standard CPU hardware.

---

## 8. Milestone 6.1: Class Imbalance Handling — Decision Record

**Status**: DECIDED — `class_weight='balanced'` retained; no external resampling applied.  
**Experiment Date**: Phase 6 (M6.1)  
**Config Setting**: `resampling_method: "none"` in `config/config.yaml`  

### 8.1 Imbalance Severity Assessment

The CPCB labeled dataset exhibits **extreme class imbalance** (re-confirmed from M3.1/M5.2):

| Class | Train Count | Train % | Imbalance Ratio vs Majority |
|:---|---:|---:|---:|
| Unsuitable | 1,436 | 54.6% | 1.0x (majority) |
| Very Poor | 735 | 28.0% | 1.95x |
| Poor | 364 | 13.8% | 3.95x |
| Good | 88 | 3.4% | 16.3x |
| Excellent | 6 | 0.2% | 239.3x |

The `Excellent` class (6 training samples) is critically underrepresented.

### 8.2 Experimental Comparison

Three imbalance-handling strategies were compared across all 4 candidate models, evaluated on the **original untouched test set** ($n = 658$):

#### Macro F1 Comparison (Primary Selection Metric)

| Model | Baseline (`class_weight='balanced'`) | SMOTE | RandomOverSampler |
|:---|---:|---:|---:|
| **Decision Tree** | **0.9067** | 0.9007 | 0.9015 |
| Random Forest | 0.8417 | 0.7121 | 0.8324 |
| XGBoost | 0.7237 | 0.7017 | 0.8544 |
| Logistic Regression | 0.5648 | 0.5763 | 0.5729 |

#### Per-Class Recall for Minority Classes (Baseline `class_weight='balanced'`)

| Model | Good (3.4%) | Excellent (0.2%) |
|:---|---:|---:|
| Decision Tree | 0.727 | 1.000 |
| Random Forest | 0.591 | 1.000 |
| XGBoost | 0.682 | 0.000 |
| Logistic Regression | 0.636 | 1.000 |

### 8.3 Decision & Justification

**Decision**: Retain `class_weight='balanced'` without external resampling (SMOTE/ROS).

**Rationale**:
1. **Best overall Macro F1**: The baseline Decision Tree with `class_weight='balanced'` achieved the highest Macro F1 (0.9067) across all 12 model×strategy combinations.
2. **SMOTE degraded the best model**: SMOTE reduced Decision Tree Macro F1 from 0.9067 → 0.9007 and severely harmed Random Forest (0.8417 → 0.7121).
3. **ROS marginal**: RandomOverSampler produced slight degradation for Decision Tree (0.9067 → 0.9015) while improving XGBoost (0.7237 → 0.8544), but the overall best model is still unimproved.
4. **Excellent class coverage**: The baseline Decision Tree already achieves perfect recall (1.000) on the `Excellent` class despite having only 6 training samples — an already strong result that resampling cannot improve.
5. **Simplicity**: `class_weight='balanced'` is parameter-free, adds no synthetic data artifacts, and has zero risk of introducing train/test contamination.

### 8.4 Leakage Audit Confirmation (TC-6.1-01)

- `resample_training_data()` accepts only `X_train` and `y_train` as inputs.
- The function is called **inside** `train_baseline_models()` **after** the M4.2 split.
- The test set (`X_test`, `y_test`) is never passed to or modified by the resampling function.
- Full experiment results saved to: `reports/m61_imbalance_experiment.json`
