# Exploratory Data Analysis (EDA) Summary Report

**Project Title**: AI-Driven Water Quality Prediction & Explainable Decision Support System  
**Deliverable**: Deliverable 2 (EDA Executive Summary Report)  
**Milestone**: 3.2 (Parameter-vs-Class Analysis & EDA Report)  
**Author**: Data Lead (Person A) & ML Lead (Person B)  
**Dataset**: Central Pollution Control Board (CPCB) Surface Water Quality Monitoring Data (2021–2025)  
**Sample Size**: 3,287 labeled observations across Maharashtra and Uttar Pradesh  

---

## 1. Executive Summary & Context

This report provides a non-technical synthesis of the exploratory data analysis conducted on the frozen CPCB surface water quality dataset. The objective is to evaluate data quality, assess statistical feature distributions, identify which chemical and biological parameters effectively distinguish clean from polluted water, and outline key constraints for subsequent machine learning and Explainable AI (SHAP) modeling.

The primary target variable is **`wqi_class`** (a 5-tier classification standard derived from CPCB Water Quality Index formulas: *Excellent*, *Good*, *Poor*, *Very Poor*, and *Unsuitable*), alongside a secondary binary target **`Potability`** (*Potable* vs. *Non-Potable*).

---

## 2. Key Findings

### 2.1 Severe Target Class Imbalance
The monitored surface water bodies exhibit substantial pollution skew, resulting in extreme class imbalance across the 3,287 records:

| Water Quality Class | Category | Count | Percentage | Imbalance Characterization |
|:---|:---|---:|---:|:---|
| **Class A (Excellent)** | Potable | 7 | **0.21%** | **Extreme Minority** (Near-zero baseline) |
| **Class B (Good)** | Potable | 110 | **3.35%** | **Severe Minority** |
| **Class C (Poor)** | Non-Potable | 455 | **13.84%** | Moderate |
| **Class D (Very Poor)** | Non-Potable | 919 | **27.96%** | High |
| **Class E (Unsuitable)** | Non-Potable | 1,796 | **54.64%** | **Dominant Majority** |
| **Total** | | **3,287** | **100.00%** | |

*Summary*: Over **82.6%** of monitored water samples are classified as *Very Poor* or *Unsuitable* for consumption without extensive conventional treatment, reflecting severe urban and industrial runoff. A naive majority-class classifier would achieve **54.64%** raw accuracy simply by predicting *Class E* for every sample.

---

### 2.2 Parameter-vs-Class Separation (Bridge to SHAP Explainability)

Evaluating how individual physical, chemical, and biological measurements distribute across classes provides early, direct expectations for model feature importance and SHAP explanations (Phase 7):

```
       [ High Class Separation ]                         [ Subtle Separation ]
   BOD  ───────────►  Fecal Coliform  ───────────►  DO  ───────────►  pH
(Strongest Driver)  (Biological Pollution)     (Oxygen Health)    (Stable Buffer)
```

1. **Biochemical Oxygen Demand (BOD) — Strongest Separator**:
   - *Observation*: Median BOD increases monotonically from pristine baseline levels ($<2.0\text{ mg/L}$) in *Good* waters to severe concentrations ($>25\text{ mg/L}$, peaking at $127\text{ mg/L}$) in *Class E (Unsuitable)* waters.
   - *SHAP Expectation*: High BOD will emerge as a dominant driver pushing predictions toward *Very Poor* and *Unsuitable*.

2. **Fecal Coliform (FC) — Strong Biological Pollution Separator**:
   - *Observation*: Spans a vast dynamic range (median $140\text{ MPN/100mL}$ in clean waters up to extreme bacterial surges exceeding $10^6 - 10^7\text{ MPN/100mL}$ in *Class E*).
   - *SHAP Expectation*: Extreme microbial counts will heavily penalize potability and serve as a primary indicator of untreated domestic sewage contamination.

3. **Dissolved Oxygen (DO) — Clear Inverted Health Separator**:
   - *Observation*: Healthy waters (*Excellent* / *Good*) maintain robust dissolved oxygen ($>6.5\text{ mg/L}$). Severely polluted waters (*Class D* / *Class E*) regularly drop into hypoxic and anoxic zones ($<2.0\text{ mg/L}$).
   - *SHAP Expectation*: Low DO values will strongly contribute to negative water quality classifications.

4. **Potential of Hydrogen (pH) — Subtle Class Separation**:
   - *Observation*: Across classes *A* through *D*, pH remains tightly buffered around neutral-to-alkaline ranges (median $7.70$, IQR $7.30 - 8.10$). Only in *Class E* does the distribution widen to capture extreme acidic ($3.45$) and alkaline ($9.20$) runoff events.
   - *SHAP Expectation*: pH will exhibit lower global SHAP attribution compared to BOD and FC, but will produce acute local impact during extreme chemical spill anomalies.

---

### 2.3 Feature Correlation & Multicollinearity
- **Redundancy Threshold Check**: Pairwise linear (Pearson $r$) and monotonic (Spearman $\rho$) correlations were audited across all features. **No pairwise correlation exceeded the $|r| > 0.85$ redundancy threshold**.
- **Physical Relationships**: A moderate negative correlation exists between Dissolved Oxygen and BOD ($\rho = -0.36$), reflecting natural microbial oxygen consumption during organic breakdown. BOD and Fecal Coliform correlate positively ($r = +0.31$, $\rho = +0.51$) due to co-occurrence in municipal sewage outfalls.
- **Conclusion**: All four candidate features contribute distinct, non-redundant signals and are retained for model training.

---

### 2.4 Spatial and Temporal Observations
- **Spatial Coverage**: The dataset spans monitoring stations across Maharashtra ($N = 1,896$, $57.7\%$) and Uttar Pradesh ($N = 1,391$, $42.3\%$). Both states exhibit high representation of *Class D* and *Class E* waters along major river stretches (e.g., Godavari, Krishna, Ganga, Yamuna basins).
- **Temporal Consistency**: Continuous sampling records from 2021 through 2025 demonstrate stable class distribution proportions over time without anomalous missing periods or single-year reporting bias.

---

## 3. Modeling & Decision Support Directives (Phase 4 – 9)

Based on these empirical findings, the downstream tracks must adhere to the following directives:

1. **Mandatory Stratified Splitting (M4.2)**: Standard random train/test splits risk leaving *Class A* ($N=7$) completely absent from test sets. Splitting must use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
2. **Primary Performance Metrics (M5.1)**: Raw classification accuracy must be deprecated in favor of **Macro F1-Score**, **Balanced Accuracy**, and **Per-Class Precision-Recall curves**.
3. **Imbalance Mitigation Benchmarking (M6.1)**: Baseline unweighted models must be compared against class-weighted loss penalties (`class_weight='balanced'`) and synthetic oversampling (SMOTE).
4. **SHAP Interpretation Sanity Check (M7.1)**: Feature importance in tree models (XGBoost, Random Forest) should reflect the empirical separation hierarchy identified in this report (BOD & FC > DO > pH).

---

## 4. Visual Artifact References
All associated figures are archived in the repository for review:
- Univariate Histograms & Boxplots: [`reports/figures/eda/`](file:///d:/Coding/College/Data%20science/CP/reports/figures/eda/)
- Parameter-vs-Class Separation Figures: [`reports/figures/parameter_vs_class/`](file:///d:/Coding/College/Data%20science/CP/reports/figures/parameter_vs_class/)
- Spatial & Temporal Distributions: [`reports/figures/parameter_vs_class/spatial_state_wqi_distribution.png`](file:///d:/Coding/College/Data%20science/CP/reports/figures/parameter_vs_class/spatial_state_wqi_distribution.png)
- Ground-Truth Class Table: [`reports/class_distribution.csv`](file:///d:/Coding/College/Data%20science/CP/reports/class_distribution.csv)
