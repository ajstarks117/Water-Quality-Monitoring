# Exploratory Data Analysis (EDA) Summary Report

**Project Title**: AI-Driven Water Quality Prediction & Explainable Decision Support System  
**Deliverable**: Deliverable 2 (EDA Executive Summary Report)  
**Milestone**: 3.2 (Parameter-vs-Class Analysis & EDA Report)  
**Author**: Data Lead (Person A) & ML Lead (Person B)  
**Dataset**: Central Pollution Control Board (CPCB) Surface Water Quality Monitoring Data (2021–2025)  
**Sample Size**: 3,287 labeled observations across Maharashtra and Uttar Pradesh  

---

## 1. Executive Summary & Context

This report provides a non-technical synthesis of the exploratory data analysis conducted on the CPCB surface water quality dataset frozen under **Milestone 1.4 (Data Dictionary & Schema Freeze)**. The objective is to evaluate data quality, assess statistical feature distributions, observe how candidate parameters vary across water-quality classes as descriptive baseline evidence, and outline key constraints for subsequent machine learning and Explainable AI (SHAP) modeling.

The primary target variable is **`wqi_class`** (a 5-tier classification standard derived from CPCB Water Quality Index formulas: *Excellent*, *Good*, *Poor*, *Very Poor*, and *Unsuitable*), alongside a secondary binary target **`Potability`** (*Potable* vs. *Non-Potable*). The candidate feature set is strictly bound to the 4 frozen features established in M1.4: `Potential of Hydrogen (pH)`, `Dissolved oxygen (mg/L)`, `Biochemical Oxygen Demand (mg/L)`, and `Fecal Coliform (MPN/100mL)`.

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

### 2.2 Descriptive Parameter-vs-Class Separation

Evaluating how individual physical, chemical, and biological measurements distribute across classes provides **descriptive exploratory evidence** to contextualize how raw readings align with ground-truth classes. 

> [!NOTE]
> **Methodological Note**: Visual class separation serves as descriptive EDA evidence only. It does **not** determine final feature importance or predictive capacity, nor is any feature selected or eliminated based on these plots. All 4 frozen features from M1.4 are retained for model training and subsequent SHAP evaluation (Phase 7).

```
          [ Clear Visual Spread ]                            [ Central Clustering ]
   BOD  ───────────►  Fecal Coliform  ───────────►  DO  ───────────►  pH
(Wide Separation)   (Microbial Spread)        (Oxygen Drop)       (Buffered Range)
```

1. **Biochemical Oxygen Demand (BOD)**:
   - *Descriptive Observation*: Median BOD increases across classes from low levels ($<2.0\text{ mg/L}$) in *Good* waters to elevated concentrations ($>25\text{ mg/L}$, reaching $127\text{ mg/L}$) in *Class E (Unsuitable)* waters.
   - *Context*: Provides a clear empirical contrast between clean and highly polluted sample groups.

2. **Fecal Coliform (FC)**:
   - *Descriptive Observation*: Spans a wide dynamic range (median $140\text{ MPN/100mL}$ in cleaner waters up to extreme bacterial surges exceeding $10^6 - 10^7\text{ MPN/100mL}$ in *Class E*).
   - *Context*: Highlights acute biological contamination in lower quality classes.

3. **Dissolved Oxygen (DO)**:
   - *Descriptive Observation*: Waters in *Excellent* / *Good* classes maintain higher median dissolved oxygen ($>6.5\text{ mg/L}$), whereas samples in *Class D* / *Class E* exhibit lower oxygen levels frequently below $2.0\text{ mg/L}$.
   - *Context*: Reflects oxygen depletion characteristic of organically degraded surface water.

4. **Potential of Hydrogen (pH)**:
   - *Descriptive Observation*: Across classes *A* through *D*, pH remains centrally clustered around neutral-to-alkaline ranges (median $7.70$, IQR $7.30 - 8.10$), while *Class E* shows broader spread ($3.45$ to $9.20$).
   - *Context*: Visual overlap across intermediate classes reflects that pH alone is a buffered parameter; its diagnostic value occurs primarily during acute acidic or alkaline discharge events.

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
