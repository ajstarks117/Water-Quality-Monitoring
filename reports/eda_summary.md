# Exploratory Data Analysis (EDA) Summary Report

**Project Title**: AI-Driven Water Quality Prediction & Explainable Decision Support System  
**Deliverable**: Deliverable 2 (EDA Executive Summary Report)  
**Milestone**: 3.2 (Parameter-vs-Class Analysis & EDA Report)  
**Author**: Data Lead (Person A) & ML Lead (Person B)  
**Dataset**: Central Pollution Control Board (CPCB) Surface Water Quality Monitoring Data  
**Analyzed Scope**: 3,287 labeled observations across Maharashtra and Uttar Pradesh (2021 Sampling Records)  
**Schema Baseline**: Frozen under **Milestone 1.4 (Data Dictionary & Schema Freeze)**  

---

## 1. Executive Summary & Objective

This report provides a concise, non-technical synthesis of the exploratory data analysis conducted on the interim CPCB surface water quality dataset. The objective is to summarize data distributions, characterize class imbalance, examine how candidate water quality parameters vary across ground-truth classes as descriptive baseline evidence, and document constraints for downstream modeling.

Per the **Milestone 1.4 Schema Freeze**, the target variable is **`wqi_class`** (a 5-tier classification standard derived from CPCB Water Quality Index formulas: *Excellent*, *Good*, *Poor*, *Very Poor*, and *Unsuitable*), alongside a secondary binary target **`Potability`** (*Potable* vs. *Non-Potable*). The candidate feature set is strictly defined as the four frozen parameters:
1. `Potential of Hydrogen (pH)`
2. `Dissolved oxygen (mg/L)`
3. `Biochemical Oxygen Demand (mg/L)`
4. `Fecal Coliform (MPN/100mL)`

All four features are retained in the preprocessing pipeline without premature feature selection or elimination based on exploratory plots.

---

## 2. Key Findings

### 2.1 Target Class Distribution & Imbalance
The analyzed surface water dataset exhibits substantial class imbalance across the 3,287 observations:

| Water Quality Class | Category | Count | Percentage | Imbalance Characterization |
|:---|:---|---:|---:|:---|
| **Class A (Excellent)** | Potable | 7 | **0.21%** | **Extreme Minority** ($N=7$) |
| **Class B (Good)** | Potable | 110 | **3.35%** | **Severe Minority** |
| **Class C (Poor)** | Non-Potable | 455 | **13.84%** | Moderate |
| **Class D (Very Poor)** | Non-Potable | 919 | **27.96%** | High |
| **Class E (Unsuitable)** | Non-Potable | 1,796 | **54.64%** | **Dominant Majority** |
| **Total** | | **3,287** | **100.00%** | |

*Summary*: Over **82.6%** of monitored observations belong to *Very Poor* (27.96%) or *Unsuitable* (54.64%) classes. A naive majority-class classifier would achieve **54.64%** raw accuracy simply by predicting *Class E* for every sample.

---

### 2.2 Descriptive Parameter-vs-Class Distributions

The box and violin plots summarize the observed empirical distributions, median shifts, overlap, and spreads across classes for the four frozen features:

```
          [ Clear Progressive Shift ]                        [ Overlapping Central Values ]
   BOD  ───────────►  Dissolved Oxygen  ───────────►  Fecal Coliform  ───────────►  pH
(Median Shift)       (Lower-Tail Drop)              (Extreme Skew)               (Buffered)
```

1. **Biochemical Oxygen Demand (BOD)**:
   - *Distribution*: Exhibits a progressive increase in central values across classes: *Good* (median $1.30\text{ mg/L}$, IQR $1.20 - 1.40$), *Poor* (median $1.80\text{ mg/L}$, IQR $1.60 - 2.10$), *Very Poor* (median $3.00\text{ mg/L}$, IQR $2.80 - 3.40$), and *Unsuitable* (median $6.80\text{ mg/L}$, IQR $4.40 - 12.00\text{ mg/L}$).
   - *Spread & Tail*: In *Class E*, while the central 50% of observations lie between $4.40$ and $12.00\text{ mg/L}$, an extended upper tail reaches up to $127.00\text{ mg/L}$.

2. **Dissolved Oxygen (DO)**:
   - *Distribution*: Central values shift downward from *Good* (median $8.95\text{ mg/L}$, IQR $8.58 - 9.28$) and *Poor* (median $7.30\text{ mg/L}$, IQR $6.90 - 8.40$) to *Very Poor* (median $7.00\text{ mg/L}$, IQR $6.20 - 7.70$) and *Unsuitable* (median $5.40\text{ mg/L}$, IQR $4.20 - 6.70$).
   - *Spread & Tail*: In *Class E*, the lower tail extends into hypoxic levels below $2.00\text{ mg/L}$ (minimum observed: $0.30\text{ mg/L}$), while occasional supersaturated readings extend up to $28.00\text{ mg/L}$.

3. **Fecal Coliform (FC)**:
   - *Distribution*: Exhibits notable overlap across classes within central quartiles (*Good* median $220.0\text{ MPN/100mL}$, *Poor* median $6.1\text{ MPN/100mL}$, *Very Poor* median $20.0\text{ MPN/100mL}$, *Unsuitable* median $220.0\text{ MPN/100mL}$).
   - *Spread & Tail*: The *Unsuitable* class displays an extremely wide, heavy-tailed distribution with an upper quartile of $11,000\text{ MPN/100mL}$ and maximum observed concentrations reaching $2.2 \times 10^7\text{ MPN/100mL}$.

4. **Potential of Hydrogen (pH)**:
   - *Distribution*: Medians remain tightly buffered between $7.20$ and $7.95$ across all five classes (*Excellent*: $7.20$, *Good*: $7.32$, *Poor*: $7.60$, *Very Poor*: $7.95$, *Unsuitable*: $7.80$).
   - *Spread & Tail*: While classes *A* through *D* remain within $6.20 - 8.93$, *Class E* exhibits the widest range ($3.45$ to $9.20$).

> [!NOTE]
> **Methodological Boundary**: Visual separation and distributional differences serve strictly as descriptive exploratory evidence. They do **not** establish predictive importance, feature rankings, or SHAP attribution. Formal model explanations will be evaluated in Phase 7.

---

### 2.3 Feature Correlation & Multicollinearity
- **Redundancy Threshold Check**: Pairwise Pearson ($r$) and Spearman ($\rho$) correlation coefficients were computed across all numeric candidate features. **No pairwise correlation exceeded the $|r| > 0.85$ redundancy threshold**.
- **Observed Associations**: Dissolved Oxygen and BOD show a moderate negative rank correlation ($\rho = -0.36$), while BOD and Fecal Coliform exhibit a positive rank correlation ($\rho = +0.51$, $r = +0.31$).
- **Conclusion**: All four candidate features contribute non-redundant measurements and are retained for model development.

---

### 2.4 Spatial and Temporal Observations
- **Spatial Breakdown by State**:
  - *Maharashtra* ($N = 1,896$, $57.7\%$ of total): Class A: $0.32\%$, Class B: $1.16\%$, Class C: $11.34\%$, Class D: $31.43\%$, Class E: $55.75\%$.
  - *Uttar Pradesh* ($N = 1,391$, $42.3\%$ of total): Class A: $0.07\%$, Class B: $6.33\%$, Class C: $17.25\%$, Class D: $23.22\%$, Class E: $53.13\%$.
  - Both state subsets show a majority of samples classified as *Class E* ($>53\%$).
- **Temporal Coverage**:
  - Analysis of the timestamp field (`Data Acquisition Time`) indicates that all timestamped records ($N = 2,610$, with $677$ missing timestamps) originate from sampling dates within the calendar year **2021** (January 2021 – December 2021).
  - **Multi-year temporal trend analysis (e.g., across 2021–2025) is not supported by the available labeled dataset.**

---

## 3. Modeling & Evaluation Directives (Phase 4 – 9)

1. **Stratified Partitioning (M4.2)**: Given the extreme scarcity of *Class A* ($N=7$), standard unstratified splits risk omitting minority samples from evaluation folds. Splitting must use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
2. **Primary Evaluation Metrics (M5.1)**: Raw classification accuracy is uninformative due to majority class dominance ($54.64\%$). Model comparisons must prioritize **Macro F1-Score**, **Balanced Accuracy**, and per-class **Precision-Recall metrics**.
3. **Imbalance Handling Benchmarking (M6.1)**: Baseline unweighted models should be compared against class-weighted loss formulations (`class_weight='balanced'`) and synthetic oversampling strategies.

---

## 4. Visual Artifact References
- Univariate Histograms & Boxplots: [`reports/figures/eda/`](file:///d:/Coding/College/Data%20science/CP/reports/figures/eda/)
- Parameter-vs-Class Separation Figures: [`reports/figures/parameter_vs_class/`](file:///d:/Coding/College/Data%20science/CP/reports/figures/parameter_vs_class/)
- Spatial & Temporal Distributions: [`reports/figures/parameter_vs_class/spatial_state_wqi_distribution.png`](file:///d:/Coding/College/Data%20science/CP/reports/figures/parameter_vs_class/spatial_state_wqi_distribution.png)
- Ground-Truth Class Distribution Table: [`reports/class_distribution.csv`](file:///d:/Coding/College/Data%20science/CP/reports/class_distribution.csv)
