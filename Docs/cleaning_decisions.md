# Data Cleaning & Missing Value Strategy Decisions

**Document ID**: `DEC-CLEANING-PREPROC-2026-V1`  
**Milestone**: Milestone 2.1 (Missing Value & Duplicate Handling)  
**Owner**: Data Lead (Person A)  
**Status**: **Approved & Implemented**  
**Applicable Dataset**: `data/interim/cpcb_labeled.csv` (3,287 records, 70 columns)  
**Last Updated**: 2026-10-05  

---

## 1. Executive Summary & Objective

This document formalizes and records the dataset cleaning decisions, missing value handling strategies, column-dropping thresholds, and duplicate investigation findings for the CPCB Surface Water Quality Monitoring pipeline.

Per the project specification, **no column or row is dropped via unprincipled blanket rules**. Every threshold is explicitly justified, documented, and encoded in `src/data/preprocessing.py`.

---

## 2. Duplicate Row Investigation & Policy

### 2.1 Audit Findings
An exhaustive duplicate scan on `data/interim/cpcb_labeled.csv` (N = 3,287) revealed:
- **Exact Full Row Duplicates**: **0 records (0.00%)**
- **Duplicate Primary Observation Keys `(Station, Data Acquisition Time)`**: **0 records (0.00%)**

### 2.2 Domain Context & Repeated Measurements Policy
- In water quality monitoring, repeated samplings at the same station on different dates/times represent **valid temporal time-series events**, not duplicates.
- The exact-timestamp join strategy executed in Milestone 1.2 successfully collapsed physical, biological, and chemical records on `(Station, Data Acquisition Time)`, eliminating Cartesian duplicates.
- **Handling Policy**: `handle_duplicates(df, subset=None, keep='first')` is applied at the entry of the preprocessing pipeline. If future data batches introduce identical duplicate records, only exact full duplicates will be deduplicated, preserving temporal granularity.

---

## 3. Missing Value Analysis & Pattern Audit

### 3.1 Global Missingness Breakdown
The 70 columns in `data/interim/cpcb_labeled.csv` fall into three distinct missingness tiers:

```
Tier 1: Core Complete Metadata & Targets (0.0% Missing)  ─── 24 columns (State, Station, wqi_class, Potability, etc.)
Tier 2: High-Coverage Candidate Features (0.0% - 8.12%)  ───  4 columns (pH, DO, BOD, Fecal Coliform)
Tier 3: Sparse / Supplementary Parameters (86.9% - 100%) ─── 42 columns (Heavy Metals, Nutrients, Anions, Sensory)
```

### 3.2 Candidate Feature Missingness & Skewness
| Candidate Feature Column | Observed Missing Count | Observed Missing % | Distribution Min / Median / Max | Distribution Skewness | Chosen Strategy |
|:---|---:|---:|:---|:---|:---|
| `Potential of Hydrogen (pH)` | 0 | 0.00% | $3.45$ / $7.70$ / $9.20$ | Mild left skew | No imputation needed |
| `Dissolved oxygen (mg/L)` | 90 | 2.74% | $0.30$ / $6.20$ / $28.00$ | Moderate right skew | **Median Imputation** |
| `Biochemical Oxygen Demand (mg/L)` | 178 | 5.42% | $1.10$ / $2.80$ / $127.00$ | Severe right skew ($>100\text{ mg/L}$) | **Median Imputation** |
| `Fecal Coliform (MPN/100mL)` | 267 | 8.12% | $2.0$ / $140.0$ / $2.2 \times 10^7$ | Extreme right skew ($10^7\text{ order}$) | **Median Imputation** |

### 3.3 Missingness Patterns Across States & Classes
- **By State**:
  - Maharashtra (N = 1,896): pH (0.0%), DO (2.48%), BOD (2.69%), FC (7.38%)
  - Uttar Pradesh (N = 1,391): pH (0.0%), DO (3.09%), BOD (9.13%), FC (9.13%)
  - *Finding*: Missingness is evenly distributed across both state jurisdictions without systematic administrative data loss.
- **By Ground-Truth WQI Class**:
  - Class A / Excellent (N = 7): pH (0.0%), DO (100% missing in these 7 rows), BOD (100% missing in these 7 rows), FC (0.0%). These pristine records were scored during M1.3 using pH, FC, and supplementary physical records under dynamic weight redistribution.
  - Class B / Good (N = 110): DO (20.0%), BOD (36.36%), FC (4.55%)
  - Class C / Poor (N = 455): DO (2.86%), BOD (17.58%), FC (11.65%)
  - Class D / Very Poor (N = 919): DO (0.54%), BOD (3.16%), FC (10.01%)
  - Class E / Unsuitable (N = 1,796): DO (2.39%), BOD (1.22%), FC (6.51%)

---

## 4. Cleaning Decisions & Thresholds

### Decision 1: Column-Level Dropping Threshold ($X = 40.0\%$)
- **Threshold**: Any non-target, non-metadata parameter column with $> 40.0\%$ missingness is dropped from the model feature set.
- **Rationale**: Imputing columns with $>85\%$ missingness (e.g. Heavy metals, Turbidity, Total Hardness) creates artificial hallucinated signals and distorts tree splits and linear gradients.
- **Impact**: All 39 supplementary sparse columns are cleanly excluded from feature matrices.

### Decision 2: Feature Imputation Strategy (Median over Mean)
- **Strategy**: **Median Imputation** calculated strictly on training folds (`X_train`).
- **Rationale**: Water quality biological/organic parameters (`Fecal Coliform`, `BOD`) follow heavy right-tailed log-normal or power-law distributions with extreme outliers ($2.2 \times 10^7\text{ MPN/100mL}$). Mean imputation would drastically shift central tendencies upward and introduce severe bias. Median imputation preserves robust non-parametric central tendency.

### Decision 3: Row-Dropping Policy (Zero Unprincipled Row Drops)
- **Policy**: No observation row is dropped due to missing feature values.
- **Rationale**: Rows in `cpcb_labeled.csv` already passed the Minimum Parameter Threshold constraint ($k \ge 3$) during M1.3 WQI calculation. Imputing the $<8.12\%$ missing values preserves all 3,287 labeled samples for downstream training.

### Decision 4: Strict Train/Test Leakage Prevention
- **Policy**: Imputation parameters (medians) must be computed **strictly on `X_train`** and transformed onto `X_test` / `X_val` without re-fitting.
- `handle_missing_values` supports a stateful transformer pattern (`fit_transform` and `transform` via `fitted_imputers`).

---

## 5. Summary Table of Preprocessing Rules

| Parameter / Column Group | Missingness % | Action | Justification |
|:---|---:|:---|:---|
| `Potential of Hydrogen (pH)` | 0.00% | Retain as feature | Complete coverage; core chemical parameter. |
| `Dissolved oxygen (mg/L)` | 2.74% | Impute via train median | Low missingness; robust to upper outliers. |
| `Biochemical Oxygen Demand (mg/L)` | 5.42% | Impute via train median | Low missingness; extreme upper skew requires median. |
| `Fecal Coliform (MPN/100mL)` | 8.12% | Impute via train median | Low missingness; heavy bacterial distribution tail. |
| Supplementary Parameters (39 cols) | 86.86% – 100.0% | Drop from feature set | $>40\%$ threshold exceeded; prevents artificial noise. |
| Metadata Columns (24 cols) | 0.00% | Retain for grouping / filter | Required for spatial analysis and map visualization. |
| `wqi_score` | 0.00% | **Strictly Exclude** | Target derivation intermediate; ML leakage guard. |
| `wqi_class`, `Potability` | 0.00% | Retain as Targets | Ground-truth multi-class & binary targets. |

---

## 6. Outlier Investigation & Domain Validity Decisions (Milestone 2.2)

### 6.1 Statistical Outlier Metrics vs Physical Plausibility

Every numeric parameter was evaluated using Interquartile Range ($1.5 \times \text{IQR}$, $3.0 \times \text{IQR}$) and standard deviation ($|z| > 3.0$):

| Parameter | Domain Plausible Range | Observed Range | IQR Outliers Count (%) | Extreme IQR (> 3.0 IQR) | $|z| > 3.0$ Count (%) | Decision & Rationale |
|:---|:---|:---|---:|---:|---:|:---|
| `Potential of Hydrogen (pH)` | $[0.0, 14.0]$ | $[3.45, 9.20]$ | 16 (0.49%) | 0 (0.00%) | 12 (0.37%) | **RETAIN ALL**: 100% within $[0, 14]$. Real acidic/alkaline stream occurrences. |
| `Dissolved oxygen (mg/L)` | $[0.0, 30.0]$ | $[0.30, 28.00]$ | 161 (5.04%) | 1 (0.03%) | 3 (0.09%) | **RETAIN ALL**: 100% within $[0, 30]$. High DO ($>14$) reflects cold/algal supersaturation. |
| `Biochemical Oxygen Demand (mg/L)` | $[0.0, 500.0]$ | $[1.10, 127.00]$ | 234 (7.53%) | 151 (4.86%) | 101 (3.25%) | **RETAIN ALL**: High BOD ($>20\text{ mg/L}$) indicates severe raw sewage/industrial contamination. Deleting would remove critical pollution signals. |
| `Fecal Coliform (MPN/100mL)` | $[0.0, 10^8]$ | $[2.0, 2.2 \times 10^7]$ | 578 (19.14%) | 530 (17.55%) | 19 (0.63%) | **RETAIN ALL**: Extreme bacterial counts ($10^5 - 10^7$) are authentic untreated urban drain signals in Class E rivers. |

### 6.2 Distinction: Measurement Error vs Genuine Environmental Extremes
- **Category A: Physically Impossible Values (Measurement Errors)**:
  - Definition: Values outside physical reality ($\text{pH} < 0.0$ or $\text{pH} > 14.0$, $\text{DO} < 0.0$, $\text{BOD} < 0.0$, $\text{FC} < 0.0$).
  - Policy: Automatically detected and filtered in `handle_outliers(df, remove_invalid=True)`.
  - Observed in official dataset: **0 records** (100% of recorded values are physically possible).
- **Category B: Genuine Environmental Extreme Events**:
  - Definition: Statistically high readings that reflect real acute pollution events (effluent discharge, seasonal low-flow hypoxia, microbial surges).
  - Policy: **Kept without deletion or arbitrary capping**.

### 6.3 Class Balance Impact Verification
- Because all 3,287 records in `data/interim/cpcb_labeled.csv` are physically valid, running `handle_outliers()` retains $100\%$ of observations across all 5 classes (`Excellent`: 7, `Good`: 110, `Poor`: 455, `Very Poor`: 919, `Unsuitable`: 1,796).
- Class loss percentage across all classes is **0.00%**.

### 6.4 Visual Audit Artifacts
Box plots visualizing distributions and class-stratified distributions are saved in:
- `reports/figures/outlier_boxplots/candidate_features_boxplots.png`
- `reports/figures/outlier_boxplots/potential_of_hydrogen_by_class_boxplot.png`
- `reports/figures/outlier_boxplots/dissolved_oxygen_by_class_boxplot.png`
- `reports/figures/outlier_boxplots/biochemical_oxygen_demand_by_class_boxplot.png`
- `reports/figures/outlier_boxplots/fecal_coliform_by_class_boxplot.png`
