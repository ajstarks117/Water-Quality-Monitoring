# CPCB Water Quality Monitoring: Data Dictionary & Frozen Interim Schema

**Document ID**: `DICT-CPCB-SCHEMA-2026-V1`  
**Milestone**: Milestone 1.4 (Data Dictionary & Schema Freeze)  
**Owner**: Data Lead (Person A)  
**Status**: **FROZEN** (Baseline Schema Locked for Track B ML, Track C XAI, Track D Dashboard)  
**Applicable Dataset**: `data/interim/cpcb_labeled.csv` (3,287 records, 70 columns)  
**Last Updated**: 2026-10-05  

---

## 1. Executive Summary & Objective

This document establishes the authoritative Data Dictionary and **Frozen Interim Schema Contract** for the CPCB Surface Water Quality monitoring records. All downstream tracks—**Track B (ML Models)**, **Track C (Explainability & Decision Support)**, and **Track D (Interactive Dashboard & API)**—must build and validate against this frozen contract.

### Key Milestones & Contracts Unblocked:
- **Unblocks ML Track (Track B)**: Exact candidate feature names, types, missingness, and target columns defined.
- **Unblocks XAI Track (Track C)**: Parameter units, regulatory baselines, and feature semantics defined.
- **Unblocks Dashboard Track (Track D)**: Station metadata, class label names, and range bounds defined for visual widgets.

---

## 2. Target Column Specifications

The dataset provides two ground-truth target variables computed according to CPCB / BIS IS 10500:2012 / CCME standards via `src/data/compute_wqi.py` (Milestone 1.3):

### 2.1 Primary Target: `wqi_class` (Multi-Class Classification)
- **Column Name**: `wqi_class`
- **Data Type**: `object` / `string`
- **Role**: Primary Supervised Multi-Class Target
- **Classes**: `Excellent`, `Good`, `Poor`, `Very Poor`, `Unsuitable`
- **Derivation Source**: Categorical mapping of continuous `wqi_score` (Weighted Arithmetic WQI)
- **M1.3 Confirmed Class Distribution** (N = 3,287):

| Class Label | CPCB Designated Category | Continuous WQI Score Range | Record Count | Percentage | Potability Status |
|:---|:---|:---|---:|---:|:---|
| **Excellent** | Class A | $0.00 \le \text{WQI} \le 25.00$ | 7 | 0.21% | Potable (`1`) |
| **Good** | Class B | $25.00 < \text{WQI} \le 50.00$ | 110 | 3.35% | Potable (`1`) |
| **Poor** | Class C | $50.00 < \text{WQI} \le 75.00$ | 455 | 13.84% | Non-Potable (`0`) |
| **Very Poor** | Class D | $75.00 < \text{WQI} \le 100.00$ | 919 | 27.96% | Non-Potable (`0`) |
| **Unsuitable** | Class E | $\text{WQI} > 100.00$ | 1,796 | 54.64% | Non-Potable (`0`) |
| **Total** | — | — | **3,287** | **100.00%** | — |

### 2.2 Secondary Target: `Potability` (Binary Classification)
- **Column Name**: `Potability`
- **Data Type**: `int64`
- **Role**: Secondary Binary Classification Target
- **Values**:
  - `1` = **Potable** (`wqi_score <= 50.0`: Excellent or Good) — **117 records (3.56%)**
  - `0` = **Non-Potable** (`wqi_score > 50.0`: Poor, Very Poor, or Unsuitable) — **3,170 records (96.44%)**

---

## 3. Strict Feature Exclusion Rules & Data Leakage Prevention

> [!CAUTION]
> ### RULE 1: MANDATORY TARGET LEAKAGE GUARD (`wqi_score`)
> - **Column**: `wqi_score` (float64)
> - **Reason for Exclusion**: `wqi_score` is the continuous numerical index calculated from water quality parameters from which `wqi_class` and `Potability` are directly derived.
> - **Rule**: `wqi_score` **MUST NEVER** be included in `feature_columns` in `config/config.yaml` or in any ML training/testing matrix (`X_train`, `X_test`).
> - **Enforcement**: Tested programmatically by TC-1.4-02 in `tests/test_data_dictionary_and_schema.py`.

> [!WARNING]
> ### RULE 2: STATION IDENTIFIERS & ADMINISTRATIVE METADATA EXCLUSION
> - **Columns (24 total)**: `SlNo`, `Station`, `Agency`, `State LGD Code`, `State`, `District LGD Code`, `District`, `Tehsil`, `Block`, `Village`, `River`, `Basin`, `Tributary`, `Subtributary`, `SubSubtributary`, `Local River`, `Latitude`, `Longitude`, `Data Acquisition Time`, `has_physical_record`, `year`, `month`, `day`, `sampling_date`.
> - **Reason for Exclusion**: Excluded from default predictive feature sets to prevent geographic memorization, administrative bias, and temporal leakage. Retained for spatial aggregation, filtering, and dashboard map visualizations.

> [!NOTE]
> ### RULE 3: HIGH MISSINGNESS / SPARSE PARAMETERS EXCLUSION
> - **Columns (40 total)**: Parameters with $>85\%$ to $100\%$ missingness in CPCB official records (e.g., Heavy Metals, Nitrate, Total Hardness, Turbidity, EC).
> - **Reason for Exclusion**: Insufficient sample density across stations; imputing $>95\%$ missingness would fabricate artificial signals.

---

## 4. Frozen Candidate Feature Set

The candidate feature set comprises high-coverage physical, chemical, and biological parameters meeting CPCB quality standards:

| Feature Column Name | Data Type | Unit | Plausible Range | Observed Range | Missing Count (%) | CPCB/BIS Permissible Limit ($S_i$) | Ideal Value ($V_0$) | Feature Role & Description |
|:---|:---|:---|:---|:---|---:|:---|:---|:---|
| `Potential of Hydrogen (pH)` | `float64` | pH scale ($0-14$) | $[0.0, 14.0]$ | $[3.45, 9.20]$ | 0 (0.00%) | $6.5 - 8.5$ ($S_i = 8.5$) | $7.0$ | **Core Chemical Feature**: Acidity/alkalinity balance. |
| `Dissolved oxygen (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 30.0]$ | $[0.30, 28.00]$ | 90 (2.74%) | $\ge 5.0$ ($S_i = 5.0$) | $14.6$ | **Core Biological Feature**: Aquatic respiratory support. |
| `Biochemical Oxygen Demand (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 200.0]$ | $[1.10, 127.00]$ | 178 (5.42%) | $\le 3.0$ ($S_i = 3.0$) | $0.0$ | **Core Organic Feature**: 3-day biochemical oxygen demand. |
| `Fecal Coliform (MPN/100mL)` | `float64` | $\text{MPN/100mL}$ | $[0.0, 10^8]$ | $[2.0, 2.2 \times 10^7]$ | 267 (8.12%) | $\le 250.0$ ($S_i = 250.0$) | $0.0$ | **Core Microbial Feature**: Pathogenic sewage indicator. |

---

## 5. Complete 70-Column Data Dictionary Table

Every column present in `data/interim/cpcb_labeled.csv` is exhaustively cataloged below:

| # | Column Name | Data Type | Unit | Plausible Range | Observed Range / Unique | % Missing | Column Status | Description |
|---:|:---|:---|:---|:---|:---|---:|:---|:---|
| 0 | `SlNo` | `int64` | Index | $[1, \infty)$ | $[1.0, 1917.0]$ | 0.00% | Excluded (Metadata) | Source file sequence serial number. |
| 1 | `Station` | `object` | String | Station Name | 337 unique names | 0.00% | Excluded (Identifier) | CPCB water quality monitoring station name. |
| 2 | `Agency` | `object` | String | Agency Name | `['CPCB']` | 0.00% | Excluded (Metadata) | Monitoring regulatory agency. |
| 3 | `State LGD Code` | `int64` | LGD Code | $[1, 99]$ | $[9.0, 27.0]$ | 0.00% | Excluded (Metadata) | MoPR Local Government Directory state code. |
| 4 | `State` | `object` | String | State Name | `['Maharashtra', 'Uttar Pradesh']` | 0.00% | Excluded (Metadata) | Indian state name. |
| 5 | `District LGD Code` | `int64` | LGD Code | $[1, 999]$ | $[118.0, 665.0]$ | 0.00% | Excluded (Metadata) | MoPR Local Government Directory district code. |
| 6 | `District` | `object` | String | District Name | 71 unique districts | 0.00% | Excluded (Metadata) | Administrative district name. |
| 7 | `Tehsil` | `object` | String | Tehsil Name | 154 unique tehsils | 0.00% | Excluded (Metadata) | Sub-district administrative division. |
| 8 | `Block` | `object` | String | Block Name | `['-']` | 0.00% | Excluded (Metadata) | Administrative block name (unpopulated in source). |
| 9 | `Village` | `object` | String | Village Name | `['-']` | 0.00% | Excluded (Metadata) | Village name (unpopulated in source). |
| 10 | `River` | `object` | String | River Name | `['-']` | 0.00% | Excluded (Metadata) | River name identifier (unpopulated in source). |
| 11 | `Basin` | `object` | String | Basin Name | `['-']` | 0.00% | Excluded (Metadata) | River basin identifier (unpopulated in source). |
| 12 | `Tributary` | `object` | String | River Name | `['-']` | 0.00% | Excluded (Metadata) | Tributary river identifier (unpopulated in source). |
| 13 | `Subtributary` | `object` | String | River Name | `['-']` | 0.00% | Excluded (Metadata) | Sub-tributary identifier (unpopulated in source). |
| 14 | `SubSubtributary` | `object` | String | River Name | `['-']` | 0.00% | Excluded (Metadata) | Sub-sub-tributary identifier (unpopulated in source). |
| 15 | `Local River` | `object` | String | River Name | `['-']` | 0.00% | Excluded (Metadata) | Local stream/river name (unpopulated in source). |
| 16 | `Latitude` | `float64` | ºN | $[8.0, 37.0]$ | $[15.75, 31.88]$ | 0.00% | Excluded (Metadata) | Station latitude in decimal degrees. |
| 17 | `Longitude` | `float64` | ºE | $[68.0, 97.5]$ | $[72.68, 83.87]$ | 0.00% | Excluded (Metadata) | Station longitude in decimal degrees. |
| 18 | `Data Acquisition Time` | `object` | Timestamp | DD-MM-YYYY HH:MM | 283 unique timestamps | 0.00% | Excluded (Metadata) | Official monitoring sampling timestamp. |
| 19 | `Biochemical Oxygen Demand (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 200.0]$ | $[1.10, 127.00]$ | 5.42% | **Candidate Feature** | Organic biological oxygen demand (3-day at 27ºC). |
| 20 | `Chemical Oxygen Demand (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 500.0]$ | `N/A` (all null) | 100.00% | Excluded (High Missingness) | Chemical oxygen demand (unpopulated). |
| 21 | `Fecal Coliform (MPN/100mL)` | `float64` | $\text{MPN/100mL}$ | $[0.0, 10^8]$ | $[2.0, 2.2 \times 10^7]$ | 8.12% | **Candidate Feature** | Faecal coliform bacterial count. |
| 22 | `Total Coliform (MPN/100mL)` | `float64` | $\text{MPN/100mL}$ | $[0.0, 10^8]$ | $[11.0, 1.4 \times 10^7]$ | 97.75% | Excluded (High Missingness) | Total coliform bacterial count. |
| 23 | `Amonia N (mgN/L)` | `float64` | $\text{mgN/L}$ | $[0.0, 50.0]$ | $[0.40, 26.90]$ | 99.85% | Excluded (High Missingness) | Ammonia nitrogen concentration. |
| 24 | `Boron (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 10.0]$ | `N/A` (all null) | 100.00% | Excluded (High Missingness) | Boron concentration. |
| 25 | `Carbonate (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 500.0]$ | `N/A` (all null) | 100.00% | Excluded (High Missingness) | Carbonate ion concentration. |
| 26 | `Calcium (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 500.0]$ | `N/A` (all null) | 100.00% | Excluded (High Missingness) | Calcium ion concentration. |
| 27 | `Chloride (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 2000.0]$ | $[6.00, 2699.20]$ | 98.54% | Excluded (High Missingness) | Chloride ion concentration. |
| 28 | `Dissolved oxygen (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 30.0]$ | $[0.30, 28.00]$ | 2.74% | **Candidate Feature** | Dissolved oxygen concentration. |
| 29 | `Total Dissolved Solids (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 10000.0]$ | $[48.00, 5462.00]$ | 98.48% | Excluded (High Missingness) | Total dissolved solids. |
| 30 | `Fluoride (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 10.0]$ | `N/A` (all null) | 100.00% | Excluded (High Missingness) | Fluoride concentration. |
| 31 | `Bicarbonate (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 1000.0]$ | `N/A` (all null) | 100.00% | Excluded (High Missingness) | Bicarbonate ion concentration. |
| 32 | `Potassium (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 100.0]$ | `N/A` (all null) | 100.00% | Excluded (High Missingness) | Potassium ion concentration. |
| 33 | `Magnesium (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 500.0]$ | `N/A` (all null) | 100.00% | Excluded (High Missingness) | Magnesium ion concentration. |
| 34 | `Sodium (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 1000.0]$ | `N/A` (all null) | 100.00% | Excluded (High Missingness) | Sodium ion concentration. |
| 35 | `Nitrite N+Nitrate N (mgN/L)` | `float64` | $\text{mgN/L}$ | $[0.0, 100.0]$ | `N/A` (all null) | 100.00% | Excluded (High Missingness) | Nitrite + Nitrate nitrogen. |
| 36 | `Potential of Hydrogen (pH)` | `float64` | pH scale | $[0.0, 14.0]$ | $[3.45, 9.20]$ | 0.00% | **Candidate Feature** | Acidity / alkalinity measure. |
| 37 | `Sulphate (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 1000.0]$ | `N/A` (all null) | 100.00% | Excluded (High Missingness) | Sulfate ion concentration. |
| 38 | `Total Phosphorus (mgP/L)` | `float64` | $\text{mgP/L}$ | $[0.0, 20.0]$ | `N/A` (all null) | 100.00% | Excluded (High Missingness) | Total phosphorus concentration. |
| 39 | `Arsenic (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 1.0]$ | `N/A` (all null) | 100.00% | Excluded (High Missingness) | Arsenic concentration. |
| 40 | `Total Alkalinity (mg/L as CaCO3)` | `float64` | $\text{mgCaCO}_3\text{/L}$ | $[0.0, 2000.0]$ | $[22.00, 1318.00]$ | 98.72% | Excluded (High Missingness) | Total alkalinity. |
| 41 | `Cadmium (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 0.5]$ | $[0.01, 0.01]$ | 99.91% | Excluded (High Missingness) | Cadmium heavy metal concentration. |
| 42 | `Chromium (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 2.0]$ | $[0.09, 0.20]$ | 99.76% | Excluded (High Missingness) | Chromium heavy metal concentration. |
| 43 | `Iron(mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 10.0]$ | $[0.01, 0.91]$ | 98.94% | Excluded (High Missingness) | Iron concentration. |
| 44 | `Hardness Calcium (mgCaCO3/L)` | `float64` | $\text{mgCaCO}_3\text{/L}$ | $[0.0, 2000.0]$ | $[8.02, 1400.00]$ | 98.84% | Excluded (High Missingness) | Calcium hardness component. |
| 45 | `Mercury(mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 0.1]$ | $[0.02, 0.02]$ | 99.97% | Excluded (High Missingness) | Mercury heavy metal concentration. |
| 46 | `Hardness_Magnesium (mg/L as CaCO3)` | `float64` | $\text{mgCaCO}_3\text{/L}$ | $[0.0, 2000.0]$ | $[6.00, 1162.00]$ | 98.72% | Excluded (High Missingness) | Magnesium hardness component. |
| 47 | `Total Hardness (mgCaCO3/L)` | `float64` | $\text{mgCaCO}_3\text{/L}$ | $[0.0, 3000.0]$ | $[26.00, 1840.00]$ | 98.51% | Excluded (High Missingness) | Total water hardness. |
| 48 | `Manganese (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 5.0]$ | `N/A` (all null) | 100.00% | Excluded (High Missingness) | Manganese concentration. |
| 49 | `Nitrate N (mgN/L)` | `float64` | $\text{mgN/L}$ | $[0.0, 100.0]$ | $[0.40, 13.59]$ | 99.09% | Excluded (High Missingness) | Nitrate nitrogen concentration. |
| 50 | `Nickel (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 1.0]$ | $[0.02, 0.25]$ | 99.54% | Excluded (High Missingness) | Nickel heavy metal concentration. |
| 51 | `Lead (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 1.0]$ | $[0.28, 0.33]$ | 99.91% | Excluded (High Missingness) | Lead heavy metal concentration. |
| 52 | `Zinc (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 10.0]$ | $[0.01, 0.96]$ | 99.18% | Excluded (High Missingness) | Zinc concentration. |
| 53 | `Sodium Adsorption Ratio (%)` | `float64` | Ratio ($\%$) | $[0.0, 100.0]$ | $[1.02, 9.60]$ | 99.88% | Excluded (High Missingness) | Sodium adsorption ratio for irrigation. |
| 54 | `Nitrate (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 100.0]$ | `N/A` (all null) | 100.00% | Excluded (High Missingness) | Nitrate concentration. |
| 55 | `Copper (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 5.0]$ | `N/A` (all null) | 100.00% | Excluded (High Missingness) | Copper heavy metal concentration. |
| 56 | `Electric Conductivity (μS/cm)` | `float64` | $\mu\text{S/cm}$ | $[0.0, 20000.0]$ | $[80.00, 9924.00]$ | 97.17% | Excluded (High Missingness) | Electrical conductivity. |
| 57 | `Colour` | `object` | String | Descriptive | 4 unique values | 86.86% | Excluded (High Missingness) | Visual color observation. |
| 58 | `Odour` | `object` | String | Descriptive | 2 unique values | 99.12% | Excluded (High Missingness) | Sensory odour observation. |
| 59 | `Turbidity (NTU)` | `float64` | NTU | $[0.0, 500.0]$ | $[1.04, 10.00]$ | 99.42% | Excluded (High Missingness) | Water clarity / turbidity. |
| 60 | `Temperature (ºC)` | `float64` | ºC | $[0.0, 50.0]$ | `N/A` (all null) | 100.00% | Excluded (High Missingness) | Water temperature. |
| 61 | `Total Solids (mg/L)` | `float64` | $\text{mg/L}$ | $[0.0, 10000.0]$ | `N/A` (all null) | 100.00% | Excluded (High Missingness) | Total solids. |
| 62 | `has_physical_record` | `bool` | Boolean | `[True, False]` | `[False, True]` | 0.00% | Excluded (Metadata) | Flag indicating physical file match in join. |
| 63 | `year` | `int64` | Year | $[2020, 2030]$ | $[2021, 2021]$ | 0.00% | Excluded (Metadata) | Observation calendar year. |
| 64 | `month` | `int64` | Month | $[1, 12]$ | $[1, 12]$ | 0.00% | Excluded (Metadata) | Observation calendar month. |
| 65 | `day` | `int64` | Day | $[1, 31]$ | $[1, 31]$ | 0.00% | Excluded (Metadata) | Observation calendar day. |
| 66 | `sampling_date` | `object` | Date String | YYYY-MM-DD | 283 unique dates | 0.00% | Excluded (Metadata) | ISO formatted sampling date. |
| 67 | `wqi_score` | `float64` | WQI Score | $[0.0, \infty)$ | $[11.89, 78649.66]$ | 0.00% | **STRICTLY EXCLUDED (LEAKAGE GUARD)** | Calculated WQI score; target derivation only. |
| 68 | `wqi_class` | `object` | Class Label | 5 Classes | 5 Classes | 0.00% | **Primary Multi-Class Target** | Ground-truth CPCB WQI classification class. |
| 69 | `Potability` | `int64` | Binary Flag | `[0, 1]` | `[0, 1]` | 0.00% | **Secondary Binary Target** | Canonical binary potability target. |

---

## 7. Schema Freeze Verification & Change Management Protocol

### 7.1 Schema Freeze Guarantees
1. **Column Names & Ordering**: Frozen as documented. No silent renames or drops.
2. **Target Definition**: `wqi_class` is the primary multi-class classification target; `Potability` is the secondary binary target.
3. **Feature List**: `feature_columns` defined in `config/config.yaml` is frozen as:
   - `Potential of Hydrogen (pH)`
   - `Dissolved oxygen (mg/L)`
   - `Biochemical Oxygen Demand (mg/L)`
   - `Fecal Coliform (MPN/100mL)`
4. **Leakage Guard**: `wqi_score` is strictly excluded from all feature sets.

### 7.2 Change Management Protocol
If an exception or schema alteration is necessary during subsequent phases (Phase 2 - Phase 8):
1. An official exception must be raised in the shared team channel (`#water-quality-core`).
2. Written confirmation from at least one engineer in each affected track (Data, ML, XAI, Dashboard) is required.
3. `Docs/data_dictionary.md`, `Docs/interface_contracts.md`, and `config/config.yaml` must be updated atomically in a dedicated pull request.

---

## 8. Sign-off & Track Handoff

| Track | Owner / Role | Status | Contract Acknowledgment |
|:---|:---|:---|:---|
| **Track A: Data Pipeline** | Person A (Data Lead) | **Completed & Frozen** | Interim dataset generated, labeled, audited, and frozen. |
| **Track B: ML Modeling** | ML Lead | **Unblocked** | Ready to ingest candidate features and target labels for baseline modeling. |
| **Track C: XAI & Decision Support** | XAI Lead | **Unblocked** | Regulatory standards, units, and parameter semantics confirmed. |
| **Track D: Dashboard & Integration** | Dashboard Lead | **Unblocked** | Station metadata, column formats, and class definitions ready for UI integration. |
