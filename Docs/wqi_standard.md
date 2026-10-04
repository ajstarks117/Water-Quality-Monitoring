# Water Quality Index (WQI) Standard & Mathematical Formulation

**Document ID**: `STD-WQI-CPCB-2026-V1`  
**Applicable Milestones**: Milestone 1.1, Milestone 1.3 (Target Labeling), Milestone 8.1 (Rule-Engine Thresholds)  
**Publishing Body / Source Standard**: 
1. **Central Pollution Control Board (CPCB)**, Ministry of Environment, Forest and Climate Change, Govt. of India (*Guidelines for Water Quality Management*, 2008 & 2017).
2. **Bureau of Indian Standards (BIS)**: `IS 10500:2012` (*Drinking Water — Specification, Second Revision*).
3. **Brown et al. (1970) / Horton (1965)**: *Weighted Arithmetic Water Quality Index (WAWQI) Method*.
4. **Canadian Council of Ministers of the Environment (CCME)**: *CCME Water Quality Index 1.0 User's Manual* (2001, updated 2017).

---

## 1. Executive Summary & Objective

This document establishes the official mathematical formulation, parameter standards, sub-index weightings, class boundaries, and missingness fallback policies used to compute the Water Quality Index ($WQI$) and derive ground-truth classification labels ($wqi\_class$ and binary $Potability$) from raw CPCB Surface Water Quality monitoring records.

Every threshold, standard limit ($S_i$), ideal value ($V_0$), and classification boundary documented here traces directly to published CPCB and BIS standards.

---

## 2. Mathematical Formulation: Weighted Arithmetic WQI Method

The **Weighted Arithmetic Water Quality Index (WAWQI)** method is selected due to its wide adoption in Indian hydrological studies, regulatory compliance monitoring by CPCB/CWC, and its capacity to handle multi-parameter chemical and biological water assessments.

### 2.1 Formula

The overall Water Quality Index ($WQI$) is defined as:

$$WQI = \frac{\sum_{i=1}^{n} w_i q_i}{\sum_{i=1}^{n} w_i}$$

where:
- $n$: Total number of water quality parameters evaluated.
- $q_i$: Quality rating (sub-index score) for the $i$-th parameter.
- $w_i$: Unit weight factor assigned to the $i$-th parameter.

---

### 2.2 Sub-Index Quality Rating ($q_i$)

The sub-index rating $q_i$ reflects the relative concentration of the parameter compared to its standard permissible limit:

$$q_i = \left| \frac{V_i - V_0}{S_i - V_0} \right| \times 100$$

where:
- $V_i$: Measured experimental concentration or value of parameter $i$.
- $S_i$: Standard permissible limit for parameter $i$ as prescribed by **BIS IS 10500:2012** / **CPCB Class A/B/C**.
- $V_0$: Ideal value of the parameter in pure water:
  - For **pH**: $V_0 = 7.00$
  - For **Dissolved Oxygen (DO)**: $V_0 = 14.60\text{ mg/L}$ (saturation at $0^\circ\text{C}$)
  - For all other physical, chemical, and biological parameters: $V_0 = 0.00$

#### Special Boundary Rules for $q_i$:
1. **pH Sub-Index**:
   - If $V_{\text{pH}} = 7.0 \implies q_{\text{pH}} = 0$
   - If $V_{\text{pH}} \ge 7.0 \implies q_{\text{pH}} = \frac{V_{\text{pH}} - 7.0}{8.5 - 7.0} \times 100 = \frac{V_{\text{pH}} - 7.0}{1.5} \times 100$
   - If $V_{\text{pH}} < 7.0 \implies q_{\text{pH}} = \frac{7.0 - V_{\text{pH}}}{7.0 - 6.5} \times 100 = \frac{7.0 - V_{\text{pH}}}{0.5} \times 100$
2. **Dissolved Oxygen (DO) Sub-Index** (higher concentration is better):
   - $q_{\text{DO}} = \left| \frac{V_{\text{DO}} - 14.6}{5.0 - 14.6} \right| \times 100 = \left| \frac{V_{\text{DO}} - 14.6}{-9.6} \right| \times 100$

---

### 2.3 Unit Weight Factor Calculation ($w_i$)

The unit weight $w_i$ is inversely proportional to the standard permissible limit $S_i$, ensuring that toxic or high-risk parameters have higher influence on the index:

$$w_i = \frac{K}{S_i}$$

where the constant of proportionality $K$ is:

$$K = \frac{1}{\sum_{i=1}^{n} \frac{1}{S_i}}$$

Normalized weights satisfy $\sum_{i=1}^{n} w_i = 1.00$.

---

## 3. Parameter Standards, CPCB Mapping & Weight Allocation

The parameters are mapped directly to columns in the **CPCB National Water Data Portal (NWDP)** surface water datasets (Physical, Chemical, and Biological monitoring files).

| Parameter Name | CPCB Source File | Raw CPCB Column Name | Standard Unit | BIS IS 10500:2012 / CPCB Limit ($S_i$) | Ideal Value ($V_0$) | Base Unit Weight ($w_i$) | Parameter Priority |
|---|---|---|---|---|---|---|---|
| **pH** | Physical | `ph` / `pH` | pH scale ($0-14$) | $6.5 - 8.5$ ($S_i = 8.5$) | $7.0$ | $0.1176$ | Mandatory |
| **Dissolved Oxygen (DO)** | Biological / Phys | `dissolved_oxygen` / `do_mg_l` | $\text{mg/L}$ | $\ge 5.0$ ($S_i = 5.0$) | $14.6$ | $0.2000$ | Mandatory |
| **Biochemical Oxygen Demand (BOD)** | Biological | `bod` / `bod_3day_27c` | $\text{mg/L}$ | $\le 3.0$ ($S_i = 3.0$) | $0.0$ | $0.3333$ | Mandatory |
| **Chemical Oxygen Demand (COD)** | Biological | `cod` / `cod_mg_l` | $\text{mg/L}$ | $\le 10.0$ ($S_i = 10.0$) | $0.0$ | $0.1000$ | High |
| **Electrical Conductivity (EC)** | Physical | `conductivity` / `ec_us_cm` | $\mu\text{S/cm}$ | $\le 750.0$ ($S_i = 750$) | $0.0$ | $0.0013$ | Standard |
| **Total Dissolved Solids (TDS)** | Physical | `solids` / `tds_mg_l` | $\text{mg/L}$ / ppm | $\le 500.0$ ($S_i = 500$) | $0.0$ | $0.0020$ | Standard |
| **Turbidity** | Physical | `turbidity` / `turbidity_ntu` | NTU | $\le 5.0$ ($S_i = 5.0$) | $0.0$ | $0.2000$ | High |
| **Total Coliform (TC)** | Biological | `total_coliform` / `tc_mpn_100ml`| MPN/100mL | $\le 500.0$ ($S_i = 500$) | $0.0$ | $0.0020$ | Critical |
| **Faecal Coliform (FC)** | Biological | `faecal_coliform` / `fc_mpn_100ml`| MPN/100mL | $\le 250.0$ ($S_i = 250$) | $0.0$ | $0.0040$ | Critical |
| **Nitrate ($NO_3^-$)** | Chemical | `nitrate` / `nitrate_mg_l` | $\text{mg/L}$ | $\le 45.0$ ($S_i = 45.0$) | $0.0$ | $0.0222$ | Standard |
| **Total Hardness ($CaCO_3$)** | Chemical | `hardness` / `hardness_mg_l` | $\text{mg/L}$ | $\le 200.0$ ($S_i = 200.0$) | $0.0$ | $0.0050$ | Standard |
| **Chloride ($Cl^-$)** | Chemical | `chloride` / `chloride_mg_l` | $\text{mg/L}$ | $\le 250.0$ ($S_i = 250.0$) | $0.0$ | $0.0040$ | Standard |
| **Sulfate ($SO_4^{2-}$)** | Chemical | `sulfate` / `sulfate_mg_l` | $\text{mg/L}$ | $\le 200.0$ ($S_i = 200.0$) | $0.0$ | $0.0050$ | Standard |

---

## 4. Water Quality Index (WQI) Class Boundaries & Target Mapping

### 4.1 CPCB & Brown et al. Categorization

| Numeric WQI Range | Water Quality Classification | CPCB Designated Best Use Category | Description / Intended Water Use |
|---|---|---|---|
| **$0 \le \text{WQI} \le 25$** | **Excellent** | **Class A** | Drinking water source without conventional treatment but after disinfection. |
| **$26 \le \text{WQI} \le 50$** | **Good** | **Class B** | Outdoor bathing, swimming, recreation; potable with conventional treatment. |
| **$51 \le \text{WQI} \le 75$** | **Poor / Moderate** | **Class C** | Drinking water source after conventional treatment and disinfection. |
| **$76 \le \text{WQI} \le 100$** | **Very Poor** | **Class D** | Propagation of wild life and fisheries. |
| **$\text{WQI} > 100$** | **Unsuitable for Drinking** | **Class E** | Irrigation, industrial cooling, controlled waste disposal; heavily contaminated. |

---

### 4.2 Binary Classification Target Mapping

For the machine learning classification track, the multi-class categorization is mapped into a canonical binary potability target:

$$\text{Potability} = \begin{cases} 1 & \text{if } WQI \le 50.0 \quad (\text{Potable / Excellent or Good}) \\ 0 & \text{if } WQI > 50.0 \quad (\text{Not Potable / Poor, Very Poor, or Unsuitable}) \end{cases}$$

---

## 5. Temporal Granularity Mismatch & Missingness Fallback Policy

### 5.1 The CPCB Granularity Challenge
In CPCB surface water data:
- **Physical & Biological monitoring**: Recorded at **monthly** intervals.
- **Chemical monitoring**: Recorded at **annual** or **seasonal** intervals.

### 5.2 Dynamic Weight Redistribution Rule
When computing monthly WQI instances where annual chemical parameters (e.g., Hardness, Sulfate, Chloride) are absent:
1. The missing parameter $j$ is excluded from the active set for that specific sample $k$:
   $$S_{\text{active}}^{(k)} = \{ i \mid V_i^{(k)} \text{ is not null} \}$$
2. The active weights are renormalized:
   $$w_i^{*(k)} = \frac{w_i}{\sum_{j \in S_{\text{active}}^{(k)}} w_j}$$
3. The dynamic $WQI$ is computed strictly over $S_{\text{active}}^{(k)}$:
   $$WQI^{(k)} = \sum_{i \in S_{\text{active}}^{(k)}} w_i^{*(k)} q_i^{(k)}$$
4. **Minimum Parameter Threshold Constraint**: A valid WQI score is only calculated if at least the core 4 parameters ($\text{pH}$, $\text{DO}$, $\text{BOD}$, and $\text{Turbidity}$ or $\text{TDS}$) are present. Records with fewer than 4 valid core parameters are rejected into `data/interim/quarantine.csv`.

---

## 6. Data Leakage Prevention Protocol (Mandatory ML Rule)

> [!CAUTION]
> **STRICT ML LEAKAGE GUARD (Milestone 1.4, 1.5, 4.2 & 6.3)**:
> The intermediate continuous calculation `wqi_score` is used **exclusively** to generate the target labels (`wqi_class` and `Potability`).
> `wqi_score` **MUST NEVER** be included as an input feature in any dataset passed to machine learning models (`X_train`, `X_test`).
> Including `wqi_score` would cause trivial 100% artificial accuracy and catastrophic test leakage.

---

## 7. Citations & References

1. **Central Pollution Control Board (CPCB)** (2017). *Guidelines for Water Quality Management*. Ministry of Environment, Forest and Climate Change, New Delhi, India.
2. **Bureau of Indian Standards (BIS)** (2012). *Indian Standard IS 10500:2012: Drinking Water — Specification (Second Revision)*. Manak Bhavan, New Delhi.
3. **Brown, R. M., McClelland, N. I., Deininger, R. A., & Tozer, R. G.** (1970). *A water quality index - do we dare?* Water & Sewage Works, 117(10), 339-343.
4. **Canadian Council of Ministers of the Environment (CCME)** (2001, updated 2017). *Canadian Water Quality Index 1.0, Technical Report and User's Manual*.
5. **Tyagi, S., Sharma, B., Singh, P., & Dobhal, R.** (2013). *Water quality assessment in terms of water quality index*. American Journal of Water Resources, 1(3), 34-38.
