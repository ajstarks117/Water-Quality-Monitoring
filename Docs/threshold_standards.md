# Rule-Engine Threshold Standards & Regulatory Mapping

**Document ID**: `STD-THRESH-CPCB-2026-V1`  
**Milestone**: Milestone 8.1: Rule-Engine Threshold Mapping (Extends Milestone 1.1)  
**Track**: Track C: Explainability & Decision Support  
**Owner**: Explainability & Decision Support Lead (Person C)  
**Status**: **FROZEN & RATIFIED** (Authoritative threshold specification for Milestone 8.2 Rule Engine, Milestone 8.3 SHAP-to-Rules Integration, and Phase 9 Streamlit Dashboard)  
**Date of Ratification**: 2026-10-18  

---

## 1. Executive Summary & Objective

This document establishes the official regulatory thresholds, normal ranges, abnormal boundaries, edge-case evaluation semantics, and action priority classifications for the **Decision-Support Rule Engine** (Specification Sections 33, 34, 35, and 48).

Per Specification Section 33, **all thresholds must trace directly to established environmental standards—no hard-coded or arbitrary values are permitted**. To preserve complete methodological consistency across the project lifecycle and avoid confusing reviewers with dual standards:
1. **Shared Parameters**: Directly reuse the official Central Pollution Control Board (CPCB) and Bureau of Indian Standards (BIS IS 10500:2012) standards established in [`Docs/wqi_standard.md`](file:///d:/Coding/College/Data%20science/CP/Docs/wqi_standard.md) (Milestone 1.1).
2. **Feature-Engineered Parameters**: Directly project the standard from raw inputs using exact mathematical transformations (Milestone 4.1).
3. **Extension Parameters**: Provide named, dated citations for physical and chemical parameters present in CPCB water monitoring records and Specification Section 34 that were not included in the 4-feature frozen ML candidate set.
4. **Fallback Protocol**: Explicitly define and document an empirical statistical fallback ($P_{10}$ / $P_{90}$ training distribution) for any candidate parameter lacking a clean regulatory cutoff, strictly labeled as a statistical fallback rather than passed off as a statutory standard.

---

## 2. Authoritative Source Standards & Citations

All thresholds in this document derive from the following published statutory and regulatory standards:

1. **Central Pollution Control Board (CPCB) Guidelines (2008 & 2017)**:
   - *Guidelines for Water Quality Management* (2008). Ministry of Environment, Forest and Climate Change, Govt. of India.
   - *Designated Best Use Water Quality Criteria* (Class A, B, C, D, E classifications), CPCB, New Delhi.
2. **Bureau of Indian Standards (BIS)**:
   - `IS 10500:2012`: *Drinking Water — Specification (Second Revision)*. Table 1 (Organoleptic and Physical Parameters) and Table 2 (General Chemical & Toxic Substances). Manak Bhavan, New Delhi.
3. **Project WQI Standard (Milestone 1.1)**:
   - [`Docs/wqi_standard.md`](file:///d:/Coding/College/Data%20science/CP/Docs/wqi_standard.md) (`STD-WQI-CPCB-2026-V1`): Official mathematical formulation and parameter limits ($S_i$, $V_0$).
4. **World Health Organization (WHO)**:
   - *Guidelines for Drinking-water Quality: Fourth Edition Incorporating the First Addendum* (2017). World Health Organization, Geneva.
5. **United States Environmental Protection Agency (USEPA)**:
   - *National Recommended Water Quality Criteria for Aquatic Life and Human Health* (2002, updated 2016). EPA-822-R-02-047.

---

## 3. Boundary Condition Semantics & Metrological Conventions (Spec Section 55)

To eliminate boundary condition ambiguity as mandated under Specification Section 55 (*Important Edge Cases*):

1. **Upper Permissible Limit ($S_i$) Parameters** (BOD, Fecal Coliform, Turbidity, TDS, Conductivity, COD, Nitrate, Ammonia):
   - A measured value **strictly less than or equal to** the standard limit ($V \le S_i$) is classified as **NORMAL / PERMISSIBLE**.
   - A measured value **strictly greater than** the standard limit ($V > S_i$) is classified as **ABNORMAL / VIOLATION**.
   - *Legal & Metrological Rationale*: In statutory compliance monitoring, meeting the ceiling exactly satisfies the permissible standard.

2. **Lower Required Limit ($S_i$) Parameters** (Dissolved Oxygen):
   - A measured value **strictly greater than or equal to** the standard limit ($V \ge S_i$) is classified as **NORMAL / SUFFICIENT**.
   - A measured value **strictly less than** the standard limit ($V < S_i$) is classified as **ABNORMAL / OXYGEN DEPLETED**.
   - *Biological Rationale*: Dissolved oxygen must be maintained at or above the minimum required threshold to support aquatic respiration.

3. **Range-Bounded Parameters** (pH):
   - A measured value within the closed interval ($6.5 \le V \le 8.5$) is classified as **NORMAL / BALANCED**.
   - $V < 6.5$ is classified as **ABNORMAL LOW (ACIDIC)**.
   - $V > 8.5$ is classified as **ABNORMAL HIGH (ALKALINE)**.

---

## 4. Primary Rule-Engine Parameters (Frozen ML Feature Set & M4.1 Pipeline)

The parameters below drive the core prediction pipeline, the local SHAP explanation module, and the primary automated decision-support rules.

### 4.1 Potential of Hydrogen (pH)
- **Dataset Column Name**: `Potential of Hydrogen (pH)`
- **Dataset Unit**: `pH scale` ($0 - 14$)
- **Normal / Permissible Range**: $[6.5, 8.5]$ (Closed interval)
- **Abnormal Conditions**:
  - `Abnormal Low (Acidic)`: $\text{pH} < 6.5$
  - `Abnormal High (Alkaline)`: $\text{pH} > 8.5$
  - `Critical Hazard (Extreme Acidity)`: $\text{pH} < 4.0$ (Severe corrosive/industrial discharge hazard)
  - `Critical Hazard (Extreme Alkalinity)`: $\text{pH} > 9.5$ (Severe caustic/industrial discharge hazard)
- **Primary Source Citation**: CPCB Designated Best Use (Class A, B, C); BIS `IS 10500:2012` Table 1, Item 1; [`Docs/wqi_standard.md`](file:///d:/Coding/College/Data%20science/CP/Docs/wqi_standard.md) Section 3.
- **Interpretation & Domain Rationale**: Disruption of acid-base equilibrium. Acidic runoff accelerates heavy metal leaching; high alkalinity impairs chlorination efficiency and precipitates carbonates.
- **Shared Parameter Consistency**: Identical to M1.1 ($S_i = 8.5$, acceptable range $6.5 - 8.5$, ideal $V_0 = 7.0$).

### 4.2 Dissolved Oxygen (DO)
- **Dataset Column Name**: `Dissolved oxygen (mg/L)`
- **Dataset Unit**: `mg/L`
- **Normal / Permissible Range**: $\ge 5.0\text{ mg/L}$
- **Abnormal Conditions**:
  - `Abnormal Low (Oxygen Depleted)`: $\text{DO} < 5.0\text{ mg/L}$ (Fails CPCB Class B bathing and Class C drinking criteria)
  - `Critical Hypoxia (Severe Ecological Stress)`: $\text{DO} < 4.0\text{ mg/L}$ (Fails CPCB Class D wildlife propagation standard)
  - `Severe Anoxia (Septic Conditions)`: $\text{DO} < 2.0\text{ mg/L}$ (Acute aquatic mortality risk; anaerobic decomposition dominant)
  - `Abnormal High (Supersaturation)`: $\text{DO} > 14.6\text{ mg/L}$ (Exceeds physical saturation at $0^\circ\text{C}$; indicates excessive algal photosynthesis / eutrophic bloom)
- **Primary Source Citation**: CPCB Designated Best Use (Class B / C); BIS `IS 10500:2012`; [`Docs/wqi_standard.md`](file:///d:/Coding/College/Data%20science/CP/Docs/wqi_standard.md) Section 3.
- **Interpretation & Domain Rationale**: Indicator of aquatic respiratory viability and organic loading capacity. Oxygen depletion accompanies biodegradation of untreated domestic and agricultural wastes.
- **Shared Parameter Consistency**: Identical to M1.1 ($S_i = 5.0\text{ mg/L}$, ideal saturation $V_0 = 14.6\text{ mg/L}$).

### 4.3 Biochemical Oxygen Demand (BOD)
- **Dataset Column Name**: `Biochemical Oxygen Demand (mg/L)`
- **Dataset Unit**: `mg/L` (3-day incubation at 27ºC)
- **Normal / Permissible Range**: $\le 3.0\text{ mg/L}$
- **Abnormal Conditions**:
  - `Abnormal High (Organic Pollution)`: $\text{BOD} > 3.0\text{ mg/L}$ (Exceeds CPCB Class B bathing standard)
  - `Critical Organic Contamination`: $\text{BOD} > 6.0\text{ mg/L}$ (Heavy organic sewage load; severe aquatic ecosystem disruption)
  - `Gross Industrial / Sewage Discharge`: $\text{BOD} > 30.0\text{ mg/L}$ (Typical raw sewage effluent regime)
- **Primary Source Citation**: CPCB Designated Best Use Class B (2008 & 2017); [`Docs/wqi_standard.md`](file:///d:/Coding/College/Data%20science/CP/Docs/wqi_standard.md) Section 3.
- **Interpretation & Domain Rationale**: Quantitative measure of oxygen required by aerobic microorganisms to stabilize biodegradable organic matter.
- **Shared Parameter Consistency**: Identical to M1.1 ($S_i = 3.0\text{ mg/L}$, ideal $V_0 = 0.0\text{ mg/L}$).

### 4.4 Fecal Coliform (FC)
- **Dataset Column Name**: `Fecal Coliform (MPN/100mL)`
- **Dataset Unit**: `MPN/100mL`
- **Normal / Permissible Range**: $\le 250.0\text{ MPN/100mL}$ (CPCB Class B Bathing Desirable Standard)
- **Abnormal Conditions**:
  - `Abnormal High (Pathogenic Contamination)`: $\text{FC} > 250.0\text{ MPN/100mL}$ (Exceeds Class B bathing criterion)
  - `Critical Pathogen Alert`: $\text{FC} > 2500.0\text{ MPN/100mL}$ (Exceeds CPCB Class C maximum permissible limit for drinking water source after conventional treatment)
  - `Severe Biohazard`: $\text{FC} > 50000.0\text{ MPN/100mL}$ (Gross untreated municipal sewer outfall)
- **Primary Source Citation**: CPCB Guidelines for Water Quality Management (2008/2017); [`Docs/wqi_standard.md`](file:///d:/Coding/College/Data%20science/CP/Docs/wqi_standard.md) Section 3.
- **Interpretation & Domain Rationale**: Primary microbiological indicator of mammalian enteric contamination and waterborne pathogens (Salmonella, Shigella, enteroviruses).
- **Shared Parameter Consistency**: Identical to M1.1 ($S_i = 250.0\text{ MPN/100mL}$, ideal $V_0 = 0.0\text{ MPN/100mL}$).

### 4.5 Log10 Fecal Coliform (Engineered Feature)
- **Dataset / Feature Name**: `Log10 Fecal Coliform`
- **Dataset Unit**: $\log_{10}(\text{MPN/100mL})$
- **Mathematical Transformation**: $\log_{10}(\max(\text{FC}, 1.0))$ (Milestone 4.1 Feature Engineering)
- **Normal / Permissible Range**: $\le \log_{10}(250.0) \approx 2.39794$ (Rounded: $\le 2.398$)
- **Abnormal Conditions**:
  - `Abnormal High`: $> 2.398$ ($\sim \text{FC} > 250\text{ MPN/100mL}$)
  - `Critical Pathogen Alert`: $> \log_{10}(2500.0) \approx 3.398$ ($\sim \text{FC} > 2500\text{ MPN/100mL}$)
- **Primary Source Citation**: Derived directly from CPCB $\text{FC} \le 250\text{ MPN/100mL}$ standard via monotonic $\log_{10}$ bijection; documented in [`Docs/interface_contracts.md`](file:///d:/Coding/College/Data%20science/CP/Docs/interface_contracts.md) Section 1.5.
- **Interpretation & Domain Rationale**: Provides linear representation of microbial risk across orders of magnitude without altering underlying statutory semantics.

---

## 5. Extension Parameters (Secondary & Diagnostic Parameters)

These parameters are recorded in CPCB monitoring and evaluated in Specification Section 34, extending coverage beyond the frozen 4-parameter ML input vector.

| Parameter Name | Exact Dataset Column Name | Dataset Unit | Normal / Permissible Limit | Abnormal Violation Condition | Critical Alert Level | Primary Statutory Citation |
|:---|:---|:---|:---|:---|:---|:---|
| **Turbidity** | `Turbidity (NTU)` | `NTU` | $\le 5.0\text{ NTU}$ | $> 5.0\text{ NTU}$ | $> 10.0\text{ NTU}$ | BIS `IS 10500:2012` Table 1, Item 3; [`Docs/wqi_standard.md`](file:///d:/Coding/College/Data%20science/CP/Docs/wqi_standard.md) |
| **Electrical Conductivity (EC)** | `Electric Conductivity (μS/cm)` | `μS/cm` | $\le 750.0\ \mu\text{S/cm}$ | $> 750.0\ \mu\text{S/cm}$ | $> 2250.0\ \mu\text{S/cm}$ | CPCB (2008) Class B; [`Docs/wqi_standard.md`](file:///d:/Coding/College/Data%20science/CP/Docs/wqi_standard.md) |
| **Total Dissolved Solids (TDS)** | `Total Dissolved Solids (mg/L)` | `mg/L` | $\le 500.0\text{ mg/L}$ | $> 500.0\text{ mg/L}$ | $> 2000.0\text{ mg/L}$ | BIS `IS 10500:2012` Table 1, Item 4; [`Docs/wqi_standard.md`](file:///d:/Coding/College/Data%20science/CP/Docs/wqi_standard.md) |
| **Chemical Oxygen Demand (COD)** | `Chemical Oxygen Demand (mg/L)` | `mg/L` | $\le 10.0\text{ mg/L}$ | $> 10.0\text{ mg/L}$ | $> 25.0\text{ mg/L}$ | CPCB Guidelines (2008); WHO (2011); [`Docs/wqi_standard.md`](file:///d:/Coding/College/Data%20science/CP/Docs/wqi_standard.md) |
| **Total Coliform (TC)** | `Total Coliform (MPN/100mL)` | `MPN/100mL` | $\le 500.0\text{ MPN/100mL}$ | $> 500.0\text{ MPN/100mL}$ | $> 5000.0\text{ MPN/100mL}$ | CPCB Class B / CPCB Class C; [`Docs/wqi_standard.md`](file:///d:/Coding/College/Data%20science/CP/Docs/wqi_standard.md) |
| **Nitrate Nitrogen** | `Nitrate N (mgN/L)` | `mgN/L` | $\le 10.0\text{ mgN/L}$ | $> 10.0\text{ mgN/L}$ | $> 20.0\text{ mgN/L}$ | BIS `IS 10500:2012` Table 2, Item 6 ($45.0\text{ mg/L}$ as $NO_3^-$); WHO (2017) |
| **Total Hardness** | `Total Hardness (mgCaCO3/L)` | `mgCaCO3/L` | $\le 200.0\text{ mg/L}$ | $> 200.0\text{ mg/L}$ | $> 600.0\text{ mg/L}$ | BIS `IS 10500:2012` Table 2, Item 2; [`Docs/wqi_standard.md`](file:///d:/Coding/College/Data%20science/CP/Docs/wqi_standard.md) |
| **Chloride** | `Chloride (mg/L)` | `mg/L` | $\le 250.0\text{ mg/L}$ | $> 250.0\text{ mg/L}$ | $> 1000.0\text{ mg/L}$ | BIS `IS 10500:2012` Table 2, Item 5; [`Docs/wqi_standard.md`](file:///d:/Coding/College/Data%20science/CP/Docs/wqi_standard.md) |
| **Sulphate** | `Sulphate (mg/L)` | `mg/L` | $\le 200.0\text{ mg/L}$ | $> 200.0\text{ mg/L}$ | $> 400.0\text{ mg/L}$ | BIS `IS 10500:2012` Table 2, Item 4; [`Docs/wqi_standard.md`](file:///d:/Coding/College/Data%20science/CP/Docs/wqi_standard.md) |
| **Ammonia Nitrogen** | `Amonia N (mgN/L)` | `mgN/L` | $\le 1.2\text{ mgN/L}$ | $> 1.2\text{ mgN/L}$ | $> 5.0\text{ mgN/L}$ | CPCB Class C Criteria (1.2 mg/L free ammonia); BIS `IS 10500:2012` |
| **Water Temperature** | `Temperature (ºC)` | `ºC` | $15.0 - 32.0^\circ\text{C}$ | $< 15.0$ or $> 32.0^\circ\text{C}$ | $> 40.0^\circ\text{C}$ | CPCB Environmental Standards (Effluent thermal limit $\Delta T \le 3^\circ\text{C}$) |

---

## 6. Documented Fallback Policy for Unstandardized Parameters

In accordance with Milestone 8.1 task requirements, where a specialized parameter recorded in the field has no clean statutory standard cutoff:

1. **Statistical Percentile-Based Baseline**:
   - The threshold defaults to the empirical **90th percentile ($P_{90}$)** (for contaminant loads) or **10th percentile ($P_{10}$)** (for depletion metrics) calculated strictly on the training partition ($X_{\text{train}}$).
2. **Explicit Non-Regulatory Labeling**:
   - The rule engine output must label any triggered recommendation originating from this fallback as:
     > `Standard Type: Empirical Statistical Fallback (P90 of Training Baseline)`
   - Under no circumstances shall an empirical fallback be passed off as a statutory or regulatory limit.

---

## 7. Action Priority Classification Architecture (Spec Section 35)

The Decision-Support Rule Engine assigns an actionable priority level to each triggered finding:

```
[Triggered Parameter Rules & Model Prediction]
                     │
       ┌─────────────┴─────────────┐
       ▼                           ▼
[Severe Violation /         [Single Non-Critical
Critical Hazard OR           Violation OR Moderate
Poor Class + Multi-Violations] Deviation]
       │                           │
       ▼                           ▼
 ┌───────────┐               ┌───────────┐               ┌───────────┐
 │   HIGH    │               │  MEDIUM   │               │    LOW    │
 │ PRIORITY  │               │ PRIORITY  │               │ PRIORITY  │
 └───────────┘               └───────────┘               └───────────┘
```

### 7.1 High Priority
A finding is classified as **High Priority** if any of the following conditions are met:
1. **Critical Microbiological Contamination**: $\text{Fecal Coliform} > 2500\text{ MPN/100mL}$ or $\text{Total Coliform} > 5000\text{ MPN/100mL}$.
2. **Acute Respiratory Depletion**: $\text{Dissolved Oxygen} < 4.0\text{ mg/L}$ (hypoxic stress) or $< 2.0\text{ mg/L}$ (anoxic state).
3. **Severe Organic Pollution**: $\text{Biochemical Oxygen Demand} > 6.0\text{ mg/L}$.
4. **Extreme Chemical Extremes**: $\text{pH} < 4.0$ or $\text{pH} > 9.5$.
5. **Compound Multi-Parameter Stress**: Water quality predicted as `Poor`, `Very Poor`, or `Unsuitable` **AND** $\ge 2$ distinct parameters abnormal simultaneously.

### 7.2 Medium Priority
A finding is classified as **Medium Priority** if:
1. Exactly one primary parameter violates permissible standards without reaching critical alert thresholds (e.g., $3.0 < \text{BOD} \le 6.0\text{ mg/L}$, $250 < \text{FC} \le 2500\text{ MPN/100mL}$, or $4.0 \le \text{DO} < 5.0\text{ mg/L}$).
2. Moderate pH imbalance ($6.0 \le \text{pH} < 6.5$ or $8.5 < \text{pH} \le 9.0$).
3. Secondary physical/chemical parameter violation (e.g., Turbidity $> 5\text{ NTU}$, TDS $> 500\text{ mg/L}$).

### 7.3 Low Priority
A finding is classified as **Low Priority** if:
1. All primary biological and chemical parameters are strictly normal.
2. Minor marginal deviation in physical indicator (e.g., slight turbidity elevation during seasonal monsoon runoff).
3. General preventive monitoring recommendations or routine sensor recalibration advisories.

---

## 8. Causation-Safe Interpretation Language Protocol (Spec Sections 31 & 48)

To enforce strict compliance with regulatory reporting standards and academic rigor:

> [!CAUTION]
> **MANDATORY WORDING DISCIPLINE**:
> Decision-support rules provide **diagnostic interpretations and recommended investigations**, NOT proof of environmental causation or culpability.
> 
> - **Permitted Phrasing**: `"is associated with"`, `"indicates potential presence of"`, `"suggests possible runoff/discharge"`, `"recommended investigative action"`.
> - **Forbidden Phrasing**: `"caused by"`, `"proven source"`, `"culprit"`, `"guilty factory"`, `"proves contamination origin"`.

---

## 9. Comprehensive Parameter-to-Rule Specification Table

The following table serves as the direct programmatic contract for `config/thresholds.yaml` and `src/decision_support/rules.py` in Milestone 8.2:

| Parameter Key | Evaluated Condition | Interpretation Message | Possible Contributing Factors | Recommended Corrective / Preventive Actions | Default Priority |
|:---|:---|:---|:---|:---|:---:|
| `pH_low` | $\text{pH} < 6.5$ | Acidic water condition below permissible range | Industrial acid discharge, acid mine drainage, organic humic runoff | Investigate upstream discharge points; inspect neutralization units; verify probe calibration before physical intervention | Medium (High if $<4.0$) |
| `pH_high` | $\text{pH} > 8.5$ | Alkaline water condition above permissible range | Industrial alkaline effluent, agricultural limestone runoff, excessive algal photosynthetic activity | Investigate industrial effluents; evaluate algal bloom density; review alkaline neutralization protocols | Medium (High if $>9.5$) |
| `DO_low` | $\text{DO} < 5.0\text{ mg/L}$ | Dissolved oxygen depletion below regulatory requirement | High organic/biodegradable waste loading, stagnant water, elevated thermal discharge | Investigate organic waste discharge; measure BOD/COD levels; consider artificial aeration in stagnant reaches | Medium (High if $<4.0$) |
| `BOD_high` | $\text{BOD} > 3.0\text{ mg/L}$ | Elevated biochemical oxygen demand indicating high organic load | Untreated or partially treated domestic sewage, dairy/food processing effluent | Audit nearby municipal wastewater treatment facilities; trace sewer outfalls; monitor downstream oxygen sag | Medium (High if $>6.0$) |
| `FC_high` | $\text{FC} > 250.0\text{ MPN/100mL}$ | Fecal coliform elevation indicating pathogenic bacterial contamination | Direct sewage discharge, septic tank leakage, open defecation runoff, livestock runoff | Issue drinking/bathing advisory; trace human sewage sources; audit disinfection and chlorination processes | High |
| `Log10_FC_high` | $\log_{10}(\text{FC}) > 2.398$ | Logarithmic fecal coliform index exceeds acceptable regulatory ceiling | Microbial contamination across multiple orders of magnitude | Execute microbiological sanitary survey; review public health warnings | High |
| `Turbidity_high` | $\text{Turbidity} > 5.0\text{ NTU}$ | Elevated suspended particulate material impairing water clarity | Surface sediment runoff, soil erosion, dredging, construction discharge | Inspect riverbank erosion control; review coagulation/filtration performance in water treatment plants | Low (Medium if $>10.0$) |
| `EC_high` | $\text{EC} > 750.0\ \mu\text{S/cm}$ | High electrical conductivity indicating elevated dissolved ions | Saline agricultural drainage, industrial brine discharge, mineral dissolution | Investigate upstream irrigation returns and industrial effluent TDS/EC ratios | Low (Medium if $>2250$) |
| `TDS_high` | $\text{TDS} > 500.0\text{ mg/L}$ | Elevated total dissolved mineral/solid concentration | Mineral weathering, industrial saline effluent, wastewater discharge | Check dissolved ion sources; verify membrane or reverse osmosis treatment efficacy | Low (Medium if $>2000$) |
| `COD_high` | $\text{COD} > 10.0\text{ mg/L}$ | Elevated chemical oxygen demand indicating non-biodegradable or industrial load | Industrial chemical effluent, refractory organic contaminants, pesticide runoff | Initiate industrial effluent compliance audit; verify advanced oxidation or tertiary treatment | Medium (High if $>25.0$) |
| `TC_high` | $\text{TC} > 500.0\text{ MPN/100mL}$ | Elevated total coliform bacterial count | Soil bacterial wash-in, decaying plant matter, sewage contamination | Correlate with Fecal Coliform to distinguish natural soil microbes from enteric human sewage | Medium (High if $>5000$) |
| `Nitrate_high` | $\text{Nitrate N} > 10.0\text{ mgN/L}$ | Elevated nitrate concentration presenting methemoglobinemia risk | Excessive agricultural nitrogen fertilizer runoff, sewage effluent | Investigate seasonal fertilizer applications; issue infant drinking water warning; trace agricultural drains | High |

---

## 10. Verification & Cross-Document Consistency Matrix

| Standard Parameter | M1.1 (`wqi_standard.md`) Limit | M8.1 (`threshold_standards.md`) Limit | Match Status | Justification for Any Difference |
|:---|:---:|:---:|:---:|:---|
| **pH** | $6.5 - 8.5$ ($S_i = 8.5$) | $6.5 - 8.5$ | **IDENTICAL** | Exact CPCB / BIS standard reused |
| **Dissolved Oxygen** | $\ge 5.0\text{ mg/L}$ ($S_i = 5.0$) | $\ge 5.0\text{ mg/L}$ | **IDENTICAL** | Exact CPCB Class B standard reused |
| **BOD** | $\le 3.0\text{ mg/L}$ ($S_i = 3.0$) | $\le 3.0\text{ mg/L}$ | **IDENTICAL** | Exact CPCB Class B standard reused |
| **Fecal Coliform** | $\le 250.0\text{ MPN/100mL}$ ($S_i = 250.0$) | $\le 250.0\text{ MPN/100mL}$ | **IDENTICAL** | Exact CPCB Class B standard reused |
| **Log10 Fecal Coliform** | N/A (M4.1 engineering) | $\le 2.398$ | **CONSISTENT** | $\log_{10}(250) = 2.39794$ (Exact mathematical bijection) |
| **Turbidity** | $\le 5.0\text{ NTU}$ ($S_i = 5.0$) | $\le 5.0\text{ NTU}$ | **IDENTICAL** | Exact BIS IS 10500:2012 limit reused |
| **Conductivity** | $\le 750.0\ \mu\text{S/cm}$ ($S_i = 750.0$) | $\le 750.0\ \mu\text{S/cm}$ | **IDENTICAL** | Exact CPCB Class B limit reused |
| **TDS** | $\le 500.0\text{ mg/L}$ ($S_i = 500.0$) | $\le 500.0\text{ mg/L}$ | **IDENTICAL** | Exact BIS IS 10500:2012 limit reused |
| **COD** | $\le 10.0\text{ mg/L}$ ($S_i = 10.0$) | $\le 10.0\text{ mg/L}$ | **IDENTICAL** | Exact CPCB limit reused |
| **Total Coliform** | $\le 500.0\text{ MPN/100mL}$ ($S_i = 500.0$) | $\le 500.0\text{ MPN/100mL}$ | **IDENTICAL** | Exact CPCB Class B limit reused |
| **Nitrate** | $\le 45.0\text{ mg/L}$ as $NO_3^-$ | $\le 10.0\text{ mgN/L}$ as $N$ | **EQUIVALENT** | Exact stoichiometric conversion ($45.0 \times \frac{14.01}{62.00} \approx 10.16\text{ mgN/L}$) |
| **Total Hardness** | $\le 200.0\text{ mg/L}$ ($S_i = 200.0$) | $\le 200.0\text{ mg/L}$ | **IDENTICAL** | Exact BIS IS 10500:2012 limit reused |
| **Chloride** | $\le 250.0\text{ mg/L}$ ($S_i = 250.0$) | $\le 250.0\text{ mg/L}$ | **IDENTICAL** | Exact BIS IS 10500:2012 limit reused |
| **Sulphate** | $\le 200.0\text{ mg/L}$ ($S_i = 200.0$) | $\le 200.0\text{ mg/L}$ | **IDENTICAL** | Exact BIS IS 10500:2012 limit reused |

---

## 11. Sign-off & Milestone Completion

| Role | Reviewer | Approval Status | Notes |
|:---|:---|:---:|:---|
| **Track C: Explainability & Decision Support** | Person C (Lead) | **APPROVED** | Authoritative threshold mapping finalized for M8.2. |
| **Track A: Data Pipeline** | Person A (Data Lead) | **RATIFIED** | Confirmed zero unexplained drift from M1.1 & M1.4. |
| **Track B: ML Modeling** | Person B (ML Lead) | **ACKNOWLEDGED** | Confirmed consistency with M4.1 feature engineering. |
| **Track D: Dashboard & Integration** | Person D (Dashboard Lead)| **READY** | Ready for configuration export to `config/thresholds.yaml`. |
