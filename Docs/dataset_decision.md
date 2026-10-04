# CPCB Dataset Decision & Exact-Timestamp Join Strategy

**Document ID**: `DEC-DATA-CPCB-2026-V3`  
**Milestone**: Milestone 1.2  
**Owner**: Data Lead (Person A)  
**Status**: Complete — Join Executed & Verified  
**Last Updated**: 2026-10-04

---

## 1. Official Data Sourcing Policy

> [!IMPORTANT]
> **Data Integrity Policy**:
> All analytical, labeling, training, and evaluation pipelines in this project operate **exclusively on authentic, official CPCB/NWDP datasets**.
> Synthetic or simulated datasets are strictly excluded from the project pipeline.

---

## 2. Target States & Monitoring Scope

- **Target States**: **Maharashtra** and **Uttar Pradesh** (highest station counts covering major river basins including Ganga, Yamuna, Godavari, Krishna, Tapi, Bhima, Panchganga).
- **Target Source**: **National Water Data Portal (NWDP)** (`nwdp.nwic.gov.in`) / CPCB Surface Water Quality Monitoring Datasets.
- **Vintage**: **2021 – 2025** official CPCB monitoring records.

---

## 3. Required Official CSV Files in `data/raw/`

| # | Category   | File Pattern                         | Key Parameters |
|---|------------|--------------------------------------|---------------|
| 1 | Physical   | `*physical*.csv` (e.g. `maharashtra_physical.csv.csv`) | Temperature, Turbidity, EC, TDS, Colour, Odour, Total Solids |
| 2 | Biological | `*biological*.csv` (e.g. `maharashtra_biological.csv.csv`) | BOD, COD, DO, Fecal Coliform, Total Coliform |
| 3 | Chemical   | `*chemical*.csv` (e.g. `maharashtra_chemical.csv.csv`) | pH, Chloride, Hardness, Nitrate, Fluoride, heavy metals, ions |

Six files total (3 categories × 2 states).

---

## 4. Why We Changed the Original Join Assumption

The original M1.2 specification assumed:

- Physical → monthly
- Biological → monthly
- Chemical → annual

and suggested joining Physical + Biological by station/month and broadcasting annual Chemical values by station/year.

**After inspecting the actual official CPCB/NWDP datasets**, this simplified description was found to be inaccurate:

### Finding 1: Chemical is NOT uniformly annual

- pH and Dissolved Oxygen have high-frequency (monthly) observations in the Chemical CSV.
- Several mineral/heavy-metal parameters are sparse or irregular.
- Some chemical columns are completely null.
- **Consequence**: Treating the entire Chemical CSV as annual and broadcasting every value across 12 months would fabricate observations that were never actually collected.

### Finding 2: Uttar Pradesh contains legitimate bi-weekly observations

- Some stations have two observations in the same calendar month.
- Joining on `Station + Year + Month` creates a many-to-many Cartesian expansion (2 physical × 2 biological = 4 rows instead of 2).
- **Consequence**: The exact `Data Acquisition Time` is required to identify the actual monitoring event and prevent artificial duplication.

### Finding 3: Biological ↔ Chemical have exact timestamp correspondence

- Maharashtra: 1,917 / 1,917 biological records match Chemical exactly.
- Uttar Pradesh: 1,414 / 1,414 biological records match Chemical exactly.
- **Consequence**: The Bio+Chem datasets can be reliably joined on exact timestamps with an INNER JOIN producing zero data loss.

### Finding 4: Physical coverage is incomplete

- Maharashtra: 803 / 1,917 exact physical matches (~42%).
- Uttar Pradesh: 362 / 1,414 exact physical matches (~26%).
- **Consequence**: Physical must be a LEFT JOIN — missing physical data is preserved as NaN, not dropped.

---

## 5. Implemented Join Strategy

### Primary Observation Grain

```
one row = one unique (Station, Data Acquisition Time) sampling event
```

### Step 1: Core Observation Table — Biological INNER JOIN Chemical

```
[ Official Biological ] ──┐
                          ├──INNER JOIN──→ [ Core Observation Table ]
[ Official Chemical ]  ───┘                  on (Station, Data Acquisition Time)
```

- Retains metadata columns (Agency, State, District, River, Basin, Lat/Long, etc.) from Biological.
- Drops duplicate metadata columns from Chemical to prevent `_x` / `_y` suffixes.
- 100% match rate in both states confirms zero data loss.

### Step 2: Physical Enrichment — LEFT JOIN

```
[ Core Observation Table ] ──┐
                              ├──LEFT JOIN──→ [ Pre-Label Table ]
[ Official Physical ]  ──────┘                  on (Station, Data Acquisition Time)
                                               + has_physical_record flag
```

- Adds `has_physical_record = True/False` flag.
- Unmatched rows retain NaN for physical parameters — no imputation at this stage.

### Step 3: Multi-State Concatenation & Temporal Columns

```
[ Maharashtra Pre-Label ] ──┐
                              ├──CONCAT──→ [ Final Pre-Label Table ]
[ Uttar Pradesh Pre-Label ] ─┘              data/interim/cpcb_joined.csv
                                            + year, month, day, sampling_date columns
```

### What This Strategy Does NOT Do

- **No broadcasting**: Chemical values are matched per-observation, not spread across a year.
- **No imputation**: Missing values remain as NaN for principled handling in Phase 2.
- **No fabrication**: Only authentic observation events appear in the output.

---

## 6. Join Audit Results

### Global Summary

| Metric | Value |
|--------|-------|
| Total Physical Input Rows | 1,228 |
| Total Biological Input Rows | 3,331 |
| Total Chemical Input Rows | 3,398 |
| Final Joined Pre-Label Observations | **3,331** |
| Unique Monitoring Stations | 347 (across 2 states) |
| Duplicate Observation Keys | **0** |

### State Breakdown

| State | Phys In | Bio In | Chem In | Bio+Chem Join | Phys Matched | Final Rows | Stations |
|-------|---------|--------|---------|---------------|--------------|------------|----------|
| Maharashtra | 828 | 1,917 | 1,942 | 1,917 | 803 | 1,917 | 216 |
| Uttar Pradesh | 400 | 1,414 | 1,456 | 1,414 | 362 | 1,414 | 131 |

### Parameter Missingness (Key WQI Parameters)

| Parameter | % Missing | Present |
|-----------|-----------|---------|
| Potential of Hydrogen (pH) | 0.8% | 3,304 |
| Dissolved Oxygen (mg/L) | 4.0% | 3,199 |
| BOD (mg/L) | 6.2% | 3,125 |
| Fecal Coliform (MPN/100mL) | 9.3% | 3,020 |
| Total Coliform (MPN/100mL) | 97.8% | 74 |
| Electric Conductivity (μS/cm) | 97.2% | 93 |
| Turbidity (NTU) | 99.4% | 19 |
| Total Hardness (mgCaCO3/L) | 98.5% | 49 |
| Chloride (mg/L) | 98.6% | 48 |
| Total Dissolved Solids (mg/L) | 98.5% | 50 |
| Nitrate N (mgN/L) | 99.1% | 30 |

### Implications for WQI Labeling (M1.3)

The four **core CPCB WQI parameters** — pH (0.8%), DO (4.0%), BOD (6.2%), Fecal Coliform (9.3%) — have strong coverage and are viable for WQI class assignment. Parameters with >95% missingness (EC, Turbidity, TDS, Hardness, Chloride, Nitrate) are effectively unavailable and should be excluded from the primary WQI computation, or treated as supplementary features only when present.

---

## 7. Output Specification

| Property | Value |
|----------|-------|
| Path | `data/interim/cpcb_joined.csv` |
| Rows | 3,331 |
| Columns | 67 |
| Primary Key | `(Station, Data Acquisition Time)` — guaranteed unique |
| Temporal Helpers | `year`, `month`, `day`, `sampling_date` |
| Physical Flag | `has_physical_record` (bool) |

---

## 8. Test Coverage

20 tests in `tests/test_join_cpcb.py` — all passing:

| ID | Test | Type |
|----|------|------|
| TC-01 | All six official files located | Integration |
| TC-01b | `locate_cpcb_raw_files` raises on missing | Unit |
| TC-02 | No duplicate keys in source data (6 parametrized) | Integration |
| TC-03 | Bio+Chem inner join produces zero duplicates | Unit |
| TC-03b | Join row count equals bio/chem intersection | Unit |
| TC-04 | Physical LEFT JOIN does not inflate rows | Unit |
| TC-04b | Physical match count is correct | Unit |
| TC-05 | Bi-weekly observations produce exactly 2 rows | Unit |
| TC-06 | Missing physical → NaN in physical columns | Unit |
| TC-06b | Matched physical → real values present | Unit |
| TC-07 | Chemical values are per-observation, not broadcast | Unit |
| TC-08 | Accounting fields match DataFrame dimensions | Unit |
| Integration | Full pipeline produces valid output | Integration |
| Integration | Maharashtra accounting | Integration |
| Integration | Uttar Pradesh accounting | Integration |
