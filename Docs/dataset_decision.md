# CPCB Dataset Decision & Official Ingestion Protocol

**Document ID**: `DEC-DATA-CPCB-2026-V2`  
**Milestone**: Milestone 1.2  
**Owner**: Data Lead (Person A)  
**Status**: Ready for Official CPCB/NWDP Upload

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
- **Recommended Vintage**: **2021 – 2025** (or most recent available multi-year continuous series).

---

## 3. Required Official CSV Files in `data/raw/`

Collaborators must place the official CPCB downloads into `data/raw/`:

1. **Physical Dataset (Monthly)**:
   - File pattern: `data/raw/*physical*.csv` (e.g. `maharashtra_physical.csv`, `cpcb_physical_monthly.csv`)
   - Monitored fields: Sampling station identifier, date/month/year, pH, Electrical Conductivity, Turbidity, Total Dissolved Solids, Temperature.
2. **Biological Dataset (Monthly)**:
   - File pattern: `data/raw/*biological*.csv` (e.g. `maharashtra_biological.csv`, `cpcb_biological_monthly.csv`)
   - Monitored fields: Station identifier, date/month/year, Dissolved Oxygen (DO), BOD, COD, Total Coliform, Faecal Coliform.
3. **Chemical Dataset (Annual)**:
   - File pattern: `data/raw/*chemical*.csv` (e.g. `maharashtra_chemical.csv`, `cpcb_chemical_annual.csv`)
   - Monitored fields: Station identifier, sampling year, Nitrate, Total Hardness, Chloride, Sulfate, Fluoride, Calcium, Magnesium.

---

## 4. Multi-Granularity Join Strategy

```
[ Official Monthly Physical ] ---\
                                 +---> [ Monthly Phys+Bio ] ---\
[ Official Monthly Biological ] -/                             |
                                                               +---> [ Joined Pre-Label Table ]
                                                               |     data/interim/cpcb_joined.csv
[ Official Annual Chemical ] -----> (Annual Broadcast) --------/
```

### Join Principles:
1. **Physical + Biological**: Merged on `(station_id, year, month)`.
2. **Annual Chemical Broadcasting**: Annual baseline mineral values for station $s$ and year $y$ are broadcast across all 12 monthly observations for $(s, y)$.
3. **Missing Chemical Fallback**: If a station-year lacks an annual chemical survey, records are marked `has_chemical_record = False` and preserved with explicit `NaN` values for principled imputation in Phase 2.

---

## 5. Next Step: Schema Inspection Upon Data Placement

Once the official CSV files are placed in `data/raw/`:
1. Inspect the exact header names and date formats across all 3 files.
2. Adapt column mappings in `src/data/join_cpcb.py` to match the exact government portal schema.
3. Execute the join engine and log official row counts and station statistics.
