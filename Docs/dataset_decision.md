# CPCB Dataset Selection, Vintage & Granularity Join Strategy

**Document ID**: `DEC-DATA-CPCB-2026-V1`  
**Milestone**: Milestone 1.2  
**Owner**: Data Lead (Person A)  
**Status**: Approved & Frozen

---

## 1. State Selection & Geographic Rationale

The project selects two representative Indian states with the highest density of CPCB water quality monitoring stations:

| State | Primary River Basins Monitored | Station Profile | Geographic & Anthropogenic Rationale |
|---|---|---|---|
| **Maharashtra** | Godavari, Krishna, Tapi, Bhima, Panchganga, Ulhas, Mithi | 8 Primary Monitoring Stations | High industrial and urban runoff (e.g. Mumbai, Pune, Kolhapur) paired with headwater origin points (Trimbakeshwar, Mahabaleshwar). |
| **Uttar Pradesh** | Ganga, Yamuna, Gomti, Betwa, Saryu | 10 Primary Monitoring Stations | Critical stretch of the National Ganga River Basin, heavily impacted by domestic sewage, agricultural runoff, and industrial discharge (Kanpur, Varanasi, Agra). |

---

## 2. Dataset Vintage & Source Origin

- **Source Portal**: **National Water Data Portal (NWDP)** (`nwdp.nwic.gov.in`) / Central Pollution Control Board (CPCB)
- **Dataset Family**: CPCB Surface Water Quality (Physical, Chemical, and Biological Monitoring Datasets)
- **Time Window / Vintage**: **2021 – 2024** (4-year continuous cycle preferred over legacy 1961–2020 archives for modern sensor/lab reporting consistency).

---

## 3. Multi-Granularity Join Architecture

The CPCB data collection protocol creates an inherent temporal granularity mismatch:
- **Physical Parameters**: Sampled **Monthly** ($12\text{ samples/year/station}$).
- **Biological Parameters**: Sampled **Monthly** ($12\text{ samples/year/station}$).
- **Chemical Parameters**: Sampled **Annually** ($1\text{ sample/year/station}$).

```
[ Monthly Physical ] ----\
                         +---> [ Monthly Phys+Bio ] ---\
[ Monthly Biological ] --/                             |
                                                       +---> [ Unified Interim Dataset ]
                                                       |     data/interim/cpcb_joined.csv
[ Annual Chemical ] -----> (Broadcast by Year) --------/
```

### 3.1 Explicit Broadcasting Policy (Annual to Monthly)
- The annual chemical record for a given station $s$ in year $y$ measures stable mineral/ionic constituents (Total Hardness, Calcium, Magnesium, Chloride, Sulfate, Nitrate).
- This single annual reading is **explicitly broadcast** across all 12 monthly physical/biological records for that station and year:
  $$\text{Chemical}_{s, y, m} = \text{Chemical}_{s, y} \quad \forall m \in \{1, 2, \dots, 12\}$$
- **Rationale**: Mineral concentrations in established river stretches change on decadal and annual hydrologic timescales, unlike volatile biological ($BOD, DO, Coliform$) and physical ($pH, Turbidity$) parameters which fluctuate monthly.

---

## 4. Join Row-Count Accounting & Missingness Fallback

### 4.1 Join Audit Metrics
- **Physical Rows Ingested**: $864$ ($18\text{ stations} \times 4\text{ years} \times 12\text{ months}$)
- **Biological Rows Ingested**: $864$ ($18\text{ stations} \times 4\text{ years} \times 12\text{ months}$)
- **Chemical Records Ingested (Annual)**: $71$ ($18\text{ stations} \times 4\text{ years} - 1\text{ missing year for UP\_STN\_009}$)
- **Monthly Phys+Bio Joined**: $864$ rows ($100.0\%$ merge rate on `station_id, year, month`)
- **Final Joined Interim Rows**: $864$ rows saved to `data/interim/cpcb_joined.csv`
- **Matched Chemical Coverage**: $852$ rows ($98.6\%$)
- **Unmatched Chemical Observations**: $12$ rows ($1.4\%$, station `UP_STN_009` in 2023)

### 4.2 Documented Fallback for Unmatched Chemical Records
- Unmatched monthly records are **flagged explicitly** via the boolean indicator `has_chemical_record = False`.
- Unmatched chemical fields are left as explicit `NaN` values rather than being silently filled with zeros or dropped.
- **Phase 2 Policy**: In Phase 2, missing chemical fields are imputed using station-level median historical values or temporal interpolation, preserving statistical validity.

---

## 5. Output Data Schema ([`data/interim/cpcb_joined.csv`](file:///d:/Coding/College/Data%20science/CP/data/interim/cpcb_joined.csv))

| Column | Data Type | Source Origin | Description |
|---|---|---|---|
| `station_id` | `object` | Shared Key | Unique station identifier (e.g. `MAH_STN_001`) |
| `station_name` | `object` | Metadata | Station descriptive location |
| `state` | `object` | Metadata | State (`Maharashtra` or `Uttar Pradesh`) |
| `district` | `object` | Metadata | District |
| `water_body` | `object` | Metadata | Monitored river or lake |
| `sampling_date` | `object` | Physical/Bio | Sampling timestamp (`YYYY-MM-DD`) |
| `year` | `int64` | Time Key | Observation year |
| `month` | `int64` | Time Key | Observation month ($1-12$) |
| `ph` | `float64` | Physical (Monthly) | pH value ($0-14$) |
| `electrical_conductivity_us_cm` | `float64` | Physical (Monthly) | Conductivity in $\mu\text{S/cm}$ |
| `turbidity_ntu` | `float64` | Physical (Monthly) | Turbidity in NTU |
| `total_dissolved_solids_mg_l` | `float64` | Physical (Monthly) | TDS in $\text{mg/L}$ |
| `water_temperature_c` | `float64` | Physical (Monthly) | Temperature in $^\circ\text{C}$ |
| `dissolved_oxygen_mg_l` | `float64` | Biological (Monthly) | DO in $\text{mg/L}$ |
| `bod_mg_l` | `float64` | Biological (Monthly) | BOD in $\text{mg/L}$ |
| `cod_mg_l` | `float64` | Biological (Monthly) | COD in $\text{mg/L}$ |
| `total_coliform_mpn_100ml` | `float64` | Biological (Monthly) | Total Coliform in MPN/100mL |
| `faecal_coliform_mpn_100ml` | `float64` | Biological (Monthly) | Faecal Coliform in MPN/100mL |
| `nitrate_mg_l` | `float64` | Chemical (Annual Broadcast) | Nitrate in $\text{mg/L}$ |
| `total_hardness_mg_l` | `float64` | Chemical (Annual Broadcast) | Hardness as $\text{CaCO}_3$ in $\text{mg/L}$ |
| `chloride_mg_l` | `float64` | Chemical (Annual Broadcast) | Chloride in $\text{mg/L}$ |
| `sulfate_mg_l` | `float64` | Chemical (Annual Broadcast) | Sulfate in $\text{mg/L}$ |
| `fluoride_mg_l` | `float64` | Chemical (Annual Broadcast) | Fluoride in $\text{mg/L}$ |
| `calcium_mg_l` | `float64` | Chemical (Annual Broadcast) | Calcium in $\text{mg/L}$ |
| `magnesium_mg_l` | `float64` | Chemical (Annual Broadcast) | Magnesium in $\text{mg/L}$ |
| `has_chemical_record` | `bool` | Join Engine Flag | Indicates whether annual chemical matched |
