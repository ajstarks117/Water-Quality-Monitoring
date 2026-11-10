"""Test Suite for CPCB Exact-Timestamp Join Engine (Milestone 1.2).

Tests verify the 3-step join strategy:
  Step 1: Biological INNER JOIN Chemical on (Station, Data Acquisition Time)
  Step 2: LEFT JOIN Physical on (Station, Data Acquisition Time)
  Step 3: Concat states, add temporal columns, verify zero duplicate keys.

Both unit tests (synthetic fixtures) and integration tests (real CPCB data)
are included.  Integration tests are skipped when official files are absent.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.join_cpcb import (
    locate_cpcb_raw_files,
    clean_cpcb_table,
    join_state_datasets,
    run_cpcb_join_pipeline,
    DatasetAccounting,
    GlobalJoinAccounting,
)

# ────────────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
INTERIM_DIR = REPO_ROOT / "data" / "interim"

OFFICIAL_FILES_PRESENT = all(
    list(RAW_DIR.glob(f"*{s}*{c}*.csv*"))
    for s in ("maharashtra", "uttar_pradesh")
    for c in ("physical", "biological", "chemical")
)


# ────────────────────────────────────────────────────────────────────────
# Fixtures — minimal synthetic tables for unit-level join-logic tests
# ────────────────────────────────────────────────────────────────────────
@pytest.fixture
def synthetic_state_csvs(tmp_path):
    """Create three small CSVs that mirror official CPCB column structure.

    Stations:
        STN_A  — has a bio+chem record AND a physical record for each timestamp
        STN_B  — has a bio+chem record but NO physical record (tests LEFT JOIN gap)
        STN_C  — has two bio+chem records in the same calendar month (tests
                 bi-weekly non-duplication)
    """
    bio_rows = [
        {"SlNo": 1, "Station": "STN_A", "State": "TestState",
         "Data Acquisition Time": "01/01/2023 10:00",
         "Biochemical Oxygen Demand (mg/L)": 2.5,
         "Dissolved oxygen (mg/L)": 7.1,
         "Fecal Coliform (MPN/100mL)": 120},
        {"SlNo": 2, "Station": "STN_B", "State": "TestState",
         "Data Acquisition Time": "15/02/2023 09:30",
         "Biochemical Oxygen Demand (mg/L)": 3.0,
         "Dissolved oxygen (mg/L)": 5.8,
         "Fecal Coliform (MPN/100mL)": 250},
        # STN_C bi-weekly: two timestamps in the same month
        {"SlNo": 3, "Station": "STN_C", "State": "TestState",
         "Data Acquisition Time": "05/03/2023 08:00",
         "Biochemical Oxygen Demand (mg/L)": 1.9,
         "Dissolved oxygen (mg/L)": 6.5,
         "Fecal Coliform (MPN/100mL)": 90},
        {"SlNo": 4, "Station": "STN_C", "State": "TestState",
         "Data Acquisition Time": "20/03/2023 08:00",
         "Biochemical Oxygen Demand (mg/L)": 2.1,
         "Dissolved oxygen (mg/L)": 6.3,
         "Fecal Coliform (MPN/100mL)": 95},
    ]

    chem_rows = [
        {"SlNo": 1, "Station": "STN_A", "State": "TestState",
         "Data Acquisition Time": "01/01/2023 10:00",
         "Potential of Hydrogen (pH)": 7.4,
         "Chloride (mg/L)": 25.0},
        {"SlNo": 2, "Station": "STN_B", "State": "TestState",
         "Data Acquisition Time": "15/02/2023 09:30",
         "Potential of Hydrogen (pH)": 7.8,
         "Chloride (mg/L)": 30.0},
        {"SlNo": 3, "Station": "STN_C", "State": "TestState",
         "Data Acquisition Time": "05/03/2023 08:00",
         "Potential of Hydrogen (pH)": 7.2,
         "Chloride (mg/L)": 18.0},
        {"SlNo": 4, "Station": "STN_C", "State": "TestState",
         "Data Acquisition Time": "20/03/2023 08:00",
         "Potential of Hydrogen (pH)": 7.3,
         "Chloride (mg/L)": 19.0},
    ]

    phys_rows = [
        # Only STN_A has a physical record; STN_B and STN_C do not
        {"SlNo": 1, "Station": "STN_A", "State": "TestState",
         "Data Acquisition Time": "01/01/2023 10:00",
         "Temperature (ºC)": 22.5,
         "Turbidity (NTU)": 4.8},
    ]

    bio_path = tmp_path / "test_biological.csv"
    chem_path = tmp_path / "test_chemical.csv"
    phys_path = tmp_path / "test_physical.csv"

    pd.DataFrame(bio_rows).to_csv(bio_path, index=False)
    pd.DataFrame(chem_rows).to_csv(chem_path, index=False)
    pd.DataFrame(phys_rows).to_csv(phys_path, index=False)

    return {
        "bio": bio_path,
        "chem": chem_path,
        "phys": phys_path,
        "tmp": tmp_path,
    }


# ────────────────────────────────────────────────────────────────────────
# TC-1: All six official files are located (integration)
# ────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not OFFICIAL_FILES_PRESENT, reason="Official CPCB files not in data/raw/")
class TestOfficialFileLocation:
    def test_tc01_all_six_files_located(self):
        """All six official CPCB raw CSV files are found and non-empty."""
        file_map = locate_cpcb_raw_files(RAW_DIR)
        for state in ("maharashtra", "uttar_pradesh"):
            for cat in ("physical", "biological", "chemical"):
                p = file_map[state][cat]
                assert p.exists(), f"Missing: {p}"
                assert p.stat().st_size > 0, f"Empty: {p}"

    def test_tc01b_locate_raises_on_missing(self, tmp_path):
        """locate_cpcb_raw_files raises FileNotFoundError when files are absent."""
        with pytest.raises(FileNotFoundError):
            locate_cpcb_raw_files(tmp_path)


# ────────────────────────────────────────────────────────────────────────
# TC-2: No duplicate (Station, Data Acquisition Time) keys in source data
# ────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not OFFICIAL_FILES_PRESENT, reason="Official CPCB files not in data/raw/")
class TestSourceKeyUniqueness:
    @pytest.mark.parametrize("state", ["maharashtra", "uttar_pradesh"])
    @pytest.mark.parametrize("cat", ["physical", "biological", "chemical"])
    def test_tc02_no_duplicate_keys(self, state, cat):
        """Each raw file has unique (Station, Data Acquisition Time) keys."""
        file_map = locate_cpcb_raw_files(RAW_DIR)
        df = pd.read_csv(file_map[state][cat], encoding="utf-8", encoding_errors="replace")
        df = clean_cpcb_table(df)
        dups = df.duplicated(subset=["Station", "Data Acquisition Time"]).sum()
        assert dups == 0, f"{state}/{cat}: {dups} duplicate keys"


# ────────────────────────────────────────────────────────────────────────
# TC-3: Bio + Chem inner join does not create duplicate observation keys
# ────────────────────────────────────────────────────────────────────────
class TestBioChemJoinUnit:
    def test_tc03_inner_join_no_duplicates(self, synthetic_state_csvs):
        """Bio INNER JOIN Chem produces zero duplicate target keys."""
        p = synthetic_state_csvs
        df, acct = join_state_datasets(
            "TestState", p["phys"], p["bio"], p["chem"]
        )
        assert acct.duplicate_target_keys == 0

    def test_tc03b_row_count_equals_bio_chem_intersection(self, synthetic_state_csvs):
        """Joined rows == bio_chem_joined_rows == 4 (the full inner intersection)."""
        p = synthetic_state_csvs
        df, acct = join_state_datasets(
            "TestState", p["phys"], p["bio"], p["chem"]
        )
        assert acct.bio_chem_joined_rows == 4
        assert acct.final_output_rows == 4
        assert len(df) == 4


# ────────────────────────────────────────────────────────────────────────
# TC-4: Physical LEFT JOIN does not increase number of core observations
# ────────────────────────────────────────────────────────────────────────
class TestPhysicalLeftJoin:
    def test_tc04_left_join_no_inflation(self, synthetic_state_csvs):
        """Physical LEFT JOIN must not inflate core row count."""
        p = synthetic_state_csvs
        df, acct = join_state_datasets(
            "TestState", p["phys"], p["bio"], p["chem"]
        )
        assert acct.final_output_rows == acct.bio_chem_joined_rows

    def test_tc04b_physical_match_count(self, synthetic_state_csvs):
        """Only STN_A should have a physical match; 3 rows lack physical data."""
        p = synthetic_state_csvs
        df, acct = join_state_datasets(
            "TestState", p["phys"], p["bio"], p["chem"]
        )
        assert acct.physical_exact_matches == 1
        assert acct.unmatched_physical_core_rows == 3


# ────────────────────────────────────────────────────────────────────────
# TC-5: UP bi-weekly observations do not produce Cartesian duplication
# ────────────────────────────────────────────────────────────────────────
class TestBiWeeklyNonDuplication:
    def test_tc05_biweekly_stays_two_rows(self, synthetic_state_csvs):
        """STN_C has 2 observations in March; join must produce exactly 2, not 4."""
        p = synthetic_state_csvs
        df, acct = join_state_datasets(
            "TestState", p["phys"], p["bio"], p["chem"]
        )
        stn_c = df[df["Station"] == "STN_C"]
        assert len(stn_c) == 2
        # Each row must be a different timestamp
        assert stn_c["Data Acquisition Time"].nunique() == 2


# ────────────────────────────────────────────────────────────────────────
# TC-6: Missing physical observations remain as rows with NaN physical fields
# ────────────────────────────────────────────────────────────────────────
class TestMissingPhysicalNaN:
    def test_tc06_missing_physical_has_nan(self, synthetic_state_csvs):
        """STN_B has no physical record → physical columns must be NaN."""
        p = synthetic_state_csvs
        df, acct = join_state_datasets(
            "TestState", p["phys"], p["bio"], p["chem"]
        )
        stn_b = df[df["Station"] == "STN_B"]
        assert len(stn_b) == 1
        assert stn_b["has_physical_record"].iloc[0] == False
        # Physical-only columns should be NaN
        if "Temperature (ºC)" in df.columns:
            assert pd.isna(stn_b["Temperature (ºC)"].iloc[0])
        if "Turbidity (NTU)" in df.columns:
            assert pd.isna(stn_b["Turbidity (NTU)"].iloc[0])

    def test_tc06b_matched_physical_has_values(self, synthetic_state_csvs):
        """STN_A has a physical match → physical columns must have real values."""
        p = synthetic_state_csvs
        df, _ = join_state_datasets(
            "TestState", p["phys"], p["bio"], p["chem"]
        )
        stn_a = df[df["Station"] == "STN_A"]
        assert stn_a["has_physical_record"].iloc[0] == True
        if "Temperature (ºC)" in df.columns:
            assert stn_a["Temperature (ºC)"].iloc[0] == pytest.approx(22.5)


# ────────────────────────────────────────────────────────────────────────
# TC-7: No sparse chemical parameter is silently broadcast / imputed
# ────────────────────────────────────────────────────────────────────────
class TestNoSilentBroadcast:
    def test_tc07_chemical_values_are_per_observation(self, synthetic_state_csvs):
        """Each observation gets its own chemical values — no broadcasting."""
        p = synthetic_state_csvs
        df, _ = join_state_datasets(
            "TestState", p["phys"], p["bio"], p["chem"]
        )
        # STN_C's two March observations should each have their own pH
        stn_c = df[df["Station"] == "STN_C"].sort_values("Data Acquisition Time")
        ph_values = stn_c["Potential of Hydrogen (pH)"].tolist()
        assert ph_values == [7.2, 7.3], "Chemical values were broadcast instead of matched per-event"


# ────────────────────────────────────────────────────────────────────────
# TC-8: Final row count exactly matches the logged accounting
# ────────────────────────────────────────────────────────────────────────
class TestAccountingConsistency:
    def test_tc08_accounting_matches_dataframe(self, synthetic_state_csvs):
        """DatasetAccounting fields must match actual DataFrame dimensions."""
        p = synthetic_state_csvs
        df, acct = join_state_datasets(
            "TestState", p["phys"], p["bio"], p["chem"]
        )
        assert acct.final_output_rows == len(df)
        assert acct.unique_stations == df["Station"].nunique()
        assert acct.unique_timestamps == df["Data Acquisition Time"].nunique()
        assert acct.physical_input_rows == 1
        assert acct.biological_input_rows == 4
        assert acct.chemical_input_rows == 4


# ────────────────────────────────────────────────────────────────────────
# Integration Tests — run against actual CPCB data
# ────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not OFFICIAL_FILES_PRESENT, reason="Official CPCB files not in data/raw/")
class TestIntegrationFullPipeline:
    def test_full_pipeline_produces_valid_output(self, tmp_path):
        """Full pipeline produces a non-empty table with zero duplicate keys."""
        out_path = tmp_path / "cpcb_joined.csv"
        df, global_acct = run_cpcb_join_pipeline(
            raw_dir=RAW_DIR,
            output_interim_path=out_path,
        )
        assert len(df) > 0
        assert out_path.exists()
        assert global_acct.total_duplicate_keys == 0
        assert global_acct.total_joined_rows == len(df)

    def test_maharashtra_accounting(self):
        """Maharashtra: Bio+Chem inner join = Bio rows (100% match)."""
        file_map = locate_cpcb_raw_files(RAW_DIR)
        df, acct = join_state_datasets(
            "Maharashtra",
            file_map["maharashtra"]["physical"],
            file_map["maharashtra"]["biological"],
            file_map["maharashtra"]["chemical"],
        )
        assert acct.bio_chem_joined_rows == acct.biological_input_rows
        assert acct.duplicate_target_keys == 0

    def test_uttar_pradesh_accounting(self):
        """Uttar Pradesh: Bio+Chem inner join = Bio rows (100% match)."""
        file_map = locate_cpcb_raw_files(RAW_DIR)
        df, acct = join_state_datasets(
            "Uttar Pradesh",
            file_map["uttar_pradesh"]["physical"],
            file_map["uttar_pradesh"]["biological"],
            file_map["uttar_pradesh"]["chemical"],
        )
        assert acct.bio_chem_joined_rows == acct.biological_input_rows
        assert acct.duplicate_target_keys == 0
