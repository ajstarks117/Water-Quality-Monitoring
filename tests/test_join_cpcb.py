"""Test Suite for CPCB Dataset Ingestion and Multi-Granularity Join (Milestone 1.2)."""
from pathlib import Path
import pandas as pd
import pytest

from src.data.join_cpcb import (
    generate_cpcb_raw_data,
    load_and_join_cpcb_datasets,
    JoinAccounting,
)


@pytest.fixture(scope="module")
def prepared_cpcb_datasets(tmp_path_factory):
    """Generate and join CPCB raw datasets for testing."""
    raw_dir = tmp_path_factory.mktemp("raw")
    interim_dir = tmp_path_factory.mktemp("interim")
    interim_output_path = interim_dir / "cpcb_joined.csv"

    paths = generate_cpcb_raw_data(output_dir=raw_dir, states=["Maharashtra", "Uttar Pradesh"], seed=42)

    df_joined, accounting = load_and_join_cpcb_datasets(
        physical_path=paths["physical"],
        biological_path=paths["biological"],
        chemical_path=paths["chemical"],
        output_interim_path=interim_output_path,
    )

    return {
        "paths": paths,
        "df_joined": df_joined,
        "accounting": accounting,
        "interim_output_path": interim_output_path,
    }


def test_tc_1_2_01_all_three_source_files_load(prepared_cpcb_datasets):
    """TC-1.2-01: Verify all three raw CPCB files load with no parse errors."""
    paths = prepared_cpcb_datasets["paths"]

    df_phys = pd.read_csv(paths["physical"])
    df_bio = pd.read_csv(paths["biological"])
    df_chem = pd.read_csv(paths["chemical"])

    assert len(df_phys) > 0, "Physical dataset is empty"
    assert len(df_bio) > 0, "Biological dataset is empty"
    assert len(df_chem) > 0, "Chemical dataset is empty"

    assert "ph" in df_phys.columns
    assert "dissolved_oxygen_mg_l" in df_bio.columns
    assert "nitrate_mg_l" in df_chem.columns


def test_tc_1_2_02_join_produces_documented_row_count(prepared_cpcb_datasets):
    """TC-1.2-02: Verify join produces exact documented row count with no unexplained row loss."""
    accounting: JoinAccounting = prepared_cpcb_datasets["accounting"]
    df_joined: pd.DataFrame = prepared_cpcb_datasets["df_joined"]

    assert accounting.physical_rows == 864, f"Expected 864 physical rows, got {accounting.physical_rows}"
    assert accounting.biological_rows == 864, f"Expected 864 biological rows, got {accounting.biological_rows}"
    assert accounting.phys_bio_merged_rows == 864, "Row loss during monthly physical+biological merge"
    assert len(df_joined) == 864, f"Expected 864 joined rows, got {len(df_joined)}"
    assert accounting.unique_stations == 18, f"Expected 18 stations, got {accounting.unique_stations}"


def test_tc_1_2_03_chemical_broadcast_strategy_applied_consistently(prepared_cpcb_datasets):
    """TC-1.2-03: Spot-check stations to confirm chemical parameters are consistently broadcast across all 12 months."""
    df_joined: pd.DataFrame = prepared_cpcb_datasets["df_joined"]

    spot_check_stations = ["MAH_STN_001", "MAH_STN_003", "UP_STN_002", "UP_STN_004", "UP_STN_007"]

    for stn_id in spot_check_stations:
        for year in [2021, 2022, 2024]:
            stn_year_df = df_joined[(df_joined["station_id"] == stn_id) & (df_joined["year"] == year)]
            assert len(stn_year_df) == 12, f"Station {stn_id} year {year} must have 12 monthly observations"

            # Check that broadcast chemical fields have zero variance across the 12 months
            for chem_col in ["nitrate_mg_l", "total_hardness_mg_l", "chloride_mg_l", "sulfate_mg_l"]:
                unique_vals = stn_year_df[chem_col].dropna().unique()
                assert len(unique_vals) == 1, (
                    f"Inconsistent broadcast for {stn_id} year {year} on {chem_col}: {unique_vals}"
                )


def test_tc_1_2_04_negative_station_with_no_chemical_match(prepared_cpcb_datasets):
    """TC-1.2-04: Verify station with missing chemical record is explicitly flagged and preserved with NaN, not zero-filled."""
    df_joined: pd.DataFrame = prepared_cpcb_datasets["df_joined"]

    # UP_STN_009 in 2023 was intentionally left without an annual chemical record
    missing_chem_df = df_joined[(df_joined["station_id"] == "UP_STN_009") & (df_joined["year"] == 2023)]

    assert len(missing_chem_df) == 12, "Unmatched chemical rows should not be dropped"
    assert (missing_chem_df["has_chemical_record"] == False).all(), "has_chemical_record flag must be False"

    # Ensure chemical columns are NaN rather than silently zeroed
    assert missing_chem_df["nitrate_mg_l"].isna().all(), "Missing chemical values should be NaN"
    assert missing_chem_df["total_hardness_mg_l"].isna().all(), "Missing chemical values should be NaN"

    # Physical and biological values must still be intact
    assert missing_chem_df["ph"].notna().all(), "Physical parameters must remain valid"
    assert missing_chem_df["dissolved_oxygen_mg_l"].notna().all(), "Biological parameters must remain valid"
