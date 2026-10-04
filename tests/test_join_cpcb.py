"""Test Suite for CPCB Dataset Ingestion and Multi-Granularity Join Engine (Milestone 1.2)."""
from pathlib import Path
import pandas as pd
import pytest

from src.data.join_cpcb import (
    load_and_join_cpcb_datasets,
    validate_raw_cpcb_files,
    JoinAccounting,
)


def test_validate_raw_cpcb_files_raises_on_missing(tmp_path):
    """Verify that validate_raw_cpcb_files raises FileNotFoundError when raw files are missing."""
    with pytest.raises(FileNotFoundError, match="Official CPCB raw datasets not found"):
        validate_raw_cpcb_files(
            physical_path=tmp_path / "nonexistent_physical.csv",
            biological_path=tmp_path / "nonexistent_biological.csv",
            chemical_path=tmp_path / "nonexistent_chemical.csv",
        )


@pytest.fixture
def mock_unit_test_tables(tmp_path):
    """Create minimal synthetic tables strictly for verifying join algorithm logic."""
    phys_path = tmp_path / "test_phys.csv"
    bio_path = tmp_path / "test_bio.csv"
    chem_path = tmp_path / "test_chem.csv"
    out_path = tmp_path / "test_joined.csv"

    # 2 stations, 2 years (2022, 2023), 12 months each = 48 monthly rows
    phys_rows = []
    bio_rows = []
    for stn in ["STN_A", "STN_B"]:
        for yr in [2022, 2023]:
            for mo in range(1, 13):
                phys_rows.append({
                    "station_id": stn,
                    "station_name": f"{stn} River Point",
                    "state": "Maharashtra",
                    "year": yr,
                    "month": mo,
                    "ph": 7.5,
                    "electrical_conductivity_us_cm": 350.0,
                })
                bio_rows.append({
                    "station_id": stn,
                    "year": yr,
                    "month": mo,
                    "dissolved_oxygen_mg_l": 6.8,
                    "bod_mg_l": 2.1,
                })

    # Chemical annual records: STN_B in 2023 is omitted to test missing chemical fallback
    chem_rows = [
        {"station_id": "STN_A", "year": 2022, "nitrate_mg_l": 12.0, "total_hardness_mg_l": 150.0},
        {"station_id": "STN_A", "year": 2023, "nitrate_mg_l": 14.5, "total_hardness_mg_l": 160.0},
        {"station_id": "STN_B", "year": 2022, "nitrate_mg_l": 8.0, "total_hardness_mg_l": 110.0},
    ]

    pd.DataFrame(phys_rows).to_csv(phys_path, index=False)
    pd.DataFrame(bio_rows).to_csv(bio_path, index=False)
    pd.DataFrame(chem_rows).to_csv(chem_path, index=False)

    return {
        "phys": phys_path,
        "bio": bio_path,
        "chem": chem_path,
        "out": out_path,
    }


def test_tc_1_2_join_algorithm_and_accounting(mock_unit_test_tables):
    """TC-1.2-01 & TC-1.2-02: Verify join algorithm, row accounting, and column preservation."""
    paths = mock_unit_test_tables

    df_joined, accounting = load_and_join_cpcb_datasets(
        physical_path=paths["phys"],
        biological_path=paths["bio"],
        chemical_path=paths["chem"],
        output_interim_path=paths["out"],
    )

    assert accounting.physical_rows == 48
    assert accounting.biological_rows == 48
    assert accounting.chemical_rows == 3
    assert accounting.phys_bio_merged_rows == 48
    assert len(df_joined) == 48
    assert paths["out"].exists()


def test_tc_1_2_chemical_broadcasting(mock_unit_test_tables):
    """TC-1.2-03: Verify annual chemical values are broadcast to all 12 monthly rows."""
    paths = mock_unit_test_tables

    df_joined, _ = load_and_join_cpcb_datasets(
        physical_path=paths["phys"],
        biological_path=paths["bio"],
        chemical_path=paths["chem"],
    )

    stn_a_2022 = df_joined[(df_joined["station_id"] == "STN_A") & (df_joined["year"] == 2022)]
    assert len(stn_a_2022) == 12
    assert (stn_a_2022["nitrate_mg_l"] == 12.0).all()
    assert (stn_a_2022["total_hardness_mg_l"] == 150.0).all()
    assert (stn_a_2022["has_chemical_record"] == True).all()


def test_tc_1_2_missing_chemical_handling(mock_unit_test_tables):
    """TC-1.2-04: Verify missing chemical record is flagged and preserved with NaN."""
    paths = mock_unit_test_tables

    df_joined, _ = load_and_join_cpcb_datasets(
        physical_path=paths["phys"],
        biological_path=paths["bio"],
        chemical_path=paths["chem"],
    )

    stn_b_2023 = df_joined[(df_joined["station_id"] == "STN_B") & (df_joined["year"] == 2023)]
    assert len(stn_b_2023) == 12
    assert (stn_b_2023["has_chemical_record"] == False).all()
    assert stn_b_2023["nitrate_mg_l"].isna().all()
    assert (stn_b_2023["ph"] == 7.5).all()
