"""CPCB Surface Water Quality Multi-Table Ingestion & Exact-Timestamp Join Engine.

This module loads and merges the official CPCB/NWDP Physical, Biological, and Chemical
monitoring datasets for Maharashtra and Uttar Pradesh into a unified pre-label interim dataset.

Primary Observation Grain:
    One row = One unique (Station, Data Acquisition Time) sampling event.
"""

from dataclasses import dataclass, field
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cpcb_join")


@dataclass
class DatasetAccounting:
    """Audit metrics for multi-state CPCB dataset ingestion and join."""
    state: str
    physical_input_rows: int = 0
    biological_input_rows: int = 0
    chemical_input_rows: int = 0
    bio_chem_joined_rows: int = 0
    physical_exact_matches: int = 0
    unmatched_physical_core_rows: int = 0
    final_output_rows: int = 0
    unique_stations: int = 0
    unique_timestamps: int = 0
    duplicate_target_keys: int = 0


@dataclass
class GlobalJoinAccounting:
    """Overall accounting across all processed states."""
    total_physical_rows: int = 0
    total_biological_rows: int = 0
    total_chemical_rows: int = 0
    total_joined_rows: int = 0
    total_unique_stations: int = 0
    total_unique_timestamps: int = 0
    total_duplicate_keys: int = 0
    state_breakdown: Dict[str, DatasetAccounting] = field(default_factory=dict)
    missingness_summary: Dict[str, float] = field(default_factory=dict)


def locate_cpcb_raw_files(raw_dir: Path) -> Dict[str, Dict[str, Path]]:
    """Locate the 6 official CPCB raw CSV files for Maharashtra and Uttar Pradesh."""
    states = ["maharashtra", "uttar_pradesh"]
    categories = ["physical", "biological", "chemical"]
    located_files: Dict[str, Dict[str, Path]] = {}

    for state in states:
        located_files[state] = {}
        for cat in categories:
            # Match pattern handling possible double extension .csv.csv or standard .csv
            pattern = str(raw_dir / f"*{state}*{cat}*.csv*")
            matches = [Path(p) for p in glob.glob(pattern) if not p.endswith(".gitkeep")]
            if not matches:
                raise FileNotFoundError(
                    f"Required raw CPCB file missing for state '{state}' category '{cat}'. "
                    f"Pattern searched: {pattern}"
                )
            located_files[state][cat] = matches[0]

    return located_files


def clean_cpcb_table(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, text whitespace, and timestamps."""
    df = df.copy()
    # Normalize column names: strip whitespace, lowercase, clean underscores
    df.columns = df.columns.str.strip()

    # Standardize metadata fields
    if "Station" in df.columns:
        df["Station"] = df["Station"].astype(str).str.strip()
    if "Data Acquisition Time" in df.columns:
        df["Data Acquisition Time"] = df["Data Acquisition Time"].astype(str).str.strip()

    return df


def join_state_datasets(
    state_name: str,
    physical_path: Path,
    biological_path: Path,
    chemical_path: Path,
) -> Tuple[pd.DataFrame, DatasetAccounting]:
    """Execute Step 1 (Bio+Chem inner join) and Step 2 (Physical left join) for a single state.

    Observation Grain: Station + Data Acquisition Time (Exact Timestamp).
    """
    logger.info("--- Processing State: %s ---", state_name)
    df_phys = pd.read_csv(physical_path, encoding="utf-8", encoding_errors="replace")
    df_bio = pd.read_csv(biological_path, encoding="utf-8", encoding_errors="replace")
    df_chem = pd.read_csv(chemical_path, encoding="utf-8", encoding_errors="replace")

    df_phys = clean_cpcb_table(df_phys)
    df_bio = clean_cpcb_table(df_bio)
    df_chem = clean_cpcb_table(df_chem)

    acct = DatasetAccounting(state=state_name)
    acct.physical_input_rows = len(df_phys)
    acct.biological_input_rows = len(df_bio)
    acct.chemical_input_rows = len(df_chem)

    # Verify zero duplicate keys in raw files
    phys_dups = int(df_phys.duplicated(subset=["Station", "Data Acquisition Time"]).sum())
    bio_dups = int(df_bio.duplicated(subset=["Station", "Data Acquisition Time"]).sum())
    chem_dups = int(df_chem.duplicated(subset=["Station", "Data Acquisition Time"]).sum())

    if phys_dups > 0 or bio_dups > 0 or chem_dups > 0:
        raise ValueError(
            f"Duplicate (Station, Data Acquisition Time) keys found in {state_name} raw files! "
            f"Physical dups: {phys_dups}, Bio dups: {bio_dups}, Chem dups: {chem_dups}"
        )

    # -------------------------------------------------------------
    # Step 1: Core Observation Table (Biological INNER JOIN Chemical)
    # -------------------------------------------------------------
    # Retain metadata from Biological, drop overlapping metadata columns in Chemical
    chem_metadata_to_drop = [
        c for c in [
            "SlNo", "Agency", "State LGD Code", "State", "District LGD Code",
            "District", "Tehsil", "Block", "Village", "River", "Basin",
            "Tributary", "Subtributary", "SubSubtributary", "Local River",
            "Latitude", "Longitude"
        ] if c in df_chem.columns and c in df_bio.columns
    ]
    df_chem_clean = df_chem.drop(columns=chem_metadata_to_drop, errors="ignore")

    df_core = pd.merge(
        df_bio,
        df_chem_clean,
        on=["Station", "Data Acquisition Time"],
        how="inner",
    )
    acct.bio_chem_joined_rows = len(df_core)

    # -------------------------------------------------------------
    # Step 2: Physical Enrichment (LEFT JOIN on Exact Timestamp)
    # -------------------------------------------------------------
    phys_metadata_to_drop = [
        c for c in [
            "SlNo", "Agency", "State LGD Code", "State", "District LGD Code",
            "District", "Tehsil", "Block", "Village", "River", "Basin",
            "Tributary", "Subtributary", "SubSubtributary", "Local River",
            "Latitude", "Longitude"
        ] if c in df_phys.columns and c in df_core.columns
    ]
    df_phys_clean = df_phys.drop(columns=phys_metadata_to_drop, errors="ignore")

    # Add flag to track physical observation match
    df_phys_clean["has_physical_record"] = True

    df_joined = pd.merge(
        df_core,
        df_phys_clean,
        on=["Station", "Data Acquisition Time"],
        how="left",
    )
    df_joined["has_physical_record"] = df_joined["has_physical_record"].eq(True)

    acct.final_output_rows = len(df_joined)
    acct.physical_exact_matches = int(df_joined["has_physical_record"].sum())
    acct.unmatched_physical_core_rows = acct.final_output_rows - acct.physical_exact_matches
    acct.unique_stations = int(df_joined["Station"].nunique())
    acct.unique_timestamps = int(df_joined["Data Acquisition Time"].nunique())
    acct.duplicate_target_keys = int(df_joined.duplicated(subset=["Station", "Data Acquisition Time"]).sum())

    # Fail fast if merge inflated row count or created duplicate target keys
    if acct.final_output_rows != acct.bio_chem_joined_rows:
        raise ValueError(
            f"Physical LEFT JOIN created unexpected Cartesian expansion in {state_name}! "
            f"Core rows: {acct.bio_chem_joined_rows}, Joined rows: {acct.final_output_rows}"
        )
    if acct.duplicate_target_keys > 0:
        raise ValueError(f"Duplicate (Station, Data Acquisition Time) keys in {state_name} output!")

    logger.info(
        "State %s accounting: Phys in=%d, Bio in=%d, Chem in=%d -> Core Bio+Chem=%d, "
        "Phys matched=%d, Final rows=%d, Stations=%d",
        state_name, acct.physical_input_rows, acct.biological_input_rows,
        acct.chemical_input_rows, acct.bio_chem_joined_rows,
        acct.physical_exact_matches, acct.final_output_rows, acct.unique_stations
    )

    return df_joined, acct


def run_cpcb_join_pipeline(
    raw_dir: Path,
    output_interim_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, GlobalJoinAccounting]:
    """Execute the complete multi-state CPCB dataset join pipeline.

    Loads official Maharashtra and Uttar Pradesh datasets, performs exact-timestamp
    joins, verifies zero key duplication, logs accounting, and saves pre-label table.
    """
    file_map = locate_cpcb_raw_files(raw_dir)
    state_dfs: List[pd.DataFrame] = []
    global_acct = GlobalJoinAccounting()

    for state_key, state_name in [("maharashtra", "Maharashtra"), ("uttar_pradesh", "Uttar Pradesh")]:
        paths = file_map[state_key]
        df_state, acct = join_state_datasets(
            state_name=state_name,
            physical_path=paths["physical"],
            biological_path=paths["biological"],
            chemical_path=paths["chemical"],
        )
        state_dfs.append(df_state)
        global_acct.state_breakdown[state_name] = acct
        global_acct.total_physical_rows += acct.physical_input_rows
        global_acct.total_biological_rows += acct.biological_input_rows
        global_acct.total_chemical_rows += acct.chemical_input_rows
        global_acct.total_joined_rows += acct.final_output_rows

    df_all = pd.concat(state_dfs, ignore_index=True)

    # Standardize temporal helper columns
    parsed_dates = pd.to_datetime(df_all["Data Acquisition Time"], format="mixed", dayfirst=True, errors="coerce")
    df_all["year"] = parsed_dates.dt.year
    df_all["month"] = parsed_dates.dt.month
    df_all["day"] = parsed_dates.dt.day
    df_all["sampling_date"] = parsed_dates.dt.date.astype(str)

    # Global key uniqueness validation
    global_acct.total_unique_stations = int(df_all["Station"].nunique())
    global_acct.total_unique_timestamps = int(df_all["Data Acquisition Time"].nunique())
    global_acct.total_duplicate_keys = int(df_all.duplicated(subset=["Station", "Data Acquisition Time"]).sum())

    if global_acct.total_duplicate_keys > 0:
        raise ValueError(f"FATAL: {global_acct.total_duplicate_keys} duplicate observation keys in final dataset!")

    # Calculate missingness for key WQI and monitoring parameters
    key_params = [
        "Potential of Hydrogen (pH)",
        "Dissolved oxygen (mg/L)",
        "Biochemical Oxygen Demand (mg/L)",
        "Fecal Coliform (MPN/100mL)",
        "Total Coliform (MPN/100mL)",
        "Electric Conductivity (μS/cm)",
        "Turbidity (NTU)",
        "Total Hardness (mgCaCO3/L)",
        "Chloride (mg/L)",
        "Total Dissolved Solids (mg/L)",
        "Nitrate N (mgN/L)",
    ]

    for param in key_params:
        if param in df_all.columns:
            null_count = int(df_all[param].isna().sum())
            null_pct = float((null_count / len(df_all)) * 100)
            global_acct.missingness_summary[param] = round(null_pct, 2)

    logger.info("================ GLOBAL JOIN AUDIT SUMMARY ================")
    logger.info("  Total Physical Input Rows: %d", global_acct.total_physical_rows)
    logger.info("  Total Biological Input Rows: %d", global_acct.total_biological_rows)
    logger.info("  Total Chemical Input Rows: %d", global_acct.total_chemical_rows)
    logger.info("  Final Joined Pre-Label Observations: %d", global_acct.total_joined_rows)
    logger.info("  Unique Monitoring Stations: %d across %d states", global_acct.total_unique_stations, len(global_acct.state_breakdown))
    logger.info("  Duplicate Observation Keys: %d", global_acct.total_duplicate_keys)
    logger.info("  Parameter Missingness in Pre-Label Table:")
    for p, pct in global_acct.missingness_summary.items():
        logger.info("    * %s: %.1f%% missing (%d present)", p, pct, len(df_all) - int((pct / 100) * len(df_all)))

    if output_interim_path is not None:
        output_interim_path.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(output_interim_path, index=False)
        logger.info("Successfully wrote final pre-label table to: %s", output_interim_path)

    return df_all, global_acct


def main():
    """CLI execution wrapper for CPCB dataset join."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = repo_root / "data" / "raw"
    interim_dir = repo_root / "data" / "interim"
    interim_output_path = interim_dir / "cpcb_joined.csv"

    df_joined, acct = run_cpcb_join_pipeline(
        raw_dir=raw_dir,
        output_interim_path=interim_output_path,
    )

    print(f"\n[CPCB Join Success] {len(df_joined)} authentic observations saved to {interim_output_path}")
    return df_joined, acct


if __name__ == "__main__":
    main()
