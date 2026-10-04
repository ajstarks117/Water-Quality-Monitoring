"""CPCB Surface Water Quality Dataset Ingestion & Multi-Granularity Join Engine.

This module loads and merges official CPCB Physical (monthly), Biological (monthly),
and Chemical (annual) monitoring datasets across selected states into a unified,
station-and-date-indexed interim dataset.

NOTE: This module operates strictly on official CPCB/NWDP datasets placed in `data/raw/`.
It contains NO synthetic data generation routines.
"""

from dataclasses import dataclass, field
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
class JoinAccounting:
    """Detailed audit metrics for the multi-granularity CPCB join process."""
    physical_rows: int = 0
    biological_rows: int = 0
    chemical_rows: int = 0
    phys_bio_merged_rows: int = 0
    final_joined_rows: int = 0
    matched_chemical_rows: int = 0
    unmatched_chemical_rows: int = 0
    unique_stations: int = 0
    unique_years: List[int] = field(default_factory=list)
    states: List[str] = field(default_factory=list)


def validate_raw_cpcb_files(
    physical_path: Path,
    biological_path: Path,
    chemical_path: Path,
) -> None:
    """Validate that official CPCB dataset files exist prior to processing."""
    missing = []
    if not physical_path.exists():
        missing.append(f"Physical dataset missing: {physical_path}")
    if not biological_path.exists():
        missing.append(f"Biological dataset missing: {biological_path}")
    if not chemical_path.exists():
        missing.append(f"Chemical dataset missing: {chemical_path}")

    if missing:
        error_msg = (
            "Official CPCB raw datasets not found in data/raw/.\n"
            "Please download and place the official CPCB/NWDP CSV files into data/raw/ before running.\n"
            + "\n".join(f"  - {m}" for m in missing)
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)


def load_and_join_cpcb_datasets(
    physical_path: Path,
    biological_path: Path,
    chemical_path: Path,
    output_interim_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, JoinAccounting]:
    """Load and join official CPCB Physical, Biological, and Chemical tables.

    Strategy:
    1. Validate presence of raw datasets.
    2. Normalize column headers (strip whitespace, lowercase).
    3. Parse station identifiers and temporal keys (year, month).
    4. Join Monthly Physical and Monthly Biological records on (station_id, year, month).
    5. Join Annual Chemical on (station_id, year) using annual-to-monthly broadcasting.
    6. Flag observations lacking annual chemical records with `has_chemical_record = False`
       and preserve explicit NaN values for downstream imputation in Phase 2.
    """
    validate_raw_cpcb_files(physical_path, biological_path, chemical_path)

    logger.info("Loading official CPCB physical data from: %s", physical_path)
    df_phys = pd.read_csv(physical_path)

    logger.info("Loading official CPCB biological data from: %s", biological_path)
    df_bio = pd.read_csv(biological_path)

    logger.info("Loading official CPCB chemical data from: %s", chemical_path)
    df_chem = pd.read_csv(chemical_path)

    accounting = JoinAccounting()
    accounting.physical_rows = len(df_phys)
    accounting.biological_rows = len(df_bio)
    accounting.chemical_rows = len(df_chem)

    # Standardize column names
    for df in [df_phys, df_bio, df_chem]:
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Resolve station identifier column
    for df in [df_phys, df_bio, df_chem]:
        stn_col = next((c for c in df.columns if "station" in c and ("id" in c or "code" in c or "no" in c)), None)
        if stn_col and stn_col != "station_id":
            df.rename(columns={stn_col: "station_id"}, inplace=True)
        elif "station_id" not in df.columns and "station" in df.columns:
            df.rename(columns={"station": "station_id"}, inplace=True)

    # Resolve temporal columns
    for df in [df_phys, df_bio]:
        df["station_id"] = df["station_id"].astype(str).str.strip()
        if "year" in df.columns:
            df["year"] = df["year"].astype(int)
        if "month" in df.columns:
            df["month"] = df["month"].astype(int)

    df_chem["station_id"] = df_chem["station_id"].astype(str).str.strip()
    if "sampling_year" in df_chem.columns:
        df_chem["year"] = df_chem["sampling_year"].astype(int)
    elif "year" in df_chem.columns:
        df_chem["year"] = df_chem["year"].astype(int)

    # Step 1: Join Physical and Biological on (station_id, year, month)
    bio_cols_to_drop = [
        c for c in ["station_name", "state", "district", "water_body", "sampling_date"]
        if c in df_bio.columns and c in df_phys.columns
    ]
    df_bio_clean = df_bio.drop(columns=bio_cols_to_drop, errors="ignore")

    df_monthly = pd.merge(
        df_phys,
        df_bio_clean,
        on=["station_id", "year", "month"],
        how="inner",
        suffixes=("", "_bio"),
    )
    accounting.phys_bio_merged_rows = len(df_monthly)

    # Step 2: Join Annual Chemical onto Monthly records on (station_id, year)
    chem_cols_to_drop = [
        c for c in ["station_name", "state", "district", "water_body", "sampling_date", "sampling_year"]
        if c in df_chem.columns and c in df_monthly.columns
    ]
    df_chem_clean = df_chem.drop(columns=chem_cols_to_drop, errors="ignore")
    df_chem_clean["has_chemical_record"] = True

    df_joined = pd.merge(
        df_monthly,
        df_chem_clean,
        on=["station_id", "year"],
        how="left",
    )

    df_joined["has_chemical_record"] = df_joined["has_chemical_record"].eq(True)

    accounting.final_joined_rows = len(df_joined)
    accounting.matched_chemical_rows = int(df_joined["has_chemical_record"].sum())
    accounting.unmatched_chemical_rows = accounting.final_joined_rows - accounting.matched_chemical_rows
    accounting.unique_stations = df_joined["station_id"].nunique() if "station_id" in df_joined.columns else 0
    accounting.unique_years = sorted(df_joined["year"].unique().tolist()) if "year" in df_joined.columns else []
    accounting.states = sorted(df_joined["state"].unique().tolist()) if "state" in df_joined.columns else []

    logger.info("CPCB Join Completed Successfully!")
    logger.info("Accounting Summary:")
    logger.info("  - Physical Rows: %d", accounting.physical_rows)
    logger.info("  - Biological Rows: %d", accounting.biological_rows)
    logger.info("  - Chemical Rows (Annual): %d", accounting.chemical_rows)
    logger.info("  - Monthly Phys+Bio Merged: %d", accounting.phys_bio_merged_rows)
    logger.info("  - Final Joined Output Rows: %d", accounting.final_joined_rows)
    logger.info("  - Matched Chemical Records: %d", accounting.matched_chemical_rows)
    logger.info("  - Unmatched Chemical Records: %d", accounting.unmatched_chemical_rows)

    if output_interim_path is not None:
        output_interim_path.parent.mkdir(parents=True, exist_ok=True)
        df_joined.to_csv(output_interim_path, index=False)
        logger.info("Saved joined interim dataset to: %s", output_interim_path)

    return df_joined, accounting


def main():
    """Main entrypoint for joining official CPCB surface water quality datasets."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = repo_root / "data" / "raw"
    interim_dir = repo_root / "data" / "interim"
    interim_output_path = interim_dir / "cpcb_joined.csv"

    # Identify CPCB raw files in data/raw
    phys_files = list(raw_dir.glob("*physical*.csv"))
    bio_files = list(raw_dir.glob("*biological*.csv"))
    chem_files = list(raw_dir.glob("*chemical*.csv"))

    if not phys_files or not bio_files or not chem_files:
        raise FileNotFoundError(
            "Official CPCB CSV files not found in data/raw/.\n"
            "Please place the downloaded CPCB files into data/raw/ with names containing "
            "'physical', 'biological', and 'chemical' (e.g. maharashtra_physical.csv)."
        )

    df_joined, accounting = load_and_join_cpcb_datasets(
        physical_path=phys_files[0],
        biological_path=bio_files[0],
        chemical_path=chem_files[0],
        output_interim_path=interim_output_path,
    )

    print(f"\n[Official CPCB Join Complete] {len(df_joined)} records written to {interim_output_path}")
    return df_joined, accounting


if __name__ == "__main__":
    main()
