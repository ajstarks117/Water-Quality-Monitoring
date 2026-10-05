"""Raw & Interim Data Ingestion and Schema Validation Module.

This module provides the authoritative, reusable dataset loader for the CPCB
Surface Water Quality monitoring dataset. Every time data is ingested, it is
automatically validated against the frozen schema contract (Milestone 1.4)
to catch schema drift, missing columns, invalid data types, and target leakage
immediately before data enters cleaning, exploratory data analysis, or modeling.

Key Functions:
    - `load_raw_data`: Ingests and validates the interim labeled CPCB dataset.
    - `validate_schema`: Verifies required columns, dtypes, missingness, and leakage guards.
    - `load_config`: Central configuration loader.
"""

from pathlib import Path
import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import yaml

# Configure structured logging
logger = logging.getLogger("load_data")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Custom Domain Exceptions
# ---------------------------------------------------------------------------
class DataIngestionError(Exception):
    """Base exception for all data ingestion and loading errors."""
    pass


class DataNotFoundError(DataIngestionError, FileNotFoundError):
    """Raised when the specified dataset path does not exist on disk."""
    pass


class SchemaValidationError(DataIngestionError):
    """Base exception for schema contract validation failures."""
    pass


class MissingColumnError(SchemaValidationError):
    """Raised when one or more required schema columns are missing from the dataset."""
    pass


class DataLeakageError(SchemaValidationError):
    """Raised when a target leakage column (e.g. `wqi_score`) is detected in feature columns."""
    pass


class InvalidDtypeError(SchemaValidationError):
    """Raised when a column's data type does not match the expected schema type."""
    pass


# ---------------------------------------------------------------------------
# Central Configuration Loader
# ---------------------------------------------------------------------------
def get_project_root() -> Path:
    """Resolve the absolute root path of the repository."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load and parse the project configuration YAML file.

    Args:
        config_path: Optional path to config YAML. Defaults to `config/config.yaml`.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.

    Raises:
        DataNotFoundError: If the configuration file cannot be found.
    """
    if config_path is None:
        resolved_path = get_project_root() / "config" / "config.yaml"
    else:
        resolved_path = Path(config_path)
        if not resolved_path.is_absolute():
            resolved_path = get_project_root() / resolved_path

    if not resolved_path.exists():
        raise DataNotFoundError(
            f"Configuration file not found at: '{resolved_path}'. "
            "Please ensure `config/config.yaml` exists in the repository root."
        )

    with open(resolved_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise SchemaValidationError(f"Invalid config format in '{resolved_path}': Expected dictionary, got {type(config).__name__}")

    return config


# ---------------------------------------------------------------------------
# Schema Validation Engine
# ---------------------------------------------------------------------------
def validate_schema(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Validate DataFrame against the frozen interim schema contract.

    Validation Checks:
        1. Non-empty DataFrame instance.
        2. Strict Data Leakage Guard: `wqi_score` must NOT be in `feature_columns`.
        3. Required target columns exist (`wqi_class`, `Potability`).
        4. Required candidate feature columns exist.
        5. Column data types match expected specifications (numeric for features).
        6. Logs dataset shape, missing value counts per column, and duplicate row count.

    Args:
        df: Input pandas DataFrame to validate.
        config: Optional configuration dictionary. Defaults to loading `config/config.yaml`.

    Returns:
        bool: True if validation succeeds.

    Raises:
        SchemaValidationError: If df is empty or invalid.
        DataLeakageError: If `wqi_score` is present in feature_columns.
        MissingColumnError: If any required feature or target column is missing.
        InvalidDtypeError: If any feature column has an incompatible data type.
    """
    if not isinstance(df, pd.DataFrame):
        raise SchemaValidationError(f"Expected pandas DataFrame, got: {type(df).__name__}")

    if df.empty:
        raise SchemaValidationError("Validation failed: Input DataFrame is empty (0 rows).")

    cfg = config if config is not None else load_config()

    feature_cols: List[str] = cfg.get("feature_columns", [])
    target_col: Optional[str] = cfg.get("target_column", "wqi_class")
    secondary_target_col: Optional[str] = cfg.get("secondary_target_column", "Potability")

    # -----------------------------------------------------------------------
    # 1. Target Leakage Guard Check (TC-1.5-04)
    # -----------------------------------------------------------------------
    for col in feature_cols:
        if "wqi_score" in str(col).lower():
            raise DataLeakageError(
                f"CRITICAL TARGET LEAKAGE GUARD VIOLATION: Column '{col}' is present in `feature_columns`! "
                "`wqi_score` is the exact mathematical value `wqi_class` and `Potability` were computed from. "
                "It MUST NEVER be included in the feature set."
            )

    # -----------------------------------------------------------------------
    # 2. Required Columns Presence Check (TC-1.5-03)
    # -----------------------------------------------------------------------
    required_cols: List[str] = []
    if target_col:
        required_cols.append(target_col)
    if secondary_target_col:
        required_cols.append(secondary_target_col)
    required_cols.extend(feature_cols)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise MissingColumnError(
            f"Schema validation failed: Missing {len(missing_cols)} required column(s): {missing_cols}. "
            f"Required schema columns: {required_cols}"
        )

    # -----------------------------------------------------------------------
    # 3. Data Type Validation Check
    # -----------------------------------------------------------------------
    for feat in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[feat]):
            raise InvalidDtypeError(
                f"Schema validation failed: Feature column '{feat}' has incompatible dtype '{df[feat].dtype}'. "
                "Expected numeric dtype (float64, int64)."
            )

    # -----------------------------------------------------------------------
    # 4. Dataset Audit & Logging
    # -----------------------------------------------------------------------
    total_rows = len(df)
    total_cols = len(df.columns)
    duplicate_rows = int(df.duplicated().sum())

    logger.info("================ DATASET SCHEMA VALIDATION AUDIT ================")
    logger.info("  Dataset Shape: %d rows x %d columns", total_rows, total_cols)
    logger.info("  Duplicate Rows: %d", duplicate_rows)
    logger.info("  Target Column ('%s'): Present (Dtype: %s)", target_col, df[target_col].dtype)
    if secondary_target_col and secondary_target_col in df.columns:
        logger.info("  Secondary Target ('%s'): Present (Dtype: %s)", secondary_target_col, df[secondary_target_col].dtype)

    logger.info("  Candidate Feature Columns Missingness Audit:")
    for feat in feature_cols:
        missing_count = int(df[feat].isna().sum())
        missing_pct = (missing_count / total_rows) * 100
        logger.info("    * %-35s : %4d missing (%5.2f%%)", feat, missing_count, missing_pct)

    logger.info("  Schema Validation Status: PASSED (Frozen Interim Schema Contract Compliant)")
    return True


# ---------------------------------------------------------------------------
# Main Data Ingestion Function
# ---------------------------------------------------------------------------
def load_raw_data(
    path: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load and validate the CPCB labeled interim dataset.

    This function wraps the output of the M1.2/M1.3 join & label pipeline,
    validating the dataset against the frozen schema contract every time it is loaded.

    Args:
        path: Path to dataset CSV file. Defaults to `dataset_path` in `config/config.yaml`
              or `data/interim/cpcb_labeled.csv`.
        config: Optional configuration dictionary.
        validate: Whether to run `validate_schema` after loading (default: True).

    Returns:
        pd.DataFrame: Validated labeled water quality dataset.

    Raises:
        DataNotFoundError: If the dataset file does not exist on disk.
        SchemaValidationError: If the dataset fails schema or leakage validation.
    """
    cfg = config if config is not None else load_config()

    if path is None:
        configured_path = cfg.get("dataset_path", "data/interim/cpcb_labeled.csv")
        target_path = Path(configured_path)
    else:
        target_path = Path(path)

    if not target_path.is_absolute():
        resolved_path = get_project_root() / target_path
    else:
        resolved_path = target_path

    if not resolved_path.exists():
        raise DataNotFoundError(
            f"Dataset file not found at: '{resolved_path}'. "
            "Please ensure the data pipeline has been executed: "
            "`python src/data/join_cpcb.py` followed by `python src/data/compute_wqi.py`."
        )

    logger.info("Loading labeled CPCB dataset from: %s", resolved_path)
    try:
        df = pd.read_csv(resolved_path, encoding="utf-8", encoding_errors="replace")
    except Exception as e:
        raise DataIngestionError(f"Failed to read CSV at '{resolved_path}': {str(e)}") from e

    if validate:
        validate_schema(df, config=cfg)

    return df


def main():
    """CLI entry point for testing dataset loading and schema validation."""
    logger.info("Executing load_raw_data CLI ingestion check...")
    df = load_raw_data()
    print(f"\n[Ingestion Success] Loaded {len(df)} rows, {len(df.columns)} columns.")
    print("Top 5 rows:")
    print(df[["Station", "Potential of Hydrogen (pH)", "Dissolved oxygen (mg/L)", "Biochemical Oxygen Demand (mg/L)", "Fecal Coliform (MPN/100mL)", "wqi_class", "Potability"]].head())


if __name__ == "__main__":
    main()
