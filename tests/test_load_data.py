"""Unit Test Suite for Milestone 1.5: Raw Data Ingestion Module (load_data.py).

Test Cases:
    - TC-1.5-01: Happy path: Load valid labeled dataset returning expected shape & dtypes.
    - TC-1.5-02: Negative: Missing file raises DataNotFoundError / FileNotFoundError with clear message.
    - TC-1.5-03: Negative: Missing required column raises MissingColumnError naming the missing column.
    - TC-1.5-04: Negative: Target leakage guard: wqi_score in feature_columns raises DataLeakageError.
    - Additional: Invalid dtypes, empty DataFrames, logging output audits.
"""

import logging
from pathlib import Path
import pytest
import pandas as pd

from src.data.load_data import (
    load_raw_data,
    validate_schema,
    load_config,
    DataNotFoundError,
    SchemaValidationError,
    MissingColumnError,
    DataLeakageError,
    InvalidDtypeError,
    get_project_root,
)


@pytest.fixture(scope="module")
def valid_labeled_csv_path() -> Path:
    """Fixture returning path to valid labeled CPCB dataset."""
    root = get_project_root()
    path = root / "data" / "interim" / "cpcb_labeled.csv"
    assert path.exists(), f"Labeled test dataset missing at {path}"
    return path


@pytest.fixture(scope="module")
def base_config() -> dict:
    """Fixture returning the standard parsed config dictionary."""
    return load_config()


# ---------------------------------------------------------------------------
# TC-1.5-01: Happy Path — Ingest Valid Labeled Dataset
# ---------------------------------------------------------------------------
def test_tc_1_5_01_load_valid_labeled_dataset(valid_labeled_csv_path: Path, base_config: dict):
    """TC-1.5-01: Call load_raw_data(path) and verify returned DataFrame matches frozen contract."""
    df = load_raw_data(valid_labeled_csv_path, config=base_config, validate=True)

    assert isinstance(df, pd.DataFrame), "load_raw_data must return a pandas DataFrame"
    assert len(df) == 3287, f"Expected 3,287 records, got {len(df)}"
    assert len(df.columns) == 70, f"Expected 70 columns, got {len(df.columns)}"

    # Verify all feature columns are present and numeric
    for feat in base_config["feature_columns"]:
        assert feat in df.columns, f"Candidate feature '{feat}' missing from loaded dataset"
        assert pd.api.types.is_numeric_dtype(df[feat]), f"Feature '{feat}' must have numeric dtype"

    # Verify target columns
    assert base_config["target_column"] in df.columns
    assert base_config["secondary_target_column"] in df.columns


def test_tc_1_5_01_load_raw_data_default_path(base_config: dict):
    """TC-1.5-01b: Call load_raw_data() with default path parameter."""
    df = load_raw_data(validate=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


# ---------------------------------------------------------------------------
# TC-1.5-02: Negative Path — Nonexistent File Handling
# ---------------------------------------------------------------------------
def test_tc_1_5_02_missing_file_raises_clear_exception():
    """TC-1.5-02: Call load_raw_data(bad_path) and verify clear DataNotFoundError is raised."""
    bad_path = "data/nonexistent_folder/missing_water_dataset.csv"

    with pytest.raises(DataNotFoundError) as exc_info:
        load_raw_data(path=bad_path)

    error_message = str(exc_info.value)
    assert "Dataset file not found at" in error_message
    assert "missing_water_dataset.csv" in error_message


def test_tc_1_5_02_missing_config_file_raises_exception():
    """TC-1.5-02b: Call load_config(bad_config_path) and verify DataNotFoundError."""
    bad_config = "config/nonexistent_config.yaml"

    with pytest.raises(DataNotFoundError) as exc_info:
        load_config(config_path=bad_config)

    assert "Configuration file not found at" in str(exc_info.value)


# ---------------------------------------------------------------------------
# TC-1.5-03: Negative Path — Missing Required Column
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "missing_col",
    [
        "Potential of Hydrogen (pH)",
        "Dissolved oxygen (mg/L)",
        "Biochemical Oxygen Demand (mg/L)",
        "Fecal Coliform (MPN/100mL)",
        "wqi_class",
        "Potability",
    ],
)
def test_tc_1_5_03_missing_required_column_raises_error(
    valid_labeled_csv_path: Path, base_config: dict, missing_col: str
):
    """TC-1.5-03: Drop a required column and verify validate_schema raises MissingColumnError naming it."""
    df = pd.read_csv(valid_labeled_csv_path)
    assert missing_col in df.columns

    # Drop the column
    df_corrupted = df.drop(columns=[missing_col])

    with pytest.raises(MissingColumnError) as exc_info:
        validate_schema(df_corrupted, config=base_config)

    error_message = str(exc_info.value)
    assert "Schema validation failed: Missing" in error_message
    assert missing_col in error_message


# ---------------------------------------------------------------------------
# TC-1.5-04: Negative Path — Data Leakage Guard (wqi_score in Features)
# ---------------------------------------------------------------------------
def test_tc_1_5_04_wqi_score_leakage_raises_explicit_exception(
    valid_labeled_csv_path: Path, base_config: dict
):
    """TC-1.5-04: Add wqi_score to feature_columns and verify validate_schema raises DataLeakageError."""
    df = pd.read_csv(valid_labeled_csv_path)

    # Corrupt config by adding wqi_score as a feature
    leaky_config = dict(base_config)
    leaky_config["feature_columns"] = list(base_config["feature_columns"]) + ["wqi_score"]

    with pytest.raises(DataLeakageError) as exc_info:
        validate_schema(df, config=leaky_config)

    error_message = str(exc_info.value)
    assert "CRITICAL TARGET LEAKAGE GUARD VIOLATION" in error_message
    assert "wqi_score" in error_message


# ---------------------------------------------------------------------------
# Additional Edge Case & Ingestion Integrity Tests
# ---------------------------------------------------------------------------
def test_invalid_dtype_for_feature_raises_error(valid_labeled_csv_path: Path, base_config: dict):
    """Verify that non-numeric data types in numeric feature columns trigger InvalidDtypeError."""
    df = pd.read_csv(valid_labeled_csv_path)

    # Corrupt a numeric feature column to string
    df["Potential of Hydrogen (pH)"] = df["Potential of Hydrogen (pH)"].astype(str) + "_invalid"

    with pytest.raises(InvalidDtypeError) as exc_info:
        validate_schema(df, config=base_config)

    error_message = str(exc_info.value)
    assert "Potential of Hydrogen (pH)" in error_message
    assert "incompatible dtype" in error_message


def test_empty_dataframe_raises_schema_validation_error(base_config: dict):
    """Verify that an empty DataFrame triggers SchemaValidationError."""
    empty_df = pd.DataFrame()

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_schema(empty_df, config=base_config)

    assert "Input DataFrame is empty" in str(exc_info.value)


def test_logging_audit_output(valid_labeled_csv_path: Path, base_config: dict, caplog):
    """Verify that validate_schema logs dataset shape, missingness per column, and duplicate row count."""
    df = pd.read_csv(valid_labeled_csv_path)

    with caplog.at_level(logging.INFO, logger="load_data"):
        result = validate_schema(df, config=base_config)

    assert result is True
    log_text = caplog.text

    # Verify audit sections in logs
    assert "DATASET SCHEMA VALIDATION AUDIT" in log_text
    assert "Dataset Shape: 3287 rows x 70 columns" in log_text
    assert "Duplicate Rows:" in log_text
    assert "Candidate Feature Columns Missingness Audit:" in log_text
    for feat in base_config["feature_columns"]:
        assert feat in log_text
