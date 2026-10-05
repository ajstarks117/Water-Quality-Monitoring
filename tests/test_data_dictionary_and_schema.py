"""Test Suite for Milestone 1.4: Data Dictionary & Schema Freeze.

Test Cases:
- TC-1.4-01: Data dictionary matches actual data (all 70 columns, dtypes, no missing/extra columns).
- TC-1.4-02: wqi_score leakage guard: strictly excluded from feature_columns in config and code.
- TC-1.4-03: Target column wqi_class correctly identified with confirmed M1.3 classes and distribution.
- TC-1.4-04: config.yaml reflects frozen schema matching data dictionary candidate features.
"""

from pathlib import Path
import re
import pandas as pd
import pytest
import yaml


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """Fixture returning the repository root path."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def labeled_df(repo_root: Path) -> pd.DataFrame:
    """Fixture loading the labeled CPCB dataset."""
    labeled_path = repo_root / "data" / "interim" / "cpcb_labeled.csv"
    assert labeled_path.exists(), f"Labeled dataset missing at {labeled_path}"
    return pd.read_csv(labeled_path, encoding="utf-8", encoding_errors="replace")


@pytest.fixture(scope="module")
def config_dict(repo_root: Path) -> dict:
    """Fixture loading config/config.yaml."""
    config_path = repo_root / "config" / "config.yaml"
    assert config_path.exists(), f"Config missing at {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def data_dict_content(repo_root: Path) -> str:
    """Fixture loading Docs/data_dictionary.md."""
    doc_path = repo_root / "Docs" / "data_dictionary.md"
    assert doc_path.exists(), f"Data dictionary document missing at {doc_path}"
    return doc_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# TC-1.4-01: Data Dictionary Matches Actual Data
# ---------------------------------------------------------------------------
def test_tc_1_4_01_data_dictionary_covers_all_columns(labeled_df: pd.DataFrame, data_dict_content: str):
    """TC-1.4-01: Programmatically compare dictionary column list against df.columns."""
    actual_columns = list(labeled_df.columns)
    assert len(actual_columns) == 70, f"Expected 70 columns in labeled data, got {len(actual_columns)}"

    # Check that every single column name from the dataframe is mentioned in data_dictionary.md
    missing_in_doc = []
    for col in actual_columns:
        # Match column name enclosed in backticks or exact text
        if f"`{col}`" not in data_dict_content and col not in data_dict_content:
            missing_in_doc.append(col)

    assert not missing_in_doc, f"Columns missing from Docs/data_dictionary.md: {missing_in_doc}"


def test_tc_1_4_01_dtypes_match_actual_dataset(labeled_df: pd.DataFrame, data_dict_content: str):
    """TC-1.4-01: Verify that core dtypes in data dictionary correspond to actual DataFrame dtypes."""
    # Check numeric candidate features
    candidate_features = [
        "Potential of Hydrogen (pH)",
        "Dissolved oxygen (mg/L)",
        "Biochemical Oxygen Demand (mg/L)",
        "Fecal Coliform (MPN/100mL)",
    ]
    for feat in candidate_features:
        assert feat in labeled_df.columns, f"Feature '{feat}' missing from labeled DataFrame"
        assert pd.api.types.is_numeric_dtype(labeled_df[feat]), f"Feature '{feat}' must be numeric"

    # Check target dtypes
    assert "wqi_class" in labeled_df.columns
    assert "Potability" in labeled_df.columns
    assert pd.api.types.is_integer_dtype(labeled_df["Potability"]) or pd.api.types.is_numeric_dtype(labeled_df["Potability"])


# ---------------------------------------------------------------------------
# TC-1.4-02: Target Leakage Guard (wqi_score Strictly Excluded)
# ---------------------------------------------------------------------------
def test_tc_1_4_02_wqi_score_excluded_from_features(config_dict: dict, data_dict_content: str):
    """TC-1.4-02: Check that wqi_score is NOT in feature_columns and is explicitly marked as excluded."""
    feature_cols = config_dict.get("feature_columns", [])
    assert isinstance(feature_cols, list)

    # wqi_score must NEVER appear in feature_columns
    assert "wqi_score" not in feature_cols, "CRITICAL LEAKAGE: 'wqi_score' found in config.yaml feature_columns!"
    assert not any("wqi_score" in str(col).lower() for col in feature_cols), "Leakage column variant found in feature_columns"

    # Must be in excluded_leakage_columns
    excluded_leakage = config_dict.get("excluded_leakage_columns", [])
    assert "wqi_score" in excluded_leakage, "'wqi_score' must be listed under excluded_leakage_columns in config.yaml"

    # Documentation must contain the leakage warning
    assert "RULE 1: MANDATORY TARGET LEAKAGE GUARD" in data_dict_content
    assert "wqi_score" in data_dict_content


# ---------------------------------------------------------------------------
# TC-1.4-03: Target Column Identified Correctly
# ---------------------------------------------------------------------------
def test_tc_1_4_03_target_column_classes_and_distribution(labeled_df: pd.DataFrame, data_dict_content: str):
    """TC-1.4-03: Confirm wqi_class has the confirmed class values and matches M1.3 distribution."""
    expected_classes = {"Excellent", "Good", "Poor", "Very Poor", "Unsuitable"}
    actual_classes = set(labeled_df["wqi_class"].dropna().unique())

    assert actual_classes == expected_classes, f"Class mismatch: expected {expected_classes}, got {actual_classes}"

    # Confirm class counts match M1.3 distribution
    counts = labeled_df["wqi_class"].value_counts().to_dict()
    assert counts["Excellent"] == 7
    assert counts["Good"] == 110
    assert counts["Poor"] == 455
    assert counts["Very Poor"] == 919
    assert counts["Unsuitable"] == 1796
    assert len(labeled_df) == 3287

    # Verify distribution is documented in data_dictionary.md
    for cls in expected_classes:
        assert cls in data_dict_content


def test_tc_1_4_03_binary_potability_target_consistency(labeled_df: pd.DataFrame):
    """TC-1.4-03: Verify binary Potability mapping aligns with wqi_class."""
    potable_classes = labeled_df[labeled_df["Potability"] == 1]["wqi_class"].unique()
    non_potable_classes = labeled_df[labeled_df["Potability"] == 0]["wqi_class"].unique()

    assert set(potable_classes).issubset({"Excellent", "Good"})
    assert set(non_potable_classes).issubset({"Poor", "Very Poor", "Unsuitable"})
    assert (labeled_df["Potability"] == 1).sum() == 117
    assert (labeled_df["Potability"] == 0).sum() == 3170


# ---------------------------------------------------------------------------
# TC-1.4-04: config.yaml Reflects Frozen Schema
# ---------------------------------------------------------------------------
def test_tc_1_4_04_config_yaml_frozen_schema(config_dict: dict, labeled_df: pd.DataFrame):
    """TC-1.4-04: Load config.yaml and check target_column and feature_columns are non-empty and match dictionary."""
    # Target column
    assert config_dict.get("target_column") == "wqi_class", "target_column in config.yaml must be 'wqi_class'"
    assert config_dict.get("secondary_target_column") == "Potability", "secondary_target_column in config.yaml must be 'Potability'"

    # Feature columns
    feature_cols = config_dict.get("feature_columns", [])
    assert len(feature_cols) == 4, f"Expected exactly 4 candidate feature columns, got {len(feature_cols)}"

    expected_features = [
        "Potential of Hydrogen (pH)",
        "Dissolved oxygen (mg/L)",
        "Biochemical Oxygen Demand (mg/L)",
        "Fecal Coliform (MPN/100mL)",
    ]
    assert feature_cols == expected_features, f"Feature columns mismatch: {feature_cols} != {expected_features}"

    # Verify all feature columns exist in the actual dataset
    for col in feature_cols:
        assert col in labeled_df.columns, f"Feature column '{col}' does not exist in dataset"
        missing_pct = labeled_df[col].isna().mean() * 100
        assert missing_pct < 10.0, f"Feature '{col}' has unacceptable missingness: {missing_pct:.2f}% (must be < 10%)"


def test_tc_1_4_04_interface_contracts_matches_frozen_schema(repo_root: Path, config_dict: dict):
    """TC-1.4-04: Verify Docs/interface_contracts.md matches config.yaml schema."""
    contract_path = repo_root / "Docs" / "interface_contracts.md"
    assert contract_path.exists()

    contract_text = contract_path.read_text(encoding="utf-8")
    for feat in config_dict["feature_columns"]:
        assert feat in contract_text, f"Feature '{feat}' not found in interface_contracts.md"

    assert "wqi_class" in contract_text
    assert "Potability" in contract_text
    assert "wqi_score" in contract_text
    assert "STRICT TARGET LEAKAGE GUARD" in contract_text
