"""Unit Test Suite for Milestone 2.1: Missing Value & Duplicate Handling (preprocessing.py).

Test Cases:
    - TC-2.1-01: Missing value strategy applied correctly (zero missing in candidate features, dropped columns match threshold).
    - TC-2.1-02: Duplicate rows handled per decision (exact row count reduction on duplicate presence).
    - TC-2.1-03: Negative: All-missing column automatically dropped with logged warning.
    - TC-2.1-04: Regression: Row count sanity (no unexplained row loss between raw and cleaned).
    - Additional: Train-fit / test-transform leakage safety, alternate imputation strategies, protected column handling.
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import pytest

from src.data.load_data import load_raw_data, load_config
from src.data.preprocessing import (
    handle_duplicates,
    handle_missing_values,
    handle_outliers,
    PreprocessingState,
    OutlierAuditReport,
    PHYSICAL_BOUNDS,
)


@pytest.fixture(scope="module")
def labeled_df() -> pd.DataFrame:
    """Fixture loading the raw labeled dataset."""
    return load_raw_data(validate=True)


@pytest.fixture(scope="module")
def candidate_features() -> list:
    """Fixture listing the 4 frozen candidate feature names."""
    return [
        "Potential of Hydrogen (pH)",
        "Dissolved oxygen (mg/L)",
        "Biochemical Oxygen Demand (mg/L)",
        "Fecal Coliform (MPN/100mL)",
    ]


# ---------------------------------------------------------------------------
# TC-2.1-01: Missing Value Strategy Applied Correctly
# ---------------------------------------------------------------------------
def test_tc_2_1_01_missing_value_strategy_applied_correctly(labeled_df: pd.DataFrame, candidate_features: list):
    """TC-2.1-01: Verify zero missing values in active features and expected dropped columns."""
    df_clean, state = handle_missing_values(labeled_df, strategy_config={"max_missing_col_pct": 40.0})

    # Zero missing values in candidate features
    for feat in candidate_features:
        assert feat in df_clean.columns, f"Active feature '{feat}' should be retained"
        assert df_clean[feat].isna().sum() == 0, f"Feature '{feat}' has remaining null values"

    # Exactly 39 sparse columns dropped (>40% missingness)
    assert len(state.dropped_columns) == 39, f"Expected 39 dropped columns, got {len(state.dropped_columns)}"
    for col in state.dropped_columns:
        assert col not in df_clean.columns, f"Dropped column '{col}' should not be in cleaned DataFrame"

    # Verify imputation values used are realistic
    assert state.imputation_values["Dissolved oxygen (mg/L)"] > 0
    assert state.imputation_values["Biochemical Oxygen Demand (mg/L)"] > 0
    assert state.imputation_values["Fecal Coliform (MPN/100mL)"] > 0


# ---------------------------------------------------------------------------
# TC-2.1-02: Duplicate Rows Handled Per Decision
# ---------------------------------------------------------------------------
def test_tc_2_1_02_duplicate_rows_handled_per_decision(labeled_df: pd.DataFrame):
    """TC-2.1-02: Verify duplicate row behavior on clean data and on injected synthetic duplicates."""
    # 1. Clean dataset has 0 duplicates
    df_dedup = handle_duplicates(labeled_df)
    assert len(df_dedup) == len(labeled_df), "Zero duplicates expected in official labeled data"

    # 2. Inject 5 duplicate rows
    duplicate_rows = labeled_df.iloc[:5].copy()
    df_with_duplicates = pd.concat([labeled_df, duplicate_rows], ignore_index=True)
    assert len(df_with_duplicates) == len(labeled_df) + 5

    # Run deduplication
    df_resolved = handle_duplicates(df_with_duplicates)
    assert len(df_resolved) == len(labeled_df), f"Expected {len(labeled_df)} rows after deduplication, got {len(df_resolved)}"


# ---------------------------------------------------------------------------
# TC-2.1-03: Negative Path — All-Missing Column Handling
# ---------------------------------------------------------------------------
def test_tc_2_1_03_all_missing_column_dropped(candidate_features: list, caplog):
    """TC-2.1-03: Synthetic DataFrame with a 100%-missing column is automatically dropped with a warning."""
    df_synthetic = pd.DataFrame({
        "Potential of Hydrogen (pH)": [7.0, 7.5, 8.0, 6.8],
        "Dissolved oxygen (mg/L)": [6.5, 5.8, 7.2, np.nan],
        "Biochemical Oxygen Demand (mg/L)": [2.0, np.nan, 3.1, 1.8],
        "Fecal Coliform (MPN/100mL)": [50.0, 100.0, 25.0, 80.0],
        "all_missing_contaminant": [np.nan, np.nan, np.nan, np.nan],
        "wqi_class": ["Good", "Good", "Excellent", "Good"],
        "Potability": [1, 1, 1, 1],
    })

    with caplog.at_level(logging.WARNING, logger="preprocessing"):
        df_clean, state = handle_missing_values(df_synthetic, strategy_config={"max_missing_col_pct": 40.0})

    assert "all_missing_contaminant" in state.dropped_columns
    assert "all_missing_contaminant" not in df_clean.columns
    assert "Dropping sparse column 'all_missing_contaminant'" in caplog.text


# ---------------------------------------------------------------------------
# TC-2.1-04: Regression — Row Count Sanity
# ---------------------------------------------------------------------------
def test_tc_2_1_04_regression_row_count_sanity(labeled_df: pd.DataFrame):
    """TC-2.1-04: Ensure no unexplained row loss occurs during missing value imputation."""
    raw_row_count = len(labeled_df)
    df_clean, state = handle_missing_values(labeled_df)

    assert len(df_clean) == raw_row_count, f"Row count changed unexpectedly: {raw_row_count} -> {len(df_clean)}"
    assert state.final_shape[0] == raw_row_count


# ---------------------------------------------------------------------------
# Additional Preprocessing & Leakage-Safety Tests
# ---------------------------------------------------------------------------
def test_train_test_imputation_leakage_safety():
    """Verify that test data is transformed using training statistics without recalculation."""
    # Training split with known medians
    train_df = pd.DataFrame({
        "Dissolved oxygen (mg/L)": [5.0, 7.0, 9.0],  # Median = 7.0
        "Biochemical Oxygen Demand (mg/L)": [2.0, 4.0, 6.0],  # Median = 4.0
    })

    # Test split with different values and missing entries
    test_df = pd.DataFrame({
        "Dissolved oxygen (mg/L)": [np.nan, 100.0, 200.0],  # Test median would be 150 if calculated!
        "Biochemical Oxygen Demand (mg/L)": [np.nan, 50.0, 60.0],
    })

    # Fit on train
    train_clean, state = handle_missing_values(train_df, strategy_config={"feature_columns": list(train_df.columns)})
    assert state.imputation_values["Dissolved oxygen (mg/L)"] == 7.0
    assert state.imputation_values["Biochemical Oxygen Demand (mg/L)"] == 4.0

    # Transform on test using fitted state
    test_clean, test_state = handle_missing_values(test_df, fitted_state=state)

    # Missing value in test must be filled with TRAIN median (7.0), NOT test median
    assert test_clean["Dissolved oxygen (mg/L)"].iloc[0] == 7.0
    assert test_clean["Biochemical Oxygen Demand (mg/L)"].iloc[0] == 4.0


def test_drop_rows_strategy():
    """Verify that 'drop_rows' strategy properly removes rows with missing values."""
    df_sample = pd.DataFrame({
        "Potential of Hydrogen (pH)": [7.0, 7.5, 8.0],
        "Dissolved oxygen (mg/L)": [6.0, np.nan, 8.0],
        "Biochemical Oxygen Demand (mg/L)": [2.0, 3.0, 4.0],
        "Fecal Coliform (MPN/100mL)": [10.0, 20.0, 30.0],
    })

    df_clean, state = handle_missing_values(
        df_sample,
        strategy_config={"imputation_strategy": "drop_rows", "feature_columns": list(df_sample.columns)},
    )

    assert len(df_clean) == 2
    assert df_clean["Dissolved oxygen (mg/L)"].isna().sum() == 0


def test_protected_columns_not_dropped():
    """Verify that protected columns are never dropped regardless of missingness."""
    df_sample = pd.DataFrame({
        "wqi_class": ["Good", np.nan, "Poor"],  # Protected target
        "sparse_unprotected": [np.nan, np.nan, 1.0],  # 66% missing
    })

    df_clean, state = handle_missing_values(
        df_sample,
        strategy_config={"max_missing_col_pct": 50.0, "protected_columns": ["wqi_class"]},
    )

    assert "wqi_class" in df_clean.columns
    assert "sparse_unprotected" not in df_clean.columns
    assert "sparse_unprotected" in state.dropped_columns


# ---------------------------------------------------------------------------
# TC-2.2-01: Physically Impossible Values Removed
# ---------------------------------------------------------------------------
def test_tc_2_2_01_physically_impossible_values_removed(candidate_features: list, caplog):
    """TC-2.2-01: Synthetic row with pH = 15.0 or negative concentration is removed as invalid error."""
    df_with_invalid = pd.DataFrame({
        "Potential of Hydrogen (pH)": [7.0, 15.0, 7.5, 6.8],  # 15.0 is physically impossible (>14)
        "Dissolved oxygen (mg/L)": [6.5, 5.8, -1.0, 7.2],      # -1.0 is physically impossible (<0)
        "Biochemical Oxygen Demand (mg/L)": [2.0, 3.5, 4.0, 1.8],
        "Fecal Coliform (MPN/100mL)": [50.0, 100.0, 25.0, 80.0],
        "wqi_class": ["Good", "Poor", "Very Poor", "Good"],
        "Potability": [1, 0, 0, 1],
    })

    with caplog.at_level(logging.WARNING, logger="preprocessing"):
        df_clean, report = handle_outliers(df_with_invalid, remove_invalid=True)

    # Both row 1 (pH=15) and row 2 (DO=-1) must be removed
    assert report.invalid_rows_removed == 2
    assert len(df_clean) == 2
    assert "INVALID PHYSICAL READINGS" in caplog.text


# ---------------------------------------------------------------------------
# TC-2.2-02: Genuine Extreme Readings Retained
# ---------------------------------------------------------------------------
def test_tc_2_2_02_genuine_extreme_values_retained(labeled_df: pd.DataFrame):
    """TC-2.2-02: High-but-plausible BOD and extreme Fecal Coliform values are retained as genuine signals."""
    df_clean, report = handle_outliers(labeled_df, remove_invalid=True)

    # In official data, 100% of samples are within physical bounds
    assert report.invalid_rows_removed == 0
    assert len(df_clean) == len(labeled_df)

    # Max BOD (127.0 mg/L) and Max FC (2.2e7 MPN/100mL) must be retained
    assert df_clean["Biochemical Oxygen Demand (mg/L)"].max() == 127.0
    assert df_clean["Fecal Coliform (MPN/100mL)"].max() == 22000000.0
    assert report.statistical_outliers_by_feature["Biochemical Oxygen Demand (mg/L)"] > 0
    assert report.statistical_outliers_by_feature["Fecal Coliform (MPN/100mL)"] > 0


# ---------------------------------------------------------------------------
# TC-2.2-03: Class Balance Check Post-Outlier-Handling
# ---------------------------------------------------------------------------
def test_tc_2_2_03_class_balance_check_post_outliers(labeled_df: pd.DataFrame):
    """TC-2.2-03: Compare class distribution before and after handle_outliers; ensure no disproportionate loss."""
    df_clean, report = handle_outliers(labeled_df, rules_config={"max_allowed_class_loss_pct": 10.0})

    assert report.is_class_imbalance_severely_distorted is False
    for cls, loss_pct in report.class_loss_percentages.items():
        assert loss_pct == 0.0, f"Class '{cls}' suffered unexpected row loss: {loss_pct}%"


def test_boxplots_artifacts_exist():
    """Verify that outlier box plot figures were generated in reports/figures/outlier_boxplots/."""
    from src.data.load_data import get_project_root

    boxplots_dir = get_project_root() / "reports" / "figures" / "outlier_boxplots"
    assert boxplots_dir.exists(), "Boxplots directory missing"

    expected_files = [
        "candidate_features_boxplots.png",
        "potential_of_hydrogen_by_class_boxplot.png",
        "dissolved_oxygen_by_class_boxplot.png",
        "biochemical_oxygen_demand_by_class_boxplot.png",
        "fecal_coliform_by_class_boxplot.png",
    ]

    for fname in expected_files:
        p = boxplots_dir / fname
        assert p.exists(), f"Missing boxplot figure: {fname}"
        assert p.stat().st_size > 1000, f"Boxplot figure '{fname}' appears empty"
