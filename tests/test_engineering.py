"""
Unit Test Suite for Milestone 4.1: Feature Engineering
======================================================
Test Cases:
- TC-4.1-01: Engineered features compute correctly (matches hand-calculations)
- TC-4.1-02: Negative leakage check (depends strictly on raw prediction-time inputs)
- TC-4.1-03: Redundancy check (|r| <= 0.85 against all raw features)
"""

import numpy as np
import pandas as pd
import pytest

from src.data.load_data import load_raw_data
from src.data.preprocessing import FROZEN_CANDIDATE_FEATURES
from src.features.engineering import (
    compute_log_fecal_coliform,
    extract_engineered_features,
    WaterQualityFeatureEngineer,
    LOG_FECAL_COLIFORM_COL,
    ENGINEERED_FEATURE_COLUMNS,
    FINAL_EXTENDED_FEATURE_COLUMNS,
)


@pytest.fixture(scope="module")
def labeled_df():
    """Load the interim labeled dataset."""
    return load_raw_data("data/interim/cpcb_labeled.csv")


def test_tc_4_1_01_log_fecal_coliform_hand_calculation():
    """
    TC-4.1-01: Verify compute_log_fecal_coliform against exact hand-calculated test cases.
    """
    # 1. Standard orders of magnitude
    assert np.isclose(compute_log_fecal_coliform(1.0), 0.0)
    assert np.isclose(compute_log_fecal_coliform(10.0), 1.0)
    assert np.isclose(compute_log_fecal_coliform(100.0), 2.0)
    assert np.isclose(compute_log_fecal_coliform(1000.0), 3.0)
    assert np.isclose(compute_log_fecal_coliform(10000.0), 4.0)
    assert np.isclose(compute_log_fecal_coliform(1000000.0), 6.0)
    
    # 2. Intermediate realistic values
    # log10(140.0) = 2.146128
    assert np.isclose(compute_log_fecal_coliform(140.0), np.log10(140.0))
    # log10(22000000.0) = 7.34242268
    assert np.isclose(compute_log_fecal_coliform(22000000.0), np.log10(22000000.0))
    
    # 3. Sub-unit & boundary values (clipping safety)
    assert np.isclose(compute_log_fecal_coliform(0.0), 0.0)
    assert np.isclose(compute_log_fecal_coliform(-5.0), 0.0)
    assert np.isclose(compute_log_fecal_coliform(0.5), 0.0)
    
    # 4. Vectorized pandas Series computation
    series_in = pd.Series([1.0, 100.0, 10000.0, 0.0])
    series_out = compute_log_fecal_coliform(series_in)
    expected_out = pd.Series([0.0, 2.0, 4.0, 0.0])
    pd.testing.assert_series_equal(series_out, expected_out)


def test_tc_4_1_02_leakage_and_prediction_time_availability(labeled_df):
    """
    TC-4.1-02: Negative leakage check.
    Confirm that engineered features require ONLY fields available at dashboard prediction time.
    """
    # 1. Extract only the 4 raw candidate features (simulating user dashboard input)
    raw_input_df = labeled_df[FROZEN_CANDIDATE_FEATURES].copy()
    
    # 2. Extract engineered features
    extended_df = extract_engineered_features(raw_input_df)
    
    # Verify exact column set
    assert list(extended_df.columns) == FINAL_EXTENDED_FEATURE_COLUMNS
    assert len(extended_df.columns) == len(FROZEN_CANDIDATE_FEATURES) + 1
    
    # Verify no target columns or intermediate scores were required
    forbidden_columns = ["wqi_score", "wqi_class", "Potability", "Data Acquisition Time"]
    for forbidden in forbidden_columns:
        assert forbidden not in extended_df.columns
        
    # 3. Verify row-independence (changing row B does not affect row A)
    row_0_orig = extended_df.iloc[0][LOG_FECAL_COLIFORM_COL]
    
    # Perturb row 1 in raw inputs
    perturbed_raw = raw_input_df.copy()
    perturbed_raw.iloc[1, perturbed_raw.columns.get_loc("Fecal Coliform (MPN/100mL)")] = 999999.0
    perturbed_extended = extract_engineered_features(perturbed_raw)
    
    # Row 0 must remain completely unchanged
    row_0_perturbed = perturbed_extended.iloc[0][LOG_FECAL_COLIFORM_COL]
    assert np.isclose(row_0_orig, row_0_perturbed), "Feature computation must be strictly row-independent"


def test_tc_4_1_03_redundancy_check(labeled_df):
    """
    TC-4.1-03: Redundancy check.
    Confirm no engineered feature exceeds |r| > 0.85 with any raw frozen feature.
    """
    raw_df = labeled_df[FROZEN_CANDIDATE_FEATURES].dropna()
    extended_df = extract_engineered_features(raw_df)
    
    corr_matrix = extended_df.corr().abs()
    
    # Check max correlation for Log10 Fecal Coliform against raw features
    fc_log_corrs = corr_matrix.loc[LOG_FECAL_COLIFORM_COL, FROZEN_CANDIDATE_FEATURES]
    max_corr = fc_log_corrs.max()
    max_corr_feature = fc_log_corrs.idxmax()
    
    # Assert non-redundant (threshold 0.85)
    assert max_corr <= 0.85, (
        f"Engineered feature '{LOG_FECAL_COLIFORM_COL}' has correlation {max_corr:.3f} "
        f"with '{max_corr_feature}', exceeding the 0.85 redundancy threshold."
    )
    assert 0.20 <= max_corr <= 0.85


def test_raw_features_preserved_intact(labeled_df):
    """
    Verify that raw features are preserved without mutation or deletion.
    """
    raw_df = labeled_df[FROZEN_CANDIDATE_FEATURES].copy()
    extended_df = extract_engineered_features(raw_df, include_raw_features=True)
    
    for col in FROZEN_CANDIDATE_FEATURES:
        assert col in extended_df.columns
        pd.testing.assert_series_equal(extended_df[col], raw_df[col])


def test_missing_input_column_raises_valueerror(labeled_df):
    """Verify that missing required raw columns raises an explicit ValueError."""
    incomplete_df = labeled_df[["Potential of Hydrogen (pH)", "Dissolved oxygen (mg/L)"]].copy()
    with pytest.raises(ValueError, match="Missing required input column"):
        extract_engineered_features(incomplete_df)


def test_sklearn_feature_engineer_transformer(labeled_df):
    """
    Verify that WaterQualityFeatureEngineer integrates cleanly as a scikit-learn transformer.
    """
    raw_df = labeled_df[FROZEN_CANDIDATE_FEATURES].dropna()
    
    transformer = WaterQualityFeatureEngineer(include_raw_features=True)
    transformer.fit(raw_df)
    
    out_df = transformer.transform(raw_df)
    assert isinstance(out_df, pd.DataFrame)
    assert list(out_df.columns) == FINAL_EXTENDED_FEATURE_COLUMNS
    assert len(out_df) == len(raw_df)
    
    # Feature names out check
    feature_names = transformer.get_feature_names_out()
    assert list(feature_names) == FINAL_EXTENDED_FEATURE_COLUMNS
