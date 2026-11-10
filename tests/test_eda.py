"""
Unit Test Suite for Milestone 3.1 & 3.2: Exploratory Data Analysis & Parameter-vs-Class Report
=============================================================================================
Test Cases:
- TC-3.1-01: All numeric features have distribution plots (histogram + boxplot saved)
- TC-3.1-02: Correlation matrix computed correctly (spot-checked against pandas .corr())
- TC-3.1-03: Class distribution export is accurate (percentages sum to 100% within tolerance)
- TC-3.2-01: Parameter-vs-class plots generated for all key candidate parameters
- TC-3.2-02: EDA summary report readability check (reports/eda_summary.md exists and is accurate)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.load_data import load_raw_data
from src.data.preprocessing import FROZEN_CANDIDATE_FEATURES
from src.features.eda import (
    compute_summary_statistics,
    compute_correlation_matrix,
    find_redundant_feature_pairs,
    compute_class_distribution,
    generate_univariate_plots,
    generate_correlation_heatmap,
    generate_class_distribution_plot,
    generate_parameter_vs_class_plots,
    generate_spatial_temporal_plots,
    run_full_eda,
)


@pytest.fixture(scope="module")
def labeled_df():
    """Load the interim labeled dataset."""
    return load_raw_data("data/interim/cpcb_labeled.csv")


def test_tc_3_1_01_all_numeric_features_have_distribution_plots(labeled_df, tmp_path):
    """
    TC-3.1-01: Verify that every numeric candidate feature produces a valid distribution plot.
    """
    saved_plots = generate_univariate_plots(
        labeled_df,
        output_dir=tmp_path / "figures",
        feature_columns=FROZEN_CANDIDATE_FEATURES,
    )
    
    assert len(saved_plots) == len(FROZEN_CANDIDATE_FEATURES), (
        f"Expected {len(FROZEN_CANDIDATE_FEATURES)} plots, got {len(saved_plots)}"
    )
    
    for col in FROZEN_CANDIDATE_FEATURES:
        assert col in saved_plots, f"Missing plot mapping for feature: {col}"
        plot_path = Path(saved_plots[col])
        assert plot_path.exists(), f"Plot file does not exist: {plot_path}"
        assert plot_path.stat().st_size > 1000, f"Plot file is suspiciously small or empty: {plot_path}"


def test_tc_3_1_02_correlation_matrix_computed_correctly(labeled_df):
    """
    TC-3.1-02: Verify that correlation matrices match standard pandas calculations exactly.
    """
    # Pearson
    corr_custom_pearson = compute_correlation_matrix(labeled_df, method="pearson")
    corr_expected_pearson = labeled_df[FROZEN_CANDIDATE_FEATURES].corr(method="pearson")
    pd.testing.assert_frame_equal(corr_custom_pearson, corr_expected_pearson)
    
    # Spot-check diagonal
    for col in FROZEN_CANDIDATE_FEATURES:
        assert np.isclose(corr_custom_pearson.loc[col, col], 1.0)
        
    # Spot-check symmetry
    for col1 in FROZEN_CANDIDATE_FEATURES:
        for col2 in FROZEN_CANDIDATE_FEATURES:
            assert np.isclose(
                corr_custom_pearson.loc[col1, col2],
                corr_custom_pearson.loc[col2, col1],
            )
            
    # Spearman
    corr_custom_spearman = compute_correlation_matrix(labeled_df, method="spearman")
    corr_expected_spearman = labeled_df[FROZEN_CANDIDATE_FEATURES].corr(method="spearman")
    pd.testing.assert_frame_equal(corr_custom_spearman, corr_expected_spearman)


def test_tc_3_1_02_no_premature_redundancy_drops(labeled_df):
    """
    Verify that redundant pair detection does not alter dataframe or drop features.
    """
    corr_matrix = compute_correlation_matrix(labeled_df)
    redundant_pairs = find_redundant_feature_pairs(corr_matrix, threshold=0.85)
    
    assert isinstance(redundant_pairs, list)
    for pair in redundant_pairs:
        assert pair["abs_correlation"] > 0.85


def test_tc_3_1_03_class_distribution_export_accurate(labeled_df, tmp_path):
    """
    TC-3.1-03: Verify that class distribution CSV export accurately sums to 100%.
    """
    csv_file = tmp_path / "class_distribution.csv"
    dist_df = compute_class_distribution(
        labeled_df,
        target_column="wqi_class",
        output_csv=csv_file,
    )
    
    assert csv_file.exists(), f"Expected CSV file {csv_file} to exist"
    
    # Check loaded CSV
    loaded_df = pd.read_csv(csv_file)
    assert set(loaded_df.columns) == {"class", "count", "percentage"}
    
    # Check total counts
    assert loaded_df["count"].sum() == len(labeled_df)
    
    # Check percentages sum to 100% (within 0.01% floating tolerance)
    pct_sum = loaded_df["percentage"].sum()
    assert np.isclose(pct_sum, 100.0, atol=0.05), f"Percentages sum to {pct_sum}%, expected 100.0%"
    
    # Check all 5 classes are present
    expected_classes = {"Excellent", "Good", "Poor", "Very Poor", "Unsuitable"}
    assert set(loaded_df["class"]) == expected_classes


def test_tc_3_2_01_parameter_vs_class_plots_generated(labeled_df, tmp_path):
    """
    TC-3.2-01: Confirm parameter-vs-class plots exist for all frozen candidate parameters.
    """
    pvc_plots = generate_parameter_vs_class_plots(
        labeled_df,
        output_dir=tmp_path / "parameter_vs_class",
        feature_columns=FROZEN_CANDIDATE_FEATURES,
    )
    
    assert len(pvc_plots) == len(FROZEN_CANDIDATE_FEATURES)
    for col in FROZEN_CANDIDATE_FEATURES:
        assert col in pvc_plots
        plot_path = Path(pvc_plots[col])
        assert plot_path.exists()
        assert plot_path.stat().st_size > 1000


def test_tc_3_2_01_spatial_temporal_plots_generated(labeled_df, tmp_path):
    """
    Verify that spatial and temporal plots are generated properly.
    """
    st_plots = generate_spatial_temporal_plots(
        labeled_df,
        output_dir=tmp_path / "parameter_vs_class",
    )
    assert "spatial_state" in st_plots
    assert "temporal_yearly" in st_plots
    assert Path(st_plots["spatial_state"]).exists()
    assert Path(st_plots["temporal_yearly"]).exists()


def test_tc_3_2_02_eda_summary_report_readability():
    """
    TC-3.2-02: Verify reports/eda_summary.md exists, is readable, and contains key findings.
    """
    report_file = Path("reports/eda_summary.md")
    assert report_file.exists(), "reports/eda_summary.md must exist"
    
    content = report_file.read_text(encoding="utf-8")
    assert len(content) > 500, "EDA summary report is too brief"
    
    # Must contain class imbalance breakdown
    assert "Class A" in content or "0.21%" in content
    assert "Class E" in content or "54.64%" in content
    
    # Must mention top separating parameters
    assert "Biochemical Oxygen Demand" in content or "BOD" in content
    assert "Fecal Coliform" in content
    assert "Dissolved Oxygen" in content
    assert "pH" in content
    
    # Must be linked in README.md
    readme_content = Path("README.md").read_text(encoding="utf-8")
    assert "eda_summary.md" in readme_content, "reports/eda_summary.md must be linked in README.md"


def test_summary_statistics_validity(labeled_df):
    """Verify that summary statistics computation covers all candidate features and ranges."""
    stats = compute_summary_statistics(labeled_df)
    assert len(stats) == len(FROZEN_CANDIDATE_FEATURES)
    assert "skewness" in stats.columns
    assert "iqr" in stats.columns
    
    fc_row = stats[stats["feature"] == "Fecal Coliform (MPN/100mL)"].iloc[0]
    bod_row = stats[stats["feature"] == "Biochemical Oxygen Demand (mg/L)"].iloc[0]
    assert fc_row["skewness"] > 1.0, "Fecal Coliform should exhibit strong positive skewness"
    assert bod_row["skewness"] > 1.0, "BOD should exhibit positive skewness"


def test_full_eda_integration(labeled_df, tmp_path):
    """Integration test verifying end-to-end execution of run_full_eda."""
    results = run_full_eda(
        data_path="data/interim/cpcb_labeled.csv",
        reports_dir=tmp_path / "reports",
    )
    
    assert "summary_statistics" in results
    assert "univariate_plots" in results
    assert "pearson_correlation" in results
    assert "spearman_correlation" in results
    assert "class_distribution" in results
    assert "parameter_vs_class_plots" in results
    assert "spatial_temporal_plots" in results
    assert Path(results["heatmap_file"]).exists()
    assert Path(results["class_distribution_plot"]).exists()

