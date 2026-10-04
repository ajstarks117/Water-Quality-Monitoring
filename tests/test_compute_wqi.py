"""Unit tests for Water Quality Index (WQI) Computation and Class Labeling.

Test Suite:
    - TC-1.3-01: Known inputs produce mathematically expected WQI score (hand-calculated test cases).
    - TC-1.3-02: Class boundary edge cases (25.0 vs 25.01/26.0, 50.0 vs 50.01/51.0, 75.0 vs 76.0, 100.0 vs 101.0, binary potability).
    - TC-1.3-03: No records with complete/sufficient parameter data produce NaN/null WQI.
    - TC-1.3-04: Class distribution produces valid counts and non-zero entries across multiple classes.
    - TC-1.3-05: Sub-index boundary calculations for pH, DO, and standard parameters.
    - TC-1.3-06: Dynamic weight redistribution and minimum parameter quarantine enforcement.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.compute_wqi import (
    WQI_PARAMETERS,
    assign_potability,
    assign_wqi_class,
    compute_sub_index,
    compute_wqi_score,
    run_wqi_labeling_pipeline,
)


# ===========================================================================
# TC-1.3-01: Hand-Calculated Mathematical Precision Test Cases
# ===========================================================================
class TestHandCalculations:
    """Validate that exact hand calculations match compute_wqi output within tight tolerance."""

    def test_hand_calculation_pure_water(self):
        """Cross-Check Case 1: Pure / Pristine Water Baseline.

        Inputs: pH=7.0, DO=14.6 mg/L, BOD=0.0 mg/L
        Expected: q_pH=0, q_DO=0, q_BOD=0 -> WQI = 0.0000 -> Excellent (Class A), Potable=1.
        """
        row = {
            "Potential of Hydrogen (pH)": 7.0,
            "Dissolved oxygen (mg/L)": 14.6,
            "Biochemical Oxygen Demand (mg/L)": 0.0,
        }
        score = compute_wqi_score(row)
        assert np.isclose(score, 0.0, atol=1e-5)
        assert assign_wqi_class(score) == "Excellent"
        assert assign_potability(score) == 1

    def test_hand_calculation_standard_limit_water(self):
        """Cross-Check Case 2: Exact BIS Standard Permissible Limit Threshold Water.

        Inputs: pH=8.5, DO=5.0 mg/L, BOD=3.0 mg/L, FC=250.0 MPN/100mL
        Expected: All q_i = 100.0 -> WQI = 100.0000 -> Very Poor (Class D), Potable=0.
        """
        row = {
            "Potential of Hydrogen (pH)": 8.5,
            "Dissolved oxygen (mg/L)": 5.0,
            "Biochemical Oxygen Demand (mg/L)": 3.0,
            "Fecal Coliform (MPN/100mL)": 250.0,
        }
        score = compute_wqi_score(row)
        assert np.isclose(score, 100.0, atol=1e-5)
        assert assign_wqi_class(score) == "Very Poor"
        assert assign_potability(score) == 0

    def test_hand_calculation_mixed_surface_water(self):
        """Cross-Check Case 3: Representative Mixed Quality Surface Water Sample.

        Inputs:
            pH = 7.6, DO = 6.8 mg/L, BOD = 2.1 mg/L, FC = 120.0 MPN/100mL
        Calculations:
            q_pH = (7.6 - 7.0) / 1.5 * 100 = 40.0
            q_DO = |6.8 - 14.6| / 9.6 * 100 = 81.25
            q_BOD = (2.1 / 3.0) * 100 = 70.0
            q_FC = (120.0 / 250.0) * 100 = 48.0
            Weights: 1/8.5, 1/5.0, 1/3.0, 1/250.0
            Expected WQI ~ 67.91228595...
        """
        row = {
            "Potential of Hydrogen (pH)": 7.6,
            "Dissolved oxygen (mg/L)": 6.8,
            "Biochemical Oxygen Demand (mg/L)": 2.1,
            "Fecal Coliform (MPN/100mL)": 120.0,
        }
        score = compute_wqi_score(row)
        inv_s = [1.0 / 8.5, 1.0 / 5.0, 1.0 / 3.0, 1.0 / 250.0]
        q = [40.0, 81.25, 70.0, 48.0]
        expected_score = sum(q_i * w_i for q_i, w_i in zip(q, inv_s)) / sum(inv_s)

        assert np.isclose(score, expected_score, atol=1e-5)
        assert np.isclose(score, 67.91228595, atol=1e-2)
        assert assign_wqi_class(score) == "Poor"
        assert assign_potability(score) == 0


# ===========================================================================
# TC-1.3-02: Class Boundary Edge Cases
# ===========================================================================
class TestClassBoundaries:
    """Validate boundary behaviors for all 5 WQI classes and binary potability."""

    @pytest.mark.parametrize(
        "score, expected_class, expected_potability",
        [
            (0.0, "Excellent", 1),
            (15.0, "Excellent", 1),
            (25.0, "Excellent", 1),
            (25.0001, "Good", 1),
            (26.0, "Good", 1),
            (50.0, "Good", 1),
            (50.0001, "Poor", 0),
            (51.0, "Poor", 0),
            (75.0, "Poor", 0),
            (75.0001, "Very Poor", 0),
            (76.0, "Very Poor", 0),
            (100.0, "Very Poor", 0),
            (100.0001, "Unsuitable", 0),
            (101.0, "Unsuitable", 0),
            (500.0, "Unsuitable", 0),
        ],
    )
    def test_class_assignment_and_potability(self, score, expected_class, expected_potability):
        assert assign_wqi_class(score) == expected_class
        assert assign_potability(score) == expected_potability

    def test_nan_handling(self):
        assert assign_wqi_class(np.nan) is None
        assert assign_potability(np.nan) is None


# ===========================================================================
# TC-1.3-05: Sub-Index Computation Specifics
# ===========================================================================
class TestSubIndices:
    """Validate sub-index logic for pH (acidic, neutral, alkaline) and DO."""

    def test_ph_sub_index(self):
        # Neutral pure water
        assert compute_sub_index("pH", 7.0) == 0.0
        # Alkaline limit (8.5 -> q=100)
        assert compute_sub_index("pH", 8.5) == 100.0
        # Alkaline midway (7.75 -> q=50)
        assert np.isclose(compute_sub_index("pH", 7.75), 50.0)
        # Acidic limit (6.5 -> q=100)
        assert np.isclose(compute_sub_index("pH", 6.5), 100.0)
        # Acidic midway (6.75 -> q=50)
        assert np.isclose(compute_sub_index("pH", 6.75), 50.0)
        # Highly acidic (5.0 -> q=400)
        assert np.isclose(compute_sub_index("pH", 5.0), 400.0)

    def test_do_sub_index(self):
        # Pristine saturated DO (14.6 mg/L -> q=0)
        assert compute_sub_index("DO", 14.6) == 0.0
        # Permissible limit DO (5.0 mg/L -> q=100)
        assert np.isclose(compute_sub_index("DO", 5.0), 100.0)
        # Anoxic DO (0.0 mg/L -> q=152.0833)
        assert np.isclose(compute_sub_index("DO", 0.0), (14.6 / 9.6) * 100.0)

    def test_general_parameter_sub_index(self):
        # BOD: S_i=3.0 -> 3.0 gives 100.0, 6.0 gives 200.0
        assert compute_sub_index("BOD", 3.0) == 100.0
        assert compute_sub_index("BOD", 6.0) == 200.0
        # EC: S_i=750 -> 750 gives 100.0
        assert compute_sub_index("EC", 750.0) == 100.0


# ===========================================================================
# TC-1.3-03 & TC-1.3-06: Dynamic Weight Redistribution & Quarantine
# ===========================================================================
class TestDynamicWeightAndQuarantine:
    """Validate dynamic weight renormalization and threshold quarantine."""

    def test_dynamic_weight_redistribution(self):
        """When optional parameters are missing, weights renormalize dynamically."""
        # Row with 3 parameters: pH=7.0, DO=14.6, BOD=3.0
        row = {
            "Potential of Hydrogen (pH)": 7.0,
            "Dissolved oxygen (mg/L)": 14.6,
            "Biochemical Oxygen Demand (mg/L)": 3.0,
        }
        score = compute_wqi_score(row)
        # q_pH=0, q_DO=0, q_BOD=100
        # w_pH = 1/8.5, w_DO = 1/5.0, w_BOD = 1/3.0
        w_sum = (1 / 8.5) + (1 / 5.0) + (1 / 3.0)
        expected = (0 + 0 + 100.0 * (1 / 3.0)) / w_sum
        assert np.isclose(score, expected, atol=1e-5)

    def test_quarantine_under_minimum_parameters(self):
        """Records with fewer than 3 valid parameters must return NaN (quarantined)."""
        # Only 2 parameters
        row = {
            "Potential of Hydrogen (pH)": 7.5,
            "Dissolved oxygen (mg/L)": 6.0,
        }
        assert np.isnan(compute_wqi_score(row, min_parameters=3))

        # 0 parameters
        assert np.isnan(compute_wqi_score({}, min_parameters=3))

    def test_short_code_or_full_column_name_support(self):
        """compute_wqi_score should work with either short codes or full CPCB column names."""
        row_short = {"pH": 7.0, "DO": 14.6, "BOD": 0.0}
        row_full = {
            "Potential of Hydrogen (pH)": 7.0,
            "Dissolved oxygen (mg/L)": 14.6,
            "Biochemical Oxygen Demand (mg/L)": 0.0,
        }
        assert compute_wqi_score(row_short) == compute_wqi_score(row_full)


# ===========================================================================
# TC-1.3-04: End-to-End Pipeline & Distribution Audit
# ===========================================================================
class TestPipelineAndDistribution:
    """Validate execution on the actual joined dataset and verify report generation."""

    def test_pipeline_on_joined_dataset(self, tmp_path):
        repo_root = Path(__file__).resolve().parent.parent
        joined_path = repo_root / "data" / "interim" / "cpcb_joined.csv"

        if not joined_path.exists():
            pytest.skip("cpcb_joined.csv not found")

        out_labeled = tmp_path / "cpcb_labeled.csv"
        out_quarantine = tmp_path / "quarantine.csv"
        out_report = tmp_path / "wqi_class_distribution.csv"

        df_labeled, df_quarantine, report = run_wqi_labeling_pipeline(
            input_joined_path=joined_path,
            output_labeled_path=out_labeled,
            output_quarantine_path=out_quarantine,
            output_report_path=out_report,
            min_parameters=3,
        )

        assert len(df_labeled) > 0
        assert "wqi_score" in df_labeled.columns
        assert "wqi_class" in df_labeled.columns
        assert "Potability" in df_labeled.columns

        # Verify no NaN scores in labeled dataset
        assert df_labeled["wqi_score"].isna().sum() == 0
        assert df_labeled["wqi_class"].isna().sum() == 0
        assert df_labeled["Potability"].isna().sum() == 0

        # Verify multi-class non-zero counts
        active_classes = [cls for cls, cnt in report.class_counts.items() if cnt > 0]
        assert len(active_classes) >= 2, "Expected at least 2 distinct classes in dataset"

        # Verify report file written and readable
        assert out_report.exists()
        df_rep = pd.read_csv(out_report)
        assert len(df_rep) == 5
        assert set(df_rep["wqi_class"]) == {"Excellent", "Good", "Poor", "Very Poor", "Unsuitable"}
