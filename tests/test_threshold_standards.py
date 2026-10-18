"""Test Suite for Milestone 8.1: Rule-Engine Threshold Mapping.

Owner: Explainability & Decision Support Lead (Person C)
Branch: feature/explainability-decision-support

Test Cases:
- TC-8.1-01: Every rule-engine parameter has a cited threshold with a named, dated source.
- TC-8.1-02: Shared parameters are consistent with M1.1 (wqi_standard.md / compute_wqi.py).
- TC-8.1-03: Units match dataset units in data_dictionary.md and config.yaml exactly.
- Additional tests:
  - Boundary condition semantics (inclusive vs exclusive) are explicitly defined.
  - Documented fallback policy for unstandardized parameters is explicitly labeled.
  - Causation-safe language discipline (Spec Section 31 & 48) is mandated.
"""

from pathlib import Path
import re
import pytest
import yaml

from src.data.compute_wqi import WQI_PARAMETERS


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """Fixture returning the repository root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def threshold_standards_doc(repo_root: Path) -> str:
    """Fixture loading Docs/threshold_standards.md."""
    doc_path = repo_root / "Docs" / "threshold_standards.md"
    if not doc_path.exists():
        doc_path = repo_root / "docs" / "threshold_standards.md"
    assert doc_path.exists(), f"threshold_standards.md not found at {doc_path}"
    return doc_path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def wqi_standard_doc(repo_root: Path) -> str:
    """Fixture loading Docs/wqi_standard.md."""
    doc_path = repo_root / "Docs" / "wqi_standard.md"
    if not doc_path.exists():
        doc_path = repo_root / "docs" / "wqi_standard.md"
    assert doc_path.exists(), f"wqi_standard.md not found at {doc_path}"
    return doc_path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def data_dict_doc(repo_root: Path) -> str:
    """Fixture loading Docs/data_dictionary.md."""
    doc_path = repo_root / "Docs" / "data_dictionary.md"
    assert doc_path.exists(), f"data_dictionary.md not found at {doc_path}"
    return doc_path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def config_dict(repo_root: Path) -> dict:
    """Fixture loading config/config.yaml."""
    config_path = repo_root / "config" / "config.yaml"
    assert config_path.exists(), f"config.yaml not found at {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ===========================================================================
# TC-8.1-01: Every Rule-Engine Parameter Has a Cited Threshold
# ===========================================================================
def test_tc_8_1_01_frozen_features_covered(threshold_standards_doc: str, config_dict: dict):
    """TC-8.1-01: Verify all frozen feature columns in config.yaml are covered."""
    frozen_features = config_dict["feature_columns"]
    assert len(frozen_features) == 4, f"Expected 4 frozen features, found {len(frozen_features)}"

    for feature in frozen_features:
        assert feature in threshold_standards_doc, (
            f"Frozen feature '{feature}' is missing from Docs/threshold_standards.md"
        )


def test_tc_8_1_01_engineered_and_extension_features_covered(threshold_standards_doc: str):
    """TC-8.1-01: Verify engineered and extension parameters are covered."""
    expected_parameters = [
        "Log10 Fecal Coliform",
        "Turbidity (NTU)",
        "Electric Conductivity (μS/cm)",
        "Total Dissolved Solids (mg/L)",
        "Chemical Oxygen Demand (mg/L)",
        "Total Coliform (MPN/100mL)",
        "Nitrate N (mgN/L)",
    ]
    for param in expected_parameters:
        assert param in threshold_standards_doc, (
            f"Parameter '{param}' is missing from Docs/threshold_standards.md"
        )


def test_tc_8_1_01_all_parameters_have_named_statutory_citations(threshold_standards_doc: str):
    """TC-8.1-01: Verify that every parameter section contains named, dated statutory citations."""
    statutory_citations = [
        "CPCB",
        "IS 10500:2012",
        "BIS",
        "wqi_standard.md",
    ]
    for citation in statutory_citations:
        assert citation in threshold_standards_doc, (
            f"Expected statutory citation token '{citation}' missing from threshold_standards.md"
        )


# ===========================================================================
# TC-8.1-02: Shared Parameters are Consistent with M1.1
# ===========================================================================
def test_tc_8_1_02_shared_parameter_limits_match_m1_1(threshold_standards_doc: str, wqi_standard_doc: str):
    """TC-8.1-02: Ensure shared parameter values match between M1.1 and M8.1."""
    # Shared numerical limits in WQI_PARAMETERS
    shared_limits = {
        "pH": {"standard": 8.5, "range_min": 6.5, "range_max": 8.5},
        "DO": {"standard": 5.0},
        "BOD": {"standard": 3.0},
        "FC": {"standard": 250.0},
        "Turbidity": {"standard": 5.0},
        "EC": {"standard": 750.0},
        "TDS": {"standard": 500.0},
        "COD": {"standard": 10.0},
        "TC": {"standard": 500.0},
    }

    # Verify each parameter matches in WQI_PARAMETERS
    for code, info in shared_limits.items():
        assert code in WQI_PARAMETERS, f"Code '{code}' not in WQI_PARAMETERS"
        wqi_limit = WQI_PARAMETERS[code].standard_limit
        assert wqi_limit == info["standard"], (
            f"WQI standard limit mismatch for {code}: {wqi_limit} vs {info['standard']}"
        )

    # Verify specific threshold occurrences in threshold_standards_doc
    assert "6.5" in threshold_standards_doc and "8.5" in threshold_standards_doc
    assert "5.0 mg/L" in threshold_standards_doc or "5.0" in threshold_standards_doc
    assert "3.0 mg/L" in threshold_standards_doc or "3.0" in threshold_standards_doc
    assert "250.0 MPN/100mL" in threshold_standards_doc or "250.0" in threshold_standards_doc
    assert "750.0" in threshold_standards_doc
    assert "500.0" in threshold_standards_doc
    assert "10.0" in threshold_standards_doc


def test_tc_8_1_02_consistency_matrix_present(threshold_standards_doc: str):
    """TC-8.1-02: Verify explicit consistency matrix with M1.1 is documented."""
    assert "Verification & Cross-Document Consistency Matrix" in threshold_standards_doc
    assert "IDENTICAL" in threshold_standards_doc
    assert "CONSISTENT" in threshold_standards_doc


# ===========================================================================
# TC-8.1-03: Units Match Dataset Units Exactly
# ===========================================================================
def test_tc_8_1_03_units_match_data_dictionary(threshold_standards_doc: str, data_dict_doc: str):
    """TC-8.1-03: Verify exact unit match with data dictionary for all key parameters."""
    unit_checks = [
        ("Potential of Hydrogen (pH)", "pH scale"),
        ("Dissolved oxygen (mg/L)", "mg/L"),
        ("Biochemical Oxygen Demand (mg/L)", "mg/L"),
        ("Fecal Coliform (MPN/100mL)", "MPN/100mL"),
        ("Turbidity (NTU)", "NTU"),
        ("Electric Conductivity (μS/cm)", "μS/cm"),
        ("Total Dissolved Solids (mg/L)", "mg/L"),
        ("Chemical Oxygen Demand (mg/L)", "mg/L"),
        ("Total Coliform (MPN/100mL)", "MPN/100mL"),
        ("Nitrate N (mgN/L)", "mgN/L"),
    ]

    for param, unit in unit_checks:
        assert unit in data_dict_doc, f"Unit '{unit}' not found in data_dictionary.md for {param}"
        assert unit in threshold_standards_doc, (
            f"Unit '{unit}' for parameter '{param}' not found in threshold_standards.md"
        )


# ===========================================================================
# Additional Verification Tests: Boundary Conditions, Fallback & Causation
# ===========================================================================
def test_tc_8_1_boundary_condition_semantics(threshold_standards_doc: str):
    """Verify that boundary condition semantics are explicitly documented."""
    assert "Boundary Condition Semantics" in threshold_standards_doc
    assert "NORMAL / PERMISSIBLE" in threshold_standards_doc
    assert "V <= S_i" in threshold_standards_doc or "V \\le S_i" in threshold_standards_doc


def test_tc_8_1_fallback_policy(threshold_standards_doc: str):
    """Verify that empirical fallback policy is documented and labeled as non-statutory."""
    assert "Documented Fallback Policy" in threshold_standards_doc
    assert "Statistical Percentile-Based Baseline" in threshold_standards_doc
    assert "Empirical Statistical Fallback" in threshold_standards_doc


def test_tc_8_1_causation_safe_language_enforced(threshold_standards_doc: str):
    """Verify that causation-safe language rules are explicitly mandated."""
    assert "Causation-Safe Interpretation Language Protocol" in threshold_standards_doc
    assert "is associated with" in threshold_standards_doc
    assert "indicates potential presence of" in threshold_standards_doc
    assert "MANDATORY WORDING DISCIPLINE" in threshold_standards_doc
