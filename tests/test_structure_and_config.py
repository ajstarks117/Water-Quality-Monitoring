"""Project Structure & Config Scaffolding Test Suite (TC-0.2-01, TC-0.2-02, TC-0.2-03)."""
import importlib
from pathlib import Path
import yaml
import pytest


def test_tc_0_2_01_folder_tree_matches_spec():
    """TC-0.2-01: Verify that all required directories exist in the project."""
    repo_root = Path(__file__).resolve().parent.parent

    required_dirs = [
        repo_root / "data" / "raw",
        repo_root / "data" / "interim",
        repo_root / "data" / "processed",
        repo_root / "notebooks",
        repo_root / "src" / "data",
        repo_root / "src" / "features",
        repo_root / "src" / "models",
        repo_root / "src" / "explainability",
        repo_root / "src" / "decision_support",
        repo_root / "models",
        repo_root / "reports" / "figures",
        repo_root / "app",
        repo_root / "config",
        repo_root / "tests",
    ]

    for directory in required_dirs:
        assert directory.exists(), f"Required directory missing: {directory.relative_to(repo_root)}"
        assert directory.is_dir(), f"Path is not a directory: {directory.relative_to(repo_root)}"


def test_tc_0_2_02_config_yaml_loads():
    """TC-0.2-02: Verify that config/config.yaml loads cleanly and contains required keys."""
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "config" / "config.yaml"

    assert config_path.exists(), "config/config.yaml does not exist"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert isinstance(config, dict), "config.yaml must parse as a dictionary"

    required_keys = [
        "dataset_path",
        "target_column",
        "feature_columns",
        "random_seed",
        "test_size",
        "model_output_path",
        "shap_output_path",
    ]

    for key in required_keys:
        assert key in config, f"Missing required key in config.yaml: {key}"

    assert isinstance(config["feature_columns"], list), "feature_columns should be a list"
    assert config["random_seed"] == 42, "random_seed should be 42"
    assert config["test_size"] == 0.2, "test_size should be 0.2"


@pytest.mark.parametrize(
    "subpackage",
    [
        "src.data",
        "src.features",
        "src.models",
        "src.explainability",
        "src.decision_support",
    ],
)
def test_tc_0_2_03_package_imports_resolve(subpackage):
    """TC-0.2-03: Verify that all src/ subpackages can be imported cleanly."""
    module = importlib.import_module(subpackage)
    assert module is not None, f"Failed to import subpackage: {subpackage}"


def test_tc_0_3_02_workflow_docs_exist():
    """TC-0.3-02: Verify CONTRIBUTING.md and Docs/interface_contracts.md exist with required sections."""
    repo_root = Path(__file__).resolve().parent.parent
    contributing_path = repo_root / "CONTRIBUTING.md"
    contracts_path = repo_root / "Docs" / "interface_contracts.md"

    assert contributing_path.exists(), "CONTRIBUTING.md missing"
    assert contracts_path.exists(), "Docs/interface_contracts.md missing"

    contributing_content = contributing_path.read_text(encoding="utf-8")
    assert "feature/data-pipeline" in contributing_content
    assert "feature/ml-models" in contributing_content
    assert "feature/explainability-decision-support" in contributing_content
    assert "feature/dashboard" in contributing_content
    assert "2 hours" in contributing_content

    contracts_content = contracts_path.read_text(encoding="utf-8")
    assert "Cleaned Data Contract" in contracts_content
    assert "Model Artifact Contract" in contracts_content
    assert "Potability" in contracts_content
    assert "best_model.pkl" in contracts_content

