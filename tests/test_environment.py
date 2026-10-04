"""Environment & Dependency Baseline Test Suite (TC-0.1-01 / TC-0.1-02)."""
import importlib
import sys
import pytest


def test_python_version():
    """Verify Python version meets requirement (>= 3.10)."""
    major = sys.version_info.major
    minor = sys.version_info.minor
    assert major == 3, f"Python 3 required, found Python {major}"
    assert minor >= 10, f"Python 3.10+ required, found Python {major}.{minor}"


@pytest.mark.parametrize(
    "package_name",
    [
        "pandas",
        "numpy",
        "sklearn",
        "xgboost",
        "shap",
        "streamlit",
        "matplotlib",
        "seaborn",
        "plotly",
        "joblib",
        "yaml",
        "imblearn",
        "pytest",
    ],
)
def test_package_imports(package_name):
    """Verify that each required package is importable without errors."""
    module = importlib.import_module(package_name)
    assert module is not None, f"Failed to import package: {package_name}"
