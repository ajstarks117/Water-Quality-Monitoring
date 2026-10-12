"""
Prediction Module & Inference Wrapper
======================================
Milestone 6.3: Final Model Selection, Freeze & Pipeline Artifact - CONTRACT FREEZE
Owner: ML Lead (Person B)
Branch: feature/ml-models

Provides a clean, stable inference wrapper around the frozen final model artifact
at `models/best_model.pkl`.

The frozen artifact encapsulates:
1. Feature engineering: computes 'Log10 Fecal Coliform' from raw 'Fecal Coliform (MPN/100mL)'
2. Preprocessing pipeline: M4.2 training-fitted StandardScaler + median imputer
3. Final classifier: Decision Tree with balanced class weights (M6.3 selected model)

Interface Guarantees:
- Accepts raw, unprocessed DataFrame with the 4 frozen candidate feature columns:
    * Potential of Hydrogen (pH)
    * Dissolved oxygen (mg/L)
    * Biochemical Oxygen Demand (mg/L)
    * Fecal Coliform (MPN/100mL)
- Exposes .predict() returning class label strings:
    ['Excellent', 'Good', 'Poor', 'Unsuitable', 'Very Poor']
- Exposes .predict_proba() returning normalized probability arrays summing to 1.0
- Raises clear, informative errors on malformed or missing input features
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from src.data.preprocessing import FROZEN_CANDIDATE_FEATURES
from src.features.engineering import (
    FINAL_EXTENDED_FEATURE_COLUMNS,
    WaterQualityFeatureEngineer,
)

logger = logging.getLogger("water_quality.models.predict")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "config" / "config.yaml"
DEFAULT_MODEL_PATH = _REPO_ROOT / "models" / "best_model.pkl"

# Frozen raw input columns expected by the model artifact
FROZEN_RAW_FEATURE_COLUMNS: List[str] = list(FROZEN_CANDIDATE_FEATURES)

# Canonical target class labels
TARGET_CLASSES: List[str] = [
    "Excellent",
    "Good",
    "Poor",
    "Unsuitable",
    "Very Poor",
]


def _resolve_model_path(model_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve model path from argument, configuration, or default."""
    if model_path is not None:
        return Path(model_path)

    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            out_path = cfg.get("model_output_path")
            if out_path:
                return _REPO_ROOT / out_path
        except Exception as exc:
            logger.warning("Failed to read model_output_path from config: %s", exc)

    return DEFAULT_MODEL_PATH


def compute_artifact_hash(artifact_path: Union[str, Path]) -> str:
    """
    Compute the SHA-256 hexadecimal digest of an artifact file.

    Parameters
    ----------
    artifact_path : Union[str, Path]
        Path to the artifact file.

    Returns
    -------
    sha256_hex : str
        Hexadecimal SHA-256 hash.
    """
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"Artifact not found at {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_raw_features(
    data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
) -> pd.DataFrame:
    """
    Validate and format raw input data against the frozen candidate feature contract.

    Parameters
    ----------
    data : Union[pd.DataFrame, Dict, List[Dict]]
        Input features as a DataFrame, single record dict, or list of dicts.

    Returns
    -------
    validated_df : pd.DataFrame
        DataFrame guaranteed to contain the frozen candidate features with numeric dtypes.

    Raises
    ------
    ValueError
        If required raw features are missing or data cannot be interpreted.
    """
    if data is None:
        raise ValueError("Input data cannot be None.")

    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        if not data:
            raise ValueError("Input list cannot be empty.")
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError(
            f"Unsupported input data type: {type(data)}. "
            f"Expected pd.DataFrame, dict, or list of dicts."
        )

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    missing_cols = [col for col in FROZEN_RAW_FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required raw feature column(s): {missing_cols}. "
            f"Input DataFrame must contain all frozen raw candidate features: {FROZEN_RAW_FEATURE_COLUMNS}. "
            f"Available columns: {list(df.columns)}"
        )

    # Validate that features are numeric
    for col in FROZEN_RAW_FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Column '{col}' must contain numeric values. Conversion failed: {exc}"
                ) from exc

    return df[FROZEN_RAW_FEATURE_COLUMNS]


def build_and_freeze_pipeline(
    preprocessor_path: Optional[Union[str, Path]] = None,
    model_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[Pipeline, str]:
    """
    Combine the M4.2 fitted preprocessing pipeline and the M6.3 selected model
    into a single end-to-end inference Pipeline, and serialize to disk.

    Parameters
    ----------
    preprocessor_path : Optional[Union[str, Path]]
        Path to fitted preprocessing pipeline (default: models/preprocessing_pipeline_v1.pkl).
    model_path : Optional[Union[str, Path]]
        Path to selected classifier (default: models/experiments/baseline_decision_tree.pkl).
    output_path : Optional[Union[str, Path]]
        Path to save frozen pipeline (default: models/best_model.pkl).

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        The combined pipeline artifact.
    artifact_hash : str
        SHA-256 hexadecimal hash of the serialized file.
    """
    prep_p = Path(preprocessor_path) if preprocessor_path else _REPO_ROOT / "models" / "preprocessing_pipeline_v1.pkl"
    mod_p = Path(model_path) if model_path else _REPO_ROOT / "models" / "experiments" / "baseline_decision_tree.pkl"
    out_p = Path(output_path) if output_path else _resolve_model_path()

    if not prep_p.is_file():
        raise FileNotFoundError(f"Preprocessing pipeline not found at {prep_p}")
    if not mod_p.is_file():
        raise FileNotFoundError(f"Selected model artifact not found at {mod_p}")

    logger.info("Loading preprocessing pipeline from: %s", prep_p)
    preprocessor_pipeline = joblib.load(prep_p)

    logger.info("Loading selected model from: %s", mod_p)
    classifier_model = joblib.load(mod_p)

    # Construct the feature engineer step
    feature_engineer = WaterQualityFeatureEngineer(include_raw_features=True)
    # Fit feature engineer on column schema to populate internal inspection names
    dummy_df = pd.DataFrame(columns=FROZEN_CANDIDATE_FEATURES)
    feature_engineer.fit(dummy_df)

    # Extract the preprocessor transformer (ColumnTransformer) from the fitted pipeline
    if hasattr(preprocessor_pipeline, "named_steps") and "preprocessor" in preprocessor_pipeline.named_steps:
        preprocessor_step = preprocessor_pipeline.named_steps["preprocessor"]
    else:
        preprocessor_step = preprocessor_pipeline

    # Build the combined end-to-end pipeline:
    # Raw 4-column input -> Feature Engineering (5 cols) -> Scaler/Imputer -> Classifier
    combined_pipeline = Pipeline(
        steps=[
            ("feature_engineering", feature_engineer),
            ("preprocessor", preprocessor_step),
            ("classifier", classifier_model),
        ],
        verbose=False,
    )

    out_p.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Serializing combined pipeline artifact to: %s", out_p)
    joblib.dump(combined_pipeline, out_p)

    artifact_hash = compute_artifact_hash(out_p)
    logger.info("Artifact serialized successfully. SHA-256: %s", artifact_hash)

    return combined_pipeline, artifact_hash


def load_model(model_path: Optional[Union[str, Path]] = None) -> Pipeline:
    """
    Load the frozen combined model pipeline from disk.

    Parameters
    ----------
    model_path : Optional[Union[str, Path]]
        Path to the model artifact. If None, resolves from config or default.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Loaded scikit-learn Pipeline instance.
    """
    resolved_path = _resolve_model_path(model_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Frozen model artifact not found at '{resolved_path}'. "
            f"Run `build_and_freeze_pipeline()` to generate it."
        )

    model = joblib.load(resolved_path)
    return model


def predict(
    data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    model: Optional[Any] = None,
    model_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Make class predictions on raw, unprocessed water quality data.

    Parameters
    ----------
    data : Union[pd.DataFrame, Dict, List[Dict]]
        Raw feature inputs.
    model : Optional[Any]
        Loaded model instance. If None, loads from `model_path` or default.
    model_path : Optional[Union[str, Path]]
        Model path used when `model` is not provided.

    Returns
    -------
    predictions : np.ndarray
        Array of predicted class labels (e.g. 'Unsuitable', 'Poor', 'Good', etc.).
    """
    df_valid = validate_raw_features(data)
    mdl = model if model is not None else load_model(model_path)
    return mdl.predict(df_valid)


def predict_proba(
    data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    model: Optional[Any] = None,
    model_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Estimate class probabilities on raw, unprocessed water quality data.

    Parameters
    ----------
    data : Union[pd.DataFrame, Dict, List[Dict]]
        Raw feature inputs.
    model : Optional[Any]
        Loaded model instance. If None, loads from `model_path` or default.
    model_path : Optional[Union[str, Path]]
        Model path used when `model` is not provided.

    Returns
    -------
    probabilities : np.ndarray
        Array of shape (n_samples, n_classes) with normalized probabilities summing to 1.0.
    """
    df_valid = validate_raw_features(data)
    mdl = model if model is not None else load_model(model_path)
    return mdl.predict_proba(df_valid)


if __name__ == "__main__":
    print("Building and freezing combined model pipeline...")
    pipe, sha = build_and_freeze_pipeline()
    print(f"Combined pipeline frozen at {DEFAULT_MODEL_PATH}")
    print(f"SHA-256 Hash: {sha}")
