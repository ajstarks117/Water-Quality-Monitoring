"""
Feature Engineering Module
==========================
Implements leakage-safe, domain-justified derived features for water quality modeling.

Milestone 4.1: Feature Engineering
Owner: ML Lead (Person B)
Branch: feature/ml-models

Design Contracts & Clarifications:
1. Preserves the 4 frozen raw candidate features from Milestone 1.4:
   - Potential of Hydrogen (pH)
   - Dissolved oxygen (mg/L)
   - Biochemical Oxygen Demand (mg/L)
   - Fecal Coliform (MPN/100mL)
   Engineered features are ADDITIONS ONLY and never replace frozen raw features.
2. Small, strictly justified shortlist:
   - 'Log10 Fecal Coliform': Logarithmic transformation linearizing extreme bacterial
     skewness (+33.98) across orders of magnitude (10^0 to 10^7 MPN/100mL).
3. Leakage-Safe:
   - Computed row-wise from prediction-time user inputs only.
   - Zero dependence on target (wqi_class, Potability), wqi_score, future data, or post-hoc aggregates.
4. Redundancy-Safe:
   - Tested to ensure pairwise correlation |r| <= 0.85 against all raw features (|r| = 0.484).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.data.preprocessing import FROZEN_CANDIDATE_FEATURES

logger = logging.getLogger("water_quality.features.engineering")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Canonical Engineered Feature Names
LOG_FECAL_COLIFORM_COL = "Log10 Fecal Coliform"

ENGINEERED_FEATURE_COLUMNS: List[str] = [
    LOG_FECAL_COLIFORM_COL,
]

FINAL_EXTENDED_FEATURE_COLUMNS: List[str] = (
    FROZEN_CANDIDATE_FEATURES + ENGINEERED_FEATURE_COLUMNS
)


def compute_log_fecal_coliform(
    fecal_coliform: Union[pd.Series, np.ndarray, float, int],
    clip_lower: float = 1.0,
) -> Union[pd.Series, np.ndarray, float]:
    """
    Compute base-10 logarithm of Fecal Coliform count with minimum clipping.
    
    Domain Justification:
    ---------------------
    In environmental microbiology and Phase 3 EDA, Fecal Coliform spans 7 orders
    of magnitude (2.0 to 22,000,000 MPN/100mL) with extreme positive skewness (+33.98).
    Applying log10 transforms exponential microbial proliferation into a linear scale
    amenable to distance, linear, and gradient-based estimators.
    
    Prediction-Time Availability & Provenance:
    -----------------------------------------
    - Input: Raw 'Fecal Coliform (MPN/100mL)' entered by user at dashboard inference.
    - Zero dependence on targets (wqi_class, Potability) or target scores (wqi_score).
    - Fully row-independent; zero data leakage across train/test folds.
    
    Parameters
    ----------
    fecal_coliform : pd.Series, np.ndarray, float
        Raw bacterial count in MPN/100mL.
    clip_lower : float, default=1.0
        Lower bound to avoid log10(0) or negative logs for sub-unit readings.
        
    Returns
    -------
    log10_fc : pd.Series, np.ndarray, float
        Log-transformed bacterial concentration (log10 MPN/100mL).
    """
    if isinstance(fecal_coliform, pd.Series):
        clipped = fecal_coliform.clip(lower=clip_lower)
        return np.log10(clipped)
    elif isinstance(fecal_coliform, np.ndarray):
        clipped = np.clip(fecal_coliform, clip_lower, None)
        return np.log10(clipped)
    else:
        clipped = max(float(fecal_coliform), clip_lower)
        return float(np.log10(clipped))


def extract_engineered_features(
    df: pd.DataFrame,
    include_raw_features: bool = True,
) -> pd.DataFrame:
    """
    Compute and append domain-justified engineered features to DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the 4 frozen candidate feature columns.
    include_raw_features : bool, default=True
        If True, returns DataFrame with [4 frozen raw features + engineered features].
        If False, returns DataFrame with only engineered features.
        
    Returns
    -------
    out_df : pd.DataFrame
        DataFrame with engineered features added.
    """
    fc_col = "Fecal Coliform (MPN/100mL)"
    if fc_col not in df.columns:
        raise ValueError(
            f"Missing required input column '{fc_col}' for feature engineering. "
            f"Available columns: {list(df.columns)}"
        )
        
    out_df = pd.DataFrame(index=df.index)
    
    if include_raw_features:
        for col in FROZEN_CANDIDATE_FEATURES:
            if col in df.columns:
                out_df[col] = df[col]
            else:
                raise ValueError(
                    f"Missing frozen raw candidate feature '{col}' in input DataFrame."
                )
                
    # 1. Log10 Fecal Coliform
    out_df[LOG_FECAL_COLIFORM_COL] = compute_log_fecal_coliform(df[fc_col])
    
    return out_df


class WaterQualityFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible feature engineering transformer.
    
    Appends justified, leakage-checked engineered features to the feature matrix.
    Safe for scikit-learn Pipeline and cross-validation workflows.
    """
    
    def __init__(self, include_raw_features: bool = True):
        self.include_raw_features = include_raw_features
        self.feature_names_in_: Optional[List[str]] = None
        self.feature_names_out_: Optional[List[str]] = None
        
    def fit(self, X: pd.DataFrame | np.ndarray, y: Optional[pd.Series | np.ndarray] = None):
        """Fit transformer (stateless; preserves schema contracts)."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"feat_{i}" for i in range(X.shape[1])]
            
        if self.include_raw_features:
            self.feature_names_out_ = (
                self.feature_names_in_ + ENGINEERED_FEATURE_COLUMNS
            )
        else:
            self.feature_names_out_ = ENGINEERED_FEATURE_COLUMNS.copy()
            
        return self
        
    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Transform input features by appending engineered features."""
        if not isinstance(X, pd.DataFrame):
            if self.feature_names_in_ and len(self.feature_names_in_) == X.shape[1]:
                X_df = pd.DataFrame(X, columns=self.feature_names_in_)
            elif X.shape[1] == len(FROZEN_CANDIDATE_FEATURES):
                X_df = pd.DataFrame(X, columns=FROZEN_CANDIDATE_FEATURES)
            else:
                raise ValueError(
                    f"Expected {len(FROZEN_CANDIDATE_FEATURES)} features, got {X.shape[1]}"
                )
        else:
            X_df = X.copy()
            
        return extract_engineered_features(
            X_df,
            include_raw_features=self.include_raw_features,
        )
        
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return output feature names for scikit-learn pipeline inspection."""
        if input_features is not None:
            if self.include_raw_features:
                return np.array(list(input_features) + ENGINEERED_FEATURE_COLUMNS)
            return np.array(ENGINEERED_FEATURE_COLUMNS)
        if self.feature_names_out_ is not None:
            return np.array(self.feature_names_out_)
        return np.array(FINAL_EXTENDED_FEATURE_COLUMNS)
