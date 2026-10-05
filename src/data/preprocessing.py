"""Data Cleaning, Missing Value Imputation, and Preprocessing Engine.

This module implements reproducible, leakage-safe data cleaning, duplicate handling,
and missing value imputation for the CPCB Surface Water Quality dataset per
`docs/cleaning_decisions.md` (DEC-CLEANING-PREPROC-2026-V1).

Key Functions:
    - `handle_duplicates`: Detects, logs, and removes exact duplicate rows.
    - `handle_missing_values`: Drops sparse columns exceeding threshold (>40%)
      and performs robust median imputation with fit/transform state preservation.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Configure structured logging
logger = logging.getLogger("preprocessing")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Preprocessing State & Audit Structures
# ---------------------------------------------------------------------------
@dataclass
class PreprocessingState:
    """Stores fitted parameters (e.g. medians, dropped columns) for test transform."""
    dropped_columns: List[str] = field(default_factory=list)
    imputation_values: Dict[str, float] = field(default_factory=dict)
    imputation_strategy: str = "median"
    max_missing_col_pct: float = 40.0
    initial_shape: Tuple[int, int] = (0, 0)
    final_shape: Tuple[int, int] = (0, 0)
    imputed_counts: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Duplicate Row Handling
# ---------------------------------------------------------------------------
def handle_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
) -> pd.DataFrame:
    """Detect, log, and handle duplicate rows in the dataset.

    Args:
        df: Input pandas DataFrame.
        subset: Optional list of column names to consider for identifying duplicates.
                If None, considers all columns (exact full-row duplicates).
        keep: Determines which duplicates to keep ('first', 'last', False). Default: 'first'.

    Returns:
        pd.DataFrame: Deduplicated DataFrame.
    """
    initial_count = len(df)
    duplicate_mask = df.duplicated(subset=subset, keep=keep)
    duplicate_count = int(duplicate_mask.sum())

    logger.info("================ DUPLICATE ROW AUDIT ================")
    logger.info("  Initial Row Count: %d", initial_count)
    logger.info("  Duplicate Subset Criteria: %s", "Exact Full Row" if subset is None else subset)
    logger.info("  Duplicate Rows Identified: %d (%.2f%%)", duplicate_count, (duplicate_count / initial_count) * 100 if initial_count > 0 else 0)

    if duplicate_count > 0:
        df_cleaned = df[~duplicate_mask].copy().reset_index(drop=True)
        logger.info("  Removed %d duplicate rows. New Row Count: %d", duplicate_count, len(df_cleaned))
    else:
        df_cleaned = df.copy().reset_index(drop=True)
        logger.info("  No duplicate rows found. Dataset preserved.")

    return df_cleaned


# ---------------------------------------------------------------------------
# Missing Value Handling Engine
# ---------------------------------------------------------------------------
def handle_missing_values(
    df: pd.DataFrame,
    strategy_config: Optional[Dict[str, Any]] = None,
    fitted_state: Optional[PreprocessingState] = None,
) -> Tuple[pd.DataFrame, PreprocessingState]:
    """Handle missing values via thresholded column dropping and robust median imputation.

    Features:
        1. Drops columns exceeding `max_missing_col_pct` (default: 40.0%) unless protected.
        2. Imputes missing values in active numeric feature columns using median (or mean).
        3. Supports train-fit and test-transform paradigm via `fitted_state` to prevent data leakage.
        4. Logs before/after shape, dropped columns, and imputed counts for complete traceability.

    Args:
        df: Input pandas DataFrame.
        strategy_config: Optional dictionary configuring cleaning options:
            - `max_missing_col_pct` (float): Threshold above which columns are dropped (default: 40.0).
            - `protected_columns` (list): Columns that must never be dropped.
            - `imputation_strategy` (str): 'median', 'mean', 'constant', or 'drop_rows' (default: 'median').
            - `fill_value` (float): Value used if strategy is 'constant' (default: 0.0).
            - `feature_columns` (list): Explicit feature columns to impute.
        fitted_state: Optional PreprocessingState from training fold for transform-only execution.

    Returns:
        Tuple[pd.DataFrame, PreprocessingState]: (Cleaned DataFrame, Fitted/Applied PreprocessingState)
    """
    config = strategy_config or {}
    max_missing_pct = float(config.get("max_missing_col_pct", 40.0))
    imputation_strategy = config.get("imputation_strategy", "median")
    fill_value = float(config.get("fill_value", 0.0))
    protected_cols = set(config.get("protected_columns", [
        "Potential of Hydrogen (pH)",
        "Dissolved oxygen (mg/L)",
        "Biochemical Oxygen Demand (mg/L)",
        "Fecal Coliform (MPN/100mL)",
        "wqi_class",
        "Potability",
        "Station",
        "State",
        "District",
        "sampling_date",
    ]))
    explicit_features = config.get("feature_columns", [
        "Potential of Hydrogen (pH)",
        "Dissolved oxygen (mg/L)",
        "Biochemical Oxygen Demand (mg/L)",
        "Fecal Coliform (MPN/100mL)",
    ])

    df_out = df.copy()
    initial_shape = df_out.shape

    state = PreprocessingState(
        initial_shape=initial_shape,
        max_missing_col_pct=max_missing_pct,
        imputation_strategy=imputation_strategy,
    )

    logger.info("================ MISSING VALUE PREPROCESSING AUDIT ================")
    logger.info("  Initial Dataset Shape: %d rows x %d columns", initial_shape[0], initial_shape[1])
    logger.info("  Column Missingness Threshold: > %.1f%% -> DROP", max_missing_pct)
    logger.info("  Imputation Strategy: %s", imputation_strategy)

    # -----------------------------------------------------------------------
    # Step 1: Threshold-based Column Dropping
    # -----------------------------------------------------------------------
    if fitted_state is not None:
        # Transform mode: Use pre-fitted dropped columns list
        cols_to_drop = [c for c in fitted_state.dropped_columns if c in df_out.columns]
        state.dropped_columns = list(fitted_state.dropped_columns)
    else:
        # Fit mode: Compute missingness per column
        cols_to_drop = []
        for col in df_out.columns:
            missing_pct = (df_out[col].isna().sum() / len(df_out)) * 100
            if missing_pct > max_missing_pct and col not in protected_cols:
                cols_to_drop.append(col)
                logger.warning("  Dropping sparse column '%s' (Missing: %.2f%% > %.1f%%)", col, missing_pct, max_missing_pct)
        state.dropped_columns = cols_to_drop

    if cols_to_drop:
        df_out.drop(columns=cols_to_drop, inplace=True)
        logger.info("  Total Columns Dropped: %d. Remaining Columns: %d", len(cols_to_drop), df_out.shape[1])

    # -----------------------------------------------------------------------
    # Step 2: Imputation for Retained Active Numeric Features
    # -----------------------------------------------------------------------
    active_features = [f for f in explicit_features if f in df_out.columns]

    if imputation_strategy == "drop_rows":
        # Drop rows with any missing value in active features
        before_rows = len(df_out)
        df_out.dropna(subset=active_features, inplace=True)
        df_out.reset_index(drop=True, inplace=True)
        dropped_rows = before_rows - len(df_out)
        logger.info("  Dropped %d rows containing missing values in features. Remaining: %d rows", dropped_rows, len(df_out))

    elif imputation_strategy in ["median", "mean", "constant"]:
        impute_values: Dict[str, float] = {}

        for feat in active_features:
            missing_count = int(df_out[feat].isna().sum())
            state.imputed_counts[feat] = missing_count

            if missing_count > 0:
                if fitted_state is not None and feat in fitted_state.imputation_values:
                    # Use pre-fitted statistic (Transform mode)
                    val = fitted_state.imputation_values[feat]
                else:
                    # Calculate statistic (Fit mode)
                    if imputation_strategy == "median":
                        val = float(df_out[feat].median())
                    elif imputation_strategy == "mean":
                        val = float(df_out[feat].mean())
                    elif imputation_strategy == "constant":
                        val = fill_value
                    else:
                        val = 0.0

                impute_values[feat] = val
                df_out[feat] = df_out[feat].fillna(val)
                logger.info("  Imputed %d missing values in '%s' with %s = %.4f", missing_count, feat, imputation_strategy, val)
            else:
                # Store existing median/mean for future transform consistency
                if pd.api.types.is_numeric_dtype(df_out[feat]) and len(df_out[feat].dropna()) > 0:
                    val = float(df_out[feat].median()) if imputation_strategy == "median" else float(df_out[feat].mean())
                    impute_values[feat] = val

        state.imputation_values = impute_values if fitted_state is None else fitted_state.imputation_values

    final_shape = df_out.shape
    state.final_shape = final_shape

    # Verify zero missing values remain in active features
    remaining_nulls = int(df_out[active_features].isna().sum().sum())
    logger.info("  Active Features Missing Values Remaining: %d", remaining_nulls)
    logger.info("  Final Output Shape: %d rows x %d columns", final_shape[0], final_shape[1])

    return df_out, state


def main():
    """CLI test execution for preprocessing module."""
    from src.data.load_data import load_raw_data

    logger.info("Executing preprocessing demonstration...")
    raw_df = load_raw_data(validate=True)
    df_dedup = handle_duplicates(raw_df)
    df_clean, state = handle_missing_values(df_dedup)

    print(f"\n[Preprocessing Success]")
    print(f"  * Initial Shape: {state.initial_shape}")
    print(f"  * Final Cleaned Shape: {state.final_shape}")
    print(f"  * Columns Dropped (>40% missing): {len(state.dropped_columns)}")
    print(f"  * Imputation Values Used: {state.imputation_values}")
    print(f"  * Imputed Counts: {state.imputed_counts}")


if __name__ == "__main__":
    main()
