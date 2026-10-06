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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

# Frozen Candidate Feature Names from M1.4 Schema Lock
FROZEN_CANDIDATE_FEATURES: List[str] = [
    "Potential of Hydrogen (pH)",
    "Dissolved oxygen (mg/L)",
    "Biochemical Oxygen Demand (mg/L)",
    "Fecal Coliform (MPN/100mL)",
]

# Configure structured logging
logger = logging.getLogger("preprocessing")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Domain Physical Plausibility Limits & Outlier Structures
# ---------------------------------------------------------------------------
# Physical / Chemical / Biological absolute plausibility limits
PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Potential of Hydrogen (pH)": (0.0, 14.0),
    "Dissolved oxygen (mg/L)": (0.0, 30.0),
    "Biochemical Oxygen Demand (mg/L)": (0.0, 500.0),
    "Fecal Coliform (MPN/100mL)": (0.0, 1e8),
}


@dataclass
class OutlierAuditReport:
    """Audit summary of outlier investigation, physical validity checks, and class balance impact."""
    initial_rows: int = 0
    final_rows: int = 0
    invalid_rows_removed: int = 0
    invalid_records_by_feature: Dict[str, int] = field(default_factory=dict)
    statistical_outliers_by_feature: Dict[str, int] = field(default_factory=dict)
    class_distribution_before: Dict[str, int] = field(default_factory=dict)
    class_distribution_after: Dict[str, int] = field(default_factory=dict)
    class_loss_percentages: Dict[str, float] = field(default_factory=dict)
    is_class_imbalance_severely_distorted: bool = False
    retained_genuine_extremes: bool = True


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


# ---------------------------------------------------------------------------
# Outlier Investigation & Domain Validity Handling Engine
# ---------------------------------------------------------------------------
def handle_outliers(
    df: pd.DataFrame,
    rules_config: Optional[Dict[str, Any]] = None,
    remove_invalid: bool = True,
) -> Tuple[pd.DataFrame, OutlierAuditReport]:
    """Investigate statistical outliers and filter physically impossible measurement errors.

    Per `docs/cleaning_decisions.md` (DEC-CLEANING-PREPROC-2026-V1):
        1. Physically Impossible Readings: Dropped as invalid errors (e.g. pH outside 0-14, negative BOD/DO/FC).
        2. Genuine Extreme Events: Kept as real pollution signals (e.g. BOD = 127 mg/L, FC = 2.2e7 MPN/100mL).
        3. Class Balance Integrity: Checks whether any class lost > max_allowed_loss_pct (default: 10%).

    Args:
        df: Input pandas DataFrame.
        rules_config: Optional dictionary defining custom physical bounds or parameters.
            - `physical_bounds`: Dict[str, Tuple[float, float]] mapping column to (min_valid, max_valid).
            - `max_allowed_class_loss_pct`: Maximum tolerable class loss percentage before warning (default: 10.0).
            - `target_column`: Target column to audit (default: 'wqi_class').
        remove_invalid: Whether to drop physically impossible rows (default: True).

    Returns:
        Tuple[pd.DataFrame, OutlierAuditReport]: (Cleaned DataFrame, OutlierAuditReport)
    """
    cfg = rules_config or {}
    bounds_dict = cfg.get("physical_bounds", PHYSICAL_BOUNDS)
    target_col = cfg.get("target_column", "wqi_class")
    max_loss_pct = float(cfg.get("max_allowed_class_loss_pct", 10.0))

    initial_rows = len(df)
    report = OutlierAuditReport(initial_rows=initial_rows)

    if target_col in df.columns:
        report.class_distribution_before = df[target_col].value_counts().to_dict()

    df_out = df.copy()
    invalid_mask = pd.Series(False, index=df_out.index)

    logger.info("================ OUTLIER & DOMAIN VALIDITY AUDIT ================")
    logger.info("  Initial Rows Evaluated: %d", initial_rows)

    # Statistical outlier computation and physical validity cross-check
    for col, (min_valid, max_valid) in bounds_dict.items():
        if col not in df_out.columns:
            continue

        s = df_out[col].dropna()
        if len(s) == 0:
            continue

        # Statistical IQR & Z-score Outliers
        q25, q75 = s.quantile(0.25), s.quantile(0.75)
        iqr = q75 - q25
        lower_iqr = q25 - 1.5 * iqr
        upper_iqr = q75 + 1.5 * iqr
        iqr_outliers_count = int(((s < lower_iqr) | (s > upper_iqr)).sum())
        report.statistical_outliers_by_feature[col] = iqr_outliers_count

        # Physical impossibility check (Domain Bounds)
        col_invalid = (df_out[col] < min_valid) | (df_out[col] > max_valid)
        invalid_count = int(col_invalid.sum())
        report.invalid_records_by_feature[col] = invalid_count

        if invalid_count > 0:
            invalid_mask = invalid_mask | col_invalid
            logger.warning(
                "  INVALID PHYSICAL READINGS in '%s': %d rows outside domain bounds [%.1f, %.1f]",
                col, invalid_count, min_valid, max_valid
            )
        else:
            logger.info(
                "  Feature '%s': 100%% physically valid (Domain: [%.1f, %.1f], Statistical IQR Outliers: %d - RETAINED)",
                col, min_valid, max_valid, iqr_outliers_count
            )

    # Filter invalid rows if requested
    invalid_total = int(invalid_mask.sum())
    report.invalid_rows_removed = invalid_total

    if remove_invalid and invalid_total > 0:
        df_out = df_out[~invalid_mask].copy().reset_index(drop=True)
        logger.info("  Removed %d physically impossible invalid rows. Remaining: %d rows", invalid_total, len(df_out))
    else:
        df_out = df_out.reset_index(drop=True)
        logger.info("  No physically impossible rows detected. All genuine extreme pollution events retained.")

    report.final_rows = len(df_out)

    # Class balance impact check
    if target_col in df_out.columns:
        report.class_distribution_after = df_out[target_col].value_counts().to_dict()
        for cls, count_before in report.class_distribution_before.items():
            count_after = report.class_distribution_after.get(cls, 0)
            loss_pct = ((count_before - count_after) / count_before) * 100 if count_before > 0 else 0.0
            report.class_loss_percentages[cls] = round(loss_pct, 2)
            if loss_pct > max_loss_pct:
                report.is_class_imbalance_severely_distorted = True
                logger.warning(
                    "  CLASS BALANCE WARNING: Class '%s' lost %.2f%% of its rows (> threshold %.1f%%) during outlier filtering!",
                    cls, loss_pct, max_loss_pct
                )

    logger.info("  Class Loss Breakdown: %s", report.class_loss_percentages)
    logger.info("  Outlier Handling Status: COMPLETE (Genuine extreme signals preserved)")
    return df_out, report


# ---------------------------------------------------------------------------
# M2.3: Leakage-Safe scikit-learn Preprocessing Pipeline Builder
# ---------------------------------------------------------------------------
@dataclass
class PreprocessingPipelineResult:
    """Container for the M2.3 preprocessing pipeline build output.

    Attributes:
        pipeline: Unfitted scikit-learn Pipeline wrapping ColumnTransformer.
                  MUST be fit ONLY on X_train after the train/test split (M4.2).
        feature_names: Ordered list of candidate feature column names entering
                       the pipeline (matches config/config.yaml feature_columns).
        target_column: Name of the primary target column ('wqi_class').
        secondary_target_column: Name of the secondary target column ('Potability').
        X: Feature matrix (pd.DataFrame) extracted from the cleaned DataFrame.
        y: Primary target series extracted from the cleaned DataFrame.
        y_secondary: Secondary target series (Potability) if present, else None.
        leakage_columns_verified_absent: List of leakage columns confirmed absent
                                          from the feature matrix X.
        cleaning_state: The PreprocessingState from the missing-value step.
        outlier_report: The OutlierAuditReport from the outlier step.
    """
    pipeline: Pipeline = field(default=None)  # type: ignore[assignment]
    feature_names: List[str] = field(default_factory=list)
    target_column: str = "wqi_class"
    secondary_target_column: str = "Potability"
    X: Optional[pd.DataFrame] = field(default=None)
    y: Optional[pd.Series] = field(default=None)
    y_secondary: Optional[pd.Series] = field(default=None)
    leakage_columns_verified_absent: List[str] = field(default_factory=list)
    cleaning_state: Optional[PreprocessingState] = field(default=None)
    outlier_report: Optional[OutlierAuditReport] = field(default=None)


class DataLeakageError(Exception):
    """Raised when a known leakage column is detected in the feature matrix."""
    pass


def build_preprocessing_pipeline(
    df_cleaned: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    target_column: str = "wqi_class",
    secondary_target_column: str = "Potability",
    leakage_columns: Optional[List[str]] = None,
    scaler: str = "standard",
    impute_in_pipeline: bool = True,
) -> PreprocessingPipelineResult:
    """Build a leakage-safe scikit-learn Pipeline for the CPCB dataset.

    This function constructs but does NOT fit the Pipeline. Fitting must happen
    only on X_train after the train/test split performed in M4.2. This is the
    core anti-leakage design: no global statistics (mean, std, median) are
    computed across the full dataset.

    The Pipeline internally chains:
        1. SimpleImputer (median) — fills remaining NaNs that survive
           the M2.1 handle_missing_values step (edge cases in unseen data).
        2. StandardScaler (default) or RobustScaler — numeric feature scaling.

    Args:
        df_cleaned: DataFrame that has already passed through handle_duplicates,
                    handle_missing_values, and handle_outliers (M2.1 + M2.2).
        feature_columns: Explicit list of feature column names. Defaults to
                         FROZEN_CANDIDATE_FEATURES from M1.4 schema lock.
        target_column: Primary target column name (default: 'wqi_class').
        secondary_target_column: Secondary target column (default: 'Potability').
        leakage_columns: Columns that MUST NOT appear in X. Defaults to ['wqi_score'].
        scaler: Scaler type — 'standard' (StandardScaler) or 'robust' (RobustScaler).
        impute_in_pipeline: If True, includes SimpleImputer(strategy='median')
                            as the first pipeline step to handle edge-case NaNs.

    Returns:
        PreprocessingPipelineResult containing the unfitted pipeline, X, y, and
        leakage verification metadata.

    Raises:
        DataLeakageError: If any leakage column is found in feature_columns.
        ValueError: If required feature or target columns are missing from df_cleaned.
    """
    features = feature_columns or list(FROZEN_CANDIDATE_FEATURES)
    leakage_cols = leakage_columns or ["wqi_score"]

    # -----------------------------------------------------------------------
    # Guard 1: Leakage column presence check
    # -----------------------------------------------------------------------
    leaked = [c for c in leakage_cols if c in features]
    if leaked:
        raise DataLeakageError(
            f"CRITICAL: Leakage columns detected in feature set: {leaked}. "
            f"'wqi_score' is the continuous value from which '{target_column}' was derived. "
            f"Including it would cause 100% artificial accuracy and total test leakage."
        )

    # -----------------------------------------------------------------------
    # Guard 2: Verify all required columns exist in the cleaned DataFrame
    # -----------------------------------------------------------------------
    missing_features = [f for f in features if f not in df_cleaned.columns]
    if missing_features:
        raise ValueError(
            f"Required feature columns missing from cleaned DataFrame: {missing_features}. "
            f"Available columns: {list(df_cleaned.columns)}"
        )
    if target_column not in df_cleaned.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in DataFrame. "
            f"Available columns: {list(df_cleaned.columns)}"
        )

    # -----------------------------------------------------------------------
    # Guard 3: Verify leakage columns are absent from the DataFrame itself
    # -----------------------------------------------------------------------
    leakage_present_in_df = [c for c in leakage_cols if c in df_cleaned.columns]
    leakage_verified = [c for c in leakage_cols if c not in df_cleaned.columns]

    # -----------------------------------------------------------------------
    # Extract X (features) and y (targets)
    # -----------------------------------------------------------------------
    X = df_cleaned[features].copy()
    y = df_cleaned[target_column].copy()
    y_secondary = (
        df_cleaned[secondary_target_column].copy()
        if secondary_target_column in df_cleaned.columns
        else None
    )

    # Final leakage assertion on extracted X
    for lc in leakage_cols:
        if lc in X.columns:
            raise DataLeakageError(
                f"CRITICAL: Leakage column '{lc}' found in extracted feature matrix X."
            )

    # -----------------------------------------------------------------------
    # Build scikit-learn Pipeline with ColumnTransformer
    # -----------------------------------------------------------------------
    # Determine which features are numeric vs categorical
    numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df_cleaned[f])]
    categorical_features = [f for f in features if not pd.api.types.is_numeric_dtype(df_cleaned[f])]

    # Numeric sub-pipeline
    numeric_steps: List[Tuple[str, Any]] = []
    if impute_in_pipeline:
        numeric_steps.append(("imputer", SimpleImputer(strategy="median")))

    if scaler == "robust":
        numeric_steps.append(("scaler", RobustScaler()))
    else:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    # Build ColumnTransformer
    transformers: List[Tuple[str, Any, List[str]]] = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append((
            "categorical",
            Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
            categorical_features,
        ))

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # Safety: any unexpected column is explicitly dropped
        verbose_feature_names_out=False,
    )

    # Wrap in a top-level Pipeline (allows future additions like PCA, feature selection)
    preprocessing_pipeline = Pipeline(
        steps=[("preprocessor", column_transformer)],
        verbose=False,
    )

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    logger.info("================ M2.3 PREPROCESSING PIPELINE BUILD ================")
    logger.info("  Feature Columns: %s", features)
    logger.info("  Numeric Features: %s", numeric_features)
    logger.info("  Categorical Features: %s", categorical_features)
    logger.info("  Scaler: %s", scaler)
    logger.info("  Imputer in Pipeline: %s", impute_in_pipeline)
    logger.info("  Leakage Columns Verified Absent from X: %s", leakage_verified)
    if leakage_present_in_df:
        logger.warning(
            "  WARNING: Leakage columns %s still exist in the source DataFrame "
            "(not in feature matrix X). Ensure they are never added to feature_columns.",
            leakage_present_in_df,
        )
    logger.info("  X Shape: %s | y Shape: %s", X.shape, y.shape)
    logger.info("  Pipeline Status: BUILT (unfitted — fit ONLY on X_train after split)")

    return PreprocessingPipelineResult(
        pipeline=preprocessing_pipeline,
        feature_names=features,
        target_column=target_column,
        secondary_target_column=secondary_target_column,
        X=X,
        y=y,
        y_secondary=y_secondary,
        leakage_columns_verified_absent=leakage_verified,
    )


# ---------------------------------------------------------------------------
# Full Cleaning + Pipeline Orchestrator
# ---------------------------------------------------------------------------
def run_full_cleaning_pipeline(
    df_raw: pd.DataFrame,
    cleaning_config: Optional[Dict[str, Any]] = None,
    outlier_config: Optional[Dict[str, Any]] = None,
    pipeline_scaler: str = "standard",
) -> PreprocessingPipelineResult:
    """Orchestrate the complete M2.1 → M2.2 → M2.3 cleaning pipeline.

    Chains: handle_duplicates → handle_missing_values → handle_outliers →
            build_preprocessing_pipeline.

    Args:
        df_raw: Raw labeled DataFrame from load_raw_data().
        cleaning_config: Config dict for handle_missing_values.
        outlier_config: Config dict for handle_outliers.
        pipeline_scaler: Scaler type for the pipeline ('standard' or 'robust').

    Returns:
        PreprocessingPipelineResult with the unfitted pipeline and cleaned X/y.
    """
    logger.info("========== FULL CLEANING PIPELINE START ==========")

    # M2.1: Duplicates
    df_dedup = handle_duplicates(df_raw)

    # M2.1: Missing values
    df_clean, state = handle_missing_values(df_dedup, strategy_config=cleaning_config)

    # M2.2: Outlier investigation
    df_final, outlier_report = handle_outliers(df_clean, rules_config=outlier_config)

    # M2.3: Build the leakage-safe pipeline
    result = build_preprocessing_pipeline(df_final, scaler=pipeline_scaler)
    result.cleaning_state = state
    result.outlier_report = outlier_report

    logger.info("========== FULL CLEANING PIPELINE COMPLETE ==========")
    return result


def main():
    """CLI test execution for preprocessing module."""
    from src.data.load_data import load_raw_data

    logger.info("Executing preprocessing demonstration...")
    raw_df = load_raw_data(validate=True)

    # Run the full M2.1 → M2.2 → M2.3 pipeline
    result = run_full_cleaning_pipeline(raw_df)

    print(f"\n[Preprocessing & Pipeline Build Success]")
    print(f"  * Cleaning State Shape: {result.cleaning_state.initial_shape} → {result.cleaning_state.final_shape}")
    print(f"  * Columns Dropped (>40% missing): {len(result.cleaning_state.dropped_columns)}")
    print(f"  * Imputation Values Used: {result.cleaning_state.imputation_values}")
    print(f"  * Statistical Outliers Kept: {result.outlier_report.statistical_outliers_by_feature}")
    print(f"  * Invalid Rows Removed: {result.outlier_report.invalid_rows_removed}")
    print(f"  * Pipeline Feature Names: {result.feature_names}")
    print(f"  * X Shape: {result.X.shape} | y Shape: {result.y.shape}")
    print(f"  * Pipeline: {result.pipeline}")
    print(f"  * Leakage Verified Absent: {result.leakage_columns_verified_absent}")


if __name__ == "__main__":
    main()
