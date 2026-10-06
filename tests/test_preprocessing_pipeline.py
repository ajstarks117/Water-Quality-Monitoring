"""Unit Test Suite for Milestone 2.3: Preprocessing Pipeline (Leakage-Safe).

Test Cases:
    - TC-2.3-01: Pipeline object is a valid scikit-learn Pipeline with ColumnTransformer.
    - TC-2.3-02: Pipeline is NOT pre-fitted; fit only on X_train proves leakage safety.
    - TC-2.3-03: wqi_score leakage guard raises DataLeakageError when injected.
    - TC-2.3-04: No manual pandas transformation outside the pipeline — all
                 scaling/imputation is encapsulated inside the Pipeline object.
    Additional:
    - Full orchestrator integration test with run_full_cleaning_pipeline.
    - RobustScaler variant test.
    - Missing feature column ValueError test.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted

from src.data.load_data import load_raw_data
from src.data.preprocessing import (
    build_preprocessing_pipeline,
    run_full_cleaning_pipeline,
    handle_duplicates,
    handle_missing_values,
    handle_outliers,
    PreprocessingPipelineResult,
    DataLeakageError,
    FROZEN_CANDIDATE_FEATURES,
)


@pytest.fixture(scope="module")
def cleaned_df() -> pd.DataFrame:
    """Fixture: full M2.1+M2.2 cleaned DataFrame ready for M2.3 pipeline build."""
    raw_df = load_raw_data(validate=True)
    df_dedup = handle_duplicates(raw_df)
    df_clean, _ = handle_missing_values(df_dedup)
    df_final, _ = handle_outliers(df_clean)
    return df_final


@pytest.fixture(scope="module")
def pipeline_result(cleaned_df: pd.DataFrame) -> PreprocessingPipelineResult:
    """Fixture: build the M2.3 pipeline result from the cleaned DataFrame."""
    return build_preprocessing_pipeline(cleaned_df)


# ---------------------------------------------------------------------------
# TC-2.3-01: Pipeline Object Is a Valid scikit-learn Pipeline
# ---------------------------------------------------------------------------
def test_tc_2_3_01_pipeline_is_sklearn_pipeline(pipeline_result: PreprocessingPipelineResult):
    """TC-2.3-01: The returned pipeline must be a scikit-learn Pipeline wrapping a ColumnTransformer."""
    pipe = pipeline_result.pipeline

    # Must be a sklearn Pipeline
    assert isinstance(pipe, Pipeline), f"Expected sklearn.pipeline.Pipeline, got {type(pipe)}"

    # Must contain a ColumnTransformer as the 'preprocessor' step
    assert "preprocessor" in dict(pipe.steps), "Pipeline must have a 'preprocessor' step"
    preprocessor = pipe.named_steps["preprocessor"]
    assert isinstance(preprocessor, ColumnTransformer), (
        f"Expected ColumnTransformer, got {type(preprocessor)}"
    )

    # ColumnTransformer must have a 'numeric' transformer
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "numeric" in transformer_names, "ColumnTransformer must have a 'numeric' transformer"

    # Feature names must match FROZEN_CANDIDATE_FEATURES
    assert pipeline_result.feature_names == list(FROZEN_CANDIDATE_FEATURES)

    # X and y must be extracted
    assert pipeline_result.X is not None
    assert pipeline_result.y is not None
    assert pipeline_result.X.shape[1] == len(FROZEN_CANDIDATE_FEATURES)
    assert len(pipeline_result.y) == pipeline_result.X.shape[0]


# ---------------------------------------------------------------------------
# TC-2.3-02: Pipeline Is NOT Pre-Fitted (Leakage Safety)
# ---------------------------------------------------------------------------
def test_tc_2_3_02_pipeline_not_prefitted_leakage_safe(pipeline_result: PreprocessingPipelineResult):
    """TC-2.3-02: Pipeline must NOT be fitted at build time — fit only after train/test split."""
    pipe = pipeline_result.pipeline

    # The pipeline must NOT be fitted yet
    with pytest.raises(Exception):
        # check_is_fitted raises NotFittedError if the estimator is not fitted
        check_is_fitted(pipe)

    # Simulate train/test split and verify fit works on train only
    X = pipeline_result.X
    n_train = int(len(X) * 0.8)
    X_train = X.iloc[:n_train]
    X_test = X.iloc[n_train:]

    # Fit on training data ONLY
    pipe.fit(X_train)

    # Now it should be fitted
    check_is_fitted(pipe)

    # Transform both train and test — test uses TRAIN statistics
    X_train_transformed = pipe.transform(X_train)
    X_test_transformed = pipe.transform(X_test)

    # Verify shapes are preserved
    assert X_train_transformed.shape == (n_train, len(FROZEN_CANDIDATE_FEATURES))
    assert X_test_transformed.shape == (len(X) - n_train, len(FROZEN_CANDIDATE_FEATURES))

    # Training data should have zero mean (StandardScaler)
    train_means = np.abs(X_train_transformed.mean(axis=0))
    assert np.all(train_means < 1e-10), (
        f"Training set means should be ~0.0 after StandardScaler, got {train_means}"
    )

    # Test data should NOT have zero mean (fitted on train, not test)
    # This is the actual leakage safety proof — if test also had zero mean,
    # the scaler would have been fit on the full dataset
    # Note: This check is probabilistic but extremely reliable with real data
    test_means = np.abs(X_test_transformed.mean(axis=0))
    # At least one feature should have non-zero mean on test set
    assert np.any(test_means > 1e-6), (
        "Test set means are all zero — suggests scaler may have been fit on full data (leakage!)"
    )


# ---------------------------------------------------------------------------
# TC-2.3-03: wqi_score Leakage Guard Raises DataLeakageError
# ---------------------------------------------------------------------------
def test_tc_2_3_03_wqi_score_leakage_guard(cleaned_df: pd.DataFrame):
    """TC-2.3-03: Attempting to include wqi_score in features must raise DataLeakageError."""
    # Attempt to include wqi_score as a feature
    leaked_features = list(FROZEN_CANDIDATE_FEATURES) + ["wqi_score"]

    with pytest.raises(DataLeakageError, match="Leakage columns detected"):
        build_preprocessing_pipeline(
            cleaned_df,
            feature_columns=leaked_features,
        )

    # Also test with wqi_score as the only leakage column in a custom list
    with pytest.raises(DataLeakageError):
        build_preprocessing_pipeline(
            cleaned_df,
            feature_columns=["wqi_score"],
            leakage_columns=["wqi_score"],
        )


# ---------------------------------------------------------------------------
# TC-2.3-04: No Manual Pandas Transformations Outside Pipeline
# ---------------------------------------------------------------------------
def test_tc_2_3_04_no_manual_transforms_outside_pipeline(
    cleaned_df: pd.DataFrame, pipeline_result: PreprocessingPipelineResult
):
    """TC-2.3-04: All scaling/imputation happens inside the Pipeline; extracted X is raw."""
    X = pipeline_result.X

    # X must contain the original (unscaled) values from the cleaned DataFrame
    for feat in FROZEN_CANDIDATE_FEATURES:
        pd.testing.assert_series_equal(
            X[feat].reset_index(drop=True),
            cleaned_df[feat].reset_index(drop=True),
            check_names=False,
            obj=f"Feature '{feat}' in X should be identical to cleaned_df (no pre-scaling)",
        )

    # Verify that the pipeline's ColumnTransformer encapsulates all transformations
    preprocessor = pipeline_result.pipeline.named_steps["preprocessor"]
    numeric_transformer = None
    for name, transformer, cols in preprocessor.transformers:
        if name == "numeric":
            numeric_transformer = transformer
            break

    assert numeric_transformer is not None, "Numeric transformer missing from ColumnTransformer"
    assert isinstance(numeric_transformer, Pipeline), "Numeric transformer should be a Pipeline"

    # Verify the numeric pipeline contains imputer + scaler steps (no manual pandas ops)
    step_names = [name for name, _ in numeric_transformer.steps]
    assert "scaler" in step_names, "Numeric pipeline must contain a 'scaler' step"
    assert "imputer" in step_names, "Numeric pipeline must contain an 'imputer' step"


# ---------------------------------------------------------------------------
# Additional: Full Orchestrator Integration Test
# ---------------------------------------------------------------------------
def test_full_cleaning_pipeline_integration():
    """Integration test: run_full_cleaning_pipeline chains M2.1 → M2.2 → M2.3 correctly."""
    raw_df = load_raw_data(validate=True)
    result = run_full_cleaning_pipeline(raw_df)

    # Verify result structure
    assert isinstance(result, PreprocessingPipelineResult)
    assert isinstance(result.pipeline, Pipeline)
    assert result.cleaning_state is not None
    assert result.outlier_report is not None
    assert result.X is not None
    assert result.y is not None

    # Verify X has no NaN values (M2.1 imputation should have handled them)
    assert result.X.isna().sum().sum() == 0, "Feature matrix should have zero NaN after cleaning"

    # Verify target has expected classes
    expected_classes = {"Excellent", "Good", "Poor", "Very Poor", "Unsuitable"}
    actual_classes = set(result.y.unique())
    assert actual_classes == expected_classes, f"Unexpected classes: {actual_classes}"


# ---------------------------------------------------------------------------
# Additional: RobustScaler Variant
# ---------------------------------------------------------------------------
def test_robust_scaler_variant(cleaned_df: pd.DataFrame):
    """Verify that the 'robust' scaler variant builds a valid pipeline with RobustScaler."""
    from sklearn.preprocessing import RobustScaler

    result = build_preprocessing_pipeline(cleaned_df, scaler="robust")
    preprocessor = result.pipeline.named_steps["preprocessor"]

    # Find the numeric transformer's scaler step
    for name, transformer, cols in preprocessor.transformers:
        if name == "numeric":
            scaler_step = transformer.named_steps["scaler"]
            assert isinstance(scaler_step, RobustScaler), (
                f"Expected RobustScaler, got {type(scaler_step)}"
            )
            break


# ---------------------------------------------------------------------------
# Additional: Missing Feature Column Error
# ---------------------------------------------------------------------------
def test_missing_feature_column_raises_valueerror(cleaned_df: pd.DataFrame):
    """Attempting to use a non-existent feature column raises ValueError."""
    with pytest.raises(ValueError, match="Required feature columns missing"):
        build_preprocessing_pipeline(
            cleaned_df,
            feature_columns=["NonExistentColumn", "AnotherFakeColumn"],
        )
