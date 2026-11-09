"""Strict schema and data integrity verification for train and test datasets."""

import numpy as np
import pandas as pd

from src.core.config import ValidationConfig


def validate_raw_dataframes(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_config: ValidationConfig,
) -> None:
    """Validate structural integrity, ID uniqueness, and target values.

    Args:
        train_df: Raw training DataFrame loaded from train.csv.
        test_df: Raw test DataFrame loaded from test.csv.
        val_config: Validation configuration specifying ID and target columns.

    Raises:
        ValueError: If any schema invariant is violated.
    """
    if train_df.empty or test_df.empty:
        raise ValueError("Train and test DataFrames must not be empty.")

    id_col = val_config.id_column
    target_col = val_config.target_column

    if id_col not in train_df.columns or id_col not in test_df.columns:
        raise ValueError(f"Missing required identifier column '{id_col}'.")

    if target_col not in train_df.columns:
        raise ValueError(f"Missing target column '{target_col}' in training data.")

    if target_col in test_df.columns:
        raise ValueError(f"Test dataset must not contain target column '{target_col}'.")

    _validate_feature_alignment(train_df, test_df, val_config)
    _validate_target_distribution(train_df[target_col])

    if train_df[id_col].duplicated().any() or test_df[id_col].duplicated().any():
        raise ValueError(f"Duplicate transaction IDs detected in column '{id_col}'.")


def _validate_feature_alignment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_config: ValidationConfig,
) -> None:
    """Ensure train and test sets share the exact same set of input feature columns."""
    excluded = {val_config.id_column, val_config.target_column}
    train_features = [c for c in train_df.columns if c not in excluded]
    test_features = [c for c in test_df.columns if c not in excluded]

    if set(train_features) != set(test_features):
        missing_in_test = set(train_features) - set(test_features)
        extra_in_test = set(test_features) - set(train_features)
        raise ValueError(
            f"Feature mismatch between train and test. "
            f"Missing in test: {missing_in_test}; Extra in test: {extra_in_test}"
        )


def _validate_target_distribution(target_series: pd.Series) -> None:
    """Verify binary {0, 1} target encoding with no missing values."""
    if target_series.isna().any():
        raise ValueError("Target column contains NaN values.")

    unique_vals = set(np.unique(target_series.to_numpy()))
    if not unique_vals.issubset({0, 1}):
        raise ValueError(f"Target column must be binary {{0, 1}}, found {unique_vals}.")

    if len(unique_vals) < 2:
        raise ValueError("Target column contains only a single class; ROC-AUC is undefined.")
