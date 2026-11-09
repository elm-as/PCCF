"""Strict validation and formatting of competition submission DataFrames."""

import numpy as np
import pandas as pd

from src.core.config import ValidationConfig


def build_validated_submission_dataframe(
    test_ids: pd.Series,
    predicted_probabilities: np.ndarray,
    val_config: ValidationConfig,
) -> pd.DataFrame:
    """Create and validate a Kaggle submission DataFrame with columns (id, is_fraud).

    Args:
        test_ids: 1D Series of unique test transaction IDs.
        predicted_probabilities: 1D numpy array of predicted fraud probabilities.
        val_config: ValidationConfig defining exact column names.

    Returns:
        Validated two-column DataFrame ready for CSV export.

    Raises:
        ValueError: If lengths mismatch, IDs contain duplicates/NaN, or probs are invalid.
    """
    if len(test_ids) == 0:
        raise ValueError("test_ids must not be empty.")

    if len(test_ids) != len(predicted_probabilities):
        raise ValueError(
            f"Submission row count mismatch: {len(test_ids)} IDs vs "
            f"{len(predicted_probabilities)} predictions."
        )

    if test_ids.isna().any() or test_ids.duplicated().any():
        raise ValueError("test_ids contains NaN or duplicate identifiers.")

    if not np.all(np.isfinite(predicted_probabilities)):
        raise ValueError("Predicted probabilities contain NaN or infinite values.")

    if np.any(predicted_probabilities < 0.0) or np.any(predicted_probabilities > 1.0):
        raise ValueError("Predicted probabilities must lie strictly within [0.0, 1.0].")

    return pd.DataFrame({
        val_config.id_column: test_ids.to_numpy(dtype=np.int64),
        val_config.target_column: np.round(predicted_probabilities, 6),
    })

