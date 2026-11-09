"""Spline-augmented Generalized Linear Model (GLM) for additive fraud risk estimation.

Why a Spline-GLM is essential in S6E11:
Because the synthetic fraud label generation process is dominated by additive main
effects and OOF target-encoded digit risks, a regularized Logistic Regression over
B-spline basis expansions of continuous features provides low-variance probability
estimates that complement axis-aligned tree splits during Rank-Logit fusion.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, SplineTransformer

from src.core.config import LinearAdditiveConfig
from src.feature_engineering.cross_fitted_encoder import EncodedFoldMatrices


def train_and_predict_spline_glm(
    encoded_fold: EncodedFoldMatrices,
    y_train: pd.Series,
    config: LinearAdditiveConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a B-Spline regularized Logistic Regression on fold training data.

    Args:
        encoded_fold: Leak-free encoded train, validation, and test feature matrices.
        y_train: Binary target series for the training fold.
        config: Hyperparameters for spline knots, degree, and L2 regularization.

    Returns:
        Tuple of (val_probabilities, test_probabilities) as 1D float64 numpy arrays.
    """
    raw_cats = set(encoded_fold.categorical_feature_names)
    continuous_cols = [
        c for c in encoded_fold.x_train_encoded.columns
        if c not in raw_cats
    ]

    x_tr = _sanitize_numeric_array(encoded_fold.x_train_encoded[continuous_cols])
    x_va = _sanitize_numeric_array(encoded_fold.x_val_encoded[continuous_cols])
    x_te = _sanitize_numeric_array(encoded_fold.x_test_encoded[continuous_cols])

    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        (
            "splines",
            SplineTransformer(
                n_knots=config.n_knots,
                degree=config.spline_degree,
                include_bias=False,
            ),
        ),
        (
            "glm",
            LogisticRegression(
                C=config.inverse_regularization_c,
                max_iter=config.max_iter,
                random_state=config.random_seed,
            ),
        ),
    ])

    pipeline.fit(x_tr, y_train.to_numpy(dtype=np.int32))
    val_preds = pipeline.predict_proba(x_va)[:, 1].astype(np.float64)
    test_preds = pipeline.predict_proba(x_te)[:, 1].astype(np.float64)
    return val_preds, test_preds


def _sanitize_numeric_array(df: pd.DataFrame) -> np.ndarray:
    """Convert DataFrame to finite float32 numpy array, replacing any NaN/Inf with 0."""
    arr = df.to_numpy(dtype=np.float32, copy=True)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
