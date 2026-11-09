"""Shallow symmetric-tree CatBoost classifier for oblivious feature interactions."""

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from src.core.config import CatBoostShallowConfig
from src.feature_engineering.cross_fitted_encoder import EncodedFoldMatrices


def train_and_predict_catboost_shallow(
    encoded_fold: EncodedFoldMatrices,
    targets: tuple[pd.Series, pd.Series],
    config: CatBoostShallowConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a shallow CatBoost classifier using symmetric oblivious trees.

    Why symmetric trees matter:
    CatBoost enforces the exact same splitting criterion across an entire tree level,
    acting as a strong regularizer against noisy synthetic interactions in S6E11.

    Args:
        encoded_fold: Leak-free encoded train, validation, and test feature matrices.
        targets: Tuple of (y_train, y_val) binary target Series.
        config: Shallow CatBoost hyperparameters (depth=4, l2_leaf_reg=12.0).

    Returns:
        Tuple of (val_probabilities, test_probabilities) as 1D float64 numpy arrays.
    """
    y_train, y_val = targets

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=config.iterations,
        learning_rate=config.learning_rate,
        depth=config.depth,
        l2_leaf_reg=config.l2_leaf_reg,
        random_strength=config.random_strength,
        subsample=config.subsample,
        random_seed=config.random_seed,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(
        encoded_fold.x_train_encoded,
        y_train.to_numpy(dtype=np.int32),
        eval_set=(encoded_fold.x_val_encoded, y_val.to_numpy(dtype=np.int32)),
        early_stopping_rounds=config.early_stopping_rounds,
        verbose=False,
    )

    val_preds = model.predict_proba(encoded_fold.x_val_encoded)[:, 1].astype(np.float64)
    test_preds = model.predict_proba(encoded_fold.x_test_encoded)[:, 1].astype(np.float64)
    return val_preds, test_preds

