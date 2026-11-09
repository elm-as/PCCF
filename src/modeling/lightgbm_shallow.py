"""Shallow-leaf LightGBM booster tailored for S6E11 additive and second-order signals."""

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.core.config import LightGBMShallowConfig
from src.feature_engineering.cross_fitted_encoder import EncodedFoldMatrices


def train_and_predict_lightgbm_shallow(
    encoded_fold: EncodedFoldMatrices,
    targets: tuple[pd.Series, pd.Series],
    config: LightGBMShallowConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a shallow LightGBM classifier on a single fold with early stopping.

    Args:
        encoded_fold: Leak-free encoded train, validation, and test feature matrices.
        targets: Tuple of (y_train, y_val) binary target Series.
        config: Shallow LightGBM hyperparameters (num_leaves=8, max_depth=3).

    Returns:
        Tuple of (val_probabilities, test_probabilities) as 1D float64 numpy arrays.
    """
    y_train, y_val = targets

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        subsample_freq=config.subsample_freq,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        random_state=config.random_seed,
        n_jobs=-1,
        verbose=-1,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=config.early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model.fit(
        encoded_fold.x_train_encoded,
        y_train.to_numpy(dtype=np.int32),
        eval_set=[(encoded_fold.x_val_encoded, y_val.to_numpy(dtype=np.int32))],
        eval_metric="auc",
        callbacks=callbacks,
    )

    val_preds = model.predict_proba(encoded_fold.x_val_encoded)[:, 1].astype(np.float64)
    test_preds = model.predict_proba(encoded_fold.x_test_encoded)[:, 1].astype(np.float64)
    return val_preds, test_preds

