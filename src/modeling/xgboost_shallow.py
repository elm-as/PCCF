"""Depth-constrained histogram XGBoost classifier."""

import numpy as np
import pandas as pd
import xgboost as xgb

from src.core.config import XGBoostShallowConfig
from src.feature_engineering.cross_fitted_encoder import EncodedFoldMatrices


def train_and_predict_xgboost_shallow(
    encoded_fold: EncodedFoldMatrices,
    targets: tuple[pd.Series, pd.Series],
    config: XGBoostShallowConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a shallow depth-3 XGBoost model with strong L1/L2 and child weight regularization.

    Args:
        encoded_fold: Leak-free encoded train, validation, and test feature matrices.
        targets: Tuple of (y_train, y_val) binary target Series.
        config: Shallow XGBoost hyperparameters.

    Returns:
        Tuple of (val_probabilities, test_probabilities) as 1D float64 numpy arrays.
    """
    y_train, y_val = targets

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        early_stopping_rounds=config.early_stopping_rounds,
        random_state=config.random_seed,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        encoded_fold.x_train_encoded,
        y_train.to_numpy(dtype=np.int32),
        eval_set=[(encoded_fold.x_val_encoded, y_val.to_numpy(dtype=np.int32))],
        verbose=False,
    )

    val_preds = model.predict_proba(encoded_fold.x_val_encoded)[:, 1].astype(np.float64)
    test_preds = model.predict_proba(encoded_fold.x_test_encoded)[:, 1].astype(np.float64)
    return val_preds, test_preds

