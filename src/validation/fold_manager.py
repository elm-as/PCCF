"""Deterministic Stratified K-Fold generator and multi-model OOF cross-validation runner."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.core.config import PipelineConfig
from src.core.types import DatasetBundle, FoldDataSplit, ModelPredictionArtifact
from src.feature_engineering.cross_fitted_encoder import (
    EncodedFoldMatrices,
    encode_fold_without_leakage,
)
from src.modeling.catboost_shallow import train_and_predict_catboost_shallow
from src.modeling.lightgbm_shallow import train_and_predict_lightgbm_shallow
from src.modeling.linear_baseline import train_and_predict_spline_glm
from src.modeling.xgboost_shallow import train_and_predict_xgboost_shallow
from src.validation.metric_calculator import compute_roc_auc_score


MODEL_NAMES: tuple[str, ...] = (
    "spline_glm",
    "lightgbm_shallow",
    "catboost_shallow",
    "xgboost_shallow",
)


def generate_stratified_folds(
    bundle: DatasetBundle,
    n_splits: int,
    random_seed: int,
) -> tuple[FoldDataSplit, ...]:
    """Create deterministic StratifiedKFold splits over the training dataset.

    Args:
        bundle: Immutable DatasetBundle holding train and test features.
        n_splits: Number of cross-validation folds.
        random_seed: Seed for deterministic shuffling.

    Returns:
        Tuple of FoldDataSplit objects.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    splits: list[FoldDataSplit] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(
        skf.split(bundle.train_features, bundle.train_target)
    ):
        splits.append(
            FoldDataSplit(
                fold_index=fold_idx,
                train_indices=tr_idx,
                val_indices=val_idx,
                x_train=bundle.train_features.iloc[tr_idx],
                y_train=bundle.train_target.iloc[tr_idx],
                x_val=bundle.train_features.iloc[val_idx],
                y_val=bundle.train_target.iloc[val_idx],
                x_test=bundle.test_features,
            )
        )
    return tuple(splits)


def run_cross_validation_suite(
    bundle: DatasetBundle,
    target_encode_cols: tuple[str, ...],
    config: PipelineConfig,
) -> tuple[ModelPredictionArtifact, ...]:
    """Execute leak-free OOF training and evaluation for all four model paradigms.

    Args:
        bundle: Engineered DatasetBundle containing all base, digit, and interaction cols.
        target_encode_cols: Columns to target-encode inside each fold without leakage.
        config: Master PipelineConfig.

    Returns:
        Tuple of ModelPredictionArtifact instances for each model family.
    """
    folds = generate_stratified_folds(
        bundle, config.validation.n_splits, config.validation.random_seed
    )
    n_train = len(bundle.train_features)
    n_test = len(bundle.test_features)
    n_splits = config.validation.n_splits

    oof_buffers = {name: np.zeros(n_train, dtype=np.float64) for name in MODEL_NAMES}
    test_buffers = {name: np.zeros(n_test, dtype=np.float64) for name in MODEL_NAMES}
    fold_scores: dict[str, list[float]] = {name: [] for name in MODEL_NAMES}

    for fold in folds:
        print(f"\n[*] Processing Fold {fold.fold_index + 1}/{n_splits}...", flush=True)
        encoded = encode_fold_without_leakage(fold, target_encode_cols, config.validation)
        print(f"    - Leak-free OOF Target Encoding complete (features: {encoded.x_train_encoded.shape[1]})", flush=True)
        fold_outputs = _train_all_models_on_fold(encoded, (fold.y_train, fold.y_val), config)

        fold_scores_log = []
        for name, (val_pred, test_pred) in fold_outputs.items():
            oof_buffers[name][fold.val_indices] = val_pred
            test_buffers[name] += test_pred / float(n_splits)
            score = compute_roc_auc_score(fold.y_val.to_numpy(), val_pred)
            fold_scores[name].append(score)
            fold_scores_log.append(f"{name}: {score:.5f}")

        print(f"    - Fold {fold.fold_index + 1} validation scores: {' | '.join(fold_scores_log)}", flush=True)

    return _build_and_save_artifacts(
        (oof_buffers, test_buffers, fold_scores),
        bundle.train_target.to_numpy(),
        config,
    )


def _train_all_models_on_fold(
    encoded: EncodedFoldMatrices,
    targets: tuple[pd.Series, pd.Series],
    config: PipelineConfig,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Run all four model families on a single encoded fold."""
    y_train, y_val = targets
    print("      -> Training Spline-GLM baseline...", flush=True)
    spline_pred = train_and_predict_spline_glm(encoded, y_train, config.linear)

    print("      -> Training LightGBM shallow...", flush=True)
    lgbm_pred = train_and_predict_lightgbm_shallow(encoded, (y_train, y_val), config.lgbm)

    print("      -> Training CatBoost shallow...", flush=True)
    cat_pred = train_and_predict_catboost_shallow(encoded, (y_train, y_val), config.catboost)

    print("      -> Training XGBoost shallow...", flush=True)
    xgb_pred = train_and_predict_xgboost_shallow(encoded, (y_train, y_val), config.xgboost)

    return {
        "spline_glm": spline_pred,
        "lightgbm_shallow": lgbm_pred,
        "catboost_shallow": cat_pred,
        "xgboost_shallow": xgb_pred,
    }


def _build_and_save_artifacts(
    buffers_triplet: tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, list[float]],
    ],
    y_true: np.ndarray,
    config: PipelineConfig,
) -> tuple[ModelPredictionArtifact, ...]:
    """Persist OOF and test arrays to disk and return ModelPredictionArtifact objects."""
    oof_buffers, test_buffers, fold_scores = buffers_triplet
    config.paths.oof_artifact_dir.mkdir(parents=True, exist_ok=True)
    config.paths.test_artifact_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[ModelPredictionArtifact] = []
    for name in MODEL_NAMES:
        oof_arr = oof_buffers[name]
        test_arr = test_buffers[name]
        overall_auc = compute_roc_auc_score(y_true, oof_arr)

        np.save(config.paths.oof_artifact_dir / f"{name}_oof.npy", oof_arr)
        np.save(config.paths.test_artifact_dir / f"{name}_test.npy", test_arr)

        artifacts.append(
            ModelPredictionArtifact(
                model_name=name,
                oof_probabilities=oof_arr,
                test_probabilities=test_arr,
                fold_auc_scores=tuple(fold_scores[name]),
                overall_oof_auc=overall_auc,
            )
        )
    return tuple(artifacts)

