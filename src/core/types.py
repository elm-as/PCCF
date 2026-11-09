"""Domain data structures and immutable value objects for the ML pipeline."""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetBundle:
    """Immutable container holding train and test dataframes alongside metadata.

    Attributes:
        train_features: Training feature matrix (excluding id and target).
        train_target: Binary target series (0 = legitimate, 1 = fraud).
        train_ids: Unique transaction IDs for training records.
        test_features: Test feature matrix (excluding id).
        test_ids: Unique transaction IDs for test records.
        numerical_columns: Tuple of continuous/numerical column names.
        categorical_columns: Tuple of categorical/boolean column names.
    """

    train_features: pd.DataFrame
    train_target: pd.Series
    train_ids: pd.Series
    test_features: pd.DataFrame
    test_ids: pd.Series
    numerical_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]


@dataclass(frozen=True)
class FoldDataSplit:
    """Immutable representation of a single cross-validation fold split."""

    fold_index: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame


@dataclass(frozen=True)
class ModelPredictionArtifact:
    """Out-of-fold and test predictions produced by a single base model family."""

    model_name: str
    oof_probabilities: np.ndarray
    test_probabilities: np.ndarray
    fold_auc_scores: tuple[float, ...]
    overall_oof_auc: float


@dataclass(frozen=True)
class EnsembleBlendResult:
    """Result of the Rank-Logit and probability ensembling stage."""

    model_names: tuple[str, ...]
    optimal_weights: tuple[float, ...]
    conservative_oof_auc: float
    rank_logit_oof_auc: float
    conservative_test_predictions: np.ndarray
    rank_logit_test_predictions: np.ndarray
