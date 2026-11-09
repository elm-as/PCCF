"""Leak-free Cross-Fitted (Out-Of-Fold) Bayesian Target Encoder.

Why strict cross-fitting is enforced:
In S6E11, encoding high-cardinality digit keys and categorical interactions without
nested fold separation leaks the target into the training features, causing a ~0.030
AUC collapse on validation/test. Here, training fold rows are encoded exclusively
using inner StratifiedKFold out-of-fold statistics with Bayesian m-estimate smoothing,
while validation and test splits are mapped from full training fold statistics.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.core.config import ValidationConfig
from src.core.types import FoldDataSplit


@dataclass(frozen=True)
class EncodedFoldMatrices:
    """Numeric feature matrices produced after leak-free fold target encoding."""

    x_train_encoded: pd.DataFrame
    x_val_encoded: pd.DataFrame
    x_test_encoded: pd.DataFrame
    categorical_feature_names: tuple[str, ...]


def encode_fold_without_leakage(
    fold_split: FoldDataSplit,
    target_encode_cols: tuple[str, ...],
    val_config: ValidationConfig,
) -> EncodedFoldMatrices:
    """Apply inner cross-fitted target encoding and frequency encoding to a fold split.

    Args:
        fold_split: Immutable fold split containing x_train, y_train, x_val, and x_test.
        target_encode_cols: Tuple of categorical and digit-key columns to target-encode.
        val_config: Validation config holding smoothing weight and inner split count.

    Returns:
        EncodedFoldMatrices with strictly leak-free target-encoded features.
    """
    x_tr = fold_split.x_train.copy()
    x_va = fold_split.x_val.copy()
    x_te = fold_split.x_test.copy()
    y_tr = fold_split.y_train.to_numpy(dtype=np.float64)

    prior_mean = float(np.mean(y_tr))
    inner_skf = StratifiedKFold(
        n_splits=val_config.inner_encoding_splits,
        shuffle=True,
        random_state=val_config.random_seed + fold_split.fold_index,
    )
    inner_splits = list(inner_skf.split(x_tr, fold_split.y_train))

    tr_te_cols: dict[str, np.ndarray] = {}
    va_te_cols: dict[str, np.ndarray] = {}
    te_te_cols: dict[str, np.ndarray] = {}

    for col in target_encode_cols:
        if col not in x_tr.columns:
            continue
        oof_tr, mapped_va, mapped_te = _compute_single_column_te(
             (x_tr[col], x_va[col], x_te[col]),
            y_tr,
            (inner_splits, prior_mean, val_config.bayesian_smoothing_weight),
        )
        feat_name = f"{col}__oof_te"
        tr_te_cols[feat_name] = oof_tr
        va_te_cols[feat_name] = mapped_va
        te_te_cols[feat_name] = mapped_te

    return _finalize_encoded_matrices(
        (x_tr, x_va, x_te),
        (tr_te_cols, va_te_cols, te_te_cols),
        target_encode_cols,
    )


def _compute_single_column_te(
    series_triplet: tuple[pd.Series, pd.Series, pd.Series],
    y_tr: np.ndarray,
    encoding_params: tuple[list[tuple[np.ndarray, np.ndarray]], float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute inner OOF target encoding for train and full-fold mapping for val/test."""
    s_tr, s_va, s_te = series_triplet
    inner_splits, prior_mean, smoothing_m = encoding_params

    oof_train = np.full(len(s_tr), prior_mean, dtype=np.float32)
    tr_values = s_tr.to_numpy()

    for in_tr_idx, in_val_idx in inner_splits:
        mapping = _fit_bayesian_smoothed_map(
            tr_values[in_tr_idx], y_tr[in_tr_idx], (prior_mean, smoothing_m)
        )
        oof_train[in_val_idx] = (
            pd.Series(tr_values[in_val_idx]).map(mapping).fillna(prior_mean).to_numpy(dtype=np.float32)
        )

    full_mapping = _fit_bayesian_smoothed_map(tr_values, y_tr, (prior_mean, smoothing_m))
    val_encoded = s_va.map(full_mapping).fillna(prior_mean).to_numpy(dtype=np.float32)
    test_encoded = s_te.map(full_mapping).fillna(prior_mean).to_numpy(dtype=np.float32)
    return oof_train, val_encoded, test_encoded


def _fit_bayesian_smoothed_map(
    keys: np.ndarray,
    targets: np.ndarray,
    smoothing_cfg: tuple[float, float],
) -> dict[object, float]:
    """Compute m-estimate smoothed target rate: (sum_y + m * prior) / (count + m)."""
    prior_mean, smoothing_m = smoothing_cfg
    df_temp = pd.DataFrame({"key": keys, "target": targets})
    stats = df_temp.groupby("key", sort=False)["target"].agg(["count", "sum"])
    smoothed = (stats["sum"] + smoothing_m * prior_mean) / (stats["count"] + smoothing_m)
    return smoothed.to_dict()


def _finalize_encoded_matrices(
    raw_triplet: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    te_dicts: tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]],
    cat_cols: tuple[str, ...],
) -> EncodedFoldMatrices:
    """Append OOF target-encoded features and ordinal-encode raw string columns."""
    x_tr, x_va, x_te = raw_triplet
    tr_te, va_te, te_te = te_dicts

    x_tr_out = pd.concat([x_tr, pd.DataFrame(tr_te, index=x_tr.index)], axis=1)
    x_va_out = pd.concat([x_va, pd.DataFrame(va_te, index=x_va.index)], axis=1)
    x_te_out = pd.concat([x_te, pd.DataFrame(te_te, index=x_te.index)], axis=1)

    present_cats = [c for c in cat_cols if c in x_tr_out.columns]
    for col in present_cats:
        unique_keys = pd.Index(x_tr_out[col].unique())
        code_map = {val: idx for idx, val in enumerate(unique_keys)}
        x_tr_out[col] = x_tr_out[col].map(code_map).fillna(-1).astype(np.int32)
        x_va_out[col] = x_va_out[col].map(code_map).fillna(-1).astype(np.int32)
        x_te_out[col] = x_te_out[col].map(code_map).fillna(-1).astype(np.int32)

    return EncodedFoldMatrices(
        x_train_encoded=x_tr_out,
        x_val_encoded=x_va_out,
        x_test_encoded=x_te_out,
        categorical_feature_names=tuple(present_cats),
    )
