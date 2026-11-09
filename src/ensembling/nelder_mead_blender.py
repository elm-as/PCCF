"""Nelder-Mead simplex and L2-regularized logit optimization for OOF ROC-AUC ensembling."""

import numpy as np
from scipy.optimize import minimize

from src.core.types import EnsembleBlendResult, ModelPredictionArtifact
from src.ensembling.rank_transformer import (
    logits_to_probabilities,
    probabilities_to_rank_logits,
    probability_to_raw_logits,
)
from src.validation.metric_calculator import compute_roc_auc_score


def optimize_ensemble_blends(
    artifacts: tuple[ModelPredictionArtifact, ...],
    y_true: np.ndarray,
) -> EnsembleBlendResult:
    """Compute both Conservative (raw logit) and Aggressive (Rank-Logit) optimal blends.

    Args:
        artifacts: Tuple of ModelPredictionArtifact instances from cross-validation.
        y_true: 1D binary ground-truth target array.

    Returns:
        EnsembleBlendResult containing weights, OOF AUC scores, and both test blends.

    Raises:
        ValueError: If artifacts tuple is empty.
    """
    if not artifacts:
        raise ValueError("Cannot optimize ensemble with zero model artifacts.")

    model_names = tuple(a.model_name for a in artifacts)
    oof_raw_logits = np.column_stack(
        [probability_to_raw_logits(a.oof_probabilities) for a in artifacts]
    )
    test_raw_logits = np.column_stack(
        [probability_to_raw_logits(a.test_probabilities) for a in artifacts]
    )

    oof_rank_logits = np.column_stack(
        [probabilities_to_rank_logits(a.oof_probabilities) for a in artifacts]
    )
    test_rank_logits = np.column_stack(
        [probabilities_to_rank_logits(a.test_probabilities) for a in artifacts]
    )

    cons_weights = _optimize_simplex_weights(oof_raw_logits, y_true, l2_penalty=1e-4)
    rank_weights = _optimize_simplex_weights(oof_rank_logits, y_true, l2_penalty=2e-5)

    cons_oof_preds = logits_to_probabilities(oof_raw_logits @ cons_weights)
    cons_test_preds = logits_to_probabilities(test_raw_logits @ cons_weights)

    rank_oof_preds = logits_to_probabilities(oof_rank_logits @ rank_weights)
    rank_test_preds = logits_to_probabilities(test_rank_logits @ rank_weights)

    return EnsembleBlendResult(
        model_names=model_names,
        optimal_weights=tuple(float(w) for w in rank_weights),
        conservative_oof_auc=compute_roc_auc_score(y_true, cons_oof_preds),
        rank_logit_oof_auc=compute_roc_auc_score(y_true, rank_oof_preds),
        conservative_test_predictions=cons_test_preds,
        rank_logit_test_predictions=rank_test_preds,
    )


def _optimize_simplex_weights(
    oof_matrix: np.ndarray,
    y_true: np.ndarray,
    l2_penalty: float,
) -> np.ndarray:
    """Find non-negative normalized weights maximizing OOF ROC-AUC via Nelder-Mead."""
    n_models = oof_matrix.shape[1]
    if n_models == 1:
        return np.array([1.0], dtype=np.float64)

    individual_aucs = np.array(
        [compute_roc_auc_score(y_true, oof_matrix[:, i]) for i in range(n_models)]
    )
    softmax_init = np.exp((individual_aucs - np.max(individual_aucs)) * 200.0)
    init_weights = softmax_init / np.sum(softmax_init)

    def objective(raw_w: np.ndarray) -> float:
        clipped_w = np.maximum(raw_w, 0.0)
        weight_sum = float(np.sum(clipped_w))
        if weight_sum <= 1e-12:
            return 0.0
        norm_w = clipped_w / weight_sum
        blended = oof_matrix @ norm_w
        auc = compute_roc_auc_score(y_true, blended)
        reg = l2_penalty * float(np.sum((norm_w - 1.0 / n_models) ** 2))
        return -auc + reg

    result = minimize(
        objective,
        init_weights,
        method="Nelder-Mead",
        options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-6},
    )
    best_w = np.maximum(result.x, 0.0)
    total = float(np.sum(best_w))
    return best_w / total if total > 1e-12 else init_weights
