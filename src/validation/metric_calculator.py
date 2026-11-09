"""Pure ROC-AUC evaluation with strict guard clauses against NaN and degenerate targets."""

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_roc_auc_score(y_true: np.ndarray, y_probabilities: np.ndarray) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC).

    Args:
        y_true: 1D binary array of ground-truth fraud labels (0 or 1).
        y_probabilities: 1D float array of predicted fraud probabilities or scores.

    Returns:
        ROC-AUC score in [0.0, 1.0].

    Raises:
        ValueError: If shapes mismatch, inputs contain NaN/Inf, or y_true is single-class.
    """
    if len(y_true) == 0 or len(y_true) != len(y_probabilities):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) vs y_probabilities ({len(y_probabilities)})."
        )

    if not np.all(np.isfinite(y_probabilities)):
        raise ValueError("Predicted probabilities contain NaN or infinite values.")

    if len(np.unique(y_true)) < 2:
        raise ValueError("Cannot compute ROC-AUC when y_true contains fewer than 2 classes.")

    return float(roc_auc_score(y_true, y_probabilities))

