"""Percentile rank normalization and Rank-Logit projection for ROC-AUC ensembling.

Why Rank-Logit fusion dominates standard probability averaging on ROC-AUC:
ROC-AUC is strictly a rank-order metric, yet raw probabilities from Spline-GLMs and
shallow GBDTs exhibit different tail compression. Converting predictions to empirical
percentile ranks R in (0, 1) removes calibration mismatch, and projecting R through
the inverse logistic function ln(R / (1 - R)) restores additive tail sensitivity when
blending strong extreme signals.
"""

import numpy as np
from scipy.stats import rankdata


CLIP_EPSILON: float = 1e-6


def probabilities_to_percentile_ranks(probabilities: np.ndarray) -> np.ndarray:
    """Transform a 1D array of probabilities into uniform percentile ranks in [eps, 1-eps].

    Args:
        probabilities: 1D numpy array of model predictions.

    Returns:
        1D float64 array of empirical percentile ranks clipped to [1e-6, 1 - 1e-6].

    Raises:
        ValueError: If input array is empty or contains non-finite values.
    """
    if probabilities.ndim != 1 or len(probabilities) == 0:
        raise ValueError("Input probabilities must be a non-empty 1D array.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Input probabilities contain NaN or infinite values.")

    n_samples = len(probabilities)
    raw_ranks = rankdata(probabilities, method="average")
    uniform_ranks = (raw_ranks - 0.5) / float(n_samples)
    return np.clip(uniform_ranks, CLIP_EPSILON, 1.0 - CLIP_EPSILON)


def probabilities_to_rank_logits(probabilities: np.ndarray) -> np.ndarray:
    """Map model probabilities to empirical percentile ranks and then to log-odds space.

    Args:
        probabilities: 1D numpy array of model predictions.

    Returns:
        1D float64 array of rank-logits z = ln(R / (1 - R)).
    """
    ranks = probabilities_to_percentile_ranks(probabilities)
    return np.log(ranks / (1.0 - ranks))


def probability_to_raw_logits(probabilities: np.ndarray) -> np.ndarray:
    """Convert calibrated probabilities directly into log-odds space with safe clipping.

    Args:
        probabilities: 1D numpy array of probabilities in [0, 1].

    Returns:
        1D float64 array of raw log-odds.
    """
    clipped = np.clip(probabilities, CLIP_EPSILON, 1.0 - CLIP_EPSILON)
    return np.log(clipped / (1.0 - clipped))


def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    """Apply the standard logistic sigmoid 1 / (1 + exp(-z)) safely without overflow.

    Args:
        logits: 1D numpy array of real-valued log-odds.

    Returns:
        1D float64 array of probabilities in (0, 1).
    """
    safe_logits = np.clip(logits, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-safe_logits))
