"""Loader and validator for external out-of-fold (OOF) and test prediction banks.

Why external artifact banks are integrated:
In S6E11, GPU-trained tabular neural architectures (such as RealMLP and Target-First
Language Models in `s6e11-fraud-series-artifacts`) share the identical 5-fold
StratifiedKFold(n_splits=5, shuffle=True, random_state=42) split. Loading and validating
their OOF and test probability pairs allows our Nelder-Mead Rank-Logit optimizer to fuse
neural representations with our shallow digit-encoded GBDTs and Spline-GLM without
retraining multi-hour GPU models on CPU.
"""

from pathlib import Path
import numpy as np

from src.core.types import ModelPredictionArtifact
from src.validation.metric_calculator import compute_roc_auc_score


DEFAULT_KAGGLEHUB_ARTIFACT_CACHE: Path = (
    Path.home()
    / ".cache"
    / "kagglehub"
    / "datasets"
    / "goodpjw2008"
    / "s6e11-fraud-series-artifacts"
)


def load_validated_external_artifacts(
    search_dirs: tuple[Path, ...],
    y_true: np.ndarray,
    expected_test_len: int,
) -> tuple[ModelPredictionArtifact, ...]:
    """Discover and validate external (*_oof.npy, *_test.npy) prediction pairs.

    Args:
        search_dirs: Directories to scan recursively for *_oof.npy files.
        y_true: Ground-truth binary target array of length n_train.
        expected_test_len: Expected number of rows in test predictions.

    Returns:
        Tuple of validated ModelPredictionArtifact objects sorted by OOF ROC-AUC.
    """
    oof_files = _discover_oof_files(search_dirs)
    if not oof_files:
        return ()

    loaded: list[ModelPredictionArtifact] = []
    for oof_path in oof_files:
        artifact = _try_load_artifact_pair(oof_path, y_true, expected_test_len)
        if artifact is not None:
            loaded.append(artifact)

    loaded.sort(key=lambda item: item.overall_oof_auc, reverse=True)
    return tuple(loaded)


def _discover_oof_files(search_dirs: tuple[Path, ...]) -> list[Path]:
    """Find all *_oof.npy files across the provided search directories."""
    found: list[Path] = []
    seen_stems: set[str] = set()

    for directory in search_dirs:
        if not directory.exists():
            continue
        for oof_path in sorted(directory.rglob("*_oof.npy")):
            stem = oof_path.name[:-8]
            if stem not in seen_stems:
                seen_stems.add(stem)
                found.append(oof_path)
    return found


def _try_load_artifact_pair(
    oof_path: Path,
    y_true: np.ndarray,
    expected_test_len: int,
) -> ModelPredictionArtifact | None:
    """Load and verify a single (*_oof.npy, *_test.npy) pair, returning None if incompatible."""
    stem = oof_path.name[:-8]
    test_path = oof_path.with_name(f"{stem}_test.npy")
    if not test_path.is_file():
        return None

    try:
        oof_arr = np.load(oof_path).astype(np.float64).ravel()
        test_arr = np.load(test_path).astype(np.float64).ravel()
    except Exception:
        return None

    if len(oof_arr) != len(y_true) or len(test_arr) != expected_test_len:
        return None

    if not (np.all(np.isfinite(oof_arr)) and np.all(np.isfinite(test_arr))):
        return None

    oof_clipped = np.clip(oof_arr, 0.0, 1.0)
    test_clipped = np.clip(test_arr, 0.0, 1.0)
    auc_score = compute_roc_auc_score(y_true, oof_clipped)

    return ModelPredictionArtifact(
        model_name=f"ext_{stem}",
        oof_probabilities=oof_clipped,
        test_probabilities=test_clipped,
        fold_auc_scores=(auc_score,),
        overall_oof_auc=auc_score,
    )

