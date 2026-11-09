"""Exporter and statistical auditor for the dual Kaggle submission strategy."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from src.core.config import PipelineConfig
from src.core.types import EnsembleBlendResult
from src.submission.submission_formatter import build_validated_submission_dataframe


@dataclass(frozen=True)
class SubmissionAuditReport:
    """Audit metrics and paths for exported submission CSV files."""

    conservative_csv_path: Path
    rank_logit_csv_path: Path
    row_count: int
    conservative_quantiles: dict[str, float]
    rank_logit_quantiles: dict[str, float]


def export_dual_submissions(
    test_ids: pd.Series,
    blend_result: EnsembleBlendResult,
    config: PipelineConfig,
) -> SubmissionAuditReport:
    """Validate, write, and audit both conservative and aggressive submission CSVs.

    Why dual submissions are generated:
    Kaggle allows selecting 2 final submissions.
    1) submission_cv_conservative.csv preserves calibrated log-odds across folds to
       maximize stability on the 80% Private Leaderboard.
    2) submission_leaderboard_blend.csv uses Nelder-Mead Rank-Logit fusion to maximize
       rank separation on the Public & Private leaderboards.

    Args:
        test_ids: Test set transaction IDs.
        blend_result: Output of the Nelder-Mead ensembling stage.
        config: Master PipelineConfig.

    Returns:
        SubmissionAuditReport summarizing exported files and distribution quantiles.
    """
    sub_dir = config.paths.submission_dir
    sub_dir.mkdir(parents=True, exist_ok=True)

    cons_df = build_validated_submission_dataframe(
        test_ids, blend_result.conservative_test_predictions, config.validation
    )
    rank_df = build_validated_submission_dataframe(
        test_ids, blend_result.rank_logit_test_predictions, config.validation
    )

    cons_path = sub_dir / "submission_cv_conservative.csv"
    rank_path = sub_dir / "submission_leaderboard_blend.csv"

    cons_df.to_csv(cons_path, index=False)
    rank_df.to_csv(rank_path, index=False)

    return SubmissionAuditReport(
        conservative_csv_path=cons_path,
        rank_logit_csv_path=rank_path,
        row_count=len(cons_df),
        conservative_quantiles=_compute_quantiles(blend_result.conservative_test_predictions),
        rank_logit_quantiles=_compute_quantiles(blend_result.rank_logit_test_predictions),
    )


def _compute_quantiles(predictions: np.ndarray) -> dict[str, float]:
    """Compute tabular summary quantiles for probability distribution verification."""
    q_vals = np.quantile(predictions, [0.0, 0.25, 0.50, 0.75, 0.95, 1.0])
    return {
        "min": float(q_vals[0]),
        "p25": float(q_vals[1]),
        "median": float(q_vals[2]),
        "p75": float(q_vals[3]),
        "p95": float(q_vals[4]),
        "max": float(q_vals[5]),
    }
