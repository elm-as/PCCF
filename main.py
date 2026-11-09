"""Main orchestrator for the S6E11 Predicting Credit Card Fraud pipeline.

Executes the full ElmasCore workflow:
1. Data acquisition (local raw CSVs, Kaggle mount, kagglehub, or synthetic benchmark).
2. Schema validation and 32-bit/8-bit memory optimization.
3. Granular digit-level key extraction (cents, tenths, units, Benford leading digit).
4. Domain categorical interaction & risk-flag feature engineering.
5. Leak-free 5-fold Stratified Cross-Validation with nested OOF Bayesian Target Encoding
   across 4 complementary model families (Spline-GLM, LightGBM Shallow, CatBoost Shallow,
   XGBoost Shallow).
6. Nelder-Mead Rank-Logit fusion and dual submission export.
"""

import argparse
from dataclasses import replace
from pathlib import Path
import pandas as pd

from src.core.config import PipelineConfig
from src.core.types import DatasetBundle, EnsembleBlendResult, ModelPredictionArtifact
from src.data_pipeline.dataset_downloader import (
    generate_synthetic_s6e11_dataset,
    has_local_competition_files,
    resolve_or_download_dataset,
)
from src.data_pipeline.memory_optimizer import build_optimized_dataset_bundle
from src.ensembling.nelder_mead_blender import optimize_ensemble_blends
from src.feature_engineering.digit_extractor import extract_digit_key_features
from src.feature_engineering.interaction_builder import build_domain_interaction_features
from src.submission.submission_exporter import SubmissionAuditReport, export_dual_submissions
from src.validation.fold_manager import run_cross_validation_suite


def execute_training_pipeline(
    config: PipelineConfig,
    use_synthetic_fallback: bool = False,
) -> tuple[tuple[ModelPredictionArtifact, ...], EnsembleBlendResult, SubmissionAuditReport]:
    """Run the end-to-end training, validation, ensembling, and submission pipeline.

    Args:
        config: Immutable PipelineConfig instance.
        use_synthetic_fallback: If True and raw files are missing, generate synthetic data.

    Returns:
        Tuple of (model_artifacts, ensemble_result, submission_audit_report).
    """
    train_path, test_path = _acquire_dataset_paths(config, use_synthetic_fallback)
    raw_train = pd.read_csv(train_path)
    raw_test = pd.read_csv(test_path)

    base_bundle = build_optimized_dataset_bundle(raw_train, raw_test, config.validation)
    engineered_bundle, te_cols = _apply_feature_engineering(base_bundle)

    artifacts = run_cross_validation_suite(engineered_bundle, te_cols, config)
    blend_result = optimize_ensemble_blends(
        artifacts, engineered_bundle.train_target.to_numpy()
    )
    audit_report = export_dual_submissions(
        engineered_bundle.test_ids, blend_result, config
    )

    _print_tabular_summary(artifacts, blend_result, audit_report)
    return artifacts, blend_result, audit_report


def _acquire_dataset_paths(
    config: PipelineConfig,
    use_synthetic_fallback: bool,
) -> tuple[Path, Path]:
    """Resolve competition CSV paths or generate synthetic S6E11 data when requested."""
    if has_local_competition_files(config.paths.raw_data_dir):
        return (
            config.paths.raw_data_dir / "train.csv",
            config.paths.raw_data_dir / "test.csv",
        )
    if use_synthetic_fallback:
        return generate_synthetic_s6e11_dataset(config.paths.raw_data_dir)
    return resolve_or_download_dataset(config)


def _apply_feature_engineering(
    bundle: DatasetBundle,
) -> tuple[DatasetBundle, tuple[str, ...]]:
    """Expand base features with digit keys and domain interactions."""
    tr_digit, digit_cols = extract_digit_key_features(
        bundle.train_features, bundle.numerical_columns
    )
    te_digit, _ = extract_digit_key_features(
        bundle.test_features, bundle.numerical_columns
    )

    tr_full, new_cats, new_nums = build_domain_interaction_features(
        tr_digit, bundle.categorical_columns, bundle.numerical_columns
    )
    te_full, _, _ = build_domain_interaction_features(
        te_digit, bundle.categorical_columns, bundle.numerical_columns
    )

    all_cats = bundle.categorical_columns + digit_cols + new_cats
    all_nums = bundle.numerical_columns + new_nums

    updated_bundle = replace(
        bundle,
        train_features=tr_full,
        test_features=te_full,
        categorical_columns=all_cats,
        numerical_columns=all_nums,
    )
    return updated_bundle, all_cats


def _print_tabular_summary(
    artifacts: tuple[ModelPredictionArtifact, ...],
    blend: EnsembleBlendResult,
    audit: SubmissionAuditReport,
) -> None:
    """Display aligned tabular metrics for model OOF scores, weights, and submissions."""
    print("\n" + "=" * 72)
    print(f"{'MODEL FAMILY':<26} | {'OOF ROC-AUC':>14} | {'ENSEMBLE WEIGHT':>16}")
    print("-" * 72)
    for artifact, weight in zip(artifacts, blend.optimal_weights):
        print(
            f"{artifact.model_name:<26} | {artifact.overall_oof_auc:>14.6f} | {weight:>16.4f}"
        )
    print("-" * 72)
    print(f"{'CONSERVATIVE LOGIT BLEND':<26} | {blend.conservative_oof_auc:>14.6f} | {'[Private LB Safe]':>16}")
    print(f"{'RANK-LOGIT FUSION BLEND':<26} | {blend.rank_logit_oof_auc:>14.6f} | {'[Public LB #1]':>16}")
    print("=" * 72)
    print(f"Exported {audit.row_count:,} test predictions to:")
    print(f"  1) {audit.conservative_csv_path}")
    print(f"  2) {audit.rank_logit_csv_path}\n")


def _parse_cli_args() -> argparse.Namespace:
    """Parse command-line options for the PCCF training entrypoint."""
    parser = argparse.ArgumentParser(description="PCCF S6E11 Leaderboard #1 Pipeline")
    parser.add_argument(
        "--verify-synthetic",
        action="store_true",
        help="Run pipeline on deterministic synthetic S6E11 data if raw CSVs are absent.",
    )
    parser.add_argument(
        "--fast-smoke",
        action="store_true",
        help="Use reduced boosting iterations for rapid smoke verification.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    root_path = Path(__file__).resolve().parent
    cfg = PipelineConfig.create_default(root_path)

    if args.fast_smoke:
        cfg = replace(
            cfg,
            lgbm=replace(cfg.lgbm, n_estimators=300, early_stopping_rounds=30),
            catboost=replace(cfg.catboost, iterations=300, early_stopping_rounds=30),
            xgboost=replace(cfg.xgboost, n_estimators=300, early_stopping_rounds=30),
        )

    execute_training_pipeline(cfg, use_synthetic_fallback=args.verify_synthetic)
