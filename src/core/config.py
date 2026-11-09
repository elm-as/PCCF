"""Immutable configuration dataclasses for the S6E11 Credit Card Fraud pipeline.

Why shallow hyperparameters are enforced here:
In Kaggle Playground Series S6E11, the underlying synthetic generation process
behaves as a noisy additive risk model. Deep trees (depth > 5) memorize synthetic
generation artifacts and lose ~0.0020 ROC-AUC on validation. Restricting trees to
4-15 leaves and depth 3-4 forces additive learning and maximizes generalization
on the 80% private leaderboard.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PathConfig:
    """Filesystem paths for raw data, out-of-fold artifacts, and submissions."""

    project_root: Path
    raw_data_dir: Path
    processed_data_dir: Path
    oof_artifact_dir: Path
    test_artifact_dir: Path
    submission_dir: Path

    @classmethod
    def from_root(cls, root_dir: Path) -> "PathConfig":
        """Build standard project paths from the repository root directory.

        Args:
            root_dir: Absolute path to the PCCF project root.

        Returns:
            Immutable PathConfig instance.
        """
        if not root_dir.exists():
            raise FileNotFoundError(f"Project root directory does not exist: {root_dir}")

        return cls(
            project_root=root_dir,
            raw_data_dir=root_dir / "data" / "raw",
            processed_data_dir=root_dir / "data" / "processed",
            oof_artifact_dir=root_dir / "artifacts" / "oof",
            test_artifact_dir=root_dir / "artifacts" / "test",
            submission_dir=root_dir / "submissions",
        )


@dataclass(frozen=True)
class ValidationConfig:
    """Cross-validation and target encoding leakage-prevention settings."""

    competition_slug: str = "playground-series-s6e11"
    id_column: str = "id"
    target_column: str = "is_fraud"
    n_splits: int = 5
    inner_encoding_splits: int = 5
    random_seed: int = 42
    bayesian_smoothing_weight: float = 20.0


@dataclass(frozen=True)
class LightGBMShallowConfig:
    """Hyperparameters for the shallow-leaf LightGBM residual booster."""

    n_estimators: int = 2500
    learning_rate: float = 0.035
    num_leaves: int = 8
    max_depth: int = 3
    min_child_samples: int = 80
    subsample: float = 0.80
    subsample_freq: int = 1
    colsample_bytree: float = 0.75
    reg_alpha: float = 2.5
    reg_lambda: float = 10.0
    early_stopping_rounds: int = 100
    random_seed: int = 42


@dataclass(frozen=True)
class CatBoostShallowConfig:
    """Hyperparameters for the shallow symmetric-tree CatBoost model."""

    iterations: int = 2200
    learning_rate: float = 0.045
    depth: int = 4
    l2_leaf_reg: float = 12.0
    random_strength: float = 1.5
    subsample: float = 0.80
    early_stopping_rounds: int = 100
    random_seed: int = 42


@dataclass(frozen=True)
class XGBoostShallowConfig:
    """Hyperparameters for the depth-constrained XGBoost hist booster."""

    n_estimators: int = 2500
    learning_rate: float = 0.035
    max_depth: int = 3
    min_child_weight: float = 60.0
    subsample: float = 0.80
    colsample_bytree: float = 0.70
    reg_alpha: float = 2.0
    reg_lambda: float = 10.0
    early_stopping_rounds: int = 100
    random_seed: int = 42


@dataclass(frozen=True)
class LinearAdditiveConfig:
    """Hyperparameters for the GLM / Spline-Logistic additive base model."""

    inverse_regularization_c: float = 0.15
    max_iter: int = 500
    n_knots: int = 6
    spline_degree: int = 2
    random_seed: int = 42


@dataclass(frozen=True)
class PipelineConfig:
    """Master immutable configuration aggregating all pipeline domain settings."""

    paths: PathConfig
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    lgbm: LightGBMShallowConfig = field(default_factory=LightGBMShallowConfig)
    catboost: CatBoostShallowConfig = field(default_factory=CatBoostShallowConfig)
    xgboost: XGBoostShallowConfig = field(default_factory=XGBoostShallowConfig)
    linear: LinearAdditiveConfig = field(default_factory=LinearAdditiveConfig)

    @classmethod
    def create_default(cls, root_dir: Path) -> "PipelineConfig":
        """Create the default production configuration for S6E11.

        Args:
            root_dir: Root directory of the PCCF project.

        Returns:
            Fully initialized PipelineConfig.
        """
        return cls(paths=PathConfig.from_root(root_dir))

