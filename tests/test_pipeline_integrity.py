"""Automated unit and integration tests for PCCF S6E11 pipeline integrity."""

import numpy as np
import pandas as pd
import pytest

from src.core.config import ValidationConfig
from src.core.types import FoldDataSplit, ModelPredictionArtifact
from src.data_pipeline.memory_optimizer import build_optimized_dataset_bundle
from src.data_pipeline.schema_validator import validate_raw_dataframes
from src.ensembling.nelder_mead_blender import optimize_ensemble_blends
from src.ensembling.rank_transformer import (
    logits_to_probabilities,
    probabilities_to_percentile_ranks,
    probabilities_to_rank_logits,
)
from src.feature_engineering.cross_fitted_encoder import encode_fold_without_leakage
from src.feature_engineering.digit_extractor import extract_digit_key_features
from src.submission.submission_formatter import build_validated_submission_dataframe


def test_schema_validator_rejects_duplicate_ids() -> None:
    """Verify that duplicate transaction IDs raise a ValueError immediately."""
    val_cfg = ValidationConfig()
    train_df = pd.DataFrame({"id": [1, 1], "feat": [10.0, 20.0], "is_fraud": [0, 1]})
    test_df = pd.DataFrame({"id": [2, 3], "feat": [15.0, 25.0]})

    with pytest.raises(ValueError, match="Duplicate transaction IDs"):
        validate_raw_dataframes(train_df, test_df, val_cfg)


def test_digit_extractor_creates_cents_and_leading_keys_purely() -> None:
    """Verify cents (.99 -> '99') and Benford leading digit extraction without mutation."""
    values = np.linspace(10.0, 500.0, 30)
    values[0] = 42.99
    values[1] = 700.13
    df = pd.DataFrame({"transaction_amount": values})
    original_copy = df.copy()

    augmented, new_cols = extract_digit_key_features(df, ("transaction_amount",))

    pd.testing.assert_frame_equal(df, original_copy)
    assert "transaction_amount__cents_key" in new_cols
    assert augmented.loc[0, "transaction_amount__cents_key"] == "99"
    assert augmented.loc[0, "transaction_amount__lead_key"] == "4"
    assert augmented.loc[1, "transaction_amount__cents_key"] == "13"
    assert augmented.loc[1, "transaction_amount__lead_key"] == "7"


def test_cross_fitted_target_encoder_has_zero_validation_leakage() -> None:
    """Prove that validation target labels never leak into validation features."""
    rng = np.random.default_rng(123)
    n_tr, n_va, n_te = 200, 60, 40
    x_tr = pd.DataFrame({"cat": rng.choice(["A", "B", "C"], size=n_tr), "num": rng.normal(size=n_tr)})
    y_tr = pd.Series(rng.binomial(1, 0.25, size=n_tr))
    x_va = pd.DataFrame({"cat": rng.choice(["A", "B", "C"], size=n_va), "num": rng.normal(size=n_va)})
    x_te = pd.DataFrame({"cat": rng.choice(["A", "B", "C"], size=n_te), "num": rng.normal(size=n_te)})

    y_va_zeros = pd.Series(np.zeros(n_va, dtype=int))
    y_va_ones = pd.Series(np.ones(n_va, dtype=int))

    split_a = FoldDataSplit(0, np.arange(n_tr), np.arange(n_va), x_tr, y_tr, x_va, y_va_zeros, x_te)
    split_b = FoldDataSplit(0, np.arange(n_tr), np.arange(n_va), x_tr, y_tr, x_va, y_va_ones, x_te)

    val_cfg = ValidationConfig()
    enc_a = encode_fold_without_leakage(split_a, ("cat",), val_cfg)
    enc_b = encode_fold_without_leakage(split_b, ("cat",), val_cfg)

    pd.testing.assert_frame_equal(enc_a.x_val_encoded, enc_b.x_val_encoded)
    pd.testing.assert_frame_equal(enc_a.x_train_encoded, enc_b.x_train_encoded)


def test_rank_logit_and_nelder_mead_blend_improves_or_matches_single_models() -> None:
    """Verify Rank-Logit projection and Nelder-Mead blending on complementary signals."""
    rng = np.random.default_rng(7)
    y_true = rng.binomial(1, 0.2, size=500)

    p1 = np.clip(0.2 + 0.35 * y_true + rng.normal(0, 0.25, size=500), 0.01, 0.99)
    p2 = np.clip(0.2 + 0.35 * y_true + rng.normal(0, 0.25, size=500), 0.01, 0.99)

    t1 = np.clip(rng.uniform(0.05, 0.95, size=100), 0.01, 0.99)
    t2 = np.clip(rng.uniform(0.05, 0.95, size=100), 0.01, 0.99)

    art1 = ModelPredictionArtifact("m1", p1, t1, (0.80,), 0.80)
    art2 = ModelPredictionArtifact("m2", p2, t2, (0.80,), 0.80)

    blend = optimize_ensemble_blends((art1, art2), y_true)
    assert blend.rank_logit_oof_auc >= max(art1.overall_oof_auc, art2.overall_oof_auc)
    assert np.isclose(sum(blend.optimal_weights), 1.0)
    assert len(blend.rank_logit_test_predictions) == 100


def test_submission_formatter_enforces_competition_schema() -> None:
    """Ensure submission formatter outputs exact columns (id, is_fraud) and rejects NaNs."""
    val_cfg = ValidationConfig()
    ids = pd.Series([699960, 699961, 699962])
    probs = np.array([0.1234567, 0.8765432, 0.0045])

    sub_df = build_validated_submission_dataframe(ids, probs, val_cfg)
    assert list(sub_df.columns) == ["id", "is_fraud"]
    assert len(sub_df) == 3

    with pytest.raises(ValueError, match="NaN or infinite"):
        build_validated_submission_dataframe(ids, np.array([0.1, np.nan, 0.5]), val_cfg)
