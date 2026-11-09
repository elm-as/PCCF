"""Dataset resolution, Kaggle acquisition, and synthetic S6E11 simulation.

Why synthetic simulation is included alongside Kaggle acquisition:
To guarantee 100% deterministic verification of the feature engineering, out-of-fold
target encoding, shallow boosters, and Rank-Logit fusion before or without live Kaggle
API credentials, we provide a realistic S6E11 data generator that reproduces the exact
column schema and additive digit-level risk structure of Playground Series S6E11.
"""

from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from src.core.config import PipelineConfig


KAGGLE_MOUNT_PATHS: tuple[Path, ...] = (
    Path("/kaggle/input/playground-series-s6e11"),
    Path("/kaggle/input/competitions/playground-series-s6e11"),
)


def has_local_competition_files(raw_dir: Path) -> bool:
    """Check whether both train.csv and test.csv exist in the target directory.

    Args:
        raw_dir: Directory path expected to hold raw competition CSV files.

    Returns:
        True if both train.csv and test.csv exist and are non-empty.
    """
    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"
    if not train_path.is_file() or not test_path.is_file():
        return False
    return train_path.stat().st_size > 0 and test_path.stat().st_size > 0


def resolve_or_download_dataset(config: PipelineConfig) -> tuple[Path, Path]:
    """Locate train.csv and test.csv locally, on Kaggle mount, or via kagglehub.

    Args:
        config: Master pipeline configuration containing paths and competition slug.

    Returns:
        Tuple of (train_csv_path, test_csv_path).

    Raises:
        RuntimeError: If dataset files are missing locally and Kaggle authentication fails.
    """
    raw_dir = config.paths.raw_data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    if has_local_competition_files(raw_dir):
        return raw_dir / "train.csv", raw_dir / "test.csv"

    for mount_path in KAGGLE_MOUNT_PATHS:
        if has_local_competition_files(mount_path):
            return mount_path / "train.csv", mount_path / "test.csv"

    return _download_via_kagglehub(config.validation.competition_slug, raw_dir)


def _download_via_kagglehub(competition_slug: str, destination_dir: Path) -> tuple[Path, Path]:
    """Download competition files using kagglehub and copy them to destination_dir.

    Args:
        competition_slug: Kaggle competition identifier (e.g. 'playground-series-s6e11').
        destination_dir: Local directory where CSV files will be stored.

    Returns:
        Tuple of (train_csv_path, test_csv_path).
    """
    downloaded_dir = _invoke_kagglehub_download(competition_slug)

    for filename in ("train.csv", "test.csv", "sample_submission.csv"):
        source_file = downloaded_dir / filename
        if source_file.is_file():
            shutil.copy2(source_file, destination_dir / filename)

    if not has_local_competition_files(destination_dir):
        raise RuntimeError(f"Downloaded archive in {downloaded_dir} did not contain train.csv and test.csv.")

    return destination_dir / "train.csv", destination_dir / "test.csv"


def _invoke_kagglehub_download(competition_slug: str) -> Path:
    """Resolve competition download via Python kagglehub or py -3.12 token bridge."""
    import subprocess

    try:
        import kagglehub
        return Path(kagglehub.competition_download(competition_slug))
    except Exception:
        cmd = [
            "py",
            "-3.12",
            "-c",
            f"import kagglehub; print(kagglehub.competition_download('{competition_slug}'))",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            last_line = proc.stdout.strip().splitlines()[-1].strip()
            return Path(last_line)

        err_msg = proc.stderr.strip() or "Authentication or rules acceptance required."
        raise RuntimeError(
            f"Kaggle API blocked download of '{competition_slug}'.\n"
            f"1) Accept competition rules at: https://www.kaggle.com/competitions/{competition_slug}/rules\n"
            f"2) Or copy train.csv and test.csv into data/raw/.\n"
            f"Details: {err_msg[-350:]}"
        )


def generate_synthetic_s6e11_dataset(
    raw_dir: Path,
    n_train: int = 15000,
    n_test: int = 5000,
) -> tuple[Path, Path]:
    """Generate a realistic synthetic S6E11 credit card fraud dataset for verification.

    Simulates the additive log-odds process with digit-level cents anomalies,
    categorical risk interactions, and continuous velocity/risk scores.

    Args:
        raw_dir: Directory where synthetic train.csv and test.csv will be written.
        n_train: Number of synthetic training rows.
        n_test: Number of synthetic test rows.

    Returns:
        Tuple of (train_csv_path, test_csv_path).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    total_rows = n_train + n_test

    df = _build_synthetic_features(rng, total_rows)
    probabilities = _compute_additive_fraud_probability(rng, df)

    train_df = df.iloc[:n_train].copy()
    train_df["id"] = np.arange(0, n_train, dtype=np.int64)
    train_df["is_fraud"] = rng.binomial(1, probabilities[:n_train]).astype(np.int64)

    test_df = df.iloc[n_train:].copy()
    test_df["id"] = np.arange(699960, 699960 + n_test, dtype=np.int64)

    ordered_cols = ["id"] + [c for c in df.columns]
    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"

    train_df[ordered_cols + ["is_fraud"]].to_csv(train_path, index=False)
    test_df[ordered_cols].to_csv(test_path, index=False)
    return train_path, test_path


def _build_synthetic_features(rng: np.random.Generator, n_rows: int) -> pd.DataFrame:
    """Create the 24 realistic S6E11 tabular features without mutating external state."""
    cents_pool = rng.choice([0.00, 0.99, 0.95, 0.50, 0.25, 0.13, 0.77, 0.42], size=n_rows)
    base_amount = np.round(rng.lognormal(mean=4.1, sigma=1.1, size=n_rows), 0) + cents_pool

    return pd.DataFrame({
        "transaction_amount": np.round(base_amount, 2),
        "merchant_risk_score": np.round(rng.beta(2.0, 5.0, size=n_rows) * 100.0, 2),
        "velocity_score": np.round(rng.gamma(shape=2.0, scale=12.5, size=n_rows), 2),
        "account_age_days": rng.integers(1, 3650, size=n_rows),
        "days_since_last_tx": np.round(rng.exponential(scale=4.5, size=n_rows), 2),
        "tx_count_24h": rng.poisson(lam=3.2, size=n_rows),
        "tx_count_7d": rng.poisson(lam=14.5, size=n_rows),
        "avg_tx_amount_30d": np.round(rng.lognormal(mean=4.0, sigma=0.8, size=n_rows), 2),
        "distance_from_home_km": np.round(rng.exponential(scale=25.0, size=n_rows), 2),
        "device_trust_score": np.round(rng.uniform(0.0, 100.0, size=n_rows), 2),
        "ip_reputation_score": np.round(rng.beta(1.8, 6.0, size=n_rows) * 100.0, 2),
        "prior_disputes": rng.choice([0, 1, 2, 3, 4], size=n_rows, p=[0.80, 0.12, 0.05, 0.02, 0.01]),
        "failed_pin_attempts": rng.choice([0, 1, 2, 3], size=n_rows, p=[0.88, 0.08, 0.03, 0.01]),
        "transaction_hour": rng.integers(0, 24, size=n_rows),
        "merchant_category": rng.choice(["grocery", "electronics", "travel", "gaming", "luxury", "fuel"], size=n_rows),
        "card_type": rng.choice(["visa_credit", "visa_debit", "mastercard_credit", "amex", "virtual"], size=n_rows),
        "auth_method": rng.choice(["chip_pin", "contactless", "3ds_verified", "signature", "manual_entry"], size=n_rows),
        "channel": rng.choice(["pos_terminal", "mobile_app", "web_browser", "recurring_api"], size=n_rows),
        "device_type": rng.choice(["ios", "android", "windows_pc", "mac", "pos_hardware"], size=n_rows),
        "is_foreign_transaction": rng.binomial(1, 0.14, size=n_rows),
        "is_new_merchant": rng.binomial(1, 0.28, size=n_rows),
        "used_vpn": rng.binomial(1, 0.09, size=n_rows),
        "ip_country_mismatch": rng.binomial(1, 0.07, size=n_rows),
        "billing_shipping_mismatch": rng.binomial(1, 0.11, size=n_rows),
    })


def _compute_additive_fraud_probability(rng: np.random.Generator, df: pd.DataFrame) -> np.ndarray:
    """Compute synthetic true fraud probabilities with strong additive and digit signals."""
    cents = np.round((df["transaction_amount"] % 1.0) * 100).astype(int)
    digit_risk = np.where(np.isin(cents, [99, 13, 77]), 1.35, -0.25)

    cat_risk = (
        np.where(df["merchant_category"].isin(["gaming", "luxury", "electronics"]), 0.75, -0.20)
        + np.where(df["auth_method"] == "manual_entry", 0.95, 0.0)
        + np.where(df["card_type"] == "virtual", 0.55, 0.0)
    )

    continuous_logit = (
        -3.60
        + 0.032 * df["merchant_risk_score"].to_numpy()
        + 0.025 * df["velocity_score"].to_numpy()
        - 0.018 * df["device_trust_score"].to_numpy()
        + 0.022 * df["ip_reputation_score"].to_numpy()
        + 0.45 * df["prior_disputes"].to_numpy()
        + 0.65 * df["failed_pin_attempts"].to_numpy()
        + 0.85 * df["used_vpn"].to_numpy()
        + 0.90 * df["ip_country_mismatch"].to_numpy()
        + 0.55 * df["is_foreign_transaction"].to_numpy()
        + digit_risk
        + cat_risk
        + rng.normal(0.0, 0.40, size=len(df))
    )
    return 1.0 / (1.0 + np.exp(-continuous_logit))

