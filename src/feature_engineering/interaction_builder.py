"""Domain-specific categorical interactions, risk flag aggregations, and financial ratios."""

import numpy as np
import pandas as pd


KNOWN_RISK_FLAG_COLUMNS: tuple[str, ...] = (
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
)

PREFERRED_CATEGORICAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("merchant_category", "card_type"),
    ("merchant_category", "channel"),
    ("merchant_category", "auth_method"),
    ("channel", "auth_method"),
    ("card_type", "channel"),
    ("device_type", "auth_method"),
)


def build_domain_interaction_features(
    df: pd.DataFrame,
    categorical_cols: tuple[str, ...],
    numerical_cols: tuple[str, ...],
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Create financial domain interactions without mutating the input DataFrame.

    Args:
        df: Input feature DataFrame.
        categorical_cols: Existing categorical column names.
        numerical_cols: Existing numerical column names.

    Returns:
        Tuple of (augmented_df, new_categorical_cols, new_numerical_cols).
    """
    cat_interactions = _build_categorical_pair_columns(df, categorical_cols)
    num_interactions = _build_numerical_domain_columns(df, numerical_cols)

    blocks = [df.copy()]
    if cat_interactions:
        blocks.append(pd.DataFrame(cat_interactions, index=df.index))
    if num_interactions:
        blocks.append(pd.DataFrame(num_interactions, index=df.index))

    combined = pd.concat(blocks, axis=1)
    return (
        combined,
        tuple(cat_interactions.keys()),
        tuple(num_interactions.keys()),
    )


def _build_categorical_pair_columns(
    df: pd.DataFrame,
    categorical_cols: tuple[str, ...],
) -> dict[str, pd.Series]:
    """Construct high-signal second-order categorical interaction keys."""
    created: dict[str, pd.Series] = {}
    available_set = set(df.columns)

    for col_a, col_b in PREFERRED_CATEGORICAL_PAIRS:
        if col_a in available_set and col_b in available_set:
            pair_name = f"{col_a}__{col_b}"
            created[pair_name] = (
                df[col_a].astype(str) + "_" + df[col_b].astype(str)
            ).astype("object")

    if not created and len(categorical_cols) >= 2:
        for idx in range(min(4, len(categorical_cols) - 1)):
            col_a = categorical_cols[idx]
            col_b = categorical_cols[idx + 1]
            pair_name = f"{col_a}__{col_b}"
            created[pair_name] = (
                df[col_a].astype(str) + "_" + df[col_b].astype(str)
            ).astype("object")

    return created


def _build_numerical_domain_columns(
    df: pd.DataFrame,
    numerical_cols: tuple[str, ...],
) -> dict[str, pd.Series]:
    """Compute additive flag counts, log transforms, and velocity/amount ratios."""
    created: dict[str, pd.Series] = {}
    cols = set(df.columns)

    present_flags = [c for c in KNOWN_RISK_FLAG_COLUMNS if c in cols]
    if not present_flags:
        present_flags = [
            c for c in numerical_cols
            if c in cols and set(df[c].dropna().unique()).issubset({0, 1})
        ]

    if present_flags:
        numeric_flags = df[present_flags].replace({"True": 1, "False": 0, True: 1, False: 0}).astype(np.int8)
        flag_sum = numeric_flags.sum(axis=1).astype(np.int8)
        created["total_risk_flags_count"] = flag_sum
        created["has_multiple_risk_flags"] = (flag_sum >= 2).astype(np.int8)

    amount_col = "amount_usd" if "amount_usd" in cols else ("transaction_amount" if "transaction_amount" in cols else None)
    if amount_col is not None:
        amt = np.maximum(df[amount_col].to_numpy(dtype=np.float32), 0.0)
        created[f"{amount_col}_log1p"] = pd.Series(
            np.log1p(amt).astype(np.float32), index=df.index
        )
        if "account_balance_usd" in cols:
            bal = np.maximum(df["account_balance_usd"].to_numpy(dtype=np.float32), 1.0)
            created["amount_to_balance_ratio"] = pd.Series(
                (amt / bal).astype(np.float32), index=df.index
            )

    if "merchant_risk_score" in cols and "velocity_score" in cols:
        m_risk = df["merchant_risk_score"].to_numpy(dtype=np.float32)
        vel = df["velocity_score"].to_numpy(dtype=np.float32)
        created["combined_merchant_velocity_risk"] = pd.Series(
            ((m_risk + vel) * 0.5).astype(np.float32), index=df.index
        )

    return created

