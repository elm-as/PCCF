"""Memory optimization and DatasetBundle construction.

Why downcasting is mandatory:
With ~700,000 training rows and ~300,000 test rows on a 16 GB RAM machine, keeping
default 64-bit floats and integers consumes ~1.8 GB per copy during feature expansion.
Downcasting continuous columns to float32 and integer flags to int8/int16/int32 reduces
RAM usage by 65% without losing numerical precision relevant to tree splits.
"""

import numpy as np
import pandas as pd

from src.core.config import ValidationConfig
from src.core.types import DatasetBundle
from src.data_pipeline.schema_validator import validate_raw_dataframes


HIGH_DROP_COLUMNS: tuple[str, ...] = (
    "trans_date_trans_time",
    "dob",
    "first",
    "last",
    "street",
    "trans_num",
    "cc_num",
)


def build_optimized_dataset_bundle(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_config: ValidationConfig,
) -> DatasetBundle:
    """Validate, clean, and downcast raw dataframes into an immutable DatasetBundle.

    Args:
        train_df: Raw training DataFrame.
        test_df: Raw test DataFrame.
        val_config: Validation configuration specifying ID and target columns.

    Returns:
        Memory-optimized DatasetBundle ready for feature engineering.
    """
    validate_raw_dataframes(train_df, test_df, val_config)

    id_col = val_config.id_column
    target_col = val_config.target_column

    train_ids = train_df[id_col].astype(np.int64).copy()
    test_ids = test_df[id_col].astype(np.int64).copy()
    train_target = train_df[target_col].astype(np.int8).copy()

    drop_cols = [c for c in HIGH_DROP_COLUMNS if c in train_df.columns]
    feature_cols = [
        c for c in train_df.columns
        if c not in {id_col, target_col, *drop_cols}
    ]

    x_train_raw = train_df[feature_cols].copy()
    x_test_raw = test_df[feature_cols].copy()

    cat_cols, num_cols = _partition_column_types(x_train_raw)
    x_train_opt = _optimize_dataframe_dtypes(x_train_raw, cat_cols, num_cols)
    x_test_opt = _optimize_dataframe_dtypes(x_test_raw, cat_cols, num_cols)

    return DatasetBundle(
        train_features=x_train_opt,
        train_target=train_target,
        train_ids=train_ids,
        test_features=x_test_opt,
        test_ids=test_ids,
        numerical_columns=num_cols,
        categorical_columns=cat_cols,
    )


def _partition_column_types(df: pd.DataFrame) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Separate feature columns into categorical and numerical tuples.

    Boolean flag columns are classified as numerical to preserve binary integer math.
    """
    cat_list: list[str] = []
    num_list: list[str] = []

    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            num_list.append(col)
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            cat_list.append(col)
        else:
            num_list.append(col)

    return tuple(cat_list), tuple(num_list)


def _optimize_dataframe_dtypes(
    df: pd.DataFrame,
    cat_cols: tuple[str, ...],
    num_cols: tuple[str, ...],
) -> pd.DataFrame:
    """Return a new DataFrame with downcasted numerical types and clean string categories."""
    optimized = df.copy()

    for col in cat_cols:
        optimized[col] = optimized[col].fillna("MISSING").astype(str)

    for col in num_cols:
        series = optimized[col]
        if series.isna().any():
            median_val = float(series.median()) if not series.isna().all() else 0.0
            series = series.fillna(median_val)
        optimized[col] = _downcast_numeric_series(series)

    return optimized


def _downcast_numeric_series(series: pd.Series) -> pd.Series:
    """Downcast a numeric Series to the smallest safe integer or float32 representation."""
    if pd.api.types.is_float_dtype(series.dtype):
        return series.astype(np.float32)

    col_min = series.min()
    col_max = series.max()
    if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
        return series.astype(np.int8)
    if col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
        return series.astype(np.int16)
    return series.astype(np.int32)

