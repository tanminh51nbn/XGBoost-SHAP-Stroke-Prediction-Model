from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from stroke_ai.utils.io import save_dataframe


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series


def split_dataset(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> DataSplits:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1)")
    if not (0.0 < val_size < 1.0):
        raise ValueError("val_size must be in (0, 1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=adjusted_val_size,
        stratify=y_train_valid,
        random_state=random_state,
    )

    return DataSplits(
        X_train=X_train.reset_index(drop=True),
        X_valid=X_valid.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_valid=y_valid.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


def save_splits(splits: DataSplits, output_dir: Path, target_col: str) -> None:
    train_df = splits.X_train.copy()
    train_df[target_col] = splits.y_train

    valid_df = splits.X_valid.copy()
    valid_df[target_col] = splits.y_valid

    test_df = splits.X_test.copy()
    test_df[target_col] = splits.y_test

    save_dataframe(train_df, output_dir / "train.csv")
    save_dataframe(valid_df, output_dir / "valid.csv")
    save_dataframe(test_df, output_dir / "test.csv")
