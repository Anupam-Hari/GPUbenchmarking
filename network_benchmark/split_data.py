from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from time import perf_counter

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE


@dataclass(frozen=True)
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: list[str]
    prepare_time: float


def _get_logger(logger: logging.Logger | None) -> logging.Logger:
    return logger if logger is not None else logging.getLogger(__name__)


def _can_stratify(target: pd.Series) -> bool:
    counts = target.value_counts(dropna=False)
    return len(counts) > 1 and counts.min() >= 2


def load_processed_data(processed_csv_path: Path | str) -> pd.DataFrame:
    return pd.read_csv(Path(processed_csv_path), low_memory=False)


def _sample_dataframe(df: pd.DataFrame, sample_size: Optional[int], random_state: int, logger: logging.Logger) -> pd.DataFrame:
    if sample_size is None:
        return df

    if sample_size >= len(df):
        logger.info("Requested sample_size=%s exceeds available rows. Using full dataset.", sample_size)
        return df

    logger.info("Sampling %d rows before splitting.", sample_size)
    return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    sample_size: Optional[int] = None,
    random_state: int = RANDOM_STATE,
    logger: logging.Logger | None = None,
) -> DataSplit:
    logger = _get_logger(logger)

    prepare_start = perf_counter()
    df = _sample_dataframe(df, sample_size, random_state, logger)

    target_column = "attack_type"
    if target_column not in df.columns:
        raise KeyError(f"Expected target column '{target_column}' not found in processed data.")

    feature_columns = [
        column
        for column in df.columns
        if column not in {"is_malicious", target_column}
    ]

    X = df[feature_columns]
    y = df[target_column]

    stratify_target = y if _can_stratify(y) else None
    if stratify_target is None:
        logger.warning(
            "Stratified split disabled because the sampled target distribution is too small."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify_target,
    )

    prepare_time = perf_counter() - prepare_start

    return DataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_columns=feature_columns,
        prepare_time=prepare_time,
    )