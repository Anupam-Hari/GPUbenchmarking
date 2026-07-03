from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional, Sequence

import pandas as pd
from sklearn.metrics import accuracy_score

from config import BACKENDS, RANDOM_STATE
from models import get_model
from split_data import split_data


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    sample_size: Optional[int]
    max_depth: Optional[int]
    n_estimators: int
    repeat: int
    train_time: float
    predict_time: float
    total_time: float
    accuracy: float


def _get_logger(logger: logging.Logger | None) -> logging.Logger:
    return logger if logger is not None else logging.getLogger(__name__)


def _normalize_backends(backend: str) -> list[str]:
    if backend == "both":
        return list(BACKENDS)
    return [backend]


def run_benchmark(
    backend: str,
    processed_csv_path: Path | str,
    sample_sizes: Sequence[Optional[int]],
    max_depths: Sequence[Optional[int]],
    n_estimators_list: Sequence[int],
    n_repeats: int,
    random_state: int = RANDOM_STATE,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    logger = _get_logger(logger)
    active_backends = _normalize_backends(backend)

    results: list[BenchmarkResult] = []
    split_cache = {
        sample_size: split_data(processed_csv_path, sample_size=sample_size, random_state=random_state, logger=logger)
        for sample_size in sample_sizes
    }
    for current_backend in active_backends:
        logger.info("Running backend: %s", current_backend)
        for sample_size in sample_sizes:
            split = split_cache[sample_size]
            for max_depth in max_depths:
                for n_estimators in n_estimators_list:
                    for repeat in range(1, n_repeats + 1):
                        model = get_model(
                            backend=current_backend,
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=random_state,
                        )

                        prepared_train_features, prepared_train_target = model.prepare_fit_data(split.X_train, split.y_train)
                        train_start = perf_counter()
                        model.fit(prepared_train_features, prepared_train_target)
                        train_time = perf_counter() - train_start

                        prepared_test_features = model.prepare_predict_data(split.X_test)
                        predict_start = perf_counter()
                        raw_predictions = model.predict_raw(prepared_test_features)
                        predict_time = perf_counter() - predict_start

                        predictions = model.postprocess_predictions(raw_predictions)
                        accuracy = accuracy_score(split.y_test, predictions)
                        total_time = train_time + predict_time

                        result = BenchmarkResult(
                            backend=current_backend,
                            sample_size=sample_size,
                            max_depth=max_depth,
                            n_estimators=n_estimators,
                            repeat=repeat,
                            train_time=train_time,
                            predict_time=predict_time,
                            total_time=total_time,
                            accuracy=accuracy,
                        )
                        results.append(result)

    return pd.DataFrame([asdict(result) for result in results])