from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional, Sequence

import pandas as pd

from config import BACKENDS, RANDOM_STATE
from models import get_model
from split_data import split_data


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    backend: str
    sample_size: Optional[int]
    parameters: dict
    repeat: int
    train_time: float
    predict_time: float
    total_time: float

def _get_logger(logger: logging.Logger | None) -> logging.Logger:
    return logger if logger is not None else logging.getLogger(__name__)


def _normalize_backends(backend: str) -> list[str]:
    if backend == "both":
        return list(BACKENDS)
    return [backend]


def run_benchmark(
    model_name: str,
    backend: str,
    processed_csv_path: Path | str,
    sample_sizes: Sequence[Optional[int]],
    model_parameters: dict,
    n_repeats: int,
    random_state: int = RANDOM_STATE,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    logger = _get_logger(logger)
    active_backends = _normalize_backends(backend)

    parameter_sets = []

    if model_name == "random_forest":
        for trees in model_parameters["n_estimators"]:
            for depth in model_parameters["max_depth"]:
                parameter_sets.append(
                    {
                        "n_estimators": trees,
                        "max_depth": depth,
                    }
                )

    elif model_name == "knn":
        for neighbors in model_parameters["n_neighbors"]:
            parameter_sets.append(
                {
                    "n_neighbors": neighbors,
                }
            )

    elif model_name == "kmeans":
        for clusters in model_parameters["n_clusters"]:
            parameter_sets.append(
                {
                    "n_clusters": clusters,
                }
            )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    results: list[BenchmarkResult] = []
    split_cache = {
        sample_size: split_data(processed_csv_path, sample_size=sample_size, random_state=random_state, logger=logger)
        for sample_size in sample_sizes
    }
    for current_backend in active_backends:
        logger.info("Running %s benchmark on %s backend",model_name,current_backend.upper(),)
        for sample_size in sample_sizes:
            split = split_cache[sample_size]
            for parameters in parameter_sets:
                for repeat in range(1, n_repeats + 1):
                    model = get_model(
                        model_name=model_name,
                        backend=current_backend,
                        parameters=parameters,
                        random_state=random_state,
                    )

                    prepared_train_features, prepared_train_target = model.prepare_fit_data(split.X_train, split.y_train)
                    train_start = perf_counter()
                    model.fit(prepared_train_features, prepared_train_target)
                    train_time = perf_counter() - train_start

                    prepared_test_features = model.prepare_predict_data(split.X_test)
                    predict_start = perf_counter()
                    model.predict_raw(prepared_test_features)
                    predict_time = perf_counter() - predict_start

                    total_time = train_time + predict_time

                    result = BenchmarkResult(
                        model=model_name,
                        backend=current_backend,
                        sample_size=sample_size,
                        parameters=parameters,
                        repeat=repeat,
                        train_time=train_time,
                        predict_time=predict_time,
                        total_time=total_time,
                    )
                    results.append(result)

    rows = []
    for result in results:
        row = {
            "model": result.model,
            "backend": result.backend,
            "sample_size": result.sample_size,
            **result.parameters,
            "repeat": result.repeat,
            "train_time": result.train_time,
            "predict_time": result.predict_time,
            "total_time": result.total_time,
        }
        rows.append(row)

    return pd.DataFrame(rows)