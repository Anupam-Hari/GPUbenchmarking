from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional, Sequence

import pandas as pd

from config import BACKENDS, RANDOM_STATE
from models import get_model
from network_benchmark import gpu_monitor
from split_data import split_data
from cpu_monitor import ProcessCPUMonitor
from gpu_monitor import GPUMonitor


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    backend: str
    sample_size: Optional[int] 
    #parameters: dict
    repeat: int
    load_time: float
    prepare_time: float
    train_time: float
    predict_time: float
    total_time: float
    train_cpu_avg: float
    train_cpu_peak: float
    predict_cpu_avg: float
    predict_cpu_peak: float
    train_gpu_avg: float
    train_gpu_peak: float
    train_gpu_mem_avg: float
    train_gpu_mem_peak: float
    predict_gpu_avg: float
    predict_gpu_peak: float
    predict_gpu_mem_avg: float
    predict_gpu_mem_peak: float

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
    sample_sizes = SAMPLE_SIZES if sample_sizes is None else sample_sizes
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

    elif model_name == "dbscan":
        for eps in model_parameters["eps"]:
            for min_samples in model_parameters["min_samples"]:
                parameter_sets.append(
                    {
                        "eps": eps,
                        "min_samples": min_samples,
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

                    cpu_monitor = ProcessCPUMonitor(interval=0.05)
                    gpu_monitor = GPUMonitor(interval=0.05)
                    cpu_monitor.start()
                    gpu_monitor.start()

                    train_start = perf_counter()
                    model.fit(prepared_train_features, prepared_train_target)
                    train_time = perf_counter() - train_start

                    cpu_monitor.stop()
                    gpu_monitor.stop()
                    train_cpu_avg = cpu_monitor.average
                    train_cpu_peak = cpu_monitor.peak
                    train_gpu_avg = gpu_monitor.average_gpu_util
                    train_gpu_peak = gpu_monitor.peak_gpu_util
                    train_gpu_mem_avg = gpu_monitor.average_memory
                    train_gpu_mem_peak = gpu_monitor.peak_memory

                    if model_name == "dbscan":
                        predict_time = 0.0
                        predict_time = 0.0
                        predict_cpu_avg = 0.0
                        predict_cpu_peak = 0.0
                        predict_gpu_avg = 0.0
                        predict_gpu_peak = 0.0
                        predict_gpu_mem_avg = 0.0
                        predict_gpu_mem_peak = 0.0
                    else:
                        prepared_test_features = model.prepare_predict_data(split.X_test)

                        cpu_monitor.start()
                        gpu_monitor.start()
                        predict_start = perf_counter()
                        model.predict_raw(prepared_test_features)
                        predict_time = perf_counter() - predict_start

                        cpu_monitor.stop()
                        gpu_monitor.stop()
                        predict_cpu_avg = cpu_monitor.average
                        predict_cpu_peak = cpu_monitor.peak
                        predict_gpu_avg = gpu_monitor.average_gpu_util
                        predict_gpu_peak = gpu_monitor.peak_gpu_util
                        predict_gpu_mem_avg = gpu_monitor.average_memory
                        predict_gpu_mem_peak = gpu_monitor.peak_memory

                    result = BenchmarkResult(
                        model=model_name,
                        backend=current_backend,
                        sample_size=sample_size,
                        #parameters=parameters,
                        repeat=repeat,
                        load_time=split.load_time,
                        prepare_time=split.prepare_time,
                        train_time=train_time,
                        predict_time=predict_time,
                        total_time=(
                            split.load_time
                            + split.prepare_time
                            + train_time
                            + predict_time
                        ),
                        train_cpu_avg=train_cpu_avg,
                        train_cpu_peak=train_cpu_peak,
                        predict_cpu_avg=predict_cpu_avg,
                        predict_cpu_peak=predict_cpu_peak,
                        train_gpu_avg=train_gpu_avg,
                        train_gpu_peak=train_gpu_peak,
                        train_gpu_mem_avg=train_gpu_mem_avg,
                        train_gpu_mem_peak=train_gpu_mem_peak,
                        predict_gpu_avg=predict_gpu_avg,
                        predict_gpu_peak=predict_gpu_peak,
                        predict_gpu_mem_avg=predict_gpu_mem_avg,
                        predict_gpu_mem_peak=predict_gpu_mem_peak,
                    )
                    results.append(result)

    rows = []
    for result in results:
        row = {
            "model": result.model,
            "backend": result.backend,
            "sample_size": result.sample_size,
            "repeat": result.repeat,
            "load_time": result.load_time,
            "prepare_time": result.prepare_time,
            "train_time": result.train_time,
            "predict_time": result.predict_time,
            "total_time": result.total_time,
            "train_cpu_avg": result.train_cpu_avg,
            "train_cpu_peak": result.train_cpu_peak,
            "predict_cpu_avg": result.predict_cpu_avg,
            "predict_cpu_peak": result.predict_cpu_peak,
            "train_gpu_avg": result.train_gpu_avg,
            "train_gpu_peak": result.train_gpu_peak,
            "train_gpu_mem_avg": result.train_gpu_mem_avg,
            "train_gpu_mem_peak": result.train_gpu_mem_peak,
            "predict_gpu_avg": result.predict_gpu_avg,
            "predict_gpu_peak": result.predict_gpu_peak,
            "predict_gpu_mem_avg": result.predict_gpu_mem_avg,
            "predict_gpu_mem_peak": result.predict_gpu_mem_peak,
        }
        rows.append(row)

    return pd.DataFrame(rows)