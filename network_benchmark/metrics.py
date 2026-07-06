from __future__ import annotations

import numpy as np
import pandas as pd


TIME_COLUMNS = [
    "train_time",
    "predict_time",
    "total_time",
]


EXCLUDED_COLUMNS = [
    "repeat",
    *TIME_COLUMNS,
]


def _safe_speedup(cpu_value: float, gpu_value: float) -> float:
    if gpu_value == 0 or pd.isna(cpu_value) or pd.isna(gpu_value):
        return np.nan
    return cpu_value / gpu_value


def _group_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]


def _aggregate_metrics(raw_results: pd.DataFrame) -> pd.DataFrame:

    group_columns = _group_columns(raw_results)

    summary = (
        raw_results
        .groupby(group_columns, dropna=False)
        .agg(
            train_time_mean=("train_time", "mean"),
            train_time_std=("train_time", "std"),
            train_time_min=("train_time", "min"),
            train_time_max=("train_time", "max"),

            predict_time_mean=("predict_time", "mean"),
            predict_time_std=("predict_time", "std"),
            predict_time_min=("predict_time", "min"),
            predict_time_max=("predict_time", "max"),

            total_time_mean=("total_time", "mean"),
            total_time_std=("total_time", "std"),
            total_time_min=("total_time", "min"),
            total_time_max=("total_time", "max"),
        )
        .reset_index()
    )

    summary.insert(0, "record_type", "backend")

    return summary


def _compute_speedup_rows(raw_results: pd.DataFrame) -> pd.DataFrame:

    if not {"cpu", "gpu"}.issubset(set(raw_results["backend"].unique())):
        return pd.DataFrame()

    group_columns = _group_columns(raw_results)

    averaged = (
        raw_results
        .groupby(group_columns, dropna=False)
        .agg(
            train_time=("train_time", "mean"),
            predict_time=("predict_time", "mean"),
        )
        .reset_index()
    )

    cpu_rows = averaged[
        averaged["backend"] == "cpu"
    ].drop(columns=["backend"])

    gpu_rows = averaged[
        averaged["backend"] == "gpu"
    ].drop(columns=["backend"])

    merge_columns = [
        column
        for column in group_columns
        if column != "backend"
    ]

    merged = cpu_rows.merge(
        gpu_rows,
        on=merge_columns,
        suffixes=("_cpu", "_gpu"),
    )

    merged["train_speedup"] = merged.apply(
        lambda row: _safe_speedup(
            row["train_time_cpu"],
            row["train_time_gpu"],
        ),
        axis=1,
    )

    merged["prediction_speedup"] = merged.apply(
        lambda row: _safe_speedup(
            row["predict_time_cpu"],
            row["predict_time_gpu"],
        ),
        axis=1,
    )

    merged.insert(0, "record_type", "speedup")
    merged.insert(1, "backend", "cpu_vs_gpu")

    return merged


def build_summary(raw_results: pd.DataFrame) -> pd.DataFrame:

    backend_summary = _aggregate_metrics(raw_results)
    speedup_summary = _compute_speedup_rows(raw_results)

    if speedup_summary.empty:
        return backend_summary

    return pd.concat(
        [backend_summary, speedup_summary],
        ignore_index=True,
        sort=False,
    )


def build_overall_summary(raw_results: pd.DataFrame) -> pd.DataFrame:

    return (
        raw_results
        .groupby(["model", "backend"], dropna=False)
        .agg(
            train_time_mean=("train_time", "mean"),
            predict_time_mean=("predict_time", "mean"),
            total_time_mean=("total_time", "mean"),
        )
        .reset_index()
    )


def format_summary_report(raw_results: pd.DataFrame) -> str:

    lines = []

    overall = build_overall_summary(raw_results)

    for _, row in overall.iterrows():

        lines.append("-----------------------------------")
        lines.append(f"Model   : {row['model']}")
        lines.append(f"Backend : {str(row['backend']).upper()}")
        lines.append(f"Average train time      : {row['train_time_mean']:.6f}")
        lines.append(f"Average prediction time : {row['predict_time_mean']:.6f}")
        lines.append(f"Average total time      : {row['total_time_mean']:.6f}")

    speedup_rows = _compute_speedup_rows(raw_results)

    if not speedup_rows.empty:

        lines.append("-----------------------------------")
        lines.append("CPU vs GPU Speedup")

        excluded = {
            "record_type",
            "backend",
            "train_time_cpu",
            "train_time_gpu",
            "predict_time_cpu",
            "predict_time_gpu",
            "train_speedup",
            "prediction_speedup",
        }

        config_columns = [
            column
            for column in speedup_rows.columns
            if column not in excluded
        ]

        for _, row in speedup_rows.iterrows():

            config = " ".join(
                f"{column}={row[column]}"
                for column in config_columns
                if pd.notna(row[column])
            )

            lines.append(config)
            lines.append(f"Train Speedup      : {row['train_speedup']:.2f}x")
            lines.append(f"Prediction Speedup : {row['prediction_speedup']:.2f}x")

    lines.append("-----------------------------------")

    return "\n".join(lines)