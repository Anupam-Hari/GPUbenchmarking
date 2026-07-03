from __future__ import annotations

import numpy as np
import pandas as pd


SUMMARY_GROUP_COLUMNS = ["backend", "sample_size", "max_depth", "n_estimators"]


def _safe_speedup(cpu_value: float, gpu_value: float) -> float:
    if gpu_value == 0 or pd.isna(cpu_value) or pd.isna(gpu_value):
        return np.nan
    return cpu_value / gpu_value


def _aggregate_metrics(raw_results: pd.DataFrame) -> pd.DataFrame:
    grouped = raw_results.groupby(SUMMARY_GROUP_COLUMNS, dropna=False)
    summary = grouped.agg(
        train_time_mean=("train_time", "mean"),
        train_time_median=("train_time", "median"),
        train_time_std=("train_time", "std"),
        train_time_min=("train_time", "min"),
        train_time_max=("train_time", "max"),
        predict_time_mean=("predict_time", "mean"),
        predict_time_median=("predict_time", "median"),
        predict_time_std=("predict_time", "std"),
        predict_time_min=("predict_time", "min"),
        predict_time_max=("predict_time", "max"),
        total_time_mean=("total_time", "mean"),
        total_time_median=("total_time", "median"),
        total_time_std=("total_time", "std"),
        total_time_min=("total_time", "min"),
        total_time_max=("total_time", "max"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_median=("accuracy", "median"),
        accuracy_std=("accuracy", "std"),
        accuracy_min=("accuracy", "min"),
        accuracy_max=("accuracy", "max"),
    ).reset_index()
    summary.insert(0, "record_type", "backend")
    return summary


def _compute_speedup_rows(raw_results: pd.DataFrame) -> pd.DataFrame:
    if not {"cpu", "gpu"}.issubset(set(raw_results["backend"].unique())):
        return pd.DataFrame(columns=["record_type", "backend", "sample_size", "max_depth", "n_estimators", "train_speedup", "prediction_speedup"])

    averaged = raw_results.groupby(SUMMARY_GROUP_COLUMNS, dropna=False).agg(
        train_time=("train_time", "mean"),
        predict_time=("predict_time", "mean"),
    ).reset_index()

    cpu_rows = averaged[averaged["backend"] == "cpu"].drop(columns=["backend"])
    gpu_rows = averaged[averaged["backend"] == "gpu"].drop(columns=["backend"])
    merged = cpu_rows.merge(gpu_rows, on=["sample_size", "max_depth", "n_estimators"], suffixes=("_cpu", "_gpu"))

    merged["train_speedup"] = merged.apply(lambda row: _safe_speedup(row["train_time_cpu"], row["train_time_gpu"]), axis=1)
    merged["prediction_speedup"] = merged.apply(lambda row: _safe_speedup(row["predict_time_cpu"], row["predict_time_gpu"]), axis=1)
    merged.insert(0, "record_type", "speedup")
    merged.insert(1, "backend", "cpu_vs_gpu")
    return merged[["record_type", "backend", "sample_size", "max_depth", "n_estimators", "train_speedup", "prediction_speedup"]]


def build_summary(raw_results: pd.DataFrame) -> pd.DataFrame:
    backend_summary = _aggregate_metrics(raw_results)
    speedup_summary = _compute_speedup_rows(raw_results)
    return pd.concat([backend_summary, speedup_summary], ignore_index=True, sort=False)


def build_overall_summary(raw_results: pd.DataFrame) -> pd.DataFrame:
    return raw_results.groupby("backend", dropna=False).agg(
        train_time_mean=("train_time", "mean"),
        predict_time_mean=("predict_time", "mean"),
        total_time_mean=("total_time", "mean"),
        accuracy_mean=("accuracy", "mean"),
    ).reset_index()


def format_summary_report(raw_results: pd.DataFrame) -> str:
    lines: list[str] = []
    overall = build_overall_summary(raw_results)

    for _, row in overall.iterrows():
        lines.append("-----------------------------------")
        lines.append(f"Backend : {str(row['backend']).upper()}")
        lines.append(f"Average train time : {row['train_time_mean']:.6f}")
        lines.append(f"Average prediction time : {row['predict_time_mean']:.6f}")
        lines.append(f"Average total time : {row['total_time_mean']:.6f}")
        lines.append(f"Average accuracy : {row['accuracy_mean']:.6f}")

    speedup_rows = _compute_speedup_rows(raw_results)
    if not speedup_rows.empty:
        lines.append("-----------------------------------")
        lines.append("CPU vs GPU Speedup")
        for _, row in speedup_rows.iterrows():
            lines.append(f"Depth={row['max_depth']} Trees={row['n_estimators']} Samples={row['sample_size']}")
            lines.append(f"Train Speedup : {row['train_speedup']:.2f}x")
            lines.append(f"Prediction Speedup : {row['prediction_speedup']:.2f}x")

    lines.append("-----------------------------------")
    return "\n".join(lines)