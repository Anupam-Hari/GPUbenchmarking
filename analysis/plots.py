from pathlib import Path
from plotly.subplots import make_subplots

import plotly.express as px
import plotly.graph_objects as go


def _save_figure(fig, output_dir: Path, filename: str):
    """
    Save both interactive HTML and static PNG.
    Requires:
        pip install kaleido
    """

    fig.update_layout(
        template="plotly_white",
        font=dict(size=18),
        width=1000,
        height=600,
        legend_title_text="",
    )

    fig.write_html(output_dir / f"{filename}.html")
    #fig.write_image(output_dir / f"{filename}.png", scale=3)

def plot_gpu_speedup(summary, output_dir: Path):
    """
    GPU Speedup = CPU Time / GPU Time

    where Time = train_time + predict_time
    """

    cpu = summary[summary.backend == "cpu"].copy()
    gpu = summary[summary.backend == "gpu"].copy()

    cpu["total_compute"] = cpu.train_time + cpu.predict_time
    gpu["total_compute"] = gpu.train_time + gpu.predict_time

    merged = cpu.merge(
        gpu,
        on=["model", "sample_size"],
        suffixes=("_cpu", "_gpu"),
    )

    merged["speedup"] = (
        merged.total_compute_cpu /
        merged.total_compute_gpu
    )

    fig = px.line(
        merged,
        x="sample_size",
        y="speedup",
        color="model",
        markers=True,
        log_x=True,
        labels={
            "sample_size": "Sample Size",
            "speedup": "GPU Speedup (CPU Time / GPU Time)",
            "model": "Model",
        },
        title="GPU Speedup Across Sample Sizes",
    )

    _save_figure(fig, output_dir, "gpu_speedup")

def plot_cpu_saved(summary, output_dir: Path):
    """
    CPU Saved (%) =
    ((CPU backend utilization - GPU backend utilization)
        / CPU backend utilization) * 100
    """

    cpu = summary[summary.backend == "cpu"].copy()
    gpu = summary[summary.backend == "gpu"].copy()

    merged = cpu.merge(
        gpu,
        on=["model", "sample_size"],
        suffixes=("_cpu", "_gpu"),
    )

    cpu_avg_cpu = (merged.train_cpu_avg_cpu + merged.predict_cpu_avg_cpu)/2
    cpu_avg_gpu = (merged.train_cpu_avg_gpu + merged.predict_cpu_avg_gpu)/2

    merged["cpu_saved"] = ((cpu_avg_cpu- cpu_avg_gpu)/cpu_avg_cpu) * 100

    fig = px.line(
        merged,
        x="sample_size",
        y="cpu_saved",
        color="model",
        markers=True,
        log_x=True,
        labels={
            "sample_size": "Sample Size",
            "cpu_saved": "CPU Saved (%)",
            "model": "Model",
        },
        title="CPU Utilization Reduction Using GPU",
    )

    fig.update_yaxes(zeroline=True)

    _save_figure(fig, output_dir, "cpu_saved")

def plot_cpu_utilization(summary, output_dir: Path):
    """
    Plot average CPU utilization for CPU and GPU backends.

    One figure with four subplots:
        - Random Forest
        - KNN
        - KMeans
        - DBSCAN
    """

    models = [
        "random_forest",
        "knn",
        "kmeans",
        "dbscan",
    ]

    titles = [
        "Random Forest",
        "KNN",
        "K-Means",
        "DBSCAN",
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=titles,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    positions = [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]

    for model, (row, col) in zip(models, positions):

        D = summary[summary.model == model]

        cpu = D[D.backend == "cpu"]
        gpu = D[D.backend == "gpu"]

        fig.add_trace(
            go.Scatter(
                x=cpu.sample_size,
                y=cpu.train_cpu_avg,
                mode="lines+markers",
                name="CPU Backend",
                legendgroup="cpu",
                showlegend=(row == 1 and col == 1),
                line=dict(color="#1f77b4", width=2),
                marker=dict(color="#1f77b4", size=8),
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=gpu.sample_size,
                y=gpu.train_cpu_avg,
                mode="lines+markers",
                name="GPU Backend",
                legendgroup="gpu",
                showlegend=(row == 1 and col == 1),
                line=dict(color="#ff7f0e", width=2),
                marker=dict(color="#ff7f0e", size=8),
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(
            type="log",
            title_text="Sample Size",
            row=row,
            col=col,
        )

        fig.update_yaxes(
            title_text="CPU Utilization (%)",
            row=row,
            col=col,
        )

    fig.update_layout(
        title="Average CPU Utilization",
        template="plotly_white",
        width=1200,
        height=900,
        font=dict(size=16),
    )

    _save_figure(fig, output_dir, "cpu_utilization")

def plot_gpu_utilization(summary, output_dir: Path):
    """
    Plot average GPU utilization.

    One figure with four subplots:
        - Random Forest
        - KNN
        - KMeans
        - DBSCAN
    """

    models = [
        "random_forest",
        "knn",
        "kmeans",
        "dbscan",
    ]

    titles = [
        "Random Forest",
        "KNN",
        "K-Means",
        "DBSCAN",
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=titles,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    positions = [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]

    for model, (row, col) in zip(models, positions):

        D = summary[
            (summary.model == model) &
            (summary.backend == "gpu")
        ]

        fig.add_trace(
            go.Scatter(
                x=D.sample_size,
                y=D.train_gpu_avg,
                mode="lines+markers",
                name="GPU Utilization",
                legendgroup="gpu",
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(
            type="log",
            title_text="Sample Size",
            row=row,
            col=col,
        )

        fig.update_yaxes(
            title_text="GPU Utilization (%)",
            range=[0, 100],
            row=row,
            col=col,
        )

    fig.update_layout(
        title="Average GPU Utilization",
        template="plotly_white",
        width=1200,
        height=900,
        font=dict(size=16),
    )

    _save_figure(fig, output_dir, "gpu_utilization")