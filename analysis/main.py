from pathlib import Path

from summary import generate_all_summaries
from utils import load_all_summaries
from plots import (
    plot_gpu_speedup,
    plot_cpu_saved,
    plot_cpu_utilization,
    plot_gpu_utilization,
)


def main():

    # Change this to your benchmark results directory
    root = Path("gpuresults/random_forest_knn_kmeans_dbscan_10072026_110129")

    # Generate data_summary.csv inside every benchmark folder
    generate_all_summaries(root)

    # Load every data_summary.csv into one dataframe
    summary = load_all_summaries(root)

    # Create output directory
    output_dir = root/"plots"
    output_dir.mkdir(exist_ok=True)

    # Generate plots
    plot_gpu_speedup(summary, output_dir)
    plot_cpu_saved(summary, output_dir)
    plot_cpu_utilization(summary, output_dir)
    plot_gpu_utilization(summary, output_dir)


if __name__ == "__main__":
    main()