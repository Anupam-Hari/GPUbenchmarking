# Network Benchmark

Reusable benchmarking framework for comparing `sklearn` RandomForestClassifier against `cuML` RandomForestClassifier on the same processed network traffic dataset.

## What This Project Does

- Runs preprocessing once and saves the processed dataset to `processed_data/processed_network_traffic.csv`.
- Benchmarks CPU and GPU Random Forest implementations across configurable depths, tree counts, sample sizes, and repeat counts.
- Measures only model training and prediction time.
- Saves raw run results, grouped summary statistics, and a readable text report.

## Project Layout

- `main.py` - CLI entrypoint.
- `config.py` - Default values and project paths.
- `prepare_data.py` - Notebook preprocessing refactored into reusable functions.
- `split_data.py` - Loads processed data and creates train/validation/test splits.
- `models.py` - Backend-specific model factory for CPU and GPU.
- `benchmark.py` - Benchmark loop and timing logic.
- `metrics.py` - Summary statistics and speedup report generation.
- `requirements.txt` - Python dependencies.

## Setup

Create and activate a virtual environment:

```bash
cd /home/anupam/Anupam/gpubenchmarking/network-benchmark
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU runs, install the RAPIDS stack on the GPU host in an environment that matches the installed CUDA version.

## Dataset

Place the raw dataset at:

```text
datasets/network.csv
```

The preprocessing step will create:

```text
processed_data/processed_network_traffic.csv
```

## CLI Usage

Run the benchmark from the project root:

```bash
python main.py \
  --backend both \
  --depths 5 10 20 None \
  --trees 50 100 200 \
  --samples 10000 50000 \
  --repeats 5 \
  --dataset datasets/network.csv
```

Supported arguments:

- `--backend cpu|gpu|both`
- `--depths 5 10 20 40 None`
- `--trees 50 100 200 500`
- `--samples 10000 25000 50000 100000`
- `--repeats 5`
- `--dataset datasets/network.csv`

CLI values override the defaults in `config.py`.

## Outputs

The benchmark writes:

- `results/csv/raw_results.csv` - one row per run.
- `results/csv/summary.csv` - grouped aggregate statistics.
- `results/logs/summary.txt` - readable report.
- `results/logs/benchmark.log` - execution log.

## Remote GPU Workflow

1. Copy the repository to the GPU machine.
2. Put the raw dataset in `datasets/network.csv`.
3. Create and activate a Python environment on the GPU host.
4. Install `requirements.txt` plus the matching RAPIDS packages for your CUDA version.
5. Run the CLI with `--backend gpu` or `--backend both`.
6. Copy the contents of `results/csv/` and `results/logs/` back to your host machine for review.

## Notes

- Preprocessing is executed once per processed dataset and is not included in benchmark timing.
- Timing uses `time.perf_counter()`.
- GPU-specific `cudf` and `cuML` handling stays inside `models.py`.