from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_CSV_DIR = RESULTS_DIR / "csv"
RESULTS_LOGS_DIR = RESULTS_DIR / "logs"

DEFAULT_DATASET_PATH = DATASETS_DIR
DEFAULT_PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_network_traffic.csv"
DEFAULT_RAW_RESULTS_PATH = RESULTS_CSV_DIR / "raw_results.csv"
DEFAULT_SUMMARY_CSV_PATH = RESULTS_CSV_DIR / "summary.csv"
DEFAULT_SUMMARY_TEXT_PATH = RESULTS_LOGS_DIR / "summary.txt"
DEFAULT_LOG_PATH = RESULTS_LOGS_DIR / "benchmark.log"

BACKENDS = ("cpu", "gpu")

SUPPORTED_MODELS = (
    "random_forest",
    "knn",
    "kmeans",
)

MODEL_PARAMETERS = {
    "random_forest": {
        "n_estimators": (50, 100, 200, 500),
        "max_depth": (5, 10, 20, 40, None),
    },
    "knn": {
        "n_neighbors": (3, 5, 10, 20, 50),
    },
    "kmeans": {
        "n_clusters": (2, 4, 8, 16, 32),
    },
}

SAMPLE_SIZES = (
    10000,
    25000,
    50000,
    100000,
)

N_REPEATS = 5
RANDOM_STATE = 42