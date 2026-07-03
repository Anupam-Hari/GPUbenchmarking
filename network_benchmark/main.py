from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from benchmark import run_benchmark
from config import (
	BACKENDS,
	DATASETS_DIR,
	DEFAULT_DATASET_PATH,
	DEFAULT_LOG_PATH,
	DEFAULT_PROCESSED_DATA_PATH,
	DEFAULT_RAW_RESULTS_PATH,
	DEFAULT_SUMMARY_CSV_PATH,
	DEFAULT_SUMMARY_TEXT_PATH,
	N_ESTIMATORS,
	N_REPEATS,
	PROCESSED_DATA_DIR,
	RANDOM_STATE,
	RESULTS_CSV_DIR,
	RESULTS_DIR,
	RESULTS_LOGS_DIR,
	SAMPLE_SIZES,
	TREE_DEPTHS,
)
from metrics import build_summary, format_summary_report
from prepare_data import prepare_data


def parse_optional_int(value: str) -> Optional[int]:
	if value.lower() in {"none", "null", "nil", ""}:
		return None
	return int(value)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Benchmark sklearn and cuML RandomForestClassifier backends.")
	parser.add_argument("--backend", choices=["cpu", "gpu", "both"], default="both")
	parser.add_argument("--depths", nargs="+", type=parse_optional_int, default=list(TREE_DEPTHS))
	parser.add_argument("--trees", nargs="+", type=int, default=list(N_ESTIMATORS))
	parser.add_argument("--samples", nargs="+", type=parse_optional_int, default=list(SAMPLE_SIZES))
	parser.add_argument("--repeats", type=int, default=N_REPEATS)
	parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
	parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
	parser.add_argument("--overwrite-processed", action="store_true")
	return parser


def setup_logging(log_path: Path) -> logging.Logger:
	logger = logging.getLogger("network_benchmark")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()

	formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)

	file_handler = logging.FileHandler(log_path)
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)

	return logger


def ensure_directories() -> None:
	for path in [DATASETS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, RESULTS_CSV_DIR, RESULTS_LOGS_DIR]:
		path.mkdir(parents=True, exist_ok=True)


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	ensure_directories()
	logger = setup_logging(DEFAULT_LOG_PATH)

	dataset_path = args.dataset.expanduser().resolve() if not args.dataset.is_absolute() else args.dataset
	logger.info("Using dataset source: %s", dataset_path)

	processed_csv_path = prepare_data(
		dataset_path=dataset_path,
		output_path=DEFAULT_PROCESSED_DATA_PATH,
		overwrite=args.overwrite_processed,
		logger=logger,
	)

	raw_results = run_benchmark(
		backend=args.backend,
		processed_csv_path=processed_csv_path,
		sample_sizes=args.samples,
		max_depths=args.depths,
		n_estimators_list=args.trees,
		n_repeats=args.repeats,
		random_state=args.random_state,
		logger=logger,
	)

	summary = build_summary(raw_results)
	raw_results.to_csv(DEFAULT_RAW_RESULTS_PATH, index=False)
	summary.to_csv(DEFAULT_SUMMARY_CSV_PATH, index=False)

	summary_text = format_summary_report(raw_results)
	DEFAULT_SUMMARY_TEXT_PATH.write_text(summary_text, encoding="utf-8")

	logger.info("Raw results saved to %s", DEFAULT_RAW_RESULTS_PATH)
	logger.info("Summary CSV saved to %s", DEFAULT_SUMMARY_CSV_PATH)
	logger.info("Summary text saved to %s", DEFAULT_SUMMARY_TEXT_PATH)


if __name__ == "__main__":
	main()
