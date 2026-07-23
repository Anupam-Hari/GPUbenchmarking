from __future__ import annotations

import argparse
import pandas as pd
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from time import perf_counter

from benchmark import run_benchmark
from split_data import load_processed_data
from config import (
    BACKENDS,
    DATASETS_DIR,
    DEFAULT_DATASET_PATH,
    DEFAULT_PROCESSED_DATA_PATH,
    MODEL_PARAMETERS,
    SUPPORTED_MODELS,
    N_REPEATS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    SAMPLE_SIZES,
)
from metrics import build_summary, format_summary_report
from prepare_data import prepare_data


def parse_optional_int(value: str) -> Optional[int]:
	if value.lower() in {"none", "null", "nil", ""}:
		return None
	return int(value)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Benchmark sklearn and cuML machine learning models.")
	parser.add_argument("--backend", choices=["cpu", "gpu", "both"], default="both")
	parser.add_argument("--model", nargs="+", choices=SUPPORTED_MODELS, default=SUPPORTED_MODELS,)
	parser.add_argument("--samples", nargs="+", type=parse_optional_int, default=SAMPLE_SIZES)
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
	for path in [DATASETS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR]:
		path.mkdir(parents=True, exist_ok=True)


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	ensure_directories()
	run_name = f"{'_'.join(args.model)}_{datetime.now().strftime('%d%m%Y_%H%M%S')}"
	run_dir = RESULTS_DIR / run_name
	run_dir.mkdir(parents=True, exist_ok=True)

	log_path = run_dir / "benchmark.log"
	raw_results_path = run_dir / "raw_results.csv"
	summary_csv_path = run_dir / "summary.csv"
	summary_text_path = run_dir / "summary.txt"

	logger = setup_logging(log_path)

	dataset_path = args.dataset.expanduser().resolve() if not args.dataset.is_absolute() else args.dataset
	logger.info("Using dataset source: %s", dataset_path)

	processed_csv_path = prepare_data(
		dataset_path=dataset_path,
		output_path=DEFAULT_PROCESSED_DATA_PATH,
		overwrite=args.overwrite_processed,
		logger=logger,
	)
	#load_start = perf_counter()
	df = load_processed_data(processed_csv_path)
	#load_time = perf_counter() - load_start

	#logger.info("Dataset loaded in %.2f seconds", load_time)

	all_results = []

	for model_name in args.model:
		model_parameters = MODEL_PARAMETERS[model_name]

		logger.info("Selected model: %s", model_name)

		results = run_benchmark(
			model_name=model_name,
			backend=args.backend,
			df=df,
			sample_sizes=args.samples,
			model_parameters=model_parameters,
			n_repeats=args.repeats,
			random_state=args.random_state,
			logger=logger,
		)

		all_results.append(results)

	raw_results = pd.concat(all_results, ignore_index=True)

	# summary = build_summary(raw_results)
	# raw_results.to_csv(raw_results_path, index=False)
	# summary.to_csv(summary_csv_path, index=False)

	# summary_text = format_summary_report(raw_results)
	# summary_text_path.write_text(summary_text, encoding="utf-8")

	# logger.info("Run directory: %s", run_dir)
	# logger.info("Raw results saved to %s", raw_results_path)
	# logger.info("Summary CSV saved to %s", summary_csv_path)
	# logger.info("Summary text saved to %s", summary_text_path)


if __name__ == "__main__":
	main()
