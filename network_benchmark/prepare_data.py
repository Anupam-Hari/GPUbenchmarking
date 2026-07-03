from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import DEFAULT_PROCESSED_DATA_PATH

COMMON_COLUMNS = [
	"ts",
	"uid",
	"id.orig_h",
	"id.orig_p",
	"id.resp_h",
	"id.resp_p",
	"proto",
	"service",
	"duration",
	"orig_bytes",
	"resp_bytes",
	"conn_state",
	"local_orig",
	"local_resp",
	"missed_bytes",
	"history",
	"orig_pkts",
	"orig_ip_bytes",
	"resp_pkts",
	"resp_ip_bytes",
	"tunnel_parents",
	"label",
	"detailed-label",
]

NUMERIC_DTYPE_MAP = {
	"ts": np.float32,
	"id.orig_p": np.int32,
	"id.resp_p": np.int32,
	"duration": np.float32,
	"orig_bytes": np.float32,
	"resp_bytes": np.float32,
	"missed_bytes": np.float32,
	"orig_pkts": np.int32,
	"orig_ip_bytes": np.float32,
	"resp_pkts": np.int32,
	"resp_ip_bytes": np.float32,
}

ZERO_IMPUTE_NUMERIC_COLUMNS = [
	"duration",
	"orig_bytes",
	"resp_bytes",
	"missed_bytes",
	"orig_pkts",
	"orig_ip_bytes",
	"resp_pkts",
	"resp_ip_bytes",
]

CATEGORICAL_COLUMNS = [
	"service",
	"local_orig",
	"local_resp",
	"tunnel_parents",
	"proto",
	"conn_state",
	"label",
	"detailed-label",
]

NUMERICAL_LOG_FEATURES = [
	"duration",
	"orig_bytes",
	"resp_bytes",
	"missed_bytes",
	"orig_pkts",
	"orig_ip_bytes",
	"resp_pkts",
	"resp_ip_bytes",
]

HISTORY_FLAGS = ["S", "h", "A", "D", "f", "R", "c", "w", "i", "q", "t", "g"]

INTERNAL_IP_PATTERNS = [
	r"^10\.",
	r"^172\.(1[6-9]|2[0-9]|3[0-1])\.",
	r"^192\.168\."
]


def _get_logger(logger: logging.Logger | None) -> logging.Logger:
	return logger if logger is not None else logging.getLogger(__name__)


def _read_raw_csv(file_path: Path) -> pd.DataFrame:
	dtype_map = {
		key: value
		for key, value in NUMERIC_DTYPE_MAP.items()
		if key not in {"duration", "orig_bytes", "resp_bytes"}
	}
	return pd.read_csv(
		file_path,
		sep="|",
		names=COMMON_COLUMNS,
		skiprows=1,
		skipinitialspace=True,
		dtype=dtype_map,
		low_memory=False,
	)


def _load_raw_frames(dataset_path: Path) -> list[pd.DataFrame]:
	if dataset_path.is_dir():
		csv_files = sorted(path for path in dataset_path.rglob("*.csv") if path.is_file())
		if not csv_files:
			raise FileNotFoundError(f"No CSV files found in directory: {dataset_path}")
		return [_read_raw_csv(path) for path in csv_files]

	if not dataset_path.is_file():
		raise FileNotFoundError(f"Dataset not found: {dataset_path}")

	return [_read_raw_csv(dataset_path)]


def _process_numeric_columns(df: pd.DataFrame, logger: logging.Logger) -> None:
	for column in ZERO_IMPUTE_NUMERIC_COLUMNS:
		if column in df.columns:
			df[column] = pd.to_numeric(df[column], errors="coerce").astype(NUMERIC_DTYPE_MAP[column])
			df[column] = df[column].fillna(0)
			logger.info("Filled NaN in %s with 0 and converted to %s.", column, NUMERIC_DTYPE_MAP[column])


def _process_categorical_columns(df: pd.DataFrame, logger: logging.Logger) -> None:
	for column in CATEGORICAL_COLUMNS:
		if column in df.columns:
			df[column] = df[column].astype(str).str.strip()
			df[column] = df[column].replace("", np.nan)
			df[column] = df[column].fillna("unknown")
			df[column] = df[column].astype("category")
			logger.info("Processed %s as category.", column)


def _engineer_ip_features(df: pd.DataFrame, logger: logging.Logger) -> None:
	combined_internal_ip_pattern = "|".join(INTERNAL_IP_PATTERNS)

	def check_internal_ips_vectorized(ip_series: pd.Series) -> pd.Series:
		ip_series_str = ip_series.astype(str)
		return ip_series_str.str.contains(combined_internal_ip_pattern, regex=True, na=False).astype(np.int8)

	if "id.orig_h" in df.columns:
		df["is_orig_internal"] = check_internal_ips_vectorized(df["id.orig_h"])
		logger.info("Created is_orig_internal feature.")

	if "id.resp_h" in df.columns:
		df["is_resp_internal"] = check_internal_ips_vectorized(df["id.resp_h"])
		logger.info("Created is_resp_internal feature.")

	df.drop(columns=["id.orig_h", "id.resp_h"], inplace=True, errors="ignore")


def _engineer_temporal_features(df: pd.DataFrame, logger: logging.Logger) -> None:
	if "ts" in df.columns:
		df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype(NUMERIC_DTYPE_MAP["ts"])
		df["ts"] = df["ts"].fillna(df["ts"].median())
		df["timestamp"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
		df["hour_of_day"] = df["timestamp"].dt.hour.astype(np.int8)
		df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(np.int8)
		df.drop(columns=["ts", "timestamp"], inplace=True, errors="ignore")
		logger.info("Created temporal features hour_of_day and day_of_week.")


def _drop_early_columns(df: pd.DataFrame, logger: logging.Logger) -> None:
	columns_to_drop = ["uid", "local_orig", "local_resp", "tunnel_parents"]
	existing_columns = [column for column in columns_to_drop if column in df.columns]
	if existing_columns:
		df.drop(columns=existing_columns, inplace=True)
		logger.info("Dropped early columns: %s.", existing_columns)


def _engineer_log_features(df: pd.DataFrame, logger: logging.Logger) -> None:
	for column in NUMERICAL_LOG_FEATURES:
		if column in df.columns:
			df[f"log_{column}"] = np.log1p(df[column]).astype(np.float32)
			df.drop(columns=[column], inplace=True)
			logger.info("Applied log1p transformation to %s.", column)


def _engineer_history_features(df: pd.DataFrame, logger: logging.Logger) -> None:
	if "history" not in df.columns:
		return

	df["history"] = df["history"].astype(str).fillna("")
	for flag in HISTORY_FLAGS:
		df[f"history_has_{flag}"] = df["history"].str.contains(flag, regex=False, na=False).astype(np.int8)
	df.drop(columns=["history"], inplace=True, errors="ignore")
	logger.info("Extracted history flags.")


def _engineer_labels(df: pd.DataFrame, logger: logging.Logger) -> None:
	if "label" not in df.columns or "detailed-label" not in df.columns:
		raise KeyError("Expected label and detailed-label columns are missing.")

	df["is_malicious"] = df["label"].apply(lambda value: 1 if "Malicious" in str(value) else 0).astype(np.int8)
	df["detailed_label_clean"] = df["detailed-label"].astype(str).str.strip()
	df["label_clean"] = df["label"].astype(str).str.strip()

	conditions = [
		(df["detailed_label_clean"] == "PartOfAHorizontalPortScan"),
		(df["detailed_label_clean"] == "C&C"),
		(df["detailed_label_clean"] == "Attack"),
		(df["detailed_label_clean"] == "HeartBeat"),
		(df["detailed_label_clean"] == "Torii"),
		(df["detailed_label_clean"] == "FileDownload"),
		(df["label_clean"].str.contains("DDoS", na=False)),
		(df["label_clean"].str.contains("PartOfAHorizontalPortScan", na=False)),
		(df["label_clean"].str.contains("C&C", na=False)),
		(df["label_clean"].str.contains("Attack", na=False)),
		(df["label_clean"].str.contains("FileDownload", na=False)),
		(df["label_clean"].str.contains("Malicious", na=False)),
	]

	choices = [
		"PartOfAHorizontalPortScan",
		"C&C",
		"Attack",
		"HeartBeat",
		"Torii",
		"FileDownload",
		"DDoS",
		"PartOfAHorizontalPortScan",
		"C&C",
		"Attack",
		"FileDownload",
		"General_Malware",
	]

	df["attack_type"] = np.select(conditions, choices, default="Benign_Traffic")
	df.drop(columns=["detailed_label_clean", "label_clean"], inplace=True, errors="ignore")
	df["attack_type"] = df["attack_type"].astype("category")
	df.drop(columns=["label", "detailed-label"], inplace=True, errors="ignore")
	logger.info("Created is_malicious and attack_type labels.")


def _scale_numeric_features(df: pd.DataFrame, logger: logging.Logger) -> None:
	features_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()
	features_to_scale = [column for column in features_to_scale if column not in ["is_malicious"]]
	if not features_to_scale:
		logger.info("No numerical features found for scaling.")
		return

	scaler = MinMaxScaler()
	df[features_to_scale] = scaler.fit_transform(df[features_to_scale]).astype(np.float32)
	logger.info("Applied MinMaxScaler to %d numerical features.", len(features_to_scale))


def process_dataframe(df: pd.DataFrame, logger: logging.Logger | None = None) -> pd.DataFrame:
	logger = _get_logger(logger)
	df = df.copy()
	_process_numeric_columns(df, logger)
	_process_categorical_columns(df, logger)
	_engineer_ip_features(df, logger)
	_engineer_temporal_features(df, logger)
	_drop_early_columns(df, logger)
	_engineer_log_features(df, logger)

	categorical_features_to_ohe = [column for column in ["proto", "service", "conn_state"] if column in df.columns]
	if categorical_features_to_ohe:
		df = pd.get_dummies(df, columns=categorical_features_to_ohe, prefix=categorical_features_to_ohe, dummy_na=False)
		logger.info("Applied one-hot encoding to %s.", categorical_features_to_ohe)

	_engineer_history_features(df, logger)
	_engineer_labels(df, logger)
	_scale_numeric_features(df, logger)
	return df


def prepare_data(
	dataset_path: Path | str,
	output_path: Path | None = None,
	overwrite: bool = False,
	logger: logging.Logger | None = None,
) -> Path:
	logger = _get_logger(logger)
	source_path = Path(dataset_path)
	destination_path = Path(output_path) if output_path is not None else DEFAULT_PROCESSED_DATA_PATH
	destination_path.parent.mkdir(parents=True, exist_ok=True)

	if destination_path.exists() and not overwrite:
		logger.info("Processed dataset already exists at %s. Reusing it.", destination_path)
		return destination_path

	logger.info("Loading raw dataset from %s.", source_path)
	frames = _load_raw_frames(source_path)
	processed_df = process_dataframe(pd.concat(frames, ignore_index=True), logger=logger)
	processed_df.to_csv(destination_path, index=False)
	logger.info("Processed data saved to %s.", destination_path)
	return destination_path
