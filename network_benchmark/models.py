from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def _to_numpy(values: Any) -> np.ndarray:
	if hasattr(values, "to_numpy"):
		return values.to_numpy()
	return np.asarray(values)


@dataclass
class RandomForestBenchmarkModel:
	backend: str
	n_estimators: int
	max_depth: Optional[int]
	random_state: int
	estimator: Any = field(init=False)
	label_encoder: LabelEncoder | None = field(default=None, init=False)

	def __post_init__(self) -> None:
		self.estimator = self._create_estimator()

	def _create_estimator(self) -> Any:
		if self.backend == "cpu":
			return SklearnRandomForestClassifier(
				n_estimators=self.n_estimators,
				max_depth=self.max_depth,
				random_state=self.random_state,
				n_jobs=-1,
			)

		if self.backend == "gpu":
			try:
				from cuml.ensemble import RandomForestClassifier as CumlRandomForestClassifier
			except ImportError as exc:
				raise ImportError("cuML is required for the GPU backend.") from exc

			return CumlRandomForestClassifier(
				n_estimators=self.n_estimators,
				max_depth=self.max_depth,
				random_state=self.random_state,
			)

		raise ValueError(f"Unsupported backend: {self.backend}")

	def _prepare_gpu_features(self, features: Any) -> Any:
		try:
			import cudf
		except ImportError as exc:
			raise ImportError("cudf is required for the GPU backend.") from exc

		if isinstance(features, cudf.DataFrame):
			return features
		if isinstance(features, pd.DataFrame):
			return cudf.from_pandas(features)
		return cudf.DataFrame(features)

	def _prepare_gpu_target(self, target: Any) -> Any:
		try:
			import cudf
		except ImportError as exc:
			raise ImportError("cudf is required for the GPU backend.") from exc

		target_array = _to_numpy(target)
		if self.label_encoder is None:
			self.label_encoder = LabelEncoder()
			encoded_target = self.label_encoder.fit_transform(target_array)
		else:
			encoded_target = self.label_encoder.transform(target_array)
		return cudf.Series(encoded_target)

	def prepare_fit_data(self, features: Any, target: Any) -> tuple[Any, Any]:
		if self.backend == "gpu":
			return self._prepare_gpu_features(features), self._prepare_gpu_target(target)
		return features, target

	def prepare_predict_data(self, features: Any) -> Any:
		if self.backend == "gpu":
			return self._prepare_gpu_features(features)
		return features

	def fit(self, features: Any, target: Any) -> None:
		self.estimator.fit(features, target)

	def predict(self, features: Any) -> np.ndarray:
		predictions = self.estimator.predict(features)
		predictions_array = _to_numpy(predictions)
		if self.label_encoder is not None:
			predictions_array = self.label_encoder.inverse_transform(predictions_array.astype(int, copy=False))
		return np.asarray(predictions_array)


def get_model(
	backend: str,
	n_estimators: int,
	max_depth: Optional[int],
	random_state: int,
) -> RandomForestBenchmarkModel:
	return RandomForestBenchmarkModel(
		backend=backend,
		n_estimators=n_estimators,
		max_depth=max_depth,
		random_state=random_state,
	)
