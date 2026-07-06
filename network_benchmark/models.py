from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class BenchmarkModel:
    model_name: str
    backend: str
    parameters: dict[str, Any]
    random_state: int

    estimator: Any = field(init=False)

    def __post_init__(self) -> None:
        self.estimator = self._create_estimator()

    def _create_estimator(self) -> Any:

        # ---------------- Random Forest ----------------

        if self.model_name == "random_forest":

            if self.backend == "cpu":
                from sklearn.ensemble import RandomForestClassifier

                return RandomForestClassifier(
                    random_state=self.random_state,
                    n_jobs=-1,
                    **self.parameters,
                )

            elif self.backend == "gpu":
                from cuml.ensemble import RandomForestClassifier

                return RandomForestClassifier(
                    random_state=self.random_state,
                    **self.parameters,
                )

        # ---------------- KNN ----------------
        
        elif self.model_name == "knn":

            if self.backend == "cpu":
                from sklearn.neighbors import KNeighborsClassifier

                return KNeighborsClassifier(
                    n_jobs=-1,
                    **self.parameters,
                )

            elif self.backend == "gpu":
                from cuml.neighbors import KNeighborsClassifier

                return KNeighborsClassifier(
                    **self.parameters,
                )

        # ---------------- KMeans ----------------

        elif self.model_name == "kmeans":

            if self.backend == "cpu":
                from sklearn.cluster import KMeans

                return KMeans(
                    random_state=self.random_state,
                    **self.parameters,
                )

            elif self.backend == "gpu":
                from cuml.cluster import KMeans

                return KMeans(
                    random_state=self.random_state,
                    **self.parameters,
                )

        raise ValueError(
            f"Unsupported model/backend combination: "
            f"{self.model_name}, {self.backend}"
        )

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

    def prepare_fit_data(
        self,
        features: Any,
        target: Any,
    ) -> tuple[Any, Any]:

        if self.backend == "gpu":
            try:
                import cudf
            except ImportError as exc:
                raise ImportError("cudf is required for the GPU backend.") from exc

            features = self._prepare_gpu_features(features)

            if self.model_name != "kmeans":
                target = cudf.Series(target)

        return features, target

    def prepare_predict_data(self, features: Any) -> Any:

        if self.backend == "gpu":
            return self._prepare_gpu_features(features)

        return features

    def fit(
        self,
        features: Any,
        target: Any,
    ) -> None:
        self.estimator.fit(features, target)

    def predict_raw(self, features: Any):

        return self.estimator.predict(features)


def get_model(
    model_name: str,
    backend: str,
    parameters: dict[str, Any],
    random_state: int,
) -> BenchmarkModel:

    return BenchmarkModel(
        model_name=model_name,
        backend=backend,
        parameters=parameters,
        random_state=random_state,
    )