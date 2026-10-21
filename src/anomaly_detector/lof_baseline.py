"""LOF baseline in novelty=True mode (scores unseen validation/test samples).

One lightweight alternative. Fit preprocessing on training only. Documented limitations."""
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from .model_contract import ModelContractError


class LOFBaseline:
    def __init__(self, feature_columns: List[str], n_neighbors: int = 20, contamination: str = "auto"):
        self.feature_columns = list(feature_columns)
        self.n_neighbors = int(n_neighbors)
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True,
                                        contamination=self.contamination, metric="minkowski")
        self.is_fitted = False
        self.score_direction = "lower_is_more_anomalous"  # sklearn LOF score_samples: lower => more abnormal

    def fit(self, X_train: np.ndarray):
        if X_train.ndim != 2 or X_train.shape[1] != len(self.feature_columns):
            raise ModelContractError("LOF: shape mismatch")
        if not np.isfinite(X_train).all():
            raise ModelContractError("LOF: non-finite values in training")
        n = X_train.shape[0]
        if self.n_neighbors >= n:
            # LOF requires n_neighbors < n_samples; adjust down defensively but log-like
            self.n_neighbors = max(1, n - 1)
            self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True,
                                            contamination=self.contamination, metric="minkowski")
        Xs = self.scaler.fit_transform(X_train.astype(float))
        self.model.fit(Xs)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ModelContractError("LOF: not fitted")
        if X.ndim != 2 or X.shape[1] != len(self.feature_columns):
            raise ModelContractError("LOF: shape mismatch")
        if not np.isfinite(X).all():
            raise ModelContractError("LOF: non-finite values")
        Xs = self.scaler.transform(X.astype(float))
        # score_samples: opposite sign to LOF; lower => more abnormal
        return self.model.score_samples(Xs)

    def contract(self) -> Dict[str, Any]:
        return {
            "type": "lof_novelty",
            "score_direction": self.score_direction,
            "n_neighbors": self.n_neighbors,
            "contamination": self.contamination,
            "feature_columns": self.feature_columns,
            "preprocessing": "StandardScaler fitted on training only",
            "novelty": True,
        }
