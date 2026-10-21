"""Robust MAD-based anomaly score (transparent statistical baseline).

Fits on training data only:
- per-feature median and MAD (median absolute deviation), with scale factor k=1.4826 for normal consistency
- handles constant/near-constant features (MAD ~ 0) explicitly
- aggregates absolute robust deviations into a row-level score by a documented rule

Score direction: higher score => more anomalous (documented)."""
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

from .model_contract import ModelContractError


@dataclass
class MADState:
    medians: Dict[str, float]
    mads: Dict[str, float]
    constant_features: List[str]
    feature_columns: List[str]
    agg: str
    eps: float


class MADBaseline:
    def __init__(self, feature_columns: List[str], agg: str = "max_abs", eps: float = 1e-12):
        self.feature_columns = list(feature_columns)
        self.agg = agg
        self.eps = float(eps)
        self.state: MADState | None = None
        self.is_fitted = False
        self.score_direction = "higher_is_more_anomalous"

    def fit(self, X_train: np.ndarray):
        if X_train.ndim != 2 or X_train.shape[1] != len(self.feature_columns):
            raise ModelContractError("MAD: shape mismatch")
        if not np.isfinite(X_train).all():
            raise ModelContractError("MAD: non-finite values in training")
        medians = {}
        mads = {}
        constant = []
        for i, col in enumerate(self.feature_columns):
            colv = X_train[:, i].astype(float)
            med = float(np.median(colv))
            mad = float(np.median(np.abs(colv - med)))
            medians[col] = med
            if mad < self.eps or not np.isfinite(mad):
                mads[col] = 0.0
                constant.append(col)
            else:
                mads[col] = mad * 1.4826  # normal consistency
        self.state = MADState(medians=medians, mads=mads, constant_features=constant,
                              feature_columns=self.feature_columns, agg=self.agg, eps=self.eps)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.state is None:
            raise ModelContractError("MAD: not fitted")
        if X.ndim != 2 or X.shape[1] != len(self.feature_columns):
            raise ModelContractError("MAD: shape mismatch")
        if not np.isfinite(X).all():
            raise ModelContractError("MAD: non-finite values")
        deviations = []
        for i, col in enumerate(self.feature_columns):
            med = self.state.medians[col]
            mad = self.state.mads[col]
            colv = X[:, i].astype(float)
            if mad <= self.eps:
                # constant/near-constant: use absolute deviation from median (raw scale)
                deviations.append(np.abs(colv - med))
            else:
                deviations.append(np.abs(colv - med) / mad)
        dev = np.stack(deviations, axis=1)  # (n, d)
        if self.agg == "max_abs":
            s = dev.max(axis=1)
        elif self.agg == "mean_abs":
            s = dev.mean(axis=1)
        elif self.agg == "sum_abs":
            s = dev.sum(axis=1)
        else:
            raise ModelContractError(f"MAD: unknown agg {self.agg}")
        return s.astype(float)

    def contract(self) -> Dict[str, Any]:
        return {
            "type": "mad_robust",
            "score_direction": self.score_direction,
            "agg": self.agg,
            "feature_columns": self.feature_columns,
            "state": self.state.__dict__ if self.state else None,
        }
