"""Isolation Forest baseline detector (Milestone 3).

Thin, explicit wrapper around :class:`sklearn.ensemble.IsolationForest` that
keeps the estimator's contract in one place:

* fitting and scoring are separate operations, so a fitted model can score new
  windows without retraining;
* the feature allowlist, the imputation statistics and the score definition
  travel with the saved artifact, so a saved model can refuse input that does
  not match the schema it was trained on;
* the score direction is stated explicitly and never guessed at.

**Score definition.** ``raw_score`` is ``IsolationForest.score_samples``: scikit-learn
defines it so that *lower values are more anomalous*. The anomaly flag is therefore
``is_anomaly = raw_score <= threshold`` (inclusive at the boundary).
``decision_function`` (positive = inlier) is also available for consumers that
prefer that convention.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .baseline_config import BaselineConfig, ModelConfig
from .model_contract import (
    MissingValueReport,
    ModelContractError,
    ModelInput,
    prepare_model_input,
    select_training_rows,
)


SCORE_NAME = "sklearn.ensemble.IsolationForest.score_samples"
SCORE_DIRECTION = "lower_is_more_anomalous"
THRESHOLD_RULE = "is_anomaly = raw_score <= threshold"
ARTIFACT_VERSION = 1


class BaselineDetector:
    """A fitted (or unfitted) Isolation Forest baseline with an explicit contract."""

    def __init__(
        self,
        model_config: ModelConfig,
        feature_columns: List[str],
        medians: Optional[Dict[str, float]] = None,
        missing_value_policy: str = "median_impute",
        expected_feature_columns: Optional[List[str]] = None,
    ) -> None:
        self.model_config = model_config
        self.feature_columns = list(feature_columns)
        # The full allowlist the model was validated against, including any
        # column that was dropped because it had no usable training values.
        self.expected_feature_columns = list(expected_feature_columns or feature_columns)
        self.medians = dict(medians or {})
        self.missing_value_policy = missing_value_policy
        self.estimator = self._build_estimator(model_config)
        self.is_fitted = False

    # -- construction ---------------------------------------------------
    @staticmethod
    def _build_estimator(config: ModelConfig) -> IsolationForest:
        return IsolationForest(
            n_estimators=config.n_estimators,
            max_samples=config.max_samples,
            contamination=config.contamination,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            bootstrap=config.bootstrap,
        )

    @property
    def score_contract(self) -> Dict[str, str]:
        return {
            "score_name": SCORE_NAME,
            "score_direction": SCORE_DIRECTION,
            "threshold_rule": THRESHOLD_RULE,
            "is_probability": "false",
        }

    # -- fitting / scoring ----------------------------------------------
    def fit(self, X: np.ndarray) -> "BaselineDetector":
        """Fit the estimator on already-prepared numeric input."""
        if X.ndim != 2:
            raise ModelContractError(f"expected a 2-D feature matrix, got shape {X.shape}")
        if X.shape[1] != len(self.feature_columns):
            raise ModelContractError(
                f"feature matrix has {X.shape[1]} column(s) but the model expects "
                f"{len(self.feature_columns)}: {self.feature_columns}"
            )
        if X.shape[0] == 0:
            raise ModelContractError("cannot fit the baseline on an empty training set")
        if not np.isfinite(X).all():
            raise ModelContractError("training matrix contains non-finite values")
        self.estimator.fit(X)
        self.is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise ModelContractError("the baseline model is not fitted")

    def score(self, X: np.ndarray) -> np.ndarray:
        """Raw anomaly score per row. LOWER means MORE anomalous."""
        self._check_fitted()
        if X.ndim != 2 or X.shape[1] != len(self.feature_columns):
            raise ModelContractError(
                f"scoring input must have {len(self.feature_columns)} column(s), got shape {X.shape}"
            )
        if not np.isfinite(X).all():
            raise ModelContractError("scoring input contains non-finite values")
        return self.estimator.score_samples(X)

    def decision(self, X: np.ndarray) -> np.ndarray:
        """scikit-learn decision function: POSITIVE means inlier, negative means outlier."""
        self._check_fitted()
        return self.estimator.decision_function(X)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Sign-flipped score: HIGHER means more anomalous (convenience only)."""
        return -self.score(X)

    @staticmethod
    def flag(scores: np.ndarray, threshold: float) -> np.ndarray:
        """Convert raw scores to boolean flags. The boundary is inclusive."""
        return np.asarray(scores, dtype="float64") <= float(threshold)

    # -- persistence ------------------------------------------------------
    def save(self, path: str) -> str:
        """Serialize the fitted model and its contract with joblib."""
        payload = {
            "artifact_version": ARTIFACT_VERSION,
            "estimator": self.estimator,
            "model_config": asdict(self.model_config),
            "feature_columns": list(self.feature_columns),
            "expected_feature_columns": list(self.expected_feature_columns),
            "medians": dict(self.medians),
            "missing_value_policy": self.missing_value_policy,
            "score_contract": self.score_contract,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path: str) -> "BaselineDetector":
        """Load a saved baseline artifact."""
        try:
            payload = joblib.load(path)
        except FileNotFoundError:
            raise ModelContractError(f"model artifact not found: {path}")
        except Exception as exc:  # noqa: BLE001 - surfaced with context
            raise ModelContractError(f"could not load model artifact {path}: {exc}") from exc
        if not isinstance(payload, dict) or "estimator" not in payload:
            raise ModelContractError(
                f"{path} is not a baseline model artifact (missing 'estimator')"
            )
        if payload.get("artifact_version") != ARTIFACT_VERSION:
            raise ModelContractError(
                f"model artifact {path} has version {payload.get('artifact_version')!r}; "
                f"expected {ARTIFACT_VERSION}"
            )
        model_config = ModelConfig(**payload["model_config"])
        detector = cls(
            model_config=model_config,
            feature_columns=payload["feature_columns"],
            medians=payload.get("medians", {}),
            missing_value_policy=payload.get("missing_value_policy", "median_impute"),
            expected_feature_columns=payload.get("expected_feature_columns"),
        )
        detector.estimator = payload["estimator"]
        detector.is_fitted = bool(payload.get("is_fitted", True))
        return detector

    def contract(self) -> Dict[str, Any]:
        return {
            "feature_columns": list(self.feature_columns),
            "expected_feature_columns": list(self.expected_feature_columns),
            "medians": dict(self.medians),
            "missing_value_policy": self.missing_value_policy,
            "model_config": asdict(self.model_config),
            **self.score_contract,
        }


@dataclass
class TrainingReport:
    """What was actually fitted, and what was left out (and why)."""
    training_data_policy: str
    missing_value_policy: str
    rows_available: int = 0
    rows_used: int = 0
    rows_excluded_anomalous: int = 0
    rows_excluded_unknown_label: int = 0
    cells_imputed: int = 0
    dropped_columns: List[str] = field(default_factory=list)
    label_distribution: Dict[str, int] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def fit_baseline(
    features: pd.DataFrame,
    labels: pd.Series,
    config: BaselineConfig,
) -> Tuple[BaselineDetector, TrainingReport]:
    """Fit the baseline on the approved training rows.

    The estimator is fitted only on the rows selected by the training-data
    policy (by default: labelled *normal* rows only). Imputation statistics are
    learned from exactly those rows and are never taken from validation/test.
    """
    data_cfg = config.data
    feature_columns = list(data_cfg.feature_columns)

    report = TrainingReport(
        training_data_policy=data_cfg.training_data_policy,
        missing_value_policy=data_cfg.missing_value_policy,
        rows_available=int(len(features)),
        feature_columns=list(feature_columns),
    )

    values = labels.to_numpy()
    positives = sum(1 for v in values if v is True or v == 1)
    negatives = sum(1 for v in values if v is False or v == 0)
    report.label_distribution = {"positive": positives, "negative": negatives}

    mask = select_training_rows(labels, policy=data_cfg.training_data_policy,
                                label_column=data_cfg.label_column)
    used = features.loc[mask]
    report.rows_used = int(mask.sum())
    report.rows_excluded_anomalous = int((~mask).sum() - report.rows_excluded_unknown_label)

    if report.rows_used == 0:
        raise ModelContractError(
            "no training rows remain after applying the training-data policy "
            f"({data_cfg.training_data_policy!r}); cannot fit the baseline"
        )

    prepared = prepare_model_input(
        used,
        feature_columns,
        medians=None,
        policy=data_cfg.missing_value_policy,
        sample_key_column=data_cfg.sample_key_column,
        context="training data",
    )
    report.cells_imputed = prepared.report.cells_imputed
    report.dropped_columns = list(prepared.report.dropped_columns)

    detector = BaselineDetector(
        model_config=config.model,
        feature_columns=prepared.feature_columns,
        medians=prepared.medians,
        missing_value_policy=data_cfg.missing_value_policy,
        expected_feature_columns=feature_columns,
    )
    detector.fit(prepared.X)
    return detector, report
