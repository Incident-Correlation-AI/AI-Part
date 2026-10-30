"""Inference boundary for the network feature schema.

The result object is the existing
:class:`anomaly_detector.inference.contract.InferenceResult`; this module adds
no competing result schema.  ``feature_schema_version`` is
``network-features-v1`` and the model-specific evidence lives in
``result.extra["network"]``.

Decision semantics
------------------
* ``is_anomaly`` is the Isolation Forest threshold decision, exactly like the
  existing contract (``raw_score <= threshold``).
* The transparent baseline contributes **evidence and severity**, never a
  silent rewrite of the model decision.  ``extra.network.baseline`` reports it
  separately, and ``severity`` is derived from the rule evidence.
* An unscored window always has ``is_anomaly = None``.  Missing telemetry is
  never reported as normal.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
import json
import time

try:
    import numpy as np
except (ImportError, ModuleNotFoundError):
    np = None

try:
    import pandas as pd
except (ImportError, ModuleNotFoundError):
    pd = None

try:
    from ..baseline_model import BaselineDetector
except (ImportError, ModuleNotFoundError):
    BaselineDetector = Any  # type: ignore[misc,assignment]
from ..model_contract import ModelContractError, prepare_model_input
from ..inference.contract import (
    CONTRACT_VERSION,
    FeatureWindowRequest,
    InferenceResult,
    InferenceStatus,
    ReasonCode,
    feature_values_in_order,
    is_aware,
)
from .baseline import BaselineDecision, score_window as score_baseline
from .config import RuleConfig
from .features import (
    NETWORK_FEATURE_COLUMNS,
    NETWORK_FEATURE_SCHEMA_VERSION,
    NetworkWindow,
)

DEFAULT_NETWORK_MODEL_PATH = "artifacts/network/model/model.joblib"
DEFAULT_NETWORK_MANIFEST_PATH = "artifacts/network/model/manifest.json"

#: Compact, non-identifying feature subset carried as evidence for explanations.
EVIDENCE_FEATURES: List[str] = [
    "event_count",
    "error_count",
    "error_ratio",
    "http_request_count",
    "http_5xx_count",
    "http_5xx_ratio",
    "http_status_eligible_count",
    "auth_failure_count",
    "auth_success_count",
    "auth_failure_ratio",
    "auth_distinct_source_count",
    "auth_distinct_account_count",
    "db_event_count",
    "db_timeout_count",
    "db_timeout_ratio",
    "db_connection_failure_count",
    "db_connection_failure_ratio",
    "latency_sample_count",
    "mean_latency_ms",
    "p95_latency_ms",
    "p95_latency_ratio",
    "restart_count",
    "invalid_event_count",
    "duplicate_event_count",
    "out_of_order_event_count",
    "telemetry_gap_seconds",
]


class NetworkModelUnavailableError(RuntimeError):
    """Raised when the network model artifact cannot be loaded."""


@dataclass
class NetworkDetectorOptions:
    measure_duration: bool = True
    max_missing_fraction: Optional[float] = 0.5


class NetworkDetector:
    """Loaded, reusable network anomaly detector (Isolation Forest + rules)."""

    def __init__(
        self,
        detector: BaselineDetector,
        manifest: Dict[str, Any],
        rules: Optional[RuleConfig] = None,
        model_path: str = "",
        manifest_path: str = "",
        options: Optional[NetworkDetectorOptions] = None,
    ) -> None:
        self._detector = detector
        self._manifest = manifest
        self.rules = rules or RuleConfig()
        self.model_path = model_path
        self.manifest_path = manifest_path
        self.options = options or NetworkDetectorOptions()

        self.feature_columns: List[str] = list(detector.feature_columns)
        self.expected_feature_columns: List[str] = list(detector.expected_feature_columns)
        self.missing_value_policy = detector.missing_value_policy
        self.threshold = float(manifest["threshold"]["threshold"])
        self.score_direction = manifest["score"]["score_direction"]
        self.model_name = manifest.get("model", {}).get("type", "unknown")
        self.model_version = str(manifest.get("code_revision") or "unknown")
        self.model_artifact_version = str(manifest.get("artifact_version", "unknown"))
        self.feature_schema_version = NETWORK_FEATURE_SCHEMA_VERSION
        self.threshold_config_version = (
            f"{manifest.get('configuration', {}).get('config_path', 'unknown')}"
            f"@{self.model_version}:{self.threshold:.12g}"
        )

    def contract(self) -> Dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "feature_schema_version": self.feature_schema_version,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_artifact_version": self.model_artifact_version,
            "threshold": self.threshold,
            "threshold_config_version": self.threshold_config_version,
            "score_direction": self.score_direction,
            "score_is_probability": False,
            "missing_value_policy": self.missing_value_policy,
            "feature_columns": list(self.feature_columns),
            "expected_feature_columns": list(self.expected_feature_columns),
            "rules": self.rules.__dict__.copy(),
            "decision_rule": "is_anomaly = raw_score <= threshold (Isolation Forest)",
        }

    def score_window(self, window: NetworkWindow) -> InferenceResult:
        if not isinstance(window, NetworkWindow):
            raise TypeError(
                f"score_window expects a NetworkWindow, got {type(window).__name__}"
            )
        started = time.perf_counter()
        result = self._score_window_inner(window)
        if self.options.measure_duration:
            result.duration_ms = round((time.perf_counter() - started) * 1000.0, 4)
        return result

    def score_windows(self, windows: Sequence[NetworkWindow]) -> List[InferenceResult]:
        results: List[InferenceResult] = []
        seen: Dict[str, int] = {}
        for window in windows:
            key = window.sample_key
            if key in seen:
                results.append(self._error_result(
                    window,
                    InferenceStatus.INVALID_INPUT,
                    ReasonCode.DUPLICATE_WINDOW_ID,
                    f"duplicate sample_key {key!r} in batch (first seen at index {seen[key]})",
                ))
                continue
            seen[key] = len(results)
            results.append(self.score_window(window))
        return results

    # -- internals --------------------------------------------------------
    def _score_window_inner(self, window: NetworkWindow) -> InferenceResult:
        metadata = window.metadata
        if not is_aware(metadata.window_start) or not is_aware(metadata.window_end):
            return self._error_result(
                window, InferenceStatus.INVALID_INPUT, ReasonCode.NAIVE_TIMESTAMP,
                "window boundaries must be timezone-aware",
            )
        if metadata.window_end <= metadata.window_start:
            return self._error_result(
                window, InferenceStatus.INVALID_INPUT, ReasonCode.INVALID_WINDOW_BOUNDARY,
                "window_end must be strictly after window_start",
            )

        features = window.features
        if not isinstance(features, Mapping):
            return self._error_result(
                window, InferenceStatus.INVALID_INPUT, ReasonCode.NON_NUMERIC_FEATURE,
                "features must be a mapping",
            )

        if _number(features.get("event_count")) in (None, 0.0):
            return self._error_result(
                window, InferenceStatus.INSUFFICIENT_DATA, ReasonCode.NO_EVENTS,
                "no accepted telemetry events in this window; the window is unscored, not normal",
            )

        event_count = _number(features.get("event_count")) or 0.0
        invalid_count = _number(features.get("invalid_event_count")) or 0.0
        if event_count < self.rules.min_window_events:
            return self._error_result(
                window, InferenceStatus.INSUFFICIENT_DATA, ReasonCode.INSUFFICIENT_COVERAGE,
                f"only {event_count:.0f} accepted event(s); the configured minimum is "
                f"{self.rules.min_window_events}",
            )
        if invalid_count > event_count:
            return self._error_result(
                window, InferenceStatus.INSUFFICIENT_DATA, ReasonCode.TELEMETRY_QUALITY,
                f"{invalid_count:.0f} invalid/duplicate observation(s) exceed the "
                f"{event_count:.0f} accepted event(s); telemetry quality is insufficient",
            )

        missing = [c for c in self.expected_feature_columns if c not in features]
        if missing:
            return self._error_result(
                window, InferenceStatus.INSUFFICIENT_DATA, ReasonCode.MISSING_FEATURES,
                f"required feature(s) missing: {missing}",
            )
        unknown = [c for c in features if c not in self.expected_feature_columns]
        if unknown:
            return self._error_result(
                window, InferenceStatus.INVALID_INPUT, ReasonCode.UNKNOWN_FEATURES,
                f"unknown feature column(s) {sorted(unknown)}",
            )

        numeric: List[float] = []
        for column, value in zip(
            self.expected_feature_columns,
            feature_values_in_order(features, self.expected_feature_columns),
        ):
            if value is None:
                numeric.append(float("nan"))
                continue
            if isinstance(value, bool):
                return self._error_result(
                    window, InferenceStatus.INVALID_INPUT, ReasonCode.NON_NUMERIC_FEATURE,
                    f"feature {column!r} is a boolean",
                )
            try:
                number = float(value)
            except (TypeError, ValueError):
                return self._error_result(
                    window, InferenceStatus.INVALID_INPUT, ReasonCode.NON_NUMERIC_FEATURE,
                    f"feature {column!r} is not numeric",
                )
            if np.isinf(number):
                return self._error_result(
                    window, InferenceStatus.INVALID_INPUT, ReasonCode.NON_FINITE_FEATURE,
                    f"feature {column!r} contains +/-infinity",
                )
            numeric.append(number)

        missing_fraction = sum(1 for v in numeric if np.isnan(v)) / max(1, len(numeric))
        if (
            self.options.max_missing_fraction is not None
            and missing_fraction > self.options.max_missing_fraction
        ):
            return self._error_result(
                window, InferenceStatus.INSUFFICIENT_DATA, ReasonCode.MISSING_FEATURES,
                f"{missing_fraction:.0%} of features are missing, above the allowed "
                f"{self.options.max_missing_fraction:.0%}",
            )

        frame = pd.DataFrame([dict(zip(self.expected_feature_columns, numeric))])
        try:
            prepared = prepare_model_input(
                frame,
                self.expected_feature_columns,
                medians=self._detector.medians,
                policy=self.missing_value_policy,
                sample_key_column=None,
                context=f"network-inference[{window.sample_key}]",
                use_columns=self.feature_columns,
            )
        except ModelContractError as exc:
            return self._error_result(
                window, InferenceStatus.INSUFFICIENT_DATA, ReasonCode.MISSING_FEATURES, str(exc),
            )

        try:
            scores = self._detector.score(prepared.X)
        except Exception as exc:  # noqa: BLE001 - surfaced as an explicit status
            return self._error_result(
                window, InferenceStatus.INFERENCE_ERROR,
                ReasonCode.INFERENCE_FAILED, f"scoring failed: {exc}",
            )

        raw_score = float(np.asarray(scores, dtype="float64").ravel()[0])
        is_anomaly = bool(self._detector.flag(np.array([raw_score]), self.threshold)[0])
        baseline = score_baseline(features, self.rules)
        severity = _severity(
            is_anomaly,
            baseline,
            self.rules,
            raw_score=raw_score,
            threshold=self.threshold,
            features=features,
        )
        evidence = _evidence(features)

        return InferenceResult(
            contract_version=CONTRACT_VERSION,
            status=InferenceStatus.SCORED,
            reason_code=ReasonCode.OK,
            reason="scored with the network Isolation Forest model",
            window_id=window.sample_key,
            window_start=metadata.window_start.isoformat(),
            window_end=metadata.window_end.isoformat(),
            host_id=metadata.host_id,
            service_name=metadata.service,
            model_name=self.model_name,
            model_version=self.model_version,
            model_artifact_version=self.model_artifact_version,
            feature_schema_version=self.feature_schema_version,
            threshold_config_version=self.threshold_config_version,
            raw_score=raw_score,
            score_direction=self.score_direction,
            score_is_probability=False,
            threshold=self.threshold,
            is_anomaly=bool(is_anomaly),
            imputed_features=[
                column for column, value in zip(self.feature_columns, numeric)
                if np.isnan(value)
            ],
            ignored_features=[],
            extra={
                "window_size_minutes": metadata.window_size_minutes,
                "network": {
                    "feature_schema_version": NETWORK_FEATURE_SCHEMA_VERSION,
                    "severity": severity,
                    "decision_source": "isolation_forest",
                    "model_only_anomaly": bool(is_anomaly and not baseline.is_anomaly),
                    "baseline": baseline.to_dict(),
                    "telemetry_quality": dict(window.quality),
                    "evidence_features": evidence,
                },
            },
        )

    def _error_result(
        self,
        window: NetworkWindow,
        status: InferenceStatus,
        reason_code: ReasonCode,
        reason: str,
    ) -> InferenceResult:
        metadata = window.metadata
        schema_version = NETWORK_FEATURE_SCHEMA_VERSION
        return InferenceResult(
            contract_version=CONTRACT_VERSION,
            status=status,
            reason_code=reason_code,
            reason=reason,
            window_id=window.sample_key,
            window_start=metadata.window_start.isoformat() if is_aware(metadata.window_start) else None,
            window_end=metadata.window_end.isoformat() if is_aware(metadata.window_end) else None,
            host_id=metadata.host_id,
            service_name=metadata.service,
            model_name=self.model_name,
            model_version=self.model_version,
            model_artifact_version=self.model_artifact_version,
            feature_schema_version=schema_version,
            threshold_config_version=self.threshold_config_version,
            raw_score=None,
            score_direction=self.score_direction,
            score_is_probability=False,
            threshold=self.threshold,
            is_anomaly=None,
            imputed_features=[],
            ignored_features=[],
            extra={
                "window_size_minutes": metadata.window_size_minutes,
                "network": {
                    "feature_schema_version": schema_version,
                    "severity": "unknown",
                    "decision_source": "none",
                    "model_only_anomaly": False,
                    "telemetry_quality": dict(window.quality),
                },
            },
        )


def load_network_detector(
    model_path: str = DEFAULT_NETWORK_MODEL_PATH,
    manifest_path: str = DEFAULT_NETWORK_MANIFEST_PATH,
    rules: Optional[RuleConfig] = None,
    options: Optional[NetworkDetectorOptions] = None,
) -> NetworkDetector:
    """Load the network model artifact and cross-check its manifest."""
    model_file = Path(model_path)
    manifest_file = Path(manifest_path)
    if not model_file.is_file():
        raise NetworkModelUnavailableError(
            f"network model artifact not found: {model_path} "
            "(regenerate with: python scripts/run_network_pipeline.py)"
        )
    if not manifest_file.is_file():
        raise NetworkModelUnavailableError(f"network model manifest not found: {manifest_path}")

    try:
        detector = BaselineDetector.load(str(model_file))
    except ModelContractError as exc:
        raise NetworkModelUnavailableError(f"could not load network model artifact: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise NetworkModelUnavailableError(
            f"could not load network model artifact {model_path}: {exc}"
        ) from exc

    try:
        with open(manifest_file, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        raise NetworkModelUnavailableError(
            f"could not read network model manifest {manifest_path}: {exc}"
        ) from exc

    _validate_manifest(manifest, detector)
    return NetworkDetector(
        detector=detector,
        manifest=manifest,
        rules=rules,
        model_path=str(model_path),
        manifest_path=str(manifest_path),
        options=options,
    )


def _validate_manifest(manifest: Dict[str, Any], detector: BaselineDetector) -> None:
    for key in ("model", "feature_schema", "threshold", "score", "code_revision"):
        if key not in manifest:
            raise NetworkModelUnavailableError(
                f"network model manifest is missing the {key!r} section"
            )
    schema = manifest["feature_schema"]
    if schema.get("feature_schema_version") != NETWORK_FEATURE_SCHEMA_VERSION:
        raise NetworkModelUnavailableError(
            f"manifest feature_schema_version "
            f"{schema.get('feature_schema_version')!r} is not "
            f"{NETWORK_FEATURE_SCHEMA_VERSION!r}"
        )
    if list(schema.get("feature_columns", [])) != list(detector.feature_columns):
        raise NetworkModelUnavailableError(
            "manifest feature columns do not match the serialized model columns"
        )
    if manifest.get("threshold", {}).get("threshold") is None:
        raise NetworkModelUnavailableError("manifest does not record a threshold")
    if manifest.get("score", {}).get("score_direction") != detector.score_contract["score_direction"]:
        raise NetworkModelUnavailableError(
            "manifest score direction does not match the serialized model"
        )


def _severity(
    is_anomaly: bool,
    baseline: BaselineDecision,
    rules: RuleConfig,
    raw_score: Optional[float] = None,
    threshold: Optional[float] = None,
    features: Optional[Mapping[str, Any]] = None,
) -> str:
    if not is_anomaly and baseline.risk_score <= 0:
        return "none"

    # 1. Transparent baseline rule hit meets high severity threshold
    if baseline.risk_score >= rules.high_severity_threshold:
        return "high"

    # 2. Isolation Forest deep tail outlier: raw_score <= threshold - isolation_forest_high_delta
    if is_anomaly and raw_score is not None and threshold is not None:
        high_if_delta = getattr(rules, "isolation_forest_high_delta", 0.10)
        if raw_score <= (threshold - high_if_delta):
            return "high"

    # 3. Robust MAD z-score check
    if features is not None:
        mad_high_z = getattr(rules, "mad_high_zscore", 4.0)
        z_scores = [
            abs(float(features[col]))
            for col in ("event_count_zscore", "http_request_count_zscore")
            if features.get(col) is not None
        ]
        if any(z >= mad_high_z for z in z_scores):
            return "high"

    if is_anomaly or baseline.is_anomaly:
        return "medium"
    if baseline.risk_score > 0:
        return "low"
    return "none"


def _evidence(features: Mapping[str, Any]) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {}
    for name in EVIDENCE_FEATURES:
        value = features.get(name)
        if value is None:
            continue
        evidence[name] = value
    return evidence


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number
