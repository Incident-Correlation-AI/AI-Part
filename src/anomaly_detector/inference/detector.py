"""Reusable ML inference component (Milestone 6).

This is the thin package the downstream evidence-correlation / explanation
component calls.  It knows nothing about incidents, alerts or explanations:
it validates one feature window, runs the *selected, already trained* model
and returns a versioned structured result.

Guarantees
----------
* **No retraining.**  The model artifact and its manifest are loaded once;
  inference never fits anything.
* **No fitting on production data.**  Imputation medians come from the
  training split and are stored inside the artifact.
* **Deterministic.**  Identical valid input + identical artifact gives
  identical output.
* **Explicit failure.**  Problems are reported as a status + reason code,
  never as a silently "normal" window.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
import os
import time

import numpy as np
import pandas as pd

from ..baseline_model import BaselineDetector
from ..model_contract import ModelContractError, prepare_model_input
from .contract import (
    CONTRACT_VERSION,
    FEATURE_SCHEMA_VERSION,
    FeatureWindowRequest,
    InferenceResult,
    InferenceStatus,
    ReasonCode,
    feature_values_in_order,
    is_aware,
)

#: Default locations of the selected model (M3 Isolation Forest baseline,
#: retained as the M4/M5 selection).  M1–M5 artifacts are never overwritten.
DEFAULT_MODEL_PATH = "artifacts/baseline/model/model.joblib"
DEFAULT_MANIFEST_PATH = "artifacts/baseline/model/manifest.json"


class ModelUnavailableError(RuntimeError):
    """Raised when the model artifact cannot be loaded or is incompatible."""


@dataclass
class DetectorOptions:
    """Runtime options for the inference component.

    Attributes
    ----------
    allow_unknown_features:
        When ``False`` (default) a request carrying feature names outside the
        allowlist is rejected as ``invalid_input``.  When ``True`` the unknown
        names are ignored and reported in ``ignored_features``.
    allow_external_model_path:
        Model paths outside the repository's ``artifacts/`` tree require this
        to be set explicitly, because loading a serialized model executes
        code.  Only load artifacts you trust.
    max_imputed_fraction:
        Optional guard: when the fraction of allowlisted features that had to
        be imputed exceeds this value the window is reported as
        ``insufficient_data`` instead of being scored.  ``None`` (default)
        keeps the trained model's own missing-value policy exactly.
    stale_after_seconds:
        Optional guard: windows whose end is older than this many seconds
        (relative to ``reference_time``) are reported as ``insufficient_data``.
        ``None`` (default) disables the check — temporal policy is owned by
        the caller.
    reference_time:
        Reference point for the staleness check.  Defaults to "now".
    measure_duration:
        Whether to record ``duration_ms``.  Timing is a convenience, not a
        guarantee, and is documented as such.
    """

    allow_unknown_features: bool = False
    allow_external_model_path: bool = False
    max_imputed_fraction: Optional[float] = None
    stale_after_seconds: Optional[float] = None
    reference_time: Optional[Any] = None
    measure_duration: bool = True


class AnomalyDetector:
    """A loaded, reusable inference component bound to one model artifact."""

    def __init__(
        self,
        detector: BaselineDetector,
        manifest: Dict[str, Any],
        model_path: str,
        manifest_path: str,
        options: Optional[DetectorOptions] = None,
    ) -> None:
        self._detector = detector
        self._manifest = manifest
        self.model_path = model_path
        self.manifest_path = manifest_path
        self.options = options or DetectorOptions()

        self.feature_columns: List[str] = list(detector.feature_columns)
        self.expected_feature_columns: List[str] = list(detector.expected_feature_columns)
        self.missing_value_policy: str = detector.missing_value_policy
        self.threshold: float = float(manifest["threshold"]["threshold"])
        self.score_direction: str = manifest["score"]["score_direction"]
        self.threshold_rule: str = manifest["score"].get("threshold_rule", "")

        model_meta = manifest.get("model", {})
        self.model_name: str = model_meta.get("type", "unknown")
        self.model_version: str = str(manifest.get("code_revision") or "unknown")
        self.model_artifact_version: str = str(
            manifest.get(
                "artifact_version",
                getattr(detector, "artifact_version", "unknown"),
            )
        )
        self.feature_schema_version: str = FEATURE_SCHEMA_VERSION
        self.threshold_config_version: str = _threshold_config_version(manifest, self.threshold)

    # -- introspection ----------------------------------------------------
    def contract(self) -> Dict[str, Any]:
        """Machine-readable description of what this component is bound to."""
        return {
            "contract_version": CONTRACT_VERSION,
            "feature_schema_version": self.feature_schema_version,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_artifact_version": self.model_artifact_version,
            "threshold": self.threshold,
            "threshold_config_version": self.threshold_config_version,
            "threshold_rule": self.threshold_rule,
            "score_direction": self.score_direction,
            "score_is_probability": False,
            "missing_value_policy": self.missing_value_policy,
            "feature_columns": list(self.feature_columns),
            "expected_feature_columns": list(self.expected_feature_columns),
            "model_path": self.model_path,
            "manifest_path": self.manifest_path,
        }

    # -- scoring ----------------------------------------------------------
    def score_window(self, request: FeatureWindowRequest) -> InferenceResult:
        """Validate and score one feature window.

        Never raises for data problems: they are reported through the
        result's ``status``/``reason_code``.  Programming errors (a request
        of the wrong type) still raise, because they are not data problems.
        """
        if not isinstance(request, FeatureWindowRequest):
            raise TypeError(
                "score_window expects a FeatureWindowRequest, got "
                f"{type(request).__name__}"
            )

        started = time.perf_counter()
        result = self._score_window_inner(request)
        if self.options.measure_duration:
            result.duration_ms = round((time.perf_counter() - started) * 1000.0, 4)
        return result

    def score_windows(self, requests: Sequence[FeatureWindowRequest]) -> List[InferenceResult]:
        """Score a batch of windows.

        One bad window never fails the batch: every row gets its own result
        with an explicit status.  Duplicate ``window_id`` values inside one
        batch are rejected as ``invalid_input`` for the later occurrence, so
        a result can always be traced back to exactly one window.
        """
        results: List[InferenceResult] = []
        seen: Dict[str, int] = {}
        for request in requests:
            if isinstance(request, FeatureWindowRequest) and request.window_id:
                key = request.window_id
                if key in seen:
                    results.append(
                        self._error_result(
                            request,
                            InferenceStatus.INVALID_INPUT,
                            ReasonCode.DUPLICATE_WINDOW_ID,
                            f"duplicate window_id {key!r} in batch "
                            f"(first seen at index {seen[key]})",
                        )
                    )
                    continue
                seen[key] = len(results)
            results.append(self.score_window(request))
        return results

    # -- internals --------------------------------------------------------
    def _score_window_inner(self, request: FeatureWindowRequest) -> InferenceResult:
        # 1. Identity / boundary validation.
        if not isinstance(request.window_id, str) or not request.window_id.strip():
            return self._error_result(
                request,
                InferenceStatus.INVALID_INPUT,
                ReasonCode.MISSING_WINDOW_ID,
                "window_id is required and must be a non-empty string",
            )
        if not isinstance(request.window_start, datetime) or not isinstance(request.window_end, datetime):
            return self._error_result(
                request,
                InferenceStatus.INVALID_INPUT,
                ReasonCode.MISSING_WINDOW_BOUNDARY,
                "window_start and window_end are required datetimes",
            )
        if not is_aware(request.window_start) or not is_aware(request.window_end):
            return self._error_result(
                request,
                InferenceStatus.INVALID_INPUT,
                ReasonCode.NAIVE_TIMESTAMP,
                "window_start/window_end must be timezone-aware (UTC recommended)",
            )
        if request.window_end <= request.window_start:
            return self._error_result(
                request,
                InferenceStatus.INVALID_INPUT,
                ReasonCode.INVALID_WINDOW_BOUNDARY,
                "window_end must be strictly after window_start",
            )

        # 2. Feature-schema version check.
        if (
            request.feature_schema_version is not None
            and request.feature_schema_version != self.feature_schema_version
        ):
            return self._error_result(
                request,
                InferenceStatus.INVALID_INPUT,
                ReasonCode.FEATURE_SCHEMA_MISMATCH,
                f"feature schema version {request.feature_schema_version!r} is not "
                f"supported by this detector (expected {self.feature_schema_version!r})",
            )

        # 3. Feature presence / unknown names.
        features = request.features
        if not isinstance(features, Mapping):
            return self._error_result(
                request,
                InferenceStatus.INVALID_INPUT,
                ReasonCode.NON_NUMERIC_FEATURE,
                "features must be a mapping of feature name to value",
            )
        missing = [c for c in self.expected_feature_columns if c not in features]
        if missing:
            return self._error_result(
                request,
                InferenceStatus.INSUFFICIENT_DATA,
                ReasonCode.MISSING_FEATURES,
                f"required feature(s) missing: {missing}",
            )
        unknown = [c for c in features if c not in self.expected_feature_columns]
        if unknown and not self.options.allow_unknown_features:
            return self._error_result(
                request,
                InferenceStatus.INVALID_INPUT,
                ReasonCode.UNKNOWN_FEATURES,
                f"unknown feature column(s) {sorted(unknown)}; the model allowlist is "
                f"{self.expected_feature_columns}",
            )

        # 4. Type / finiteness validation of the values themselves.
        values = feature_values_in_order(features, self.expected_feature_columns)
        numeric: List[float] = []
        for column, value in zip(self.expected_feature_columns, values):
            if value is None:
                numeric.append(float("nan"))  # missing, handled by the model's policy
                continue
            if isinstance(value, bool):
                return self._error_result(
                    request,
                    InferenceStatus.INVALID_INPUT,
                    ReasonCode.NON_NUMERIC_FEATURE,
                    f"feature {column!r} is a boolean, which is not a valid numeric value",
                )
            try:
                number = float(value)
            except (TypeError, ValueError):
                return self._error_result(
                    request,
                    InferenceStatus.INVALID_INPUT,
                    ReasonCode.NON_NUMERIC_FEATURE,
                    f"feature {column!r} is not numeric (got {type(value).__name__})",
                )
            if np.isinf(number):
                return self._error_result(
                    request,
                    InferenceStatus.INVALID_INPUT,
                    ReasonCode.NON_FINITE_FEATURE,
                    f"feature {column!r} contains +/-infinity; infinities are never imputed",
                )
            numeric.append(number)

        # 5. Optional staleness policy.
        if self.options.stale_after_seconds is not None:
            reference = self.options.reference_time
            if reference is None:
                reference = datetime.now(timezone.utc)
            age = (reference - request.window_end).total_seconds()
            if age > float(self.options.stale_after_seconds):
                return self._error_result(
                    request,
                    InferenceStatus.INSUFFICIENT_DATA,
                    ReasonCode.STALE_WINDOW,
                    f"window end is {age:.0f}s old, older than the allowed "
                    f"{self.options.stale_after_seconds:.0f}s",
                )

        # 6. Score with the trained model's own preprocessing.
        frame = pd.DataFrame([dict(zip(self.expected_feature_columns, numeric))])
        try:
            prepared = prepare_model_input(
                frame,
                self.expected_feature_columns,
                medians=self._detector.medians,
                policy=self.missing_value_policy,
                sample_key_column=None,
                context=f"inference[{request.window_id}]",
                use_columns=self.feature_columns,
            )
        except ModelContractError as exc:
            return self._error_result(
                request,
                InferenceStatus.INSUFFICIENT_DATA,
                ReasonCode.MISSING_FEATURES,
                str(exc),
            )

        if len(prepared.X) == 0:
            return self._error_result(
                request,
                InferenceStatus.INSUFFICIENT_DATA,
                ReasonCode.MISSING_FEATURES,
                "no rows could be prepared for scoring under the missing-value policy",
            )

        imputed = list(prepared.report.dropped_columns)  # unused here, kept explicit
        imputed_features = [
            column
            for column in self.feature_columns
            if np.isnan(numeric[self.expected_feature_columns.index(column)])
        ]

        if (
            self.options.max_imputed_fraction is not None
            and self.feature_columns
        ):
            fraction = len(imputed_features) / len(self.feature_columns)
            if fraction > float(self.options.max_imputed_fraction):
                return self._error_result(
                    request,
                    InferenceStatus.INSUFFICIENT_DATA,
                    ReasonCode.MISSING_FEATURES,
                    f"{len(imputed_features)} of {len(self.feature_columns)} features were "
                    f"missing ({fraction:.0%}), above the allowed "
                    f"{self.options.max_imputed_fraction:.0%}",
                )

        try:
            scores = self._detector.score(prepared.X)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            return self._error_result(
                request,
                InferenceStatus.INFERENCE_ERROR,
                ReasonCode.INFERENCE_FAILED,
                f"scoring failed: {exc}",
            )

        raw_score = float(np.asarray(scores, dtype="float64").ravel()[0])
        # Inclusive boundary, exactly as the M3 baseline defines it.
        is_anomaly = bool(self._detector.flag(np.array([raw_score]), self.threshold)[0])

        result = InferenceResult(
            contract_version=CONTRACT_VERSION,
            status=InferenceStatus.SCORED,
            reason_code=ReasonCode.OK,
            reason="scored with the selected model",
            window_id=request.window_id,
            window_start=request.window_start.isoformat(),
            window_end=request.window_end.isoformat(),
            host_id=request.host_id,
            service_name=request.service_name,
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
            imputed_features=imputed_features,
            ignored_features=sorted(unknown),
            extra=dict(request.extra or {}),
        )
        del imputed
        return result

    def _error_result(
        self,
        request: FeatureWindowRequest,
        status: InferenceStatus,
        reason_code: ReasonCode,
        reason: str,
    ) -> InferenceResult:
        """Build a non-scored result.  ``is_anomaly`` stays ``None``."""
        window_id = request.window_id if isinstance(getattr(request, "window_id", None), str) else None
        window_start = getattr(request, "window_start", None)
        window_end = getattr(request, "window_end", None)
        return InferenceResult(
            contract_version=CONTRACT_VERSION,
            status=status,
            reason_code=reason_code,
            reason=reason,
            window_id=window_id or None,
            window_start=window_start.isoformat() if isinstance(window_start, datetime) else None,
            window_end=window_end.isoformat() if isinstance(window_end, datetime) else None,
            host_id=getattr(request, "host_id", None),
            service_name=getattr(request, "service_name", None),
            model_name=self.model_name,
            model_version=self.model_version,
            model_artifact_version=self.model_artifact_version,
            feature_schema_version=self.feature_schema_version,
            threshold_config_version=self.threshold_config_version,
            raw_score=None,
            score_direction=self.score_direction,
            score_is_probability=False,
            threshold=self.threshold,
            is_anomaly=None,  # never "False" for an unscored window
            extra=dict(getattr(request, "extra", {}) or {}),
        )


def _threshold_config_version(manifest: Dict[str, Any], threshold: float) -> str:
    """Stable identifier for the threshold + configuration used."""
    config = manifest.get("configuration", {}) or {}
    config_path = config.get("config_path", "unknown")
    revision = manifest.get("code_revision", "unknown")
    return f"{config_path}@{revision}:{threshold:.12g}"


def load_detector(
    model_path: str = DEFAULT_MODEL_PATH,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    options: Optional[DetectorOptions] = None,
) -> AnomalyDetector:
    """Load the selected model artifact and its manifest once.

    The returned object is reusable: call :meth:`AnomalyDetector.score_window`
    as often as needed without reloading anything.

    Raises
    ------
    ModelUnavailableError
        If the artifact or manifest is missing, unreadable, or incompatible
        with this contract.
    """
    options = options or DetectorOptions()
    model_file = Path(model_path)
    manifest_file = Path(manifest_path)

    if not model_file.is_file():
        raise ModelUnavailableError(
            f"model artifact not found: {model_path} "
            f"(regenerate with: python scripts/run_baseline.py run)"
        )
    if not manifest_file.is_file():
        raise ModelUnavailableError(f"model manifest not found: {manifest_path}")
    if not options.allow_external_model_path and not _is_inside_repo_artifacts(model_file):
        raise ModelUnavailableError(
            f"refusing to load a model artifact from outside the repository artifacts "
            f"tree: {model_path}. Serialized models execute code on load; pass "
            f"DetectorOptions(allow_external_model_path=True) to accept a trusted "
            f"external artifact."
        )

    try:
        detector = BaselineDetector.load(str(model_file))
    except ModelContractError as exc:
        raise ModelUnavailableError(f"could not load model artifact: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surfaced with context
        raise ModelUnavailableError(
            f"could not load model artifact {model_path}: {exc}"
        ) from exc

    try:
        import json

        with open(manifest_file, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        raise ModelUnavailableError(
            f"could not read model manifest {manifest_path}: {exc}"
        ) from exc

    _validate_manifest(manifest, detector, model_path, manifest_path)

    return AnomalyDetector(
        detector=detector,
        manifest=manifest,
        model_path=str(model_path),
        manifest_path=str(manifest_path),
        options=options,
    )


def _validate_manifest(
    manifest: Dict[str, Any],
    detector: BaselineDetector,
    model_path: str,
    manifest_path: str,
) -> None:
    """Cross-check manifest against the artifact before any scoring happens."""
    for key in ("model", "feature_schema", "threshold", "score", "code_revision"):
        if key not in manifest:
            raise ModelUnavailableError(
                f"model manifest {manifest_path} is missing the {key!r} section"
            )

    manifest_columns = list(manifest["feature_schema"]["feature_columns"])
    artifact_columns = list(detector.feature_columns)
    if manifest_columns != artifact_columns:
        raise ModelUnavailableError(
            f"model manifest {manifest_path} declares feature columns "
            f"{manifest_columns} but the artifact {model_path} was trained on "
            f"{artifact_columns}; refusing to score with a mismatched schema"
        )

    threshold = manifest.get("threshold", {}).get("threshold")
    if threshold is None:
        raise ModelUnavailableError(
            f"model manifest {manifest_path} does not record a threshold"
        )
    direction = manifest.get("score", {}).get("score_direction")
    if direction != detector.score_contract["score_direction"]:
        raise ModelUnavailableError(
            f"model manifest {manifest_path} declares score direction {direction!r} but "
            f"the artifact produces {detector.score_contract['score_direction']!r}"
        )


def _is_inside_repo_artifacts(path: Path) -> bool:
    """True when the path lives under the repository's ``artifacts/`` tree."""
    try:
        repo = Path(__file__).resolve().parents[3]
    except IndexError:  # pragma: no cover - defensive
        return False
    try:
        path.resolve().relative_to(repo / "artifacts")
        return True
    except ValueError:
        return False
