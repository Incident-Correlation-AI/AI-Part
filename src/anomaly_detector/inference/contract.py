"""Versioned ML inference contract (Milestone 6).

This module defines the **stable, documented boundary** between the ML
anomaly-detection component and the rest of the defensive log-analysis
project. It is intentionally small and dependency-free (standard library
only) so that any downstream Python code can depend on the contract without
knowing anything about the estimator internals.

Design rules enforced here
---------------------------
* **One window in, one structured result out.**  The input is an
  already-aggregated *feature window* produced by the M1 feature pipeline —
  never raw log text and never an arbitrary raw event.
* **Model features are separated from window metadata.**  Only the
  allowlisted numeric features reach the estimator.  Identifiers, timestamps,
  host/service names and any label-like field stay metadata.
* **Feature order is explicit.**  The matrix is built by *name* from the
  detector's expected column list; dictionary iteration order is never used.
* **Unscored is not normal.**  A window that could not be scored carries
  ``is_anomaly = None`` and an explicit status/reason.  It is never reported
  as "not anomalous".
* **Scores are scores, not probabilities.**  The raw score keeps the selected
  model's own direction and is explicitly flagged as non-probabilistic.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

#: Version of *this* request/result contract.  Bump the minor version for
#: additive, backward-compatible changes and the major version for
#: incompatible changes (removed/renamed fields, changed semantics).
CONTRACT_VERSION = "1.0.0"

#: Identifier of the feature schema (the ordered allowlist) the contract
#: speaks.  It matches the feature list recorded in the model manifest.
FEATURE_SCHEMA_VERSION = "features-v1"


class InferenceStatus(str, Enum):
    """Status vocabulary for a single inference result.

    ``scored``
        The input was valid and a score was produced.  ``is_anomaly`` holds
        the threshold decision (either ``True`` or ``False``).
    ``insufficient_data``
        The input is incomplete or unusable under the approved policy (a
        required feature is absent, or the window is unusable under an
        explicit staleness/coverage policy).  No decision is made.
    ``invalid_input``
        Contract/type validation failed (bad types, non-finite values,
        impossible window boundaries, unknown feature names, schema
        mismatch).  No decision is made.
    ``model_unavailable``
        The model artifact is missing, unreadable or incompatible with this
        contract.  This is a *deployment* problem, not a data problem.
    ``inference_error``
        The model/runtime could not produce a trustworthy score for an
        otherwise valid window.
    """

    SCORED = "scored"
    INSUFFICIENT_DATA = "insufficient_data"
    INVALID_INPUT = "invalid_input"
    MODEL_UNAVAILABLE = "model_unavailable"
    INFERENCE_ERROR = "inference_error"

    @property
    def is_scored(self) -> bool:
        return self is InferenceStatus.SCORED


class ReasonCode(str, Enum):
    """Stable machine-readable reason codes.

    The downstream component should branch on these rather than on message
    text.
    """

    OK = "ok"
    MISSING_WINDOW_ID = "missing_window_id"
    MISSING_WINDOW_BOUNDARY = "missing_window_boundary"
    NAIVE_TIMESTAMP = "naive_timestamp"
    INVALID_WINDOW_BOUNDARY = "invalid_window_boundary"
    MISSING_FEATURES = "missing_features"
    UNKNOWN_FEATURES = "unknown_features"
    NON_NUMERIC_FEATURE = "non_numeric_feature"
    NON_FINITE_FEATURE = "non_finite_feature"
    FEATURE_SCHEMA_MISMATCH = "feature_schema_mismatch"
    STALE_WINDOW = "stale_window"
    DUPLICATE_WINDOW_ID = "duplicate_window_id"
    MODEL_ARTIFACT_MISSING = "model_artifact_missing"
    MODEL_ARTIFACT_INVALID = "model_artifact_invalid"
    MODEL_MANIFEST_MISMATCH = "model_manifest_mismatch"
    MODEL_UNAVAILABLE = "model_unavailable"
    INFERENCE_FAILED = "inference_failed"


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


@dataclass
class FeatureWindowRequest:
    """One already-aggregated feature window, ready to be scored.

    Attributes
    ----------
    window_id:
        Required, stable, traceable identifier for the window.  It should be
        the key used by the rest of the project so a result can be linked
        back to its source window.  In this repository the M1/M2 sample key
        (``host|service|window_start``) is the canonical form — see
        :func:`make_window_id`.
    window_start:
        Required, timezone-aware start of the window (UTC recommended).
    window_end:
        Required, timezone-aware end of the window.  Must be strictly after
        ``window_start``.
    host_id:
        Optional source machine identifier.  Metadata only — never a model
        feature.
    service_name:
        Optional source service identifier.  Metadata only — never a model
        feature.
    features:
        Mapping of feature name to numeric value.  Every allowlisted feature
        must be present.  ``None`` means *missing* (not zero) and is handled
        by the model's documented missing-value policy.
    feature_schema_version:
        Optional schema identifier.  When supplied it must match the
        detector's schema version, otherwise the request is rejected.
    extra:
        Optional caller-supplied metadata carried through untouched.  It is
        never used as a model feature and never logged by default.
    """

    window_id: str
    window_start: datetime
    window_end: datetime
    features: Dict[str, Any]
    host_id: Optional[str] = None
    service_name: Optional[str] = None
    feature_schema_version: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_feature_window(cls, window: Any, window_id: Optional[str] = None) -> "FeatureWindowRequest":
        """Build a request from an M1 :class:`~anomaly_detector.schemas.FeatureWindow`.

        Reuses the established M1 schema so no competing window schema is
        introduced.
        """
        metadata = window.metadata
        features = dict(window.to_features_dict())
        if window_id is None:
            window_id = make_window_id(metadata.host_id, metadata.service, metadata.window_start)
        return cls(
            window_id=window_id,
            window_start=metadata.window_start,
            window_end=metadata.window_end,
            features=features,
            host_id=metadata.host_id,
            service_name=metadata.service,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            extra={"window_size_minutes": metadata.window_size_minutes},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "window_start": _iso(self.window_start),
            "window_end": _iso(self.window_end),
            "host_id": self.host_id,
            "service_name": self.service_name,
            "feature_schema_version": self.feature_schema_version,
            "features": dict(self.features),
            "extra": dict(self.extra),
        }


def make_window_id(
    host_id: Optional[str],
    service_name: Optional[str],
    window_start: datetime,
) -> str:
    """Build the canonical window id used across M1–M3 artifacts.

    The format matches the M1/M2 ``sample_key`` (``host|service|window_start``)
    so downstream code can join inference results back to existing splits,
    predictions and evidence tables.
    """
    start = _iso(window_start)
    return f"{host_id or ''}|{service_name or ''}|{start}"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    """Structured result of scoring one feature window.

    ``is_anomaly`` is ``None`` for every non-``scored`` status — an unscored
    window is never reported as normal.
    """

    contract_version: str
    status: InferenceStatus
    reason_code: ReasonCode
    reason: str = ""

    window_id: Optional[str] = None
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    host_id: Optional[str] = None
    service_name: Optional[str] = None

    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_artifact_version: Optional[str] = None
    feature_schema_version: Optional[str] = None
    threshold_config_version: Optional[str] = None

    raw_score: Optional[float] = None
    score_direction: Optional[str] = None
    score_is_probability: bool = False
    threshold: Optional[float] = None
    is_anomaly: Optional[bool] = None

    imputed_features: List[str] = field(default_factory=list)
    ignored_features: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None

    @property
    def is_scored(self) -> bool:
        return self.status is InferenceStatus.SCORED

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable representation (matches the documented examples)."""
        return {
            "contract_version": self.contract_version,
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "reason": self.reason,
            "window_id": self.window_id,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "host_id": self.host_id,
            "service_name": self.service_name,
            "model": {
                "name": self.model_name,
                "version": self.model_version,
                "artifact_version": self.model_artifact_version,
                "feature_schema_version": self.feature_schema_version,
                "threshold_config_version": self.threshold_config_version,
            },
            "score": {
                "raw_score": self.raw_score,
                "score_direction": self.score_direction,
                "is_probability": self.score_is_probability,
            },
            "decision": {
                "threshold": self.threshold,
                "is_anomaly": self.is_anomaly,
            },
            "data_quality": {
                "imputed_features": list(self.imputed_features),
                "ignored_features": list(self.ignored_features),
            },
            "extra": dict(self.extra),
            "duration_ms": self.duration_ms,
        }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def is_aware(value: datetime) -> bool:
    """True when a datetime carries timezone information."""
    return value.tzinfo is not None and value.tzinfo.utcoffset(value) is not None


def to_utc(value: datetime) -> datetime:
    """Normalise an aware datetime to UTC; naive datetimes are rejected."""
    if not is_aware(value):
        raise ValueError("naive datetime is not accepted; use timezone-aware UTC")
    return value.astimezone(timezone.utc)


def feature_values_in_order(
    features: Mapping[str, Any],
    expected_columns: Sequence[str],
) -> List[Any]:
    """Return feature values in the *expected* order.

    The lookup is by name; dictionary iteration order is never relied upon.
    """
    return [features[name] for name in expected_columns]
