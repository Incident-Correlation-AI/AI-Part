"""Milestone 6 — stable ML inference interface.

Public API for the downstream (evidence-correlation / explanation) component::

    from anomaly_detector.inference import load_detector, score_window

    detector = load_detector()                     # loads once, reuses
    result = detector.score_window(request)        # one feature window in
    if result.status == InferenceStatus.SCORED:
        ...  # result.raw_score, result.is_anomaly
    else:
        ...  # result.reason_code explains why; is_anomaly is None

Nothing in this package retrieves evidence, correlates events, explains
incidents, delivers alerts or remediates anything.
"""
from .contract import (
    CONTRACT_VERSION,
    FEATURE_SCHEMA_VERSION,
    FeatureWindowRequest,
    InferenceResult,
    InferenceStatus,
    ReasonCode,
    feature_values_in_order,
    is_aware,
    make_window_id,
    to_utc,
)
from .detector import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_MODEL_PATH,
    AnomalyDetector,
    DetectorOptions,
    ModelUnavailableError,
    load_detector,
)
from .adapter import (
    AdapterOutcome,
    DownstreamSink,
    EvidenceCorrelationAdapter,
    RecordingSink,
    RejectedWindow,
    ScoredWindow,
)

__all__ = [
    "AnomalyDetector",
    "AdapterOutcome",
    "CONTRACT_VERSION",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_MODEL_PATH",
    "DetectorOptions",
    "DownstreamSink",
    "EvidenceCorrelationAdapter",
    "FEATURE_SCHEMA_VERSION",
    "FeatureWindowRequest",
    "InferenceResult",
    "InferenceStatus",
    "ModelUnavailableError",
    "ReasonCode",
    "RecordingSink",
    "RejectedWindow",
    "ScoredWindow",
    "feature_values_in_order",
    "is_aware",
    "load_detector",
    "make_window_id",
    "to_utc",
]
