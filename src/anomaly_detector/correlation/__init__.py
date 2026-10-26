"""Evidence-correlation integration.

This package is the smallest compatible adapter between the ML inference
contract and the canonical LAD incident contract.  It reuses the upstream
``correlate_anomalies()`` implementation (vendored with provenance) and the
existing optional ``anomaly_detector.network.llm`` explanation adapter.

Upstream contains **no LLM**; the "evidence-correlation LLM" premise of the
integration prompt is false.  Correlation is deterministic and rule-based.
Explanations are optional, disabled by default and never change a decision.
See ``docs/ml_llm_integration.md`` and
``docs/ml_llm_integration_reconnaissance.md``.
"""
from .config import CorrelationConfig, load_correlation_config
from .evidence import (
    WindowEvidence,
    is_safe_event_id,
    is_safe_identity,
    is_safe_redacted_ref,
    normalise_evidence_index,
    validate_window_evidence,
)
from .incident import (
    CorrelationCandidate,
    build_incident,
    correlate_candidates,
    deterministic_incident_id,
)
from .pipeline import (
    CorrelationError,
    CorrelationOutcome,
    CorrelationRecord,
    EvidenceCorrelationPipeline,
    build_explanation_provider,
)
from .upstream import UPSTREAM_MODULE, UPSTREAM_REVISION

__all__ = [
    "CorrelationCandidate",
    "CorrelationConfig",
    "CorrelationError",
    "CorrelationOutcome",
    "CorrelationRecord",
    "EvidenceCorrelationPipeline",
    "UPSTREAM_MODULE",
    "UPSTREAM_REVISION",
    "WindowEvidence",
    "build_explanation_provider",
    "build_incident",
    "correlate_candidates",
    "deterministic_incident_id",
    "is_safe_event_id",
    "is_safe_identity",
    "is_safe_redacted_ref",
    "load_correlation_config",
    "normalise_evidence_index",
    "validate_window_evidence",
]
