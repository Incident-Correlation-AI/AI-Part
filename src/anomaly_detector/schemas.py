"""Event schema definitions and metadata structures.

This module defines the **common event representation** (the ML data contract)
and the derived window / label structures used throughout Milestone 1. These
fields are an initial ML data contract, not a replacement for the team's
eventual shared API contract.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum


class EventLevel(str, Enum):
    """Normalized severity vocabulary.

    Missing severities normalize to ``UNKNOWN`` (representing absence), never to
    ``INFO``. ``WARN`` is an accepted alias for ``WARNING``; ``FATAL``/``CRIT``
    are treated as ``ERROR`` at normalization time.
    """
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    WARN = "WARNING"  # alias
    ERROR = "ERROR"
    FATAL = "FATAL"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class EventType(str, Enum):
    """Documented event categories used by the feature set."""
    HTTP_REQUEST = "http_request"
    AUTH_FAILURE = "auth_failure"
    AUTH_SUCCESS = "auth_success"
    SERVICE_RESTART = "service_restart"
    SYSTEM_EVENT = "system_event"
    DATABASE_QUERY = "database_query"
    APPLICATION_LOG = "application_log"
    OTHER = "other"


@dataclass
class StructuredEvent:
    """Common structured event representation.

    Attributes
    ----------
    event_id:
        Optional stable identifier used for deduplication. When absent, a
        synthetic id (``evt-<index>``) is assigned; no deduplication occurs
        without a genuine source id.
    timestamp:
        Required, timezone-aware event time normalized to UTC internally.
    host_id:
        Required non-empty source machine identifier.
    service:
        Required non-empty application/service identifier.
    event_type:
        Required normalized category (see :class:`EventType`).
    level:
        Optional normalized severity (see :class:`EventLevel`). Missing becomes
        ``UNKNOWN``.
    attributes:
        Optional mapping of event-specific structured values (e.g.
        ``status_code``, ``latency_ms``). Invalid optional measurements are
        dropped during validation rather than coerced.
    source_metadata:
        Optional, deliberately small, source-specific metadata kept only when it
        is useful and safe. Never contains secrets, tokens, or PII.
    """
    event_id: Optional[str] = None
    timestamp: datetime = None
    host_id: str = ""
    service: str = ""
    event_type: str = "other"
    level: str = "INFO"
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "host_id": self.host_id,
            "service": self.service,
            "event_type": self.event_type,
            "level": self.level,
            "attributes": self.attributes,
            "source_metadata": self.source_metadata
        }

@dataclass
class WindowMetadata:
    """Metadata for a feature window."""
    window_start: datetime
    window_end: datetime
    host_id: str
    service: str
    window_size_minutes: int

@dataclass
class FeatureWindow:
    """Numerical feature window with metadata.

    Availability semantics
    ----------------------
    Counts (``*_count``) are genuine counts and may legitimately be zero.
    Measurements and ratios default to ``None`` when unavailable; they are never
    substituted with a meaningful-looking zero. ``*_eligible_count`` and
    ``latency_sample_count`` record the denominator/sample size so downstream
    modelers can tell whether a ratio rests on few or many observations.
    """
    metadata: WindowMetadata
    event_count: int = 0
    error_count: int = 0
    error_ratio: Optional[float] = None
    http_request_count: int = 0
    http_5xx_count: int = 0
    http_5xx_ratio: Optional[float] = None
    auth_failure_count: int = 0
    restart_count: int = 0
    mean_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    # Sample/denominator counts supporting ratio and statistic interpretation.
    http_request_eligible_count: int = 0
    latency_sample_count: int = 0
    error_eligible_count: int = 0

    def to_features_dict(self) -> Dict[str, Any]:
        """Return only numerical model features (no metadata, no labels).

        This is the single source of truth for the model feature matrix. It
        deliberately excludes ``host_id``, ``service``, raw timestamps, window
        boundaries and any ground-truth label/scenario information.
        """
        return {
            "event_count": self.event_count,
            "error_count": self.error_count,
            "error_ratio": self.error_ratio,
            "http_request_count": self.http_request_count,
            "http_5xx_count": self.http_5xx_count,
            "http_5xx_ratio": self.http_5xx_ratio,
            "auth_failure_count": self.auth_failure_count,
            "restart_count": self.restart_count,
            "mean_latency_ms": self.mean_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "latency_sample_count": self.latency_sample_count,
            "http_request_eligible_count": self.http_request_eligible_count,
            "error_eligible_count": self.error_eligible_count
        }
    
    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "window_start": self.metadata.window_start.isoformat(),
            "window_end": self.metadata.window_end.isoformat(),
            "host_id": self.metadata.host_id,
            "service": self.metadata.service,
            "window_size_minutes": self.metadata.window_size_minutes
        }

@dataclass
class LabelInfo:
    """Ground-truth label kept strictly separate from the feature matrix.

    Labels are stored in their own table/metadata structure with a clear
    relationship to the affected time windows. They must never enter the model
    feature matrix; ``tests/test_no_label_leakage.py`` enforces this.
    """
    window_start: datetime
    window_end: datetime
    host_id: str
    service: str
    is_anomaly: bool
    scenario_name: str
    intensity: Optional[float] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "host_id": self.host_id,
            "service": self.service,
            "is_anomaly": self.is_anomaly,
            "scenario_name": self.scenario_name,
            "intensity": self.intensity,
            "description": self.description
        }
