"""Adapter from the canonical LAD event envelope to the ML event schema.

The canonical contract is ``LAD/contracts/schemas/event.schema.json`` (schema
version ``1.0``).  The ML pipeline's own event representation is
:class:`anomaly_detector.schemas.StructuredEvent`.  Rather than inventing a
third event format, this module performs an explicit, testable mapping and
reports every rejection/partial loss instead of silently dropping data.

Mapping rules
-------------
=====================  =====================================================
LAD field              ML field
=====================  =====================================================
``event_id``           ``event_id`` (deduplication key)
``event_ts``           ``timestamp`` (timezone-aware, normalised to UTC)
``host``               ``host_id``
``service``            ``service`` (``null`` -> ``"unknown"`` + quality flag)
``category``/``type``  ``event_type`` via :func:`map_event_type`
``severity``           ``level`` (missing/unknown -> ``UNKNOWN``)
``attributes``         allowlisted numeric/opaque attributes only
=====================  =====================================================

Privacy: ``auth_source_id`` / ``auth_account_id`` are *only* used to produce
distinct counts.  Any identifier supplied by a collector is SHA-256 hashed and
truncated before it reaches the feature layer; raw IPs, usernames or tokens are
never stored in the event attributes by this adapter and are never logged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import hashlib
import math

from ..preprocessing import _window_start
from ..schemas import EventLevel, EventType, StructuredEvent

#: The only canonical envelope version this adapter understands.
LAD_EVENT_SCHEMA_VERSION = "1.0"

#: Quality reasons that mean the event was *entirely* rejected.
REJECTED_REASONS = frozenset({
    "invalid_envelope",
    "unsupported_schema_version",
    "missing_event_id",
    "missing_host",
    "missing_type",
    "invalid_timestamp",
    "naive_timestamp",
    "duplicate_event_id",
})

#: Quality reasons that mean the event was accepted but a field was degraded.
PARTIAL_REASONS = frozenset({
    "missing_observed_ts",
    "missing_service",
    "unknown_category",
    "unknown_severity",
    "invalid_attributes",
    "invalid_status_code",
    "invalid_latency",
    "unmapped_auth_type",
    "status_code_derived_from_type",
    "out_of_order_event",
})

_KNOWN_CATEGORIES = frozenset({
    "http", "auth", "db", "service", "network", "syslog", "system",
})

_SEVERITY_MAP = {
    "info": EventLevel.INFO.value,
    "warning": EventLevel.WARNING.value,
    "warn": EventLevel.WARNING.value,
    "error": EventLevel.ERROR.value,
    "critical": EventLevel.CRITICAL.value,
    "fatal": EventLevel.FATAL.value,
}


@dataclass(frozen=True)
class RejectedLadEvent:
    """One envelope that could not enter the ML event stream."""

    event_id: Optional[str]
    reason: str
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {"event_id": self.event_id, "reason": self.reason, "detail": self.detail}


@dataclass(frozen=True)
class QualityRecord:
    """A non-fatal data-quality observation tied to an event/window.

    ``window_start`` is present only when the event had a valid UTC timestamp
    and host; otherwise the observation is batch-level.  No message text is
    retained.
    """

    reason: str
    host_id: Optional[str] = None
    service: Optional[str] = None
    window_start: Optional[datetime] = None
    event_id: Optional[str] = None

    def bucket_key(self) -> Optional[Tuple[str, str, datetime]]:
        if self.window_start is None or not self.host_id:
            return None
        return (self.host_id, self.service or "unknown", self.window_start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "host_id": self.host_id,
            "service": self.service,
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "event_id": self.event_id,
        }


@dataclass
class LadParseResult:
    """Accepted events plus explicit quality bookkeeping."""

    events: List[StructuredEvent] = field(default_factory=list)
    rejections: List[RejectedLadEvent] = field(default_factory=list)
    quality: List[QualityRecord] = field(default_factory=list)
    duplicate_event_ids: List[str] = field(default_factory=list)
    out_of_order_count: int = 0

    @property
    def accepted_count(self) -> int:
        return len(self.events)

    @property
    def rejected_count(self) -> int:
        return len(self.rejections)

    def quality_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self.quality:
            counts[record.reason] = counts.get(record.reason, 0) + 1
        return counts

    def quality_by_window(self, window_size_minutes: int = 1) -> Dict[Tuple[str, str, datetime], Dict[str, int]]:
        """Count quality observations per ``(host, service, window_start)``."""
        if window_size_minutes <= 0:
            raise ValueError("window_size_minutes must be positive")
        seconds = window_size_minutes * 60
        out: Dict[Tuple[str, str, datetime], Dict[str, int]] = {}
        for record in self.quality:
            key = record.bucket_key()
            if key is None:
                continue
            # Normalise again so callers cannot pass a non-aligned bucket.
            aligned = _window_start(key[2], seconds)
            full = (key[0], key[1], aligned)
            bucket = out.setdefault(full, {})
            bucket[record.reason] = bucket.get(record.reason, 0) + 1
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted_count,
            "rejected": self.rejected_count,
            "duplicates": len(self.duplicate_event_ids),
            "out_of_order": self.out_of_order_count,
            "quality_counts": self.quality_counts(),
            "rejection_reasons": _count_reasons(self.rejections),
        }


@dataclass
class LadParseOptions:
    """Bounded, explicit parsing policy."""

    allowed_schema_version: str = LAD_EVENT_SCHEMA_VERSION
    max_events: int = 250_000
    max_rejections: int = 5_000
    reject_duplicates: bool = True


def parse_lad_events(
    raw_events: Iterable[Mapping[str, Any]],
    options: Optional[LadParseOptions] = None,
) -> LadParseResult:
    """Parse a bounded batch of canonical LAD envelopes.

    The function never raises for a malformed individual event; every problem
    becomes a :class:`RejectedLadEvent` or a :class:`QualityRecord`.  It raises
    only for a programming error (a non-iterable input or an invalid bound).
    """
    options = options or LadParseOptions()
    if options.max_events < 1:
        raise ValueError("max_events must be >= 1")

    result = LadParseResult()
    seen_ids = set()
    last_ts_by_stream: Dict[Tuple[str, str], datetime] = {}

    for index, raw in enumerate(raw_events):
        if index >= options.max_events:
            result.quality.append(QualityRecord(reason="batch_bounded"))
            break
        _parse_one(raw, index, options, result, seen_ids, last_ts_by_stream)

    return result


def _parse_one(
    raw: Any,
    index: int,
    options: LadParseOptions,
    result: LadParseResult,
    seen_ids: set,
    last_ts_by_stream: Dict[Tuple[str, str], datetime],
) -> None:
    if not isinstance(raw, Mapping):
        _reject(result, None, "invalid_envelope", f"item {index} is not a mapping")
        return

    schema_version = raw.get("schema_version")
    if schema_version != options.allowed_schema_version:
        _reject(
            result,
            _safe_id(raw.get("event_id")),
            "unsupported_schema_version",
            f"schema_version {schema_version!r} is not {options.allowed_schema_version!r}",
        )
        return

    event_id = raw.get("event_id")
    if not isinstance(event_id, str) or not event_id.strip():
        _reject(result, None, "missing_event_id", "event_id is required")
        return
    event_id = event_id.strip()

    if event_id in seen_ids:
        result.duplicate_event_ids.append(event_id)
        _reject(result, event_id, "duplicate_event_id", "event_id already seen in this batch")
        return
    seen_ids.add(event_id)

    host = raw.get("host")
    if not isinstance(host, str) or not host.strip():
        _reject(result, event_id, "missing_host", "host is required")
        return
    host = host.strip()

    event_ts, ts_reason = _parse_ts(raw.get("event_ts"))
    if ts_reason is not None:
        _reject(result, event_id, ts_reason, "event_ts must be a timezone-aware ISO-8601 timestamp")
        return
    assert event_ts is not None

    observed_ts, observed_reason = _parse_ts(raw.get("observed_ts"))
    quality: List[QualityRecord] = []
    if observed_reason is not None:
        observed_ts = event_ts
        quality.append(QualityRecord("missing_observed_ts", host, _service(raw), event_ts, event_id))

    category_raw = raw.get("category")
    if not isinstance(category_raw, str) or not category_raw.strip():
        _reject(result, event_id, "missing_type", "category is required")
        return
    category = category_raw.strip().lower()

    type_raw = raw.get("type")
    if not isinstance(type_raw, str) or not type_raw.strip():
        _reject(result, event_id, "missing_type", "type is required")
        return
    event_type_raw = type_raw.strip().lower()

    if category not in _KNOWN_CATEGORIES:
        quality.append(QualityRecord("unknown_category", host, _service(raw), event_ts, event_id))

    service = _service(raw)
    if not service:
        service = "unknown"
        quality.append(QualityRecord("missing_service", host, service, event_ts, event_id))

    severity_raw = raw.get("severity")
    level = _SEVERITY_MAP.get(str(severity_raw).lower()) if severity_raw is not None else None
    if level is None:
        level = EventLevel.UNKNOWN.value
        quality.append(QualityRecord("unknown_severity", host, service, event_ts, event_id))

    attributes_raw = raw.get("attributes")
    if attributes_raw is None:
        attributes_raw = {}
    if not isinstance(attributes_raw, Mapping):
        attributes_raw = {}
        quality.append(QualityRecord("invalid_attributes", host, service, event_ts, event_id))
    attributes = dict(attributes_raw)

    event_type, mapped_attrs, map_quality = map_event_type(
        category=category,
        event_type_raw=event_type_raw,
        attributes=attributes,
    )
    for reason in map_quality:
        quality.append(QualityRecord(reason, host, service, event_ts, event_id))

    # Out-of-order is tracked per (host, service) stream in *arrival* order.
    stream = (host, service)
    previous_max = last_ts_by_stream.get(stream)
    if previous_max is not None and event_ts < previous_max:
        result.out_of_order_count += 1
        quality.append(QualityRecord("out_of_order_event", host, service, event_ts, event_id))
    if previous_max is None or event_ts > previous_max:
        last_ts_by_stream[stream] = event_ts

    result.events.append(
        StructuredEvent(
            event_id=event_id,
            timestamp=event_ts,
            host_id=host,
            service=service,
            event_type=event_type,
            level=level,
            attributes=mapped_attrs,
            source_metadata={
                "schema_version": options.allowed_schema_version,
                "lad_category": category,
                "lad_type": event_type_raw,
                "lad_source": str(raw.get("source") or "unknown")[:64],
                "observed_ts": observed_ts.isoformat() if observed_ts else None,
            },
        )
    )
    result.quality.extend(quality)


def map_event_type(
    category: str,
    event_type_raw: str,
    attributes: Mapping[str, Any],
) -> Tuple[str, Dict[str, Any], List[str]]:
    """Map a canonical ``category``/``type`` pair onto the ML event vocabulary.

    Returns ``(event_type, safe_attributes, quality_reasons)``.  The attribute
    mapping keeps only fields with a documented feature use; everything else is
    dropped.  Opaque auth identifiers are hashed before being stored.
    """
    attrs: Dict[str, Any] = {}
    quality: List[str] = []

    status = _parse_status(attributes.get("status_code"))
    if status is None:
        if "status_code" in attributes and attributes.get("status_code") not in (None, ""):
            quality.append("invalid_status_code")
        derived = _status_from_type(event_type_raw)
        if derived is not None:
            status = derived
            quality.append("status_code_derived_from_type")
    elif not (100 <= status <= 599):
        quality.append("invalid_status_code")
        status = None
    if status is not None:
        attrs["status_code"] = status

    latency = _parse_latency(attributes)
    if latency is not None:
        attrs["latency_ms"] = latency
    elif any(key in attributes for key in ("latency_ms", "duration_ms", "duration")):
        quality.append("invalid_latency")

    for key in ("db_timeout", "db_connection_failure"):
        if key in attributes:
            attrs[key] = bool(attributes.get(key))

    source_id = attributes.get("auth_source_id", attributes.get("source_id"))
    account_id = attributes.get("auth_account_id", attributes.get("account_id"))
    if source_id not in (None, ""):
        attrs["auth_source_id"] = _pseudonymise(source_id, "source")
    if account_id not in (None, ""):
        attrs["auth_account_id"] = _pseudonymise(account_id, "account")

    if category == "http" or event_type_raw.startswith("http"):
        return EventType.HTTP_REQUEST.value, attrs, quality

    if category == "auth" or event_type_raw.startswith("auth") or "login" in event_type_raw:
        if any(token in event_type_raw for token in ("fail", "denied", "invalid", "reject")):
            return EventType.AUTH_FAILURE.value, attrs, quality
        if any(token in event_type_raw for token in ("success", "accept", "granted", "ok")):
            return EventType.AUTH_SUCCESS.value, attrs, quality
        quality.append("unmapped_auth_type")
        return EventType.OTHER.value, attrs, quality

    if category == "db" or event_type_raw.startswith("db"):
        if "timeout" in event_type_raw or "timed_out" in event_type_raw:
            attrs["db_timeout"] = True
        if ("connection" in event_type_raw or "connect" in event_type_raw) and any(
            token in event_type_raw for token in ("fail", "error", "refused", "lost")
        ):
            attrs["db_connection_failure"] = True
        return EventType.DATABASE_QUERY.value, attrs, quality

    if "restart" in event_type_raw:
        return EventType.SERVICE_RESTART.value, attrs, quality

    if category == "service":
        return EventType.SYSTEM_EVENT.value, attrs, quality
    if category in ("network", "system", "syslog"):
        return EventType.APPLICATION_LOG.value, attrs, quality
    return EventType.OTHER.value, attrs, quality


def quality_invalid_reasons() -> frozenset:
    """Quality reasons that count as invalid/degraded telemetry."""
    return PARTIAL_REASONS | REJECTED_REASONS


def _service(raw: Mapping[str, Any]) -> Optional[str]:
    value = raw.get("service")
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _parse_status(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _status_from_type(event_type_raw: str) -> Optional[int]:
    if "5xx" in event_type_raw or "server_error" in event_type_raw:
        return 500
    if "4xx" in event_type_raw or "client_error" in event_type_raw:
        return 400
    return None


def _parse_latency(attributes: Mapping[str, Any]) -> Optional[float]:
    for key in ("latency_ms", "duration_ms", "duration"):
        if key not in attributes:
            continue
        value = attributes.get(key)
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number < 0:
            return None
        return float(number)
    return None


def _pseudonymise(value: Any, salt: str) -> str:
    """Stable, non-reversible key for distinct counting.

    A fixed, public salt is used because this is pseudonymisation, not
    encryption; it prevents raw identifiers entering the feature table while
    keeping distinct counts stable within a deployment.
    """
    text = f"lad-network:{salt}:{str(value)[:512]}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _parse_ts(value: Any) -> Tuple[Optional[datetime], Optional[str]]:
    if not isinstance(value, str) or not value.strip():
        return None, "invalid_timestamp"
    text = value.strip()
    # ``Z`` is valid ISO-8601 but ``datetime.fromisoformat`` only accepts it
    # from Python 3.11; normalise explicitly for the 3.9 floor.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None, "invalid_timestamp"
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None, "naive_timestamp"
    return parsed.astimezone(timezone.utc), None


def _safe_id(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()[:128]
    return None


def _reject(result: LadParseResult, event_id: Optional[str], reason: str, detail: str) -> None:
    if len(result.rejections) < 5_000:
        result.rejections.append(RejectedLadEvent(event_id=event_id, reason=reason, detail=detail))
    result.quality.append(QualityRecord(reason=reason, event_id=event_id))


def _count_reasons(rejections: List[RejectedLadEvent]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rejection in rejections:
        counts[rejection.reason] = counts.get(rejection.reason, 0) + 1
    return counts
