"""Validation and normalization of structured events.

Policy summary
--------------
* **Rejected** — the record cannot be salvaged as an event: missing/unparseable
  ``timestamp``, missing/empty ``host_id`` or ``service``, or a non-mapping
  ``attributes`` value.
* **Duplicated** — an ``event_id`` that has already been seen is dropped to avoid
  double counting. Deduplication only happens when a stable ``event_id`` exists.
* **Partially usable** — the event is accepted and counted, but one or more
  optional measurements were invalid and had to be dropped (for example a
  negative latency or a non-numeric status code). The event is *not* silently
  discarded and the invalid measurement is *not* coerced to zero.
* **Accepted** — everything else.
"""
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
import logging

from .schemas import StructuredEvent

logger = logging.getLogger(__name__)

ERROR_LEVELS = {"ERROR", "FATAL", "CRITICAL"}


class ValidationResult:
    """Container for the outcome of validating a batch of events."""

    def __init__(self) -> None:
        self.accepted: List[StructuredEvent] = []
        self.rejected: List[Dict[str, Any]] = []
        self.duplicated: List[Dict[str, Any]] = []
        self.partially_usable: List[Dict[str, Any]] = []
        self.total_processed = 0


def normalize_level(level: Any) -> str:
    """Normalize a severity string to a small documented vocabulary."""
    if level is None:
        return "INFO"
    level_str = str(level).strip().upper()
    if level_str in ("WARN", "WARNING"):
        return "WARNING"
    if level_str in ("CRIT", "CRITICAL", "FATAL", "ERR", "ERROR"):
        return "ERROR"
    if level_str in ("INFO", "INFORMATION"):
        return "INFO"
    if level_str in ("DEBUG", "TRACE", "VERBOSE"):
        return "DEBUG"
    return level_str or "UNKNOWN"


def normalize_event_type(event_type: Any) -> str:
    """Normalize an event-type string to a documented category."""
    if event_type is None:
        return "other"
    et = str(event_type).strip().lower()
    mapping = {
        "http": "http_request",
        "http_request": "http_request",
        "request": "http_request",
        "auth": "auth_failure",
        "auth_failure": "auth_failure",
        "authentication_failure": "auth_failure",
        "login_failure": "auth_failure",
        "restart": "service_restart",
        "service_restart": "service_restart",
        "system": "system_event",
        "db": "database_query",
        "database": "database_query",
        "app": "application_log",
    }
    return mapping.get(event_type_str := et, et)


def parse_timestamp(ts: Any) -> Tuple[bool, datetime, str]:
    """Parse a timestamp and normalize it to UTC.

    Returns ``(success, datetime_utc, reason)``. Naive timestamps are assumed to
    be UTC; aware timestamps are converted to UTC.
    """
    if ts is None:
        return False, None, "Missing timestamp"

    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return True, ts.replace(tzinfo=timezone.utc), ""
        return True, ts.astimezone(timezone.utc), ""

    ts_str = str(ts).strip()
    if not ts_str:
        return False, None, "Empty timestamp"

    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%z",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
        if dt.tzinfo is None:
            return True, dt.replace(tzinfo=timezone.utc), ""
        return True, dt.astimezone(timezone.utc), ""

    # Fallback to ISO parsing (handles offsets and fractional seconds).
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return True, dt.replace(tzinfo=timezone.utc), ""
        return True, dt.astimezone(timezone.utc), ""
    except ValueError:
        pass

    return False, None, f"Unparseable timestamp: {ts_str[:50]}"


def _clean_attributes(attributes: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Drop invalid optional measurements, returning (clean_attrs, warnings)."""
    warnings: List[str] = []
    clean: Dict[str, Any] = {}

    for key, value in attributes.items():
        if key == "latency_ms":
            try:
                lat = float(value)
            except (TypeError, ValueError):
                warnings.append(f"Dropped non-numeric latency_ms={value!r}")
                continue
            if lat < 0:
                warnings.append(f"Dropped negative latency_ms={lat}")
                continue
            clean[key] = lat
        elif key == "status_code":
            try:
                sc = int(value)
            except (TypeError, ValueError):
                warnings.append(f"Dropped non-integer status_code={value!r}")
                continue
            if sc < 100 or sc > 599:
                warnings.append(f"Dropped out-of-range status_code={sc}")
                continue
            clean[key] = sc
        else:
            clean[key] = value

    return clean, warnings


def validate_and_normalize_events(events: List[Dict[str, Any]]) -> ValidationResult:
    """Validate and normalize a list of raw event dictionaries."""
    result = ValidationResult()
    seen_event_ids = set()

    for i, ev in enumerate(events):
        result.total_processed += 1
        reasons: List[str] = []

        ev_norm = dict(ev) if isinstance(ev, dict) else {}

        # --- Required: timestamp ---
        ts_ok, ts_dt, ts_reason = parse_timestamp(ev_norm.get("timestamp"))
        if not ts_ok:
            reasons.append(ts_reason)

        # --- Required: host_id ---
        host_id = ev_norm.get("host_id")
        if host_id is None or str(host_id).strip() == "":
            reasons.append("Missing or empty host_id")

        # --- Required: service ---
        service = ev_norm.get("service")
        if service is None or str(service).strip() == "":
            reasons.append("Missing or empty service")

        # --- Optional/required-normalized fields ---
        ev_norm["event_type"] = normalize_event_type(ev_norm.get("event_type"))
        ev_norm["level"] = normalize_level(ev_norm.get("level"))

        # --- attributes must be a mapping when present ---
        raw_attrs = ev_norm.get("attributes")
        if raw_attrs is None:
            raw_attrs = {}
        if not isinstance(raw_attrs, dict):
            reasons.append(f"Invalid attributes type: {type(raw_attrs).__name__}")
            raw_attrs = {}

        clean_attrs, attr_warnings = _clean_attributes(raw_attrs)

        # Reject unusable records before dedup so counts are explicit.
        if reasons:
            result.rejected.append({
                "index": i,
                "event": ev,
                "normalized": ev_norm,
                "reasons": reasons,
                "warnings": attr_warnings,
            })
            continue

        # --- Deduplication on stable event_id only ---
        event_id = ev_norm.get("event_id")
        if event_id is not None and str(event_id) in seen_event_ids:
            result.duplicated.append({
                "index": i,
                "event_id": str(event_id),
                "reason": "Duplicate event_id",
            })
            continue
        if event_id is not None:
            seen_event_ids.add(str(event_id))

        ev_norm["timestamp"] = ts_dt
        ev_norm["host_id"] = str(host_id).strip()
        ev_norm["service"] = str(service).strip()
        ev_norm["attributes"] = clean_attrs

        structured = StructuredEvent(
            event_id=str(event_id) if event_id is not None else f"evt-{i:06d}",
            timestamp=ts_dt,
            host_id=ev_norm["host_id"],
            service=ev_norm["service"],
            event_type=ev_norm["event_type"],
            level=ev_norm["level"],
            attributes=clean_attrs,
            source_metadata=ev_norm.get("source_metadata", {}) or {},
        )
        result.accepted.append(structured)

        if attr_warnings:
            result.partially_usable.append({
                "index": i,
                "event_id": structured.event_id,
                "reasons": attr_warnings,
            })

    return result
