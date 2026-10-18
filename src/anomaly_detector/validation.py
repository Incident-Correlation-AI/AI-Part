"""Validation and normalization of structured events."""
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
import logging

from .schemas import StructuredEvent, EventLevel, EventType

logger = logging.getLogger(__name__)

class ValidationResult:
    def __init__(self):
        self.accepted: List[StructuredEvent] = []
        self.rejected: List[Dict[str, Any]] = []
        self.duplicated: List[Dict[str, Any]] = []
        self.partially_usable: List[Dict[str, Any]] = []
        self.total_processed = 0

def normalize_level(level: Any) -> str:
    if level is None:
        return "INFO"
    level_str = str(level).strip().upper()
    # Map common variations
    if level_str in ("WARN", "WARNING"):
        return "WARNING"
    if level_str in ("CRIT", "CRITICAL", "FATAL"):
        return "ERROR"  # Normalize to common levels
    if level_str in ("ERR", "ERROR"):
        return "ERROR"
    if level_str in ("INFO", "INFORMATION"):
        return "INFO"
    if level_str in ("DEBUG", "TRACE", "VERBOSE"):
        return "DEBUG"
    return level_str  # Keep as-is but document

def normalize_event_type(event_type: Any) -> str:
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
    return mapping.get(et, et)

def parse_timestamp(ts: Any) -> Tuple[bool, datetime, str]:
    """Parse timestamp to UTC. Returns (success, dt_utc, reason)."""
    if ts is None:
        return False, None, "Missing timestamp"
    
    # If already datetime
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            # Assume UTC if naive
            return True, ts.replace(tzinfo=timezone.utc), ""
        else:
            return True, ts.astimezone(timezone.utc), ""
    
    ts_str = str(ts).strip()
    if not ts_str:
        return False, None, "Empty timestamp"
    
    # Common formats
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
            dt = datetime.strptime(ts_str.replace('+00:00', ''), fmt)
            if fmt.endswith('%z') or 'Z' in ts_str or '+' in ts_str:
                dt_utc = dt.astimezone(timezone.utc)
            else:
                dt_utc = dt.replace(tzinfo=timezone.utc)
            return True, dt_utc, ""
        except Exception:
            continue
    
    # Try fromisoformat
    try:
        if 'Z' in ts_str:
            ts_str_clean = ts_str.replace('Z', '+00:00')
        dt = datetime.fromisoformat(ts_str_clean) if 'Z' in ts_str else datetime.fromisoformat(ts_str)
        dt_utc = dt.astimezone(timezone.utc)
        return True, dt_utc, ""
    except Exception:
        pass
    
    return False, None, f"Unparseable timestamp format: {ts_str[:50]}"

def validate_and_normalize_events(events: List[Dict[str, Any]], 
                                  strict: bool = True) -> ValidationResult:
    """Validate and normalize a list of events."""
    result = ValidationResult()
    seen_event_ids = set()
    
    for i, ev in enumerate(events):
        result.total_processed += 1
        reasons = []
        warnings = []
        
        # Copy event
        ev_norm = dict(ev) if ev else {}
        
        # Parse timestamp
        ts_ok, ts_dt, ts_reason = parse_timestamp(ev_norm.get("timestamp"))
        if not ts_ok:
            reasons.append(ts_reason)
        else:
            ev_norm["timestamp"] = ts_dt
        
        # Validate required fields
        host_id = ev_norm.get("host_id")
        if host_id is None or str(host_id).strip() == "":
            reasons.append("Missing or empty host_id")
        else:
            ev_norm["host_id"] = str(host_id).strip()
        
        service = ev_norm.get("service")
        if service is None or str(service).strip() == "":
            reasons.append("Missing or empty service")
        else:
            ev_norm["service"] = str(service).strip()
        
        # Normalize event_type and level
        ev_norm["event_type"] = normalize_event_type(ev_norm.get("event_type"))
        ev_norm["level"] = normalize_level(ev_norm.get("level"))
        
        # Validate attributes
        attrs = ev_norm.get("attributes", {})
        if attrs is None:
            ev_norm["attributes"] = {}
        elif not isinstance(attrs, dict):
            reasons.append(f"Invalid attributes type: {type(attrs)}")
            ev_norm["attributes"] = {}
        
        # Check for negative latency
        latency = ev_norm.get("attributes", {}).get("latency_ms") if ev_norm.get("attributes") else None
        if latency is not None:
            try:
                lat_val = float(latency)
                if lat_val < 0:
                    reasons.append(f"Invalid negative latency: {lat_val}")
            except Exception:
                reasons.append(f"Invalid latency type: {latency}")
        
        # Check status code validity if present
        status = ev_norm.get("attributes", {}).get("status_code") if ev_norm.get("attributes") else None
        if status is not None:
            try:
                sc = int(status)
                if sc < 100 or sc > 599:  # HTTP status codes range
                    warnings.append(f"Unusual status code: {sc}")
            except Exception:
                warnings.append(f"Invalid status code type: {status}")
        
        event_id = ev_norm.get("event_id")
        if event_id is not None and str(event_id) in seen_event_ids:
            result.duplicated.append({
                "index": i,
                "event": ev_norm,
                "reason": "Duplicate event_id"
            })
            continue
        elif event_id is not None:
            seen_event_ids.add(str(event_id))
        
        if reasons:
            result.rejected.append({
                "index": i,
                "event": ev,
                "normalized": ev_norm,
                "reasons": reasons,
                "warnings": warnings
            })
            if not strict and len(reasons) == 0:  # shouldn't happen
                pass
        else:
            structured = StructuredEvent(
                event_id=str(event_id) if event_id else f"evt-{i:06d}",
                timestamp=ev_norm["timestamp"],
                host_id=ev_norm["host_id"],
                service=ev_norm["service"],
                event_type=ev_norm["event_type"],
                level=ev_norm["level"],
                attributes=ev_norm.get("attributes", {}),
                source_metadata=ev_norm.get("source_metadata", {})
            )
            result.accepted.append(structured)
        
    return result
