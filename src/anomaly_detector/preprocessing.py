"""Preprocessing: aggregate structured events into per-host/service time windows.

Window semantics
----------------
* Windows are aligned to Unix-epoch boundaries: a window starts at the largest
  multiple of the window size (in seconds) that is ``<=`` the event timestamp.
  This makes boundaries reproducible regardless of the first event's clock time.
* A window is the half-open interval ``[window_start, window_end)``.
* A window is **materialized** only when it contains at least one accepted
  event, unless ``allow_empty_windows=True``, in which case every window between
  the first and last observed window for a host/service is emitted.
* Empty windows are *not* the same as absent telemetry. When materialized they
  carry ``event_count == 0`` and all measurements default to missing (``None``),
  never to a meaningful-looking zero.
* The offline pipeline has no wall-clock concept of "late", so all events are
  processed; events are simply assigned to the window their timestamp falls in.
  Out-of-order input is handled by sorting per host/service before bucketing.
* Events missing ``host_id``/``service`` never reach this stage; they are
  rejected during validation.
"""
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from collections import defaultdict
import logging

from .schemas import StructuredEvent, FeatureWindow, WindowMetadata

logger = logging.getLogger(__name__)

ERROR_LEVELS = {"ERROR", "FATAL", "CRITICAL"}
UNKNOWN_LEVEL = "UNKNOWN"


def _window_start(ts: datetime, window_seconds: int) -> datetime:
    """Return the epoch-aligned start of the window containing ``ts``."""
    epoch_seconds = int(ts.timestamp())
    aligned = (epoch_seconds // window_seconds) * window_seconds
    return datetime.fromtimestamp(aligned, tz=timezone.utc)


def aggregate_events_to_windows(events: List[StructuredEvent],
                                window_size_minutes: int = 1,
                                allow_empty_windows: bool = False) -> List[FeatureWindow]:
    """Aggregate events into time windows per (host_id, service) pair.

    Parameters
    ----------
    events:
        Accepted, normalized events.
    window_size_minutes:
        Window size in minutes. Must be a positive integer.
    allow_empty_windows:
        When True, materialize windows without observations between the first
        and last observed window for each host/service pair. Documented as a
        distinct behaviour from absent telemetry.
    """
    if window_size_minutes <= 0:
        raise ValueError("window_size_minutes must be a positive integer")

    if not events:
        return []

    window_seconds = window_size_minutes * 60

    # Group by host/service and by aligned window start.
    buckets: Dict[tuple, Dict[datetime, List[StructuredEvent]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for ev in events:
        key = (ev.host_id, ev.service)
        start = _window_start(ev.timestamp, window_seconds)
        buckets[key][start].append(ev)

    windows: List[FeatureWindow] = []
    for (host_id, service), by_start in buckets.items():
        observed_starts = sorted(by_start.keys())

        starts_to_emit = list(observed_starts)
        if allow_empty_windows and len(observed_starts) > 1:
            starts_to_emit = []
            cursor = observed_starts[0]
            last = observed_starts[-1]
            delta = timedelta(seconds=window_seconds)
            while cursor <= last:
                starts_to_emit.append(cursor)
                cursor = cursor + delta

        for start in starts_to_emit:
            end = start + timedelta(seconds=window_seconds)
            metadata = WindowMetadata(
                window_start=start,
                window_end=end,
                host_id=host_id,
                service=service,
                window_size_minutes=window_size_minutes,
            )
            windows.append(create_feature_window(metadata, by_start.get(start, [])))

    windows.sort(key=lambda w: (w.metadata.host_id, w.metadata.service, w.metadata.window_start))
    return windows


def create_feature_window(metadata: WindowMetadata,
                          events: List[StructuredEvent]) -> FeatureWindow:
    """Compute the documented feature set for a single window."""
    fw = FeatureWindow(metadata=metadata)
    fw.event_count = len(events)

    if not events:
        # Empty window: distinguish from absent telemetry. All measurements
        # stay missing (None); counts are genuine zeros.
        return fw

    # --- error count / ratio -------------------------------------------------
    # Only events with a *known* severity are eligible for error classification.
    eligible_for_error = [e for e in events if e.level != UNKNOWN_LEVEL]
    fw.error_count = sum(1 for e in eligible_for_error if e.level in ERROR_LEVELS)
    fw.error_eligible_count = len(eligible_for_error)
    fw.error_ratio = (fw.error_count / fw.error_eligible_count) if fw.error_eligible_count > 0 else None

    # --- HTTP requests ------------------------------------------------------
    http_events = [e for e in events if _is_http_request(e)]
    fw.http_request_count = len(http_events)

    # --- HTTP 5xx -----------------------------------------------------------
    # Only requests carrying a status code are eligible for 5xx classification.
    eligible_http = [e for e in http_events if _status_code(e) is not None]
    fw.http_request_eligible_count = len(eligible_http)
    fw.http_5xx_count = sum(1 for e in eligible_http if 500 <= _status_code(e) < 600)
    fw.http_5xx_ratio = (fw.http_5xx_count / fw.http_request_eligible_count) \
        if fw.http_request_eligible_count > 0 else None

    # --- auth failures / restarts ------------------------------------------
    fw.auth_failure_count = sum(1 for e in events if e.event_type == "auth_failure")
    fw.restart_count = sum(1 for e in events if e.event_type == "service_restart")

    # --- latency ------------------------------------------------------------
    latencies = [e.attributes["latency_ms"] for e in events
                 if isinstance(e.attributes.get("latency_ms"), (int, float))]
    fw.latency_sample_count = len(latencies)
    if latencies:
        fw.mean_latency_ms = sum(latencies) / len(latencies)
        fw.p95_latency_ms = _percentile(latencies, 95.0)
    else:
        fw.mean_latency_ms = None
        fw.p95_latency_ms = None

    return fw


def _is_http_request(event: StructuredEvent) -> bool:
    return event.event_type == "http_request"


def _status_code(event: StructuredEvent) -> Optional[int]:
    sc = event.attributes.get("status_code")
    if isinstance(sc, bool):  # bool is a subclass of int; reject it
        return None
    return sc if isinstance(sc, int) else None


def _percentile(values: List[float], percentile: float) -> float:
    """Linear-interpolation percentile (equivalent to numpy default)."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    frac = rank - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * frac)
