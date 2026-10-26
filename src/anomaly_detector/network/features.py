"""Network-oriented window features derived from canonical LAD events.

The existing ``features-v1`` allowlist (13 columns) is deliberately left
untouched.  This module defines a *separate*, explicit schema
(``network-features-v1``) for network/service operational incidents.

Feature rules
-------------
* Counts are genuine counts and may be zero.
* Ratios/measurements are ``None`` when their denominator is absent; they are
  never replaced with a meaningful-looking zero.
* Every rolling feature uses **only prior windows of the same host/service**.
  The current window is never part of its own baseline.
* Invalid/duplicate/out-of-order telemetry is counted, not silently discarded.
* Labels, scenario names and run identifiers never enter the feature mapping.

The window boundaries are the same epoch-aligned, half-open ``[start, end)``
windows used by :mod:`anomaly_detector.preprocessing`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import statistics

from ..inference.contract import FeatureWindowRequest
from ..preprocessing import _window_start, create_feature_window
from ..schemas import FeatureWindow, StructuredEvent, WindowMetadata

#: Feature-schema identifier carried in inference requests/results/manifests.
NETWORK_FEATURE_SCHEMA_VERSION = "network-features-v1"

#: The explicit, ordered model-input allowlist.  Order is significant.
NETWORK_FEATURE_COLUMNS: List[str] = [
    "event_count",
    "error_count",
    "error_ratio",
    "error_eligible_count",
    "http_request_count",
    "http_5xx_count",
    "http_5xx_ratio",
    "http_4xx_count",
    "http_4xx_ratio",
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
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "max_latency_ms",
    "restart_count",
    "invalid_event_count",
    "duplicate_event_count",
    "out_of_order_event_count",
    "missing_service_count",
    "unknown_category_count",
    "telemetry_gap_seconds",
    "event_count_zscore",
    "http_request_count_zscore",
    "http_5xx_ratio_delta",
    "auth_failure_ratio_delta",
    "db_timeout_ratio_delta",
    "p95_latency_ratio",
]

_COUNT_FEATURES = frozenset({
    "event_count", "error_count", "error_eligible_count",
    "http_request_count", "http_5xx_count", "http_4xx_count",
    "http_status_eligible_count", "auth_failure_count", "auth_success_count",
    "auth_distinct_source_count", "auth_distinct_account_count",
    "db_event_count", "db_timeout_count", "db_connection_failure_count",
    "latency_sample_count", "restart_count", "invalid_event_count",
    "duplicate_event_count", "out_of_order_event_count",
    "missing_service_count", "unknown_category_count",
})


@dataclass
class NetworkLabel:
    """Evaluation-only ground truth.  Never a model input."""

    window_start: datetime
    window_end: datetime
    host_id: str
    service: str
    is_anomaly: bool
    scenario_name: str
    run_id: str = ""
    telemetry_issue: bool = False
    description: Optional[str] = None

    @property
    def key(self) -> Tuple[str, str, datetime]:
        return (self.host_id, self.service, self.window_start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "host_id": self.host_id,
            "service": self.service,
            "is_anomaly": bool(self.is_anomaly),
            "scenario_name": self.scenario_name,
            "run_id": self.run_id,
            "telemetry_issue": bool(self.telemetry_issue),
            "description": self.description,
        }


@dataclass
class NetworkWindow:
    """One network feature window plus safe provenance and quality counts."""

    metadata: WindowMetadata
    features: Dict[str, Optional[float]]
    quality: Dict[str, int] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[str, str, datetime]:
        return (
            self.metadata.host_id,
            self.metadata.service,
            self.metadata.window_start,
        )

    @property
    def sample_key(self) -> str:
        return (
            f"{self.metadata.host_id}|{self.metadata.service}|"
            f"{self.metadata.window_start.isoformat()}"
        )

    def feature_vector(self) -> List[Optional[float]]:
        """Values in the documented order (name lookup, never dict order)."""
        return [self.features.get(name) for name in NETWORK_FEATURE_COLUMNS]

    def to_request(self) -> FeatureWindowRequest:
        return FeatureWindowRequest(
            window_id=self.sample_key,
            window_start=self.metadata.window_start,
            window_end=self.metadata.window_end,
            features=dict(self.features),
            host_id=self.metadata.host_id,
            service_name=self.metadata.service,
            feature_schema_version=NETWORK_FEATURE_SCHEMA_VERSION,
            extra={
                "window_size_minutes": self.metadata.window_size_minutes,
                "network": {
                    "feature_schema_version": NETWORK_FEATURE_SCHEMA_VERSION,
                    "quality": dict(self.quality),
                },
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_key": self.sample_key,
            "window_start": self.metadata.window_start.isoformat(),
            "window_end": self.metadata.window_end.isoformat(),
            "host_id": self.metadata.host_id,
            "service": self.metadata.service,
            "window_size_minutes": self.metadata.window_size_minutes,
            "features": dict(self.features),
            "quality": dict(self.quality),
        }


def build_network_windows(
    events: Sequence[StructuredEvent],
    quality_by_window: Optional[Mapping[Tuple[str, str, datetime], Mapping[str, int]]] = None,
    window_size_minutes: int = 1,
    rolling_windows: int = 30,
    rolling_min_history: int = 5,
    window_grid: Optional[Mapping[Tuple[str, str], Sequence[datetime]]] = None,
) -> List[NetworkWindow]:
    """Aggregate events into network feature windows.

    ``window_grid`` can materialise windows that have no accepted events (for
    example because every event in that minute was malformed).  Such a window
    carries ``event_count == 0`` and is later reported as ``insufficient_data``
    rather than as normal.
    """
    if window_size_minutes <= 0:
        raise ValueError("window_size_minutes must be positive")
    if rolling_windows < 1:
        raise ValueError("rolling_windows must be >= 1")
    if rolling_min_history < 1:
        raise ValueError("rolling_min_history must be >= 1")

    seconds = window_size_minutes * 60
    buckets: Dict[Tuple[str, str], Dict[datetime, List[StructuredEvent]]] = {}
    for event in events:
        key = (event.host_id, event.service)
        start = _window_start(event.timestamp, seconds)
        buckets.setdefault(key, {}).setdefault(start, []).append(event)

    if window_grid:
        for key, starts in window_grid.items():
            bucket = buckets.setdefault(key, {})
            for start in starts:
                bucket.setdefault(_window_start(start, seconds), [])

    quality_by_window = quality_by_window or {}
    windows: List[NetworkWindow] = []
    for (host, service), by_start in buckets.items():
        for start in sorted(by_start):
            end = start + timedelta(seconds=seconds)
            metadata = WindowMetadata(
                window_start=start,
                window_end=end,
                host_id=host,
                service=service,
                window_size_minutes=window_size_minutes,
            )
            window_events = by_start[start]
            quality = dict(quality_by_window.get((host, service, start), {}))
            windows.append(
                _compute_features(metadata, window_events, quality, seconds)
            )

    windows.sort(key=lambda w: (w.metadata.host_id, w.metadata.service, w.metadata.window_start))
    _add_rolling_features(windows, rolling_windows, rolling_min_history)
    return windows


def join_windows_with_labels(
    windows: Sequence[NetworkWindow],
    labels: Sequence[NetworkLabel],
) -> Tuple[List[NetworkWindow], List[NetworkLabel], List[str]]:
    """Align windows with evaluation labels, materialising telemetry gaps.

    Returns ``(windows, labels, missing_label_keys)``.  A label whose window
    produced no accepted events still gets an empty window, so the downstream
    detector can explicitly return ``insufficient_data`` instead of treating
    the window as normal.  Windows without a label are excluded from evaluation
    and their keys are returned for reporting.
    """
    label_by_key = {label.key: label for label in labels}
    window_by_key = {window.key: window for window in windows}

    aligned_windows: List[NetworkWindow] = []
    aligned_labels: List[NetworkLabel] = []
    missing_labels: List[str] = []

    for window in windows:
        label = label_by_key.get(window.key)
        if label is None:
            missing_labels.append(window.sample_key)
            continue
        aligned_windows.append(window)
        aligned_labels.append(label)

    for label in labels:
        if label.key in window_by_key:
            continue
        aligned_windows.append(_empty_window(label))
        aligned_labels.append(label)

    order = sorted(
        range(len(aligned_windows)),
        key=lambda i: (
            aligned_windows[i].metadata.host_id,
            aligned_windows[i].metadata.service,
            aligned_windows[i].metadata.window_start,
        ),
    )
    return (
        [aligned_windows[i] for i in order],
        [aligned_labels[i] for i in order],
        missing_labels,
    )


def windows_to_dataframe(
    windows: Sequence[NetworkWindow],
) -> Tuple["Any", "Any"]:
    """Return ``(features, metadata)`` DataFrames with the stable schema order."""
    import pandas as pd

    feature_rows: List[Dict[str, Optional[float]]] = []
    metadata_rows: List[Dict[str, Any]] = []
    for window in windows:
        feature_rows.append({name: window.features.get(name) for name in NETWORK_FEATURE_COLUMNS})
        metadata_rows.append({
            "sample_key": window.sample_key,
            "window_start": window.metadata.window_start.isoformat(),
            "window_end": window.metadata.window_end.isoformat(),
            "host_id": window.metadata.host_id,
            "service": window.metadata.service,
            "window_size_minutes": window.metadata.window_size_minutes,
        })
    features = pd.DataFrame(feature_rows, columns=NETWORK_FEATURE_COLUMNS)
    metadata = pd.DataFrame(metadata_rows)
    return features, metadata


def labels_to_dataframe(labels: Sequence[NetworkLabel]) -> "Any":
    import pandas as pd

    rows = [label.to_dict() for label in labels]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["sample_key"] = [
            f"{label.host_id}|{label.service}|{label.window_start.isoformat()}"
            for label in labels
        ]
    return frame


def _compute_features(
    metadata: WindowMetadata,
    events: Sequence[StructuredEvent],
    quality: Mapping[str, int],
    window_seconds: int,
) -> NetworkWindow:
    base: FeatureWindow = create_feature_window(metadata, events)
    features: Dict[str, Optional[float]] = {
        "event_count": float(base.event_count),
        "error_count": float(base.error_count),
        "error_ratio": base.error_ratio,
        "error_eligible_count": float(base.error_eligible_count),
        "http_request_count": float(base.http_request_count),
        "http_5xx_count": float(base.http_5xx_count),
        "http_5xx_ratio": base.http_5xx_ratio,
        "http_4xx_count": 0.0,
        "http_4xx_ratio": None,
        "http_status_eligible_count": float(base.http_request_eligible_count),
        "auth_failure_count": float(base.auth_failure_count),
        "auth_success_count": 0.0,
        "auth_failure_ratio": None,
        "auth_distinct_source_count": 0.0,
        "auth_distinct_account_count": 0.0,
        "db_event_count": 0.0,
        "db_timeout_count": 0.0,
        "db_timeout_ratio": None,
        "db_connection_failure_count": 0.0,
        "db_connection_failure_ratio": None,
        "latency_sample_count": float(base.latency_sample_count),
        "mean_latency_ms": base.mean_latency_ms,
        "p50_latency_ms": None,
        "p95_latency_ms": base.p95_latency_ms,
        "p99_latency_ms": None,
        "max_latency_ms": None,
        "restart_count": float(base.restart_count),
        "invalid_event_count": float(sum(quality.values())),
        "duplicate_event_count": float(quality.get("duplicate_event_id", 0)),
        "out_of_order_event_count": float(quality.get("out_of_order_event", 0)),
        "missing_service_count": float(quality.get("missing_service", 0)),
        "unknown_category_count": float(quality.get("unknown_category", 0)),
        "telemetry_gap_seconds": None,
        "event_count_zscore": None,
        "http_request_count_zscore": None,
        "http_5xx_ratio_delta": None,
        "auth_failure_ratio_delta": None,
        "db_timeout_ratio_delta": None,
        "p95_latency_ratio": None,
    }

    if not events:
        return NetworkWindow(metadata=metadata, features=features, quality=dict(quality))

    http_4xx = 0
    auth_success = 0
    auth_sources = set()
    auth_accounts = set()
    db_events = 0
    db_timeouts = 0
    db_conn_failures = 0
    latencies: List[float] = []

    for event in events:
        attrs = event.attributes
        status = attrs.get("status_code")
        if event.event_type == "http_request" and isinstance(status, int) and not isinstance(status, bool):
            if 400 <= status < 500:
                http_4xx += 1

        if event.event_type == "auth_success":
            auth_success += 1
        if event.event_type in ("auth_success", "auth_failure"):
            source = attrs.get("auth_source_id")
            account = attrs.get("auth_account_id")
            if isinstance(source, str) and source:
                auth_sources.add(source)
            if isinstance(account, str) and account:
                auth_accounts.add(account)

        if event.event_type == "database_query":
            db_events += 1
            if bool(attrs.get("db_timeout")):
                db_timeouts += 1
            if bool(attrs.get("db_connection_failure")):
                db_conn_failures += 1

        value = attrs.get("latency_ms")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            latencies.append(float(value))

    eligible_http = int(features["http_status_eligible_count"] or 0)
    features["http_4xx_count"] = float(http_4xx)
    features["http_4xx_ratio"] = (http_4xx / eligible_http) if eligible_http else None
    features["auth_success_count"] = float(auth_success)
    auth_total = int(base.auth_failure_count) + auth_success
    features["auth_failure_ratio"] = (base.auth_failure_count / auth_total) if auth_total else None
    features["auth_distinct_source_count"] = float(len(auth_sources))
    features["auth_distinct_account_count"] = float(len(auth_accounts))
    features["db_event_count"] = float(db_events)
    features["db_timeout_count"] = float(db_timeouts)
    features["db_timeout_ratio"] = (db_timeouts / db_events) if db_events else None
    features["db_connection_failure_count"] = float(db_conn_failures)
    features["db_connection_failure_ratio"] = (db_conn_failures / db_events) if db_events else None

    if latencies:
        features["p50_latency_ms"] = _percentile(latencies, 50.0)
        features["p95_latency_ms"] = _percentile(latencies, 95.0)
        features["p99_latency_ms"] = _percentile(latencies, 99.0)
        features["max_latency_ms"] = float(max(latencies))
        features["mean_latency_ms"] = float(sum(latencies) / len(latencies))

    return NetworkWindow(metadata=metadata, features=features, quality=dict(quality))


def _add_rolling_features(
    windows: Sequence[NetworkWindow],
    rolling_windows: int,
    min_history: int,
) -> None:
    """Add causal rolling features in place.

    Windows must already be sorted by ``(host, service, start)``.  For each
    stream only earlier rows are consulted; this is enforced by construction.
    """
    history: Dict[Tuple[str, str], List[NetworkWindow]] = {}
    previous_start: Dict[Tuple[str, str], datetime] = {}

    for window in windows:
        stream = (window.metadata.host_id, window.metadata.service)
        prior = history.setdefault(stream, [])
        previous = prior[-rolling_windows:] if prior else []
        current_start = window.metadata.window_start

        if encountered := previous_start.get(stream):
            gap = (current_start - encountered).total_seconds() - (
                window.metadata.window_size_minutes * 60
            )
            window.features["telemetry_gap_seconds"] = max(0.0, float(gap))
        previous_start[stream] = current_start

        if len(previous) >= min_history:
            current_event_count = _number(window.features.get("event_count"))
            current_http_count = _number(window.features.get("http_request_count"))
            window.features["event_count_zscore"] = _robust_zscore(
                current_event_count,
                [_number(w.features.get("event_count")) for w in previous],
            )
            window.features["http_request_count_zscore"] = _robust_zscore(
                current_http_count,
                [_number(w.features.get("http_request_count")) for w in previous],
            )
            window.features["http_5xx_ratio_delta"] = _ratio_delta(
                window.features.get("http_5xx_ratio"),
                [w.features.get("http_5xx_ratio") for w in previous],
            )
            window.features["auth_failure_ratio_delta"] = _ratio_delta(
                window.features.get("auth_failure_ratio"),
                [w.features.get("auth_failure_ratio") for w in previous],
            )
            window.features["db_timeout_ratio_delta"] = _ratio_delta(
                window.features.get("db_timeout_ratio"),
                [w.features.get("db_timeout_ratio") for w in previous],
            )
            window.features["p95_latency_ratio"] = _ratio_to_median(
                window.features.get("p95_latency_ms"),
                [w.features.get("p95_latency_ms") for w in previous],
            )

        prior.append(window)


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _robust_zscore(current: Optional[float], history: Sequence[Any]) -> Optional[float]:
    if current is None:
        return None
    values = [v for v in ( _number(x) for x in history) if v is not None]
    if len(values) < 2:
        return None
    centre = statistics.median(values)
    deviations = [abs(v - centre) for v in values]
    mad = statistics.median(deviations)
    scale = 1.4826 * mad
    if scale <= 0:
        mean = statistics.fmean(values)
        variance = statistics.fmean([(v - mean) ** 2 for v in values])
        scale = math.sqrt(variance)
    if scale <= 0:
        if current == centre:
            return 0.0
        return 10.0 if current > centre else -10.0
    return _clip((current - centre) / scale, -10.0, 10.0)


def _ratio_delta(current: Optional[float], history: Sequence[Any]) -> Optional[float]:
    current_value = _number(current)
    if current_value is None:
        return None
    values = [v for v in (_number(x) for x in history) if v is not None]
    if not values:
        return None
    return current_value - statistics.median(values)


def _ratio_to_median(current: Optional[float], history: Sequence[Any]) -> Optional[float]:
    current_value = _number(current)
    if current_value is None:
        return None
    values = [v for v in (_number(x) for x in history) if v is not None and v > 0]
    if not values:
        return None
    median = statistics.median(values)
    if median <= 0:
        return None
    return _clip(current_value / median, 0.0, 50.0)


def _empty_window(label: NetworkLabel) -> NetworkWindow:
    metadata = WindowMetadata(
        window_start=label.window_start,
        window_end=label.window_end,
        host_id=label.host_id,
        service=label.service,
        window_size_minutes=max(
            1,
            int(round((label.window_end - label.window_start).total_seconds() / 60.0)),
        ),
    )
    features: Dict[str, Optional[float]] = {
        name: (0.0 if name in _COUNT_FEATURES else None)
        for name in NETWORK_FEATURE_COLUMNS
    }
    features["event_count"] = 0.0
    features["invalid_event_count"] = 0.0
    return NetworkWindow(metadata=metadata, features=features, quality={})


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def align_window_start(value: datetime, window_size_minutes: int) -> datetime:
    """Public, testable wrapper around the epoch-aligned window rule."""
    if window_size_minutes <= 0:
        raise ValueError("window_size_minutes must be positive")
    return _window_start(value, window_size_minutes * 60)
