"""End-to-end scenario fixtures for Milestone 7.

Each scenario produces **raw event dictionaries** (the M1 input format), not
pre-aggregated features. The full path is therefore exercised:

    raw events -> validate_and_normalize_events -> aggregate_events_to_windows
              -> FeatureWindow -> M6 inference -> neutral downstream sink

Scenario ground truth lives in a separate structure and is used by the test
harness only. It never enters the model feature vector.

All generators are deterministic (no RNG, or a seeded RNG).
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

BASE = datetime(2026, 11, 2, 9, 0, tzinfo=timezone.utc)


@dataclass
class E2EScenario:
    """One end-to-end scenario: raw events plus the expected behaviour."""

    name: str
    description: str
    events: List[Dict[str, Any]]
    expected_behavior: str
    #: Windows the harness believes are anomalous (by window index).
    anomalous_windows: List[int] = field(default_factory=list)
    #: Windows that must be rejected or otherwise not scorable.
    unscorable_windows: List[int] = field(default_factory=list)
    #: True when the scenario represents a benign change, not an attack.
    benign: bool = False
    notes: List[str] = field(default_factory=list)


def _event(
    minute: int,
    *,
    host: str = "app-01",
    service: str = "checkout-api",
    level: str = "INFO",
    event_type: str = "http_request",
    status: Optional[int] = 200,
    latency: Optional[float] = 42.0,
    event_id: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {}
    if status is not None:
        attributes["status_code"] = status
    if latency is not None:
        attributes["latency_ms"] = latency
    return {
        "event_id": event_id,
        "timestamp": timestamp or (BASE + timedelta(minutes=minute)).isoformat(),
        "host_id": host,
        "service": service,
        "event_type": event_type,
        "level": level,
        "attributes": attributes,
    }


def _healthy_window(minute: int, n: int = 12) -> List[Dict[str, Any]]:
    """A normal minute of successful requests."""
    events = []
    for i in range(n):
        events.append(_event(minute, latency=40.0 + (i % 5) * 2.0, status=200))
    return events


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenario_healthy_traffic() -> E2EScenario:
    """Ordinary successful traffic across six one-minute windows."""
    events: List[Dict[str, Any]] = []
    for minute in range(6):
        events.extend(_healthy_window(minute))
    return E2EScenario(
        name="healthy_traffic",
        description="Six normal minutes of successful requests.",
        events=events,
        expected_behavior="scored, not flagged",
        benign=True,
    )


def scenario_http_5xx_spike() -> E2EScenario:
    """One minute where most requests return HTTP 5xx."""
    events: List[Dict[str, Any]] = []
    for minute in range(5):
        events.extend(_healthy_window(minute))
    spike = []
    for i in range(12):
        spike.append(_event(5, level="ERROR", status=503, latency=210.0))
    events.extend(spike)
    return E2EScenario(
        name="http_5xx_spike",
        description="One minute with 12 HTTP 503 responses and elevated latency.",
        events=events,
        expected_behavior="flagged as anomalous",
        anomalous_windows=[5],
    )


def scenario_latency_degradation() -> E2EScenario:
    """One minute with a sustained latency spike but normal status codes."""
    events: List[Dict[str, Any]] = []
    for minute in range(5):
        events.extend(_healthy_window(minute))
    slow = [_event(5, latency=1800.0 + i * 40.0, status=200) for i in range(12)]
    events.extend(slow)
    return E2EScenario(
        name="latency_degradation",
        description="One minute with sustained ~1.8-2.3 s latencies and HTTP 200.",
        events=events,
        expected_behavior="flagged as anomalous",
        anomalous_windows=[5],
    )


def scenario_database_timeout() -> E2EScenario:
    """A minute of database query timeouts (dependency failure)."""
    events: List[Dict[str, Any]] = []
    for minute in range(5):
        events.extend(_healthy_window(minute))
    db = [
        _event(5, level="ERROR", event_type="database_query", status=None, latency=5000.0)
        for _ in range(10)
    ]
    events.extend(db)
    return E2EScenario(
        name="database_timeout",
        description="One minute of ten database_query timeouts (no HTTP status).",
        events=events,
        expected_behavior="flagged as anomalous (or at minimum scored, not dropped)",
        anomalous_windows=[5],
        notes=[
            "database_query events carry no status_code and no http_request_count; "
            "the anomaly signal is the event count and the latency."
        ],
    )


def scenario_missing_latency() -> E2EScenario:
    """A minute where the latency telemetry is absent (collection gap)."""
    events: List[Dict[str, Any]] = []
    for minute in range(6):
        if minute == 3:
            # latency_ms is absent entirely -> missing, not zero
            events.extend(_event(minute, latency=None, status=200) for _ in range(12))
        else:
            events.extend(_healthy_window(minute))
    return E2EScenario(
        name="missing_latency",
        description="One minute with no latency measurements (telemetry collection gap).",
        events=events,
        expected_behavior="scored with imputation reported; NOT necessarily flagged",
        benign=True,
        notes=[
            "The missing value must be imputed with the training median and reported "
            "in data_quality.imputed_features - never silently treated as zero."
        ],
    )


def scenario_malformed_event() -> E2EScenario:
    """A malformed / out-of-range event mixed into otherwise normal traffic."""
    events: List[Dict[str, Any]] = []
    for minute in range(4):
        events.extend(_healthy_window(minute))
    # Malformed: no timestamp, empty host -> rejected by the M1 validator
    events.append({
        "event_id": "bad-1",
        "timestamp": None,
        "host_id": "",
        "service": "checkout-api",
        "event_type": "http_request",
        "level": "INFO",
        "attributes": {"status_code": 200, "latency_ms": 30.0},
    })
    # Out-of-range: negative latency -> invalid measurement, dropped not coerced
    events.append(_event(4, latency=-500.0, status=200))
    events.extend(_healthy_window(4))
    return E2EScenario(
        name="malformed_event",
        description="One rejected malformed event and one invalid (negative) latency.",
        events=events,
        expected_behavior="the malformed event is rejected, the window is still scored",
        benign=True,
        notes=[
            "The M1 validator must reject the malformed record and drop the invalid "
            "measurement rather than coercing it."
        ],
    )


def scenario_duplicate_and_late_events() -> E2EScenario:
    """Duplicate delivery and a late (out-of-order) event."""
    events: List[Dict[str, Any]] = []
    for minute in range(5):
        events.extend(_healthy_window(minute))
    original = _event(5, event_id="dup-1", latency=45.0, status=200)
    duplicate = dict(original)
    # A late delivery: the same event id arrives again with the same id
    events.append(original)
    events.append(duplicate)
    # A late event: an event whose timestamp belongs to an earlier window
    events.append(_event(5, event_id="late-1", latency=47.0, status=200,
                         timestamp=(BASE + timedelta(minutes=2, seconds=30)).isoformat()))
    events.extend(_healthy_window(5))
    return E2EScenario(
        name="duplicate_and_late_events",
        description="One duplicated event id and one event delivered late.",
        events=events,
        expected_behavior="the duplicate is de-duplicated; the late event lands in its own window",
        benign=True,
        notes=[
            "De-duplication follows the M1 policy (event_id based). A late event is "
            "assigned to the window its event time belongs to, not to its arrival time."
        ],
    )


def scenario_benign_distribution_shift() -> E2EScenario:
    """A legitimate step up in traffic volume and latency baseline."""
    events: List[Dict[str, Any]] = []
    for minute in range(4):
        events.extend(_healthy_window(minute, n=12))
    # A legitimate release: more traffic, higher but still plausible latency
    for minute in range(4, 8):
        events.extend([
            _event(minute, latency=95.0 + (i % 6) * 4.0, status=200) for i in range(26)
        ])
    return E2EScenario(
        name="benign_distribution_shift",
        description="A legitimate step change to ~26 events/minute and ~95 ms latency.",
        events=events,
        expected_behavior="scored; a high flag rate here is a false-positive risk, not a detection",
        benign=True,
        notes=[
            "M5 measured a 100% flag rate on a benign shift. This scenario exercises "
            "the same weakness through the real pipeline."
        ],
    )


def all_scenarios() -> List[E2EScenario]:
    """All end-to-end scenarios, in a stable order."""
    return [
        scenario_healthy_traffic(),
        scenario_http_5xx_spike(),
        scenario_latency_degradation(),
        scenario_database_timeout(),
        scenario_missing_latency(),
        scenario_malformed_event(),
        scenario_duplicate_and_late_events(),
        scenario_benign_distribution_shift(),
    ]
