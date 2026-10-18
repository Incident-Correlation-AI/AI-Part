"""Test validation and normalization."""
from anomaly_detector.validation import validate_and_normalize_events, parse_timestamp

def test_parse_timestamp_utc():
    ok, dt, reason = parse_timestamp("2026-10-18T10:00:12Z")
    assert ok
    assert dt.tzinfo is not None

def test_parse_timestamp_naive():
    ok, dt, reason = parse_timestamp("2026-10-18 10:00:12")
    assert ok
    assert dt.tzinfo is not None

def test_validate_valid_event():
    events = [{
        "event_id": "evt-001",
        "timestamp": "2026-10-18T10:00:12Z",
        "host_id": "web-01",
        "service": "web-api",
        "event_type": "http_request",
        "level": "INFO",
        "attributes": {"status_code": 200, "latency_ms": 50}
    }]
    res = validate_and_normalize_events(events)
    assert len(res.accepted) == 1
    assert len(res.rejected) == 0

def test_validate_missing_host():
    events = [{
        "timestamp": "2026-10-18T10:00:12Z",
        "service": "web-api",
        "event_type": "http_request"
    }]
    res = validate_and_normalize_events(events)
    assert len(res.accepted) == 0
    assert len(res.rejected) == 1

def test_validate_missing_timestamp():
    events = [{
        "host_id": "web-01",
        "service": "web-api",
        "event_type": "http_request"
    }]
    res = validate_and_normalize_events(events)
    assert len(res.rejected) == 1

def test_negative_latency_rejected_or_flagged():
    events = [{
        "timestamp": "2026-10-18T10:00:12Z",
        "host_id": "web-01",
        "service": "web-api",
        "event_type": "http_request",
        "attributes": {"latency_ms": -10}
    }]
    res = validate_and_normalize_events(events)
    assert len(res.rejected) == 1

def test_duplicate_handling():
    events = [
        {"event_id": "evt-001", "timestamp": "2026-10-18T10:00:12Z", "host_id": "web-01", "service": "web-api", "event_type": "http_request"},
        {"event_id": "evt-001", "timestamp": "2026-10-18T10:00:13Z", "host_id": "web-01", "service": "web-api", "event_type": "http_request"}
    ]
    res = validate_and_normalize_events(events)
    assert len(res.accepted) == 1
    assert len(res.duplicated) == 1
