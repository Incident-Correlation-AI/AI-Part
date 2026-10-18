"""Test feature generation."""
from datetime import datetime, timezone
from anomaly_detector.schemas import StructuredEvent
from anomaly_detector.preprocessing import aggregate_events_to_windows

def test_error_ratio_correct():
    base = datetime(2026, 10, 18, 10, 0, 0, tzinfo=timezone.utc)
    events = [
        StructuredEvent("e1", base, "web-01", "web-api", "http_request", "ERROR", {"status_code": 500}),
        StructuredEvent("e2", base.replace(second=10), "web-01", "web-api", "http_request", "INFO", {"status_code": 200}),
    ]
    windows = aggregate_events_to_windows(events, window_size_minutes=1)
    assert len(windows) == 1
    w = windows[0]
    assert w.error_count == 1
    assert w.error_ratio == 0.5

def test_http_5xx_ratio():
    base = datetime(2026, 10, 18, 10, 0, 0, tzinfo=timezone.utc)
    events = [
        StructuredEvent("e1", base, "web-01", "web-api", "http_request", "ERROR", {"status_code": 500, "latency_ms": 100}),
        StructuredEvent("e2", base.replace(second=5), "web-01", "web-api", "http_request", "ERROR", {"status_code": 502, "latency_ms": 150}),
        StructuredEvent("e3", base.replace(second=10), "web-01", "web-api", "http_request", "INFO", {"status_code": 200, "latency_ms": 50}),
    ]
    windows = aggregate_events_to_windows(events, window_size_minutes=1)
    w = windows[0]
    assert w.http_5xx_count == 2
    assert w.http_5xx_ratio == 2/3

def test_latency_stats():
    base = datetime(2026, 10, 18, 10, 0, 0, tzinfo=timezone.utc)
    events = []
    for i, lat in enumerate([10, 20, 30, 40, 50]):
        events.append(StructuredEvent(f"e{i}", base.replace(second=i*2), "web-01", "web-api", "http_request", "INFO", {"latency_ms": lat}))
    windows = aggregate_events_to_windows(events, window_size_minutes=1)
    w = windows[0]
    assert w.mean_latency_ms == 30
    assert w.latency_sample_count == 5

def test_ratio_with_zero_denominator():
    base = datetime(2026, 10, 18, 10, 0, 0, tzinfo=timezone.utc)
    events = [
        StructuredEvent("e1", base, "web-01", "db", "database_query", "INFO", {}),  # No HTTP, no latency
    ]
    windows = aggregate_events_to_windows(events, window_size_minutes=1)
    w = windows[0]
    assert w.http_5xx_ratio is None  # No eligible HTTP requests
    assert w.mean_latency_ms is None  # No valid latency samples
