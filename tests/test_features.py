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

def test_auth_failure_and_restart_count():
    from datetime import datetime, timezone, timedelta
    from anomaly_detector.schemas import StructuredEvent
    base = datetime(2026, 10, 18, 10, 0, 0, tzinfo=timezone.utc)
    events = [
        StructuredEvent("a1", base, "web-01", "web-api", "auth_failure", "ERROR", {}),
        StructuredEvent("a2", base.replace(second=10), "web-01", "web-api", "auth_failure", "ERROR", {}),
        StructuredEvent("r1", base.replace(second=20), "web-01", "web-api", "service_restart", "WARNING", {}),
    ]
    windows = aggregate_events_to_windows(events, window_size_minutes=1)
    assert windows[0].auth_failure_count == 2
    assert windows[0].restart_count == 1

def test_p95_latency():
    from datetime import datetime, timezone, timedelta
    from anomaly_detector.schemas import StructuredEvent
    base = datetime(2026, 10, 18, 10, 0, 0, tzinfo=timezone.utc)
    events = []
    # 100 samples: 1-100ms
    for i in range(100):
        events.append(StructuredEvent(f"l{i}", base.replace(second=i % 60), "web-01", "web-api", "http_request", "INFO", {"latency_ms": i + 1}))
    windows = aggregate_events_to_windows(events, window_size_minutes=1)
    w = windows[0]
    assert w.latency_sample_count == 100
    assert w.mean_latency_ms == 50.5
    # p95 should be around 95-96
    assert 90 <= w.p95_latency_ms <= 100
