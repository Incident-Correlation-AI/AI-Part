"""Test preprocessing and window aggregation."""
from datetime import datetime, timedelta, timezone
from anomaly_detector.schemas import StructuredEvent
from anomaly_detector.preprocessing import aggregate_events_to_windows

def test_window_aggregation_basic():
    base = datetime(2026, 10, 18, 10, 0, 0, tzinfo=timezone.utc)
    events = []
    for i in range(5):
        events.append(StructuredEvent(
            event_id=f"e{i}",
            timestamp=base + timedelta(seconds=i*10),
            host_id="web-01",
            service="web-api",
            event_type="http_request",
            level="INFO",
            attributes={"status_code": 200}
        ))
    windows = aggregate_events_to_windows(events, window_size_minutes=1)
    assert len(windows) >= 1
    assert windows[0].event_count == 5

def test_out_of_order_events():
    base = datetime(2026, 10, 18, 10, 0, 0, tzinfo=timezone.utc)
    events = []
    # Add out of order
    events.append(StructuredEvent("e2", base + timedelta(seconds=30), "web-01", "web-api", "http_request", "INFO", {}))
    events.append(StructuredEvent("e0", base + timedelta(seconds=0), "web-01", "web-api", "http_request", "INFO", {}))
    events.append(StructuredEvent("e1", base + timedelta(seconds=10), "web-01", "web-api", "http_request", "INFO", {}))
    windows = aggregate_events_to_windows(events, window_size_minutes=1)
    assert len(windows) >= 1
