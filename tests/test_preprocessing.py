"""Test empty and low-volume windows, and window boundary cases."""
import pytest
from datetime import datetime, timedelta, timezone
from anomaly_detector.schemas import StructuredEvent
from anomaly_detector.preprocessing import aggregate_events_to_windows

def test_empty_events_produce_no_windows():
    windows = aggregate_events_to_windows([], window_size_minutes=1)
    assert windows == []

def test_single_event_produces_window():
    ts = datetime(2026, 10, 18, 10, 0, 30, tzinfo=timezone.utc)
    events = [StructuredEvent("e1", ts, "web-01", "web-api", "http_request", "INFO", {})]
    windows = aggregate_events_to_windows(events, window_size_minutes=1)
    assert len(windows) == 1
    assert windows[0].event_count == 1

def test_window_boundary_inclusive_start_exclusive_end():
    """Events at boundary: start inclusive, end exclusive."""
    base = datetime(2026, 10, 18, 10, 0, 0, tzinfo=timezone.utc)
    events = [
        StructuredEvent("e0", base, "web-01", "web-api", "http_request", "INFO", {}),  # 10:00:00 -> window 1
        StructuredEvent("e1", base + timedelta(seconds=59), "web-01", "web-api", "http_request", "INFO", {}),  # 10:00:59 -> window 1
        StructuredEvent("e2", base + timedelta(seconds=60), "web-01", "web-api", "http_request", "INFO", {}),  # 10:01:00 -> window 2
    ]
    windows = aggregate_events_to_windows(events, window_size_minutes=1)
    assert len(windows) >= 2
    # First window should have 2 events (at 10:00:00 and 10:00:59)
    assert windows[0].event_count == 2
    # Second window should have 1 event (at 10:01:00)
    assert windows[1].event_count == 1

def test_low_volume_window():
    """Single event in window is valid."""
    ts = datetime(2026, 10, 18, 10, 0, 0, tzinfo=timezone.utc)
    events = [StructuredEvent("e1", ts, "db", "postgres", "database_query", "INFO", {})]
    windows = aggregate_events_to_windows(events, window_size_minutes=1)
    assert len(windows) == 1
    assert windows[0].event_count == 1
    assert windows[0].http_request_count == 0  # Not HTTP
    assert windows[0].http_5xx_ratio is None  # undefined denominator

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
    events = [
        StructuredEvent("e2", base + timedelta(seconds=30), "web-01", "web-api", "http_request", "INFO", {}),
        StructuredEvent("e0", base + timedelta(seconds=0), "web-01", "web-api", "http_request", "INFO", {}),
        StructuredEvent("e1", base + timedelta(seconds=10), "web-01", "web-api", "http_request", "INFO", {}),
    ]
    windows = aggregate_events_to_windows(events, window_size_minutes=1)
    assert len(windows) >= 1
    assert windows[0].event_count == 3
