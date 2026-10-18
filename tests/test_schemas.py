"""Test event schemas and data structures."""
import pytest
from datetime import datetime, timezone
from anomaly_detector.schemas import StructuredEvent, FeatureWindow, WindowMetadata, LabelInfo

def test_structured_event_creation():
    ts = datetime(2026, 10, 18, 10, 0, 0, tzinfo=timezone.utc)
    ev = StructuredEvent(
        event_id="evt-001",
        timestamp=ts,
        host_id="web-01",
        service="web-api",
        event_type="http_request",
        level="INFO",
        attributes={"status_code": 200, "latency_ms": 50}
    )
    assert ev.host_id == "web-01"
    assert ev.service == "web-api"
    d = ev.to_dict()
    assert d["timestamp"] == "2026-10-18T10:00:00+00:00"

def test_feature_window_separation():
    ts = datetime(2026, 10, 18, 10, 0, 0, tzinfo=timezone.utc)
    meta = WindowMetadata(ts, ts.replace(minute=1), "web-01", "web-api", 1)
    fw = FeatureWindow(meta, event_count=10, error_count=2, error_ratio=0.2)
    features = fw.to_features_dict()
    # Ensure metadata fields not in features
    assert "host_id" not in features
    assert "service" not in features
    assert "window_start" not in features
    assert "is_anomaly" not in features
    assert features["event_count"] == 10

def test_label_info():
    ts = datetime(2026, 10, 18, 10, 0, 0, tzinfo=timezone.utc)
    lbl = LabelInfo(ts, ts.replace(minute=1), "web-01", "web-api", True, "http_5xx_surge")
    d = lbl.to_dict()
    assert d["is_anomaly"] == True
    assert d["scenario_name"] == "http_5xx_surge"
