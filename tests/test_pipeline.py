"""Test the end-to-end pipeline wiring (validation -> windows -> exports)."""
from anomaly_detector.pipeline import run_pipeline


def _make(event_id, **over):
    d = {
        "event_id": event_id,
        "timestamp": "2026-10-18T10:00:12Z",
        "host_id": "web-01",
        "service": "web-api",
        "event_type": "http_request",
        "level": "INFO",
        "attributes": {"status_code": 200, "latency_ms": 50},
    }
    d.update(over)
    return d


def test_pipeline_counts_accepted_rejected_duplicated_and_partial():
    events = [
        _make("e1"),                                 # accepted
        _make("e2", attributes={"status_code": 200, "latency_ms": -5}),  # partial
        _make("e1"),                                 # duplicate
        {k: v for k, v in _make("e3").items() if k != "timestamp"},  # rejected
        _make("e4", host_id=""),                     # rejected
    ]
    result = run_pipeline(events, window_size_minutes=1)
    r = result.report
    assert r["input"]["total_processed"] == 5
    assert r["input"]["accepted_count"] == 2
    assert r["input"]["rejected_count"] == 2
    assert r["input"]["duplicated_count"] == 1
    assert r["input"]["partially_usable_count"] == 1


def test_pipeline_raw_field_missingness_measured():
    events = [
        _make("e1"),
        {k: v for k, v in _make("e2").items() if k != "level"},      # missing level
        {k: v for k, v in _make("e3").items() if k != "event_id"},   # missing event_id
    ]
    result = run_pipeline(events, window_size_minutes=1)
    miss = result.report["missingness_by_field"]
    assert miss["level"]["missing"] == 1
    assert miss["event_id"]["missing"] == 1
    assert miss["timestamp"]["missing"] == 0


def test_pipeline_feature_matrix_excludes_metadata_and_labels():
    events = [_make("e1"), _make("e2")]
    result = run_pipeline(events, window_size_minutes=1)
    forbidden = {"host_id", "service", "window_start", "window_end",
                 "is_anomaly", "scenario_name", "window_size_minutes"}
    assert forbidden.isdisjoint(set(result.feature_df.columns))
    assert "host_id" in result.metadata_df.columns
