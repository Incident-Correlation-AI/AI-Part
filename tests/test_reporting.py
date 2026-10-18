"""Test output schema stability and data quality report counts."""
from datetime import datetime, timezone
from anomaly_detector.synthetic_data import SyntheticDataGenerator
from anomaly_detector.validation import validate_and_normalize_events
from anomaly_detector.preprocessing import aggregate_events_to_windows
from anomaly_detector.features import windows_to_feature_dataframe, windows_to_metadata_dataframe, labels_to_dataframe
from anomaly_detector.reporting import generate_data_quality_report

def test_output_schema_stability():
    """Feature schema must be stable across runs."""
    expected_feature_cols = [
        "event_count", "error_count", "error_ratio",
        "http_request_count", "http_5xx_count", "http_5xx_ratio",
        "auth_failure_count", "restart_count",
        "mean_latency_ms", "p95_latency_ms",
        "latency_sample_count", "http_request_eligible_count", "error_eligible_count"
    ]
    expected_meta_cols = ["window_start", "window_end", "host_id", "service", "window_size_minutes"]
    
    gen = SyntheticDataGenerator(seed=42)
    data = gen.generate_full_synthetic_dataset()
    events_dict = [e.to_dict() for e in data["events"]]
    res = validate_and_normalize_events(events_dict)
    windows = aggregate_events_to_windows(res.accepted, window_size_minutes=1)
    
    feat_df = windows_to_feature_dataframe(windows)
    meta_df = windows_to_metadata_dataframe(windows)
    lbl_df = labels_to_dataframe(data["labels"])
    
    assert list(feat_df.columns) == expected_feature_cols
    assert list(meta_df.columns) == expected_meta_cols
    assert "is_anomaly" in lbl_df.columns
    assert "scenario_name" in lbl_df.columns

def test_data_quality_report_counts():
    """Data quality report should have correct counts."""
    gen = SyntheticDataGenerator(seed=42)
    data = gen.generate_full_synthetic_dataset()
    events_dict = [e.to_dict() for e in data["events"]]
    res = validate_and_normalize_events(events_dict)
    windows = aggregate_events_to_windows(res.accepted, window_size_minutes=1)
    report = generate_data_quality_report(res, windows, data["labels"])
    
    assert report["input"]["total_processed"] == res.total_processed
    assert report["input"]["accepted_count"] == len(res.accepted)
    assert report["input"]["rejected_count"] == len(res.rejected)
    assert report["input"]["duplicated_count"] == len(res.duplicated)
    assert report["windows"]["count"] == len(windows)
    assert report["labels"]["count"] == len(data["labels"])
    assert report["labels"]["anomaly_count"] + report["labels"]["normal_count"] == len(data["labels"])
    assert report["windows"]["num_hosts"] >= 1
    assert report["windows"]["num_services"] >= 1
    # Scenario distribution should cover all scenarios
    assert len(report["labels"]["scenario_distribution"]) >= 5
