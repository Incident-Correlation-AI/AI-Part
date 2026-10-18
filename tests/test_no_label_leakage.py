"""Test that labels are not leaked into feature matrix."""
from anomaly_detector.synthetic_data import SyntheticDataGenerator
from anomaly_detector.validation import validate_and_normalize_events
from anomaly_detector.preprocessing import aggregate_events_to_windows
from anomaly_detector.features import windows_to_feature_dataframe, check_label_leakage

def test_no_label_leakage():
    gen = SyntheticDataGenerator(seed=42)
    data = gen.generate_full_synthetic_dataset()
    events_dict = [e.to_dict() for e in data["events"]]
    res = validate_and_normalize_events(events_dict)
    windows = aggregate_events_to_windows(res.accepted, window_size_minutes=1)
    feat_df = windows_to_feature_dataframe(windows)
    leaks = check_label_leakage(feat_df)
    assert len(leaks) == 0, f"Label leakage detected in columns: {leaks}"

def test_feature_columns_clean():
    gen = SyntheticDataGenerator(seed=7)
    data = gen.generate_full_synthetic_dataset()
    events_dict = [e.to_dict() for e in data["events"]]
    res = validate_and_normalize_events(events_dict)
    windows = aggregate_events_to_windows(res.accepted, window_size_minutes=1)
    feat_df = windows_to_feature_dataframe(windows)
    # Check no metadata fields
    forbidden = ["host_id", "service", "window_start", "window_end", "scenario_name", "is_anomaly"]
    for col in feat_df.columns:
        assert col not in forbidden
