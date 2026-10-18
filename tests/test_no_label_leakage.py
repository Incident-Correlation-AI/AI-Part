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

def test_exported_feature_file_has_no_label_columns(tmp_path):
    """The exported feature CSV must not contain labels or metadata columns."""
    from anomaly_detector.pipeline import run_pipeline, export_result

    gen = SyntheticDataGenerator(seed=11)
    data = gen.generate_full_synthetic_dataset()
    result = run_pipeline(
        [e.to_dict() for e in data["events"]],
        window_size_minutes=1,
        labels=data["labels"],
    )
    paths = export_result(result, str(tmp_path))
    with open(paths["features"], "r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    forbidden = {"is_anomaly", "scenario_name", "host_id", "service",
                 "window_start", "window_end", "intensity", "description"}
    assert forbidden.isdisjoint(set(header))
    # And the labels file genuinely carries the labels.
    with open(paths["labels"], "r", encoding="utf-8") as fh:
        label_header = fh.readline().strip().split(",")
    assert "is_anomaly" in label_header
    assert "scenario_name" in label_header
