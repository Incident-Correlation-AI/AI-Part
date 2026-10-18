"""Test synthetic data generation."""
from anomaly_detector.synthetic_data import SyntheticDataGenerator
from anomaly_detector.validation import validate_and_normalize_events
from anomaly_detector.preprocessing import aggregate_events_to_windows
from anomaly_detector.features import windows_to_feature_dataframe, check_label_leakage

def test_reproducible_generation():
    gen1 = SyntheticDataGenerator(seed=42)
    gen2 = SyntheticDataGenerator(seed=42)
    data1 = gen1.generate_full_synthetic_dataset()
    data2 = gen2.generate_full_synthetic_dataset()
    assert len(data1["events"]) == len(data2["events"])
    assert data1["metadata"]["seed"] == 42

def test_all_scenarios_present():
    gen = SyntheticDataGenerator(seed=123)
    data = gen.generate_full_synthetic_dataset()
    scenarios = set(l.scenario_name for l in data["labels"])
    expected = {
        "stable_traffic", "occasional_errors", "low_traffic", 
        "missing_delayed", "normal_restart", "http_5xx_surge",
        "auth_failure_surge", "restart_surge", "latency_increase", "volume_spike"
    }
    assert expected.issubset(scenarios)

def test_synthetic_data_validates():
    gen = SyntheticDataGenerator(seed=42)
    data = gen.generate_full_synthetic_dataset()
    events_dict = [e.to_dict() for e in data["events"]]
    res = validate_and_normalize_events(events_dict)
    assert res.total_processed > 0

def test_windows_generated():
    gen = SyntheticDataGenerator(seed=42)
    data = gen.generate_full_synthetic_dataset()
    events_dict = [e.to_dict() for e in data["events"]]
    res = validate_and_normalize_events(events_dict)
    windows = aggregate_events_to_windows(res.accepted, window_size_minutes=1)
    assert len(windows) > 0

def test_same_seed_identical_values():
    """Same seed must yield byte-identical event payloads."""
    import json
    d1 = SyntheticDataGenerator(seed=99).generate_full_synthetic_dataset()
    d2 = SyntheticDataGenerator(seed=99).generate_full_synthetic_dataset()
    s1 = json.dumps([e.to_dict() for e in d1["events"]], sort_keys=True, default=str)
    s2 = json.dumps([e.to_dict() for e in d2["events"]], sort_keys=True, default=str)
    assert s1 == s2


def test_different_seeds_differ():
    import json
    d1 = SyntheticDataGenerator(seed=1).generate_full_synthetic_dataset()
    d2 = SyntheticDataGenerator(seed=2).generate_full_synthetic_dataset()
    s1 = json.dumps([e.to_dict() for e in d1["events"]], sort_keys=True, default=str)
    s2 = json.dumps([e.to_dict() for e in d2["events"]], sort_keys=True, default=str)
    assert s1 != s2


def test_scenarios_have_both_normal_and_anomaly_labels():
    data = SyntheticDataGenerator(seed=42).generate_full_synthetic_dataset()
    has_normal = any(not l.is_anomaly for l in data["labels"])
    has_anomaly = any(l.is_anomaly for l in data["labels"])
    assert has_normal and has_anomaly
