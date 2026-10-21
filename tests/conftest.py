"""Pytest configuration, path setup and shared fixtures."""
import os
import sys

import pandas as pd
import pytest

# Add src directory to path so anomaly_detector package can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from anomaly_detector.dataset import align_dataset  # noqa: E402
from anomaly_detector.readiness_config import (  # noqa: E402
    DataPolicy,
    FeatureRoleConfig,
    ReadinessConfig,
    RoleRule,
    SplitConfig,
)


def _role_rules():
    return [
        RoleRule("label", ["is_anomaly", "scenario_name", "label", "ground_truth"]),
        RoleRule("timestamp", ["window_start", "window_end"]),
        RoleRule("identifier", ["host_id", "service", "sample_key"]),
        RoleRule("future_derived", ["future", "next_"]),
        RoleRule("feature", ["count", "ratio", "mean_", "p95_"]),
    ]


@pytest.fixture
def readiness_config():
    """A small, fully explicit readiness configuration for tests."""
    cfg = ReadinessConfig()
    cfg.feature_roles = FeatureRoleConfig(
        explicit={},
        patterns=_role_rules(),
        eligible_features=["event_count", "error_count", "error_ratio", "mean_latency_ms"],
        unknown_role_policy="error",
    )
    cfg.data_policy = DataPolicy(
        non_negative_columns=["event_count", "error_count", "mean_latency_ms"],
        unit_interval_columns=["error_ratio"],
        integer_columns=["event_count", "error_count"],
        constant_max_unique=1,
        near_constant_ratio=0.99,
        missing_value_action="report",
        sample_key_columns=["host_id", "service", "window_start"],
        group_columns=["host_id", "service", "scenario_name"],
    )
    cfg.split = SplitConfig(
        strategy="random",
        seed=7,
        ratios={"train": 0.6, "validation": 0.2, "test": 0.2},
        group_column="host_id",
        time_column="window_start",
        label_column="is_anomaly",
    )
    cfg.paths = {
        "reports_dir": "reports",
        "splits_dir": "data/processed/splits",
        "training_ready_dir": "data/processed/training_ready",
    }
    return cfg


@pytest.fixture
def tiny_frames():
    """Small, well-formed features/metadata/labels frames (10 rows)."""
    starts = [f"2026-01-01T00:0{i}:00+00:00" for i in range(5)] * 2
    hosts = ["h1"] * 5 + ["h2"] * 5
    services = ["svc"] * 5 + ["auth"] * 5
    features = pd.DataFrame({
        "event_count": [10, 11, 12, 13, 14, 20, 210, 22, 23, 24],
        "error_count": [1, 0, 2, 1, 0, 1, 40, 2, 1, 0],
        "error_ratio": [0.1, 0.0, 0.16, 0.077, 0.0, 0.05, 0.19, 0.09, 0.04, 0.0],
        "mean_latency_ms": [120.0, 118.0, 121.0, 119.0, 120.0, 200.0, 900.0, 205.0, 199.0, 201.0],
    })
    metadata = pd.DataFrame({
        "window_start": starts,
        "window_end": [s.replace(":0", ":1", 1) for s in starts],
        "host_id": hosts,
        "service": services,
    })
    labels = pd.DataFrame({
        "window_start": list(starts),
        "host_id": hosts,
        "service": services,
        "is_anomaly": [False, False, True, False, False, False, True, False, False, False],
        "scenario_name": ["normal"] * 5 + ["normal"] * 2 + ["cpu_spike"] * 3,
    })
    return features, metadata, labels


@pytest.fixture
def tiny_aligned(tiny_frames, readiness_config):
    features, metadata, labels = tiny_frames
    return align_dataset(
        features, metadata, labels,
        key_columns=readiness_config.data_policy.sample_key_columns,
    )


# ---------------------------------------------------------------------------
# Milestone 3 fixtures
# ---------------------------------------------------------------------------

BASELINE_FEATURES = ["event_count", "error_count", "error_ratio", "mean_latency_ms"]


def _make_baseline_split(n_rows: int, n_anomalies: int, offset: int):
    """Build a deterministic small split: anomalies have clearly higher counts."""
    features = pd.DataFrame({
        "event_count": [10.0 + i % 5 for i in range(n_rows)],
        "error_count": [0.0] * n_rows,
        "error_ratio": [0.0] * n_rows,
        "mean_latency_ms": [50.0 + (i % 3) for i in range(n_rows)],
    })
    labels = []
    for i in range(n_rows):
        anomalous = i < n_anomalies
        if anomalous:
            features.loc[i, "event_count"] = 500.0 + i
            features.loc[i, "error_count"] = 120.0 + i
            features.loc[i, "error_ratio"] = 0.6
            features.loc[i, "mean_latency_ms"] = 900.0 + i
        labels.append(anomalous)
    starts = pd.date_range("2026-02-01", periods=n_rows, freq="min", tz="UTC")
    metadata = pd.DataFrame({
        "sample_key": [f"h1|svc|{s.isoformat()}" for s in starts],
        "window_start": [s.isoformat() for s in starts],
        "window_end": [(s + pd.Timedelta(minutes=1)).isoformat() for s in starts],
        "host_id": ["h1"] * n_rows,
        "service": ["svc"] * n_rows,
        "window_size_minutes": [1] * n_rows,
    })
    label_frame = pd.DataFrame({
        "sample_key": metadata["sample_key"].tolist(),
        "is_anomaly": labels,
    })
    return features, label_frame, metadata


def _write_split(root, name, features, labels, metadata):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    features.to_csv(directory / "features.csv", index=False)
    labels.to_csv(directory / "labels.csv", index=False)
    metadata.to_csv(directory / "metadata.csv", index=False)


@pytest.fixture
def baseline_config(tmp_path):
    """A baseline config pointing at a temporary output directory."""
    from anomaly_detector.baseline_config import load_baseline_config

    cfg = load_baseline_config(None)
    cfg.data.feature_columns = list(BASELINE_FEATURES)
    cfg.data.training_data_policy = "normal_only"
    cfg.data.missing_value_policy = "median_impute"
    cfg.data.metadata_columns = ["window_start", "window_end", "host_id", "service"]
    cfg.model.n_estimators = 20
    cfg.model.random_state = 42
    cfg.paths["splits_dir"] = str(tmp_path / "splits")
    cfg.paths["output_dir"] = str(tmp_path / "out")
    return cfg


@pytest.fixture
def baseline_splits(tmp_path, baseline_config):
    """Write deterministic train/validation/test split tables and return the root."""
    root = tmp_path / "splits" / "random"
    for name, (n_rows, n_anomalies) in {
        "train": (30, 5),
        "validation": (12, 3),
        "test": (10, 2),
    }.items():
        features, labels, metadata = _make_baseline_split(n_rows, n_anomalies, 0)
        _write_split(root, name, features, labels, metadata)
    return root
