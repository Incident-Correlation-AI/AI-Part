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
