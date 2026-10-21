"""Tests for the Milestone 3 baseline configuration."""
import os

import pytest

from anomaly_detector.baseline_config import load_baseline_config


def test_loads_repo_config():
    cfg = load_baseline_config(os.path.join("configs", "baseline.yaml"))
    assert cfg.model.n_estimators == 100
    assert cfg.model.max_samples == 256
    assert cfg.model.contamination == "auto"
    assert cfg.model.random_state == 42
    assert cfg.model.n_jobs == 2
    assert cfg.data.training_data_policy == "normal_only"
    assert cfg.data.missing_value_policy == "median_impute"
    assert cfg.threshold.method == "validation_f1"
    assert "event_count" in cfg.data.feature_columns


def test_repo_config_allowlist_excludes_metadata():
    cfg = load_baseline_config(os.path.join("configs", "baseline.yaml"))
    for forbidden in ["is_anomaly", "scenario_name", "host_id", "service",
                      "window_start", "window_end", "sample_key", "split"]:
        assert forbidden not in cfg.data.feature_columns


def test_defaults_without_file():
    cfg = load_baseline_config(None)
    assert cfg.model.n_estimators == 100
    assert cfg.split.strategy == "random"


def _write(tmp_path, text):
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_empty_allowlist_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_baseline_config(_write(tmp_path, "data:\n  feature_columns: []\n"))


def test_duplicate_allowlist_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_baseline_config(
            _write(tmp_path, "data:\n  feature_columns: [event_count, event_count]\n")
        )


def test_forbidden_column_in_allowlist_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_baseline_config(
            _write(tmp_path, "data:\n  feature_columns: [event_count, is_anomaly]\n")
        )


def test_identifier_in_allowlist_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_baseline_config(
            _write(tmp_path, "data:\n  feature_columns: [event_count, host_id]\n")
        )


def test_invalid_contamination_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_baseline_config(_write(tmp_path, "model:\n  contamination: 0.9\n"))
    with pytest.raises(ValueError):
        load_baseline_config(_write(tmp_path, "model:\n  contamination: 'high'\n"))


def test_invalid_missing_policy_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_baseline_config(
            _write(tmp_path, "data:\n  feature_columns: [a]\n  missing_value_policy: zero_fill\n")
        )


def test_invalid_training_policy_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_baseline_config(
            _write(tmp_path, "data:\n  feature_columns: [a]\n  training_data_policy: sneaky\n")
        )


def test_invalid_threshold_method_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_baseline_config(
            _write(tmp_path, "data:\n  feature_columns: [a]\nthreshold:\n  method: magic\n")
        )


def test_invalid_quantile_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_baseline_config(
            _write(tmp_path, "data:\n  feature_columns: [a]\nthreshold:\n  fallback_quantile: 0\n")
        )
