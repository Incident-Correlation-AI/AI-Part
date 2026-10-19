"""Tests for reproducible dataset splits."""
import numpy as np
import pandas as pd
import pytest

from anomaly_detector.splits import make_splits


def _all_indices(result):
    return sorted(i for name in result.split_order for i in result.assignments[name])


def test_random_split_covers_every_row_once(tiny_aligned, readiness_config):
    cfg = readiness_config.split
    cfg.strategy = "random"
    result = make_splits(tiny_aligned.metadata, cfg, tiny_aligned.labels)
    assert _all_indices(result) == list(range(10))
    assert result.size("train") == 6
    assert result.size("validation") == 2
    assert result.size("test") == 2


def test_random_split_is_reproducible(tiny_aligned, readiness_config):
    cfg = readiness_config.split
    cfg.strategy = "random"
    a = make_splits(tiny_aligned.metadata, cfg, tiny_aligned.labels)
    b = make_splits(tiny_aligned.metadata, cfg, tiny_aligned.labels)
    assert a.assignments == b.assignments


def test_different_seed_changes_assignment(readiness_config):
    n = 100
    metadata = pd.DataFrame({
        "host_id": ["h1"] * n,
        "service": ["svc"] * n,
        "window_start": pd.date_range("2026-01-01", periods=n, freq="min").astype(str),
    })
    cfg = readiness_config.split
    cfg.strategy = "random"
    cfg.seed = 1
    a = make_splits(metadata, cfg)
    cfg.seed = 2
    b = make_splits(metadata, cfg)
    assert a.assignments != b.assignments


def test_time_split_is_chronological(tiny_aligned, readiness_config):
    cfg = readiness_config.split
    cfg.strategy = "time"
    cfg.time_column = "window_start"
    result = make_splits(tiny_aligned.metadata, cfg, tiny_aligned.labels)
    train = tiny_aligned.metadata["window_start"].iloc[result.assignments["train"]]
    test = tiny_aligned.metadata["window_start"].iloc[result.assignments["test"]]
    assert train.max() <= test.min()


def test_group_split_has_no_group_overlap(tiny_aligned, readiness_config):
    cfg = readiness_config.split
    cfg.strategy = "group"
    cfg.group_column = "host_id"
    result = make_splits(tiny_aligned.metadata, cfg, tiny_aligned.labels)
    host = tiny_aligned.metadata["host_id"]
    train_hosts = set(host.iloc[result.assignments["train"]])
    val_hosts = set(host.iloc[result.assignments["validation"]])
    test_hosts = set(host.iloc[result.assignments["test"]])
    assert not (train_hosts & val_hosts)
    assert not (train_hosts & test_hosts)
    assert not (val_hosts & test_hosts)


def test_group_split_resolves_scenario_from_labels(tiny_aligned, readiness_config):
    cfg = readiness_config.split
    cfg.strategy = "group"
    cfg.group_column = "scenario_name"  # lives in the labels table, not metadata
    result = make_splits(tiny_aligned.metadata, cfg, tiny_aligned.labels)
    assert result.group_column == "scenario_name"
    assert _all_indices(result) == list(range(10))


def test_empty_split_warns(readiness_config):
    metadata = pd.DataFrame({
        "host_id": ["h1"] * 10,
        "service": ["svc"] * 10,
        "window_start": pd.date_range("2026-01-01", periods=10, freq="min").astype(str),
    })
    cfg = readiness_config.split
    cfg.strategy = "group"
    cfg.group_column = "host_id"  # only one group -> everything to train
    result = make_splits(metadata, cfg)
    assert any("empty" in w for w in result.warnings)


def test_single_class_split_warns(readiness_config):
    n = 10
    metadata = pd.DataFrame({
        "host_id": ["h1"] * n,
        "service": ["svc"] * n,
        "window_start": pd.date_range("2026-01-01", periods=n, freq="min").astype(str),
    })
    labels = pd.DataFrame({"is_anomaly": [True] * n})
    cfg = readiness_config.split
    cfg.strategy = "random"
    result = make_splits(metadata, cfg, labels)
    assert any("single label class" in w for w in result.warnings)


def test_missing_group_column_raises(tiny_aligned, readiness_config):
    cfg = readiness_config.split
    cfg.strategy = "group"
    cfg.group_column = "does_not_exist"
    with pytest.raises(ValueError):
        make_splits(tiny_aligned.metadata, cfg, tiny_aligned.labels)


def test_unknown_strategy_raises(tiny_aligned, readiness_config):
    cfg = readiness_config.split
    cfg.strategy = "magic"
    with pytest.raises(ValueError):
        make_splits(tiny_aligned.metadata, cfg)
