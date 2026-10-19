"""Tests for the training-ready export contract."""
import json
import os

import pandas as pd
import pytest

from anomaly_detector.dataset import AlignedDataset
from anomaly_detector.splits import make_splits
from anomaly_detector.training_ready import (
    build_training_ready,
    verify_alignment,
    write_split,
    write_training_ready,
)


def _rebuild(aligned, features):
    return AlignedDataset(
        features=features,
        metadata=aligned.metadata,
        labels=aligned.labels,
        sample_key=aligned.sample_key,
        key_columns=aligned.key_columns,
    )


def test_features_exclude_labels_and_identifiers(tiny_aligned, readiness_config):
    tr = build_training_ready(tiny_aligned, readiness_config)
    cols = set(tr.features.columns)
    for forbidden in ["is_anomaly", "scenario_name", "host_id", "service", "window_start", "window_end"]:
        assert forbidden not in cols
    assert set(tr.eligible_features) == cols


def test_alignment_mismatch_fails_loudly(tiny_aligned, readiness_config):
    tr = build_training_ready(tiny_aligned, readiness_config)
    tr.labels = tr.labels.iloc[:-1].copy()
    with pytest.raises(ValueError):
        verify_alignment(tr, readiness_config.data_policy.sample_key_columns)


def test_sample_key_mismatch_fails(tiny_aligned, readiness_config):
    tr = build_training_ready(tiny_aligned, readiness_config)
    tr.sample_index.loc[0, "sample_key"] = "tampered"
    with pytest.raises(ValueError):
        verify_alignment(tr, readiness_config.data_policy.sample_key_columns)


def test_export_refuses_on_leakage(tiny_aligned, readiness_config):
    features = tiny_aligned.features.copy()
    features["is_anomaly"] = tiny_aligned.labels["is_anomaly"].astype(int).to_numpy()
    with pytest.raises(ValueError):
        build_training_ready(_rebuild(tiny_aligned, features), readiness_config)


def test_export_allow_leakage_override(tiny_aligned, readiness_config):
    features = tiny_aligned.features.copy()
    features["is_anomaly"] = tiny_aligned.labels["is_anomaly"].astype(int).to_numpy()
    tr = build_training_ready(_rebuild(tiny_aligned, features), readiness_config, strict_leakage=False)
    assert not tr.leakage.is_clean  # exported, but the problem is recorded


def test_write_training_ready_files_and_manifest(tiny_aligned, readiness_config, tmp_path):
    tr = build_training_ready(tiny_aligned, readiness_config)
    manifest_path = write_training_ready(
        tr, str(tmp_path), readiness_config.data_policy.sample_key_columns,
        readiness_config.split.label_column,
    )
    for name in ["features.csv", "labels.csv", "metadata.csv", "sample_index.csv", "manifest.json"]:
        assert os.path.isfile(os.path.join(str(tmp_path), name))
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    assert manifest["row_count"] == 10
    assert set(manifest["feature_columns"]) == set(tr.eligible_features)
    assert manifest["leakage"]["clean"] is True
    assert manifest["label_distribution"]["positive"] == 2
    # Every listed file has a checksum.
    for info in manifest["files"].values():
        assert len(info["sha256"]) == 64


def test_row_alignment_is_preserved_across_exports(tiny_aligned, readiness_config, tmp_path):
    tr = build_training_ready(tiny_aligned, readiness_config)
    write_training_ready(tr, str(tmp_path), readiness_config.data_policy.sample_key_columns,
                         readiness_config.split.label_column)
    features = pd.read_csv(os.path.join(str(tmp_path), "features.csv"))
    labels = pd.read_csv(os.path.join(str(tmp_path), "labels.csv"))
    index = pd.read_csv(os.path.join(str(tmp_path), "sample_index.csv"))
    assert len(features) == len(labels) == len(index) == 10
    assert list(index["row_index"]) == list(range(10))
    # The injected anomaly rows survive the round-trip in the same positions.
    assert labels["is_anomaly"].tolist() == tiny_aligned.labels["is_anomaly"].tolist()


def test_write_split_creates_per_split_tables(tiny_aligned, readiness_config, tmp_path):
    tr = build_training_ready(tiny_aligned, readiness_config)
    cfg = readiness_config.split
    cfg.strategy = "random"
    split = make_splits(tiny_aligned.metadata, cfg, tiny_aligned.labels)
    manifest_path = write_split(
        tr, split, str(tmp_path), readiness_config.data_policy.sample_key_columns,
        readiness_config.split.label_column,
    )
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    assert set(manifest["splits"].keys()) == {"train", "validation", "test"}
    for name in split.split_order:
        for f in ["features.csv", "labels.csv", "metadata.csv", "sample_index.csv"]:
            assert os.path.isfile(os.path.join(str(tmp_path), name, f))
    assert manifest["splits"]["train"]["rows"] == 6
