"""Tests for the dataset inventory/integrity audit."""
import json

import numpy as np
import pandas as pd

from anomaly_detector.audit import audit_dataset, write_audit
from anomaly_detector.dataset import AlignedDataset, align_dataset


def _rebuild(aligned, features):
    return AlignedDataset(
        features=features,
        metadata=aligned.metadata,
        labels=aligned.labels,
        sample_key=aligned.sample_key,
        key_columns=aligned.key_columns,
    )


def test_clean_dataset_is_ready(tiny_aligned, readiness_config):
    audit = audit_dataset(tiny_aligned, readiness_config)
    assert audit.is_ready
    assert audit.error_count == 0
    assert audit.inventory["feature_rows"] == 10
    assert audit.label_summary["positive"] == 2


def test_negative_value_detected(tiny_aligned, readiness_config):
    features = tiny_aligned.features.copy()
    features.loc[0, "event_count"] = -5
    audit = audit_dataset(_rebuild(tiny_aligned, features), readiness_config)
    assert any(i.code == "negative_value" and i.column == "event_count" for i in audit.issues)
    assert not audit.is_ready


def test_out_of_unit_interval_detected(tiny_aligned, readiness_config):
    features = tiny_aligned.features.copy()
    features.loc[0, "error_ratio"] = 1.5
    audit = audit_dataset(_rebuild(tiny_aligned, features), readiness_config)
    assert any(i.code == "out_of_unit_interval" and i.column == "error_ratio" for i in audit.issues)


def test_infinite_value_is_error_but_nan_is_not(tiny_aligned, readiness_config):
    features = tiny_aligned.features.copy()
    features.loc[0, "mean_latency_ms"] = np.inf
    audit = audit_dataset(_rebuild(tiny_aligned, features), readiness_config)
    assert any(i.code == "infinite_values" and i.column == "mean_latency_ms" for i in audit.issues)

    features2 = tiny_aligned.features.copy()
    features2.loc[0, "mean_latency_ms"] = np.nan
    audit2 = audit_dataset(_rebuild(tiny_aligned, features2), readiness_config)
    assert any(i.code == "missing_values" for i in audit2.issues)
    assert not any(i.code == "infinite_values" for i in audit2.issues)
    assert audit2.is_ready  # missingness is report-only, not an error


def test_constant_feature_reported(tiny_aligned, readiness_config):
    features = tiny_aligned.features.copy()
    features["event_count"] = 7
    audit = audit_dataset(_rebuild(tiny_aligned, features), readiness_config)
    assert any(i.code == "constant_feature" and i.column == "event_count" for i in audit.issues)


def test_duplicate_rows_reported(tiny_frames, readiness_config):
    features, metadata, labels = tiny_frames
    features = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    metadata = pd.concat([metadata, metadata.iloc[[0]]], ignore_index=True)
    labels = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
    bad = align_dataset(features, metadata, labels,
                        key_columns=readiness_config.data_policy.sample_key_columns)
    audit = audit_dataset(bad, readiness_config)
    assert audit.duplicates["duplicate_feature_rows"] >= 1


def test_injected_label_column_is_error(tiny_aligned, readiness_config):
    features = tiny_aligned.features.copy()
    features["is_anomaly"] = tiny_aligned.labels["is_anomaly"].astype(int).to_numpy()
    audit = audit_dataset(_rebuild(tiny_aligned, features), readiness_config)
    assert any(i.code == "forbidden_column_in_features" and i.column == "is_anomaly" for i in audit.issues)
    assert not audit.is_ready


def test_audit_does_not_mutate_source(tiny_aligned, readiness_config):
    before = tiny_aligned.features.copy(deep=True)
    audit_dataset(tiny_aligned, readiness_config)
    pd.testing.assert_frame_equal(before, tiny_aligned.features)


def test_write_audit_outputs_json_and_markdown(tiny_aligned, readiness_config, tmp_path):
    audit = audit_dataset(tiny_aligned, readiness_config)
    j, m = write_audit(audit, str(tmp_path))
    assert j.endswith(".json") and m.endswith(".md")
    payload = json.load(open(j, encoding="utf-8"))
    assert payload["summary"]["is_ready"] is True
    assert "Dataset Audit Report" in open(m, encoding="utf-8").read()
