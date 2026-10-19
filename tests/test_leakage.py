"""Tests for the leakage audit."""
import json

import numpy as np
import pandas as pd
import pytest

from anomaly_detector.dataset import AlignedDataset
from anomaly_detector.leakage import (
    assert_no_leakage,
    audit_leakage,
    write_leakage_report,
)


def _rebuild(aligned, features):
    return AlignedDataset(
        features=features,
        metadata=aligned.metadata,
        labels=aligned.labels,
        sample_key=aligned.sample_key,
        key_columns=aligned.key_columns,
    )


def test_clean_dataset_has_no_findings(tiny_aligned, readiness_config):
    report = audit_leakage(tiny_aligned, readiness_config)
    assert report.is_clean
    assert report.error_count == 0
    assert set(report.eligible_features) == {
        "event_count", "error_count", "error_ratio", "mean_latency_ms"
    }
    assert_no_leakage(report)  # must not raise


def test_injected_label_column_detected(tiny_aligned, readiness_config):
    features = tiny_aligned.features.copy()
    features["is_anomaly"] = tiny_aligned.labels["is_anomaly"].astype(int).to_numpy()
    report = audit_leakage(_rebuild(tiny_aligned, features), readiness_config)
    assert not report.is_clean
    codes = {f.code for f in report.findings}
    assert "forbidden_column_in_features" in codes
    with pytest.raises(ValueError):
        assert_no_leakage(report)


def test_injected_identifier_detected(tiny_aligned, readiness_config):
    features = tiny_aligned.features.copy()
    features["host_id"] = tiny_aligned.metadata["host_id"].to_numpy()
    report = audit_leakage(_rebuild(tiny_aligned, features), readiness_config)
    assert any(f.column == "host_id" for f in report.findings)
    assert not report.is_clean


def test_future_derived_allowlisted_is_violation(tiny_aligned, readiness_config):
    cfg = readiness_config
    cfg.feature_roles.eligible_features = list(cfg.feature_roles.eligible_features) + ["event_next_count"]
    features = tiny_aligned.features.copy()
    features["event_next_count"] = features["event_count"].shift(-1).fillna(0)
    report = audit_leakage(_rebuild(tiny_aligned, features), cfg)
    assert any(f.code == "allowlisted_but_forbidden" for f in report.findings)


def test_perfect_label_correlation_flagged(tiny_aligned, readiness_config):
    features = tiny_aligned.features.copy()
    # `error_count` becomes a perfect proxy for the label -> target leakage.
    features["error_count"] = tiny_aligned.labels["is_anomaly"].astype(int).to_numpy() * 3
    report = audit_leakage(_rebuild(tiny_aligned, features), readiness_config)
    assert any(f.code == "perfect_label_correlation" for f in report.findings)


def test_report_written(tiny_aligned, readiness_config, tmp_path):
    report = audit_leakage(tiny_aligned, readiness_config)
    j, m = write_leakage_report(report, str(tmp_path))
    payload = json.load(open(j, encoding="utf-8"))
    assert payload["summary"]["is_clean"] is True
    assert "Leakage Audit Report" in open(m, encoding="utf-8").read()
