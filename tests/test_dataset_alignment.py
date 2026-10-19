"""Tests for dataset loading, sample keys and row alignment."""
import pandas as pd
import pytest

from anomaly_detector.dataset import (
    align_dataset,
    has_complete_key,
    load_table,
    make_sample_key,
)


def test_make_sample_key_stable_and_order_preserving():
    df = pd.DataFrame({
        "host_id": ["h1", "h2"],
        "service": ["svc", "auth"],
        "window_start": ["2026-01-01T00:00:00+00:00", "2026-01-01T00:01:00+00:00"],
    })
    key = make_sample_key(df)
    assert list(key) == [
        "h1|svc|2026-01-01T00:00:00+00:00",
        "h2|auth|2026-01-01T00:01:00+00:00",
    ]


def test_make_sample_key_missing_column_raises():
    df = pd.DataFrame({"host_id": ["h1"], "service": ["svc"]})
    with pytest.raises(KeyError):
        make_sample_key(df)


def test_has_complete_key_flags_blanks():
    df = pd.DataFrame({
        "host_id": ["h1", ""],
        "service": ["svc", "auth"],
        "window_start": ["t0", "t1"],
    })
    assert list(has_complete_key(df)) == [True, False]


def test_row_count_mismatch_raises(tiny_frames):
    features, metadata, labels = tiny_frames
    with pytest.raises(ValueError):
        align_dataset(features, metadata.iloc[:-1], labels)


def test_alignment_joins_labels_by_key(tiny_frames, readiness_config):
    features, metadata, labels = tiny_frames
    aligned = align_dataset(features, metadata, labels,
                            key_columns=readiness_config.data_policy.sample_key_columns)
    assert len(aligned.features) == 10
    assert len(aligned.labels) == 10
    assert aligned.labels["is_anomaly"].sum() == 2
    assert len(aligned.sample_key) == 10


def test_unmatched_labels_reported(tiny_frames, readiness_config):
    features, metadata, labels = tiny_frames
    extra = labels.iloc[[0]].copy()
    extra["window_start"] = "2099-01-01T00:00:00+00:00"
    labels_with_extra = pd.concat([labels, extra], ignore_index=True)
    aligned = align_dataset(features, metadata, labels_with_extra,
                            key_columns=readiness_config.data_policy.sample_key_columns)
    assert len(aligned.unmatched_labels) == 1
    assert any("no matching feature window" in i for i in aligned.issues)


def test_load_table_rejects_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_table(str(tmp_path / "nope.csv"))
