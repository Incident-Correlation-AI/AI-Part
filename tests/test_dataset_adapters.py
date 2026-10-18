"""Test the public-dataset adapter interface using an offline HDFS fixture.

No network access is required: the fixture mimics the Loghub HDFS raw log
format and its block-level label CSV.
"""
import os
from anomaly_detector.dataset_adapters import BaseDatasetAdapter, LoghubHDFSAdapter

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "hdfs_sample")


def test_adapter_is_a_base_adapter():
    adapter = LoghubHDFSAdapter()
    assert isinstance(adapter, BaseDatasetAdapter)


def test_can_load_detects_valid_fixture():
    adapter = LoghubHDFSAdapter()
    assert adapter.can_load(os.path.join(FIXTURE_DIR, "HDFS.log")) is True
    assert adapter.can_load(os.path.join(FIXTURE_DIR, "does_not_exist.log")) is False


def test_load_data_parses_events():
    adapter = LoghubHDFSAdapter()
    events, labels = adapter.load_data(FIXTURE_DIR)
    assert len(events) == 8
    assert len(labels) == 3
    # Events carry the assumed year and UTC tz.
    first = events[0]
    assert first.timestamp.year == 2008
    assert first.timestamp.tzinfo is not None
    assert first.service == "hdfs"
    assert first.event_type == "application_log"
    # Level normalization happened at parse time value level (raw preserved).
    assert events[4].level == "ERROR"


def test_block_id_extracted_from_message():
    adapter = LoghubHDFSAdapter()
    events, _ = adapter.load_data(FIXTURE_DIR)
    block_ids = {e.attributes.get("block_id") for e in events}
    assert "blk_38865049064139660" in block_ids
    assert "blk_-56789012345678901" in block_ids


def test_labels_are_separate_and_typed():
    adapter = LoghubHDFSAdapter()
    events, labels = adapter.load_data(FIXTURE_DIR)
    assert {l.is_anomaly for l in labels} == {True, False}
    assert all(l.scenario_name == "hdfs_block_label" for l in labels)
    # Labels are not embedded in events.
    for ev in events:
        assert "is_anomaly" not in ev.attributes
        assert "scenario_name" not in ev.source_metadata


def test_get_dataset_info_documents_provenance():
    info = LoghubHDFSAdapter().get_dataset_info()
    for key in ("name", "source", "license", "label_semantics", "limitations",
                "mapping", "lossy_conversions"):
        assert key in info
    assert info["year_assumption"] == 2008
