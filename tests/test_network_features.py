"""Network feature schema, causality, scenarios, leakage and split tests.

These tests use the deterministic 3-run simulator fixture from ``conftest.py``.
They are offline and do not touch the repository's real artifacts.
"""
from datetime import timedelta
import math

import pandas as pd
import pytest

from anomaly_detector.baseline_config import FORBIDDEN_FEATURE_NAMES
from anomaly_detector.network.features import (
    NETWORK_FEATURE_COLUMNS,
    NETWORK_FEATURE_SCHEMA_VERSION,
    NetworkLabel,
    build_network_windows,
    join_windows_with_labels,
    windows_to_dataframe,
)
from anomaly_detector.network.simulator import SimulatorConfig, simulate
from anomaly_detector.network.splits import SplitError, make_split_assignment


# ---------------------------------------------------------------------------
# Simulator and labels
# ---------------------------------------------------------------------------

def test_simulator_is_deterministic_and_labelled_separately():
    a = simulate(SimulatorConfig(seed=7, run_count=3, run_minutes=90))
    b = simulate(SimulatorConfig(seed=7, run_count=3, run_minutes=90))

    assert a.events == b.events
    assert [label.to_dict() for label in a.labels] == [
        label.to_dict() for label in b.labels
    ]
    assert a.event_count == b.event_count
    # Every label carries evaluation metadata that is absent from an event and
    # therefore cannot enter the feature matrix.
    assert all(label.scenario_name for label in a.labels)
    assert all(label.run_id for label in a.labels)
    assert not any("scenario_name" in event for event in a.events)


def test_scenario_schedule_covers_all_required_families(network_simulation):
    summary = network_simulation.summary
    # 3 runs x 90 one-minute windows x 3 streams.
    expected = {
        "normal": 15 * 3 * 3,
        "http_5xx_burst": 10 * 3 * 3,
        "malformed_telemetry": 10 * 3 * 3,
        "db_timeout_latency": 10 * 3 * 3,
        "normal_recovery": 10 * 3 * 3,
        "auth_failure_burst": 10 * 3 * 3,
        "benign_high_traffic": 10 * 3 * 3,
        "benign_latency_shift": 10 * 3 * 3,
        "benign_restart": 5 * 3 * 3,
    }
    assert summary["scenarios"] == expected
    assert summary["label_count"] == sum(expected.values())
    assert summary["incident_windows"] > 0
    assert summary["truncated"] is False


def test_incident_and_benign_labels_are_correct(network_simulation):
    incident = {"http_5xx_burst", "db_timeout_latency", "auth_failure_burst"}
    for label in network_simulation.labels:
        if label.is_anomaly:
            assert label.scenario_name in incident
            assert not label.telemetry_issue
        if label.scenario_name in incident:
            # Only the affected stream is labelled anomalous.
            expected = (
                (label.scenario_name == "http_5xx_burst"
                 and label.service in {"web-api", "auth-service"})
                or (label.scenario_name == "db_timeout_latency"
                    and label.service == "orders-db")
                or (label.scenario_name == "auth_failure_burst"
                    and label.service == "auth-service")
            )
            assert label.is_anomaly is expected
        if label.scenario_name.startswith("benign_") or label.scenario_name in {
            "normal", "normal_recovery",
        }:
            assert label.is_anomaly is False
        assert label.telemetry_issue is (label.scenario_name == "malformed_telemetry")


def test_malformed_db_window_has_no_accepted_events(network_windows):
    windows, labels, missing = network_windows
    assert missing == []
    by_key = {(w.metadata.host_id, w.metadata.service,
               w.metadata.window_start): w for w in windows}
    empty = [w for w in windows if w.features["event_count"] == 0]
    assert len(empty) == 3, "one empty orders-db window per simulated run"
    for window in empty:
        assert window.metadata.host_id == "sim-db-01"
        assert window.metadata.service == "orders-db"
        assert window.metadata.window_start.minute == 33
        assert by_key[window.key] is window


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

def test_network_schema_is_explicit_and_label_free():
    assert NETWORK_FEATURE_SCHEMA_VERSION == "network-features-v1"
    assert len(NETWORK_FEATURE_COLUMNS) == 39
    assert len(set(NETWORK_FEATURE_COLUMNS)) == len(NETWORK_FEATURE_COLUMNS)
    lowered = {name.lower() for name in NETWORK_FEATURE_COLUMNS}
    for forbidden in FORBIDDEN_FEATURE_NAMES:
        assert forbidden.lower() not in lowered
    for token in ("scenario", "label", "ground_truth", "run_id", "event_id",
                  "host_id", "sample_key"):
        assert not any(token in name for name in lowered)


def test_windows_expose_exactly_the_documented_schema(network_windows):
    windows, labels, _ = network_windows
    features, metadata = windows_to_dataframe(windows)
    assert list(features.columns) == NETWORK_FEATURE_COLUMNS
    assert list(metadata.columns) == [
        "sample_key", "window_start", "window_end", "host_id", "service",
        "window_size_minutes",
    ]
    for name in ("scenario_name", "run_id", "is_anomaly", "telemetry_issue"):
        assert name not in features.columns
        assert name not in metadata.columns
    for window in windows:
        assert set(window.features) == set(NETWORK_FEATURE_COLUMNS)
        assert all(isinstance(v, (int, float)) or v is None
                   for v in window.features.values())


def test_zero_denominator_ratios_are_none_not_zero(network_windows):
    windows, _, _ = network_windows
    idle_db = [w for w in windows if w.features["db_event_count"] == 0]
    assert idle_db
    for window in idle_db:
        assert window.features["db_timeout_ratio"] is None
        assert window.features["db_connection_failure_ratio"] is None

    idle_http = [w for w in windows if w.features["http_request_count"] == 0]
    assert idle_http
    for window in idle_http:
        assert window.features["http_5xx_ratio"] is None

    idle_auth = [w for w in windows
                 if not w.features["auth_failure_count"]
                 and not w.features["auth_success_count"]]
    assert idle_auth
    for window in idle_auth:
        assert window.features["auth_failure_ratio"] is None


# ---------------------------------------------------------------------------
# Causality / leakage
# ---------------------------------------------------------------------------

def test_future_events_cannot_change_past_windows(
    network_simulation, network_parsed, network_windows
):
    windows, _, _ = network_windows
    first_start = min(w.metadata.window_start for w in windows)
    cutoff = first_start + timedelta(minutes=45)

    early = [w for w in windows if w.metadata.window_end <= cutoff]
    assert len(early) == 3 * 45  # three streams, forty-five one-minute windows

    truncated_events = [e for e in network_parsed.events if e.timestamp < cutoff]
    grid = {}
    for window in early:
        grid.setdefault(
            (window.metadata.host_id, window.metadata.service), []
        ).append(window.metadata.window_start)
    grid = {key: sorted(set(starts)) for key, starts in grid.items()}

    rebuilt = build_network_windows(
        truncated_events,
        network_parsed.quality_by_window(1),
        window_size_minutes=1,
        rolling_windows=30,
        rolling_min_history=5,
        window_grid=grid,
    )
    rebuilt_by_key = {window.key: window for window in rebuilt}
    assert len(rebuilt_by_key) == len(early)
    for original in early:
        assert original.features == rebuilt_by_key[original.key].features


def test_rolling_features_require_prior_history(network_windows):
    windows, _, _ = network_windows
    streams = {}
    for window in windows:
        streams.setdefault(
            (window.metadata.host_id, window.metadata.service), []
        ).append(window)
    for stream_windows in streams.values():
        stream_windows.sort(key=lambda w: w.metadata.window_start)
        for window in stream_windows[:4]:
            assert window.features["event_count_zscore"] is None
            assert window.features["p95_latency_ratio"] is None
        for window in stream_windows[5:10]:
            if window.features["event_count"] is not None:
                assert window.features["event_count_zscore"] is not None


def test_features_never_carry_evaluation_labels(network_windows):
    windows, labels, _ = network_windows
    assert len(windows) == len(labels)
    for window, label in zip(windows, labels):
        assert window.metadata.host_id == label.host_id
        assert window.metadata.service == label.service
        for name, value in window.features.items():
            if isinstance(value, str):
                assert value not in {label.scenario_name, label.run_id}
        # The scenario vocabulary is structural: it cannot collide with a
        # numeric feature value.
        assert all(
            not isinstance(value, str) for value in window.features.values()
        )


def test_join_reports_windows_without_labels(network_windows):
    windows, labels, _ = network_windows
    _, aligned, missing = join_windows_with_labels(windows, labels[:-1])
    assert len(missing) == 1
    assert len(aligned) == len(labels) - 1


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_time_split_is_strictly_chronological(network_windows):
    windows, labels, _ = network_windows
    assignment = make_split_assignment(
        windows, labels, strategy="time",
        ratios={"train": 0.6, "validation": 0.2, "test": 0.2}, seed=42,
    )
    train_latest = max(windows[i].metadata.window_start for i in assignment.indices["train"])
    val_earliest = min(windows[i].metadata.window_start for i in assignment.indices["validation"])
    val_latest = max(windows[i].metadata.window_start for i in assignment.indices["validation"])
    test_earliest = min(windows[i].metadata.window_start for i in assignment.indices["test"])
    assert train_latest < val_earliest
    assert val_latest < test_earliest
    assert assignment.integrity["chronological"] is True
    assert assignment.integrity["no_timestamp_overlap"] is True

    all_indices = sum((list(v) for v in assignment.indices.values()), [])
    assert sorted(all_indices) == list(range(len(windows)))


def test_group_split_keeps_runs_disjoint(network_windows):
    windows, labels, _ = network_windows
    assignment = make_split_assignment(
        windows, labels, strategy="group",
        ratios={"train": 0.6, "validation": 0.2, "test": 0.2}, seed=42,
    )
    run_sets = {
        name: {labels[i].run_id for i in assignment.indices[name]}
        for name in ("train", "validation", "test")
    }
    assert run_sets["train"].isdisjoint(run_sets["validation"])
    assert run_sets["train"].isdisjoint(run_sets["test"])
    assert run_sets["validation"].isdisjoint(run_sets["test"])
    assert assignment.integrity["group_disjoint"] is True
    assert set().union(*run_sets.values()) == {label.run_id for label in labels}


def test_invalid_split_policies_are_rejected(network_windows):
    windows, labels, _ = network_windows
    with pytest.raises(SplitError):
        make_split_assignment(windows, labels, strategy="random")
    with pytest.raises(SplitError):
        make_split_assignment(
            windows, labels, strategy="time",
            ratios={"train": 0.7, "validation": 0.2, "test": 0.2},
        )
    with pytest.raises(SplitError):
        make_split_assignment(windows, labels[:-1], strategy="time")
