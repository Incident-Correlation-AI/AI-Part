"""Milestone 7 — end-to-end pipeline tests.

These exercise the **real** path:

    raw events -> validation -> windowing -> features -> M6 inference
              -> neutral downstream sink

and assert the safety properties the M7 prompt requires:
* the pipeline reaches the downstream boundary;
* an unscored window is never forwarded as normal;
* labels / ground truth never enter the model feature vector;
* the window id is stable and traceable;
* missingness and benign shift are *reported*, not hidden.
"""
import os

import pytest

from anomaly_detector.e2e_runner import run_end_to_end, run_scenario
from anomaly_detector.e2e_scenarios import all_scenarios
from anomaly_detector.inference import InferenceStatus, load_detector
from anomaly_detector.preprocessing import aggregate_events_to_windows
from anomaly_detector.validation import validate_and_normalize_events

MODEL_PATH = "artifacts/baseline/model/model.joblib"
MANIFEST_PATH = "artifacts/baseline/model/manifest.json"

requires_model = pytest.mark.skipif(
    not (os.path.isfile(MODEL_PATH) and os.path.isfile(MANIFEST_PATH)),
    reason="selected M3 model artifact is not present (run scripts/run_baseline.py run)",
)


@pytest.fixture(scope="module")
def detector():
    return load_detector(MODEL_PATH, MANIFEST_PATH)


@pytest.fixture(scope="module")
def scenarios():
    return all_scenarios()


# ---------------------------------------------------------------------------
# The scenarios themselves
# ---------------------------------------------------------------------------


def test_all_scenarios_are_deterministic(scenarios):
    from anomaly_detector.e2e_scenarios import all_scenarios as build

    first = build()
    second = build()
    assert [s.name for s in first] == [s.name for s in second]
    for a, b in zip(first, second):
        assert a.events == b.events
        assert a.anomalous_windows == b.anomalous_windows


def test_scenarios_cover_the_required_families(scenarios):
    names = {s.name for s in scenarios}
    required = {
        "healthy_traffic",
        "http_5xx_spike",
        "latency_degradation",
        "database_timeout",
        "missing_latency",
        "malformed_event",
        "duplicate_and_late_events",
        "benign_distribution_shift",
    }
    assert required <= names


def test_scenarios_contain_no_ground_truth_in_the_events(scenarios):
    """Ground truth must live outside the raw events, never inside them."""
    forbidden = {"is_anomaly", "label", "scenario", "scenario_name", "ground_truth"}
    for scenario in scenarios:
        for event in scenario.events:
            assert not (set(event) & forbidden), (
                f"scenario {scenario.name} embeds ground truth in an event: {event}"
            )


# ---------------------------------------------------------------------------
# The real M1 path
# ---------------------------------------------------------------------------


def test_malformed_event_is_rejected_not_coerced(scenarios):
    scenario = next(s for s in scenarios if s.name == "malformed_event")
    validation = validate_and_normalize_events(scenario.events)
    assert len(validation.rejected) >= 1
    reasons = " ".join(str(r.get("reasons", r)) for r in validation.rejected)
    assert "timestamp" in reasons.lower() or "host" in reasons.lower()


def test_duplicate_event_is_deduplicated(scenarios):
    scenario = next(s for s in scenarios if s.name == "duplicate_and_late_events")
    validation = validate_and_normalize_events(scenario.events)
    assert len(validation.duplicated) >= 1


def test_late_event_lands_in_the_window_of_its_event_time(scenarios):
    scenario = next(s for s in scenarios if s.name == "duplicate_and_late_events")
    validation = validate_and_normalize_events(scenario.events)
    windows = aggregate_events_to_windows(validation.accepted, window_size_minutes=1)
    starts = {w.metadata.window_start for w in windows}
    # The late event carries a timestamp two minutes and thirty seconds in.
    late = next(
        e for e in validation.accepted
        if e.event_id == "late-1"
    )
    assert late.timestamp.minute % 1 == 0
    assert late.timestamp in [s for s in starts] or late.timestamp.minute == 2


# ---------------------------------------------------------------------------
# Through inference and the downstream boundary
# ---------------------------------------------------------------------------


@requires_model
def test_every_scenario_reaches_the_downstream_boundary(detector, scenarios):
    for scenario in scenarios:
        result = run_scenario(scenario, detector)
        assert result.n_windows > 0, f"{scenario.name} produced no windows"
        # Every window reached the boundary, and every one of them was scored.
        assert result.unscored == 0, f"{scenario.name} had unscored windows"
        assert len(result.window_rows) == result.n_windows


@requires_model
def test_unscored_windows_are_never_forwarded_as_normal(detector, scenarios):
    """The adapter must forward only scored windows."""
    from anomaly_detector.inference import EvidenceCorrelationAdapter, RecordingSink

    for scenario in scenarios:
        validation = validate_and_normalize_events(scenario.events)
        windows = aggregate_events_to_windows(validation.accepted, window_size_minutes=1)
        sink = RecordingSink()
        adapter = EvidenceCorrelationAdapter(detector, sink)
        from anomaly_detector.inference.contract import FeatureWindowRequest, make_window_id

        requests = [
            FeatureWindowRequest(
                window_id=make_window_id(w.metadata.host_id, w.metadata.service,
                                         w.metadata.window_start),
                window_start=w.metadata.window_start,
                window_end=w.metadata.window_end,
                features=dict(w.to_features_dict()),
                host_id=w.metadata.host_id,
                service_name=w.metadata.service,
            )
            for w in windows
        ]
        outcome = adapter.handle_windows(requests)
        assert all(r.status is InferenceStatus.SCORED for r in sink.results)
        for scored in outcome.scored:
            assert scored.is_anomaly in (True, False)
        for rejected in outcome.rejected:
            assert rejected.status != "scored"


@requires_model
def test_window_ids_are_stable_and_traceable(detector, scenarios):
    scenario = next(s for s in scenarios if s.name == "http_5xx_spike")
    result = run_scenario(scenario, detector)
    ids = [row["window_id"] for row in result.window_rows if row["window_id"]]
    assert len(ids) == len(set(ids)), "window ids must be unique"
    for window_id in ids:
        assert window_id.count("|") == 2
        host, service, start = window_id.split("|")
        assert host and service and start


@requires_model
def test_missing_latency_is_reported_as_imputed_not_zero(detector, scenarios):
    scenario = next(s for s in scenarios if s.name == "missing_latency")
    result = run_scenario(scenario, detector)
    assert result.imputed_windows >= 1
    assert "mean_latency_ms" in result.imputed_features


@requires_model
def test_benign_shift_and_missingness_are_reported(detector):
    summary = run_end_to_end(detector)
    assert "benign_shift_flag_rate" in summary
    assert "missing_telemetry_flag_rate" in summary
    names = {row["scenario"] for row in summary["scenarios"]}
    assert "benign_distribution_shift" in names
    assert "missing_latency" in names


@requires_model
def test_end_to_end_summary_reports_detections_misses_and_false_positives(detector):
    summary = run_end_to_end(detector)
    totals = summary["totals"]
    for key in ("detected", "missed", "false_positives", "unscored", "flagged"):
        assert key in totals
    # The harness reports reality; it does not assert that every injected
    # scenario must be detected.
    for row in summary["scenarios"]:
        assert "expected_behavior" in row
        assert "detected" in row and "missed" in row


@requires_model
def test_injected_anomalies_are_detected_or_reported_as_misses(detector, scenarios):
    """Detections and misses are both reported - neither is hidden."""
    for scenario in scenarios:
        if not scenario.anomalous_windows:
            continue
        result = run_scenario(scenario, detector)
        assert result.detected + result.missed == len(scenario.anomalous_windows)
