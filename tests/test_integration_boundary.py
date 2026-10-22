"""Milestone 8 — integration-boundary tests.

These pin the non-negotiable contract rules of the ML -> downstream boundary:

* only a result whose status is exactly ``scored`` may be forwarded;
* every other status is handled separately and remains distinguishable;
* ``None``, missing results, unknown/malformed statuses and exceptions all
  **fail closed** and are never turned into a normal result;
* a sink failure is explicit and never reported as a successful delivery;
* model / threshold / score-direction / provenance survive routing;
* the harness runs offline.

The real downstream interface is pending, so lightweight fake sinks stand in
for it. The inference types and the canonical routing rule are the real ones.
"""
import os
import subprocess
import sys

import pytest

from anomaly_detector.inference import (
    CONTRACT_VERSION,
    EvidenceCorrelationAdapter,
    InferenceResult,
    InferenceStatus,
    ReasonCode,
    RecordingSink,
    SinkDeliveryError,
    UnscoredWindow,
    is_forwardable,
    route_result,
)
from anomaly_detector.integration_harness import (
    IntegrationHarness,
    UnscoredCollector,
    run_rehearsal,
)

MODEL_PATH = "artifacts/baseline/model/model.joblib"
MANIFEST_PATH = "artifacts/baseline/model/manifest.json"

requires_model = pytest.mark.skipif(
    not (os.path.isfile(MODEL_PATH) and os.path.isfile(MANIFEST_PATH)),
    reason="selected M3 model artifact is not present (run scripts/run_baseline.py run)",
)


# ---------------------------------------------------------------------------
# Test doubles built on the REAL request/result types
# ---------------------------------------------------------------------------


def _result(status, *, window_id="w1", is_anomaly=None, raw_score=None,
            reason_code=ReasonCode.OK, reason="test"):
    if raw_score is None and status is InferenceStatus.SCORED and is_anomaly is not None:
        # A genuinely scored result always carries a score and a decision.
        raw_score = -0.9 if is_anomaly else -0.3
    return InferenceResult(
        contract_version=CONTRACT_VERSION,
        status=status,
        reason_code=reason_code,
        reason=reason,
        window_id=window_id,
        raw_score=raw_score,
        score_direction="lower_is_more_anomalous",
        threshold=-0.6202666407504878,
        is_anomaly=is_anomaly,
        model_name="sklearn.ensemble.IsolationForest",
        model_version="c210b7a",
        model_artifact_version="1",
        feature_schema_version="features-v1",
        threshold_config_version="configs/baseline.yaml@c210b7a:-0.62026664075",
    )


class _StubDetector:
    """A detector that returns a scripted result per request."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def score_window(self, request):
        self.calls.append(request)
        if not self._results:
            raise AssertionError("no scripted result left")
        item = self._results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _ExplodingSink:
    """A sink that always fails."""

    def __init__(self):
        self.attempts = 0

    def handle_scored_window(self, result):
        self.attempts += 1
        raise RuntimeError("downstream unavailable")


def _adapter(results, sink=None, unscored=None):
    return EvidenceCorrelationAdapter(
        _StubDetector(results), sink=sink, unscored_handler=unscored
    )


# ---------------------------------------------------------------------------
# 1 & 2 — scored results are forwarded
# ---------------------------------------------------------------------------


def test_scored_non_anomalous_result_is_forwarded():
    sink = RecordingSink()
    outcome = _adapter([_result(InferenceStatus.SCORED, is_anomaly=False, raw_score=-0.4)],
                       sink=sink).handle_windows(["w1"])
    assert outcome.forwarded_count == 1
    assert outcome.unscored_count == 0
    assert len(sink.results) == 1
    assert sink.results[0].is_anomaly is False


def test_scored_anomalous_result_is_forwarded():
    sink = RecordingSink()
    outcome = _adapter([_result(InferenceStatus.SCORED, is_anomaly=True, raw_score=-0.9)],
                       sink=sink).handle_windows(["w1"])
    assert outcome.forwarded_count == 1
    assert sink.results[0].is_anomaly is True


# ---------------------------------------------------------------------------
# 3-6 — each unscored status is not forwarded and is explicitly recorded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,reason_code",
    [
        (InferenceStatus.INSUFFICIENT_DATA, ReasonCode.MISSING_FEATURES),
        (InferenceStatus.INVALID_INPUT, ReasonCode.NON_FINITE_FEATURE),
        (InferenceStatus.MODEL_UNAVAILABLE, ReasonCode.MODEL_ARTIFACT_MISSING),
        (InferenceStatus.INFERENCE_ERROR, ReasonCode.INFERENCE_FAILED),
    ],
)
def test_unscored_status_is_not_forwarded_and_is_recorded(status, reason_code):
    sink = RecordingSink()
    collector = UnscoredCollector()
    outcome = _adapter(
        [_result(status, reason_code=reason_code, reason="n/a")],
        sink=sink, unscored=collector,
    ).handle_windows(["w1"])

    assert outcome.forwarded_count == 0
    assert outcome.unscored_count == 1
    assert outcome.unscored[0].status == status.value
    assert outcome.unscored[0].reason_code == reason_code.value
    # The explicit handler saw it too.
    assert len(collector.records) == 1
    assert collector.records[0].status == status.value
    # And nothing reached the correlation sink.
    assert sink.results == []


def test_unscored_statuses_remain_distinguishable():
    sink = RecordingSink()
    statuses = list(InferenceStatus)
    results = [_result(s) for s in statuses if s is not InferenceStatus.SCORED]
    outcome = _adapter(results, sink=sink).handle_windows(
        [f"w{i}" for i in range(len(results))]
    )
    counts = outcome.unscored_by_status()
    for status in statuses:
        if status is InferenceStatus.SCORED:
            continue
        assert counts.get(status.value) == 1
    assert outcome.forwarded_count == 0


# ---------------------------------------------------------------------------
# 7 — unknown / malformed status fails closed
# ---------------------------------------------------------------------------


def test_unknown_status_is_not_forwarded():
    sink = RecordingSink()
    collector = UnscoredCollector()
    bad = _result(InferenceStatus.SCORED, is_anomaly=True)
    object.__setattr__(bad, "status", "scored")  # a bare string, not the enum
    outcome = _adapter([bad], sink=sink, unscored=collector).handle_windows(["w1"])
    assert outcome.forwarded_count == 0
    assert sink.results == []
    assert outcome.unscored_count == 1


def test_none_status_is_not_forwarded():
    sink = RecordingSink()
    bad = _result(InferenceStatus.SCORED, is_anomaly=True)
    object.__setattr__(bad, "status", None)
    outcome = _adapter([bad], sink=sink).handle_windows(["w1"])
    assert outcome.forwarded_count == 0
    assert sink.results == []


def test_malformed_status_does_not_crash_the_adapter():
    """A malformed status must fail closed, not raise."""
    sink = RecordingSink()
    bad = _result(InferenceStatus.SCORED, is_anomaly=True)
    object.__setattr__(bad, "status", None)
    outcome = _adapter([bad], sink=sink).handle_windows(["w1"])
    assert outcome.unscored_count == 1
    assert outcome.unscored[0].status == "unknown"


# ---------------------------------------------------------------------------
# 8 — None / missing result / exception never becomes a normal result
# ---------------------------------------------------------------------------


def test_none_result_is_not_forwarded():
    sink = RecordingSink()
    outcome = _adapter([None], sink=sink).handle_windows(["w1"])
    assert outcome.forwarded_count == 0
    assert sink.results == []
    assert outcome.unscored[0].status == "unknown"


def test_scoring_exception_is_not_converted_to_a_normal_result():
    sink = RecordingSink()
    collector = UnscoredCollector()
    outcome = _adapter([RuntimeError("boom")], sink=sink, unscored=collector).handle_windows(["w1"])
    assert outcome.forwarded_count == 0
    assert sink.results == []
    assert outcome.unscored_count == 1
    assert outcome.unscored[0].status == InferenceStatus.INFERENCE_ERROR.value
    assert "boom" in outcome.unscored[0].reason


def test_routing_never_converts_anything_to_is_anomaly_false():
    for bad in (None, "not a result", 42):
        decision = route_result(bad)
        assert decision.forwardable is False
        assert decision.is_anomaly is None


# ---------------------------------------------------------------------------
# 9 — sink failures are explicit
# ---------------------------------------------------------------------------


def test_sink_exception_is_recorded_and_not_a_false_success():
    sink = _ExplodingSink()
    outcome = _adapter([_result(InferenceStatus.SCORED, is_anomaly=True, raw_score=-0.9)],
                       sink=sink).handle_windows(["w1"])
    assert sink.attempts == 1
    # Not reported as delivered.
    assert outcome.forwarded_count == 0
    assert outcome.sink_failure_count == 1
    assert outcome.sink_failures[0].error_type == "RuntimeError"
    assert outcome.sink_failures[0].window_id == "w1"


def test_sink_exception_on_a_single_window_raises_explicitly():
    sink = _ExplodingSink()
    with pytest.raises(SinkDeliveryError):
        _adapter([_result(InferenceStatus.SCORED, is_anomaly=True)], sink=sink).handle_window("w1")


def test_sink_failure_does_not_abort_the_rest_of_the_batch():
    class FlakySink:
        def __init__(self):
            self.seen = []

        def handle_scored_window(self, result):
            self.seen.append(result.window_id)
            if result.window_id == "w1":
                raise RuntimeError("transient")

    sink = FlakySink()
    results = [
        _result(InferenceStatus.SCORED, window_id="w1", is_anomaly=True),
        _result(InferenceStatus.SCORED, window_id="w2", is_anomaly=False),
        _result(InferenceStatus.SCORED, window_id="w3", is_anomaly=False),
    ]
    outcome = _adapter(results, sink=sink).handle_windows(["w1", "w2", "w3"])
    assert outcome.sink_failure_count == 1
    assert outcome.forwarded_count == 2
    assert [w.window_id for w in outcome.scored] == ["w2", "w3"]


# ---------------------------------------------------------------------------
# 10 — mixed batch
# ---------------------------------------------------------------------------


def test_mixed_batch_routes_each_outcome_correctly():
    sink = RecordingSink()
    collector = UnscoredCollector()
    results = [
        _result(InferenceStatus.SCORED, window_id="w1", is_anomaly=True, raw_score=-0.9),
        _result(InferenceStatus.INSUFFICIENT_DATA, window_id="w2",
                reason_code=ReasonCode.MISSING_FEATURES),
        _result(InferenceStatus.SCORED, window_id="w3", is_anomaly=False, raw_score=-0.3),
        _result(InferenceStatus.INVALID_INPUT, window_id="w4",
                reason_code=ReasonCode.NON_NUMERIC_FEATURE),
        _result(InferenceStatus.INFERENCE_ERROR, window_id="w5",
                reason_code=ReasonCode.INFERENCE_FAILED),
    ]
    outcome = _adapter(results, sink=sink, unscored=collector).handle_windows(
        ["w1", "w2", "w3", "w4", "w5"]
    )
    assert outcome.forwarded_count == 2
    assert outcome.unscored_count == 3
    assert len(sink.results) == 2
    assert {w.window_id for w in outcome.scored} == {"w1", "w3"}
    assert {u.window_id for u in outcome.unscored} == {"w2", "w4", "w5"}


def test_one_failure_does_not_corrupt_unrelated_windows():
    sink = RecordingSink()
    results = [
        _result(InferenceStatus.INVALID_INPUT, window_id="bad",
                reason_code=ReasonCode.INVALID_WINDOW_BOUNDARY),
        _result(InferenceStatus.SCORED, window_id="good", is_anomaly=False, raw_score=-0.3),
    ]
    outcome = _adapter(results, sink=sink).handle_windows(["bad", "good"])
    assert outcome.forwarded_count == 1
    assert outcome.scored[0].window_id == "good"
    assert outcome.scored[0].is_anomaly is False


# ---------------------------------------------------------------------------
# 11 — duplicate window ids follow the existing contract
# ---------------------------------------------------------------------------


@requires_model
def test_duplicate_window_id_follows_the_existing_contract():
    """The later occurrence of a duplicate id is rejected, not forwarded.

    This matches the existing M6 behaviour
    (``AnomalyDetector.score_windows`` rejects the duplicate). No stronger,
    exactly-once or distributed guarantee is claimed: the check is per batch
    and in-process.
    """
    from datetime import datetime, timezone

    from anomaly_detector.inference import load_detector
    from anomaly_detector.inference.contract import FeatureWindowRequest

    real = load_detector(MODEL_PATH, MANIFEST_PATH)
    sink = RecordingSink()
    adapter = EvidenceCorrelationAdapter(real, sink=sink)
    request = FeatureWindowRequest(
        window_id="dup",
        window_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        window_end=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
        features={c: 1.0 for c in real.feature_columns},
    )
    outcome = adapter.handle_windows([request, request])
    # The first occurrence is scored and forwarded; the duplicate is rejected.
    assert outcome.forwarded_count == 1
    assert outcome.unscored_count == 1
    assert outcome.unscored[0].reason_code == ReasonCode.DUPLICATE_WINDOW_ID.value
    assert len(sink.results) == 1


# ---------------------------------------------------------------------------
# 12 & 13 — provenance survives routing
# ---------------------------------------------------------------------------


def test_model_threshold_and_score_direction_survive_routing():
    sink = RecordingSink()
    result = _result(InferenceStatus.SCORED, is_anomaly=True, raw_score=-0.91)
    outcome = _adapter([result], sink=sink).handle_windows(["w1"])
    forwarded = outcome.scored[0]
    assert forwarded.model_name == "sklearn.ensemble.IsolationForest"
    assert forwarded.model_version == "c210b7a"
    assert forwarded.feature_schema_version == "features-v1"
    assert forwarded.threshold == -0.6202666407504878
    assert forwarded.score_direction == "lower_is_more_anomalous"
    assert forwarded.raw_score == -0.91
    assert forwarded.is_anomaly is True
    # The sink sees the full result, including the provenance.
    assert sink.results[0].threshold_config_version


def test_window_identifiers_remain_traceable():
    sink = RecordingSink()
    outcome = _adapter([_result(InferenceStatus.SCORED, window_id="host|svc|2026-01-01T00:00:00+00:00",
                                 is_anomaly=False, raw_score=-0.3)], sink=sink).handle_windows(["x"])
    assert outcome.scored[0].window_id == "host|svc|2026-01-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# 14 — no downstream re-thresholding, no probability interpretation
# ---------------------------------------------------------------------------


def test_forwarded_result_keeps_the_original_threshold_and_score():
    sink = RecordingSink()
    outcome = _adapter([_result(InferenceStatus.SCORED, is_anomaly=False, raw_score=-0.55)],
                       sink=sink).handle_windows(["w1"])
    forwarded = outcome.scored[0]
    # The threshold was applied once, by the ML component.
    assert forwarded.threshold == -0.6202666407504878
    assert forwarded.raw_score == -0.55
    assert forwarded.is_anomaly is (forwarded.raw_score <= forwarded.threshold)
    # The score is explicitly not a probability.
    assert sink.results[0].score_is_probability is False


# ---------------------------------------------------------------------------
# 15 — offline
# ---------------------------------------------------------------------------


def test_harness_package_has_no_network_imports():
    package_dir = os.path.join("src", "anomaly_detector", "inference")
    banned = ("import requests", "import urllib", "import socket", "import http",
              "from fastapi", "import flask", "import uvicorn", "import boto3")
    for name in os.listdir(package_dir):
        if not name.endswith(".py"):
            continue
        text = open(os.path.join(package_dir, name), encoding="utf-8").read()
        for token in banned:
            assert token not in text, f"{name} must not import {token}"


def test_rehearsal_script_runs_offline():
    if not (os.path.isfile(MODEL_PATH) and os.path.isfile(MANIFEST_PATH)):
        pytest.skip("model artifact not present")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(os.getcwd(), "src")
    completed = subprocess.run(
        [sys.executable, "scripts/run_integration_rehearsal.py"],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    assert "every forwarded item scored : True" in completed.stdout
    assert "any unscored reached the sink: False" in completed.stdout


# ---------------------------------------------------------------------------
# 16 — the invariant, stated as one test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("is_anomaly_value", [None, False, True])
def test_unscored_results_never_reach_the_sink_whatever_is_anomaly_says(is_anomaly_value):
    sink = RecordingSink()
    for status in (InferenceStatus.INSUFFICIENT_DATA, InferenceStatus.INVALID_INPUT,
                   InferenceStatus.MODEL_UNAVAILABLE, InferenceStatus.INFERENCE_ERROR):
        outcome = _adapter([_result(status, is_anomaly=is_anomaly_value)], sink=sink).handle_windows(["w"])
        assert outcome.forwarded_count == 0
    assert sink.results == []


def test_the_forwarding_rule_is_centralised_in_one_function():
    """Callers must not implement their own check."""
    from anomaly_detector.inference import contract as contract_module

    source = open(contract_module.__file__, encoding="utf-8").read()
    assert "def route_result(" in source
    assert "def is_forwardable(" in source
    # The adapter uses the canonical rule rather than re-implementing it.
    adapter_source = open(
        os.path.join("src", "anomaly_detector", "inference", "adapter.py"), encoding="utf-8"
    ).read()
    assert "route_result(" in adapter_source
    assert "status is InferenceStatus.SCORED" not in adapter_source


def test_is_forwardable_agrees_with_route_result():
    for status in InferenceStatus:
        result = _result(status, is_anomaly=False)
        assert is_forwardable(result) is route_result(result).forwardable
    assert is_forwardable(None) is False


# ---------------------------------------------------------------------------
# The harness itself
# ---------------------------------------------------------------------------


@requires_model
def test_harness_loads_the_model_once_and_routes():
    harness = IntegrationHarness(model_path=MODEL_PATH, manifest_path=MANIFEST_PATH)
    contract = harness.contract()
    assert contract["model_name"] == "sklearn.ensemble.IsolationForest"
    assert contract["score_direction"] == "lower_is_more_anomalous"
    assert contract["threshold"] == -0.6202666407504878
    # The same detector instance is reused for every call.
    assert harness.detector is harness.adapter._detector


@requires_model
def test_harness_rehearsal_reports_routing_separately_from_detection():
    report = run_rehearsal(IntegrationHarness(model_path=MODEL_PATH, manifest_path=MANIFEST_PATH))
    routing = report["routing"]
    assert routing["every_forwarded_item_was_scored"] is True
    assert routing["any_unscored_item_reached_the_sink"] is False
    assert routing["all_scenarios_routing_ok"] is True
    # Only "scored" may appear in the forwarded statuses.
    assert set(routing["forwarded_statuses"]) == {"scored"}
    # Detection outcomes are reported separately.
    detection = report["ml_detection"]
    assert "detected" in detection and "missed" in detection and "false_positives" in detection


@requires_model
def test_harness_rehearsal_covers_the_expected_scenarios():
    report = run_rehearsal(IntegrationHarness(model_path=MODEL_PATH, manifest_path=MANIFEST_PATH))
    names = {row["scenario"] for row in report["scenarios"]}
    assert names == {
        "healthy_traffic",
        "http_5xx_spike",
        "latency_degradation",
        "database_timeout",
        "missing_latency",
        "malformed_event",
        "duplicate_and_late_events",
        "benign_distribution_shift",
    }


@requires_model
def test_harness_rehearsal_reports_every_window():
    report = run_rehearsal(IntegrationHarness(model_path=MODEL_PATH, manifest_path=MANIFEST_PATH))
    totals = report["totals"]
    # Every window is accounted for: scored + unscored == attempted.
    assert totals["scored"] + totals["unscored"] == totals["windows_attempted"]
    assert totals["forwarded"] + totals["sink_failures"] <= totals["scored"]
    assert totals["sink_failures"] == 0


@requires_model
def test_harness_uses_the_real_inference_entry_point():
    """The harness must not implement a second scoring path."""
    harness = IntegrationHarness(model_path=MODEL_PATH, manifest_path=MANIFEST_PATH)
    assert harness.detector.__class__.__name__ == "AnomalyDetector"
    assert hasattr(harness.detector, "score_window")
    # The adapter delegates to the detector, not to a local copy of the model.
    assert harness.adapter._detector is harness.detector


def test_unscored_collector_preserves_status_and_provenance():
    collector = UnscoredCollector()
    record = UnscoredWindow(
        window_id="host|svc|2026-01-01T00:00:00+00:00",
        status="insufficient_data",
        reason_code="missing_features",
        reason="required feature(s) missing: ['p95_latency_ms']",
    )
    collector(record)
    assert len(collector) == 1
    assert collector.by_status() == {"insufficient_data": 1}
    assert collector.records[0].window_id.startswith("host|svc|")
    assert collector.records[0].reason_code == "missing_features"
