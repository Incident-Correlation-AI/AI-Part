"""Milestone 8 — local integration harness for the ML → downstream boundary.

The downstream evidence-correlation / incident-explanation component is being
built by another teammate and is **not available yet**. This module prepares
and tests the boundary without implementing that component or depending on its
code.

What this harness is
--------------------
* A thin, offline, in-process wrapper around the M6 inference interface.
* It loads the selected model **once** per session.
* It applies the **canonical** forwarding rule
  (:func:`~anomaly_detector.inference.contract.route_result`) — the single
  place where "may this be forwarded?" is decided.
* It routes scored results to a neutral sink and unscored results to a
  separate, explicit handler that preserves the status and safe provenance.

What this harness is **not**
----------------------------
* Not an API, service, queue, dashboard, cloud deployment or alerting system.
* Not a correlation, grouping or explanation component.
* Not a second scoring implementation: it uses the real M6 entry point and
  the real M3 artifact.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import time

from .e2e_scenarios import E2EScenario, all_scenarios
from .inference import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_MODEL_PATH,
    AdapterOutcome,
    EvidenceCorrelationAdapter,
    InferenceResult,
    InferenceStatus,
    RecordingSink,
    RoutingDecision,
    UnscoredWindow,
    load_detector,
    route_result,
)
from .inference.contract import FeatureWindowRequest, make_window_id
from .preprocessing import aggregate_events_to_windows
from .validation import validate_and_normalize_events


class UnscoredCollector:
    """Explicit handler for windows that were **not** forwarded.

    It preserves the status, the reason code and the safe provenance of every
    unscored outcome, so the caller can distinguish *insufficient data* from
    *invalid input* from *model unavailable* from *inference error* — and
    from a malformed result that has no status at all.
    """

    def __init__(self) -> None:
        self.records: List[UnscoredWindow] = []

    def __call__(self, record: UnscoredWindow) -> None:
        self.records.append(record)

    def by_status(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self.records:
            counts[record.status] = counts.get(record.status, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self.records)


class IntegrationHarness:
    """Local, offline harness for the ML → downstream boundary.

    Usage::

        harness = IntegrationHarness()          # loads the model once
        harness.route_windows(requests)          # -> AdapterOutcome
        harness.unscored.records                 # every unscored window
        harness.sink.results                     # every forwarded window
    """

    def __init__(
        self,
        detector: Any = None,
        sink: Any = None,
        unscored_handler: Any = None,
        model_path: str = DEFAULT_MODEL_PATH,
        manifest_path: str = DEFAULT_MANIFEST_PATH,
    ) -> None:
        # The model is loaded once per harness/session and reused.
        self.detector = detector if detector is not None else load_detector(model_path, manifest_path)
        self.sink = sink if sink is not None else RecordingSink()
        self.unscored = unscored_handler if unscored_handler is not None else UnscoredCollector()
        self.adapter = EvidenceCorrelationAdapter(
            self.detector, sink=self.sink, unscored_handler=self.unscored
        )

    # -- routing ---------------------------------------------------------
    def route_window(self, request: FeatureWindowRequest) -> RoutingDecision:
        """Score one window and return the routing decision.

        This does **not** forward anything; use :meth:`route_windows` for
        that. It exists so a caller can ask "would this be forwarded?" using
        exactly the same rule the adapter uses.
        """
        return route_result(self.detector.score_window(request))

    def route_windows(self, requests: Sequence[FeatureWindowRequest]) -> AdapterOutcome:
        """Score a batch and route each outcome to the sink or the handler."""
        return self.adapter.handle_windows(requests)

    # -- introspection ---------------------------------------------------
    def contract(self) -> Dict[str, Any]:
        """The bound model's contract, for provenance reporting."""
        return self.detector.contract()

    def forwarded_statuses(self) -> Dict[str, int]:
        """Status counts of everything that reached the sink.

        This must contain only ``scored`` — it is the evidence that nothing
        else leaked into the correlation path.
        """
        counts: Dict[str, int] = {}
        for result in getattr(self.sink, "results", []):
            status = getattr(result, "status", None)
            key = getattr(status, "value", None) or str(status)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def sink_received_only_scored(self) -> bool:
        """True when every item that reached the sink had status ``scored``."""
        return all(
            getattr(result, "status", None) is InferenceStatus.SCORED
            for result in getattr(self.sink, "results", [])
        )


# ---------------------------------------------------------------------------
# Phase 5 — controlled local rehearsal
# ---------------------------------------------------------------------------


@dataclass
class RehearsalScenarioResult:
    """Routing and detection outcome for one rehearsal scenario.

    Contract-routing success and ML detection success are reported
    **separately**: a scenario can be routed perfectly and still not be
    detected (or vice versa).
    """

    scenario: str
    description: str
    expected_behavior: str
    benign: bool
    n_events: int
    n_accepted: int
    n_rejected_events: int
    n_duplicated_events: int
    n_windows: int
    scored: int
    forwarded: int
    unscored: int
    unscored_by_status: Dict[str, int]
    sink_failures: int
    flagged: int
    detected: int
    missed: int
    false_positives: int
    duplicate_window_ids: int
    routing_ok: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "description": self.description,
            "expected_behavior": self.expected_behavior,
            "benign": self.benign,
            "n_events": self.n_events,
            "n_accepted": self.n_accepted,
            "n_rejected_events": self.n_rejected_events,
            "n_duplicated_events": self.n_duplicated_events,
            "n_windows": self.n_windows,
            "scored": self.scored,
            "forwarded": self.forwarded,
            "unscored": self.unscored,
            "unscored_by_status": dict(self.unscored_by_status),
            "sink_failures": self.sink_failures,
            "flagged": self.flagged,
            "detected": self.detected,
            "missed": self.missed,
            "false_positives": self.false_positives,
            "duplicate_window_ids": self.duplicate_window_ids,
            "routing_ok": self.routing_ok,
            "notes": list(self.notes),
        }


def _requests_from_events(events: List[Dict[str, Any]]):
    """Run the real M1 path and build one inference request per window."""
    validation = validate_and_normalize_events(events)
    windows = aggregate_events_to_windows(validation.accepted, window_size_minutes=1)
    requests = [
        FeatureWindowRequest(
            window_id=make_window_id(
                w.metadata.host_id, w.metadata.service, w.metadata.window_start
            ),
            window_start=w.metadata.window_start,
            window_end=w.metadata.window_end,
            features=dict(w.to_features_dict()),
            host_id=w.metadata.host_id,
            service_name=w.metadata.service,
        )
        for w in windows
    ]
    return validation, windows, requests


def run_rehearsal(
    harness: Optional[IntegrationHarness] = None,
    scenarios: Optional[List[E2EScenario]] = None,
) -> Dict[str, Any]:
    """Run the offline rehearsal over the real M1 → M6 path.

    Reports routing counts and detection outcomes separately, and records
    misses and false positives honestly.
    """
    harness = harness or IntegrationHarness()
    scenarios = scenarios if scenarios is not None else all_scenarios()

    started = time.perf_counter()
    scenario_results: List[RehearsalScenarioResult] = []

    for scenario in scenarios:
        validation, windows, requests = _requests_from_events(scenario.events)
        outcome = harness.route_windows(requests)

        anomalous = set(scenario.anomalous_windows)
        detected = missed = false_positives = 0
        for index, scored in enumerate(outcome.scored):
            if index in anomalous:
                detected += int(scored.is_anomaly)
                missed += int(not scored.is_anomaly)
            else:
                false_positives += int(scored.is_anomaly)

        # Duplicate window ids inside the batch (the M6 contract rejects the
        # later occurrence).
        ids = [r.window_id for r in requests]
        duplicates = len(ids) - len(set(ids))

        # Contract-routing check: everything that reached the sink was scored.
        sink_results = getattr(harness.sink, "results", [])
        routed_ok = all(
            getattr(r, "status", None) is InferenceStatus.SCORED for r in sink_results
        )
        forwarded_now = len(outcome.scored)
        scored_now = forwarded_now + outcome.sink_failure_count
        routing_ok = routed_ok and (outcome.unscored_count + forwarded_now + outcome.sink_failure_count == len(windows))

        scenario_results.append(RehearsalScenarioResult(
            scenario=scenario.name,
            description=scenario.description,
            expected_behavior=scenario.expected_behavior,
            benign=scenario.benign,
            n_events=len(scenario.events),
            n_accepted=len(validation.accepted),
            n_rejected_events=len(validation.rejected),
            n_duplicated_events=len(validation.duplicated),
            n_windows=len(windows),
            scored=scored_now,
            forwarded=forwarded_now,
            unscored=outcome.unscored_count,
            unscored_by_status=outcome.unscored_by_status(),
            sink_failures=outcome.sink_failure_count,
            flagged=sum(1 for s in outcome.scored if s.is_anomaly),
            detected=detected,
            missed=missed,
            false_positives=false_positives,
            duplicate_window_ids=duplicates,
            routing_ok=routing_ok,
            notes=list(scenario.notes),
        ))

    elapsed = time.perf_counter() - started
    all_forwarded_scored = harness.sink_received_only_scored()
    unscored_reached_sink = not all_forwarded_scored

    totals = {
        "scenarios": len(scenario_results),
        "events_attempted": sum(s.n_events for s in scenario_results),
        "events_accepted": sum(s.n_accepted for s in scenario_results),
        "events_rejected": sum(s.n_rejected_events for s in scenario_results),
        "events_duplicated": sum(s.n_duplicated_events for s in scenario_results),
        "windows_attempted": sum(s.n_windows for s in scenario_results),
        "scored": sum(s.scored for s in scenario_results),
        "forwarded": sum(s.forwarded for s in scenario_results),
        "unscored": sum(s.unscored for s in scenario_results),
        "sink_failures": sum(s.sink_failures for s in scenario_results),
        "flagged": sum(s.flagged for s in scenario_results),
        "detected": sum(s.detected for s in scenario_results),
        "missed": sum(s.missed for s in scenario_results),
        "false_positives": sum(s.false_positives for s in scenario_results),
        "duplicate_window_ids": sum(s.duplicate_window_ids for s in scenario_results),
    }

    unscored_by_status: Dict[str, int] = {}
    for record in harness.unscored.records:
        unscored_by_status[record.status] = unscored_by_status.get(record.status, 0) + 1

    return {
        "harness": {
            "model_name": harness.contract().get("model_name"),
            "model_version": harness.contract().get("model_version"),
            "feature_schema_version": harness.contract().get("feature_schema_version"),
            "threshold": harness.contract().get("threshold"),
            "threshold_rule": harness.contract().get("threshold_rule"),
            "score_direction": harness.contract().get("score_direction"),
            "contract_version": harness.contract().get("contract_version"),
        },
        "routing": {
            "canonical_rule": "anomaly_detector.inference.contract.route_result",
            "forwardable_status": "scored",
            "unscored_by_status": unscored_by_status,
            "forwarded_statuses": harness.forwarded_statuses(),
            "every_forwarded_item_was_scored": all_forwarded_scored,
            "any_unscored_item_reached_the_sink": unscored_reached_sink,
            "all_scenarios_routing_ok": all(s.routing_ok for s in scenario_results),
        },
        "contract_routing": {
            "note": (
                "Contract-routing success and ML detection success are reported "
                "separately. A scenario can be routed perfectly and still not be "
                "detected."
            ),
            "routing_ok": all(s.routing_ok for s in scenario_results),
            "forwarded_all_scored": all_forwarded_scored,
            "unscored_reached_sink": unscored_reached_sink,
        },
        "ml_detection": {
            "note": "Detection outcomes are synthetic-only. They validate the pipeline, "
                    "not real-world detection accuracy.",
            "detected": totals["detected"],
            "missed": totals["missed"],
            "false_positives": totals["false_positives"],
            "flagged": totals["flagged"],
        },
        "totals": totals,
        "scenarios": [s.to_dict() for s in scenario_results],
        "runtime_seconds": round(elapsed, 3),
    }
