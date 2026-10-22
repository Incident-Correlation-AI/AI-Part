"""End-to-end pipeline driver for Milestone 7.

Drives the **real** path, not a parallel implementation:

    raw events
      -> anomaly_detector.validation.validate_and_normalize_events
      -> anomaly_detector.preprocessing.aggregate_events_to_windows
      -> FeatureWindow.to_features_dict()          (features only)
      -> anomaly_detector.inference.AnomalyDetector.score_window
      -> anomaly_detector.inference.adapter (neutral downstream sink)

Ground truth is used by the harness only, never as a model input.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .e2e_scenarios import E2EScenario, all_scenarios
from .features import windows_to_feature_dataframe, windows_to_metadata_dataframe
from .inference import (
    EvidenceCorrelationAdapter,
    InferenceStatus,
    RecordingSink,
    load_detector,
)
from .inference.contract import FeatureWindowRequest, make_window_id
from .preprocessing import aggregate_events_to_windows
from .validation import validate_and_normalize_events


@dataclass
class E2EScenarioResult:
    """Outcome of one scenario through the real pipeline."""

    scenario: str
    description: str
    expected_behavior: str
    benign: bool
    n_events: int
    n_accepted: int
    n_rejected: int
    n_duplicated: int
    n_partially_usable: int
    n_windows: int
    flagged: int
    detected: int
    missed: int
    false_positives: int
    unscored: int
    imputed_windows: int
    imputed_features: Dict[str, int] = field(default_factory=dict)
    window_rows: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "description": self.description,
            "expected_behavior": self.expected_behavior,
            "benign": self.benign,
            "n_events": self.n_events,
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "n_duplicated": self.n_duplicated,
            "n_partially_usable": self.n_partially_usable,
            "n_windows": self.n_windows,
            "flagged": self.flagged,
            "detected": self.detected,
            "missed": self.missed,
            "false_positives": self.false_positives,
            "unscored": self.unscored,
            "imputed_windows": self.imputed_windows,
            "imputed_features": dict(self.imputed_features),
            "window_rows": self.window_rows,
            "notes": list(self.notes),
        }


def run_scenario(scenario: E2EScenario, detector) -> E2EScenarioResult:
    """Run one scenario through the real pipeline and score the outcome."""
    # 1. Normalize / validate the raw events (M1).
    validation = validate_and_normalize_events(scenario.events)

    # 2. Aggregate into feature windows (M1).
    windows = aggregate_events_to_windows(validation.accepted, window_size_minutes=1)

    # 3. Build the feature table exactly as the M1 pipeline does.
    features = windows_to_feature_dataframe(windows)
    metadata = windows_to_metadata_dataframe(windows)

    # 3b. Build one inference request per window, from the M1 feature window.
    requests: List[FeatureWindowRequest] = []
    for i in range(len(windows)):
        window = windows[i]
        requests.append(
            FeatureWindowRequest(
                window_id=make_window_id(
                    window.metadata.host_id, window.metadata.service, window.metadata.window_start
                ),
                window_start=window.metadata.window_start,
                window_end=window.metadata.window_end,
                features=dict(window.to_features_dict()),
                host_id=window.metadata.host_id,
                service_name=window.metadata.service,
            )
        )

    # 4. Score through the M6 inference interface.
    results = detector.score_windows(requests)

    # 4b. Drive the same windows through the neutral downstream boundary and
    #     assert that only scored windows ever reach the sink.
    sink = RecordingSink()
    adapter = EvidenceCorrelationAdapter(detector, sink)
    outcome = adapter.handle_windows(requests)
    assert all(r.status is InferenceStatus.SCORED for r in sink.results), (
        "an unscored result reached the downstream sink"
    )
    assert outcome.forwarded_count == sum(
        1 for r in results if r.status is InferenceStatus.SCORED
    ), "the adapter forwarded a different number of windows than were scored"

    # 5. Tally the outcome against the scenario's ground truth.
    anomalous = set(scenario.anomalous_windows)
    detected = 0
    missed = 0
    false_positives = 0
    imputed_windows = 0
    imputed_features: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    flagged_total = 0

    for index, result in enumerate(results):
        if result.status is not InferenceStatus.SCORED:
            rows.append({
                "window_index": index,
                "window_id": result.window_id,
                "status": result.status.value,
                "raw_score": None,
                "is_anomaly": None,
                "imputed_features": [],
                "reason_code": result.reason_code.value,
            })
            continue
        is_anomaly = bool(result.is_anomaly)
        flagged_total += int(is_anomaly)
        if index in anomalous:
            detected += int(is_anomaly)
            missed += int(not is_anomaly)
        else:
            false_positives += int(is_anomaly)
        if result.imputed_features:
            imputed_windows += 1
            for feature in result.imputed_features:
                imputed_features[feature] = imputed_features.get(feature, 0) + 1
        rows.append({
            "window_index": index,
            "window_id": result.window_id,
            "status": "scored",
            "raw_score": result.raw_score,
            "is_anomaly": is_anomaly,
            "imputed_features": list(result.imputed_features),
            "host_id": result.host_id,
            "service": result.service_name,
            "window_start": result.window_start,
        })

    # The downstream sink must only ever have seen scored windows.
    assert all(r.status is InferenceStatus.SCORED for r in sink.results), (
        "an unscored result reached the downstream sink"
    )

    return E2EScenarioResult(
        scenario=scenario.name,
        description=scenario.description,
        expected_behavior=scenario.expected_behavior,
        benign=scenario.benign,
        n_events=len(scenario.events),
        n_accepted=len(validation.accepted),
        n_rejected=len(validation.rejected),
        n_duplicated=len(validation.duplicated),
        n_partially_usable=len(validation.partially_usable),
        n_windows=len(windows),
        flagged=flagged_total,
        detected=detected,
        missed=missed,
        false_positives=false_positives,
        unscored=outcome.rejected_count,
        imputed_windows=imputed_windows,
        imputed_features=imputed_features,
        window_rows=rows,
        notes=list(scenario.notes),
    )


def run_end_to_end(detector=None, scenarios: Optional[List[E2EScenario]] = None) -> Dict[str, Any]:
    """Run every scenario through the real pipeline.

    Returns a summary that reports actual detections, misses, false positives
    and abstentions. It does **not** assert that every injected scenario
    must be detected.
    """
    if detector is None:
        detector = load_detector()
    scenarios = scenarios if scenarios is not None else all_scenarios()

    results = [run_scenario(scenario, detector) for scenario in scenarios]

    anomalous_scenarios = [r for r in results if not r.benign]
    benign_scenarios = [r for r in results if r.benign]

    total_detected = sum(r.detected for r in results)
    total_missed = sum(r.missed for r in results)
    total_fp = sum(r.false_positives for r in results)
    total_unscored = sum(r.unscored for r in results)

    # Benign-scenario flag rate: the false-alert burden of the detector.
    benign_windows = sum(r.n_windows for r in benign_scenarios)
    benign_flagged = sum(int(r.flagged or 0) for r in benign_scenarios)
    benign_flag_rate = benign_flagged / benign_windows if benign_windows else 0.0

    shift = next((r for r in results if r.scenario == "benign_distribution_shift"), None)
    missing = next((r for r in results if r.scenario == "missing_latency"), None)

    return {
        "pipeline": (
            "raw events -> validate_and_normalize_events -> aggregate_events_to_windows "
            "-> FeatureWindow.to_features_dict -> inference.score_window "
            "-> EvidenceCorrelationAdapter -> neutral sink"
        ),
        "scenarios": [r.to_dict() for r in results],
        "totals": {
            "scenarios": len(results),
            "windows": sum(r.n_windows for r in results),
            "flagged": sum(int(r.flagged or 0) for r in results),
            "detected": total_detected,
            "missed": total_missed,
            "false_positives": total_fp,
            "unscored": total_unscored,
            "imputed_windows": sum(r.imputed_windows for r in results),
        },
        "anomalous_scenarios": len(anomalous_scenarios),
        "benign_scenarios": len(benign_scenarios),
        "benign_flag_rate": benign_flag_rate,
        "benign_shift_flag_rate": (
            (int(shift.flagged or 0) / shift.n_windows) if shift and shift.n_windows else None
        ),
        "missing_telemetry_flag_rate": (
            (int(missing.flagged or 0) / missing.n_windows) if missing and missing.n_windows else None
        ),
        "notes": [
            "Scenario ground truth is used by the harness only; it never enters the "
            "model feature vector.",
            "A high flag rate on a benign scenario is a false-alert risk, not a detection.",
            "These are synthetic scenarios; they validate the pipeline, not real-world "
            "detection quality.",
        ],
    }
