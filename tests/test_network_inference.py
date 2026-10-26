"""Network inference boundary: statuses, score semantics, serialization.

The tests load the artifact produced by the real pipeline (three runs, small
forest) through the public ``load_network_detector`` boundary.  They never use
labels as model input and never treat an unscored window as normal.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from anomaly_detector.inference.contract import (
    CONTRACT_VERSION,
    InferenceStatus,
    ReasonCode,
)
from anomaly_detector.network.config import load_network_config
from anomaly_detector.network.features import (
    NETWORK_FEATURE_SCHEMA_VERSION,
    NetworkWindow,
)
from anomaly_detector.network.inference import (
    NetworkDetector,
    NetworkModelUnavailableError,
    load_network_detector,
)
from anomaly_detector.schemas import WindowMetadata


@pytest.fixture(scope="module")
def loaded(network_pipeline_run):
    result, cfg = network_pipeline_run
    detector = load_network_detector(
        model_path=f"{cfg.paths['model_dir']}/model.joblib",
        manifest_path=f"{cfg.paths['model_dir']}/manifest.json",
        rules=cfg.rules,
    )
    return result, cfg, detector


def _sample_window(result) -> NetworkWindow:
    for window in result.windows:
        if window.features["event_count"] >= 10:
            return deepcopy(window)
    raise AssertionError("the simulated dataset contains no suitable window")


def _mutate(window: NetworkWindow, **overrides) -> NetworkWindow:
    clone = deepcopy(window)
    clone.features.update(overrides)
    return clone


# ---------------------------------------------------------------------------
# Contract and serialization
# ---------------------------------------------------------------------------

def test_public_package_exports_the_documented_inference_api():
    from anomaly_detector import network

    assert network.load_network_detector is load_network_detector
    assert network.NetworkDetector is NetworkDetector
    assert network.NetworkModelUnavailableError is NetworkModelUnavailableError
    for name in ("NetworkWindow", "explain_result", "EvidenceBundle"):
        assert name in network.__all__
        assert hasattr(network, name)


def test_contract_version_and_schema_are_extended_additively(loaded):
    _, _, detector = loaded
    contract = detector.contract()
    assert contract["contract_version"] == CONTRACT_VERSION == "1.1.0"
    assert contract["feature_schema_version"] == NETWORK_FEATURE_SCHEMA_VERSION
    assert contract["score_direction"] == "lower_is_more_anomalous"
    assert contract["decision_rule"].startswith("is_anomaly = raw_score <= threshold")
    assert len(contract["feature_columns"]) == 39
    assert contract["feature_columns"] == contract["expected_feature_columns"]


def test_reloaded_model_reproduces_recorded_predictions(loaded):
    result, cfg, detector = loaded
    predictions = result.strategies["time"].predictions.set_index("sample_key")
    for window in result.windows[:150]:
        scored = detector.score_window(window)
        recorded = predictions.loc[window.sample_key]
        assert scored.status.value == recorded["status"]
        if scored.status is InferenceStatus.SCORED:
            assert scored.is_anomaly == bool(recorded["is_anomaly"])
            assert scored.raw_score == pytest.approx(float(recorded["raw_score"]))
            assert scored.contract_version == CONTRACT_VERSION
            assert scored.feature_schema_version == NETWORK_FEATURE_SCHEMA_VERSION


def test_manifest_schema_mismatch_is_refused(loaded, tmp_path):
    _, cfg, _ = loaded
    with open(f"{cfg.paths['model_dir']}/manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest["feature_schema"]["feature_schema_version"] = "features-v1"
    bad_manifest = tmp_path / "manifest.json"
    bad_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(NetworkModelUnavailableError):
        load_network_detector(
            model_path=f"{cfg.paths['model_dir']}/model.joblib",
            manifest_path=str(bad_manifest),
            rules=cfg.rules,
        )


def test_missing_artifact_is_an_explicit_error(tmp_path):
    with pytest.raises(NetworkModelUnavailableError):
        load_network_detector(
            model_path=str(tmp_path / "absent.joblib"),
            manifest_path=str(tmp_path / "absent.json"),
        )


def test_score_semantics_are_model_only(loaded):
    result, _, detector = loaded
    checked = 0
    for window in result.windows[:250]:
        scored = detector.score_window(window)
        if scored.status is not InferenceStatus.SCORED:
            continue
        checked += 1
        assert scored.is_anomaly == (scored.raw_score <= scored.threshold)
        assert scored.score_direction == "lower_is_more_anomalous"
        network = scored.extra["network"]
        assert network["decision_source"] == "isolation_forest"
        assert isinstance(network["baseline"]["is_anomaly"], bool)
        assert network["severity"] in {"none", "low", "medium", "high"}
    assert checked > 0

    predictions = result.strategies["time"].predictions
    scored_predictions = predictions[predictions["status"] == "scored"]
    disagreements = scored_predictions[
        scored_predictions["is_anomaly"].astype(bool)
        != scored_predictions["baseline_is_anomaly"].astype(bool)
    ]
    assert not disagreements.empty, (
        "the rule baseline and the model must be reported independently"
    )


# ---------------------------------------------------------------------------
# Unscored is not normal
# ---------------------------------------------------------------------------

def test_empty_window_is_insufficient_data_not_normal(loaded):
    _, _, detector = loaded
    result, cfg, _ = loaded
    window = _sample_window(result)
    window = _mutate(window, event_count=0.0, invalid_event_count=0.0)
    scored = detector.score_window(window)
    assert scored.status is InferenceStatus.INSUFFICIENT_DATA
    assert scored.reason_code is ReasonCode.NO_EVENTS
    assert scored.is_anomaly is None
    assert scored.raw_score is None


def test_low_volume_window_is_insufficient_coverage(loaded):
    _, cfg, detector = loaded
    result, _, _ = loaded
    window = _mutate(
        _sample_window(result),
        event_count=float(max(1, cfg.rules.min_window_events - 1)),
        invalid_event_count=0.0,
    )
    scored = detector.score_window(window)
    assert scored.status is InferenceStatus.INSUFFICIENT_DATA
    assert scored.reason_code is ReasonCode.INSUFFICIENT_COVERAGE
    assert scored.is_anomaly is None


def test_degraded_telemetry_is_insufficient_data(loaded):
    _, _, detector = loaded
    result, _, _ = loaded
    window = _mutate(_sample_window(result), event_count=5.0, invalid_event_count=10.0)
    scored = detector.score_window(window)
    assert scored.status is InferenceStatus.INSUFFICIENT_DATA
    assert scored.reason_code is ReasonCode.TELEMETRY_QUALITY
    assert scored.is_anomaly is None


def test_naive_timestamp_is_invalid_input(loaded):
    _, _, detector = loaded
    result, _, _ = loaded
    window = _sample_window(result)
    window.metadata.window_start = datetime(2026, 1, 1, 0, 0)
    window.metadata.window_end = datetime(2026, 1, 1, 0, 1)
    scored = detector.score_window(window)
    assert scored.status is InferenceStatus.INVALID_INPUT
    assert scored.reason_code is ReasonCode.NAIVE_TIMESTAMP
    assert scored.is_anomaly is None


def test_invalid_window_boundary_is_rejected(loaded):
    _, _, detector = loaded
    result, _, _ = loaded
    window = _sample_window(result)
    window.metadata.window_end = window.metadata.window_start
    scored = detector.score_window(window)
    assert scored.status is InferenceStatus.INVALID_INPUT
    assert scored.reason_code is ReasonCode.INVALID_WINDOW_BOUNDARY


def test_missing_unknown_and_non_numeric_features_are_rejected(loaded):
    _, _, detector = loaded
    result, _, _ = loaded
    base = _sample_window(result)

    missing = deepcopy(base)
    del missing.features["error_count"]
    scored = detector.score_window(missing)
    assert scored.status is InferenceStatus.INSUFFICIENT_DATA
    assert scored.reason_code is ReasonCode.MISSING_FEATURES

    unknown = _mutate(base, mystery_feature=1.0)
    scored = detector.score_window(unknown)
    assert scored.status is InferenceStatus.INVALID_INPUT
    assert scored.reason_code is ReasonCode.UNKNOWN_FEATURES

    non_numeric = _mutate(base, error_count="many")
    scored = detector.score_window(non_numeric)
    assert scored.status is InferenceStatus.INVALID_INPUT
    assert scored.reason_code is ReasonCode.NON_NUMERIC_FEATURE

    boolean = _mutate(base, restart_count=True)
    scored = detector.score_window(boolean)
    assert scored.status is InferenceStatus.INVALID_INPUT
    assert scored.reason_code is ReasonCode.NON_NUMERIC_FEATURE

    infinite = _mutate(base, mean_latency_ms=float("inf"))
    scored = detector.score_window(infinite)
    assert scored.status is InferenceStatus.INVALID_INPUT
    assert scored.reason_code is ReasonCode.NON_FINITE_FEATURE


def test_duplicate_window_ids_are_reported_not_scored_twice(loaded):
    _, _, detector = loaded
    result, _, _ = loaded
    window = _sample_window(result)
    first, second = detector.score_windows([window, deepcopy(window)])
    assert first.status is InferenceStatus.SCORED
    assert second.status is InferenceStatus.INVALID_INPUT
    assert second.reason_code is ReasonCode.DUPLICATE_WINDOW_ID
    assert second.is_anomaly is None


# ---------------------------------------------------------------------------
# Downstream safety
# ---------------------------------------------------------------------------

def test_result_is_json_safe_and_unscored_is_explicit(loaded):
    _, _, detector = loaded
    result, _, _ = loaded
    scored = detector.score_window(_sample_window(result))
    payload = scored.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["status"] == "scored"
    assert payload["contract_version"] == CONTRACT_VERSION
    assert payload["model"]["feature_schema_version"] == NETWORK_FEATURE_SCHEMA_VERSION
    assert "baseline" in payload["extra"]["network"]

    empty = _mutate(_sample_window(result), event_count=0.0, invalid_event_count=0.0)
    unscored = detector.score_window(empty).to_dict()
    assert unscored["status"] == "insufficient_data"
    assert unscored["reason_code"] == "no_events"
    assert unscored["decision"]["is_anomaly"] is None
    assert unscored["score"]["raw_score"] is None


def test_window_request_keeps_metadata_out_of_features(loaded):
    _, _, _ = loaded
    result, _, _ = loaded
    window = _sample_window(result)
    request = window.to_request()
    assert request.feature_schema_version == NETWORK_FEATURE_SCHEMA_VERSION
    assert request.window_id == window.sample_key
    assert "scenario_name" not in request.features
    assert "is_anomaly" not in request.features
    assert request.host_id == window.metadata.host_id
