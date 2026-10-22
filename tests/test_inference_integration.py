"""Milestone 6 — integration-boundary and end-to-end tests.

These tests exercise the path a real window takes: existing M1/M2 feature
output -> inference interface -> downstream boundary.  They also pin the
ownership rules: the ML component scores, the downstream component reasons.
"""
from datetime import datetime, timedelta, timezone
import os
import subprocess
import sys

import pandas as pd
import pytest

from anomaly_detector.inference import (
    EvidenceCorrelationAdapter,
    InferenceStatus,
    RecordingSink,
    ScoredWindow,
    load_detector,
)
from anomaly_detector.inference.adapter import RejectedWindow
from anomaly_detector.inference.contract import FeatureWindowRequest

MODEL_PATH = "artifacts/baseline/model/model.joblib"
MANIFEST_PATH = "artifacts/baseline/model/manifest.json"
SPLITS_DIR = "data/processed/splits"
STRATEGY = "random"

requires_model = pytest.mark.skipif(
    not (os.path.isfile(MODEL_PATH) and os.path.isfile(MANIFEST_PATH)),
    reason="selected M3 model artifact is not present (run scripts/run_baseline.py run)",
)
requires_splits = pytest.mark.skipif(
    not os.path.isdir(os.path.join(SPLITS_DIR, STRATEGY, "test")),
    reason="processed splits are not present (run scripts/prepare_experiments.py)",
)


@pytest.fixture(scope="module")
def detector():
    return load_detector(MODEL_PATH, MANIFEST_PATH)


def _request_from_split(split: str, index: int = 0) -> FeatureWindowRequest:
    base = os.path.join(SPLITS_DIR, STRATEGY, split)
    features = pd.read_csv(os.path.join(base, "features.csv"))
    metadata = pd.read_csv(os.path.join(base, "metadata.csv"))
    row = features.iloc[index]
    meta = metadata.iloc[index]
    return FeatureWindowRequest(
        window_id=str(meta["sample_key"]),
        window_start=datetime.fromisoformat(str(meta["window_start"])),
        window_end=datetime.fromisoformat(str(meta["window_end"])),
        features={c: row[c] for c in features.columns},
        host_id=str(meta.get("host_id", "")) or None,
        service_name=str(meta.get("service", "")) or None,
    )


# ---------------------------------------------------------------------------
# Downstream boundary
# ---------------------------------------------------------------------------


@requires_model
def test_adapter_forwards_only_scored_windows(detector):
    sink = RecordingSink()
    adapter = EvidenceCorrelationAdapter(detector, sink)
    outcome = adapter.handle_windows(
        [
            _request_from_split("test", 0) if os.path.isdir(os.path.join(SPLITS_DIR, STRATEGY, "test"))
            else _synthetic_request("ok"),
            _synthetic_request("broken", features={"event_count": 1.0}),
        ]
    )
    assert outcome.forwarded_count == 1
    assert outcome.rejected_count == 1
    assert isinstance(outcome.rejected[0], RejectedWindow)
    assert outcome.rejected[0].status == InferenceStatus.INSUFFICIENT_DATA.value
    # only the scored window reached the sink
    assert len(sink.results) == 1
    assert sink.results[0].status is InferenceStatus.SCORED


@requires_model
def test_adapter_never_forwards_an_unscored_window_as_normal(detector):
    sink = RecordingSink()
    adapter = EvidenceCorrelationAdapter(detector, sink)
    forwarded = adapter.handle_window(_synthetic_request("broken", features={"event_count": 1.0}))
    assert forwarded is None
    assert sink.results == []


@requires_model
def test_scored_window_refuses_to_build_from_a_failed_result(detector):
    result = detector.score_window(_synthetic_request("broken", features={"event_count": 1.0}))
    with pytest.raises(ValueError):
        ScoredWindow.from_result(result)


@requires_model
def test_window_id_and_metadata_survive_the_adapter(detector):
    sink = RecordingSink()
    adapter = EvidenceCorrelationAdapter(detector, sink)
    request = _synthetic_request("meta", host_id="db-07", service_name="postgres")
    outcome = adapter.handle_windows([request])
    assert outcome.scored[0].window_id == request.window_id
    assert outcome.scored[0].host_id == "db-07"
    assert outcome.scored[0].service_name == "postgres"
    assert outcome.scored[0].window_start == request.window_start.isoformat()
    assert outcome.scored[0].window_end == request.window_end.isoformat()
    assert sink.results[0].window_id == request.window_id


@requires_model
def test_scored_normal_and_scored_anomalous_are_distinguishable(detector):
    normal = detector.score_window(_synthetic_request("n"))
    anomalous = detector.score_window(_synthetic_request("a", anomalous=True))
    assert normal.status is anomalous.status is InferenceStatus.SCORED
    assert normal.is_anomaly is False
    assert anomalous.is_anomaly is True
    assert anomalous.raw_score < normal.raw_score


@requires_model
def test_score_is_not_represented_as_a_probability(detector):
    result = detector.score_window(_synthetic_request("n"))
    payload = result.to_dict()
    assert payload["score"]["is_probability"] is False
    # IsolationForest.score_samples lives in (-1, 0]; it is a ranking
    # statistic, not a probability, and it keeps the model's own sign.
    assert -1.0 < payload["score"]["raw_score"] <= 0.0
    assert payload["score"]["score_direction"] == "lower_is_more_anomalous"
    # the score is deliberately NOT the sign-flipped convenience variant
    assert payload["score"]["raw_score"] < 0.0


@requires_model
def test_threshold_decision_uses_the_manifest_threshold(detector):
    result = detector.score_window(_synthetic_request("n"))
    assert result.threshold == detector.threshold
    expected = result.raw_score <= detector.threshold
    assert result.is_anomaly is expected


# ---------------------------------------------------------------------------
# End to end: existing feature output -> inference -> boundary
# ---------------------------------------------------------------------------


@requires_model
@requires_splits
def test_representative_window_travels_from_feature_output_to_boundary(detector):
    sink = RecordingSink()
    adapter = EvidenceCorrelationAdapter(detector, sink)
    request = _request_from_split("test", 0)
    outcome = adapter.handle_windows([request])
    assert outcome.forwarded_count == 1
    assert outcome.rejected_count == 0
    scored = outcome.scored[0]
    assert scored.window_id == request.window_id
    assert scored.model_name == "sklearn.ensemble.IsolationForest"
    assert scored.feature_schema_version == "features-v1"
    assert isinstance(scored.raw_score, float)


@requires_model
@requires_splits
def test_whole_test_split_scores_without_error(detector):
    base = os.path.join(SPLITS_DIR, STRATEGY, "test")
    features = pd.read_csv(os.path.join(base, "features.csv"))
    metadata = pd.read_csv(os.path.join(base, "metadata.csv"))
    requests = [
        FeatureWindowRequest(
            window_id=str(meta["sample_key"]),
            window_start=datetime.fromisoformat(str(meta["window_start"])),
            window_end=datetime.fromisoformat(str(meta["window_end"])),
            features={c: row[c] for c in features.columns},
            host_id=str(meta.get("host_id", "")) or None,
            service_name=str(meta.get("service", "")) or None,
        )
        for (_, row), (_, meta) in zip(features.iterrows(), metadata.iterrows())
    ]
    results = detector.score_windows(requests)
    assert len(results) == len(requests)
    assert all(r.status is InferenceStatus.SCORED for r in results)
    assert all(r.is_anomaly is not None for r in results)


# ---------------------------------------------------------------------------
# Usability / regression
# ---------------------------------------------------------------------------


def test_smoke_test_script_runs_offline():
    """The documented smoke test must work without network access."""
    if not (os.path.isfile(MODEL_PATH) and os.path.isfile(MANIFEST_PATH)):
        pytest.skip("model artifact not present")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(os.getcwd(), "src")
    env["NO_PROXY"] = "*"
    completed = subprocess.run(
        [sys.executable, "scripts/smoke_test_inference.py", "--all"],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    assert "contract_version" in completed.stdout
    assert "scored" in completed.stdout


def test_inference_package_has_no_network_imports():
    """The inference package must not pull in a web/network stack."""
    package_dir = os.path.join("src", "anomaly_detector", "inference")
    banned = ("import requests", "import urllib", "import socket", "import http",
              "from fastapi", "import flask", "import uvicorn")
    for name in os.listdir(package_dir):
        if not name.endswith(".py"):
            continue
        text = open(os.path.join(package_dir, name), encoding="utf-8").read()
        for token in banned:
            assert token not in text, f"{name} must not import {token}"


def _synthetic_request(
    window_id: str,
    features=None,
    anomalous: bool = False,
    host_id: str = "web-01",
    service_name: str = "web-api",
) -> FeatureWindowRequest:
    start = datetime(2026, 10, 18, 10, 1, tzinfo=timezone.utc)
    base = {
        "event_count": 20.0,
        "error_count": 1.0,
        "error_ratio": 0.05,
        "http_request_count": 20.0,
        "http_5xx_count": 0.0,
        "http_5xx_ratio": 0.0,
        "auth_failure_count": 0.0,
        "restart_count": 0.0,
        "mean_latency_ms": 52.0,
        "p95_latency_ms": 70.0,
        "latency_sample_count": 20.0,
        "http_request_eligible_count": 20.0,
        "error_eligible_count": 20.0,
    }
    if anomalous:
        base.update({
            "event_count": 500.0,
            "error_count": 380.0,
            "error_ratio": 0.76,
            "http_request_count": 470.0,
            "http_5xx_count": 380.0,
            "http_5xx_ratio": 0.81,
            "auth_failure_count": 9.0,
            "restart_count": 4.0,
            "mean_latency_ms": 3100.0,
            "p95_latency_ms": 7000.0,
            "latency_sample_count": 470.0,
            "http_request_eligible_count": 470.0,
            "error_eligible_count": 500.0,
        })
    if features is not None:
        base = dict(features)
    return FeatureWindowRequest(
        window_id=window_id,
        window_start=start,
        window_end=start + timedelta(minutes=1),
        features=base,
        host_id=host_id,
        service_name=service_name,
    )
