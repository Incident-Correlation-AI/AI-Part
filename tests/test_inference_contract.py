"""Milestone 6 — contract, model-compatibility and failure-mode tests.

These tests exercise the public inference interface exactly as the downstream
component would.  They are offline and deterministic: no network, no
training, no large fixtures.
"""
from datetime import datetime, timedelta, timezone
import json
import os

import pytest

from anomaly_detector.inference import (
    CONTRACT_VERSION,
    DetectorOptions,
    FeatureWindowRequest,
    InferenceStatus,
    ModelUnavailableError,
    ReasonCode,
    load_detector,
    make_window_id,
)
from anomaly_detector.inference.contract import FEATURE_SCHEMA_VERSION

MODEL_PATH = "artifacts/baseline/model/model.joblib"
MANIFEST_PATH = "artifacts/baseline/model/manifest.json"

FEATURES = [
    "event_count",
    "error_count",
    "error_ratio",
    "http_request_count",
    "http_5xx_count",
    "http_5xx_ratio",
    "auth_failure_count",
    "restart_count",
    "mean_latency_ms",
    "p95_latency_ms",
    "latency_sample_count",
    "http_request_eligible_count",
    "error_eligible_count",
]

NORMAL_FEATURES = {
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

ANOMALOUS_FEATURES = {
    "event_count": 400.0,
    "error_count": 300.0,
    "error_ratio": 0.75,
    "http_request_count": 380.0,
    "http_5xx_count": 300.0,
    "http_5xx_ratio": 0.79,
    "auth_failure_count": 5.0,
    "restart_count": 3.0,
    "mean_latency_ms": 2400.0,
    "p95_latency_ms": 6000.0,
    "latency_sample_count": 380.0,
    "http_request_eligible_count": 380.0,
    "error_eligible_count": 400.0,
}

START = datetime(2026, 10, 18, 10, 1, tzinfo=timezone.utc)
END = START + timedelta(minutes=1)


def _request(features=None, **kwargs):
    params = dict(
        window_id="web-01|web-api|2026-10-18T10:01:00+00:00",
        window_start=START,
        window_end=END,
        features=dict(NORMAL_FEATURES if features is None else features),
        host_id="web-01",
        service_name="web-api",
    )
    params.update(kwargs)
    return FeatureWindowRequest(**params)


requires_model = pytest.mark.skipif(
    not (os.path.isfile(MODEL_PATH) and os.path.isfile(MANIFEST_PATH)),
    reason="selected M3 model artifact is not present (run scripts/run_baseline.py run)",
)


@pytest.fixture(scope="module")
def detector():
    return load_detector(MODEL_PATH, MANIFEST_PATH)


# ---------------------------------------------------------------------------
# Loading / compatibility
# ---------------------------------------------------------------------------


@requires_model
def test_load_detector_returns_reusable_component(detector):
    assert detector.feature_columns == FEATURES
    assert detector.score_direction == "lower_is_more_anomalous"
    assert detector.threshold is not None
    contract = detector.contract()
    assert contract["contract_version"] == CONTRACT_VERSION
    assert contract["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert contract["score_is_probability"] is False


@requires_model
def test_missing_model_artifact_raises_explicit_error(tmp_path):
    with pytest.raises(ModelUnavailableError):
        load_detector(str(tmp_path / "nope.joblib"), MANIFEST_PATH)


@requires_model
def test_missing_manifest_raises_explicit_error(tmp_path):
    with pytest.raises(ModelUnavailableError):
        load_detector(MODEL_PATH, str(tmp_path / "nope.json"))


@requires_model
def test_external_model_path_requires_explicit_opt_in(tmp_path):
    outside = tmp_path / "model.joblib"
    outside.write_bytes(b"not-a-model")
    with pytest.raises(ModelUnavailableError) as excinfo:
        load_detector(str(outside), MANIFEST_PATH)
    assert "outside the repository artifacts tree" in str(excinfo.value)


@requires_model
def test_manifest_feature_mismatch_fails_clearly(tmp_path):
    import shutil

    manifest = json.load(open(MANIFEST_PATH))
    manifest["feature_schema"]["feature_columns"] = ["event_count", "error_count"]
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps(manifest))
    shutil.copy(MODEL_PATH, tmp_path / "model.joblib")
    with pytest.raises(ModelUnavailableError) as excinfo:
        load_detector(
            str(tmp_path / "model.joblib"),
            str(bad),
            DetectorOptions(allow_external_model_path=True),
        )
    assert "mismatched schema" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Valid input
# ---------------------------------------------------------------------------


@requires_model
def test_valid_window_produces_documented_result(detector):
    result = detector.score_window(_request())
    assert result.status is InferenceStatus.SCORED
    assert result.reason_code is ReasonCode.OK
    assert result.is_anomaly is False
    assert isinstance(result.raw_score, float)
    assert result.score_direction == "lower_is_more_anomalous"
    assert result.score_is_probability is False
    assert result.contract_version == CONTRACT_VERSION
    payload = result.to_dict()
    assert payload["status"] == "scored"
    assert payload["decision"]["is_anomaly"] is False
    assert payload["model"]["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert payload["model"]["threshold_config_version"]
    json.dumps(payload)  # must be JSON serialisable


@requires_model
def test_anomalous_window_is_flagged(detector):
    result = detector.score_window(_request(ANOMALOUS_FEATURES))
    assert result.status is InferenceStatus.SCORED
    assert result.is_anomaly is True
    # lower score == more anomalous
    assert result.raw_score < detector.score_window(_request()).raw_score


@requires_model
def test_result_carries_window_provenance(detector):
    result = detector.score_window(_request(extra={"tenant": "acme"}))
    assert result.window_id == "web-01|web-api|2026-10-18T10:01:00+00:00"
    assert result.host_id == "web-01"
    assert result.service_name == "web-api"
    assert result.window_start == START.isoformat()
    assert result.window_end == END.isoformat()
    assert result.extra == {"tenant": "acme"}


@requires_model
def test_repeated_calls_are_deterministic(detector):
    first = detector.score_window(_request())
    second = detector.score_window(_request())
    assert first.raw_score == second.raw_score
    assert first.is_anomaly == second.is_anomaly


# ---------------------------------------------------------------------------
# Feature order / schema
# ---------------------------------------------------------------------------


@requires_model
def test_feature_order_is_enforced_not_dict_order(detector):
    shuffled = dict(reversed(list(NORMAL_FEATURES.items())))
    result = detector.score_window(_request(shuffled))
    assert result.status is InferenceStatus.SCORED
    assert result.raw_score == detector.score_window(_request()).raw_score


@requires_model
def test_unknown_feature_is_rejected_by_default(detector):
    features = dict(NORMAL_FEATURES)
    features["secret_feature"] = 1.0
    result = detector.score_window(_request(features))
    assert result.status is InferenceStatus.INVALID_INPUT
    assert result.reason_code is ReasonCode.UNKNOWN_FEATURES
    assert result.is_anomaly is None


@requires_model
def test_unknown_feature_can_be_allowed_and_is_reported(tmp_path):
    detector = load_detector(
        MODEL_PATH, MANIFEST_PATH, DetectorOptions(allow_unknown_features=True)
    )
    features = dict(NORMAL_FEATURES)
    features["secret_feature"] = 1.0
    result = detector.score_window(_request(features))
    assert result.status is InferenceStatus.SCORED
    assert result.ignored_features == ["secret_feature"]
    # the unknown column must not change the score
    assert result.raw_score == detector.score_window(_request()).raw_score


@requires_model
def test_missing_feature_is_insufficient_data(detector):
    features = dict(NORMAL_FEATURES)
    features.pop("p95_latency_ms")
    result = detector.score_window(_request(features))
    assert result.status is InferenceStatus.INSUFFICIENT_DATA
    assert result.reason_code is ReasonCode.MISSING_FEATURES
    assert result.is_anomaly is None
    assert result.raw_score is None


@requires_model
def test_feature_schema_version_mismatch_is_rejected(detector):
    result = detector.score_window(_request(feature_schema_version="features-v99"))
    assert result.status is InferenceStatus.INVALID_INPUT
    assert result.reason_code is ReasonCode.FEATURE_SCHEMA_MISMATCH
    assert result.is_anomaly is None


# ---------------------------------------------------------------------------
# Types, finiteness, boundaries
# ---------------------------------------------------------------------------


@requires_model
def test_non_numeric_feature_is_invalid_input(detector):
    features = dict(NORMAL_FEATURES)
    features["event_count"] = "many"
    result = detector.score_window(_request(features))
    assert result.status is InferenceStatus.INVALID_INPUT
    assert result.reason_code is ReasonCode.NON_NUMERIC_FEATURE
    assert result.is_anomaly is None


@requires_model
def test_infinity_is_rejected_and_never_scored_as_normal(detector):
    features = dict(NORMAL_FEATURES)
    features["mean_latency_ms"] = float("inf")
    result = detector.score_window(_request(features))
    assert result.status is InferenceStatus.INVALID_INPUT
    assert result.reason_code is ReasonCode.NON_FINITE_FEATURE
    assert result.is_anomaly is None
    assert result.raw_score is None


@requires_model
def test_nan_is_missing_not_zero_and_is_reported(detector):
    features = dict(NORMAL_FEATURES)
    features["mean_latency_ms"] = None
    result = detector.score_window(_request(features))
    assert result.status is InferenceStatus.SCORED
    assert "mean_latency_ms" in result.imputed_features
    assert result.is_anomaly is not None
    # imputation must use the training median, not zero
    zero_version = dict(NORMAL_FEATURES)
    zero_version["mean_latency_ms"] = 0.0
    assert result.raw_score != detector.score_window(_request(zero_version)).raw_score


@requires_model
def test_invalid_window_boundary_is_rejected(detector):
    result = detector.score_window(_request(window_end=START))
    assert result.status is InferenceStatus.INVALID_INPUT
    assert result.reason_code is ReasonCode.INVALID_WINDOW_BOUNDARY
    assert result.is_anomaly is None


@requires_model
def test_naive_timestamp_is_rejected(detector):
    result = detector.score_window(
        _request(window_start=datetime(2026, 10, 18, 10, 1), window_end=datetime(2026, 10, 18, 10, 2))
    )
    assert result.status is InferenceStatus.INVALID_INPUT
    assert result.reason_code is ReasonCode.NAIVE_TIMESTAMP
    assert result.is_anomaly is None


@requires_model
def test_missing_window_id_is_rejected(detector):
    result = detector.score_window(_request(window_id="   "))
    assert result.status is InferenceStatus.INVALID_INPUT
    assert result.reason_code is ReasonCode.MISSING_WINDOW_ID
    assert result.is_anomaly is None


# ---------------------------------------------------------------------------
# Labels and forbidden metadata never reach the model
# ---------------------------------------------------------------------------


@requires_model
def test_labels_and_metadata_never_enter_the_feature_vector(detector):
    features = dict(NORMAL_FEATURES)
    # A caller that tries to smuggle a label/identifier in as a feature is
    # rejected, because those names are not in the allowlist.
    for forbidden in ("is_anomaly", "scenario_name", "host_id", "service", "sample_key"):
        polluted = dict(features)
        polluted[forbidden] = 1.0
        result = detector.score_window(_request(polluted))
        assert result.status is InferenceStatus.INVALID_INPUT
        assert result.reason_code is ReasonCode.UNKNOWN_FEATURES

    # Metadata fields are carried as metadata and never as features.
    result = detector.score_window(_request())
    assert result.host_id == "web-01"
    assert result.service_name == "web-api"
    assert "host_id" not in detector.feature_columns
    assert "service" not in detector.feature_columns


@requires_model
def test_missing_window_id_is_not_invented(detector):
    result = detector.score_window(_request(window_id=""))
    assert result.window_id is None
    assert result.status is InferenceStatus.INVALID_INPUT


# ---------------------------------------------------------------------------
# Batch semantics
# ---------------------------------------------------------------------------


@requires_model
def test_batch_does_not_fail_on_one_bad_row(detector):
    bad = _request(window_id="bad", features={"event_count": 1.0})
    good = _request(window_id="good")
    results = detector.score_windows([bad, good])
    assert len(results) == 2
    assert results[0].status is InferenceStatus.INSUFFICIENT_DATA
    assert results[0].is_anomaly is None
    assert results[1].status is InferenceStatus.SCORED
    assert results[1].is_anomaly is not None


@requires_model
def test_duplicate_window_id_in_batch_is_rejected(detector):
    results = detector.score_windows([_request(), _request()])
    assert results[0].status is InferenceStatus.SCORED
    assert results[1].status is InferenceStatus.INVALID_INPUT
    assert results[1].reason_code is ReasonCode.DUPLICATE_WINDOW_ID
    assert results[1].is_anomaly is None


# ---------------------------------------------------------------------------
# Optional policies
# ---------------------------------------------------------------------------


@requires_model
def test_stale_window_policy_is_opt_in():
    detector = load_detector(
        MODEL_PATH,
        MANIFEST_PATH,
        DetectorOptions(stale_after_seconds=60, reference_time=START + timedelta(hours=1)),
    )
    result = detector.score_window(_request())
    assert result.status is InferenceStatus.INSUFFICIENT_DATA
    assert result.reason_code is ReasonCode.STALE_WINDOW
    assert result.is_anomaly is None


@requires_model
def test_max_imputed_fraction_policy_is_opt_in():
    detector = load_detector(MODEL_PATH, MANIFEST_PATH, DetectorOptions(max_imputed_fraction=0.1))
    features = dict(NORMAL_FEATURES)
    features["mean_latency_ms"] = None
    features["p95_latency_ms"] = None
    result = detector.score_window(_request(features))
    assert result.status is InferenceStatus.INSUFFICIENT_DATA
    assert result.is_anomaly is None


# ---------------------------------------------------------------------------
# Contract helpers
# ---------------------------------------------------------------------------


def test_make_window_id_matches_repository_sample_key_format():
    window_id = make_window_id("web-01", "web-api", START)
    assert window_id == "web-01|web-api|2026-10-18T10:01:00+00:00"


def test_request_round_trips_to_dict():
    payload = _request().to_dict()
    assert payload["window_id"].startswith("web-01|web-api|")
    assert payload["features"]["event_count"] == 20.0
    json.dumps(payload)
