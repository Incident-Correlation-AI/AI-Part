"""Contract and behaviour tests for the evidence-correlation adapter.

All providers are local/injected.  No test contacts a network, a cloud service
or a live LLM.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import uuid

import pytest
from jsonschema import validate as validate_json_schema

from anomaly_detector.correlation import (
    CorrelationConfig,
    EvidenceCorrelationPipeline,
    WindowEvidence,
    build_explanation_provider,
    is_safe_event_id,
    is_safe_identity,
    is_safe_redacted_ref,
    load_correlation_config,
)
from anomaly_detector.correlation import pipeline as pipeline_module
from anomaly_detector.inference.contract import (
    CONTRACT_VERSION,
    InferenceResult,
    InferenceStatus,
    ReasonCode,
)
from anomaly_detector.network.llm import (
    DraftExplanation,
    HttpExplanationProvider,
)

BASE = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
SCHEMA = json.loads(
    (Path(__file__).parent / "fixtures" / "contracts" / "incident.schema.json").read_text()
)


def _result(
    window_id: str = "host-1|web-api|2026-01-01T10:00:00+00:00",
    *,
    service: str = "web-api",
    host: str = "host-1",
    start: datetime = BASE,
    end: datetime | None = None,
    is_anomaly: bool | None = True,
    status: InferenceStatus = InferenceStatus.SCORED,
    reason_code: ReasonCode = ReasonCode.OK,
    raw_score: float | None = -0.62,
    threshold: float | None = 0.5,
    severity: str = "high",
    rule_code: str = "http_5xx_rate",
    extra: dict | None = None,
) -> InferenceResult:
    end = end or (start + timedelta(minutes=1))
    network = {
        "severity": severity,
        "decision_source": "isolation_forest" if status is InferenceStatus.SCORED else "none",
        "baseline": {
            "is_anomaly": True,
            "risk_score": 1.0,
            "severity": severity,
            "rule_hits": [
                {
                    "code": rule_code,
                    "strength": "strong",
                    "weight": 1.0,
                    "observed": {"http_5xx_ratio": 0.4},
                    "threshold": {"ratio": 0.05},
                    "message": "HTTP 5xx rate exceeds the configured threshold",
                }
            ],
        },
        "evidence_features": {
            "event_count": 25.0,
            "http_5xx_ratio": 0.4,
            "http_5xx_count": 10.0,
            "p95_latency_ms": 220.0,
        },
        "telemetry_quality": {"duplicate_event_id": 0},
    }
    payload = dict(extra or {})
    if "network" in payload:
        network.update(payload.pop("network"))
    payload["network"] = network
    return InferenceResult(
        contract_version=CONTRACT_VERSION,
        status=status,
        reason_code=reason_code,
        reason="scored" if status is InferenceStatus.SCORED else "not scored",
        window_id=window_id,
        window_start=start.isoformat(),
        window_end=end.isoformat(),
        host_id=host,
        service_name=service,
        model_name="sklearn.ensemble.IsolationForest",
        model_version="rev-123",
        model_artifact_version="artifact-7",
        feature_schema_version="network-features-v1",
        threshold_config_version="config@rev-123:0.5",
        raw_score=raw_score if status is InferenceStatus.SCORED else None,
        score_direction="lower_is_more_anomalous",
        threshold=threshold if status is InferenceStatus.SCORED else None,
        is_anomaly=is_anomaly if status is InferenceStatus.SCORED else None,
        extra=payload,
    )


def _evidence(window_id: str, *event_ids: str, **kwargs) -> dict:
    return {
        window_id: WindowEvidence(
            window_id=window_id,
            supporting_event_ids=tuple(event_ids or (f"evt-{abs(hash(window_id)) % 10000}",)),
            contradicting_event_ids=kwargs.get("contradicting", ()),
            redacted_refs=kwargs.get("redacted", ()),
        )
    }


def _valid_draft(bundle) -> DraftExplanation:
    return DraftExplanation(
        summary=(
            f"The Isolation Forest flagged this window: raw score "
            f"{bundle.model_raw_score:.4f} against threshold {bundle.threshold:.4f}."
        ),
        evidence=[f"http_5xx_ratio = {bundle.features.get('http_5xx_ratio', 0.0):.4f}"],
        recommended_checks=["Check the upstream HTTP dependency."],
        citations=["model_score", "threshold", "http_5xx_ratio"],
        asserts_anomaly=True,
    )


class RecordingProvider:
    name = "recording"

    def __init__(self, draft_factory=None, error: Exception | None = None):
        self.bundles = []
        self._draft_factory = draft_factory or _valid_draft
        self._error = error

    def explain(self, bundle):
        self.bundles.append(bundle)
        if self._error is not None:
            raise self._error
        return self._draft_factory(bundle)


def _fixed_clock(value: datetime = BASE):
    return lambda: value


# ---------------------------------------------------------------------------
# Incident construction and contract
# ---------------------------------------------------------------------------


def test_scored_flagged_result_becomes_a_schema_valid_incident():
    result = _result()
    outcome = EvidenceCorrelationPipeline(
        CorrelationConfig(), clock=_fixed_clock()
    ).correlate([result], _evidence(result.window_id, "evt-1", "evt-2"))

    assert outcome.counts()["incidents"] == 1
    assert outcome.counts()["candidates"] == 1
    incident = outcome.incidents[0]
    validate_json_schema(incident, SCHEMA)

    assert incident["schema_version"] == "1.0"
    assert incident["status"] == "open"
    assert incident["severity"] == "high"
    assert incident["anomaly_score"] == result.raw_score
    assert incident["detector"] == "isolation_forest"
    assert incident["affected_entities"] == [
        {"host": "host-1", "service": "web-api", "source": None}
    ]
    assert incident["evidence"]["supporting_event_ids"] == ["evt-1", "evt-2"]
    assert incident["evidence"]["contradicting_event_ids"] == []
    assert incident["windows"][0]["start"] == result.window_start
    assert incident["model_versions"]["model_version"] == "rev-123"
    assert incident["model_versions"]["feature_schema_version"] == "network-features-v1"
    assert incident["model_versions"]["contract_version"] == CONTRACT_VERSION
    assert incident["created_at"] == BASE.isoformat()
    assert incident["audit"] == []
    assert incident["correlation"]["upstream_revision"].startswith("b155a05")
    assert incident["correlation"]["window_seconds"] == 120
    assert all(check["executes"] is False for check in incident["diagnostic_checks"])
    assert incident["hypotheses"][0]["confidence"] in {"low", "unknown"}
    assert "not a confirmed root cause" in incident["hypotheses"][0]["statement"]


def test_same_service_nearby_anomalies_are_correlated():
    first = _result("h1|web|2026-01-01T10:00:00+00:00", start=BASE)
    second = _result(
        "h1|web|2026-01-01T10:01:00+00:00",
        start=BASE + timedelta(minutes=1),
    )
    evidence = {
        **_evidence(first.window_id, "evt-1"),
        **_evidence(second.window_id, "evt-2"),
    }
    outcome = EvidenceCorrelationPipeline(CorrelationConfig()).correlate(
        [first, second], evidence
    )
    assert outcome.counts()["incidents"] == 1
    incident = outcome.incidents[0]
    validate_json_schema(incident, SCHEMA)
    assert incident["correlation"]["candidate_count"] == 2
    assert incident["evidence"]["supporting_event_ids"] == ["evt-1", "evt-2"]
    assert incident["hypotheses"][0]["confidence"] == "low"


def test_different_services_and_distant_windows_stay_separate():
    first = _result("h1|web|2026-01-01T10:00:00+00:00", service="web")
    second = _result("h1|auth|2026-01-01T10:00:30+00:00", service="auth",
                     start=BASE + timedelta(seconds=30))
    third = _result("h1|web|2026-01-01T10:10:00+00:00", service="web",
                    start=BASE + timedelta(minutes=10))
    evidence = {
        **_evidence(first.window_id, "evt-1"),
        **_evidence(second.window_id, "evt-2"),
        **_evidence(third.window_id, "evt-3"),
    }
    outcome = EvidenceCorrelationPipeline(CorrelationConfig()).correlate(
        [first, second, third], evidence
    )
    assert outcome.counts()["incidents"] == 3


def test_out_of_order_results_are_grouped_and_windows_sorted():
    later = _result("h1|web|2026-01-01T10:02:00+00:00",
                    start=BASE + timedelta(minutes=2))
    earlier = _result("h1|web|2026-01-01T10:00:00+00:00", start=BASE)
    evidence = {
        **_evidence(later.window_id, "evt-2"),
        **_evidence(earlier.window_id, "evt-1"),
    }
    outcome = EvidenceCorrelationPipeline(CorrelationConfig()).correlate(
        [later, earlier], evidence
    )
    incident = outcome.incidents[0]
    assert [w["start"] for w in incident["windows"]] == [
        earlier.window_start, later.window_start
    ]
    assert incident["correlation"]["source_window_ids"] == [
        earlier.window_id, later.window_id
    ]


def test_incident_id_is_deterministic_for_replayed_evidence():
    first = _result("h1|web|2026-01-01T10:00:00+00:00")
    second = _result("h1|web|2026-01-01T10:01:00+00:00",
                     start=BASE + timedelta(minutes=1))
    evidence = {
        **_evidence(first.window_id, "evt-1"),
        **_evidence(second.window_id, "evt-2"),
    }
    a = EvidenceCorrelationPipeline(CorrelationConfig()).correlate([first, second], evidence)
    b = EvidenceCorrelationPipeline(CorrelationConfig()).correlate([second, first], evidence)
    assert a.incidents[0]["incident_id"] == b.incidents[0]["incident_id"]
    # The ID is a UUID derived from the sorted window ids, not a random value.
    expected = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            "lad-correlation|" + "|".join(sorted([first.window_id, second.window_id])),
        )
    )
    assert a.incidents[0]["incident_id"] == expected


# ---------------------------------------------------------------------------
# Explicit status handling
# ---------------------------------------------------------------------------


def test_unscored_statuses_never_enter_correlation():
    results = [
        _result("w1|svc|t", status=InferenceStatus.INSUFFICIENT_DATA,
                reason_code=ReasonCode.NO_EVENTS, is_anomaly=None),
        _result("w2|svc|t", status=InferenceStatus.INVALID_INPUT,
                reason_code=ReasonCode.NON_NUMERIC_FEATURE, is_anomaly=None),
        _result("w3|svc|t", status=InferenceStatus.MODEL_UNAVAILABLE,
                reason_code=ReasonCode.MODEL_UNAVAILABLE, is_anomaly=None),
        _result("w4|svc|t", status=InferenceStatus.INFERENCE_ERROR,
                reason_code=ReasonCode.INFERENCE_FAILED, is_anomaly=None),
    ]
    outcome = EvidenceCorrelationPipeline(CorrelationConfig()).correlate(results, {})
    assert outcome.counts()["incidents"] == 0
    assert outcome.counts()["unscored"] == 4
    assert outcome.unscored_by_status() == {
        "insufficient_data": 1,
        "invalid_input": 1,
        "model_unavailable": 1,
        "inference_error": 1,
    }
    assert all(record.is_anomaly is None for record in outcome.unscored)


def test_scored_but_not_flagged_is_not_correlated():
    result = _result(is_anomaly=False)
    outcome = EvidenceCorrelationPipeline(CorrelationConfig()).correlate(
        [result], _evidence(result.window_id, "evt-1")
    )
    assert outcome.counts()["incidents"] == 0
    assert outcome.counts()["scored_not_flagged"] == 1
    assert outcome.scored_not_flagged[0].is_anomaly is False


def test_none_and_malformed_results_fail_closed():
    class Malformed:
        status = "scored"
        window_id = "bad-status"

    outcome = EvidenceCorrelationPipeline(CorrelationConfig()).correlate(
        [None, Malformed()], {}
    )
    assert outcome.counts()["incidents"] == 0
    assert outcome.counts()["unscored"] == 2
    assert {record.status for record in outcome.unscored} == {"unknown"}


def test_scored_without_a_decision_is_rejected():
    result = _result(is_anomaly=None)
    outcome = EvidenceCorrelationPipeline(CorrelationConfig()).correlate(
        [result], _evidence(result.window_id, "evt-1")
    )
    assert outcome.counts()["rejected"] == 1
    assert outcome.rejected[0].integration_reason == "missing_decision"


def test_missing_evidence_and_missing_identity_are_explicit():
    no_evidence = _result("h1|web|2026-01-01T10:00:00+00:00")
    no_host = _result("h2|web|2026-01-01T10:00:00+00:00", host="")
    no_service = _result("h3|web|2026-01-01T10:00:00+00:00", service="")
    outcome = EvidenceCorrelationPipeline(CorrelationConfig()).correlate(
        [no_evidence, no_host, no_service], {}
    )
    reasons = {record.integration_reason for record in outcome.rejected}
    assert reasons == {"missing_event_evidence", "missing_host_id", "missing_service"}
    assert outcome.counts()["incidents"] == 0


def test_naive_timestamp_and_invalid_boundary_are_rejected():
    naive = _result("h1|web|2026-01-01T10:00:00")
    naive.window_start = "2026-01-01T10:00:00"
    backwards = _result("h2|web|2026-01-01T10:00:00+00:00")
    backwards.window_end = backwards.window_start
    outcome = EvidenceCorrelationPipeline(CorrelationConfig()).correlate(
        [naive, backwards],
        {
            **{naive.window_id: WindowEvidence(naive.window_id, ("evt-1",))},
            **{backwards.window_id: WindowEvidence(backwards.window_id, ("evt-2",))},
        },
    )
    reasons = {record.integration_reason for record in outcome.rejected}
    assert "invalid_window_start" in reasons
    assert "invalid_window_boundary" in reasons


def test_duplicate_window_ids_are_not_correlated_twice():
    first = _result("h1|web|2026-01-01T10:00:00+00:00")
    second = _result("h1|web|2026-01-01T10:00:00+00:00", raw_score=-0.7)
    outcome = EvidenceCorrelationPipeline(CorrelationConfig()).correlate(
        [first, second], _evidence(first.window_id, "evt-1")
    )
    assert outcome.counts()["incidents"] == 1
    assert outcome.counts()["duplicates"] == 1
    assert outcome.duplicates[0].integration_reason == "duplicate_window_id"


def test_unsafe_event_reference_quarantines_the_candidate():
    for unsafe in ("analyst@example.com", "10.0.0.1", "password=hunter2"):
        result = _result(f"h|web|{unsafe}")
        outcome = EvidenceCorrelationPipeline(CorrelationConfig()).correlate(
            [result], _evidence(result.window_id, unsafe)
        )
        assert outcome.counts()["incidents"] == 0
        assert outcome.rejected[0].integration_reason == "unsafe_supporting_event_reference"


def test_unsafe_optional_redacted_ref_is_dropped_not_copied():
    result = _result()
    outcome = EvidenceCorrelationPipeline(CorrelationConfig()).correlate(
        [result],
        _evidence(
            result.window_id,
            "evt-1",
            redacted=("s3://redacted/2026/incident.json", "10.0.0.1",
                      "analyst@example.com"),
        ),
    )
    incident = outcome.incidents[0]
    assert incident["evidence"]["redacted_refs"] == [
        "s3://redacted/2026/incident.json"
    ]
    assert "10.0.0.1" not in json.dumps(incident)
    assert "analyst@example.com" not in json.dumps(incident)


def test_evidence_bounds_are_applied_and_recorded():
    result = _result()
    outcome = EvidenceCorrelationPipeline(
        CorrelationConfig(max_supporting_events_per_window=2, max_events_per_incident=2)
    ).correlate([result], _evidence(result.window_id, "e1", "e2", "e3", "e4"))
    incident = outcome.incidents[0]
    assert incident["evidence"]["supporting_event_ids"] == ["e1", "e2"]
    assert incident["correlation"]["evidence_truncated"] is True
    assert any(
        "truncated" in item for item in incident["hypotheses"][0]["missing_evidence"]
    )


def test_orphan_and_unused_evidence_are_reported():
    scored = _result("h1|web|2026-01-01T10:00:00+00:00")
    unscored = _result(
        "h2|web|2026-01-01T10:00:00+00:00",
        status=InferenceStatus.INSUFFICIENT_DATA,
        reason_code=ReasonCode.NO_EVENTS,
        is_anomaly=None,
    )
    evidence = {
        **_evidence(scored.window_id, "evt-1"),
        **_evidence(unscored.window_id, "evt-2"),
        **_evidence("not-in-results", "evt-3"),
    }
    outcome = EvidenceCorrelationPipeline(CorrelationConfig()).correlate(
        [scored, unscored], evidence
    )
    assert outcome.orphan_evidence == ["not-in-results"]
    assert outcome.unused_evidence == sorted([unscored.window_id, "not-in-results"])


def test_upstream_failure_is_recorded_without_losing_candidates():
    def broken(candidates, window_seconds=120):
        raise RuntimeError("upstream exploded")

    result = _result()
    outcome = EvidenceCorrelationPipeline(
        CorrelationConfig(), correlator=broken
    ).correlate([result], _evidence(result.window_id, "evt-1"))
    assert outcome.counts()["candidates"] == 1
    assert outcome.counts()["incidents"] == 0
    assert outcome.errors[0].stage == "upstream_correlation"


def test_incident_build_failure_is_recorded_without_losing_explanations(monkeypatch):
    result = _result()
    provider = RecordingProvider()

    def broken(*args, **kwargs):
        raise RuntimeError("builder exploded")

    monkeypatch.setattr(pipeline_module, "build_incident", broken)
    outcome = EvidenceCorrelationPipeline(
        CorrelationConfig(explain=True), provider=provider
    ).correlate([result], _evidence(result.window_id, "evt-1"))
    assert outcome.counts()["incidents"] == 0
    assert outcome.errors[0].stage == "incident_build"
    assert len(outcome.llm_interpretations) == 1
    assert outcome.llm_interpretations[0]["grounded"] is True


# ---------------------------------------------------------------------------
# LLM orchestration and failure isolation
# ---------------------------------------------------------------------------


def test_real_explanation_orchestration_is_invoked_with_preserved_evidence():
    result = _result()
    provider = RecordingProvider()
    outcome = EvidenceCorrelationPipeline(
        CorrelationConfig(explain=True), provider=provider
    ).correlate([result], _evidence(result.window_id, "evt-1"))

    assert len(provider.bundles) == 1
    bundle = provider.bundles[0]
    assert bundle.window_id == result.window_id
    assert bundle.service_name == "web-api"
    assert bundle.host_id == "host-1"
    assert bundle.model_name == result.model_name
    assert bundle.model_version == "rev-123"
    assert bundle.model_raw_score == result.raw_score
    assert bundle.threshold == result.threshold
    assert bundle.feature_schema_version == "network-features-v1"

    incident = outcome.incidents[0]
    validate_json_schema(incident, SCHEMA)
    assert incident["anomaly_score"] == result.raw_score
    assert len(incident["hypotheses"]) == 2  # deterministic correlation + LLM
    assert incident["hypotheses"][1]["confidence"] == "low"
    assert incident["llm_interpretations"][0]["provider"] == "recording"
    assert incident["llm_interpretations"][0]["attached_hypothesis"] is True
    assert any(
        check["instruction"] == "Check the upstream HTTP dependency."
        for check in incident["diagnostic_checks"]
    )


def test_llm_timeout_falls_back_and_preserves_the_ml_score():
    result = _result()
    outcome = EvidenceCorrelationPipeline(
        CorrelationConfig(explain=True),
        provider=RecordingProvider(error=TimeoutError("timed out")),
    ).correlate([result], _evidence(result.window_id, "evt-1"))

    incident = outcome.incidents[0]
    assert incident["anomaly_score"] == result.raw_score
    assert incident["evidence"]["supporting_event_ids"] == ["evt-1"]
    entry = incident["llm_interpretations"][0]
    assert entry["fallback_used"] is True
    assert "provider error" in entry["rejected_reason"]
    assert outcome.llm_summary()["fallback_used"] == 1


def test_provider_error_text_is_redacted_in_output_and_logs(caplog):
    leak = (
        "POST https://api.example.com/v1?token=abc123 from 10.1.2.3 "
        "analyst@example.com password=hunter2"
    )
    result = _result()
    with caplog.at_level("WARNING"):
        outcome = EvidenceCorrelationPipeline(
            CorrelationConfig(explain=True),
            provider=RecordingProvider(error=RuntimeError(leak)),
        ).correlate([result], _evidence(result.window_id, "evt-1"))

    blob = json.dumps(outcome.to_dict(), sort_keys=True)
    for sensitive in ("abc123", "10.1.2.3", "analyst@example.com", "hunter2"):
        assert sensitive not in blob
        assert sensitive not in caplog.text
    entry = outcome.llm_interpretations[0]
    assert "provider error" in entry["rejected_reason"]
    assert "<ip>" in entry["rejected_reason"]
    assert "<email>" in entry["rejected_reason"]


def test_llm_disabled_makes_no_provider_call(monkeypatch):
    def must_not_be_built(config):
        raise AssertionError("a provider must not be built when explain is disabled")

    monkeypatch.setattr(
        pipeline_module, "build_explanation_provider", must_not_be_built
    )
    result = _result()
    outcome = EvidenceCorrelationPipeline(CorrelationConfig(explain=False)).correlate(
        [result], _evidence(result.window_id, "evt-1")
    )
    assert outcome.llm_interpretations == []
    assert len(outcome.incidents[0]["hypotheses"]) == 1
    assert outcome.llm_summary()["attempted"] == 0


def test_llm_unavailable_without_fallback_keeps_the_incident():
    result = _result()
    outcome = EvidenceCorrelationPipeline(
        CorrelationConfig(explain=True, explanation_fallback=False),
        provider=RecordingProvider(error=RuntimeError("unavailable")),
    ).correlate([result], _evidence(result.window_id, "evt-1"))
    incident = outcome.incidents[0]
    assert incident["anomaly_score"] == result.raw_score
    assert len(incident["hypotheses"]) == 1
    assert incident["llm_interpretations"][0]["grounded"] is False
    assert incident["llm_interpretations"][0]["attached_hypothesis"] is False


def test_http_rate_limit_and_malformed_response_are_safe():
    result = _result()
    for transport in (
        lambda url, payload, headers, timeout: (429, "rate limited"),
        lambda url, payload, headers, timeout: (200, "not-json"),
    ):
        provider = HttpExplanationProvider(
            "http://127.0.0.1:9/v1/chat/completions",
            model="test",
            api_key_env="LAD_TEST_KEY_NOT_SET",
            transport=transport,
        )
        outcome = EvidenceCorrelationPipeline(
            CorrelationConfig(explain=True), provider=provider
        ).correlate([result], _evidence(result.window_id, "evt-1"))
        incident = outcome.incidents[0]
        assert incident["anomaly_score"] == result.raw_score
        assert incident["llm_interpretations"][0]["fallback_used"] is True


def test_fabricated_llm_claim_is_rejected_and_repaired():
    def fabricated(bundle):
        return DraftExplanation(
            summary="The raw score -999.0 proves an attack.",
            citations=["model_score"],
            asserts_anomaly=True,
        )

    result = _result()
    outcome = EvidenceCorrelationPipeline(
        CorrelationConfig(explain=True), provider=RecordingProvider(fabricated)
    ).correlate([result], _evidence(result.window_id, "evt-1"))
    entry = outcome.llm_interpretations[0]
    assert entry["fallback_used"] is True
    assert "numeric claim" in entry["rejected_reason"]
    assert outcome.incidents[0]["anomaly_score"] == result.raw_score


def test_http_provider_requires_an_endpoint_when_enabled():
    with pytest.raises(ValueError):
        CorrelationConfig(explain=True, llm_provider="http", llm_endpoint="")
    with pytest.raises(ValueError):
        build_explanation_provider(
            CorrelationConfig(llm_provider="http", llm_endpoint="")
        )


def test_labels_and_evaluation_metadata_do_not_reach_the_prompt():
    result = _result(extra={
        "scenario_name": "http_5xx_burst",
        "run_id": "run-42",
        "is_anomaly_true": True,
        "ground_truth": "attack",
    })
    provider = RecordingProvider()
    outcome = EvidenceCorrelationPipeline(
        CorrelationConfig(explain=True), provider=provider
    ).correlate([result], _evidence(result.window_id, "evt-1"))

    prompt = json.dumps(provider.bundles[0].to_prompt_dict(), sort_keys=True)
    serialised = json.dumps(outcome.to_dict(), sort_keys=True)
    for forbidden in ("scenario_name", "http_5xx_burst", "run-42",
                      "is_anomaly_true", "ground_truth"):
        assert forbidden not in prompt
        assert forbidden not in serialised


def test_safe_reference_helpers():
    assert is_safe_event_id("0192f0c6-5555-7b2a-9f4e-0a1b2c3d4e5f")
    assert not is_safe_event_id("analyst@example.com")
    assert not is_safe_event_id("10.0.0.1")
    assert not is_safe_event_id("evt with spaces")
    assert is_safe_redacted_ref("s3://redacted/2026/incident.json")
    assert not is_safe_redacted_ref("token=abc")
    assert is_safe_identity("web-api")
    assert is_safe_identity("sim-web-01.example.internal")
    assert is_safe_identity("10.0.0.7")  # a legitimate host identity
    assert not is_safe_identity("analyst@example.com")
    assert not is_safe_identity("password=hunter2")
    assert not is_safe_identity("web api with spaces")


def test_unsafe_host_and_service_identifiers_are_rejected():
    bad_host = _result("h1|web|2026-01-01T10:00:00+00:00", host="analyst@example.com")
    bad_service = _result(
        "h2|web|2026-01-01T10:00:00+00:00", service="password=hunter2"
    )
    outcome = EvidenceCorrelationPipeline(CorrelationConfig()).correlate(
        [bad_host, bad_service],
        {
            **_evidence(bad_host.window_id, "evt-1"),
            **_evidence(bad_service.window_id, "evt-2"),
        },
    )
    reasons = {record.integration_reason for record in outcome.rejected}
    assert reasons == {"unsafe_host_identifier", "unsafe_service_identifier"}
    assert outcome.counts()["incidents"] == 0


def test_missing_evidence_can_be_allowed_explicitly():
    result = _result()
    outcome = EvidenceCorrelationPipeline(
        CorrelationConfig(require_supporting_events=False)
    ).correlate([result], {})
    assert outcome.counts()["incidents"] == 1
    incident = outcome.incidents[0]
    validate_json_schema(incident, SCHEMA)
    assert incident["evidence"]["supporting_event_ids"] == []
    assert any(
        "No supporting event references were supplied" in item
        for item in incident["hypotheses"][0]["missing_evidence"]
    )


def test_vendored_upstream_correlator_is_used_and_preserves_time_window():
    from anomaly_detector.correlation.upstream import (
        UPSTREAM_MODULE,
        UPSTREAM_REVISION,
        correlate_anomalies,
    )

    assert UPSTREAM_MODULE == "correlator.py"
    assert UPSTREAM_REVISION == "b155a05c00741cc5f129610d2d90af0d18ca48ed"

    anomalies = [
        {"service": "web", "timestamp": "2026-01-01T10:00:00+00:00"},
        {"service": "web", "timestamp": "2026-01-01T10:05:00+00:00"},
        {"service": "auth", "timestamp": "2026-01-01T10:00:30+00:00"},
        {"service": "web", "timestamp": "2026-01-01T10:00:00+00:00"},
        {"timestamp": "2026-01-01T10:00:00+00:00"},  # ignored: no service
        {"service": "web"},  # ignored: no timestamp
    ]
    tight = correlate_anomalies(anomalies, window_seconds=120)
    wide = correlate_anomalies(anomalies, window_seconds=600)
    assert [len(group) for group in tight] == [2, 1, 1]
    assert sorted(len(group) for group in wide) == [1, 3]
    with pytest.raises(ValueError):
        correlate_anomalies(anomalies, window_seconds=-1)


def test_configured_window_seconds_changes_grouping():
    first = _result("h1|web|2026-01-01T10:00:00+00:00", start=BASE)
    second = _result(
        "h1|web|2026-01-01T10:05:00+00:00", start=BASE + timedelta(minutes=5)
    )
    evidence = {
        **_evidence(first.window_id, "evt-1"),
        **_evidence(second.window_id, "evt-2"),
    }
    tight = EvidenceCorrelationPipeline(
        CorrelationConfig(window_seconds=120)
    ).correlate([first, second], evidence)
    wide = EvidenceCorrelationPipeline(
        CorrelationConfig(window_seconds=600)
    ).correlate([first, second], evidence)
    assert tight.counts()["incidents"] == 2
    assert wide.counts()["incidents"] == 1


def test_llm_hypothesis_limit_is_enforced_across_a_group():
    first = _result("h1|web|2026-01-01T10:00:00+00:00", start=BASE)
    second = _result(
        "h1|web|2026-01-01T10:01:00+00:00", start=BASE + timedelta(minutes=1)
    )
    evidence = {
        **_evidence(first.window_id, "evt-1"),
        **_evidence(second.window_id, "evt-2"),
    }
    outcome = EvidenceCorrelationPipeline(
        CorrelationConfig(explain=True, max_llm_hypotheses_per_incident=1),
        provider=RecordingProvider(),
    ).correlate([first, second], evidence)
    incident = outcome.incidents[0]
    # One deterministic correlation hypothesis plus at most one LLM hypothesis.
    assert len(incident["hypotheses"]) == 2
    assert outcome.llm_summary()["attached_hypotheses"] == 1
    assert any(
        entry.get("attachment_reason") ==
        "the per-incident LLM hypothesis limit was reached"
        for entry in incident["llm_interpretations"]
    )


def test_local_end_to_end_fixture_events_to_validated_incident(network_pipeline_run):
    """Full path: simulated LAD events → ML inference → adapter → mocked LLM."""
    from anomaly_detector.network.inference import load_network_detector

    run, cfg = network_pipeline_run
    detector = load_network_detector(
        model_path=f"{cfg.paths['model_dir']}/model.joblib",
        manifest_path=f"{cfg.paths['model_dir']}/manifest.json",
        rules=cfg.rules,
    )
    results = detector.score_windows(run.windows[:400])
    assert any(
        r.status is InferenceStatus.SCORED and r.is_anomaly for r in results
    ), "the fixture must contain at least one flagged window"

    events = []
    for event in run.dataset.events:
        try:
            parsed = datetime.fromisoformat(
                str(event["event_ts"]).replace("Z", "+00:00")
            )
        except (KeyError, ValueError):
            continue
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            continue
        events.append((parsed, event))

    evidence = {}
    for result in results:
        if result.status is not InferenceStatus.SCORED or not result.is_anomaly:
            continue
        start = datetime.fromisoformat(result.window_start.replace("Z", "+00:00"))
        end = datetime.fromisoformat(result.window_end.replace("Z", "+00:00"))
        ids = [
            event["event_id"]
            for timestamp, event in events
            if event.get("host") == result.host_id
            and event.get("service") == result.service_name
            and start <= timestamp < end
        ][:5]
        if ids:
            evidence[result.window_id] = WindowEvidence(
                window_id=result.window_id, supporting_event_ids=tuple(ids)
            )

    provider = RecordingProvider()
    outcome = EvidenceCorrelationPipeline(
        CorrelationConfig(explain=True), provider=provider, clock=_fixed_clock()
    ).correlate(results, evidence)

    assert outcome.counts()["incidents"] >= 1
    assert outcome.counts()["unscored"] == sum(
        1 for r in results if r.status is not InferenceStatus.SCORED
    )
    assert outcome.llm_summary()["attempted"] >= 1
    assert len(provider.bundles) == outcome.llm_summary()["attempted"]
    for incident in outcome.incidents:
        validate_json_schema(incident, SCHEMA)
        assert incident["detector"] == "isolation_forest"
        assert incident["model_versions"]["feature_schema_version"] == (
            "network-features-v1"
        )
        assert incident["correlation"]["upstream_revision"].startswith("b155a05")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_environment_configuration_and_file_loading(tmp_path, monkeypatch):
    monkeypatch.setenv("LAD_CORRELATION_WINDOW_SECONDS", "45")
    monkeypatch.setenv("LAD_CORRELATION_EXPLAIN", "true")
    monkeypatch.setenv("LAD_LLM_PROVIDER", "mock")
    cfg = CorrelationConfig.from_env()
    assert cfg.window_seconds == 45
    assert cfg.explain is True

    path = tmp_path / "correlation.yaml"
    path.write_text(
        "correlation:\n"
        "  window_seconds: 90\n"
        "  explain: false\n"
        "llm:\n"
        "  provider: mock\n"
        "  api_key_env: LAD_TEST_KEY\n",
        encoding="utf-8",
    )
    loaded = load_correlation_config(str(path))
    assert loaded.window_seconds == 45  # environment wins over the file
    assert loaded.explain is True
    assert loaded.llm_provider == "mock"

    monkeypatch.delenv("LAD_CORRELATION_WINDOW_SECONDS")
    monkeypatch.delenv("LAD_CORRELATION_EXPLAIN")
    loaded = load_correlation_config(str(path))
    assert loaded.window_seconds == 90
    assert loaded.explain is False
    assert loaded.llm_api_key_env == "LAD_TEST_KEY"


def test_missing_explicit_config_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_correlation_config(str(tmp_path / "absent.yaml"))
