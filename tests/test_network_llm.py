"""Evidence-grounded explanation adapter tests.

Every test is offline.  Provider calls use an injected transport or a local
mock object; no HTTP request leaves the process.
"""
import json

import pytest

from anomaly_detector.inference.contract import (
    CONTRACT_VERSION,
    InferenceResult,
    InferenceStatus,
    ReasonCode,
)
from anomaly_detector.network.config import load_network_config
from anomaly_detector.network.llm import (
    DraftExplanation,
    EvidenceBundle,
    HttpExplanationProvider,
    MockExplanationProvider,
    explain_result,
    validate_draft,
)


def _scored_result(is_anomaly: bool = True) -> InferenceResult:
    baseline = {
        "is_anomaly": True,
        "risk_score": 1.0,
        "severity": "medium",
        "rule_hits": [
            {
                "code": "http_5xx_ratio",
                "message": "5xx ratio 0.4 is above 0.05",
                "observed": {"http_5xx_ratio": 0.4},
                "threshold": 0.05,
                "weight": 1.0,
            }
        ],
    }
    return InferenceResult(
        contract_version=CONTRACT_VERSION,
        status=InferenceStatus.SCORED,
        reason_code=ReasonCode.OK,
        reason="scored",
        window_id="sim-web-01|web-api|2026-01-01T00:15:00+00:00",
        window_start="2026-01-01T00:15:00+00:00",
        window_end="2026-01-01T00:16:00+00:00",
        host_id="sim-web-01",
        service_name="web-api",
        model_name="sklearn.ensemble.IsolationForest",
        model_version="test-revision",
        feature_schema_version="network-features-v1",
        raw_score=-0.6,
        score_direction="lower_is_more_anomalous",
        threshold=0.5,
        is_anomaly=is_anomaly,
        extra={
            "network": {
                "severity": "medium",
                "decision_source": "isolation_forest",
                "baseline": baseline,
                "evidence_features": {
                    "http_5xx_ratio": 0.4,
                    "http_5xx_count": 10.0,
                    "event_count": 25.0,
                },
                "telemetry_quality": {"duplicate_event_id": 2},
            }
        },
    )


class ScriptedProvider:
    name = "scripted"

    def __init__(self, draft):
        self._draft = draft

    def explain(self, bundle):
        if isinstance(self._draft, Exception):
            raise self._draft
        return self._draft


# ---------------------------------------------------------------------------
# Fallback and grounding
# ---------------------------------------------------------------------------

def test_offline_fallback_is_grounded_and_matches_the_decision():
    result = _scored_result(is_anomaly=True)
    explanation = explain_result(result)
    assert explanation.explained is True
    assert explanation.grounded is True
    assert explanation.fallback_used is False
    assert explanation.asserts_anomaly is True
    assert "model_score" in explanation.citations
    assert "threshold" in explanation.citations
    assert "http_5xx_ratio" in explanation.citations
    assert explanation.to_dict()["window_id"] == result.window_id


def test_unscored_window_is_never_explained_as_a_detection():
    result = InferenceResult(
        contract_version=CONTRACT_VERSION,
        status=InferenceStatus.INSUFFICIENT_DATA,
        reason_code=ReasonCode.NO_EVENTS,
        window_id="sim-db-01|orders-db|2026-01-01T00:33:00+00:00",
        is_anomaly=None,
    )
    explanation = explain_result(result)
    assert explanation.explained is False
    assert explanation.grounded is False
    assert explanation.asserts_anomaly is None
    assert "not scored" in explanation.rejected_reason
    assert explanation.to_dict()["window_id"] == result.window_id


def test_provider_failure_falls_back_to_evidence():
    explanation = explain_result(
        _scored_result(), provider=ScriptedProvider(RuntimeError("boom"))
    )
    assert explanation.explained is True
    assert explanation.fallback_used is True
    assert "provider error" in explanation.rejected_reason
    assert explanation.grounded is True


def test_fabricated_number_is_rejected_then_repaired():
    draft = DraftExplanation(
        summary="The raw score -0.7777 crossed the threshold.",
        citations=["model_score", "threshold"],
        asserts_anomaly=True,
    )
    explanation = explain_result(_scored_result(), provider=ScriptedProvider(draft))
    assert explanation.grounded is True
    assert explanation.fallback_used is True
    assert "numeric claim" in explanation.rejected_reason


def test_contradicting_anomaly_assertion_is_rejected():
    draft = DraftExplanation(
        summary="The window is normal with raw score -0.6000.",
        citations=["model_score", "threshold"],
        asserts_anomaly=False,
    )
    explanation = explain_result(_scored_result(is_anomaly=True), provider=ScriptedProvider(draft))
    assert explanation.fallback_used is True
    assert "contradicts the model decision" in explanation.rejected_reason


def test_unknown_citation_is_rejected():
    draft = DraftExplanation(
        summary="The window is anomalous with raw score -0.6000.",
        citations=["made_up_evidence"],
        asserts_anomaly=True,
    )
    explanation = explain_result(_scored_result(), provider=ScriptedProvider(draft))
    assert explanation.fallback_used is True
    assert "unknown citation" in explanation.rejected_reason


def test_evaluation_vocabulary_and_pii_are_rejected():
    for summary in (
        "The http_5xx_burst scenario explains the raw score -0.6000.",
        "Contact analyst@example.com about raw score -0.6000.",
        "The source 10.0.0.1 caused raw score -0.6000.",
        "The password was leaked; raw score -0.6000.",
    ):
        draft = DraftExplanation(
            summary=summary,
            citations=["model_score", "threshold"],
            asserts_anomaly=True,
        )
        explanation = explain_result(_scored_result(), provider=ScriptedProvider(draft))
        assert explanation.fallback_used is True, summary
        assert explanation.rejected_reason


def test_valid_scripted_draft_is_used_without_fallback():
    draft = DraftExplanation(
        summary=(
            "The Isolation Forest flagged this window: raw score -0.6000 is "
            "below the threshold 0.5000, and the HTTP 5xx ratio is 0.4000."
        ),
        evidence=["http_5xx_ratio = 0.4000"],
        recommended_checks=["Check the upstream HTTP dependency."],
        citations=["model_score", "threshold", "http_5xx_ratio"],
        asserts_anomaly=True,
    )
    explanation = explain_result(_scored_result(), provider=ScriptedProvider(draft))
    assert explanation.grounded is True
    assert explanation.fallback_used is False
    assert explanation.provider == "scripted"
    assert explanation.text.startswith("The Isolation Forest flagged")


# ---------------------------------------------------------------------------
# Evidence bundle privacy boundary
# ---------------------------------------------------------------------------

def test_prompt_payload_contains_only_evidence_and_constraints():
    bundle = EvidenceBundle.from_result(_scored_result())
    payload = bundle.to_prompt_dict()
    names = json.dumps(payload)
    for forbidden in ("scenario_name", "run_id", "ground_truth", "is_anomaly_true"):
        assert forbidden not in names
    assert payload["decision"]["is_anomaly"] is True
    assert payload["measurements"]["http_5xx_ratio"] == 0.4
    assert payload["constraints"]


def test_validate_draft_requires_an_explicit_assertion():
    bundle = EvidenceBundle.from_result(_scored_result())
    violations = validate_draft(
        DraftExplanation(
            summary="Raw score -0.6000.",
            citations=["model_score"],
            asserts_anomaly=None,
        ),
        bundle,
    )
    assert any("explicit anomaly assertion" in v for v in violations)


# ---------------------------------------------------------------------------
# HTTP provider (injected transport, still offline)
# ---------------------------------------------------------------------------

def test_http_provider_parses_openai_compatible_response():
    captured = {}

    def transport(url, payload, headers, timeout):
        captured.update(url=url, payload=payload, headers=headers)
        content = json.dumps({
            "summary": "Raw score -0.6000 is below the threshold 0.5000.",
            "evidence": ["http_5xx_ratio = 0.4000"],
            "recommended_checks": ["Check the upstream HTTP dependency."],
            "citations": ["model_score", "threshold", "http_5xx_ratio"],
            "asserts_anomaly": True,
        })
        return 200, json.dumps({"choices": [{"message": {"content": content}}]})

    provider = HttpExplanationProvider(
        "http://127.0.0.1:9/v1/chat/completions",
        model="small-test-model",
        api_key_env="LAD_TEST_LLM_KEY",
        transport=transport,
    )
    explanation = explain_result(_scored_result(), provider=provider)
    assert explanation.grounded is True
    assert explanation.provider == "http"
    assert captured["url"].startswith("http://127.0.0.1:9/")
    assert "Authorization" not in captured["headers"]
    assert "messages" in captured["payload"]
    assert "scenario_name" not in json.dumps(captured["payload"])


def test_http_provider_error_is_recoverable():
    provider = HttpExplanationProvider(
        "http://127.0.0.1:9/v1/chat/completions",
        transport=lambda *args: (503, "unavailable"),
    )
    explanation = explain_result(_scored_result(), provider=provider)
    assert explanation.explained is True
    assert explanation.fallback_used is True
    assert "HTTP 503" in explanation.rejected_reason


def test_llm_is_disabled_by_default_and_never_serialises_the_key():
    cfg = load_network_config("configs/network.yaml")
    assert cfg.llm.enabled is False
    serialised = cfg.llm.to_dict()
    assert "api_key" not in serialised
    assert serialised["api_key_env"] == "LAD_LLM_API_KEY"
