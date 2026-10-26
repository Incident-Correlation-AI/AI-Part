"""HTTP contract and safety tests for the dual-deployment audit binding.

These tests exercise the **real** service: real artifact, real parser, real
window builder, real Isolation Forest detector and the real correlation
pipeline.  No scoring or threshold logic is mocked.

The model artifact is a local build output and is not committed.  When it is
absent the model-dependent tests skip with an explicit reason instead of
fabricating a model.
"""
from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.service import (  # noqa: E402
    AuditHTTPServer,
    Engine,
    ServiceConfig,
    redact_text,
)
MODEL_PATH = REPO_ROOT / "artifacts" / "network" / "model" / "model.joblib"
MANIFEST_PATH = REPO_ROOT / "artifacts" / "network" / "model" / "manifest.json"
BASE = datetime(2026, 10, 26, 8, 0, tzinfo=timezone.utc)

needs_model = pytest.mark.skipif(
    not (MODEL_PATH.is_file() and MANIFEST_PATH.is_file()),
    reason="local model artifact is not present (build output, intentionally not committed)",
)


def _config(**overrides: Any) -> ServiceConfig:
    values: Dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 0,
        "model_path": str(MODEL_PATH),
        "manifest_path": str(MANIFEST_PATH),
        "network_config_path": str(REPO_ROOT / "configs" / "network.yaml"),
        "correlation_config_path": str(REPO_ROOT / "configs" / "correlation.yaml"),
        "deployment_mode": "test",
        "max_request_bytes": 256 * 1024,
        "max_events_per_request": 500,
        "max_buffer_events": 2_000,
        "max_windows_per_request": 2_000,
        "log_level": "WARNING",
    }
    values.update(overrides)
    return ServiceConfig(**values)


@contextmanager
def _service(**overrides: Any) -> Iterator[Tuple[str, Engine]]:
    engine = Engine(_config(**overrides))
    server = AuditHTTPServer(("127.0.0.1", 0), engine)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", engine
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def _request(
    url: str,
    method: str = "GET",
    body: Any = None,
    content_type: str = "application/json",
    raw: Optional[bytes] = None,
    headers: Optional[Mapping[str, str]] = None,
) -> Tuple[int, Dict[str, Any]]:
    data = raw
    if data is None and body is not None:
        data = json.dumps(body).encode("utf-8")
    request_headers = dict(headers or {})
    if data is not None:
        request_headers.setdefault("Content-Type", content_type)
    request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            return int(response.status), json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return int(exc.code), json.loads(exc.read().decode("utf-8"))


def _event(
    event_id: str,
    ts: datetime,
    *,
    host: str = "host-1",
    service: str = "checkout-api",
    status: int = 200,
    latency: float = 20.0,
    attributes: Optional[Mapping[str, Any]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    payload = {
        "schema_version": "1.0",
        "event_id": event_id,
        "event_ts": ts.isoformat(),
        "host": host,
        "service": service,
        "category": "http",
        "type": "http_request",
        "severity": "error" if status >= 500 else "info",
        "source": "pytest",
        "attributes": {"status_code": status, "latency_ms": latency},
    }
    if attributes is not None:
        payload["attributes"] = dict(attributes)
    payload.update(extra)
    return payload


def _incident_batch(host: str = "corr-host", prefix: str = "corr") -> List[Dict[str, Any]]:
    """A deterministic 5xx burst that the frozen detector flags."""
    events: List[Dict[str, Any]] = []
    counter = 0
    for minute in range(12):
        for index in range(15):
            counter += 1
            burst = minute >= 8
            events.append(
                _event(
                    f"{prefix}-{counter:05d}",
                    BASE + timedelta(minutes=minute, seconds=index * 3),
                    host=host,
                    status=500 if burst and index % 2 == 0 else 200,
                    latency=320.0 if burst else 20.0,
                )
            )
    return events


# ---------------------------------------------------------------------------
# Liveness / readiness / model identity
# ---------------------------------------------------------------------------


@needs_model
def test_health_readiness_and_model_identity() -> None:
    with _service() as (url, _engine):
        status, health = _request(f"{url}/healthz")
        assert status == 200
        assert health["status"] == "ok"
        assert health["deployment"]["binding"] == "http"

        status, ready = _request(f"{url}/readyz")
        assert status == 200
        assert ready["status"] == "ready"
        assert ready["model"]["artifact_sha256"] == _sha256(MODEL_PATH)
        assert ready["model"]["manifest_sha256"] == _sha256(MANIFEST_PATH)

        status, model = _request(f"{url}/v1/model")
        assert status == 200
        contract = model["model"]["contract"]
        assert contract["contract_version"] == "1.1.0"
        assert contract["score_direction"] == "lower_is_more_anomalous"
        assert contract["decision_rule"].startswith("is_anomaly = raw_score <= threshold")
        assert contract["feature_schema_version"] == "network-features-v1"
        assert contract["threshold"] == pytest.approx(
            _manifest_threshold(), rel=0, abs=1e-12
        )


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_threshold() -> float:
    return float(json.loads(MANIFEST_PATH.read_text())["threshold"]["threshold"])


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


@needs_model
def test_ingest_infer_and_label_key_stripping() -> None:
    with _service() as (url, _engine):
        events = []
        for index in range(6):
            event = _event(
                f"leet-{index:04d}",
                BASE + timedelta(seconds=index * 5),
                attributes={
                    "status_code": 200,
                    "latency_ms": 18.0,
                    "label": "anomaly",
                    "ground_truth": True,
                    "scenario_name": "http_5xx_burst",
                },
            )
            event["test_id"] = "T-999"
            event["expected"] = "anomaly"
            events.append(event)

        status, ingest = _request(f"{url}/v1/events", "POST", {"events": events})
        assert status == 200
        assert ingest["accepted_now"] == 6
        assert ingest["dropped_metadata_keys"]["evaluation_key"] >= 18

        status, result = _request(f"{url}/v1/infer", "POST", {"use_buffer": True})
        assert status == 200
        assert result["windows"]["total"] == 1
        assert result["windows"]["scored"] == 1

        serialized = json.dumps(result["results"])
        for needle in ("ground_truth", "scenario_name", "T-999", "http_5xx_burst", '"label"'):
            assert needle not in serialized
        # The infer request itself carried no metadata, so nothing was dropped
        # at this hop; the earlier ingest summary holds the accounting.
        assert result["ingest"]["dropped_metadata_keys"] == {}


def test_empty_event_list_is_rejected() -> None:
    with _service() as (url, _engine):
        status, body = _request(f"{url}/v1/events", "POST", {"events": []})
        assert status == 400
        assert body["error"]["code"] == "no_events"


def test_empty_body_is_rejected() -> None:
    with _service() as (url, _engine):
        status, body = _request(f"{url}/v1/infer", "POST", raw=b"")
        assert status == 400
        assert body["error"]["code"] == "empty_body"


def test_wrong_content_type_is_rejected() -> None:
    with _service() as (url, _engine):
        status, body = _request(
            f"{url}/v1/infer",
            "POST",
            raw=b'{"events": []}',
            content_type="text/plain",
        )
        assert status == 415
        assert body["error"]["code"] == "unsupported_media_type"


def test_invalid_json_is_rejected() -> None:
    with _service() as (url, _engine):
        status, body = _request(f"{url}/v1/infer", "POST", raw=b"{not json")
        assert status == 400
        assert body["error"]["code"] == "invalid_json"


def test_request_size_limit() -> None:
    with _service(max_request_bytes=512) as (url, _engine):
        status, body = _request(f"{url}/v1/events", "POST", raw=b"{" + b" " * 2048 + b"}")
        assert status == 413
        assert body["error"]["code"] == "request_too_large"


def test_unknown_path() -> None:
    with _service() as (url, _engine):
        status, body = _request(f"{url}/v1/nope")
        assert status == 404
        assert body["error"]["code"] == "not_found"


# ---------------------------------------------------------------------------
# Malformed / duplicate / out-of-order telemetry
# ---------------------------------------------------------------------------


@needs_model
def test_malformed_events_are_counted_not_fatal() -> None:
    with _service() as (url, _engine):
        valid = [
            _event(f"ok-{index}", BASE + timedelta(seconds=index * 4))
            for index in range(6)
        ]
        malformed = [
            {"event_id": "bad-version", "schema_version": "0.9"},
            {"event_id": "bad-host", "schema_version": "1.0"},
            {"event_id": "bad-ts", "schema_version": "1.0", "host": "h", "event_ts": "not-a-time"},
            42,
        ]
        status, result = _request(
            f"{url}/v1/infer", "POST", {"events": valid + malformed, "use_buffer": False}
        )
        assert status == 200
        summary = result["ingest"]["request_parse"]
        assert summary["accepted"] == 6
        assert summary["rejected"] == len(malformed)
        assert result["windows"]["scored"] == 1


@needs_model
def test_duplicate_event_ids_are_counted() -> None:
    with _service() as (url, _engine):
        events = [
            _event(f"dup-{index}", BASE + timedelta(seconds=index * 4)) for index in range(6)
        ]
        status, first = _request(f"{url}/v1/events", "POST", {"events": events})
        assert status == 200
        status, second = _request(f"{url}/v1/events", "POST", {"events": events})
        assert status == 200
        assert second["accepted_now"] == 0
        assert second["duplicates_now"] == 6
        assert second["buffer"]["duplicate_event_ids"] == 6


@needs_model
def test_out_of_order_events_are_counted() -> None:
    with _service() as (url, _engine):
        events = [
            _event("ooo-late", BASE + timedelta(minutes=5)),
            _event("ooo-early", BASE + timedelta(minutes=1)),
        ]
        status, result = _request(
            f"{url}/v1/infer", "POST", {"events": events, "use_buffer": False}
        )
        assert status == 200
        assert result["ingest"]["request_parse"]["out_of_order"] >= 1


def test_unsafe_event_id_is_rejected_at_ingress() -> None:
    with _service() as (url, _engine):
        unsafe = _event("user@example.com", BASE)
        status, result = _request(
            f"{url}/v1/infer", "POST", {"events": [unsafe], "use_buffer": False}
        )
        assert status == 400
        assert result["error"]["code"] == "no_events"
        assert result["error"]["detail"]["request_parse"]["rejected"] == 1


# ---------------------------------------------------------------------------
# Correlation / evidence / explanation
# ---------------------------------------------------------------------------


@needs_model
def test_analyze_produces_correlation_with_event_evidence() -> None:
    with _service() as (url, _engine):
        status, result = _request(
            f"{url}/v1/analyze",
            "POST",
            {"events": _incident_batch(), "use_buffer": False, "explain": False},
        )
        assert status == 200
        assert result["windows"]["scored"] >= 10
        correlation = result["correlation"]
        assert correlation is not None
        assert correlation["counts"]["candidates"] >= 1
        assert correlation["counts"]["incidents"] >= 1
        assert correlation["errors"] == []
        assert correlation["upstream_revision"]
        incident = correlation["incidents"][0]
        assert incident["evidence"]["supporting_event_ids"]
        assert incident["affected_entities"]
        assert result["explanation"]["enabled"] is False


@needs_model
def test_http_mock_provider_exercises_real_transport() -> None:
    from deploy import mock_llm
    from anomaly_detector.network.llm import HttpExplanationProvider

    mock_server = mock_llm.ThreadingHTTPServer(("127.0.0.1", 0), mock_llm.Handler)
    mock_server.daemon_threads = True
    thread = threading.Thread(
        target=mock_server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    endpoint = (
        f"http://127.0.0.1:{mock_server.server_address[1]}/v1/chat/completions"
    )
    try:
        provider = HttpExplanationProvider(endpoint=endpoint, model="lad-grounded-mock-v1")
        with _service() as (url, _engine):
            engine = _engine
            engine._provider_override = provider
            status, result = _request(
                f"{url}/v1/analyze",
                "POST",
                {"events": _incident_batch(), "use_buffer": False, "explain": True},
            )
            assert status == 200
            assert result["explanation"]["enabled"] is True
            assert result["explanation"]["provider"] == "http"
            llm = result["correlation"]["llm"]
            assert llm["attempted"] >= 1
            assert llm["grounded"] >= 1
            assert llm["errors"] == 0
            for interpretation in result["correlation"]["llm_interpretations"]:
                assert interpretation["grounded"] is True
                assert interpretation["citations"]
    finally:
        mock_server.shutdown()
        mock_server.server_close()
        thread.join(timeout=10)


@needs_model
def test_prompt_payload_contains_no_evaluation_metadata() -> None:
    """The provider boundary sees only allowlisted evidence, never labels."""
    from anomaly_detector.network.llm import DraftExplanation

    captured: List[Dict[str, Any]] = []

    class CaptureProvider:
        name = "capture"

        def explain(self, bundle: Any) -> Any:
            payload = bundle.to_prompt_dict()
            captured.append(payload)
            raw = float(payload["decision"]["raw_score"])
            threshold = float(payload["decision"]["threshold"])
            return DraftExplanation(
                summary=(
                    "The detector flagged this window with raw score "
                    f"{raw:.4f} against threshold {threshold:.4f}."
                ),
                evidence=[f"model_score = {raw:.4f}", f"threshold = {threshold:.4f}"],
                recommended_checks=["Compare with the preceding rolling baseline."],
                asserts_anomaly=bool(payload["decision"]["is_anomaly"]),
                citations=["model_score", "threshold"],
            )

    with _service() as (url, _engine):
        _engine._provider_override = CaptureProvider()
        status, result = _request(
            f"{url}/v1/analyze",
            "POST",
            {"events": _incident_batch(host="prompt-host", prefix="prompt"), "use_buffer": False, "explain": True},
        )
        assert status == 200
        assert captured, "the provider was never called"
        serialized = json.dumps(captured)
        for needle in (
            "scenario_name", "ground_truth", "groundtruth", "is_anomaly_true",
            "test_id", '"expected"', "http_5xx_burst", "run-0", '"label"',
            '"labels"', "T-999",
        ):
            assert needle not in serialized
        assert result["explanation"]["grounded"] >= 1


# ---------------------------------------------------------------------------
# Failure modes / redaction
# ---------------------------------------------------------------------------


def test_model_unavailable_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "missing.joblib"
    missing_manifest = tmp_path / "missing.json"
    with _service(
        model_path=str(missing),
        manifest_path=str(missing_manifest),
    ) as (url, _engine):
        status, ready = _request(f"{url}/readyz")
        assert status == 503
        assert ready["status"] == "not_ready"
        assert ready["model"]["ready"] is False

        status, result = _request(
            f"{url}/v1/infer",
            "POST",
            {"events": [_event("model-down-1", BASE)], "use_buffer": False},
        )
        assert status == 503
        assert result["error"]["code"] == "model_unavailable"
        assert "results" not in result


def test_redaction_removes_secrets_and_pii() -> None:
    text = redact_text(
        "password=hunter2 api_key=abcd1234 bearer abc.def.ghi "
        "user@example.com 10.1.2.3"
    )
    for secret in ("hunter2", "abcd1234", "abc.def.ghi", "user@example.com", "10.1.2.3"):
        assert secret not in text
    assert "***" in text


def test_deployment_mode_is_reported() -> None:
    with _service(deployment_mode="container") as (url, _engine):
        status, health = _request(f"{url}/healthz")
        assert status == 200
        assert health["deployment"]["mode"] == "container"
