"""Deterministic, local OpenAI-compatible mock explanation provider.

This process exists **only** to exercise the real
``anomaly_detector.network.llm.HttpExplanationProvider`` transport boundary in
Path C without contacting an external model.  It is not a real LLM and its
output is never presented as real-provider evidence.

The response is derived from the supplied ``EvidenceBundle`` JSON (score,
threshold, severity, rule codes and measurement keys).  It contains no canned
scenario text, labels or test identifiers.

Run::

    python -m deploy.mock_llm --port 8081
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Mapping, Optional, Sequence

LOGGER = logging.getLogger("lad.deploy.mock_llm")
MODEL_NAME = "lad-grounded-mock-v1"


def _latest_bundle(payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            parsed = json.loads(content)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_draft(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    decision = bundle.get("decision") or {}
    window = bundle.get("window") or {}
    rule_evidence = bundle.get("rule_evidence") or {}
    measurements = bundle.get("measurements") or {}

    raw_score = _number(decision.get("raw_score"))
    threshold = _number(decision.get("threshold"))
    severity = str(decision.get("severity") or "unknown")
    service = str(window.get("service") or "unknown-service")
    host = str(window.get("host") or "unknown-host")
    flagged = bool(decision.get("is_anomaly"))

    hits = rule_evidence.get("hits") or []
    codes: List[str] = []
    for hit in hits:
        if isinstance(hit, Mapping) and hit.get("code"):
            code = str(hit["code"])
            if code not in codes:
                codes.append(code)
    codes = codes[:3]

    summary_parts = [
        f"The Isolation Forest {'flagged' if flagged else 'did not flag'} the "
        f"{service} window on {host} with raw score "
        f"{raw_score:.4f} against threshold {threshold:.4f} "
        f"(severity {severity})."
    ]
    if codes:
        summary_parts.append(f"Transparent rule evidence: {', '.join(codes)}.")
    else:
        summary_parts.append("No transparent rule exceeded its configured threshold.")

    evidence = [
        f"model_score = {raw_score:.4f}",
        f"threshold = {threshold:.4f}",
        f"severity = {severity}",
    ]
    citations = ["model_score", "threshold", "severity"] + list(codes)

    if isinstance(measurements, Mapping):
        for key in sorted(measurements, key=str)[:3]:
            value = _number(measurements.get(key))
            if value is None:
                continue
            evidence.append(f"{key} = {value:.4f}")
            citations.append(str(key))

    checks = [
        "Compare this window with the preceding rolling baseline for the same service.",
        "Check the health of the dependency named by the rule evidence, if any.",
    ]
    return {
        "summary": " ".join(summary_parts),
        "evidence": evidence,
        "recommended_checks": checks,
        "asserts_anomaly": flagged,
        "citations": list(dict.fromkeys(citations)),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "LADMockLLM/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] == "/healthz":
            self._send(HTTPStatus.OK, {"status": "ok", "mock": True, "model": MODEL_NAME})
            return
        self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] not in {
            "/v1/chat/completions",
            "/chat/completions",
        }:
            self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        length_header = self.headers.get("Content-Length")
        if not length_header:
            self._send(HTTPStatus.LENGTH_REQUIRED, {"error": "content_length_required"})
            return
        try:
            length = int(length_header)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return
        if not isinstance(payload, Mapping):
            self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid_body"})
            return

        bundle = _latest_bundle(payload)
        if bundle is None:
            self._send(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "no evidence bundle in the user message"},
            )
            return

        draft = build_draft(bundle)
        self._send(
            HTTPStatus.OK,
            {
                "id": "mock-completion",
                "object": "chat.completion",
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(draft, sort_keys=True),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "note": "deterministic mock provider; not a real LLM",
            },
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("LAD_MOCK_LLM_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("LAD_MOCK_LLM_PORT", "8081"))
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, os.environ.get("LAD_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    LOGGER.info(
        json.dumps(
            {
                "event": "mock_llm_start",
                "host": args.host,
                "port": args.port,
                "model": MODEL_NAME,
                "warning": "mock provider only; not a real LLM",
            },
            sort_keys=True,
        )
    )

    def _shutdown(signum: int, frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
