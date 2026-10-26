"""Minimal, bounded HTTP binding for the real network anomaly-detection pipeline.

This module is an **audit / deployment adapter**, not a new production
architecture.  It exposes the already-implemented path::

    canonical LAD envelopes
        -> anomaly_detector.network.lad_adapter.parse_lad_events
        -> anomaly_detector.network.features.build_network_windows
        -> anomaly_detector.network.inference.NetworkDetector.score_windows
        -> anomaly_detector.correlation.EvidenceCorrelationPipeline
        -> optional network.llm explanation provider

No scoring, thresholding, routing or correlation rule is reimplemented here.
The service only adds HTTP transport, bounded in-memory buffering, request
validation, redaction of diagnostics and an artifact-identity endpoint.

Run locally (Path L)::

    python -m deploy.service

Run in a container (Path C): the image entrypoint is the same command; see
``deploy/README.md``.

Endpoints
---------
``GET  /healthz``      liveness only, no model access
``GET  /readyz``       readiness + loaded artifact hashes
``GET  /v1/model``     verified model/feature/contract metadata
``POST /v1/events``    parse and buffer canonical LAD events
``POST /v1/infer``     parse/score events (ML interface only)
``POST /v1/analyze``   score + evidence correlation + optional explanation

The API never accepts labels, scenario names, test IDs or expected outcomes.
Those keys are dropped at ingress and counted (names only) in the response.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import signal
import socket
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Make ``python -m deploy.service`` work from a source checkout as well as from
# an installed wheel inside the container.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from anomaly_detector.correlation import (  # noqa: E402
    CorrelationConfig,
    EvidenceCorrelationPipeline,
    WindowEvidence,
    build_explanation_provider,
    is_safe_event_id,
    load_correlation_config,
)
from anomaly_detector.inference.contract import InferenceStatus  # noqa: E402
from anomaly_detector.network.config import load_network_config  # noqa: E402
from anomaly_detector.network.features import (  # noqa: E402
    NETWORK_FEATURE_SCHEMA_VERSION,
    build_network_windows,
)
from anomaly_detector.network.inference import (  # noqa: E402
    NetworkDetector,
    load_network_detector,
)
from anomaly_detector.network.lad_adapter import (  # noqa: E402
    LadParseOptions,
    LadParseResult,
    QualityRecord,
    RejectedLadEvent,
    parse_lad_events,
)
from anomaly_detector.schemas import StructuredEvent  # noqa: E402

LOGGER = logging.getLogger("lad.deploy.service")

SERVICE_VERSION = "1.0.0-audit"
CONTRACT_VERSION = "1.1.0"

#: Top-level canonical envelope fields.  Everything else is dropped before the
#: parser sees it.  This is the primary label/answer-key ingress control.
_ALLOWED_ENVELOPE_KEYS = frozenset({
    "schema_version", "event_id", "event_ts", "observed_ts", "host",
    "service", "category", "type", "severity", "source", "attributes",
})

#: Attribute keys with a documented feature use in ``network-features-v1``.
_ALLOWED_ATTRIBUTE_KEYS = frozenset({
    "status_code", "latency_ms", "duration_ms", "duration",
    "db_timeout", "db_connection_failure",
    "auth_source_id", "source_id", "auth_account_id", "account_id",
})

#: Keys that must never reach features, model inputs or LLM prompts.
_EVALUATION_KEYS = frozenset({
    "label", "labels", "is_anomaly", "is_anomaly_true", "ground_truth",
    "groundtruth", "answer", "answer_key", "expected", "expected_output",
    "scenario", "scenario_name", "test_id", "testid", "run_id", "future",
    "future_events", "anomaly_label", "anomaly_score_true",
})

_SECRET_WORD_RE = re.compile(
    r"(?i)\b(password|passwd|secret|api[_-]?key|apikey|access_token|"
    r"refresh_token|session_token|token|authorization|bearer|cookie|"
    r"private_key)\b"
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(password|passwd|secret|api[_-]?key|apikey|access_token|"
    r"refresh_token|session_token|token|authorization|bearer|cookie|"
    r"private_key)\b\s*[=:]\s*([^\s,;]+)"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(value: str, default_relative: str) -> str:
    candidate = value or default_relative
    if os.path.isabs(candidate):
        return candidate
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    rooted = os.path.join(_REPO_ROOT, candidate)
    return rooted


@dataclass(frozen=True)
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    model_path: str = ""
    manifest_path: str = ""
    network_config_path: str = ""
    correlation_config_path: str = ""
    deployment_mode: str = "local"
    max_request_bytes: int = 8 * 1024 * 1024
    max_events_per_request: int = 5_000
    max_buffer_events: int = 250_000
    max_windows_per_request: int = 50_000
    max_incidents_in_response: int = 50
    max_records_in_response: int = 500
    log_level: str = "INFO"
    service_version: str = SERVICE_VERSION

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        return cls(
            host=_env_str("LAD_HOST", "0.0.0.0"),
            port=_env_int("LAD_PORT", 8080),
            model_path=_resolve_path(
                _env_str("LAD_MODEL_PATH", ""),
                "artifacts/network/model/model.joblib",
            ),
            manifest_path=_resolve_path(
                _env_str("LAD_MANIFEST_PATH", ""),
                "artifacts/network/model/manifest.json",
            ),
            network_config_path=_resolve_path(
                _env_str("LAD_NETWORK_CONFIG", ""),
                "configs/network.yaml",
            ),
            correlation_config_path=_resolve_path(
                _env_str("LAD_CORRELATION_CONFIG", ""),
                "configs/correlation.yaml",
            ),
            deployment_mode=_env_str("LAD_DEPLOYMENT_MODE", "local").lower(),
            max_request_bytes=_env_int("LAD_MAX_REQUEST_BYTES", 8 * 1024 * 1024),
            max_events_per_request=_env_int("LAD_MAX_EVENTS_PER_REQUEST", 5_000),
            max_buffer_events=_env_int("LAD_MAX_BUFFER_EVENTS", 250_000),
            max_windows_per_request=_env_int("LAD_MAX_WINDOWS_PER_REQUEST", 50_000),
            max_incidents_in_response=_env_int("LAD_MAX_INCIDENTS_IN_RESPONSE", 50),
            max_records_in_response=_env_int("LAD_MAX_RECORDS_IN_RESPONSE", 500),
            log_level=_env_str("LAD_LOG_LEVEL", "INFO").upper(),
            service_version=_env_str("LAD_SERVICE_VERSION", SERVICE_VERSION),
        )


# ---------------------------------------------------------------------------
# Redaction helpers
# ---------------------------------------------------------------------------


def redact_text(value: Any, limit: int = 500) -> str:
    """Return a bounded diagnostic with obvious secrets/PII removed."""
    text = str(value or "")
    text = _BEARER_RE.sub("bearer ***", text)
    text = _SECRET_ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}=***", text)
    text = _EMAIL_RE.sub("<email>", text)
    text = _IPV4_RE.sub("<ipv4>", text)
    text = _JWT_RE.sub("<jwt>", text)
    if len(text) > limit:
        text = text[:limit] + "..."
    return text


def _safe_key_name(key: Any) -> str:
    name = str(key)
    if _SECRET_WORD_RE.search(name) or _EMAIL_RE.search(name):
        return "redacted_key"
    return name[:64]


def _safe_source(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    text = value.strip()
    if (
        len(text) > 64
        or _SECRET_WORD_RE.search(text)
        or _EMAIL_RE.search(text)
        or _IPV4_RE.search(text)
        or _JWT_RE.search(text)
    ):
        return "redacted"
    return text


def _sanitize_event(
    raw: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[RejectedLadEvent], Dict[str, int]]:
    """Return ``(clean_event, rejection, dropped_key_counts)``.

    The function never inspects or logs values of dropped keys.  Only key names
    are counted, and sensitive-looking names are collapsed to ``redacted_key``.
    """
    dropped: Dict[str, int] = {}
    if not isinstance(raw, Mapping):
        # Let the canonical parser produce its own invalid_envelope record.
        return raw, None, dropped

    clean: Dict[str, Any] = {}
    for key, value in raw.items():
        if key not in _ALLOWED_ENVELOPE_KEYS:
            label = "evaluation_key" if str(key).lower() in _EVALUATION_KEYS else _safe_key_name(key)
            dropped[label] = dropped.get(label, 0) + 1
            continue
        clean[key] = value

    event_id = clean.get("event_id")
    if isinstance(event_id, str) and event_id.strip():
        if not is_safe_event_id(event_id.strip()):
            return (
                None,
                RejectedLadEvent(
                    event_id=None,
                    reason="unsafe_event_id",
                    detail="event_id must be an opaque identifier without secrets or PII",
                ),
                dropped,
            )

    if isinstance(clean.get("source"), str):
        clean["source"] = _safe_source(clean.get("source"))

    attributes = clean.get("attributes")
    if isinstance(attributes, Mapping):
        safe_attributes: Dict[str, Any] = {}
        for key, value in attributes.items():
            if key not in _ALLOWED_ATTRIBUTE_KEYS:
                label = "evaluation_key" if str(key).lower() in _EVALUATION_KEYS else _safe_key_name(key)
                dropped[label] = dropped.get(label, 0) + 1
                continue
            safe_attributes[key] = value
        clean["attributes"] = safe_attributes
    elif attributes is not None:
        # Preserve the parser's explicit invalid_attributes quality record.
        clean["attributes"] = attributes

    return clean, None, dropped


def _sanitize_batch(
    raw_events: Iterable[Any],
) -> Tuple[List[Any], List[RejectedLadEvent], Dict[str, int]]:
    cleaned: List[Any] = []
    rejections: List[RejectedLadEvent] = []
    dropped: Dict[str, int] = {}
    for raw in raw_events:
        clean, rejection, counts = _sanitize_event(raw)
        for key, count in counts.items():
            dropped[key] = dropped.get(key, 0) + count
        if rejection is not None:
            rejections.append(rejection)
            continue
        if clean is not None:
            cleaned.append(clean)
    return cleaned, rejections, dropped


# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------


def _sha256_file(path: str) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ModelState:
    model_path: str
    manifest_path: str
    detector: Optional[NetworkDetector] = None
    error: str = ""
    artifact_sha256: Optional[str] = None
    manifest_sha256: Optional[str] = None
    loaded_at: Optional[str] = None
    load_duration_ms: Optional[float] = None

    @property
    def ready(self) -> bool:
        return self.detector is not None

    def load(self, rules: Any) -> None:
        started = time.perf_counter()
        self.artifact_sha256 = _sha256_file(self.model_path)
        self.manifest_sha256 = _sha256_file(self.manifest_path)
        try:
            self.detector = load_network_detector(
                model_path=self.model_path,
                manifest_path=self.manifest_path,
                rules=rules,
            )
            self.error = ""
            self.loaded_at = datetime.now(timezone.utc).isoformat()
            LOGGER.info(
                "model loaded artifact_sha256=%s manifest_sha256=%s",
                self.artifact_sha256,
                self.manifest_sha256,
            )
        except Exception as exc:  # noqa: BLE001 - reported through /readyz
            self.detector = None
            self.error = redact_text(exc)
            LOGGER.error("model load failed: %s", self.error)
        finally:
            self.load_duration_ms = round((time.perf_counter() - started) * 1000.0, 3)

    def metadata(self, include_contract: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "ready": self.ready,
            "model_path": self.model_path,
            "manifest_path": self.manifest_path,
            "artifact_sha256": self.artifact_sha256,
            "manifest_sha256": self.manifest_sha256,
            "loaded_at": self.loaded_at,
            "load_duration_ms": self.load_duration_ms,
            "error": self.error,
        }
        if include_contract and self.detector is not None:
            payload["contract"] = self.detector.contract()
        return payload


# ---------------------------------------------------------------------------
# Bounded event buffer
# ---------------------------------------------------------------------------


class EventBuffer:
    """Bounded, insertion-ordered in-memory event store.

    The service is deliberately not a durable ingestion system.  The buffer is
    an audit binding so a single deployment can accept events over time and
    then score the resulting windows.  When the bound is reached the oldest
    event is evicted and the eviction count is reported; events are never
    silently treated as normal.
    """

    def __init__(self, max_events: int) -> None:
        if max_events < 1:
            raise ValueError("max_buffer_events must be >= 1")
        self.max_events = max_events
        self._events: "OrderedDict[str, StructuredEvent]" = OrderedDict()
        self._quality: Dict[str, List[QualityRecord]] = {}
        self._batch_quality: List[QualityRecord] = []
        self._duplicate_ids: set = set()
        self.evicted = 0

    def add(self, parse: LadParseResult) -> Tuple[int, int]:
        quality_by_id: Dict[str, List[QualityRecord]] = {}
        for record in parse.quality:
            if record.event_id:
                quality_by_id.setdefault(record.event_id, []).append(record)

        added = 0
        duplicates = 0
        for event in parse.events:
            if event.event_id in self._events:
                duplicates += 1
                self._duplicate_ids.add(str(event.event_id))
                self._quality.setdefault(str(event.event_id), []).append(
                    QualityRecord(
                        reason="duplicate_event_id",
                        host_id=event.host_id,
                        service=event.service,
                        window_start=event.timestamp,
                        event_id=str(event.event_id),
                    )
                )
                continue
            self._events[str(event.event_id)] = event
            self._quality[str(event.event_id)] = [
                record
                for record in quality_by_id.get(str(event.event_id), [])
                if record.reason != "out_of_order_event"
            ]
            added += 1

        for record in parse.quality:
            if not record.event_id:
                self._batch_quality.append(record)

        while len(self._events) > self.max_events:
            oldest_id, _ = self._events.popitem(last=False)
            self._quality.pop(str(oldest_id), None)
            self.evicted += 1

        if len(self._batch_quality) > self.max_events:
            self._batch_quality = self._batch_quality[-self.max_events:]

        return added, duplicates

    def snapshot(self) -> List[StructuredEvent]:
        return list(self._events.values())

    def combined_parse(self) -> LadParseResult:
        events = self.snapshot()
        quality: List[QualityRecord] = []
        for event in events:
            quality.extend(self._quality.get(str(event.event_id), []))
        quality.extend(self._batch_quality)

        out_of_order: List[QualityRecord] = []
        last_by_stream: Dict[Tuple[str, str], datetime] = {}
        for event in events:
            stream = (event.host_id, event.service)
            previous = last_by_stream.get(stream)
            if previous is not None and event.timestamp < previous:
                out_of_order.append(
                    QualityRecord(
                        reason="out_of_order_event",
                        host_id=event.host_id,
                        service=event.service,
                        window_start=event.timestamp,
                        event_id=str(event.event_id),
                    )
                )
            if previous is None or event.timestamp > previous:
                last_by_stream[stream] = event.timestamp
        quality.extend(out_of_order)

        return LadParseResult(
            events=events,
            quality=quality,
            duplicate_event_ids=sorted(self._duplicate_ids),
            out_of_order_count=len(out_of_order),
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "buffered_events": len(self._events),
            "buffer_capacity": self.max_events,
            "evicted_events": self.evicted,
            "duplicate_event_ids": len(self._duplicate_ids),
        }

    def clear(self) -> None:
        self._events.clear()
        self._quality.clear()
        self._batch_quality.clear()
        self._duplicate_ids.clear()
        self.evicted = 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ApiError(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.detail = detail or {}


def _parse_aware(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


class Engine:
    """Transport-independent implementation used by the HTTP handler and tests."""

    def __init__(
        self,
        config: ServiceConfig,
        detector: Optional[NetworkDetector] = None,
        provider: Optional[Any] = None,
    ) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._started_monotonic = time.monotonic()
        self._buffer = EventBuffer(config.max_buffer_events)
        self._provider_override = provider

        network_config = load_network_config(config.network_config_path)
        self._rules = network_config.rules
        self._window_size_minutes = network_config.simulator.window_size_minutes

        self._model = ModelState(
            model_path=config.model_path,
            manifest_path=config.manifest_path,
        )
        if detector is not None:
            self._model.detector = detector
            self._model.artifact_sha256 = _sha256_file(config.model_path)
            self._model.manifest_sha256 = _sha256_file(config.manifest_path)
            self._model.loaded_at = datetime.now(timezone.utc).isoformat()
        else:
            self._model.load(self._rules)

        try:
            self._correlation_config = load_correlation_config(
                config.correlation_config_path or None
            )
        except Exception as exc:  # noqa: BLE001 - reported per request
            LOGGER.warning("correlation config unavailable: %s", redact_text(exc))
            self._correlation_config = CorrelationConfig()

    # -- metadata ---------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "service": "lad-anomaly-detector",
            "service_version": self.config.service_version,
            "deployment": self.deployment_info(),
            "uptime_seconds": round(time.monotonic() - self._started_monotonic, 3),
        }

    def deployment_info(self) -> Dict[str, Any]:
        return {
            "mode": self.config.deployment_mode,
            "containerized": os.path.exists("/.dockerenv")
            or os.environ.get("container") == "docker",
            "binding": "http",
            "service_version": self.config.service_version,
            "contract_version": CONTRACT_VERSION,
            "feature_schema_version": NETWORK_FEATURE_SCHEMA_VERSION,
        }

    def readiness(self) -> Tuple[int, Dict[str, Any]]:
        model = self._model.metadata(include_contract=False)
        if self._model.ready:
            return HTTPStatus.OK, {
                "status": "ready",
                "deployment": self.deployment_info(),
                "model": model,
                "buffer": self._buffer.stats(),
            }
        return HTTPStatus.SERVICE_UNAVAILABLE, {
            "status": "not_ready",
            "deployment": self.deployment_info(),
            "model": model,
            "buffer": self._buffer.stats(),
        }

    def model_info(self) -> Tuple[int, Dict[str, Any]]:
        if not self._model.ready:
            return HTTPStatus.SERVICE_UNAVAILABLE, {
                "status": "not_ready",
                "model": self._model.metadata(include_contract=False),
            }
        return HTTPStatus.OK, {
            "status": "ready",
            "deployment": self.deployment_info(),
            "model": self._model.metadata(include_contract=True),
        }

    # -- ingestion / inference -------------------------------------------
    def ingest(
        self,
        raw_events: Sequence[Any],
        request_id: str = "",
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        sanitized, ingress_rejections, dropped = _sanitize_batch(raw_events)
        if len(sanitized) + len(ingress_rejections) > self.config.max_events_per_request:
            raise ApiError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "too_many_events",
                "the request exceeds the configured event bound",
                {
                    "limit": self.config.max_events_per_request,
                    "received": len(sanitized) + len(ingress_rejections),
                },
            )

        parse = parse_lad_events(
            sanitized,
            LadParseOptions(max_events=self.config.max_events_per_request),
        )
        if ingress_rejections:
            parse.rejections[:0] = ingress_rejections

        with self._lock:
            added, duplicates = self._buffer.add(parse)
            stats = self._buffer.stats()

        return {
            "request_id": request_id,
            "accepted_now": added,
            "duplicates_now": duplicates,
            "rejected_now": parse.rejected_count,
            "request_parse": parse.summary(),
            "dropped_metadata_keys": dict(sorted(dropped.items())),
            "buffer": stats,
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }

    def infer(
        self,
        raw_events: Optional[Sequence[Any]] = None,
        use_buffer: bool = True,
        request_id: str = "",
    ) -> Dict[str, Any]:
        return self._run(
            raw_events or [],
            use_buffer=use_buffer,
            correlate=False,
            explain=None,
            request_id=request_id,
        )

    def analyze(
        self,
        raw_events: Optional[Sequence[Any]] = None,
        use_buffer: bool = True,
        explain: Optional[bool] = None,
        request_id: str = "",
    ) -> Dict[str, Any]:
        return self._run(
            raw_events or [],
            use_buffer=use_buffer,
            correlate=True,
            explain=explain,
            request_id=request_id,
        )

    # -- internals --------------------------------------------------------
    def _run(
        self,
        raw_events: Sequence[Any],
        use_buffer: bool,
        correlate: bool,
        explain: Optional[bool],
        request_id: str,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        if len(raw_events) > self.config.max_events_per_request:
            raise ApiError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "too_many_events",
                "the request exceeds the configured event bound",
                {
                    "limit": self.config.max_events_per_request,
                    "received": len(raw_events),
                },
            )

        sanitized, ingress_rejections, dropped = _sanitize_batch(raw_events)
        request_parse = parse_lad_events(
            sanitized,
            LadParseOptions(max_events=self.config.max_events_per_request),
        )
        if ingress_rejections:
            request_parse.rejections[:0] = ingress_rejections

        if not use_buffer and not request_parse.events:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "no_events",
                "the request contained no accepted events and use_buffer is false",
                {"request_parse": request_parse.summary()},
            )

        with self._lock:
            if use_buffer:
                added, duplicates = self._buffer.add(request_parse)
                combined = self._buffer.combined_parse()
                buffer_stats = self._buffer.stats()
            else:
                added, duplicates = 0, 0
                combined = request_parse
                buffer_stats = self._buffer.stats()

            if not combined.events:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    "no_events",
                    "no accepted events are available to score",
                    {
                        "request_parse": request_parse.summary(),
                        "buffer": buffer_stats,
                    },
                )

            quality_by_window = combined.quality_by_window(self._window_size_minutes)
            window_grid: Dict[Tuple[str, str], List[datetime]] = {}
            for host, service, start in quality_by_window:
                window_grid.setdefault((host, service), []).append(start)

            windows = build_network_windows(
                combined.events,
                quality_by_window=quality_by_window,
                window_size_minutes=self._window_size_minutes,
                rolling_windows=self._rules.rolling_windows,
                rolling_min_history=self._rules.rolling_min_history,
                window_grid=window_grid,
            )
            if len(windows) > self.config.max_windows_per_request:
                raise ApiError(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "too_many_windows",
                    "the derived window count exceeds the configured bound",
                    {
                        "limit": self.config.max_windows_per_request,
                        "windows": len(windows),
                    },
                )

            if not self._model.ready:
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "model_unavailable",
                    "the model artifact could not be loaded; no window was scored",
                    {
                        "model": self._model.metadata(include_contract=False),
                        "request_parse": request_parse.summary(),
                        "buffer": buffer_stats,
                    },
                )

            results = self._model.detector.score_windows(windows)  # type: ignore[union-attr]
            evidence = self._build_evidence(results, combined.events)

            correlation: Optional[Dict[str, Any]] = None
            explanation_state: Optional[Dict[str, Any]] = None
            if correlate:
                correlation, explanation_state = self._correlate(
                    results, evidence, explain=explain
                )

        by_status: Dict[str, int] = {}
        for result in results:
            by_status[result.status.value] = by_status.get(result.status.value, 0) + 1

        payload: Dict[str, Any] = {
            "request_id": request_id,
            "deployment": self.deployment_info(),
            "model": self._model.metadata(include_contract=False),
            "ingest": {
                "request_events": len(raw_events),
                "accepted_now": added,
                "duplicates_now": duplicates,
                "request_parse": request_parse.summary(),
                "dropped_metadata_keys": dict(sorted(dropped.items())),
                "buffer": buffer_stats,
            },
            "windows": {
                "total": len(windows),
                "window_size_minutes": self._window_size_minutes,
                "by_status": by_status,
                "scored": by_status.get(InferenceStatus.SCORED.value, 0),
                "unscored": len(windows) - by_status.get(InferenceStatus.SCORED.value, 0),
            },
            "results": [result.to_dict() for result in results],
            "correlation": correlation,
            "explanation": explanation_state,
            "timing_ms": {
                "total": round((time.perf_counter() - started) * 1000.0, 3),
            },
        }
        return payload

    def _build_evidence(
        self,
        results: Sequence[Any],
        events: Sequence[StructuredEvent],
    ) -> Dict[str, WindowEvidence]:
        index: Dict[Tuple[str, str], List[Tuple[datetime, str]]] = {}
        for event in events:
            index.setdefault((event.host_id, event.service), []).append(
                (event.timestamp, str(event.event_id))
            )
        for stream in index:
            index[stream].sort(key=lambda item: item[0])

        limit = self._correlation_config.max_supporting_events_per_window
        evidence: Dict[str, WindowEvidence] = {}
        for result in results:
            if result.status is not InferenceStatus.SCORED or not result.is_anomaly:
                continue
            start = _parse_aware(result.window_start)
            end = _parse_aware(result.window_end)
            if start is None or end is None or not result.host_id or not result.service_name:
                continue
            ids = [
                event_id
                for timestamp, event_id in index.get(
                    (result.host_id, result.service_name), []
                )
                if start <= timestamp < end
            ][:limit]
            if ids:
                evidence[str(result.window_id)] = WindowEvidence(
                    window_id=str(result.window_id),
                    supporting_event_ids=tuple(ids),
                )
        return evidence

    def _correlate(
        self,
        results: Sequence[Any],
        evidence: Mapping[str, WindowEvidence],
        explain: Optional[bool],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        correlation_config = self._correlation_config
        provider = None
        explanation_state: Dict[str, Any] = {
            "requested": bool(explain) if explain is not None else bool(correlation_config.explain),
            "enabled": False,
            "provider": None,
            "note": "",
        }

        requested = (
            bool(explain) if explain is not None else bool(correlation_config.explain)
        )
        if requested:
            probe = replace(correlation_config, explain=True)
            if self._provider_override is not None:
                provider = self._provider_override
                correlation_config = probe
                explanation_state.update(
                    {"enabled": True, "provider": getattr(provider, "name", type(provider).__name__)}
                )
            else:
                try:
                    provider = build_explanation_provider(probe)
                except Exception as exc:  # noqa: BLE001 - explicit, no silent mock
                    explanation_state["note"] = redact_text(exc)
                    provider = None
                if provider is None:
                    explanation_state["note"] = (
                        "no explanation provider is configured; the ML decision is unchanged"
                    )
                else:
                    correlation_config = probe
                    explanation_state.update(
                        {
                            "enabled": True,
                            "provider": getattr(
                                provider, "name", type(provider).__name__
                            ),
                        }
                    )
        else:
            correlation_config = replace(correlation_config, explain=False)

        pipeline = EvidenceCorrelationPipeline(
            correlation_config,
            provider=provider,
        )
        outcome = pipeline.correlate(results, evidence)
        data = outcome.to_dict()

        cap_incidents = self.config.max_incidents_in_response
        cap_records = self.config.max_records_in_response
        data["incidents_truncated"] = len(data.get("incidents", [])) > cap_incidents
        data["incidents"] = data.get("incidents", [])[:cap_incidents]
        for key in (
            "candidates",
            "unscored",
            "scored_not_flagged",
            "rejected",
            "duplicates",
            "llm_interpretations",
        ):
            values = data.get(key) or []
            data[f"{key}_truncated"] = len(values) > cap_records
            data[key] = values[:cap_records]

        explanation_state["grounded"] = data.get("llm", {}).get("grounded", 0)
        explanation_state["attempted"] = data.get("llm", {}).get("attempted", 0)
        explanation_state["fallback_used"] = data.get("llm", {}).get(
            "fallback_used", 0
        )
        return data, explanation_state


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "LADAuditService/1.0"
    protocol_version = "HTTP/1.1"

    # -- plumbing ---------------------------------------------------------
    def _request_id(self) -> str:
        supplied = self.headers.get("X-Request-ID", "")
        if supplied and _REQUEST_ID_RE.fullmatch(supplied.strip()):
            return supplied.strip()
        return uuid.uuid4().hex

    def _send_json(self, status: int, payload: Mapping[str, Any], request_id: str) -> None:
        body = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Request-ID", request_id)
        self.end_headers()
        self.wfile.write(body)

    def _send_error(
        self,
        status: int,
        code: str,
        message: str,
        request_id: str,
        detail: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._send_json(
            status,
            {
                "request_id": request_id,
                "error": {
                    "code": code,
                    "message": redact_text(message),
                    "detail": dict(detail or {}),
                },
            },
            request_id,
        )

    def _read_json(self) -> Any:
        if self.headers.get("Transfer-Encoding"):
            raise ApiError(
                HTTPStatus.LENGTH_REQUIRED,
                "chunked_not_supported",
                "a Content-Length body is required",
            )
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type and content_type != "application/json":
            raise ApiError(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
                "Content-Type must be application/json",
            )
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            raise ApiError(
                HTTPStatus.LENGTH_REQUIRED,
                "content_length_required",
                "a Content-Length header is required",
            )
        try:
            length = int(length_header)
        except ValueError as exc:
            raise ApiError(
                HTTPStatus.BAD_REQUEST, "invalid_content_length", "Content-Length is not an integer"
            ) from exc
        if length <= 0:
            raise ApiError(HTTPStatus.BAD_REQUEST, "empty_body", "the request body is empty")
        if length > self.server.max_request_bytes:  # type: ignore[attr-defined]
            raise ApiError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request_too_large",
                "the request body exceeds the configured bound",
                {"limit": self.server.max_request_bytes},  # type: ignore[attr-defined]
            )
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ApiError(
                HTTPStatus.BAD_REQUEST, "invalid_json", "the request body is not valid JSON"
            ) from exc

    def _events_from_body(self, body: Any) -> Tuple[List[Any], bool, Optional[bool]]:
        if isinstance(body, list):
            return body, True, None
        if not isinstance(body, Mapping):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "invalid_body",
                "the body must be a JSON object or an array of LAD events",
            )
        events = body.get("events")
        if events is None:
            events = []
        if not isinstance(events, list):
            raise ApiError(
                HTTPStatus.BAD_REQUEST, "invalid_events", "events must be an array"
            )
        use_buffer = body.get("use_buffer", True)
        if not isinstance(use_buffer, bool):
            raise ApiError(
                HTTPStatus.BAD_REQUEST, "invalid_use_buffer", "use_buffer must be a boolean"
            )
        explain = body.get("explain")
        if explain is not None and not isinstance(explain, bool):
            raise ApiError(
                HTTPStatus.BAD_REQUEST, "invalid_explain", "explain must be a boolean"
            )
        return events, use_buffer, explain

    # -- request logging --------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:
        # Suppress the default unstructured access log; log_request emits JSON.
        return

    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        LOGGER.info(
            json.dumps(
                {
                    "event": "http_request",
                    "method": self.command,
                    "path": self.path.split("?", 1)[0],
                    "status": int(code) if str(code).isdigit() else str(code),
                    "bytes": size,
                    "client": self.client_address[0] if self.client_address else "",
                },
                sort_keys=True,
            )
        )

    # -- routing ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        request_id = self._request_id()
        path = self.path.split("?", 1)[0]
        try:
            engine: Engine = self.server.engine  # type: ignore[attr-defined]
            if path == "/healthz":
                self._send_json(HTTPStatus.OK, engine.health(), request_id)
                return
            if path == "/readyz":
                status, payload = engine.readiness()
                self._send_json(status, payload, request_id)
                return
            if path in {"/v1/model", "/v1/contract"}:
                status, payload = engine.model_info()
                self._send_json(status, payload, request_id)
                return
            if path == "/":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "service": "lad-anomaly-detector",
                        "service_version": engine.config.service_version,
                        "endpoints": [
                            "GET /healthz",
                            "GET /readyz",
                            "GET /v1/model",
                            "POST /v1/events",
                            "POST /v1/infer",
                            "POST /v1/analyze",
                        ],
                        "note": "audit/deployment binding; not a production API",
                    },
                    request_id,
                )
                return
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "unknown path", request_id)
        except ApiError as exc:
            self._send_error(exc.status, exc.code, exc.message, request_id, exc.detail)
        except Exception as exc:  # noqa: BLE001 - never leak a traceback
            LOGGER.exception("unhandled GET error")
            self._send_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "internal_error",
                f"unexpected error: {redact_text(exc)}",
                request_id,
            )

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        request_id = self._request_id()
        path = self.path.split("?", 1)[0]
        try:
            engine: Engine = self.server.engine  # type: ignore[attr-defined]
            if path not in {"/v1/events", "/v1/infer", "/v1/analyze"}:
                self._send_error(HTTPStatus.NOT_FOUND, "not_found", "unknown path", request_id)
                return

            body = self._read_json()
            events, use_buffer, explain = self._events_from_body(body)

            if path == "/v1/events":
                if not events:
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "no_events",
                        "events must contain at least one LAD envelope",
                    )
                payload = engine.ingest(events, request_id=request_id)
            elif path == "/v1/infer":
                payload = engine.infer(
                    events, use_buffer=use_buffer, request_id=request_id
                )
            else:
                payload = engine.analyze(
                    events,
                    use_buffer=use_buffer,
                    explain=explain,
                    request_id=request_id,
                )

            self._send_json(HTTPStatus.OK, payload, request_id)
        except ApiError as exc:
            self._send_error(exc.status, exc.code, exc.message, request_id, exc.detail)
        except Exception as exc:  # noqa: BLE001 - never leak a traceback
            LOGGER.exception("unhandled POST error")
            self._send_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "internal_error",
                f"unexpected error: {redact_text(exc)}",
                request_id,
            )


class AuditHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: Tuple[str, int], engine: Engine) -> None:
        self.engine = engine
        self.max_request_bytes = engine.config.max_request_bytes
        super().__init__(address, RequestHandler)

    def server_bind(self) -> None:
        # Avoid a long reverse-DNS lookup on startup in restricted containers.
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        super().server_bind()


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        return value.to_dict()
    return str(value)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def build_server(config: ServiceConfig) -> AuditHTTPServer:
    engine = Engine(config)
    return AuditHTTPServer((config.host, config.port), engine)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("LAD_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LAD_PORT", "8080")))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--deployment-mode", default=None)
    args = parser.parse_args(argv)

    config = ServiceConfig.from_env()
    if args.host:
        config = replace(config, host=args.host)
    if args.port:
        config = replace(config, port=args.port)
    if args.model_path:
        config = replace(config, model_path=args.model_path)
    if args.manifest_path:
        config = replace(config, manifest_path=args.manifest_path)
    if args.deployment_mode:
        config = replace(config, deployment_mode=args.deployment_mode)

    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    engine = Engine(config)
    server = AuditHTTPServer((config.host, config.port), engine)
    actual_host, actual_port = server.server_address[:2]
    LOGGER.info(
        json.dumps(
            {
                "event": "service_start",
                "host": actual_host,
                "port": actual_port,
                "deployment": engine.deployment_info(),
                "model_ready": engine._model.ready,  # noqa: SLF001 - startup report
                "artifact_sha256": engine._model.artifact_sha256,  # noqa: SLF001
            },
            sort_keys=True,
        )
    )

    def _shutdown(signum: int, frame: Any) -> None:
        LOGGER.info("signal received signum=%s; shutting down", signum)
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
