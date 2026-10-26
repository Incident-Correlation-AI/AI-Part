"""Evidence-correlation pipeline: ML result → upstream correlation → incident.

The flow is deliberately one-way and offline by default::

    InferenceResult batch + WindowEvidence index
        → canonical route_result() (scored only)
        → candidate validation and reference redaction
        → upstream correlate_anomalies()
        → canonical LAD incident draft
        → optional evidence-grounded explanation (reused network.llm adapter)

Nothing here re-scores a window, changes a model decision, or turns an
unscored result into an anomaly.  An LLM outage cannot erase a valid ML score:
the incident draft is built first from deterministic evidence and only then
augmented with validated explanations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import logging
import re

from ..inference.contract import (
    RoutingDecision,
    route_result,
)
from ..network.llm import (
    HttpExplanationProvider,
    MockExplanationProvider,
    explain_result,
)
from .config import CorrelationConfig
from .evidence import (
    is_safe_identity,
    WindowEvidence,
    normalise_evidence_index,
    validate_window_evidence,
)
from .incident import (
    CorrelationCandidate,
    build_incident,
    correlate_candidates,
    deterministic_incident_id,
    is_finite_number,
)
from .upstream import UPSTREAM_REVISION


_SEVERITIES = frozenset({"none", "low", "medium", "high", "critical", "unknown"})
_MAX_ENTITY_TEXT = 255
_MAX_STATEMENT = 2048
_MAX_DIAGNOSTIC = 300

logger = logging.getLogger(__name__)

_DIAGNOSTIC_SECRET_RE = re.compile(
    r"(?i)\b(password|passwd|secret|api[_-]?key|access_token|refresh_token|"
    r"session_token|token|authorization|bearer|cookie|private_key)\b\s*[=:]\s*\S+"
)


@dataclass
class CorrelationRecord:
    """A result that did **not** become a correlated anomaly candidate.

    ``status`` and ``reason_code`` are the ML contract's values.  The extra
    ``integration_reason`` explains the correlation-specific decision without
    inventing a competing status vocabulary.
    """

    window_id: Optional[str]
    status: str
    reason_code: str = ""
    reason: str = ""
    integration_reason: str = ""
    detail: str = ""
    is_anomaly: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "integration_reason": self.integration_reason,
            "detail": self.detail,
            "is_anomaly": self.is_anomaly,
        }


@dataclass
class CorrelationError:
    """An integration-stage failure, kept separate from ML failures."""

    stage: str
    error_type: str
    message: str
    window_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "error_type": self.error_type,
            "message": self.message,
            "window_id": self.window_id,
        }


@dataclass
class CorrelationOutcome:
    """Complete, explicit outcome of one correlation run."""

    config: Dict[str, Any] = field(default_factory=dict)
    incidents: List[Dict[str, Any]] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    unscored: List[CorrelationRecord] = field(default_factory=list)
    scored_not_flagged: List[CorrelationRecord] = field(default_factory=list)
    rejected: List[CorrelationRecord] = field(default_factory=list)
    duplicates: List[CorrelationRecord] = field(default_factory=list)
    unused_evidence: List[str] = field(default_factory=list)
    orphan_evidence: List[str] = field(default_factory=list)
    errors: List[CorrelationError] = field(default_factory=list)
    llm_interpretations: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def incident_count(self) -> int:
        return len(self.incidents)

    def counts(self) -> Dict[str, int]:
        return {
            "incidents": len(self.incidents),
            "candidates": len(self.candidates),
            "unscored": len(self.unscored),
            "scored_not_flagged": len(self.scored_not_flagged),
            "rejected": len(self.rejected),
            "duplicates": len(self.duplicates),
            "unused_evidence": len(self.unused_evidence),
            "orphan_evidence": len(self.orphan_evidence),
            "errors": len(self.errors),
            "llm_interpretations": len(self.llm_interpretations),
        }

    def unscored_by_status(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self.unscored:
            counts[record.status] = counts.get(record.status, 0) + 1
        return counts

    def llm_summary(self) -> Dict[str, Any]:
        entries = self.llm_interpretations
        return {
            "attempted": len(entries),
            "grounded": sum(1 for e in entries if e.get("grounded")),
            "explained": sum(1 for e in entries if e.get("explained")),
            "fallback_used": sum(1 for e in entries if e.get("fallback_used")),
            "attached_hypotheses": sum(
                1 for e in entries if e.get("attached_hypothesis")
            ),
            "attached_checks": sum(int(e.get("attached_checks", 0)) for e in entries),
            "ungrounded": sum(1 for e in entries if not e.get("grounded")),
            "errors": sum(1 for e in self.errors if e.stage == "explanation"),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": dict(self.config),
            "counts": self.counts(),
            "unscored_by_status": self.unscored_by_status(),
            "upstream_revision": UPSTREAM_REVISION,
            "incidents": list(self.incidents),
            "candidates": list(self.candidates),
            "unscored": [r.to_dict() for r in self.unscored],
            "scored_not_flagged": [r.to_dict() for r in self.scored_not_flagged],
            "rejected": [r.to_dict() for r in self.rejected],
            "duplicates": [r.to_dict() for r in self.duplicates],
            "unused_evidence": list(self.unused_evidence),
            "orphan_evidence": list(self.orphan_evidence),
            "errors": [e.to_dict() for e in self.errors],
            "llm": self.llm_summary(),
            "llm_interpretations": list(self.llm_interpretations),
        }


def build_explanation_provider(config: CorrelationConfig) -> Optional[Any]:
    """Build the configured provider; no network call is made here."""
    if config.llm_provider == "none":
        return None
    if config.llm_provider == "mock":
        return MockExplanationProvider()
    if config.llm_provider == "http":
        if not config.llm_endpoint:
            raise ValueError("llm_endpoint is required for the http provider")
        return HttpExplanationProvider(
            endpoint=config.llm_endpoint,
            model=config.llm_model,
            api_key_env=config.llm_api_key_env,
            timeout_seconds=config.llm_timeout_seconds,
        )
    raise ValueError(f"unsupported LLM provider {config.llm_provider!r}")


class EvidenceCorrelationPipeline:
    """Correlate already-scored ML results and build canonical incidents.

    Parameters
    ----------
    config:
        Bounds and switches.  Defaults are offline: explanations are off and
        ``llm_provider="mock"`` never contacts a network.
    provider:
        Optional provider injection (used by tests and by callers that already
        own a provider).  Supplying one enables explanation even when
        ``config.explain`` is false.
    clock:
        Optional timezone-aware clock for deterministic ``created_at`` /
        ``updated_at`` values.
    correlator:
        Optional injection point for the upstream correlation callable.  The
        default is the vendored upstream implementation.  It is injectable so
        an upstream failure can be tested without modifying the code.
    """

    def __init__(
        self,
        config: Optional[CorrelationConfig] = None,
        provider: Optional[Any] = None,
        clock: Optional[Callable[[], datetime]] = None,
        correlator: Optional[Callable[..., List[List[CorrelationCandidate]]]] = None,
    ) -> None:
        self.config = config or CorrelationConfig.from_env()
        self._provider = provider
        self._explain = bool(self.config.explain or provider is not None)
        if self._explain and self._provider is None:
            self._provider = build_explanation_provider(self.config)
            if self._provider is None:
                self._explain = False
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._correlator = correlator or correlate_candidates

    # -- public API -------------------------------------------------------
    def correlate(
        self,
        results: Sequence[Any],
        evidence: Any = None,
    ) -> CorrelationOutcome:
        """Correlate a batch and return an explicit outcome.

        ``evidence`` is a mapping ``window_id -> WindowEvidence`` (or an
        accepted shorthand) and is matched by the canonical ``window_id``.
        Missing evidence never causes an event or ID to be invented.
        """
        evidence_index = normalise_evidence_index(evidence)
        outcome = CorrelationOutcome(config=self.config.to_dict())
        seen_ids = set()
        result_ids = set()
        candidate_windows = set()
        candidates: List[CorrelationCandidate] = []

        for raw in results:
            decision = route_result(raw)
            window_id = decision.window_id or _text(getattr(raw, "window_id", None))
            if window_id:
                result_ids.add(window_id)

            if window_id and window_id in seen_ids:
                outcome.duplicates.append(
                    CorrelationRecord(
                        window_id=window_id,
                        status=decision.status,
                        reason_code=decision.reason_code,
                        reason=decision.reason,
                        integration_reason="duplicate_window_id",
                        detail=(
                            "the same window_id appeared earlier in this batch; "
                            "the later occurrence was not correlated"
                        ),
                        is_anomaly=decision.is_anomaly,
                    )
                )
                continue
            if window_id:
                seen_ids.add(window_id)

            if not decision.forwardable:
                outcome.unscored.append(
                    CorrelationRecord(
                        window_id=decision.window_id,
                        status=decision.status,
                        reason_code=decision.reason_code,
                        reason=decision.reason,
                        integration_reason="not_forwardable",
                        detail=decision.detail,
                        is_anomaly=None,
                    )
                )
                continue

            is_anomaly = getattr(raw, "is_anomaly", None)
            if is_anomaly is False:
                outcome.scored_not_flagged.append(
                    CorrelationRecord(
                        window_id=decision.window_id,
                        status=decision.status,
                        reason_code=decision.reason_code,
                        reason=decision.reason,
                        integration_reason="scored_not_flagged",
                        detail=(
                            "scored successfully and not flagged; it is not an "
                            "anomaly candidate"
                        ),
                        is_anomaly=False,
                    )
                )
                continue
            if is_anomaly is not True:
                outcome.rejected.append(
                    CorrelationRecord(
                        window_id=decision.window_id,
                        status=decision.status,
                        reason_code=decision.reason_code,
                        reason=decision.reason,
                        integration_reason="missing_decision",
                        detail="a scored result must carry is_anomaly=True/False",
                        is_anomaly=None,
                    )
                )
                continue

            candidate, rejection = self._build_candidate(raw, decision, evidence_index)
            if candidate is None:
                outcome.rejected.append(rejection)  # type: ignore[arg-type]
                continue
            candidates.append(candidate)
            candidate_windows.add(candidate.window_id)
            outcome.candidates.append(candidate.to_dict())

        outcome.orphan_evidence = sorted(set(evidence_index) - result_ids)
        outcome.unused_evidence = sorted(set(evidence_index) - candidate_windows)

        if not candidates:
            logger.info(
                "correlation run had no eligible candidate: %s",
                outcome.counts(),
            )
            return outcome

        try:
            groups = self._correlator(
                candidates, window_seconds=self.config.window_seconds
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as an explicit error
            outcome.errors.append(
                CorrelationError(
                    stage="upstream_correlation",
                    error_type=type(exc).__name__,
                    message=_sanitise_diagnostic(
                        f"upstream correlation failed: {exc}"
                    ),
                )
            )
            logger.warning(
                "correlation stage=upstream_correlation error_type=%s",
                type(exc).__name__,
            )
            return outcome

        explanations = self._explain_candidates(groups, outcome)

        for group in groups:
            if not group:
                continue
            incident_id = deterministic_incident_id(group)
            contributing: List[Dict[str, Any]] = []
            extra_hypotheses: List[Dict[str, Any]] = []
            extra_checks: List[Dict[str, Any]] = []
            remaining_hypotheses = self.config.max_llm_hypotheses_per_incident

            for candidate in group:
                entry = explanations.get(candidate.window_id)
                if entry is None:
                    continue
                attached = _attach_explanation(
                    candidate,
                    entry,
                    1 if remaining_hypotheses > 0 else 0,
                    self.config.include_llm_checks,
                )
                remaining_hypotheses -= len(attached["hypotheses"])
                extra_hypotheses.extend(attached["hypotheses"])
                extra_checks.extend(attached["checks"])
                entry["incident_id"] = incident_id
                entry["attached_hypothesis"] = bool(attached["hypotheses"])
                entry["attached_checks"] = len(attached["checks"])
                if attached["attachment_reason"]:
                    entry["attachment_reason"] = attached["attachment_reason"]
                contributing.append(entry)

            try:
                incident = build_incident(
                    group,
                    self.config,
                    now=self._clock(),
                    extra_hypotheses=extra_hypotheses,
                    extra_checks=extra_checks,
                    llm_metadata=contributing,
                )
            except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
                outcome.errors.append(
                    CorrelationError(
                        stage="incident_build",
                        error_type=type(exc).__name__,
                        message=_sanitise_diagnostic(
                            f"could not build an incident: {exc}"
                        ),
                        window_id=group[0].window_id,
                    )
                )
                logger.warning(
                    "correlation stage=incident_build error_type=%s window_id=%s",
                    type(exc).__name__,
                    group[0].window_id,
                )
                continue
            outcome.incidents.append(incident)

        # Preserve every explanation in the outcome even when the incident
        # build failed; the ML score is never discarded because of an LLM.
        outcome.llm_interpretations = list(explanations.values())
        logger.info("correlation run complete: %s", outcome.counts())
        return outcome

    # -- candidate construction ------------------------------------------
    def _build_candidate(
        self,
        result: Any,
        decision: RoutingDecision,
        evidence_index: Mapping[str, WindowEvidence],
    ) -> Tuple[Optional[CorrelationCandidate], Optional[CorrelationRecord]]:
        window_id = decision.window_id or _text(getattr(result, "window_id", None))
        if not window_id:
            return None, self._reject(
                result, decision, "missing_window_id", "window_id is required"
            )
        host_id = _bounded_text(getattr(result, "host_id", None))
        if not host_id:
            return None, self._reject(
                result,
                decision,
                "missing_host_id",
                "a scored anomaly needs a host for the incident's affected entity",
            )
        if not is_safe_identity(host_id):
            return None, self._reject(
                result,
                decision,
                "unsafe_host_identifier",
                "the host identifier looked like sensitive data or free-form text",
            )
        service = _bounded_text(getattr(result, "service_name", None))
        if not service:
            return None, self._reject(
                result,
                decision,
                "missing_service",
                "upstream correlation requires a service identifier",
            )
        if not is_safe_identity(service):
            return None, self._reject(
                result,
                decision,
                "unsafe_service_identifier",
                "the service identifier looked like sensitive data or free-form text",
            )

        start = _parse_aware(getattr(result, "window_start", None))
        end = _parse_aware(getattr(result, "window_end", None))
        if start is None:
            return None, self._reject(
                result, decision, "invalid_window_start",
                "window_start must be a timezone-aware ISO timestamp",
            )
        if end is None:
            return None, self._reject(
                result, decision, "missing_window_end",
                "window_end must be a timezone-aware ISO timestamp",
            )
        if end <= start:
            return None, self._reject(
                result, decision, "invalid_window_boundary",
                "window_end must be strictly after window_start",
            )

        raw_score = getattr(result, "raw_score", None)
        threshold = getattr(result, "threshold", None)
        if not is_finite_number(raw_score):
            return None, self._reject(
                result, decision, "missing_score",
                "a scored anomaly needs a finite raw_score",
            )
        if not is_finite_number(threshold):
            return None, self._reject(
                result, decision, "missing_threshold",
                "a scored anomaly needs a finite threshold",
            )

        record = evidence_index.get(window_id)
        if record is None:
            if self.config.require_supporting_events:
                return None, self._reject(
                    result,
                    decision,
                    "missing_event_evidence",
                    "no supporting event IDs were supplied for this window",
                )
            record = WindowEvidence(window_id=window_id)
        validation = validate_window_evidence(
            record,
            max_events=self.config.max_supporting_events_per_window,
            max_redacted_refs=self.config.max_redacted_refs_per_incident,
        )
        unsafe = validation.reject_reason()
        if unsafe:
            return None, self._reject(
                result,
                decision,
                unsafe,
                "an evidence reference looked like sensitive or free-form data; "
                "the candidate was quarantined instead of sending it downstream",
            )
        if self.config.require_supporting_events and not validation.has_supporting:
            return None, self._reject(
                result,
                decision,
                "missing_event_evidence",
                "no safe supporting event IDs remain for this window",
            )

        network = _mapping(getattr(result, "extra", None)).get("network")
        network_map = _mapping(network)
        baseline = _mapping(network_map.get("baseline"))
        rule_hits = tuple(
            dict(hit)
            for hit in (baseline.get("rule_hits") or [])
            if isinstance(hit, Mapping)
        )
        features: Dict[str, float] = {}
        for key, value in _mapping(network_map.get("evidence_features")).items():
            if is_finite_number(value):
                features[str(key)] = float(value)
        severity = str(network_map.get("severity") or "unknown")
        if severity not in _SEVERITIES:
            severity = "unknown"

        candidate = CorrelationCandidate(
            result=result,
            window_id=window_id,
            service=service,
            host_id=host_id,
            window_start=start,
            window_end=end,
            raw_score=float(raw_score),
            threshold=float(threshold),
            score_direction=str(getattr(result, "score_direction", None) or "unknown"),
            severity=severity,
            model_name=str(getattr(result, "model_name", None) or "unknown"),
            model_version=str(getattr(result, "model_version", None) or "unknown"),
            model_artifact_version=str(
                getattr(result, "model_artifact_version", None) or "unknown"
            ),
            feature_schema_version=str(
                getattr(result, "feature_schema_version", None) or "unknown"
            ),
            contract_version=str(
                getattr(result, "contract_version", None) or "unknown"
            ),
            threshold_config_version=str(
                getattr(result, "threshold_config_version", None) or "unknown"
            ),
            rule_hits=rule_hits,
            evidence_features=features,
            evidence=validation,
        )
        return candidate, None

    def _reject(
        self,
        result: Any,
        decision: RoutingDecision,
        integration_reason: str,
        detail: str,
    ) -> CorrelationRecord:
        return CorrelationRecord(
            window_id=decision.window_id or _text(getattr(result, "window_id", None)),
            status=decision.status,
            reason_code=decision.reason_code,
            reason=decision.reason,
            integration_reason=integration_reason,
            detail=detail,
            is_anomaly=True,
        )

    # -- explanation ------------------------------------------------------
    def _explain_candidates(
        self,
        groups: Sequence[Sequence[CorrelationCandidate]],
        outcome: CorrelationOutcome,
    ) -> Dict[str, Dict[str, Any]]:
        if not self._explain:
            return {}
        entries: Dict[str, Dict[str, Any]] = {}
        for group in groups:
            for candidate in group:
                try:
                    explanation = explain_result(
                        candidate.result,
                        provider=self._provider,
                        fallback=self.config.explanation_fallback,
                    )
                    entry = explanation.to_dict()
                    entry["rejected_reason"] = _sanitise_diagnostic(
                        entry.get("rejected_reason", "")
                    )
                    entry["candidate_feature"] = candidate.primary_feature
                    entry["supporting_event_count"] = len(
                        candidate.supporting_event_ids
                    )
                    entries[candidate.window_id] = entry
                except Exception as exc:  # noqa: BLE001 - ML result must survive
                    outcome.errors.append(
                        CorrelationError(
                            stage="explanation",
                            error_type=type(exc).__name__,
                            message=_sanitise_diagnostic(
                                f"explanation adapter failed: {exc}"
                            ),
                            window_id=candidate.window_id,
                        )
                    )
                    logger.warning(
                        "correlation stage=explanation error_type=%s window_id=%s",
                        type(exc).__name__,
                        candidate.window_id,
                    )
                    entries[candidate.window_id] = {
                        "window_id": candidate.window_id,
                        "explained": False,
                        "text": "",
                        "evidence": [],
                        "citations": [],
                        "recommended_checks": [],
                        "asserts_anomaly": None,
                        "grounded": False,
                        "provider": getattr(
                            self._provider, "name", type(self._provider).__name__
                        ),
                        "fallback_used": False,
                        "rejected_reason": _sanitise_diagnostic(
                            f"adapter error: {exc}"
                        ),
                    }
        return entries


# ---------------------------------------------------------------------------
# Explanation → incident contribution
# ---------------------------------------------------------------------------


def _attach_explanation(
    candidate: CorrelationCandidate,
    entry: Mapping[str, Any],
    max_hypotheses: int,
    include_checks: bool,
) -> Dict[str, Any]:
    """Convert one validated explanation into schema-safe hypothesis/checks."""
    result: Dict[str, Any] = {
        "hypotheses": [],
        "checks": [],
        "attachment_reason": "",
    }
    if not entry.get("explained") or not entry.get("grounded"):
        result["attachment_reason"] = "the explanation was not grounded"
        return result
    text = str(entry.get("text") or "").strip()
    if not text:
        result["attachment_reason"] = "the explanation was empty"
        return result
    if len(text) > _MAX_STATEMENT:
        result["attachment_reason"] = (
            f"the explanation is {len(text)} characters; the incident contract "
            f"limits hypothesis statements to {_MAX_STATEMENT}"
        )
        return result

    if max_hypotheses > 0:
        result["hypotheses"].append({
            "statement": text,
            "confidence": "low",
            "supporting_event_ids": candidate.supporting_event_ids,
            "contradicting_event_ids": candidate.contradicting_event_ids,
            "missing_evidence": [
                "The LLM interpretation is derived only from the supplied evidence.",
                "It is not a root-cause finding or proof of malicious intent.",
            ],
        })
    else:
        result["attachment_reason"] = (
            "the per-incident LLM hypothesis limit was reached"
        )

    if include_checks:
        links = candidate.supporting_event_ids[:10]
        for check in entry.get("recommended_checks") or []:
            instruction = str(check).strip()
            if not instruction or len(instruction) > _MAX_STATEMENT:
                continue
            result["checks"].append({
                "instruction": instruction,
                "evidence_links": links,
                "executes": False,
            })
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    return parsed.astimezone(timezone.utc)


def _bounded_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > _MAX_ENTITY_TEXT:
        return None
    return text


def _text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value.strip() else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sanitise_diagnostic(value: Any, limit: int = _MAX_DIAGNOSTIC) -> str:
    """Keep a short diagnostic without URLs, addresses or secret values.

    Provider and upstream exception text can echo the request, a URL containing
    credentials, an address or a token.  Only a bounded, redacted summary is
    stored in the result or written to a log.
    """
    text = " ".join(str(value or "").split())
    text = re.sub(r"https?://\S+", "<url>", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "<email>", text)
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<ip>", text)
    text = _DIAGNOSTIC_SECRET_RE.sub(r"\1=<redacted>", text)
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


__all__ = [
    "CorrelationError",
    "CorrelationOutcome",
    "CorrelationRecord",
    "EvidenceCorrelationPipeline",
    "build_explanation_provider",
]
