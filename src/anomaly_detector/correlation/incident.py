"""Map upstream correlation groups to the canonical LAD incident contract.

Upstream ``correlate_anomalies()`` decides which anomalies are related.  This
module does **not** re-implement correlation and does not ask an LLM to decide
it.  It only converts each upstream group into an incident draft that satisfies
``LAD/contracts/schemas/incident.schema.json``.

The draft keeps the five layers separate:

* observed evidence — event IDs, measurements and quality counts;
* ML assessment — raw score, threshold, direction and model metadata;
* correlation — the upstream group and its time/identity basis;
* interpretation — a deterministic hypothesis plus optional validated LLM
  interpretation;
* recommended investigation — non-executing, evidence-linked checks.

The LLM never changes ``anomaly_score``, ``detector``, the group membership or
the severity.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import uuid

from ..inference.contract import InferenceResult
from .config import CorrelationConfig
from .evidence import EvidenceValidation
from .upstream import UPSTREAM_REVISION, correlate_anomalies


_SEVERITY_ORDER = {
    "none": 0,
    "unknown": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}
_SCHEMA_SEVERITIES = frozenset({"low", "medium", "high", "critical"})
_EVIDENCE_PRIORITY = (
    "http_5xx_ratio",
    "http_5xx_count",
    "db_timeout_ratio",
    "db_connection_failure_ratio",
    "auth_failure_ratio",
    "auth_distinct_source_count",
    "p95_latency_ratio",
    "p95_latency_ms",
    "restart_count",
    "event_count",
)

#: Deterministic diagnostic checks keyed by the upstream rule/evidence code.
_CHECK_BY_CODE = {
    "http_5xx_rate": (
        "Compare the service's 5xx rate with its upstream dependency's error "
        "rate and with recent deployments."
    ),
    "http_5xx_rate_delta": (
        "Compare the 5xx-rate change with the preceding baseline and the "
        "upstream dependency's health."
    ),
    "db_timeout_rate": (
        "Check database latency, connection-pool saturation and timeout "
        "settings for the affected service."
    ),
    "db_connection_failure_rate": (
        "Check database reachability, credential rotation and connection-pool "
        "health for the affected service."
    ),
    "auth_failure_rate": (
        "Review the redacted authentication-failure sources and the account "
        "lockout policy."
    ),
    "auth_distinct_sources": (
        "Check whether the authentication failures originate from an expected "
        "client pool or a misconfigured retry loop."
    ),
    "latency_p95_ratio": (
        "Compare p95 latency with the service's rolling baseline and its "
        "upstream dependency's latency."
    ),
    "restart_surge": (
        "Inspect restart/health events and deployment history for the "
        "affected service."
    ),
    "mad_robust_zscore": (
        "Inspect event and request volume for an extreme surge or anomaly "
        "diverging significantly from the robust rolling baseline."
    ),
    "model_score": (
        "The model flagged this window without a transparent rule hit; compare "
        "the listed measurements with the preceding baseline before acting."
    ),
}


@dataclass(frozen=True)
class CorrelationCandidate:
    """One eligible, scored anomaly ready for upstream correlation."""

    result: InferenceResult
    window_id: str
    service: str
    host_id: str
    window_start: datetime
    window_end: datetime
    raw_score: float
    threshold: float
    score_direction: str
    severity: str
    model_name: str
    model_version: str
    model_artifact_version: str
    feature_schema_version: str
    contract_version: str
    threshold_config_version: str
    rule_hits: Tuple[Dict[str, Any], ...]
    evidence_features: Dict[str, float]
    evidence: EvidenceValidation

    @property
    def primary_feature(self) -> str:
        for hit in self.rule_hits:
            code = hit.get("code")
            if isinstance(code, str) and code:
                return code
        for key in _EVIDENCE_PRIORITY:
            if key in self.evidence_features:
                return key
        return "model_score"

    @property
    def supporting_event_ids(self) -> List[str]:
        return list(self.evidence.supporting_event_ids)

    @property
    def contradicting_event_ids(self) -> List[str]:
        return list(self.evidence.contradicting_event_ids)

    @property
    def redacted_refs(self) -> List[str]:
        return list(self.evidence.redacted_refs)

    def to_upstream_anomaly(self) -> Dict[str, Any]:
        """The dictionary shape expected by upstream ``correlate_anomalies``.

        Upstream reads only ``service`` and ``timestamp``; every other field is
        carried so the group can be mapped back without a second lookup.
        """
        return {
            "feature": self.primary_feature,
            "service": self.service,
            "timestamp": self.window_start.isoformat(),
            "window_id": self.window_id,
            "host_id": self.host_id,
            "supporting_event_ids": self.supporting_event_ids,
            "contradicting_event_ids": self.contradicting_event_ids,
            "redacted_refs": self.redacted_refs,
            "raw_score": self.raw_score,
            "threshold": self.threshold,
            "score_direction": self.score_direction,
            "severity": self.severity,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "service": self.service,
            "host_id": self.host_id,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "raw_score": self.raw_score,
            "threshold": self.threshold,
            "score_direction": self.score_direction,
            "severity": self.severity,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_artifact_version": self.model_artifact_version,
            "feature_schema_version": self.feature_schema_version,
            "contract_version": self.contract_version,
            "threshold_config_version": self.threshold_config_version,
            "rule_hits": [dict(hit) for hit in self.rule_hits],
            "evidence_features": dict(self.evidence_features),
            "evidence": self.evidence.to_dict(),
        }


def correlate_candidates(
    candidates: Sequence[CorrelationCandidate],
    window_seconds: int = 120,
) -> List[List[CorrelationCandidate]]:
    """Call the **real upstream** correlator and map groups back to candidates."""
    by_window: Dict[str, CorrelationCandidate] = {
        candidate.window_id: candidate for candidate in candidates
    }
    anomalies = [candidate.to_upstream_anomaly() for candidate in candidates]
    groups = correlate_anomalies(anomalies, window_seconds=window_seconds)
    mapped: List[List[CorrelationCandidate]] = []
    for group in groups:
        mapped_group: List[CorrelationCandidate] = []
        for anomaly in group:
            window_id = anomaly.get("window_id")
            if window_id in by_window:
                mapped_group.append(by_window[window_id])
        if mapped_group:
            mapped.append(mapped_group)
    return mapped


def build_incident(
    group: Sequence[CorrelationCandidate],
    config: CorrelationConfig,
    now: Optional[datetime] = None,
    extra_hypotheses: Optional[Sequence[Mapping[str, Any]]] = None,
    extra_checks: Optional[Sequence[Mapping[str, Any]]] = None,
    llm_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build one canonical incident draft from an upstream group."""
    if not group:
        raise ValueError("an incident requires at least one candidate")
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("incident timestamps must be timezone-aware")

    ordered = sorted(group, key=lambda c: (c.window_start, c.window_id))
    incident_id = deterministic_incident_id(ordered)
    services = sorted({c.service for c in ordered})
    entities = _affected_entities(ordered)
    supporting, supporting_truncated = _bounded_union(
        (c.supporting_event_ids for c in ordered), config.max_events_per_incident
    )
    contradicting, contradicting_truncated = _bounded_union(
        (c.contradicting_event_ids for c in ordered),
        config.max_events_per_incident,
    )
    redacted_refs, refs_truncated = _bounded_union(
        (c.redacted_refs for c in ordered), config.max_redacted_refs_per_incident
    )
    evidence_truncated = (
        supporting_truncated
        or contradicting_truncated
        or refs_truncated
        or any(c.evidence.truncated for c in ordered)
    )

    score, comparable = _representative_score(ordered)
    severity, severity_source = _group_severity(ordered)

    hypotheses: List[Dict[str, Any]] = [
        _correlation_hypothesis(
            ordered, config, supporting, contradicting, evidence_truncated
        )
    ]
    for hypothesis in extra_hypotheses or ():
        item = _bounded_hypothesis(hypothesis)
        if item is not None and item not in hypotheses:
            hypotheses.append(item)

    checks: List[Dict[str, Any]] = []
    for check in _deterministic_checks(ordered, config, supporting):
        if check not in checks:
            checks.append(check)
    for check in extra_checks or ():
        item = _bounded_check(check)
        if item is not None and item not in checks:
            checks.append(item)
    checks = checks[: config.max_diagnostic_checks_per_incident]

    incident: Dict[str, Any] = {
        "incident_id": incident_id,
        "schema_version": "1.0",
        "status": "open",
        "severity": severity,
        "affected_entities": entities,
        "anomaly_score": score,
        "detector": _detector_name(ordered),
        "model_versions": _model_versions(ordered, config),
        "windows": _windows(ordered),
        "evidence": {
            "supporting_event_ids": supporting,
            "contradicting_event_ids": contradicting,
            "redacted_refs": redacted_refs,
        },
        "hypotheses": hypotheses,
        "diagnostic_checks": checks,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "audit": [],
        "correlation": {
            "method": "upstream.correlator.correlate_anomalies",
            "upstream_revision": UPSTREAM_REVISION,
            "window_seconds": config.window_seconds,
            "candidate_count": len(ordered),
            "source_window_ids": [c.window_id for c in ordered],
            "services": services,
            "basis": (
                "same service and timestamp proximity; association only, "
                "not a confirmed root cause"
            ),
            "score_comparable": comparable,
            "severity_source": severity_source,
            "evidence_truncated": evidence_truncated,
        },
    }
    if llm_metadata:
        incident["llm_interpretations"] = [dict(item) for item in llm_metadata]
    return incident


# ---------------------------------------------------------------------------
# Deterministic pieces
# ---------------------------------------------------------------------------


def _correlation_hypothesis(
    group: Sequence[CorrelationCandidate],
    config: CorrelationConfig,
    supporting: Sequence[str],
    contradicting: Sequence[str],
    truncated: bool,
) -> Dict[str, Any]:
    services = sorted({c.service for c in group})
    hosts = sorted({c.host_id for c in group})
    if len(group) == 1:
        statement = (
            f"One scored window for {services[0]} on {hosts[0]} exceeded the "
            f"model threshold. No cross-event correlation was established "
            f"within the {config.window_seconds}s window; this is not a "
            f"confirmed root cause."
        )
        confidence = "unknown"
    else:
        statement = (
            f"{len(group)} scored windows for {', '.join(services)} on "
            f"{', '.join(hosts)} fall within the {config.window_seconds}s "
            "correlation window. This is temporal and identity association, "
            "not a confirmed root cause."
        )
        confidence = "low"
    missing = [
        "A trace or dependency link connecting the included events.",
        "Confirmation from the service owner or a deployment record.",
        "Evidence from outside the correlation window.",
    ]
    if not supporting:
        missing.append(
            "No supporting event references were supplied for the scored "
            "window(s); this hypothesis rests on the model assessment and its "
            "numeric features alone."
        )
    if truncated:
        missing.append(
            "The evidence reference list was truncated at the configured bound; "
            "inspect the incident store or redacted evidence pointers for the "
            "full set."
        )
    return {
        "statement": statement[:2048],
        "confidence": confidence,
        "supporting_event_ids": list(supporting),
        "contradicting_event_ids": list(contradicting),
        "missing_evidence": missing,
    }


def _deterministic_checks(
    group: Sequence[CorrelationCandidate],
    config: CorrelationConfig,
    supporting: Sequence[str],
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for candidate in group:
        codes = [str(hit.get("code")) for hit in candidate.rule_hits if hit.get("code")]
        if not codes:
            codes = ["model_score"]
        links = candidate.supporting_event_ids[:10]
        for code in codes:
            instruction = _CHECK_BY_CODE.get(
                code,
                "Inspect the rule/feature evidence for this window and compare "
                "it with the preceding baseline.",
            )
            checks.append({
                "instruction": instruction,
                "evidence_links": links,
                "executes": False,
            })
    return checks


def _representative_score(
    group: Sequence[CorrelationCandidate],
) -> Tuple[float, bool]:
    directions = {c.score_direction for c in group}
    if len(directions) == 1:
        direction = next(iter(directions))
        if direction == "lower_is_more_anomalous":
            return min(c.raw_score for c in group), True
        if direction == "higher_is_more_anomalous":
            return max(c.raw_score for c in group), True
    # Mixed or unknown direction: keep the first score but say it is not
    # comparable rather than pretending the numbers share a scale.
    return group[0].raw_score, False


def _group_severity(
    group: Sequence[CorrelationCandidate],
) -> Tuple[str, str]:
    known = [c.severity for c in group if c.severity in _SEVERITY_ORDER]
    if not known or all(_SEVERITY_ORDER[s] == 0 for s in known):
        return "low", "ML severity was unknown/none; mapped to the schema minimum"
    highest = max(known, key=lambda s: _SEVERITY_ORDER[s])
    if highest in _SCHEMA_SEVERITIES:
        return highest, "maximum of the scored windows' ML severity"
    return "low", f"unmapped ML severity {highest!r}; mapped to the schema minimum"


def _detector_name(group: Sequence[CorrelationCandidate]) -> str:
    names = {c.model_name for c in group if c.model_name}
    if any("IsolationForest" in name for name in names):
        return "isolation_forest"
    if len(names) == 1:
        return next(iter(names))
    return "unknown"


def _model_versions(
    group: Sequence[CorrelationCandidate],
    config: CorrelationConfig,
) -> Dict[str, str]:
    return {
        "contract_version": _common(c.contract_version for c in group),
        "feature_schema_version": _common(c.feature_schema_version for c in group),
        "model_name": _common(c.model_name for c in group),
        "model_version": _common(c.model_version for c in group),
        "model_artifact_version": _common(
            c.model_artifact_version for c in group
        ),
        "threshold_config_version": _common(
            c.threshold_config_version for c in group
        ),
        "score_direction": _common(c.score_direction for c in group),
        "correlation_window_seconds": str(config.window_seconds),
        "correlation_method": "upstream.correlator.correlate_anomalies",
        "upstream_revision": UPSTREAM_REVISION,
    }


def _common(values: Iterable[str]) -> str:
    items = {str(value) if value not in (None, "") else "unknown" for value in values}
    if len(items) == 1:
        return next(iter(items))
    return "mixed"


def _affected_entities(
    group: Sequence[CorrelationCandidate],
) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    seen = set()
    for candidate in group:
        key = (candidate.host_id, candidate.service)
        if key in seen:
            continue
        seen.add(key)
        entities.append({
            "host": candidate.host_id,
            "service": candidate.service,
            "source": None,
        })
    return entities


def _windows(group: Sequence[CorrelationCandidate]) -> List[Dict[str, str]]:
    windows: List[Dict[str, str]] = []
    seen = set()
    for candidate in sorted(group, key=lambda c: c.window_start):
        key = (candidate.window_start.isoformat(), candidate.window_end.isoformat())
        if key in seen:
            continue
        seen.add(key)
        windows.append({"start": key[0], "end": key[1]})
    return windows


def _bounded_union(
    sequences: Iterable[Sequence[str]],
    limit: int,
) -> Tuple[List[str], bool]:
    result: List[str] = []
    seen = set()
    truncated = False
    for sequence in sequences:
        for value in sequence:
            if value in seen:
                continue
            if len(result) >= limit:
                truncated = True
                continue
            seen.add(value)
            result.append(value)
    return result, truncated


def deterministic_incident_id(group: Sequence[CorrelationCandidate]) -> str:
    """Stable ID for the same scored-window set (replay-safe, not random)."""
    identity = "lad-correlation|" + "|".join(
        sorted(candidate.window_id for candidate in group)
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def _bounded_hypothesis(
    value: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    statement = str(value.get("statement", "")).strip()
    confidence = str(value.get("confidence", "unknown"))
    if not statement or len(statement) > 2048:
        return None
    if confidence not in {"low", "medium", "high", "unknown"}:
        confidence = "unknown"
    return {
        "statement": statement,
        "confidence": confidence,
        "supporting_event_ids": _string_list(value.get("supporting_event_ids")),
        "contradicting_event_ids": _string_list(
            value.get("contradicting_event_ids")
        ),
        "missing_evidence": [
            str(item)[:1024]
            for item in _string_list(value.get("missing_evidence"))
        ],
    }


def _bounded_check(value: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    instruction = str(value.get("instruction", "")).strip()
    if not instruction or len(instruction) > 2048:
        return None
    return {
        "instruction": instruction,
        "evidence_links": _string_list(value.get("evidence_links")),
        "executes": False,
    }


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if isinstance(item, (str, int, float))]


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
