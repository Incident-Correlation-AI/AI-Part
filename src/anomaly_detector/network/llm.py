"""Evidence-grounded LLM explanation adapter.

This is a **narrow adapter**, not a new reasoning system.  It takes one
*scored* :class:`~anomaly_detector.inference.contract.InferenceResult`, builds a
small evidence bundle that contains only allowlisted measurements, and asks a
provider for a short explanation.  Every draft is validated before it is
returned:

* an unscored window is **never** explained as a detection;
* the explanation cannot claim an anomaly when the model decision is
  ``is_anomaly = False`` (and vice versa);
* every citation must name a real evidence key or rule code;
* every numeric claim must match a number present in the bundle;
* labels, scenario names, run ids and PII/secret-like tokens are rejected.

If a provider fails or produces an ungrounded draft, a deterministic fallback
explanation is generated from the same evidence.  The fallback is marked
``fallback_used = True`` and carries the rejection reason.

No provider is contacted by default.  ``LlmSettings.enabled`` is ``False`` in
``configs/network.yaml``; the test suite uses a mock provider or an injected
transport and never makes a network request.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import math
import os
import re
import urllib.error
import urllib.request

from ..inference.contract import InferenceResult, InferenceStatus

#: Tokens that must never appear in a generated explanation.  Scenario names
#: are evaluation-only; run ids and label vocabulary are evaluation metadata.
FORBIDDEN_TOKENS: Tuple[str, ...] = (
    "http_5xx_burst",
    "db_timeout_latency",
    "auth_failure_burst",
    "malformed_telemetry",
    "benign_high_traffic",
    "benign_latency_shift",
    "benign_restart",
    "normal_recovery",
    "ground truth",
    "scenario_name",
    "run-",
    "is_anomaly",
    "acct-",
    "src-",
    "password",
    "api key",
    "secret",
)

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?%?")

#: Compact evidence keys exposed to an explanation provider.
EVIDENCE_KEYS: Tuple[str, ...] = (
    "event_count",
    "error_count",
    "error_ratio",
    "http_request_count",
    "http_5xx_count",
    "http_5xx_ratio",
    "http_status_eligible_count",
    "auth_failure_count",
    "auth_success_count",
    "auth_failure_ratio",
    "auth_distinct_source_count",
    "auth_distinct_account_count",
    "db_event_count",
    "db_timeout_count",
    "db_timeout_ratio",
    "db_connection_failure_count",
    "db_connection_failure_ratio",
    "latency_sample_count",
    "mean_latency_ms",
    "p95_latency_ms",
    "p95_latency_ratio",
    "restart_count",
    "invalid_event_count",
    "duplicate_event_count",
    "out_of_order_event_count",
    "telemetry_gap_seconds",
)


class ExplanationProviderError(RuntimeError):
    """Raised when a provider cannot return a usable draft."""


@dataclass(frozen=True)
class EvidenceBundle:
    """The complete, allowlisted evidence an explanation may cite."""

    window_id: str
    window_start: Optional[str]
    window_end: Optional[str]
    host_id: Optional[str]
    service_name: Optional[str]
    severity: str
    model_is_anomaly: bool
    model_raw_score: float
    threshold: float
    score_direction: str
    baseline_is_anomaly: bool
    baseline_risk_score: float
    rule_hits: List[Dict[str, Any]] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)
    telemetry_quality: Dict[str, int] = field(default_factory=dict)
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    feature_schema_version: Optional[str] = None
    contract_version: Optional[str] = None

    @classmethod
    def from_result(cls, result: InferenceResult) -> "EvidenceBundle":
        return _bundle_from_result(result)

    def to_prompt_dict(self) -> Dict[str, Any]:
        """Provider-facing payload.  Contains no labels or identifiers beyond
        the operational host/service attached to the window."""
        return {
            "window": {
                "window_id": self.window_id,
                "window_start": self.window_start,
                "window_end": self.window_end,
                "host": self.host_id,
                "service": self.service_name,
            },
            "model": {
                "name": self.model_name,
                "version": self.model_version,
                "feature_schema_version": self.feature_schema_version,
                "contract_version": self.contract_version,
            },
            "decision": {
                "is_anomaly": self.model_is_anomaly,
                "raw_score": self.model_raw_score,
                "threshold": self.threshold,
                "score_direction": self.score_direction,
                "severity": self.severity,
            },
            "rule_evidence": {
                "is_anomaly": self.baseline_is_anomaly,
                "risk_score": self.baseline_risk_score,
                "hits": [dict(hit) for hit in self.rule_hits],
            },
            "measurements": dict(self.features),
            "telemetry_quality": dict(self.telemetry_quality),
            "constraints": [
                "Explain only the evidence in this payload.",
                "Do not invent numbers, hosts, services, users, IPs or labels.",
                "Do not claim an anomaly when is_anomaly is false.",
                "Cite evidence keys or rule codes for every claim.",
                "Return JSON with summary, evidence, recommended_checks, "
                "asserts_anomaly and citations.",
            ],
        }

    def allowed_citations(self) -> set:
        citations = set(self.features) | {"model_score", "threshold",
                                           "baseline_risk_score",
                                           "telemetry_quality", "severity"}
        citations |= {str(hit.get("code")) for hit in self.rule_hits if hit.get("code")}
        return citations

    def allowed_numbers(self) -> set:
        values: List[float] = [self.model_raw_score, self.threshold,
                               self.baseline_risk_score]
        values.extend(float(v) for v in self.features.values()
                      if isinstance(v, (int, float)) and not isinstance(v, bool))
        for hit in self.rule_hits:
            for section in ("observed", "threshold", "weight"):
                payload = hit.get(section)
                if isinstance(payload, Mapping):
                    values.extend(_numeric_values(payload))
                elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
                    values.append(float(payload))
        expanded = set(values)
        for value in values:
            if 0.0 <= value <= 1.0:
                expanded.add(value * 100.0)
        return expanded


@dataclass
class DraftExplanation:
    summary: str
    evidence: List[str] = field(default_factory=list)
    recommended_checks: List[str] = field(default_factory=list)
    asserts_anomaly: Optional[bool] = None
    citations: List[str] = field(default_factory=list)

    @classmethod
    def from_any(cls, value: Any) -> "DraftExplanation":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            parsed = _try_json(value)
            if isinstance(parsed, Mapping):
                return cls.from_any(parsed)
            return cls(summary=value)
        if isinstance(value, Mapping):
            summary = value.get("summary", value.get("text", value.get("explanation", "")))
            evidence = value.get("evidence", []) or []
            checks = value.get("recommended_checks", value.get("checks", [])) or []
            citations = value.get("citations", []) or []
            assertion = value.get("asserts_anomaly", value.get("is_anomaly"))
            return cls(
                summary=str(summary or ""),
                evidence=[str(v) for v in evidence],
                recommended_checks=[str(v) for v in checks],
                asserts_anomaly=assertion if isinstance(assertion, bool) else None,
                citations=[str(v) for v in citations],
            )
        raise ExplanationProviderError(
            f"provider returned an unsupported draft type {type(value).__name__}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "evidence": list(self.evidence),
            "recommended_checks": list(self.recommended_checks),
            "asserts_anomaly": self.asserts_anomaly,
            "citations": list(self.citations),
        }


@dataclass
class Explanation:
    """Validated explanation plus the provenance needed to audit it."""

    window_id: Optional[str]
    text: str
    evidence: List[str]
    citations: List[str]
    recommended_checks: List[str]
    asserts_anomaly: Optional[bool]
    grounded: bool
    provider: str
    fallback_used: bool = False
    rejected_reason: str = ""
    explained: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "explained": self.explained,
            "text": self.text,
            "evidence": list(self.evidence),
            "citations": list(self.citations),
            "recommended_checks": list(self.recommended_checks),
            "asserts_anomaly": self.asserts_anomaly,
            "grounded": self.grounded,
            "provider": self.provider,
            "fallback_used": self.fallback_used,
            "rejected_reason": self.rejected_reason,
        }


class MockExplanationProvider:
    """Deterministic provider used by tests and by the offline default."""

    name = "mock"

    def explain(self, bundle: EvidenceBundle) -> DraftExplanation:
        return build_fallback_draft(bundle)


class HttpExplanationProvider:
    """Minimal JSON-over-HTTP provider.

    The endpoint is expected to accept a JSON body and return either
    ``{"summary": ...}`` or an OpenAI-compatible
    ``{"choices": [{"message": {"content": "<json>"}}]}`` response.  The API key
    is read from ``api_key_env`` at call time and is never logged or serialised.
    A transport callable may be injected for offline tests.
    """

    name = "http"

    def __init__(
        self,
        endpoint: str,
        model: str = "",
        api_key_env: str = "LAD_LLM_API_KEY",
        timeout_seconds: float = 20.0,
        transport: Optional[Callable[[str, Dict[str, Any], Dict[str, str], float], Tuple[int, str]]] = None,
    ) -> None:
        if not endpoint:
            raise ValueError("an endpoint is required for the http provider")
        self.endpoint = endpoint
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_seconds = float(timeout_seconds)
        self._transport = transport or _urllib_transport

    def explain(self, bundle: EvidenceBundle) -> DraftExplanation:
        payload = {
            "model": self.model or "unspecified",
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a defensive log-analysis assistant. Explain only "
                        "the supplied evidence. Return JSON with the fields summary, "
                        "evidence, recommended_checks, asserts_anomaly, citations. "
                        "Never invent measurements, labels, users, hosts or IPs."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(bundle.to_prompt_dict(), sort_keys=True),
                },
            ],
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env) if self.api_key_env else None
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            status, body = self._transport(
                self.endpoint, payload, headers, self.timeout_seconds
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a provider failure
            raise ExplanationProviderError(f"provider request failed: {exc}") from exc
        if status < 200 or status >= 300:
            raise ExplanationProviderError(f"provider returned HTTP {status}")
        try:
            response = json.loads(body)
        except ValueError as exc:
            raise ExplanationProviderError("provider response was not JSON") from exc
        content = _extract_content(response)
        return DraftExplanation.from_any(content)


def explain_result(
    result: InferenceResult,
    provider: Optional[Any] = None,
    fallback: bool = True,
) -> Explanation:
    """Build, validate and (if necessary) repair an explanation for one result.

    An unscored result is returned with ``explained = False`` and an explicit
    reason; it is never turned into an anomaly explanation.
    """
    provider = provider or MockExplanationProvider()
    provider_name = getattr(provider, "name", provider.__class__.__name__)

    if result is None:
        return Explanation(
            window_id=None, text="", evidence=[], citations=[],
            recommended_checks=[], asserts_anomaly=None, grounded=False,
            provider=provider_name, explained=False,
            rejected_reason="no inference result was supplied",
        )
    if result.status is not InferenceStatus.SCORED or result.is_anomaly is None:
        return Explanation(
            window_id=result.window_id, text="", evidence=[], citations=[],
            recommended_checks=[], asserts_anomaly=None, grounded=False,
            provider=provider_name, explained=False,
            rejected_reason=(
                f"status {getattr(result.status, 'value', result.status)!r} is not "
                "scored; an unscored window cannot be explained as a detection"
            ),
        )

    bundle = EvidenceBundle.from_result(result)
    rejection = ""
    try:
        draft = DraftExplanation.from_any(provider.explain(bundle))
    except Exception as exc:  # noqa: BLE001 - provider failures are recoverable
        draft = None
        rejection = f"provider error: {exc}"

    if draft is not None:
        violations = validate_draft(draft, bundle)
        if not violations:
            return Explanation(
                window_id=bundle.window_id,
                text=draft.summary,
                evidence=list(draft.evidence),
                citations=list(draft.citations),
                recommended_checks=list(draft.recommended_checks),
                asserts_anomaly=draft.asserts_anomaly,
                grounded=True,
                provider=provider_name,
            )
        rejection = "; ".join(violations)

    if fallback:
        fallback_draft = build_fallback_draft(bundle)
        violations = validate_draft(fallback_draft, bundle)
        if violations:  # pragma: no cover - the fallback is tested to be valid
            return Explanation(
                window_id=bundle.window_id,
                text="",
                evidence=[],
                citations=[],
                recommended_checks=[],
                asserts_anomaly=None,
                grounded=False,
                provider=provider_name,
                fallback_used=True,
                rejected_reason="fallback failed validation: " + "; ".join(violations),
                explained=False,
            )
        return Explanation(
            window_id=bundle.window_id,
            text=fallback_draft.summary,
            evidence=list(fallback_draft.evidence),
            citations=list(fallback_draft.citations),
            recommended_checks=list(fallback_draft.recommended_checks),
            asserts_anomaly=fallback_draft.asserts_anomaly,
            grounded=True,
            provider=provider_name,
            fallback_used=True,
            rejected_reason=rejection,
        )

    return Explanation(
        window_id=bundle.window_id,
        text="",
        evidence=[],
        citations=[],
        recommended_checks=[],
        asserts_anomaly=None,
        grounded=False,
        provider=provider_name,
        rejected_reason=rejection,
    )


def validate_draft(draft: DraftExplanation, bundle: EvidenceBundle) -> List[str]:
    """Return a list of grounding violations (empty means valid)."""
    violations: List[str] = []

    if not draft.summary or not draft.summary.strip():
        violations.append("the explanation is empty")
    if len(draft.summary) > 4000:
        violations.append("the explanation exceeds the 4000-character bound")
    if draft.asserts_anomaly is None:
        violations.append("the draft does not state an explicit anomaly assertion")
    elif bool(draft.asserts_anomaly) != bool(bundle.model_is_anomaly):
        violations.append(
            "the draft's anomaly assertion contradicts the model decision "
            f"({draft.asserts_anomaly!r} vs {bundle.model_is_anomaly!r})"
        )

    allowed_citations = bundle.allowed_citations()
    unknown = [c for c in draft.citations if c not in allowed_citations]
    if unknown:
        violations.append(f"unknown citation(s): {sorted(set(unknown))}")
    if not draft.citations:
        violations.append("the draft cites no evidence")

    text = "\n".join([draft.summary, *draft.evidence, *draft.recommended_checks])
    lowered = text.lower()
    for token in FORBIDDEN_TOKENS:
        if token.lower() in lowered:
            violations.append(f"forbidden evaluation/PII token {token!r} in the explanation")
    if _EMAIL_RE.search(text):
        violations.append("the explanation contains an email-like identifier")
    if _IPV4_RE.search(text):
        violations.append("the explanation contains an IP-address-like identifier")

    allowed_numbers = bundle.allowed_numbers()
    fabricated = sorted(
        token for token in _NUMBER_RE.findall(text)
        if not _number_is_supported(token, allowed_numbers)
    )
    if fabricated:
        violations.append(f"numeric claim(s) not present in the evidence: {fabricated}")
    return violations


def build_fallback_draft(bundle: EvidenceBundle) -> DraftExplanation:
    """Deterministic, evidence-only explanation."""
    decision = "flagged" if bundle.model_is_anomaly else "did not flag"
    summary_parts = [
        f"The Isolation Forest {decision} the {bundle.service_name or 'service'} "
        f"window on {bundle.host_id or 'host'} with raw score "
        f"{bundle.model_raw_score:.4f} against threshold {bundle.threshold:.4f}."
    ]
    citations = ["model_score", "threshold"]
    evidence: List[str] = []
    checks: List[str] = []

    for hit in bundle.rule_hits:
        code = str(hit.get("code", "rule"))
        message = str(hit.get("message", "rule threshold exceeded"))
        summary_parts.append(f"Rule evidence: {message}.")
        evidence.append(f"{code}: {message}")
        citations.append(code)

    if bundle.model_is_anomaly and not bundle.rule_hits:
        summary_parts.append(
            "The flag is model-only: no transparent rule exceeded its threshold."
        )

    if not bundle.model_is_anomaly and bundle.rule_hits:
        summary_parts.append(
            "The transparent rules produced evidence, but the model decision "
            "remains not anomalous; treat this as a low-severity lead only."
        )

    for key in _feature_priority(bundle):
        value = bundle.features.get(key)
        if value is None:
            continue
        evidence.append(f"{key} = {value:.4f}")
        citations.append(key)
        if len(evidence) >= 4:
            break

    if bundle.telemetry_quality:
        summary_parts.append(
            "Telemetry quality flags were recorded for this window."
        )
        citations.append("telemetry_quality")

    if bundle.model_is_anomaly:
        checks.append(
            "Compare this window with the preceding rolling baseline for the same service."
        )
        checks.append(
            "Check the health of the upstream dependency named by the rule evidence."
        )
    else:
        checks.append(
            "No immediate action is required; keep the window for trend comparison."
        )

    return DraftExplanation(
        summary=" ".join(summary_parts),
        evidence=evidence,
        recommended_checks=checks,
        asserts_anomaly=bool(bundle.model_is_anomaly),
        citations=_dedupe(citations),
    )


def _feature_priority(bundle: EvidenceBundle) -> List[str]:
    priority = [
        "http_5xx_ratio",
        "http_5xx_count",
        "db_timeout_ratio",
        "db_connection_failure_ratio",
        "auth_failure_ratio",
        "auth_distinct_source_count",
        "p95_latency_ratio",
        "p95_latency_ms",
        "event_count_zscore",
        "event_count",
    ]
    return [key for key in priority if key in bundle.features]


def _extract_content(response: Any) -> Any:
    if isinstance(response, Mapping):
        if "choices" in response:
            try:
                content = response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ExplanationProviderError(
                    "OpenAI-compatible response is missing choices[0].message.content"
                ) from exc
            return content
        if "summary" in response or "explanation" in response:
            return response
        if "content" in response:
            return response["content"]
    raise ExplanationProviderError("provider response did not contain an explanation")


def _urllib_transport(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float,
) -> Tuple[int, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return int(response.status), response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return int(exc.code), body


def _try_json(value: str) -> Any:
    text = value.strip()
    if not text:
        return None
    # Tolerate a fenced ```json block.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except ValueError:
        return None


def _numeric_values(payload: Mapping[str, Any]) -> List[float]:
    values: List[float] = []
    for value in payload.values():
        if isinstance(value, Mapping):
            values.extend(_numeric_values(value))
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Mapping):
                    values.extend(_numeric_values(item))
                elif isinstance(item, (int, float)) and not isinstance(item, bool):
                    values.append(float(item))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _number_is_supported(token: str, allowed: set) -> bool:
    percent = token.endswith("%")
    raw = token[:-1] if percent else token
    try:
        value = float(raw)
    except ValueError:
        return False
    candidate = value if not percent else value
    # Small ordinals/counts and calendar years are not metric claims.
    if not percent and float(value).is_integer():
        if 0 <= value <= 10 or 1900 <= value <= 2100:
            return True
    for permitted in allowed:
        tolerance = max(1e-6, abs(permitted) * 0.01)
        if abs(candidate - permitted) <= tolerance:
            return True
    return False


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


# Attach the bundle constructor after the dataclass definition so the module
# stays a single, readable unit.
def _bundle_from_result(result: InferenceResult) -> EvidenceBundle:
    if result.status is not InferenceStatus.SCORED or result.is_anomaly is None:
        raise ValueError("an evidence bundle requires a scored result")
    network = dict(result.extra.get("network") or {})
    baseline = dict(network.get("baseline") or {})
    features = {
        str(key): float(value)
        for key, value in (network.get("evidence_features") or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return EvidenceBundle(
        window_id=str(result.window_id or ""),
        window_start=result.window_start,
        window_end=result.window_end,
        host_id=result.host_id,
        service_name=result.service_name,
        severity=str(network.get("severity") or "unknown"),
        model_is_anomaly=bool(result.is_anomaly),
        model_raw_score=float(result.raw_score if result.raw_score is not None else float("nan")),
        threshold=float(result.threshold if result.threshold is not None else float("nan")),
        score_direction=str(result.score_direction or ""),
        baseline_is_anomaly=bool(baseline.get("is_anomaly", False)),
        baseline_risk_score=float(baseline.get("risk_score", 0.0)),
        rule_hits=list(baseline.get("rule_hits") or []),
        features=features,
        telemetry_quality={
            str(key): int(value)
            for key, value in (network.get("telemetry_quality") or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        },
        model_name=result.model_name,
        model_version=result.model_version,
        feature_schema_version=result.feature_schema_version,
        contract_version=result.contract_version,
    )
