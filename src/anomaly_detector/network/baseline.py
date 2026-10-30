"""Transparent, configurable baseline for network/service incidents.

The baseline is deliberately *not* a black box.  It combines:

* explicit domain rules with minimum-volume safeguards (HTTP 5xx rate, DB
  timeout/connection-failure rate, authentication-failure burst, latency
  degradation, restart surge); and
* causal robust-deviation features (rolling median/MAD z-scores and deltas).

Score semantics match the repository's existing Isolation Forest contract:
``raw_score = -risk_score`` so **lower is more anomalous**, and
``is_anomaly = risk_score >= decision_threshold``.  A rule hit is evidence for
investigation, not proof of an attack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence
import json
import math

from .config import RuleConfig


@dataclass(frozen=True)
class RuleHit:
    code: str
    strength: str          # weak | strong
    weight: float
    observed: Dict[str, Any]
    threshold: Dict[str, Any]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "strength": self.strength,
            "weight": self.weight,
            "observed": dict(self.observed),
            "threshold": dict(self.threshold),
            "message": self.message,
        }


@dataclass
class BaselineDecision:
    risk_score: float
    raw_score: float
    is_anomaly: bool
    severity: str
    hits: List[RuleHit] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "raw_score": self.raw_score,
            "is_anomaly": self.is_anomaly,
            "severity": self.severity,
            "rule_hits": [hit.to_dict() for hit in self.hits],
        }


def score_window(features: Mapping[str, Any], rules: Optional[RuleConfig] = None) -> BaselineDecision:
    """Apply the transparent baseline to one feature mapping."""
    rules = rules or RuleConfig()
    hits: List[RuleHit] = []

    eligible_http = _get(features, "http_status_eligible_count")
    five_count = _get(features, "http_5xx_count")
    five_ratio = _get(features, "http_5xx_ratio")
    five_delta = _get(features, "http_5xx_ratio_delta")

    if (
        eligible_http is not None and five_count is not None and five_ratio is not None
        and eligible_http >= rules.http_5xx_min_requests
        and five_count >= rules.http_5xx_min_count
        and five_ratio >= rules.http_5xx_ratio_threshold
    ):
        strong = five_ratio >= rules.http_5xx_strong_ratio
        hits.append(RuleHit(
            code="http_5xx_rate",
            strength="strong" if strong else "weak",
            weight=rules.strong_weight if strong else rules.weak_weight,
            observed={
                "http_5xx_count": five_count,
                "http_status_eligible_count": eligible_http,
                "http_5xx_ratio": round(five_ratio, 6),
            },
            threshold={
                "min_requests": rules.http_5xx_min_requests,
                "min_5xx": rules.http_5xx_min_count,
                "ratio": rules.http_5xx_ratio_threshold,
                "strong_ratio": rules.http_5xx_strong_ratio,
            },
            message="HTTP 5xx rate exceeds the configured threshold with enough volume",
        ))
    elif (
        eligible_http is not None and eligible_http >= rules.http_5xx_min_requests
        and five_delta is not None and five_delta >= rules.http_5xx_delta_threshold
    ):
        hits.append(RuleHit(
            code="http_5xx_rate_delta",
            strength="weak",
            weight=rules.weak_weight,
            observed={"http_5xx_ratio_delta": round(five_delta, 6),
                      "http_status_eligible_count": eligible_http},
            threshold={"delta": rules.http_5xx_delta_threshold,
                       "min_requests": rules.http_5xx_min_requests},
            message="HTTP 5xx rate rose sharply relative to this service's recent baseline",
        ))

    db_events = _get(features, "db_event_count")
    timeout_ratio = _get(features, "db_timeout_ratio")
    conn_ratio = _get(features, "db_connection_failure_ratio")
    if db_events is not None and db_events >= rules.db_min_events:
        if timeout_ratio is not None and timeout_ratio >= rules.db_timeout_ratio_threshold:
            strong = timeout_ratio >= rules.db_timeout_strong_ratio
            hits.append(RuleHit(
                code="db_timeout_rate",
                strength="strong" if strong else "weak",
                weight=rules.strong_weight if strong else rules.weak_weight,
                observed={"db_event_count": db_events,
                          "db_timeout_ratio": round(timeout_ratio, 6)},
                threshold={"min_events": rules.db_min_events,
                           "ratio": rules.db_timeout_ratio_threshold,
                           "strong_ratio": rules.db_timeout_strong_ratio},
                message="Database timeout rate exceeds the configured threshold",
            ))
        if conn_ratio is not None and conn_ratio >= rules.db_connection_failure_ratio_threshold:
            strong = conn_ratio >= rules.db_connection_failure_strong_ratio
            hits.append(RuleHit(
                code="db_connection_failure_rate",
                strength="strong" if strong else "weak",
                weight=rules.strong_weight if strong else rules.weak_weight,
                observed={"db_event_count": db_events,
                          "db_connection_failure_ratio": round(conn_ratio, 6)},
                threshold={"min_events": rules.db_min_events,
                           "ratio": rules.db_connection_failure_ratio_threshold,
                           "strong_ratio": rules.db_connection_failure_strong_ratio},
                message="Database connection-failure rate exceeds the configured threshold",
            ))

    failures = _get(features, "auth_failure_count")
    successes = _get(features, "auth_success_count")
    auth_ratio = _get(features, "auth_failure_ratio")
    distinct_sources = _get(features, "auth_distinct_source_count")
    if failures is not None and successes is not None:
        total_auth = failures + successes
        if (
            total_auth >= rules.auth_min_events
            and failures >= rules.auth_min_failures
            and auth_ratio is not None
            and auth_ratio >= rules.auth_failure_ratio_threshold
        ):
            strong = auth_ratio >= rules.auth_failure_strong_ratio
            hits.append(RuleHit(
                code="auth_failure_rate",
                strength="strong" if strong else "weak",
                weight=rules.strong_weight if strong else rules.weak_weight,
                observed={"auth_failure_count": failures,
                          "auth_success_count": successes,
                          "auth_failure_ratio": round(auth_ratio, 6)},
                threshold={"min_events": rules.auth_min_events,
                           "min_failures": rules.auth_min_failures,
                           "ratio": rules.auth_failure_ratio_threshold,
                           "strong_ratio": rules.auth_failure_strong_ratio},
                message="Authentication-failure burst exceeds the configured threshold",
            ))
        if (
            distinct_sources is not None
            and distinct_sources >= rules.auth_distinct_source_threshold
            and failures >= rules.auth_min_failures
        ):
            hits.append(RuleHit(
                code="auth_distinct_sources",
                strength="weak",
                weight=rules.weak_weight,
                observed={"auth_distinct_source_count": distinct_sources,
                          "auth_failure_count": failures},
                threshold={"distinct_sources": rules.auth_distinct_source_threshold,
                           "min_failures": rules.auth_min_failures},
                message="Failures originate from an unusually diverse set of sources",
            ))

    p95 = _get(features, "p95_latency_ms")
    p95_ratio = _get(features, "p95_latency_ratio")
    samples = _get(features, "latency_sample_count")
    if (
        p95 is not None and samples is not None and p95_ratio is not None
        and samples >= rules.latency_min_samples
        and p95 >= rules.latency_p95_absolute_floor_ms
        and p95_ratio >= rules.latency_p95_ratio_threshold
    ):
        strong = p95_ratio >= rules.latency_p95_strong_ratio
        hits.append(RuleHit(
            code="latency_p95_ratio",
            strength="strong" if strong else "weak",
            weight=rules.strong_weight if strong else rules.weak_weight,
            observed={"p95_latency_ms": round(p95, 3),
                      "p95_latency_ratio": round(p95_ratio, 6),
                      "latency_sample_count": samples},
            threshold={"ratio": rules.latency_p95_ratio_threshold,
                       "strong_ratio": rules.latency_p95_strong_ratio,
                       "absolute_floor_ms": rules.latency_p95_absolute_floor_ms,
                       "min_samples": rules.latency_min_samples},
            message="p95 latency rose sharply relative to this service's recent baseline",
        ))

    restarts = _get(features, "restart_count")
    if restarts is not None and restarts > rules.max_routine_restarts:
        hits.append(RuleHit(
            code="restart_surge",
            strength="weak",
            weight=rules.weak_weight,
            observed={"restart_count": restarts},
            threshold={"max_routine_restarts": rules.max_routine_restarts},
            message="More restarts than the configured routine allowance",
        ))

    # MAD robust deviation check on rolling metrics
    event_z = abs(_get(features, "event_count_zscore") or 0.0)
    http_z = abs(_get(features, "http_request_count_zscore") or 0.0)
    max_mad_z = max(event_z, http_z)
    mad_threshold = getattr(rules, "mad_high_zscore", 4.0)
    if max_mad_z >= mad_threshold:
        hits.append(RuleHit(
            code="mad_robust_zscore",
            strength="strong",
            weight=rules.high_severity_threshold,
            observed={"max_mad_zscore": round(max_mad_z, 3)},
            threshold={"mad_high_zscore": mad_threshold},
            message="Robust MAD deviation indicates extreme divergence from rolling baseline",
        ))

    risk = round(sum(hit.weight for hit in hits), 6)
    is_anomaly = risk >= rules.decision_threshold
    if risk >= rules.high_severity_threshold:
        severity = "high"
    elif is_anomaly:
        severity = "medium"
    elif risk > 0:
        severity = "low"
    else:
        severity = "none"
    return BaselineDecision(
        risk_score=risk,
        raw_score=-risk,
        is_anomaly=bool(is_anomaly),
        severity=severity,
        hits=hits,
    )


def score_dataframe(features: "Any", rules: Optional[RuleConfig] = None) -> "Any":
    """Score a feature DataFrame and return evaluation columns."""
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for record in features.to_dict(orient="records"):
        decision = score_window(record, rules)
        rows.append({
            "baseline_risk_score": decision.risk_score,
            "baseline_raw_score": decision.raw_score,
            "baseline_is_anomaly": decision.is_anomaly,
            "baseline_severity": decision.severity,
            "baseline_rule_hits": json.dumps([h.to_dict() for h in decision.hits], sort_keys=True),
        })
    return pd.DataFrame(rows, index=features.index)


def _get(features: Mapping[str, Any], name: str) -> Optional[float]:
    value = features.get(name)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
