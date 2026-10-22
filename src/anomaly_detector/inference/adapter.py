"""Integration boundary between the ML component and the downstream consumer.

The downstream teammate owns evidence correlation, incident grouping,
explanation and root-cause hypotheses.  This module defines the *boundary*
only: a small, neutral protocol plus a reference adapter that shows how to
call the ML interface correctly.

What this module deliberately does **not** do
---------------------------------------------
* It does not retrieve or correlate events.
* It does not build incident narratives or explanations.
* It does not deliver alerts or trigger remediation.
* It does not invent or modify another workstream's architecture.

The reference adapter simply demonstrates the contract obligations: a
scored result is forwarded with its traceable metadata, and an unscored or
errored result is **not** forwarded as if it were a normal prediction.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from .contract import FeatureWindowRequest, InferenceResult, InferenceStatus


class DownstreamSink(Protocol):
    """The neutral boundary the downstream component must implement.

    The downstream component receives a *scored* result plus the traceable
    window metadata it needs to find evidence.  It never receives an
    unscored window presented as normal.
    """

    def handle_scored_window(self, result: InferenceResult) -> None:
        """Consume one scored window result."""
        ...


@dataclass
class ScoredWindow:
    """A scored result handed across the boundary.

    Only successfully scored windows produce this object, which makes it
    impossible for the caller to forward an unscored window as normal by
    accident.
    """

    window_id: str
    raw_score: float
    score_direction: str
    threshold: float
    is_anomaly: bool
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    host_id: Optional[str] = None
    service_name: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    feature_schema_version: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: InferenceResult) -> "ScoredWindow":
        if result.status is not InferenceStatus.SCORED:
            raise ValueError(
                f"refusing to build a ScoredWindow from a {result.status.value} result "
                f"(reason: {result.reason_code.value})"
            )
        assert result.raw_score is not None and result.is_anomaly is not None
        return cls(
            window_id=result.window_id or "",
            raw_score=float(result.raw_score),
            score_direction=result.score_direction or "",
            threshold=float(result.threshold or 0.0),
            is_anomaly=bool(result.is_anomaly),
            window_start=result.window_start,
            window_end=result.window_end,
            host_id=result.host_id,
            service_name=result.service_name,
            model_name=result.model_name,
            model_version=result.model_version,
            feature_schema_version=result.feature_schema_version,
            extra=dict(result.extra),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "raw_score": self.raw_score,
            "score_direction": self.score_direction,
            "threshold": self.threshold,
            "is_anomaly": self.is_anomaly,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "host_id": self.host_id,
            "service_name": self.service_name,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "extra": dict(self.extra),
        }


@dataclass
class RejectedWindow:
    """A window that could not be scored, kept for reporting only."""

    window_id: Optional[str]
    status: str
    reason_code: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "reason": self.reason,
        }


@dataclass
class AdapterOutcome:
    """What happened to a batch of windows at the boundary."""

    scored: List[ScoredWindow] = field(default_factory=list)
    rejected: List[RejectedWindow] = field(default_factory=list)

    @property
    def forwarded_count(self) -> int:
        return len(self.scored)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forwarded": [w.to_dict() for w in self.scored],
            "rejected": [r.to_dict() for r in self.rejected],
            "forwarded_count": self.forwarded_count,
            "rejected_count": self.rejected_count,
        }


class EvidenceCorrelationAdapter:
    """Reference adapter showing the expected call pattern.

    The downstream component implements :class:`DownstreamSink`.  This
    adapter calls the ML interface, branches on the explicit status and
    forwards only scored windows.
    """

    def __init__(self, detector: Any, sink: Optional[DownstreamSink] = None) -> None:
        self._detector = detector
        self._sink = sink

    def handle_window(self, request: FeatureWindowRequest) -> Optional[ScoredWindow]:
        """Score one window and forward it when it was actually scored."""
        result = self._detector.score_window(request)
        if result.status is not InferenceStatus.SCORED:
            return None
        scored = ScoredWindow.from_result(result)
        if self._sink is not None:
            self._sink.handle_scored_window(result)
        return scored

    def handle_windows(self, requests: Sequence[FeatureWindowRequest]) -> AdapterOutcome:
        """Score a batch and split the outcomes into forwarded / rejected."""
        outcome = AdapterOutcome()
        for request in requests:
            result = self._detector.score_window(request)
            if result.status is not InferenceStatus.SCORED:
                outcome.rejected.append(
                    RejectedWindow(
                        window_id=result.window_id,
                        status=result.status.value,
                        reason_code=result.reason_code.value,
                        reason=result.reason,
                    )
                )
                continue
            outcome.scored.append(ScoredWindow.from_result(result))
            if self._sink is not None:
                self._sink.handle_scored_window(result)
        return outcome


class RecordingSink:
    """Simple in-memory sink used by the example script and the tests."""

    def __init__(self) -> None:
        self.results: List[InferenceResult] = []

    def handle_scored_window(self, result: InferenceResult) -> None:
        self.results.append(result)


def make_example_request(
    features: Dict[str, Any],
    window_id: str = "web-01|web-api|2026-10-18T10:01:00+00:00",
    window_start: str = "2026-10-18T10:01:00+00:00",
    window_end: str = "2026-10-18T10:02:00+00:00",
    host_id: str = "web-01",
    service_name: str = "web-api",
) -> FeatureWindowRequest:
    """Build a request from ISO strings (used by the CLI smoke test)."""
    from datetime import datetime

    return FeatureWindowRequest(
        window_id=window_id,
        window_start=datetime.fromisoformat(window_start),
        window_end=datetime.fromisoformat(window_end),
        features=dict(features),
        host_id=host_id,
        service_name=service_name,
    )


__all__ = [
    "AdapterOutcome",
    "DownstreamSink",
    "EvidenceCorrelationAdapter",
    "RecordingSink",
    "RejectedWindow",
    "ScoredWindow",
    "make_example_request",
]
