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
from .contract import (
    CONTRACT_VERSION,
    FeatureWindowRequest,
    InferenceResult,
    InferenceStatus,
    ReasonCode,
    RoutingDecision,
    route_result,
)


class SinkDeliveryError(RuntimeError):
    """Raised when the downstream sink fails while accepting a scored window.

    The window *was* scored, but it was **not** delivered. This is never
    reported as a successful delivery, and it never turns into a normal
    result.
    """


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
        """Build a forwarded window from a scored result.

        Raises ``ValueError`` for anything that is not a genuine scored
        result. An explicit check is used rather than ``assert`` so the
        guarantee also holds when Python runs with ``-O``.
        """
        if result.status is not InferenceStatus.SCORED:
            raise ValueError(
                f"refusing to build a ScoredWindow from a {result.status.value} result "
                f"(reason: {result.reason_code.value})"
            )
        if result.raw_score is None or result.is_anomaly is None:
            raise ValueError(
                "refusing to build a ScoredWindow from a scored result without a "
                "raw score and a decision"
            )
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
class SinkFailure:
    """A scored window that the downstream sink refused or failed to accept.

    The window was scored successfully, but it was **not** delivered. It is
    recorded here rather than in the forwarded list, so a delivery failure can
    never be mistaken for a successful delivery.
    """

    window_id: Optional[str]
    error_type: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass
class UnscoredWindow:
    """A window that was not forwarded, with its status and safe provenance.

    This is the record kept for every non-``scored`` outcome. The status and
    the reason code are preserved so the caller can distinguish
    *insufficient data* from *invalid input* from *model unavailable* from
    *inference error* — and from a malformed result that has no status at all.
    """

    window_id: Optional[str]
    status: str
    reason_code: str
    reason: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass
class AdapterOutcome:
    """What happened to a batch of windows at the boundary.

    ``unscored`` is the single source of truth for every non-``scored``
    outcome. ``rejected`` is retained as a **backward-compatible alias**
    (Milestone 6 API) over the same records.
    """

    scored: List[ScoredWindow] = field(default_factory=list)
    unscored: List[UnscoredWindow] = field(default_factory=list)
    sink_failures: List[SinkFailure] = field(default_factory=list)

    @property
    def rejected(self) -> List["RejectedWindow"]:
        """Backward-compatible alias for :attr:`unscored` (Milestone 6 API)."""
        return [
            RejectedWindow(
                window_id=item.window_id,
                status=item.status,
                reason_code=item.reason_code,
                reason=item.reason,
            )
            for item in self.unscored
        ]

    @property
    def forwarded_count(self) -> int:
        return len(self.scored)

    @property
    def rejected_count(self) -> int:
        """Backward-compatible alias for :meth:`unscored_count`."""
        return len(self.unscored)

    @property
    def unscored_count(self) -> int:
        return len(self.unscored)

    @property
    def sink_failure_count(self) -> int:
        return len(self.sink_failures)

    def unscored_by_status(self) -> Dict[str, int]:
        """Count of unscored outcomes per status."""
        counts: Dict[str, int] = {}
        for item in self.unscored:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forwarded": [w.to_dict() for w in self.scored],
            "rejected": [r.to_dict() for r in self.rejected],
            "unscored": [u.to_dict() for u in self.unscored],
            "sink_failures": [f.to_dict() for f in self.sink_failures],
            "forwarded_count": self.forwarded_count,
            "rejected_count": self.rejected_count,
            "unscored_count": self.unscored_count,
            "sink_failure_count": self.sink_failure_count,
        }


class EvidenceCorrelationAdapter:
    """Reference adapter showing the expected call pattern.

    The downstream component implements :class:`DownstreamSink`.  This
    adapter calls the ML interface, applies the **canonical** forwarding rule
    (:func:`~anomaly_detector.inference.contract.route_result`) and forwards
    only scored windows.

    Milestone 8 hardening:

    * the forwarding decision is made in exactly one place
      (:func:`route_result`) so no caller can implement a subtly different
      check;
    * an unscored result is handed to an explicit ``unscored_handler`` and
      recorded with its status and safe provenance;
    * a malformed or missing status **fails closed** — it is never forwarded;
    * a sink failure is recorded explicitly as a :class:`SinkFailure`; it is
      never reported as a successful delivery, and it never aborts the rest
      of the batch.
    """

    def __init__(
        self,
        detector: Any,
        sink: Optional[DownstreamSink] = None,
        unscored_handler: Optional[Callable[[UnscoredWindow], None]] = None,
    ) -> None:
        self._detector = detector
        self._sink = sink
        self._unscored_handler = unscored_handler

    # -- single window ---------------------------------------------------
    def handle_window(self, request: FeatureWindowRequest) -> Optional[ScoredWindow]:
        """Score one window and forward it when it was actually scored.

        Returns the forwarded :class:`ScoredWindow`, or ``None`` when the
        window was not forwarded (not scored, or the sink failed).  An
        unscored window is recorded through the ``unscored_handler``.

        Raises
        ------
        SinkDeliveryError
            When the window was scored but the sink failed to accept it. The
            window is *not* reported as delivered.
        """
        result = self._score(request)
        decision = route_result(result)

        if not decision.forwardable:
            self._record_unscored(result, decision)
            return None

        scored = ScoredWindow.from_result(result)
        if self._sink is not None:
            try:
                self._sink.handle_scored_window(result)
            except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
                raise SinkDeliveryError(
                    f"downstream sink failed to accept window {scored.window_id!r}: {exc}"
                ) from exc
        return scored

    # -- batch -----------------------------------------------------------
    def handle_windows(self, requests: Sequence[FeatureWindowRequest]) -> AdapterOutcome:
        """Score a batch and route each outcome.

        One bad window never fails the batch: every window gets its own
        outcome. A sink failure is recorded in ``outcome.sink_failures`` and
        the remaining windows are still processed.
        """
        outcome = AdapterOutcome()
        seen: Dict[str, int] = {}
        for index, request in enumerate(requests):
            # Duplicate window ids inside one batch are rejected for the later
            # occurrence, matching the existing M6 contract behaviour
            # (AnomalyDetector.score_windows). No stronger, exactly-once
            # guarantee is claimed: this is a per-batch, in-process check.
            window_id = getattr(request, "window_id", None)
            if isinstance(window_id, str) and window_id:
                if window_id in seen:
                    record = UnscoredWindow(
                        window_id=window_id,
                        status=InferenceStatus.INVALID_INPUT.value,
                        reason_code=ReasonCode.DUPLICATE_WINDOW_ID.value,
                        reason=f"duplicate window_id {window_id!r} in batch "
                               f"(first seen at index {seen[window_id]})",
                        detail="duplicate window id",
                    )
                    outcome.unscored.append(record)
                    self._notify_unscored(record)
                    continue
                seen[window_id] = index

            result = self._score(request)
            decision = route_result(result)

            if not decision.forwardable:
                self._record_unscored(result, decision, outcome)
                continue

            try:
                scored = ScoredWindow.from_result(result)
            except Exception as exc:  # noqa: BLE001 - fail closed
                outcome.unscored.append(
                    UnscoredWindow(
                        window_id=decision.window_id,
                        status=decision.status,
                        reason_code=decision.reason_code,
                        reason=decision.reason,
                        detail=f"scored result could not be converted: {exc}",
                    )
                )
                self._notify_unscored(outcome.unscored[-1])
                continue

            if self._sink is not None:
                try:
                    self._sink.handle_scored_window(result)
                except Exception as exc:  # noqa: BLE001 - explicit, not a success
                    outcome.sink_failures.append(
                        SinkFailure(
                            window_id=scored.window_id,
                            error_type=type(exc).__name__,
                            message=str(exc),
                        )
                    )
                    # The window was NOT delivered: it must not appear in the
                    # forwarded list.
                    continue

            outcome.scored.append(scored)
        return outcome

    # -- internals -------------------------------------------------------
    def _score(self, request: FeatureWindowRequest) -> Any:
        """Score one window, converting an unexpected failure into a result.

        An exception escaping the detector is turned into an explicit
        ``inference_error`` result. It is never silently dropped, and it is
        never converted into a normal (not-anomalous) outcome.
        """
        try:
            return self._detector.score_window(request)
        except Exception as exc:  # noqa: BLE001 - reported as inference_error
            return InferenceResult(
                contract_version=CONTRACT_VERSION,
                status=InferenceStatus.INFERENCE_ERROR,
                reason_code=ReasonCode.INFERENCE_FAILED,
                reason=f"scoring raised {type(exc).__name__}: {exc}",
                window_id=getattr(request, "window_id", None),
            )

    def _record_unscored(
        self,
        result: Any,
        decision: RoutingDecision,
        outcome: Optional[AdapterOutcome] = None,
    ) -> UnscoredWindow:
        record = UnscoredWindow(
            window_id=decision.window_id,
            status=decision.status,
            reason_code=decision.reason_code,
            reason=decision.reason,
            detail=decision.detail,
        )
        if outcome is not None:
            outcome.unscored.append(record)
        self._notify_unscored(record)
        return record

    def _notify_unscored(self, record: UnscoredWindow) -> None:
        if self._unscored_handler is None:
            return
        try:
            self._unscored_handler(record)
        except Exception:  # noqa: BLE001 - a handler failure must not abort routing
            # The record is already in the outcome; a broken handler must not
            # stop the batch or turn anything into a normal result.
            pass


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
