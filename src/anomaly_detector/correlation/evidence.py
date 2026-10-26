"""Evidence references and the narrow redaction boundary.

The correlation adapter never accepts raw log text.  It accepts only:

* stable event IDs from the canonical LAD envelope;
* optional pointers to already-redacted evidence (``evidence_ref``);
* the numeric evidence already present in a scored inference result.

This keeps the LLM/incident path free of messages, request bodies, credentials,
cookies, IP addresses and account identifiers.  References are validated, not
silently copied: an unsafe event ID rejects the candidate, while an unsafe
optional redacted pointer is dropped and recorded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import re


_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.")
_SECRET_WORDS = (
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "session_token",
    "authorization",
    "bearer",
    "cookie",
    "private_key",
)

#: Stable event IDs are opaque tokens, not free-form text.
_EVENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
#: Already-redacted evidence pointers may use a URI-like form.
_REDACTED_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,511}$")
#: Operational identity values (hostnames, service names, FQDNs).
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,254}$")


@dataclass(frozen=True)
class WindowEvidence:
    """Event references attached to one scored window.

    ``supporting_event_ids`` and ``contradicting_event_ids`` are kept separate
    because the incident contract requires both arrays and a record that cites
    only supporting evidence is not honest.
    """

    window_id: str
    supporting_event_ids: Tuple[str, ...] = ()
    contradicting_event_ids: Tuple[str, ...] = ()
    redacted_refs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.window_id, str) or not self.window_id.strip():
            raise ValueError("WindowEvidence requires a non-empty window_id")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "supporting_event_ids": list(self.supporting_event_ids),
            "contradicting_event_ids": list(self.contradicting_event_ids),
            "redacted_refs": list(self.redacted_refs),
        }


@dataclass
class EvidenceValidation:
    """Sanitised references plus what had to be dropped or truncated."""

    window_id: str
    supporting_event_ids: List[str] = field(default_factory=list)
    contradicting_event_ids: List[str] = field(default_factory=list)
    redacted_refs: List[str] = field(default_factory=list)
    unsafe_event_ids: List[str] = field(default_factory=list)
    unsafe_contradicting_event_ids: List[str] = field(default_factory=list)
    dropped_redacted_refs: int = 0
    truncated: bool = False

    @property
    def has_supporting(self) -> bool:
        return bool(self.supporting_event_ids)

    def reject_reason(self) -> Optional[str]:
        if self.unsafe_event_ids:
            return "unsafe_supporting_event_reference"
        if self.unsafe_contradicting_event_ids:
            return "unsafe_contradicting_event_reference"
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "supporting_event_ids": list(self.supporting_event_ids),
            "contradicting_event_ids": list(self.contradicting_event_ids),
            "redacted_refs": list(self.redacted_refs),
            "unsafe_event_ids": list(self.unsafe_event_ids),
            "unsafe_contradicting_event_ids": list(
                self.unsafe_contradicting_event_ids
            ),
            "dropped_redacted_refs": self.dropped_redacted_refs,
            "truncated": self.truncated,
        }


def is_safe_event_id(value: Any) -> bool:
    """True only for an opaque, non-sensitive event identifier."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or text != value:
        return False
    if not _EVENT_ID_RE.fullmatch(text):
        return False
    lowered = text.lower()
    if any(word in lowered for word in _SECRET_WORDS):
        return False
    if _EMAIL_RE.search(text) or _IPV4_RE.search(text) or _JWT_RE.search(text):
        return False
    return True


def is_safe_redacted_ref(value: Any) -> bool:
    """True only for a bounded pointer that does not look like a secret."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or text != value or len(text) > 512:
        return False
    if not _REDACTED_REF_RE.fullmatch(text):
        return False
    lowered = text.lower()
    if any(word in lowered for word in _SECRET_WORDS):
        return False
    if _EMAIL_RE.search(text) or _IPV4_RE.search(text) or _JWT_RE.search(text):
        return False
    return True


def is_safe_identity(value: Any) -> bool:
    """True for a bounded operational identifier such as a host or service.

    Hostnames and service names are required by the incident contract, so they
    are kept — but they must not smuggle an email address, a JWT, a
    ``key=value`` credential or free-form message text.  A bare IP address is
    permitted because it is a legitimate host identity; it is **not** accepted
    as an event reference or a redacted-evidence pointer.
    """
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or text != value or len(text) > 255:
        return False
    if not _IDENTITY_RE.fullmatch(text):
        return False
    lowered = text.lower()
    if any(word in lowered for word in _SECRET_WORDS):
        return False
    if _EMAIL_RE.search(text) or _JWT_RE.search(text):
        return False
    if "=" in text or any(ch.isspace() for ch in text):
        return False
    return True


def validate_window_evidence(
    evidence: WindowEvidence,
    max_events: int,
    max_redacted_refs: int,
) -> EvidenceValidation:
    """Sanitise one evidence record and apply hard bounds.

    Unsafe supporting or contradicting IDs are **not** dropped silently: they
    make the candidate ineligible.  Unsafe optional redacted pointers are
    dropped and counted because they are not required for correlation.
    """
    result = EvidenceValidation(window_id=evidence.window_id)

    supporting: List[str] = []
    for value in evidence.supporting_event_ids:
        if is_safe_event_id(value):
            item = value.strip()
            if item not in supporting:
                supporting.append(item)
        else:
            result.unsafe_event_ids.append(_safe_preview(value))
    if len(supporting) > max_events:
        supporting = supporting[:max_events]
        result.truncated = True
    result.supporting_event_ids = supporting

    contradicting: List[str] = []
    for value in evidence.contradicting_event_ids:
        if is_safe_event_id(value):
            item = value.strip()
            if item not in contradicting:
                contradicting.append(item)
        else:
            result.unsafe_contradicting_event_ids.append(_safe_preview(value))
    if len(contradicting) > max_events:
        contradicting = contradicting[:max_events]
        result.truncated = True
    result.contradicting_event_ids = contradicting

    refs: List[str] = []
    for value in evidence.redacted_refs:
        if is_safe_redacted_ref(value):
            item = value.strip()
            if item not in refs:
                refs.append(item)
        else:
            result.dropped_redacted_refs += 1
    if len(refs) > max_redacted_refs:
        refs = refs[:max_redacted_refs]
        result.truncated = True
    result.redacted_refs = refs
    return result


def normalise_evidence_index(
    evidence: Any,
) -> Dict[str, WindowEvidence]:
    """Accept a mapping or sequence of :class:`WindowEvidence` records.

    Accepted forms::

        {"window-id": WindowEvidence(...)}
        {"window-id": ["event-1", "event-2"]}
        [WindowEvidence(...), ...]

    A mapping value that is a bare string is treated as one supporting ID.
    Duplicate window IDs in a sequence are rejected instead of merged, because
    merging would silently change the evidence for a scored window.
    """
    if evidence is None:
        return {}
    if isinstance(evidence, Mapping):
        index: Dict[str, WindowEvidence] = {}
        for key, value in evidence.items():
            if not isinstance(key, str) or not key:
                raise ValueError("evidence keys must be non-empty window IDs")
            if isinstance(value, WindowEvidence):
                record = value
                if record.window_id != key:
                    raise ValueError(
                        f"evidence key {key!r} does not match "
                        f"WindowEvidence.window_id {record.window_id!r}"
                    )
            elif isinstance(value, str):
                record = WindowEvidence(window_id=key, supporting_event_ids=(value,))
            elif isinstance(value, Iterable):
                record = WindowEvidence(
                    window_id=key,
                    supporting_event_ids=tuple(value),
                )
            else:
                raise TypeError(
                    f"unsupported evidence value for {key!r}: {type(value).__name__}"
                )
            index[key] = record
        return index

    if isinstance(evidence, Iterable):
        index = {}
        for record in evidence:
            if not isinstance(record, WindowEvidence):
                raise TypeError(
                    "evidence sequences must contain WindowEvidence records"
                )
            if record.window_id in index:
                raise ValueError(
                    f"duplicate evidence for window_id {record.window_id!r}"
                )
            index[record.window_id] = record
        return index
    raise TypeError("evidence must be a mapping, a sequence or None")


def _safe_preview(value: Any) -> str:
    """A short, non-reversible label for a rejected reference.

    The rejected value itself is not copied into the result because it may be
    the sensitive token that caused the rejection.
    """
    if value is None:
        return "<none>"
    if isinstance(value, str):
        return f"<string:{len(value)} chars>"
    return f"<{type(value).__name__}>"
