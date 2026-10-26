"""Canonical LAD envelope adapter and telemetry-quality tests."""
from datetime import datetime, timezone

import pytest

from anomaly_detector.inference.contract import CONTRACT_VERSION, ReasonCode
from anomaly_detector.network.lad_adapter import (
    LAD_EVENT_SCHEMA_VERSION,
    LadParseOptions,
    parse_lad_events,
    quality_invalid_reasons,
)
from anomaly_detector.schemas import EventType


def _envelope(event_id="e1", **overrides):
    envelope = {
        "schema_version": LAD_EVENT_SCHEMA_VERSION,
        "event_id": event_id,
        "event_ts": "2026-01-01T00:00:30+00:00",
        "observed_ts": "2026-01-01T00:00:31+00:00",
        "host": "web-01",
        "source": "nginx",
        "service": "web-api",
        "category": "http",
        "type": "http_request",
        "severity": "error",
        "message": "upstream failed with password=hunter2",
        "attributes": {
            "status_code": 503,
            "latency_ms": 42.0,
            "auth_source_id": "src-1",
            "auth_account_id": "acct-1",
            "password": "hunter2",
            "email": "person@example.com",
        },
    }
    envelope.update(overrides)
    return envelope


def test_canonical_event_maps_and_drops_pii():
    parsed = parse_lad_events([_envelope()])
    assert parsed.accepted_count == 1
    assert parsed.rejected_count == 0
    event = parsed.events[0]
    assert event.event_id == "e1"
    assert event.host_id == "web-01"
    assert event.service == "web-api"
    assert event.event_type == EventType.HTTP_REQUEST.value
    assert event.attributes["status_code"] == 503
    assert event.attributes["latency_ms"] == 42.0

    # Raw identifiers, credentials and message text never reach the event.
    serialised = str(event.__dict__)
    assert "hunter2" not in serialised
    assert "person@example.com" not in serialised
    assert "src-1" not in serialised
    assert "acct-1" not in serialised
    source = event.attributes["auth_source_id"]
    account = event.attributes["auth_account_id"]
    assert len(source) == 20 and len(account) == 20
    assert source != "src-1" and account != "acct-1"


def test_event_type_mapping_covers_auth_and_database():
    auth_fail = parse_lad_events([_envelope(
        category="auth", type="auth_failure", service="auth-service",
        attributes={"auth_source_id": "src-2"},
    )]).events[0]
    auth_ok = parse_lad_events([_envelope(
        event_id="e2", category="auth", type="auth_success",
        service="auth-service", attributes={"auth_account_id": "acct-2"},
    )]).events[0]
    db = parse_lad_events([_envelope(
        event_id="e3", category="db", type="db_query_timeout",
        service="orders-db", attributes={"latency_ms": 2500.0},
    )]).events[0]

    assert auth_fail.event_type == EventType.AUTH_FAILURE.value
    assert auth_ok.event_type == EventType.AUTH_SUCCESS.value
    assert db.event_type == EventType.DATABASE_QUERY.value
    assert db.attributes["db_timeout"] is True


def test_fatal_envelope_errors_are_rejected_individually():
    cases = [
        (_envelope(schema_version="9.9"), "unsupported_schema_version"),
        (_envelope(host=""), "missing_host"),
        (_envelope(event_id=""), "missing_event_id"),
        (_envelope(event_ts="2026-01-01T00:00:30"), "naive_timestamp"),
        (_envelope(event_ts="not-a-timestamp"), "invalid_timestamp"),
        (_envelope(category=""), "missing_type"),
        (42, "invalid_envelope"),
    ]
    for index, (raw, reason) in enumerate(cases):
        parsed = parse_lad_events([raw])
        assert parsed.accepted_count == 0
        assert parsed.rejected_count == 1
        assert parsed.rejections[0].reason == reason, (index, parsed.summary())


def test_duplicate_event_ids_are_counted_not_double_counted():
    parsed = parse_lad_events([_envelope(), _envelope()])
    assert parsed.accepted_count == 1
    assert parsed.rejected_count == 1
    assert parsed.duplicate_event_ids == ["e1"]
    assert parsed.rejections[0].reason == "duplicate_event_id"


def test_partial_quality_problems_are_recorded_without_dropping_the_event():
    parsed = parse_lad_events([
        _envelope(
            event_id="bad-status",
            attributes={"status_code": 999, "latency_ms": "slow"},
        ),
        _envelope(
            event_id="weird",
            category="unexpected",
            type="unknown_thing",
            service="",
            severity="noidea",
            event_ts="2026-01-01T00:01:30+00:00",
        ),
    ])
    assert parsed.accepted_count == 2
    assert parsed.rejected_count == 0
    counts = parsed.quality_counts()
    for reason in (
        "invalid_status_code",
        "invalid_latency",
        "unknown_category",
        "missing_service",
        "unknown_severity",
    ):
        assert counts[reason] >= 1, counts
    assert parsed.events[1].service == "unknown"
    assert "status_code" not in parsed.events[0].attributes


def test_out_of_order_events_are_flagged_per_stream():
    parsed = parse_lad_events([
        _envelope(event_id="late", event_ts="2026-01-01T00:02:30+00:00"),
        _envelope(event_id="early", event_ts="2026-01-01T00:02:00+00:00"),
    ])
    assert parsed.accepted_count == 2
    assert parsed.out_of_order_count == 1
    assert parsed.quality_counts()["out_of_order_event"] == 1
    bucket = parsed.quality_by_window(1)
    key = ("web-01", "web-api", datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc))
    assert bucket[key]["out_of_order_event"] == 1


def test_event_bound_is_enforced():
    parsed = parse_lad_events(
        [_envelope(event_id="a"), _envelope(event_id="b")],
        LadParseOptions(max_events=1),
    )
    assert parsed.accepted_count == 1
    assert parsed.quality_counts()["batch_bounded"] == 1


def test_quality_invalid_reasons_is_a_documented_superset():
    invalid = quality_invalid_reasons()
    for reason in ("duplicate_event_id", "missing_host", "invalid_status_code",
                   "out_of_order_event", "missing_service"):
        assert reason in invalid


# ---------------------------------------------------------------------------
# Additive contract extensions
# ---------------------------------------------------------------------------

def test_contract_extensions_are_additive_and_versioned():
    assert LAD_EVENT_SCHEMA_VERSION == "1.0"
    assert CONTRACT_VERSION == "1.1.0"
    assert ReasonCode.NO_EVENTS.value == "no_events"
    assert ReasonCode.INSUFFICIENT_COVERAGE.value == "insufficient_coverage"
    assert ReasonCode.TELEMETRY_QUALITY.value == "telemetry_quality"
    assert EventType.AUTH_SUCCESS.value == "auth_success"
