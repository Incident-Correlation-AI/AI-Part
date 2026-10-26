"""Deterministic, bounded, local-only incident simulator.

The simulator emits **canonical LAD event envelopes** (schema ``1.0``) and a
*separate* evaluation label channel.  It never opens a socket, never contacts a
host and never generates exploit traffic.  Authentication records are synthetic
log lines recording an outcome; no login is attempted anywhere.

Scenario families per run
-------------------------
======================  ==========================================
minutes                 scenario
======================  ==========================================
0-15                    normal
15-25                   HTTP 5xx burst
25-35                   malformed/duplicate/out-of-order telemetry
35-45                   database timeouts + latency rise
45-55                   normal recovery
55-65                   authentication-failure burst
65-75                   benign high traffic
75-85                   benign latency shift
85-90                   benign deployment/restart
======================  ==========================================

Only the first four incident families are labelled anomalous.  Benign shifts
remain ``is_anomaly=False`` so false positives are measurable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import json
import os
import random

from .features import NetworkLabel, labels_to_dataframe

SCENARIO_SCHEDULE: Tuple[Tuple[str, int, int], ...] = (
    ("normal", 0, 15),
    ("http_5xx_burst", 15, 25),
    ("malformed_telemetry", 25, 35),
    ("db_timeout_latency", 35, 45),
    ("normal_recovery", 45, 55),
    ("auth_failure_burst", 55, 65),
    ("benign_high_traffic", 65, 75),
    ("benign_latency_shift", 75, 85),
    ("benign_restart", 85, 90),
)

STREAMS: Tuple[Tuple[str, str, str], ...] = (
    ("sim-web-01", "web-api", "http"),
    ("sim-web-01", "auth-service", "auth"),
    ("sim-db-01", "orders-db", "db"),
)

_MALFORMED_REASONS = (
    "duplicate_event_id",
    "out_of_order_event",
    "naive_timestamp",
    "missing_host",
    "missing_event_id",
    "unsupported_schema_version",
    "invalid_status_code",
    "invalid_latency",
    "unknown_category",
)


@dataclass
class SimulatorConfig:
    seed: int = 42
    run_count: int = 10
    run_minutes: int = 90
    window_size_minutes: int = 1
    base_http_per_minute: int = 24
    base_auth_per_minute: int = 10
    base_db_per_minute: int = 24
    inject_malformed: bool = True
    max_events: int = 250_000
    start_time: str = "2026-01-01T00:00:00+00:00"

    def validate(self) -> None:
        if self.run_count < 3:
            raise ValueError("run_count must be >= 3")
        if self.run_minutes < 90:
            raise ValueError("run_minutes must be >= 90 (the scenario schedule is 90 minutes)")
        if self.window_size_minutes < 1:
            raise ValueError("window_size_minutes must be >= 1")
        if self.max_events < 1:
            raise ValueError("max_events must be >= 1")
        try:
            parsed = datetime.fromisoformat(self.start_time.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"start_time is not a valid ISO-8601 timestamp: {exc}") from exc
        if parsed.tzinfo is None:
            raise ValueError("start_time must be timezone-aware")


@dataclass
class ScenarioOnset:
    scenario: str
    start: datetime
    end: datetime
    is_anomaly: bool


@dataclass
class RunMetadata:
    run_id: str
    start: datetime
    end: datetime
    onset: Dict[str, ScenarioOnset] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "onset": {
                name: {
                    "start": value.start.isoformat(),
                    "end": value.end.isoformat(),
                    "is_anomaly": value.is_anomaly,
                }
                for name, value in self.onset.items()
            },
        }


@dataclass
class SimulatedDataset:
    events: List[Dict[str, Any]]
    labels: List[NetworkLabel]
    runs: Dict[str, RunMetadata]
    summary: Dict[str, Any]

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def label_count(self) -> int:
        return len(self.labels)

    def write(self, directory: str) -> Dict[str, str]:
        """Write events and labels to *separate* files under ``directory``."""
        os.makedirs(directory, exist_ok=True)
        events_path = os.path.join(directory, "events.jsonl")
        labels_path = os.path.join(directory, "labels.csv")
        manifest_path = os.path.join(directory, "simulator_manifest.json")

        with open(events_path, "w", encoding="utf-8") as fh:
            for event in self.events:
                fh.write(json.dumps(event, sort_keys=True))
                fh.write("\n")
        labels_to_dataframe(self.labels).to_csv(labels_path, index=False)
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "summary": self.summary,
                    "runs": {k: v.to_dict() for k, v in self.runs.items()},
                    "note": (
                        "Labels are evaluation-only. They are written to a separate "
                        "channel and are never joined into model features."
                    ),
                },
                fh,
                indent=2,
            )
        return {"events": events_path, "labels": labels_path, "manifest": manifest_path}


class _RunGenerator:
    def __init__(self, config: SimulatorConfig, run_index: int, base_start: datetime):
        config.validate()
        self.config = config
        self.run_index = run_index
        self.run_id = f"run-{run_index:02d}"
        self.rng = random.Random(config.seed * 100_003 + run_index * 7_919)
        self.start = base_start + timedelta(days=run_index)
        self.events: List[Dict[str, Any]] = []
        self.labels: List[NetworkLabel] = []
        self.expected_quality: Dict[str, int] = {reason: 0 for reason in _MALFORMED_REASONS}
        self._counter = 0
        self._max_event_ts: Dict[Tuple[str, str], datetime] = {}
        self.truncated = False

    # -- public ----------------------------------------------------------
    def generate(self) -> Tuple[List[Dict[str, Any]], List[NetworkLabel], RunMetadata, Dict[str, int], bool]:
        metadata = RunMetadata(
            run_id=self.run_id,
            start=self.start,
            end=self.start + timedelta(minutes=self.config.run_minutes),
        )
        for scenario, start_min, end_min in SCENARIO_SCHEDULE:
            metadata.onset[scenario] = ScenarioOnset(
                scenario=scenario,
                start=self.start + timedelta(minutes=start_min),
                end=self.start + timedelta(minutes=end_min),
                is_anomaly=scenario in {"http_5xx_burst", "db_timeout_latency", "auth_failure_burst"},
            )

        for minute in range(self.config.run_minutes):
            if self.truncated:
                break
            scenario = _scenario_for_minute(minute)
            minute_start = self.start + timedelta(minutes=minute)
            self._emit_minute(minute, minute_start, scenario)
            self._emit_labels(minute_start, scenario)
            if self.config.inject_malformed and scenario == "malformed_telemetry":
                self._emit_malformed(minute, minute_start)

        return self.events, self.labels, metadata, dict(self.expected_quality), self.truncated

    # -- event generation -------------------------------------------------
    def _emit_minute(self, minute: int, minute_start: datetime, scenario: str) -> None:
        high_traffic = scenario == "benign_high_traffic"
        restart_minute = scenario == "benign_restart"
        for host, service, kind in STREAMS:
            # Make one telemetry window deliberately empty so the detector must
            # report `insufficient_data`, not "normal".
            if (
                self.config.inject_malformed
                and scenario == "malformed_telemetry"
                and minute == 33
                and kind == "db"
            ):
                self._emit_malformed_db_window(minute_start, host, service)
                continue

            if kind == "http":
                self._emit_http(minute_start, host, service, scenario, high_traffic, restart_minute)
            elif kind == "auth":
                self._emit_auth(minute_start, host, service, scenario, high_traffic, restart_minute)
                self._emit_http(
                    minute_start, host, service, scenario, high_traffic, restart_minute,
                    request_count_scale=0.4, is_primary=False,
                )
            else:
                self._emit_db(minute_start, host, service, scenario, restart_minute)

            if restart_minute and minute % self.config.run_minutes == 85:
                self._emit_restart(minute_start, host, service)

    def _emit_http(
        self,
        ts: datetime,
        host: str,
        service: str,
        scenario: str,
        high_traffic: bool,
        restart_minute: bool,
        request_count_scale: float = 1.0,
        is_primary: bool = True,
    ) -> None:
        base = self.config.base_http_per_minute
        if high_traffic:
            count = int(base * 3.0 * request_count_scale)
        elif restart_minute:
            count = max(3, int(base * 0.35 * request_count_scale))
        else:
            count = max(3, int(base * request_count_scale + self.rng.randint(-2, 2)))

        if scenario == "http_5xx_burst":
            five_ratio = 0.38 if is_primary else 0.25
        else:
            five_ratio = 0.02 if is_primary else 0.01
        four_ratio = 0.03

        latency_scale = 1.0
        if scenario == "db_timeout_latency":
            latency_scale = 1.15
        elif scenario == "benign_latency_shift":
            latency_scale = 1.35

        for _ in range(count):
            status = 200
            roll = self.rng.random()
            if roll < five_ratio:
                status = self.rng.choice([500, 502, 503, 504])
            elif roll < five_ratio + four_ratio:
                status = self.rng.choice([400, 401, 403, 404, 429])
            latency = max(1.0, self.rng.gauss(120.0, 28.0) * latency_scale)
            severity = "error" if status >= 500 else ("warning" if status >= 400 else "info")
            self._add_event(
                ts=ts,
                host=host,
                service=service,
                source="app-nginx",
                category="http",
                event_type="http_request",
                severity=severity,
                message=f"GET /api/v1/orders {status}",
                attributes={
                    "status_code": status,
                    "latency_ms": round(latency, 2),
                    "method": "GET",
                },
            )

    def _emit_auth(
        self,
        ts: datetime,
        host: str,
        service: str,
        scenario: str,
        high_traffic: bool,
        restart_minute: bool,
    ) -> None:
        if high_traffic:
            successes = self.config.base_auth_per_minute * 3
            failures = 2
            distinct = 12
        elif scenario == "auth_failure_burst":
            successes = self.rng.randint(2, 5)
            failures = self.rng.randint(35, 55)
            distinct = self.rng.randint(20, 32)
        elif restart_minute:
            successes = 3
            failures = 0
            distinct = 3
        else:
            successes = max(1, self.config.base_auth_per_minute + self.rng.randint(-2, 2))
            failures = self.rng.randint(0, 2)
            distinct = 4

        for i in range(successes):
            self._add_event(
                ts=ts,
                host=host,
                service=service,
                source="sshd",
                category="auth",
                event_type="auth_success",
                severity="info",
                message="authentication success",
                attributes={
                    "auth_source_id": f"src-{self.rng.randrange(max(1, distinct))}",
                    "auth_account_id": f"acct-{self.rng.randrange(max(1, distinct))}",
                    "result": "success",
                },
            )
        for i in range(failures):
            self._add_event(
                ts=ts,
                host=host,
                service=service,
                source="sshd",
                category="auth",
                event_type="auth_failure",
                severity="warning",
                message="authentication failure",
                attributes={
                    "auth_source_id": f"src-{self.rng.randrange(max(1, distinct))}",
                    "auth_account_id": f"acct-{self.rng.randrange(max(1, distinct))}",
                    "result": "failure",
                },
            )

    def _emit_db(
        self,
        ts: datetime,
        host: str,
        service: str,
        scenario: str,
        restart_minute: bool,
    ) -> None:
        count = max(3, self.config.base_db_per_minute + self.rng.randint(-2, 2))
        if restart_minute:
            count = 8
        if scenario == "db_timeout_latency":
            timeout_ratio = 0.35
            conn_failure_ratio = 0.15
            latency_base = 210.0
        else:
            timeout_ratio = 0.02
            conn_failure_ratio = 0.01
            latency_base = 18.0

        for _ in range(count):
            is_timeout = self.rng.random() < timeout_ratio
            is_conn_failure = self.rng.random() < conn_failure_ratio
            if is_timeout:
                event_type = "db_timeout"
                severity = "warning"
                latency = self.rng.gauss(650.0, 120.0)
            elif is_conn_failure:
                event_type = "db_connection_failure"
                severity = "error"
                latency = self.rng.gauss(900.0, 150.0)
            else:
                event_type = "db_query"
                severity = "info"
                latency = self.rng.gauss(latency_base, latency_base * 0.25)
            self._add_event(
                ts=ts,
                host=host,
                service=service,
                source="postgres",
                category="db",
                event_type=event_type,
                severity=severity,
                message=f"query {event_type}",
                attributes={
                    "latency_ms": round(max(0.5, latency), 2),
                    "db_timeout": bool(is_timeout),
                    "db_connection_failure": bool(is_conn_failure),
                    "query_template": "select_orders",
                },
            )

    def _emit_restart(self, ts: datetime, host: str, service: str) -> None:
        self._add_event(
            ts=ts,
            host=host,
            service=service,
            source="systemd",
            category="service",
            event_type="service_restart",
            severity="warning",
            message="scheduled restart completed",
            attributes={"restart_reason": "deployment"},
        )

    # -- malformed telemetry ---------------------------------------------
    def _emit_malformed(self, minute: int, minute_start: datetime) -> None:
        if not self.events:
            return
        # Duplicate a handful of already-emitted events (same event_id).
        for source in self.rng.sample(self.events[-25:], k=min(4, len(self.events[-25:]))):
            clone = dict(source)
            self.events.append(clone)
            self.expected_quality["duplicate_event_id"] += 1

        # Replay an older event with a *new* id after newer events were seen.
        replay_pool = self.events[:-1]
        if replay_pool:
            for source in self.rng.sample(replay_pool, k=min(4, len(replay_pool))):
                clone = dict(source)
                clone["event_id"] = self._next_id()
                clone["observed_ts"] = _iso(minute_start + timedelta(seconds=59))
                self.events.append(clone)
                self.expected_quality["out_of_order_event"] += 1

        # Naive timestamp.
        clone = dict(self.events[-1])
        clone["event_id"] = self._next_id()
        clone["event_ts"] = clone["event_ts"].replace("Z", "").replace("+00:00", "")
        self.events.append(clone)
        self.expected_quality["naive_timestamp"] += 1

        # Missing host.
        clone = dict(self.events[-1])
        clone["event_id"] = self._next_id()
        clone.pop("host", None)
        self.events.append(clone)
        self.expected_quality["missing_host"] += 1

        # Missing event id.
        clone = dict(self.events[-1])
        clone.pop("event_id", None)
        self.events.append(clone)
        self.expected_quality["missing_event_id"] += 1

        # Unsupported envelope version.
        clone = dict(self.events[-1])
        clone["event_id"] = self._next_id()
        clone["schema_version"] = "9.9"
        self.events.append(clone)
        self.expected_quality["unsupported_schema_version"] += 1

        # Accepted event with an invalid status code and invalid latency.
        self._add_event(
            ts=minute_start,
            host="sim-web-01",
            service="web-api",
            source="app-nginx",
            category="http",
            event_type="http_request",
            severity="info",
            message="malformed measurement",
            attributes={"status_code": "not-a-number", "latency_ms": "slow"},
        )
        self.expected_quality["invalid_status_code"] += 1
        self.expected_quality["invalid_latency"] += 1

        # Unknown categorical values are accepted but counted, never silently
        # treated as a known category.
        self._add_event(
            ts=minute_start,
            host="sim-web-01",
            service="web-api",
            source="custom-agent",
            category="mystery",
            event_type="mystery_event",
            severity="info",
            message="unknown category",
            attributes={},
        )
        self.expected_quality["unknown_category"] += 1

    def _emit_malformed_db_window(self, ts: datetime, host: str, service: str) -> None:
        """Emit only unusable DB records for one minute (empty valid window)."""
        for _ in range(6):
            clone = {
                "schema_version": "1.0",
                "event_id": self._next_id(),
                "event_ts": _iso(ts),
                "observed_ts": _iso(ts),
                "source": "postgres",
                "service": service,
                "category": "db",
                "type": "db_query",
                "severity": "info",
                "message": "query",
                "attributes": {"latency_ms": 12.0},
            }
            # Missing host -> rejected, so no accepted event exists for the window.
            self.events.append(clone)
            self.expected_quality["missing_host"] += 1

    # -- labels -----------------------------------------------------------
    def _emit_labels(self, minute_start: datetime, scenario: str) -> None:
        incident = scenario in {"http_5xx_burst", "db_timeout_latency", "auth_failure_burst"}
        for host, service, kind in STREAMS:
            affected = (
                (scenario == "http_5xx_burst" and kind in {"http", "auth"})
                or (scenario == "db_timeout_latency" and kind == "db")
                or (scenario == "auth_failure_burst" and kind == "auth")
            )
            self.labels.append(
                NetworkLabel(
                    window_start=minute_start,
                    window_end=minute_start + timedelta(minutes=1),
                    host_id=host,
                    service=service,
                    is_anomaly=bool(incident and affected),
                    scenario_name=scenario,
                    run_id=self.run_id,
                    telemetry_issue=scenario == "malformed_telemetry",
                    description=f"{self.run_id}: {scenario}",
                )
            )

    # -- helpers ----------------------------------------------------------
    def _add_event(
        self,
        ts: datetime,
        host: str,
        service: str,
        source: str,
        category: str,
        event_type: str,
        severity: str,
        message: str,
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if len(self.events) >= self.config.max_events:
            self.truncated = True
            return
        stream = (host, service)
        previous = self._max_event_ts.get(stream)
        if previous is not None and ts < previous:
            self.expected_quality["out_of_order_event"] += 1
        if previous is None or ts > previous:
            self._max_event_ts[stream] = ts

        self.events.append({
            "schema_version": "1.0",
            "event_id": self._next_id(),
            "event_ts": _iso(ts),
            "observed_ts": _iso(ts + timedelta(seconds=self.rng.uniform(0.1, 2.0))),
            "host": host,
            "source": source,
            "service": service,
            "category": category,
            "type": event_type,
            "severity": severity,
            "message": message,
            "template": None,
            "attributes": dict(attributes or {}),
        })

    def _next_id(self) -> str:
        self._counter += 1
        return f"{self.run_id}-evt-{self._counter:07d}"


def simulate(config: Optional[SimulatorConfig] = None) -> SimulatedDataset:
    """Generate all configured runs deterministically."""
    config = config or SimulatorConfig()
    config.validate()
    try:
        base_start = datetime.fromisoformat(config.start_time.replace("Z", "+00:00"))
    except ValueError as exc:  # pragma: no cover - validate() already checks
        raise ValueError(f"invalid start_time: {exc}") from exc

    all_events: List[Dict[str, Any]] = []
    all_labels: List[NetworkLabel] = []
    runs: Dict[str, RunMetadata] = {}
    expected_quality: Dict[str, int] = {reason: 0 for reason in _MALFORMED_REASONS}
    truncated = False

    for index in range(config.run_count):
        generator = _RunGenerator(config, index, base_start)
        events, labels, metadata, quality, run_truncated = generator.generate()
        all_events.extend(events)
        all_labels.extend(labels)
        runs[metadata.run_id] = metadata
        for reason, count in quality.items():
            expected_quality[reason] = expected_quality.get(reason, 0) + count
        if run_truncated:
            truncated = True
            break

    scenario_counts: Dict[str, int] = {}
    for label in all_labels:
        scenario_counts[label.scenario_name] = scenario_counts.get(label.scenario_name, 0) + 1

    summary = {
        "seed": config.seed,
        "run_count": len(runs),
        "run_minutes": config.run_minutes,
        "window_size_minutes": config.window_size_minutes,
        "event_count": len(all_events),
        "label_count": len(all_labels),
        "truncated": truncated,
        "max_events": config.max_events,
        "scenarios": scenario_counts,
        "incident_windows": sum(1 for label in all_labels if label.is_anomaly),
        "telemetry_issue_windows": sum(1 for label in all_labels if label.telemetry_issue),
        "expected_quality_reasons": expected_quality,
        "note": (
            "All events are synthetic and local. No network request was made. "
            "Authentication records are log lines, not login attempts."
        ),
    }
    return SimulatedDataset(events=all_events, labels=all_labels, runs=runs, summary=summary)


def _scenario_for_minute(minute: int) -> str:
    for name, start, end in SCENARIO_SCHEDULE:
        if start <= minute < end:
            return name
    return "normal"


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("simulator timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
