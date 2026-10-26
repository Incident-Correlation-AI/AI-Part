"""Controlled local-service telemetry lab for Layer B.

This script starts a **real local HTTP application** backed by a **real local
SQLite database**, drives ordinary and faulty traffic through it, collects the
application's structured telemetry, and streams canonical LAD envelopes into
the detector over HTTP at a bounded pace.  It is the Layer B harness: the
events are produced by actual HTTP requests and SQLite calls, not by a
pre-written feature table.

Ground truth is written to a separate ``labels.csv`` file.  It is never sent to
the detector, never included in a request body and never placed in an LLM
prompt.

Logical time is accelerated (default 20x) so a 25-minute operational schedule
finishes in about 75 seconds.  This is documented in the manifest and the
report; timestamps are the application's logical clock, not wall-clock time.

Example::

    # terminal 1 (Path L)
    python -m deploy.service --port 8080

    # terminal 2
    python -m deploy.local_telemetry_lab \
        --detector-url http://127.0.0.1:8080 \
        --out /tmp/opencode/dual-e2e/layer-b
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import queue
import random
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

LOGGER = logging.getLogger("lad.deploy.local_lab")

HOST_ID = "lab-node-1"
LOGICAL_START = datetime(2026, 10, 26, 8, 0, 0, tzinfo=timezone.utc)

#: Per-minute schedule.  ``is_anomaly`` is evaluator-only ground truth.
SCHEDULE: Tuple[Tuple[str, bool, str], ...] = (
    ("normal", False, "warm-up baseline"),
    ("normal", False, "warm-up baseline"),
    ("normal", False, "warm-up baseline"),
    ("normal", False, "warm-up baseline"),
    ("normal", False, "warm-up baseline"),
    ("normal", False, "warm-up baseline"),
    ("benign_high_traffic", False, "benign 3x traffic, all 200"),
    ("benign_high_traffic", False, "benign 3x traffic, all 200"),
    ("benign_latency_shift", False, "benign +50 ms latency"),
    ("benign_latency_shift", False, "benign +50 ms latency"),
    ("http_5xx_burst", True, "checkout 5xx spike"),
    ("http_5xx_burst", True, "checkout 5xx spike"),
    ("normal_recovery", False, "recovery after 5xx spike"),
    ("normal_recovery", False, "recovery after 5xx spike"),
    ("db_timeout_latency", True, "SQLite timeout and high latency"),
    ("db_timeout_latency", True, "SQLite timeout and high latency"),
    ("normal_recovery", False, "recovery after DB timeout"),
    ("normal_recovery", False, "recovery after DB timeout"),
    ("auth_failure_burst", True, "auth failure burst, many sources"),
    ("auth_failure_burst", True, "auth failure burst, many sources"),
    ("normal_recovery", False, "recovery after auth burst"),
    ("normal_recovery", False, "recovery after auth burst"),
    ("benign_restart", False, "single routine restart"),
    ("malformed_telemetry", False, "malformed/duplicate/out-of-order injection"),
    ("normal", False, "final normal minute"),
    ("normal", False, "final normal minute"),
)


class LogicalClock:
    def __init__(self, scale: float, start: datetime = LOGICAL_START) -> None:
        if scale <= 0:
            raise ValueError("time scale must be positive")
        self.scale = float(scale)
        self.start_logical = start
        self.start_wall = time.monotonic()

    def now(self) -> datetime:
        elapsed = time.monotonic() - self.start_wall
        return self.start_logical + timedelta(seconds=elapsed * self.scale)

    def minute(self, index: int) -> datetime:
        return self.start_logical + timedelta(minutes=index)

    def wait_until(self, value: datetime) -> None:
        target = self.start_wall + (value - self.start_logical).total_seconds() / self.scale
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(delay)


@dataclass
class LabState:
    scenario: str = "normal"
    http_5xx_ratio: float = 0.0
    db_timeout_ratio: float = 0.0
    auth_failure_ratio: float = 0.0
    latency_ms: float = 15.0
    rate_multiplier: float = 1.0
    restart_pending: bool = False


class TelemetrySink:
    """Writes every emitted envelope to JSONL and queues it for ingestion."""

    def __init__(self, path: str, max_queue: int = 50_000) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._handle = open(path, "w", encoding="utf-8")
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max_queue)
        self.sequence = 0
        self.last_event_id: Optional[str] = None

    def emit(
        self,
        clock: LogicalClock,
        *,
        service: str,
        category: str,
        event_type: str,
        severity: str,
        attributes: Optional[Mapping[str, Any]] = None,
        source: str = "local-lab-collector",
    ) -> Dict[str, Any]:
        with self._lock:
            self.sequence += 1
            event_id = f"lab-{self.sequence:06d}"
            timestamp = clock.now().isoformat()
            event = {
                "schema_version": "1.0",
                "event_id": event_id,
                "event_ts": timestamp,
                "observed_ts": timestamp,
                "host": HOST_ID,
                "service": service,
                "category": category,
                "type": event_type,
                "severity": severity,
                "source": source,
                "attributes": dict(attributes or {}),
            }
            self._handle.write(json.dumps(event, sort_keys=True) + "\n")
            self._handle.flush()
            self.last_event_id = event_id
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            LOGGER.warning("telemetry queue full; event kept only in JSONL")
        return event

    def close(self) -> None:
        with self._lock:
            self._handle.close()


class LabServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: Tuple[str, int],
        clock: LogicalClock,
        state: LabState,
        sink: TelemetrySink,
        seed: int,
    ) -> None:
        self.clock = clock
        self.state = state
        self.sink = sink
        self.db_path = os.path.join(os.path.dirname(sink.path), "lab.sqlite3")
        self.rng = random.Random(seed)
        self._db_lock = threading.Lock()
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.execute("CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY, amount REAL)")
        connection.execute("DELETE FROM orders")
        for index in range(100):
            connection.execute("INSERT INTO orders (id, amount) VALUES (?, ?)", (index, index * 1.5))
        connection.commit()
        self.db = connection
        super().__init__(address, Handler)

    def query_db(self, timeout: bool) -> Tuple[bool, float]:
        started = time.perf_counter()
        with self._db_lock:
            try:
                if timeout:
                    time.sleep(min(0.35, 0.05 + self.rng.random() * 0.30))
                    raise sqlite3.OperationalError("database is locked")
                self.db.execute("SELECT COUNT(*), SUM(amount) FROM orders").fetchone()
                return True, (time.perf_counter() - started) * 1000.0
            except sqlite3.OperationalError:
                return False, (time.perf_counter() - started) * 1000.0

    def close_lab(self) -> None:
        try:
            self.db.close()
        finally:
            self.server_close()


class Handler(BaseHTTPRequestHandler):
    server_version = "LADLocalLab/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        server: LabServer = self.server  # type: ignore[assignment]
        path = self.path.split("?", 1)[0]
        state = server.state
        if path == "/health":
            self._json(HTTPStatus.OK, {"status": "ok", "scenario": state.scenario})
            return
        if path == "/checkout":
            latency = state.latency_ms + server.rng.random() * 8.0
            time.sleep(latency / 1000.0)
            db_ok, db_latency = server.query_db(
                state.db_timeout_ratio > 0 and server.rng.random() < state.db_timeout_ratio
            )
            if not db_ok:
                server.sink.emit(
                    server.clock,
                    service="orders-db",
                    category="db",
                    event_type="db_timeout",
                    severity="error",
                    attributes={"latency_ms": db_latency, "db_timeout": True},
                )
                status = 500
            else:
                status = 500 if server.rng.random() < state.http_5xx_ratio else 200
            server.sink.emit(
                server.clock,
                service="checkout-api",
                category="http",
                event_type="http_request",
                severity="error" if status >= 500 else "info",
                attributes={"status_code": status, "latency_ms": latency},
            )
            self._json(
                HTTPStatus.OK if status < 500 else HTTPStatus.INTERNAL_SERVER_ERROR,
                {"status": status, "latency_ms": round(latency, 3)},
            )
            return
        if path == "/auth":
            failed = server.rng.random() < state.auth_failure_ratio
            server.sink.emit(
                server.clock,
                service="auth-service",
                category="auth",
                event_type="auth_failure" if failed else "auth_success",
                severity="warning" if failed else "info",
                attributes={
                    "status_code": 401 if failed else 200,
                    "auth_source_id": f"src-{server.rng.randrange(1000, 9999)}",
                    "auth_account_id": f"acct-{server.rng.randrange(100, 999)}",
                },
            )
            self._json(
                HTTPStatus.UNAUTHORIZED if failed else HTTPStatus.OK,
                {"authenticated": not failed},
            )
            return
        if path == "/db":
            db_ok, db_latency = server.query_db(
                state.db_timeout_ratio > 0 and server.rng.random() < state.db_timeout_ratio
            )
            server.sink.emit(
                server.clock,
                service="orders-db",
                category="db",
                event_type="db_query" if db_ok else "db_timeout",
                severity="info" if db_ok else "error",
                attributes={
                    "latency_ms": db_latency,
                    "db_timeout": not db_ok,
                },
            )
            self._json(
                HTTPStatus.OK if db_ok else HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": db_ok, "latency_ms": round(db_latency, 3)},
            )
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        server: LabServer = self.server  # type: ignore[assignment]
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length") or "0")
        if length:
            self.rfile.read(length)
        if path == "/restart":
            server.sink.emit(
                server.clock,
                service="checkout-api",
                category="service",
                event_type="service_restart",
                severity="warning",
                attributes={},
            )
            self._json(HTTPStatus.OK, {"restarted": True})
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})


def _post_json(url: str, payload: Mapping[str, Any], timeout: float = 30.0) -> Tuple[int, Dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return int(response.status), json.loads(response.read().decode("utf-8"))


def _get_json(url: str, timeout: float = 10.0) -> Tuple[int, Dict[str, Any]]:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return int(response.status), json.loads(response.read().decode("utf-8"))


def collector_loop(
    sink: TelemetrySink,
    detector_url: str,
    stop_event: threading.Event,
    errors: List[str],
    batch_size: int = 40,
    flush_seconds: float = 0.5,
) -> None:
    batch: List[Dict[str, Any]] = []
    last_flush = time.monotonic()
    while not stop_event.is_set() or not sink.queue.empty():
        timeout = max(0.05, flush_seconds - (time.monotonic() - last_flush))
        try:
            batch.append(sink.queue.get(timeout=timeout))
        except queue.Empty:
            pass
        due = len(batch) >= batch_size or (time.monotonic() - last_flush) >= flush_seconds
        if not due or not batch:
            continue
        payload = {"events": batch}
        try:
            status, _ = _post_json(f"{detector_url}/v1/events", payload)
            if status != 200:
                errors.append(f"ingest returned HTTP {status}")
        except Exception as exc:  # noqa: BLE001 - recorded, run continues
            errors.append(f"ingest failed: {type(exc).__name__}: {exc}")
            LOGGER.warning("ingest failed: %s", exc)
        batch = []
        last_flush = time.monotonic()
    if batch:
        try:
            _post_json(f"{detector_url}/v1/events", {"events": batch})
        except Exception as exc:  # noqa: BLE001
            errors.append(f"final ingest failed: {type(exc).__name__}: {exc}")


def _http_call(url: str, method: str = "GET", timeout: float = 5.0) -> None:
    request = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout):  # noqa: S310
            pass
    except urllib.error.HTTPError as exc:
        # 4xx/5xx application responses are expected fault traffic.
        if exc.code not in {401, 500}:
            LOGGER.warning("lab request %s returned HTTP %s", url, exc.code)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("lab request %s failed: %s", url, exc)


def _apply_scenario(state: LabState, scenario: str) -> None:
    state.scenario = scenario
    state.http_5xx_ratio = 0.0
    state.db_timeout_ratio = 0.0
    state.auth_failure_ratio = 0.0
    state.latency_ms = 15.0
    state.rate_multiplier = 1.0
    if scenario == "benign_high_traffic":
        state.rate_multiplier = 3.0
    elif scenario == "benign_latency_shift":
        state.latency_ms = 65.0
    elif scenario == "http_5xx_burst":
        state.http_5xx_ratio = 0.45
    elif scenario == "db_timeout_latency":
        state.db_timeout_ratio = 0.55
        state.latency_ms = 220.0
    elif scenario == "auth_failure_burst":
        state.auth_failure_ratio = 0.85
    elif scenario == "benign_restart":
        state.restart_pending = True


def run_schedule(
    app_url: str,
    detector_url: str,
    clock: LogicalClock,
    state: LabState,
    server: LabServer,
    stop_event: threading.Event,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    labels: List[Dict[str, Any]] = []
    for index, (scenario, is_anomaly, description) in enumerate(SCHEDULE):
        if stop_event.is_set():
            break
        minute_start = clock.minute(index)
        clock.wait_until(minute_start)
        _apply_scenario(state, scenario)

        minute_end = minute_start + timedelta(minutes=1)
        wall_budget = 60.0 / clock.scale * 0.80
        started_wall = time.monotonic()

        checkout_rate = int(20 * state.rate_multiplier)
        auth_rate = int(12 * state.rate_multiplier)
        db_rate = int(12 * state.rate_multiplier)

        calls: List[str] = []
        calls.extend([f"{app_url}/checkout"] * checkout_rate)
        calls.extend([f"{app_url}/auth"] * auth_rate)
        calls.extend([f"{app_url}/db"] * db_rate)
        rng.shuffle(calls)

        for position, url in enumerate(calls):
            if clock.now() >= minute_end:
                break
            _http_call(url)
            remaining = wall_budget - (time.monotonic() - started_wall)
            left = len(calls) - position - 1
            if left > 0 and remaining > 0:
                time.sleep(remaining / left)

        if state.restart_pending:
            _http_call(f"{app_url}/restart", method="POST")
            state.restart_pending = False

        if scenario == "malformed_telemetry":
            _inject_telemetry_faults(detector_url, clock, server.sink)

        affected = {
            "http_5xx_burst": ["checkout-api"],
            "db_timeout_latency": ["orders-db", "checkout-api"],
            "auth_failure_burst": ["auth-service"],
            "benign_high_traffic": ["checkout-api", "auth-service", "orders-db"],
            "benign_latency_shift": ["checkout-api", "orders-db"],
            "benign_restart": ["checkout-api"],
            "malformed_telemetry": ["checkout-api"],
        }.get(scenario, ["checkout-api", "auth-service", "orders-db"])

        for service in ("checkout-api", "auth-service", "orders-db"):
            labels.append(
                {
                    "window_start": minute_start.isoformat(),
                    "window_end": minute_end.isoformat(),
                    "host_id": HOST_ID,
                    "service": service,
                    "is_anomaly": bool(is_anomaly and service in affected),
                    "scenario_name": scenario,
                    "run_id": "layer-b-local-lab",
                    "telemetry_issue": bool(
                        scenario == "malformed_telemetry" and service == "checkout-api"
                    ),
                    "description": description,
                }
            )
    return labels


def _inject_telemetry_faults(detector_url: str, clock: LogicalClock, sink: TelemetrySink) -> None:
    """Send malformed, duplicate and out-of-order envelopes through the API."""
    now = clock.now().isoformat()
    faults: List[Any] = [
        {"schema_version": "0.9", "event_id": "lab-bad-version", "event_ts": now,
         "host": HOST_ID, "service": "checkout-api", "category": "http",
         "type": "http_request", "severity": "info", "attributes": {}},
        {"schema_version": "1.0", "event_id": "lab-bad-host", "event_ts": now,
         "service": "checkout-api", "category": "http", "type": "http_request",
         "severity": "info", "attributes": {}},
        {"schema_version": "1.0", "event_id": "lab-bad-ts",
         "event_ts": "2026-10-26T08:00:00", "host": HOST_ID,
         "service": "checkout-api", "category": "http", "type": "http_request",
         "severity": "info", "attributes": {}},
        {"schema_version": "1.0", "event_id": "lab-bad-attributes", "event_ts": now,
         "host": HOST_ID, "service": "checkout-api", "category": "http",
         "type": "http_request", "severity": "info", "attributes": "not-a-mapping"},
        42,
        # Out-of-order: an earlier timestamp sent after later events.
        {"schema_version": "1.0", "event_id": "lab-out-of-order",
         "event_ts": (clock.now() - timedelta(minutes=3)).isoformat(),
         "host": HOST_ID, "service": "checkout-api", "category": "http",
         "type": "http_request", "severity": "info",
         "attributes": {"status_code": 200, "latency_ms": 12.0}},
        # Label/scenario leakage attempt: dropped at ingress, never scored.
        {"schema_version": "1.0", "event_id": "lab-leak-attempt", "event_ts": now,
         "host": HOST_ID, "service": "checkout-api", "category": "http",
         "type": "http_request", "severity": "info",
         "label": "anomaly", "scenario_name": "http_5xx_burst", "test_id": "T-999",
         "attributes": {"status_code": 200, "latency_ms": 12.0,
                        "ground_truth": "anomaly"}},
    ]
    # Duplicate of a real event id (across the request boundary).
    if sink.last_event_id:
        faults.append(
            {
                "schema_version": "1.0",
                "event_id": sink.last_event_id,
                "event_ts": now,
                "host": HOST_ID,
                "service": "checkout-api",
                "category": "http",
                "type": "http_request",
                "severity": "info",
                "attributes": {"status_code": 200, "latency_ms": 10.0},
            }
        )
    try:
        _post_json(f"{detector_url.rstrip('/')}/v1/events", {"events": faults})
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("fault injection failed: %s", exc)


def _write_labels(path: str, labels: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "window_start", "window_end", "host_id", "service", "is_anomaly",
        "scenario_name", "run_id", "telemetry_issue", "description",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label in labels:
            writer.writerow({key: label.get(key, "") for key in fields})


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-url", default=os.environ.get("LAD_DETECTOR_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--out", default=os.path.join("artifacts", "audit", "layer_b"))
    parser.add_argument("--scale", type=float, default=20.0, help="logical seconds per wall second")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lab-port", type=int, default=18080)
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, os.environ.get("LAD_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    os.makedirs(args.out, exist_ok=True)
    clock = LogicalClock(args.scale)
    state = LabState()
    sink = TelemetrySink(os.path.join(args.out, "telemetry.jsonl"))
    server = LabServer(("127.0.0.1", args.lab_port), clock, state, sink, args.seed)
    app_thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    app_thread.start()
    app_url = f"http://127.0.0.1:{args.lab_port}"

    errors: List[str] = []
    stop_event = threading.Event()
    collector = threading.Thread(
        target=collector_loop,
        args=(sink, args.detector_url, stop_event, errors),
        daemon=True,
    )
    collector.start()

    started_wall = time.monotonic()
    labels: List[Dict[str, Any]] = []
    analysis: Optional[Dict[str, Any]] = None
    ready: Optional[Dict[str, Any]] = None
    try:
        _, ready = _get_json(f"{args.detector_url}/readyz")
        LOGGER.info("detector ready: %s", json.dumps(ready.get("model", {}), sort_keys=True))
        labels = run_schedule(
            app_url, args.detector_url, clock, state, server, stop_event, args.seed
        )
    finally:
        stop_event.set()
        collector.join(timeout=30)
        sink.close()
        server.shutdown()
        server.close_lab()

    _write_labels(os.path.join(args.out, "labels.csv"), labels)

    try:
        status, analysis = _post_json(
            f"{args.detector_url}/v1/analyze",
            {"use_buffer": True, "explain": bool(args.explain)},
            timeout=120.0,
        )
        with open(os.path.join(args.out, "detector_response.json"), "w", encoding="utf-8") as handle:
            json.dump({"status": status, "response": analysis}, handle, indent=2, sort_keys=True)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"analyze failed: {type(exc).__name__}: {exc}")
        LOGGER.exception("analyze failed")

    scenario_counts: Dict[str, int] = {}
    for label in labels:
        key = str(label["scenario_name"])
        scenario_counts[key] = scenario_counts.get(key, 0) + 1

    manifest = {
        "layer": "B",
        "note": (
            "Controlled local web app + SQLite telemetry collected through the real "
            "HTTP ingestion and inference path. Logical time is accelerated; labels "
            "are evaluator-only and were never sent to the detector."
        ),
        "detector_url": args.detector_url,
        "lab_url": app_url,
        "time_scale": args.scale,
        "seed": args.seed,
        "schedule_minutes": len(SCHEDULE),
        "wall_duration_seconds": round(time.monotonic() - started_wall, 3),
        "event_count": sink.sequence,
        "label_count": len(labels),
        "scenario_counts": scenario_counts,
        "collector_errors": errors,
        "detector_ready": ready,
        "analysis_status": analysis.get("windows") if analysis else None,
    }
    with open(os.path.join(args.out, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if analysis is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
