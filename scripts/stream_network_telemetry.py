#!/usr/bin/env python3
"""Stream one simulated run through the real HTTP binding (Layer A regression).

The simulator's events and labels are kept separate.  Events are sent to
``/v1/events``; labels are written only to the evaluator-side ``labels.csv`` and
are never included in a request body or an LLM prompt.

This is developer-authored synthetic data and is reported as Layer A, not as
independent real-world validation.

Example::

    python -m deploy.service --port 8080 &
    python scripts/stream_network_telemetry.py \
        --url http://127.0.0.1:8080 --run-index 2 \
        --out /tmp/opencode/dual-e2e/layer-a
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from anomaly_detector.network.config import load_network_config  # noqa: E402
from anomaly_detector.network.features import NetworkLabel  # noqa: E402
from anomaly_detector.network.simulator import simulate  # noqa: E402


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
    return parsed


def _post_json(url: str, payload: Any, timeout: float = 120.0) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, timeout: float = 10.0) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _write_labels(path: str, labels: Sequence[NetworkLabel]) -> None:
    fields = [
        "window_start", "window_end", "host_id", "service", "is_anomaly",
        "scenario_name", "run_id", "telemetry_issue", "description",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label in labels:
            writer.writerow({key: label.to_dict().get(key, "") for key in fields})


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("LAD_DETECTOR_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--config", default=os.path.join(_REPO_ROOT, "configs", "network.yaml"))
    parser.add_argument("--run-index", type=int, default=2)
    parser.add_argument("--out", default=os.path.join(_REPO_ROOT, "artifacts", "audit", "layer_a"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--pause", type=float, default=0.02, help="seconds between batches")
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    config = load_network_config(args.config)
    simulator_config = config.simulator.to_simulator_config()
    dataset = simulate(simulator_config)

    run_id = f"run-{args.run_index:02d}"
    if run_id not in dataset.runs:
        raise SystemExit(f"run {run_id} not present; available: {sorted(dataset.runs)}")
    run = dataset.runs[run_id]
    events = [
        event
        for event in dataset.events
        if (timestamp := _parse_aware(event.get("event_ts"))) is not None
        and run.start <= timestamp < run.end
    ]
    labels = [label for label in dataset.labels if label.run_id == run_id]

    with open(os.path.join(args.out, "events.jsonl"), "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    _write_labels(os.path.join(args.out, "labels.csv"), labels)

    ready = _get_json(f"{args.url}/readyz")
    ingest_summaries: List[Dict[str, Any]] = []
    started = time.monotonic()
    for index in range(0, len(events), args.batch_size):
        batch = events[index:index + args.batch_size]
        summary = _post_json(f"{args.url}/v1/events", {"events": batch})
        ingest_summaries.append(summary)
        if args.pause:
            time.sleep(args.pause)

    analyze = _post_json(
        f"{args.url}/v1/analyze",
        {"use_buffer": True, "explain": bool(args.explain)},
        timeout=300.0,
    )
    with open(os.path.join(args.out, "detector_response.json"), "w", encoding="utf-8") as handle:
        json.dump(analyze, handle, indent=2, sort_keys=True)

    accepted = sum(int(item.get("accepted_now", 0)) for item in ingest_summaries)
    duplicates = sum(int(item.get("duplicates_now", 0)) for item in ingest_summaries)
    rejected = sum(int(item.get("rejected_now", 0)) for item in ingest_summaries)
    manifest = {
        "layer": "A",
        "note": (
            "Developer-authored synthetic simulator data streamed through the real "
            "HTTP ingestion and inference path. Labels are evaluator-only. This is "
            "regression evidence, not independent or real-world validation."
        ),
        "url": args.url,
        "run_id": run_id,
        "run_start": run.start.isoformat(),
        "run_end": run.end.isoformat(),
        "events_sent": len(events),
        "labels_written": len(labels),
        "accepted": accepted,
        "duplicates": duplicates,
        "rejected": rejected,
        "wall_duration_seconds": round(time.monotonic() - started, 3),
        "detector_ready": ready,
        "windows": analyze.get("windows"),
        "model": analyze.get("model"),
        "explanation": analyze.get("explanation"),
    }
    with open(os.path.join(args.out, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
