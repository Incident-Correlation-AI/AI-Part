#!/usr/bin/env python3
"""Offline end-to-end rehearsal: simulated events → ML → incidents.

Runs the deterministic network simulator into a temporary directory, reloads the
persisted model through the public inference boundary, scores the windows,
builds a validated evidence index from the simulated canonical LAD events, and
passes the results through :class:`EvidenceCorrelationPipeline`.

The optional explanation layer uses the offline mock provider.  No network
request, credential or external service is involved.

Examples
--------
    python scripts/run_correlation_rehearsal.py
    python scripts/run_correlation_rehearsal.py --explain --limit 200
    python scripts/run_correlation_rehearsal.py --config configs/network.yaml
"""
import argparse
import json
import os
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from anomaly_detector.correlation import (
    CorrelationConfig,
    EvidenceCorrelationPipeline,
    WindowEvidence,
)
from anomaly_detector.inference.contract import InferenceStatus
from anomaly_detector.network.config import load_network_config
from anomaly_detector.network.inference import load_network_detector
from anomaly_detector.network.pipeline import run_network_pipeline


def _parse(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def build_evidence(results, events, limit_per_window=5):
    """Map scored anomalies to the real simulated event IDs in their window."""
    parsed_events = []
    for event in events:
        timestamp = _parse(event.get("event_ts"))
        if timestamp is None:
            continue
        parsed_events.append((timestamp, event))

    evidence = {}
    for result in results:
        if result.status is not InferenceStatus.SCORED or not result.is_anomaly:
            continue
        start = _parse(result.window_start)
        end = _parse(result.window_end)
        if start is None or end is None:
            continue
        ids = [
            event["event_id"]
            for timestamp, event in parsed_events
            if event.get("host") == result.host_id
            and event.get("service") == result.service_name
            and start <= timestamp < end
        ][:limit_per_window]
        if ids:
            evidence[result.window_id] = WindowEvidence(
                window_id=result.window_id,
                supporting_event_ids=tuple(ids),
            )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/network.yaml", help="Network pipeline config"
    )
    parser.add_argument(
        "--limit", type=int, default=400, help="Maximum windows to score (0 = all)"
    )
    parser.add_argument(
        "--explain", action="store_true", help="Use the offline mock explanation provider"
    )
    parser.add_argument(
        "--window-seconds", type=int, default=120, help="Upstream correlation window"
    )
    parser.add_argument(
        "--dump-incidents",
        action="store_true",
        help="Print full incident drafts (may be large)",
    )
    args = parser.parse_args()

    config = load_network_config(args.config)
    # Keep the rehearsal fast and self-contained; the schedule still needs three
    # runs because the simulator's scenario plan is 90 minutes long.
    config.simulator.run_count = 3
    config.model.n_estimators = min(config.model.n_estimators, 50)
    config.model.max_samples = min(config.model.max_samples, 128)

    with tempfile.TemporaryDirectory(prefix="lad-correlation-rehearsal-") as tmp:
        config.paths["synthetic_dir"] = os.path.join(tmp, "synthetic")
        config.paths["splits_dir"] = os.path.join(tmp, "splits")
        config.paths["output_dir"] = os.path.join(tmp, "artifacts")
        config.paths["model_dir"] = os.path.join(tmp, "artifacts", "model")
        config.paths["reports_dir"] = os.path.join(tmp, "artifacts", "reports")

        run = run_network_pipeline(config, strategies=["time"])
        detector = load_network_detector(
            model_path=os.path.join(config.paths["model_dir"], "model.joblib"),
            manifest_path=os.path.join(config.paths["model_dir"], "manifest.json"),
            rules=config.rules,
        )

        windows = run.windows if args.limit <= 0 else run.windows[: args.limit]
        results = detector.score_windows(windows)
        evidence = build_evidence(results, run.dataset.events)

        outcome = EvidenceCorrelationPipeline(
            CorrelationConfig(
                window_seconds=args.window_seconds,
                explain=args.explain,
                llm_provider="mock",
            )
        ).correlate(results, evidence)

    summary = {
        "scored_windows": len(results),
        "evidence_windows": len(evidence),
        "counts": outcome.counts(),
        "unscored_by_status": outcome.unscored_by_status(),
        "llm": outcome.llm_summary(),
        "upstream_revision": outcome.to_dict()["upstream_revision"],
        "incidents": [
            {
                "incident_id": incident["incident_id"],
                "severity": incident["severity"],
                "anomaly_score": incident["anomaly_score"],
                "detector": incident["detector"],
                "services": [
                    entity["service"] for entity in incident["affected_entities"]
                ],
                "supporting_event_count": len(
                    incident["evidence"]["supporting_event_ids"]
                ),
                "hypotheses": len(incident["hypotheses"]),
                "checks": len(incident["diagnostic_checks"]),
            }
            for incident in outcome.incidents
        ],
    }
    if args.dump_incidents:
        summary["incident_drafts"] = outcome.incidents
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
