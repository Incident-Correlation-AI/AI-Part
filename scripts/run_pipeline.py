#!/usr/bin/env python3
"""Run the full Milestone 1 pipeline on an existing raw-events JSON file.

Input: a JSON file containing a list of raw event objects (the common event
representation). Optional sidecar labels file (JSON list of label objects or a
CSV with the label columns).

Example
-------
    python scripts/run_pipeline.py \
        --events data/synthetic/synthetic_events.json \
        --labels data/synthetic/synthetic_labels.json \
        --window-size 1 \
        --output-dir data/processed \
        --artifacts-dir artifacts
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from datetime import datetime, timezone

from anomaly_detector.config import load_config
from anomaly_detector.features import check_label_leakage
from anomaly_detector.pipeline import run_pipeline, export_result
from anomaly_detector.schemas import LabelInfo


def _load_labels(path):
    if not path:
        return []
    if not os.path.isfile(path):
        raise FileNotFoundError(f"labels file not found: {path}")
    labels = []
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            rows = json.load(fh)
        for r in rows:
            labels.append(LabelInfo(
                window_start=datetime.fromisoformat(r["window_start"]),
                window_end=datetime.fromisoformat(r["window_end"]),
                host_id=r["host_id"],
                service=r["service"],
                is_anomaly=bool(r["is_anomaly"]),
                scenario_name=r.get("scenario_name", "unknown"),
                intensity=r.get("intensity"),
                description=r.get("description"),
            ))
    else:
        import csv
        with open(path, "r", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                labels.append(LabelInfo(
                    window_start=datetime.fromisoformat(r["window_start"]),
                    window_end=datetime.fromisoformat(r["window_end"]),
                    host_id=r["host_id"],
                    service=r["service"],
                    is_anomaly=str(r["is_anomaly"]).lower() in ("1", "true", "yes"),
                    scenario_name=r.get("scenario_name", "unknown"),
                    intensity=float(r["intensity"]) if r.get("intensity") else None,
                    description=r.get("description"),
                ))
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Milestone 1 pipeline on raw event JSON")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--events", required=True, help="Path to raw events JSON (list of objects)")
    parser.add_argument("--labels", default=None, help="Optional labels JSON/CSV")
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--artifacts-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    window_size = args.window_size or cfg.window_size_minutes
    output_dir = args.output_dir or cfg.output_dir
    artifacts_dir = cfg.artifacts_dir if args.artifacts_dir is None else args.artifacts_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    with open(args.events, "r", encoding="utf-8") as fh:
        raw_events = json.load(fh)

    labels = _load_labels(args.labels)

    result = run_pipeline(
        raw_events,
        window_size_minutes=window_size,
        labels=labels,
        allow_empty_windows=cfg.allow_empty_windows,
    )

    leaks = check_label_leakage(result.feature_df)
    if leaks:
        print(f"ERROR: label leakage detected: {leaks}", file=sys.stderr)
        return 2

    paths = export_result(result, cfg.output_dir)
    with open(os.path.join(artifacts_dir, "data_quality_report.json"), "w", encoding="utf-8") as fh:
        json.dump(result.report, fh, indent=2)

    print(f"Processed {result.report['input']['total_processed']} events "
          f"(accepted={result.report['input']['accepted_count']}, "
          f"rejected={result.report['input']['rejected_count']}, "
          f"duplicated={result.report['input']['duplicated_count']})")
    print(f"Windows: {result.report['windows']['count']}  "
          f"Labels: {result.report['labels']['count']}")
    print(f"Features -> {paths['features']}")
    print(f"Report   -> {os.path.join(artifacts_dir, 'data_quality_report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
