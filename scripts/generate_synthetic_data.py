#!/usr/bin/env python3
"""Generate reproducible synthetic data and run the full Milestone 1 pipeline.

Example
-------
    python scripts/generate_synthetic_data.py \
        --config configs/default.yaml \
        --seed 42 --window-size 1 \
        --output-dir data/synthetic
"""
import argparse
import json
import os
import sys

# Allow running from a clean checkout without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from anomaly_detector.config import load_config
from anomaly_detector.features import check_label_leakage
from anomaly_detector.pipeline import run_pipeline, export_result
from anomaly_detector.synthetic_data import SyntheticDataGenerator


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic data for the ML anomaly detector")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--window-size", type=int, default=None, help="Window size in minutes (overrides config)")
    parser.add_argument("--output-dir", default=None, help="Synthetic events output directory")
    parser.add_argument("--processed-dir", default=None, help="Where to write features/labels")
    parser.add_argument("--artifacts-dir", default=None, help="Where to write the report")
    parser.add_argument("--allow-empty-windows", action="store_true", help="Materialize empty windows")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.random_seed
    window_size = args.window_size if args.window_size is not None else cfg.window_size_minutes
    synthetic_dir = args.output_dir or cfg.synthetic_dir
    processed_dir = args.processed_dir or cfg.output_dir
    artifacts_dir = args.artifacts_dir or cfg.artifacts_dir
    allow_empty = args.allow_empty_windows or cfg.allow_empty_windows

    for d in (synthetic_dir, processed_dir, artifacts_dir):
        os.makedirs(d, exist_ok=True)

    print(f"Generating synthetic data (seed={seed}, window={window_size}m)")
    gen = SyntheticDataGenerator(seed=seed)
    data = gen.generate_full_synthetic_dataset()

    events_dict = [e.to_dict() for e in data["events"]]
    with open(os.path.join(synthetic_dir, "synthetic_events.json"), "w", encoding="utf-8") as fh:
        json.dump(events_dict, fh, indent=2)
    with open(os.path.join(synthetic_dir, "synthetic_labels.json"), "w", encoding="utf-8") as fh:
        json.dump([l.to_dict() for l in data["labels"]], fh, indent=2)

    result = run_pipeline(
        events_dict,
        window_size_minutes=window_size,
        labels=data["labels"],
        allow_empty_windows=allow_empty,
    )

    # Guard: fail loudly if labels ever leak into the feature matrix.
    leaks = check_label_leakage(result.feature_df)
    if leaks:
        print(f"ERROR: label leakage detected in feature columns: {leaks}", file=sys.stderr)
        return 2

    paths = export_result(result, processed_dir)
    with open(os.path.join(artifacts_dir, "data_quality_report.json"), "w", encoding="utf-8") as fh:
        json.dump(result.report, fh, indent=2)

    print(f"  processed input events : {result.report['input']['total_processed']}")
    print(f"  accepted events        : {result.report['input']['accepted_count']}")
    print(f"  rejected events        : {result.report['input']['rejected_count']}")
    print(f"  feature windows        : {result.report['windows']['count']}")
    print(f"  labels (separate)      : {result.report['labels']['count']} "
          f"(anomaly={result.report['labels']['anomaly_count']})")
    print(f"  features  -> {paths['features']}")
    print(f"  metadata  -> {paths['metadata']}")
    print(f"  labels    -> {paths['labels']}")
    print(f"  report    -> {os.path.join(artifacts_dir, 'data_quality_report.json')}")
    print(f"  scenarios : {', '.join(data['scenarios'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
