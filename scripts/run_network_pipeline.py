#!/usr/bin/env python3
"""Train, score and evaluate the network anomaly detector.

Runs the complete local pipeline: simulator -> LAD envelope parsing ->
``network-features-v1`` windows -> time/group splits -> Isolation Forest +
transparent rule baseline -> evaluation -> artifacts.

The test split is never used for model fitting or threshold selection.

Examples
--------
    python scripts/run_network_pipeline.py
    python scripts/run_network_pipeline.py --config configs/network.yaml
    python scripts/run_network_pipeline.py --strategies time
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from anomaly_detector.network.config import load_network_config
from anomaly_detector.network.pipeline import run_network_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the network anomaly-detection pipeline")
    parser.add_argument("--config", default="configs/network.yaml", help="Path to YAML config")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Split strategies to run (default: config value, e.g. 'time group')",
    )
    args = parser.parse_args()

    config = load_network_config(args.config)
    result = run_network_pipeline(config, strategies=args.strategies)

    summary = {
        "events": result.dataset.event_count,
        "windows": len(result.windows),
        "labels": len(result.labels),
        "primary_strategy": result.primary_strategy,
        "strategies": {},
        "artifacts": result.shared_paths,
    }
    for name, run in result.strategies.items():
        splits = {}
        for split, payload in run.metrics.get("splits", {}).items():
            overall = payload.get("overall") or {}
            splits[split] = {
                "n_windows": payload.get("n_windows"),
                "n_scored": payload.get("n_scored"),
                "n_unscored": payload.get("n_unscored"),
                "precision": overall.get("precision"),
                "recall": overall.get("recall"),
                "f1": overall.get("f1"),
            }
        summary["strategies"][name] = {
            "threshold": run.threshold.to_dict(),
            "splits": splits,
            "model": run.paths.get("model"),
            "metrics": run.paths.get("metrics_json"),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
