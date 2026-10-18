#!/usr/bin/env python3
"""Convert a public dataset into the common event format using an adapter.

This script never downloads data. If no dataset is available it records the
documented blocker and the adapter requirements. When a supported dataset is
available locally, it converts it and runs the same Milestone 1 pipeline used
for synthetic data.

Examples
--------
    # Record the adapter decision / blocker (no dataset needed)
    python scripts/prepare_dataset.py

    # Convert a local Loghub HDFS directory (HDFS.log [+ anomaly_label.csv])
    python scripts/prepare_dataset.py --dataset loghub_hdfs --path /data/HDFS
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from anomaly_detector.config import load_config
from anomaly_detector.pipeline import run_pipeline, export_result
from anomaly_detector.dataset_adapters import LoghubHDFSAdapter

ADAPTERS = {
    "loghub_hdfs": LoghubHDFSAdapter,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a public dataset via an adapter")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default=None, help="Adapter name (e.g. loghub_hdfs)")
    parser.add_argument("--path", default=None, help="Path to the dataset file or directory")
    parser.add_argument("--window-size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = cfg.artifacts_dir
    os.makedirs(artifacts_dir, exist_ok=True)
    status_path = os.path.join(artifacts_dir, "dataset_adapter_status.json")

    if not args.dataset:
        status = {
            "status": "no_dataset_selected",
            "available_adapters": sorted(ADAPTERS.keys()),
            "adapter_interface": "src/anomaly_detector/dataset_adapters/base.py",
            "note": (
                "No public dataset is bundled with this milestone. Provide --dataset "
                "and --path to convert a locally available dataset. Downloads must be "
                "obtained separately so the milestone never depends on network access."
            ),
            "requirements": [
                "Verify the license/usage restrictions at the official source.",
                "Confirm the field mapping into the common event format.",
                "Document label semantics (block-level vs window-level).",
                "Keep dataset files out of version control.",
            ],
            "candidates": [
                {"name": "Loghub HDFS", "adapter": "loghub_hdfs",
                 "source": "https://github.com/logpai/loghub"},
                {"name": "Loghub BGL", "adapter": "not_implemented",
                 "note": "Timestamps present; labeled; not yet adapted."},
                {"name": "Thunderbird", "source": "https://github.com/logpai/loghub"},
            ],
        }
        with open(status_path, "w", encoding="utf-8") as fh:
            json.dump(status, fh, indent=2)
        print("No dataset selected. Documented adapter path and blockers:")
        print(f"  -> {status_path}")
        return 0

    adapter_cls = ADAPTERS.get(args.dataset)
    if adapter_cls is None:
        print(f"Unknown adapter '{args.dataset}'. Available: {sorted(ADAPTERS)}", file=sys.stderr)
        return 2

    adapter = adapter_cls()
    if not adapter.can_load(args.path):
        status = {
            "status": "dataset_unavailable",
            "dataset": args.dataset,
            "path": args.path,
            "blocker": "adapter.can_load() returned False (file missing or unexpected format)",
            "info": adapter.get_dataset_info(),
        }
        with open(status_path, "w", encoding="utf-8") as fh:
            json.dump(status, fh, indent=2)
        print(f"Could not load dataset at {args.path!r}.", file=sys.stderr)
        print(f"Recorded blocker -> {status_path}")
        return 1

    events, labels = adapter.load_data(args.path)
    result = run_pipeline(
        [e.to_dict() for e in events],
        window_size_minutes=args.window_size or cfg.window_size_minutes,
        labels=labels,
        allow_empty_windows=cfg.allow_empty_windows,
    )
    paths = export_result(result, cfg.output_dir, prefix=args.dataset)
    status = {
        "status": "converted",
        "dataset": args.dataset,
        "info": adapter.get_dataset_info(),
        "events_loaded": len(events),
        "labels_loaded": len(labels),
        "windows": len(result.windows),
        "paths": paths,
        "note": "Block/instance-level labels require careful projection onto windows.",
    }
    with open(status_path, "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2)
    print(f"Converted {len(events)} events, {len(labels)} labels, {len(result.windows)} windows.")
    print(f"  -> {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
