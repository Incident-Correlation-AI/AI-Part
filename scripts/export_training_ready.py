#!/usr/bin/env python3
"""Export the alignable, leakage-audited training-ready dataset (Milestone 2).

Writes ``features.csv`` (eligible features only), ``labels.csv``,
``metadata.csv``, ``sample_index.csv`` and ``manifest.json`` to the output
directory. The export refuses to run if the leakage audit finds errors.

Example
-------
    python scripts/export_training_ready.py --out-dir data/processed/training_ready
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from anomaly_detector.dataset import align_dataset, load_table
from anomaly_detector.readiness_config import load_readiness_config
from anomaly_detector.training_ready import build_training_ready, write_training_ready


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the training-ready dataset")
    parser.add_argument("--config", default="configs/data_readiness.yaml")
    parser.add_argument("--features", default="data/processed/features.csv")
    parser.add_argument("--metadata", default="data/processed/metadata.csv")
    parser.add_argument("--labels", default="data/processed/labels.csv")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--allow-leakage", action="store_true",
                        help="Export even if the leakage audit reports errors (not recommended)")
    args = parser.parse_args()

    cfg = load_readiness_config(args.config)
    out_dir = args.out_dir or cfg.paths["training_ready_dir"]

    features = load_table(args.features)
    metadata = load_table(args.metadata)
    labels = load_table(args.labels) if args.labels and os.path.isfile(args.labels) else None

    aligned = align_dataset(features, metadata, labels, key_columns=cfg.data_policy.sample_key_columns)
    for issue in aligned.issues:
        print(f"[align] {issue}")

    dataset = build_training_ready(
        aligned, cfg,
        strict_leakage=not args.allow_leakage,
        source_paths={"features": args.features, "metadata": args.metadata, "labels": args.labels},
    )
    manifest_path = write_training_ready(
        dataset, out_dir, cfg.data_policy.sample_key_columns, cfg.split.label_column
    )
    manifest = json.load(open(manifest_path, "r", encoding="utf-8"))

    print(f"Training-ready rows: {dataset.row_count}")
    print(f"Feature columns ({len(dataset.features.columns)}): {', '.join(dataset.features.columns)}")
    print(f"Labels: {manifest['label_distribution']}")
    print(f"Manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
