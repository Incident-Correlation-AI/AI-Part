#!/usr/bin/env python3
"""Create reproducible train/validation/test splits (Milestone 2).

Supports ``random``, ``time`` and ``group`` strategies. Writes per-split
feature/label/metadata tables plus a ``split_manifest.json``. The same config
and seed always reproduce identical assignments.

Example
-------
    python scripts/make_splits.py --strategy group --group-column scenario_name
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from anomaly_detector.dataset import align_dataset, load_table
from anomaly_detector.readiness_config import load_readiness_config
from anomaly_detector.splits import make_splits
from anomaly_detector.training_ready import build_training_ready, write_split


def main() -> int:
    parser = argparse.ArgumentParser(description="Create reproducible dataset splits")
    parser.add_argument("--config", default="configs/data_readiness.yaml")
    parser.add_argument("--features", default="data/processed/features.csv")
    parser.add_argument("--metadata", default="data/processed/metadata.csv")
    parser.add_argument("--labels", default="data/processed/labels.csv")
    parser.add_argument("--strategy", default=None, choices=["random", "time", "group"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_readiness_config(args.config)
    if args.strategy:
        cfg.split.strategy = args.strategy
    if args.seed is not None:
        cfg.split.seed = args.seed
    if args.group_column:
        cfg.split.group_column = args.group_column
    if args.time_column:
        cfg.split.time_column = args.time_column

    out_dir = args.out_dir or os.path.join(cfg.paths["splits_dir"], cfg.split.strategy)

    features = load_table(args.features)
    metadata = load_table(args.metadata)
    labels = load_table(args.labels) if args.labels and os.path.isfile(args.labels) else None

    aligned = align_dataset(features, metadata, labels, key_columns=cfg.data_policy.sample_key_columns)
    for issue in aligned.issues:
        print(f"[align] {issue}")

    dataset = build_training_ready(aligned, cfg, strict_leakage=True,
                                   source_paths={"features": args.features})
    split = make_splits(aligned.metadata, cfg.split, aligned.labels)
    manifest_path = write_split(
        dataset, split, out_dir, cfg.data_policy.sample_key_columns, cfg.split.label_column
    )
    manifest = json.load(open(manifest_path, "r", encoding="utf-8"))

    print(f"Strategy: {split.strategy}  seed: {split.seed}")
    for name in split.split_order:
        dist = manifest["splits"][name].get("label_distribution", {})
        print(f"  {name:11s} rows={manifest['splits'][name]['rows']:5d} "
              f"positives={dist.get('positive', 'n/a')}")
    for w in split.warnings:
        print(f"  [warn] {w}")
    print(f"Manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
