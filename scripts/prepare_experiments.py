#!/usr/bin/env python3
"""One-shot Milestone 2 experiment prep: audit, export, and split.

Runs the full readiness flow on the processed dataset:

1. Data-quality + leakage audit  -> ``reports/``
2. Training-ready export         -> ``data/processed/training_ready/``
3. Random, time and group splits -> ``data/processed/splits/<strategy>/``

Aborts (non-zero) if the audit reports errors or the leakage audit is not clean,
so it is safe to wire into CI.

Example
-------
    python scripts/prepare_experiments.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from anomaly_detector.audit import audit_dataset, write_audit
from anomaly_detector.dataset import align_dataset, load_table
from anomaly_detector.leakage import audit_leakage, write_leakage_report
from anomaly_detector.readiness_config import load_readiness_config
from anomaly_detector.splits import make_splits
from anomaly_detector.training_ready import (
    build_training_ready,
    write_split,
    write_training_ready,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full Milestone 2 experiment prep")
    parser.add_argument("--config", default="configs/data_readiness.yaml")
    parser.add_argument("--features", default="data/processed/features.csv")
    parser.add_argument("--metadata", default="data/processed/metadata.csv")
    parser.add_argument("--labels", default="data/processed/labels.csv")
    parser.add_argument("--strategies", nargs="*", default=["random", "time", "group"])
    args = parser.parse_args()

    cfg = load_readiness_config(args.config)
    reports_dir = cfg.paths["reports_dir"]

    features = load_table(args.features)
    metadata = load_table(args.metadata)
    labels = load_table(args.labels) if args.labels and os.path.isfile(args.labels) else None
    aligned = align_dataset(features, metadata, labels,
                            key_columns=cfg.data_policy.sample_key_columns)

    # 1) Audit.
    audit = audit_dataset(aligned, cfg, source_paths={
        "features": args.features, "metadata": args.metadata, "labels": args.labels})
    leakage = audit_leakage(aligned, cfg)
    write_audit(audit, reports_dir)
    write_leakage_report(leakage, reports_dir)
    print(f"Audit   ready={audit.is_ready} errors={audit.error_count} warnings={audit.warning_count}")
    print(f"Leakage clean={leakage.is_clean} errors={leakage.error_count}")
    if not audit.is_ready or not leakage.is_clean:
        print("ERROR: refusing to export/split until audit errors are resolved", file=sys.stderr)
        return 2

    # 2) Training-ready export.
    dataset = build_training_ready(aligned, cfg, strict_leakage=True,
                                   source_paths={"features": args.features})
    manifest_path = write_training_ready(
        dataset, cfg.paths["training_ready_dir"],
        cfg.data_policy.sample_key_columns, cfg.split.label_column,
    )
    print(f"Training-ready rows={dataset.row_count} -> {os.path.dirname(manifest_path)}")

    # 3) Splits.
    for strategy in args.strategies:
        cfg.split.strategy = strategy
        split = make_splits(aligned.metadata, cfg.split, aligned.labels)
        out_dir = os.path.join(cfg.paths["splits_dir"], strategy)
        write_split(dataset, split, out_dir,
                    cfg.data_policy.sample_key_columns, cfg.split.label_column)
        sizes = {n: split.size(n) for n in split.split_order}
        print(f"Split   {strategy}: {sizes} -> {out_dir}")
        for w in split.warnings:
            print(f"        [warn] {w}")

    print(f"Reports -> {reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
