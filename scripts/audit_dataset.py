#!/usr/bin/env python3
"""Audit the processed dataset for quality and leakage (Milestone 2).

Reads the Milestone 1 exports (features/labels/metadata CSVs), aligns them on a
stable sample key, then writes a data-quality report and a leakage report (each
as JSON and Markdown) into the reports directory.

The source tables are never modified. The script exits non-zero when the audit
finds errors or the leakage audit is not clean, so it can gate a build.

Example
-------
    python scripts/audit_dataset.py --reports-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from anomaly_detector.audit import audit_dataset, write_audit
from anomaly_detector.dataset import align_dataset, load_table
from anomaly_detector.leakage import audit_leakage, write_leakage_report
from anomaly_detector.readiness_config import load_readiness_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the processed dataset (quality + leakage)")
    parser.add_argument("--config", default="configs/data_readiness.yaml")
    parser.add_argument("--features", default="data/processed/features.csv")
    parser.add_argument("--metadata", default="data/processed/metadata.csv")
    parser.add_argument("--labels", default="data/processed/labels.csv")
    parser.add_argument("--reports-dir", default=None)
    args = parser.parse_args()

    cfg = load_readiness_config(args.config)
    reports_dir = args.reports_dir or cfg.paths["reports_dir"]

    features = load_table(args.features)
    metadata = load_table(args.metadata)
    labels = load_table(args.labels) if args.labels and os.path.isfile(args.labels) else None

    aligned = align_dataset(features, metadata, labels, key_columns=cfg.data_policy.sample_key_columns)
    for issue in aligned.issues:
        print(f"[align] {issue}")

    audit = audit_dataset(aligned, cfg, source_paths={
        "features": args.features,
        "metadata": args.metadata,
        "labels": args.labels if labels is not None else "",
    })
    leakage = audit_leakage(aligned, cfg)

    q_json, q_md = write_audit(audit, reports_dir)
    l_json, l_md = write_leakage_report(leakage, reports_dir)

    print(f"Rows: {audit.inventory['feature_rows']}  "
          f"features: {audit.inventory['eligible_feature_count']} eligible")
    print(f"Audit    errors={audit.error_count} warnings={audit.warning_count} "
          f"ready={audit.is_ready}")
    print(f"Leakage  errors={leakage.error_count} clean={leakage.is_clean}")
    print(f"Quality  -> {q_json}")
    print(f"         -> {q_md}")
    print(f"Leakage  -> {l_json}")
    print(f"         -> {l_md}")

    if not audit.is_ready:
        print("ERROR: dataset audit reported errors", file=sys.stderr)
        return 2
    if not leakage.is_clean:
        print("ERROR: leakage audit reported errors", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
