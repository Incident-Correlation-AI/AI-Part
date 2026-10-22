#!/usr/bin/env python3
"""Milestone 8 — local integration rehearsal.

Runs the real M1 -> M6 path over the M7 scenarios and reports the boundary
routing behaviour separately from the ML detection outcomes.

Offline: no network, no AWS, no downstream service. The real downstream
interface is still pending, so a neutral in-memory sink stands in for it.

    python scripts/run_integration_rehearsal.py
    python scripts/run_integration_rehearsal.py --out-root artifacts/rehearsal
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from anomaly_detector.integration_harness import IntegrationHarness, run_rehearsal


def main() -> int:
    parser = argparse.ArgumentParser(description="M8 local integration rehearsal")
    parser.add_argument("--model", default="artifacts/baseline/model/model.joblib")
    parser.add_argument("--manifest", default="artifacts/baseline/model/manifest.json")
    parser.add_argument("--out-root", default="artifacts/rehearsal")
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(args.out_root, f"run_{run_id}")
    reports_dir = os.path.join(run_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 78)
    print("Milestone 8 — local integration rehearsal")
    print("=" * 78)

    harness = IntegrationHarness(
        model_path=args.model, manifest_path=args.manifest
    )
    contract = harness.contract()
    print(f"model              : {contract['model_name']} (version {contract['model_version']})")
    print(f"feature schema     : {contract['feature_schema_version']}")
    print(f"score direction    : {contract['score_direction']} (not a probability)")
    print(f"threshold          : {contract['threshold']}  [{contract['threshold_rule']}]")
    print()

    report = run_rehearsal(harness)

    print("--- routing (contract behaviour) ---")
    routing = report["routing"]
    print(f"  canonical rule        : {routing['canonical_rule']}")
    print(f"  forwardable status    : {routing['forwardable_status']}")
    print(f"  forwarded statuses    : {routing['forwarded_statuses']}")
    print(f"  every forwarded item scored : {routing['every_forwarded_item_was_scored']}")
    print(f"  any unscored reached the sink: {routing['any_unscored_item_reached_the_sink']}")
    print(f"  all scenarios routing ok    : {routing['all_scenarios_routing_ok']}")
    print()

    print("--- per scenario ---")
    header = (f"  {'scenario':30s} {'win':>4s} {'scored':>6s} {'fwd':>4s} "
              f"{'unsc':>5s} {'flag':>5s} {'det':>4s} {'miss':>5s} {'fp':>3s}  routing")
    print(header)
    for row in report["scenarios"]:
        print(f"  {row['scenario']:30s} {row['n_windows']:4d} {row['scored']:6d} "
              f"{row['forwarded']:4d} {row['unscored']:5d} {row['flagged']:5d} "
              f"{row['detected']:4d} {row['missed']:5d} {row['false_positives']:3d}  "
              f"{'ok' if row['routing_ok'] else 'FAILED'}")
    print()

    totals = report["totals"]
    print("--- totals ---")
    for key in ("events_attempted", "events_accepted", "events_rejected",
                "events_duplicated", "windows_attempted", "scored", "forwarded",
                "unscored", "sink_failures", "flagged", "detected", "missed",
                "false_positives", "duplicate_window_ids"):
        print(f"  {key:24s} {totals[key]}")
    print()

    detection = report["ml_detection"]
    print("--- ML detection (synthetic-only) ---")
    print(f"  detected {detection['detected']} · missed {detection['missed']} · "
          f"false positives {detection['false_positives']}")
    print("  a scenario is never called detected merely because the harness ran")
    print()

    json_path = os.path.join(reports_dir, "rehearsal_report.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    md_path = os.path.join(reports_dir, "rehearsal_report.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(_render(report))

    manifest_path = os.path.join(run_dir, "rehearsal_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({
            "milestone": 8,
            "run_dir": run_dir,
            "generated_at": report.get("generated_at", run_id),
            "routing_rule": report["routing"]["canonical_rule"],
            "every_forwarded_item_was_scored": report["routing"]["every_forwarded_item_was_scored"],
            "any_unscored_item_reached_the_sink": report["routing"]["any_unscored_item_reached_the_sink"],
            "downstream_interface": "pending - a neutral in-memory sink stands in",
        }, handle, indent=2)

    print(f"Report (json) : {json_path}")
    print(f"Report (md)   : {md_path}")
    return 0


def _render(report) -> str:
    lines = ["# Milestone 8 — Local Integration Rehearsal", ""]
    harness = report["harness"]
    lines.append(f"- Model: `{harness['model_name']}` (version `{harness['model_version']}`)")
    lines.append(f"- Feature schema: `{harness['feature_schema_version']}`")
    lines.append(f"- Contract version: `{harness['contract_version']}`")
    lines.append(f"- Score direction: `{harness['score_direction']}` (not a probability)")
    lines.append(f"- Threshold: `{harness['threshold']}` ({harness['threshold_rule']})")
    lines.append("")
    routing = report["routing"]
    lines.append("## Routing (contract behaviour)")
    lines.append("")
    lines.append(f"- Canonical rule: `{routing['canonical_rule']}`")
    lines.append(f"- Forwardable status: `{routing['forwardable_status']}`")
    lines.append(f"- Forwarded statuses: `{routing['forwarded_statuses']}`")
    lines.append(f"- Every forwarded item had status `scored`: "
                 f"**{routing['every_forwarded_item_was_scored']}**")
    lines.append(f"- Any unscored item reached the correlation sink: "
                 f"**{routing['any_unscored_item_reached_the_sink']}**")
    lines.append("")
    lines.append("## Per-scenario results")
    lines.append("")
    lines.append("| Scenario | Windows | Scored | Forwarded | Unscored | Flagged | Detected | Missed | False positives | Routing |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in report["scenarios"]:
        lines.append(
            f"| {row['scenario']} | {row['n_windows']} | {row['scored']} | {row['forwarded']} | "
            f"{row['unscored']} | {row['flagged']} | {row['detected']} | {row['missed']} | "
            f"{row['false_positives']} | {'ok' if row['routing_ok'] else 'FAILED'} |"
        )
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    for key, value in report["totals"].items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- {report['contract_routing']['note']}")
    lines.append(f"- {report['ml_detection']['note']}")
    lines.append("- The real downstream interface is pending; a neutral in-memory sink stands in.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
