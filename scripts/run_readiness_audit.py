#!/usr/bin/env python3
"""Milestone 7 — ML readiness audit and end-to-end local validation.

Runs, in order:
  Phase 1  validation-only model-selection comparison (IF vs LOF, same rows)
  Phase 2  threshold / abstention audit (validation only)
  Phase 3  end-to-end pipeline through the real M1 -> M6 path
  Phase 4  JSON + Markdown reports

Test metrics are computed only AFTER the selection decision is locked.
Existing M1-M6 artifacts and reports are never overwritten; this writes to
its own run directory under artifacts/readiness/.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from anomaly_detector.comparison_config import load_comparison_config
from anomaly_detector.e2e_runner import run_end_to_end
from anomaly_detector.inference import load_detector
from anomaly_detector.readiness_audit import (
    INCUMBENT,
    audit_threshold_policies,
    build_evidence,
    build_readiness_report,
    run_fair_validation_comparison,
    save_readiness_artifacts,
)
from anomaly_detector.selection_policy import SelectionPolicy


#: Pre-declared threshold policies evaluated on VALIDATION data only.
THRESHOLD_POLICIES = [
    {"name": "incumbent_validation_f1", "kind": "fixed", "value": -0.6202666407504878},
    {"name": "train_quantile_0.01", "kind": "train_quantile", "value": 0.01},
    {"name": "train_quantile_0.02", "kind": "train_quantile", "value": 0.02},
    {"name": "train_quantile_0.05", "kind": "train_quantile", "value": 0.05},
]


def main() -> int:
    parser = argparse.ArgumentParser(description="M7 ML readiness audit")
    parser.add_argument("--config", default="configs/comparison.yaml")
    parser.add_argument("--selection-policy", default="configs/selection.yaml")
    parser.add_argument("--out-root", default="artifacts/readiness")
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(args.out_root, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    cfg = load_comparison_config(args.config)
    policy = SelectionPolicy.load(args.selection_policy)

    print("=" * 72)
    print("Milestone 7 — ML readiness audit")
    print("=" * 72)
    print(f"run dir           : {run_dir}")
    print(f"selection policy  : {policy.name}")
    print(f"eligible          : {policy.eligible_candidates}")
    print()

    # -- Phase 1: validation-only comparison ----------------------------
    t0 = time.time()
    runs, provenance = run_fair_validation_comparison(cfg)
    print(f"[phase 1] fair validation comparison: {len(runs)} candidate(s), "
          f"{provenance['training_rows_used']}/{provenance['training_rows_available']} training rows used")
    for run in runs:
        m = run.validation_metrics.get("metrics", {})
        print(f"           {run.name:34s} val f1={_f(m.get('f1'))} prec={_f(m.get('precision'))} "
              f"recall={_f(m.get('recall'))} thr={run.threshold:.6f}")

    # -- Phase 1b: robustness evidence (M5 scenarios, validation-only models)
    from anomaly_detector.readiness_audit import collect_robustness_evidence
    robustness_summary = collect_robustness_evidence(cfg, runs)
    print(f"[phase 1] robustness floors: "
          f"{ {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in robustness_summary.items()} }")

    # -- Phase 1c: the selection decision (validation only) --------------
    evidence = build_evidence(runs, robustness_summary)
    decision = policy.decide(evidence)
    print(f"[phase 1] SELECTED: {decision.selected}")
    print(f"           {decision.reason}")
    print(f"           test metrics used for selection: {decision.used_test_metrics}")
    print()

    # -- Phase 2: threshold / abstention audit (validation only) ---------
    incumbent_run = next((r for r in runs if r.name == INCUMBENT), runs[0])
    from anomaly_detector.baseline_pipeline import load_split_tables
    from anomaly_detector.baseline_config import load_baseline_config

    bcfg = load_baseline_config("configs/baseline.yaml")
    bcfg.data.feature_columns = list(cfg.data["feature_columns"])
    bcfg.paths["splits_dir"] = cfg.data.get("splits_dir", "data/processed/splits")
    val_tables = load_split_tables(bcfg, "validation")
    y_val = val_tables.labels[cfg.data.get("label_column", "is_anomaly")]

    threshold_audit = audit_threshold_policies(incumbent_run, y_val, THRESHOLD_POLICIES)
    print("[phase 2] threshold policies (validation only):")
    for entry in threshold_audit["policies"]:
        if "error" in entry:
            print(f"           {entry['policy']:24s} ERROR {entry['error']}")
        else:
            print(f"           {entry['policy']:24s} thr={entry['threshold']:.6f} "
                  f"tp={entry.get('tp')} fp={entry.get('fp')} f1={_f(entry.get('f1'))} "
                  f"fpr={_f(entry.get('false_positive_rate'))}")
    print()

    # -- Phase 3: end-to-end pipeline -----------------------------------
    t1 = time.time()
    detector = load_detector()
    e2e = run_end_to_end(detector)
    e2e_seconds = time.time() - t1
    print("[phase 3] end-to-end scenarios:")
    for row in e2e["scenarios"]:
        print(f"           {row['scenario']:28s} n={row['n_windows']:3d} flagged={row['flagged']:3d} "
              f"detected={row['detected']} missed={row['missed']} fp={row['false_positives']} "
              f"unscored={row['unscored']}")
    print(f"           benign flag rate: {_f(e2e['benign_flag_rate'])}")
    print()

    # -- Phase 4: reports -----------------------------------------------
    abstention = _abstention_findings(e2e, threshold_audit)
    runtime = {
        "phase_1_comparison_seconds": round(time.time() - t0, 3),
        "phase_3_end_to_end_seconds": round(e2e_seconds, 3),
        "windows_scored_end_to_end": e2e["totals"]["windows"],
        "note": (
            "Wall-clock observations on the local development machine "
            "(8 GB RAM / 4 logical CPU threads). These are measurements, not SLAs. "
            "No large-scale extrapolation is made."
        ),
    }
    root_cause = _root_cause()

    report = build_readiness_report(
        cfg, decision, runs, provenance, threshold_audit, e2e, abstention, runtime, root_cause
    )
    paths = save_readiness_artifacts(report, run_dir)

    print("=" * 72)
    print(f"Selected model : {decision.selected}")
    print(f"Readiness      : {report['readiness']['verdict']}")
    print(f"  {report['readiness']['rationale']}")
    for caution in report["readiness"]["cautions"]:
        print(f"  - {caution}")
    print(f"Report (json)  : {paths['json']}")
    print(f"Report (md)    : {paths['markdown']}")
    print("=" * 72)
    return 0


def _f(value):
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _abstention_findings(e2e, threshold_audit):
    """Findings on the abstention / missingness behaviour."""
    shift = e2e.get("benign_shift_flag_rate")
    missing = e2e.get("missing_telemetry_flag_rate")
    findings = {
        "imputation_policy": (
            "median_impute with training-split medians (the trained model's own policy). "
            "A missing measurement is never replaced by zero."
        ),
        "imputed_windows_end_to_end": e2e["totals"]["imputed_windows"],
        "unscored_windows_end_to_end": e2e["totals"]["unscored"],
        "benign_shift_flag_rate": (
            f"{shift:.3f}" if shift is not None else "n/a"
        ),
        "missing_telemetry_flag_rate": (
            f"{missing:.3f}" if missing is not None else "n/a"
        ),
        "max_imputed_fraction_default": (
            "NOT SET. The available validation evidence is too thin (45 windows / 11 "
            "positives) to justify an abstention threshold; the option remains available "
            "to callers via DetectorOptions(max_imputed_fraction=...)."
        ),
        "stale_window_default": (
            "NOT SET. Temporal freshness is a caller-owned policy; the option remains "
            "available via DetectorOptions(stale_after_seconds=...)."
        ),
        "status_distinction_preserved": (
            "scored / insufficient_data / invalid_input / model_unavailable / "
            "inference_error remain distinct; no automatic retraining or threshold "
            "adjustment is performed."
        ),
    }
    return findings


def _root_cause():
    """Root cause and resolution of the M4 discrepancy."""
    return {
        "symptom": (
            "M4's machine-generated report recorded `recommended: lof_novelty` while its "
            "written report recommended retaining the M3 Isolation Forest."
        ),
        "root_cause_1": (
            "src/anomaly_detector/comparison.py :: build_comparison_report() keyed the "
            "winner on the TEST metrics (test_f1, test_precision, test_fp, test_recall), "
            "directly contradicting the pre-declared rule in configs/comparison.yaml "
            "('evaluate test once after selection is fixed') and the rationale string the "
            "same function emitted. The pre-declared weights were never applied and the "
            "declared 'retain the baseline unless clearly outperformed' rule was never "
            "implemented."
        ),
        "root_cause_2": (
            "Candidates B (MAD) and C (LOF) were fitted on ALL training rows, including "
            "the 26 labelled anomalies, because select_training_rows() was imported but "
            "never called. The Isolation Forest incumbent was fitted on the 110 normal-only "
            "rows. The comparison was therefore unfair and violated the declared "
            "training_data_policy: normal_only."
        ),
        "resolution_1": (
            "The M4 auto-picker now selects on VALIDATION metrics only, applies the "
            "pre-declared weights, and prefers the incumbent. The report carries "
            "`selection_evidence_split: validation` and `used_test_metrics_for_selection: "
            "false`."
        ),
        "resolution_2": (
            "MAD and LOF now apply the same normal_only training-row selection as the "
            "incumbent, with the mask re-indexed through the prepared matrix so a "
            "drop-policy cannot misalign rows."
        ),
        "resolution_3": (
            "A dedicated pre-declared selection policy (configs/selection.yaml, "
            "src/anomaly_detector/selection_policy.py) decides between the two "
            "candidates using validation evidence only, with a required improvement "
            "margin and robustness floors. The decision is structural: the selection "
            "evidence type rejects any test metric."
        ),
        "historical_artifacts": (
            "The original M4 run directory and its report are preserved unchanged; this "
            "audit writes to its own run directory under artifacts/readiness/."
        ),
        "model_switched": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())
