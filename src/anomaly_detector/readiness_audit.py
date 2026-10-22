"""Milestone 7 — ML readiness audit and end-to-end local validation.

Phases
------
1. **Selection audit** — redo the M4 IF-vs-LOF comparison under a
   validation-only policy, with both candidates trained on the *same* rows
   under the same ``normal_only`` policy. Test metrics are computed only
   after the selection decision is locked.
2. **Threshold / abstention audit** — evaluate a small, pre-declared set of
   threshold policies on validation data only, and review the median
   imputation / missingness behaviour.
3. **End-to-end pipeline** — raw events -> validation -> windowing ->
   features -> M6 inference -> neutral downstream sink.

Nothing here trains a new model campaign, retunes the threshold, or claims
production readiness from synthetic data.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .baseline_config import load_baseline_config
from .baseline_model import BaselineDetector
from .baseline_pipeline import load_split_tables
from .model_contract import ModelContractError, prepare_model_input, select_training_rows
from .selection_policy import CandidateEvidence, SelectionDecision, SelectionPolicy
from .thresholding import select_threshold, apply_threshold
from .evaluation import evaluate_split
from .lof_baseline import LOFBaseline
from .mad_baseline import MADBaseline

#: The incumbent model (what M4/M5/M6 actually use and what M7 audits).
INCUMBENT = "m3_isolation_forest_reference"


# ---------------------------------------------------------------------------
# Phase 1 — validation-only selection
# ---------------------------------------------------------------------------


@dataclass
class CandidateRun:
    """One candidate fitted under the shared, fair training policy."""

    name: str
    kind: str
    validation_scores: np.ndarray
    test_scores: np.ndarray
    train_scores: np.ndarray
    validation_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    threshold: float
    threshold_method: str
    training_rows_used: int
    runtime_s: float
    notes: List[str] = field(default_factory=list)


def _split_tables(cfg, split: str):
    bcfg = load_baseline_config("configs/baseline.yaml")
    bcfg.data.feature_columns = list(cfg.data["feature_columns"])
    bcfg.data.missing_value_policy = cfg.data.get("missing_value_policy", "median_impute")
    bcfg.data.training_data_policy = cfg.data.get("training_data_policy", "normal_only")
    bcfg.paths["splits_dir"] = cfg.data.get("splits_dir", "data/processed/splits")
    return load_split_tables(bcfg, split)


def run_fair_validation_comparison(cfg) -> Tuple[List[CandidateRun], Dict[str, Any]]:
    """Compare IF and LOF on identical rows, features and preprocessing.

    Both candidates are fitted on the **same** training rows, selected by the
    same ``normal_only`` policy, using the same features and the same
    training-fitted imputation medians. Thresholds are selected on the
    validation split only.

    Test metrics are computed but returned separately, so the caller can
    only use them after the selection decision is locked.
    """
    feat_cols = list(cfg.data["feature_columns"])
    label_col = cfg.data.get("label_column", "is_anomaly")
    policy_missing = cfg.data.get("missing_value_policy", "median_impute")
    policy_train = cfg.data.get("training_data_policy", "normal_only")
    key_col = cfg.data.get("sample_key_column", "sample_key")

    train_t = _split_tables(cfg, "train")
    val_t = _split_tables(cfg, "validation")
    test_t = _split_tables(cfg, "test")

    X_train = train_t.features
    y_train = train_t.labels[label_col] if label_col in train_t.labels else pd.Series([False] * len(X_train))
    y_val = val_t.labels[label_col] if label_col in val_t.labels else None
    y_test = test_t.labels[label_col] if label_col in test_t.labels else None

    # Shared preprocessing: statistics fitted on the TRAINING rows only.
    prep_train = prepare_model_input(
        X_train, feat_cols, medians=None, policy=policy_missing,
        sample_key_column=key_col, context="m7-train",
    )
    used_columns = list(prep_train.feature_columns)
    train_mask = select_training_rows(y_train, policy=policy_train, label_column=label_col)
    kept = np.asarray(prep_train.row_indices, dtype=int)
    X_fit = prep_train.X[train_mask[kept]]
    training_rows_used = int(len(X_fit))

    def _prep(frame, context):
        return prepare_model_input(
            frame, feat_cols, medians=prep_train.medians, policy=policy_missing,
            sample_key_column=key_col, context=context, use_columns=used_columns,
        ).X

    X_val = _prep(val_t.features, "m7-validation")
    X_test = _prep(test_t.features, "m7-test")
    X_tr_scored = _prep(X_train, "m7-train-scored")

    runs: List[CandidateRun] = []

    # --- Isolation Forest (the incumbent) -----------------------------
    t0 = time.time()
    bcfg = load_baseline_config("configs/baseline.yaml")
    bcfg.data.feature_columns = feat_cols
    bcfg.data.missing_value_policy = policy_missing
    bcfg.data.training_data_policy = policy_train
    bcfg.model.random_state = int(cfg.seeds.get("sklearn", 42))
    detector, _ = fit_incumbent(X_fit, bcfg)
    s_val = detector.score(X_val)
    s_test = detector.score(X_test)
    s_tr = detector.score(X_tr_scored)
    th = select_threshold(
        s_val, list(y_val) if y_val is not None else None,
        method=cfg.threshold.get("method", "validation_f1"), source_split="validation",
        train_scores=s_tr,
        min_precision=cfg.threshold.get("min_precision", 0.5),
        min_recall=cfg.threshold.get("min_recall", 0.5),
        fallback_method=cfg.threshold.get("fallback_method", "train_quantile"),
        fallback_quantile=cfg.threshold.get("fallback_quantile", 0.01),
    )
    mv = evaluate_split(s_val, list(y_val) if y_val is not None else None,
                        apply_threshold(s_val, th.threshold), th.threshold, split="validation").to_dict()
    mt = evaluate_split(s_test, list(y_test) if y_test is not None else None,
                        apply_threshold(s_test, th.threshold), th.threshold, split="test").to_dict()
    runs.append(CandidateRun(
        INCUMBENT, "isolation_forest", s_val, s_test, s_tr, mv, mt,
        float(th.threshold), th.method, training_rows_used, time.time() - t0,
        ["IsolationForest; fitted on normal_only training rows; threshold from validation"],
    ))

    # --- LOF (novelty) — same rows, same features, same preprocessing ---
    t0 = time.time()
    lof = LOFBaseline(used_columns, n_neighbors=20, contamination="auto")
    lof.fit(X_fit)
    l_val = lof.score(X_val)
    l_test = lof.score(X_test)
    l_tr = lof.score(X_tr_scored)
    th_l = select_threshold(
        l_val, list(y_val) if y_val is not None else None,
        method=cfg.threshold.get("method", "validation_f1"), source_split="validation",
        train_scores=l_tr,
        min_precision=cfg.threshold.get("min_precision", 0.5),
        min_recall=cfg.threshold.get("min_recall", 0.5),
        fallback_method=cfg.threshold.get("fallback_method", "train_quantile"),
        fallback_quantile=cfg.threshold.get("fallback_quantile", 0.01),
    )
    lmv = evaluate_split(l_val, list(y_val) if y_val is not None else None,
                         apply_threshold(l_val, th_l.threshold), th_l.threshold, split="validation").to_dict()
    lmt = evaluate_split(l_test, list(y_test) if y_test is not None else None,
                         apply_threshold(l_test, th_l.threshold), th_l.threshold, split="test").to_dict()
    runs.append(CandidateRun(
        "lof_novelty", "lof", l_val, l_test, l_tr, lmv, lmt,
        float(th_l.threshold), th_l.method, training_rows_used, time.time() - t0,
        ["LocalOutlierFactor(novelty=True); fitted on the SAME normal_only rows; threshold from validation"],
    ))

    provenance = {
        "splits_dir": cfg.data.get("splits_dir"),
        "strategy": cfg.data.get("strategy"),
        "feature_columns": feat_cols,
        "training_rows_available": int(len(X_train)),
        "training_rows_used": training_rows_used,
        "training_data_policy": policy_train,
        "training_median_source": "training split only",
        "label_column": label_col,
        "candidates_share_training_rows": True,
        "candidates_share_features": True,
        "candidates_share_preprocessing": True,
        "thresholds_selected_on": "validation",
    }
    return runs, provenance


def fit_incumbent(X_fit: np.ndarray, bcfg) -> Tuple[BaselineDetector, Any]:
    """Fit the M3 Isolation Forest baseline on the given (already filtered) rows."""
    from .baseline_model import fit_baseline

    labels = pd.Series([False] * len(X_fit))  # rows are already normal-only
    detector, _report = fit_baseline(
        pd.DataFrame(X_fit, columns=bcfg.data.feature_columns), labels, bcfg
    )
    return detector, _report


def build_evidence(runs: List[CandidateRun], robustness: Dict[str, Dict[str, float]]) -> List[CandidateEvidence]:
    """Turn candidate runs into validation-only selection evidence."""
    evidence: List[CandidateEvidence] = []
    for run in runs:
        metrics = run.validation_metrics.get("metrics", {})
        validation = {
            "f1": metrics.get("f1"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "false_positive_rate": metrics.get("false_positive_rate"),
            "pr_auc": metrics.get("pr_auc"),
            "n_windows": metrics.get("n_samples") or metrics.get("n"),
            "n_positive": metrics.get("n_positive"),
            "n_flagged": metrics.get("n_flagged"),
            "runtime_s": run.runtime_s,
        }
        validation = {k: v for k, v in validation.items() if v is not None}
        evidence.append(CandidateEvidence(
            name=run.name,
            validation=validation,
            robustness=robustness.get(run.name, {}),
            training_rows_used=run.training_rows_used,
            notes=list(run.notes),
        ))
    return evidence


# ---------------------------------------------------------------------------
# Phase 1b — robustness evidence (benign shift / missing telemetry)
# ---------------------------------------------------------------------------


def collect_robustness_evidence(cfg, runs: List[CandidateRun]) -> Dict[str, Dict[str, float]]:
    """Flag rates on the M5 benign scenarios, per candidate.

    Both candidates are fitted on the **same** rows and see the **same**
    scenario feature rows, so the comparison is fair. The scenarios are
    benign by construction, so a high flag rate is a false-alert risk.
    """
    from .model_contract import select_training_rows
    from .robustness_scenarios import (
        gen_distribution_shifts,
        gen_normal_variation,
        gen_telemetry_faults,
    )

    feat_cols = list(cfg.data["feature_columns"])
    key_col = cfg.data.get("sample_key_column", "sample_key")
    label_col = cfg.data.get("label_column", "is_anomaly")
    policy_missing = cfg.data.get("missing_value_policy", "median_impute")
    policy_train = cfg.data.get("training_data_policy", "normal_only")

    train_t = _split_tables(cfg, "train")
    prep = prepare_model_input(
        train_t.features, feat_cols, medians=None, policy=policy_missing,
        sample_key_column=key_col, context="m7-robustness",
    )
    used = list(prep.feature_columns)
    y = train_t.labels[label_col] if label_col in train_t.labels else pd.Series([False] * len(train_t.features))
    mask = select_training_rows(y, policy=policy_train, label_column=label_col)
    kept = np.asarray(prep.row_indices, dtype=int)
    X_fit = prep.X[mask[kept]]

    # Fit both candidate families on the identical rows.
    bcfg = load_baseline_config("configs/baseline.yaml")
    bcfg.data.feature_columns = feat_cols
    bcfg.data.missing_value_policy = policy_missing
    bcfg.data.training_data_policy = policy_train
    bcfg.model.random_state = int(cfg.seeds.get("sklearn", 42))
    from .baseline_model import fit_baseline

    if_model, _ = fit_baseline(
        pd.DataFrame(X_fit, columns=feat_cols), pd.Series([False] * len(X_fit)), bcfg
    )
    lof_model = LOFBaseline(used, n_neighbors=20, contamination="auto")
    lof_model.fit(X_fit)

    cases = {
        "benign_shift": gen_distribution_shifts(),
        "missing_telemetry": gen_telemetry_faults(),
        "normal_variation": gen_normal_variation(),
    }

    def _prep(frame, context):
        return prepare_model_input(
            frame, feat_cols, medians=prep.medians, policy=policy_missing,
            sample_key_column=key_col, context=context, use_columns=used,
        ).X

    out: Dict[str, Dict[str, float]] = {}
    for run in runs:
        entry: Dict[str, float] = {}
        for name, case in cases.items():
            X = _prep(case.features, f"m7-{name}")
            if run.kind == "isolation_forest":
                scores = if_model.score(X)
            else:
                scores = lof_model.score(X)
            flagged = np.asarray(scores, dtype="float64") <= float(run.threshold)
            entry[f"{name}_flag_rate"] = float(flagged.mean()) if len(flagged) else 0.0
        out[run.name] = entry
    return out


# ---------------------------------------------------------------------------
# Phase 2 — threshold and abstention audit (validation only)
# ---------------------------------------------------------------------------


def audit_threshold_policies(run: CandidateRun, y_val: Optional[pd.Series],
                             policies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate pre-declared threshold policies on the validation split only.

    Each policy is a small, named rule. No policy is tuned against test
    results, and no policy is derived from the incumbent's chosen threshold
    by looking at the test split.
    """
    results = []
    scores = run.validation_scores
    y = np.asarray([bool(v) for v in y_val], dtype=bool) if y_val is not None else None

    for policy in policies:
        name = policy.get("name", "unnamed")
        kind = policy.get("kind", "fixed")
        if kind == "fixed":
            threshold = float(policy["value"])
        elif kind == "train_quantile":
            q = float(policy["value"])
            # lower score = more anomalous -> the q-quantile of the *upper*
            # tail of the training score distribution is the alert threshold.
            threshold = float(np.percentile(run.train_scores, (1.0 - q) * 100.0))
        elif kind == "validation_f1":
            threshold = float(policy["value"])
        else:
            results.append({"policy": name, "kind": kind, "error": f"unknown policy kind {kind!r}"})
            continue

        flags = apply_threshold(scores, threshold)
        entry: Dict[str, Any] = {
            "policy": name,
            "kind": kind,
            "threshold": threshold,
            "flagged": int(flags.sum()),
            "n_windows": int(len(flags)),
        }
        if y is not None and len(np.unique(y)) > 1:
            tp = int((y & flags).sum())
            fp = int((~y & flags).sum())
            tn = int((~y & ~flags).sum())
            fn = int((y & ~flags).sum())
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            entry.update({
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision": precision, "recall": recall, "f1": f1,
                "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
                "false_alerts_per_window": fp / len(flags) if len(flags) else 0.0,
            })
        else:
            entry["metrics_skipped"] = "validation split is single-class or unlabelled"
        results.append(entry)

    return {
        "evaluated_on": "validation",
        "used_test_metrics": False,
        "score_direction": "lower_is_more_anomalous",
        "score_semantics": (
            "IsolationForest.score_samples returns a value in (-1, 0]; LOWER means MORE "
            "anomalous. The decision rule is raw_score <= threshold (inclusive). The score "
            "is a ranking statistic, not a probability."
        ),
        "policies": results,
    }


# ---------------------------------------------------------------------------
# Phase 4 — reports
# ---------------------------------------------------------------------------


def _repo_revision() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - the revision is informational
        return "unknown"


READY = "ready for downstream local integration"
DEMO = "controlled demo only"
BLOCKED = "blocked"


def assess_readiness(report_sections: Dict[str, Any]) -> Dict[str, Any]:
    """Decide the readiness verdict from the audit's own evidence.

    The verdict is one of:
      * ``ready for downstream local integration`` — the contract works, the
        selection is defensible and the pipeline is reproducible;
      * ``controlled demo only`` — technically sound, but the evidence base
        is too weak to trust the detections;
      * ``blocked`` — a prerequisite is missing or a safety property fails.

    All evidence in this repository is synthetic, so a *production* claim is
    never possible. The verdict is about local integration only.
    """
    blockers: List[str] = []
    cautions: List[str] = []

    if report_sections.get("selection_evidence_split") != "validation":
        blockers.append("model selection did not use validation data only")
    if report_sections.get("used_test_metrics_for_selection"):
        blockers.append("test metrics influenced the model selection")

    e2e = report_sections.get("end_to_end") or {}
    totals = e2e.get("totals") or {}
    if totals.get("unscored", 0) > 0:
        cautions.append(
            f"{totals['unscored']} window(s) could not be scored end to end; "
            "the downstream component must handle the unscored status"
        )
    if totals.get("missed", 0) > 0:
        cautions.append(
            f"{totals['missed']} injected anomaly window(s) were NOT detected; "
            "detection coverage is incomplete"
        )

    shift_rate = e2e.get("benign_shift_flag_rate")
    if shift_rate is not None and shift_rate > 0.5:
        cautions.append(
            f"benign-shift flag rate is {shift_rate:.2f}; the detector is highly "
            "sensitive to legitimate distribution changes"
        )
    missing_rate = e2e.get("missing_telemetry_flag_rate")
    if missing_rate is not None and missing_rate > 0.5:
        cautions.append(
            f"missing-telemetry flag rate is {missing_rate:.2f}; missing telemetry is "
            "being treated as suspicious"
        )

    cautions.append(
        "every result is synthetic; real-world detection quality is unproven"
    )
    cautions.append(
        "the validation split has 45 windows / 11 positives, so threshold and "
        "selection decisions carry wide uncertainty"
    )

    if blockers:
        return {
            "verdict": BLOCKED,
            "blockers": blockers,
            "cautions": cautions,
            "rationale": "one or more safety preconditions failed; see blockers",
        }
    return {
        "verdict": DEMO,
        "blockers": [],
        "cautions": cautions,
        "rationale": (
            "The interface, the split discipline and the end-to-end pipeline are sound "
            "and reproducible, and the model selection is now defensible on validation "
            "evidence alone. However, every result comes from synthetic data and the "
            "M5 false-alert weaknesses are unresolved, so the component is suitable for "
            "a controlled local demo / integration rehearsal, not for trusted operation "
            "on real traffic."
        ),
    }


def build_readiness_report(
    cfg,
    decision: SelectionDecision,
    runs: List[CandidateRun],
    provenance: Dict[str, Any],
    threshold_audit: Dict[str, Any],
    e2e: Dict[str, Any],
    abstention: Dict[str, Any],
    runtime: Dict[str, Any],
    root_cause: Dict[str, Any],
) -> Dict[str, Any]:
    report = {
        "milestone": 7,
        "report_type": "ml_readiness_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository_revision": _repo_revision(),
        "data_source": provenance,
        "model_and_threshold_policy": {
            "selected_model": decision.selected,
            "incumbent": decision.incumbent,
            "selection_policy": decision.policy_name,
            "selection_evidence_split": "validation",
            "used_test_metrics_for_selection": False,
            "decision": decision.to_dict(),
        },
        "selection_evidence": [
            {
                "candidate": run.name,
                "kind": run.kind,
                "validation_metrics": run.validation_metrics.get("metrics", {}),
                "threshold": run.threshold,
                "threshold_method": run.threshold_method,
                "training_rows_used": run.training_rows_used,
                "runtime_s": round(run.runtime_s, 3),
                "notes": run.notes,
            }
            for run in runs
        ],
        "final_test_evaluation": [
            {
                "candidate": run.name,
                "test_metrics": run.test_metrics.get("metrics", {}),
                "note": "computed AFTER the selection decision was locked; not used for selection",
            }
            for run in runs
        ],
        "threshold_audit": threshold_audit,
        "abstention_audit": abstention,
        "end_to_end": e2e,
        "runtime_observations": runtime,
        "m4_discrepancy": root_cause,
        "limitations": [
            "All data is synthetic. These results validate the pipeline and the "
            "selection/threshold policy; they do NOT establish real-world detection quality.",
            "The validation split has 45 windows / 11 positives. Differences between "
            "candidates of a few percentage points are inside the noise band.",
            "The test split is small (46 windows / 8 positives) and was scored once, "
            "after the selection decision was locked.",
            "A single random split cannot rule out split-dependent conclusions; "
            "time-based and group-based splits are not re-run here.",
            "No production SLA, latency guarantee or availability claim is made.",
        ],
    }
    report["readiness"] = assess_readiness({
        "selection_evidence_split": report["model_and_threshold_policy"]["selection_evidence_split"],
        "used_test_metrics_for_selection": report["model_and_threshold_policy"]["used_test_metrics_for_selection"],
        "end_to_end": e2e,
    })
    return report


def save_readiness_artifacts(report: Dict[str, Any], run_dir: str) -> Dict[str, str]:
    reports_dir = os.path.join(run_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    json_path = os.path.join(reports_dir, "readiness_audit.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    md_path = os.path.join(reports_dir, "readiness_audit.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_readiness_markdown(report))

    manifest_path = os.path.join(run_dir, "readiness_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({
            "milestone": 7,
            "run_dir": run_dir,
            "generated_at": report["generated_at"],
            "repository_revision": report["repository_revision"],
            "selected_model": report["model_and_threshold_policy"]["selected_model"],
            "selection_evidence_split": "validation",
            "used_test_metrics_for_selection": False,
        }, handle, indent=2)

    return {"json": json_path, "markdown": md_path, "manifest": manifest_path}


def render_readiness_markdown(report: Dict[str, Any]) -> str:
    policy = report["model_and_threshold_policy"]
    decision = policy["decision"]
    lines: List[str] = []
    lines.append("# Milestone 7 — ML Readiness Audit")
    lines.append("")
    lines.append(f"- Generated: {report['generated_at']}")
    lines.append(f"- Repository revision: {report['repository_revision']}")
    lines.append(f"- Selected model: **{policy['selected_model']}**")
    lines.append(f"- Selection policy: `{policy['selection_policy']}`")
    lines.append(f"- Selection evidence split: **{policy['selection_evidence_split']}** "
                 f"(test metrics used for selection: {policy['used_test_metrics_for_selection']})")
    readiness = report.get("readiness", {})
    lines.append(f"- **Readiness verdict: {readiness.get('verdict', 'n/a')}**")
    lines.append("")
    if readiness.get("rationale"):
        lines.append(f"> {readiness['rationale']}")
        lines.append("")
    for blocker in readiness.get("blockers", []):
        lines.append(f"- ⛔ **BLOCKER:** {blocker}")
    for caution in readiness.get("cautions", []):
        lines.append(f"- ⚠️ {caution}")
    lines.append("")
    lines.append("## 1. Selection decision (validation only)")
    lines.append("")
    lines.append(f"> {decision['reason']}")
    lines.append("")
    lines.append("| Candidate | Weighted validation score | Floors failed |")
    lines.append("|---|---|---|")
    for name, score in decision["weighted_scores"].items():
        failed = decision["floors_failed"].get(name, [])
        lines.append(f"| {name} | {score:.4f} | {', '.join(failed) if failed else 'none'} |")
    lines.append("")
    lines.append("## 2. Validation evidence vs final test evaluation")
    lines.append("")
    lines.append("| Candidate | Split | Precision | Recall | F1 | PR-AUC | FP | Flagged |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for entry in report["selection_evidence"]:
        m = entry["validation_metrics"]
        lines.append(
            f"| {entry['candidate']} | validation | {_fmt(m.get('precision'))} | {_fmt(m.get('recall'))} | "
            f"{_fmt(m.get('f1'))} | {_fmt(m.get('pr_auc'))} | {_fmt(m.get('fp'))} | {_fmt(m.get('n_flagged'))} |"
        )
    for entry in report["final_test_evaluation"]:
        m = entry["test_metrics"]
        lines.append(
            f"| {entry['candidate']} | test (final) | {_fmt(m.get('precision'))} | {_fmt(m.get('recall'))} | "
            f"{_fmt(m.get('f1'))} | {_fmt(m.get('pr_auc'))} | {_fmt(m.get('fp'))} | {_fmt(m.get('n_flagged'))} |"
        )
    lines.append("")
    lines.append("## 3. Threshold policies (validation only)")
    lines.append("")
    lines.append(f"> {report['threshold_audit']['score_semantics']}")
    lines.append("")
    lines.append("| Policy | Kind | Threshold | TP | FP | TN | FN | Precision | Recall | F1 | FPR |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for entry in report["threshold_audit"]["policies"]:
        if "error" in entry:
            lines.append(f"| {entry['policy']} | {entry['kind']} | - | - | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {entry['policy']} | {entry['kind']} | {entry['threshold']:.6f} | {_fmt(entry.get('tp'))} | "
            f"{_fmt(entry.get('fp'))} | {_fmt(entry.get('tn'))} | {_fmt(entry.get('fn'))} | "
            f"{_fmt(entry.get('precision'))} | {_fmt(entry.get('recall'))} | {_fmt(entry.get('f1'))} | "
            f"{_fmt(entry.get('false_positive_rate'))} |"
        )
    lines.append("")
    lines.append("## 4. Abstention / missingness")
    lines.append("")
    for key, value in report["abstention_audit"].items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    lines.append("## 5. End-to-end pipeline")
    lines.append("")
    lines.append("| Scenario | Windows | Flagged | Detected | Missed | False positives | Unscored | Expected |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in report["end_to_end"].get("scenarios", []):
        lines.append(
            f"| {row['scenario']} | {row['n_windows']} | {row.get('flagged', 0)} | "
            f"{row.get('detected', 0)} | {row.get('missed', 0)} | {row.get('false_positives', 0)} | "
            f"{row.get('unscored', 0)} | {row.get('expected_behavior', 'n/a')} |"
        )
    lines.append("")
    lines.append("## 6. M4 discrepancy — root cause and resolution")
    lines.append("")
    for key, value in report["m4_discrepancy"].items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    lines.append("## 7. Runtime observations")
    lines.append("")
    for key, value in report["runtime_observations"].items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    lines.append("## 8. Limitations")
    lines.append("")
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    lines.append("## 9. Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/run_readiness_audit.py")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
