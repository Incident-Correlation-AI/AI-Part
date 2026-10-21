"""Milestone 5 robustness validation orchestration.

Uses the SELECTED M3 Isolation Forest model (from artifacts/baseline/model/manifest.json)
without retraining/tuning on test scenarios. All scenarios use deterministic seeds.
Never encodes scenario labels into features.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import os, json, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from .comparison_config import load_comparison_config
from .baseline_config import load_baseline_config
from .baseline_model import BaselineDetector
from .model_contract import prepare_model_input, ModelContractError
from .thresholding import apply_threshold
from .robustness_scenarios import (
    gen_normal_variation, gen_service_anomalies, gen_telemetry_faults,
    gen_low_volume_boundaries, gen_distribution_shifts, FEATURES
)
import joblib


@dataclass
class ScenarioResult:
    scenario: str
    group: str
    n: int
    scores: np.ndarray
    flags: np.ndarray
    labels: pd.DataFrame
    notes: List[str]
    metrics: Dict[str, Any]


def _ensure_dir(p): os.makedirs(p, exist_ok=True); return p


def load_selected_model():
    """Load the selected M3 IF model and its contract."""
    model_path = "artifacts/baseline/model/model.joblib"
    manifest_path = "artifacts/baseline/model/manifest.json"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Selected model not found: {model_path}")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Model manifest not found: {manifest_path}")
    
    obj = joblib.load(model_path)
    from .baseline_model import BaselineDetector
    det = BaselineDetector.__new__(BaselineDetector)
    det.estimator = obj['estimator']
    det.feature_columns = obj['feature_columns']
    det.expected_feature_columns = obj['expected_feature_columns']
    det.medians = obj['medians']
    det.missing_value_policy = obj['missing_value_policy']
    det.is_fitted = obj['is_fitted']
    det.model_config = obj['model_config']
    contract = obj.get("score_contract") or det.score_contract
    medians = obj.get("medians")
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    return det, contract, medians, manifest


def run_robustness(out_root: Optional[str] = None) -> Dict[str, Any]:
    out_root = out_root or "artifacts/robustness"
    _ensure_dir(out_root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(out_root, f"run_{run_id}")
    _ensure_dir(run_dir)
    preds_dir = os.path.join(run_dir, "predictions")
    _ensure_dir(preds_dir)
    reports_dir = os.path.join(run_dir, "reports")
    _ensure_dir(reports_dir)
    
    # Load selected model - NO RETRAINING
    detector, contract, medians, manifest = load_selected_model()
    feat_cols = manifest["feature_schema"]["feature_columns"]
    threshold = manifest["threshold"]["threshold"]
    score_dir = manifest["score"]["score_direction"]
    
    results: List[ScenarioResult] = []
    
    # Generate all scenarios with deterministic seeds
    scenarios = [
        ("normal_variation", "normal", gen_normal_variation),
        ("service_anomalies", "anomaly", gen_service_anomalies),
        ("telemetry_faults", "fault", gen_telemetry_faults),
        ("low_volume_boundaries", "boundary", gen_low_volume_boundaries),
        ("distribution_shifts", "shift", gen_distribution_shifts),
    ]
    
    for name, group, gen_fn in scenarios:
        try:
            sc = gen_fn()
            X = sc.features[feat_cols].copy()
            
            # Score with fixed model - NO retraining
            prep = prepare_model_input(X, feat_cols, medians=medians, policy="median_impute",
                                      sample_key_column="sample_key", context=f"m5-{name}",
                                      use_columns=feat_cols)
            Xp = prep.X
            scores = detector.score(Xp)
            flags = apply_threshold(scores, threshold)  # lower_is_more_anomalous: score <= threshold
            
            # Compute metrics if labels exist and are trustworthy
            metrics = {}
            y = sc.labels["is_anomaly"].values if "is_anomaly" in sc.labels else None
            if y is not None and group in ["anomaly"]:  # only for clearly labeled anomalies
                tp = int(((y == True) & flags).sum())
                fp = int(((y == False) & flags).sum())
                tn = int(((y == False) & ~flags).sum())
                fn = int(((y == True) & ~flags).sum())
                p = tp/(tp+fp) if (tp+fp) else 0.0
                r = tp/(tp+fn) if (tp+fn) else 0.0
                f1 = 2*p*r/(p+r) if (p+r) else 0.0
                metrics = {"tp":tp, "fp":fp, "tn":tn, "fn":fn, "precision":p, "recall":r, "f1":f1}
            elif y is not None and group in ["normal", "fault", "boundary", "shift"]:
                fp = int(((y == False) & flags).sum())
                fp_rate = fp / len(flags) if len(flags) else 0.0
                metrics = {"false_positives": fp, "false_positive_rate": fp_rate, "benign_flagged": fp}
            
            results.append(ScenarioResult(name, group, len(X), scores, flags, sc.labels, [sc.notes], metrics))
            
            # Save predictions
            pred_df = pd.DataFrame({
                "sample_key": sc.labels["sample_key"],
                "scenario": sc.labels["scenario"],
                "scenario_group": group,
                "raw_score": scores,
                "flagged": flags,
                "is_anomaly_label": y if y is not None else None
            })
            pred_df.to_csv(os.path.join(preds_dir, f"{name}_predictions.csv"), index=False)
            
        except Exception as e:
            results.append(ScenarioResult(name, group, 0, np.array([]), np.array([]), 
                                        pd.DataFrame(), [f"ERROR: {e}"], {"error": str(e)}))
    
    report = build_robustness_report(manifest, results, run_dir)
    save_robustness_artifacts(report, results, run_dir)
    return report


def build_robustness_report(manifest, results, run_dir):
    table = []
    for r in results:
        if r.n == 0:
            table.append({"scenario": r.scenario, "group": r.group, "n": 0, "flagged": 0, 
                         "flag_rate": 0, "notes": "; ".join(r.notes), "metrics": r.metrics})
            continue
        flag_rate = float(r.flags.sum()) / len(r.flags)
        table.append({
            "scenario": r.scenario, "group": r.group, "n": r.n,
            "flagged": int(r.flags.sum()), "flag_rate": flag_rate,
            "score_mean": float(r.scores.mean()), "score_std": float(r.scores.std()),
            "score_min": float(r.scores.min()), "score_max": float(r.scores.max()),
            "notes": "; ".join(r.notes), "metrics": r.metrics
        })
    
    return {
        "run_dir": run_dir,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": {
            "type": manifest["model"]["type"],
            "manifest": "artifacts/baseline/model/manifest.json",
            "threshold": manifest["threshold"]["threshold"],
            "score_direction": manifest["score"]["score_direction"],
            "threshold_rule": manifest["score"]["threshold_rule"],
            "feature_columns": manifest["feature_schema"]["feature_columns"]
        },
        "scenarios_tested": [r.scenario for r in results],
        "results": table,
        "caveats": [
            "All results are synthetic-only; do not prove real-world effectiveness.",
            "Model not retrained or tuned on scenarios - fixed M3 parameters and threshold.",
            "Test labels used only where scenarios are defensibly labeled (service anomalies).",
            "Benign faults/shifts are labeled benign; flagging them may indicate sensitivity issues.",
            "Small sample sizes limit metric reliability."
        ]
    }


def save_robustness_artifacts(report, results, run_dir):
    reports_dir = os.path.join(run_dir, "reports")
    preds_dir = os.path.join(run_dir, "predictions")
    
    with open(os.path.join(reports_dir, "robustness_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # CSV summary
    rows = []
    for t in report["results"]:
        row = {"scenario": t["scenario"], "group": t["group"], "n": t["n"],
               "flagged": t.get("flagged", 0), "flag_rate": t.get("flag_rate", 0)}
        row.update({k: t.get(k) for k in ["score_mean", "score_std", "score_min", "score_max"] if k in t})
        m = t.get("metrics", {})
        for k in ["precision", "recall", "f1", "false_positives", "false_positive_rate"]:
            if k in m: row[k] = m[k]
        rows.append(row)
    
    pd.DataFrame(rows).to_csv(os.path.join(reports_dir, "scenario_summary.csv"), index=False)
    
    # Markdown report
    md = ["# Milestone 5 — Robustness, Drift, and Realistic Validation\n"]
    md.append(f"- Generated: {report['generated_at']}")
    md.append(f"- Selected model: {report['selected_model']['type']} (from artifacts/baseline/model/manifest.json)")
    md.append(f"- Threshold: {report['selected_model']['threshold']} ({report['selected_model']['threshold_rule']})")
    md.append("\n## Scenario Results\n")
    md.append("| Scenario | Group | N | Flagged | Flag Rate | Score (mean±std) | Metrics | Notes |")
    md.append("|---|---|---|---|---|---|---|---|")
    for t in report["results"]:
        if t["n"] == 0:
            md.append(f"| {t['scenario']} | {t['group']} | 0 | 0 | 0 | - | {t['metrics']} | {t['notes']} |")
            continue
        score_str = f"{t['score_mean']:.4f}±{t['score_std']:.4f}"
        mstr = json.dumps(t['metrics']) if t['metrics'] else "-"
        md.append(f"| {t['scenario']} | {t['group']} | {t['n']} | {t['flagged']} | {t['flag_rate']:.3f} | {score_str} | {mstr} | {t['notes']} |")
    
    md.append("\n## False Alert / Miss Analysis\n")
    md.append("- **Normal variation:** Benign traffic fluctuations should mostly not be flagged. High flag rates indicate over-sensitivity.")
    md.append("- **Service anomalies:** HTTP 5xx/latency spikes labeled anomalous - tests detection capability.")
    md.append("- **Telemetry faults:** Missing/invalid data labeled benign - should not be treated as attacks; flagging suggests poor handling of missingness.")
    md.append("- **Low volume/boundaries:** Edge cases (zeros, small counts) - model must handle without crashing.")
    md.append("- **Distribution shifts:** Legitimate shifts labeled benign - should tolerate reasonable shifts.\n")
    md.append("## Caveats\n")
    for c in report["caveats"]:
        md.append(f"- {c}")
    md.append("\n## Reproduction\n```bash\npython scripts/run_robustness.py\n```")
    
    with open(os.path.join(reports_dir, "robustness_report.md"), "w") as f:
        f.write("\n".join(md))
    
    # Manifest
    manifest_out = {
        "milestone": 5,
        "run_dir": run_dir,
        "generated_at": report["generated_at"],
        "selected_model_manifest": "artifacts/baseline/model/manifest.json",
        "model_not_retrained": True,
        "threshold_not_tuned": True,
        "scenarios": report["scenarios_tested"]
    }
    with open(os.path.join(run_dir, "robustness_manifest.json"), "w") as f:
        json.dump(manifest_out, f, indent=2)
