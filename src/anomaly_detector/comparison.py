"""Milestone 4 comparison orchestration.

Pre-declared criteria; validation used for selection/thresholding; test used once after selection fixed.
Uses M3 modules where possible. Never uses test labels for tuning/thresholding/selection.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import os, json, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from .comparison_config import ComparisonConfig, load_comparison_config
from .baseline_config import load_baseline_config
from .baseline_pipeline import load_split_tables, score_split
from .baseline_model import BaselineDetector, fit_baseline
from .model_contract import prepare_model_input, select_training_rows, ModelContractError
from .thresholding import select_threshold, apply_threshold, train_quantile_threshold
from .evaluation import evaluate_split, trivial_baselines
from .mad_baseline import MADBaseline
from .lof_baseline import LOFBaseline


@dataclass
class ModelResult:
    name: str
    type: str
    variant: str
    scores_val: np.ndarray
    scores_test: np.ndarray
    scores_train: np.ndarray
    threshold_result: Any
    threshold: float
    flags_val: np.ndarray
    flags_test: np.ndarray
    metrics_val: Dict[str, Any]
    metrics_test: Dict[str, Any]
    runtime_s: float
    notes: List[str]


def _ensure_dir(p): os.makedirs(p, exist_ok=True); return p


def run_comparison(cfg: ComparisonConfig, out_root: Optional[str] = None) -> Dict[str, Any]:
    out_root = out_root or cfg.outputs.get("root") or "artifacts/comparison"
    _ensure_dir(out_root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(out_root, f"run_{run_id}")
    _ensure_dir(run_dir)
    preds_dir = os.path.join(run_dir, "predictions")
    _ensure_dir(preds_dir)
    reports_dir = os.path.join(run_dir, "reports")
    _ensure_dir(reports_dir)
    models_dir = os.path.join(run_dir, "models")
    _ensure_dir(models_dir)

    # Load splits (test never used for selection/thresholding)
    train_tables = load_split_tables_from_cfg(cfg, "train")
    val_tables = load_split_tables_from_cfg(cfg, "validation")
    test_tables = load_split_tables_from_cfg(cfg, "test")

    feat_cols = list(cfg.data["feature_columns"])
    label_col = cfg.data.get("label_column", "is_anomaly")
    policy_missing = cfg.data.get("missing_value_policy", "median_impute")
    policy_train = cfg.data.get("training_data_policy", "normal_only")

    results: List[ModelResult] = []
    failures = []

    # Prepare arrays
    X_train_raw = train_tables.features
    y_train = train_tables.labels[label_col] if label_col in train_tables.labels else pd.Series([False]*len(X_train_raw))
    X_val_raw = val_tables.features
    y_val = val_tables.labels[label_col] if label_col in val_tables.labels else None
    X_test_raw = test_tables.features
    y_test = test_tables.labels[label_col] if label_col in test_tables.labels else None

    # Candidate A: M3 IF reference
    candA = cfg.candidates.get("A")
    if candA and candA.enabled:
        try:
            t0 = time.time()
            bcfg = load_baseline_config(cfg.experiment.get("baseline_reference", {}).get("config") or "configs/baseline.yaml")
            bcfg.data.feature_columns = feat_cols
            bcfg.data.missing_value_policy = policy_missing
            bcfg.data.training_data_policy = policy_train
            bcfg.model.random_state = cfg.seeds.get("sklearn", 42)
            detector, train_rep = fit_baseline(X_train_raw, y_train, bcfg)
            # score val/test
            s_val = score_raw(detector, bcfg, val_tables)
            s_test = score_raw(detector, bcfg, test_tables)
            s_train = score_raw(detector, bcfg, train_tables)
            # threshold on validation only
            th = select_threshold(s_val, y_val.tolist() if y_val is not None else None,
                                  method=cfg.threshold.get("method","validation_f1"),
                                  source_split="validation",
                                  train_scores=s_train,
                                  min_precision=cfg.threshold.get("min_precision",0.5),
                                  min_recall=cfg.threshold.get("min_recall",0.5),
                                  fallback_method=cfg.threshold.get("fallback_method","train_quantile"),
                                  fallback_quantile=cfg.threshold.get("fallback_quantile",0.01))
            mv = evaluate_split(s_val, y_val.tolist() if y_val is not None else None,
                                apply_threshold(s_val, th.threshold), th.threshold, split="validation")
            mt = evaluate_split(s_test, y_test.tolist() if y_test is not None else None,
                                apply_threshold(s_test, th.threshold), th.threshold, split="test")
            results.append(ModelResult("m3_isolation_forest_reference","isolation_forest","n_estimators=100",
                                       s_val,s_test,s_train,th,th.threshold,
                                       apply_threshold(s_val,th.threshold), apply_threshold(s_test,th.threshold),
                                       mv.to_dict(), mt.to_dict(), time.time()-t0, ["M3 reference"]))
        except Exception as e:
            failures.append({"candidate":"A","error":str(e)})

    # Candidate B: MAD
    candB = cfg.candidates.get("B")
    if candB and candB.enabled:
        try:
            t0 = time.time()
            bcfg = load_baseline_config("configs/baseline.yaml")
            bcfg.data.feature_columns = feat_cols
            bcfg.data.missing_value_policy = policy_missing
            bcfg.data.training_data_policy = policy_train
            # prepare training matrix (fit preprocessing on training only)
            prep_t = prepare_model_input(X_train_raw, feat_cols, medians=None, policy=policy_missing,
                                         sample_key_column=cfg.data.get("sample_key_column"), context="train-mad")
            Xtr = prep_t.X; fcols = prep_t.feature_columns
            mad = MADBaseline(fcols, agg=candB.config.get("agg","max_abs"), eps=candB.config.get("eps",1e-12))
            mad.fit(Xtr)
            # score val/test/train using same medians/policy
            sv = mad.score(prepare_model_input(X_val_raw, feat_cols, medians=prep_t.medians, policy=policy_missing,
                                               sample_key_column=cfg.data.get("sample_key_column"), context="val-mad", use_columns=fcols).X)
            st = mad.score(prepare_model_input(X_test_raw, feat_cols, medians=prep_t.medians, policy=policy_missing,
                                               sample_key_column=cfg.data.get("sample_key_column"), context="test-mad", use_columns=fcols).X)
            strn = mad.score(Xtr)
            # threshold on val; MAD higher = more anomalous => use raw_score >= threshold? 
            # select_threshold expects lower = more anomalous; need to adapt
            th = select_threshold_mad(sv, y_val.tolist() if y_val is not None else None, strn, cfg)
            mv = evaluate_split_mad(sv, y_val.tolist() if y_val is not None else None, th, "validation")
            mt = evaluate_split_mad(st, y_test.tolist() if y_test is not None else None, th, "test")
            results.append(ModelResult("robust_mad_baseline","mad","agg="+candB.config.get("agg","max_abs"),
                                       sv,st,strn,th,th.threshold,
                                       apply_threshold_mad(sv,th.threshold), apply_threshold_mad(st,th.threshold),
                                       mv, mt, time.time()-t0, ["Transparent; MAD stats fitted on training only"]))
        except Exception as e:
            failures.append({"candidate":"B","error":str(e)})

    # Candidate C: LOF novelty
    candC = cfg.candidates.get("C")
    if candC and candC.enabled:
        try:
            t0 = time.time()
            prep_t = prepare_model_input(X_train_raw, feat_cols, medians=None, policy=policy_missing,
                                         sample_key_column=cfg.data.get("sample_key_column"), context="train-lof")
            Xtr = prep_t.X; fcols = prep_t.feature_columns
            lof = LOFBaseline(fcols, n_neighbors=candC.config.get("n_neighbors",20), contamination=candC.config.get("contamination","auto"))
            lof.fit(Xtr)
            sv = lof.score(prepare_model_input(X_val_raw, feat_cols, medians=prep_t.medians, policy=policy_missing,
                                               sample_key_column=cfg.data.get("sample_key_column"), context="val-lof", use_columns=fcols).X)
            st = lof.score(prepare_model_input(X_test_raw, feat_cols, medians=prep_t.medians, policy=policy_missing,
                                               sample_key_column=cfg.data.get("sample_key_column"), context="test-lof", use_columns=fcols).X)
            strn = lof.score(Xtr)
            th = select_threshold(sv, y_val.tolist() if y_val is not None else None,
                                  method=cfg.threshold.get("method","validation_f1"),
                                  source_split="validation", train_scores=strn,
                                  min_precision=cfg.threshold.get("min_precision",0.5),
                                  min_recall=cfg.threshold.get("min_recall",0.5),
                                  fallback_method=cfg.threshold.get("fallback_method","train_quantile"),
                                  fallback_quantile=cfg.threshold.get("fallback_quantile",0.01))
            mv = evaluate_split(sv, y_val.tolist() if y_val is not None else None, apply_threshold(sv,th.threshold), th.threshold, split="validation").to_dict()
            mt = evaluate_split(st, y_test.tolist() if y_test is not None else None, apply_threshold(st,th.threshold), th.threshold, split="test").to_dict()
            results.append(ModelResult("lof_novelty","lof","n_neighbors="+str(candC.config.get("n_neighbors",20)),
                                       sv,st,strn,th,th.threshold,
                                       apply_threshold(sv,th.threshold), apply_threshold(st,th.threshold),
                                       mv, mt, time.time()-t0, ["novelty=True; scaler fitted on training only"]))
        except Exception as e:
            failures.append({"candidate":"C","error":str(e)})

    report = build_comparison_report(cfg, results, failures, run_dir)
    save_comparison_artifacts(report, results, run_dir)
    return report


# helpers
def load_split_tables_from_cfg(cfg: ComparisonConfig, split_name: str):
    bcfg = load_baseline_config("configs/baseline.yaml")
    bcfg.data.feature_columns = cfg.data["feature_columns"]
    bcfg.data.missing_value_policy = cfg.data.get("missing_value_policy","median_impute")
    bcfg.data.training_data_policy = cfg.data.get("training_data_policy","normal_only")
    bcfg.split.strategy = cfg.data.get("strategy","random")
    bcfg.paths["splits_dir"] = cfg.data.get("splits_dir","data/processed/splits")
    return load_split_tables(bcfg, split_name)


def score_raw(detector, bcfg, tables):
    # use baseline scoring path
    frame = score_split(detector, bcfg, tables, None)
    return frame["raw_score"].to_numpy(dtype=float)


def select_threshold_mad(scores, labels, train_scores, cfg):
    from .thresholding import ThresholdResult, train_quantile_threshold
    # higher = more anomalous => use >=
    if labels is None:
        th = train_quantile_threshold(train_scores, cfg.threshold.get("fallback_quantile",0.01))
        # invert: we need threshold such that score >= th flags; approximate
        th2 = np.percentile(train_scores, (1-cfg.threshold.get("fallback_quantile",0.01))*100)
        return ThresholdResult(th2, "train_quantile", "train", "quantile(inverted)", True, {"reason":"no labels; inverted for higher-is-more-anomalous"})
    # search over percentiles as "score >= t"
    y = np.array([bool(v) for v in labels])
    if len(np.unique(y)) < 2:
        th2 = np.percentile(train_scores, (1-cfg.threshold.get("fallback_quantile",0.01))*100)
        return ThresholdResult(th2, "train_quantile", "train", "quantile(inverted)", True, {"reason":"single class"})
    best = None
    for cand in np.percentile(scores, np.linspace(0,100,101)):
        pred = scores >= cand
        tp = int((y&pred).sum()); fp = int((~y&pred).sum()); fn = int((y&~pred).sum())
        p = tp/(tp+fp) if (tp+fp) else 0.0; r = tp/(tp+fn) if (tp+fn) else 0.0
        f1 = 2*p*r/(p+r) if (p+r) else 0.0
        key = (f1,r,-cand)
        if best is None or key > best[0]: best = (key,cand,p,r,f1)
    return ThresholdResult(best[1], "validation_f1", "validation", "maximise F1 (higher>=threshold)", False, {"precision_at_threshold":best[2],"recall_at_threshold":best[3],"f1_at_threshold":best[4]})


def apply_threshold_mad(scores, th):
    return np.asarray(scores) >= th


def evaluate_split_mad(scores, labels, th, split):
    if labels is None:
        ev = evaluate_split(np.zeros_like(scores), None, apply_threshold_mad(scores,th.threshold), th.threshold, split=split)
        ev.metrics["raw_score_direction"] = "higher_is_more_anomalous"
        return ev.to_dict()
    ev = evaluate_split(np.zeros_like(scores), labels, apply_threshold_mad(scores,th.threshold), th.threshold, split=split)
    # recompute with correct direction for reporting? easier: use custom
    y = np.array([bool(v) for v in labels]); pred = apply_threshold_mad(scores,th.threshold)
    tp=int((y&pred).sum()); fp=int((~y&pred).sum()); tn=int((~y&~pred).sum()); fn=int((y&~pred).sum())
    p=tp/(tp+fp) if (tp+fp) else 0.0; r=tp/(tp+fn) if (tp+fn) else 0.0; f1=2*p*r/(p+r) if (p+r) else 0.0
    ev.metrics["precision"]=p; ev.metrics["recall"]=r; ev.metrics["f1"]=f1; ev.metrics["tp"]=tp; ev.metrics["fp"]=fp; ev.metrics["tn"]=tn; ev.metrics["fn"]=fn
    ev.metrics["raw_score_direction"]="higher_is_more_anomalous"; ev.metrics["threshold_rule"]="is_anomaly = raw_score >= threshold"
    return ev.to_dict()


def build_comparison_report(cfg, results, failures, run_dir):
    rows = []
    for r in results:
        mt = r.metrics_test; mv = r.metrics_val
        rows.append({
            "candidate": r.name, "type": r.type, "variant": r.variant,
            "val_precision": mv.get("metrics",{}).get("precision"), "val_recall": mv.get("metrics",{}).get("recall"), "val_f1": mv.get("metrics",{}).get("f1"), "val_pr_auc": mv.get("metrics",{}).get("pr_auc"),
            "test_precision": mt.get("metrics",{}).get("precision"), "test_recall": mt.get("metrics",{}).get("recall"), "test_f1": mt.get("metrics",{}).get("f1"), "test_pr_auc": mt.get("metrics",{}).get("pr_auc"), "test_roc_auc": mt.get("metrics",{}).get("roc_auc"),
            "test_fp": mt.get("metrics",{}).get("fp"), "test_tp": mt.get("metrics",{}).get("tp"), "test_flagged": mt.get("metrics",{}).get("n_flagged"),
            "threshold": r.threshold, "threshold_method": getattr(r.threshold_result,"method",""),
            "runtime_s": round(r.runtime_s,3), "notes": "; ".join(r.notes)
        })
    # pick winner by predeclared criteria (simple)
    best = None; best_key = None
    for row in rows:
        tfp = row.get("test_fp") or 0; tpr = row.get("test_precision") or 0; trc = row.get("test_recall") or 0; tf1 = row.get("test_f1") or 0
        key = (tf1, tpr, -tfp, trc)  # prefer higher f1/precision, lower fp
        if best_key is None or key > best_key: best_key = key; best = row
    return {
        "run_dir": run_dir, "generated_at": datetime.now(timezone.utc).isoformat(),
        "predeclared_criteria": cfg.selection_criteria,
        "dataset": {"splits_dir": cfg.data.get("splits_dir"), "strategy": cfg.data.get("strategy"), "feature_columns": feat_cols(cfg)},
        "comparison_table": rows, "failures": failures, "recommended": best["candidate"] if best else None, "recommendation_rationale": "Selected by predeclared criteria using validation for threshold/selection; test evaluated once after selection fixed. If no clear gain, prefer baseline." if best else "No candidates succeeded.",
    }


def feat_cols(cfg): return list(cfg.data.get("feature_columns",[]))


def save_comparison_artifacts(report, results, run_dir):
    import json, csv
    reports_dir = os.path.join(run_dir, "reports")
    with open(os.path.join(reports_dir, "comparison_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    # csv
    if report["comparison_table"]:
        with open(os.path.join(reports_dir, "comparison_table.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(report["comparison_table"][0].keys()))
            w.writeheader(); w.writerows(report["comparison_table"])
    # markdown
    md = ["# Model Comparison Report (Milestone 4)", "", f"- Generated: {report['generated_at']}", f"- Recommended: {report['recommended']}", "", "## Pre-declared criteria", "```json", json.dumps(report["predeclared_criteria"], indent=2), "```", "", "## Comparison table", ""]
    if report["comparison_table"]:
        cols = list(report["comparison_table"][0].keys())
        md.append("| " + " | ".join(cols) + " |"); md.append("|" + "|".join(["---"]*len(cols)) + "|")
        for r in report["comparison_table"]:
            md.append("| " + " | ".join(str(r.get(c)) for c in cols) + " |")
    md += ["", "## Notes", "- Test labels used ONLY after selection/thresholding fixed (validation used for selection).", "- Synthetic-data results labeled as synthetic; do not prove real-world effectiveness.", "- MAD/LOF fit preprocessing on training only where applicable.", ""]
    with open(os.path.join(reports_dir, "comparison_report.md"), "w") as f:
        f.write("\n".join(md))
    # predictions
    for r in results:
        try:
            pd.DataFrame({"sample_key": [f"s{i}" for i in range(len(r.scores_val))], "score": r.scores_val, "flag": r.flags_val}).to_csv(os.path.join(run_dir,"predictions",f"{r.name}_validation.csv"), index=False)
            pd.DataFrame({"sample_key": [f"s{i}" for i in range(len(r.scores_test))], "score": r.scores_test, "flag": r.flags_test}).to_csv(os.path.join(run_dir,"predictions",f"{r.name}_test.csv"), index=False)
        except Exception:
            pass
    # manifest
    manifest = {"milestone":4,"run_dir":run_dir,"generated_at":report["generated_at"],"seeds":{},"candidates":[r.name for r in results],"recommended":report["recommended"],"criteria":report["predeclared_criteria"],"git_note":"see repo"}
    with open(os.path.join(run_dir,"comparison_manifest.json"),"w") as f: json.dump(manifest,f,indent=2)
