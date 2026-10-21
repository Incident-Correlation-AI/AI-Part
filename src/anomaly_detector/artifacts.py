"""Artifact writing for the Milestone 3 baseline (Milestone 3).

Produces, under a configurable output directory:

* ``model/model.joblib``   — the fitted estimator plus its input contract
* ``model/manifest.json``  — the machine-readable model manifest
* ``predictions_<split>.csv`` — per-sample scores, threshold and flags
* ``reports/metrics.json`` / ``reports/metrics.md`` — evaluation output
* ``experiment_note.md``   — limitations and the exact reproduction commands

No raw logs are stored inside the artifacts, and no absolute paths or
credentials are written.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
import json
import os
import subprocess

import pandas as pd

from .baseline_config import BaselineConfig
from .baseline_model import BaselineDetector, TrainingReport
from .thresholding import ThresholdResult


def git_revision() -> str:
    """Best-effort short git revision; ``"unknown"`` when it cannot be read."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - a missing git binary must not break a run
        pass
    return "unknown"


def _library_versions() -> Dict[str, str]:
    versions = {"python": ".".join(str(p) for p in __import__("sys").version_info[:3])}
    for name in ("sklearn", "numpy", "pandas", "joblib"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - optional at report time
            versions[name] = "not installed"
    return versions


def build_manifest(
    detector: BaselineDetector,
    config: BaselineConfig,
    training_report: TrainingReport,
    threshold_result: Optional[ThresholdResult],
    split_info: Dict[str, Any],
    source: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the model manifest."""
    versions = _library_versions()
    manifest: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(),
        "model": {
            "type": "sklearn.ensemble.IsolationForest",
            "library": "scikit-learn",
            "library_version": versions.get("sklearn", "unknown"),
            "versions": versions,
            "parameters": asdict(detector.model_config),
        },
        "configuration": {
            "config_path": config.source_path,
            "model": asdict(config.model),
            "data": asdict(config.data),
            "threshold": asdict(config.threshold),
            "split": asdict(config.split),
            "paths": dict(config.paths),
        },
        "random_seed": config.model.random_state,
        "feature_schema": {
            "feature_columns": list(detector.feature_columns),
            "n_features": len(detector.feature_columns),
            "order_is_significant": True,
        },
        "data_policy": {
            "missing_value_policy": detector.missing_value_policy,
            "non_finite_policy": "reject (infinities raise; NaN handled by the missing-value policy)",
            "missing_is_not_zero": True,
            "imputation_statistics": dict(detector.medians),
            "imputation_fitted_on": "training rows only",
            "training_data_policy": training_report.training_data_policy,
        },
        "training_data": {
            "rows_available": training_report.rows_available,
            "rows_used_for_fitting": training_report.rows_used,
            "rows_excluded": training_report.rows_excluded_anomalous,
            "label_distribution_in_training_split": training_report.label_distribution,
            "cells_imputed": training_report.cells_imputed,
            "dropped_columns": list(training_report.dropped_columns),
            **dict(source),
        },
        "splits": dict(split_info),
        "score": dict(detector.score_contract),
        "threshold": (threshold_result.to_dict() if threshold_result else {
            "threshold": None,
            "method": "not selected",
            "source_split": None,
            "objective": "",
            "is_heuristic": False,
            "details": {},
        }),
        "interpretation": {
            "anomaly_score_is_not_a_probability": True,
            "root_cause_analysis_included": False,
            "note": (
                "A flagged window is a lead for investigation, not proof of an attack; "
                "this milestone performs detection only and performs no reasoning."
            ),
        },
    }
    return manifest


def write_manifest(manifest: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def write_predictions(predictions: pd.DataFrame, path: str) -> str:
    """Write per-sample predictions (stable metadata, score, threshold, flag)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    predictions.to_csv(path, index=False)
    return path


def write_metrics(report: Dict[str, Any], reports_dir: str) -> Dict[str, str]:
    from .evaluation import format_metrics_markdown

    os.makedirs(reports_dir, exist_ok=True)
    json_path = os.path.join(reports_dir, "metrics.json")
    md_path = os.path.join(reports_dir, "metrics.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(format_metrics_markdown(report))
    return {"json": json_path, "markdown": md_path}


def write_experiment_note(
    path: str,
    commands: Sequence[str],
    limitations: Sequence[str],
    artifacts: Dict[str, str],
    manifest: Dict[str, Any],
) -> str:
    """Write the short experiment note (limitations + exact commands)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    threshold = manifest.get("threshold", {})
    training = manifest.get("training_data", {})
    lines: List[str] = []
    lines.append("# Experiment Note — Isolation Forest Baseline")
    lines.append("")
    lines.append(f"- Trained at (UTC): {manifest.get('trained_at_utc')}")
    lines.append(f"- Code revision: {manifest.get('code_revision')}")
    lines.append(f"- Model: {manifest.get('model', {}).get('type')} "
                 f"({manifest.get('model', {}).get('library_version')})")
    lines.append(f"- Seed: {manifest.get('random_seed')}")
    lines.append(f"- Training rows used: {training.get('rows_used_for_fitting')} "
                 f"of {training.get('rows_available')} "
                 f"(policy: {manifest.get('data_policy', {}).get('training_data_policy')})")
    lines.append(f"- Threshold: {threshold.get('threshold')} "
                 f"(method: {threshold.get('method')}, source split: {threshold.get('source_split')}, "
                 f"heuristic: {threshold.get('is_heuristic')})")
    lines.append(f"- Score: {manifest.get('score', {}).get('score_name')} "
                 f"({manifest.get('score', {}).get('score_direction')})")
    lines.append("")
    lines.append("## Reproduction commands")
    lines.append("")
    lines.append("```bash")
    for command in commands:
        lines.append(command)
    lines.append("```")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for name, value in artifacts.items():
        lines.append(f"- `{name}`: {value}")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for limitation in limitations:
        lines.append(f"- {limitation}")
    lines.append("")
    lines.append("## What this is not")
    lines.append("")
    lines.append("- Not root-cause analysis: no evidence correlation, incident narrative or")
    lines.append("  explanation is produced. A flagged window is a lead for investigation.")
    lines.append("- Not a probability: the anomaly score is a ranking statistic, not a")
    lines.append("  confidence percentage.")
    lines.append("- Not proof of real-world detection quality: the evaluation uses synthetic")
    lines.append("  labels, which validate the pipeline rather than the approach.")
    lines.append("")
    return _write_text(lines, path)


def _write_text(lines: List[str], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path
