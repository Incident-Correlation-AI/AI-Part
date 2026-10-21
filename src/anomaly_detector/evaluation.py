"""Evaluation of the Milestone 3 baseline (honest, per-split, reproducible).

Metrics are computed **per split**. Training metrics are never presented as test
performance, and a metric that cannot be computed is skipped with a stated
reason rather than reported as zero.

Metric orientation: the baseline's raw score is *lower = more anomalous*, while
scikit-learn's ranking metrics expect *higher = more positive*. The evaluation
therefore uses the sign-flipped anomaly score (``-raw_score``) for ROC-AUC and
average precision so the numbers read the intuitive way.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .model_contract import ModelContractError


@dataclass
class SplitEvaluation:
    """Metrics and context for one evaluated split."""
    split: str
    n_samples: int
    n_positive: int
    n_negative: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)
    baselines: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "metrics": dict(self.metrics),
            "skipped_metrics": dict(self.skipped),
            "baselines": {k: dict(v) for k, v in self.baselines.items()},
            "warnings": list(self.warnings),
        }


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _scores_from_flags(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Precision/recall/F1 computed from flags (no dependency on ranking metrics)."""
    counts = _confusion(y_true, y_pred)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": counts["tn"],
        "fn": fn,
    }


def evaluate_split(
    scores: Sequence[float],
    labels: Optional[Sequence[Any]],
    is_anomaly: Optional[Sequence[bool]] = None,
    threshold: Optional[float] = None,
    split: str = "test",
    metadata: Optional[pd.DataFrame] = None,
    window_minutes: Optional[float] = None,
) -> SplitEvaluation:
    """Evaluate one split.

    ``labels`` may be ``None`` (no trustworthy ground truth). In that case only
    unsupervised/operational figures are reported and every supervised metric is
    skipped with a reason.
    """
    score_array = np.asarray(scores, dtype="float64")
    if score_array.size and not np.isfinite(score_array).all():
        raise ModelContractError("scores contain non-finite values")

    if is_anomaly is None:
        if threshold is None:
            raise ModelContractError("a threshold is required to flag samples")
        flags = np.asarray(score_array, dtype="float64") <= float(threshold)
    else:
        flags = np.asarray(is_anomaly, dtype=bool)
        if flags.size != score_array.size:
            raise ModelContractError("flags and scores have different lengths")

    evaluation = SplitEvaluation(
        split=split,
        n_samples=int(score_array.size),
        n_positive=0,
        n_negative=0,
    )

    # -- unsupervised / operational figures (always available) --------------
    evaluation.metrics["n_samples"] = int(score_array.size)
    evaluation.metrics["n_flagged"] = int(flags.sum())
    evaluation.metrics["flagged_rate"] = float(flags.mean()) if score_array.size else 0.0
    if score_array.size:
        evaluation.metrics["score_min"] = float(score_array.min())
        evaluation.metrics["score_max"] = float(score_array.max())
        evaluation.metrics["score_mean"] = float(score_array.mean())
        evaluation.metrics["score_median"] = float(np.median(score_array))
    if threshold is not None:
        evaluation.metrics["threshold"] = float(threshold)

    # -- supervised metrics -------------------------------------------------
    if labels is None:
        evaluation.skipped["precision"] = "no ground-truth labels for this split"
        evaluation.skipped["recall"] = "no ground-truth labels for this split"
        evaluation.skipped["f1"] = "no ground-truth labels for this split"
        evaluation.skipped["roc_auc"] = "no ground-truth labels for this split"
        evaluation.skipped["pr_auc"] = "no ground-truth labels for this split"
        evaluation.skipped["confusion_matrix"] = "no ground-truth labels for this split"
        evaluation.skipped["prevalence"] = "no ground-truth labels for this split"
    else:
        y_true = np.asarray([bool(v) if v is not None and not (
            isinstance(v, float) and np.isnan(v)) else False for v in labels], dtype=bool)
        if y_true.size != score_array.size:
            raise ModelContractError("labels and scores have different lengths")
        n_pos = int(y_true.sum())
        n_neg = int((~y_true).sum())
        evaluation.n_positive = n_pos
        evaluation.n_negative = n_neg
        evaluation.metrics["prevalence"] = (n_pos / len(y_true)) if len(y_true) else 0.0
        evaluation.metrics["n_positive"] = n_pos
        evaluation.metrics["n_negative"] = n_neg

        scored = _scores_from_flags(y_true, flags)
        evaluation.metrics.update(scored)
        evaluation.metrics["false_positive_count"] = scored["fp"]
        evaluation.metrics["false_positive_rate"] = (
            scored["fp"] / n_neg if n_neg else 0.0
        )
        evaluation.metrics["false_negative_count"] = scored["fn"]
        evaluation.metrics["true_positive_rate"] = scored["recall"]

        if len(np.unique(y_true)) < 2:
            reason = "only one class present in this split"
            evaluation.skipped["roc_auc"] = reason
            evaluation.skipped["pr_auc"] = reason
            evaluation.metrics["roc_auc"] = None
            evaluation.metrics["pr_auc"] = None
        else:
            anomaly_score = -score_array  # higher == more anomalous
            try:
                evaluation.metrics["roc_auc"] = float(roc_auc_score(y_true, anomaly_score))
            except Exception as exc:  # noqa: BLE001 - reported, never hidden
                evaluation.metrics["roc_auc"] = None
                evaluation.skipped["roc_auc"] = f"could not be computed: {exc}"
            try:
                evaluation.metrics["pr_auc"] = float(average_precision_score(y_true, anomaly_score))
            except Exception as exc:  # noqa: BLE001 - reported, never hidden
                evaluation.metrics["pr_auc"] = None
                evaluation.skipped["pr_auc"] = f"could not be computed: {exc}"

    # -- alert rate (needs reliable host/time coverage) ---------------------
    if metadata is not None and window_minutes and window_minutes > 0:
        if len(metadata) == len(score_array):
            host_hours = (len(metadata) * float(window_minutes)) / 60.0
            evaluation.metrics["observed_host_hours"] = host_hours
            evaluation.metrics["alerts_per_host_hour"] = (
                float(flags.sum()) / host_hours if host_hours else 0.0
            )
        else:
            evaluation.skipped["alerts_per_host_hour"] = (
                "metadata row count does not match the number of scored samples"
            )
    else:
        evaluation.skipped["alerts_per_host_hour"] = (
            "window duration metadata (window_size_minutes) is not available for this "
            "split, so the alert rate per host-hour cannot be computed"
        )

    # -- detection delay ----------------------------------------------------
    evaluation.skipped["detection_delay"] = (
        "not computed: the label contract carries no anomaly onset timestamp, and "
        "deriving one from the same features that are being modelled would leak the answer"
    )

    return evaluation


def trivial_baselines(
    y_true: Optional[np.ndarray],
    n_samples: int,
    quantile_threshold: Optional[float] = None,
    scores: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Any]]:
    """Reference baselines used for context (never presented as the model)."""
    results: Dict[str, Dict[str, Any]] = {}

    no_alert = np.zeros(n_samples, dtype=bool)
    if y_true is not None and len(y_true) == n_samples:
        results["no_alert"] = {
            "description": "flag nothing (no detector at all)",
            **_scores_from_flags(y_true, no_alert),
        }
    else:
        results["no_alert"] = {
            "description": "flag nothing (no detector at all)",
            "n_flagged": 0,
        }

    if quantile_threshold is not None and scores is not None and len(scores) == n_samples:
        flagged = np.asarray(scores, dtype="float64") <= float(quantile_threshold)
        entry = {
            "description": "train-quantile threshold (operational heuristic, no labels used)",
            "threshold": float(quantile_threshold),
            "n_flagged": int(flagged.sum()),
        }
        if y_true is not None and len(y_true) == n_samples:
            entry.update(_scores_from_flags(y_true, flagged))
        results["train_quantile"] = entry

    return results


def format_metrics_markdown(report: Dict[str, Any]) -> str:
    """Human-readable summary of a metrics report."""
    lines: List[str] = []
    lines.append("# Baseline Evaluation Report")
    lines.append("")
    lines.append(f"- Model: {report.get('model', {}).get('type', 'unknown')}")
    lines.append(f"- Score: {report.get('score', {}).get('score_name', 'unknown')} "
                 f"({report.get('score', {}).get('score_direction', 'unknown')})")
    lines.append(f"- Threshold: {report.get('threshold', {}).get('threshold', 'n/a')} "
                 f"(method: {report.get('threshold', {}).get('method', 'n/a')}, "
                 f"source: {report.get('threshold', {}).get('source_split', 'n/a')})")
    lines.append(f"- Training policy: {report.get('training', {}).get('data_policy', 'n/a')}")
    lines.append(f"- Generated: {report.get('generated_at', 'n/a')}")
    lines.append("")
    lines.append("> These are **synthetic-data** results. They demonstrate that the pipeline")
    lines.append("> works end to end; they do **not** establish real-world detection quality.")
    lines.append("")

    for split, payload in report.get("splits", {}).items():
        metrics = payload.get("metrics", {})
        lines.append(f"## Split: {split}")
        lines.append("")
        lines.append(f"- Samples: {payload.get('n_samples')} | positives: {payload.get('n_positive')} "
                     f"| negatives: {payload.get('n_negative')}")
        lines.append(f"- Flagged: {metrics.get('n_flagged')} "
                     f"({metrics.get('flagged_rate', 0.0):.1%})")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        for key in ["precision", "recall", "f1", "roc_auc", "pr_auc",
                    "tp", "fp", "tn", "fn",
                    "false_positive_count", "false_positive_rate", "prevalence",
                    "alerts_per_host_hour", "score_min", "score_max", "score_mean"]:
            if key in metrics and metrics[key] is not None:
                value = metrics[key]
                lines.append(f"| {key} | {value:.4f} |" if isinstance(value, float)
                             else f"| {key} | {value} |")
        skipped = payload.get("skipped_metrics", {})
        if skipped:
            lines.append("")
            lines.append("**Skipped metrics**")
            lines.append("")
            for key, reason in skipped.items():
                lines.append(f"- `{key}`: {reason}")
        baselines = payload.get("baselines", {})
        if baselines:
            lines.append("")
            lines.append("**Reference baselines**")
            lines.append("")
            lines.append("| Baseline | Precision | Recall | F1 | Flagged |")
            lines.append("| --- | --- | --- | --- | --- |")
            for name, entry in baselines.items():
                lines.append(
                    f"| {name} | {entry.get('precision', 'n/a')} | {entry.get('recall', 'n/a')} "
                    f"| {entry.get('f1', 'n/a')} | {entry.get('n_flagged', 'n/a')} |"
                )
        lines.append("")
    return "\n".join(lines)
