"""Evaluation for network/service anomaly detection.

The evaluation is deliberately split into separate questions:

* **How well does the detector rank/flag labelled windows?**  (overall
  precision/recall/F1/ROC-AUC/PR-AUC, computed only on *scored* windows)
* **Which scenario families drive the result?**  (per-scenario precision,
  recall and false-positive rate)
* **What does benign traffic cost?**  (false-positive rate on benign
  scenarios, with telemetry-damaged windows reported separately)
* **How fast is detection?**  (minutes from scenario onset to first flagged
  window, per run)
* **What could not be scored, and why?**  (status/reason accounting; an
  unscored window is never counted as a true negative)

Two detectors are compared: the Isolation Forest decision and the transparent
rule/MAD baseline.  They are reported separately; the baseline never rewrites
the model decision.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import math

import numpy as np
import pandas as pd

from ..evaluation import evaluate_split

#: Scenario families.  Incident scenarios contain positive labels; benign
#: scenarios contain only negatives (and are therefore the false-positive
#: test); telemetry scenarios describe damaged telemetry rather than traffic.
INCIDENT_SCENARIOS = frozenset({
    "http_5xx_burst",
    "db_timeout_latency",
    "auth_failure_burst",
})
BENIGN_SCENARIOS = frozenset({
    "normal",
    "normal_recovery",
    "benign_high_traffic",
    "benign_latency_shift",
    "benign_restart",
})
TELEMETRY_SCENARIOS = frozenset({"malformed_telemetry"})
ALL_SCENARIOS = INCIDENT_SCENARIOS | BENIGN_SCENARIOS | TELEMETRY_SCENARIOS

SPLIT_NAMES = ("train", "validation", "test")
_REQUIRED_COLUMNS = frozenset({
    "split", "sample_key", "window_start", "run_id", "scenario_name",
    "is_anomaly_true", "telemetry_issue", "status", "reason_code",
    "raw_score", "is_anomaly", "baseline_is_anomaly", "baseline_raw_score",
})


def _truthy(value: Any) -> bool:
    """Convert a scalar flag to ``bool`` without pandas' downcast warning.

    ``None``, ``NaN`` and ``pd.NA`` all mean "flag absent" and become
    ``False``.  A missing flag is never evidence of an anomaly.
    """
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):  # pragma: no cover - non-scalar input
        pass
    return bool(value)


def _as_bool(values: Any) -> pd.Series:
    """Coerce a column of nullable flags to a plain boolean ``Series``."""
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if series.dtype == bool:
        return series
    return series.map(_truthy).astype(bool)


def evaluate_split_predictions(
    frame: pd.DataFrame,
    split_name: str,
) -> Dict[str, Any]:
    """Evaluate one split's prediction table.

    ``frame`` must contain the columns produced by
    :func:`anomaly_detector.network.pipeline.build_predictions`.  Rows whose
    status is not ``scored`` are excluded from supervised metrics and reported
    separately — they are *unknown*, not normal.
    """
    missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"prediction table is missing required columns: {missing}")
    if frame.empty:
        return {
            "split": split_name,
            "n_windows": 0,
            "n_scored": 0,
            "n_unscored": 0,
            "overall": None,
            "warnings": ["no windows in this split"],
        }

    frame = frame.copy()
    frame["is_anomaly_true"] = _as_bool(frame["is_anomaly_true"])
    frame["telemetry_issue"] = _as_bool(frame["telemetry_issue"])
    scored_mask = frame["status"].astype(str) == "scored"
    scored = frame[scored_mask].copy()
    unscored = frame[~scored_mask].copy()

    report: Dict[str, Any] = {
        "split": split_name,
        "n_windows": int(len(frame)),
        "n_scored": int(len(scored)),
        "n_unscored": int(len(unscored)),
        "scored_fraction": float(len(scored) / len(frame)) if len(frame) else 0.0,
        "n_incident_windows": int(frame["is_anomaly_true"].sum()),
        "status_counts": _counts(frame["status"].astype(str)),
        "reason_counts": _counts(frame["reason_code"].astype(str)),
        "scenario_counts": _counts(frame["scenario_name"].astype(str)),
        "warnings": [],
    }

    if scored.empty:
        report["overall"] = None
        report["warnings"].append(
            "no window in this split could be scored; supervised metrics are not defined"
        )
    else:
        report["overall"] = _overall(scored, "is_anomaly", "raw_score")
        report["overall"]["coverage"] = {
            "n_windows": int(len(frame)),
            "n_scored": int(len(scored)),
            "n_unscored": int(len(unscored)),
            "scored_fraction": report["scored_fraction"],
        }

    report["baseline_overall"] = (
        _overall(scored, "baseline_is_anomaly", "baseline_raw_score")
        if not scored.empty else None
    )
    report["per_scenario"] = _per_scenario(scored)
    report["benign"] = _benign(scored)
    report["incident"] = _incident(scored)
    report["telemetry"] = _telemetry(frame, scored, unscored)
    report["detection_delay"] = detection_delays(scored)
    report["model_vs_baseline"] = _model_vs_baseline(scored)
    report["unscored_incident_windows"] = _unscored_positives(unscored)

    if report["unscored_incident_windows"]:
        report["warnings"].append(
            f"{len(report['unscored_incident_windows'])} labelled incident window(s) "
            "could not be scored; they are reported as unknown, never as true negatives"
        )
    if len(scored) and scored["is_anomaly_true"].nunique() < 2:
        report["warnings"].append(
            "the scored subset contains only one ground-truth class; "
            "ROC-AUC and PR-AUC are not defined"
        )
    return report


def evaluate_strategy_predictions(predictions: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate every split in a strategy's prediction table."""
    if "split" not in predictions.columns:
        raise ValueError("prediction table is missing the 'split' column")
    splits: Dict[str, Any] = {}
    for name in SPLIT_NAMES:
        subset = predictions[predictions["split"] == name]
        if subset.empty:
            continue
        splits[name] = evaluate_split_predictions(subset.reset_index(drop=True), name)

    combined = {
        "generated_at_utc": None,
        "score_definition": (
            "IsolationForest.score_samples: lower is more anomalous; "
            "is_anomaly = raw_score <= threshold"
        ),
        "splits": splits,
        "comparison": _strategy_comparison(splits),
        "notes": [
            "Metrics are computed on scored windows only.",
            "Unscored windows are reported separately and never treated as normal.",
            "The transparent rule baseline is evaluated independently; it does not "
            "change the Isolation Forest decision.",
            "Synthetic labels validate the pipeline, not real-world detection quality.",
        ],
    }
    return combined


def detection_delays(scored: pd.DataFrame) -> Dict[str, Any]:
    """Minutes from a scenario's first labelled window to the first flag.

    Only scored rows participate.  A missed run is counted as missed, not as a
    zero-minute delay.  Pre-onset flags are ignored for the delay calculation
    (they are captured by the false-positive metrics instead).
    """
    groups: List[Dict[str, Any]] = []
    if scored.empty:
        return {"n_groups": 0, "n_detected": 0, "n_missed": 0, "delays_minutes": [], "runs": []}

    scored = scored.copy()
    scored["window_start"] = pd.to_datetime(scored["window_start"], utc=True)
    positives = scored[scored["is_anomaly_true"].astype(bool)]
    for (run_id, scenario), group in positives.groupby(["run_id", "scenario_name"], dropna=False):
        if scenario not in INCIDENT_SCENARIOS:
            continue
        onset = group["window_start"].min()
        end = group["window_start"].max()
        detected = scored[
            (scored["run_id"] == run_id)
            & _as_bool(scored["is_anomaly"])
            & (scored["window_start"] >= onset)
            & (scored["window_start"] <= end)
        ].sort_values("window_start")
        if detected.empty:
            groups.append({
                "run_id": str(run_id),
                "scenario": str(scenario),
                "onset": onset.isoformat(),
                "delay_minutes": None,
                "detected": False,
            })
        else:
            delay = (detected.iloc[0]["window_start"] - onset).total_seconds() / 60.0
            groups.append({
                "run_id": str(run_id),
                "scenario": str(scenario),
                "onset": onset.isoformat(),
                "first_detection": detected.iloc[0]["window_start"].isoformat(),
                "delay_minutes": float(delay),
                "detected": True,
            })

    delays = [g["delay_minutes"] for g in groups if g["detected"]]
    return {
        "n_groups": len(groups),
        "n_detected": len(delays),
        "n_missed": len(groups) - len(delays),
        "delays_minutes": [float(v) for v in delays],
        "mean_delay_minutes": float(np.mean(delays)) if delays else None,
        "median_delay_minutes": float(np.median(delays)) if delays else None,
        "max_delay_minutes": float(np.max(delays)) if delays else None,
        "runs": groups,
    }


def format_network_metrics_markdown(report: Mapping[str, Any]) -> str:
    """Human-readable report for the network detector."""
    lines: List[str] = []
    lines.append("# Network Anomaly Detection — Evaluation")
    lines.append("")
    lines.append("> Synthetic, deterministic data. These numbers validate the pipeline")
    lines.append("> and the evaluation discipline; they do not establish real-world quality.")
    lines.append("")
    if report.get("strategy"):
        lines.append(f"- Strategy: `{report['strategy']}`")
    if report.get("feature_schema_version"):
        lines.append(f"- Feature schema: `{report['feature_schema_version']}`")
    if report.get("contract_version"):
        lines.append(f"- Contract version: `{report['contract_version']}`")
    lines.append("")

    for split, payload in (report.get("splits") or {}).items():
        lines.append(f"## Split: {split}")
        lines.append("")
        lines.append(
            f"- Windows: {payload.get('n_windows')} | scored: {payload.get('n_scored')} "
            f"| unscored: {payload.get('n_unscored')} "
            f"({payload.get('scored_fraction', 0.0):.1%} coverage)"
        )
        overall = payload.get("overall") or {}
        if overall:
            lines.append(
                f"- Model: precision {_fmt(overall.get('precision'))}, "
                f"recall {_fmt(overall.get('recall'))}, "
                f"F1 {_fmt(overall.get('f1'))}, "
                f"ROC-AUC {_fmt(overall.get('roc_auc'))}, "
                f"PR-AUC {_fmt(overall.get('pr_auc'))}"
            )
            lines.append(
                f"- Confusion: TP {overall.get('tp')} | FP {overall.get('fp')} "
                f"| FN {overall.get('fn')} | TN {overall.get('tn')}"
            )
        baseline = payload.get("baseline_overall") or {}
        if baseline:
            lines.append(
                f"- Rule baseline: precision {_fmt(baseline.get('precision'))}, "
                f"recall {_fmt(baseline.get('recall'))}, "
                f"F1 {_fmt(baseline.get('f1'))}"
            )
        lines.append("")

        benign = payload.get("benign") or {}
        if benign:
            lines.append(
                f"**Benign false-positive rate:** {_fmt(benign.get('false_positive_rate'))} "
                f"({benign.get('false_positive_count')} of {benign.get('n_negatives')} "
                "clean benign windows)"
            )
            lines.append("")

        per_scenario = payload.get("per_scenario") or {}
        if per_scenario:
            lines.append("| Scenario | Windows | Positives | Flagged | Precision | Recall | F1 | FPR |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for name in sorted(per_scenario):
                entry = per_scenario[name]
                lines.append(
                    f"| `{name}` | {entry.get('n')} | {entry.get('n_positive')} "
                    f"| {entry.get('n_flagged')} | {_fmt(entry.get('precision'))} "
                    f"| {_fmt(entry.get('recall'))} | {_fmt(entry.get('f1'))} "
                    f"| {_fmt(entry.get('false_positive_rate'))} |"
                )
            lines.append("")

        delay = payload.get("detection_delay") or {}
        if delay.get("n_groups"):
            lines.append(
                f"**Detection delay:** {delay.get('n_detected')}/{delay.get('n_groups')} "
                f"incident runs detected; mean {_fmt(delay.get('mean_delay_minutes'))} min, "
                f"median {_fmt(delay.get('median_delay_minutes'))} min, "
                f"max {_fmt(delay.get('max_delay_minutes'))} min"
            )
            lines.append("")

        telemetry = payload.get("telemetry") or {}
        if telemetry:
            lines.append(
                f"**Telemetry-damaged windows:** {telemetry.get('n_windows')} total, "
                f"{telemetry.get('n_scored')} scored, "
                f"{telemetry.get('n_unscored')} unscored, "
                f"{telemetry.get('n_flagged')} flagged"
            )
            lines.append("")

        warnings = payload.get("warnings") or []
        if warnings:
            lines.append("**Warnings**")
            lines.append("")
            for warning in warnings:
                lines.append(f"- {warning}")
            lines.append("")

    comparison = report.get("comparison") or {}
    if comparison:
        lines.append("## Strategy comparison")
        lines.append("")
        lines.append("| Strategy | Split | Precision | Recall | F1 | Flagged | Scored |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for name, entry in comparison.items():
            for split, values in entry.items():
                lines.append(
                    f"| `{name}` | `{split}` | {_fmt(values.get('precision'))} "
                    f"| {_fmt(values.get('recall'))} | {_fmt(values.get('f1'))} "
                    f"| {values.get('n_flagged')} | {values.get('n_scored')} |"
                )
        lines.append("")

    notes = report.get("notes") or []
    if notes:
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _overall(scored: pd.DataFrame, flag_column: str, score_column: str) -> Dict[str, Any]:
    scores = pd.to_numeric(scored[score_column], errors="coerce")
    flags = _as_bool(scored[flag_column])
    evaluation = evaluate_split(
        scores=scores.to_numpy(dtype="float64"),
        labels=scored["is_anomaly_true"].astype(bool).tolist(),
        is_anomaly=flags.to_numpy(dtype=bool),
        split=str(scored["split"].iloc[0]) if "split" in scored else "unknown",
    )
    payload = _json_safe(evaluation.to_dict())
    # ``evaluate_split`` nests the numeric metrics under ``metrics``; flatten
    # them so every report consumer reads the same top-level shape while the
    # skipped-metric explanations remain available.
    metrics = payload.pop("metrics", {}) or {}
    return {**payload, **metrics}


def _per_scenario(scored: pd.DataFrame) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if scored.empty:
        return result
    for scenario, group in scored.groupby("scenario_name", dropna=False):
        y_true = group["is_anomaly_true"].astype(bool).to_numpy()
        y_pred = _as_bool(group["is_anomaly"]).to_numpy()
        y_base = _as_bool(group["baseline_is_anomaly"]).to_numpy()
        result[str(scenario)] = {
            "family": _family(str(scenario)),
            "n": int(len(group)),
            "n_positive": int(y_true.sum()),
            "n_negative": int((~y_true).sum()),
            "n_flagged": int(y_pred.sum()),
            "flag_rate": float(y_pred.mean()) if len(y_pred) else 0.0,
            **_flag_metrics(y_true, y_pred),
            "baseline": {
                "n_flagged": int(y_base.sum()),
                **_flag_metrics(y_true, y_base),
            },
        }
    return result


def _benign(scored: pd.DataFrame) -> Dict[str, Any]:
    if scored.empty:
        return {}
    benign = scored[
        scored["scenario_name"].isin(BENIGN_SCENARIOS)
        & ~scored["telemetry_issue"].astype(bool)
    ]
    if benign.empty:
        return {"n": 0, "n_negatives": 0, "false_positive_count": 0,
                "false_positive_rate": None}
    y_true = benign["is_anomaly_true"].astype(bool).to_numpy()
    y_pred = _as_bool(benign["is_anomaly"]).to_numpy()
    y_base = _as_bool(benign["baseline_is_anomaly"]).to_numpy()
    metrics = _flag_metrics(y_true, y_pred)
    return {
        "n": int(len(benign)),
        "n_negatives": int((~y_true).sum()),
        "n_flagged": int(y_pred.sum()),
        "false_positive_count": int(metrics["fp"]),
        "false_positive_rate": (
            float(metrics["fp"] / (~y_true).sum()) if (~y_true).sum() else None
        ),
        "baseline": {
            "n_flagged": int(y_base.sum()),
            "false_positive_count": int(_flag_metrics(y_true, y_base)["fp"]),
            "false_positive_rate": (
                float(_flag_metrics(y_true, y_base)["fp"] / (~y_true).sum())
                if (~y_true).sum() else None
            ),
        },
    }


def _incident(scored: pd.DataFrame) -> Dict[str, Any]:
    if scored.empty:
        return {}
    incident = scored[scored["scenario_name"].isin(INCIDENT_SCENARIOS)]
    if incident.empty:
        return {"n": 0, "n_positive": 0, "recall": None}
    y_true = incident["is_anomaly_true"].astype(bool).to_numpy()
    y_pred = _as_bool(incident["is_anomaly"]).to_numpy()
    y_base = _as_bool(incident["baseline_is_anomaly"]).to_numpy()
    return {
        "n": int(len(incident)),
        "n_positive": int(y_true.sum()),
        "n_missed": int((y_true & ~y_pred).sum()),
        **_flag_metrics(y_true, y_pred),
        "baseline": _flag_metrics(y_true, y_base),
    }


def _telemetry(
    frame: pd.DataFrame,
    scored: pd.DataFrame,
    unscored: pd.DataFrame,
) -> Dict[str, Any]:
    damaged = frame[frame["telemetry_issue"].astype(bool)]
    damaged_scored = scored[scored["telemetry_issue"].astype(bool)]
    damaged_unscored = unscored[unscored["telemetry_issue"].astype(bool)]
    other_unscored = unscored[~unscored["telemetry_issue"].astype(bool)]
    return {
        "n_windows": int(len(damaged)),
        "n_scored": int(len(damaged_scored)),
        "n_unscored": int(len(damaged_unscored)),
        "n_flagged": int(_as_bool(damaged_scored["is_anomaly"]).sum()),
        "scored_status_counts": _counts(damaged_scored["status"].astype(str)),
        "unscored_reason_counts": _counts(damaged_unscored["reason_code"].astype(str)),
        "non_telemetry_unscored": {
            "n": int(len(other_unscored)),
            "reason_counts": _counts(other_unscored["reason_code"].astype(str)),
        },
    }


def _model_vs_baseline(scored: pd.DataFrame) -> Dict[str, Any]:
    if scored.empty:
        return {}
    model = _as_bool(scored["is_anomaly"])
    baseline = _as_bool(scored["baseline_is_anomaly"])
    return {
        "both": int((model & baseline).sum()),
        "model_only": int((model & ~baseline).sum()),
        "baseline_only": int((~model & baseline).sum()),
        "neither": int((~model & ~baseline).sum()),
    }


def _unscored_positives(unscored: pd.DataFrame) -> List[Dict[str, Any]]:
    if unscored.empty:
        return []
    positives = unscored[unscored["is_anomaly_true"].astype(bool)]
    return [
        {
            "sample_key": str(row.get("sample_key")),
            "status": str(row.get("status")),
            "reason_code": str(row.get("reason_code")),
        }
        for row in positives.to_dict(orient="records")
    ]


def _strategy_comparison(splits: Mapping[str, Any]) -> Dict[str, Any]:
    comparison: Dict[str, Any] = {}
    for detector, prefix in (("isolation_forest", "overall"),
                             ("rule_baseline", "baseline_overall")):
        comparison[detector] = {
            name: {
                "precision": (payload.get(prefix) or {}).get("precision"),
                "recall": (payload.get(prefix) or {}).get("recall"),
                "f1": (payload.get(prefix) or {}).get("f1"),
                "n_flagged": (payload.get(prefix) or {}).get("n_flagged"),
                "n_scored": payload.get("n_scored"),
            }
            for name, payload in splits.items()
        }
    return comparison


def _flag_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": (fp / (fp + tn)) if (fp + tn) else None,
    }


def _family(scenario: str) -> str:
    if scenario in INCIDENT_SCENARIOS:
        return "incident"
    if scenario in BENIGN_SCENARIOS:
        return "benign"
    if scenario in TELEMETRY_SCENARIOS:
        return "telemetry"
    return "unknown"


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    return dict(Counter(str(value) for value in values))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def dumps(report: Mapping[str, Any]) -> str:
    """JSON-serialise a report with a stable, readable layout."""
    return json.dumps(_json_safe(report), indent=2, sort_keys=True)
