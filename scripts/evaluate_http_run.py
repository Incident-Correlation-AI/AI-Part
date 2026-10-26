#!/usr/bin/env python3
"""Evaluate an HTTP run against evaluator-only labels.

The labels file is never sent to the detector.  This script joins it to the
detector response **after** the run and reports separate metrics for:

* scored vs unscored windows (unscored is never treated as normal);
* each scenario;
* each service;
* detection delay per (run, scenario).

It is used for Layer A (synthetic simulator regression) and Layer B
(controlled local-service telemetry).  It does not manufacture labels.

Example::

    python scripts/evaluate_http_run.py \
        --labels /tmp/opencode/dual-e2e/layer-a/labels.csv \
        --response /tmp/opencode/dual-e2e/layer-a/detector_response.json \
        --out /tmp/opencode/dual-e2e/layer-a/evaluation.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _key(host: Any, service: Any, window_start: Any) -> str:
    return f"{host}|{service}|{window_start}"


def _aware(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _prf(tp: int, fp: int, fn: int, tn: int) -> Dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )
    fpr = fp / (fp + tn) if (fp + tn) else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
    }


def evaluate(
    labels: Sequence[Mapping[str, Any]],
    response: Mapping[str, Any],
) -> Dict[str, Any]:
    results = response.get("results") or []
    result_by_key = {str(item.get("window_id")): item for item in results}

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for label in labels:
        key = _key(label.get("host_id"), label.get("service"), label.get("window_start"))
        result = result_by_key.get(key)
        if result is None:
            missing.append(key)
            continue
        status = str(result.get("status") or "unknown")
        scored = status == "scored"
        flagged = bool(result.get("decision", {}).get("is_anomaly")) if scored else None
        true = _bool(label.get("is_anomaly"))
        rows.append(
            {
                "key": key,
                "host_id": label.get("host_id"),
                "service": label.get("service"),
                "window_start": label.get("window_start"),
                "scenario": label.get("scenario_name"),
                "run_id": label.get("run_id"),
                "telemetry_issue": _bool(label.get("telemetry_issue")),
                "is_anomaly_true": true,
                "status": status,
                "is_anomaly_pred": flagged,
                "raw_score": result.get("score", {}).get("raw_score"),
                "threshold": result.get("decision", {}).get("threshold"),
                "severity": result.get("extra", {}).get("network", {}).get("severity"),
            }
        )

    scored_rows = [row for row in rows if row["status"] == "scored"]
    unscored_rows = [row for row in rows if row["status"] != "scored"]

    tp = sum(1 for row in scored_rows if row["is_anomaly_true"] and row["is_anomaly_pred"])
    fp = sum(1 for row in scored_rows if not row["is_anomaly_true"] and row["is_anomaly_pred"])
    fn = sum(1 for row in scored_rows if row["is_anomaly_true"] and not row["is_anomaly_pred"])
    tn = sum(1 for row in scored_rows if not row["is_anomaly_true"] and not row["is_anomaly_pred"])
    overall = _prf(tp, fp, fn, tn)
    overall["n_scored"] = len(scored_rows)
    overall["n_unscored"] = len(unscored_rows)
    overall["n_missing"] = len(missing)
    overall["n_labeled_windows"] = len(labels)

    by_scenario: Dict[str, Dict[str, Any]] = {}
    for row in scored_rows:
        scenario = str(row["scenario"] or "unknown")
        entry = by_scenario.setdefault(
            scenario,
            {"n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "flagged": 0},
        )
        entry["n"] += 1
        entry["flagged"] += int(bool(row["is_anomaly_pred"]))
        if row["is_anomaly_true"] and row["is_anomaly_pred"]:
            entry["tp"] += 1
        elif not row["is_anomaly_true"] and row["is_anomaly_pred"]:
            entry["fp"] += 1
        elif row["is_anomaly_true"] and not row["is_anomaly_pred"]:
            entry["fn"] += 1
        else:
            entry["tn"] += 1
    for entry in by_scenario.values():
        entry.update(_prf(entry["tp"], entry["fp"], entry["fn"], entry["tn"]))

    by_service: Dict[str, Dict[str, Any]] = {}
    for row in scored_rows:
        service = str(row["service"] or "unknown")
        entry = by_service.setdefault(
            service, {"n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
        )
        entry["n"] += 1
        if row["is_anomaly_true"] and row["is_anomaly_pred"]:
            entry["tp"] += 1
        elif not row["is_anomaly_true"] and row["is_anomaly_pred"]:
            entry["fp"] += 1
        elif row["is_anomaly_true"] and not row["is_anomaly_pred"]:
            entry["fn"] += 1
        else:
            entry["tn"] += 1
    for entry in by_service.values():
        entry.update(_prf(entry["tp"], entry["fp"], entry["fn"], entry["tn"]))

    delays: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in scored_rows:
        if not row["is_anomaly_true"]:
            continue
        groups.setdefault((str(row["run_id"]), str(row["scenario"])), []).append(row)
    for (run_id, scenario), group in sorted(groups.items()):
        onset = min(_aware(row["window_start"]) for row in group if _aware(row["window_start"]))
        detected = [
            row for row in group if row["is_anomaly_pred"] and _aware(row["window_start"])
        ]
        if not detected:
            delays.append(
                {
                    "run_id": run_id,
                    "scenario": scenario,
                    "detected": False,
                    "delay_minutes": None,
                }
            )
            continue
        first = min(_aware(row["window_start"]) for row in detected)
        delays.append(
            {
                "run_id": run_id,
                "scenario": scenario,
                "detected": True,
                "first_detection": first.isoformat() if first else None,
                "delay_minutes": (first - onset).total_seconds() / 60.0 if first and onset else None,
            }
        )

    telemetry_rows = [row for row in rows if row["telemetry_issue"]]
    telemetry_summary = {
        "n": len(telemetry_rows),
        "scored": sum(1 for row in telemetry_rows if row["status"] == "scored"),
        "unscored": sum(1 for row in telemetry_rows if row["status"] != "scored"),
        "flagged": sum(1 for row in telemetry_rows if row["is_anomaly_pred"]),
        "by_status": {},
    }
    for row in telemetry_rows:
        key = str(row["status"])
        telemetry_summary["by_status"][key] = telemetry_summary["by_status"].get(key, 0) + 1

    return {
        "note": (
            "Unscored windows are excluded from supervised metrics and reported "
            "separately; they are never treated as normal. Labels were not sent "
            "to the detector."
        ),
        "overall": overall,
        "by_scenario": by_scenario,
        "by_service": by_service,
        "detection_delays": delays,
        "telemetry_issue_windows": telemetry_summary,
        "missing_labeled_windows": missing[:200],
        "missing_labeled_windows_count": len(missing),
        "unscored_by_status": _count(unscored_rows),
        "response_window_counts": response.get("windows"),
        "correlation_counts": (response.get("correlation") or {}).get("counts"),
        "explanation": response.get("explanation"),
        "rows": rows[:5000],
    }


def _count(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row["status"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def read_labels(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    labels = read_labels(args.labels)
    with open(args.response, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    response = payload.get("response", payload)
    report = evaluate(labels, response)

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
