#!/usr/bin/env python3
"""Layer D — blind holdout on an unseen simulator seed.

The **already-deployed** artifact in ``artifacts/network/model`` is loaded and
scored against a freshly simulated dataset generated with a seed that was not
used for model fitting or threshold selection.  Nothing is retrained, no
threshold is re-selected, and no label is passed to the detector.

This is still developer-authored synthetic data.  It tests generalisation to a
new random draw of the same generator, **not** to a real production system.

Example::

    python scripts/run_blind_holdout.py \
        --seed 20261026 \
        --model artifacts/network/model/model.joblib \
        --manifest artifacts/network/model/manifest.json \
        --out /tmp/opencode/dual-e2e/layer-d
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from anomaly_detector.network.config import load_network_config  # noqa: E402
from anomaly_detector.network.evaluation import (  # noqa: E402
    detection_delays,
    evaluate_split_predictions,
)
from anomaly_detector.network.features import (  # noqa: E402
    build_network_windows,
    join_windows_with_labels,
)
from anomaly_detector.network.inference import load_network_detector  # noqa: E402
from anomaly_detector.network.lad_adapter import (  # noqa: E402
    LadParseOptions,
    parse_lad_events,
)
from anomaly_detector.network.pipeline import build_predictions  # noqa: E402
from anomaly_detector.network.simulator import simulate  # noqa: E402
from anomaly_detector.network.splits import SplitAssignment  # noqa: E402


def _window_grid(labels: Sequence[Any], window_size_minutes: int) -> Dict[Any, Any]:
    grid: Dict[Any, Any] = {}
    for label in labels:
        grid.setdefault((label.host_id, label.service), []).append(label.window_start)
    return {key: sorted(set(starts)) for key, starts in grid.items()}


def _hash_file(path: str) -> Optional[str]:
    import hashlib

    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.path.join(_REPO_ROOT, "configs", "network.yaml"))
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--run-count", type=int, default=3)
    parser.add_argument(
        "--model",
        default=os.path.join(_REPO_ROOT, "artifacts", "network", "model", "model.joblib"),
    )
    parser.add_argument(
        "--manifest",
        default=os.path.join(_REPO_ROOT, "artifacts", "network", "model", "manifest.json"),
    )
    parser.add_argument("--out", default=os.path.join(_REPO_ROOT, "artifacts", "audit", "layer_d"))
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    config = load_network_config(args.config)
    simulator_config = replace(
        config.simulator.to_simulator_config(),
        seed=args.seed,
        run_count=max(3, args.run_count),
    )

    started = datetime.now().astimezone()
    dataset = simulate(simulator_config)

    parse_result = parse_lad_events(
        dataset.events,
        LadParseOptions(max_events=simulator_config.max_events),
    )
    grid = _window_grid(dataset.labels, simulator_config.window_size_minutes)
    windows = build_network_windows(
        parse_result.events,
        quality_by_window=parse_result.quality_by_window(
            simulator_config.window_size_minutes
        ),
        window_size_minutes=simulator_config.window_size_minutes,
        rolling_windows=config.rules.rolling_windows,
        rolling_min_history=config.rules.rolling_min_history,
        window_grid=grid,
    )
    aligned_windows, aligned_labels, missing = join_windows_with_labels(
        windows, dataset.labels
    )

    detector = load_network_detector(
        model_path=args.model,
        manifest_path=args.manifest,
        rules=config.rules,
    )
    results = detector.score_windows(aligned_windows)

    assignment = SplitAssignment(
        strategy="blind-holdout",
        seed=args.seed,
        indices={"train": [], "validation": [], "test": list(range(len(aligned_windows)))},
        runs={"train": [], "validation": [], "test": sorted(dataset.runs)},
        integrity={"note": "no row was used for fitting or threshold selection"},
    )
    frame = build_predictions(
        aligned_windows,
        aligned_labels,
        results,
        assignment,
        strategy="blind-holdout",
    )

    metrics = evaluate_split_predictions(frame, "blind_holdout")
    delays = detection_delays(frame[frame["status"].astype(str) == "scored"])
    report = {
        "layer": "D",
        "note": (
            "Blind holdout: the deployed artifact scored a new simulator seed. "
            "No training, tuning or threshold selection was performed. "
            "Developer-authored synthetic data, not real-world validation."
        ),
        "seed": args.seed,
        "run_count": simulator_config.run_count,
        "started_at": started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "model": {
            "path": args.model,
            "model_sha256": _hash_file(args.model),
            "manifest_path": args.manifest,
            "manifest_sha256": _hash_file(args.manifest),
            "name": detector.model_name,
            "version": detector.model_version,
            "artifact_version": detector.model_artifact_version,
            "threshold": detector.threshold,
            "score_direction": detector.score_direction,
            "feature_schema_version": detector.feature_schema_version,
            "contract_version": detector.contract()["contract_version"],
        },
        "dataset": {
            "runs": sorted(dataset.runs),
            "events": dataset.event_count,
            "labels": len(dataset.labels),
            "lad_parse": parse_result.summary(),
        },
        "alignment": {
            "aligned_windows": len(aligned_windows),
            "missing_label_windows": len(missing),
        },
        "metrics": metrics,
        "detection_delays": delays,
    }

    with open(os.path.join(args.out, "blind_results.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
    frame.to_csv(os.path.join(args.out, "predictions.csv"), index=False)

    summary = {
        "seed": args.seed,
        "model_sha256": report["model"]["model_sha256"],
        "n_windows": metrics.get("n_windows"),
        "n_scored": metrics.get("n_scored"),
        "n_unscored": metrics.get("n_unscored"),
        "overall": (metrics.get("overall") or {}),
        "benign": metrics.get("benign"),
        "incident": metrics.get("incident"),
        "detection_delays": {
            key: delays.get(key)
            for key in ("n_groups", "n_detected", "n_missed", "mean_delay_minutes")
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
