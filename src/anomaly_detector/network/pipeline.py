"""End-to-end network anomaly-detection pipeline.

Stages
------
1. generate the deterministic local simulator dataset (events + separate labels);
2. parse the canonical LAD envelopes with explicit quality bookkeeping;
3. aggregate network feature windows (``network-features-v1``);
4. build time-aware and group-aware splits;
5. fit the Isolation Forest on **training-only, normal-only, clean** windows;
6. select the threshold on **validation** data only;
7. persist and reload the artifact, then score every window;
8. evaluate model and transparent baseline, write reproducible artifacts.

No step trains on validation/test data.  No label, scenario name, run id or
future value enters the feature matrix.  Unscored windows are kept as
``insufficient_data`` / ``invalid_input`` and are never counted as normal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import json
import os
import shutil

import numpy as np
import pandas as pd

from ..artifacts import build_manifest, git_revision, write_manifest, write_predictions
from ..baseline_model import BaselineDetector, TrainingReport, fit_baseline
from ..inference.contract import CONTRACT_VERSION, InferenceStatus
from ..model_contract import prepare_model_input
from ..thresholding import ThresholdResult, select_threshold
from .baseline import score_dataframe as score_baseline_dataframe
from .config import NetworkConfig, load_network_config
from .evaluation import (
    evaluate_strategy_predictions,
    format_network_metrics_markdown,
)
from .features import (
    NETWORK_FEATURE_COLUMNS,
    NETWORK_FEATURE_SCHEMA_VERSION,
    NetworkLabel,
    NetworkWindow,
    build_network_windows,
    join_windows_with_labels,
    labels_to_dataframe,
    windows_to_dataframe,
)
from .inference import (
    NetworkDetector,
    load_network_detector,
)
from .lad_adapter import LadParseOptions, LadParseResult, parse_lad_events
from .simulator import SimulatedDataset, simulate
from .splits import SplitAssignment, make_split_assignment

PRIMARY_STRATEGY = "time"


@dataclass
class StrategyRun:
    strategy: str
    assignment: SplitAssignment
    detector: NetworkDetector
    manifest: Dict[str, Any]
    threshold: ThresholdResult
    training_report: TrainingReport
    predictions: pd.DataFrame
    metrics: Dict[str, Any]
    paths: Dict[str, str] = field(default_factory=dict)


@dataclass
class NetworkRunResult:
    config: NetworkConfig
    dataset: SimulatedDataset
    parse_result: LadParseResult
    windows: List[NetworkWindow]
    labels: List[NetworkLabel]
    features: pd.DataFrame
    metadata: pd.DataFrame
    labels_frame: pd.DataFrame
    strategies: Dict[str, StrategyRun]
    shared_paths: Dict[str, str] = field(default_factory=dict)
    primary_strategy: str = PRIMARY_STRATEGY


def run_network_pipeline(
    config: Optional[NetworkConfig] = None,
    strategies: Optional[Sequence[str]] = None,
) -> NetworkRunResult:
    """Run the complete network pipeline.  Deterministic for a fixed config."""
    config = config or load_network_config()
    strategies = list(strategies or config.split_strategies)
    if not strategies:
        raise ValueError("at least one split strategy is required")
    if PRIMARY_STRATEGY not in strategies:
        strategies = [PRIMARY_STRATEGY] + strategies

    _prepare_directories(config)
    dataset = simulate(config.simulator.to_simulator_config())
    shared = dataset.write(config.paths["synthetic_dir"])

    parse_result = parse_lad_events(
        dataset.events,
        LadParseOptions(max_events=config.simulator.max_events),
    )
    grid = _window_grid(dataset.labels, config.simulator.window_size_minutes)
    windows = build_network_windows(
        parse_result.events,
        quality_by_window=parse_result.quality_by_window(
            config.simulator.window_size_minutes
        ),
        window_size_minutes=config.simulator.window_size_minutes,
        rolling_windows=config.rules.rolling_windows,
        rolling_min_history=config.rules.rolling_min_history,
        window_grid=grid,
    )
    aligned_windows, aligned_labels, missing_labels = join_windows_with_labels(
        windows, dataset.labels
    )
    features, metadata = windows_to_dataframe(aligned_windows)
    labels_frame = labels_to_dataframe(aligned_labels)
    _assert_alignment(features, metadata, labels_frame)

    # Shared, non-secret inputs and join tables.
    feature_path = os.path.join(config.paths["output_dir"], "features.csv")
    metadata_path = os.path.join(config.paths["output_dir"], "metadata.csv")
    labels_path = os.path.join(config.paths["output_dir"], "labels.csv")
    features.to_csv(feature_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    labels_frame.to_csv(labels_path, index=False)
    _write_json(
        os.path.join(config.paths["output_dir"], "lad_parse_summary.json"),
        {
            "summary": parse_result.summary(),
            "missing_label_windows": missing_labels,
            "note": "Labels are evaluation-only and are not model inputs.",
        },
    )

    runs: Dict[str, StrategyRun] = {}
    for strategy in strategies:
        runs[strategy] = _run_strategy(
            config=config,
            strategy=strategy,
            windows=aligned_windows,
            labels=aligned_labels,
            features=features,
            metadata=metadata,
            dataset=dataset,
            parse_result=parse_result,
        )

    shared_paths = {
        "events": shared.get("events", ""),
        "labels": shared.get("labels", ""),
        "simulator_manifest": shared.get("manifest", ""),
        "features": feature_path,
        "metadata": metadata_path,
        "labels_join": labels_path,
        "output_dir": config.paths["output_dir"],
    }
    _write_combined_report(config, runs, shared_paths)
    return NetworkRunResult(
        config=config,
        dataset=dataset,
        parse_result=parse_result,
        windows=aligned_windows,
        labels=aligned_labels,
        features=features,
        metadata=metadata,
        labels_frame=labels_frame,
        strategies=runs,
        shared_paths=shared_paths,
        primary_strategy=PRIMARY_STRATEGY,
    )


def build_predictions(
    windows: Sequence[NetworkWindow],
    labels: Sequence[NetworkLabel],
    results: Sequence[Any],
    assignment: SplitAssignment,
    strategy: str,
) -> pd.DataFrame:
    """Join window metadata, evaluation labels and inference results.

    The label columns live only in this evaluation table.  They are never part
    of a feature window and are never read by the detector.
    """
    if not (len(windows) == len(labels) == len(results)):
        raise ValueError("windows, labels and results must have the same length")
    split_by_index: Dict[int, str] = {}
    for name, indices in assignment.indices.items():
        for index in indices:
            split_by_index[index] = name

    rows: List[Dict[str, Any]] = []
    for index, (window, label, result) in enumerate(zip(windows, labels, results)):
        network = dict((result.extra or {}).get("network") or {})
        baseline = dict(network.get("baseline") or {})
        rows.append({
            "strategy": strategy,
            "split": split_by_index.get(index, "unassigned"),
            "sample_key": window.sample_key,
            "host_id": window.metadata.host_id,
            "service": window.metadata.service,
            "window_start": window.metadata.window_start.isoformat(),
            "window_end": window.metadata.window_end.isoformat(),
            "window_size_minutes": window.metadata.window_size_minutes,
            "run_id": label.run_id,
            "scenario_name": label.scenario_name,
            "is_anomaly_true": bool(label.is_anomaly),
            "telemetry_issue": bool(label.telemetry_issue),
            "status": result.status.value,
            "reason_code": result.reason_code.value,
            "raw_score": result.raw_score,
            "is_anomaly": result.is_anomaly,
            "severity": network.get("severity"),
            "model_only_anomaly": network.get("model_only_anomaly"),
            "imputed_features": json.dumps(result.imputed_features, sort_keys=True),
            "baseline_is_anomaly": baseline.get("is_anomaly"),
            "baseline_risk_score": baseline.get("risk_score"),
            "baseline_raw_score": (
                -float(baseline["risk_score"])
                if baseline.get("risk_score") is not None else None
            ),
            "baseline_severity": baseline.get("severity"),
            "rule_hits": json.dumps(baseline.get("rule_hits") or [], sort_keys=True),
            "evidence_features": json.dumps(
                network.get("evidence_features") or {}, sort_keys=True
            ),
            "telemetry_quality": json.dumps(
                network.get("telemetry_quality") or {}, sort_keys=True
            ),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# One strategy
# ---------------------------------------------------------------------------


def _run_strategy(
    config: NetworkConfig,
    strategy: str,
    windows: Sequence[NetworkWindow],
    labels: Sequence[NetworkLabel],
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    dataset: SimulatedDataset,
    parse_result: LadParseResult,
) -> StrategyRun:
    assignment = make_split_assignment(
        windows,
        labels,
        strategy=strategy,
        ratios=config.split_ratios,
        seed=config.simulator.seed,
    )

    telemetry_issue = np.asarray([label.telemetry_issue for label in labels], dtype=bool)
    label_values = np.asarray([bool(label.is_anomaly) for label in labels], dtype=bool)
    event_counts = pd.to_numeric(features["event_count"], errors="coerce").fillna(0).to_numpy()

    train_indices = np.asarray(assignment.indices["train"], dtype=int)
    coverage = event_counts[train_indices] >= config.rules.min_window_events
    clean = ~telemetry_issue[train_indices]
    normal = ~label_values[train_indices]
    train_mask = coverage & clean & normal
    used_train = train_indices[train_mask]

    disease = {
        "rows_in_split": int(len(train_indices)),
        "excluded_labelled_anomaly": int(np.sum(~normal)),
        "excluded_telemetry_issue": int(np.sum(~clean)),
        "excluded_low_coverage": int(np.sum(~coverage)),
        "rows_fitted": int(len(used_train)),
    }
    if len(used_train) == 0:
        raise ValueError(
            "no clean, normal, adequately-covered training windows remain; "
            "the split policy or minimum-volume guard is too strict"
        )

    base_config = config.baseline_config()
    train_features = features.iloc[used_train]
    train_labels = pd.Series(label_values[used_train], index=train_features.index)
    detector, training_report = fit_baseline(train_features, train_labels, base_config)

    train_scores = _score_features(detector, train_features)
    val_indices = np.asarray(assignment.indices["validation"], dtype=int)
    val_mask = (
        (event_counts[val_indices] >= config.rules.min_window_events)
        & (~telemetry_issue[val_indices])
    )
    val_indices = val_indices[val_mask]
    if len(val_indices) == 0:
        raise ValueError("no scoreable validation windows remain for threshold selection")
    val_features = features.iloc[val_indices]
    val_scores = _score_features(detector, val_features)
    val_labels = label_values[val_indices].tolist()

    threshold_result = select_threshold(
        scores=val_scores,
        labels=val_labels,
        method=config.threshold.method,
        source_split="validation",
        train_scores=train_scores,
        min_precision=config.threshold.min_precision,
        min_recall=config.threshold.min_recall,
        fallback_method=config.threshold.fallback_method,
        fallback_quantile=config.threshold.fallback_quantile,
    )

    experiment_dir = os.path.join(
        config.paths["output_dir"], "experiments", strategy
    )
    model_dir = os.path.join(experiment_dir, "model")
    reports_dir = os.path.join(experiment_dir, "reports")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    manifest_path = os.path.join(model_dir, "manifest.json")

    split_info = {
        **assignment.to_dict(),
        "training_selection": disease,
        "threshold_source_split": threshold_result.source_split,
        "test_used_for_selection": False,
    }
    source = {
        "simulator": dataset.summary,
        "lad_parse": parse_result.summary(),
        "feature_schema_version": NETWORK_FEATURE_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
    }
    manifest = build_manifest(
        detector=detector,
        config=base_config,
        training_report=training_report,
        threshold_result=threshold_result,
        split_info=split_info,
        source=source,
    )
    manifest["feature_schema"]["feature_schema_version"] = NETWORK_FEATURE_SCHEMA_VERSION
    manifest["network"] = {
        "contract_version": CONTRACT_VERSION,
        "feature_schema_version": NETWORK_FEATURE_SCHEMA_VERSION,
        "window_size_minutes": config.simulator.window_size_minutes,
        "split_strategy": strategy,
        "simulator": dataset.summary,
        "lad_parse": parse_result.summary(),
        "rules": _asdict(config.rules),
        "training_selection": disease,
        "llm": config.llm.to_dict(),
        "explanation_policy": (
            "optional adapter; evidence-grounded; never changes the detection decision"
        ),
        "label_policy": (
            "labels/scenario names/run ids are evaluation-only and are not model inputs"
        ),
    }
    manifest["interpretation"]["llm_explanation_adapter"] = (
        "available as an optional, evidence-grounded downstream adapter; "
        "it does not run as part of scoring"
    )

    detector.save(model_path)
    write_manifest(manifest, manifest_path)

    # Reload through the public inference boundary so serialization and
    # manifest cross-checks are exercised on every run.
    loaded = load_network_detector(
        model_path=model_path,
        manifest_path=manifest_path,
        rules=config.rules,
    )
    results = loaded.score_windows(windows)
    predictions = build_predictions(windows, labels, results, assignment, strategy)

    metrics = evaluate_strategy_predictions(predictions)
    metrics["strategy"] = strategy
    metrics["feature_schema_version"] = NETWORK_FEATURE_SCHEMA_VERSION
    metrics["contract_version"] = CONTRACT_VERSION
    metrics["threshold"] = threshold_result.to_dict()
    metrics["split_integrity"] = {
        "strategy": strategy,
        "sizes": {name: len(assignment.indices[name]) for name in assignment.indices},
        "integrity": dict(assignment.integrity),
        "training_selection": disease,
    }
    metrics["model"] = {
        "type": manifest["model"]["type"],
        "parameters": manifest["model"]["parameters"],
        "random_seed": manifest.get("random_seed"),
        "code_revision": manifest.get("code_revision"),
        "feature_columns": list(detector.feature_columns),
        "n_features": len(detector.feature_columns),
        "score_direction": manifest["score"]["score_direction"],
    }
    metrics["notes"] = [
        "The train split is scored in-sample; its metrics are diagnostic only.",
        "Only scored windows enter supervised metrics; unscored windows are unknown.",
        "The rule baseline is evaluated independently and never rewrites the model decision.",
        "Synthetic labels validate the pipeline, not real-world detection quality.",
    ]
    metrics["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    paths = _write_strategy_artifacts(
        config=config,
        strategy=strategy,
        predictions=predictions,
        metrics=metrics,
        manifest=manifest,
        threshold_result=threshold_result,
        experiment_dir=experiment_dir,
        reports_dir=reports_dir,
        model_path=model_path,
        manifest_path=manifest_path,
        split_info=split_info,
    )
    return StrategyRun(
        strategy=strategy,
        assignment=assignment,
        detector=loaded,
        manifest=manifest,
        threshold=threshold_result,
        training_report=training_report,
        predictions=predictions,
        metrics=metrics,
        paths=paths,
    )


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def _write_strategy_artifacts(
    config: NetworkConfig,
    strategy: str,
    predictions: pd.DataFrame,
    metrics: Dict[str, Any],
    manifest: Dict[str, Any],
    threshold_result: ThresholdResult,
    experiment_dir: str,
    reports_dir: str,
    model_path: str,
    manifest_path: str,
    split_info: Dict[str, Any],
) -> Dict[str, str]:
    paths: Dict[str, str] = {
        "model": model_path,
        "manifest": manifest_path,
        "experiment_dir": experiment_dir,
    }
    for split in ("train", "validation", "test"):
        subset = predictions[predictions["split"] == split]
        if subset.empty:
            continue
        path = os.path.join(experiment_dir, f"predictions_{split}.csv")
        write_predictions(subset, path)
        paths[f"predictions_{split}"] = path

    all_predictions = os.path.join(experiment_dir, "predictions_all.csv")
    write_predictions(predictions, all_predictions)
    paths["predictions_all"] = all_predictions

    threshold_path = os.path.join(experiment_dir, "threshold.json")
    _write_json(threshold_path, threshold_result.to_dict())
    paths["threshold"] = threshold_path

    split_path = os.path.join(
        config.paths["splits_dir"], f"{strategy}.json"
    )
    _write_json(split_path, split_info)
    paths["split"] = split_path

    metrics_json = os.path.join(reports_dir, "metrics.json")
    metrics_md = os.path.join(reports_dir, "metrics.md")
    _write_json(metrics_json, metrics)
    _write_text(
        metrics_md,
        format_network_metrics_markdown(
            {
                **metrics,
                "strategy": strategy,
            }
        ),
    )
    paths["metrics_json"] = metrics_json
    paths["metrics_markdown"] = metrics_md

    note_path = os.path.join(experiment_dir, "experiment_note.md")
    _write_text(note_path, _strategy_note(strategy, config, manifest, metrics, paths))
    paths["experiment_note"] = note_path

    # The primary strategy is the deployed default for load_network_detector().
    if strategy == PRIMARY_STRATEGY:
        os.makedirs(config.paths["model_dir"], exist_ok=True)
        deployed_model = os.path.join(config.paths["model_dir"], "model.joblib")
        deployed_manifest = os.path.join(config.paths["model_dir"], "manifest.json")
        shutil.copyfile(model_path, deployed_model)
        shutil.copyfile(manifest_path, deployed_manifest)
        paths["deployed_model"] = deployed_model
        paths["deployed_manifest"] = deployed_manifest
    return paths


def _write_combined_report(
    config: NetworkConfig,
    runs: Mapping[str, StrategyRun],
    shared_paths: Mapping[str, str],
) -> None:
    os.makedirs(config.paths["reports_dir"], exist_ok=True)
    combined = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(),
        "contract_version": CONTRACT_VERSION,
        "feature_schema_version": NETWORK_FEATURE_SCHEMA_VERSION,
        "primary_strategy": PRIMARY_STRATEGY,
        "strategies": {name: run.metrics for name, run in runs.items()},
        "comparison": {name: run.metrics.get("comparison", {}) for name, run in runs.items()},
        "shared_artifacts": dict(shared_paths),
        "notes": [
            "Both split strategies are trained and evaluated independently.",
            f"`{PRIMARY_STRATEGY}` is the deployed default under "
            f"`{config.paths['model_dir']}`.",
            "The test split is never used for model or threshold selection.",
            "Synthetic-data metrics validate the pipeline, not real-world quality.",
        ],
    }
    _write_json(os.path.join(config.paths["reports_dir"], "metrics.json"), combined)
    markdown = ["# Network Anomaly Detection — All Strategies", ""]
    markdown.append(
        "> Synthetic, deterministic data. Train metrics are in-sample. "
        "Unscored windows are unknown, never normal."
    )
    markdown.append("")
    for name, run in runs.items():
        markdown.append("---")
        markdown.append("")
        markdown.append(format_network_metrics_markdown({**run.metrics, "strategy": name}))
        markdown.append("")
    _write_text(os.path.join(config.paths["reports_dir"], "metrics.md"), "\n".join(markdown))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_directories(config: NetworkConfig) -> None:
    for key in ("synthetic_dir", "splits_dir", "output_dir", "model_dir", "reports_dir"):
        os.makedirs(config.paths[key], exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(config.paths["model_dir"])), exist_ok=True)


def _window_grid(
    labels: Sequence[NetworkLabel],
    window_size_minutes: int,
) -> Dict[Tuple[str, str], List[datetime]]:
    grid: Dict[Tuple[str, str], List[datetime]] = {}
    for label in labels:
        key = (label.host_id, label.service)
        grid.setdefault(key, []).append(label.window_start)
    return {key: sorted(set(starts)) for key, starts in grid.items()}


def _assert_alignment(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    if not (len(features) == len(metadata) == len(labels)):
        raise ValueError("feature, metadata and label tables are not row-aligned")
    if "sample_key" not in labels.columns:
        raise ValueError("label table is missing the sample_key join column")
    if list(metadata["sample_key"]) != list(labels["sample_key"]):
        raise ValueError("metadata and label sample keys are not aligned")


def _score_features(detector: BaselineDetector, features: pd.DataFrame) -> np.ndarray:
    prepared = prepare_model_input(
        features,
        detector.expected_feature_columns,
        medians=detector.medians,
        policy=detector.missing_value_policy,
        sample_key_column=None,
        context="network-threshold",
        use_columns=detector.feature_columns,
    )
    return detector.score(prepared.X)


def _strategy_note(
    strategy: str,
    config: NetworkConfig,
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    paths: Mapping[str, str],
) -> str:
    lines = [
        f"# Network Anomaly Detection — Experiment Note ({strategy} split)",
        "",
        f"- Contract: `{CONTRACT_VERSION}`",
        f"- Feature schema: `{NETWORK_FEATURE_SCHEMA_VERSION}`",
        f"- Model: `{manifest.get('model', {}).get('type')}`",
        f"- Seed: `{manifest.get('random_seed')}`",
        f"- Code revision: `{manifest.get('code_revision')}`",
        f"- Threshold: `{metrics.get('threshold', {}).get('threshold')}` "
        f"({metrics.get('threshold', {}).get('method')}, "
        f"source `{metrics.get('threshold', {}).get('source_split')}`)",
        f"- Feature count: `{metrics.get('model', {}).get('n_features')}`",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python scripts/run_network_pipeline.py --config configs/network.yaml",
        "```",
        "",
        "## Artifacts",
        "",
    ]
    for name, value in sorted(paths.items()):
        lines.append(f"- `{name}`: `{value}`")
    lines.extend([
        "",
        "## Limitations",
        "",
        "- Synthetic data: the labels validate the pipeline, not real-world accuracy.",
        "- The train split is scored in-sample; do not use its metrics as generalisation evidence.",
        "- The transparent rule baseline is evidence for an analyst; it does not change "
        "the Isolation Forest decision.",
        "- The LLM adapter is disabled by default and, when enabled, explains evidence only; "
        "it never creates or upgrades a detection.",
        "",
    ])
    return "\n".join(lines)


def _asdict(value: Any) -> Dict[str, Any]:
    from dataclasses import asdict

    return asdict(value)


def _write_json(path: str, payload: Any) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    return path


def _write_text(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")
    return path
