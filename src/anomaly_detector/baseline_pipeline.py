"""End-to-end orchestration for the Milestone 3 baseline.

The flow is deliberately split into small, individually testable steps:

``load_split_tables`` -> ``train_baseline`` -> ``select_threshold`` ->
``score_split`` -> ``evaluate_splits`` -> ``write_*`` artifacts

Training, threshold selection, scoring and evaluation are separate so that a
fitted model can score new windows without retraining, and so that the test
split never influences the threshold.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os

import numpy as np
import pandas as pd

from .artifacts import (
    build_manifest,
    write_experiment_note,
    write_manifest,
    write_metrics,
    write_predictions,
)
from .baseline_config import BaselineConfig
from .baseline_model import BaselineDetector, TrainingReport, fit_baseline
from .dataset import load_table
from .evaluation import evaluate_split, trivial_baselines
from .model_contract import (
    ModelContractError,
    assert_unique_sample_keys,
    prepare_model_input,
)
from .thresholding import (
    ThresholdResult,
    apply_threshold,
    select_threshold,
    train_quantile_threshold,
)


@dataclass
class SplitTables:
    """One split's three aligned tables."""
    name: str
    features: pd.DataFrame
    labels: pd.DataFrame
    metadata: pd.DataFrame
    path: str

    def label_series(self, label_column: str) -> Optional[pd.Series]:
        if label_column in self.labels.columns:
            return self.labels[label_column]
        return None


@dataclass
class TrainOutcome:
    detector: BaselineDetector
    training_report: TrainingReport
    threshold: Optional[ThresholdResult] = None
    split_info: Dict[str, Any] = field(default_factory=dict)
    threshold_details: Dict[str, Any] = field(default_factory=dict)


def load_split_tables(config: BaselineConfig, split_name: str) -> SplitTables:
    """Load one split's features/labels/metadata and verify alignment."""
    path = config.split_path(split_name)
    if not os.path.isdir(path):
        raise ModelContractError(
            f"split directory not found: {path} — run "
            f"'python scripts/prepare_experiments.py' first"
        )
    features_path = os.path.join(path, "features.csv")
    labels_path = os.path.join(path, "labels.csv")
    metadata_path = os.path.join(path, "metadata.csv")
    for required in (features_path, labels_path, metadata_path):
        if not os.path.isfile(required):
            raise ModelContractError(f"missing split table: {required}")

    features = load_table(features_path)
    labels = load_table(labels_path)
    metadata = load_table(metadata_path)
    if not (len(features) == len(labels) == len(metadata)):
        raise ModelContractError(
            f"split {split_name!r} is not row-aligned: features={len(features)}, "
            f"labels={len(labels)}, metadata={len(metadata)}"
        )
    return SplitTables(
        name=split_name,
        features=features,
        labels=labels,
        metadata=metadata,
        path=path,
    )


def train_baseline(
    config: BaselineConfig,
    validation_tables: Optional[SplitTables] = None,
) -> TrainOutcome:
    """Fit the baseline on the training split and choose a threshold.

    The estimator only ever sees training rows. The threshold is chosen on the
    validation split (or falls back to a training-score quantile). The test
    split is never read here.
    """
    train = load_split_tables(config, config.split.train)
    label_column = config.data.label_column
    labels = train.label_series(label_column)
    if labels is None:
        raise ModelContractError(
            f"training split has no {label_column!r} column; the training-data policy "
            f"{config.data.training_data_policy!r} needs labels"
        )

    detector, training_report = fit_baseline(train.features, labels, config)

    split_info: Dict[str, Any] = {
        "strategy": config.split.strategy,
        "train": {"name": config.split.train, "path": train.path, "rows": len(train.features)},
    }

    threshold_result: Optional[ThresholdResult] = None
    threshold_details: Dict[str, Any] = {}

    if validation_tables is not None:
        split_info["validation"] = {
            "name": config.split.validation,
            "path": validation_tables.path,
            "rows": len(validation_tables.features),
        }
        val_labels = validation_tables.label_series(label_column)
        prepared = prepare_model_input(
            validation_tables.features,
            detector.expected_feature_columns,
            medians=detector.medians,
            policy=detector.missing_value_policy,
            sample_key_column=config.data.sample_key_column,
            context="validation data",
            use_columns=detector.feature_columns,
        )
        if prepared.feature_columns != detector.feature_columns:
            raise ModelContractError(
                "validation feature schema does not match the trained model: "
                f"{prepared.feature_columns} != {detector.feature_columns}"
            )
        val_scores = detector.score(prepared.X)
        threshold_result = select_threshold(
            val_scores,
            None if val_labels is None else val_labels.tolist(),
            method=config.threshold.method,
            source_split=config.split.validation,
            train_scores=_train_scores(detector, train, config),
            min_precision=config.threshold.min_precision,
            min_recall=config.threshold.min_recall,
            fallback_method=config.threshold.fallback_method,
            fallback_quantile=config.threshold.fallback_quantile,
        )
        threshold_details = {
            "validation_rows_scored": int(len(val_scores)),
            "validation_rows_dropped": prepared.report.rows_dropped,
        }

    return TrainOutcome(
        detector=detector,
        training_report=training_report,
        threshold=threshold_result,
        split_info=split_info,
        threshold_details=threshold_details,
    )


def _train_scores(
    detector: BaselineDetector,
    train: SplitTables,
    config: BaselineConfig,
) -> np.ndarray:
    """Scores of the training rows (used by the quantile fallback)."""
    prepared = prepare_model_input(
        train.features,
        detector.expected_feature_columns,
        medians=detector.medians,
        policy=detector.missing_value_policy,
        sample_key_column=config.data.sample_key_column,
        context="training data",
        use_columns=detector.feature_columns,
    )
    if prepared.feature_columns != detector.feature_columns:
        raise ModelContractError(
            "training feature schema does not match the fitted model"
        )
    return detector.score(prepared.X)


def score_split(
    detector: BaselineDetector,
    config: BaselineConfig,
    tables: SplitTables,
    threshold: Optional[float],
) -> pd.DataFrame:
    """Score one split and build the per-sample predictions frame."""
    features = tables.features
    key_column = config.data.sample_key_column
    if key_column and key_column not in features.columns:
        # The split's feature table holds only the feature columns; the sample
        # key lives in the metadata table. It is copied in for traceability
        # (it never reaches the estimator).
        source = tables.metadata if key_column in tables.metadata.columns else tables.labels
        if key_column in source.columns:
            features = features.copy()
            features[key_column] = source.reset_index(drop=True)[key_column].tolist()

    prepared = prepare_model_input(
        features,
        detector.expected_feature_columns,
        medians=detector.medians,
        policy=detector.missing_value_policy,
        sample_key_column=key_column,
        context=f"{tables.name} data",
        use_columns=detector.feature_columns,
    )
    if prepared.feature_columns != detector.feature_columns:
        raise ModelContractError(
            f"{tables.name}: feature schema does not match the trained model — "
            f"expected {detector.feature_columns}, got {prepared.feature_columns}"
        )

    raw_scores = detector.score(prepared.X)
    decision = detector.decision(prepared.X)

    metadata = tables.metadata.reset_index(drop=True)
    metadata = metadata.loc[prepared.row_indices].reset_index(drop=True)

    predictions = pd.DataFrame({
        "sample_key": metadata[config.data.sample_key_column].astype(str).tolist()
        if config.data.sample_key_column in metadata.columns
        else prepared.sample_keys,
        "split": tables.name,
    })
    for column in config.data.metadata_columns:
        if column in metadata.columns:
            predictions[column] = metadata[column].astype(str).tolist()
    predictions["raw_score"] = raw_scores
    predictions["anomaly_score"] = -raw_scores
    predictions["decision_function"] = decision
    predictions["threshold"] = float(threshold) if threshold is not None else np.nan
    predictions["is_anomaly"] = (
        apply_threshold(raw_scores, threshold) if threshold is not None
        else np.zeros(len(raw_scores), dtype=bool)
    )

    label_column = config.data.label_column
    if label_column in tables.labels.columns:
        labels = tables.labels.reset_index(drop=True).loc[prepared.row_indices]
        predictions["true_label"] = labels[label_column].astype(bool).tolist()

    assert_unique_sample_keys(predictions["sample_key"].tolist(), context=f"{tables.name} predictions")

    # Samples excluded by the documented missing-value rule are reported, never
    # silently counted as normal windows.
    predictions.attrs["rows_dropped"] = prepared.report.rows_dropped
    predictions.attrs["dropped_sample_keys"] = list(prepared.report.dropped_sample_keys)
    predictions.attrs["cells_imputed"] = prepared.report.cells_imputed
    return predictions


def evaluate_splits(
    config: BaselineConfig,
    predictions: Dict[str, pd.DataFrame],
    quantile_threshold: Optional[float] = None,
    quantile_scores: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Evaluate every split that has predictions, per split."""
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "splits": {},
    }
    for name, frame in predictions.items():
        scores = frame["raw_score"].to_numpy(dtype="float64")
        flags = frame["is_anomaly"].to_numpy(dtype=bool)
        labels = frame["true_label"].tolist() if "true_label" in frame.columns else None

        window_minutes = None
        if "window_size_minutes" in frame.columns:
            try:
                window_minutes = float(frame["window_size_minutes"].iloc[0])
            except (TypeError, ValueError):
                window_minutes = None

        threshold = None
        if frame["threshold"].notna().any():
            threshold = float(frame["threshold"].iloc[0])

        evaluation = evaluate_split(
            scores=scores,
            labels=labels,
            is_anomaly=flags,
            threshold=threshold,
            split=name,
            metadata=frame,
            window_minutes=window_minutes,
        )

        rows_dropped = int(frame.attrs.get("rows_dropped", 0) or 0)
        dropped_keys = list(frame.attrs.get("dropped_sample_keys", []) or [])
        evaluation.metrics["rows_excluded_by_policy"] = rows_dropped
        if rows_dropped:
            evaluation.warnings.append(
                f"{rows_dropped} sample(s) were excluded by the missing-value policy "
                f"and are not part of these metrics: {dropped_keys[:5]}"
            )

        y_true = None
        if labels is not None:
            y_true = np.asarray([bool(v) for v in labels], dtype=bool)
        evaluation.baselines = trivial_baselines(
            y_true=y_true,
            n_samples=len(frame),
            quantile_threshold=quantile_threshold,
            scores=quantile_scores.get(name) if quantile_scores else None,
        )
        report["splits"][name] = evaluation.to_dict()
    return report


def run_baseline_experiment(
    config: BaselineConfig,
    splits: Sequence[str] = ("train", "validation", "test"),
    output_dir: Optional[str] = None,
    commands: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Train, score, evaluate and write every artifact of the baseline run."""
    output_dir = output_dir or config.paths["output_dir"]
    model_dir = os.path.join(output_dir, "model")
    reports_dir = os.path.join(output_dir, "reports")

    validation_tables: Optional[SplitTables] = None
    try:
        validation_tables = load_split_tables(config, config.split.validation)
    except ModelContractError:
        validation_tables = None

    outcome = train_baseline(config, validation_tables=validation_tables)
    detector = outcome.detector
    threshold_value = outcome.threshold.threshold if outcome.threshold else None

    # A training-score quantile is kept as a trivial reference baseline.
    train_tables = load_split_tables(config, config.split.train)
    train_scores = _train_scores(detector, train_tables, config)
    quantile_threshold = train_quantile_threshold(
        train_scores, config.threshold.fallback_quantile
    )

    predictions: Dict[str, pd.DataFrame] = {}
    prediction_paths: Dict[str, str] = {}
    excluded_paths: Dict[str, str] = {}
    for name in splits:
        tables = load_split_tables(config, name)
        frame = score_split(detector, config, tables, threshold_value)
        path = write_predictions(frame, os.path.join(output_dir, f"predictions_{name}.csv"))
        predictions[name] = frame
        prediction_paths[name] = path
        dropped_keys = list(frame.attrs.get("dropped_sample_keys", []) or [])
        if dropped_keys:
            excluded_path = os.path.join(output_dir, f"excluded_{name}.csv")
            with open(excluded_path, "w", encoding="utf-8") as fh:
                fh.write("sample_key,reason\n")
                for key in dropped_keys:
                    fh.write(f"{key},missing_value_policy\n")
            excluded_paths[name] = excluded_path

    report = evaluate_splits(
        config,
        predictions,
        quantile_threshold=quantile_threshold,
        quantile_scores={name: frame["raw_score"].to_numpy(dtype="float64")
                         for name, frame in predictions.items()},
    )
    report["model"] = {
        "type": "sklearn.ensemble.IsolationForest",
        "parameters": outcome.detector.model_config.__dict__,
        "feature_columns": list(detector.feature_columns),
    }
    report["score"] = dict(detector.score_contract)
    report["threshold"] = (outcome.threshold.to_dict() if outcome.threshold else {
        "threshold": None, "method": "not selected", "source_split": None,
        "objective": "", "is_heuristic": False, "details": {},
    })
    report["training"] = {
        "data_policy": outcome.training_report.training_data_policy,
        "missing_value_policy": outcome.training_report.missing_value_policy,
        "rows_used_for_fitting": outcome.training_report.rows_used,
        "rows_available": outcome.training_report.rows_available,
        "label_distribution": outcome.training_report.label_distribution,
    }
    report["baseline_comparison"] = {
        "train_quantile_threshold": quantile_threshold,
        "note": (
            "The trivial 'no_alert' and 'train_quantile' references are reported for "
            "context only; they are not anomaly detectors."
        ),
    }
    report["data_nature"] = "synthetic"
    report["caveats"] = [
        "Results are computed on synthetic data and demonstrate the pipeline, not real-world "
        "detection quality.",
        "Anomaly detection is not root-cause analysis: a flagged window is a lead for "
        "investigation, not proof of an attack.",
        "The anomaly score is a ranking statistic, not a probability.",
    ]

    metrics_paths = write_metrics(report, reports_dir)

    manifest = build_manifest(
        detector=detector,
        config=config,
        training_report=outcome.training_report,
        threshold_result=outcome.threshold,
        split_info={**outcome.split_info, "scored_splits": list(predictions.keys())},
        source={
            "split_source": config.paths["splits_dir"],
            "split_strategy": config.split.strategy,
            "train_features": os.path.join(train_tables.path, "features.csv"),
        },
    )
    manifest_path = write_manifest(manifest, os.path.join(model_dir, "manifest.json"))
    model_path = detector.save(os.path.join(model_dir, "model.joblib"))

    artifacts = {
        "model": model_path,
        "manifest": manifest_path,
        "metrics_json": metrics_paths["json"],
        "metrics_markdown": metrics_paths["markdown"],
        **{f"predictions_{name}": path for name, path in prediction_paths.items()},
        **{f"excluded_{name}": path for name, path in excluded_paths.items()},
    }
    note_path = write_experiment_note(
        os.path.join(output_dir, "experiment_note.md"),
        commands=list(commands or []),
        limitations=[
            "Trained on synthetic windows; metrics are not evidence of real-world detection quality.",
            "The threshold is chosen on the validation split only; the test split is never used "
            "for training or threshold selection.",
            "Missing values are imputed with training-split medians; the imputation is learned "
            "on training rows only and is never a zero-fill.",
            "No hyperparameter search was performed (that is a later milestone).",
            "Detection delay is not reported: the label contract has no anomaly onset time.",
        ],
        artifacts=artifacts,
        manifest=manifest,
    )
    artifacts["experiment_note"] = note_path

    return {
        "artifacts": artifacts,
        "manifest": manifest,
        "metrics": report,
        "threshold": outcome.threshold,
        "training_report": outcome.training_report,
        "predictions": predictions,
    }
