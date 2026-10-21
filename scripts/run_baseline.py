#!/usr/bin/env python3
"""Train, score and evaluate the Milestone 3 Isolation Forest baseline.

Subcommands
-----------
``run``       train → score every split → evaluate → write all artifacts
``train``     fit the model, choose the threshold, save the model + manifest
``score``     score one split with a saved model, writing a predictions file
``evaluate``  evaluate saved predictions against the labels, writing metrics

Examples
--------
    python scripts/run_baseline.py run
    python scripts/run_baseline.py train --output-dir artifacts/baseline
    python scripts/run_baseline.py score --split test
    python scripts/run_baseline.py evaluate --split test
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pandas as pd

from anomaly_detector.baseline_config import load_baseline_config
from anomaly_detector.baseline_model import BaselineDetector
from anomaly_detector.baseline_pipeline import (
    evaluate_splits,
    load_split_tables,
    run_baseline_experiment,
    score_split,
    train_baseline,
)
from anomaly_detector.artifacts import write_metrics, write_predictions
from anomaly_detector.model_contract import ModelContractError


def _resolve_paths(args, config):
    output_dir = args.output_dir or config.paths["output_dir"]
    model_dir = os.path.join(output_dir, "model")
    reports_dir = os.path.join(output_dir, "reports")
    return output_dir, model_dir, reports_dir


def cmd_run(args, config) -> int:
    output_dir, model_dir, reports_dir = _resolve_paths(args, config)
    commands = [
        "python scripts/generate_synthetic_data.py --config configs/default.yaml",
        "python scripts/prepare_experiments.py",
        "python scripts/run_baseline.py run --config " + (args.config or "configs/baseline.yaml"),
    ]
    result = run_baseline_experiment(
        config,
        splits=tuple(args.splits) if args.splits else ("train", "validation", "test"),
        output_dir=output_dir,
        commands=commands,
    )
    _print_summary(result)
    return 0


def cmd_train(args, config) -> int:
    output_dir, model_dir, reports_dir = _resolve_paths(args, config)
    validation = None
    try:
        validation = load_split_tables(config, config.split.validation)
    except ModelContractError:
        pass
    outcome = train_baseline(config, validation_tables=validation)
    os.makedirs(model_dir, exist_ok=True)
    model_path = outcome.detector.save(os.path.join(model_dir, "model.joblib"))
    from anomaly_detector.artifacts import build_manifest, write_manifest

    manifest = build_manifest(
        detector=outcome.detector,
        config=config,
        training_report=outcome.training_report,
        threshold_result=outcome.threshold,
        split_info=outcome.split_info,
        source={"split_source": config.paths["splits_dir"], "split_strategy": config.split.strategy},
    )
    manifest_path = write_manifest(manifest, os.path.join(model_dir, "manifest.json"))
    print(f"Model    -> {model_path}")
    print(f"Manifest -> {manifest_path}")
    if outcome.threshold:
        print(f"Threshold: {outcome.threshold.threshold} "
              f"(method={outcome.threshold.method}, source={outcome.threshold.source_split})")
    return 0


def cmd_score(args, config) -> int:
    output_dir, model_dir, reports_dir = _resolve_paths(args, config)
    model_path = os.path.join(model_dir, "model.joblib")
    manifest_path = os.path.join(model_dir, "manifest.json")
    if not os.path.isfile(model_path):
        print(f"ERROR: no saved model at {model_path}; run 'train' first", file=sys.stderr)
        return 2
    detector = BaselineDetector.load(model_path)
    threshold = None
    if os.path.isfile(manifest_path):
        manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
        threshold = manifest.get("threshold", {}).get("threshold")
    tables = load_split_tables(config, args.split)
    frame = score_split(detector, config, tables, threshold)
    path = write_predictions(frame, os.path.join(output_dir, f"predictions_{args.split}.csv"))
    print(f"Scored {len(frame)} row(s) of split {args.split!r} -> {path}")
    return 0


def cmd_evaluate(args, config) -> int:
    output_dir, model_dir, reports_dir = _resolve_paths(args, config)
    splits = [args.split] if args.split else ["train", "validation", "test"]
    predictions = {}
    for name in splits:
        path = os.path.join(output_dir, f"predictions_{name}.csv")
        if not os.path.isfile(path):
            print(f"ERROR: predictions file not found: {path}; run 'score' first", file=sys.stderr)
            return 2
        predictions[name] = pd.read_csv(path)
    report = evaluate_splits(config, predictions)
    paths = write_metrics(report, reports_dir)
    for name, payload in report["splits"].items():
        metrics = payload["metrics"]
        print(f"{name:11s} n={payload['n_samples']:4d} pos={payload['n_positive']:3d} "
              f"precision={metrics.get('precision')} recall={metrics.get('recall')} "
              f"f1={metrics.get('f1')} auc={metrics.get('roc_auc')}")
    print(f"Metrics -> {paths['json']}")
    print(f"        -> {paths['markdown']}")
    return 0


def _print_summary(result) -> None:
    manifest = result["manifest"]
    threshold = result["threshold"]
    print(f"Model       : {manifest['model']['type']} "
          f"({manifest['model']['library_version']})")
    print(f"Seed        : {manifest['random_seed']}")
    print(f"Training    : {manifest['training_data']['rows_used_for_fitting']} row(s) "
          f"({manifest['data_policy']['training_data_policy']})")
    if threshold:
        print(f"Threshold   : {threshold.threshold} (method={threshold.method}, "
              f"source={threshold.source_split}, heuristic={threshold.is_heuristic})")
    for name, payload in result["metrics"]["splits"].items():
        metrics = payload["metrics"]
        print(f"  {name:11s} n={payload['n_samples']:4d} pos={payload['n_positive']:3d} "
              f"precision={metrics.get('precision')} recall={metrics.get('recall')} "
              f"f1={metrics.get('f1')} roc_auc={metrics.get('roc_auc')} pr_auc={metrics.get('pr_auc')}")
    for name, path in result["artifacts"].items():
        print(f"  {name:18s} -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Milestone 3 Isolation Forest baseline")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output-dir", default=None)

    # Shared options so they may be given before *or* after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS)
    common.add_argument("--output-dir", default=argparse.SUPPRESS)

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", parents=[common], help="train, score and evaluate everything")
    run_parser.add_argument("--splits", nargs="*", default=None)
    run_parser.set_defaults(func=cmd_run)

    train_parser = subparsers.add_parser(
        "train", parents=[common], help="fit the model and save it")
    train_parser.set_defaults(func=cmd_train)

    score_parser = subparsers.add_parser(
        "score", parents=[common], help="score a split with the saved model")
    score_parser.add_argument("--split", default="test")
    score_parser.set_defaults(func=cmd_score)

    eval_parser = subparsers.add_parser(
        "evaluate", parents=[common], help="evaluate saved predictions")
    eval_parser.add_argument("--split", default=None)
    eval_parser.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    config = load_baseline_config(args.config)
    try:
        return args.func(args, config)
    except ModelContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
