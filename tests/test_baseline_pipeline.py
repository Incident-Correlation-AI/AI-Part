"""Tests for the Milestone 3 pipeline, split discipline and CLI."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from anomaly_detector.baseline_pipeline import (
    load_split_tables,
    run_baseline_experiment,
    score_split,
    train_baseline,
)
from anomaly_detector.model_contract import ModelContractError


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "run_baseline.py")


def _validation_tables(config):
    return load_split_tables(config, config.split.validation)


def test_train_selects_threshold_on_validation_only(baseline_config, baseline_splits):
    outcome = train_baseline(baseline_config, _validation_tables(baseline_config))
    assert outcome.threshold is not None
    assert outcome.threshold.source_split == "validation"
    assert outcome.threshold.method == "validation_f1"


def test_test_split_never_influences_the_model_or_threshold(baseline_config, baseline_splits):
    first = train_baseline(baseline_config, _validation_tables(baseline_config))
    train_features = pd.read_csv(
        os.path.join(str(baseline_splits), "train", "features.csv")
    )
    first_scores = first.detector.score(train_features.to_numpy(dtype="float64"))

    # Poison the test split (features and labels).
    test_dir = os.path.join(str(baseline_splits), "test")
    features = pd.read_csv(os.path.join(test_dir, "features.csv"))
    features.iloc[:, :] = 1e9
    features.to_csv(os.path.join(test_dir, "features.csv"), index=False)
    labels = pd.read_csv(os.path.join(test_dir, "labels.csv"))
    labels["is_anomaly"] = ~labels["is_anomaly"]
    labels.to_csv(os.path.join(test_dir, "labels.csv"), index=False)

    second = train_baseline(baseline_config, _validation_tables(baseline_config))
    second_scores = second.detector.score(train_features.to_numpy(dtype="float64"))
    assert first.threshold.threshold == second.threshold.threshold
    assert np.allclose(first_scores, second_scores)


def test_validation_labels_do_influence_the_threshold(baseline_config, baseline_splits):
    first = train_baseline(baseline_config, _validation_tables(baseline_config))
    validation_dir = os.path.join(str(baseline_splits), "validation")
    labels = pd.read_csv(os.path.join(validation_dir, "labels.csv"))
    labels["is_anomaly"] = ~labels["is_anomaly"]
    labels.to_csv(os.path.join(validation_dir, "labels.csv"), index=False)
    second = train_baseline(baseline_config, _validation_tables(baseline_config))
    assert first.threshold.threshold != second.threshold.threshold


def test_predictions_preserve_identifiers_and_alignment(baseline_config, baseline_splits):
    outcome = train_baseline(baseline_config, _validation_tables(baseline_config))
    tables = load_split_tables(baseline_config, "test")
    frame = score_split(outcome.detector, baseline_config, tables,
                        outcome.threshold.threshold)
    expected = pd.read_csv(os.path.join(str(baseline_splits), "test", "metadata.csv"))
    assert list(frame["sample_key"]) == list(expected["sample_key"])
    assert list(frame["split"]) == ["test"] * len(frame)
    assert frame["raw_score"].notna().all()
    assert frame["is_anomaly"].dtype == bool


def test_scoring_rejects_a_mismatched_schema(baseline_config, baseline_splits):
    outcome = train_baseline(baseline_config, _validation_tables(baseline_config))
    test_dir = os.path.join(str(baseline_splits), "test")
    features = pd.read_csv(os.path.join(test_dir, "features.csv"))
    features = features.drop(columns=["error_ratio"])
    features.to_csv(os.path.join(test_dir, "features.csv"), index=False)
    tables = load_split_tables(baseline_config, "test")
    with pytest.raises(ModelContractError):
        score_split(outcome.detector, baseline_config, tables, 0.0)


def test_run_experiment_writes_all_artifacts(baseline_config, baseline_splits):
    result = run_baseline_experiment(baseline_config, output_dir=baseline_config.paths["output_dir"])
    artifacts = result["artifacts"]
    for name in ["model", "manifest", "metrics_json", "metrics_markdown",
                 "predictions_train", "predictions_validation", "predictions_test",
                 "experiment_note"]:
        assert os.path.isfile(artifacts[name]), name

    manifest = json.load(open(artifacts["manifest"], encoding="utf-8"))
    assert manifest["model"]["type"] == "sklearn.ensemble.IsolationForest"
    assert manifest["score"]["score_direction"] == "lower_is_more_anomalous"
    assert manifest["threshold"]["method"] == "validation_f1"
    assert manifest["data_policy"]["missing_is_not_zero"] is True
    assert manifest["feature_schema"]["feature_columns"] == baseline_config.data.feature_columns
    assert manifest["random_seed"] == baseline_config.model.random_state

    metrics = json.load(open(artifacts["metrics_json"], encoding="utf-8"))
    assert set(metrics["splits"].keys()) == {"train", "validation", "test"}
    assert metrics["data_nature"] == "synthetic"


def test_manifest_contains_no_secrets_or_per_sample_content(baseline_config, baseline_splits):
    result = run_baseline_experiment(baseline_config, output_dir=baseline_config.paths["output_dir"])
    text = json.dumps(result["manifest"]).lower()
    for secret in ("password", "secret", "api_key", "authorization", "bearer "):
        assert secret not in text
    # No per-sample identifiers (raw log content) may leak into the manifest.
    metadata = pd.read_csv(os.path.join(str(baseline_splits), "train", "metadata.csv"))
    assert not any(key in text for key in metadata["sample_key"].tolist())


def test_manifest_records_the_split_provenance(baseline_config, baseline_splits):
    result = run_baseline_experiment(baseline_config, output_dir=baseline_config.paths["output_dir"])
    manifest = result["manifest"]
    assert manifest["training_data"]["split_strategy"] == "random"
    assert manifest["splits"]["train"]["rows"] == 30
    assert manifest["splits"]["scored_splits"] == ["train", "validation", "test"]


def _inject_missing_values(root, split="train"):
    """Put a missing value into a *normal* training row (row 0 is an anomaly)."""
    path = os.path.join(str(root), split, "features.csv")
    features = pd.read_csv(path)
    features.loc[10, "mean_latency_ms"] = np.nan
    features.to_csv(path, index=False)


def test_drop_policy_excludes_and_reports_the_samples(baseline_config, baseline_splits):
    baseline_config.data.missing_value_policy = "drop"
    _inject_missing_values(baseline_splits, "train")
    result = run_baseline_experiment(baseline_config, output_dir=baseline_config.paths["output_dir"])
    train = result["metrics"]["splits"]["train"]
    assert train["metrics"]["rows_excluded_by_policy"] == 1
    assert train["warnings"], "excluding a sample must be reported, never silent"
    assert "excluded_train" in result["artifacts"]
    excluded = pd.read_csv(result["artifacts"]["excluded_train"])
    assert len(excluded) == 1
    assert excluded["reason"].iloc[0] == "missing_value_policy"


def test_error_policy_fails_loudly(baseline_config, baseline_splits):
    baseline_config.data.missing_value_policy = "error"
    _inject_missing_values(baseline_splits, "train")
    with pytest.raises(ModelContractError) as exc:
        run_baseline_experiment(baseline_config, output_dir=baseline_config.paths["output_dir"])
    assert "missing" in str(exc.value).lower()


def test_median_impute_never_zero_fills(baseline_config, baseline_splits):
    _inject_missing_values(baseline_splits, "train")
    result = run_baseline_experiment(baseline_config, output_dir=baseline_config.paths["output_dir"])
    manifest = result["manifest"]
    assert manifest["data_policy"]["missing_is_not_zero"] is True
    assert manifest["training_data"]["cells_imputed"] == 1
    # The imputation statistic is a learned median, not zero.
    assert manifest["data_policy"]["imputation_statistics"]["mean_latency_ms"] != 0.0


def _write_config(tmp_path, baseline_config) -> str:
    path = tmp_path / "baseline_test.yaml"
    path.write_text(
        "model:\n"
        f"  n_estimators: {baseline_config.model.n_estimators}\n"
        f"  random_state: {baseline_config.model.random_state}\n"
        "data:\n"
        "  feature_columns:\n"
        + "".join(f"    - {c}\n" for c in baseline_config.data.feature_columns)
        + "  training_data_policy: normal_only\n"
        "  missing_value_policy: median_impute\n"
        "threshold:\n"
        "  method: validation_f1\n"
        "split:\n"
        "  strategy: random\n"
        "paths:\n"
        f"  splits_dir: {baseline_config.paths['splits_dir']}\n"
        f"  output_dir: {baseline_config.paths['output_dir']}\n",
        encoding="utf-8",
    )
    return str(path)


def test_cli_run_succeeds_and_writes_outputs(baseline_config, baseline_splits, tmp_path):
    config_path = _write_config(tmp_path, baseline_config)
    completed = subprocess.run(
        [sys.executable, SCRIPT, "run", "--config", config_path],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    assert os.path.isfile(os.path.join(baseline_config.paths["output_dir"], "model", "model.joblib"))
    assert os.path.isfile(os.path.join(baseline_config.paths["output_dir"], "reports", "metrics.json"))


def test_cli_score_and_evaluate(baseline_config, baseline_splits, tmp_path):
    config_path = _write_config(tmp_path, baseline_config)
    run = subprocess.run([sys.executable, SCRIPT, "run", "--config", config_path],
                         capture_output=True, text=True, cwd=REPO_ROOT)
    assert run.returncode == 0, run.stderr
    score = subprocess.run([sys.executable, SCRIPT, "score", "--split", "test", "--config", config_path],
                           capture_output=True, text=True, cwd=REPO_ROOT)
    assert score.returncode == 0, score.stderr
    evaluate = subprocess.run([sys.executable, SCRIPT, "evaluate", "--config", config_path],
                              capture_output=True, text=True, cwd=REPO_ROOT)
    assert evaluate.returncode == 0, evaluate.stderr
    assert "test" in evaluate.stdout


def test_cli_fails_when_the_model_is_missing(baseline_config, baseline_splits, tmp_path):
    config_path = _write_config(tmp_path, baseline_config)
    completed = subprocess.run(
        [sys.executable, SCRIPT, "score", "--split", "test", "--config", config_path],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert completed.returncode != 0
    assert "no saved model" in completed.stderr


def test_cli_fails_when_a_split_is_missing(baseline_config, baseline_splits, tmp_path):
    config_path = _write_config(tmp_path, baseline_config)
    import shutil

    shutil.rmtree(os.path.join(str(baseline_splits), "test"))
    completed = subprocess.run(
        [sys.executable, SCRIPT, "run", "--config", config_path],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert completed.returncode != 0
    assert "ERROR" in completed.stderr


def test_cli_evaluate_fails_without_predictions(baseline_config, baseline_splits, tmp_path):
    config_path = _write_config(tmp_path, baseline_config)
    completed = subprocess.run(
        [sys.executable, SCRIPT, "evaluate", "--config", config_path],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert completed.returncode != 0
    assert "predictions file not found" in completed.stderr
