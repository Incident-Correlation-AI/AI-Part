"""Tests for the evaluation metrics and their edge cases."""
import numpy as np
import pandas as pd
import pytest

from anomaly_detector.evaluation import evaluate_split, format_metrics_markdown, trivial_baselines
from anomaly_detector.model_contract import ModelContractError


def test_perfect_prediction():
    scores = [-0.9, -0.8, 0.1, 0.2]
    labels = [True, True, False, False]
    flags = [True, True, False, False]
    result = evaluate_split(scores, labels, flags, threshold=-0.5, split="test")
    metrics = result.metrics
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert (metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]) == (2, 0, 2, 0)
    assert metrics["roc_auc"] == 1.0
    assert result.n_positive == 2 and result.n_negative == 2


def test_no_predicted_anomalies():
    scores = [0.1, 0.2, 0.3]
    labels = [True, False, False]
    flags = [False, False, False]
    result = evaluate_split(scores, labels, flags, threshold=-1.0, split="test")
    assert result.metrics["precision"] == 0.0
    assert result.metrics["recall"] == 0.0
    assert result.metrics["f1"] == 0.0
    assert result.metrics["fn"] == 1
    assert result.metrics["n_flagged"] == 0


def test_single_class_split_skips_auc():
    scores = [-0.9, -0.8, -0.7]
    labels = [False, False, False]
    flags = [True, False, False]
    result = evaluate_split(scores, labels, flags, threshold=0.0, split="validation")
    assert result.metrics["roc_auc"] is None
    assert result.metrics["pr_auc"] is None
    assert "one class" in result.skipped["roc_auc"]
    assert result.metrics["prevalence"] == 0.0


def test_no_positive_labels_skips_auc():
    scores = [-0.9, 0.2, 0.3]
    labels = [False, False, False]
    flags = [True, False, False]
    result = evaluate_split(scores, labels, flags, threshold=0.0, split="validation")
    assert "one class" in result.skipped["roc_auc"]


def test_missing_labels_skip_supervised_metrics():
    scores = [-0.9, 0.2]
    flags = [True, False]
    result = evaluate_split(scores, None, flags, threshold=0.0, split="unlabelled")
    assert "no ground-truth labels" in result.skipped["precision"]
    assert result.metrics["n_flagged"] == 1


def test_false_positive_metrics():
    scores = [-0.9, -0.8, 0.1]
    labels = [True, False, False]
    flags = [True, True, False]
    result = evaluate_split(scores, labels, flags, threshold=-0.85, split="test")
    assert result.metrics["false_positive_count"] == 1
    assert result.metrics["false_positive_rate"] == pytest.approx(0.5)
    assert result.metrics["prevalence"] == pytest.approx(1 / 3)


def test_alert_rate_per_host_hour():
    scores = [-0.9, 0.1, 0.2]
    flags = [True, False, False]
    metadata = pd.DataFrame({"host_id": ["h1"] * 3, "window_start": ["t"] * 3})
    result = evaluate_split(scores, None, flags, threshold=0.0, split="test",
                            metadata=metadata, window_minutes=60)
    # 3 windows of 60 minutes == 3 host-hours, 1 alert.
    assert result.metrics["observed_host_hours"] == pytest.approx(3.0)
    assert result.metrics["alerts_per_host_hour"] == pytest.approx(1 / 3)


def test_alert_rate_skipped_without_metadata():
    result = evaluate_split([-0.9], None, [True], threshold=0.0, split="test")
    assert "alerts_per_host_hour" in result.skipped


def test_detection_delay_is_skipped_with_a_reason():
    result = evaluate_split([-0.9], [True], [True], threshold=0.0, split="test")
    assert "detection_delay" in result.skipped
    assert "onset" in result.skipped["detection_delay"]


def test_flags_and_scores_length_mismatch_fails():
    with pytest.raises(ModelContractError):
        evaluate_split([-0.1, 0.1], [True, False], [True], threshold=0.0, split="test")


def test_non_finite_scores_fail():
    with pytest.raises(ModelContractError):
        evaluate_split([-0.1, np.inf], [True, False], [True, False], threshold=0.0, split="test")


def test_trivial_baselines():
    y = np.array([True, False, False])
    baselines = trivial_baselines(y, n_samples=3, quantile_threshold=-0.5,
                                  scores=np.array([-0.9, 0.1, 0.2]))
    assert baselines["no_alert"]["recall"] == 0.0
    assert baselines["no_alert"]["f1"] == 0.0
    assert baselines["train_quantile"]["n_flagged"] == 1


def test_roc_auc_uses_anomaly_orientation():
    # Lower raw score == more anomalous, so AUC must be computed on -score.
    scores = [-0.9, -0.8, 0.1, 0.2]
    labels = [True, True, False, False]
    flags = [True, True, False, False]
    result = evaluate_split(scores, labels, flags, threshold=0.0, split="test")
    assert result.metrics["roc_auc"] == pytest.approx(1.0)
    assert result.metrics["pr_auc"] == pytest.approx(1.0)


def test_markdown_report_states_the_score_contract():
    report = {
        "model": {"type": "sklearn.ensemble.IsolationForest"},
        "score": {"score_name": "x.score_samples", "score_direction": "lower_is_more_anomalous"},
        "threshold": {"threshold": -0.5, "method": "validation_f1", "source_split": "validation"},
        "training": {"data_policy": "normal_only"},
        "generated_at": "now",
        "splits": {},
    }
    text = format_metrics_markdown(report)
    assert "x.score_samples" in text
    assert "lower_is_more_anomalous" in text
    assert "validation_f1" in text
    assert "synthetic" in text.lower()
