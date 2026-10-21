"""Tests for threshold selection and its direction."""
import numpy as np
import pytest

from anomaly_detector.model_contract import ModelContractError
from anomaly_detector.thresholding import (
    apply_threshold,
    select_threshold,
    train_quantile_threshold,
)


def test_flag_direction_is_lower_score_is_anomalous():
    scores = [-0.9, -0.4, 0.1, 0.6]
    flags = apply_threshold(scores, -0.4)
    assert list(flags) == [True, True, False, False]


def test_threshold_boundary_is_inclusive():
    # A score exactly equal to the threshold is flagged.
    assert list(apply_threshold([-0.5, -0.4], -0.5)) == [True, False]
    assert list(apply_threshold([-0.5, -0.4], -0.4)) == [True, True]
    assert list(apply_threshold([-0.5, -0.4], -0.6)) == [False, False]


def test_validation_f1_maximises_f1():
    scores = [-0.9, -0.8, 0.1, 0.2, 0.3]
    labels = [True, True, False, False, False]
    result = select_threshold(scores, labels, method="validation_f1", source_split="validation")
    assert result.method == "validation_f1"
    assert result.source_split == "validation"
    assert result.is_heuristic is False
    flags = apply_threshold(scores, result.threshold)
    # A perfect separation is reachable, so F1 must be 1.
    assert flags.tolist() == [True, True, False, False, False]


def test_test_split_is_refused():
    with pytest.raises(ModelContractError) as exc:
        select_threshold([-0.1, 0.1], [True, False], source_split="test")
    assert "test" in str(exc.value).lower()


def test_missing_labels_fall_back_to_train_quantile():
    train_scores = list(np.linspace(-0.5, 0.5, 100))
    result = select_threshold(
        [-0.1, 0.1], None, method="validation_f1",
        source_split="validation", train_scores=train_scores, fallback_quantile=0.01,
    )
    assert result.method == "train_quantile"
    assert result.is_heuristic is True
    assert "not a validated optimum" in result.details["note"]


def test_single_class_labels_fall_back():
    train_scores = list(np.linspace(-0.5, 0.5, 100))
    result = select_threshold(
        [-0.1, 0.2], [False, False], method="validation_f1",
        source_split="validation", train_scores=train_scores,
    )
    assert result.is_heuristic is True
    assert "single class" in result.details["reason"]


def test_train_quantile_threshold_flags_roughly_the_quantile():
    train_scores = list(np.linspace(0.0, 1.0, 100))
    threshold = train_quantile_threshold(train_scores, 0.05)
    flagged = (np.asarray(train_scores) <= threshold).sum()
    assert 3 <= flagged <= 7


def test_train_quantile_requires_scores():
    with pytest.raises(ModelContractError):
        select_threshold([-0.1], None, method="train_quantile", source_split="validation")


def test_validation_precision_respects_the_constraint():
    scores = [-0.9, -0.8, -0.7, 0.1, 0.2]
    labels = [True, True, False, False, False]
    result = select_threshold(
        scores, labels, method="validation_precision", source_split="validation",
        min_precision=0.9,
    )
    flags = apply_threshold(scores, result.threshold)
    tp = sum(1 for f, l in zip(flags, labels) if f and l)
    fp = sum(1 for f, l in zip(flags, labels) if f and not l)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    assert precision >= 0.9
    assert result.details["constraint_met"] is True


def test_validation_precision_falls_back_when_unreachable():
    scores = [-0.9, -0.8, -0.7, 0.1, 0.2]
    labels = [False, False, False, False, True]
    train_scores = list(np.linspace(-0.5, 0.5, 100))
    result = select_threshold(
        scores, labels, method="validation_precision", source_split="validation",
        min_precision=0.99, train_scores=train_scores,
    )
    assert result.is_heuristic is True


def test_validation_recall_constraint():
    scores = [-0.9, -0.8, -0.7, 0.1, 0.2]
    labels = [True, True, False, False, False]
    result = select_threshold(
        scores, labels, method="validation_recall", source_split="validation",
        min_recall=0.5,
    )
    flags = apply_threshold(scores, result.threshold)
    tp = sum(1 for f, l in zip(flags, labels) if f and l)
    fn = sum(1 for f, l in zip(flags, labels) if not f and l)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    assert recall >= 0.5


def test_selection_is_reproducible():
    scores = [-0.9, -0.8, 0.1, 0.2, 0.3, 0.05]
    labels = [True, False, False, True, False, True]
    a = select_threshold(scores, labels, method="validation_f1", source_split="validation")
    b = select_threshold(scores, labels, method="validation_f1", source_split="validation")
    assert a.threshold == b.threshold
    assert a.details == b.details


def test_length_mismatch_fails():
    with pytest.raises(ModelContractError):
        select_threshold([-0.1, 0.1], [True], source_split="validation")


def test_non_finite_scores_fail():
    with pytest.raises(ModelContractError):
        select_threshold([-0.1, np.nan], [True, False], source_split="validation")


def test_unknown_method_fails():
    with pytest.raises(ModelContractError):
        select_threshold([-0.1], [True], method="magic", source_split="validation")
