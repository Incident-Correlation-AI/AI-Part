"""Milestone 7 — selection-policy tests.

These tests pin the central M7 guarantee: **the model selection is
validation-only**. A test metric must not be able to influence the decision,
even by accident.
"""
import os

import pytest

from anomaly_detector.selection_policy import (
    ALLOWED_SELECTION_METRICS,
    FORBIDDEN_SELECTION_METRICS,
    CandidateEvidence,
    SelectionPolicy,
    SelectionPolicyError,
)

POLICY_PATH = os.path.join("configs", "selection.yaml")


@pytest.fixture(scope="module")
def policy():
    return SelectionPolicy.load(POLICY_PATH)


# ---------------------------------------------------------------------------
# The policy file
# ---------------------------------------------------------------------------


def test_policy_loads_from_config(policy):
    assert policy.name
    assert policy.weights
    assert policy.incumbent
    assert policy.eligible_candidates
    assert policy.min_improvement_margin >= 0


def test_policy_file_declares_the_guards():
    import yaml

    with open(POLICY_PATH, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    guards = raw["policy"]["guards"]
    assert guards["forbid_test_metrics_in_selection"] is True
    assert guards["forbid_test_labels_in_selection"] is True
    assert guards["forbid_test_split_as_threshold_source"] is True
    fairness = raw["policy"]["fairness"]
    assert fairness["training_data_policy"] == "normal_only"
    assert fairness["enforce_matching_training_rows"] is True


# ---------------------------------------------------------------------------
# Structural guarantee: test metrics cannot enter the evidence
# ---------------------------------------------------------------------------


def test_evidence_rejects_test_metrics():
    for forbidden in sorted(FORBIDDEN_SELECTION_METRICS):
        with pytest.raises(SelectionPolicyError):
            CandidateEvidence(name="x", validation={forbidden: 1.0})


def test_evidence_rejects_unknown_metrics():
    with pytest.raises(SelectionPolicyError):
        CandidateEvidence(name="x", validation={"some_random_metric": 1.0})


def test_allowed_metrics_are_the_validation_flavour():
    assert "f1" in ALLOWED_SELECTION_METRICS
    assert "test_f1" not in ALLOWED_SELECTION_METRICS
    assert FORBIDDEN_SELECTION_METRICS.isdisjoint(ALLOWED_SELECTION_METRICS)


# ---------------------------------------------------------------------------
# Decision behaviour
# ---------------------------------------------------------------------------


def _evidence(name, f1, precision, recall, fpr=0.0, pr_auc=1.0, robust=None):
    return CandidateEvidence(
        name=name,
        validation={
            "f1": f1, "precision": precision, "recall": recall,
            "false_positive_rate": fpr, "pr_auc": pr_auc,
        },
        robustness=robust or {},
    )


def test_incumbent_is_retained_when_no_challenger_beats_it(policy):
    evidence = [
        _evidence(policy.incumbent, 1.0, 1.0, 1.0),
        _evidence("lof_novelty", 0.9, 0.9, 0.9),
    ]
    decision = policy.decide(evidence)
    assert decision.selected == policy.incumbent
    assert decision.used_test_metrics is False
    assert "retained" in decision.reason


def test_challenger_wins_only_with_a_clear_margin(policy):
    evidence = [
        _evidence(policy.incumbent, 0.80, 0.80, 0.80),
        _evidence("lof_novelty", 0.80, 0.80, 0.80),
    ]
    # Exactly tied -> incumbent retained (no clear gain).
    assert policy.decide(evidence).selected == policy.incumbent

    # A clear margin -> challenger wins.
    evidence = [
        _evidence(policy.incumbent, 0.80, 0.80, 0.80),
        _evidence("lof_novelty", 0.99, 0.99, 0.99),
    ]
    decision = policy.decide(evidence)
    assert decision.selected == "lof_novelty"
    assert decision.margin >= decision.required_margin


def test_challenger_failing_a_robustness_floor_cannot_win(policy):
    evidence = [
        _evidence(policy.incumbent, 0.80, 0.80, 0.80),
        # Great validation metrics, but it flags every benign shift window.
        _evidence("lof_novelty", 0.99, 0.99, 0.99,
                  robust={"benign_shift_flag_rate": 1.0}),
    ]
    decision = policy.decide(evidence)
    assert decision.selected == policy.incumbent
    assert "lof_novelty" in decision.floors_failed
    assert any("benign_shift_flag_rate" in f for f in decision.floors_failed["lof_novelty"])


def test_false_positive_rate_floor_blocks_a_challenger(policy):
    evidence = [
        _evidence(policy.incumbent, 0.80, 0.80, 0.80),
        _evidence("lof_novelty", 0.99, 0.99, 0.99, fpr=0.9),
    ]
    decision = policy.decide(evidence)
    assert decision.selected == policy.incumbent
    assert any("false_positive_rate" in f for f in decision.floors_failed["lof_novelty"])


def test_decision_requires_evidence_for_the_incumbent(policy):
    evidence = [_evidence("lof_novelty", 0.9, 0.9, 0.9)]
    with pytest.raises(SelectionPolicyError):
        policy.decide(evidence)


def test_decision_is_deterministic(policy):
    evidence = [
        _evidence(policy.incumbent, 0.90, 0.90, 0.90),
        _evidence("lof_novelty", 0.91, 0.91, 0.91),
    ]
    first = policy.decide(evidence)
    second = policy.decide(evidence)
    assert first.selected == second.selected
    assert first.weighted_scores == second.weighted_scores


def test_decision_records_the_evidence_split(policy):
    evidence = [
        _evidence(policy.incumbent, 0.9, 0.9, 0.9),
        _evidence("lof_novelty", 0.9, 0.9, 0.9),
    ]
    payload = policy.decide(evidence).to_dict()
    assert payload["selection_evidence_split"] == "validation"
    assert payload["used_test_metrics"] is False


def test_weighted_score_inverts_the_false_positive_rate(policy):
    low = policy.weighted_score({"f1": 1.0, "precision": 1.0, "recall": 1.0,
                                  "false_positive_rate": 0.0, "pr_auc": 1.0})
    high = policy.weighted_score({"f1": 1.0, "precision": 1.0, "recall": 1.0,
                                   "false_positive_rate": 1.0, "pr_auc": 1.0})
    assert low > high


def test_ineligible_candidates_are_ignored(policy):
    evidence = [
        _evidence(policy.incumbent, 0.9, 0.9, 0.9),
        _evidence("some_other_model", 1.0, 1.0, 1.0),
    ]
    decision = policy.decide(evidence)
    assert decision.selected == policy.incumbent
    assert "some_other_model" not in decision.candidates_considered
