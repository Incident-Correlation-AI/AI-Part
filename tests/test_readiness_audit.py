"""Milestone 7 — readiness-audit tests.

These cover the audit itself: the fair comparison, the validation-only
threshold audit, the M4 root-cause fix, and the report contents.
"""
import json
import os

import numpy as np
import pytest

from anomaly_detector.comparison_config import load_comparison_config
from anomaly_detector.readiness_audit import (
    INCUMBENT,
    assess_readiness,
    audit_threshold_policies,
    build_evidence,
    collect_robustness_evidence,
    run_fair_validation_comparison,
)
from anomaly_detector.selection_policy import SelectionPolicy

MODEL_PATH = "artifacts/baseline/model/model.joblib"
MANIFEST_PATH = "artifacts/baseline/model/manifest.json"
SPLITS_DIR = "data/processed/splits"

requires_splits = pytest.mark.skipif(
    not os.path.isdir(os.path.join(SPLITS_DIR, "random", "test")),
    reason="processed splits are not present (run scripts/prepare_experiments.py)",
)


@pytest.fixture(scope="module")
def cfg():
    return load_comparison_config("configs/comparison.yaml")


@pytest.fixture(scope="module")
def policy():
    return SelectionPolicy.load("configs/selection.yaml")


# ---------------------------------------------------------------------------
# Phase 1 — fair, validation-only comparison
# ---------------------------------------------------------------------------


@requires_splits
def test_both_candidates_use_the_same_training_rows(cfg):
    runs, provenance = run_fair_validation_comparison(cfg)
    assert len(runs) == 2
    used = {r.training_rows_used for r in runs}
    assert len(used) == 1, "candidates were trained on different row counts"
    assert provenance["candidates_share_training_rows"] is True
    assert provenance["training_data_policy"] == "normal_only"
    # The training split holds 136 rows of which 26 are anomalies, so the
    # normal_only policy must have excluded them.
    assert provenance["training_rows_used"] < provenance["training_rows_available"]


@requires_splits
def test_thresholds_come_from_the_validation_split_only(cfg):
    runs, provenance = run_fair_validation_comparison(cfg)
    assert provenance["thresholds_selected_on"] == "validation"
    for run in runs:
        assert run.threshold_method == "validation_f1"
        assert run.threshold is not None


@requires_splits
def test_selection_evidence_contains_no_test_metrics(cfg):
    runs, _ = run_fair_validation_comparison(cfg)
    evidence = build_evidence(runs, {})
    for candidate in evidence:
        for key in candidate.validation:
            assert not key.startswith("test_")
            assert "test" not in key.lower()


@requires_splits
def test_decision_retains_the_incumbent_and_uses_validation_only(cfg, policy):
    runs, _ = run_fair_validation_comparison(cfg)
    robustness = collect_robustness_evidence(cfg, runs)
    evidence = build_evidence(runs, robustness)
    decision = policy.decide(evidence)
    assert decision.selected == INCUMBENT
    assert decision.used_test_metrics is False
    assert decision.to_dict()["selection_evidence_split"] == "validation"


@requires_splits
def test_robustness_evidence_measures_benign_scenarios(cfg):
    runs, _ = run_fair_validation_comparison(cfg)
    robustness = collect_robustness_evidence(cfg, runs)
    for run in runs:
        entry = robustness[run.name]
        assert "benign_shift_flag_rate" in entry
        assert "missing_telemetry_flag_rate" in entry
        assert 0.0 <= entry["benign_shift_flag_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Phase 2 — threshold audit
# ---------------------------------------------------------------------------


class _FakeRun:
    def __init__(self, validation_scores, train_scores, threshold):
        self.validation_scores = validation_scores
        self.train_scores = train_scores
        self.threshold = threshold


def test_threshold_audit_uses_validation_only():
    run = _FakeRun(
        validation_scores=np.array([-0.9, -0.8, -0.2, -0.1, 0.0]),
        train_scores=np.array([-0.7, -0.6, -0.5, -0.4]),
        threshold=-0.5,
    )
    y = [True, True, False, False, False]
    audit = audit_threshold_policies(run, y, [
        {"name": "fixed", "kind": "fixed", "value": -0.75},
        {"name": "quantile", "kind": "train_quantile", "value": 0.25},
    ])
    assert audit["evaluated_on"] == "validation"
    assert audit["used_test_metrics"] is False
    assert "lower_is_more_anomalous" in audit["score_direction"]
    assert "not a probability" in audit["score_semantics"]
    for entry in audit["policies"]:
        assert "tp" in entry and "fp" in entry and "tn" in entry and "fn" in entry
        assert "false_positive_rate" in entry
        assert "false_alerts_per_window" in entry


def test_threshold_audit_reports_skips_for_single_class():
    run = _FakeRun(
        validation_scores=np.array([-0.9, -0.8]),
        train_scores=np.array([-0.7]),
        threshold=-0.5,
    )
    audit = audit_threshold_policies(run, [False, False], [
        {"name": "fixed", "kind": "fixed", "value": -0.75},
    ])
    assert "metrics_skipped" in audit["policies"][0]


def test_threshold_audit_rejects_unknown_policy_kinds():
    run = _FakeRun(np.array([0.0]), np.array([0.0]), 0.0)
    audit = audit_threshold_policies(run, [True], [
        {"name": "weird", "kind": "not_a_kind", "value": 1},
    ])
    assert "error" in audit["policies"][0]


# ---------------------------------------------------------------------------
# Readiness verdict
# ---------------------------------------------------------------------------


def test_readiness_is_blocked_when_selection_used_test_metrics():
    verdict = assess_readiness({
        "selection_evidence_split": "test",
        "used_test_metrics_for_selection": True,
        "end_to_end": {"totals": {}, "benign_shift_flag_rate": 0.0,
                       "missing_telemetry_flag_rate": 0.0},
    })
    assert verdict["verdict"] == "blocked"
    assert verdict["blockers"]


def test_readiness_is_demo_only_with_synthetic_evidence():
    verdict = assess_readiness({
        "selection_evidence_split": "validation",
        "used_test_metrics_for_selection": False,
        "end_to_end": {"totals": {"missed": 1, "unscored": 0},
                       "benign_shift_flag_rate": 0.0,
                       "missing_telemetry_flag_rate": 0.0},
    })
    assert verdict["verdict"] == "controlled demo only"
    assert not verdict["blockers"]
    assert any("synthetic" in c for c in verdict["cautions"])


def test_readiness_flags_a_high_benign_shift_flag_rate():
    verdict = assess_readiness({
        "selection_evidence_split": "validation",
        "used_test_metrics_for_selection": False,
        "end_to_end": {"totals": {}, "benign_shift_flag_rate": 1.0,
                       "missing_telemetry_flag_rate": 0.15},
    })
    assert any("benign-shift" in c for c in verdict["cautions"])


# ---------------------------------------------------------------------------
# The M4 root-cause fix
# ---------------------------------------------------------------------------


def test_m4_comparison_no_longer_selects_on_test_metrics():
    """The M4 auto-picker must not key the winner on test metrics.

    This is a behavioural test: a candidate with perfect TEST metrics but
    worse VALIDATION metrics must not win.
    """
    from types import SimpleNamespace

    from anomaly_detector import comparison

    class _Threshold:
        method = "validation_f1"

    class _Run:
        def __init__(self, name, val, test):
            self.name = name
            self.type = "stub"
            self.variant = "v"
            self.metrics_val = {"metrics": val}
            self.metrics_test = {"metrics": test}
            self.threshold = 0.0
            self.threshold_result = _Threshold()
            self.runtime_s = 0.0
            self.notes = []

    weights = {"f1": 1.0, "precision": 0.0, "recall": 0.0,
               "false_positive_rate": 0.0, "pr_auc": 0.0}
    cfg = SimpleNamespace(
        selection_criteria={"weights": weights},
        data={"feature_columns": ["a"], "splits_dir": "x", "strategy": "random"},
    )

    runs = [
        # The incumbent: weaker on test, stronger on validation.
        _Run("m3_isolation_forest_reference",
             {"f1": 0.90, "precision": 0.90, "recall": 0.90, "pr_auc": 0.90,
              "false_positive_rate": 0.10},
             {"f1": 0.10, "precision": 0.10, "recall": 0.10, "pr_auc": 0.10,
              "fp": 9, "tp": 1, "n_flagged": 10}),
        # The challenger: perfect on test, worse on validation.
        _Run("lof_novelty",
             {"f1": 0.50, "precision": 0.50, "recall": 0.50, "pr_auc": 0.50,
              "false_positive_rate": 0.50},
             {"f1": 1.00, "precision": 1.00, "recall": 1.00, "pr_auc": 1.00,
              "fp": 0, "tp": 8, "n_flagged": 8}),
    ]

    report = comparison.build_comparison_report(cfg, runs, [], "unused")
    assert report["recommended"] == "m3_isolation_forest_reference"
    assert report["selection_evidence_split"] == "validation"
    assert report["used_test_metrics_for_selection"] is False


def test_m4_comparison_reports_the_validation_weighted_score():
    from types import SimpleNamespace

    from anomaly_detector import comparison

    class _Threshold:
        method = "validation_f1"

    class _Run:
        def __init__(self, name, val):
            self.name = name
            self.type = "stub"
            self.variant = "v"
            self.metrics_val = {"metrics": val}
            self.metrics_test = {"metrics": {}}
            self.threshold = 0.0
            self.threshold_result = _Threshold()
            self.runtime_s = 0.0
            self.notes = []

    weights = {"f1": 1.0, "precision": 0.0, "recall": 0.0,
               "false_positive_rate": 0.0, "pr_auc": 0.0}
    cfg = SimpleNamespace(
        selection_criteria={"weights": weights},
        data={"feature_columns": ["a"], "splits_dir": "x", "strategy": "random"},
    )
    runs = [_Run("m3_isolation_forest_reference",
                 {"f1": 0.75, "precision": 0.75, "recall": 0.75, "pr_auc": 0.75,
                  "false_positive_rate": 0.25})]
    report = comparison.build_comparison_report(cfg, runs, [], "unused")
    row = report["comparison_table"][0]
    assert row["validation_weighted_score"] == pytest.approx(0.75)


def test_m4_comparison_applies_the_normal_only_policy_to_every_candidate():
    import inspect

    from anomaly_detector import comparison

    source = inspect.getsource(comparison.run_comparison)
    # Both the MAD and the LOF candidate must apply the training-row policy.
    assert source.count("select_training_rows(") >= 2


def test_m4_comparison_preserves_the_historical_run():
    """The original M4 run directory must still exist untouched."""
    root = "artifacts/comparison"
    if not os.path.isdir(root):
        pytest.skip("no M4 comparison artifacts present")
    runs = sorted(d for d in os.listdir(root) if d.startswith("run_"))
    assert runs, "the original M4 run directory was removed"
    original = os.path.join(root, "run_20261021T201001Z")
    if os.path.isdir(original):
        report = json.load(open(os.path.join(original, "reports", "comparison_report.json")))
        # The historical artifact keeps its original content.
        assert report["recommended"] == "lof_novelty"
        assert "selection_evidence_split" not in report
