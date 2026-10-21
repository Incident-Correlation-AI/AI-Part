"""Tests for Milestone 4 comparison plumbing."""
import os, json
import numpy as np
import pandas as pd

from anomaly_detector.comparison_config import load_comparison_config
from anomaly_detector.mad_baseline import MADBaseline
from anomaly_detector.lof_baseline import LOFBaseline
from anomaly_detector.model_contract import ModelContractError


def test_comparison_config_loads_and_validates():
    cfg = load_comparison_config(os.path.join("configs", "comparison.yaml"))
    assert cfg.data["feature_columns"]
    assert cfg.evaluation["use_test_only_after_selection"] is True
    assert cfg.threshold["method"] in {"validation_f1", "validation_precision", "validation_recall", "train_quantile"}


def test_mad_fits_on_training_only():
    Xtr = np.random.RandomState(42).randn(50, 4)
    Xval = np.random.RandomState(7).randn(20, 4)
    mad = MADBaseline(["a", "b", "c", "d"], agg="max_abs")
    mad.fit(Xtr)
    s = mad.score(Xval)
    assert s.shape == (20,)
    assert mad.score_direction == "higher_is_more_anomalous"
    # constant feature
    Xc = Xtr.copy(); Xc[:, 0] = 5.0
    mad2 = MADBaseline(["a", "b", "c", "d"])
    mad2.fit(Xc)
    assert "a" in mad2.state.constant_features


def test_mad_shape_and_finite_checks():
    mad = MADBaseline(["x"], agg="max_abs")
    mad.fit(np.array([[1.0],[2.0],[3.0]]))
    try:
        mad.score(np.array([[1.0, 2.0]]))
        assert False
    except ModelContractError:
        pass
    try:
        mad.fit(np.array([[np.nan]]))
        assert False
    except ModelContractError:
        pass


def test_lof_novelty_can_score_unseen():
    Xtr = np.random.RandomState(1).randn(40, 3)
    Xte = np.random.RandomState(2).randn(15, 3)
    lof = LOFBaseline(["a", "b", "c"], n_neighbors=5)
    lof.fit(Xtr)
    s = lof.score(Xte)
    assert s.shape == (15,)
    assert lof.contract()["novelty"] is True
    assert "StandardScaler" in lof.contract()["preprocessing"]


def test_lof_handles_small_train():
    Xtr = np.array([[1.0, 2.0], [1.1, 2.1]])
    lof = LOFBaseline(["a", "b"], n_neighbors=20)
    lof.fit(Xtr)
    assert lof.n_neighbors == 1


def test_comparison_artifacts_exist():
    # find latest run
    root = "artifacts/comparison"
    if not os.path.isdir(root):
        return
    runs = sorted([os.path.join(root, d) for d in os.listdir(root) if d.startswith("run_")])
    if not runs:
        return
    rdir = runs[-1]
    assert os.path.isfile(os.path.join(rdir, "reports", "comparison_report.json"))
    assert os.path.isfile(os.path.join(rdir, "reports", "comparison_report.md"))
    assert os.path.isfile(os.path.join(rdir, "reports", "comparison_table.csv"))
    assert os.path.isfile(os.path.join(rdir, "comparison_manifest.json"))
    rep = json.load(open(os.path.join(rdir, "reports", "comparison_report.json")))
    assert rep["predeclared_criteria"]
    assert rep["recommended"] in {"m3_isolation_forest_reference", "robust_mad_baseline", "lof_novelty"}
