"""Tests for the Isolation Forest baseline model."""
import numpy as np
import pandas as pd
import pytest

from anomaly_detector.baseline_config import ModelConfig, load_baseline_config
from anomaly_detector.baseline_model import BaselineDetector, fit_baseline
from anomaly_detector.model_contract import ModelContractError

FEATURES = ["event_count", "error_count", "error_ratio", "mean_latency_ms"]


def _config(**overrides):
    cfg = load_baseline_config(None)
    cfg.data.feature_columns = list(FEATURES)
    cfg.model.n_estimators = 20
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _training_frame(rows=40):
    features = pd.DataFrame({
        "event_count": [10.0 + (i % 7) for i in range(rows)],
        "error_count": [0.0 + (i % 3) for i in range(rows)],
        "error_ratio": [0.0] * rows,
        "mean_latency_ms": [50.0 + (i % 5) for i in range(rows)],
    })
    labels = pd.Series([False] * rows)
    return features, labels


def test_fit_and_score_small_dataset():
    features, labels = _training_frame()
    cfg = _config()
    detector, report = fit_baseline(features, labels, cfg)
    assert detector.is_fitted
    assert report.rows_used == 40
    scores = detector.score(features[FEATURES].to_numpy(dtype="float64"))
    assert scores.shape == (40,)
    assert np.isfinite(scores).all()


def test_lower_score_means_more_anomalous():
    features, labels = _training_frame()
    cfg = _config()
    detector, _ = fit_baseline(features, labels, cfg)
    normal = np.array([[11.0, 1.0, 0.0, 51.0]])
    outlier = np.array([[5000.0, 900.0, 0.9, 9000.0]])
    normal_score = detector.score(normal)[0]
    outlier_score = detector.score(outlier)[0]
    assert outlier_score < normal_score


def test_flag_boundary_is_inclusive():
    scores = np.array([-1.0, -0.5, 0.0, 0.5])
    flags = BaselineDetector.flag(scores, 0.0)
    assert list(flags) == [True, True, True, False]


def test_anomaly_score_is_sign_flipped():
    features, labels = _training_frame()
    detector, _ = fit_baseline(features, labels, _config())
    X = np.array([[11.0, 1.0, 0.0, 51.0]])
    assert detector.anomaly_score(X)[0] == pytest.approx(-detector.score(X)[0])


def test_normal_only_policy_excludes_anomalous_rows():
    features, labels = _training_frame()
    labels.iloc[:5] = True
    detector, report = fit_baseline(features, labels, _config())
    assert report.rows_used == 35
    assert report.rows_excluded_anomalous == 5
    assert report.label_distribution == {"positive": 5, "negative": 35}
    # The estimator was fitted on 35 rows, so its effective sample budget is 35.
    assert detector.estimator.max_samples_ == 35


def test_all_policy_uses_every_row():
    features, labels = _training_frame()
    labels.iloc[:5] = True
    cfg = _config()
    cfg.data.training_data_policy = "all"
    detector, report = fit_baseline(features, labels, cfg)
    assert report.rows_used == 40
    assert report.rows_excluded_anomalous == 0
    assert detector.estimator.max_samples_ == 40


def test_fit_with_no_usable_rows_fails():
    features, labels = _training_frame(10)
    labels[:] = True
    with pytest.raises(ModelContractError):
        fit_baseline(features, labels, _config())


def test_save_and_load_round_trip(tmp_path):
    features, labels = _training_frame()
    detector, _ = fit_baseline(features, labels, _config())
    path = str(tmp_path / "model.joblib")
    detector.save(path)
    loaded = BaselineDetector.load(path)
    assert loaded.feature_columns == detector.feature_columns
    assert loaded.expected_feature_columns == detector.expected_feature_columns
    X = np.array([[11.0, 1.0, 0.0, 51.0], [12.0, 2.0, 0.0, 52.0]])
    assert loaded.score(X) == pytest.approx(detector.score(X))


def test_load_rejects_missing_file(tmp_path):
    with pytest.raises(ModelContractError):
        BaselineDetector.load(str(tmp_path / "nope.joblib"))


def test_load_rejects_foreign_file(tmp_path):
    import joblib

    path = str(tmp_path / "other.joblib")
    joblib.dump({"not": "a model"}, path)
    with pytest.raises(ModelContractError):
        BaselineDetector.load(path)


def test_scoring_rejects_wrong_column_count():
    features, labels = _training_frame()
    detector, _ = fit_baseline(features, labels, _config())
    with pytest.raises(ModelContractError):
        detector.score(np.zeros((2, 3)))


def test_scoring_rejects_non_finite():
    features, labels = _training_frame()
    detector, _ = fit_baseline(features, labels, _config())
    with pytest.raises(ModelContractError):
        detector.score(np.array([[np.nan, 1.0, 0.0, 1.0]]))


def test_unfitted_detector_cannot_score():
    detector = BaselineDetector(ModelConfig(), FEATURES)
    with pytest.raises(ModelContractError):
        detector.score(np.zeros((1, 4)))


def test_score_contract_is_documented():
    detector = BaselineDetector(ModelConfig(), FEATURES)
    contract = detector.score_contract
    assert contract["score_direction"] == "lower_is_more_anomalous"
    assert contract["is_probability"] == "false"
    assert "score_samples" in contract["score_name"]
