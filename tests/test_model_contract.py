"""Tests for the model-input contract (schema, missing values, non-finite)."""
import numpy as np
import pandas as pd
import pytest

from anomaly_detector.model_contract import (
    ModelContractError,
    assert_unique_sample_keys,
    check_finite,
    fit_missing_statistics,
    prepare_model_input,
    select_training_rows,
    validate_feature_schema,
)


FEATURES = ["event_count", "error_count", "error_ratio", "mean_latency_ms"]


def _frame(rows=6, with_missing=False):
    data = {
        "event_count": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0][:rows],
        "error_count": [1.0, 0.0, 2.0, 1.0, 0.0, 3.0][:rows],
        "error_ratio": [0.1, 0.0, 0.16, 0.07, 0.0, 0.2][:rows],
        "mean_latency_ms": [120.0, 118.0, 121.0, 119.0, 120.0, 122.0][:rows],
    }
    frame = pd.DataFrame(data)
    frame.insert(0, "sample_key", [f"k{i}" for i in range(rows)])
    if with_missing:
        frame.loc[1, "mean_latency_ms"] = np.nan
    return frame


def test_schema_missing_column_fails():
    frame = _frame().drop(columns=["error_ratio"])
    with pytest.raises(ModelContractError) as exc:
        validate_feature_schema(frame, FEATURES)
    assert "error_ratio" in str(exc.value)


def test_schema_extra_column_is_tolerated():
    # A feature table legitimately carries the sample key and other metadata;
    # only the allowlisted columns ever reach the estimator.
    frame = _frame()
    frame["extra"] = 1.0
    assert validate_feature_schema(frame, FEATURES) == FEATURES


def test_schema_extra_column_fails_in_strict_mode():
    frame = _frame()
    frame["extra"] = 1.0
    with pytest.raises(ModelContractError) as exc:
        validate_feature_schema(frame, FEATURES, allow_extra=False)
    assert "extra" in str(exc.value)


def test_schema_order_is_normalised():
    frame = _frame()[list(reversed(FEATURES))]
    assert validate_feature_schema(frame, FEATURES) == FEATURES


def test_schema_strict_order_can_fail():
    frame = _frame()[list(reversed(FEATURES))]
    with pytest.raises(ModelContractError):
        validate_feature_schema(frame, FEATURES, allow_reorder=False)


def test_median_impute_uses_training_median_not_zero():
    frame = _frame(with_missing=True)
    medians = {
        "event_count": 12.0, "error_count": 1.0,
        "error_ratio": 0.1, "mean_latency_ms": 118.0,
    }
    prepared = prepare_model_input(frame, FEATURES, medians=medians, policy="median_impute")
    row = prepared.X[1]
    # The imputed value must be the supplied statistic, never 0.
    assert row[FEATURES.index("mean_latency_ms")] == pytest.approx(118.0)
    assert row[FEATURES.index("mean_latency_ms")] != 0.0
    assert prepared.report.cells_imputed == 1
    assert np.isfinite(prepared.X).all()


def test_median_impute_without_a_statistic_fails():
    frame = _frame(with_missing=True)
    with pytest.raises(ModelContractError) as exc:
        prepare_model_input(
            frame, FEATURES,
            medians={"event_count": 12.0},
            use_columns=FEATURES,
            policy="median_impute",
        )
    assert "mean_latency_ms" in str(exc.value)


def test_columns_outside_the_model_are_dropped():
    frame = _frame(with_missing=True)
    prepared = prepare_model_input(
        frame, FEATURES,
        medians={"mean_latency_ms": 118.0},
        use_columns=["mean_latency_ms"],
        policy="median_impute",
    )
    assert prepared.feature_columns == ["mean_latency_ms"]
    assert prepared.report.dropped_columns == ["event_count", "error_count", "error_ratio"]


def test_median_impute_learns_statistics_from_given_rows():
    frame = _frame(with_missing=True)
    medians, all_missing = fit_missing_statistics(frame, FEATURES)
    assert all_missing == []
    # Non-missing latency values are [120, 121, 119, 120, 122] -> median 120.
    assert medians["mean_latency_ms"] == pytest.approx(120.0)


def test_all_missing_column_is_dropped_and_reported():
    frame = _frame()
    frame["mean_latency_ms"] = np.nan
    medians, all_missing = fit_missing_statistics(frame, FEATURES)
    assert all_missing == ["mean_latency_ms"]
    prepared = prepare_model_input(frame, FEATURES, medians=medians, policy="median_impute")
    assert "mean_latency_ms" not in prepared.feature_columns
    assert prepared.report.dropped_columns == ["mean_latency_ms"]


def test_drop_policy_removes_rows_and_reports_them():
    frame = _frame(with_missing=True)
    prepared = prepare_model_input(frame, FEATURES, policy="drop")
    assert len(prepared.X) == 5
    assert prepared.report.rows_dropped == 1
    assert prepared.report.dropped_sample_keys == ["k1"]


def test_error_policy_raises_on_missing():
    frame = _frame(with_missing=True)
    with pytest.raises(ModelContractError) as exc:
        prepare_model_input(frame, FEATURES, policy="error")
    assert "missing" in str(exc.value).lower()


def test_infinite_values_are_rejected():
    frame = _frame()
    frame.loc[0, "event_count"] = np.inf
    with pytest.raises(ModelContractError) as exc:
        prepare_model_input(frame, FEATURES, policy="median_impute")
    assert "infinity" in str(exc.value).lower()


def test_negative_infinity_is_rejected():
    frame = _frame()
    frame.loc[0, "event_count"] = -np.inf
    with pytest.raises(ModelContractError):
        prepare_model_input(frame, FEATURES, policy="median_impute")


def test_check_finite_reports_column():
    matrix = np.array([[1.0, np.inf], [2.0, 3.0]])
    with pytest.raises(ModelContractError) as exc:
        check_finite(matrix, ["a", "b"])
    assert "'b'" in str(exc.value)


def test_unknown_policy_rejected():
    with pytest.raises(ModelContractError):
        prepare_model_input(_frame(), FEATURES, policy="zero_fill")


def test_select_training_rows_normal_only():
    labels = pd.Series([False, True, False, None, True, False])
    mask = select_training_rows(labels, policy="normal_only")
    assert list(mask) == [True, False, True, False, False, True]


def test_select_training_rows_all():
    labels = pd.Series([False, True, False])
    assert list(select_training_rows(labels, policy="all")) == [True, True, True]


def test_select_training_rows_unknown_policy():
    with pytest.raises(ModelContractError):
        select_training_rows(pd.Series([False]), policy="whatever")


def test_select_training_rows_rejects_non_binary_labels():
    with pytest.raises(ModelContractError):
        select_training_rows(pd.Series(["yes"]), policy="normal_only")


def test_duplicate_sample_keys_rejected():
    with pytest.raises(ModelContractError):
        assert_unique_sample_keys(["a", "b", "a"])
    assert_unique_sample_keys(["a", "b"])
