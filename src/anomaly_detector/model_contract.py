"""Model-input contract for the Milestone 3 baseline.

This module is the single place where raw feature tables become estimator input.
It enforces the rules that keep the baseline honest:

* **Explicit allowlist.** Only the columns listed in the configuration are used.
  Labels, identifiers, timestamps, scenario names and split metadata can never
  enter the model matrix.
* **Stable schema.** The feature set and its order are validated. A scoring
  table whose columns differ from the training schema fails with a clear error.
* **Documented missing-value policy.** Missing values are *never* replaced with
  zero. They are either imputed with a statistic learned from the training rows
  only, or the affected samples are excluded under a documented rule.
* **Non-finite rejection.** Infinite values are always rejected with an error.
* **No leakage of splits.** The imputer statistics are fitted on training rows
  only and then applied unchanged to validation/test rows.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class ModelContractError(ValueError):
    """Raised when data violates the model-input contract."""


@dataclass
class MissingValueReport:
    policy: str
    rows_total: int = 0
    rows_dropped: int = 0
    cells_imputed: int = 0
    imputation_values: Dict[str, float] = field(default_factory=dict)
    dropped_columns: List[str] = field(default_factory=list)
    dropped_sample_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "rows_total": self.rows_total,
            "rows_dropped": self.rows_dropped,
            "cells_imputed": self.cells_imputed,
            "imputation_values": dict(self.imputation_values),
            "dropped_columns": list(self.dropped_columns),
            "dropped_sample_keys": list(self.dropped_sample_keys),
        }


@dataclass
class ModelInput:
    """Estimator-ready numeric input plus the provenance of its preparation."""
    X: np.ndarray
    feature_columns: List[str]
    medians: Dict[str, float]
    report: MissingValueReport
    sample_keys: List[str] = field(default_factory=list)
    row_indices: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.X)


def validate_feature_schema(
    frame: pd.DataFrame,
    expected_columns: Sequence[str],
    context: str = "data",
    allow_reorder: bool = True,
    allow_extra: bool = True,
) -> List[str]:
    """Validate that ``frame`` carries the expected feature columns.

    Returns the ordered column list to use.

    * A **missing** feature column always raises :class:`ModelContractError` —
      the model cannot be fed a column it was not trained on.
    * **Extra** columns are tolerated by default: a feature table legitimately
      carries the sample key and other metadata next to the features, and only
      the allowlisted columns ever reach the estimator. Pass
      ``allow_extra=False`` for a strict check.
    * A different column *order* is normalised to the expected order unless
      ``allow_reorder`` is disabled.
    """
    expected = list(expected_columns)
    if not expected:
        raise ModelContractError("expected feature column list is empty")
    if len(set(expected)) != len(expected):
        raise ModelContractError("expected feature column list contains duplicates")

    present = list(frame.columns)
    missing = [c for c in expected if c not in present]
    unexpected = [c for c in present if c not in expected]
    if missing:
        raise ModelContractError(
            f"{context}: feature schema mismatch — missing columns: {missing}; "
            f"expected: {expected}"
        )
    if unexpected and not allow_extra:
        raise ModelContractError(
            f"{context}: feature schema mismatch — unexpected columns: {unexpected}; "
            f"expected: {expected}"
        )
    if not allow_reorder and [c for c in present if c in expected] != expected:
        raise ModelContractError(
            f"{context}: feature column order mismatch — expected {expected}, got {present}"
        )
    return expected


def fit_missing_statistics(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Tuple[Dict[str, float], List[str]]:
    """Learn per-column medians from the given (training) rows.

    Returns ``(medians, all_missing_columns)``. A column whose median cannot be
    computed (every value missing) is reported so it can be dropped from the
    model input instead of being imputed with a meaningless constant.
    """
    medians: Dict[str, float] = {}
    all_missing: List[str] = []
    for col in feature_columns:
        series = pd.to_numeric(frame[col], errors="coerce")
        value = series.median()
        if pd.isna(value):
            all_missing.append(col)
        else:
            medians[col] = float(value)
    return medians, all_missing


def check_finite(matrix: np.ndarray, feature_columns: Sequence[str]) -> None:
    """Reject infinite values in model input. NaN is handled by the policy."""
    if matrix.size == 0:
        return
    infinite_cols = [
        feature_columns[i]
        for i in range(matrix.shape[1])
        if np.isinf(matrix[:, i]).any()
    ]
    if infinite_cols:
        raise ModelContractError(
            f"model input contains infinite values in column(s) {infinite_cols}; "
            "infinities are never imputed — fix the upstream data"
        )


def prepare_model_input(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    medians: Optional[Dict[str, float]] = None,
    policy: str = "median_impute",
    sample_key_column: Optional[str] = "sample_key",
    context: str = "data",
    allow_reorder: bool = True,
    use_columns: Optional[Sequence[str]] = None,
) -> ModelInput:
    """Turn a feature table into estimator input under the missing-value policy.

    Parameters
    ----------
    frame:
        Table containing the feature columns (and optionally a sample key).
    feature_columns:
        The explicit allowlist, in model order.
    medians:
        Imputation statistics learned from the training rows. Required for
        ``median_impute`` when preparing validation/test rows; when ``None``
        they are learned from ``frame`` itself (training use only).
    policy:
        ``median_impute`` | ``drop`` | ``error``.
    use_columns:
        The columns the *fitted* model actually uses. Scoring passes the
        trained model's column list so that a column dropped at training time
        (no usable values) stays dropped, instead of silently re-entering the
        matrix.
    """
    columns = validate_feature_schema(frame, feature_columns, context=context,
                                       allow_reorder=allow_reorder)
    report = MissingValueReport(policy=policy, rows_total=len(frame))

    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")

    # Infinities are never acceptable, whatever the missing-value policy.
    inf_mask = np.isinf(numeric.to_numpy(dtype="float64", na_value=np.nan))
    if inf_mask.any():
        offending = [columns[i] for i in range(len(columns)) if inf_mask[:, i].any()]
        raise ModelContractError(
            f"{context}: feature column(s) {offending} contain +/-infinity; "
            "non-finite values must be fixed before scoring"
        )

    missing_mask = numeric.isna()
    n_missing_cells = int(missing_mask.to_numpy().sum())

    if policy == "error":
        if n_missing_cells:
            bad_cols = [c for c in columns if missing_mask[c].any()]
            raise ModelContractError(
                f"{context}: {n_missing_cells} missing value(s) in column(s) {bad_cols}; "
                "the missing-value policy is 'error'"
            )
        used_medians: Dict[str, float] = {}
        dropped_columns: List[str] = []
        keep = np.ones(len(frame), dtype=bool)
    elif policy == "drop":
        keep = ~missing_mask.any(axis=1)
        dropped = np.where(~keep)[0]
        used_medians = {}
        dropped_columns = []
        report.rows_dropped = int(len(dropped))
        if len(dropped):
            if sample_key_column and sample_key_column in frame.columns:
                keys = frame[sample_key_column].astype(str).tolist()
                report.dropped_sample_keys = [keys[i] for i in dropped]
            else:
                report.dropped_sample_keys = [str(i) for i in dropped]
    elif policy == "median_impute":
        if use_columns is not None:
            requested = list(use_columns)
            unknown = [c for c in requested if c not in columns]
            if unknown:
                raise ModelContractError(
                    f"{context}: the model expects column(s) {unknown} which are not in "
                    f"the data; expected columns: {columns}"
                )
            dropped_columns = [c for c in columns if c not in requested]
            used_medians = dict(medians or {})
        elif medians is None:
            used_medians, dropped_columns = fit_missing_statistics(numeric, columns)
        else:
            used_medians = dict(medians)
            dropped_columns = [c for c in columns if c not in used_medians]
        keep = np.ones(len(frame), dtype=bool)
        report.imputation_values = dict(used_medians)
    else:
        raise ModelContractError(
            f"unknown missing-value policy {policy!r}; expected one of "
            "median_impute, drop, error"
        )

    report.dropped_columns = list(dropped_columns)

    kept_columns = [c for c in columns if c not in dropped_columns]
    numeric = numeric[kept_columns]
    missing_mask = numeric.isna()

    if policy == "median_impute" and n_missing_cells:
        without_statistic = [
            col for col in kept_columns
            if missing_mask[col].any() and col not in used_medians
        ]
        if without_statistic:
            raise ModelContractError(
                f"{context}: no imputation statistic is available for column(s) "
                f"{without_statistic}; the missing-value policy is 'median_impute' but "
                "the statistic must be learned from the training split"
            )
        imputed = numeric.copy()
        for col in kept_columns:
            if missing_mask[col].any():
                imputed[col] = imputed[col].fillna(used_medians[col])
        report.cells_imputed = int(missing_mask.to_numpy().sum())
        numeric = imputed

    numeric = numeric.loc[keep]
    matrix = numeric.to_numpy(dtype="float64")

    # Final gate: the estimator must never see NaN or infinity.
    if matrix.size and not np.isfinite(matrix).all():
        raise ModelContractError(
            f"{context}: non-finite values remain after applying the missing-value policy"
        )
    check_finite(matrix, kept_columns)

    sample_keys: List[str] = []
    if sample_key_column and sample_key_column in frame.columns:
        sample_keys = [str(v) for v in frame.loc[keep, sample_key_column].tolist()]
    row_indices = [int(i) for i in np.where(keep)[0]]

    return ModelInput(
        X=matrix,
        feature_columns=kept_columns,
        medians=used_medians,
        report=report,
        sample_keys=sample_keys,
        row_indices=row_indices,
    )


def select_training_rows(
    labels: pd.Series,
    policy: str = "normal_only",
    label_column: str = "is_anomaly",
) -> np.ndarray:
    """Return the boolean mask of rows the estimator may be fitted on.

    ``normal_only`` keeps rows whose label is explicitly False (labelled
    anomalies are excluded so the model learns normal behaviour). Rows with an
    unknown label (NaN) are always excluded.
    """
    if policy == "all":
        return np.ones(len(labels), dtype=bool)
    if policy != "normal_only":
        raise ModelContractError(
            f"unknown training-data policy {policy!r}; expected 'normal_only' or 'all'"
        )
    values = labels.to_numpy()
    mask = np.zeros(len(values), dtype=bool)
    for i, value in enumerate(values):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        if isinstance(value, (bool, np.bool_)):
            mask[i] = not value
        elif value in (0, 1):
            mask[i] = value == 0
        else:
            raise ModelContractError(
                f"{label_column!r} must contain booleans or 0/1, got {value!r}"
            )
    return mask


def assert_unique_sample_keys(keys: Sequence[str], context: str = "predictions") -> None:
    """Fail when sample identifiers are not unique (they address rows)."""
    seen = set()
    duplicates = []
    for key in keys:
        if key in seen:
            duplicates.append(key)
        else:
            seen.add(key)
    if duplicates:
        unique_dupes = sorted(set(duplicates))
        raise ModelContractError(
            f"{context}: duplicate sample identifiers detected "
            f"({len(unique_dupes)} unique value(s), e.g. {unique_dupes[:3]})"
        )
