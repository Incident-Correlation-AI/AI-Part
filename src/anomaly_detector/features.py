"""Feature extraction utilities.

Three distinct tables are produced and must never be conflated:

* ``features`` — the numerical model feature matrix (no ids, no timestamps, no labels)
* ``metadata`` — window traceability (host/service/time range/window size)
* ``labels``   — ground-truth labels, kept separate by design

``FEATURE_COLUMNS`` is the documented, stable contract for the model matrix.
"""
from typing import List, Any
import pandas as pd

from .schemas import FeatureWindow, LabelInfo

# The stable order/contract of the numerical feature matrix.
FEATURE_COLUMNS = [
    "event_count",
    "error_count",
    "error_ratio",
    "http_request_count",
    "http_5xx_count",
    "http_5xx_ratio",
    "auth_failure_count",
    "restart_count",
    "mean_latency_ms",
    "p95_latency_ms",
    "latency_sample_count",
    "http_request_eligible_count",
    "error_eligible_count",
]

# Substrings that must never appear in a feature column name.
LEAKAGE_INDICATORS = ["is_anomaly", "anomaly", "label", "scenario",
                      "ground_truth", "true_label", "host_id", "service"]


def windows_to_feature_dataframe(windows: List[FeatureWindow]) -> pd.DataFrame:
    """Convert windows to the numerical feature matrix (no metadata)."""
    if not windows:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    return pd.DataFrame([w.to_features_dict() for w in windows],
                        columns=FEATURE_COLUMNS)


def windows_to_metadata_dataframe(windows: List[FeatureWindow]) -> pd.DataFrame:
    """Convert windows to a metadata table for traceability."""
    columns = ["window_start", "window_end", "host_id", "service", "window_size_minutes"]
    if not windows:
        return pd.DataFrame(columns=metadata_columns())
    return pd.DataFrame([w.to_metadata_dict() for w in windows], columns=columns)


def labels_to_dataframe(labels: List[LabelInfo]) -> pd.DataFrame:
    """Convert ground-truth labels to a separate dataframe."""
    columns = ["window_start", "window_end", "host_id", "service",
               "is_anomaly", "scenario_name", "intensity", "description"]
    if not labels:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([l.to_dict() for l in labels], columns=columns)


def check_label_leakage(feature_df: pd.DataFrame) -> List[str]:
    """Return feature columns that look like leaked labels or metadata.

    An empty list means the matrix is clean. Used by tests and by the CLI
    scripts, which abort if any leakage is detected.
    """
    leaks: List[str] = []
    for col in feature_df.columns:
        col_lower = str(col).lower()
        if any(ind in col_lower for ind in LEAKAGE_INDICATORS):
            leaks.append(col)
    return leaks
