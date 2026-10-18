"""Data-quality reporting."""
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from .schemas import FeatureWindow, LabelInfo
from .validation import ValidationResult


# Fields whose missingness is worth tracking in the input report.
IMPORTANT_EVENT_FIELDS = ["event_id", "timestamp", "host_id", "service",
                          "event_type", "level", "attributes"]

# Numerical feature columns whose availability/missingness we report.
NUMERICAL_FEATURE_COLUMNS = [
    "event_count", "error_count", "error_ratio", "http_request_count",
    "http_5xx_count", "http_5xx_ratio", "auth_failure_count", "restart_count",
    "mean_latency_ms", "p95_latency_ms", "latency_sample_count",
    "http_request_eligible_count", "error_eligible_count",
]


def _missingness_by_field(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute missingness counts for important fields over raw event dicts."""
    total = len(events)
    out: Dict[str, Dict[str, Any]] = {}
    for field in IMPORTANT_EVENT_FIELDS:
        missing = 0
        for ev in events:
            val = ev.get(field) if isinstance(ev, dict) else None
            if val is None or (isinstance(val, str) and val.strip() == ""):
                missing += 1
        out[field] = {
            "missing": missing,
            "present": total - missing,
            "missing_ratio": (missing / total) if total else None,
        }
    return out


def _feature_availability(windows: List[FeatureWindow]) -> Dict[str, Dict[str, Any]]:
    """Compute availability/missingness for numerical feature columns."""
    total = len(windows)
    out: Dict[str, Dict[str, Any]] = {}
    if total == 0:
        for col in NUMERICAL_FEATURE_COLUMNS:
            out[col] = {"available": 0, "missing": 0, "missing_ratio": None}
        return out

    feature_dicts = [w.to_features_dict() for w in windows]
    for col in NUMERICAL_FEATURE_COLUMNS:
        missing = sum(1 for fd in feature_dicts if fd.get(col) is None)
        out[col] = {
            "available": total - missing,
            "missing": missing,
            "missing_ratio": missing / total,
        }
    return out


def generate_data_quality_report(validation_result: ValidationResult,
                                 windows: List[FeatureWindow],
                                 labels: Optional[List[LabelInfo]] = None) -> Dict[str, Any]:
    """Generate a comprehensive data-quality report.

    Parameters
    ----------
    validation_result:
        Output of ``validate_and_normalize_events`` (accepted/rejected/etc.).
    windows:
        Feature windows produced by the aggregation step.
    labels:
        Optional ground-truth labels (only for labeled synthetic data).
    """
    labels = labels or []
    accepted_events = [e.to_dict() for e in validation_result.accepted]

    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": {
            "total_processed": validation_result.total_processed,
            "accepted_count": len(validation_result.accepted),
            "rejected_count": len(validation_result.rejected),
            "duplicated_count": len(validation_result.duplicated),
            "partially_usable_count": len(validation_result.partially_usable),
        },
        "rejection_reasons": {},
        "missingness_by_field": _missingness_by_field(
            [e.to_dict() for e in validation_result.accepted]
            + [r.get("normalized", {}) for r in validation_result.rejected]
        ),
        "windows": {
            "count": len(windows),
            "hosts": sorted({w.metadata.host_id for w in windows}),
            "services": sorted({w.metadata.service for w in windows}),
            "num_hosts": len({w.metadata.host_id for w in windows}),
            "num_services": len({w.metadata.service for w in windows}),
        },
        "labels": {
            "count": len(labels),
            "anomaly_count": sum(1 for l in labels if l.is_anomaly),
            "normal_count": sum(1 for l in labels if not l.is_anomaly),
            "scenario_distribution": {},
        },
        "features": {
            "availability": _feature_availability(windows),
        },
    }

    # Count rejection reasons
    for rej in validation_result.rejected:
        for reason in rej.get("reasons", []):
            key = reason[:80]  # Truncate long reasons
            report["rejection_reasons"][key] = report["rejection_reasons"].get(key, 0) + 1

    # Window time range
    if windows:
        report["windows"]["time_range"] = {
            "start": min(w.metadata.window_start for w in windows).isoformat(),
            "end": max(w.metadata.window_end for w in windows).isoformat(),
        }
        report["windows"]["window_size_minutes"] = windows[0].metadata.window_size_minutes

    # Label distribution
    for lbl in labels:
        sc = lbl.scenario_name
        report["labels"]["scenario_distribution"][sc] = \
            report["labels"]["scenario_distribution"].get(sc, 0) + 1

    return report
