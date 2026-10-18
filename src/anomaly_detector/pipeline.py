"""End-to-end (offline) pipeline: events -> validated -> windows -> exports.

This module wires the Milestone 1 stages together so scripts and tests share one
implementation. It performs **no** model training or anomaly scoring.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os
import json

from .schemas import StructuredEvent, FeatureWindow, LabelInfo
from .validation import validate_and_normalize_events, ValidationResult
from .preprocessing import aggregate_events_to_windows
from .features import (
    windows_to_feature_dataframe,
    windows_to_metadata_dataframe,
    labels_to_dataframe,
)
from .reporting import generate_data_quality_report


@dataclass
class PipelineResult:
    validation: ValidationResult
    windows: List[FeatureWindow]
    feature_df: Any
    metadata_df: Any
    label_df: Any
    report: Dict[str, Any]
    paths: Dict[str, str] = field(default_factory=dict)


def run_pipeline(raw_events: List[Dict[str, Any]],
                 window_size_minutes: int = 1,
                 labels: Optional[List[LabelInfo]] = None,
                 allow_empty_windows: bool = False) -> PipelineResult:
    """Validate, window, featurize and report a batch of raw event dicts."""
    validation = validate_and_normalize_events(raw_events)
    windows = aggregate_events_to_windows(
        validation.accepted,
        window_size_minutes=window_size_minutes,
        allow_empty_windows=allow_empty_windows,
    )
    feature_df = windows_to_feature_dataframe(windows)
    metadata_df = windows_to_metadata_dataframe(windows)
    label_df = labels_to_dataframe(labels or [])
    report = generate_data_quality_report(validation, windows, labels or [], raw_events=raw_events)
    return PipelineResult(
        validation=validation,
        windows=windows,
        feature_df=feature_df,
        metadata_df=metadata_df,
        label_df=label_df,
        report=report,
    )


def export_result(result: PipelineResult, output_dir: str, prefix: str = "") -> Dict[str, str]:
    """Export features, metadata, labels and the report as CSV/JSON.

    Features are written to ``*_features.csv`` (numerical matrix only) while
    metadata and labels are written to separate files to keep them out of the
    model feature matrix.
    """
    os.makedirs(output_dir, exist_ok=True)
    p = (prefix + "_") if prefix else ""
    paths = {
        "features": os.path.join(output_dir, f"{p}features.csv"),
        "metadata": os.path.join(output_dir, f"{p}metadata.csv"),
        "labels": os.path.join(output_dir, f"{p}labels.csv"),
        "report": os.path.join(output_dir, f"{p}data_quality_report.json"),
    }
    result.feature_df.to_csv(paths["features"], index=False)
    result.metadata_df.to_csv(paths["metadata"], index=False)
    result.label_df.to_csv(paths["labels"], index=False)
    with open(paths["report"], "w", encoding="utf-8") as fh:
        json.dump(result.report, fh, indent=2)
    return paths
