"""Training-ready dataset construction and export (Milestone 2).

Produces a *modelling-ready* dataset with three strictly separate tables:

* ``features.csv``      — eligible model features only (no labels, no ids, no timestamps)
* ``labels.csv``        — ground truth, aligned by row to the features
* ``metadata.csv``      — descriptive/identifier columns, aligned by row

Plus ``sample_index.csv`` (an explicit ``row_index -> sample_key`` map) and
``manifest.json`` (seed, sizes, feature list, leakage status and SHA-256
checksums). Alignment is verified before anything is written: a mismatch fails
loudly rather than producing silently misaligned data.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import hashlib
import json
import os

import pandas as pd

from .dataset import AlignedDataset
from .leakage import LeakageReport, audit_leakage
from .readiness_config import ReadinessConfig
from .splits import SplitResult


@dataclass
class TrainingReadyDataset:
    features: pd.DataFrame
    labels: Optional[pd.DataFrame]
    metadata: pd.DataFrame
    sample_index: pd.DataFrame
    eligible_features: List[str]
    label_columns: List[str]
    metadata_columns: List[str]
    leakage: LeakageReport
    seed: int
    source_paths: Dict[str, str] = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        return len(self.features)


def verify_alignment(dataset: TrainingReadyDataset, key_columns: List[str]) -> None:
    """Fail loudly if features/labels/metadata/index are not row-aligned."""
    n = len(dataset.features)
    if len(dataset.metadata) != n:
        raise ValueError(f"features ({n}) and metadata ({len(dataset.metadata)}) row counts differ")
    if dataset.labels is not None and len(dataset.labels) != n:
        raise ValueError(f"features ({n}) and labels ({len(dataset.labels)}) row counts differ")
    if len(dataset.sample_index) != n:
        raise ValueError(f"features ({n}) and sample index ({len(dataset.sample_index)}) row counts differ")

    idx = dataset.sample_index.reset_index(drop=True)
    if list(idx["row_index"]) != list(range(n)):
        raise ValueError("sample index row_index is not the contiguous range 0..n-1")

    # Recompute the key from metadata and compare to the stored key.
    from .dataset import make_sample_key
    recomputed = make_sample_key(dataset.metadata.reset_index(drop=True), key_columns).tolist()
    stored = idx["sample_key"].tolist()
    if recomputed != stored:
        mismatch = next((i for i, (a, b) in enumerate(zip(recomputed, stored)) if a != b), None)
        raise ValueError(f"sample key mismatch at row {mismatch}")


def build_training_ready(
    aligned: AlignedDataset,
    config: ReadinessConfig,
    strict_leakage: bool = True,
    source_paths: Optional[Dict[str, str]] = None,
) -> TrainingReadyDataset:
    """Build the aligned, leakage-audited training-ready dataset."""
    leakage = audit_leakage(aligned, config)
    if strict_leakage and not leakage.is_clean:
        details = "; ".join(f"{f.code}:{f.column}" for f in leakage.findings if f.severity == "error")
        raise ValueError(f"refusing to export: leakage audit found errors ({details})")

    features = aligned.features[leakage.eligible_features].copy()
    metadata = aligned.metadata.copy()
    labels = aligned.labels.copy() if aligned.labels is not None else None

    n = len(features)
    sample_index = pd.DataFrame({
        "row_index": list(range(n)),
        "sample_key": aligned.sample_key.reset_index(drop=True).tolist(),
    })

    dataset = TrainingReadyDataset(
        features=features,
        labels=labels,
        metadata=metadata,
        sample_index=sample_index,
        eligible_features=list(leakage.eligible_features),
        label_columns=[c for c in (labels.columns if labels is not None else []) if c != "sample_key"],
        metadata_columns=list(metadata.columns),
        leakage=leakage,
        seed=config.split.seed,
        source_paths=dict(source_paths or {}),
    )
    verify_alignment(dataset, aligned.key_columns)
    return dataset


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_table(df: pd.DataFrame, path: str) -> Dict[str, Any]:
    df.to_csv(path, index=False)
    with open(path, "r", encoding="utf-8") as fh:
        rows = sum(1 for _ in fh) - 1
    return {
        "sha256": _sha256(path),
        "rows": rows,
        "columns": list(df.columns),
    }


def _label_distribution(labels: Optional[pd.DataFrame], label_column: str) -> Dict[str, Any]:
    if labels is None or label_column not in labels.columns:
        return {}
    positive = int((labels[label_column] == True).sum())  # noqa: E712
    negative = int((labels[label_column] == False).sum())  # noqa: E712
    total = positive + negative
    return {
        "positive": positive,
        "negative": negative,
        "missing": int(labels[label_column].isna().sum()),
        "positive_ratio": (positive / total) if total else 0.0,
    }


def _manifest(
    dataset: TrainingReadyDataset,
    files: Dict[str, Dict[str, Any]],
    key_columns: List[str],
    label_column: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": dataset.seed,
        "row_count": dataset.row_count,
        "feature_columns": list(dataset.features.columns),
        "label_columns": list(dataset.label_columns),
        "metadata_columns": list(dataset.metadata_columns),
        "alignment": {
            "method": "row order; row i is the same sample across features/labels/metadata",
            "sample_index_file": "sample_index.csv",
            "key_columns": list(key_columns),
        },
        "leakage": {
            "clean": dataset.leakage.is_clean,
            "error_count": dataset.leakage.error_count,
            "eligible_feature_count": len(dataset.eligible_features),
        },
        "label_distribution": _label_distribution(dataset.labels, label_column),
        "source_paths": dataset.source_paths,
        "files": files,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_training_ready(
    dataset: TrainingReadyDataset,
    out_dir: str,
    key_columns: List[str],
    label_column: str,
) -> str:
    """Write the full training-ready dataset. Returns the manifest path."""
    os.makedirs(out_dir, exist_ok=True)
    files: Dict[str, Dict[str, Any]] = {}
    files["features.csv"] = _write_table(dataset.features, os.path.join(out_dir, "features.csv"))
    files["metadata.csv"] = _write_table(dataset.metadata, os.path.join(out_dir, "metadata.csv"))
    if dataset.labels is not None:
        files["labels.csv"] = _write_table(dataset.labels, os.path.join(out_dir, "labels.csv"))
    files["sample_index.csv"] = _write_table(dataset.sample_index, os.path.join(out_dir, "sample_index.csv"))

    manifest = _manifest(dataset, files, key_columns, label_column)
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest_path


def write_split(
    dataset: TrainingReadyDataset,
    split: SplitResult,
    out_dir: str,
    key_columns: List[str],
    label_column: str,
) -> str:
    """Write per-split feature/label/metadata tables plus a split manifest.

    Returns the split manifest path.
    """
    os.makedirs(out_dir, exist_ok=True)
    split_files: Dict[str, Dict[str, Any]] = {}

    for name in split.split_order:
        idx = split.assignments.get(name, [])
        sub_dir = os.path.join(out_dir, name)
        os.makedirs(sub_dir, exist_ok=True)
        entry: Dict[str, Any] = {"rows": len(idx)}

        feat = dataset.features.iloc[idx].reset_index(drop=True)
        meta = dataset.metadata.iloc[idx].reset_index(drop=True)
        index = dataset.sample_index.iloc[idx].reset_index(drop=True)
        entry["files"] = {}
        entry["files"]["features.csv"] = _write_table(feat, os.path.join(sub_dir, "features.csv"))
        entry["files"]["metadata.csv"] = _write_table(meta, os.path.join(sub_dir, "metadata.csv"))
        entry["files"]["sample_index.csv"] = _write_table(index, os.path.join(sub_dir, "sample_index.csv"))
        if dataset.labels is not None:
            lab = dataset.labels.iloc[idx].reset_index(drop=True)
            entry["files"]["labels.csv"] = _write_table(lab, os.path.join(sub_dir, "labels.csv"))
            entry["label_distribution"] = _label_distribution(lab, label_column)
        split_files[name] = entry

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": split.seed,
        "strategy": split.strategy,
        "ratios": split.ratios,
        "split_order": split.split_order,
        "group_column": split.group_column,
        "time_column": split.time_column,
        "total_rows": dataset.row_count,
        "splits": split_files,
        "checks": list(split.checks),
        "warnings": list(split.warnings),
        "leakage": {
            "clean": dataset.leakage.is_clean,
            "error_count": dataset.leakage.error_count,
        },
        "feature_columns": list(dataset.features.columns),
    }
    manifest_path = os.path.join(out_dir, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest_path
