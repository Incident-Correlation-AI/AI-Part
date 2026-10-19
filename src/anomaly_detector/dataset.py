"""Dataset loading and row alignment helpers.

The Milestone 1 pipeline exports three separate tables (features, metadata,
labels). Milestone 2 needs to reason about them **jointly** while keeping them
separate for modelling. This module loads those tables and aligns them on a
stable sample key without merging labels or identifiers into the features.

Alignment is explicit: unmatched rows are reported, never silently dropped.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os

import pandas as pd


DEFAULT_KEY_COLUMNS = ["host_id", "service", "window_start"]


def load_table(path: str) -> pd.DataFrame:
    """Load a CSV or Parquet table. Raises FileNotFoundError with a clear message."""
    if not path:
        raise ValueError("A path is required to load a table")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Table not found: {path}")
    lower = path.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format (expected .csv or .parquet): {path}")


def make_sample_key(df: pd.DataFrame, key_columns: Optional[List[str]] = None) -> pd.Series:
    """Build a stable string sample key from one or more columns.

    Missing/null components are rendered as ``""`` so the key stays buildable;
    callers can detect incomplete keys via :func:`has_complete_key`.
    """
    key_columns = key_columns or DEFAULT_KEY_COLUMNS
    missing = [c for c in key_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Cannot build sample key; missing columns: {missing}")
    parts = [df[c].astype("string").fillna("") for c in key_columns]
    key = parts[0]
    for extra in parts[1:]:
        key = key.str.cat(extra, sep="|")
    return key


def has_complete_key(df: pd.DataFrame, key_columns: Optional[List[str]] = None) -> pd.Series:
    """True where every key component is present and non-empty."""
    key_columns = key_columns or DEFAULT_KEY_COLUMNS
    missing = [c for c in key_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Cannot check sample key; missing columns: {missing}")
    complete = pd.Series(True, index=df.index)
    for c in key_columns:
        complete &= df[c].notna() & (df[c].astype("string").str.strip() != "")
    return complete


@dataclass
class AlignedDataset:
    """Features, metadata and (optionally) labels aligned by row/sample key."""
    features: pd.DataFrame
    metadata: pd.DataFrame
    labels: Optional[pd.DataFrame]
    sample_key: pd.Series
    key_columns: List[str]
    issues: List[str] = field(default_factory=list)
    unmatched_labels: List[str] = field(default_factory=list)
    unmatched_metadata: List[str] = field(default_factory=list)


def align_dataset(features: pd.DataFrame,
                  metadata: pd.DataFrame,
                  labels: Optional[pd.DataFrame] = None,
                  key_columns: Optional[List[str]] = None) -> AlignedDataset:
    """Align features, metadata and labels on a stable sample key.

    * ``features`` and ``metadata`` must have the same number of rows and are
      assumed to be in the same order (as Milestone 1 exports them). This is
      verified; a mismatch raises ValueError.
    * ``labels`` are joined onto the feature windows by key. Label rows that do
      not correspond to a materialized window are reported in
      ``unmatched_labels`` (they are expected whenever empty windows are not
      materialized). Windows without a label keep ``is_anomaly = <NA>``.
    """
    key_columns = key_columns or DEFAULT_KEY_COLUMNS
    issues: List[str] = []

    if len(features) != len(metadata):
        raise ValueError(
            f"features and metadata row counts differ ({len(features)} vs {len(metadata)}); "
            "cannot align by position"
        )

    meta = metadata.reset_index(drop=True).copy()
    feat = features.reset_index(drop=True).copy()
    meta_key = make_sample_key(meta, key_columns)
    meta = meta.assign(sample_key=meta_key)

    aligned_labels = None
    unmatched_labels: List[str] = []
    if labels is not None and len(labels) > 0:
        lab = labels.reset_index(drop=True).copy()
        lab_key = make_sample_key(lab, key_columns)
        lab = lab.assign(sample_key=lab_key)
        # Keep only the first label per key; duplicates are an issue to report.
        dup_counts = lab["sample_key"].duplicated().sum()
        if dup_counts:
            issues.append(f"{dup_counts} duplicate label row(s) for the same sample key")
        lab_first = lab.drop_duplicates(subset="sample_key", keep="first")
        label_cols = [c for c in lab_first.columns if c not in key_columns and c != "sample_key"]
        merged = meta[["sample_key"]].merge(
            lab_first[["sample_key"] + label_cols], on="sample_key", how="left"
        )
        aligned_labels = merged
        meta_keys = set(meta["sample_key"])
        unmatched_labels = [k for k in lab_first["sample_key"] if k not in meta_keys]
        if unmatched_labels:
            issues.append(
                f"{len(unmatched_labels)} label row(s) have no matching feature window "
                "(expected when empty windows are not materialized)"
            )

    return AlignedDataset(
        features=feat,
        metadata=meta,
        labels=aligned_labels,
        sample_key=meta["sample_key"],
        key_columns=key_columns,
        issues=issues,
        unmatched_labels=unmatched_labels,
    )
