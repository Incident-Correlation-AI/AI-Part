"""Dataset inventory and integrity audit (Milestone 2).

The audit is **read-only**: it never mutates the input tables. It profiles every
column, checks the configured policy (non-negative counts, unit-interval ratios,
integer counts, constant/near-constant features), detects duplicate rows and
duplicate sample keys, and summarises label distribution.

Results are emitted as both JSON and Markdown so a human can review readiness
before any model is trained. Nothing is silently coerced or dropped: every
problem becomes an :class:`AuditIssue` with a severity.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json
import math
import os

import numpy as np
import pandas as pd

from .dataset import AlignedDataset
from .readiness_config import ReadinessConfig
from .roles import FORBIDDEN_FEATURE_ROLES, classify_columns


ERROR = "error"
WARNING = "warning"
INFO = "info"


@dataclass
class AuditIssue:
    severity: str
    code: str
    message: str
    column: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetAudit:
    generated_at: str
    inventory: Dict[str, Any] = field(default_factory=dict)
    columns: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[AuditIssue] = field(default_factory=list)
    duplicates: Dict[str, Any] = field(default_factory=dict)
    label_summary: Dict[str, Any] = field(default_factory=dict)
    role_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == WARNING)

    @property
    def is_ready(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "inventory": self.inventory,
            "columns": self.columns,
            "duplicates": self.duplicates,
            "label_summary": self.label_summary,
            "role_summary": self.role_summary,
            "issues": [i.to_dict() for i in self.issues],
            "summary": {
                "error_count": self.error_count,
                "warning_count": self.warning_count,
                "is_ready": self.is_ready,
            },
        }

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append("# Dataset Audit Report")
        lines.append("")
        lines.append(f"- Generated: {self.generated_at}")
        inv = self.inventory
        lines.append(f"- Features rows: {inv.get('feature_rows')}  columns: {inv.get('feature_columns')}")
        lines.append(f"- Metadata rows: {inv.get('metadata_rows')}")
        lines.append(f"- Labels rows (aligned): {inv.get('label_rows')}")
        lines.append(f"- Errors: {self.error_count}  Warnings: {self.warning_count}")
        lines.append(f"- Verdict: {'READY' if self.is_ready else 'NOT READY'}")
        lines.append("")

        lines.append("## Column profile")
        lines.append("")
        lines.append("| Column | Role | dtype | Missing | Infinite | Unique | Constant | Min | Max |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for c in self.columns:
            lines.append(
                "| {column} | {role} | {dtype} | {missing} | {infinite} | {unique} | {constant} | {min} | {max} |".format(
                    column=c["column"],
                    role=c["role"],
                    dtype=c["dtype"],
                    missing=f"{c['missing_count']} ({c['missing_ratio']:.1%})",
                    infinite=c["infinite_count"] if c["infinite_count"] is not None else "-",
                    unique=c["unique_count"],
                    constant=("yes" if c["is_constant"] else ("near" if c["is_near_constant"] else "no")),
                    min=_fmt(c.get("min")),
                    max=_fmt(c.get("max")),
                )
            )
        lines.append("")

        if self.role_summary:
            lines.append("## Feature roles")
            lines.append("")
            for role, cols in sorted(self.role_summary.items()):
                lines.append(f"- **{role}**: {', '.join(cols) if cols else '(none)'}")
            lines.append("")

        lines.append("## Duplicates")
        lines.append("")
        lines.append(f"- Duplicate feature rows: {self.duplicates.get('duplicate_feature_rows', 0)}")
        lines.append(f"- Duplicate sample keys: {self.duplicates.get('duplicate_sample_keys', 0)}")
        lines.append("")

        if self.label_summary:
            lines.append("## Labels")
            lines.append("")
            for k, v in self.label_summary.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        lines.append("## Issues")
        lines.append("")
        if not self.issues:
            lines.append("No issues detected.")
        else:
            lines.append("| Severity | Code | Column | Message |")
            lines.append("| --- | --- | --- | --- |")
            for i in self.issues:
                lines.append(f"| {i.severity} | {i.code} | {i.column or '-'} | {i.message} |")
        lines.append("")
        return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.4g}"
    return str(value)


def _profile_columns(
    df: pd.DataFrame,
    roles: Dict[str, Any],
    policy_non_negative: set,
    policy_unit_interval: set,
    policy_integer: set,
    constant_max_unique: int,
    near_constant_ratio: float,
    issues: List[AuditIssue],
    prefix: str,
) -> List[Dict[str, Any]]:
    profiles: List[Dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        n = len(series)
        missing = int(series.isna().sum())
        missing_ratio = (missing / n) if n else 0.0

        numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)

        # Distinguish true infinities (a model cannot consume them) from missing
        # values (NaN, reported separately and never auto-imputed).
        infinite_count = 0
        if numeric:
            arr = pd.to_numeric(series, errors="coerce")
            finite = np.isfinite(arr.to_numpy(dtype="float64", na_value=np.nan))
            missing_mask = arr.isna().to_numpy()
            infinite_count = int((~finite & ~missing_mask).sum())

        unique_count = int(series.nunique(dropna=True))
        try:
            value_counts = series.value_counts(dropna=True)
            dominant_ratio = float(value_counts.iloc[0] / value_counts.sum()) if value_counts.sum() else 0.0
        except Exception:
            dominant_ratio = 0.0

        is_constant = unique_count <= constant_max_unique
        is_near_constant = (not is_constant) and dominant_ratio >= near_constant_ratio

        min_v = max_v = mean_v = std_v = None
        if numeric:
            num = pd.to_numeric(series, errors="coerce")
            finite = num[np.isfinite(num.to_numpy(dtype="float64", na_value=np.nan))]
            if len(finite):
                min_v = float(finite.min())
                max_v = float(finite.max())
                mean_v = float(finite.mean())
                std_v = float(finite.std(ddof=0)) if len(finite) > 1 else 0.0

        role_info = roles.get(col)
        role = role_info.role if role_info else "unknown"

        profile = {
            "column": col,
            "table": prefix,
            "role": role,
            "dtype": str(series.dtype),
            "missing_count": missing,
            "missing_ratio": missing_ratio,
            "infinite_count": infinite_count if numeric else None,
            "unique_count": unique_count,
            "dominant_value_ratio": dominant_ratio,
            "is_constant": bool(is_constant),
            "is_near_constant": bool(is_near_constant),
            "min": min_v,
            "max": max_v,
            "mean": mean_v,
            "std": std_v,
        }
        profiles.append(profile)

        # Policy checks only apply to the model feature table.
        if prefix != "features":
            continue

        if missing > 0:
            issues.append(AuditIssue(
                WARNING, "missing_values",
                f"{col!r} has {missing} missing value(s) ({missing_ratio:.1%}); policy is report-only",
                column=col,
            ))
        if numeric and infinite_count > 0:
            issues.append(AuditIssue(
                ERROR, "infinite_values",
                f"{col!r} contains {infinite_count} infinite value(s) (+/-inf) that a model cannot consume",
                column=col,
            ))
        if col in policy_non_negative and min_v is not None and min_v < 0:
            issues.append(AuditIssue(
                ERROR, "negative_value",
                f"{col!r} is declared non-negative but has min {min_v}",
                column=col,
            ))
        if col in policy_unit_interval and (min_v is not None and (min_v < -1e-9 or max_v > 1 + 1e-9)):
            issues.append(AuditIssue(
                ERROR, "out_of_unit_interval",
                f"{col!r} is declared a ratio in [0,1] but has range [{min_v}, {max_v}]",
                column=col,
            ))
        if col in policy_integer and numeric:
            num = pd.to_numeric(series, errors="coerce").dropna()
            if len(num) and not np.all(np.isclose(num % 1, 0.0)):
                issues.append(AuditIssue(
                    WARNING, "non_integer_count",
                    f"{col!r} is declared an integer count but has non-integer values",
                    column=col,
                ))
        if is_constant:
            issues.append(AuditIssue(
                WARNING, "constant_feature",
                f"{col!r} is constant ({unique_count} distinct value(s)); it carries no signal",
                column=col,
            ))
        elif is_near_constant:
            issues.append(AuditIssue(
                INFO, "near_constant_feature",
                f"{col!r} is near-constant (dominant value {dominant_ratio:.1%})",
                column=col,
            ))
    return profiles


def audit_dataset(
    aligned: AlignedDataset,
    config: ReadinessConfig,
    source_paths: Optional[Dict[str, str]] = None,
) -> DatasetAudit:
    """Audit an aligned dataset against the configured policy (read-only)."""
    # Work on copies so the caller's frames are never mutated.
    features = aligned.features.copy()
    metadata = aligned.metadata.copy()
    labels = aligned.labels.copy() if aligned.labels is not None else None

    role_cfg = config.feature_roles
    policy = config.data_policy
    role_report = classify_columns(list(features.columns), role_cfg)

    issues: List[AuditIssue] = []

    # Fail-fast role problems first.
    for violation in role_report.violations:
        issues.append(AuditIssue(
            ERROR, "feature_role_violation", violation["message"], column=violation["column"]
        ))
    # Defense in depth: any forbidden-role column present in the features is an
    # error, even if it was not explicitly allow-listed.
    already_flagged = {v["column"] for v in role_report.violations}
    for col, info in role_report.roles.items():
        forbidden = [r for r in info.all_roles if r in FORBIDDEN_FEATURE_ROLES]
        if forbidden and col not in already_flagged:
            issues.append(AuditIssue(
                ERROR, "forbidden_column_in_features",
                f"{col!r} carries forbidden role(s) {forbidden} but is present in the feature table",
                column=col,
            ))
    for col in role_report.unknown:
        sev = ERROR if role_cfg.unknown_role_policy == "error" else WARNING
        issues.append(AuditIssue(
            sev, "unknown_role",
            f"column {col!r} has no determinable role; decide explicitly in feature_roles",
            column=col,
        ))

    profiles = _profile_columns(
        features,
        role_report.roles,
        set(policy.non_negative_columns),
        set(policy.unit_interval_columns),
        set(policy.integer_columns),
        policy.constant_max_unique,
        policy.near_constant_ratio,
        issues,
        prefix="features",
    )

    # Duplicate feature rows.
    dup_feature_rows = int(features.duplicated().sum())
    if dup_feature_rows:
        issues.append(AuditIssue(
            WARNING, "duplicate_feature_rows",
            f"{dup_feature_rows} duplicate feature row(s) detected",
        ))

    # Duplicate sample keys (from metadata).
    dup_keys = int(aligned.sample_key.duplicated().sum()) if len(aligned.sample_key) else 0
    if dup_keys:
        issues.append(AuditIssue(
            ERROR, "duplicate_sample_keys",
            f"{dup_keys} duplicate sample key(s); row alignment is ambiguous",
        ))

    # Labels: alignment and distribution.
    label_summary: Dict[str, Any] = {}
    if labels is not None:
        label_col = config.split.label_column
        missing_labels = int(labels[label_col].isna().sum()) if label_col in labels.columns else 0
        if missing_labels:
            issues.append(AuditIssue(
                WARNING, "missing_labels",
                f"{missing_labels} window(s) have no ground-truth label (`{label_col}` = NA)",
            ))
        if label_col in labels.columns:
            positive = int((labels[label_col] == True).sum())  # noqa: E712
            negative = int((labels[label_col] == False).sum())  # noqa: E712
            total_labelled = positive + negative
            label_summary = {
                "label_column": label_col,
                "labelled_rows": total_labelled,
                "positive": positive,
                "negative": negative,
                "positive_ratio": (positive / total_labelled) if total_labelled else 0.0,
                "missing": missing_labels,
            }
            if total_labelled and positive == 0:
                issues.append(AuditIssue(
                    ERROR, "no_positive_labels",
                    "there are no positive (anomaly) labels; supervised evaluation is impossible",
                ))
        if aligned.unmatched_labels:
            issues.append(AuditIssue(
                INFO, "unmatched_label_rows",
                f"{len(aligned.unmatched_labels)} label row(s) reference windows that were not "
                "materialized (expected when empty windows are skipped)",
            ))

    role_summary: Dict[str, List[str]] = {}
    for col, info in role_report.roles.items():
        role_summary.setdefault(info.role, []).append(col)
    for role in role_summary:
        role_summary[role].sort()

    inventory = {
        "feature_rows": int(len(features)),
        "feature_columns": int(features.shape[1]),
        "metadata_rows": int(len(metadata)),
        "label_rows": int(len(labels)) if labels is not None else 0,
        "eligible_feature_count": len(role_report.eligible_features),
        "source_paths": dict(source_paths or {}),
    }

    return DatasetAudit(
        generated_at=datetime.now(timezone.utc).isoformat(),
        inventory=inventory,
        columns=profiles,
        issues=issues,
        duplicates={
            "duplicate_feature_rows": dup_feature_rows,
            "duplicate_sample_keys": dup_keys,
        },
        label_summary=label_summary,
        role_summary=role_summary,
    )


def write_audit(audit: DatasetAudit, reports_dir: str, basename: str = "data_quality_report"):
    """Write the audit as JSON and Markdown. Returns the two paths."""
    os.makedirs(reports_dir, exist_ok=True)
    json_path = os.path.join(reports_dir, f"{basename}.json")
    md_path = os.path.join(reports_dir, f"{basename}.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(audit.to_dict(), fh, indent=2, sort_keys=False, default=_json_default)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(audit.to_markdown())
    return json_path, md_path


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
