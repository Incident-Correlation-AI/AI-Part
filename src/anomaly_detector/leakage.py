"""Leakage audit for the model input matrix (Milestone 2).

The audit answers one question: *is anything in the candidate feature table that
the model must not see?* Labels, identifiers, timestamps, scenario metadata and
future-derived fields are all leakage risks. They are detected by role, by
cross-table comparison with the label table, and by a small set of statistical
heuristics (high-cardinality numeric columns, features perfectly correlated with
the label).

The audit never silently drops a column: non-eligible columns are listed with a
reason, and violations are reported as errors so a caller can refuse to export.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json
import os

import numpy as np
import pandas as pd

from .dataset import AlignedDataset
from .readiness_config import ReadinessConfig
from .roles import FORBIDDEN_FEATURE_ROLES, classify_columns


@dataclass
class LeakageFinding:
    severity: str            # error | warning | info
    code: str
    message: str
    column: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LeakageReport:
    generated_at: str
    eligible_features: List[str] = field(default_factory=list)
    excluded_columns: List[Dict[str, str]] = field(default_factory=list)
    findings: List[LeakageFinding] = field(default_factory=list)
    correlations: Dict[str, float] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "error")

    @property
    def is_clean(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "eligible_features": self.eligible_features,
            "excluded_columns": self.excluded_columns,
            "findings": [f.to_dict() for f in self.findings],
            "correlations": self.correlations,
            "summary": {
                "error_count": self.error_count,
                "is_clean": self.is_clean,
                "eligible_feature_count": len(self.eligible_features),
            },
        }

    def to_markdown(self) -> str:
        lines = ["# Leakage Audit Report", ""]
        lines.append(f"- Generated: {self.generated_at}")
        lines.append(f"- Eligible features ({len(self.eligible_features)}): {', '.join(self.eligible_features) or '(none)'}")
        lines.append(f"- Errors: {self.error_count}")
        lines.append(f"- Verdict: {'CLEAN' if self.is_clean else 'LEAKAGE DETECTED'}")
        lines.append("")

        lines.append("## Excluded columns")
        lines.append("")
        if not self.excluded_columns:
            lines.append("None — every column in the feature table is eligible.")
        else:
            lines.append("| Column | Role | Reason |")
            lines.append("| --- | --- | --- |")
            for e in self.excluded_columns:
                lines.append(f"| {e['column']} | {e['role']} | {e['reason']} |")
        lines.append("")

        lines.append("## Findings")
        lines.append("")
        if not self.findings:
            lines.append("No leakage findings.")
        else:
            lines.append("| Severity | Code | Column | Message |")
            lines.append("| --- | --- | --- | --- |")
            for f in self.findings:
                lines.append(f"| {f.severity} | {f.code} | {f.column or '-'} | {f.message} |")
        lines.append("")

        if self.correlations:
            lines.append("## Label correlation (|r| > 0.99 flagged)")
            lines.append("")
            lines.append("| Feature | Correlation with label |")
            lines.append("| --- | --- |")
            for col, r in sorted(self.correlations.items(), key=lambda kv: abs(kv[1]), reverse=True):
                lines.append(f"| {col} | {r:.4f} |")
            lines.append("")
        return "\n".join(lines)


def audit_leakage(
    aligned: AlignedDataset,
    config: ReadinessConfig,
    high_cardinality_ratio: float = 0.95,
) -> LeakageReport:
    """Audit the candidate feature table for anything the model must not see."""
    features = aligned.features
    role_cfg = config.feature_roles
    report = LeakageReport(generated_at=datetime.now(timezone.utc).isoformat())

    role_report = classify_columns(list(features.columns), role_cfg)

    # Eligible features and excluded columns with reasons.
    for col, info in role_report.roles.items():
        if info.eligible:
            report.eligible_features.append(col)
        else:
            report.excluded_columns.append({
                "column": col,
                "role": info.role,
                "reason": info.reason or "not eligible",
            })
    report.eligible_features.sort()
    report.excluded_columns.sort(key=lambda e: e["column"])

    # 1) Any forbidden-role column present in the candidate features is leakage.
    for col, info in role_report.roles.items():
        forbidden = [r for r in info.all_roles if r in FORBIDDEN_FEATURE_ROLES]
        if forbidden:
            report.findings.append(LeakageFinding(
                "error", "forbidden_column_in_features",
                f"{col!r} carries forbidden role(s) {forbidden} but is present in the feature table",
                column=col,
            ))

    # 2) An allow-listed feature that also matches a forbidden role.
    for v in role_report.violations:
        report.findings.append(LeakageFinding(
            "error", "allowlisted_but_forbidden",
            v["message"],
            column=v["column"],
        ))

    # 3) Cross-table leak: does any label column name appear in the features?
    if aligned.labels is not None:
        label_cols = set(aligned.labels.columns) - {"sample_key"}
        overlap = label_cols & set(features.columns)
        for col in sorted(overlap):
            report.findings.append(LeakageFinding(
                "error", "label_column_in_features",
                f"label column {col!r} is present in the feature table",
                column=col,
            ))

    # 4) High-cardinality numeric heuristic: possible identifier disguised as a
    #    numeric feature (e.g. raw host index or monotonic id).
    n = len(features)
    if n > 1:
        for col in features.columns:
            if not pd.api.types.is_numeric_dtype(features[col]):
                continue
            if not role_report.roles[col].eligible:
                continue
            values = pd.to_numeric(features[col], errors="coerce").dropna()
            if len(values) < n:
                continue
            # Only integer-valued columns are identifier-shaped. Continuous
            # measurements (latency, ratios) are legitimately high-cardinality.
            if not np.all(np.isclose(values.to_numpy(dtype="float64") % 1, 0.0)):
                continue
            unique_ratio = values.nunique() / n
            if unique_ratio >= high_cardinality_ratio:
                report.findings.append(LeakageFinding(
                    "warning", "high_cardinality_feature",
                    f"{col!r} is integer-valued with {unique_ratio:.0%} distinct values; verify it is not an identifier",
                    column=col,
                ))

    # 5) Perfect correlation with the label: numeric eligible feature whose
    #    absolute Pearson correlation with the aligned label is ~1.
    label_col = config.split.label_column
    if aligned.labels is not None and label_col in aligned.labels.columns:
        y = aligned.labels[label_col]
        y_bool = y.astype("boolean")
        if y_bool.notna().all():
            y_num = y_bool.astype(float).to_numpy()
            for col in report.eligible_features:
                x = pd.to_numeric(features[col], errors="coerce")
                if len(x) != len(y_num):
                    continue
                valid = np.isfinite(x.to_numpy(dtype="float64", na_value=np.nan))
                if valid.sum() < 3:
                    continue
                xs = x.to_numpy(dtype="float64")[valid]
                ys = y_num[valid]
                if np.std(xs) == 0 or np.std(ys) == 0:
                    continue
                r = float(np.corrcoef(xs, ys)[0, 1])
                report.correlations[col] = r
                if abs(r) > 0.999:
                    report.findings.append(LeakageFinding(
                        "warning", "perfect_label_correlation",
                        f"{col!r} correlates {r:.4f} with the label; possible target leakage",
                        column=col,
                    ))

    return report


def assert_no_leakage(report: LeakageReport) -> None:
    """Raise ValueError if the leakage audit found any error-severity finding."""
    if not report.is_clean:
        details = "; ".join(f"{f.code}:{f.column}" for f in report.findings if f.severity == "error")
        raise ValueError(f"Leakage audit failed with {report.error_count} error(s): {details}")


def write_leakage_report(report: LeakageReport, reports_dir: str, basename: str = "leakage_audit"):
    """Write the leakage audit as JSON and Markdown. Returns the two paths."""
    os.makedirs(reports_dir, exist_ok=True)
    json_path = os.path.join(reports_dir, f"{basename}.json")
    md_path = os.path.join(reports_dir, f"{basename}.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(report.to_markdown())
    return json_path, md_path
