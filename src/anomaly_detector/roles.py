"""Feature-role classification and leakage-risk detection.

Every column in a candidate feature table is assigned a *role*:

* ``feature``        — a model-eligible numeric predictor
* ``label``          — ground truth (never a model input)
* ``identifier``     — host/service/event/sample identifiers
* ``timestamp``      — timestamps and window boundaries
* ``scenario``       — scenario/generation metadata
* ``future_derived`` — fields computed using future observations

Roles come from the validated :class:`~anomaly_detector.readiness_config.FeatureRoleConfig`.
Model eligibility is decided by an **explicit allowlist** (``eligible_features``),
never by "is it numeric". A column that is allow-listed *and* also matches a
non-feature role is a **leakage violation** and is reported, not silently dropped.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .readiness_config import FeatureRoleConfig


# Non-feature roles that must never appear in the model feature matrix.
FORBIDDEN_FEATURE_ROLES = {"label", "identifier", "timestamp", "scenario", "future_derived"}


@dataclass
class ColumnRole:
    """The assigned role(s) for a single column."""
    column: str
    role: str                       # primary role
    all_roles: List[str] = field(default_factory=list)
    eligible: bool = False
    reason: str = ""


@dataclass
class RoleReport:
    """Result of classifying a set of candidate feature columns."""
    roles: Dict[str, ColumnRole] = field(default_factory=dict)
    eligible_features: List[str] = field(default_factory=list)
    violations: List[Dict[str, str]] = field(default_factory=list)   # leakage risks in features
    ambiguous: List[Dict[str, object]] = field(default_factory=list)  # multi-role columns
    unknown: List[str] = field(default_factory=list)                 # undetermined role


def classify_columns(columns: List[str], config: FeatureRoleConfig) -> RoleReport:
    """Classify candidate feature columns into roles.

    Parameters
    ----------
    columns:
        Candidate column names (e.g. the columns of a features table).
    config:
        Role rules and the explicit eligibility allowlist.
    """
    report = RoleReport()
    allowlist = set(config.eligible_features)

    for col in columns:
        matched_roles: List[str] = []

        # 1) Explicit override wins.
        explicit_role = config.explicit.get(col)

        # 2) Ordered substring patterns; record every role that matches.
        for rule in config.patterns:
            for token in rule.match:
                if token and token.lower() in col.lower():
                    if rule.role not in matched_roles:
                        matched_roles.append(rule.role)
                    break

        if explicit_role is not None:
            primary = explicit_role
        elif matched_roles:
            primary = matched_roles[0]
        else:
            primary = "unknown"

        all_roles = list(dict.fromkeys(([explicit_role] if explicit_role else []) + matched_roles))

        # Eligibility is allowlist-driven, plus a safety net: never eligible if
        # the column carries a forbidden role.
        in_allowlist = col in allowlist
        forbidden = [r for r in all_roles if r in FORBIDDEN_FEATURE_ROLES]
        eligible = in_allowlist and not forbidden

        reason_bits = []
        if in_allowlist:
            reason_bits.append("in allowlist")
        if forbidden:
            reason_bits.append(f"forbidden role(s): {','.join(forbidden)}")
        if not in_allowlist and primary == "feature":
            reason_bits.append("looks numeric but not allow-listed")
        if primary == "unknown":
            reason_bits.append("role undetermined")

        role = ColumnRole(
            column=col,
            role=primary,
            all_roles=all_roles,
            eligible=eligible,
            reason="; ".join(reason_bits) or "not eligible",
        )
        report.roles[col] = role

        if eligible:
            report.eligible_features.append(col)

        # Leakage violations: allow-listed but also carrying a forbidden role.
        if in_allowlist and forbidden:
            report.violations.append({
                "column": col,
                "primary_role": primary,
                "forbidden_roles": ",".join(forbidden),
                "message": f"{col!r} is allow-listed but also matches {forbidden}",
            })

        # Ambiguity: more than one distinct role matched.
        if len(all_roles) > 1:
            report.ambiguous.append({"column": col, "roles": all_roles})

        if primary == "unknown":
            report.unknown.append(col)

    report.eligible_features.sort()
    return report


def detect_leaked_columns(columns: List[str], config: FeatureRoleConfig) -> List[str]:
    """Return columns that carry a forbidden role (leakage candidates)."""
    report = classify_columns(columns, config)
    leaked = []
    for col, role in report.roles.items():
        if any(r in FORBIDDEN_FEATURE_ROLES for r in role.all_roles):
            leaked.append(col)
    return leaked
