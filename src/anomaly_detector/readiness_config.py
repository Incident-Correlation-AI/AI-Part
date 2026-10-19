"""Configuration for Milestone 2 data-readiness tooling.

Loads ``configs/data_readiness.yaml`` into validated dataclasses. Configuration
is explicit: unknown roles, invalid ratios and impossible split settings raise a
clear error rather than being silently ignored.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import os

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a declared dependency
    yaml = None


VALID_ROLES = {"feature", "label", "identifier", "timestamp", "scenario", "future_derived"}
VALID_UNKNOWN_POLICIES = {"error", "warn", "exclude"}
VALID_SPLIT_STRATEGIES = {"random", "time", "group"}


@dataclass
class RoleRule:
    role: str
    match: List[str]


@dataclass
class FeatureRoleConfig:
    explicit: Dict[str, str] = field(default_factory=dict)
    patterns: List[RoleRule] = field(default_factory=list)
    eligible_features: List[str] = field(default_factory=list)
    unknown_role_policy: str = "error"


@dataclass
class DataPolicy:
    non_negative_columns: List[str] = field(default_factory=list)
    unit_interval_columns: List[str] = field(default_factory=list)
    integer_columns: List[str] = field(default_factory=list)
    constant_max_unique: int = 1
    near_constant_ratio: float = 0.99
    missing_value_action: str = "report"
    sample_key_columns: List[str] = field(default_factory=lambda: ["host_id", "service", "window_start"])
    group_columns: List[str] = field(default_factory=list)


@dataclass
class SplitConfig:
    strategy: str = "random"
    seed: int = 42
    ratios: Dict[str, float] = field(default_factory=lambda: {"train": 0.6, "validation": 0.2, "test": 0.2})
    group_column: str = "host_id"
    time_column: str = "window_start"
    label_column: str = "is_anomaly"


@dataclass
class ReadinessConfig:
    feature_roles: FeatureRoleConfig = field(default_factory=FeatureRoleConfig)
    data_policy: DataPolicy = field(default_factory=DataPolicy)
    split: SplitConfig = field(default_factory=SplitConfig)
    paths: Dict[str, str] = field(default_factory=lambda: {
        "reports_dir": "reports",
        "splits_dir": "data/processed/splits",
        "training_ready_dir": "data/processed/training_ready",
    })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate(cfg: ReadinessConfig) -> None:
    roles = cfg.feature_roles
    if roles.unknown_role_policy not in VALID_UNKNOWN_POLICIES:
        raise ValueError(
            f"feature_roles.unknown_role_policy={roles.unknown_role_policy!r} "
            f"must be one of {sorted(VALID_UNKNOWN_POLICIES)}"
        )
    for rule in roles.patterns:
        if rule.role not in VALID_ROLES:
            raise ValueError(f"Unknown role {rule.role!r}; valid roles: {sorted(VALID_ROLES)}")
    for col, role in roles.explicit.items():
        if role not in VALID_ROLES:
            raise ValueError(f"explicit role for {col!r} is {role!r}; valid roles: {sorted(VALID_ROLES)}")

    if cfg.split.strategy not in VALID_SPLIT_STRATEGIES:
        raise ValueError(
            f"split.strategy={cfg.split.strategy!r} must be one of {sorted(VALID_SPLIT_STRATEGIES)}"
        )
    ratios = cfg.split.ratios
    if any(v < 0 for v in ratios.values()):
        raise ValueError("split ratios must be non-negative")
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")


def load_readiness_config(path: Optional[str] = None) -> ReadinessConfig:
    """Load and validate the readiness configuration.

    A missing file yields documented defaults. Invalid values raise ValueError.
    """
    cfg = ReadinessConfig()
    if path and os.path.isfile(path):
        if yaml is None:
            raise RuntimeError("PyYAML is required to load a config file")
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        fr = raw.get("feature_roles", {}) or {}
        cfg.feature_roles = FeatureRoleConfig(
            explicit=dict(fr.get("explicit", {}) or {}),
            patterns=[RoleRule(r["role"], list(r.get("match", []))) for r in (fr.get("patterns", []) or [])],
            eligible_features=list(fr.get("eligible_features", []) or []),
            unknown_role_policy=fr.get("unknown_role_policy", "error"),
        )

        dp = raw.get("data_policy", {}) or {}
        cfg.data_policy = DataPolicy(
            non_negative_columns=list(dp.get("non_negative_columns", []) or []),
            unit_interval_columns=list(dp.get("unit_interval_columns", []) or []),
            integer_columns=list(dp.get("integer_columns", []) or []),
            constant_max_unique=int(dp.get("constant_max_unique", 1)),
            near_constant_ratio=float(dp.get("near_constant_ratio", 0.99)),
            missing_value_action=dp.get("missing_value_action", "report"),
            sample_key_columns=list(dp.get("sample_key_columns", ["host_id", "service", "window_start"])),
            group_columns=list(dp.get("group_columns", []) or []),
        )

        sp = raw.get("split", {}) or {}
        cfg.split = SplitConfig(
            strategy=sp.get("strategy", "random"),
            seed=int(sp.get("seed", 42)),
            ratios={k: float(v) for k, v in (sp.get("ratios", {}) or {}).items()} or {"train": 0.6, "validation": 0.2, "test": 0.2},
            group_column=sp.get("group_column", "host_id"),
            time_column=sp.get("time_column", "window_start"),
            label_column=sp.get("label_column", "is_anomaly"),
        )

        paths = raw.get("paths", {}) or {}
        cfg.paths.update({k: v for k, v in paths.items() if k in cfg.paths})

    _validate(cfg)
    return cfg
