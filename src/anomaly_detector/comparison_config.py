"""Configuration loader for Milestone 4 model comparison.

Predeclares selection criteria and validates candidates/configuration.
Fails loudly on invalid settings; never auto-tunes to maximize test metrics.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import os

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class CandidateConfig:
    name: str
    type: str  # isolation_forest | mad | lof
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    variants: List[Dict[str, Any]] = field(default_factory=list)
    note: str = ""


@dataclass
class ComparisonConfig:
    experiment: Dict[str, Any] = field(default_factory=dict)
    selection_criteria: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    candidates: Dict[str, CandidateConfig] = field(default_factory=dict)
    threshold: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    runtime_limits: Dict[str, Any] = field(default_factory=dict)
    seeds: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    source_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def load_comparison_config(path: Optional[str] = None) -> ComparisonConfig:
    cfg = ComparisonConfig()
    if path and os.path.isfile(path):
        if yaml is None:
            raise RuntimeError("PyYAML required")
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg.source_path = path
        cfg.experiment = raw.get("experiment", {}) or {}
        cfg.selection_criteria = raw.get("selection_criteria", {}) or {}
        cfg.data = raw.get("data", {}) or {}
        cfg.threshold = raw.get("threshold", {}) or {}
        cfg.evaluation = raw.get("evaluation", {}) or {}
        cfg.runtime_limits = raw.get("runtime_limits", {}) or {}
        cfg.seeds = raw.get("seeds", {}) or {}
        cfg.outputs = raw.get("outputs", {}) or {}
        cand = raw.get("candidates", {}) or {}
        for key, val in cand.items():
            if not isinstance(val, dict):
                continue
            cfg.candidates[key] = CandidateConfig(
                name=val.get("name", key),
                type=val.get("type", key),
                enabled=bool(val.get("enabled", True)),
                config=val.get("config", {}) or {},
                variants=val.get("variants", []) or [],
                note=val.get("note", ""),
            )
    _validate(cfg)
    return cfg


def _validate(cfg: ComparisonConfig):
    if not cfg.data.get("feature_columns"):
        raise ValueError("comparison config requires data.feature_columns (explicit allowlist)")
    # forbid leakage-ish names
    forbidden = {"is_anomaly", "scenario_name", "scenario", "host_id", "service", "sample_key",
                "window_start", "window_end", "split", "description", "intensity"}
    for c in cfg.data.get("feature_columns", []):
        if str(c).lower() in forbidden:
            raise ValueError(f"feature allowlist contains forbidden/metadata column: {c}")
    if cfg.threshold.get("method") not in {"validation_f1", "validation_precision", "validation_recall", "train_quantile"}:
        raise ValueError("invalid threshold method")
    if cfg.data.get("missing_value_policy") not in {"median_impute", "drop", "error"}:
        raise ValueError("invalid missing_value_policy")
    if cfg.data.get("training_data_policy") not in {"normal_only", "all"}:
        raise ValueError("invalid training_data_policy")
    # must not allow test-based selection by accident
    if cfg.evaluation.get("use_test_only_after_selection") is False:
        raise ValueError("evaluation.use_test_only_after_selection must be true")
