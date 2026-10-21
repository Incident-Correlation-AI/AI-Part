"""Configuration for the Milestone 3 Isolation Forest baseline.

The baseline is fully configuration-driven: model hyper-parameters, the explicit
feature allowlist, the missing-value policy, the training-data policy, the
thresholding method and all output paths live in ``configs/baseline.yaml`` and
are loaded into validated dataclasses here.

Invalid configuration fails loudly at load time rather than silently degrading
the experiment.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import os

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a declared dependency
    yaml = None


#: Column names that must never appear in the model-input allowlist.
FORBIDDEN_FEATURE_NAMES = {
    "is_anomaly",
    "label",
    "labels",
    "anomaly",
    "is_anomalous",
    "ground_truth",
    "true_label",
    "attack_type",
    "scenario",
    "scenario_name",
    "scenario_id",
    "intensity",
    "description",
    "host_id",
    "service",
    "event_id",
    "sample_key",
    "window_id",
    "window_start",
    "window_end",
    "window_size_minutes",
    "timestamp",
    "split",
    "split_name",
    "filename",
    "file_name",
}

VALID_MISSING_POLICIES = {"median_impute", "drop", "error"}
VALID_TRAINING_POLICIES = {"normal_only", "all"}
VALID_THRESHOLD_METHODS = {
    "validation_f1",
    "validation_precision",
    "validation_recall",
    "train_quantile",
}

#: The Milestone 2 eligible-feature allowlist, used as the documented default.
#: It is still an explicit list — a column never reaches the estimator just
#: because it happens to be numeric.
DEFAULT_FEATURE_COLUMNS = [
    "auth_failure_count",
    "error_count",
    "error_eligible_count",
    "error_ratio",
    "event_count",
    "http_5xx_count",
    "http_5xx_ratio",
    "http_request_count",
    "http_request_eligible_count",
    "latency_sample_count",
    "mean_latency_ms",
    "p95_latency_ms",
    "restart_count",
]


@dataclass
class ModelConfig:
    n_estimators: int = 100
    max_samples: int = 256
    contamination: Any = "auto"
    random_state: int = 42
    n_jobs: int = 2
    bootstrap: bool = False


@dataclass
class DataConfig:
    feature_columns: List[str] = field(default_factory=lambda: list(DEFAULT_FEATURE_COLUMNS))
    missing_value_policy: str = "median_impute"
    training_data_policy: str = "normal_only"
    label_column: str = "is_anomaly"
    sample_key_column: str = "sample_key"
    metadata_columns: List[str] = field(default_factory=list)


@dataclass
class ThresholdConfig:
    method: str = "validation_f1"
    min_precision: float = 0.5
    min_recall: float = 0.5
    fallback_method: str = "train_quantile"
    fallback_quantile: float = 0.01


@dataclass
class SplitConfig:
    strategy: str = "random"
    train: str = "train"
    validation: str = "validation"
    test: str = "test"


@dataclass
class BaselineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    paths: Dict[str, str] = field(default_factory=lambda: {
        "splits_dir": "data/processed/splits",
        "output_dir": "artifacts/baseline",
        "model_dir": "artifacts/baseline/model",
        "reports_dir": "artifacts/baseline/reports",
    })
    source_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def split_path(self, name: str) -> str:
        """Directory holding one split's tables."""
        return os.path.join(self.paths["splits_dir"], self.split.strategy, name)


def _validate(cfg: BaselineConfig) -> None:
    model = cfg.model
    if model.n_estimators < 1:
        raise ValueError("model.n_estimators must be >= 1")
    if model.max_samples < 1:
        raise ValueError("model.max_samples must be >= 1")
    if not (model.n_jobs == -1 or model.n_jobs >= 1):
        raise ValueError("model.n_jobs must be -1 or >= 1")
    contamination = model.contamination
    if isinstance(contamination, str):
        if contamination != "auto":
            raise ValueError(
                f"model.contamination must be 'auto' or a float in (0, 0.5], got {contamination!r}"
            )
    else:
        contamination = float(contamination)
        if not (0.0 < contamination <= 0.5):
            raise ValueError(f"model.contamination must be in (0, 0.5], got {contamination}")
        model.contamination = contamination

    data = cfg.data
    if not data.feature_columns:
        raise ValueError("data.feature_columns must be a non-empty explicit allowlist")
    if len(set(data.feature_columns)) != len(data.feature_columns):
        raise ValueError("data.feature_columns contains duplicates")
    for col in data.feature_columns:
        if not isinstance(col, str) or not col.strip():
            raise ValueError(f"data.feature_columns contains an invalid name: {col!r}")
        if col.lower() in FORBIDDEN_FEATURE_NAMES:
            raise ValueError(
                f"data.feature_columns may not contain {col!r}: labels, identifiers, "
                "timestamps and scenario metadata are not model inputs"
            )
    if data.missing_value_policy not in VALID_MISSING_POLICIES:
        raise ValueError(
            f"data.missing_value_policy={data.missing_value_policy!r} must be one of "
            f"{sorted(VALID_MISSING_POLICIES)}"
        )
    if data.training_data_policy not in VALID_TRAINING_POLICIES:
        raise ValueError(
            f"data.training_data_policy={data.training_data_policy!r} must be one of "
            f"{sorted(VALID_TRAINING_POLICIES)}"
        )

    threshold = cfg.threshold
    if threshold.method not in VALID_THRESHOLD_METHODS:
        raise ValueError(
            f"threshold.method={threshold.method!r} must be one of {sorted(VALID_THRESHOLD_METHODS)}"
        )
    if threshold.fallback_method not in VALID_THRESHOLD_METHODS:
        raise ValueError(
            f"threshold.fallback_method={threshold.fallback_method!r} must be one of "
            f"{sorted(VALID_THRESHOLD_METHODS)}"
        )
    if not (0.0 < threshold.fallback_quantile < 1.0):
        raise ValueError("threshold.fallback_quantile must be in (0, 1)")
    if not (0.0 <= threshold.min_precision <= 1.0):
        raise ValueError("threshold.min_precision must be in [0, 1]")
    if not (0.0 <= threshold.min_recall <= 1.0):
        raise ValueError("threshold.min_recall must be in [0, 1]")


def load_baseline_config(path: Optional[str] = None) -> BaselineConfig:
    """Load and validate the baseline configuration.

    A missing file yields documented defaults. Invalid values raise ValueError.
    """
    cfg = BaselineConfig()
    if path and os.path.isfile(path):
        if yaml is None:
            raise RuntimeError("PyYAML is required to load a config file")
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg.source_path = path

        m = raw.get("model", {}) or {}
        cfg.model = ModelConfig(
            n_estimators=int(m.get("n_estimators", 100)),
            max_samples=int(m.get("max_samples", 256)),
            contamination=m.get("contamination", "auto"),
            random_state=int(m.get("random_state", 42)),
            n_jobs=int(m.get("n_jobs", 2)),
            bootstrap=bool(m.get("bootstrap", False)),
        )

        d = raw.get("data", {}) or {}
        if "feature_columns" in d:
            feature_columns = [str(c) for c in (d.get("feature_columns") or [])]
        else:
            feature_columns = list(DEFAULT_FEATURE_COLUMNS)
        cfg.data = DataConfig(
            feature_columns=feature_columns,
            missing_value_policy=d.get("missing_value_policy", "median_impute"),
            training_data_policy=d.get("training_data_policy", "normal_only"),
            label_column=d.get("label_column", "is_anomaly"),
            sample_key_column=d.get("sample_key_column", "sample_key"),
            metadata_columns=[str(c) for c in (d.get("metadata_columns", []) or [])],
        )

        t = raw.get("threshold", {}) or {}
        cfg.threshold = ThresholdConfig(
            method=t.get("method", "validation_f1"),
            min_precision=float(t.get("min_precision", 0.5)),
            min_recall=float(t.get("min_recall", 0.5)),
            fallback_method=t.get("fallback_method", "train_quantile"),
            fallback_quantile=float(t.get("fallback_quantile", 0.01)),
        )

        s = raw.get("split", {}) or {}
        cfg.split = SplitConfig(
            strategy=s.get("strategy", "random"),
            train=s.get("train", "train"),
            validation=s.get("validation", "validation"),
            test=s.get("test", "test"),
        )

        p = raw.get("paths", {}) or {}
        for key, value in p.items():
            if key in cfg.paths:
                cfg.paths[key] = str(value)

    _validate(cfg)
    return cfg
