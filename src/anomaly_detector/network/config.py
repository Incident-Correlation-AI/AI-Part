"""Configuration for the network/service anomaly-detection extension.

Everything that influences a result lives in ``configs/network.yaml`` (or an
explicit :class:`NetworkConfig`), including simulator bounds, rolling-window
settings, transparent-baseline thresholds, model hyper-parameters, split
policy, artifact paths and the LLM privacy boundary.  Invalid configuration
fails loudly at load time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is a declared dependency
    yaml = None

from ..baseline_config import (
    BaselineConfig,
    DataConfig,
    ModelConfig,
    SplitConfig,
    ThresholdConfig,
)
from ..baseline_config import _validate as _validate_baseline
from ..inference.contract import CONTRACT_VERSION
from .features import NETWORK_FEATURE_COLUMNS, NETWORK_FEATURE_SCHEMA_VERSION

VALID_POLICIES = {"median_impute", "drop", "error"}


@dataclass
class SimulatorSettings:
    seed: int = 42
    run_count: int = 10
    run_minutes: int = 90
    window_size_minutes: int = 1
    base_http_per_minute: int = 24
    base_auth_per_minute: int = 10
    base_db_per_minute: int = 24
    inject_malformed: bool = True
    max_events: int = 250_000
    start_time: str = "2026-01-01T00:00:00+00:00"

    def to_simulator_config(self):
        """Build the simulator's own config object (kept import-cycle free)."""
        from .simulator import SimulatorConfig

        return SimulatorConfig(
            seed=self.seed,
            run_count=self.run_count,
            run_minutes=self.run_minutes,
            window_size_minutes=self.window_size_minutes,
            base_http_per_minute=self.base_http_per_minute,
            base_auth_per_minute=self.base_auth_per_minute,
            base_db_per_minute=self.base_db_per_minute,
            inject_malformed=self.inject_malformed,
            max_events=self.max_events,
            start_time=self.start_time,
        )


@dataclass
class RuleConfig:
    """Transparent-baseline thresholds.

    Units are explicit in the names.  All rules have a minimum-volume guard so
    a single failure on a nearly idle service cannot become a severe anomaly.
    """

    rolling_windows: int = 30
    rolling_min_history: int = 5
    min_window_events: int = 5

    http_5xx_min_requests: int = 20
    http_5xx_min_count: int = 3
    http_5xx_ratio_threshold: float = 0.05
    http_5xx_strong_ratio: float = 0.20
    http_5xx_delta_threshold: float = 0.05

    db_min_events: int = 8
    db_timeout_ratio_threshold: float = 0.05
    db_timeout_strong_ratio: float = 0.20
    db_connection_failure_ratio_threshold: float = 0.02
    db_connection_failure_strong_ratio: float = 0.10

    auth_min_events: int = 10
    auth_min_failures: int = 8
    auth_failure_ratio_threshold: float = 0.50
    auth_failure_strong_ratio: float = 0.80
    auth_distinct_source_threshold: int = 8

    latency_min_samples: int = 20
    latency_p95_ratio_threshold: float = 2.0
    latency_p95_strong_ratio: float = 4.0
    latency_p95_absolute_floor_ms: float = 50.0

    max_routine_restarts: int = 1
    weak_weight: float = 0.5
    strong_weight: float = 1.0
    decision_threshold: float = 1.0
    high_severity_threshold: float = 2.0
    isolation_forest_high_delta: float = 0.10
    mad_high_zscore: float = 4.0


@dataclass
class LlmSettings:
    enabled: bool = False
    provider: str = "mock"           # mock | http | none
    endpoint: str = ""
    model: str = ""
    api_key_env: str = "LAD_LLM_API_KEY"
    timeout_seconds: float = 20.0

    def to_dict(self) -> Dict[str, Any]:
        # The API-key env *name* is configuration; its value is never serialised.
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "endpoint": self.endpoint,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class NetworkConfig:
    simulator: SimulatorSettings = field(default_factory=SimulatorSettings)
    rules: RuleConfig = field(default_factory=RuleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    split_strategies: List[str] = field(default_factory=lambda: ["time", "group"])
    split_ratios: Dict[str, float] = field(default_factory=lambda: {
        "train": 0.6, "validation": 0.2, "test": 0.2,
    })
    paths: Dict[str, str] = field(default_factory=lambda: {
        "synthetic_dir": "data/synthetic/network",
        "splits_dir": "data/processed/network_splits",
        "output_dir": "artifacts/network",
        "model_dir": "artifacts/network/model",
        "reports_dir": "artifacts/network/reports",
        "existing_model": "artifacts/baseline/model/model.joblib",
        "existing_manifest": "artifacts/baseline/model/manifest.json",
    })
    llm: LlmSettings = field(default_factory=LlmSettings)
    source_path: str = ""

    @property
    def feature_schema_version(self) -> str:
        return NETWORK_FEATURE_SCHEMA_VERSION

    @property
    def feature_columns(self) -> List[str]:
        return list(NETWORK_FEATURE_COLUMNS)

    def data_config(self) -> DataConfig:
        return DataConfig(
            feature_columns=list(NETWORK_FEATURE_COLUMNS),
            missing_value_policy="median_impute",
            training_data_policy="normal_only",
            label_column="is_anomaly",
            sample_key_column="sample_key",
            metadata_columns=[
                "window_start", "window_end", "host_id", "service",
                "window_size_minutes",
            ],
        )

    def baseline_config(self) -> BaselineConfig:
        """Build the existing M3 config object so artifacts stay compatible."""
        cfg = BaselineConfig(
            model=self.model,
            data=self.data_config(),
            threshold=self.threshold,
            split=SplitConfig(strategy="time", train="train",
                              validation="validation", test="test"),
            paths={
                "splits_dir": self.paths["splits_dir"],
                "output_dir": self.paths["output_dir"],
                "model_dir": self.paths["model_dir"],
                "reports_dir": self.paths["reports_dir"],
            },
            source_path=self.source_path,
        )
        _validate_baseline(cfg)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_path": self.source_path,
            "contract_version": CONTRACT_VERSION,
            "feature_schema_version": NETWORK_FEATURE_SCHEMA_VERSION,
            "simulator": {
                "seed": self.simulator.seed,
                "run_count": self.simulator.run_count,
                "run_minutes": self.simulator.run_minutes,
                "window_size_minutes": self.simulator.window_size_minutes,
                "base_http_per_minute": self.simulator.base_http_per_minute,
                "base_auth_per_minute": self.simulator.base_auth_per_minute,
                "base_db_per_minute": self.simulator.base_db_per_minute,
                "inject_malformed": self.simulator.inject_malformed,
                "max_events": self.simulator.max_events,
                "start_time": self.simulator.start_time,
            },
            "rules": _asdict(self.rules),
            "model": _asdict(self.model),
            "threshold": _asdict(self.threshold),
            "split": {"strategies": list(self.split_strategies),
                      "ratios": dict(self.split_ratios)},
            "paths": dict(self.paths),
            "llm": self.llm.to_dict(),
        }


def _asdict(value: Any) -> Dict[str, Any]:
    from dataclasses import asdict

    return asdict(value)


def load_network_config(path: Optional[str] = None) -> NetworkConfig:
    cfg = NetworkConfig()
    if path and os.path.isfile(path):
        if yaml is None:
            raise RuntimeError("PyYAML is required to load a config file")
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg.source_path = path

        sim = raw.get("simulator", {}) or {}
        cfg.simulator = SimulatorSettings(
            seed=int(sim.get("seed", 42)),
            run_count=int(sim.get("run_count", 10)),
            run_minutes=int(sim.get("run_minutes", 90)),
            window_size_minutes=int(sim.get("window_size_minutes", 1)),
            base_http_per_minute=int(sim.get("base_http_per_minute", 24)),
            base_auth_per_minute=int(sim.get("base_auth_per_minute", 10)),
            base_db_per_minute=int(sim.get("base_db_per_minute", 24)),
            inject_malformed=bool(sim.get("inject_malformed", True)),
            max_events=int(sim.get("max_events", 250_000)),
            start_time=str(sim.get("start_time", "2026-01-01T00:00:00+00:00")),
        )

        r = raw.get("rules", {}) or {}
        cfg.rules = RuleConfig(
            rolling_windows=int(r.get("rolling_windows", 30)),
            rolling_min_history=int(r.get("rolling_min_history", 5)),
            min_window_events=int(r.get("min_window_events", 5)),
            http_5xx_min_requests=int(r.get("http_5xx_min_requests", 20)),
            http_5xx_min_count=int(r.get("http_5xx_min_count", 3)),
            http_5xx_ratio_threshold=float(r.get("http_5xx_ratio_threshold", 0.05)),
            http_5xx_strong_ratio=float(r.get("http_5xx_strong_ratio", 0.20)),
            http_5xx_delta_threshold=float(r.get("http_5xx_delta_threshold", 0.05)),
            db_min_events=int(r.get("db_min_events", 8)),
            db_timeout_ratio_threshold=float(r.get("db_timeout_ratio_threshold", 0.05)),
            db_timeout_strong_ratio=float(r.get("db_timeout_strong_ratio", 0.20)),
            db_connection_failure_ratio_threshold=float(r.get("db_connection_failure_ratio_threshold", 0.02)),
            db_connection_failure_strong_ratio=float(r.get("db_connection_failure_strong_ratio", 0.10)),
            auth_min_events=int(r.get("auth_min_events", 10)),
            auth_min_failures=int(r.get("auth_min_failures", 8)),
            auth_failure_ratio_threshold=float(r.get("auth_failure_ratio_threshold", 0.50)),
            auth_failure_strong_ratio=float(r.get("auth_failure_strong_ratio", 0.80)),
            auth_distinct_source_threshold=int(r.get("auth_distinct_source_threshold", 8)),
            latency_min_samples=int(r.get("latency_min_samples", 20)),
            latency_p95_ratio_threshold=float(r.get("latency_p95_ratio_threshold", 2.0)),
            latency_p95_strong_ratio=float(r.get("latency_p95_strong_ratio", 4.0)),
            latency_p95_absolute_floor_ms=float(r.get("latency_p95_absolute_floor_ms", 50.0)),
            max_routine_restarts=int(r.get("max_routine_restarts", 1)),
            weak_weight=float(r.get("weak_weight", 0.5)),
            strong_weight=float(r.get("strong_weight", 1.0)),
            decision_threshold=float(r.get("decision_threshold", 1.0)),
            high_severity_threshold=float(r.get("high_severity_threshold", 2.0)),
        )

        m = raw.get("model", {}) or {}
        cfg.model = ModelConfig(
            n_estimators=int(m.get("n_estimators", 200)),
            max_samples=int(m.get("max_samples", 256)),
            contamination=m.get("contamination", "auto"),
            random_state=int(m.get("random_state", 42)),
            n_jobs=int(m.get("n_jobs", 2)),
            bootstrap=bool(m.get("bootstrap", False)),
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
        cfg.split_strategies = [str(v) for v in (s.get("strategies") or ["time", "group"])]
        ratios = s.get("ratios") or {}
        cfg.split_ratios = {
            "train": float(ratios.get("train", 0.6)),
            "validation": float(ratios.get("validation", 0.2)),
            "test": float(ratios.get("test", 0.2)),
        }

        p = raw.get("paths", {}) or {}
        for key, value in p.items():
            if key in cfg.paths:
                cfg.paths[key] = str(value)

        l = raw.get("llm", {}) or {}
        cfg.llm = LlmSettings(
            enabled=bool(l.get("enabled", False)),
            provider=str(l.get("provider", "mock")),
            endpoint=str(l.get("endpoint", "")),
            model=str(l.get("model", "")),
            api_key_env=str(l.get("api_key_env", "LAD_LLM_API_KEY")),
            timeout_seconds=float(l.get("timeout_seconds", 20.0)),
        )

    _validate(cfg)
    return cfg


def _validate(cfg: NetworkConfig) -> None:
    if cfg.simulator.run_count < 3:
        raise ValueError("simulator.run_count must be >= 3 (train/validation/test need runs)")
    if cfg.simulator.run_minutes < 10:
        raise ValueError("simulator.run_minutes must be >= 10")
    if cfg.simulator.window_size_minutes < 1:
        raise ValueError("simulator.window_size_minutes must be >= 1")
    if cfg.simulator.max_events < 1:
        raise ValueError("simulator.max_events must be >= 1")
    if cfg.rules.rolling_windows < 1:
        raise ValueError("rules.rolling_windows must be >= 1")
    if cfg.rules.rolling_min_history < 1:
        raise ValueError("rules.rolling_min_history must be >= 1")
    if cfg.rules.min_window_events < 1:
        raise ValueError("rules.min_window_events must be >= 1")
    for name in ("http_5xx_ratio_threshold", "http_5xx_strong_ratio",
                 "db_timeout_ratio_threshold", "db_timeout_strong_ratio",
                 "db_connection_failure_ratio_threshold",
                 "db_connection_failure_strong_ratio",
                 "auth_failure_ratio_threshold", "auth_failure_strong_ratio"):
        value = getattr(cfg.rules, name)
        if not (0.0 < value <= 1.0):
            raise ValueError(f"rules.{name} must be in (0, 1]")
    if cfg.rules.latency_p95_ratio_threshold < 1.0:
        raise ValueError("rules.latency_p95_ratio_threshold must be >= 1")
    if cfg.rules.high_severity_threshold < cfg.rules.decision_threshold:
        raise ValueError("rules.high_severity_threshold must be >= decision_threshold")
    if len(cfg.feature_columns) != len(set(cfg.feature_columns)):
        raise ValueError("network feature columns contain duplicates")
    if not set(cfg.split_strategies).issubset({"time", "group"}):
        raise ValueError("split.strategies may only contain 'time' and 'group'")
    total = sum(cfg.split_ratios.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    if any(v <= 0 for v in cfg.split_ratios.values()):
        raise ValueError("every split ratio must be > 0")
    if cfg.llm.enabled and cfg.llm.provider not in {"mock", "http"}:
        raise ValueError("llm.provider must be 'mock' or 'http'")
    if cfg.llm.provider == "http" and cfg.llm.enabled and not cfg.llm.endpoint:
        raise ValueError("llm.endpoint is required when the http provider is enabled")

    _validate_baseline(cfg.baseline_config())
