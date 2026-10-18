"""Configuration loading for the pipeline.

Defaults are documented in ``configs/default.yaml``. Paths in the config are
resolved relative to the current working directory so the project stays
portable and free of hard-coded machine-specific paths.
"""
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import os

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a declared dependency
    yaml = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "window_size_minutes": 1,
    "random_seed": 42,
    "allow_empty_windows": False,
    "output_dir": "data/processed",
    "synthetic_dir": "data/synthetic",
    "artifacts_dir": "artifacts",
    "scenario_duration_hours": 1,
}


@dataclass
class Config:
    window_size_minutes: int = 1
    random_seed: int = 42
    allow_empty_windows: bool = False
    output_dir: str = "data/processed"
    synthetic_dir: str = "data/synthetic"
    artifacts_dir: str = "artifacts"
    scenario_duration_hours: float = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from YAML, falling back to documented defaults."""
    data = dict(DEFAULT_CONFIG)
    if path and os.path.isfile(path):
        if yaml is None:
            raise RuntimeError("PyYAML is required to load a config file")
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        # Accept both `random_seed` and `seed` keys for convenience.
        if "seed" in loaded and "random_seed" not in loaded:
            loaded["random_seed"] = loaded.pop("seed")
        data.update({k: v for k, v in loaded.items() if k in DEFAULT_CONFIG})
    return Config(**data)
