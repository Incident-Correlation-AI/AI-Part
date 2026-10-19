"""Tests for the Milestone 2 readiness configuration loader."""
import os
import pytest

from anomaly_detector.readiness_config import (
    ReadinessConfig,
    load_readiness_config,
)


def test_loads_repo_config():
    cfg = load_readiness_config(os.path.join("configs", "data_readiness.yaml"))
    assert cfg.split.strategy in {"random", "time", "group"}
    assert abs(sum(cfg.split.ratios.values()) - 1.0) < 1e-9
    assert "event_count" in cfg.feature_roles.eligible_features
    assert cfg.feature_roles.unknown_role_policy == "error"


def test_defaults_when_no_file():
    cfg = load_readiness_config(None)
    assert cfg.split.strategy == "random"
    assert cfg.data_policy.missing_value_action == "report"


def test_invalid_ratios_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "split:\n  strategy: random\n  ratios:\n    train: 0.5\n    test: 0.2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_readiness_config(str(p))


def test_invalid_strategy_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("split:\n  strategy: chronological\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_readiness_config(str(p))


def test_invalid_role_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "feature_roles:\n  patterns:\n    - role: mystery\n      match: [x]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_readiness_config(str(p))


def test_invalid_unknown_policy_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("feature_roles:\n  unknown_role_policy: ignore\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_readiness_config(str(p))
