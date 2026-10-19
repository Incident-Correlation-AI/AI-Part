"""Tests for feature-role classification."""
from anomaly_detector.readiness_config import FeatureRoleConfig, RoleRule
from anomaly_detector.roles import classify_columns, detect_leaked_columns


def _config():
    return FeatureRoleConfig(
        explicit={"legacy_flag": "label"},
        patterns=[
            RoleRule("label", ["is_anomaly", "scenario_name", "label"]),
            RoleRule("timestamp", ["window_start", "window_end"]),
            RoleRule("identifier", ["host_id", "service", "sample_key"]),
            RoleRule("future_derived", ["future", "next_"]),
            RoleRule("feature", ["count", "ratio"]),
        ],
        eligible_features=["event_count", "error_ratio"],
        unknown_role_policy="error",
    )


def test_roles_assigned_and_eligibility_is_allowlist_driven():
    cols = ["event_count", "error_ratio", "is_anomaly", "scenario_name",
            "host_id", "window_start", "mean_latency_ms"]
    report = classify_columns(cols, _config())
    assert report.roles["is_anomaly"].role == "label"
    assert report.roles["scenario_name"].role == "label"
    assert report.roles["host_id"].role == "identifier"
    assert report.roles["window_start"].role == "timestamp"
    # eligible regardless of role: `error_ratio` is a feature; `event_count` too
    assert report.eligible_features == ["error_ratio", "event_count"]
    # `mean_latency_ms` is numeric-looking but not allow-listed
    assert "mean_latency_ms" not in report.eligible_features
    assert report.roles["mean_latency_ms"].eligible is False


def test_unknown_role_reported():
    report = classify_columns(["event_count", "mystery_field"], _config())
    assert report.unknown == ["mystery_field"]
    assert report.roles["event_count"].eligible is True


def test_explicit_override_wins():
    report = classify_columns(["legacy_flag"], _config())
    assert report.roles["legacy_flag"].role == "label"


def test_future_derived_not_eligible():
    cfg = _config()
    cfg.eligible_features = ["event_next_count"]
    report = classify_columns(["event_next_count"], cfg)
    assert report.roles["event_next_count"].eligible is False
    assert report.violations and report.violations[0]["column"] == "event_next_count"


def test_detect_leaked_columns():
    cols = ["event_count", "is_anomaly", "host_id", "window_start", "scenario_name"]
    leaked = set(detect_leaked_columns(cols, _config()))
    assert leaked == {"is_anomaly", "host_id", "window_start", "scenario_name"}


def test_ambiguous_column_recorded():
    # A column matching both identifier and feature is flagged ambiguous but
    # still classified by the first matching role.
    cfg = _config()
    cfg.patterns = [
        RoleRule("identifier", ["count"]),
        RoleRule("feature", ["count"]),
    ]
    cfg.eligible_features = ["event_count"]
    report = classify_columns(["event_count"], cfg)
    assert report.roles["event_count"].role == "identifier"
    assert report.roles["event_count"].eligible is False
    assert report.ambiguous
