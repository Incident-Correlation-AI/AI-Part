"""Configuration for the evidence-correlation / incident-draft adapter.

The adapter has no import-time network calls and is disabled by default with
respect to explanations.  Configuration is explicit and can come from a YAML
file and/or the environment.  No secret values are read here: the HTTP provider
reads its API key from the named environment variable at call time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import os

try:  # PyYAML is already a declared project dependency
    import yaml
except ImportError:  # pragma: no cover - only when the dependency is missing
    yaml = None


_VALID_PROVIDERS = {"none", "mock", "http"}


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value)


@dataclass(frozen=True)
class CorrelationConfig:
    """Bounds and switches for one correlation run.

    ``window_seconds`` is passed to the upstream correlator unchanged.  All
    bounds are hard limits, not performance hints: exceeding them truncates
    evidence and is recorded in the result.
    """

    window_seconds: int = 120
    require_supporting_events: bool = True
    max_supporting_events_per_window: int = 100
    max_events_per_incident: int = 200
    max_redacted_refs_per_incident: int = 50
    max_llm_hypotheses_per_incident: int = 3
    max_diagnostic_checks_per_incident: int = 8
    explain: bool = False
    include_llm_checks: bool = True
    explanation_fallback: bool = True
    llm_provider: str = "mock"
    llm_endpoint: str = ""
    llm_model: str = ""
    llm_api_key_env: str = "LAD_LLM_API_KEY"
    llm_timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        if self.window_seconds < 0:
            raise ValueError("window_seconds cannot be negative")
        for name in (
            "max_supporting_events_per_window",
            "max_events_per_incident",
            "max_redacted_refs_per_incident",
            "max_diagnostic_checks_per_incident",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.max_llm_hypotheses_per_incident < 0:
            raise ValueError("max_llm_hypotheses_per_incident cannot be negative")
        if self.llm_provider not in _VALID_PROVIDERS:
            raise ValueError(
                f"llm_provider must be one of {sorted(_VALID_PROVIDERS)}, "
                f"got {self.llm_provider!r}"
            )
        if self.llm_timeout_seconds <= 0:
            raise ValueError("llm_timeout_seconds must be > 0")
        if self.explain and self.llm_provider == "http" and not self.llm_endpoint:
            raise ValueError("llm_endpoint is required when the http provider is enabled")

    @classmethod
    def from_env(cls) -> "CorrelationConfig":
        """Build a config from ``LAD_CORRELATION_*`` / ``LAD_LLM_*`` variables."""
        return cls(
            window_seconds=_env_int("LAD_CORRELATION_WINDOW_SECONDS", 120),
            require_supporting_events=_env_bool(
                "LAD_CORRELATION_REQUIRE_EVENT_EVIDENCE", True
            ),
            max_supporting_events_per_window=_env_int(
                "LAD_CORRELATION_MAX_EVENTS_PER_WINDOW", 100
            ),
            max_events_per_incident=_env_int(
                "LAD_CORRELATION_MAX_EVENTS_PER_INCIDENT", 200
            ),
            max_redacted_refs_per_incident=_env_int(
                "LAD_CORRELATION_MAX_REDACTED_REFS", 50
            ),
            max_llm_hypotheses_per_incident=_env_int(
                "LAD_CORRELATION_MAX_LLM_HYPOTHESES", 3
            ),
            max_diagnostic_checks_per_incident=_env_int(
                "LAD_CORRELATION_MAX_DIAGNOSTIC_CHECKS", 8
            ),
            explain=_env_bool("LAD_CORRELATION_EXPLAIN", False),
            include_llm_checks=_env_bool("LAD_CORRELATION_INCLUDE_LLM_CHECKS", True),
            explanation_fallback=_env_bool("LAD_CORRELATION_EXPLANATION_FALLBACK", True),
            llm_provider=os.environ.get("LAD_LLM_PROVIDER", "mock").strip().lower(),
            llm_endpoint=os.environ.get("LAD_LLM_ENDPOINT", "").strip(),
            llm_model=os.environ.get("LAD_LLM_MODEL", "").strip(),
            llm_api_key_env=os.environ.get("LAD_LLM_API_KEY_ENV", "LAD_LLM_API_KEY").strip(),
            llm_timeout_seconds=_env_float("LAD_LLM_TIMEOUT_SECONDS", 20.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable configuration; never contains a secret value."""
        return {
            "window_seconds": self.window_seconds,
            "require_supporting_events": self.require_supporting_events,
            "max_supporting_events_per_window": self.max_supporting_events_per_window,
            "max_events_per_incident": self.max_events_per_incident,
            "max_redacted_refs_per_incident": self.max_redacted_refs_per_incident,
            "max_llm_hypotheses_per_incident": self.max_llm_hypotheses_per_incident,
            "max_diagnostic_checks_per_incident": self.max_diagnostic_checks_per_incident,
            "explain": self.explain,
            "include_llm_checks": self.include_llm_checks,
            "explanation_fallback": self.explanation_fallback,
            "llm": {
                "provider": self.llm_provider,
                "endpoint": self.llm_endpoint,
                "model": self.llm_model,
                "api_key_env": self.llm_api_key_env,
                "timeout_seconds": self.llm_timeout_seconds,
            },
        }


def load_correlation_config(path: Optional[str] = None) -> CorrelationConfig:
    """Load ``configs/correlation.yaml`` and apply environment overrides.

    A missing path is not an error: defaults plus environment variables are
    used.  An explicitly supplied path that does not exist is an error, so a
    typo cannot silently fall back to defaults.
    """
    explicit = path or os.environ.get("LAD_CORRELATION_CONFIG") or ""
    raw: Dict[str, Any] = {}
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"correlation config not found: {explicit}")
        if yaml is None:
            raise RuntimeError("PyYAML is required to load a correlation config file")
        with open(explicit, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    section = raw.get("correlation", raw) or {}
    llm = raw.get("llm", {}) or {}
    base = CorrelationConfig()
    provider = os.environ.get("LAD_LLM_PROVIDER")
    endpoint = os.environ.get("LAD_LLM_ENDPOINT")
    model = os.environ.get("LAD_LLM_MODEL")
    api_key_env = os.environ.get("LAD_LLM_API_KEY_ENV")
    timeout = os.environ.get("LAD_LLM_TIMEOUT_SECONDS")

    def env_bool(name: str, value: bool) -> bool:
        return _env_bool(name, value) if name in os.environ else value

    return CorrelationConfig(
        window_seconds=_env_int(
            "LAD_CORRELATION_WINDOW_SECONDS",
            int(section.get("window_seconds", base.window_seconds)),
        ),
        require_supporting_events=env_bool(
            "LAD_CORRELATION_REQUIRE_EVENT_EVIDENCE",
            bool(section.get("require_supporting_events", base.require_supporting_events)),
        ),
        max_supporting_events_per_window=_env_int(
            "LAD_CORRELATION_MAX_EVENTS_PER_WINDOW",
            int(
                section.get(
                    "max_supporting_events_per_window",
                    base.max_supporting_events_per_window,
                )
            ),
        ),
        max_events_per_incident=_env_int(
            "LAD_CORRELATION_MAX_EVENTS_PER_INCIDENT",
            int(section.get("max_events_per_incident", base.max_events_per_incident)),
        ),
        max_redacted_refs_per_incident=_env_int(
            "LAD_CORRELATION_MAX_REDACTED_REFS",
            int(
                section.get(
                    "max_redacted_refs_per_incident",
                    base.max_redacted_refs_per_incident,
                )
            ),
        ),
        max_llm_hypotheses_per_incident=_env_int(
            "LAD_CORRELATION_MAX_LLM_HYPOTHESES",
            int(
                section.get(
                    "max_llm_hypotheses_per_incident",
                    base.max_llm_hypotheses_per_incident,
                )
            ),
        ),
        max_diagnostic_checks_per_incident=_env_int(
            "LAD_CORRELATION_MAX_DIAGNOSTIC_CHECKS",
            int(
                section.get(
                    "max_diagnostic_checks_per_incident",
                    base.max_diagnostic_checks_per_incident,
                )
            ),
        ),
        explain=env_bool(
            "LAD_CORRELATION_EXPLAIN", bool(section.get("explain", base.explain))
        ),
        include_llm_checks=env_bool(
            "LAD_CORRELATION_INCLUDE_LLM_CHECKS",
            bool(section.get("include_llm_checks", base.include_llm_checks)),
        ),
        explanation_fallback=env_bool(
            "LAD_CORRELATION_EXPLANATION_FALLBACK",
            bool(section.get("explanation_fallback", base.explanation_fallback)),
        ),
        llm_provider=str(provider if provider is not None else llm.get("provider", base.llm_provider)).strip().lower(),
        llm_endpoint=str(endpoint if endpoint is not None else llm.get("endpoint", base.llm_endpoint)).strip(),
        llm_model=str(model if model is not None else llm.get("model", base.llm_model)).strip(),
        llm_api_key_env=str(
            api_key_env if api_key_env is not None else llm.get("api_key_env", base.llm_api_key_env)
        ).strip(),
        llm_timeout_seconds=float(
            timeout if timeout is not None else llm.get("timeout_seconds", base.llm_timeout_seconds)
        ),
    )
