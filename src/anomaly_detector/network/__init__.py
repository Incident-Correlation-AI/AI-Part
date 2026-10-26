"""Network/service-log anomaly detection extension.

This package sits *beside* the existing ``features-v1`` pipeline.  It adds a
network-oriented feature schema (``network-features-v1``), a local incident
simulator, a transparent rule/MAD baseline, an Isolation Forest trained on the
same schema, and a narrow evidence-grounded LLM explanation adapter.

Nothing in this package changes the existing ``features-v1`` contract.  The
result envelope is still :class:`anomaly_detector.inference.contract.InferenceResult`;
only an additive feature-schema identifier and telemetry reason codes were
introduced (see ``docs/ml_network_integration.md``).
"""
from .config import NetworkConfig, load_network_config
from .evaluation import evaluate_split_predictions, evaluate_strategy_predictions
from .features import (
    NETWORK_FEATURE_COLUMNS,
    NETWORK_FEATURE_SCHEMA_VERSION,
    NetworkLabel,
    NetworkWindow,
    build_network_windows,
    join_windows_with_labels,
    windows_to_dataframe,
)
from .lad_adapter import (
    LAD_EVENT_SCHEMA_VERSION,
    LadParseOptions,
    LadParseResult,
    QualityRecord,
    RejectedLadEvent,
    parse_lad_events,
)
from .inference import (
    NetworkDetector,
    NetworkDetectorOptions,
    NetworkModelUnavailableError,
    load_network_detector,
)
from .llm import (
    DraftExplanation,
    EvidenceBundle,
    Explanation,
    HttpExplanationProvider,
    MockExplanationProvider,
    explain_result,
    validate_draft,
)
from .pipeline import NetworkRunResult, StrategyRun, build_predictions, run_network_pipeline
from .simulator import SimulatedDataset, SimulatorConfig, simulate
from .splits import SplitAssignment, make_split_assignment

__all__ = [
    "DraftExplanation",
    "EvidenceBundle",
    "Explanation",
    "HttpExplanationProvider",
    "LAD_EVENT_SCHEMA_VERSION",
    "LadParseOptions",
    "LadParseResult",
    "MockExplanationProvider",
    "NETWORK_FEATURE_COLUMNS",
    "NETWORK_FEATURE_SCHEMA_VERSION",
    "NetworkConfig",
    "NetworkDetector",
    "NetworkDetectorOptions",
    "NetworkLabel",
    "NetworkModelUnavailableError",
    "NetworkRunResult",
    "NetworkWindow",
    "QualityRecord",
    "RejectedLadEvent",
    "SimulatedDataset",
    "SimulatorConfig",
    "SplitAssignment",
    "StrategyRun",
    "build_network_windows",
    "build_predictions",
    "evaluate_split_predictions",
    "evaluate_strategy_predictions",
    "explain_result",
    "join_windows_with_labels",
    "load_network_config",
    "load_network_detector",
    "make_split_assignment",
    "parse_lad_events",
    "run_network_pipeline",
    "simulate",
    "validate_draft",
    "windows_to_dataframe",
]
