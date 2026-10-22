"""Model-selection policy for Milestone 7 (validation-only).

This module replaces the M4 auto-picker, which keyed the winner on **test**
metrics. The root cause is documented in ``docs/READINESS_AUDIT.md``.

Guarantees
----------
* Selection reads **validation** metrics only. Test metrics are structurally
  absent from :class:`CandidateEvidence` used for selection.
* A challenger must beat the incumbent by a pre-declared margin *and* pass
  every pre-declared robustness floor. Otherwise the incumbent is retained.
* The weights, the margin and the floors all come from an explicit config
  file (``configs/selection.yaml``), never from code.
* The policy is deterministic: the same evidence always yields the same
  decision.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is a declared dependency
    yaml = None


#: Metrics the policy is allowed to read. Anything else is rejected.
ALLOWED_SELECTION_METRICS = {
    "f1",
    "precision",
    "recall",
    "false_positive_rate",
    "pr_auc",
    "n_windows",
    "n_positive",
    "n_flagged",
    "runtime_s",
}

#: Metrics that must never influence the decision.
FORBIDDEN_SELECTION_METRICS = {
    "test_f1",
    "test_precision",
    "test_recall",
    "test_pr_auc",
    "test_roc_auc",
    "test_fp",
    "test_tp",
    "test_flagged",
}


class SelectionPolicyError(ValueError):
    """Raised when the policy or the evidence violates the contract."""


@dataclass
class CandidateEvidence:
    """Validation-split evidence for one candidate.

    Only validation metrics are representable here. There is no field for a
    test metric — that is the structural guarantee that the test split cannot
    leak into selection.
    """

    name: str
    validation: Dict[str, Any] = field(default_factory=dict)
    robustness: Dict[str, Any] = field(default_factory=dict)
    training_rows_used: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        offending = set(self.validation) & FORBIDDEN_SELECTION_METRICS
        if offending:
            raise SelectionPolicyError(
                f"candidate {self.name!r} carries test-split metrics {sorted(offending)}; "
                "selection evidence must be validation-only"
            )
        unknown = set(self.validation) - ALLOWED_SELECTION_METRICS
        if unknown:
            raise SelectionPolicyError(
                f"candidate {self.name!r} carries unsupported selection metrics {sorted(unknown)}; "
                f"allowed: {sorted(ALLOWED_SELECTION_METRICS)}"
            )


@dataclass
class SelectionDecision:
    """The outcome of applying the policy to the evidence."""

    selected: Optional[str]
    incumbent: str
    reason: str
    weighted_scores: Dict[str, float] = field(default_factory=dict)
    margin: float = 0.0
    required_margin: float = 0.0
    floors_failed: Dict[str, List[str]] = field(default_factory=dict)
    candidates_considered: List[str] = field(default_factory=list)
    policy_name: str = ""
    used_test_metrics: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected": self.selected,
            "incumbent": self.incumbent,
            "reason": self.reason,
            "weighted_scores": dict(self.weighted_scores),
            "margin": self.margin,
            "required_margin": self.required_margin,
            "floors_failed": {k: list(v) for k, v in self.floors_failed.items()},
            "candidates_considered": list(self.candidates_considered),
            "policy_name": self.policy_name,
            "used_test_metrics": self.used_test_metrics,
            "selection_evidence_split": "validation",
        }


@dataclass
class SelectionPolicy:
    """Pre-declared, config-driven selection policy."""

    name: str
    weights: Dict[str, float]
    min_improvement_margin: float
    incumbent: str
    robustness_floors: Dict[str, float]
    eligible_candidates: List[str]
    tie_break: str = ""
    source_path: str = ""

    # -- construction ----------------------------------------------------
    @classmethod
    def from_dict(cls, raw: Dict[str, Any], source_path: str = "") -> "SelectionPolicy":
        policy = raw.get("policy", raw) or {}
        weights = {k: float(v) for k, v in (policy.get("weights") or {}).items()}
        if not weights:
            raise SelectionPolicyError("selection policy requires non-empty weights")
        if any(v < 0 for v in weights.values()):
            raise SelectionPolicyError("selection policy weights must be non-negative")
        floors = {k: float(v) for k, v in (policy.get("robustness_floors") or {}).items()}
        eligible = list(policy.get("eligible_candidates") or [])
        if not eligible:
            raise SelectionPolicyError("selection policy requires eligible_candidates")
        return cls(
            name=str(policy.get("name", "unnamed")),
            weights=weights,
            min_improvement_margin=float(policy.get("min_improvement_margin", 0.0)),
            incumbent=str(policy.get("incumbent", eligible[0])),
            robustness_floors=floors,
            eligible_candidates=eligible,
            tie_break=str(policy.get("tie_break", "")),
            source_path=source_path,
        )

    @classmethod
    def load(cls, path: Optional[str] = None) -> "SelectionPolicy":
        path = path or os.path.join("configs", "selection.yaml")
        if not os.path.isfile(path):
            raise SelectionPolicyError(f"selection policy file not found: {path}")
        if yaml is None:  # pragma: no cover
            raise SelectionPolicyError("PyYAML is required to read the selection policy")
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.from_dict(raw, source_path=path)

    # -- scoring ---------------------------------------------------------
    def weighted_score(self, validation: Dict[str, Any]) -> float:
        """Weighted validation score. Higher is better.

        ``false_positive_rate`` is inverted because lower is better for a
        false-positive rate. Missing components contribute 0.0 and are
        reported, so a candidate cannot win by omitting a metric.
        """
        total = 0.0
        for name, weight in self.weights.items():
            value = validation.get(name)
            if value is None:
                continue
            value = float(value)
            if name == "false_positive_rate":
                value = 1.0 - value
            total += weight * value
        return total

    def failed_floors(self, evidence: CandidateEvidence) -> List[str]:
        """Which robustness floors (if any) this candidate fails."""
        failed: List[str] = []
        validation = evidence.validation
        robustness = evidence.robustness

        floor = self.robustness_floors.get("max_validation_false_positive_rate")
        if floor is not None and validation.get("false_positive_rate") is not None:
            if float(validation["false_positive_rate"]) > floor:
                failed.append(
                    f"validation_false_positive_rate {validation['false_positive_rate']:.3f} > {floor}"
                )
        floor = self.robustness_floors.get("min_validation_recall")
        if floor is not None and validation.get("recall") is not None:
            if float(validation["recall"]) < floor:
                failed.append(f"validation_recall {validation['recall']:.3f} < {floor}")
        floor = self.robustness_floors.get("min_validation_precision")
        if floor is not None and validation.get("precision") is not None:
            if float(validation["precision"]) < floor:
                failed.append(f"validation_precision {validation['precision']:.3f} < {floor}")
        floor = self.robustness_floors.get("max_benign_shift_flag_rate")
        if floor is not None and robustness.get("benign_shift_flag_rate") is not None:
            if float(robustness["benign_shift_flag_rate"]) > floor:
                failed.append(
                    f"benign_shift_flag_rate {robustness['benign_shift_flag_rate']:.3f} > {floor}"
                )
        floor = self.robustness_floors.get("max_missing_telemetry_flag_rate")
        if floor is not None and robustness.get("missing_telemetry_flag_rate") is not None:
            if float(robustness["missing_telemetry_flag_rate"]) > floor:
                failed.append(
                    f"missing_telemetry_flag_rate {robustness['missing_telemetry_flag_rate']:.3f} > {floor}"
                )
        return failed

    # -- decision --------------------------------------------------------
    def decide(self, evidence: List[CandidateEvidence]) -> SelectionDecision:
        """Apply the policy. Reads validation evidence only."""
        by_name = {e.name: e for e in evidence}
        considered = [name for name in self.eligible_candidates if name in by_name]
        if not considered:
            raise SelectionPolicyError(
                f"no eligible candidate has evidence; eligible={self.eligible_candidates}, "
                f"evidence={sorted(by_name)}"
            )
        if self.incumbent not in by_name:
            raise SelectionPolicyError(
                f"the incumbent {self.incumbent!r} has no evidence; a selection decision "
                "cannot be made without the baseline"
            )

        scores = {name: self.weighted_score(by_name[name].validation) for name in considered}
        floors_failed = {name: self.failed_floors(by_name[name]) for name in considered}

        incumbent_score = scores[self.incumbent]
        best_name = self.incumbent
        best_score = incumbent_score
        best_reason = (
            f"incumbent {self.incumbent!r} retained: no eligible challenger beat it by the "
            f"required margin of {self.min_improvement_margin:.3f} on validation-only evidence"
        )

        for name in considered:
            if name == self.incumbent:
                continue
            margin = scores[name] - incumbent_score
            if margin <= self.min_improvement_margin:
                continue
            if floors_failed.get(name):
                continue
            if margin > best_score - incumbent_score or best_name == self.incumbent:
                best_name = name
                best_score = scores[name]
                best_reason = (
                    f"challenger {name!r} beat the incumbent {self.incumbent!r} by "
                    f"{margin:.4f} (>= {self.min_improvement_margin:.3f}) on validation-only "
                    "evidence and passed every robustness floor"
                )

        margin_achieved = best_score - incumbent_score
        return SelectionDecision(
            selected=best_name,
            incumbent=self.incumbent,
            reason=best_reason,
            weighted_scores=scores,
            margin=margin_achieved,
            required_margin=self.min_improvement_margin,
            floors_failed={k: v for k, v in floors_failed.items() if v},
            candidates_considered=considered,
            policy_name=self.name,
            used_test_metrics=False,
        )
