"""Threshold selection for the Milestone 3 baseline.

The raw anomaly score and the final flag are separate things. This module turns
a continuous score into a boolean ``is_anomaly`` under an explicit, reproducible
rule.

**Direction.** ``raw_score`` comes from ``IsolationForest.score_samples``, where
*lower means more anomalous*. The flag is therefore ``is_anomaly = raw_score <=
threshold`` and the boundary is **inclusive** (a score exactly equal to the
threshold is flagged).

**Data discipline.** The threshold may only be selected from training and
validation information. Selecting it on the test split raises
:class:`~anomaly_detector.model_contract.ModelContractError` — the guard is
part of the code, not a convention.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .model_contract import ModelContractError


@dataclass
class ThresholdResult:
    """The chosen threshold plus everything needed to reproduce and explain it."""
    threshold: float
    method: str
    source_split: str
    objective: str = ""
    is_heuristic: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "method": self.method,
            "source_split": self.source_split,
            "objective": self.objective,
            "is_heuristic": self.is_heuristic,
            "details": dict(self.details),
        }


def apply_threshold(scores: Sequence[float], threshold: float) -> np.ndarray:
    """Convert raw scores to flags. The boundary is inclusive."""
    return np.asarray(scores, dtype="float64") <= float(threshold)


def _labels_as_bool(labels: Sequence[Any]) -> np.ndarray:
    values = np.asarray(labels)
    out = np.zeros(len(values), dtype=bool)
    for i, value in enumerate(values):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise ModelContractError("labels contain missing values; cannot select a threshold")
        if isinstance(value, (bool, np.bool_)):
            out[i] = bool(value)
        elif value in (0, 1):
            out[i] = bool(value)
        else:
            raise ModelContractError(f"labels must be booleans or 0/1, got {value!r}")
    return out


def _precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def _candidate_thresholds(scores: np.ndarray) -> List[float]:
    """Candidate cut-offs, including "flag nothing" and "flag everything"."""
    unique = np.unique(scores)
    below = float(unique.min()) - 1.0
    above = float(unique.max()) + 1.0
    return [below] + [float(v) for v in unique] + [above]


def _select_by_f1(scores: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    best = None
    for candidate in _candidate_thresholds(scores):
        predicted = apply_threshold(scores, candidate)
        precision, recall, f1 = _precision_recall_f1(y, predicted)
        key = (f1, recall, -candidate)  # maximise F1, then recall, then prefer the lower cut
        if best is None or key > best[0]:
            best = (key, candidate, precision, recall, f1, int(predicted.sum()))
    assert best is not None
    _, candidate, precision, recall, f1, flagged = best
    details = {
        "precision_at_threshold": precision,
        "recall_at_threshold": recall,
        "f1_at_threshold": f1,
        "flagged_windows": flagged,
        "candidates_evaluated": len(_candidate_thresholds(scores)),
        "tie_break": "max F1, then max recall, then lowest threshold",
    }
    return candidate, details


def _select_by_precision(scores: np.ndarray, y: np.ndarray, min_precision: float):
    best_feasible = None
    best_any = None
    for candidate in _candidate_thresholds(scores):
        predicted = apply_threshold(scores, candidate)
        precision, recall, f1 = _precision_recall_f1(y, predicted)
        key = (precision, recall, -candidate)
        if best_any is None or key > best_any[0]:
            best_any = (key, candidate, precision, recall, f1, int(predicted.sum()))
        if precision >= min_precision:
            # Among feasible thresholds prefer the most sensitive (highest recall).
            feasible_key = (recall, precision, -candidate)
            if best_feasible is None or feasible_key > best_feasible[0]:
                best_feasible = (feasible_key, candidate, precision, recall, f1, int(predicted.sum()))
    if best_feasible is not None:
        return best_feasible[1], {
            "precision_at_threshold": best_feasible[2],
            "recall_at_threshold": best_feasible[3],
            "f1_at_threshold": best_feasible[4],
            "flagged_windows": best_feasible[5],
            "constraint": f"precision >= {min_precision}",
            "constraint_met": True,
            "selection": "highest recall among thresholds meeting the precision constraint",
        }
    if best_any is None:
        raise ModelContractError("could not evaluate any candidate threshold")
    return None, {
        "precision_at_threshold": best_any[2],
        "recall_at_threshold": best_any[3],
        "f1_at_threshold": best_any[4],
        "flagged_windows": best_any[5],
        "constraint": f"precision >= {min_precision}",
        "constraint_met": False,
        "selection": "no threshold met the precision constraint",
    }


def _select_by_recall(scores: np.ndarray, y: np.ndarray, min_recall: float):
    best_feasible = None
    best_any = None
    for candidate in _candidate_thresholds(scores):
        predicted = apply_threshold(scores, candidate)
        precision, recall, f1 = _precision_recall_f1(y, predicted)
        key = (recall, precision, -candidate)
        if best_any is None or key > best_any[0]:
            best_any = (key, candidate, precision, recall, f1, int(predicted.sum()))
        if recall >= min_recall:
            feasible_key = (precision, recall, -candidate)
            if best_feasible is None or feasible_key > best_feasible[0]:
                best_feasible = (feasible_key, candidate, precision, recall, f1, int(predicted.sum()))
    if best_feasible is not None:
        return best_feasible[1], {
            "precision_at_threshold": best_feasible[2],
            "recall_at_threshold": best_feasible[3],
            "f1_at_threshold": best_feasible[4],
            "flagged_windows": best_feasible[5],
            "constraint": f"recall >= {min_recall}",
            "constraint_met": True,
            "selection": "highest precision among thresholds meeting the recall constraint",
        }
    if best_any is None:
        raise ModelContractError("could not evaluate any candidate threshold")
    return None, {
        "precision_at_threshold": best_any[2],
        "recall_at_threshold": best_any[3],
        "f1_at_threshold": best_any[4],
        "flagged_windows": best_any[5],
        "constraint": f"recall >= {min_recall}",
        "constraint_met": False,
        "selection": "no threshold met the recall constraint",
    }


def train_quantile_threshold(train_scores: Sequence[float], quantile: float) -> float:
    """Operational heuristic: flag the ``quantile`` fraction of training scores.

    Because lower scores are more anomalous, the ``quantile``-quantile of the
    training scores is the cut-off that flags roughly that fraction of rows.
    This is a *heuristic*, not a validated optimum.
    """
    scores = np.asarray(train_scores, dtype="float64")
    if scores.size == 0:
        raise ModelContractError("cannot derive a quantile threshold from an empty score list")
    if not (0.0 < quantile < 1.0):
        raise ModelContractError(f"quantile must be in (0, 1), got {quantile}")
    return float(np.quantile(scores, quantile))


def select_threshold(
    scores: Sequence[float],
    labels: Optional[Sequence[Any]],
    method: str = "validation_f1",
    source_split: str = "validation",
    train_scores: Optional[Sequence[float]] = None,
    min_precision: float = 0.5,
    min_recall: float = 0.5,
    fallback_method: str = "train_quantile",
    fallback_quantile: float = 0.01,
) -> ThresholdResult:
    """Select a threshold using training/validation information only.

    The ``test`` split is refused outright. When the requested method cannot be
    applied (labels missing, single-class, constraint unmet) the documented
    fallback is used and the result is marked as a heuristic.
    """
    if source_split == "test":
        raise ModelContractError(
            "the test split must never be used to select a threshold; "
            "use validation or training information"
        )

    score_array = np.asarray(scores, dtype="float64")
    if score_array.size == 0:
        raise ModelContractError("cannot select a threshold from an empty score list")
    if not np.isfinite(score_array).all():
        raise ModelContractError("scores contain non-finite values")

    def _fallback(reason: str) -> ThresholdResult:
        if train_scores is None:
            raise ModelContractError(
                f"threshold fallback {fallback_method!r} requires training scores, which were not provided"
            )
        if fallback_method == "train_quantile":
            threshold = train_quantile_threshold(train_scores, fallback_quantile)
            return ThresholdResult(
                threshold=threshold,
                method="train_quantile",
                source_split="train",
                objective=f"quantile({fallback_quantile}) of training scores",
                is_heuristic=True,
                details={
                    "reason": reason,
                    "quantile": fallback_quantile,
                    "note": "operational heuristic, not a validated optimum",
                },
            )
        raise ModelContractError(f"unsupported fallback method {fallback_method!r}")

    if method == "train_quantile":
        if train_scores is None:
            raise ModelContractError("the train_quantile method requires training scores")
        threshold = train_quantile_threshold(train_scores, fallback_quantile)
        return ThresholdResult(
            threshold=threshold,
            method="train_quantile",
            source_split="train",
            objective=f"quantile({fallback_quantile}) of training scores",
            is_heuristic=True,
            details={"quantile": fallback_quantile,
                     "note": "operational heuristic, not a validated optimum"},
        )

    if labels is None:
        return _fallback("no labels available for threshold selection")
    y = _labels_as_bool(labels)
    if y.size != score_array.size:
        raise ModelContractError(
            f"scores ({score_array.size}) and labels ({y.size}) have different lengths"
        )
    if len(np.unique(y)) < 2:
        return _fallback("validation labels contain a single class")

    if method == "validation_f1":
        threshold, details = _select_by_f1(score_array, y)
        return ThresholdResult(
            threshold=threshold,
            method="validation_f1",
            source_split=source_split,
            objective="maximise F1 on the validation split",
            is_heuristic=False,
            details=details,
        )
    if method == "validation_precision":
        threshold, details = _select_by_precision(score_array, y, min_precision)
        if threshold is None:
            fallback = _fallback(details.get("selection", "precision constraint unmet"))
            fallback.details.update(details)
            return fallback
        return ThresholdResult(
            threshold=threshold,
            method="validation_precision",
            source_split=source_split,
            objective=f"precision >= {min_precision}",
            is_heuristic=False,
            details=details,
        )
    if method == "validation_recall":
        threshold, details = _select_by_recall(score_array, y, min_recall)
        if threshold is None:
            fallback = _fallback(details.get("selection", "recall constraint unmet"))
            fallback.details.update(details)
            return fallback
        return ThresholdResult(
            threshold=threshold,
            method="validation_recall",
            source_split=source_split,
            objective=f"recall >= {min_recall}",
            is_heuristic=False,
            details=details,
        )

    raise ModelContractError(f"unknown threshold method {method!r}")
