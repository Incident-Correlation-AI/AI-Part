"""Time-aware and group-aware splits for network windows.

Two strategies are supported and both are *leakage-resistant* by construction:

``time``
    Windows are ordered by their start timestamp and cut into
    train/validation/test by unique timestamp.  Every training timestamp is
    strictly earlier than every validation timestamp, and every validation
    timestamp is strictly earlier than every test timestamp.

``group``
    Whole simulated runs are assigned to one split.  No run contributes
    windows to more than one split, so run-specific noise cannot leak across
    the boundary.

Both strategies are deterministic for a given seed and dataset.  The split
assignment is a data-partitioning concern only: labels are never part of the
features and are only used here to identify run ids.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import random

from .features import NetworkLabel, NetworkWindow

VALID_STRATEGIES = frozenset({"time", "group"})
SPLIT_NAMES = ("train", "validation", "test")


class SplitError(ValueError):
    """Raised when a split cannot be produced under the requested policy."""


@dataclass
class SplitAssignment:
    strategy: str
    seed: int
    indices: Dict[str, List[int]]
    runs: Dict[str, List[str]]
    integrity: Dict[str, Any] = field(default_factory=dict)

    def indices_for(self, split: str) -> List[int]:
        return list(self.indices[split])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "sizes": {name: len(self.indices[name]) for name in SPLIT_NAMES},
            "runs": {name: list(self.runs[name]) for name in SPLIT_NAMES},
            "integrity": dict(self.integrity),
        }


def make_split_assignment(
    windows: Sequence[NetworkWindow],
    labels: Sequence[NetworkLabel],
    strategy: str = "time",
    ratios: Optional[Mapping[str, float]] = None,
    seed: int = 42,
) -> SplitAssignment:
    """Partition window indices into train/validation/test.

    ``windows`` and ``labels`` must be aligned (same length and same order).
    """
    if strategy not in VALID_STRATEGIES:
        raise SplitError(
            f"unknown split strategy {strategy!r}; expected one of {sorted(VALID_STRATEGIES)}"
        )
    if len(windows) != len(labels):
        raise SplitError(
            f"windows ({len(windows)}) and labels ({len(labels)}) are not aligned"
        )
    if not windows:
        raise SplitError("cannot split an empty window set")

    ratios = dict(ratios or {"train": 0.6, "validation": 0.2, "test": 0.2})
    _validate_ratios(ratios)

    if strategy == "time":
        assignment = _time_split(windows, ratios, seed)
    else:
        assignment = _group_split(windows, labels, ratios, seed)

    _check_integrity(assignment, windows, labels, strategy)
    return assignment


def _validate_ratios(ratios: Mapping[str, float]) -> None:
    missing = [name for name in SPLIT_NAMES if name not in ratios]
    if missing:
        raise SplitError(f"split ratios are missing {missing}")
    if any(float(ratios[name]) <= 0 for name in SPLIT_NAMES):
        raise SplitError("every split ratio must be > 0")
    total = sum(float(ratios[name]) for name in SPLIT_NAMES)
    if abs(total - 1.0) > 1e-9:
        raise SplitError(f"split ratios must sum to 1.0, got {total}")


def _time_split(
    windows: Sequence[NetworkWindow],
    ratios: Mapping[str, float],
    seed: int,
) -> SplitAssignment:
    timestamps = sorted({window.metadata.window_start for window in windows})
    if len(timestamps) < 3:
        raise SplitError(
            "the time strategy needs at least three distinct window timestamps"
        )

    train_end, validation_end = _cut_points(len(timestamps), ratios)
    train_times = set(timestamps[:train_end])
    validation_times = set(timestamps[train_end:validation_end])
    test_times = set(timestamps[validation_end:])

    indices: Dict[str, List[int]] = {name: [] for name in SPLIT_NAMES}
    for index, window in enumerate(windows):
        start = window.metadata.window_start
        if start in train_times:
            indices["train"].append(index)
        elif start in validation_times:
            indices["validation"].append(index)
        else:
            indices["test"].append(index)

    runs = {name: [] for name in SPLIT_NAMES}
    return SplitAssignment(
        strategy="time",
        seed=seed,
        indices=indices,
        runs=runs,
        integrity={
            "rule": "unique window timestamps, chronological cut",
            "n_unique_timestamps": len(timestamps),
            "train_latest": timestamps[train_end - 1].isoformat(),
            "validation_earliest": timestamps[train_end].isoformat(),
            "validation_latest": timestamps[validation_end - 1].isoformat(),
            "test_earliest": timestamps[validation_end].isoformat(),
        },
    )


def _group_split(
    windows: Sequence[NetworkWindow],
    labels: Sequence[NetworkLabel],
    ratios: Mapping[str, float],
    seed: int,
) -> SplitAssignment:
    run_ids = sorted({label.run_id or "unassigned" for label in labels})
    if len(run_ids) < 3:
        raise SplitError(
            "the group strategy needs at least three runs (train/validation/test)"
        )

    shuffled = list(run_ids)
    random.Random(seed).shuffle(shuffled)
    train_end, validation_end = _cut_points(len(shuffled), ratios)
    run_sets = {
        "train": set(shuffled[:train_end]),
        "validation": set(shuffled[train_end:validation_end]),
        "test": set(shuffled[validation_end:]),
    }

    indices: Dict[str, List[int]] = {name: [] for name in SPLIT_NAMES}
    for index, label in enumerate(labels):
        run_id = label.run_id or "unassigned"
        for name in SPLIT_NAMES:
            if run_id in run_sets[name]:
                indices[name].append(index)
                break
        else:  # pragma: no cover - every run is assigned by construction
            raise SplitError(f"run {run_id!r} was not assigned to a split")

    return SplitAssignment(
        strategy="group",
        seed=seed,
        indices=indices,
        runs={name: sorted(run_sets[name]) for name in SPLIT_NAMES},
        integrity={
            "rule": "whole runs assigned to exactly one split",
            "n_runs": len(run_ids),
        },
    )


def _cut_points(total: int, ratios: Mapping[str, float]) -> Tuple[int, int]:
    """Return ``(train_end, validation_end)`` ensuring every split is non-empty."""
    train_end = int(round(total * float(ratios["train"])))
    validation_end = train_end + int(round(total * float(ratios["validation"])))
    train_end = max(1, min(train_end, total - 2))
    validation_end = max(train_end + 1, min(validation_end, total - 1))
    return train_end, validation_end


def _check_integrity(
    assignment: SplitAssignment,
    windows: Sequence[NetworkWindow],
    labels: Sequence[NetworkLabel],
    strategy: str,
) -> None:
    all_indices: List[int] = []
    for name in SPLIT_NAMES:
        if not assignment.indices[name]:
            raise SplitError(f"split {name!r} is empty; adjust the ratios or dataset size")
        all_indices.extend(assignment.indices[name])

    if len(all_indices) != len(windows) or sorted(all_indices) != list(range(len(windows))):
        raise SplitError("split indices do not cover every window exactly once")

    if strategy == "time":
        train_latest = max(windows[i].metadata.window_start for i in assignment.indices["train"])
        val_earliest = min(windows[i].metadata.window_start for i in assignment.indices["validation"])
        val_latest = max(windows[i].metadata.window_start for i in assignment.indices["validation"])
        test_earliest = min(windows[i].metadata.window_start for i in assignment.indices["test"])
        if not (train_latest < val_earliest and val_latest < test_earliest):
            raise SplitError("time split is not strictly chronological")
        assignment.integrity["chronological"] = True
        assignment.integrity["no_timestamp_overlap"] = True
    else:
        run_sets = {
            name: {labels[i].run_id or "unassigned" for i in assignment.indices[name]}
            for name in SPLIT_NAMES
        }
        overlaps = (
            run_sets["train"] & run_sets["validation"]
            | run_sets["train"] & run_sets["test"]
            | run_sets["validation"] & run_sets["test"]
        )
        if overlaps:
            raise SplitError(f"group split leaks runs across splits: {sorted(overlaps)}")
        assignment.integrity["group_disjoint"] = True
        assignment.integrity["runs"] = {name: sorted(run_sets[name]) for name in SPLIT_NAMES}

    assignment.integrity["counts"] = {
        name: len(assignment.indices[name]) for name in SPLIT_NAMES
    }
