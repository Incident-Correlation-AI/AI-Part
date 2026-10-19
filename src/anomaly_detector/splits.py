"""Reproducible dataset splits (Milestone 2).

Three strategies are supported:

* ``random`` — shuffle rows with a seeded RNG and slice by ratio.
* ``time``   — sort rows by a time column and slice chronologically
               (train = earliest, test = latest).
* ``group``  — assign whole groups (e.g. a host or scenario) to a single split
               so no group leaks across train/validation/test.

Every split is **reproducible**: the same configuration and seed always produce
identical assignments. Integrity checks (no row dropped, no row duplicated, no
group overlap, chronological ordering) run on every split and are recorded.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .readiness_config import SplitConfig


@dataclass
class SplitResult:
    strategy: str
    seed: int
    ratios: Dict[str, float]
    assignments: Dict[str, List[int]] = field(default_factory=dict)
    split_order: List[str] = field(default_factory=list)
    group_column: Optional[str] = None
    time_column: Optional[str] = None
    checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def size(self, name: str) -> int:
        return len(self.assignments.get(name, []))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "ratios": self.ratios,
            "split_order": self.split_order,
            "group_column": self.group_column,
            "time_column": self.time_column,
            "sizes": {n: self.size(n) for n in self.split_order},
            "assignments": {n: list(self.assignments.get(n, [])) for n in self.split_order},
            "checks": list(self.checks),
            "warnings": list(self.warnings),
        }

    def to_markdown(self) -> str:
        lines = ["# Dataset Split Report", ""]
        lines.append(f"- Strategy: {self.strategy}")
        lines.append(f"- Seed: {self.seed}")
        lines.append(f"- Ratios: {self.ratios}")
        if self.group_column:
            lines.append(f"- Group column: {self.group_column}")
        if self.time_column:
            lines.append(f"- Time column: {self.time_column}")
        lines.append("")
        lines.append("## Sizes")
        lines.append("")
        lines.append("| Split | Rows |")
        lines.append("| --- | --- |")
        for n in self.split_order:
            lines.append(f"| {n} | {self.size(n)} |")
        lines.append("")
        lines.append("## Integrity checks")
        lines.append("")
        for c in self.checks:
            lines.append(f"- PASS: {c}")
        for w in self.warnings:
            lines.append(f"- WARN: {w}")
        lines.append("")
        return "\n".join(lines)


def _split_order(ratios: Dict[str, float]) -> List[str]:
    return list(ratios.keys())


def _ratio_sizes(n: int, ratios: Dict[str, float], order: List[str]) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    remaining = n
    for i, name in enumerate(order):
        if i == len(order) - 1:
            sizes[name] = remaining
        else:
            sizes[name] = int(round(n * ratios[name]))
            remaining -= sizes[name]
    # Guard against negative rounding artefacts.
    for name in order:
        if sizes[name] < 0:
            sizes[name] = 0
    return sizes


def _check_coverage(result: SplitResult, n: int) -> None:
    seen = np.zeros(n, dtype=bool)
    for name in result.split_order:
        for idx in result.assignments[name]:
            if not (0 <= idx < n):
                raise ValueError(f"split {name!r} contains out-of-range index {idx}")
            if seen[idx]:
                raise ValueError(f"row index {idx} was assigned to more than one split")
            seen[idx] = True
    if not seen.all():
        missing = int((~seen).sum())
        raise ValueError(f"{missing} row(s) were not assigned to any split")


def _resolve_column(
    metadata: pd.DataFrame,
    labels: Optional[pd.DataFrame],
    column: str,
    role: str,
) -> pd.Series:
    """Find a column in metadata first, then in the (row-aligned) labels table.

    Group/time keys such as ``scenario_name`` live in the label table rather
    than in the feature metadata, so both tables are searched. The values are
    used only to partition rows, never as model inputs.
    """
    if column in metadata.columns:
        return metadata[column]
    if labels is not None and column in labels.columns and len(labels) == len(metadata):
        return labels[column].reset_index(drop=True)
    raise ValueError(
        f"{role} column {column!r} not found in metadata or aligned labels"
    )


def make_splits(
    metadata: pd.DataFrame,
    config: SplitConfig,
    labels: Optional[pd.DataFrame] = None,
) -> SplitResult:
    """Partition rows of ``metadata`` into reproducible splits.

    The group/time keys may live in either the metadata or the aligned labels
    table. When a key comes from the labels table (e.g. ``scenario_name``) it is
    used *only* to partition rows, never as a model feature, and the label
    column itself is never used to choose the split.
    """
    n = len(metadata)
    order = _split_order(config.ratios)
    ratios = {k: float(v) for k, v in config.ratios.items()}
    sizes = _ratio_sizes(n, ratios, order)

    result = SplitResult(
        strategy=config.strategy,
        seed=config.seed,
        ratios=ratios,
        split_order=order,
    )

    if n == 0:
        result.warnings.append("dataset is empty; no splits produced")
        result.assignments = {name: [] for name in order}
        return result

    if config.strategy == "random":
        rng = np.random.default_rng(config.seed)
        perm = rng.permutation(n)
        start = 0
        for name in order:
            size = sizes[name]
            result.assignments[name] = sorted(int(i) for i in perm[start:start + size])
            start += size
        result.checks.append("random split seeded; identical seed reproduces identical assignment")

    elif config.strategy == "time":
        col = config.time_column
        times = pd.to_datetime(_resolve_column(metadata, labels, col, "time"))
        ordered = list(times.sort_values(kind="mergesort").index)
        start = 0
        for name in order:
            size = sizes[name]
            result.assignments[name] = sorted(int(i) for i in ordered[start:start + size])
            start += size
        result.time_column = col

        # Chronological non-overlap check.
        maxima = {}
        minima = {}
        for name in order:
            idx = result.assignments[name]
            if idx:
                maxima[name] = times.iloc[idx].max()
                minima[name] = times.iloc[idx].min()
        for a, b in zip(order, order[1:]):
            if a in maxima and b in minima and maxima[a] > minima[b]:
                raise ValueError(
                    f"time split is not chronological: max({a})={maxima[a]} > min({b})={minima[b]}"
                )
        result.checks.append("time split is strictly chronological (train earlier than test)")

    elif config.strategy == "group":
        col = config.group_column
        result.group_column = col
        groups = _resolve_column(metadata, labels, col, "group").astype("string").fillna("<missing>")
        unique_groups = sorted(groups.unique().tolist())
        rng = np.random.default_rng(config.seed)
        rng.shuffle(unique_groups)  # seed-controlled tie-break for equal-sized groups
        size_map = groups.value_counts().to_dict()
        # Largest groups first (LPT-style) balances row counts across splits.
        unique_groups = sorted(unique_groups, key=lambda g: -size_map.get(g, 0))

        targets = {name: max(n * ratios[name], 1e-9) for name in order}
        assigned: Dict[str, int] = {name: 0 for name in order}
        buckets: Dict[str, List[int]] = {name: [] for name in order}

        for g in unique_groups:
            # Assign to the split with the largest *relative* remaining capacity.
            best = max(order, key=lambda name: (targets[name] - assigned[name]) / targets[name])
            idx = list(groups.index[groups == g])
            buckets[best].extend(int(i) for i in idx)
            assigned[best] += len(idx)

        for name in order:
            result.assignments[name] = sorted(buckets[name])

        # Group disjointness check.
        group_sets = {}
        for name in order:
            idx = result.assignments[name]
            group_sets[name] = set(groups.iloc[idx].tolist()) if idx else set()
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                overlap = group_sets[a] & group_sets[b]
                if overlap:
                    raise ValueError(f"group overlap between {a} and {b}: {sorted(overlap)[:5]}")
        covered = set().union(*group_sets.values()) if group_sets else set()
        if covered != set(unique_groups):
            raise ValueError("group-aware split did not cover all groups")
        result.checks.append(f"no {col!r} value appears in more than one split")
        result.checks.append(f"all {len(unique_groups)} group(s) assigned exactly once")

    else:
        raise ValueError(f"unknown split strategy {config.strategy!r}")

    _check_coverage(result, n)

    for name in order:
        if result.size(name) == 0:
            result.warnings.append(f"split {name!r} is empty; consider adjusting ratios")

    if labels is not None and config.label_column in labels.columns:
        for name in order:
            idx = result.assignments[name]
            if not idx:
                continue
            col = labels[config.label_column]
            positives = int((col.iloc[idx] == True).sum())   # noqa: E712
            negatives = int((col.iloc[idx] == False).sum())  # noqa: E712
            ratio = positives / len(idx)
            result.checks.append(f"{name}: {positives}/{len(idx)} positive ({ratio:.1%})")
            if (positives + negatives) > 0 and (positives == 0 or negatives == 0):
                result.warnings.append(
                    f"split {name!r} has a single label class "
                    f"(positives={positives}, negatives={negatives}); evaluation may be degenerate"
                )

    return result
