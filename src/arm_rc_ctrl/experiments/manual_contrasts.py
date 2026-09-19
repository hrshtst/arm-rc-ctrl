# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-010: the paired comparisons and class summaries of the manual study (plan section 6).

Every comparison is made within one configuration and one tracker, over the
scenarios of one perturbation class, and never pooled across them. Two measures
are reported side by side, as the owner decided on 2026-09-19: the difference
in success counts over the scenarios both arms were run on, with that
denominator stated, and the per-scenario tally of improved, worsened and tied
cases. Across the ten parents, the ten differences are listed in parent order
with their median and range and the number of parents improved, worsened or
tied. The all-ten arm is one model, so it is counted once in any arm total,
however many comparisons reuse it.

A run that does not exist is unavailable, never a failure: a comparison that is
missing runs shrinks to the scenarios both arms share and says so, and a
comparison with nothing shared says it is unavailable rather than reporting a
tie. These are descriptive paired counts. Copies, synthetic episodes and
overlapping training sets are not independent demonstrations, so no confidence
interval is attached to them.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from arm_rc_ctrl.experiments.manual_recipes import ASSIGNMENTS
from arm_rc_ctrl.experiments.perturbations import CLASS_ORDER

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "ALL_CLASSES",
    "ARM_KINDS",
    "CONTRASTS",
    "CONTRAST_STATUSES",
    "PARENTS",
    "ManualArmSummary",
    "ManualContrastRow",
    "ManualContrastSummary",
    "ScenarioVerdict",
    "arm_label",
    "arm_summaries",
    "contrast_rows",
    "contrast_summaries",
]

PARENTS: Final = ASSIGNMENTS
"""The ten bank positions, in the order every per-parent list follows."""
ALL_CLASSES: Final = "all"
"""The class label of a comparison over every scenario of the locked set together."""
ARM_KINDS: Final = ("S", "M10", "R10", "C10", "replay")
"""The four training arms and the direct replay of a demonstration."""
CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("M10-S", "M10", "S"),
    ("M10-R10", "M10", "R10"),
    ("C10-R10", "C10", "R10"),
    ("M10-C10", "M10", "C10"),
    ("S-replay", "S", "replay"),
    ("R10-replay", "R10", "replay"),
    ("C10-replay", "C10", "replay"),
    ("M10-replay", "M10", "replay"),
)
"""``(name, arm a, arm b)``: the plan's four arm contrasts, then every arm against its parent's replay.

Each is made per parent: arm ``a`` minus arm ``b``, where a parented arm is the
one of that parent and the all-ten arm is the same model for every parent.
"""
CONTRAST_STATUSES: Final = ("complete", "partial", "unavailable")
_ALL_TEN: Final = "M10"


def _median(values: Sequence[int]) -> float | None:
    """The median as a float, so an odd count and an even count serialise alike."""
    return float(statistics.median(values)) if values else None


def _classes() -> tuple[str, ...]:
    return (*CLASS_ORDER, ALL_CLASSES)


def arm_label(arm_kind: str, parent: str | None) -> str:
    """The arm as the study names it: ``M10``, ``S/D01``, or ``replay/D01`` for a parent's direct replay."""
    return arm_kind if parent is None else f"{arm_kind}/{parent}"


@dataclass(frozen=True)
class ScenarioVerdict:
    """Whether one arm succeeded on one scenario under one tracker, or that its run does not exist."""

    configuration: str
    tracker: str
    arm_kind: str
    parent: str | None
    """The demonstration this arm belongs to; the all-ten arm has none."""
    scenario_id: str
    scenario_class: str
    success: bool | None
    """``None`` when the run is unavailable: never executed, or its model has no evidence."""

    def __post_init__(self) -> None:
        """The all-ten arm has no parent; every other arm has exactly one of the ten."""
        if self.arm_kind not in ARM_KINDS:
            msg = f"unknown arm {self.arm_kind!r}; the study's arms are {ARM_KINDS}"
            raise ValueError(msg)
        if (self.arm_kind == _ALL_TEN) != (self.parent is None):
            msg = f"arm {self.arm_kind} with parent {self.parent!r}: only the all-ten arm has no parent"
            raise ValueError(msg)
        if self.parent is not None and self.parent not in PARENTS:
            msg = f"parent must be one of {PARENTS}, got {self.parent!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ManualContrastRow:
    """One paired comparison: arm a against arm b for one parent, over one class of scenarios."""

    configuration: str
    tracker: str
    scenario_class: str
    contrast: str
    parent: str
    arm_a: str
    arm_b: str
    status: str
    n_scenarios: int
    n_shared: int
    successes_a: int
    successes_b: int
    difference: int
    improved: int
    worsened: int
    tied_success: int
    tied_failure: int

    def __post_init__(self) -> None:
        """The two measures agree with each other and with their denominator."""
        if self.contrast not in {name for name, _, _ in CONTRASTS} or self.scenario_class not in _classes():
            msg = f"unknown contrast {self.contrast!r} or class {self.scenario_class!r}"
            raise ValueError(msg)
        expected = (
            "unavailable" if self.n_shared == 0 else "complete" if self.n_shared == self.n_scenarios else "partial"
        )
        if self.status != expected or not 0 <= self.n_shared <= self.n_scenarios:
            msg = f"{self.contrast} {self.parent}: status {self.status!r} with {self.n_shared} of {self.n_scenarios}"
            raise ValueError(msg)
        tallies = (self.improved, self.worsened, self.tied_success, self.tied_failure)
        if min(tallies) < 0 or sum(tallies) != self.n_shared:
            msg = f"{self.contrast} {self.parent}: the tallies {tallies} do not partition {self.n_shared} scenarios"
            raise ValueError(msg)
        derived = (self.improved + self.tied_success, self.worsened + self.tied_success)
        if (self.successes_a, self.successes_b) != derived or self.difference != self.successes_a - self.successes_b:
            msg = f"{self.contrast} {self.parent}: the success counts contradict the tallies"
            raise ValueError(msg)


@dataclass(frozen=True)
class ManualContrastSummary:
    """One contrast across the ten parents: every difference, their median and range, and their signs."""

    configuration: str
    tracker: str
    scenario_class: str
    contrast: str
    arm_a: str
    arm_b: str
    n_scenarios: int
    n_parents: int
    differences: tuple[int | None, ...]
    """One per parent in parent order; ``None`` where that parent's comparison is not complete."""
    median: float | None
    minimum: int | None
    maximum: int | None
    parents_improved: int
    parents_worsened: int
    parents_tied: int

    def __post_init__(self) -> None:
        """Every figure re-derives from the listed differences."""
        present = [d for d in self.differences if d is not None]
        if len(self.differences) != len(PARENTS) or self.n_parents != len(present):
            msg = f"{self.contrast}: {self.n_parents} parents stated, {len(present)} differences listed"
            raise ValueError(msg)
        derived = (
            _median(present),
            min(present, default=None),
            max(present, default=None),
            sum(1 for d in present if d > 0),
            sum(1 for d in present if d < 0),
            sum(1 for d in present if d == 0),
        )
        recorded = (
            self.median,
            self.minimum,
            self.maximum,
            self.parents_improved,
            self.parents_worsened,
            self.parents_tied,
        )
        if derived != recorded:
            msg = f"{self.contrast}: the summary {recorded} does not re-derive from its differences ({derived})"
            raise ValueError(msg)


@dataclass(frozen=True)
class ManualArmSummary:
    """One arm's successes over one class of scenarios, with the all-ten arm counted as the single model it is."""

    configuration: str
    tracker: str
    scenario_class: str
    arm_kind: str
    n_scenarios: int
    n_models: int
    n_runs: int
    excluded_runs: int
    unavailable_runs: int
    successes: int
    per_model: tuple[int | None, ...]
    """Successes of each model over the class, in parent order (one entry for the all-ten arm)."""
    median: float | None
    minimum: int | None
    maximum: int | None

    def __post_init__(self) -> None:
        """The totals re-derive from the per-model counts, and the all-ten arm is one model."""
        expected_models = 1 if self.arm_kind == _ALL_TEN else len(PARENTS)
        if self.arm_kind not in ARM_KINDS or len(self.per_model) != expected_models:
            msg = f"{self.arm_kind}: expected {expected_models} per-model counts, got {len(self.per_model)}"
            raise ValueError(msg)
        present = [count for count in self.per_model if count is not None]
        derived = (
            len(present),
            sum(present),
            _median(present),
            min(present, default=None),
            max(present, default=None),
        )
        recorded = (self.n_models, self.successes, self.median, self.minimum, self.maximum)
        accounted = self.n_runs + self.excluded_runs + self.unavailable_runs
        if (
            derived != recorded
            or self.n_runs != self.n_models * self.n_scenarios
            or accounted != expected_models * self.n_scenarios
        ):
            msg = f"{self.arm_kind} {self.scenario_class}: the totals do not re-derive from the per-model counts"
            raise ValueError(msg)


# --- indexing ------------------------------------------------------------------------------------

type _ArmKey = tuple[str, str, str, str | None]
"""(configuration, tracker, arm kind, parent)."""


def _index(
    verdicts: Iterable[ScenarioVerdict], scenarios: Sequence[tuple[str, str]]
) -> tuple[dict[_ArmKey, dict[str, bool | None]], list[tuple[str, str]]]:
    """Group verdicts by arm, refusing a repeat, a stray scenario, or a class the frozen order contradicts."""
    classes = dict(scenarios)
    if len(classes) != len(scenarios):
        msg = "the frozen scenario order names a scenario more than once"
        raise ValueError(msg)
    grouped: dict[_ArmKey, dict[str, bool | None]] = {}
    keys: list[tuple[str, str]] = []
    for verdict in verdicts:
        if verdict.scenario_id not in classes:
            msg = f"{verdict.scenario_id!r} is outside the frozen order of {len(classes)} scenarios"
            raise ValueError(msg)
        if classes[verdict.scenario_id] != verdict.scenario_class:
            msg = (
                f"{verdict.scenario_id} is recorded in class {verdict.scenario_class!r}, "
                f"but the frozen order puts it in {classes[verdict.scenario_id]!r}"
            )
            raise ValueError(msg)
        key = (verdict.configuration, verdict.tracker, verdict.arm_kind, verdict.parent)
        arm = grouped.setdefault(key, {})
        if verdict.scenario_id in arm:
            msg = f"{arm_label(verdict.arm_kind, verdict.parent)} on {verdict.scenario_id} is recorded more than once"
            raise ValueError(msg)
        arm[verdict.scenario_id] = verdict.success
        if (verdict.configuration, verdict.tracker) not in keys:
            keys.append((verdict.configuration, verdict.tracker))
    return grouped, keys


def _members(scenarios: Sequence[tuple[str, str]], scenario_class: str) -> tuple[str, ...]:
    return tuple(sid for sid, kind in scenarios if scenario_class in (ALL_CLASSES, kind))


def _present_classes(scenarios: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    """The classes that occur in the frozen order, in class order, then every scenario together."""
    kinds = {kind for _, kind in scenarios}
    return (*(kind for kind in CLASS_ORDER if kind in kinds), ALL_CLASSES)


# --- contrasts -----------------------------------------------------------------------------------


def _compare(a: dict[str, bool | None], b: dict[str, bool | None], members: Sequence[str]) -> tuple[int, ...]:
    """``(n_shared, improved, worsened, tied_success, tied_failure)`` over the scenarios both arms have."""
    improved = worsened = tied_success = tied_failure = shared = 0
    for sid in members:
        left, right = a.get(sid), b.get(sid)
        if left is None or right is None:
            continue
        shared += 1
        if left and not right:
            improved += 1
        elif right and not left:
            worsened += 1
        elif left:
            tied_success += 1
        else:
            tied_failure += 1
    return shared, improved, worsened, tied_success, tied_failure


def contrast_rows(
    verdicts: Iterable[ScenarioVerdict], *, scenarios: Sequence[tuple[str, str]]
) -> tuple[ManualContrastRow, ...]:
    """Every planned contrast for every parent, class, configuration and tracker the verdicts cover.

    ``scenarios`` is the frozen ``(scenario id, class)`` order; it fixes each
    class's denominator, so a scenario missing from the verdicts shrinks a
    comparison instead of disappearing from its denominator.
    """
    grouped, keys = _index(verdicts, scenarios)
    rows: list[ManualContrastRow] = []
    for configuration, tracker in keys:
        for scenario_class in _present_classes(scenarios):
            members = _members(scenarios, scenario_class)
            for name, kind_a, kind_b in CONTRASTS:
                for parent in PARENTS:
                    parent_a = None if kind_a == _ALL_TEN else parent
                    parent_b = parent  # arm b of every planned contrast has a parent
                    a = grouped.get((configuration, tracker, kind_a, parent_a), {})
                    b = grouped.get((configuration, tracker, kind_b, parent_b), {})
                    shared, improved, worsened, tied_success, tied_failure = _compare(a, b, members)
                    status = "unavailable" if shared == 0 else "complete" if shared == len(members) else "partial"
                    rows.append(
                        ManualContrastRow(
                            configuration=configuration,
                            tracker=tracker,
                            scenario_class=scenario_class,
                            contrast=name,
                            parent=parent,
                            arm_a=arm_label(kind_a, parent_a),
                            arm_b=arm_label(kind_b, parent_b),
                            status=status,
                            n_scenarios=len(members),
                            n_shared=shared,
                            successes_a=improved + tied_success,
                            successes_b=worsened + tied_success,
                            difference=improved - worsened,
                            improved=improved,
                            worsened=worsened,
                            tied_success=tied_success,
                            tied_failure=tied_failure,
                        )
                    )
    return tuple(rows)


def contrast_summaries(rows: Sequence[ManualContrastRow]) -> tuple[ManualContrastSummary, ...]:
    """Each contrast across the ten parents, over complete comparisons only, in the rows' own order."""
    groups: dict[tuple[str, str, str, str], dict[str, ManualContrastRow]] = {}
    for row in rows:
        group = groups.setdefault((row.configuration, row.tracker, row.scenario_class, row.contrast), {})
        if row.parent in group:
            msg = f"{row.contrast} {row.parent} appears more than once for one class, configuration and tracker"
            raise ValueError(msg)
        group[row.parent] = row
    kinds = {name: (a, b) for name, a, b in CONTRASTS}
    summaries: list[ManualContrastSummary] = []
    for (configuration, tracker, scenario_class, contrast), by_parent in groups.items():
        differences = tuple(
            by_parent[p].difference if p in by_parent and by_parent[p].status == "complete" else None for p in PARENTS
        )
        present = [d for d in differences if d is not None]
        n_scenarios = {r.n_scenarios for r in by_parent.values()}
        if len(n_scenarios) != 1:
            msg = f"{contrast} {scenario_class}: parents disagree on the class size {sorted(n_scenarios)}"
            raise ValueError(msg)
        kind_a, kind_b = kinds[contrast]
        summaries.append(
            ManualContrastSummary(
                configuration=configuration,
                tracker=tracker,
                scenario_class=scenario_class,
                contrast=contrast,
                arm_a=kind_a,
                arm_b=kind_b,
                n_scenarios=n_scenarios.pop(),
                n_parents=len(present),
                differences=differences,
                median=_median(present),
                minimum=min(present, default=None),
                maximum=max(present, default=None),
                parents_improved=sum(1 for d in present if d > 0),
                parents_worsened=sum(1 for d in present if d < 0),
                parents_tied=sum(1 for d in present if d == 0),
            )
        )
    return tuple(summaries)


# --- class summaries -----------------------------------------------------------------------------


def arm_summaries(
    verdicts: Iterable[ScenarioVerdict], *, scenarios: Sequence[tuple[str, str]]
) -> tuple[ManualArmSummary, ...]:
    """Each arm's successes per class, per configuration and tracker, counting only models complete over the class.

    A model with a missing run in a class is left out of that class's totals and
    its runs are counted as unavailable, so a success rate is never computed over
    runs that do not exist.
    """
    grouped, keys = _index(verdicts, scenarios)
    summaries: list[ManualArmSummary] = []
    for configuration, tracker in keys:
        for scenario_class in _present_classes(scenarios):
            members = _members(scenarios, scenario_class)
            for kind in ARM_KINDS:
                parents: tuple[str | None, ...] = (None,) if kind == _ALL_TEN else PARENTS
                per_model: list[int | None] = []
                unavailable = excluded = 0
                for parent in parents:
                    outcomes = [grouped.get((configuration, tracker, kind, parent), {}).get(sid) for sid in members]
                    missing = sum(1 for outcome in outcomes if outcome is None)
                    unavailable += missing
                    excluded += len(members) - missing if missing else 0
                    per_model.append(None if missing else sum(1 for outcome in outcomes if outcome))
                present = [count for count in per_model if count is not None]
                summaries.append(
                    ManualArmSummary(
                        configuration=configuration,
                        tracker=tracker,
                        scenario_class=scenario_class,
                        arm_kind=kind,
                        n_scenarios=len(members),
                        n_models=len(present),
                        n_runs=len(present) * len(members),
                        excluded_runs=excluded,
                        unavailable_runs=unavailable,
                        successes=sum(present),
                        per_model=tuple(per_model),
                        median=_median(present),
                        minimum=min(present, default=None),
                        maximum=max(present, default=None),
                    )
                )
    return tuple(summaries)
