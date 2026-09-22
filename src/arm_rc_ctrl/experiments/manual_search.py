# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MS-001: the frozen nominal-only ESN search protocol for the manual demonstrations.

The experiment plan (``docs/experiments/task_1a_manual_esn_search/plan.md``) is
the approved protocol; this module is the machine-readable form of it. A
protocol file (``configs/studies/manual_esn_search_*.toml``) binds the searched
ESN ranges and the warm-up set, the conditions that are deliberately *not*
searched (reservoir seed, both trackers, the inherited v4 filter policy), the
nominal-only objective, the rule that selects three configurations before any
perturbed case is evaluated, the approved budgets, and the scope of the final
five-arm comparison.

Two properties are enforced here rather than left to the execution code, because
they are what the design rests on: the optimizer's scenario list is exactly the
nominal case, and the approved caps can be tightened but never exceeded. The
search space is a closed schema, so a protocol that tries to tune a tracker, a
filter cutoff or the reservoir seed is refused rather than quietly accepted.

Nothing here fits a model, simulates a run, or reads the external store.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from arm_rc_ctrl.config import ConfigError, load_config
from arm_rc_ctrl.experiments.closed_loop import NominalConfig
from arm_rc_ctrl.experiments.esn_search import FloatRange, IntRange
from arm_rc_ctrl.experiments.studies import SamplerSpec
from arm_rc_ctrl.provenance import config_digest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "APPROVED_GIB",
    "APPROVED_HOURS",
    "APPROVED_TRIALS",
    "NOMINAL_SCENARIOS",
    "SELECTION_LABEL",
    "ManualComparisonScope",
    "ManualFixedConditions",
    "ManualSearchBudget",
    "ManualSearchObjective",
    "ManualSearchProtocol",
    "ManualSearchSpace",
    "ManualSelection",
    "ManualSelectionRule",
    "ScoredTrial",
    "filter_mismatches",
    "load_manual_search",
    "nominal_success_fraction",
    "protocol_digest",
    "scope_mismatches",
    "select_configurations",
]

NOMINAL_SCENARIOS: Final = ("nominal",)
"""The only scenarios the optimizer may see; every perturbed case is withheld from it."""
APPROVED_TRIALS: Final = 100
"""Owner-approved trial cap, including queued anchors, pilot candidates and failures."""
APPROVED_HOURS: Final = 10.0
"""Owner-approved elapsed-execution cap, applied as a shared ceiling with the comparison."""
APPROVED_GIB: Final = 20.0
"""Owner-approved artifact cap, applied as a shared ceiling with the comparison."""
SELECTION_LABEL: Final = "highest nominal scores"
"""How the frozen configurations are named: a coarse two-run score does not rank quality."""
_ARMS: Final = ("S", "M10", "R10", "C10", "replay")
_LEARNED_MODELS: Final = 31
"""Ten singletons, one all-ten model, ten copies and ten synthetic arms per configuration."""


@dataclass(frozen=True)
class ManualSearchSpace:
    """Every parameter this search tunes, and nothing else.

    A closed schema is the point: the loader refuses a protocol that adds a
    tracker, a filter cutoff or the reservoir seed here, because those are the
    conditions the experiment holds fixed to isolate ESN tuning.
    """

    n_neurons: IntRange
    spectral_radius: FloatRange
    sparsity: FloatRange
    leak_rate: FloatRange
    input_scaling: FloatRange
    alpha_0: FloatRange
    warmup_s: tuple[float, ...]
    """Categorical warm-up choices in seconds; zero means no warm-up phase."""

    def __post_init__(self) -> None:
        """The warm-up set is a sorted, distinct, finite set that offers zero."""
        if len(set(self.warmup_s)) != len(self.warmup_s):
            msg = f"warmup_s must be distinct, got {self.warmup_s}"
            raise ValueError(msg)
        if list(self.warmup_s) != sorted(self.warmup_s):
            msg = f"warmup_s must be ordered, got {self.warmup_s}"
            raise ValueError(msg)
        if not all(math.isfinite(value) and value >= 0.0 for value in self.warmup_s):
            msg = f"warmup_s must be finite and non-negative, got {self.warmup_s}"
            raise ValueError(msg)
        if 0.0 not in self.warmup_s:
            msg = "warmup_s must offer zero, which the owner approved as a searched choice"
            raise ValueError(msg)


@dataclass(frozen=True)
class ManualFixedConditions:
    """What the search inherits and never varies.

    The filter cutoffs are recorded beside the configuration they come from so
    ``filter_mismatches`` can compare them with their source; a restated value
    that drifts from the policy it claims to inherit is the failure this guards.
    """

    reservoir_seed: int
    trackers: tuple[str, ...]
    filters: Path
    velocity_cutoff_hz: float
    acceleration_cutoff_hz: float
    max_dt_ratio: float

    def __post_init__(self) -> None:
        """Validate the fixed conditions that do not depend on another file."""
        if self.reservoir_seed < 0:
            msg = f"reservoir_seed must be non-negative, got {self.reservoir_seed}"
            raise ValueError(msg)
        if len(set(self.trackers)) != len(self.trackers) or not self.trackers:
            msg = f"trackers must be a non-empty distinct list, got {self.trackers}"
            raise ValueError(msg)
        values = (self.velocity_cutoff_hz, self.acceleration_cutoff_hz, self.max_dt_ratio)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            msg = f"the filter policy must be finite and positive, got {values}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ManualSearchObjective:
    """The nominal success fraction, maximized, over the fixed trackers."""

    scenarios: tuple[str, ...] = NOMINAL_SCENARIOS
    runs_per_candidate: int = 2
    kind: Literal["nominal_success_fraction"] = "nominal_success_fraction"
    direction: Literal["maximize"] = "maximize"

    def __post_init__(self) -> None:
        """The optimizer's scenarios are exactly the nominal case."""
        if self.scenarios != NOMINAL_SCENARIOS:
            msg = (
                f"the optimizer's scenarios must be exactly {NOMINAL_SCENARIOS}, got {self.scenarios}: "
                "no perturbed scenario may reach the objective, its tie-break or its pruning"
            )
            raise ValueError(msg)
        if self.runs_per_candidate < 1:
            msg = f"runs_per_candidate must be >= 1, got {self.runs_per_candidate}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ManualSelectionRule:
    """How configurations are frozen, decided before any of them is evaluated."""

    n_configurations: int = 3
    order: Literal["descending_score_then_ascending_trial"] = "descending_score_then_ascending_trial"
    representative: Literal["earliest_trial"] = "earliest_trial"
    label: str = SELECTION_LABEL

    def __post_init__(self) -> None:
        """The count is positive and the label does not claim a ranking the score cannot support."""
        if self.n_configurations < 1:
            msg = f"n_configurations must be >= 1, got {self.n_configurations}"
            raise ValueError(msg)
        if self.label != SELECTION_LABEL:
            msg = f"the selection label is fixed at {SELECTION_LABEL!r}, got {self.label!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ManualSearchBudget:
    """The approved caps. A protocol may tighten them; it may never exceed them."""

    trials: int = APPROVED_TRIALS
    hours: float = APPROVED_HOURS
    gib: float = APPROVED_GIB
    shared_with_comparison: bool = True
    parallel_trials: int = 1

    def __post_init__(self) -> None:
        """Refuse a protocol that grants itself more than the owner approved."""
        for name, value, approved in (
            ("trials", self.trials, APPROVED_TRIALS),
            ("hours", self.hours, APPROVED_HOURS),
            ("gib", self.gib, APPROVED_GIB),
        ):
            if value <= 0:
                msg = f"{name} must be positive, got {value}"
                raise ValueError(msg)
            if value > approved:
                msg = f"{name} is {value}, above the approved {approved}; the cap is not enlarged here"
                raise ValueError(msg)
        if self.parallel_trials < 1:
            msg = f"parallel_trials must be >= 1, got {self.parallel_trials}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ManualComparisonScope:
    """The five-arm comparison that follows the freeze, sequential by configuration."""

    evaluation: Path
    scenarios: int
    arms: tuple[str, ...] = _ARMS
    models_per_configuration: int = _LEARNED_MODELS
    replay_banks_per_configuration: int = 10
    sequential: bool = True

    def __post_init__(self) -> None:
        """The arms and their counts are the closed experiment's, and the order is sequential."""
        if self.arms != _ARMS:
            msg = f"the comparison arms are fixed at {_ARMS}, got {self.arms}"
            raise ValueError(msg)
        if self.models_per_configuration != _LEARNED_MODELS:
            msg = f"a configuration carries {_LEARNED_MODELS} learned models, got {self.models_per_configuration}"
            raise ValueError(msg)
        if self.scenarios < 1 or self.replay_banks_per_configuration < 1:
            msg = "the comparison needs at least one scenario and one replay bank"
            raise ValueError(msg)
        if not self.sequential:
            msg = "configurations are evaluated sequentially in selection order (plan section 4)"
            raise ValueError(msg)


@dataclass(frozen=True)
class ManualSearchProtocol:
    """The frozen search: what is tuned, what is fixed, how it is scored and what follows."""

    name: str
    study: Path
    """The closed study manifest whose demonstrations, recipes and banks this search inherits."""
    model: Path
    scenario: Path
    sampler: SamplerSpec
    fixed: ManualFixedConditions
    space: ManualSearchSpace
    comparison: ManualComparisonScope
    objective: ManualSearchObjective = field(default_factory=ManualSearchObjective)
    selection: ManualSelectionRule = field(default_factory=ManualSelectionRule)
    budget: ManualSearchBudget = field(default_factory=ManualSearchBudget)

    def __post_init__(self) -> None:
        """Cross-field agreement that neither part can check alone."""
        if self.objective.runs_per_candidate != len(self.fixed.trackers):
            msg = (
                f"the objective scores {self.objective.runs_per_candidate} runs per candidate, but the protocol "
                f"fixes {len(self.fixed.trackers)} trackers"
            )
            raise ValueError(msg)
        if not self.name.strip():
            msg = "the protocol needs a name"
            raise ValueError(msg)


def load_manual_search(path: Path) -> ManualSearchProtocol:
    """Load and validate a search protocol, refusing unknown keys and widened scope."""
    return load_config(path, ManualSearchProtocol)


def protocol_digest(protocol: ManualSearchProtocol) -> str:
    """SHA-256 of the canonical, machine-independent protocol: this search's identity."""
    return config_digest(replace(protocol, study=Path(protocol.study.name)))[1]


def filter_mismatches(protocol: ManualSearchProtocol) -> list[str]:
    """Every fixed filter value that is not the one its own source configuration records.

    The protocol names the policy it inherits and repeats its three values so a
    reader sees them; this compares those values with the file through the
    evaluation configuration's own loader rather than trusting the repetition
    or restating that file's schema here.
    """
    try:
        source = load_config(protocol.fixed.filters, NominalConfig)
    except (ConfigError, OSError) as error:
        return [f"{protocol.fixed.filters.name}: cannot be read: {type(error).__name__}: {str(error)[:200]}"]
    pairs = (
        ("velocity_cutoff_hz", protocol.fixed.velocity_cutoff_hz, source.estimator.velocity_cutoff_hz),
        ("acceleration_cutoff_hz", protocol.fixed.acceleration_cutoff_hz, source.estimator.acceleration_cutoff_hz),
        ("max_dt_ratio", protocol.fixed.max_dt_ratio, source.estimator.max_dt_ratio),
    )
    return [
        f"{name}: the protocol records {recorded!r}, {protocol.fixed.filters.name} has {actual!r}"
        for name, recorded, actual in pairs
        if recorded != actual
    ]


def scope_mismatches(protocol: ManualSearchProtocol) -> list[str]:
    """Every way the protocol's parts disagree about what may be evaluated or scored."""
    failures: list[str] = []
    if protocol.objective.scenarios != NOMINAL_SCENARIOS:
        failures.append(f"the objective may only see {NOMINAL_SCENARIOS}, not {protocol.objective.scenarios}")
    if protocol.comparison.scenarios <= len(NOMINAL_SCENARIOS):
        failures.append("the final comparison must cover more than the optimizer's nominal case")
    if protocol.budget.trials * protocol.objective.runs_per_candidate > protocol.budget.trials * 2:
        failures.append("a candidate is scored on the two fixed trackers alone")
    return failures


def nominal_success_fraction(successes: int, *, runs: int) -> float:
    """The objective: successful nominal tracker runs over the runs that were judged.

    With two trackers this is 0, 0.5 or 1. It is deliberately coarse, and no
    settling time, tracking error or perturbed result may refine it.
    """
    if runs < 1:
        msg = f"a candidate is scored over at least one run, got runs={runs}"
        raise ValueError(msg)
    if not 0 <= successes <= runs:
        msg = f"successes must lie in [0, {runs}], got {successes}"
        raise ValueError(msg)
    return successes / runs


@dataclass(frozen=True)
class ScoredTrial:
    """One search trial as the selection rule sees it."""

    number: int
    score: float | None
    """``None`` for a fit failure or an infrastructure interruption: retained, never selected."""
    point: Mapping[str, float]
    """The parameter point, which decides whether two trials are the same configuration."""

    @property
    def identity(self) -> tuple[tuple[str, float], ...]:
        """The parameter point as a comparable, order-independent key."""
        return tuple(sorted((str(name), float(value)) for name, value in self.point.items()))


@dataclass(frozen=True)
class ManualSelection:
    """The frozen configurations, and how far the search fell short of the asked-for count."""

    chosen: tuple[ScoredTrial, ...]
    shortfall: int
    label: str = SELECTION_LABEL


def select_configurations(trials: Sequence[ScoredTrial], *, n_configurations: int = 3) -> ManualSelection:
    """Freeze configurations by descending nominal score, then earliest trial number.

    A repeated parameter point is one configuration represented by its earliest
    trial, and a candidate without a score is never selected. Fewer than the
    asked-for count is reported as a shortfall rather than raised or filled from
    somewhere else: the cap does not grow because the search was unlucky.
    """
    if n_configurations < 1:
        msg = f"n_configurations must be >= 1, got {n_configurations}"
        raise ValueError(msg)
    best: dict[tuple[tuple[str, float], ...], ScoredTrial] = {}
    for trial in sorted((t for t in trials if t.score is not None), key=lambda t: t.number):
        best.setdefault(trial.identity, trial)
    ordered = sorted(best.values(), key=lambda t: (-(t.score or 0.0), t.number))
    chosen = tuple(ordered[:n_configurations])
    return ManualSelection(chosen=chosen, shortfall=n_configurations - len(chosen))
