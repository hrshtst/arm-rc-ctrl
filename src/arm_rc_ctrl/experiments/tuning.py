# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Equal-budget tuning of the direct-replay tracker gains (``docs/PLAN.md`` sections 6 and 10).

The protocol is a versioned TOML (``configs/studies/*.toml``) fixing the
search space per tracker, the objective, the sampler seed, the trial budget,
the infeasibility penalty, and the development scenarios. A study samples
gains with a seeded generator (log-uniform per joint), simulates every
development scenario headlessly, and scores the trial by the median
movement-window joint RMSE; trials that terminate early, violate limits, or
fail the scenario's versioned dwell criteria (in-tolerance fraction and
stationarity over the dwell window) receive the documented penalty. Every
objective component is kept, and the run is deterministic for a given seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.controllers.reference import DemonstrationReference
from arm_rc_ctrl.controllers.tracking import TrackerConfig, TrackerType
from arm_rc_ctrl.data.phases import intervals_from_phases
from arm_rc_ctrl.data.samples import SampleSet
from arm_rc_ctrl.experiments.replay import dwell_outcome, simulate_tracking
from arm_rc_ctrl.metrics.joint import JointAnglePolicy, joint_rmse
from arm_rc_ctrl.scenario import ScenarioConfig, load_scenario
from arm_rc_ctrl.validation import require_finite

__all__ = [
    "DevelopmentScenarios",
    "GainRange",
    "Objective",
    "SearchSpaces",
    "StudyResult",
    "Trial",
    "TrialScenario",
    "TuningProtocol",
    "evaluate_gains",
    "load_protocol",
    "run_study",
    "sample_gains",
]


@dataclass(frozen=True)
class GainRange:
    """Per-joint sampling range of one gain."""

    low: float
    high: float
    log: bool = True

    def __post_init__(self) -> None:
        """Validate the bounds."""
        require_finite((self.low, self.high), "gain range")
        if not 0 < self.low < self.high:
            msg = f"gain range must satisfy 0 < low < high, got {self.low}, {self.high}"
            raise ValueError(msg)


@dataclass(frozen=True)
class TrackerSearch:
    """Search space of one tracker."""

    kp: GainRange
    kd: GainRange


@dataclass(frozen=True)
class SearchSpaces:
    """Search spaces per tracker type."""

    pd: TrackerSearch
    computed_torque: TrackerSearch

    def for_type(self, tracker_type: TrackerType) -> TrackerSearch:
        """Search space of the given tracker."""
        return self.pd if tracker_type == "pd" else self.computed_torque


@dataclass(frozen=True)
class Objective:
    """Scalar objective definition and infeasibility penalty."""

    kind: Literal["median_move_joint_rmse"]
    infeasible_penalty: float

    def __post_init__(self) -> None:
        """Require a positive finite penalty."""
        if not (self.infeasible_penalty > 0 and math.isfinite(self.infeasible_penalty)):
            msg = f"infeasible_penalty must be positive and finite, got {self.infeasible_penalty!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class DevelopmentScenarios:
    """Development-only scenario perturbations."""

    initial_posture_offsets: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        """Require at least one offset with finite entries."""
        if not self.initial_posture_offsets:
            msg = "development.initial_posture_offsets must not be empty"
            raise ValueError(msg)
        for i, offset in enumerate(self.initial_posture_offsets):
            require_finite(offset, f"development.initial_posture_offsets[{i}]")


@dataclass(frozen=True)
class TuningProtocol:
    """Complete tuning protocol (``configs/studies/*.toml``)."""

    name: str
    scenario: Path
    budget: int
    sampler_seed: int
    objective: Objective
    search: SearchSpaces
    development: DevelopmentScenarios

    def __post_init__(self) -> None:
        """Validate name, budget, and seed."""
        if not self.name.strip():
            msg = "name must not be empty"
            raise ValueError(msg)
        if self.budget < 1:
            msg = f"budget must be >= 1, got {self.budget}"
            raise ValueError(msg)
        if self.sampler_seed < 0:
            msg = f"sampler_seed must be non-negative, got {self.sampler_seed}"
            raise ValueError(msg)


def load_protocol(path: Path) -> TuningProtocol:
    """Load and validate a tuning protocol."""
    return load_config(path, TuningProtocol)


@dataclass(frozen=True)
class TrialScenario:
    """Objective components of one development scenario."""

    index: int
    initial_q: tuple[float, ...]
    termination: str
    move_joint_rmse: float | None
    criteria: dict[str, bool]
    """The scenario's success criteria (completion and the versioned dwell criteria)."""
    feasible: bool


@dataclass(frozen=True)
class Trial:
    """One sampled gain set and its evaluation."""

    number: int
    gains: TrackerConfig
    objective: float
    feasible: bool
    scenarios: tuple[TrialScenario, ...]


@dataclass(frozen=True)
class StudyResult:
    """Every trial of a study plus the selected one."""

    protocol: str
    tracker_type: TrackerType
    budget: int
    sampler_seed: int
    objective: str
    trials: tuple[Trial, ...]
    best: Trial
    feasible_trials: int = field(default=0)

    def __post_init__(self) -> None:
        """Consistency: budget equals the trial count, best is the minimum objective."""
        if len(self.trials) != self.budget:
            msg = f"study has {len(self.trials)} trials but budget {self.budget}"
            raise ValueError(msg)
        if self.best != min(self.trials, key=lambda t: (t.objective, t.number)):
            msg = "best must be the trial with the lowest objective (ties by number)"
            raise ValueError(msg)


def sample_gains(
    protocol: TuningProtocol, tracker_type: TrackerType, dof: int, rng: np.random.Generator
) -> TrackerConfig:
    """Draw per-joint gains from the tracker's search space (log-uniform when ``log``)."""
    space = protocol.search.for_type(tracker_type)

    def draw(gain: GainRange) -> tuple[float, ...]:
        if gain.log:
            values = np.exp(rng.uniform(math.log(gain.low), math.log(gain.high), size=dof))
        else:
            values = rng.uniform(gain.low, gain.high, size=dof)
        return tuple(float(v) for v in values)

    return TrackerConfig(type=tracker_type, kp=draw(space.kp), kd=draw(space.kd))


def evaluate_gains(
    protocol: TuningProtocol,
    scenario: ScenarioConfig,
    reference: SampleSet,
    gains: TrackerConfig,
) -> tuple[float, bool, tuple[TrialScenario, ...]]:
    """Simulate every development scenario and return ``(objective, feasible, components)``."""
    demo = DemonstrationReference.from_samples(reference)
    intervals = intervals_from_phases(reference.t, reference.phase)
    move = (reference.t >= intervals.move[0]) & (reference.t < intervals.move[1])
    policy = JointAnglePolicy.limited(scenario.dof)
    duration = float(reference.t[-1])
    components: list[TrialScenario] = []
    for index, offset in enumerate(protocol.development.initial_posture_offsets):
        if len(offset) != scenario.dof:
            msg = f"development offset {index} has {len(offset)} entries, expected {scenario.dof}"
            raise ValueError(msg)
        initial_q = tuple(float(a + b) for a, b in zip(scenario.task.initial_q, offset, strict=True))
        arrays, termination = simulate_tracking(scenario, demo, gains, duration_s=duration, initial_q=initial_q)
        q = cast("NDArray[np.float64]", arrays.arrays["q"])
        n = min(q.shape[0], reference.n_samples)
        rmse: float | None = None
        if termination.is_completed and move[:n].any():
            rmse = joint_rmse(q[:n][move[:n]], reference.q[:n][move[:n]], policy).aggregate
        criteria = dwell_outcome(scenario, reference, arrays, termination)
        feasible = rmse is not None and all(criteria.values())
        components.append(TrialScenario(index, initial_q, termination.kind, rmse, criteria, feasible))
    feasible_all = all(c.feasible for c in components)
    if feasible_all:
        objective = float(np.median([cast("float", c.move_joint_rmse) for c in components]))
    else:
        objective = protocol.objective.infeasible_penalty
    return objective, feasible_all, tuple(components)


def run_study(
    protocol: TuningProtocol,
    reference: SampleSet,
    tracker_type: TrackerType,
    *,
    scenario: ScenarioConfig | None = None,
) -> StudyResult:
    """Run the full seeded search for one tracker type (same seed and budget for every tracker)."""
    scenario = load_scenario(protocol.scenario) if scenario is None else scenario
    rng = np.random.default_rng(protocol.sampler_seed)
    trials: list[Trial] = []
    for number in range(protocol.budget):
        gains = sample_gains(protocol, tracker_type, scenario.dof, rng)
        objective, feasible, components = evaluate_gains(protocol, scenario, reference, gains)
        trials.append(Trial(number, gains, objective, feasible, components))
    best = min(trials, key=lambda t: (t.objective, t.number))
    return StudyResult(
        protocol=protocol.name,
        tracker_type=tracker_type,
        budget=protocol.budget,
        sampler_seed=protocol.sampler_seed,
        objective=protocol.objective.kind,
        trials=tuple(trials),
        best=best,
        feasible_trials=sum(1 for t in trials if t.feasible),
    )
