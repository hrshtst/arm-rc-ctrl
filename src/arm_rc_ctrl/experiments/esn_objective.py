# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Evaluating one ESN search point: feasibility rules and the documented penalty (``docs/PLAN.md`` section 10; M3-004).

A point is trained once in memory (the recipe is not written), wrapped as the
RC target generator with the point's estimator cutoffs, and run closed loop
with the protocol's frozen tracker through every development scenario. A
scenario is *feasible* when the run completes, meets every dwell criterion of
the scenario, and keeps actuator saturation within the protocol's bound; the
trial is feasible when every scenario is. The scalar objective is the median
movement-window joint RMSE over the scenarios of a feasible trial and the
protocol's ``infeasible_penalty`` otherwise (divergence, a state/torque/
endpoint limit violation, any other early termination, a missed dwell
criterion, excess saturation, or a training failure). Scenarios stop at the
first infeasible one — the objective is already decided — and every component
computed so far is kept with the trial, together with the reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import optuna

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.controllers.adapter import GeneratorTrackingController
from arm_rc_ctrl.controllers.estimator import CausalDerivativeEstimator
from arm_rc_ctrl.data.phases import intervals_from_phases
from arm_rc_ctrl.data.records import ProcessedDatasetRecord, load_record, verify_payload
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.experiments.baselines import load_frozen_baseline
from arm_rc_ctrl.experiments.esn_search import suggest_point
from arm_rc_ctrl.experiments.replay import bind_dataset, dwell_outcome
from arm_rc_ctrl.experiments.simulation import GENERATOR_CHANNELS, simulate
from arm_rc_ctrl.metrics.joint import JointAnglePolicy, joint_rmse
from arm_rc_ctrl.rc.generator import RcTargetGenerator
from arm_rc_ctrl.rc.recipe import DatasetSource, create_recipe
from arm_rc_ctrl.rc.teacher_forcing import InputTransform
from arm_rc_ctrl.scenario import load_scenario

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from numpy.typing import NDArray

    from arm_rc_ctrl.controllers.tracking import TrackerConfig
    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.experiments.esn_search import EsnSearchProtocol, TrialPoint
    from arm_rc_ctrl.experiments.run_record import RunArrays
    from arm_rc_ctrl.experiments.termination import Termination
    from arm_rc_ctrl.experiments.tuning import DevelopmentPulse
    from arm_rc_ctrl.rc.esn import EsnModel
    from arm_rc_ctrl.rc.recipe import ModelRecipe
    from arm_rc_ctrl.rc.train import ModelConfig
    from arm_rc_ctrl.scenario import ScenarioConfig
    from arm_rc_ctrl.storage import StorageRoot

__all__ = [
    "ScenarioComponent",
    "TrialContext",
    "TrialEvaluation",
    "classify",
    "development_cases",
    "evaluate_point",
    "make_objective",
    "train_point",
]

type ReportCallback = Callable[[int, float], bool]
"""Called after every scenario with ``(step, running_objective)``; returning ``True`` stops the evaluation."""


@dataclass(frozen=True)
class ScenarioComponent:
    """Objective components of one development scenario."""

    index: int
    kind: Literal["posture", "force"]
    initial_q: tuple[float, ...]
    termination: str
    move_joint_rmse: float | None
    criteria: dict[str, bool]
    feasible: bool
    reason: str | None
    """Why the scenario is infeasible (``None`` when feasible)."""
    failure: str | None = None
    limit: str | None = None
    saturation_fraction: float | None = None
    boundary_jump: float | None = None
    pulse: DevelopmentPulse | None = None


@dataclass(frozen=True)
class TrialEvaluation:
    """Everything one evaluated point yields; the scalar objective is never the only record."""

    point: TrialPoint
    objective: float
    feasible: bool
    penalized: bool
    reason: str | None
    fit_rmse: float | None
    scenarios_total: int
    components: tuple[ScenarioComponent, ...]
    running: tuple[float, ...]
    """The running objective after each evaluated scenario (what the pruner sees)."""
    stopped_early: bool = False
    """Whether a report callback stopped the evaluation before its natural end."""

    def attrs(self) -> dict[str, object]:
        """The evaluation as (nested) trial attributes; ``studies.summarize`` flattens them."""
        return {
            "feasible": self.feasible,
            "penalized": self.penalized,
            "reason": "" if self.reason is None else self.reason,
            "fit_rmse": self.fit_rmse,
            "scenarios_total": self.scenarios_total,
            "scenarios_evaluated": len(self.components),
            "stopped_early": self.stopped_early,
            "running": list(self.running),
            "components": [to_mapping(c) for c in self.components],
        }


@dataclass(frozen=True)
class TrialContext:
    """The loaded, bound inputs every trial of a study shares."""

    scenario: ScenarioConfig
    scenario_file: Path
    reference: SampleSet
    dataset: ProcessedDatasetRecord
    source: DatasetSource
    tracker: TrackerConfig
    base_model: ModelConfig

    @classmethod
    def load(
        cls, protocol: EsnSearchProtocol, *, store: StorageRoot, dataset_file: Path, records_root: Path
    ) -> TrialContext:
        """Load and bind the scenario, dataset, frozen tracker, and base model of ``protocol``."""
        scenario = load_scenario(protocol.scenario)
        dataset = load_record(dataset_file, ProcessedDatasetRecord)
        reference = load_samples(verify_payload(store, dataset.artifact))
        bind_dataset(scenario, protocol.scenario, dataset, reference)
        if dataset.normalization is None:
            msg = f"dataset {dataset.artifact.artifact_id} records no normalization statistics"
            raise ValueError(msg)
        relative = dataset_file.resolve().relative_to(records_root.resolve()).as_posix()
        source = DatasetSource(dataset.artifact.artifact_id, dataset.artifact.payload.sha256, relative)
        tracker = load_frozen_baseline(protocol.tracker)
        if tracker.dof != scenario.dof:
            msg = f"dof mismatch: scenario {scenario.dof}, tracker {tracker.dof}"
            raise ValueError(msg)
        return cls(scenario, protocol.scenario, reference, dataset, source, tracker, protocol.base_model())

    @property
    def samples(self) -> Mapping[str, SampleSet]:
        """Training samples keyed by dataset ID."""
        return {self.dataset.artifact.artifact_id: self.reference}


def development_cases(
    protocol: EsnSearchProtocol, scenario: ScenarioConfig
) -> list[tuple[tuple[float, ...], DevelopmentPulse | None]]:
    """The development scenarios as ``(initial_q, pulse)`` pairs: posture offsets first, then force pulses."""
    cases: list[tuple[tuple[float, ...], DevelopmentPulse | None]] = []
    for index, offset in enumerate(protocol.development.initial_posture_offsets):
        if len(offset) != scenario.dof:
            msg = f"development offset {index} has {len(offset)} entries, expected {scenario.dof}"
            raise ValueError(msg)
        cases.append((tuple(float(a + b) for a, b in zip(scenario.task.initial_q, offset, strict=True)), None))
    cases.extend((tuple(scenario.task.initial_q), pulse) for pulse in protocol.development.force_pulses)
    return cases


def classify(
    termination: Termination, criteria: Mapping[str, bool], saturation: float | None, bound: float
) -> str | None:
    """The infeasibility reason of a scenario, or ``None`` when it is feasible."""
    if termination.kind == "divergence":
        return "divergence"
    if termination.kind == "limit_violation":
        return f"limit_violation:{termination.limit}"
    if not termination.is_completed:
        failure = "" if termination.failure is None else f":{termination.failure}"
        return f"early_termination:{termination.kind}{failure}"
    missed = [name for name, ok in criteria.items() if not ok]
    if missed:
        return "dwell:" + ",".join(missed)
    if saturation is None or saturation > bound:
        return "saturation"
    return None


def train_point(
    protocol: EsnSearchProtocol, context: TrialContext, point: TrialPoint
) -> tuple[ModelRecipe, EsnModel] | str:
    """Train ``point`` in memory; return the recipe and model, or the infeasibility reason when training fails."""
    base = context.base_model
    normalization = context.dataset.normalization
    if normalization is None:  # pragma: no cover - TrialContext.load rejects such datasets
        return "training_failure:no_normalization"
    transform = InputTransform.derive(
        base.input_transform.policy, normalization, fixed_scales=base.input_transform.fixed_scales
    )
    config = point.model_config(base, name=f"{protocol.name}-trial")
    try:
        recipe, model = create_recipe(
            config.name,
            config.esn,
            sources=[context.source],
            samples=context.samples,
            dof=context.dataset.dof,
            task_code_dim=context.dataset.task_code_dim,
            preprocessing=context.dataset.preprocessing,
            transform=transform,
        )
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return f"training_failure:{type(exc).__name__}"
    if not math.isfinite(recipe.fit.rmse):
        return "training_failure:non_finite_fit"
    return recipe, model


def _component(
    index: int,
    initial_q: tuple[float, ...],
    pulse: DevelopmentPulse | None,
    arrays_and_termination: tuple[RunArrays, Termination],
    *,
    context: TrialContext,
    bound: float,
    boundary_jump: float | None,
) -> ScenarioComponent:
    arrays, termination = arrays_and_termination
    reference = context.reference
    intervals = intervals_from_phases(reference.t, reference.phase)
    move = (reference.t >= intervals.move[0]) & (reference.t < intervals.move[1])
    q = cast("NDArray[np.float64]", arrays.arrays["q"])
    n = min(q.shape[0], reference.n_samples)
    rmse: float | None = None
    if termination.is_completed and move[:n].any():
        policy = JointAnglePolicy.limited(context.scenario.dof)
        rmse = joint_rmse(q[:n][move[:n]], reference.q[:n][move[:n]], policy).aggregate
    criteria = dwell_outcome(context.scenario, reference, arrays, termination)
    saturation = float(np.mean(cast("NDArray[np.int64]", arrays.arrays["saturation"]))) if n else None
    reason = classify(termination, criteria, saturation, bound)
    if reason is None and rmse is None:
        reason = "no_movement_samples"
    return ScenarioComponent(
        index=index,
        kind="posture" if pulse is None else "force",
        initial_q=initial_q,
        termination=termination.kind,
        move_joint_rmse=rmse,
        criteria=criteria,
        feasible=reason is None,
        reason=reason,
        failure=termination.failure,
        limit=termination.limit,
        saturation_fraction=saturation,
        boundary_jump=boundary_jump,
        pulse=pulse,
    )


def evaluate_point(
    protocol: EsnSearchProtocol,
    context: TrialContext,
    point: TrialPoint,
    *,
    report: ReportCallback | None = None,
) -> TrialEvaluation:
    """Train ``point`` and run every development scenario until one is infeasible or ``report`` stops it."""
    cases = development_cases(protocol, context.scenario)
    penalty = protocol.objective.infeasible_penalty
    trained = train_point(protocol, context, point)
    if isinstance(trained, str):
        return TrialEvaluation(
            point=point,
            objective=penalty,
            feasible=False,
            penalized=True,
            reason=trained,
            fit_rmse=None,
            scenarios_total=len(cases),
            components=(),
            running=(penalty,),
        )
    recipe, model = trained
    scenario = context.scenario
    lower = np.array([link.q_min for link in scenario.robot.links], dtype=np.float64)
    upper = np.array([link.q_max for link in scenario.robot.links], dtype=np.float64)
    estimator = CausalDerivativeEstimator(
        point.estimator(max_dt_ratio=protocol.max_dt_ratio).config(scenario.timing.dt), scenario.dof
    )
    generator = RcTargetGenerator(model, recipe.encoder(), estimator, position_bounds=(lower, upper))
    controller = GeneratorTrackingController(
        generator, context.tracker, scenario.limits.torque, hold_until_s=scenario.timing.intervals.prime[1]
    )
    duration = float(context.reference.t[-1])
    bound = protocol.feasibility.max_saturation_fraction
    components: list[ScenarioComponent] = []
    running: list[float] = []
    stopped = False
    for index, (initial_q, pulse) in enumerate(cases):
        outcome = simulate(
            scenario,
            controller,
            duration_s=duration,
            initial_q=initial_q,
            force=None if pulse is None else pulse.pulse(),
            channels=GENERATOR_CHANNELS,
        )
        component = _component(
            index, initial_q, pulse, outcome, context=context, bound=bound, boundary_jump=controller.boundary_jump
        )
        components.append(component)
        if not component.feasible:
            running.append(penalty)
            break
        running.append(float(np.median([cast("float", c.move_joint_rmse) for c in components])))
        if report is not None and report(index, running[-1]):
            stopped = True
            break
    feasible = all(c.feasible for c in components) and len(components) == len(cases)
    infeasible = next((c for c in components if not c.feasible), None)
    reason = None if infeasible is None else f"scenario {infeasible.index}: {infeasible.reason}"
    if stopped and infeasible is None:
        reason = "stopped by the pruner"
    objective = running[-1] if feasible or stopped else penalty
    return TrialEvaluation(
        point=point,
        objective=objective,
        feasible=feasible,
        penalized=not feasible and not stopped,
        reason=reason,
        fit_rmse=recipe.fit.rmse,
        scenarios_total=len(cases),
        components=tuple(components),
        running=tuple(running),
        stopped_early=stopped,
    )


def make_objective(
    protocol: EsnSearchProtocol,
    context: TrialContext,
    *,
    on_evaluation: Callable[[optuna.Trial, TrialEvaluation], None] | None = None,
) -> Callable[[optuna.Trial], float]:
    """The Optuna objective: sample a point, evaluate it with pruning reports, store every component."""

    def objective(trial: optuna.Trial) -> float:
        point = suggest_point(protocol.search, trial)

        def report(step: int, value: float) -> bool:
            trial.report(value, step)
            return trial.should_prune()

        evaluation = evaluate_point(protocol, context, point, report=report)
        for key, value in evaluation.attrs().items():
            trial.set_user_attr(key, value)
        if on_evaluation is not None:
            on_evaluation(trial, evaluation)
        if evaluation.stopped_early:
            raise optuna.TrialPruned
        return evaluation.objective

    return objective
