# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Evaluating one recovery search point: paired feasibility and the worst-cell gap-ratio objective (M3R-012).

A point is trained once in memory through its formulation's training spec
(augmented families regenerate their synthetic episodes deterministically) and
run closed loop through every locked development scenario **twice per
scenario** — once under each frozen tracker; the tracker is never a search
parameter. Replay on the identical schedule (tracker, warm-up, start,
disturbance) is the paired baseline; replay runs depend only on
``(scenario, tracker, warm-up)`` and are cached on the context across trials.
A component is *feasible* when the run completes, meets the actual-motion
dwell criteria, stays within the protocol's saturation bound, and the
generated reference passes its own dwell gates (recovery plan section 7.3);
posture-class components additionally need a feasible replay baseline to
define their early command-gap ratio. The scalar objective of a feasible
trial is the worst (largest) of the four class-by-tracker cell medians of the
early command-gap ratio; every infeasible trial receives the protocol's
penalty with the reason and all components computed so far. Confirmatory
locks and outcomes are unreachable from here.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import optuna

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.controllers.adapter import GeneratorTrackingController
from arm_rc_ctrl.controllers.estimator import CausalDerivativeEstimator
from arm_rc_ctrl.controllers.tracking import LimitedTracker
from arm_rc_ctrl.data.records import verify_payload
from arm_rc_ctrl.data.recovery import RecoveryDatasetRecord, load_processed_record
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.experiments.baselines import load_frozen_baseline
from arm_rc_ctrl.experiments.disturbances import ForcePulse
from arm_rc_ctrl.experiments.esn_objective import classify
from arm_rc_ctrl.experiments.perturbations import load_development_robustness, robustness_scenarios
from arm_rc_ctrl.experiments.recovery_search import RECOVERY_TRACKERS, suggest_recovery_point, training_spec_for
from arm_rc_ctrl.experiments.recovery_slice import (
    DEFAULT_SETTLING_BAND_RAD,
    HeldTaskReference,
    recovery_outcome,
    recovery_report_from_arrays,
)
from arm_rc_ctrl.experiments.simulation import GENERATOR_CHANNELS, simulate
from arm_rc_ctrl.metrics.effort import effort_metrics
from arm_rc_ctrl.metrics.recovery import EARLY_WINDOW_S, SATURATION_BOUND, activation_jump, gap_series, gap_summary
from arm_rc_ctrl.rc.generator import RcTargetGenerator
from arm_rc_ctrl.rc.recipe import DatasetSource, create_recipe
from arm_rc_ctrl.rc.teacher_forcing import InputTransform
from arm_rc_ctrl.rc.warmup import WarmupConfig
from arm_rc_ctrl.scenario import load_scenario

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from arm_rc_ctrl.controllers.tracking import TrackerConfig
    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.experiments.perturbations import PerturbationClass, RobustnessScenario
    from arm_rc_ctrl.experiments.recovery_search import RecoverySearchProtocol, RecoveryTrialPoint
    from arm_rc_ctrl.experiments.run_record import RunArrays
    from arm_rc_ctrl.experiments.termination import Termination
    from arm_rc_ctrl.rc.esn import EsnModel
    from arm_rc_ctrl.rc.recipe import ModelRecipe
    from arm_rc_ctrl.rc.train import ModelConfig
    from arm_rc_ctrl.scenario import ScenarioConfig
    from arm_rc_ctrl.storage import StorageRoot

__all__ = [
    "RATIO_CLASSES",
    "RecoveryComponent",
    "RecoveryTrialContext",
    "RecoveryTrialEvaluation",
    "ReplayComponent",
    "evaluate_recovery_point",
    "make_recovery_objective",
    "train_recovery_point",
]

type ReportCallback = Callable[[int, float], bool]
"""Called after every fully feasible scenario with ``(index, running_objective)``; ``True`` stops the evaluation."""

RATIO_CLASSES: Final = ("posture_small", "posture_large")
"""The classes whose paired early command-gap ratios form the four objective cells (plan section 7.3)."""

_GRID_TOLERANCE_S: Final = 1e-9


@dataclass(frozen=True)
class ReplayComponent:
    """The paired replay baseline of one (scenario, tracker) on the common recovery schedule."""

    index: int
    scenario_id: str
    kind: PerturbationClass
    tracker: str
    initial_q: tuple[float, ...]
    termination: str
    criteria: dict[str, bool]
    saturation_fraction: float | None
    activation_jump_rad: float | None
    early_gap_integral: float | None
    feasible: bool
    reason: str | None

    def __post_init__(self) -> None:
        """The reason exists exactly for infeasible baselines; feasible baselines carry both paired values."""
        if self.feasible != (self.reason is None):
            msg = f"feasible must mean no reason, got feasible={self.feasible} reason={self.reason!r}"
            raise ValueError(msg)
        if self.feasible and (self.activation_jump_rad is None or self.early_gap_integral is None):
            msg = f"a feasible replay baseline needs its paired values, got {self}"
            raise ValueError(msg)


@dataclass(frozen=True)
class RecoveryComponent:
    """Objective components of one development (scenario, tracker) pair of the RC arm."""

    index: int
    scenario_id: str
    kind: PerturbationClass
    tracker: str
    initial_q: tuple[float, ...]
    termination: str
    criteria: dict[str, bool]
    generated_criteria: dict[str, bool] | None
    feasible: bool
    reason: str | None
    """Why the component is infeasible (``None`` when feasible)."""
    failure: str | None = None
    limit: str | None = None
    saturation_fraction: float | None = None
    activation_jump_rad: float | None = None
    early_gap_integral: float | None = None
    replay_early_gap_integral: float | None = None
    gap_ratio: float | None = None
    """Early command-gap ratio against replay (posture classes of a feasible component only)."""
    settling_time_s: float | None = None
    torque_rms: float | None = None
    boundary_jump: float | None = None

    def __post_init__(self) -> None:
        """The reason exists exactly for infeasible components; ratios belong to feasible ones."""
        if self.feasible != (self.reason is None):
            msg = f"feasible must mean no reason, got feasible={self.feasible} reason={self.reason!r}"
            raise ValueError(msg)
        if self.gap_ratio is not None and not self.feasible:
            msg = "an infeasible component cannot carry a gap ratio"
            raise ValueError(msg)


@dataclass(frozen=True)
class RecoveryTrialEvaluation:
    """Everything one evaluated recovery point yields; the scalar objective is never the only record."""

    point: RecoveryTrialPoint
    objective: float
    feasible: bool
    penalized: bool
    reason: str | None
    fit_rmse: float | None
    scenarios_total: int
    """Development (scenario, tracker) pairs of a complete trial."""
    components: tuple[RecoveryComponent, ...]
    cells: dict[str, float]
    """Median early command-gap ratio per ``{class}:{tracker}`` cell (complete for feasible trials only)."""
    running: tuple[float, ...]
    """The running objective after each fully feasible scenario with ratio data (what the pruner sees)."""
    stopped_early: bool = False
    """Whether a report callback stopped the evaluation before its natural end."""

    def __post_init__(self) -> None:
        """A feasible trial has complete cells and no penalty; infeasible trials never carry cells."""
        if self.feasible and (self.penalized or not self.cells):
            msg = f"a feasible trial has cells and no penalty, got {self}"
            raise ValueError(msg)
        if not self.feasible and self.cells:
            msg = "cell medians are recorded for feasible trials only"
            raise ValueError(msg)

    def attrs(self) -> dict[str, object]:
        """The evaluation as (nested) trial attributes; ``studies.summarize`` flattens them."""
        return {
            "feasible": self.feasible,
            "penalized": self.penalized,
            "reason": "" if self.reason is None else self.reason,
            "fit_rmse": self.fit_rmse,
            "scenarios_total": self.scenarios_total,
            "components_evaluated": len(self.components),
            "stopped_early": self.stopped_early,
            "cells": dict(self.cells),
            "running": list(self.running),
            "components": [to_mapping(c) for c in self.components],
        }


@dataclass(frozen=True)
class RecoveryTrialContext:
    """The loaded, bound inputs every trial of a recovery study shares, plus the cross-trial replay cache."""

    scenario: ScenarioConfig
    scenario_file: Path
    reference: SampleSet
    dataset: RecoveryDatasetRecord
    source: DatasetSource
    trackers: dict[str, TrackerConfig]
    base_model: ModelConfig
    scenarios: tuple[RobustnessScenario, ...]
    replay_cache: dict[tuple[str, float], tuple[ReplayComponent, ...]] = field(
        default_factory=dict, repr=False, compare=False
    )

    @classmethod
    def load(
        cls, protocol: RecoverySearchProtocol, *, store: StorageRoot, dataset_file: Path, records_root: Path
    ) -> RecoveryTrialContext:
        """Load and bind the scenario, recovery dataset, both frozen trackers, and the locked development levels."""
        if "confirmatory" in protocol.development.name:
            msg = (
                f"protocol {protocol.name!r} points its development levels at {protocol.development.name!r}; "
                "the confirmatory lock must stay unreachable from the search"
            )
            raise ValueError(msg)
        scenario = load_scenario(protocol.scenario)
        dataset = load_processed_record(dataset_file)
        if not isinstance(dataset, RecoveryDatasetRecord):
            msg = f"{dataset_file} is not a recovery dataset record, got {type(dataset).__name__}"
            raise TypeError(msg)
        reference = load_samples(verify_payload(store, dataset.artifact))
        dataset.check_scenario(protocol.scenario)
        dataset.check_samples(reference)
        if scenario.dof != dataset.dof:
            msg = f"dof mismatch: scenario {scenario.dof}, dataset {dataset.dof}"
            raise ValueError(msg)
        if dataset.normalization is None:
            msg = f"dataset {dataset.artifact.artifact_id} records no normalization statistics"
            raise ValueError(msg)
        relative = dataset_file.resolve().relative_to(records_root.resolve()).as_posix()
        source = DatasetSource(dataset.artifact.artifact_id, dataset.artifact.payload.sha256, relative)
        trackers: dict[str, TrackerConfig] = {}
        for name in RECOVERY_TRACKERS:
            tracker = load_frozen_baseline(name)
            if tracker.dof != scenario.dof:
                msg = f"dof mismatch: scenario {scenario.dof}, tracker {name!r} {tracker.dof}"
                raise ValueError(msg)
            trackers[name] = tracker
        levels = load_development_robustness(protocol.development)
        lower = tuple(link.q_min for link in scenario.robot.links)
        upper = tuple(link.q_max for link in scenario.robot.links)
        scenarios = robustness_scenarios(levels, nominal=dataset.q0_ref, lower=lower, upper=upper)
        return cls(scenario, protocol.scenario, reference, dataset, source, trackers, protocol.base_model(), scenarios)

    @property
    def samples(self) -> Mapping[str, SampleSet]:
        """Training samples keyed by dataset ID."""
        return {self.dataset.artifact.artifact_id: self.reference}

    def replay_components(
        self, tracker: str, warmup_s: float, *, bound: float = SATURATION_BOUND
    ) -> tuple[ReplayComponent, ...]:
        """Replay baselines of every development scenario under ``tracker`` at ``warmup_s`` (cached across trials).

        The saturation ``bound`` is protocol-fixed (the recovery protocols reject any other value), so it is
        deliberately not part of the cache key.
        """
        key = (tracker, warmup_s)
        cached = self.replay_cache.get(key)
        if cached is not None:
            return cached
        gains = self.trackers[tracker]
        activation = WarmupConfig(warmup_s).duration_s
        duration = activation + float(self.reference.t[-1])
        components: list[ReplayComponent] = []
        for index, case in enumerate(self.scenarios):
            start = case.initial_q(self.dataset.q0_ref)
            run_force = _run_force(case.pulse, activation)
            held = HeldTaskReference.from_samples(
                self.reference,
                activation_s=activation,
                interpolation=self.dataset.preprocessing.interpolation,
                hold=np.asarray(start, dtype=np.float64),
            )
            controller = LimitedTracker(cast("Any", held), gains, self.scenario.limits.torque)
            arrays, termination = simulate(
                self.scenario, controller, duration_s=duration, initial_q=start, force=run_force
            )
            components.append(
                _replay_component(
                    index,
                    case,
                    tracker,
                    start,
                    self.scenario,
                    self.reference,
                    arrays,
                    termination,
                    activation_s=activation,
                    bound=bound,
                )
            )
        result = tuple(components)
        self.replay_cache[key] = result
        return result


def _run_force(pulse: ForcePulse | None, activation_s: float) -> ForcePulse | None:
    """The task-clock pulse shifted onto the run clock (it can never land in the pre-task hold)."""
    if pulse is None:
        return None
    return ForcePulse(start_s=pulse.start_s + activation_s, duration_s=pulse.duration_s, force=pulse.force)


def _saturation(arrays: RunArrays) -> float | None:
    values = cast("NDArray[np.int64]", arrays.arrays["saturation"])
    return float(np.mean(values)) if values.shape[0] else None


def _replay_component(
    index: int,
    case: RobustnessScenario,
    tracker: str,
    start: tuple[float, ...],
    scenario: ScenarioConfig,
    reference: SampleSet,
    arrays: RunArrays,
    termination: Termination,
    *,
    activation_s: float,
    bound: float,
) -> ReplayComponent:
    criteria = recovery_outcome(scenario, reference, arrays, termination, activation_s=activation_s)
    saturation = _saturation(arrays)
    reason = classify(termination, criteria, saturation, bound)
    jump: float | None = None
    early: float | None = None
    if reason is None:
        run_t = cast("NDArray[np.float64]", arrays.arrays["t"])
        active = run_t >= activation_s - _GRID_TOLERANCE_S
        q_active = cast("NDArray[np.float64]", arrays.arrays["q"])[active]
        if q_active.shape[0] != reference.n_samples:
            reason = "no_active_segment"
        else:
            task_t = run_t[active] - run_t[active][0]
            gap = gap_series(reference.q, q_active)
            jump = activation_jump(reference.q[0], q_active[0])
            early = float(gap_summary(task_t, gap, window=(0.0, EARLY_WINDOW_S)).integral)
            if not early > 0.0:
                reason = "degenerate_replay_gap"
                jump = None
                early = None
    return ReplayComponent(
        index=index,
        scenario_id=case.scenario_id,
        kind=case.kind,
        tracker=tracker,
        initial_q=start,
        termination=termination.kind,
        criteria=criteria,
        saturation_fraction=saturation,
        activation_jump_rad=jump,
        early_gap_integral=early,
        feasible=reason is None,
        reason=reason,
    )


def train_recovery_point(
    protocol: RecoverySearchProtocol, context: RecoveryTrialContext, point: RecoveryTrialPoint
) -> tuple[ModelRecipe, EsnModel] | str:
    """Train ``point`` in memory through its formulation's training spec, or return the infeasibility reason."""
    base = context.base_model
    normalization = context.dataset.normalization
    if normalization is None:  # pragma: no cover - RecoveryTrialContext.load rejects such datasets
        return "training_failure:no_normalization"
    transform = InputTransform.derive(
        base.input_transform.policy, normalization, fixed_scales=base.input_transform.fixed_scales
    )
    config = point.esn.model_config(base, name=f"{protocol.name}-trial")
    training = training_spec_for(protocol, point)
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
            training=training,
            scenario=context.scenario,
        )
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return f"training_failure:{type(exc).__name__}"
    if not math.isfinite(recipe.fit.rmse):
        return "training_failure:non_finite_fit"
    return recipe, model


def _blocked_component(
    index: int, case: RobustnessScenario, tracker: str, start: tuple[float, ...], replay: ReplayComponent
) -> RecoveryComponent:
    """A posture-class pair whose replay baseline is infeasible: the ratio is undefined, so the RC run is skipped."""
    return RecoveryComponent(
        index=index,
        scenario_id=case.scenario_id,
        kind=case.kind,
        tracker=tracker,
        initial_q=start,
        termination="not_simulated",
        criteria={},
        generated_criteria=None,
        feasible=False,
        reason=f"replay_infeasible:{replay.reason}",
    )


def _component(
    index: int,
    case: RobustnessScenario,
    tracker: str,
    start: tuple[float, ...],
    arrays_and_termination: tuple[RunArrays, Termination],
    *,
    context: RecoveryTrialContext,
    activation_s: float,
    bound: float,
    replay: ReplayComponent,
    boundary_jump: float | None,
    settling_band_rad: float,
) -> RecoveryComponent:
    arrays, termination = arrays_and_termination
    criteria = recovery_outcome(context.scenario, context.reference, arrays, termination, activation_s=activation_s)
    saturation = _saturation(arrays)
    reason = classify(termination, criteria, saturation, bound)
    generated: dict[str, bool] | None = None
    jump: float | None = None
    early: float | None = None
    ratio: float | None = None
    settling: float | None = None
    torque: float | None = None
    if reason is None:
        report = recovery_report_from_arrays(
            context.scenario, context.reference, arrays, activation_s=activation_s, settling_band_rad=settling_band_rad
        )
        if report is None:
            reason = "no_active_segment"
        else:
            generated = dict(report.generated_dwell_criteria)
            missed = sorted(name for name, ok in generated.items() if not ok)
            if missed:
                reason = "generated_dwell:" + ",".join(missed)
            else:
                jump = report.activation_jump_rad
                early = report.command_gap_early.integral
                settling = report.reference_settling.settling_time_s
                run_t = cast("NDArray[np.float64]", arrays.arrays["t"])
                source = "tau_applied" if "tau_applied" in arrays.arrays else "tau_requested"
                tau = cast("NDArray[np.float64]", arrays.arrays[source])
                effort = effort_metrics(
                    run_t, tau, context.scenario.limits.torque, window=(activation_s, float(run_t[-1]))
                )
                torque = float(effort.torque_rms)
                if case.kind in RATIO_CLASSES:
                    integral = replay.early_gap_integral
                    assert integral is not None  # a blocked replay never reaches _component
                    ratio = float(early / integral)
    return RecoveryComponent(
        index=index,
        scenario_id=case.scenario_id,
        kind=case.kind,
        tracker=tracker,
        initial_q=start,
        termination=termination.kind,
        criteria=criteria,
        generated_criteria=generated,
        feasible=reason is None,
        reason=reason,
        failure=termination.failure,
        limit=termination.limit,
        saturation_fraction=saturation,
        activation_jump_rad=jump,
        early_gap_integral=early,
        replay_early_gap_integral=replay.early_gap_integral,
        gap_ratio=ratio,
        settling_time_s=settling,
        torque_rms=torque,
        boundary_jump=boundary_jump,
    )


def _worst_cell(ratios: Mapping[tuple[str, str], Sequence[float]]) -> float:
    """The largest cell median over the ratio cells collected so far."""
    return max(float(statistics.median(values)) for values in ratios.values())


def _run_development(
    context: RecoveryTrialContext,
    controllers: dict[str, GeneratorTrackingController],
    replays: dict[str, tuple[ReplayComponent, ...]],
    *,
    duration: float,
    activation: float,
    bound: float,
    penalty: float,
    settling_band_rad: float,
    report: ReportCallback | None,
) -> tuple[list[RecoveryComponent], dict[tuple[str, str], list[float]], list[float], bool, bool]:
    """Run the (scenario, tracker) grid until a pair is infeasible or ``report`` stops the evaluation."""
    components: list[RecoveryComponent] = []
    ratios: dict[tuple[str, str], list[float]] = {}
    running: list[float] = []
    stopped = False
    infeasible_hit = False
    for index, case in enumerate(context.scenarios):
        start = case.initial_q(context.dataset.q0_ref)
        run_force = _run_force(case.pulse, activation)
        for name, controller in controllers.items():
            replay = replays[name][index]
            if case.kind in RATIO_CLASSES and not replay.feasible:
                component = _blocked_component(index, case, name, start, replay)
            else:
                outcome = simulate(
                    context.scenario,
                    controller,
                    duration_s=duration,
                    initial_q=start,
                    force=run_force,
                    channels=GENERATOR_CHANNELS,
                )
                component = _component(
                    index,
                    case,
                    name,
                    start,
                    outcome,
                    context=context,
                    activation_s=activation,
                    bound=bound,
                    replay=replay,
                    boundary_jump=controller.boundary_jump,
                    settling_band_rad=settling_band_rad,
                )
            components.append(component)
            if not component.feasible:
                infeasible_hit = True
                break
            if component.gap_ratio is not None:
                ratios.setdefault((str(case.kind), name), []).append(component.gap_ratio)
        if infeasible_hit:
            running.append(penalty)
            break
        if ratios:
            running.append(_worst_cell(ratios))
            if report is not None and report(index, running[-1]):
                stopped = True
                break
    return components, ratios, running, stopped, infeasible_hit


def evaluate_recovery_point(
    protocol: RecoverySearchProtocol,
    context: RecoveryTrialContext,
    point: RecoveryTrialPoint,
    *,
    report: ReportCallback | None = None,
    settling_band_rad: float = DEFAULT_SETTLING_BAND_RAD,
) -> RecoveryTrialEvaluation:
    """Train ``point`` and run every (scenario, tracker) pair until one is infeasible or ``report`` stops it."""
    kinds = {case.kind for case in context.scenarios}
    missing = [c for c in RATIO_CLASSES if c not in kinds]
    if missing:
        msg = f"development scenarios must cover both posture classes; missing {missing}"
        raise ValueError(msg)
    total = len(context.scenarios) * len(context.trackers)
    penalty = protocol.objective.infeasible_penalty
    trained = train_recovery_point(protocol, context, point)
    if isinstance(trained, str):
        return RecoveryTrialEvaluation(
            point=point,
            objective=penalty,
            feasible=False,
            penalized=True,
            reason=trained,
            fit_rmse=None,
            scenarios_total=total,
            components=(),
            cells={},
            running=(penalty,),
        )
    recipe, model = trained
    scenario = context.scenario
    activation = point.warmup_s
    duration = activation + float(context.reference.t[-1])
    bound = protocol.feasibility.max_saturation_fraction
    lower = np.array([link.q_min for link in scenario.robot.links], dtype=np.float64)
    upper = np.array([link.q_max for link in scenario.robot.links], dtype=np.float64)
    controllers: dict[str, GeneratorTrackingController] = {}
    for name, gains in context.trackers.items():
        estimator = CausalDerivativeEstimator(
            point.esn.estimator(max_dt_ratio=protocol.max_dt_ratio).config(scenario.timing.dt), scenario.dof
        )
        generator = RcTargetGenerator(model, recipe.encoder(), estimator, position_bounds=(lower, upper))
        controllers[name] = GeneratorTrackingController(
            generator, gains, scenario.limits.torque, hold_until_s=activation
        )
    replays = {name: context.replay_components(name, activation, bound=bound) for name in context.trackers}
    components, ratios, running, stopped, infeasible_hit = _run_development(
        context,
        controllers,
        replays,
        duration=duration,
        activation=activation,
        bound=bound,
        penalty=penalty,
        settling_band_rad=settling_band_rad,
        report=report,
    )
    feasible = not infeasible_hit and len(components) == total
    infeasible = next((c for c in components if not c.feasible), None)
    reason = None if infeasible is None else f"scenario {infeasible.index} [{infeasible.tracker}]: {infeasible.reason}"
    if stopped and infeasible is None:
        reason = "stopped by the pruner"
    cells: dict[str, float] = {}
    if feasible:
        cells = {
            f"{kind}:{tracker}": float(statistics.median(values)) for (kind, tracker), values in sorted(ratios.items())
        }
    objective = max(cells.values()) if feasible else (running[-1] if stopped else penalty)
    return RecoveryTrialEvaluation(
        point=point,
        objective=objective,
        feasible=feasible,
        penalized=not feasible and not stopped,
        reason=reason,
        fit_rmse=recipe.fit.rmse,
        scenarios_total=total,
        components=tuple(components),
        cells=cells,
        running=tuple(running),
        stopped_early=stopped,
    )


def make_recovery_objective(
    protocol: RecoverySearchProtocol,
    context: RecoveryTrialContext,
    *,
    on_evaluation: Callable[[optuna.Trial, RecoveryTrialEvaluation], None] | None = None,
) -> Callable[[optuna.Trial], float]:
    """The Optuna objective: sample a point, evaluate it with pruning reports, store every component."""

    def objective(trial: optuna.Trial) -> float:
        point = suggest_recovery_point(protocol, trial)

        def report(step: int, value: float) -> bool:
            trial.report(value, step)
            return trial.should_prune()

        evaluation = evaluate_recovery_point(protocol, context, point, report=report)
        for key, value in evaluation.attrs().items():
            trial.set_user_attr(key, value)
        if on_evaluation is not None:
            on_evaluation(trial, evaluation)
        if evaluation.stopped_early:
            raise optuna.TrialPruned
        return evaluation.objective

    return objective
