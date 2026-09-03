# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""No-augmentation timing vertical slice (M3R-009; recovery plan sections 4.2 and 8, gate 2).

Replay and RC run the same scenario on the common recovery schedule: both hold
the identical initial posture during the pre-task interval ``[0, T_w)`` of the
run clock (replay through :class:`HeldTaskReference`, RC by priming the
generator with measured state), then activate **simultaneously** at
``activation_s = T_w`` — task time zero. Both arms share the tracker, the
torque limits, the initial posture, and any disturbance; runs record the
activation boundary and the versioned task-time telemetry, and the RC run's
active segment is evaluated with the M3R-008 recovery metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.controllers.adapter import GeneratorTrackingController
from arm_rc_ctrl.controllers.estimator import EstimatorConfig
from arm_rc_ctrl.controllers.reference import DemonstrationReference
from arm_rc_ctrl.controllers.tracking import LimitedTracker
from arm_rc_ctrl.data.recovery import RecoveryDatasetRecord, task_intervals_from_phases
from arm_rc_ctrl.experiments.run_record import LoadedRun, RunPointerRecord, RunSummary, load_run, write_run
from arm_rc_ctrl.experiments.simulation import GENERATOR_CHANNELS, simulate
from arm_rc_ctrl.experiments.termination import Outcome
from arm_rc_ctrl.metrics.dwell import dwell_metrics
from arm_rc_ctrl.metrics.recovery import RecoveryMetricsReport, compute_recovery_metrics
from arm_rc_ctrl.provenance import ArtifactReference, collect_provenance, require_clean_for_confirmatory
from arm_rc_ctrl.rc.runtime import generator_from_recipe
from arm_rc_ctrl.scenario import endpoint_positions

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from numpy.typing import NDArray

    from arm_rc_ctrl.controllers.tracking import TrackerConfig
    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.experiments.disturbances import ForcePulse
    from arm_rc_ctrl.experiments.run_record import RunArrays
    from arm_rc_ctrl.experiments.termination import Termination
    from arm_rc_ctrl.rc.recipe import ModelRecipe
    from arm_rc_ctrl.rc.warmup import WarmupConfig
    from arm_rc_ctrl.scenario import ScenarioConfig
    from arm_rc_ctrl.storage import StorageRoot

__all__ = ["HeldTaskReference", "RecoveryPair", "SliceRunResult", "run_recovery_pair"]

_GRID_TOLERANCE_S = 1e-9
_DEFAULT_SETTLING_BAND_RAD = 0.05


class HeldTaskReference:
    """``skelarm`` joint reference: hold the task initial posture until activation, then play the task reference.

    Before ``activation_s`` the reference is ``(q_0, 0, 0)`` — the common
    pre-task hold; from activation on it is the cropped task reference shifted
    onto the run clock (``t - activation_s``).
    """

    def __init__(self, reference: DemonstrationReference, *, activation_s: float) -> None:
        if not (math.isfinite(activation_s) and activation_s >= 0):
            msg = f"activation_s must be finite and non-negative, got {activation_s!r}"
            raise ValueError(msg)
        self._reference = reference
        self._activation = activation_s
        self._hold: NDArray[np.float64] = np.array(reference.q[0], dtype=np.float64)
        self._zeros: NDArray[np.float64] = np.zeros_like(self._hold)

    @classmethod
    def from_samples(
        cls, samples: SampleSet, *, activation_s: float, interpolation: str = "linear"
    ) -> HeldTaskReference:
        """Build from a cropped task dataset (task clock starting at zero)."""
        reference = DemonstrationReference.from_samples(samples, cast("Any", interpolation))
        return cls(reference, activation_s=activation_s)

    @property
    def activation_s(self) -> float:
        """The common activation boundary on the run clock."""
        return self._activation

    def sample(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """The held posture before activation; the shifted task reference from activation on."""
        if t < self._activation:
            return self._hold, self._zeros, self._zeros
        return self._reference.sample(t - self._activation)


@dataclass(frozen=True)
class SliceRunResult:
    """One persisted arm of the paired slice."""

    pointer: RunPointerRecord
    summary: RunSummary
    directory: Path
    run: LoadedRun


@dataclass(frozen=True)
class RecoveryPair:
    """Both arms of one scenario on the common recovery schedule."""

    replay: SliceRunResult
    rc: SliceRunResult
    activation_s: float
    recovery: RecoveryMetricsReport | None
    """M3R-008 metrics of the RC run's active segment; ``None`` when the RC run did not complete."""


def _bind(scenario: ScenarioConfig, scenario_file: Path, dataset: RecoveryDatasetRecord, reference: SampleSet) -> None:
    dataset.check_scenario(scenario_file)
    dataset.check_samples(reference)
    if scenario.dof != dataset.dof:
        msg = f"dof mismatch: scenario {scenario.dof}, dataset {dataset.dof}"
        raise ValueError(msg)


def _outcome(
    scenario: ScenarioConfig,
    reference: SampleSet,
    arrays: RunArrays,
    termination: Termination,
    *,
    activation_s: float,
) -> dict[str, bool]:
    """Completion plus the scenario dwell criteria over the task dwell window shifted onto the run clock."""
    criteria = scenario.task.dwell_criteria
    result = {"completed": termination.is_completed, **dict.fromkeys(criteria.names, False)}
    if not termination.is_completed:
        return result
    task = task_intervals_from_phases(reference.t, reference.phase)
    window = (activation_s + task.dwell[0], activation_s + task.dwell[1])
    run_t = cast("NDArray[np.float64]", arrays.arrays["t"])
    if not bool(np.any((run_t >= window[0]) & (run_t <= window[1]))):
        return result
    metrics = dwell_metrics(
        run_t,
        cast("NDArray[np.float64]", arrays.arrays["tip"]),
        cast("NDArray[np.float64]", arrays.arrays["dq"]),
        np.asarray(scenario.task.target, dtype=np.float64),
        criteria.tolerance,
        window=window,
    )
    result.update(criteria.evaluate(metrics))
    return result


def _recovery_report(
    scenario: ScenarioConfig,
    reference: SampleSet,
    arrays: RunArrays,
    *,
    activation_s: float,
    settling_band_rad: float,
) -> RecoveryMetricsReport | None:
    """The M3R-008 metrics of the active segment (``None`` when it does not cover the task)."""
    run_t = cast("NDArray[np.float64]", arrays.arrays["t"])
    active = run_t >= activation_s - _GRID_TOLERANCE_S
    if int(np.count_nonzero(active)) != reference.n_samples:
        return None
    task_t = run_t[active] - run_t[active][0]
    q_desired = cast("NDArray[np.float64]", arrays.arrays["generator_output_q"])[active]
    dq_desired = cast("NDArray[np.float64]", arrays.arrays["dq_desired"])[active]
    task = task_intervals_from_phases(reference.t, reference.phase)
    return compute_recovery_metrics(
        task_t,
        cast("NDArray[np.float64]", arrays.arrays["q"])[active],
        cast("NDArray[np.float64]", arrays.arrays["dq"])[active],
        q_desired,
        dq_desired,
        endpoint_positions(scenario, q_desired),
        reference.q,
        target=np.asarray(scenario.task.target, dtype=np.float64),
        dwell_window=(task.dwell[0], task.dwell[1]),
        settling_band_rad=settling_band_rad,
    )


def run_recovery_pair(
    scenario: ScenarioConfig,
    scenario_file: Path,
    dataset: RecoveryDatasetRecord,
    reference: SampleSet,
    recipe: ModelRecipe,
    tracker: TrackerConfig,
    *,
    store: StorageRoot,
    warmup: WarmupConfig,
    exploratory: bool,
    estimator: EstimatorConfig | None = None,
    initial_q: tuple[float, ...] | None = None,
    force: ForcePulse | None = None,
    settling_band_rad: float = _DEFAULT_SETTLING_BAND_RAD,
    now: datetime | None = None,
    command: str = "python -m arm_rc_ctrl.experiments.recovery_slice",
    license_label: str = "LicenseRef-Private",
    access: str = "private",
) -> RecoveryPair:
    """Run both arms of one scenario on the common schedule, persist them, and evaluate the RC segment.

    ``force`` acts on the run clock; place task-relative pulses at
    ``activation_s + t``. The recipe's training warm-up must equal the run
    warm-up so the evaluation schedule matches what the model saw.
    """
    _bind(scenario, scenario_file, dataset, reference)
    if tracker.dof != scenario.dof or recipe.dof != scenario.dof:
        msg = f"dof mismatch: scenario {scenario.dof}, tracker {tracker.dof}, recipe {recipe.dof}"
        raise ValueError(msg)
    if recipe.training.washout != "warmup_hold" or recipe.training.warmup_s != warmup.duration_s:
        msg = (
            f"the run warm-up {warmup.duration_s!r} s must equal the recipe's training warm-up "
            f"({recipe.training.washout!r}, {recipe.training.warmup_s!r})"
        )
        raise ValueError(msg)
    activation = warmup.duration_s
    duration = activation + float(reference.t[-1])
    reference_artifact = dataset.artifact.artifact_id
    payload = dataset.artifact.payload
    est = EstimatorConfig(nominal_dt_s=scenario.timing.dt) if estimator is None else estimator
    resolved: dict[str, object] = {
        "scenario": to_mapping(scenario),
        "tracker": to_mapping(tracker),
        "recipe": to_mapping(recipe),
        "estimator": to_mapping(est),
        "warmup": to_mapping(warmup),
        "activation_s": activation,
        "reference_artifact": reference_artifact,
        "initial_q": list(scenario.task.initial_q if initial_q is None else initial_q),
        "duration_s": duration,
        "force": None if force is None else to_mapping(force),
        "command": command,
    }
    reference_payload = ArtifactReference(payload.uri, payload.sha256, payload.size)

    def _persist(
        arrays: RunArrays, termination: Termination, *, method: str, seeds: dict[str, int], arm: str
    ) -> SliceRunResult:
        provenance = collect_provenance(
            {**resolved, "arm": arm}, seeds=seeds, artifacts=[reference_payload], exploratory=exploratory, now=now
        )
        require_clean_for_confirmatory(provenance)
        outcome = Outcome(termination, _outcome(scenario, reference, arrays, termination, activation_s=activation))
        pointer, summary, directory = write_run(
            store,
            arrays,
            kind="simulation",
            method=method,
            scenario=scenario.name,
            control_period_s=scenario.timing.dt,
            duration_s=duration,
            target=scenario.task.target,
            task_code=(),
            disturbances=() if force is None else (force.to_disturbance(),),
            termination=termination,
            outcome=outcome,
            provenance=provenance,
            license_label=license_label,
            access=cast("Any", access),
            command=command,
            sources=(reference_artifact,),
            activation_s=activation,
            notes=f"Recovery slice {arm} arm: common hold until {activation} s, activation at task time zero.",
        )
        return SliceRunResult(pointer, summary, directory, load_run(store, pointer))

    held = HeldTaskReference.from_samples(
        reference, activation_s=activation, interpolation=dataset.preprocessing.interpolation
    )
    replay_controller = LimitedTracker(cast("Any", held), tracker, scenario.limits.torque)
    replay_arrays, replay_termination = simulate(
        scenario, replay_controller, duration_s=duration, initial_q=initial_q, force=force
    )
    replay = _persist(replay_arrays, replay_termination, method=f"replay+{tracker.method}", seeds={}, arm="replay")

    lower = np.array([link.q_min for link in scenario.robot.links], dtype=np.float64)
    upper = np.array([link.q_max for link in scenario.robot.links], dtype=np.float64)
    generator = generator_from_recipe(
        recipe, {reference_artifact: reference}, estimator=est, position_bounds=(lower, upper)
    )
    rc_controller = GeneratorTrackingController(generator, tracker, scenario.limits.torque, hold_until_s=activation)
    rc_arrays, rc_termination = simulate(
        scenario, rc_controller, duration_s=duration, initial_q=initial_q, force=force, channels=GENERATOR_CHANNELS
    )
    rc = _persist(
        rc_arrays,
        rc_termination,
        method=f"rc+{tracker.method}",
        seeds={"reservoir": recipe.esn.reservoir.seed},
        arm="rc",
    )
    recovery = None
    if rc_termination.is_completed:
        recovery = _recovery_report(
            scenario, reference, rc_arrays, activation_s=activation, settling_band_rad=settling_band_rad
        )
    return RecoveryPair(replay=replay, rc=rc, activation_s=activation, recovery=recovery)
