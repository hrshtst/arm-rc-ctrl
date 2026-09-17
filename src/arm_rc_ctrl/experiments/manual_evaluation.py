# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-008: the manual protocol's run verdicts (plan section 6, clarification I5).

Three rules separate this protocol from the inherited recovery one. Completion
is judged against the **configured evaluation horizon** rather than the
demonstration's length, so reaching the target and then drifting away for the
rest of the horizon cannot count as success, and an aborted run still keeps
its partial metrics. The dwell is the acquisition rule -- one uninterrupted
second inside the target radius with every joint slow, any excursion
restarting the timer -- rather than the old occupancy fraction over a
phase-derived window. And the force pulse is triggered by the measured motion,
so its verdict is about whether it fired at all and early enough to leave room
for the final dwell it is meant to disturb.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path  # a run-time import: the configuration loader resolves field types at run time
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np

from arm_rc_ctrl.config import from_mapping, load_config, to_mapping
from arm_rc_ctrl.controllers.adapter import GeneratorTrackingController
from arm_rc_ctrl.controllers.estimator import CausalDerivativeEstimator, EstimatorConfig
from arm_rc_ctrl.controllers.tracking import LimitedTracker
from arm_rc_ctrl.data.manual import DwellPredicate, continuous_dwell, dwell_runs
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario, manual_endpoint_positions
from arm_rc_ctrl.experiments.baselines import frozen_baseline_digest
from arm_rc_ctrl.experiments.manual_fits import ManualFitStore
from arm_rc_ctrl.experiments.recovery_search import RECOVERY_TRACKERS
from arm_rc_ctrl.experiments.recovery_slice import HeldTaskReference
from arm_rc_ctrl.experiments.run_record import write_run
from arm_rc_ctrl.experiments.simulation import GENERATOR_CHANNELS, RESIDUAL_CHANNELS, DwellTrigger, simulate
from arm_rc_ctrl.experiments.termination import Outcome
from arm_rc_ctrl.metrics.recovery import SATURATION_BOUND
from arm_rc_ctrl.provenance import ArtifactReference, canonical_json, sha256_bytes, sha256_file
from arm_rc_ctrl.rc.generator import RcTargetGenerator
from arm_rc_ctrl.validation import is_hex

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from arm_rc_ctrl.controllers.tracking import TrackerConfig
    from arm_rc_ctrl.data.manual_scenario import ManualScenarioConfig
    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.execution import ExecutionRecord
    from arm_rc_ctrl.experiments.disturbances import ForcePulse
    from arm_rc_ctrl.experiments.manual_fits import CachedFit, ManualFitInputs
    from arm_rc_ctrl.experiments.manual_study import StudyConfiguration, StudyModel
    from arm_rc_ctrl.experiments.perturbations import RobustnessScenario
    from arm_rc_ctrl.experiments.run_record import RunArrays
    from arm_rc_ctrl.experiments.termination import Termination
    from arm_rc_ctrl.provenance import ProvenanceRecord
    from arm_rc_ctrl.storage import StorageRoot

__all__ = [
    "PROGRESS_FILE",
    "GeneratedReferenceReport",
    "ManualDwellReport",
    "ManualEvaluationConfig",
    "ManualEvaluationRunner",
    "ManualFitBinding",
    "ManualModelEvidence",
    "ManualPairRecord",
    "ManualReplayBank",
    "ManualRunConditions",
    "ManualRunOutcome",
    "ManualSimulationLimits",
    "ManualTriggerRule",
    "SimulateFn",
    "TriggerOutcome",
    "horizon_completed",
    "load_manual_evaluation_config",
    "load_manual_model_evidence",
    "load_manual_replay_bank",
    "manual_bank_to_json",
    "manual_conditions",
    "manual_dwell_report",
    "manual_evidence_to_json",
    "manual_run_outcome",
    "manual_trigger",
    "model_uri",
    "replay_bank_uri",
    "trigger_outcome",
]

_SHA256_HEX: Final = 64
_SHORT: Final = 12
_GRID_TOLERANCE_S: Final = 1e-9
"""Slack for comparing times that are exact multiples of the control period."""

ASSIGNMENT_ALL: Final = "M10"
"""The all-ten arm trains on the whole bank, so its runs are sourced from it rather than one parent."""

REPORTS_PREFIX: Final = "armrc://reports/task_1a_manual_v1"
"""Where this experiment's evidence lives in the store."""

EVALUATION_SCHEMA_VERSION: Final = 1
PROGRESS_FILE: Final = "progress.json"
"""Run-granular progress beside a manifest, so an interrupted sweep keeps the runs it paid for."""
MODEL_STATUSES: Final = ("feasible", "infeasible")
"""Every scenario is attempted, so a model is feasible or not; nothing is left unexecuted by an earlier failure."""

type SimulateFn = Callable[..., tuple[RunArrays, Termination]]


def horizon_completed(
    t: NDArray[np.float64], termination: Termination, *, activation_s: float, horizon_s: float
) -> bool:
    """Whether the run terminated normally **and** ran the whole configured horizon from activation (I5).

    The inherited check compares the active sample count with the reference
    length, which this protocol cannot use: the horizon is a finite timeout for
    non-convergence, deliberately far longer than any demonstration.
    """
    times = np.asarray(t, dtype=np.float64)
    if times.ndim != 1 or times.shape[0] == 0:
        msg = f"t must be a non-empty 1-D array, got shape {times.shape}"
        raise ValueError(msg)
    if not termination.is_completed:
        return False
    return float(times[-1]) >= activation_s + horizon_s - _GRID_TOLERANCE_S


@dataclass(frozen=True)
class ManualDwellReport:
    """What the continuous dwell rule measured on one run (plan section 6's four reported quantities)."""

    ok: bool
    """The run ending at the last sample satisfies the rule for its full required duration."""
    final_samples: int
    final_duration_s: float
    longest_samples: int
    longest_duration_s: float
    earliest_start_s: float | None
    """Start of the first run that reached the required length, if any."""
    departures_after_hold: int
    """Qualifying holds that ended before the last sample: the arm reached the target and then left it."""

    def __post_init__(self) -> None:
        """Counts are non-negative and a successful report ends in a qualifying run."""
        if self.final_samples < 0 or self.longest_samples < self.final_samples or self.departures_after_hold < 0:
            msg = f"inconsistent dwell counts: {self}"
            raise ValueError(msg)
        if self.ok and self.final_samples == 0:
            msg = "a successful dwell report must end in a qualifying run"
            raise ValueError(msg)


def manual_dwell_report(
    t: NDArray[np.float64],
    tip: NDArray[np.float64],
    dq: NDArray[np.float64],
    *,
    target: NDArray[np.float64],
    predicate: DwellPredicate,
    since_s: float | None = None,
) -> ManualDwellReport:
    """Measure the dwell of a run under the acquisition predicate.

    ``since_s`` restarts the measurement at that time, which is how a force
    case is judged: the pulse resets the success timer, so only the dwell after
    the pulse ends can satisfy the rule.
    """
    times = np.asarray(t, dtype=np.float64)
    tips = np.asarray(tip, dtype=np.float64)
    speeds = np.asarray(dq, dtype=np.float64)
    if since_s is not None:
        first = int(np.searchsorted(times, since_s - _GRID_TOLERANCE_S, side="left"))
        times, tips, speeds = times[first:], tips[first:], speeds[first:]
    measurement = continuous_dwell(times, tips, speeds, target=target, predicate=predicate)
    runs = dwell_runs(times, tips, speeds, target=target, predicate=predicate)
    n = times.shape[0]
    departures = sum(1 for start, end in runs if end < n and end - start >= predicate.min_samples)
    longest = max((float(times[end - 1] - times[start]) for start, end in runs), default=0.0)
    return ManualDwellReport(
        ok=measurement.ok,
        final_samples=measurement.final_samples,
        final_duration_s=measurement.final_duration_s,
        longest_samples=measurement.longest_samples,
        longest_duration_s=longest,
        earliest_start_s=measurement.earliest_start_s,
        departures_after_hold=departures,
    )


@dataclass(frozen=True)
class TriggerOutcome:
    """Whether the dwell-triggered pulse fired, and early enough to leave room for the dwell it disturbs."""

    ok: bool
    triggered: bool
    pulse_start_s: float | None
    pulse_end_s: float | None
    reason: str | None
    """Why the case cannot count as a disturbance-recovery test, or ``None`` when the timing rule was met."""

    def __post_init__(self) -> None:
        """A met rule carries its timing and no reason; a failed one always names itself."""
        if self.ok and (self.reason is not None or not self.triggered):
            msg = f"a satisfied trigger rule carries no reason and did fire: {self}"
            raise ValueError(msg)
        if not self.ok and not self.reason:
            msg = "a failed trigger rule must give its reason"
            raise ValueError(msg)


def trigger_outcome(
    pulse: ForcePulse | None, *, activation_s: float, horizon_s: float, dwell_min_duration_s: float
) -> TriggerOutcome:
    """The verdict on a state-triggered pulse: it must have fired, and ended a full dwell before the horizon.

    A missing or too-late trigger is reported explicitly and never counts as a
    successful disturbance-recovery test, because the run never carried the
    disturbance the case exists to apply.
    """
    horizon_end = activation_s + horizon_s
    if pulse is None:
        return TriggerOutcome(
            ok=False,
            triggered=False,
            pulse_start_s=None,
            pulse_end_s=None,
            reason="the target dwell never qualified for the trigger hold, so the pulse never fired",
        )
    deadline = horizon_end - dwell_min_duration_s
    if pulse.end_s > deadline + _GRID_TOLERANCE_S:
        return TriggerOutcome(
            ok=False,
            triggered=True,
            pulse_start_s=pulse.start_s,
            pulse_end_s=pulse.end_s,
            reason=(
                f"the pulse ended at {pulse.end_s:.3f} s, too late for the {dwell_min_duration_s:.3f} s final "
                f"dwell to complete before the horizon at {horizon_end:.3f} s"
            ),
        )
    return TriggerOutcome(ok=True, triggered=True, pulse_start_s=pulse.start_s, pulse_end_s=pulse.end_s, reason=None)


# --- the evaluation configuration (I6) -------------------------------------------------------


@dataclass(frozen=True)
class ManualTriggerRule:
    """The state trigger that supersedes the development levels' fixed pulse start (plan section 6)."""

    hold_s: float
    """Continuous target dwell that arms the pulse."""
    duration_s: float
    magnitude_n: float

    def __post_init__(self) -> None:
        """Every part of the rule is a real positive quantity."""
        for name, value in (
            ("hold_s", self.hold_s),
            ("duration_s", self.duration_s),
            ("magnitude_n", self.magnitude_n),
        ):
            if not (value > 0 and math.isfinite(value)):
                msg = f"trigger.{name} must be positive and finite, got {value!r}"
                raise ValueError(msg)


@dataclass(frozen=True)
class ManualSimulationLimits:
    """The D3 abort of this protocol: the task configuration's own per-joint bound, never a relaxation."""

    velocity_abort: tuple[float, ...]

    def __post_init__(self) -> None:
        """Positive, finite bounds."""
        if not self.velocity_abort or any(not (math.isfinite(v) and v > 0) for v in self.velocity_abort):
            msg = f"simulation.velocity_abort must be positive finite per-joint bounds, got {self.velocity_abort!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ManualEvaluationConfig:
    """``configs/evaluations/task_1a_manual_dev_v1.toml``: the locked draws under the revised protocol."""

    name: str
    development: Path
    """The locked development levels, reused by reference so this pilot adds no seed of its own."""
    scenario: Path
    """The manual task configuration, which also carries the continuous dwell rule."""
    horizon_s: float
    """The D2 common evaluation horizon from activation (I5), never the demonstration's length."""
    trigger: ManualTriggerRule
    simulation: ManualSimulationLimits

    def __post_init__(self) -> None:
        """Named, pointed at development levels, and running for a real horizon."""
        if not self.name.strip():
            msg = "name must not be empty"
            raise ValueError(msg)
        if "confirmatory" in self.development.name:
            msg = f"the pilot evaluates development levels only, not {self.development.name!r}"
            raise ValueError(msg)
        if not (self.horizon_s > 0 and math.isfinite(self.horizon_s)):
            msg = f"horizon_s must be positive and finite, got {self.horizon_s!r}"
            raise ValueError(msg)


def load_manual_evaluation_config(path: Path) -> ManualEvaluationConfig:
    """Load the configuration and check it against the task configuration it names.

    The two cross-checks are the ones a wrong file would otherwise pass
    silently: an abort that quietly relaxes the approved limit, and a horizon
    too short for a force case to complete the dwell the pulse disturbs.
    """
    config = load_config(path, ManualEvaluationConfig)
    scenario = load_manual_scenario(config.scenario)
    canonical = tuple(float(v) for v in scenario.limits.velocity)
    if config.simulation.velocity_abort != canonical:
        msg = (
            f"simulation.velocity_abort {list(config.simulation.velocity_abort)} is not the canonical per-joint "
            f"limit {list(canonical)} of {config.scenario.name}: D3 applies the scenario's own bound to all "
            "evaluation, and the repetition pilot's relaxation does not carry over"
        )
        raise ValueError(msg)
    needed = config.trigger.hold_s + config.trigger.duration_s + scenario.task.dwell_min_duration_s
    if config.horizon_s <= needed:
        msg = (
            f"horizon_s {config.horizon_s} leaves a force case no room: the trigger hold ({config.trigger.hold_s} s), "
            f"the pulse ({config.trigger.duration_s} s) and the final dwell "
            f"({scenario.task.dwell_min_duration_s} s) need more than {needed} s"
        )
        raise ValueError(msg)
    return config


def manual_trigger(
    config: ManualEvaluationConfig, scenario: ManualScenarioConfig, *, direction_deg: float
) -> DwellTrigger:
    """The dwell trigger of one force case: the scenario's predicate with the configuration's hold and pulse.

    The predicate is read from the task configuration rather than restated
    here, so the rule that arms the pulse cannot drift from the one the takes
    were accepted under.
    """
    return DwellTrigger.from_polar(
        target=scenario.task.target,
        tolerance_m=scenario.task.tolerance,
        max_velocity_rad_s=scenario.task.dwell_max_velocity,
        hold_s=config.trigger.hold_s,
        duration_s=config.trigger.duration_s,
        magnitude_n=config.trigger.magnitude_n,
        direction_deg=direction_deg,
    )


# --- the conditions every run is keyed by ----------------------------------------------------


@dataclass(frozen=True)
class ManualRunConditions:
    """Everything the revised protocol decides about a run; the identity of these keys every cache.

    Historical scenario identities and replay caches cannot be reused here
    (plan section 6), and the reason is exactly this: a cached run is only the
    same run if the horizon, the dwell rule, the trigger, the abort, the
    trackers, the bound files, and the environment are all the same. Each of
    those is a field, so changing any of them changes the key rather than
    silently serving a run from a different experiment.
    """

    evaluation_name: str
    evaluation_file: str
    evaluation_sha256: str
    development_file: str
    development_sha256: str
    """The locked draws, bound by digest: the scenarios are only these if this file is."""
    scenario_file: str
    scenario_sha256: str
    horizon_s: float
    trigger_hold_s: float
    trigger_duration_s: float
    trigger_magnitude_n: float
    dwell_min_duration_s: float
    dwell_tolerance_m: float
    dwell_max_velocity_rad_s: float
    """The acquisition dwell rule, carried explicitly so a run records the rule it was judged by."""
    velocity_abort: tuple[float, ...]
    trackers: dict[str, str]
    """SHA-256 of each frozen tracker's gains, by name."""
    tracker_order: tuple[str, ...]
    """The trackers in evaluation order (the JSON form sorts mapping keys, so the order is explicit)."""
    scenario_ids: tuple[str, ...]
    warmup_s: float
    execution_identity: str

    def __post_init__(self) -> None:
        """Digests, quantities, and the evaluated set are each checked in turn."""
        self._check_digests()
        self._check_quantities()
        self._check_evaluated_set()

    def _check_digests(self) -> None:
        """Every bound file, the environment, and each tracker is named by a full digest."""
        for name in ("evaluation_sha256", "development_sha256", "scenario_sha256", "execution_identity"):
            value = getattr(self, name)
            if not is_hex(value, _SHA256_HEX):
                msg = f"{name} must be 64 lowercase hex characters, got {value!r}"
                raise ValueError(msg)
        for name, digest in self.trackers.items():
            if not is_hex(digest, _SHA256_HEX):
                msg = f"trackers[{name!r}] must be 64 lowercase hex characters, got {digest!r}"
                raise ValueError(msg)

    def _check_quantities(self) -> None:
        """The protocol's own quantities are real positive durations, rates, and bounds."""
        if not self.velocity_abort or any(not (math.isfinite(v) and v > 0) for v in self.velocity_abort):
            msg = f"velocity_abort must be positive finite per-joint bounds, got {self.velocity_abort!r}"
            raise ValueError(msg)
        for name, value in (
            ("horizon_s", self.horizon_s),
            ("trigger_hold_s", self.trigger_hold_s),
            ("trigger_duration_s", self.trigger_duration_s),
            ("trigger_magnitude_n", self.trigger_magnitude_n),
            ("dwell_min_duration_s", self.dwell_min_duration_s),
            ("dwell_tolerance_m", self.dwell_tolerance_m),
            ("dwell_max_velocity_rad_s", self.dwell_max_velocity_rad_s),
        ):
            if not (value > 0 and math.isfinite(value)):
                msg = f"{name} must be positive and finite, got {value!r}"
                raise ValueError(msg)
        if not (math.isfinite(self.warmup_s) and self.warmup_s >= 0):
            msg = f"warmup_s must be finite and non-negative, got {self.warmup_s!r}"
            raise ValueError(msg)

    def _check_evaluated_set(self) -> None:
        """Each case is attempted once per tracker, in a declared order."""
        if not self.trackers or not self.scenario_ids or len(set(self.scenario_ids)) != len(self.scenario_ids):
            msg = "conditions need at least one tracker and distinct scenario ids"
            raise ValueError(msg)
        if len(self.tracker_order) != len(self.trackers) or set(self.tracker_order) != set(self.trackers):
            msg = "tracker_order must list every tracker exactly once"
            raise ValueError(msg)

    @property
    def identity(self) -> str:
        """SHA-256 of the canonical JSON of these conditions (the cache key of every run under them)."""
        return sha256_bytes(canonical_json(to_mapping(self)).encode("utf-8"))

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        """Every ``(scenario_id, tracker)`` in evaluation order: scenario major, tracker minor."""
        return tuple((scenario, tracker) for scenario in self.scenario_ids for tracker in self.tracker_order)


def _relative(path: Path, root: Path) -> str:
    """``path`` as a repository-relative POSIX path; recorded evidence never carries a machine path."""
    resolved, base = path.resolve(), root.resolve()
    if not resolved.is_relative_to(base):
        msg = f"{path} lies outside the repository {root}"
        raise ValueError(msg)
    return resolved.relative_to(base).as_posix()


def manual_conditions(
    config: ManualEvaluationConfig,
    evaluation_file: Path,
    *,
    scenario_ids: Sequence[str],
    warmup_s: float,
    execution_identity: str,
    root: Path,
) -> ManualRunConditions:
    """Assemble the conditions of one evaluation from the files it binds and the environment it runs in.

    The dwell rule is copied from the task configuration and the tracker
    digests are derived from the frozen gains, so neither can be passed in
    wrongly; the caller supplies only what it has itself verified, namely the
    cases to run, the warm-up of the entry, and the environment identity.
    """
    scenario = load_manual_scenario(config.scenario)
    return ManualRunConditions(
        evaluation_name=config.name,
        evaluation_file=_relative(evaluation_file, root),
        evaluation_sha256=sha256_file(evaluation_file),
        development_file=_relative(config.development, root),
        development_sha256=sha256_file(config.development),
        scenario_file=_relative(config.scenario, root),
        scenario_sha256=sha256_file(config.scenario),
        horizon_s=config.horizon_s,
        trigger_hold_s=config.trigger.hold_s,
        trigger_duration_s=config.trigger.duration_s,
        trigger_magnitude_n=config.trigger.magnitude_n,
        dwell_min_duration_s=scenario.task.dwell_min_duration_s,
        dwell_tolerance_m=scenario.task.tolerance,
        dwell_max_velocity_rad_s=scenario.task.dwell_max_velocity,
        velocity_abort=config.simulation.velocity_abort,
        trackers={name: frozen_baseline_digest(name) for name in RECOVERY_TRACKERS},
        tracker_order=tuple(RECOVERY_TRACKERS),
        scenario_ids=tuple(scenario_ids),
        warmup_s=warmup_s,
        execution_identity=execution_identity,
    )


# --- the verdict of one run ------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratedReferenceReport:
    """The generated reference judged by the same rules as the actual motion (plan section 6).

    A command that already leaves the joint limits is infeasible on its own
    terms, and its endpoint cannot be computed: the forward kinematics would
    clamp the posture and answer for a trajectory the generator never
    commanded. The workspace and dwell results are therefore ``None`` in that
    case -- not evaluated -- rather than a fabricated failure.
    """

    within_position_limits: bool
    within_speed_limits: bool
    within_workspace: bool | None
    dwell: ManualDwellReport | None

    def __post_init__(self) -> None:
        """Endpoint-derived results exist exactly when the posture was inside its limits."""
        evaluated = self.within_workspace is not None
        if evaluated != (self.dwell is not None):
            msg = f"the workspace and dwell results are evaluated together, got {self}"
            raise ValueError(msg)
        if evaluated and not self.within_position_limits:
            msg = "an out-of-limit command is never evaluated through forward kinematics"
            raise ValueError(msg)

    @property
    def ok(self) -> bool:
        """Whether the generated command is itself a valid, holding trajectory, on every measured count."""
        return (
            self.within_position_limits
            and self.within_speed_limits
            and self.within_workspace is True
            and self.dwell is not None
            and self.dwell.ok
        )


@dataclass(frozen=True)
class ManualRunOutcome:
    """What one run achieved, and why it did not succeed when it did not."""

    completed: bool
    dwell: ManualDwellReport
    """The dwell over the whole active segment: earliest, longest, final, and departures after holds."""
    post_pulse_dwell: ManualDwellReport | None
    """The dwell measured from the pulse end, which is what a force case must satisfy."""
    generated: GeneratedReferenceReport | None
    """``None`` for a replay run, which carries no readout."""
    saturation_fraction: float
    torque_rms: float | None
    trigger: TriggerOutcome | None
    success: bool
    reason: str | None

    def __post_init__(self) -> None:
        """Success and the reason are exactly complementary."""
        if self.success != (self.reason is None):
            msg = f"success must mean no reason, got success={self.success} reason={self.reason!r}"
            raise ValueError(msg)


def _dwell_predicate(scenario: ManualScenarioConfig) -> DwellPredicate:
    """The acquisition rule of this task, applied unchanged to evaluation runs."""
    return DwellPredicate(
        tolerance_m=scenario.task.tolerance,
        max_velocity_rad_s=scenario.task.dwell_max_velocity,
        min_duration_s=scenario.task.dwell_min_duration_s,
        min_samples=scenario.dwell_min_samples,
    )


def _generated_report(
    arrays: RunArrays, *, scenario: ManualScenarioConfig, activation_s: float, predicate: DwellPredicate
) -> GeneratedReferenceReport | None:
    """Judge the generated command, or ``None`` when the run carries no readout.

    The readout is NaN before activation, so everything here is measured over
    the active samples only; feeding the hold's NaNs to the dwell rule would
    raise rather than report.
    """
    readout = arrays.arrays.get("generator_output_q")
    if readout is None:
        return None
    run_t = np.asarray(arrays.arrays["t"], dtype=np.float64)
    active = run_t >= activation_s - _GRID_TOLERANCE_S
    generated_q = np.asarray(readout, dtype=np.float64)[active]
    generated_dq = np.asarray(arrays.arrays["dq_desired"], dtype=np.float64)[active]
    if not generated_q.shape[0]:
        # The run ended before activation, so the readout never produced anything to judge.
        return None
    if not np.all(np.isfinite(generated_q)):
        msg = "the generated reference is not finite over the active segment"
        raise ValueError(msg)
    lower = np.array([link.q_min for link in scenario.robot.links], dtype=np.float64)
    upper = np.array([link.q_max for link in scenario.robot.links], dtype=np.float64)
    speed = np.asarray(scenario.limits.velocity, dtype=np.float64)
    within_limits = bool(np.all(generated_q >= lower) and np.all(generated_q <= upper))
    within_speed = bool(np.all(np.abs(generated_dq) <= speed))
    if not within_limits:
        # Forward kinematics would clamp the posture and answer for a trajectory that was never
        # commanded, so the endpoint-derived checks are left unevaluated rather than fabricated.
        return GeneratedReferenceReport(
            within_position_limits=False, within_speed_limits=within_speed, within_workspace=None, dwell=None
        )
    tip = manual_endpoint_positions(scenario, generated_q)
    dwell = manual_dwell_report(
        run_t[active],
        tip,
        generated_dq,
        target=np.asarray(scenario.task.target, dtype=np.float64),
        predicate=predicate,
    )
    return GeneratedReferenceReport(
        within_position_limits=True,
        within_speed_limits=within_speed,
        within_workspace=bool(np.all(np.hypot(tip[:, 0], tip[:, 1]) <= scenario.limits.endpoint_radius)),
        dwell=dwell,
    )


def _reason(
    termination: Termination,
    *,
    completed: bool,
    dwell_ok: bool,
    generated: GeneratedReferenceReport | None,
    trigger: TriggerOutcome | None,
    saturation_fraction: float,
    bound: float,
) -> str | None:
    """Why this run did not succeed, in the vocabulary the sweep reports, or ``None`` when it did."""
    if termination.kind == "divergence":
        return "divergence"
    if termination.kind == "limit_violation":
        return f"limit_violation:{termination.limit}"
    if not completed:
        return f"incomplete_horizon:{termination.kind}"
    if trigger is not None and not trigger.ok:
        return f"trigger:{trigger.reason}"
    if not dwell_ok:
        return "dwell:no_final_dwell"
    if generated is not None and not generated.ok:
        # Only what was actually measured is named; an unevaluated check is not reported as a failure.
        missed = [
            name
            for name, ok in (
                ("position_limits", generated.within_position_limits),
                ("speed_limits", generated.within_speed_limits),
                ("workspace", generated.within_workspace is not False),
                ("dwell", generated.dwell is None or generated.dwell.ok),
            )
            if not ok
        ]
        return "generated_reference:" + ",".join(missed)
    if saturation_fraction > bound:
        return "saturation"
    return None


def manual_run_outcome(
    arrays: RunArrays,
    termination: Termination,
    *,
    scenario: ManualScenarioConfig,
    activation_s: float,
    horizon_s: float,
    pulse: ForcePulse | None = None,
    force_case: bool = False,
) -> ManualRunOutcome:
    """Judge one run: the horizon, the dwell, the generated reference, the pulse, and the effort.

    ``force_case`` says the scenario was supposed to carry a pulse, so a run
    that never triggered one is reported as a failed disturbance test rather
    than as an ordinary success; ``pulse`` is the pulse that actually fired.
    """
    run_t = np.asarray(arrays.arrays["t"], dtype=np.float64)
    active = run_t >= activation_s - _GRID_TOLERANCE_S
    predicate = _dwell_predicate(scenario)
    target = np.asarray(scenario.task.target, dtype=np.float64)
    dwell = manual_dwell_report(
        run_t,
        np.asarray(arrays.arrays["tip"], dtype=np.float64),
        np.asarray(arrays.arrays["dq"], dtype=np.float64),
        target=target,
        predicate=predicate,
        since_s=activation_s,
    )
    post_pulse = (
        None
        if pulse is None
        else manual_dwell_report(
            run_t,
            np.asarray(arrays.arrays["tip"], dtype=np.float64),
            np.asarray(arrays.arrays["dq"], dtype=np.float64),
            target=target,
            predicate=predicate,
            since_s=pulse.end_s,
        )
    )
    trigger = (
        trigger_outcome(
            pulse,
            activation_s=activation_s,
            horizon_s=horizon_s,
            dwell_min_duration_s=scenario.task.dwell_min_duration_s,
        )
        if force_case
        else None
    )
    generated = _generated_report(arrays, scenario=scenario, activation_s=activation_s, predicate=predicate)
    flags = np.asarray(arrays.arrays["saturation"], dtype=np.float64)[active]
    saturation_fraction = float(np.mean(flags)) if flags.shape[0] else 0.0
    torque = np.asarray(arrays.arrays.get("tau_applied", arrays.arrays["tau_requested"]), dtype=np.float64)[active]
    torque_rms = float(np.sqrt(np.mean(np.sum(torque**2, axis=1)))) if torque.shape[0] else None
    completed = horizon_completed(run_t, termination, activation_s=activation_s, horizon_s=horizon_s)
    deciding = dwell if post_pulse is None else post_pulse
    reason = _reason(
        termination,
        completed=completed,
        dwell_ok=deciding.ok,
        generated=generated,
        trigger=trigger,
        saturation_fraction=saturation_fraction,
        bound=SATURATION_BOUND,
    )
    return ManualRunOutcome(
        completed=completed,
        dwell=dwell,
        post_pulse_dwell=post_pulse,
        generated=generated,
        saturation_fraction=saturation_fraction,
        torque_rms=torque_rms,
        trigger=trigger,
        success=reason is None,
        reason=reason,
    )


# --- the evidence of one model ---------------------------------------------------------------


@dataclass(frozen=True)
class ManualFitBinding:
    """The fit a model's runs were produced from, bound by the study's own identity and digests."""

    identity: str
    configuration: str
    arm: str
    solver_alpha: float
    recipe_sha256: str
    weights_sha256: str

    def __post_init__(self) -> None:
        """Digests are well formed and the ridge parameter is a real positive number."""
        for name in ("identity", "recipe_sha256", "weights_sha256"):
            value = getattr(self, name)
            if not is_hex(value, _SHA256_HEX):
                msg = f"{name} must be 64 lowercase hex characters, got {value!r}"
                raise ValueError(msg)
        if not (self.solver_alpha > 0 and math.isfinite(self.solver_alpha)):
            msg = f"solver_alpha must be positive and finite, got {self.solver_alpha!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ManualModelEvidence:
    """Everything one model produced under one protocol, with its counts re-derived from its runs."""

    identity: str
    conditions: ManualRunConditions
    fit: ManualFitBinding | None
    assignment: str | None
    """The parent this model is paired against; ``None`` for the all-ten arm, which has no single parent."""
    replay_bank: str | None
    status: str
    pairs: tuple[ManualPairRecord, ...]
    n_pairs: int
    n_completed: int
    n_infeasible: int
    n_unexecuted: int
    schema_version: int = EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """The recorded summary must be exactly what the pairs say, and the pairs the conditions' own."""
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            msg = f"unsupported evidence schema {self.schema_version}"
            raise ValueError(msg)
        if self.status not in MODEL_STATUSES:
            msg = f"status must be one of {MODEL_STATUSES}, got {self.status!r}"
            raise ValueError(msg)
        if [(p.scenario_id, p.tracker) for p in self.pairs] != list(self.conditions.pairs):
            msg = "the pairs are every scenario and tracker of the conditions, in evaluation order"
            raise ValueError(msg)
        counts = Counter(p.status for p in self.pairs)
        recorded = (self.n_pairs, self.n_completed, self.n_infeasible, self.n_unexecuted)
        actual = (len(self.pairs), counts["completed"], counts["infeasible"], counts["unexecuted"])
        if recorded != actual:
            msg = f"recorded counts {recorded} contradict the pairs {actual}"
            raise ValueError(msg)
        if (self.status == "feasible") != (self.n_completed == self.n_pairs):
            msg = "a feasible model completed every pair"
            raise ValueError(msg)


# --- storing and rebuilding the evidence -----------------------------------------------------


def manual_bank_to_json(bank: ManualReplayBank) -> str:
    """The canonical JSON of one replay bank."""
    return canonical_json(to_mapping(bank)) + "\n"


def manual_evidence_to_json(evidence: ManualModelEvidence) -> str:
    """The canonical JSON of one model's evidence."""
    return canonical_json(to_mapping(evidence)) + "\n"


def load_manual_replay_bank(path: Path) -> ManualReplayBank:
    """Strictly rebuild a replay bank from its manifest."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), ManualReplayBank)


def load_manual_model_evidence(path: Path) -> ManualModelEvidence:
    """Strictly rebuild one model's evidence from its manifest."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), ManualModelEvidence)


def _write_atomic(path: Path, text: str) -> None:
    """Stage beside the target and replace it, so a reader never sees a half-written manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.staging-{os.getpid()}")
    staged.write_text(text, encoding="utf-8")
    staged.replace(path)


def _manifest_dir(store: StorageRoot, directory_uri: str) -> Path:
    return store.path(f"{directory_uri}/manifest.json", mode="write").parent


def _install_manifest(store: StorageRoot, directory_uri: str, text: str) -> ArtifactReference:
    """Install a content-addressed manifest; identical content is reused and differing content refused."""
    data = text.encode("utf-8")
    digest = sha256_bytes(data)
    uri = f"{directory_uri}/manifest-{digest[:_SHORT]}.json"
    target = store.path(uri, mode="write")
    if target.exists():
        if sha256_file(target) != digest:
            msg = f"{uri} exists with other content; completed evidence is never overwritten"
            raise ValueError(msg)
    else:
        _write_atomic(target, text)
    return ArtifactReference(uri, digest, len(data))


def _existing_manifest(store: StorageRoot, directory_uri: str) -> Path | None:
    """The manifest already stored for these conditions, if any; two would be a corrupted store."""
    directory = _manifest_dir(store, directory_uri)
    if not directory.exists():
        return None
    manifests = sorted(directory.glob("manifest-*.json"))
    if len(manifests) > 1:
        msg = f"{directory} holds {len(manifests)} manifests; completed evidence is written once"
        raise ValueError(msg)
    return manifests[0] if manifests else None


def model_uri(identity: str) -> str:
    """The store directory of one model's evidence under one protocol."""
    return f"{REPORTS_PREFIX}/model/{identity}"


class _ManualProgress:
    """The run-granular progress of one manifest, re-verified against the store before it is trusted."""

    def __init__(self, store: StorageRoot, directory_uri: str, identity: str) -> None:
        """Load any progress already recorded here, refusing one that belongs elsewhere."""
        self.store = store
        self.identity = identity
        self.path = store.path(f"{directory_uri}/{PROGRESS_FILE}", mode="write")
        self.pairs: list[ManualPairRecord] = []
        if not self.path.exists():
            return
        mapping = cast("dict[str, object]", json.loads(self.path.read_text(encoding="utf-8")))
        if mapping.get("identity") != identity or mapping.get("schema_version") != EVALUATION_SCHEMA_VERSION:
            msg = f"{self.path} belongs to another evaluation or schema"
            raise ValueError(msg)
        self.pairs = [
            from_mapping(cast("dict[str, object]", p), ManualPairRecord) for p in cast("list[object]", mapping["pairs"])
        ]
        self._verify_runs()

    def _verify_runs(self) -> None:
        """A completed run is served only while its stored arrays still match what the record claims."""
        for pair in self.pairs:
            if pair.run is None:
                continue
            path = self.store.path(pair.run.uri, mode="read")
            if path.stat().st_size != pair.run.size or sha256_file(path) != pair.run.sha256:
                msg = f"run {pair.run.artifact_id} of {pair.scenario_id} [{pair.tracker}] no longer matches its record"
                raise ValueError(msg)

    def get(self, scenario_id: str, tracker: str, arm: str) -> ManualPairRecord | None:
        """The pair already recorded for this scenario, tracker, and arm, if any."""
        for pair in self.pairs:
            if (pair.scenario_id, pair.tracker, pair.arm) == (scenario_id, tracker, arm):
                return pair
        return None

    def add(self, pair: ManualPairRecord) -> None:
        """Record one completed pair; recorded evidence is never replaced."""
        if self.get(pair.scenario_id, pair.tracker, pair.arm) is not None:
            msg = f"{pair.scenario_id} [{pair.tracker}] {pair.arm} is already recorded; completed evidence is immutable"
            raise ValueError(msg)
        self.pairs.append(pair)
        self.save()

    def save(self) -> None:
        """Write the progress atomically, so an interruption never leaves a half-written file."""
        mapping = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "identity": self.identity,
            "pairs": [to_mapping(pair) for pair in self.pairs],
        }
        _write_atomic(self.path, json.dumps(mapping, indent=1, sort_keys=True))


# --- the replay baselines of one demonstration -----------------------------------------------


def replay_bank_uri(conditions: ManualRunConditions, assignment: str) -> str:
    """The store directory of one parent's replay baselines under one protocol."""
    return f"{REPORTS_PREFIX}/replay/{_bank_identity(conditions, assignment)}"


def _bank_identity(conditions: ManualRunConditions, assignment: str) -> str:
    """Conditions and parent together: the same protocol over another recording is another bank."""
    return sha256_bytes(f"{conditions.identity}:{assignment}".encode("ascii"))


@dataclass(frozen=True)
class ManualRunArtifact:
    """Where one persisted run lives and what it contains."""

    artifact_id: str
    uri: str
    sha256: str
    size: int
    arrays_sha256: str


@dataclass(frozen=True)
class ManualPairRecord:
    """One (scenario, tracker) run of one arm, with the posture it started from and how it was judged."""

    index: int
    scenario_id: str
    kind: str
    tracker: str
    arm: str
    status: str
    initial_q: tuple[float, ...]
    outcome: ManualRunOutcome | None = None
    run: ManualRunArtifact | None = None
    pulse_start_s: float | None = None
    """When the pulse actually fired on the run clock, or ``None`` when none did."""

    def __post_init__(self) -> None:
        """A simulated pair carries the run it produced and the verdict it was given."""
        if self.arm not in ("rc", "replay"):
            msg = f"arm must be 'rc' or 'replay', got {self.arm!r}"
            raise ValueError(msg)
        simulated = self.status in ("completed", "infeasible")
        if simulated != (self.run is not None) or simulated != (self.outcome is not None):
            msg = f"{self.scenario_id} [{self.tracker}] {self.arm}: a simulated pair carries its run and outcome"
            raise ValueError(msg)


@dataclass(frozen=True)
class ManualReplayBank:
    """Every direct-replay baseline of one demonstration under one set of conditions."""

    conditions: ManualRunConditions
    assignment: str
    pairs: tuple[ManualPairRecord, ...]

    @property
    def identity(self) -> str:
        """The key this bank is stored and served under."""
        return _bank_identity(self.conditions, self.assignment)


class ManualEvaluationRunner:
    """Runs the manual protocol's scenarios, paired against direct replay of each demonstration.

    Every scenario is attempted independently from a fresh reset, so an unsafe
    run aborts alone and the next scenario still runs (D6); the sweep never
    stops at the first infeasible case the way the predecessor's model sweep
    did.
    """

    def __init__(
        self,
        *,
        store: StorageRoot,
        inputs: ManualFitInputs,
        config: ManualEvaluationConfig,
        evaluation_file: Path,
        scenarios: Sequence[RobustnessScenario],
        trackers: dict[str, TrackerConfig],
        root: Path,
        execution: ExecutionRecord,
        provenance: ProvenanceRecord,
        simulate_fn: SimulateFn | None = None,
        log: Callable[[str], None] = lambda _message: None,
        license_label: str = "LicenseRef-Private",
        access: str = "private",
        command: str = "python -m arm_rc_ctrl.experiments.manual_evaluation",
    ) -> None:
        """Bind the store, the frozen study inputs, and the protocol this sweep runs under."""
        self.store = store
        self.inputs = inputs
        self.config = config
        self.evaluation_file = evaluation_file
        self.scenarios = tuple(scenarios)
        self.trackers = dict(trackers)
        self.root = root
        self.execution = execution
        self.provenance = provenance
        self.log = log
        self.simulate: SimulateFn = simulate if simulate_fn is None else simulate_fn
        self.license_label = license_label
        self.access = access
        self.command = command
        self.scenario = load_manual_scenario(config.scenario)
        self._banks: dict[str, ManualReplayBank] = {}
        self._models: dict[str, ManualModelEvidence] = {}

    def conditions(self, warmup_s: float) -> ManualRunConditions:
        """The protocol conditions at one warm-up; banks are shared by every model that shares it."""
        return manual_conditions(
            self.config,
            self.evaluation_file,
            scenario_ids=tuple(case.scenario_id for case in self.scenarios),
            warmup_s=warmup_s,
            execution_identity=self.execution.identity,
            root=self.root,
        )

    def _start(self, case: RobustnessScenario) -> tuple[float, ...]:
        """The perturbed reset posture: every take begins at the configured one, so the offsets are shared."""
        return case.initial_q(self.scenario.task.initial_q)

    def _samples(self, assignment: str) -> SampleSet:
        """The locked demonstration of one bank position."""
        return self.inputs.samples[self.inputs.sources[assignment].artifact_id]

    def _replay_pair(
        self, index: int, case: RobustnessScenario, tracker: str, *, assignment: str, warmup_s: float
    ) -> ManualPairRecord:
        """Replay one demonstration through one tracker under one scenario, from a fresh reset."""
        start = self._start(case)
        samples = self._samples(assignment)
        held = HeldTaskReference.from_samples(
            samples,
            activation_s=warmup_s,
            interpolation=cast("Any", self.inputs.preprocessing.interpolation),
            hold=np.asarray(start, dtype=np.float64),
        )
        controller = LimitedTracker(cast("Any", held), self.trackers[tracker], self.scenario.limits.torque)
        trigger = (
            None
            if case.pulse is None
            else manual_trigger(self.config, self.scenario, direction_deg=case.direction_deg or 0.0)
        )
        fired: list[ForcePulse] = []
        arrays, termination = self.simulate(
            self.scenario,
            controller,
            duration_s=warmup_s + self.config.horizon_s,
            initial_q=start,
            force_trigger=trigger,
            triggered=fired,
            velocity_abort=self.config.simulation.velocity_abort,
        )
        pulse = fired[0] if fired else None
        outcome = manual_run_outcome(
            arrays,
            termination,
            scenario=self.scenario,
            activation_s=warmup_s,
            horizon_s=self.config.horizon_s,
            pulse=pulse,
            force_case=case.pulse is not None,
        )
        run = self._persist(
            arrays,
            termination,
            outcome=outcome,
            case=case,
            tracker=tracker,
            arm="replay",
            assignment=assignment,
            warmup_s=warmup_s,
            pulse=pulse,
        )
        return ManualPairRecord(
            index=index,
            scenario_id=case.scenario_id,
            kind=str(case.kind),
            tracker=tracker,
            arm="replay",
            status="completed" if outcome.success else "infeasible",
            initial_q=start,
            outcome=outcome,
            run=run,
            pulse_start_s=None if pulse is None else pulse.start_s,
        )

    def _persist(
        self,
        arrays: RunArrays,
        termination: Termination,
        *,
        outcome: ManualRunOutcome,
        case: RobustnessScenario,
        tracker: str,
        arm: str,
        assignment: str,
        warmup_s: float,
        pulse: ForcePulse | None,
    ) -> ManualRunArtifact:
        """Store the run with the disturbance that actually fired, not the one the levels prescribed."""
        criteria = {
            "completed": outcome.completed,
            "dwell": outcome.dwell.ok if outcome.post_pulse_dwell is None else outcome.post_pulse_dwell.ok,
            "generated_reference": outcome.generated is None or outcome.generated.ok,
            "trigger": outcome.trigger is None or outcome.trigger.ok,
        }
        pointer, summary, directory = write_run(
            self.store,
            arrays,
            kind="simulation",
            method=f"{arm}+{tracker}",
            scenario=self.scenario.name,
            control_period_s=self.scenario.timing.dt,
            duration_s=warmup_s + self.config.horizon_s,
            target=self.scenario.task.target,
            task_code=(),
            disturbances=() if pulse is None else (pulse.to_disturbance(),),
            termination=termination,
            outcome=Outcome(termination, criteria),
            provenance=self.provenance,
            license_label=self.license_label,
            access=cast("Any", self.access),
            command=self.command,
            sources=(self.inputs.sources[assignment].artifact_id,),
            activation_s=warmup_s,
            reuse_identical=True,
            notes=f"{self.config.name} {arm} arm: {case.scenario_id} [{tracker}] of {assignment}.",
        )
        del directory
        return ManualRunArtifact(
            artifact_id=pointer.artifact.artifact_id,
            uri=pointer.artifact.payload.uri,
            sha256=pointer.artifact.payload.sha256,
            size=pointer.artifact.payload.size,
            arrays_sha256=summary.arrays_sha256,
        )

    def model_identity(self, entry: StudyModel, *, warmup_s: float) -> str:
        """The key one model's evidence is stored under, without refitting it.

        The study's own fit identity is the cache key of the fit, so the model
        key is derivable from the manifest alone.
        """
        return sha256_bytes(f"{entry.fit_identity}:{self.conditions(warmup_s).identity}".encode("ascii"))

    def _configuration(self, entry: StudyModel) -> StudyConfiguration:
        """The inherited configuration of ``entry``, which carries its evaluation-side estimator cutoffs."""
        return self.inputs.configuration(entry)

    def _controllers(
        self, entry: StudyModel, cached: CachedFit, warmup_s: float
    ) -> dict[str, GeneratorTrackingController]:
        """One generator-tracking controller per tracker, holding the reset posture until activation."""
        configuration = self._configuration(entry)
        lower = np.array([link.q_min for link in self.scenario.robot.links], dtype=np.float64)
        upper = np.array([link.q_max for link in self.scenario.robot.links], dtype=np.float64)
        controllers: dict[str, GeneratorTrackingController] = {}
        for name, gains in self.trackers.items():
            estimator = CausalDerivativeEstimator(
                EstimatorConfig(
                    nominal_dt_s=self.scenario.timing.dt,
                    velocity_cutoff_hz=configuration.velocity_cutoff_hz,
                    acceleration_cutoff_hz=configuration.acceleration_cutoff_hz,
                ),
                self.inputs.dof,
            )
            generator = RcTargetGenerator(
                cached.model,
                cached.recipe.encoder(),
                estimator,
                position_bounds=(lower, upper),
                output=cached.recipe.output,
            )
            controllers[name] = GeneratorTrackingController(
                generator, gains, self.scenario.limits.torque, hold_until_s=warmup_s
            )
        return controllers

    def _rc_pair(
        self,
        index: int,
        case: RobustnessScenario,
        tracker: str,
        *,
        controller: GeneratorTrackingController,
        cached: CachedFit,
        assignment: str | None,
        warmup_s: float,
    ) -> ManualPairRecord:
        """Run one scenario under one tracker from a fresh reset, whatever earlier scenarios did."""
        start = self._start(case)
        trigger = (
            None
            if case.pulse is None
            else manual_trigger(self.config, self.scenario, direction_deg=case.direction_deg or 0.0)
        )
        fired: list[ForcePulse] = []
        channels = RESIDUAL_CHANNELS if cached.recipe.output == "increment" else GENERATOR_CHANNELS
        arrays, termination = self.simulate(
            self.scenario,
            controller,
            duration_s=warmup_s + self.config.horizon_s,
            initial_q=start,
            force_trigger=trigger,
            triggered=fired,
            channels=channels,
            velocity_abort=self.config.simulation.velocity_abort,
        )
        pulse = fired[0] if fired else None
        outcome = manual_run_outcome(
            arrays,
            termination,
            scenario=self.scenario,
            activation_s=warmup_s,
            horizon_s=self.config.horizon_s,
            pulse=pulse,
            force_case=case.pulse is not None,
        )
        run = self._persist(
            arrays,
            termination,
            outcome=outcome,
            case=case,
            tracker=tracker,
            arm="rc",
            assignment=assignment or ASSIGNMENT_ALL,
            warmup_s=warmup_s,
            pulse=pulse,
        )
        return ManualPairRecord(
            index=index,
            scenario_id=case.scenario_id,
            kind=str(case.kind),
            tracker=tracker,
            arm="rc",
            status="completed" if outcome.success else "infeasible",
            initial_q=start,
            outcome=outcome,
            run=run,
            pulse_start_s=None if pulse is None else pulse.start_s,
        )

    def evaluate(self, entry: StudyModel, *, warmup_s: float) -> ManualModelEvidence:
        """Evaluate one model over every scenario and tracker, paired against its parent's replay baselines.

        Every scenario is attempted from a fresh reset: an unsafe run aborts
        alone and the next one still runs (D6), so per-class success rates
        exist. The predecessor's sweep stopped at the first infeasible pair and
        marked the rest unexecuted; nothing here does.
        """
        conditions = self.conditions(warmup_s)
        cached = ManualFitStore(self.store).fit_or_load(entry, self.inputs)
        identity = sha256_bytes(f"{cached.record.identity}:{conditions.identity}".encode("ascii"))
        existing = self._models.get(identity)
        if existing is not None:
            return existing
        uri = model_uri(identity)
        stored = _existing_manifest(self.store, uri)
        if stored is not None:
            evidence = load_manual_model_evidence(stored)
            if evidence.identity != identity:
                msg = f"{stored} holds the evidence {evidence.identity[:_SHORT]}, not {identity[:_SHORT]}"
                raise ValueError(msg)
            self._models[identity] = evidence
            return evidence
        assignment = entry.arm.assignment
        bank = None if assignment is None else self.replay_bank(assignment, warmup_s=warmup_s)
        controllers = self._controllers(entry, cached, warmup_s)
        progress = _ManualProgress(self.store, uri, identity)
        pairs: list[ManualPairRecord] = []
        for index, case in enumerate(self.scenarios):
            for tracker in conditions.tracker_order:
                recorded = progress.get(case.scenario_id, tracker, "rc")
                if recorded is not None:
                    pairs.append(recorded)
                    continue
                self.log(f"{entry.label}: {case.scenario_id} [{tracker}]")
                pair = self._rc_pair(
                    index,
                    case,
                    tracker,
                    controller=controllers[tracker],
                    cached=cached,
                    assignment=assignment,
                    warmup_s=warmup_s,
                )
                progress.add(pair)
                pairs.append(pair)
        counts = Counter(pair.status for pair in pairs)
        evidence = ManualModelEvidence(
            identity=identity,
            conditions=conditions,
            fit=ManualFitBinding(
                identity=cached.record.identity,
                configuration=entry.configuration,
                arm=entry.arm.label,
                solver_alpha=cached.recipe.solver_alpha,
                recipe_sha256=cached.record.recipe_sha256,
                weights_sha256=cached.record.weights_sha256,
            ),
            assignment=assignment,
            replay_bank=None if bank is None else bank.identity,
            status="feasible" if counts["completed"] == len(pairs) else "infeasible",
            pairs=tuple(pairs),
            n_pairs=len(pairs),
            n_completed=counts["completed"],
            n_infeasible=counts["infeasible"],
            n_unexecuted=counts["unexecuted"],
        )
        _install_manifest(self.store, uri, manual_evidence_to_json(evidence))
        self._models[identity] = evidence
        return evidence

    def replay_bank(self, assignment: str, *, warmup_s: float) -> ManualReplayBank:
        """Every replay baseline of one demonstration, run once and shared by the models that pair against it."""
        conditions = self.conditions(warmup_s)
        identity = _bank_identity(conditions, assignment)
        cached = self._banks.get(identity)
        if cached is not None:
            return cached
        uri = replay_bank_uri(conditions, assignment)
        stored = _existing_manifest(self.store, uri)
        if stored is not None:
            bank = load_manual_replay_bank(stored)
            if bank.identity != identity:
                msg = f"{stored} holds the bank {bank.identity[:_SHORT]}, not {identity[:_SHORT]}"
                raise ValueError(msg)
            self._banks[identity] = bank
            return bank
        progress = _ManualProgress(self.store, uri, identity)
        pairs: list[ManualPairRecord] = []
        for index, case in enumerate(self.scenarios):
            for tracker in conditions.tracker_order:
                recorded = progress.get(case.scenario_id, tracker, "replay")
                if recorded is not None:
                    pairs.append(recorded)
                    continue
                self.log(f"replay {identity[:_SHORT]}: {case.scenario_id} [{tracker}] of {assignment}")
                pair = self._replay_pair(index, case, tracker, assignment=assignment, warmup_s=warmup_s)
                progress.add(pair)
                pairs.append(pair)
        bank = ManualReplayBank(conditions=conditions, assignment=assignment, pairs=tuple(pairs))
        _install_manifest(self.store, uri, manual_bank_to_json(bank))
        self._banks[identity] = bank
        return bank
