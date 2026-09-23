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

import argparse
import importlib
import io
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path  # a run-time import: the configuration loader resolves field types at run time
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np

from arm_rc_ctrl.config import from_mapping, load_config, to_mapping
from arm_rc_ctrl.controllers.adapter import GeneratorTrackingController
from arm_rc_ctrl.controllers.estimator import CausalDerivativeEstimator, EstimatorConfig
from arm_rc_ctrl.controllers.reference import DemonstrationReference
from arm_rc_ctrl.controllers.tracking import LimitedTracker
from arm_rc_ctrl.data.manual import DwellPredicate, continuous_dwell, dwell_runs
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario, manual_endpoint_positions
from arm_rc_ctrl.data.records import load_record, write_record
from arm_rc_ctrl.execution import collect_execution, require_canonical
from arm_rc_ctrl.experiments.baselines import frozen_baseline_digest, load_frozen_baseline
from arm_rc_ctrl.experiments.manual_fits import EvidenceIntegrityError, ManualFitStore, recipe_mismatches
from arm_rc_ctrl.experiments.manual_numerics import ManualStudyContext
from arm_rc_ctrl.experiments.manual_recipes import ASSIGNMENTS
from arm_rc_ctrl.experiments.manual_study import EXPERIMENT_LABEL
from arm_rc_ctrl.experiments.perturbations import load_development_robustness, robustness_scenarios
from arm_rc_ctrl.experiments.recovery_search import RECOVERY_TRACKERS
from arm_rc_ctrl.experiments.run_record import RUN_ARRAYS_FILE, RUN_SUMMARY_FILE, RunSummary, write_run
from arm_rc_ctrl.experiments.simulation import GENERATOR_CHANNELS, RESIDUAL_CHANNELS, DwellTrigger, simulate
from arm_rc_ctrl.experiments.termination import Outcome
from arm_rc_ctrl.metrics.recovery import SATURATION_BOUND
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    canonical_json,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
    sha256_bytes,
    sha256_file,
)
from arm_rc_ctrl.rc.esn import ensure_single_thread
from arm_rc_ctrl.rc.generator import RcTargetGenerator
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import open_storage
from arm_rc_ctrl.validation import is_hex

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray

    from arm_rc_ctrl.controllers.estimator import DerivativeEstimate
    from arm_rc_ctrl.controllers.tracking import TrackerConfig
    from arm_rc_ctrl.data.manual_scenario import ManualScenarioConfig
    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.execution import ExecutionRecord
    from arm_rc_ctrl.experiments.disturbances import ForcePulse
    from arm_rc_ctrl.experiments.manual_fits import CachedFit, ManualFitInputs, ManualFitRecord
    from arm_rc_ctrl.experiments.manual_study import StudyConfiguration, StudyManifest, StudyModel
    from arm_rc_ctrl.experiments.perturbations import DevelopmentRobustness, RobustnessScenario
    from arm_rc_ctrl.experiments.run_record import RunArrays
    from arm_rc_ctrl.experiments.termination import Termination
    from arm_rc_ctrl.provenance import ProvenanceRecord
    from arm_rc_ctrl.rc.recipe import ModelRecipe
    from arm_rc_ctrl.storage import StorageRoot

__all__ = [
    "GRID_TOLERANCE_S",
    "POINTER_SCHEMA",
    "PROGRESS_FILE",
    "RUN_CLAIMS_FILE",
    "CausalReplayReference",
    "EvidenceIntegrityError",
    "GeneratedReferenceReport",
    "ManualDwellReport",
    "ManualEvaluationConfig",
    "ManualEvaluationRunner",
    "ManualEvidencePointer",
    "ManualFitBinding",
    "ManualModelEvidence",
    "ManualModelTiming",
    "ManualPairRecord",
    "ManualReplayBank",
    "ManualRunArtifact",
    "ManualRunConditions",
    "ManualRunOutcome",
    "ManualRunTiming",
    "ManualSimulationLimits",
    "ManualTriggerRule",
    "RunClaims",
    "SimulateFn",
    "TriggerOutcome",
    "WorkerSpawn",
    "bank_identity",
    "check_bank_manifest",
    "check_model_manifest",
    "check_training_sources",
    "evaluate_in_parallel",
    "evaluation_entries",
    "evaluation_scenarios",
    "fit_binding",
    "horizon_completed",
    "load_manual_evaluation_config",
    "load_manual_model_evidence",
    "load_manual_pointer",
    "load_manual_replay_bank",
    "load_verified_payload",
    "load_verified_run",
    "main",
    "manual_bank_to_json",
    "manual_conditions",
    "manual_dwell_report",
    "manual_evidence_to_json",
    "manual_pointer_name",
    "manual_run_outcome",
    "manual_trigger",
    "model_uri",
    "prepare_runner",
    "read_progress_runs",
    "read_run_claims",
    "replay_bank_uri",
    "spawn_worker",
    "stored_manifest",
    "stored_reference",
    "trigger_outcome",
    "verify_model_evidence",
]

_SHA256_HEX: Final = 64
_SHORT: Final = 12
GRID_TOLERANCE_S: Final = 1e-9
"""Slack for comparing times that are exact multiples of the control period."""
_GRID_TOLERANCE_S: Final = GRID_TOLERANCE_S

REPORTS_PREFIX: Final = "armrc://reports/task_1a_manual_v1"
"""Where this experiment's evidence lives in the store."""

EVALUATION_SCHEMA_VERSION: Final = 1
PROGRESS_FILE: Final = "progress.json"
RUN_CLAIMS_FILE: Final = "runs.json"
"""Every run an evidence directory owns, claimed before its payload is published."""
POINTER_SCHEMA: Final = "task-1a-manual-evidence"
POINTER_KINDS: Final = ("model", "replay")
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
    replay_velocity_cutoff_hz: float
    replay_acceleration_cutoff_hz: float
    """The causal derivative policy replay is driven through: its model configuration's own cutoffs, so both
    arms of a pair filter alike and a difference between them is the generator's, not the filter's."""
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
            ("replay_velocity_cutoff_hz", self.replay_velocity_cutoff_hz),
            ("replay_acceleration_cutoff_hz", self.replay_acceleration_cutoff_hz),
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
    replay_cutoffs: tuple[float, float],
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
        replay_velocity_cutoff_hz=replay_cutoffs[0],
        replay_acceleration_cutoff_hz=replay_cutoffs[1],
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
    label: str
    """``<configuration>/<arm>``, the model this evidence is of."""
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
    if not manifests:
        return None
    found = manifests[0]
    # The name is a claim about the content, so it is checked before anything reads the file: a
    # manifest edited after it was written is not the evidence this directory says it holds.
    named = found.stem.split("-", 1)[1]
    if sha256_file(found)[:_SHORT] != named:
        msg = f"{found.name} does not match its content digest; a manifest is named by what it holds"
        raise ValueError(msg)
    return found


def stored_manifest(store: StorageRoot, directory_uri: str) -> Path | None:
    """The manifest installed under ``directory_uri``, checked against its name, or ``None`` when there is none."""
    return _existing_manifest(store, directory_uri)


def stored_reference(path: Path, store: StorageRoot) -> ArtifactReference:
    """The store reference of an installed manifest: its URI, content digest and size."""
    return _reference_of(path, store)


def model_uri(identity: str) -> str:
    """The store directory of one model's evidence under one protocol."""
    return f"{REPORTS_PREFIX}/model/{identity}"


def _outcome_criteria(outcome: ManualRunOutcome) -> dict[str, bool]:
    """The named criteria one run's verdict is recorded with.

    One construction serves writing and checking alike: a run is stored with
    exactly these, and a served or resumed pair is compared against exactly
    these, so a recorded verdict cannot drift from the verdict the sweep
    actually reached.
    """
    return {
        "completed": outcome.completed,
        "dwell": outcome.dwell.ok if outcome.post_pulse_dwell is None else outcome.post_pulse_dwell.ok,
        "generated_reference": outcome.generated is None or outcome.generated.ok,
        "trigger": outcome.trigger is None or outcome.trigger.ok,
        "saturation": outcome.saturation_fraction <= SATURATION_BOUND,
    }


def _verify_run_summary(where: str, summary: Path, run: ManualRunArtifact) -> RunSummary:
    """The stored summary a record cites, checked against the digest that record kept for it."""
    if not summary.is_file():
        msg = f"{where}: {RUN_SUMMARY_FILE} is missing from the store"
        raise EvidenceIntegrityError(msg)
    if summary.stat().st_size != run.size or sha256_file(summary) != run.sha256:
        msg = f"{where}: {RUN_SUMMARY_FILE} no longer matches its record"
        raise EvidenceIntegrityError(msg)
    return RunSummary.from_json(summary.read_text(encoding="utf-8"))


def _verify_run_arrays(where: str, arrays: Path, run: ManualRunArtifact, stored: RunSummary) -> None:
    """The arrays, against both references that describe them.

    The manifest keeps its own copy of the digest, so it cannot answer for the
    arrays alone; the run's summary carries an independent reference, and the
    payload has to satisfy both of them.
    """
    if not arrays.is_file():
        msg = f"{where}: {RUN_ARRAYS_FILE} is missing from the store"
        raise EvidenceIntegrityError(msg)
    _check_arrays_digest(where, sha256_file(arrays), run, stored)


def _check_arrays_digest(where: str, digest: str, run: ManualRunArtifact, stored: RunSummary) -> None:
    """One arrays digest, against the record's copy and the summary's own reference."""
    if digest != run.arrays_sha256:
        msg = f"{where}: {RUN_ARRAYS_FILE} no longer matches its record"
        raise EvidenceIntegrityError(msg)
    if digest != stored.arrays_sha256:
        msg = f"{where}: {RUN_ARRAYS_FILE} no longer matches its run summary"
        raise EvidenceIntegrityError(msg)


def _verify_run_verdict(where: str, pair: ManualPairRecord, stored: RunSummary) -> None:
    """The recorded verdict, grounded in the run rather than in the record's agreement with itself.

    Digests bind a payload to what a record says its digest is, and re-derived
    counts only make a record agree with its own pairs. So the status is checked
    against the pair's own outcome, and that outcome against the verdict its
    summary computes: ``success`` is a stored field, and cannot vouch for itself.
    """
    outcome = pair.outcome
    if outcome is None:
        return
    expected_status = "completed" if outcome.success else "infeasible"
    if pair.status != expected_status:
        msg = f"{where}: recorded status {pair.status!r}, but its own outcome gives the verdict {expected_status!r}"
        raise EvidenceIntegrityError(msg)
    if outcome.success != stored.outcome.success:
        msg = (
            f"{where}: the pair records success {outcome.success}, but the verdict its run summary "
            f"computes is {stored.outcome.success}"
        )
        raise EvidenceIntegrityError(msg)
    recorded = _outcome_criteria(outcome)
    criteria = dict(stored.outcome.criteria)
    if criteria != recorded:
        msg = f"{where}: the recorded verdict {recorded} is not the run's own stored verdict {criteria}"
        raise EvidenceIntegrityError(msg)


def _verify_run_payload(store: StorageRoot, pair: ManualPairRecord) -> None:
    """Check one recorded run is still in the store exactly as the record describes it.

    A manifest cites runs; it does not contain them. Serving evidence whose
    payloads were deleted or rewritten would report an experiment that can no
    longer be inspected, so both files are checked: the summary the record is
    digest-bound to, and the arrays, which that summary references independently
    of the record citing them. The verdict is then checked against the summary
    too, which is what makes a recorded success answerable to the run that
    produced it.
    """
    run = pair.run
    if run is None:
        return
    where = f"run {run.artifact_id} of {pair.scenario_id} [{pair.tracker}] {pair.arm}"
    try:
        summary = store.path(run.uri, mode="read")
    except FileNotFoundError as error:
        msg = f"{where}: {RUN_SUMMARY_FILE} is missing from the store"
        raise EvidenceIntegrityError(msg) from error
    stored = _verify_run_summary(where, summary, run)
    _verify_run_arrays(where, summary.parent / RUN_ARRAYS_FILE, run, stored)
    _verify_run_verdict(where, pair, stored)


def load_verified_payload(
    store: StorageRoot, run: ManualRunArtifact, *, where: str
) -> tuple[RunSummary, dict[str, NDArray[Any]]]:
    """One stored run's summary and arrays, each checked against every digest that describes it.

    The arrays file is read once: its digest is checked against both references
    and the same bytes are then parsed, so what a caller measures is what was
    verified.
    """
    try:
        summary = store.path(run.uri, mode="read")
    except FileNotFoundError as error:
        msg = f"{where}: {RUN_SUMMARY_FILE} is missing from the store"
        raise ValueError(msg) from error
    stored = _verify_run_summary(where, summary, run)
    arrays_file = summary.parent / RUN_ARRAYS_FILE
    if not arrays_file.is_file():
        msg = f"{where}: {RUN_ARRAYS_FILE} is missing from the store"
        raise ValueError(msg)
    data = arrays_file.read_bytes()
    _check_arrays_digest(where, sha256_bytes(data), run, stored)
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if set(archive.files) != set(stored.arrays):
            msg = f"{where}: {RUN_ARRAYS_FILE} holds {sorted(archive.files)}, not {sorted(stored.arrays)}"
            raise ValueError(msg)
        arrays = {name: cast("NDArray[Any]", archive[name]) for name in stored.arrays}
    return stored, arrays


def load_verified_run(store: StorageRoot, pair: ManualPairRecord) -> tuple[RunSummary, dict[str, NDArray[Any]]]:
    """One recorded run's summary and arrays, after exactly the checks a resume applies before serving it."""
    run = pair.run
    where = f"{pair.scenario_id} [{pair.tracker}] {pair.arm}"
    if run is None:
        msg = f"{where} was not simulated, so it has no run to load"
        raise ValueError(msg)
    where = f"run {run.artifact_id} of {where}"
    stored, arrays = load_verified_payload(store, run, where=where)
    _verify_run_verdict(where, pair, stored)
    return stored, arrays


def _verify_stored_runs(store: StorageRoot, pairs: Sequence[ManualPairRecord]) -> None:
    """Every run behind a set of pairs, before they are served as completed evidence."""
    for pair in pairs:
        _verify_run_payload(store, pair)


class RunClaims:
    """Every run one evidence directory owns, recorded before each payload is published.

    A run payload becomes visible by renaming a staging directory into the
    runs bucket, and the progress record names it only once its pair is
    complete. An owner that reads progress alone therefore cannot discover a
    run whose process died between the two, which is how stored bytes escape
    a resource ceiling. The claim is written first, so ownership of a payload
    exists before the payload does.
    """

    def __init__(self, store: StorageRoot, directory_uri: str, identity: str) -> None:
        """Load any claims already recorded here, refusing a record that belongs elsewhere."""
        self.path = store.path(f"{directory_uri}/{RUN_CLAIMS_FILE}", mode="write")
        self.identity = identity
        self.runs: list[str] = list(read_run_claims(self.path, identity=identity)) if self.path.is_file() else []

    def add(self, artifact_id: str) -> None:
        """Record one run before its payload is published; claims accumulate and are never dropped."""
        if artifact_id in self.runs:
            return
        self.runs.append(artifact_id)
        mapping = {"schema_version": EVALUATION_SCHEMA_VERSION, "identity": self.identity, "runs": self.runs}
        _write_atomic(self.path, json.dumps(mapping, indent=1, sort_keys=True))


def read_run_claims(path: Path, *, identity: str | None = None) -> tuple[str, ...]:
    """The runs one evidence directory claimed, refusing a record that cannot be read whole.

    An unreadable inventory is never an empty one: a caller that charges
    stored bytes must refuse a truncated record rather than read no runs from
    it and spend a ceiling on work it cannot see.
    """
    try:
        mapping = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        claimed = tuple(cast("list[object]", mapping["runs"]))
        recorded = cast("str | None", mapping.get("identity"))
    except (OSError, ValueError, KeyError, TypeError) as error:
        msg = f"{path} cannot be read, so the runs it claims cannot be counted: {error}"
        raise ValueError(msg) from error
    if identity is not None and recorded != identity:
        msg = f"{path} claims runs for {recorded}, not for {identity}"
        raise ValueError(msg)
    if not all(isinstance(run, str) and run.strip() for run in claimed):
        msg = f"{path} claims something that is not a run identity"
        raise ValueError(msg)
    return cast("tuple[str, ...]", claimed)


def read_progress_runs(path: Path) -> tuple[str, ...]:
    """Every run URI the run-granular progress record names, refusing one that cannot be read whole."""
    try:
        mapping = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        pairs = cast("list[dict[str, object]]", mapping["pairs"])
        runs = [cast("dict[str, object]", pair["run"]) for pair in pairs if pair.get("run") is not None]
        return tuple(cast("str", run["uri"]) for run in runs)
    except (OSError, ValueError, KeyError, TypeError) as error:
        msg = f"{path} cannot be read, so the runs it records cannot be counted: {error}"
        raise ValueError(msg) from error


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
        """A recorded run is resumed onto only while its stored payloads still match what it claims."""
        _verify_stored_runs(self.store, self.pairs)

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


# --- the Git-tracked pointers to stored evidence ---------------------------------------------


@dataclass(frozen=True)
class ManualEvidencePointer:
    """The repository's pointer to one stored manifest: what it is, where it lives, and its digest.

    Payloads stay in the external store, so this is what Git holds: enough to
    find the evidence a report cites and to verify it is the evidence that was
    recorded. Every field is required, because a pointer written to TOML drops
    unset values and a half-described pointer is worse than none.
    """

    schema: str
    experiment: str
    kind: str
    identity: str
    label: str
    status: str
    payload: ArtifactReference
    n_pairs: int
    n_completed: int
    n_infeasible: int

    def __post_init__(self) -> None:
        """The schema, experiment, and kind are known, and the counts agree with each other."""
        if self.schema != POINTER_SCHEMA or self.kind not in POINTER_KINDS:
            msg = f"unsupported evidence pointer {self.schema!r}/{self.kind!r}"
            raise ValueError(msg)
        if not is_hex(self.identity, _SHA256_HEX) or not self.label.strip() or not self.experiment.strip():
            msg = "a pointer names its experiment, its identity, and its label"
            raise ValueError(msg)
        if self.n_completed + self.n_infeasible != self.n_pairs:
            msg = f"{self.label}: {self.n_completed} completed and {self.n_infeasible} infeasible is not {self.n_pairs}"
            raise ValueError(msg)


def manual_pointer_name(kind: str, label: str) -> str:
    """The pointer file name: ``<kind>__<label with / replaced by __>.toml``."""
    return f"{kind}__{label.replace('/', '__')}.toml"


def load_manual_pointer(path: Path) -> ManualEvidencePointer:
    """Load one pointer record."""
    return load_record(path, ManualEvidencePointer)


def _pointer_of(
    kind: str, label: str, identity: str, pairs: Sequence[ManualPairRecord], payload: ArtifactReference
) -> ManualEvidencePointer:
    """Describe one stored manifest for the repository."""
    counts = Counter(pair.status for pair in pairs)
    return ManualEvidencePointer(
        schema=POINTER_SCHEMA,
        experiment=EXPERIMENT_LABEL,
        kind=kind,
        identity=identity,
        label=label,
        status="feasible" if counts["completed"] == len(pairs) else "infeasible",
        payload=payload,
        n_pairs=len(pairs),
        n_completed=counts["completed"],
        n_infeasible=counts["infeasible"],
    )


def _reference_of(path: Path, store: StorageRoot) -> ArtifactReference:
    """The store reference of an already-installed manifest."""
    return ArtifactReference(str(store.uri_for(path)), sha256_file(path), path.stat().st_size)


# --- the replay reference (its derivative policy is the generator's) -------------------------


class CausalReplayReference:
    """Command a recording's positions, deriving velocity and acceleration causally from them.

    The paired comparison is only about the generator when both arms are driven
    the same way. A reference that hands the tracker the recording's own
    ``dq``/``ddq`` would give replay derivatives computed from the whole
    trajectory, including its future, which the generator cannot have -- and
    would then switch them to zero the instant the log ends. Here the tracker
    receives derivatives of the positions actually commanded, in every regime:
    the hold before activation, the recording itself, and the continuation at
    the final recorded posture afterwards.
    """

    def __init__(
        self,
        reference: DemonstrationReference,
        *,
        activation_s: float,
        hold: NDArray[np.float64],
        estimator: CausalDerivativeEstimator,
    ) -> None:
        """Bind the recording, the activation boundary, the held posture, and the estimator."""
        if not (math.isfinite(activation_s) and activation_s >= 0):
            msg = f"activation_s must be finite and non-negative, got {activation_s!r}"
            raise ValueError(msg)
        posture = np.asarray(hold, dtype=np.float64)
        if posture.shape != (reference.dof,) or not bool(np.all(np.isfinite(posture))):
            msg = f"hold must be a finite ({reference.dof},) posture, got {posture!r}"
            raise ValueError(msg)
        if estimator.dof != reference.dof:
            msg = f"the estimator covers {estimator.dof} joints and the recording {reference.dof}: same dof required"
            raise ValueError(msg)
        self._reference = reference
        self._activation = activation_s
        self._hold: NDArray[np.float64] = np.array(posture, dtype=np.float64)
        self._estimator = estimator
        self._last: tuple[float, DerivativeEstimate] | None = None

    @classmethod
    def from_samples(
        cls,
        samples: SampleSet,
        *,
        activation_s: float,
        hold: NDArray[np.float64],
        estimator: CausalDerivativeEstimator,
        interpolation: str = "linear",
    ) -> CausalReplayReference:
        """Build from a processed take, reusing the repository's own interpolation of its positions."""
        reference = DemonstrationReference.from_samples(samples, cast("Any", interpolation))
        return cls(reference, activation_s=activation_s, hold=hold, estimator=estimator)

    @property
    def activation_s(self) -> float:
        """The activation boundary on the run clock."""
        return self._activation

    def sample(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """The commanded posture at ``t`` with its causally estimated derivatives."""
        if t < self._activation:
            q = self._hold
        else:
            # Past the end of the log this clips to the final recorded sample, so the continuation
            # commands the final posture and its derivatives follow from that, not from a jump.
            q, _dq, _ddq = self._reference.sample(t - self._activation)
        # LimitedTracker samples the reference for telemetry and the skelarm tracker it wraps samples
        # the same instant again, so a causal estimator must not advance on a repeated question: it
        # answers with what it already estimated, and only a new instant moves it forward.
        cached = self._last
        if cached is not None and t == cached[0]:
            estimate = cached[1]
        else:
            estimate = self._estimator.update(t, np.asarray(q, dtype=np.float64))
            self._last = (t, estimate)
        return estimate.q, estimate.dq, estimate.ddq


# --- the replay baselines of one demonstration -----------------------------------------------


def replay_bank_uri(conditions: ManualRunConditions, assignment: str) -> str:
    """The store directory of one parent's replay baselines under one protocol."""
    return f"{REPORTS_PREFIX}/replay/{_bank_identity(conditions, assignment)}"


def bank_identity(conditions: ManualRunConditions, assignment: str) -> str:
    """Conditions and parent together: the same protocol over another recording is another bank."""
    return sha256_bytes(f"{conditions.identity}:{assignment}".encode("ascii"))


_bank_identity = bank_identity


@dataclass(frozen=True)
class ManualRunTiming:
    """What one run this invocation simulated cost, in wall time and stored bytes.

    Kept out of the evidence on purpose: wall time describes this machine, not
    the experiment, and a content-addressed manifest carrying it would differ
    between machines and between re-runs of the same sweep.
    """

    arm: str
    scenario_id: str
    tracker: str
    rows: int
    simulate_seconds: float
    persist_seconds: float
    run_bytes: int
    """The run's summary and its arrays together: what storing it actually costs."""

    def __post_init__(self) -> None:
        """Figures are non-negative."""
        if min(self.rows, self.simulate_seconds, self.persist_seconds, self.run_bytes) < 0:
            msg = f"run timings are non-negative, got {self}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ManualModelTiming:
    """What one model cost this invocation: its fit, and the sweep over its pairs."""

    label: str
    fit_seconds: float
    """As the fit cache recorded it: measured now for a miss, at its original fitting for a hit."""
    fit_cache_hit: bool
    sweep_seconds: float
    runs: int


def _instant(moment: datetime) -> str:
    """A UTC instant on a format two processes can compare as text."""
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


PHASE_PREPARE: Final = "prepare"
PHASE_FIT: Final = "fit"
PHASE_SERVE_FIT: Final = "serve_fit"
PHASE_VERIFY_MODEL: Final = "verify_served_model"
PHASE_VERIFY_BANK: Final = "verify_served_bank"
PHASE_SWEEP: Final = "sweep"
PHASE_BUILD_BANK: Final = "build_replay_bank"

INCLUSIVE_PHASES: Final = (PHASE_SWEEP, PHASE_BUILD_BANK)
"""Spans that enclose other measured work; summing them with their contents double-counts seconds."""


@dataclass(frozen=True)
class ManualPhaseTiming:
    """One measured interval, saying whether it encloses other measurements.

    Elapsed time is not the sum of everything measured: a sweep contains the
    runs it simulates, and a bank build contains its own. Adding those together
    would count the same seconds twice, so each interval states which kind it
    is and only the disjoint ones are ever summed.
    """

    phase: str
    seconds: float
    inclusive: bool
    """True when this span contains other measured intervals (runs, or nested phases)."""
    label: str = ""
    """The model or parent this interval belongs to, where one applies."""

    def __post_init__(self) -> None:
        """A known phase, a non-negative duration, and the enclosure flag its phase implies."""
        if self.seconds < 0:
            msg = f"a phase lasts a non-negative time, got {self.seconds} for {self.phase!r}"
            raise ValueError(msg)
        if self.inclusive != (self.phase in INCLUSIVE_PHASES):
            msg = f"{self.phase!r} is {'an inclusive span' if self.phase in INCLUSIVE_PHASES else 'disjoint'}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ManualWorkerSpan:
    """Which process a worker was and when it ran, on the clock its parent shares.

    Overlap is a fact about processes and instants. Dividing summed work by the
    worker count is not elapsed time and cannot establish whether anything
    overlapped, so the instants are recorded instead of inferred.
    """

    pid: int
    label: str
    started_at: str
    """Worker ``main()`` entry. Interpreter startup precedes this and is not covered."""
    prepared_at: str
    """When ``prepare_runner`` returned: the preparation phase ends here."""
    finished_at: str
    """When ``evaluate`` returned: the evaluation phase ends here."""
    covers: str = "worker main() entry to evaluate() return; interpreter startup precedes started_at and is not covered"

    def __post_init__(self) -> None:
        """A real process, and instants that do not run backwards."""
        if self.pid <= 0:
            msg = f"a worker span names a process, got pid {self.pid}"
            raise ValueError(msg)
        if not (self.started_at <= self.prepared_at <= self.finished_at):
            msg = f"{self.label}: worker instants are out of order"
            raise ValueError(msg)


@dataclass(frozen=True)
class ManualWorkerTimings:
    """What one worker process measured, carried back to the parent that reports it.

    The runner's instrumentation is per-process. Under workers the parent
    simulates only the replay banks it builds before fanning out and then
    serves the rest back, so reading its timings alone reports the runs it did
    not make as none at all. Transient: written beside the sweep, read once by
    the parent, and never part of the evidence.
    """

    runs: tuple[ManualRunTiming, ...]
    models: tuple[ManualModelTiming, ...]
    manifest_bytes: int
    phases: tuple[ManualPhaseTiming, ...] = ()
    span: ManualWorkerSpan | None = None

    def __post_init__(self) -> None:
        """Bytes are non-negative; each timing validates itself."""
        if self.manifest_bytes < 0:
            msg = f"manifest bytes are non-negative, got {self.manifest_bytes}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ManualRunArtifact:
    """Where one persisted run lives and what it contains."""

    artifact_id: str
    uri: str
    sha256: str
    size: int
    arrays_sha256: str
    sources: tuple[str, ...] = ()
    """The demonstrations this run's model trained on: all ten for the all-ten arm, one otherwise."""


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

    def __post_init__(self) -> None:
        """The pairs are every scenario and tracker of the conditions, each a replay of this bank's parent."""
        if self.assignment not in ASSIGNMENTS:
            msg = f"assignment must be a bank position in {list(ASSIGNMENTS)}, got {self.assignment!r}"
            raise ValueError(msg)
        if [(p.scenario_id, p.tracker) for p in self.pairs] != list(self.conditions.pairs):
            msg = "the bank's pairs are every scenario and tracker of its conditions, in evaluation order"
            raise ValueError(msg)
        if any(p.arm != "replay" for p in self.pairs):
            msg = "every pair of a replay bank is a replay pair"
            raise ValueError(msg)

    @property
    def identity(self) -> str:
        """The key this bank is stored and served under."""
        return _bank_identity(self.conditions, self.assignment)


def fit_binding(entry: StudyModel, record: ManualFitRecord, recipe: ModelRecipe) -> ManualFitBinding:
    """The complete fit binding the evidence of ``entry`` must carry, from the fit recorded for it.

    One construction serves writing and checking alike: evidence records
    exactly this, and stored evidence is compared against exactly this, so the
    two cannot drift apart and no recorded field is left vouching for itself.
    """
    return ManualFitBinding(
        identity=record.identity,
        configuration=entry.configuration,
        arm=entry.arm.label,
        solver_alpha=recipe.solver_alpha,
        recipe_sha256=record.recipe_sha256,
        weights_sha256=record.weights_sha256,
    )


def verify_model_evidence(
    store: StorageRoot,
    inputs: ManualFitInputs,
    entry: StudyModel,
    evidence: ManualModelEvidence,
    *,
    conditions: ManualRunConditions,
    where: str,
) -> None:
    """Check one stored model manifest against the study's trusted inputs, without refitting the model.

    A resume refits a served model before trusting it; a reader of finished
    evidence need not. The binding is read instead from the fit cache under the
    study's own fit identity, with its recipe's and weights' digests verified
    and the recipe's construction compared with the one the entry demands, and
    the manifest is then checked whole. The runs themselves are verified
    separately, when read.

    This is the one implementation: the sweep's resume, the audit and the
    search's parent all call it, so no reader of this evidence checks less than
    another.
    """
    fits = ManualFitStore(store)
    record = fits.read_record(entry.fit_identity)
    recipe = fits.read_recipe(record)
    fits.read_weights(record)
    mismatches = recipe_mismatches(entry, recipe, inputs)
    if mismatches:
        msg = f"{entry.label}: the cached recipe is not the construction the study froze: {'; '.join(mismatches)}"
        raise ValueError(msg)
    check_model_manifest(
        evidence,
        entry=entry,
        fit=fit_binding(entry, record, recipe),
        conditions=conditions,
        sources=tuple(
            inputs.sources[name].artifact_id
            for name in (ASSIGNMENTS if entry.arm.assignment is None else (entry.arm.assignment,))
        ),
        where=where,
    )
    parent = entry.arm.assignment
    expected = None if parent is None else _bank_identity(conditions, parent)
    if evidence.replay_bank != expected:
        msg = f"{where} cites another replay bank than its parent and protocol produce"
        raise ValueError(msg)


def check_training_sources(pairs: Sequence[ManualPairRecord], expected: tuple[str, ...], described: str) -> None:
    """Check every run names the demonstrations its arm trained on, as the study demands them.

    Renaming an altered manifest to its own digest satisfies the name check,
    so what the runs claim is compared with what the arm's assignment demands:
    all ten for the all-ten arm, the replayed or singleton parent otherwise.
    """
    for pair in pairs:
        if pair.run is None:
            continue
        if pair.run.sources != expected:
            msg = (
                f"{described}: {pair.scenario_id} [{pair.tracker}] {pair.arm} names "
                f"{len(pair.run.sources)} training sources {list(pair.run.sources)}, not the {len(expected)} "
                f"its assignment demands"
            )
            raise ValueError(msg)


def check_model_manifest(
    evidence: ManualModelEvidence,
    *,
    entry: StudyModel,
    fit: ManualFitBinding,
    conditions: ManualRunConditions,
    sources: tuple[str, ...],
    where: str,
) -> None:
    """Check a stored model manifest against what the study demands for ``entry``, never against itself.

    The recorded identity binds the fit and the conditions and nothing else,
    so every other recorded field is unconstrained by it: an alteration that
    rewrote them together, and renamed the file to its new digest, would be
    checked against its own claims and pass. Each is compared with a trusted
    source instead -- the study entry, the fit recorded under the study's own
    fit identity, the conditions derived from the evaluation configuration,
    and the demonstrations the arm trained on. The conditions and the fit are
    compared whole, because an identity is a recorded field too and cannot
    vouch for the bindings beside it.
    """
    identity = sha256_bytes(f"{fit.identity}:{conditions.identity}".encode("ascii"))
    if evidence.identity != identity:
        msg = f"{where} holds the evidence {evidence.identity[:_SHORT]}, not {identity[:_SHORT]}"
        raise ValueError(msg)
    if evidence.conditions != conditions:
        msg = (
            f"{where} records conditions {evidence.conditions.identity[:_SHORT]}, not the "
            f"{conditions.identity[:_SHORT]} this study entry demands"
        )
        raise ValueError(msg)
    checks: tuple[tuple[str, object, object], ...] = (
        ("label", evidence.label, entry.label),
        ("assignment", evidence.assignment, entry.arm.assignment),
        ("fit", evidence.fit, fit),
    )
    for field_name, found, expected in checks:
        if found != expected:
            msg = f"{where} records {field_name} {found!r}, not the {expected!r} this study entry demands"
            raise ValueError(msg)
    check_training_sources(evidence.pairs, sources, f"the evidence of {entry.label}")


def check_bank_manifest(
    bank: ManualReplayBank, *, assignment: str, conditions: ManualRunConditions, sources: tuple[str, ...], where: str
) -> None:
    """Check a stored replay bank against the conditions and parent the study keys it by.

    The bank's identity is derived from its own conditions and parent, so
    matching the identity the trusted conditions produce compares both whole;
    the runs' training sources are not in the identity and are checked apart.
    """
    identity = _bank_identity(conditions, assignment)
    if bank.identity != identity:
        msg = f"{where} holds the bank {bank.identity[:_SHORT]}, not {identity[:_SHORT]}"
        raise ValueError(msg)
    check_training_sources(bank.pairs, sources, f"the replay bank of {assignment}")


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
        self._pointers: dict[tuple[str, str], ManualEvidencePointer] = {}
        self._run_timings: list[ManualRunTiming] = []
        self._model_timings: dict[str, ManualModelTiming] = {}
        self._phase_timings: list[ManualPhaseTiming] = []
        self._manifest_bytes = 0

    def conditions(self, warmup_s: float, replay_cutoffs: tuple[float, float]) -> ManualRunConditions:
        """The protocol conditions at one warm-up and derivative policy.

        Baselines are shared only by models that share both: replay derives its
        own velocity and acceleration, so a different filter is a different run.
        """
        return manual_conditions(
            self.config,
            self.evaluation_file,
            scenario_ids=tuple(case.scenario_id for case in self.scenarios),
            warmup_s=warmup_s,
            replay_cutoffs=replay_cutoffs,
            execution_identity=self.execution.identity,
            root=self.root,
        )

    def replay_cutoffs(self, entry: StudyModel) -> tuple[float, float]:
        """The causal estimator cutoffs of ``entry``'s configuration, used by both arms of its pairs."""
        configuration = self._configuration(entry)
        return (configuration.velocity_cutoff_hz, configuration.acceleration_cutoff_hz)

    def _start(self, case: RobustnessScenario) -> tuple[float, ...]:
        """The perturbed reset posture: every take begins at the configured one, so the offsets are shared."""
        return case.initial_q(self.scenario.task.initial_q)

    def _samples(self, assignment: str) -> SampleSet:
        """The locked demonstration of one bank position."""
        return self.inputs.samples[self.inputs.sources[assignment].artifact_id]

    def _replay_pair(
        self,
        index: int,
        case: RobustnessScenario,
        tracker: str,
        *,
        assignment: str,
        warmup_s: float,
        replay_cutoffs: tuple[float, float],
        claims: RunClaims,
    ) -> ManualPairRecord:
        """Replay one demonstration through one tracker under one scenario, from a fresh reset."""
        start = self._start(case)
        samples = self._samples(assignment)
        # Replay is driven through the same causal derivative policy as the generator it is paired
        # against: handing the tracker the recording's own derivatives would give it the trajectory's
        # future, which the generator never has.
        held = CausalReplayReference.from_samples(
            samples,
            activation_s=warmup_s,
            hold=np.asarray(start, dtype=np.float64),
            estimator=CausalDerivativeEstimator(
                EstimatorConfig(
                    nominal_dt_s=self.scenario.timing.dt,
                    velocity_cutoff_hz=replay_cutoffs[0],
                    acceleration_cutoff_hz=replay_cutoffs[1],
                ),
                self.inputs.dof,
            ),
            interpolation=cast("Any", self.inputs.preprocessing.interpolation),
        )
        controller = LimitedTracker(cast("Any", held), self.trackers[tracker], self.scenario.limits.torque)
        trigger = (
            None
            if case.pulse is None
            else manual_trigger(self.config, self.scenario, direction_deg=case.direction_deg or 0.0)
        )
        fired: list[ForcePulse] = []
        started = time.perf_counter()
        arrays, termination = self.simulate(
            self.scenario,
            controller,
            duration_s=warmup_s + self.config.horizon_s,
            initial_q=start,
            force_trigger=trigger,
            triggered=fired,
            velocity_abort=self.config.simulation.velocity_abort,
        )
        simulated = time.perf_counter() - started
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
            simulate_seconds=simulated,
            claims=claims,
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
        assignment: str | None,
        warmup_s: float,
        pulse: ForcePulse | None,
        simulate_seconds: float,
        claims: RunClaims,
    ) -> ManualRunArtifact:
        """Store the run with the disturbance that actually fired, not the one the levels prescribed."""
        # The stored verdict must be the pair record's verdict: every criterion the outcome judges
        # appears here, or a run this sweep calls infeasible reads as a success in its own record.
        criteria = _outcome_criteria(outcome)
        trained_on = ASSIGNMENTS if assignment is None else (assignment,)
        described = "the whole bank" if assignment is None else assignment
        persist_started = time.perf_counter()
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
            sources=tuple(self.inputs.sources[name].artifact_id for name in trained_on),
            activation_s=warmup_s,
            reuse_identical=True,
            notes=f"{self.config.name} {arm} arm: {case.scenario_id} [{tracker}] of {described}.",
            claim=claims.add,
        )
        persisted = time.perf_counter() - persist_started
        del directory
        artifact = ManualRunArtifact(
            artifact_id=pointer.artifact.artifact_id,
            uri=pointer.artifact.payload.uri,
            sha256=pointer.artifact.payload.sha256,
            size=pointer.artifact.payload.size,
            arrays_sha256=summary.arrays_sha256,
            sources=tuple(self.inputs.sources[name].artifact_id for name in trained_on),
        )
        stored_arrays = self.store.path(artifact.uri, mode="read").parent / RUN_ARRAYS_FILE
        self._run_timings.append(
            ManualRunTiming(
                arm=arm,
                scenario_id=case.scenario_id,
                tracker=tracker,
                rows=arrays.n_samples,
                simulate_seconds=simulate_seconds,
                persist_seconds=persisted,
                run_bytes=artifact.size + stored_arrays.stat().st_size,
            )
        )
        return artifact

    @property
    def run_timings(self) -> tuple[ManualRunTiming, ...]:
        """Every run this invocation simulated, in the order it simulated them."""
        return tuple(self._run_timings)

    @property
    def model_timings(self) -> dict[str, ManualModelTiming]:
        """What each model reached in this invocation cost, by label."""
        return dict(self._model_timings)

    @property
    def phase_timings(self) -> tuple[ManualPhaseTiming, ...]:
        """Every interval this invocation measured, each saying whether it encloses others."""
        return tuple(self._phase_timings)

    @property
    def manifest_bytes(self) -> int:
        """Bytes of evidence this invocation installed."""
        return self._manifest_bytes

    @property
    def pointers(self) -> tuple[ManualEvidencePointer, ...]:
        """Every manifest this invocation produced or stood behind, in the order it was reached."""
        return tuple(self._pointers.values())

    def _register(
        self,
        evidence: ManualReplayBank | ManualModelEvidence,
        payload: ArtifactReference,
        *,
        warmup_s: float | None = None,
    ) -> None:
        """Remember the pointer of one manifest, whether it was installed now or served from the store."""
        if isinstance(evidence, ManualReplayBank):
            # The policy is part of the bank, so two banks of one parent and warm-up that filtered
            # differently must not collide on one pointer name.
            kind, label = "replay", f"{evidence.assignment}-warmup-{warmup_s:g}s-{evidence.identity[:8]}"
        else:
            kind, label = "model", evidence.label
        self._pointers[kind, label] = _pointer_of(kind, label, evidence.identity, evidence.pairs, payload)

    def write_pointers(self, evidence_dir: Path) -> list[Path]:
        """Write the Git pointer of every manifest this runner reached; identical pointers are idempotent."""
        written: list[Path] = []
        for (kind, label), pointer in self._pointers.items():
            path = evidence_dir / manual_pointer_name(kind, label)
            if path.exists():
                if load_manual_pointer(path) != pointer:
                    msg = f"{path} exists with another pointer; completed evidence is never overwritten"
                    raise FileExistsError(msg)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            write_record(path, pointer)
            written.append(path)
        return written

    def model_identity(self, entry: StudyModel, *, warmup_s: float) -> str:
        """The key one model's evidence is stored under, without refitting it.

        The study's own fit identity is the cache key of the fit, so the model
        key is derivable from the manifest alone.
        """
        conditions = self.conditions(warmup_s, self.replay_cutoffs(entry))
        return sha256_bytes(f"{entry.fit_identity}:{conditions.identity}".encode("ascii"))

    def _fit_binding(self, entry: StudyModel, cached: CachedFit) -> ManualFitBinding:
        """The complete fit binding this entry's evidence must carry, from the fit this runner loaded."""
        return fit_binding(entry, cached.record, cached.recipe)

    def _sources(self, assignment: str | None) -> tuple[str, ...]:
        """The demonstrations an arm trained on: all ten for the all-ten arm, its own parent otherwise."""
        names = ASSIGNMENTS if assignment is None else (assignment,)
        return tuple(self.inputs.sources[name].artifact_id for name in names)

    def _verify_sources(self, pairs: Sequence[ManualPairRecord], assignment: str | None, described: str) -> None:
        """Check runs name the demonstrations their arm trained on, before they become or remain evidence."""
        check_training_sources(pairs, self._sources(assignment), described)

    def verify_stored_model(self, entry: StudyModel, evidence: ManualModelEvidence, *, where: str) -> None:
        """Apply a resume's checks to one stored model manifest, without refitting the model."""
        warmup_s = self.inputs.configuration(entry).warmup_s
        verify_model_evidence(
            self.store,
            self.inputs,
            entry,
            evidence,
            conditions=self.conditions(warmup_s, self.replay_cutoffs(entry)),
            where=where,
        )

    def verify_stored_bank(
        self,
        bank: ManualReplayBank,
        *,
        assignment: str,
        warmup_s: float,
        replay_cutoffs: tuple[float, float],
        where: str,
    ) -> None:
        """Apply a resume's checks to one stored replay bank, under the conditions the study keys it by."""
        check_bank_manifest(
            bank,
            assignment=assignment,
            conditions=self.conditions(warmup_s, replay_cutoffs),
            sources=self._sources(assignment),
            where=where,
        )

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
        claims: RunClaims,
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
        started = time.perf_counter()
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
        simulated = time.perf_counter() - started
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
            assignment=assignment,
            warmup_s=warmup_s,
            pulse=pulse,
            simulate_seconds=simulated,
            claims=claims,
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
        cutoffs = self.replay_cutoffs(entry)
        conditions = self.conditions(warmup_s, cutoffs)
        fit_started = time.perf_counter()
        cached = ManualFitStore(self.store).fit_or_load(entry, self.inputs)
        # A cache hit refits the model to verify it. That is this invocation's own work, and
        # fit_seconds reports the ORIGINAL fit, so the two are measured apart.
        self._phase_timings.append(
            ManualPhaseTiming(
                phase=PHASE_SERVE_FIT if cached.cache_hit else PHASE_FIT,
                seconds=time.perf_counter() - fit_started,
                inclusive=False,
                label=entry.label,
            )
        )
        sweep_started = time.perf_counter()
        self._model_timings[entry.label] = ManualModelTiming(
            label=entry.label,
            fit_seconds=cached.record.fit_seconds,
            fit_cache_hit=cached.cache_hit,
            sweep_seconds=0.0,
            runs=0,
        )
        identity = sha256_bytes(f"{cached.record.identity}:{conditions.identity}".encode("ascii"))
        existing = self._models.get(identity)
        if existing is not None:
            return existing
        uri = model_uri(identity)
        stored = _existing_manifest(self.store, uri)
        if stored is not None:
            verify_started = time.perf_counter()
            evidence = load_manual_model_evidence(stored)
            check_model_manifest(
                evidence,
                entry=entry,
                fit=self._fit_binding(entry, cached),
                conditions=conditions,
                sources=self._sources(entry.arm.assignment),
                where=str(stored),
            )
            _verify_stored_runs(self.store, evidence.pairs)
            nested = 0.0
            if entry.arm.assignment is not None:
                # Also point at the baselines this model was compared against: they are equally
                # part of the evidence, and a served model would otherwise cite a bank the
                # repository has no pointer to. That call measures itself, so its time is taken
                # out of this phase rather than counted in both.
                nested_started = time.perf_counter()
                served = self.replay_bank(entry.arm.assignment, warmup_s=warmup_s, replay_cutoffs=cutoffs)
                nested = time.perf_counter() - nested_started
                if evidence.replay_bank != served.identity:
                    msg = f"{stored} cites another replay bank than its parent and protocol produce"
                    raise ValueError(msg)
            elif evidence.replay_bank is not None:
                msg = f"{stored} cites a replay bank, but the all-ten arm is paired against no single one"
                raise ValueError(msg)
            self._phase_timings.append(
                ManualPhaseTiming(
                    phase=PHASE_VERIFY_MODEL,
                    seconds=max(0.0, time.perf_counter() - verify_started - nested),
                    inclusive=False,
                    label=entry.label,
                )
            )
            self._register(evidence, _reference_of(stored, self.store))
            self._models[identity] = evidence
            return evidence
        assignment = entry.arm.assignment
        bank = None if assignment is None else self.replay_bank(assignment, warmup_s=warmup_s, replay_cutoffs=cutoffs)
        controllers = self._controllers(entry, cached, warmup_s)
        progress = _ManualProgress(self.store, uri, identity)
        claims = RunClaims(self.store, uri, identity)
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
                    claims=claims,
                )
                progress.add(pair)
                pairs.append(pair)
        swept = time.perf_counter() - sweep_started
        self._model_timings[entry.label] = ManualModelTiming(
            label=entry.label,
            fit_seconds=cached.record.fit_seconds,
            fit_cache_hit=cached.cache_hit,
            sweep_seconds=swept,
            runs=sum(1 for pair in pairs if pair.run is not None),
        )
        self._phase_timings.append(
            ManualPhaseTiming(phase=PHASE_SWEEP, seconds=swept, inclusive=True, label=entry.label)
        )
        self._verify_sources(pairs, assignment, f"the evidence of {entry.label}")
        counts = Counter(pair.status for pair in pairs)
        evidence = ManualModelEvidence(
            identity=identity,
            label=entry.label,
            conditions=conditions,
            fit=self._fit_binding(entry, cached),
            assignment=assignment,
            replay_bank=None if bank is None else bank.identity,
            status="feasible" if counts["completed"] == len(pairs) else "infeasible",
            pairs=tuple(pairs),
            n_pairs=len(pairs),
            n_completed=counts["completed"],
            n_infeasible=counts["infeasible"],
            n_unexecuted=counts["unexecuted"],
        )
        payload = _install_manifest(self.store, uri, manual_evidence_to_json(evidence))
        self._manifest_bytes += payload.size
        self._register(evidence, payload)
        self._models[identity] = evidence
        return evidence

    def replay_bank(self, assignment: str, *, warmup_s: float, replay_cutoffs: tuple[float, float]) -> ManualReplayBank:
        """Every replay baseline of one demonstration under one protocol and derivative policy."""
        conditions = self.conditions(warmup_s, replay_cutoffs)
        identity = _bank_identity(conditions, assignment)
        cached = self._banks.get(identity)
        if cached is not None:
            return cached
        uri = replay_bank_uri(conditions, assignment)
        stored = _existing_manifest(self.store, uri)
        if stored is not None:
            bank_verify_started = time.perf_counter()
            bank = load_manual_replay_bank(stored)
            # The requested parent, not the stored one: deriving the expectation from the file being
            # checked would be circular.
            check_bank_manifest(
                bank, assignment=assignment, conditions=conditions, sources=self._sources(assignment), where=str(stored)
            )
            _verify_stored_runs(self.store, bank.pairs)
            self._register(bank, _reference_of(stored, self.store), warmup_s=warmup_s)
            self._banks[identity] = bank
            self._phase_timings.append(
                ManualPhaseTiming(
                    phase=PHASE_VERIFY_BANK,
                    seconds=time.perf_counter() - bank_verify_started,
                    inclusive=False,
                    label=assignment,
                )
            )
            return bank
        build_started = time.perf_counter()
        progress = _ManualProgress(self.store, uri, identity)
        claims = RunClaims(self.store, uri, identity)
        pairs: list[ManualPairRecord] = []
        for index, case in enumerate(self.scenarios):
            for tracker in conditions.tracker_order:
                recorded = progress.get(case.scenario_id, tracker, "replay")
                if recorded is not None:
                    pairs.append(recorded)
                    continue
                self.log(f"replay {identity[:_SHORT]}: {case.scenario_id} [{tracker}] of {assignment}")
                pair = self._replay_pair(
                    index,
                    case,
                    tracker,
                    assignment=assignment,
                    warmup_s=warmup_s,
                    replay_cutoffs=replay_cutoffs,
                    claims=claims,
                )
                progress.add(pair)
                pairs.append(pair)
        # Recorded progress becomes a manifest here, so the bindings are checked before it does: a run
        # summary does not record what its model trained on, so nothing downstream would notice.
        self._verify_sources(pairs, assignment, f"the replay bank of {assignment}")
        bank = ManualReplayBank(conditions=conditions, assignment=assignment, pairs=tuple(pairs))
        payload = _install_manifest(self.store, uri, manual_bank_to_json(bank))
        self._manifest_bytes += payload.size
        self._register(bank, payload, warmup_s=warmup_s)
        self._banks[identity] = bank
        self._phase_timings.append(
            ManualPhaseTiming(
                phase=PHASE_BUILD_BANK,
                seconds=time.perf_counter() - build_started,
                inclusive=True,
                label=assignment,
            )
        )
        return bank


# --- bounded parallel execution (I1) ---------------------------------------------------------

type WorkerSpawn = Callable[..., None]
"""``(entry, *, warmup_s, env) -> None``: does one model's work, in a worker process or in this one."""


def evaluate_in_parallel(
    runner: ManualEvaluationRunner,
    entries: Sequence[StudyModel],
    *,
    warmup_s: float,
    workers: int,
    env: Mapping[str, str],
    spawn: WorkerSpawn,
) -> tuple[ManualModelEvidence, ...]:
    """Run ``entries`` through at most ``workers`` concurrent workers, then serve what they produced.

    The threads here only supervise workers; each worker is a separate
    interpreter with one numerical thread, inheriting the parent's declared
    affinity and thread settings, so no numerical work is ever shared between
    threads of this process. Sharing the parent's whole CPU set is deliberate:
    the execution identity a run is keyed by includes the declared policy and
    CPU set, so a worker pinned to its own subset would key its runs to another
    environment and they would not be this study's at all.

    Once the workers are done the evidence is in the store, so serving it back
    costs nothing and gives this invocation the pointers it must write.
    """
    if workers < 1:
        msg = f"workers must be at least 1, got {workers}"
        raise ValueError(msg)
    selected = tuple(entries)
    if selected:
        # Shared replay banks are built here, in this process, before any worker starts. Two models
        # of one parent, warm-up and derivative policy are paired against the same bank, and two
        # workers building it at once interleave their runs into one progress file and install two
        # differing manifests -- after which the bank can no longer be read at all.
        for assignment, cutoffs in dict.fromkeys(
            (entry.arm.assignment, runner.replay_cutoffs(entry))
            for entry in selected
            if entry.arm.assignment is not None
        ):
            runner.replay_bank(assignment, warmup_s=warmup_s, replay_cutoffs=cutoffs)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(spawn, entry, warmup_s=warmup_s, env=dict(env)) for entry in selected]
            for future in futures:
                future.result()
    return tuple(runner.evaluate(entry, warmup_s=warmup_s) for entry in selected)


# --- the command --------------------------------------------------------------------------

_MODULE: Final = "arm_rc_ctrl.experiments.manual_evaluation"


def evaluation_entries(manifest: StudyManifest, labels: Sequence[str] | None = None) -> tuple[StudyModel, ...]:
    """The models this invocation evaluates: every one of the study, or the named ones in manifest order."""
    if not labels:
        return tuple(manifest.entries)
    wanted = set(labels)
    chosen = tuple(entry for entry in manifest.entries if entry.label in wanted)
    unknown = sorted(wanted - {entry.label for entry in chosen})
    if unknown:
        msg = f"unknown model labels {unknown}"
        raise ValueError(msg)
    return chosen


def evaluation_scenarios(
    levels: DevelopmentRobustness, scenario: ManualScenarioConfig
) -> tuple[RobustnessScenario, ...]:
    """The locked development cases at this task's reset posture, checked against its joint bounds."""
    lower = tuple(link.q_min for link in scenario.robot.links)
    upper = tuple(link.q_max for link in scenario.robot.links)
    return robustness_scenarios(levels, nominal=scenario.task.initial_q, lower=lower, upper=upper)


def _load_runtimes() -> None:
    """Import the numerical runtimes before the environment is probed (C10: the probe must see them)."""
    for name in ("numpy", "rclib"):
        importlib.import_module(name)


def spawn_worker(
    entry: StudyModel,
    *,
    warmup_s: float,
    env: Mapping[str, str],
    study_file: Path,
    evaluation_file: Path,
    root: Path,
    exploratory: bool,
    scenario_ids: Sequence[str] | None = None,
    timings_path: Path | None = None,
    python: str = sys.executable,
) -> None:
    """Evaluate one model in a fresh interpreter that inherits this environment.

    The worker is given the same study and configuration, so it derives the
    same conditions; it inherits the parent's declared affinity and thread
    settings through ``env``, which is what makes its runs this study's.
    """
    command = [
        python,
        "-m",
        _MODULE,
        "evaluate-model",
        "--study",
        str(study_file),
        "--evaluation",
        str(evaluation_file),
        "--entry",
        entry.label,
        "--warmup-s",
        repr(float(warmup_s)),
        "--root",
        str(root),
    ]
    if scenario_ids is not None:
        # The worker rebuilds its own scope from the configuration, so a selection held only in
        # the parent reaches it not at all: a benchmark bounded to 60 nominal runs ran 416.
        command += ["--scenarios", *scenario_ids]
    if timings_path is not None:
        command += ["--timings", str(timings_path)]
    if exploratory:
        command.append("--exploratory")
    subprocess.run(command, check=True, env=dict(env))


@dataclass(frozen=True)
class _Prepared:
    """What both the command and a worker need: the bound study, the protocol, and the runner."""

    context: ManualStudyContext
    config: ManualEvaluationConfig
    runner: ManualEvaluationRunner
    execution: ExecutionRecord
    locked_scenarios: int
    """Cases the protocol locks, before any restriction: what a budget projection scales."""


def prepare_runner(
    args: argparse.Namespace,
    *,
    role: str,
    root: Path,
    module: str = _MODULE,
    scenario_ids: Sequence[str] | None = None,
    sampled: Sequence[StudyConfiguration] = (),
) -> _Prepared:
    """Verify the environment, bind the study and the configuration, and build the runner.

    ``module`` names the caller, because the command a run records must be the
    command that was actually invoked: the timing smoke check is its own entry
    point, and provenance claiming otherwise would misdescribe the evidence.

    ``sampled`` carries the configurations a search drew (M3MS-003). They are
    resolved beside the frozen ones, so the search evaluates through this same
    runner instead of a second path, and the frozen manifest is never written
    to.
    """
    require_canonical()
    ensure_single_thread()
    _load_runtimes()
    command = command_line(module, cast("list[str]", args.argv))
    execution = collect_execution(command=command, role=role, now=datetime.now(tz=UTC))
    execution.check_canonical()
    store = open_storage()
    context = ManualStudyContext.load(Path(cast("str", args.study)), store=store, root=root, execution=execution)
    evaluation_file = Path(cast("str", args.evaluation))
    config = load_manual_evaluation_config(evaluation_file)
    scenario = load_manual_scenario(config.scenario)
    cases = evaluation_scenarios(load_development_robustness(config.development), scenario)
    locked_scenarios = len(cases)
    if scenario_ids is not None:
        # The selection is part of the conditions a run is keyed by, so a restricted sweep keys its
        # evidence differently and can never serve a broader sweep's runs back as its own.
        wanted = tuple(scenario_ids)
        available = {case.scenario_id: case for case in cases}
        unknown = sorted(set(wanted) - set(available))
        if unknown:
            msg = f"unknown scenario ids {unknown}; the locked set has {len(available)} cases"
            raise ValueError(msg)
        cases = tuple(available[scenario_id] for scenario_id in wanted)
    resolved: dict[str, object] = {
        "study_manifest": context.manifest_sha256,
        "evaluation": {evaluation_file.name: sha256_file(evaluation_file)},
        "scenarios": len(cases),
        "execution_identity": execution.identity,
        "command": command,
    }
    provenance = collect_provenance(
        resolved,
        seeds={"contractive_seed_bank": context.manifest.seed_bank},
        artifacts=list(context.payloads),
        exploratory=bool(args.exploratory),
        now=datetime.now(tz=UTC),
    )
    require_clean_for_confirmatory(provenance)
    runner = ManualEvaluationRunner(
        store=store,
        inputs=replace(context.inputs, sampled=tuple(sampled)),
        config=config,
        evaluation_file=evaluation_file,
        scenarios=cases,
        trackers={name: load_frozen_baseline(name) for name in RECOVERY_TRACKERS},
        root=root,
        execution=execution,
        provenance=provenance,
        command=command,
    )
    return _Prepared(
        context=context,
        config=config,
        runner=runner,
        execution=execution,
        locked_scenarios=locked_scenarios,
    )


def _evaluate_model(args: argparse.Namespace) -> int:
    """Worker subcommand: evaluate one model and leave its evidence in the store."""
    started_at = datetime.now(tz=UTC)
    prepared = prepare_runner(
        args,
        role="worker",
        root=Path(cast("str", args.root)),
        scenario_ids=cast("list[str] | None", args.scenarios),
    )
    prepared_at = datetime.now(tz=UTC)
    entries = evaluation_entries(prepared.context.manifest, [cast("str", args.entry)])
    prepared.runner.evaluate(entries[0], warmup_s=float(cast("str", args.warmup_s)))
    destination = cast("str | None", args.timings)
    if destination is not None:
        runner = prepared.runner
        carried = ManualWorkerTimings(
            runs=runner.run_timings,
            models=tuple(runner.model_timings.values()),
            manifest_bytes=runner.manifest_bytes,
            phases=runner.phase_timings,
            span=ManualWorkerSpan(
                pid=os.getpid(),
                label=cast("str", args.entry),
                started_at=_instant(started_at),
                prepared_at=_instant(prepared_at),
                finished_at=_instant(datetime.now(tz=UTC)),
            ),
        )
        Path(destination).write_text(json.dumps(to_mapping(carried), sort_keys=True), encoding="utf-8")
    return 0


def _run(args: argparse.Namespace) -> int:
    """Evaluate the selected models of the frozen study and leave pointers to what was produced."""
    workers = int(cast("int", args.workers))
    if workers < 1:
        # Checked before anything expensive: zero workers would otherwise quietly evaluate nothing.
        msg = f"workers must be at least 1, got {workers}"
        raise ValueError(msg)
    prepared = prepare_runner(args, role="main", root=repository_root())
    context, runner, execution = prepared.context, prepared.runner, prepared.execution
    entries = evaluation_entries(context.manifest, cast("list[str] | None", args.entries))
    study_file = Path(cast("str", args.study))
    if workers == 1:
        evidences = [runner.evaluate(entry, warmup_s=context.inputs.configuration(entry).warmup_s) for entry in entries]
    else:
        # Models are dispatched by their inherited warm-up, which is part of the conditions. Within
        # a group, replay baselines are shared only by models that also share a parent and a
        # derivative policy, so a group may need several banks; the parallel path builds them first.
        groups: dict[float, list[StudyModel]] = {}
        for entry in entries:
            groups.setdefault(context.inputs.configuration(entry).warmup_s, []).append(entry)
        evidences: list[ManualModelEvidence] = []
        for warmup_s, group in sorted(groups.items()):
            evidences.extend(
                evaluate_in_parallel(
                    runner,
                    group,
                    warmup_s=warmup_s,
                    workers=workers,
                    env=os.environ,
                    spawn=partial(
                        spawn_worker,
                        study_file=study_file,
                        evaluation_file=Path(cast("str", args.evaluation)),
                        root=repository_root(),
                        exploratory=bool(args.exploratory),
                    ),
                )
            )
    written = runner.write_pointers(Path(cast("str", args.evidence_dir)))
    print(
        json.dumps(
            {
                "models": len(evidences),
                "pairs": sum(e.n_pairs for e in evidences),
                "completed": sum(e.n_completed for e in evidences),
                "infeasible": sum(e.n_infeasible for e in evidences),
                "statuses": {s: sum(1 for e in evidences if e.status == s) for s in MODEL_STATUSES},
                "pointers_written": len(written),
                "workers": workers,
                "study_manifest": sha256_file(study_file),
                "execution_identity": execution.identity,
                "exploratory": bool(args.exploratory),
            },
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Evaluate the manual-demonstration study against paired replay.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    run = subparsers.add_parser("run", help="evaluate study models with paired replay of their parents")
    run.add_argument("--study", type=str, required=True, help="frozen study manifest JSON")
    run.add_argument("--evaluation", type=str, required=True, help="manual evaluation config TOML")
    run.add_argument("--evidence-dir", type=str, required=True, help="directory of the Git pointer records")
    run.add_argument("--entries", type=str, nargs="*", default=None, help="model labels (default: every model)")
    run.add_argument("--workers", type=int, default=1, help="models evaluated at once in worker processes")
    run.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    worker = subparsers.add_parser("evaluate-model", help="evaluate one model (spawned by the run command)")
    worker.add_argument("--study", type=str, required=True, help="frozen study manifest JSON")
    worker.add_argument("--evaluation", type=str, required=True, help="manual evaluation config TOML")
    worker.add_argument("--entry", type=str, required=True, help="model label, e.g. feasible-best/S/D01")
    worker.add_argument("--warmup-s", type=str, required=True, help="the model configuration's warm-up")
    worker.add_argument("--root", type=str, required=True, help="repository root the study is bound to")
    worker.add_argument(
        "--scenarios",
        type=str,
        nargs="*",
        default=None,
        help="scenario ids this worker evaluates (default: every locked case)",
    )
    worker.add_argument(
        "--timings",
        type=str,
        default=None,
        help="write this worker's measured timings here, for the parent that reports them",
    )
    worker.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    args.argv = argv
    return _evaluate_model(args) if args.subcommand == "evaluate-model" else _run(args)


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
