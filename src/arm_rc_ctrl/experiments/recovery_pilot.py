# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

r"""Recovery safety pilot: method-independent perturbation levels on the recovery schedule (M3R-011).

Starting from M3's levels, initial-posture offsets and endpoint force pulses
are swept with the frozen direct-replay baselines under the recovery timing
protocol: every run holds its own (possibly offset) initial posture
``q0_ref + delta`` through the common pre-task hold, activates the cropped
reference at task time zero, and takes force pulses on the **task clock**.
The M3 selection machinery (grids, safe/nontrivial rules, level summaries,
selection) is reused unchanged over these recovery-schedule cases; the report
keeps every case, and the selected envelope justifies the locked recovery
evaluation protocols.

Usage::

    python -m arm_rc_ctrl.experiments.recovery_pilot \\
        --protocol configs/studies/task_1a_recovery_pilot_v1.toml \\
        --dataset data/records/processed/<id>.toml \\
        --report docs/experiments/task_1a_state_conditioned_recovery/recovery_pilot_v1.json \\
        --markdown docs/experiments/task_1a_state_conditioned_recovery/recovery_pilot_v1.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np

from arm_rc_ctrl.config import from_mapping, load_config, to_mapping
from arm_rc_ctrl.controllers.tracking import LimitedTracker, TrackerConfig
from arm_rc_ctrl.data.records import load_record, verify_payload
from arm_rc_ctrl.data.recovery import RecoveryDatasetRecord, TaskIntervals, task_intervals_from_phases
from arm_rc_ctrl.data.samples import SampleSet, load_samples
from arm_rc_ctrl.experiments.baselines import load_frozen_baseline
from arm_rc_ctrl.experiments.disturbances import ForcePulse
from arm_rc_ctrl.experiments.perturbation_pilot import (
    ForceSweep,
    LevelOutcome,
    PilotCase,
    PilotProtocol,
    PostureSweep,
    Selection,
    SelectionRules,
    render_markdown,
    select_levels,
    summarize_levels,
)
from arm_rc_ctrl.experiments.perturbation_pilot import (
    PilotReport as _CorePilotReport,
)
from arm_rc_ctrl.experiments.recovery_slice import HeldTaskReference
from arm_rc_ctrl.experiments.simulation import simulate
from arm_rc_ctrl.metrics.dwell import dwell_metrics
from arm_rc_ctrl.metrics.joint import JointAnglePolicy, joint_rmse
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    ProvenanceRecord,
    canonical_json,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
)
from arm_rc_ctrl.rc.warmup import WarmupConfig
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import ScenarioConfig, load_scenario
from arm_rc_ctrl.storage import open_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from arm_rc_ctrl.experiments.run_record import RunArrays
    from arm_rc_ctrl.experiments.termination import Termination

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "RecoveryPilotProtocol",
    "RecoveryPilotReport",
    "as_core",
    "force_pulse_on_run_clock",
    "load_recovery_pilot_protocol",
    "load_recovery_pilot_report",
    "main",
    "render_recovery_markdown",
    "run_recovery_cases",
    "run_recovery_pilot",
]

REPORT_SCHEMA_VERSION: Final = 1
_GRID_TOLERANCE_S: Final = 1e-9


@dataclass(frozen=True)
class RecoveryPilotProtocol:
    """A versioned recovery perturbation pilot (force timing on the task clock)."""

    name: str
    scenario: Path
    baselines: tuple[str, ...]
    warmup_s: float
    """The common pre-task hold of every pilot run (an approved D2 duration)."""
    posture: PostureSweep
    force: ForceSweep
    """``start_s`` is task-relative; :func:`force_pulse_on_run_clock` shifts it by the warm-up."""
    selection: SelectionRules

    def __post_init__(self) -> None:
        """Name and baselines are non-empty and unique; the warm-up is an approved duration."""
        if not self.name.strip():
            msg = "name must not be empty"
            raise ValueError(msg)
        if not self.baselines or len(set(self.baselines)) != len(self.baselines):
            msg = f"baselines must be a non-empty list of distinct methods, got {self.baselines}"
            raise ValueError(msg)
        WarmupConfig(self.warmup_s)  # rejects durations outside the approved D2 set


def load_recovery_pilot_protocol(path: Path) -> RecoveryPilotProtocol:
    """Load and validate a recovery pilot protocol."""
    return load_config(path, RecoveryPilotProtocol)


def as_core(protocol: RecoveryPilotProtocol) -> PilotProtocol:
    """The M3-shaped core the shared level summariser and selector operate on."""
    return PilotProtocol(
        name=protocol.name,
        scenario=protocol.scenario,
        baselines=protocol.baselines,
        posture=protocol.posture,
        force=protocol.force,
        selection=protocol.selection,
    )


def force_pulse_on_run_clock(protocol: RecoveryPilotProtocol, magnitude: float, direction_deg: float) -> ForcePulse:
    """The task-relative pulse of one case shifted onto the run clock (never inside the hold)."""
    pulse = protocol.force.pulse(magnitude, direction_deg)
    return ForcePulse(start_s=pulse.start_s + protocol.warmup_s, duration_s=pulse.duration_s, force=pulse.force)


@dataclass(frozen=True)
class RecoveryPilotReport:
    """Every case, every level classification, the selection, and provenance of one recovery pilot."""

    protocol: str
    scenario_file: str
    dataset: str
    warmup_s: float
    baselines: dict[str, TrackerConfig]
    rules: SelectionRules
    cases: tuple[PilotCase, ...]
    levels: tuple[LevelOutcome, ...]
    selection: Selection
    provenance: ProvenanceRecord
    schema_version: int = field(default=REPORT_SCHEMA_VERSION)


def load_recovery_pilot_report(path: Path) -> RecoveryPilotReport:
    """Load a report written by :func:`main`."""
    data = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    return from_mapping(data, RecoveryPilotReport)


@dataclass(frozen=True)
class _Reference:
    scenario: ScenarioConfig
    samples: SampleSet
    task: TaskIntervals
    move: NDArray[np.bool_]
    policy: JointAnglePolicy
    warmup_s: float
    interpolation: str

    @property
    def duration_s(self) -> float:
        """Hold plus the full task episode."""
        return self.warmup_s + float(self.samples.t[-1])


def _reference(
    scenario: ScenarioConfig, samples: SampleSet, dataset: RecoveryDatasetRecord, warmup_s: float
) -> _Reference:
    task = task_intervals_from_phases(samples.t, samples.phase)
    move = (samples.t >= task.move[0]) & (samples.t < task.move[1])
    return _Reference(
        scenario,
        samples,
        task,
        move,
        JointAnglePolicy.limited(scenario.dof),
        warmup_s,
        dataset.preprocessing.interpolation,
    )


def _criteria(ref: _Reference, arrays: RunArrays, termination: Termination) -> dict[str, bool]:
    """Completion plus the scenario dwell criteria over the task dwell window on the run clock."""
    rules = ref.scenario.task.dwell_criteria
    result = {"completed": termination.is_completed, **dict.fromkeys(rules.names, False)}
    if not termination.is_completed:
        return result
    window = (ref.warmup_s + ref.task.dwell[0], ref.warmup_s + ref.task.dwell[1])
    run_t = cast("NDArray[np.float64]", arrays.arrays["t"])
    if not bool(np.any((run_t >= window[0]) & (run_t <= window[1]))):
        return result
    metrics = dwell_metrics(
        run_t,
        cast("NDArray[np.float64]", arrays.arrays["tip"]),
        cast("NDArray[np.float64]", arrays.arrays["dq"]),
        np.asarray(ref.scenario.task.target, dtype=np.float64),
        rules.tolerance,
        window=window,
    )
    result.update(rules.evaluate(metrics))
    return result


def _evaluate(
    ref: _Reference,
    gains: TrackerConfig,
    *,
    initial_q: tuple[float, ...],
    force_run: ForcePulse | None,
    onset_task_s: float,
) -> tuple[Termination, dict[str, bool], float | None, float, float | None, float, float, float]:
    """One replay run on the recovery schedule; onset times are task-relative."""
    held = HeldTaskReference.from_samples(
        ref.samples,
        activation_s=ref.warmup_s,
        interpolation=ref.interpolation,
        hold=np.asarray(initial_q, dtype=np.float64),
    )
    controller = LimitedTracker(cast("Any", held), gains, ref.scenario.limits.torque)
    arrays, termination = simulate(
        ref.scenario, controller, duration_s=ref.duration_s, initial_q=initial_q, force=force_run
    )
    a = arrays.arrays
    run_t = cast("NDArray[np.float64]", a["t"])
    active = run_t >= ref.warmup_s - _GRID_TOLERANCE_S
    q_active = cast("NDArray[np.float64]", a["q"])[active]
    tip_active = cast("NDArray[np.float64]", a["tip"])[active]
    n = min(q_active.shape[0], ref.samples.n_samples)
    task_t = run_t[active][:n] - ref.warmup_s
    rmse: float | None = None
    move = ref.move[:n]
    if termination.is_completed and bool(move.any()):
        rmse = joint_rmse(q_active[:n][move], ref.samples.q[:n][move], ref.policy).aggregate
    criteria = _criteria(ref, arrays, termination)
    difference = tip_active[:n] - ref.samples.tip[:n]
    deviation = np.sqrt(np.sum(difference * difference, axis=1))
    after = task_t >= onset_task_s - _GRID_TOLERANCE_S
    peak_deviation = float(deviation[after].max()) if bool(after.any()) else 0.0
    outside = after & (deviation >= ref.scenario.task.tolerance)
    recovery: float | None
    if not termination.is_completed:
        recovery = None
    elif bool(outside.any()):
        recovery = float(task_t[np.flatnonzero(outside)[-1]] + ref.scenario.timing.dt - onset_task_s)
    else:
        recovery = 0.0
    limits = np.asarray(ref.scenario.limits.torque, dtype=np.float64)
    torque_fraction = float((np.abs(cast("NDArray[np.float64]", a["tau_applied"])) / limits).max())
    saturation = float(cast("NDArray[np.int64]", a["saturation"]).mean())
    velocity = float(np.abs(cast("NDArray[np.float64]", a["dq"])).max())
    return termination, criteria, rmse, peak_deviation, recovery, torque_fraction, saturation, velocity


def _case(
    ref: _Reference,
    baseline: str,
    gains: TrackerConfig,
    kind: str,
    magnitude: float,
    direction: tuple[float, ...],
    *,
    initial_q: tuple[float, ...],
    force_run: ForcePulse | None,
    onset_task_s: float,
) -> PilotCase:
    termination, criteria, rmse, deviation, recovery, torque, saturation, velocity = _evaluate(
        ref, gains, initial_q=initial_q, force_run=force_run, onset_task_s=onset_task_s
    )
    return PilotCase(
        baseline=baseline,
        kind=cast("Any", kind),
        magnitude=magnitude,
        direction=direction,
        initial_q=initial_q,
        termination=termination,
        criteria=criteria,
        success=termination.is_completed and all(criteria.values()),
        move_joint_rmse=rmse,
        peak_deviation_m=deviation,
        recovery_time_s=recovery,
        peak_torque_fraction=torque,
        saturation_fraction=saturation,
        peak_velocity=velocity,
    )


def run_recovery_cases(
    protocol: RecoveryPilotProtocol,
    scenario: ScenarioConfig,
    samples: SampleSet,
    dataset: RecoveryDatasetRecord,
    baselines: dict[str, TrackerConfig],
) -> tuple[PilotCase, ...]:
    """Simulate every (baseline, kind, magnitude, direction) case on the recovery schedule.

    Posture offsets are added to ``q0_ref`` (never the scenario metadata) and
    held through the pre-task interval; their recovery clock starts at
    activation. Force cases start from ``q0_ref`` with the pulse shifted onto
    the run clock; their recovery clock starts at the task-relative onset.
    """
    ref = _reference(scenario, samples, dataset, protocol.warmup_s)
    q0 = np.asarray(dataset.q0_ref, dtype=np.float64)
    cases: list[PilotCase] = []
    for baseline, gains in baselines.items():
        if gains.dof != scenario.dof:
            msg = f"baseline {baseline!r} has {gains.dof} joints, scenario {scenario.dof}"
            raise ValueError(msg)
        for magnitude in protocol.posture.magnitudes:
            for index in range(len(protocol.posture.directions)):
                unit = protocol.posture.unit(index)
                if unit.shape[0] != scenario.dof:
                    msg = f"posture direction {index} has {unit.shape[0]} entries, expected {scenario.dof}"
                    raise ValueError(msg)
                initial_q = tuple(float(v) for v in q0 + magnitude * unit)
                cases.append(
                    _case(
                        ref,
                        baseline,
                        gains,
                        "posture",
                        magnitude,
                        tuple(float(v) for v in unit),
                        initial_q=initial_q,
                        force_run=None,
                        onset_task_s=0.0,
                    )
                )
        cases.extend(
            _case(
                ref,
                baseline,
                gains,
                "force",
                magnitude,
                (direction_deg,),
                initial_q=tuple(float(v) for v in dataset.q0_ref),
                force_run=force_pulse_on_run_clock(protocol, magnitude, direction_deg),
                onset_task_s=protocol.force.start_s,
            )
            for magnitude in protocol.force.magnitudes
            for direction_deg in protocol.force.directions_deg
        )
    return tuple(cases)


def _repo_relative(path: Path) -> str:
    """Repository-relative POSIX path, or the bare name for files outside the checkout (no machine paths)."""
    root = repository_root()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name


def run_recovery_pilot(
    protocol: RecoveryPilotProtocol,
    protocol_file: Path,
    dataset: RecoveryDatasetRecord,
    samples: SampleSet,
    *,
    exploratory: bool,
    now: datetime | None = None,
    command: str = "python -m arm_rc_ctrl.experiments.recovery_pilot",
) -> RecoveryPilotReport:
    """Run the whole recovery pilot with the frozen baselines and build the report."""
    scenario = load_scenario(protocol.scenario)
    dataset.check_scenario(protocol.scenario)
    dataset.check_samples(samples)
    if scenario.dof != dataset.dof:
        msg = f"dof mismatch: scenario {scenario.dof}, dataset {dataset.dof}"
        raise ValueError(msg)
    baselines = {method: load_frozen_baseline(method) for method in protocol.baselines}
    scenario_file = _repo_relative(protocol.scenario)
    protocol_mapping = to_mapping(protocol)
    protocol_mapping["scenario"] = scenario_file  # records never carry machine paths
    resolved = {
        "protocol": protocol_mapping,
        "protocol_file": _repo_relative(protocol_file),
        "baselines": {method: to_mapping(gains) for method, gains in baselines.items()},
        "dataset": dataset.artifact.artifact_id,
        "warmup_s": protocol.warmup_s,
        "command": command,
    }
    payload = dataset.artifact.payload
    provenance = collect_provenance(
        resolved,
        seeds={},
        artifacts=[ArtifactReference(payload.uri, payload.sha256, payload.size)],
        exploratory=exploratory,
        now=now,
    )
    require_clean_for_confirmatory(provenance)
    cases = run_recovery_cases(protocol, scenario, samples, dataset, baselines)
    core = as_core(protocol)
    levels = summarize_levels(core, cases)
    return RecoveryPilotReport(
        protocol=protocol.name,
        scenario_file=scenario_file,
        dataset=dataset.artifact.artifact_id,
        warmup_s=protocol.warmup_s,
        baselines=baselines,
        rules=protocol.selection,
        cases=cases,
        levels=levels,
        selection=select_levels(core, levels),
        provenance=provenance,
    )


def render_recovery_markdown(report: RecoveryPilotReport) -> str:
    """The M3 pilot tables with the recovery-schedule preamble."""
    core = _CorePilotReport(
        protocol=report.protocol,
        scenario_file=report.scenario_file,
        dataset=report.dataset,
        baselines=report.baselines,
        rules=report.rules,
        cases=report.cases,
        levels=report.levels,
        selection=report.selection,
        provenance=report.provenance,
    )
    lines = render_markdown(core).splitlines()
    schedule = (
        f"Recovery schedule: every run holds its own initial posture (q0_ref + offset) for the common "
        f"T_w = {report.warmup_s:g} s hold, activates the cropped reference at task time zero, and takes "
        f"force pulses on the task clock (start {report.selection.force_start_s:g} s after activation); "
        "recovery times are measured on the task clock."
    )
    lines.insert(2, schedule)
    lines.insert(3, "")
    return "\n".join(lines) + ("\n" if lines[-1] != "" else "")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run the recovery perturbation pilot with the frozen baselines.")
    parser.add_argument("--protocol", type=Path, required=True, help="pilot protocol (configs/studies/*.toml)")
    parser.add_argument("--dataset", type=Path, required=True, help="recovery dataset record (TOML)")
    parser.add_argument("--report", type=Path, required=True, help="JSON report to write (must not exist)")
    parser.add_argument("--markdown", type=Path, default=None, help="optional Markdown to write (must not exist)")
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    for target in (args.report, args.markdown):
        if target is not None and Path(target).exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    protocol = load_recovery_pilot_protocol(Path(args.protocol))
    store = open_storage()
    dataset = load_record(Path(args.dataset), RecoveryDatasetRecord)
    samples = load_samples(verify_payload(store, dataset.artifact))
    report = run_recovery_pilot(
        protocol,
        Path(args.protocol),
        dataset,
        samples,
        exploratory=bool(args.exploratory),
        now=datetime.now(tz=UTC),
        command=command_line("arm_rc_ctrl.experiments.recovery_pilot", sys.argv[1:] if argv is None else argv),
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(canonical_json(to_mapping(report)) + "\n", encoding="utf-8")
    if args.markdown is not None:
        Path(args.markdown).write_text(render_recovery_markdown(report), encoding="utf-8")
    selection = report.selection
    print(
        json.dumps(
            {
                "cases": len(report.cases),
                "posture_small_rad": selection.posture_small_rad,
                "posture_large_rad": selection.posture_large_rad,
                "force_magnitude_n": selection.force_magnitude_n,
                "warmup_s": report.warmup_s,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
