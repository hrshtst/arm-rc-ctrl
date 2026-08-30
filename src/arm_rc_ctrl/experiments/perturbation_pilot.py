# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

r"""Pilot that calibrates confirmatory perturbation levels with the frozen baselines (M1-028).

Following docs/PLAN.md section 9.2, initial-posture offsets and endpoint force
pulses are swept over a grid of magnitudes and directions with every frozen
direct-replay baseline. Explicit, versioned rules then classify each level as
safe and/or nontrivial and select the levels that the confirmatory protocol
locks. Every case is kept in the report; the selection is never the only
saved result.

Usage::

    python -m arm_rc_ctrl.experiments.perturbation_pilot --protocol configs/studies/perturbation_pilot_1a.toml \\
        --dataset data/records/processed/<id>.toml --report docs/experiments/task_1a/perturbation_pilot.json \\
        --markdown docs/experiments/task_1a/perturbation_pilot.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, cast

import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.config import from_mapping, load_config, to_mapping
from arm_rc_ctrl.controllers.reference import DemonstrationReference
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.phases import intervals_from_phases
from arm_rc_ctrl.data.records import ProcessedDatasetRecord, load_record, verify_payload
from arm_rc_ctrl.data.samples import SampleSet, load_samples
from arm_rc_ctrl.experiments.baselines import load_frozen_baseline
from arm_rc_ctrl.experiments.disturbances import ForcePulse
from arm_rc_ctrl.experiments.replay import bind_dataset, dwell_outcome, simulate_tracking
from arm_rc_ctrl.experiments.termination import Termination
from arm_rc_ctrl.metrics.joint import JointAnglePolicy, joint_rmse
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    ProvenanceRecord,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import ScenarioConfig, load_scenario
from arm_rc_ctrl.storage import open_storage
from arm_rc_ctrl.validation import require_finite

__all__ = [
    "BaselineLevel",
    "ForceSweep",
    "LevelOutcome",
    "PilotCase",
    "PilotProtocol",
    "PilotReport",
    "PostureSweep",
    "Selection",
    "SelectionRules",
    "load_pilot_report",
    "load_protocol",
    "main",
    "render_markdown",
    "run_pilot",
    "select_levels",
    "summarize_levels",
]

REPORT_SCHEMA_VERSION: Final = 1
type PerturbationKind = Literal["posture", "force"]


def _positive_increasing(values: tuple[float, ...], name: str) -> None:
    require_finite(values, name)
    if not values or values[0] <= 0 or any(b <= a for a, b in itertools.pairwise(values)):
        msg = f"{name} must be positive and strictly increasing, got {values}"
        raise ValueError(msg)


@dataclass(frozen=True)
class PostureSweep:
    """Initial-posture offsets: magnitudes (rad) along normalized joint-space directions."""

    magnitudes: tuple[float, ...]
    directions: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        """Magnitudes increase; directions are non-zero and share one dimension."""
        _positive_increasing(self.magnitudes, "posture.magnitudes")
        if not self.directions:
            msg = "posture.directions must not be empty"
            raise ValueError(msg)
        for direction in self.directions:
            require_finite(direction, "posture.directions")
            if len(direction) != len(self.directions[0]) or not any(direction):
                msg = f"posture directions must be non-zero vectors of one dimension, got {direction}"
                raise ValueError(msg)

    def unit(self, index: int) -> NDArray[np.float64]:
        """Direction ``index`` normalized to unit length."""
        vector = np.asarray(self.directions[index], dtype=np.float64)
        return vector / np.linalg.norm(vector)


@dataclass(frozen=True)
class ForceSweep:
    """Endpoint force pulses: magnitudes (N) along base-frame directions during the movement."""

    magnitudes: tuple[float, ...]
    directions_deg: tuple[float, ...]
    start_s: float
    duration_s: float

    def __post_init__(self) -> None:
        """Magnitudes increase; timing is a valid pulse window."""
        _positive_increasing(self.magnitudes, "force.magnitudes")
        require_finite(self.directions_deg, "force.directions_deg")
        if not self.directions_deg:
            msg = "force.directions_deg must not be empty"
            raise ValueError(msg)
        ForcePulse(self.start_s, self.duration_s, (0.0, 0.0))  # validates the window

    def pulse(self, magnitude: float, direction_deg: float) -> ForcePulse:
        """The pulse of one case."""
        return ForcePulse.from_polar(self.start_s, self.duration_s, magnitude, direction_deg)


@dataclass(frozen=True)
class SelectionRules:
    """Versioned definitions of *safe* and *nontrivial* (see the pilot protocol file)."""

    posture_recovery_min_s: float
    posture_recovery_max_s: float
    force_recovery_max_s: float
    force_deviation_min_m: float
    force_max_saturation_fraction: float
    """Largest fraction of saturated samples a *safe* force level may show (0 = full actuator headroom)."""

    def __post_init__(self) -> None:
        """Bounds are positive and consistent."""
        values = (
            self.posture_recovery_min_s,
            self.posture_recovery_max_s,
            self.force_recovery_max_s,
            self.force_deviation_min_m,
        )
        require_finite((*values, self.force_max_saturation_fraction), "selection")
        if any(v <= 0 for v in values):
            msg = f"selection bounds must be positive, got {values}"
            raise ValueError(msg)
        if not 0 <= self.force_max_saturation_fraction <= 1:
            msg = f"force_max_saturation_fraction must lie in [0, 1], got {self.force_max_saturation_fraction}"
            raise ValueError(msg)
        if self.posture_recovery_min_s > self.posture_recovery_max_s:
            msg = "posture_recovery_min_s must not exceed posture_recovery_max_s"
            raise ValueError(msg)


@dataclass(frozen=True)
class PilotProtocol:
    """A versioned perturbation pilot."""

    name: str
    scenario: Path
    baselines: tuple[str, ...]
    posture: PostureSweep
    force: ForceSweep
    selection: SelectionRules

    def __post_init__(self) -> None:
        """Name and baselines are non-empty and unique."""
        if not self.name.strip():
            msg = "name must not be empty"
            raise ValueError(msg)
        if not self.baselines or len(set(self.baselines)) != len(self.baselines):
            msg = f"baselines must be a non-empty list of distinct methods, got {self.baselines}"
            raise ValueError(msg)


def load_protocol(path: Path) -> PilotProtocol:
    """Load and validate a pilot protocol."""
    return load_config(path, PilotProtocol)


@dataclass(frozen=True)
class PilotCase:
    """One baseline under one perturbation."""

    baseline: str
    kind: PerturbationKind
    magnitude: float
    direction: tuple[float, ...]
    """Unit joint-space direction (posture) or ``(direction_deg,)`` (force)."""
    initial_q: tuple[float, ...]
    termination: str
    criteria: dict[str, bool]
    success: bool
    move_joint_rmse: float | None
    peak_deviation_m: float
    """Largest endpoint distance from the reference endpoint from the perturbation onset on."""
    recovery_time_s: float | None
    """Time after onset until the endpoint stays within the task tolerance; ``None`` when it never does."""
    peak_torque_fraction: float
    """Largest ``|tau_applied| / limit`` over joints and samples."""
    saturation_fraction: float
    peak_velocity: float


@dataclass(frozen=True)
class BaselineLevel:
    """Worst case of one baseline over the directions of one level."""

    success_all: bool
    terminations: tuple[str, ...]
    recovered_all: bool
    max_recovery_time_s: float | None
    """Slowest recovery over the directions; ``None`` when any direction never recovered."""
    max_peak_deviation_m: float
    max_peak_torque_fraction: float
    max_saturation_fraction: float
    max_peak_velocity: float


@dataclass(frozen=True)
class LevelOutcome:
    """Classification of one magnitude of one perturbation kind."""

    kind: PerturbationKind
    magnitude: float
    safe: bool
    nontrivial: bool
    baselines: dict[str, BaselineLevel]


@dataclass(frozen=True)
class Selection:
    """Levels the confirmatory protocol locks."""

    posture_small_rad: float
    posture_large_rad: float
    force_magnitude_n: float
    force_start_s: float
    force_duration_s: float
    force_directions_deg: tuple[float, ...]

    def __post_init__(self) -> None:
        """Small never exceeds large."""
        if self.posture_small_rad > self.posture_large_rad:
            msg = "posture_small_rad must not exceed posture_large_rad"
            raise ValueError(msg)


@dataclass(frozen=True)
class PilotReport:
    """Every case, every level classification, the selection, and provenance."""

    protocol: str
    scenario_file: str
    dataset: str
    baselines: dict[str, TrackerConfig]
    rules: SelectionRules
    cases: tuple[PilotCase, ...]
    levels: tuple[LevelOutcome, ...]
    selection: Selection
    provenance: ProvenanceRecord
    schema_version: int = field(default=REPORT_SCHEMA_VERSION)


def load_pilot_report(path: Path) -> PilotReport:
    """Load a report written by :func:`main`."""
    data = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    return from_mapping(data, PilotReport)


@dataclass(frozen=True)
class _Reference:
    scenario: ScenarioConfig
    samples: SampleSet
    demo: DemonstrationReference
    move: NDArray[np.bool_]
    policy: JointAnglePolicy

    @property
    def duration_s(self) -> float:
        return float(self.samples.t[-1])


def _evaluate(
    ref: _Reference,
    gains: TrackerConfig,
    *,
    initial_q: tuple[float, ...] | None,
    force: ForcePulse | None,
    onset_s: float,
) -> tuple[Termination, dict[str, bool], float | None, float, float | None, float, float, float]:
    scenario, samples = ref.scenario, ref.samples
    arrays, termination = simulate_tracking(
        scenario, ref.demo, gains, duration_s=ref.duration_s, initial_q=initial_q, force=force
    )
    a = arrays.arrays
    q = cast("NDArray[np.float64]", a["q"])
    n = min(q.shape[0], samples.n_samples)
    move = ref.move[:n]
    rmse: float | None = None
    if termination.is_completed and move.any():
        rmse = joint_rmse(q[:n][move], samples.q[:n][move], ref.policy).aggregate
    criteria = dwell_outcome(scenario, samples, arrays, termination)
    t = cast("NDArray[np.float64]", a["t"])[:n]
    deviation = np.linalg.norm(cast("NDArray[np.float64]", a["tip"])[:n] - samples.tip[:n], axis=1)
    after = t >= onset_s
    peak_deviation = float(deviation[after].max()) if after.any() else 0.0
    outside = after & (deviation >= scenario.task.tolerance)
    recovery: float | None
    if not termination.is_completed:
        recovery = None
    elif outside.any():
        recovery = float(t[np.flatnonzero(outside)[-1]] + scenario.timing.dt - onset_s)
    else:
        recovery = 0.0
    limits = np.asarray(scenario.limits.torque, dtype=np.float64)
    torque_fraction = float((np.abs(cast("NDArray[np.float64]", a["tau_applied"])) / limits).max())
    saturation = float(cast("NDArray[np.int64]", a["saturation"]).mean())
    velocity = float(np.abs(cast("NDArray[np.float64]", a["dq"])).max())
    return termination, criteria, rmse, peak_deviation, recovery, torque_fraction, saturation, velocity


def _reference(scenario: ScenarioConfig, samples: SampleSet, dataset: ProcessedDatasetRecord) -> _Reference:
    demo = DemonstrationReference.from_samples(
        samples, cast("Literal['linear', 'cubic']", dataset.preprocessing.interpolation)
    )
    intervals = intervals_from_phases(samples.t, samples.phase)
    move = (samples.t >= intervals.move[0]) & (samples.t < intervals.move[1])
    return _Reference(scenario, samples, demo, move, JointAnglePolicy.limited(scenario.dof))


def _case(
    ref: _Reference,
    baseline: str,
    gains: TrackerConfig,
    kind: PerturbationKind,
    magnitude: float,
    direction: tuple[float, ...],
    *,
    initial_q: tuple[float, ...],
    force: ForcePulse | None,
    onset_s: float,
) -> PilotCase:
    termination, criteria, rmse, deviation, recovery, torque, saturation, velocity = _evaluate(
        ref, gains, initial_q=initial_q, force=force, onset_s=onset_s
    )
    return PilotCase(
        baseline=baseline,
        kind=kind,
        magnitude=magnitude,
        direction=direction,
        initial_q=initial_q,
        termination=termination.kind,
        criteria=criteria,
        success=termination.is_completed and all(criteria.values()),
        move_joint_rmse=rmse,
        peak_deviation_m=deviation,
        recovery_time_s=recovery,
        peak_torque_fraction=torque,
        saturation_fraction=saturation,
        peak_velocity=velocity,
    )


def run_cases(
    protocol: PilotProtocol,
    scenario: ScenarioConfig,
    samples: SampleSet,
    dataset: ProcessedDatasetRecord,
    baselines: dict[str, TrackerConfig],
) -> tuple[PilotCase, ...]:
    """Simulate every (baseline, kind, magnitude, direction) case of the protocol."""
    ref = _reference(scenario, samples, dataset)
    nominal = np.asarray(scenario.task.initial_q, dtype=np.float64)
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
                initial_q = tuple(float(v) for v in nominal + magnitude * unit)
                cases.append(
                    _case(
                        ref,
                        baseline,
                        gains,
                        "posture",
                        magnitude,
                        tuple(float(v) for v in unit),
                        initial_q=initial_q,
                        force=None,
                        onset_s=0.0,
                    )
                )
        for magnitude in protocol.force.magnitudes:
            for direction_deg in protocol.force.directions_deg:
                pulse = protocol.force.pulse(magnitude, direction_deg)
                cases.append(
                    _case(
                        ref,
                        baseline,
                        gains,
                        "force",
                        magnitude,
                        (direction_deg,),
                        initial_q=tuple(scenario.task.initial_q),
                        force=pulse,
                        onset_s=pulse.start_s,
                    )
                )
    return tuple(cases)


def _baseline_level(cases: Sequence[PilotCase]) -> BaselineLevel:
    recoveries = [c.recovery_time_s for c in cases]
    recovered = all(r is not None for r in recoveries)
    return BaselineLevel(
        success_all=all(c.success for c in cases),
        terminations=tuple(sorted({c.termination for c in cases})),
        recovered_all=recovered,
        max_recovery_time_s=max(r for r in recoveries if r is not None) if recovered else None,
        max_peak_deviation_m=max(c.peak_deviation_m for c in cases),
        max_peak_torque_fraction=max(c.peak_torque_fraction for c in cases),
        max_saturation_fraction=max(c.saturation_fraction for c in cases),
        max_peak_velocity=max(c.peak_velocity for c in cases),
    )


def _within(recovery: float | None, bound: float) -> bool:
    """Whether every direction recovered, and the slowest one within ``bound`` seconds."""
    return recovery is not None and recovery <= bound


def summarize_levels(
    protocol: PilotProtocol, cases: Sequence[PilotCase], rules: SelectionRules | None = None
) -> tuple[LevelOutcome, ...]:
    """Classify every level of both kinds from the cases, per the selection rules."""
    rules = protocol.selection if rules is None else rules
    levels: list[LevelOutcome] = []
    for kind, magnitudes in (("posture", protocol.posture.magnitudes), ("force", protocol.force.magnitudes)):
        for magnitude in magnitudes:
            per_baseline: dict[str, BaselineLevel] = {}
            for baseline in protocol.baselines:
                subset = [c for c in cases if c.kind == kind and c.magnitude == magnitude and c.baseline == baseline]
                if not subset:
                    msg = f"no cases for {kind} level {magnitude:g} of baseline {baseline!r}"
                    raise ValueError(msg)
                per_baseline[baseline] = _baseline_level(subset)
            if kind == "posture":
                safe = all(
                    b.success_all and _within(b.max_recovery_time_s, rules.posture_recovery_max_s)
                    for b in per_baseline.values()
                )
                nontrivial = any(
                    b.max_recovery_time_s is None or b.max_recovery_time_s >= rules.posture_recovery_min_s
                    for b in per_baseline.values()
                )
            else:
                safe = all(
                    b.success_all
                    and _within(b.max_recovery_time_s, rules.force_recovery_max_s)
                    and b.max_saturation_fraction <= rules.force_max_saturation_fraction
                    for b in per_baseline.values()
                )
                nontrivial = any(b.max_peak_deviation_m >= rules.force_deviation_min_m for b in per_baseline.values())
            levels.append(LevelOutcome(cast("PerturbationKind", kind), magnitude, safe, nontrivial, per_baseline))
    return tuple(levels)


def select_levels(protocol: PilotProtocol, levels: Sequence[LevelOutcome]) -> Selection:
    """Apply the selection rule; fail when the grid holds no qualifying level."""

    def pick(kind: PerturbationKind, *, nontrivial: bool, largest: bool) -> float:
        candidates = [
            lv.magnitude for lv in levels if lv.kind == kind and lv.safe and (lv.nontrivial or not nontrivial)
        ]
        if not candidates:
            wanted = "safe and nontrivial" if nontrivial else "safe"
            msg = f"no {wanted} {kind} level in the pilot grid; widen or refine the grid before locking"
            raise ValueError(msg)
        return max(candidates) if largest else min(candidates)

    return Selection(
        posture_small_rad=pick("posture", nontrivial=True, largest=False),
        posture_large_rad=pick("posture", nontrivial=False, largest=True),
        force_magnitude_n=pick("force", nontrivial=True, largest=True),
        force_start_s=protocol.force.start_s,
        force_duration_s=protocol.force.duration_s,
        force_directions_deg=protocol.force.directions_deg,
    )


def _repo_relative(path: Path) -> str:
    """Repository-relative POSIX path, or the bare name for files outside the checkout (no machine paths)."""
    root = repository_root()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name


def run_pilot(
    protocol: PilotProtocol,
    protocol_file: Path,
    dataset: ProcessedDatasetRecord,
    samples: SampleSet,
    *,
    exploratory: bool,
    now: datetime | None = None,
    command: str = "python -m arm_rc_ctrl.experiments.perturbation_pilot",
) -> PilotReport:
    """Run the whole pilot on the dataset with the frozen baselines and build the report."""
    scenario = load_scenario(protocol.scenario)
    bind_dataset(scenario, protocol.scenario, dataset, samples)
    baselines = {method: load_frozen_baseline(method) for method in protocol.baselines}
    scenario_file = _repo_relative(protocol.scenario)
    protocol_mapping = to_mapping(protocol)
    protocol_mapping["scenario"] = scenario_file  # records never carry machine paths
    resolved = {
        "protocol": protocol_mapping,
        "protocol_file": _repo_relative(protocol_file),
        "baselines": {method: to_mapping(gains) for method, gains in baselines.items()},
        "dataset": dataset.artifact.artifact_id,
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
    cases = run_cases(protocol, scenario, samples, dataset, baselines)
    levels = summarize_levels(protocol, cases)
    return PilotReport(
        protocol=protocol.name,
        scenario_file=scenario_file,
        dataset=dataset.artifact.artifact_id,
        baselines=baselines,
        rules=protocol.selection,
        cases=cases,
        levels=levels,
        selection=select_levels(protocol, levels),
        provenance=provenance,
    )


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or math.isinf(value):
        return "—"
    return f"{value:.{digits}g}"


def render_markdown(report: PilotReport) -> str:
    """Human-readable tables of every level plus the selection and its rules."""
    rules = report.rules
    gains = ", ".join(f"`{m}` (kp {list(g.kp)}, kd {list(g.kd)})" for m, g in sorted(report.baselines.items()))
    dirty = " (dirty)" if report.provenance.project_dirty else ""
    lines = [
        f"# Perturbation pilot `{report.protocol}`",
        "",
        (
            f"Dataset `{report.dataset}`, scenario `{report.scenario_file}`, commit "
            f"`{report.provenance.project_commit[:12]}`{dirty}, baselines {gains}."
        ),
        "",
        (
            "Rules: a level is *safe* when every baseline completes, meets every dwell criterion, and recovers "
            f"(endpoint within tolerance of the reference) within {rules.posture_recovery_max_s:g} s (posture) / "
            f"{rules.force_recovery_max_s:g} s after the pulse (force, with at most "
            f"{rules.force_max_saturation_fraction:g} of the samples saturated); "
            f"*nontrivial* when a baseline needs >= {rules.posture_recovery_min_s:g} s to recover (posture) or "
            f"the endpoint deviates >= {rules.force_deviation_min_m:g} m (force)."
        ),
    ]
    columns = (
        "Baseline | Terminations | Success | Recovery max (s) | Peak deviation (m) | Peak torque fraction | "
        "Saturation | Peak velocity (rad/s) | Safe | Nontrivial"
    )
    for kind, unit in (("posture", "rad"), ("force", "N")):
        lines += ["", f"## {kind.capitalize()} levels", "", f"| Magnitude ({unit}) | {columns} |", "|---" * 11 + "|"]
        for level in (lv for lv in report.levels if lv.kind == kind):
            for baseline, b in sorted(level.baselines.items()):  # order-independent of the JSON/protocol
                cells = (
                    f"{level.magnitude:g}",
                    baseline,
                    ", ".join(b.terminations),
                    "yes" if b.success_all else "no",
                    _fmt(b.max_recovery_time_s),
                    _fmt(b.max_peak_deviation_m),
                    _fmt(b.max_peak_torque_fraction),
                    _fmt(b.max_saturation_fraction),
                    _fmt(b.max_peak_velocity),
                    "yes" if level.safe else "no",
                    "yes" if level.nontrivial else "no",
                )
                lines.append("| " + " | ".join(cells) + " |")
    s = report.selection
    pulse = (
        f"{s.force_magnitude_n:g} N for {s.force_duration_s:g} s from t = {s.force_start_s:g} s, "
        f"directions {list(s.force_directions_deg)} deg"
    )
    lines += [
        "",
        "## Selection",
        "",
        f"- small posture perturbation: {s.posture_small_rad:g} rad (smallest safe nontrivial level)",
        f"- large held-out posture perturbation: {s.posture_large_rad:g} rad (largest safe level)",
        f"- endpoint force pulse: {pulse} (largest safe nontrivial level)",
        "",
        "These values are locked in `configs/evaluations/task_1a_confirmatory.toml` and may not be used for tuning.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Calibrate confirmatory perturbation levels with the frozen baselines."
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="processed dataset record (TOML)")
    parser.add_argument("--report", type=Path, required=True, help="JSON report to write (must not exist)")
    parser.add_argument(
        "--markdown", type=Path, default=None, help="optional Markdown summary to write (must not exist)"
    )
    parser.add_argument(
        "--exploratory", action="store_true", help="allow a dirty worktree (result is not confirmatory)"
    )
    args = parser.parse_args(argv)
    for target in (args.report, args.markdown):
        if target is not None and target.exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    protocol_file = cast("Path", args.protocol).resolve()
    protocol = load_protocol(protocol_file)
    store = open_storage()
    dataset = load_record(args.dataset, ProcessedDatasetRecord)
    samples = load_samples(verify_payload(store, dataset.artifact))
    report = run_pilot(
        protocol,
        protocol_file,
        dataset,
        samples,
        exploratory=bool(args.exploratory),
        now=datetime.now(tz=UTC),
        command=command_line("arm_rc_ctrl.experiments.perturbation_pilot", sys.argv[1:] if argv is None else argv),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_mapping(report), indent=2, sort_keys=True, allow_nan=False)  # inf/NaN cannot be recorded
    args.report.write_text(text + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    summary = {
        "cases": len(report.cases),
        "safe_levels": [f"{lv.kind}:{lv.magnitude:g}" for lv in report.levels if lv.safe],
        "selection": to_mapping(report.selection),
        "report": args.report.name,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
