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

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import numpy as np
import tomli_w
from numpy.typing import NDArray

from arm_rc_ctrl.config import load_config, to_mapping
from arm_rc_ctrl.controllers.reference import DemonstrationReference
from arm_rc_ctrl.controllers.tracking import TrackerConfig, TrackerType
from arm_rc_ctrl.data.phases import intervals_from_phases
from arm_rc_ctrl.data.records import ProcessedDatasetRecord, load_record, verify_payload
from arm_rc_ctrl.data.samples import SampleSet, load_samples
from arm_rc_ctrl.experiments.disturbances import ForcePulse
from arm_rc_ctrl.experiments.replay import bind_dataset, dwell_outcome, simulate_tracking
from arm_rc_ctrl.metrics.joint import JointAnglePolicy, joint_rmse
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    ProvenanceRecord,
    collect_provenance,
    require_clean_for_confirmatory,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import ScenarioConfig, load_scenario
from arm_rc_ctrl.storage import open_storage
from arm_rc_ctrl.validation import require_finite

__all__ = [
    "DevelopmentPulse",
    "DevelopmentScenarios",
    "Feasibility",
    "GainRange",
    "Objective",
    "SearchSpaces",
    "StudyReport",
    "StudyResult",
    "Trial",
    "TrialScenario",
    "TuningProtocol",
    "evaluate_gains",
    "freeze_config_toml",
    "load_protocol",
    "main",
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

    kind: Literal["median_move_joint_rmse", "nominal_move_joint_rmse"]
    """Median over development scenarios, or the nominal (unperturbed) scenario's RMSE among robust-feasible trials."""
    infeasible_penalty: float

    def __post_init__(self) -> None:
        """Require a positive finite penalty."""
        if not (self.infeasible_penalty > 0 and math.isfinite(self.infeasible_penalty)):
            msg = f"infeasible_penalty must be positive and finite, got {self.infeasible_penalty!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class DevelopmentPulse:
    """A development-only endpoint force pulse applied at the nominal posture."""

    start_s: float
    duration_s: float
    magnitude_n: float
    direction_deg: float

    def __post_init__(self) -> None:
        """Validate through the pulse it describes."""
        self.pulse()

    def pulse(self) -> ForcePulse:
        """The disturbance to apply."""
        return ForcePulse.from_polar(self.start_s, self.duration_s, self.magnitude_n, self.direction_deg)


@dataclass(frozen=True)
class DevelopmentScenarios:
    """Development-only scenario perturbations: posture offsets and, optionally, force pulses."""

    initial_posture_offsets: tuple[tuple[float, ...], ...]
    force_pulses: tuple[DevelopmentPulse, ...] = ()

    def __post_init__(self) -> None:
        """Require at least one offset with finite entries."""
        if not self.initial_posture_offsets:
            msg = "development.initial_posture_offsets must not be empty"
            raise ValueError(msg)
        for i, offset in enumerate(self.initial_posture_offsets):
            require_finite(offset, f"development.initial_posture_offsets[{i}]")

    @property
    def nominal_first(self) -> bool:
        """Whether the first offset is the unperturbed posture."""
        return not any(self.initial_posture_offsets[0])


@dataclass(frozen=True)
class Feasibility:
    """Robustness constraints every development scenario of a feasible trial must satisfy."""

    max_saturation_fraction: float = 1.0
    """Largest fraction of control samples with a saturated actuator (1.0 = no headroom requirement)."""

    def __post_init__(self) -> None:
        """The bound lies in [0, 1]."""
        if not (0 <= self.max_saturation_fraction <= 1):
            msg = f"feasibility.max_saturation_fraction must lie in [0, 1], got {self.max_saturation_fraction!r}"
            raise ValueError(msg)


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
    feasibility: Feasibility = field(default_factory=Feasibility)

    def __post_init__(self) -> None:
        """Validate name, budget, seed, and the objective's requirements."""
        if not self.name.strip():
            msg = "name must not be empty"
            raise ValueError(msg)
        if self.objective.kind == "nominal_move_joint_rmse" and not self.development.nominal_first:
            msg = "the nominal_move_joint_rmse objective needs the unperturbed posture as the first development offset"
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
    kind: Literal["posture", "force"] = "posture"
    saturation_fraction: float | None = None
    """Fraction of logged control samples with a saturated actuator (``None`` when nothing was logged)."""
    pulse: DevelopmentPulse | None = None


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
    *,
    interpolation: Literal["linear", "cubic"] = "linear",
) -> tuple[float, bool, tuple[TrialScenario, ...]]:
    """Simulate every development scenario and return ``(objective, feasible, components)``."""
    demo = DemonstrationReference.from_samples(reference, interpolation)
    intervals = intervals_from_phases(reference.t, reference.phase)
    move = (reference.t >= intervals.move[0]) & (reference.t < intervals.move[1])
    policy = JointAnglePolicy.limited(scenario.dof)
    duration = float(reference.t[-1])
    components: list[TrialScenario] = []
    cases: list[tuple[tuple[float, ...], DevelopmentPulse | None]] = []
    for index, offset in enumerate(protocol.development.initial_posture_offsets):
        if len(offset) != scenario.dof:
            msg = f"development offset {index} has {len(offset)} entries, expected {scenario.dof}"
            raise ValueError(msg)
        cases.append((tuple(float(a + b) for a, b in zip(scenario.task.initial_q, offset, strict=True)), None))
    cases.extend((tuple(scenario.task.initial_q), pulse) for pulse in protocol.development.force_pulses)
    bound = protocol.feasibility.max_saturation_fraction
    for index, (initial_q, pulse) in enumerate(cases):
        arrays, termination = simulate_tracking(
            scenario,
            demo,
            gains,
            duration_s=duration,
            initial_q=initial_q,
            force=None if pulse is None else pulse.pulse(),
        )
        q = cast("NDArray[np.float64]", arrays.arrays["q"])
        n = min(q.shape[0], reference.n_samples)
        rmse: float | None = None
        if termination.is_completed and move[:n].any():
            rmse = joint_rmse(q[:n][move[:n]], reference.q[:n][move[:n]], policy).aggregate
        criteria = dwell_outcome(scenario, reference, arrays, termination)
        saturation = float(np.mean(cast("NDArray[np.int64]", arrays.arrays["saturation"]))) if n else None
        feasible = rmse is not None and all(criteria.values()) and saturation is not None and saturation <= bound
        components.append(
            TrialScenario(
                index,
                initial_q,
                termination.kind,
                rmse,
                criteria,
                feasible,
                kind="posture" if pulse is None else "force",
                saturation_fraction=saturation,
                pulse=pulse,
            )
        )
    feasible_all = all(c.feasible for c in components)
    if not feasible_all:
        objective = protocol.objective.infeasible_penalty
    elif protocol.objective.kind == "nominal_move_joint_rmse":
        objective = cast("float", components[0].move_joint_rmse)
    else:
        objective = float(np.median([cast("float", c.move_joint_rmse) for c in components]))
    return objective, feasible_all, tuple(components)


def run_study(
    protocol: TuningProtocol,
    dataset: ProcessedDatasetRecord,
    reference: SampleSet,
    tracker_type: TrackerType,
    *,
    scenario_file: Path | None = None,
    scenario: ScenarioConfig | None = None,
) -> StudyResult:
    """Run the full seeded search for one tracker type (same seed and budget for every tracker).

    The dataset must have been derived under the protocol's scenario file (or
    ``scenario_file`` when given) and match its record.
    """
    scenario_file = protocol.scenario if scenario_file is None else scenario_file
    scenario = load_scenario(scenario_file) if scenario is None else scenario
    bind_dataset(scenario, scenario_file, dataset, reference)
    interpolation = cast('Literal["linear", "cubic"]', dataset.preprocessing.interpolation)
    rng = np.random.default_rng(protocol.sampler_seed)
    trials: list[Trial] = []
    for number in range(protocol.budget):
        gains = sample_gains(protocol, tracker_type, scenario.dof, rng)
        objective, feasible, components = evaluate_gains(
            protocol, scenario, reference, gains, interpolation=interpolation
        )
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


@dataclass(frozen=True)
class StudyReport:
    """Curated, Git-tracked study summary: every trial plus the provenance of the run."""

    result: StudyResult
    dataset: str
    scenario_file: str
    provenance: ProvenanceRecord
    schema_version: int = 1


def freeze_config_toml(gains: TrackerConfig, *, study: str, dataset: str, best_objective: float) -> str:
    """Render the selected gains as a tracker TOML with a header naming the study that chose them."""
    header = (
        f"# Frozen by study {study!r} on dataset {dataset} (objective {best_objective:.6g} rad).\n"
        "# Do not edit: baseline gains are fixed before any ESN tuning (docs/PLAN.md section 6).\n"
    )
    return header + tomli_w.dumps(to_mapping(gains))


def main(argv: Sequence[str] | None = None) -> int:
    """Run one tracker's study on a processed dataset, write the report, and freeze the selected gains."""
    parser = argparse.ArgumentParser(description="Tune and freeze direct-replay tracker gains.")
    parser.add_argument("--protocol", type=Path, required=True, help="study TOML (configs/studies/*.toml)")
    parser.add_argument("--dataset", type=Path, required=True, help="processed dataset record (TOML)")
    parser.add_argument("--tracker", required=True, choices=("pd", "computed_torque"))
    parser.add_argument("--report", type=Path, required=True, help="study report JSON to write")
    parser.add_argument("--freeze", type=Path, required=True, help="frozen tracker TOML to write")
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    if Path(args.report).exists() or Path(args.freeze).exists():
        msg = f"{args.report} or {args.freeze} already exists; studies and frozen configs are immutable"
        raise FileExistsError(msg)
    protocol = load_protocol(Path(args.protocol))
    store = open_storage()
    dataset = load_record(Path(args.dataset), ProcessedDatasetRecord)
    samples = load_samples(verify_payload(store, dataset.artifact))
    scenario_file = protocol.scenario.relative_to(repository_root()).as_posix()
    protocol_mapping = to_mapping(protocol)
    protocol_mapping["scenario"] = scenario_file  # records never carry machine paths
    resolved = {"protocol": protocol_mapping, "dataset": dataset.artifact.artifact_id, "tracker": args.tracker}
    payload = dataset.artifact.payload
    provenance = collect_provenance(
        resolved,
        seeds={"sampler": protocol.sampler_seed},
        artifacts=[ArtifactReference(payload.uri, payload.sha256, payload.size)],
        exploratory=args.exploratory,
        now=datetime.now(UTC),
    )
    require_clean_for_confirmatory(provenance)
    result = run_study(protocol, dataset, samples, cast("TrackerType", args.tracker))
    report = StudyReport(
        result=result,
        dataset=dataset.artifact.artifact_id,
        scenario_file=scenario_file,
        provenance=provenance,
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(to_mapping(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result.feasible_trials == 0:
        msg = f"no trial of {protocol.name!r} satisfied every development scenario; report written, nothing frozen"
        raise RuntimeError(msg)
    Path(args.freeze).write_text(
        freeze_config_toml(
            result.best.gains, study=protocol.name, dataset=report.dataset, best_objective=result.best.objective
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "tracker": args.tracker,
                "budget": result.budget,
                "feasible_trials": result.feasible_trials,
                "best_trial": result.best.number,
                "best_objective": result.best.objective,
                "kp": list(result.best.gains.kp),
                "kd": list(result.best.gains.kd),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
