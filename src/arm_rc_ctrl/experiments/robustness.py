# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Paired robustness suite across methods (``docs/PLAN.md`` section 9.2; M3-009).

Every arm (an RC generator or the direct replay, each with a frozen tracker)
runs exactly the same scenarios — identical IDs, initial postures, and force
pulses from :mod:`arm_rc_ctrl.experiments.perturbations` — as persisted,
provenance-complete run records. Failures stay in the aggregation: a run that
terminates early or misses a success criterion counts as a failure of its arm
and class, its metrics are reported where they exist, and paired effects are
taken only over scenarios where both runs of a pair completed, with the number
of failed pairs reported next to them. The suite report recomputes its
aggregates and effects from the stored runs on load and rejects stored values
that disagree.

Command line::

    python -m arm_rc_ctrl.experiments.robustness (--confirmatory LOCK.toml | --development LEVELS.toml)
        --dataset RECORD.toml --recipe RECIPE.toml --evaluation configs/evaluations/task_1a_nominal_v3.toml
        --label development --report REPORT.json [--markdown REPORT.md] [--trackers pd_v2 computed_torque]
        [--classes nominal posture_small ...] [--records-root ROOT] [--no-pointer] [--no-mlflow] [--exploratory]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.data.records import ProcessedDatasetRecord, load_record, verify_payload
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.experiments.baselines import baseline_method, load_frozen_baseline
from arm_rc_ctrl.experiments.closed_loop import EstimatorSpec, load_nominal_config, run_nominal
from arm_rc_ctrl.experiments.confirmatory import ConfirmatoryProtocol, load_confirmatory
from arm_rc_ctrl.experiments.paired import compare_reports
from arm_rc_ctrl.experiments.perturbations import (
    CLASS_ORDER,
    DevelopmentRobustness,
    PerturbationClass,
    RobustnessScenario,
    load_development_robustness,
    robustness_scenarios,
)
from arm_rc_ctrl.experiments.replay import run_replay
from arm_rc_ctrl.experiments.run_record import record_run_pointer
from arm_rc_ctrl.experiments.tracking import MlflowTracker
from arm_rc_ctrl.metrics.report import RunReport
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    ProvenanceRecord,
    canonical_json,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
)
from arm_rc_ctrl.rc.esn import ensure_single_thread
from arm_rc_ctrl.rc.recipe import load_recipe
from arm_rc_ctrl.rc.runtime import load_training_samples
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario
from arm_rc_ctrl.storage import open_storage

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from arm_rc_ctrl.controllers.estimator import EstimatorConfig
    from arm_rc_ctrl.controllers.tracking import TrackerConfig
    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.rc.recipe import ModelRecipe
    from arm_rc_ctrl.scenario import ScenarioConfig
    from arm_rc_ctrl.storage import StorageRoot

__all__ = [
    "SUITE_SCHEMA_VERSION",
    "Arm",
    "ArmRun",
    "ClassAggregate",
    "PairedEffect",
    "RobustnessSuite",
    "aggregate_runs",
    "default_arms",
    "load_suite",
    "main",
    "paired_effects",
    "run_robustness",
    "suite_to_json",
    "suite_to_markdown",
]

SUITE_SCHEMA_VERSION: Final = 1
type Generator = Literal["rc", "replay"]
type SuiteLabel = Literal["development", "confirmatory", "confirmatory-rerun"]
_EFFECT_METRICS: Final = ("joint_rmse", "dwell_endpoint_rms", "effort_torque_rms", "effort_saturation_fraction")


@dataclass(frozen=True)
class Arm:
    """One method: a target source and a frozen tracker (registry name)."""

    name: str
    generator: Generator
    tracker: str

    def __post_init__(self) -> None:
        """The name is non-empty."""
        if not self.name.strip() or not self.tracker.strip():
            msg = "arm name and tracker must not be empty"
            raise ValueError(msg)


def default_arms(trackers: Sequence[str]) -> tuple[Arm, ...]:
    """RC and replay arms for every tracker, RC first."""
    arms: list[Arm] = []
    for tracker in trackers:
        arms.append(Arm(f"rc+{tracker}", "rc", tracker))
        arms.append(Arm(f"replay+{tracker}", "replay", tracker))
    return tuple(arms)


@dataclass(frozen=True)
class ArmRun:
    """One arm on one scenario."""

    arm: str
    scenario_id: str
    kind: PerturbationClass
    run_id: str
    report: RunReport
    pointer: str | None = None
    """Repository-relative pointer record, when tracked."""
    mlflow_run_id: str | None = None
    boundary_jump: float | None = None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


@dataclass(frozen=True)
class ClassAggregate:
    """Outcomes of one arm over one perturbation class; failures stay counted."""

    arm: str
    kind: PerturbationClass
    n: int
    completed: int
    successes: int
    failures: dict[str, int]
    """Failed runs by termination kind (``completed`` = finished but missed a success criterion)."""
    joint_rmse_median: float | None
    """Median movement-window joint RMSE over successful runs (``None`` without any)."""
    joint_rmse_max: float | None
    saturation_max: float | None
    dwell_in_tolerance_median: float | None


def aggregate_runs(runs: Sequence[ArmRun], arms: Sequence[str]) -> tuple[ClassAggregate, ...]:
    """Per arm and class, in arm then class order."""
    aggregates: list[ClassAggregate] = []
    for arm in arms:
        for kind in CLASS_ORDER:
            group = [r for r in runs if r.arm == arm and r.kind == kind]
            if not group:
                continue
            successes = [r for r in group if r.report.success]
            failures: dict[str, int] = {}
            for run in group:
                if not run.report.success:
                    failures[run.report.termination_kind] = failures.get(run.report.termination_kind, 0) + 1
            rmse = [r.report.joint_rmse.aggregate for r in successes if r.report.joint_rmse is not None]
            saturation = [r.report.effort.saturation_fraction for r in group if r.report.effort is not None]
            dwell = [r.report.dwell.in_tolerance_fraction for r in successes if r.report.dwell is not None]
            aggregates.append(
                ClassAggregate(
                    arm=arm,
                    kind=kind,
                    n=len(group),
                    completed=sum(1 for r in group if r.report.termination_kind == "completed"),
                    successes=len(successes),
                    failures=dict(sorted(failures.items())),
                    joint_rmse_median=_median(rmse),
                    joint_rmse_max=max(rmse) if rmse else None,
                    saturation_max=max(saturation) if saturation else None,
                    dwell_in_tolerance_median=_median(dwell),
                )
            )
    return tuple(aggregates)


@dataclass(frozen=True)
class PairedEffect:
    """RC minus replay under one tracker and class, over scenarios where both runs succeeded."""

    tracker: str
    kind: PerturbationClass
    metric: str
    unit: str
    n_pairs: int
    n_both_success: int
    rc_failures: int
    replay_failures: int
    median_difference: float | None
    median_rc: float | None
    median_replay: float | None


def paired_effects(runs: Sequence[ArmRun], arms: Sequence[Arm]) -> tuple[PairedEffect, ...]:
    """Pair every RC arm with the replay arm of the same tracker, scenario by scenario."""
    effects: list[PairedEffect] = []
    by_key = {(r.arm, r.scenario_id): r for r in runs}
    for rc_arm in (a for a in arms if a.generator == "rc"):
        replay_arm = next((a for a in arms if a.generator == "replay" and a.tracker == rc_arm.tracker), None)
        if replay_arm is None:
            continue
        for kind in CLASS_ORDER:
            ids = [r.scenario_id for r in runs if r.arm == rc_arm.name and r.kind == kind]
            pairs = [
                (by_key[(rc_arm.name, i)], by_key[(replay_arm.name, i)]) for i in ids if (replay_arm.name, i) in by_key
            ]
            if not pairs:
                continue
            both = [(rc, rp) for rc, rp in pairs if rc.report.success and rp.report.success]
            for metric in _EFFECT_METRICS:
                rc_values: list[float] = []
                replay_values: list[float] = []
                for rc, rp in both:
                    comparison = next(c for c in compare_reports(rc.report, rp.report) if c.name == metric)
                    if comparison.rc is not None and comparison.replay is not None:
                        rc_values.append(comparison.rc)
                        replay_values.append(comparison.replay)
                unit = next(c.unit for c in compare_reports(pairs[0][0].report, pairs[0][1].report) if c.name == metric)
                diffs = [a - b for a, b in zip(rc_values, replay_values, strict=True)]
                effects.append(
                    PairedEffect(
                        tracker=rc_arm.tracker,
                        kind=kind,
                        metric=metric,
                        unit=unit,
                        n_pairs=len(pairs),
                        n_both_success=len(both),
                        rc_failures=sum(1 for rc, _ in pairs if not rc.report.success),
                        replay_failures=sum(1 for _, rp in pairs if not rp.report.success),
                        median_difference=_median(diffs),
                        median_rc=_median(rc_values),
                        median_replay=_median(replay_values),
                    )
                )
    return tuple(effects)


@dataclass(frozen=True)
class RobustnessSuite:
    """The complete suite: arms, scenarios, every run, and the derived aggregates and paired effects."""

    label: SuiteLabel
    protocol: str
    protocol_file: str
    scenario: str
    reference_artifact: str
    recipe: str
    recipe_file: str
    estimator: EstimatorSpec
    arms: tuple[Arm, ...]
    scenarios: tuple[RobustnessScenario, ...]
    runs: tuple[ArmRun, ...]
    provenance: ProvenanceRecord
    aggregates: tuple[ClassAggregate, ...] = ()
    effects: tuple[PairedEffect, ...] = ()
    schema_version: int = field(default=SUITE_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Every arm ran every scenario exactly once; derived tables are recomputed and must match stored ones."""
        ids = [s.scenario_id for s in self.scenarios]
        if len(set(ids)) != len(ids):
            msg = "scenario IDs must be unique"
            raise ValueError(msg)
        expected = {(a.name, i) for a in self.arms for i in ids}
        actual = [(r.arm, r.scenario_id) for r in self.runs]
        if len(set(actual)) != len(actual) or set(actual) != expected:
            msg = "every arm must run every scenario exactly once"
            raise ValueError(msg)
        kinds = {s.scenario_id: s.kind for s in self.scenarios}
        methods = {a.name: f"{a.generator}+{baseline_method(a.tracker)}" for a in self.arms}
        for run in self.runs:
            if run.report.run_id != run.run_id:
                msg = f"run {run.run_id} carries the report of {run.report.run_id}"
                raise ValueError(msg)
            if run.kind != kinds[run.scenario_id]:
                kind = kinds[run.scenario_id]
                msg = f"run {run.run_id} is filed under class {run.kind!r} but scenario {run.scenario_id} is {kind!r}"
                raise ValueError(msg)
            if run.report.method != methods[run.arm]:
                msg = (
                    f"run {run.run_id} reports method {run.report.method!r}, arm {run.arm} expects {methods[run.arm]!r}"
                )
                raise ValueError(msg)
            if run.report.scenario != self.scenario or run.report.reference_artifact != self.reference_artifact:
                msg = f"run {run.run_id} was evaluated on another scenario or reference than the suite"
                raise ValueError(msg)
        aggregates = aggregate_runs(self.runs, [a.name for a in self.arms])
        effects = paired_effects(self.runs, self.arms)
        if self.aggregates and self.aggregates != aggregates:
            msg = "stored aggregates do not match the runs they were derived from"
            raise ValueError(msg)
        if self.effects and self.effects != effects:
            msg = "stored paired effects do not match the runs they were derived from"
            raise ValueError(msg)
        object.__setattr__(self, "aggregates", aggregates)
        object.__setattr__(self, "effects", effects)


def suite_to_json(suite: RobustnessSuite) -> str:
    """Canonical JSON of the suite."""
    return canonical_json(to_mapping(suite))


def load_suite(path: Path) -> RobustnessSuite:
    """Strictly rebuild a suite from JSON (derived tables are recomputed and checked)."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), RobustnessSuite)


def _relative(path: Path) -> str:
    root = repository_root()
    resolved = path.resolve()
    return resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else path.name


def run_robustness(
    protocol: ConfirmatoryProtocol | DevelopmentRobustness,
    protocol_file: Path,
    *,
    label: SuiteLabel,
    scenario: ScenarioConfig,
    scenario_file: Path,
    dataset: ProcessedDatasetRecord,
    reference: SampleSet,
    recipe: ModelRecipe,
    recipe_file: Path,
    estimator: EstimatorSpec,
    trackers: Mapping[str, TrackerConfig],
    training_samples: Mapping[str, SampleSet],
    store: StorageRoot,
    exploratory: bool,
    arms: Sequence[Arm] | None = None,
    classes: Sequence[PerturbationClass] = CLASS_ORDER,
    now: datetime | None = None,
    command: str = "python -m arm_rc_ctrl.experiments.robustness",
    on_run: Callable[[ArmRun, object], ArmRun] | None = None,
) -> RobustnessSuite:
    """Run every arm on every scenario of ``protocol`` (persisting each run) and assemble the suite.

    ``on_run`` (pointer recording, tracking) is applied to every run after the
    last simulation, so nothing is written into the repository while runs that
    require a clean worktree are still to come.
    """
    confirmatory = isinstance(protocol, ConfirmatoryProtocol)
    if label == "development" and confirmatory:
        msg = "the locked confirmatory protocol cannot be run under the development label"
        raise ValueError(msg)
    if label != "development" and (exploratory or not confirmatory):
        msg = "a confirmatory suite needs the locked confirmatory protocol and a clean worktree"
        raise ValueError(msg)
    arms = default_arms(list(trackers)) if arms is None else tuple(arms)
    missing = sorted({a.tracker for a in arms} - set(trackers))
    if missing:
        msg = f"arms reference trackers without a configuration: {missing}"
        raise ValueError(msg)
    lower = [link.q_min for link in scenario.robot.links]
    upper = [link.q_max for link in scenario.robot.links]
    scenarios = robustness_scenarios(
        protocol, nominal=scenario.task.initial_q, lower=lower, upper=upper, classes=classes
    )
    payload = dataset.artifact.payload
    resolved = {
        "label": label,
        "protocol": to_mapping(protocol),
        "protocol_file": _relative(protocol_file),
        "recipe": recipe.name,
        "recipe_file": _relative(recipe_file),
        "estimator": to_mapping(estimator),
        "arms": [to_mapping(a) for a in arms],
        "trackers": {name: to_mapping(cfg) for name, cfg in trackers.items()},
        "classes": list(classes),
        "scenario_ids": [s.scenario_id for s in scenarios],
        "command": command,
    }
    provenance = collect_provenance(
        resolved,
        seeds={f"protocol.{i}": seed for i, seed in enumerate(protocol.seeds)},
        artifacts=[ArtifactReference(payload.uri, payload.sha256, payload.size)],
        exploratory=exploratory,
        now=now,
    )
    require_clean_for_confirmatory(provenance)
    estimator_config: EstimatorConfig = estimator.config(scenario.timing.dt)
    completed: list[tuple[ArmRun, object]] = []
    for robustness in scenarios:
        initial_q = robustness.initial_q(scenario.task.initial_q)
        for arm in arms:
            tracker = trackers[arm.tracker]
            if arm.generator == "rc":
                rc = run_nominal(
                    scenario,
                    scenario_file,
                    dataset,
                    reference,
                    recipe,
                    tracker,
                    store=store,
                    estimator=estimator_config,
                    training_samples=training_samples,
                    exploratory=exploratory,
                    now=now,
                    command=command,
                    initial_q=initial_q,
                    force=robustness.pulse,
                )
                run = ArmRun(
                    arm.name,
                    robustness.scenario_id,
                    robustness.kind,
                    rc.pointer.artifact.artifact_id,
                    rc.report,
                    boundary_jump=rc.boundary_jump,
                )
                result: object = rc
            else:
                rp = run_replay(
                    scenario,
                    scenario_file,
                    dataset,
                    reference,
                    tracker,
                    store=store,
                    exploratory=exploratory,
                    now=now,
                    command=command,
                    initial_q=initial_q,
                    force=robustness.pulse,
                )
                run = ArmRun(
                    arm.name, robustness.scenario_id, robustness.kind, rp.pointer.artifact.artifact_id, rp.report
                )
                result = rp
            completed.append((run, result))
    # Pointers and tracking are recorded only now: writing into the repository mid-suite would dirty the
    # worktree that the remaining (non-exploratory) runs must find clean.
    runs = [run if on_run is None else on_run(run, result) for run, result in completed]
    return RobustnessSuite(
        label=label,
        protocol=protocol.name,
        protocol_file=_relative(protocol_file),
        scenario=scenario.name,
        reference_artifact=dataset.artifact.artifact_id,
        recipe=recipe.name,
        recipe_file=_relative(recipe_file),
        estimator=estimator,
        arms=arms,
        scenarios=scenarios,
        runs=tuple(runs),
        provenance=provenance,
    )


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}g}"


def _row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _aggregate_rows(suite: RobustnessSuite) -> list[str]:
    header = [
        "arm", "class", "n", "completed", "successes", "failures",
        "joint RMSE median (rad)", "joint RMSE max", "saturation max", "dwell in-tolerance median",
    ]  # fmt: skip
    rows = [_row(header), _row(["---"] * len(header))]
    for a in suite.aggregates:
        failures = ", ".join(f"{k} x{v}" for k, v in a.failures.items()) or "-"
        cells = [
            a.arm, a.kind, a.n, a.completed, a.successes, failures,
            _fmt(a.joint_rmse_median), _fmt(a.joint_rmse_max),
            _fmt(a.saturation_max), _fmt(a.dwell_in_tolerance_median),
        ]  # fmt: skip
        rows.append(_row(cells))
    return rows


def _effect_rows(suite: RobustnessSuite) -> list[str]:
    header = [
        "tracker", "class", "metric", "pairs", "both succeeded", "RC failures", "replay failures",
        "median RC", "median replay", "median difference",
    ]  # fmt: skip
    rows = [_row(header), _row(["---"] * len(header))]
    for e in suite.effects:
        unit = f" {e.unit}" if e.unit else ""
        cells = [
            e.tracker, e.kind, e.metric, e.n_pairs, e.n_both_success, e.rc_failures, e.replay_failures,
            f"{_fmt(e.median_rc)}{unit}", f"{_fmt(e.median_replay)}{unit}", f"{_fmt(e.median_difference)}{unit}",
        ]  # fmt: skip
        rows.append(_row(cells))
    return rows


def suite_to_markdown(suite: RobustnessSuite) -> str:
    """Per-class outcomes of every arm, paired RC-minus-replay effects, and the failed runs."""
    dirty = " (dirty)" if suite.provenance.project_dirty else ""
    cutoffs = f"{_fmt(suite.estimator.velocity_cutoff_hz)}/{_fmt(suite.estimator.acceleration_cutoff_hz)} Hz"
    lines = [
        f"# Robustness suite `{suite.protocol}` ({suite.label})",
        "",
        f"- Scenario `{suite.scenario}`, reference `{suite.reference_artifact}`, recipe `{suite.recipe}`",
        f"  (`{suite.recipe_file}`), estimator cutoffs {cutoffs}.",
        f"- {len(suite.scenarios)} scenarios x {len(suite.arms)} arms = {len(suite.runs)} runs;",
        f"  protocol `{suite.protocol_file}`; commit `{suite.provenance.project_commit[:12]}`{dirty}.",
        "",
        "## Outcomes by arm and class",
        "",
        *_aggregate_rows(suite),
        "",
        "## Paired effects (RC minus replay, same tracker and scenario)",
        "",
        *_effect_rows(suite),
    ]
    failed = [r for r in suite.runs if not r.report.success]
    lines.extend(["", f"## Failed runs ({len(failed)})", ""])
    if failed:
        lines.extend([_row(["arm", "scenario", "termination", "failed criteria", "run"]), _row(["---"] * 5)])
        lines.extend(
            _row(
                [
                    r.arm,
                    r.scenario_id,
                    r.report.termination_kind,
                    ", ".join(r.report.failed_criteria) or "-",
                    f"`{r.run_id}`",
                ]
            )
            for r in failed
        )
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run the paired robustness suite of a task across methods.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--confirmatory", type=Path, help="locked confirmatory protocol (configs/evaluations/*.toml)")
    group.add_argument("--development", type=Path, help="development robustness levels (configs/evaluations/*.toml)")
    parser.add_argument("--dataset", type=Path, required=True, help="processed dataset record (the reference)")
    parser.add_argument("--recipe", type=Path, required=True, help="model recipe (TOML)")
    parser.add_argument("--evaluation", type=Path, required=True, help="evaluation config providing the estimator")
    parser.add_argument("--trackers", nargs="+", default=["pd_v2", "computed_torque"], help="frozen baseline names")
    parser.add_argument("--classes", nargs="+", default=list(CLASS_ORDER), choices=list(CLASS_ORDER))
    parser.add_argument("--label", choices=["development", "confirmatory", "confirmatory-rerun"], required=True)
    parser.add_argument("--report", type=Path, required=True, help="suite JSON to write (must not exist)")
    parser.add_argument("--markdown", type=Path, default=None, help="optional Markdown to write (must not exist)")
    parser.add_argument("--records-root", type=Path, default=None)
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree (development only)")
    parser.add_argument("--no-pointer", action="store_true", help="do not track the runs under data/records/runs")
    parser.add_argument("--no-mlflow", action="store_true", help="skip the mandatory MLflow logging (scratch only)")
    args = parser.parse_args(argv)
    ensure_single_thread()  # before rclib is imported and provenance is collected
    for target in (args.report, args.markdown):
        if target is not None and Path(target).exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    if args.confirmatory is not None:
        protocol: ConfirmatoryProtocol | DevelopmentRobustness = load_confirmatory(Path(args.confirmatory))
        protocol_file = Path(args.confirmatory)
    else:
        protocol = load_development_robustness(Path(args.development))
        protocol_file = Path(args.development)
    store = open_storage()
    scenario = load_scenario(protocol.scenario)
    dataset = load_record(Path(args.dataset), ProcessedDatasetRecord)
    reference = load_samples(verify_payload(store, dataset.artifact))
    recipe = load_recipe(Path(args.recipe))
    records_root = repository_root() if args.records_root is None else Path(args.records_root)
    training = load_training_samples(recipe, store, records_root=None if args.records_root is None else records_root)
    evaluation = load_nominal_config(Path(args.evaluation))
    trackers = {name: load_frozen_baseline(name) for name in cast("list[str]", args.trackers)}
    tracker_logger = None if args.no_mlflow else MlflowTracker(store)
    label = cast("SuiteLabel", args.label)
    experiment = f"{protocol.name}-{label}"

    def on_run(run: ArmRun, result: object) -> ArmRun:
        pointer_obj = getattr(result, "pointer")  # noqa: B009 - ClosedLoopResult | ReplayResult
        loaded = getattr(result, "run")  # noqa: B009
        report = getattr(result, "report")  # noqa: B009
        pointer_file = None if args.no_pointer else record_run_pointer(records_root, pointer_obj)
        mlflow_id = None
        if tracker_logger is not None:
            is_rc = run.arm.startswith("rc")
            mlflow_id = tracker_logger.log_run(
                loaded, report, experiment=experiment, recipe=recipe if is_rc else None, pointer_file=pointer_file
            ).mlflow_run_id
        pointer = None if pointer_file is None else pointer_file.relative_to(records_root).as_posix()
        return ArmRun(run.arm, run.scenario_id, run.kind, run.run_id, run.report, pointer, mlflow_id, run.boundary_jump)

    suite = run_robustness(
        protocol,
        protocol_file,
        label=label,
        scenario=scenario,
        scenario_file=protocol.scenario,
        dataset=dataset,
        reference=reference,
        recipe=recipe,
        recipe_file=Path(args.recipe),
        estimator=evaluation.estimator,
        trackers=trackers,
        training_samples=training,
        store=store,
        exploratory=bool(args.exploratory),
        classes=cast("list[PerturbationClass]", args.classes),
        now=datetime.now(tz=UTC),
        command=command_line("arm_rc_ctrl.experiments.robustness", sys.argv[1:] if argv is None else argv),
        on_run=on_run,
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(suite_to_json(suite) + "\n", encoding="utf-8")
    if args.markdown is not None:
        Path(args.markdown).write_text(suite_to_markdown(suite), encoding="utf-8")
    summary = {
        "label": suite.label,
        "runs": len(suite.runs),
        "failed": sum(1 for r in suite.runs if not r.report.success),
        "arms": {a.arm + "/" + a.kind: f"{a.successes}/{a.n}" for a in suite.aggregates},
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
