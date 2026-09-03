# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

r"""Nominal RC closed-loop experiment: recipe + frozen tracker in ``skelarm`` with a full run record (M2-012).

Usage::

    python -m arm_rc_ctrl.experiments.closed_loop --config configs/evaluations/task_1a_nominal.toml \\
        --scenario configs/tasks/task_1a.toml --dataset data/records/processed/<id>.toml \\
        --recipe data/records/models/<id>.toml [--report <json>] [--exploratory]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from arm_rc_ctrl.config import load_config, to_mapping
from arm_rc_ctrl.controllers.adapter import GeneratorTrackingController
from arm_rc_ctrl.controllers.estimator import EstimatorConfig
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.records import ProcessedDatasetRecord, load_record, verify_payload
from arm_rc_ctrl.data.samples import SampleSet, load_samples
from arm_rc_ctrl.experiments.replay import bind_dataset, dwell_outcome
from arm_rc_ctrl.experiments.run_record import (
    LoadedRun,
    RunPointerRecord,
    RunSummary,
    load_run,
    record_run_pointer,
    write_run,
)
from arm_rc_ctrl.experiments.simulation import GENERATOR_CHANNELS, simulate
from arm_rc_ctrl.experiments.termination import Outcome
from arm_rc_ctrl.experiments.tracking import MlflowTracker
from arm_rc_ctrl.metrics.joint import JointAnglePolicy
from arm_rc_ctrl.metrics.report import RunReport, build_report, report_to_json
from arm_rc_ctrl.provenance import ArtifactReference, collect_provenance, command_line, require_clean_for_confirmatory
from arm_rc_ctrl.rc.esn import ensure_single_thread
from arm_rc_ctrl.rc.recipe import ModelRecipe, load_recipe
from arm_rc_ctrl.rc.runtime import generator_from_recipe, load_training_samples
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import ScenarioConfig, load_scenario
from arm_rc_ctrl.storage import StorageRoot, open_storage

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from arm_rc_ctrl.experiments.disturbances import ForcePulse

__all__ = ["ClosedLoopResult", "EstimatorSpec", "NominalConfig", "load_nominal_config", "main", "run_nominal"]


@dataclass(frozen=True)
class EstimatorSpec:
    """Estimator settings of an evaluation config (the nominal period comes from the scenario)."""

    velocity_cutoff_hz: float | None = None
    acceleration_cutoff_hz: float | None = None
    max_dt_ratio: float = 3.0

    def config(self, dt: float) -> EstimatorConfig:
        """The estimator configuration at the scenario's control period."""
        return EstimatorConfig(
            nominal_dt_s=dt,
            max_dt_ratio=self.max_dt_ratio,
            velocity_cutoff_hz=self.velocity_cutoff_hz,
            acceleration_cutoff_hz=self.acceleration_cutoff_hz,
        )


@dataclass(frozen=True)
class NominalConfig:
    """Evaluation configuration (``configs/evaluations/*.toml``): tracker gains and estimator settings."""

    name: str
    tracker: Path
    estimator: EstimatorSpec

    def __post_init__(self) -> None:
        """The name identifies the evaluation."""
        if not self.name.strip():
            msg = "name must not be empty"
            raise ValueError(msg)


def load_nominal_config(path: Path) -> NominalConfig:
    """Load and validate an evaluation configuration."""
    return load_config(path, NominalConfig)


@dataclass(frozen=True)
class ClosedLoopResult:
    """Outputs of one closed-loop run."""

    pointer: RunPointerRecord
    summary: RunSummary
    directory: Path
    run: LoadedRun
    report: RunReport
    boundary_jump: float | None


def run_nominal(
    scenario: ScenarioConfig,
    scenario_file: Path,
    dataset: ProcessedDatasetRecord,
    reference: SampleSet,
    recipe: ModelRecipe,
    tracker: TrackerConfig,
    *,
    store: StorageRoot,
    estimator: EstimatorConfig,
    training_samples: Mapping[str, SampleSet],
    exploratory: bool,
    now: datetime | None = None,
    command: str = "python -m arm_rc_ctrl.experiments.closed_loop",
    initial_q: tuple[float, ...] | None = None,
    force: ForcePulse | None = None,
    license_label: str = "LicenseRef-Private",
    access: str = "private",
) -> ClosedLoopResult:
    """Rebuild the generator from the recipe, run it closed loop with the tracker, persist and evaluate the run.

    The demonstration ``reference`` (bound to the scenario) defines the run
    duration, the priming boundary (its prime interval), and the metric
    windows; the generator never sees it during the run.
    """
    bind_dataset(scenario, scenario_file, dataset, reference)
    if tracker.dof != scenario.dof or recipe.dof != scenario.dof:
        msg = f"dof mismatch: scenario {scenario.dof}, tracker {tracker.dof}, recipe {recipe.dof}"
        raise ValueError(msg)
    lower = np.array([link.q_min for link in scenario.robot.links], dtype=np.float64)
    upper = np.array([link.q_max for link in scenario.robot.links], dtype=np.float64)
    generator = generator_from_recipe(recipe, training_samples, estimator=estimator, position_bounds=(lower, upper))
    hold_until = scenario.timing.intervals.prime[1]
    controller = GeneratorTrackingController(generator, tracker, scenario.limits.torque, hold_until_s=hold_until)
    reference_artifact = dataset.artifact.artifact_id
    payload = dataset.artifact.payload
    resolved = {
        "scenario": to_mapping(scenario),
        "tracker": to_mapping(tracker),
        "recipe": to_mapping(recipe),
        "estimator": to_mapping(estimator),
        "hold_until_s": hold_until,
        "reference_artifact": reference_artifact,
        "initial_q": list(scenario.task.initial_q if initial_q is None else initial_q),
        "force": None if force is None else to_mapping(force),
        "duration_s": float(reference.t[-1]),
    }
    provenance = collect_provenance(
        resolved,
        seeds={"reservoir": recipe.esn.reservoir.seed},
        artifacts=[ArtifactReference(payload.uri, payload.sha256, payload.size)],
        exploratory=exploratory,
        now=now,
    )
    require_clean_for_confirmatory(provenance)
    duration = float(reference.t[-1])
    arrays, termination = simulate(
        scenario, controller, duration_s=duration, initial_q=initial_q, force=force, channels=GENERATOR_CHANNELS
    )
    outcome = Outcome(termination, dwell_outcome(scenario, reference, arrays, termination))
    jump = controller.boundary_jump
    pointer, summary, directory = write_run(
        store,
        arrays,
        kind="simulation",
        method=f"rc+{tracker.method}",
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
        activation_s=hold_until,
        notes=(
            f"RC target generator from recipe {recipe.name!r} tracked by {tracker.method}; "
            f"priming until {hold_until} s; boundary jump {'n/a' if jump is None else f'{jump:.3g} rad'}."
        ),
    )
    run = load_run(store, pointer)
    report = build_report(
        run,
        reference,
        reference_artifact,
        tolerance=scenario.task.tolerance,
        torque_limits=scenario.limits.torque,
        policy=JointAnglePolicy.limited(scenario.dof),
    )
    return ClosedLoopResult(pointer, summary, directory, run, report, jump)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run the nominal RC closed loop for task 1-a.")
    parser.add_argument("--config", type=Path, required=True, help="evaluation config (configs/evaluations/*.toml)")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="processed dataset record (the reference)")
    parser.add_argument("--recipe", type=Path, required=True, help="model recipe (TOML)")
    parser.add_argument(
        "--records-root", type=Path, default=None, help="root the recipe's dataset records are relative to"
    )
    parser.add_argument("--report", type=Path, default=None, help="write the metric report JSON here")
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    parser.add_argument(
        "--no-pointer", action="store_true", help="do not track the run under data/records/runs (exploratory scratch)"
    )
    parser.add_argument("--no-mlflow", action="store_true", help="skip the mandatory MLflow logging (scratch only)")
    parser.add_argument("--experiment", default=None, help="MLflow experiment name (default: the scenario name)")
    args = parser.parse_args(argv)
    ensure_single_thread()  # before rclib is imported and provenance is collected
    if args.report is not None and Path(args.report).exists():
        msg = f"refusing to overwrite {args.report}"
        raise FileExistsError(msg)
    config = load_nominal_config(Path(args.config))
    scenario = load_scenario(Path(args.scenario))
    store = open_storage()
    dataset = load_record(Path(args.dataset), ProcessedDatasetRecord)
    reference = load_samples(verify_payload(store, dataset.artifact))
    recipe = load_recipe(Path(args.recipe))
    training = load_training_samples(
        recipe, store, records_root=None if args.records_root is None else Path(args.records_root)
    )
    result = run_nominal(
        scenario,
        Path(args.scenario),
        dataset,
        reference,
        recipe,
        load_config(config.tracker, TrackerConfig),
        store=store,
        estimator=config.estimator.config(scenario.timing.dt),
        training_samples=training,
        exploratory=bool(args.exploratory),
        now=datetime.now(tz=UTC),
        command=command_line("arm_rc_ctrl.experiments.closed_loop", sys.argv[1:] if argv is None else argv),
    )
    records_root = repository_root() if args.records_root is None else Path(args.records_root)
    pointer_file = None if args.no_pointer else record_run_pointer(records_root, result.pointer)
    tracked = None
    if not args.no_mlflow:
        experiment = scenario.name if args.experiment is None else str(args.experiment)
        tracked = MlflowTracker(store).log_run(
            result.run, result.report, experiment=experiment, recipe=recipe, pointer_file=pointer_file
        )
    if args.report is not None:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report_to_json(result.report) + "\n", encoding="utf-8")
    summary = {
        "run_id": result.pointer.artifact.artifact_id,
        "run_dir": result.directory.relative_to(store.root).as_posix(),
        "pointer": None if pointer_file is None else pointer_file.relative_to(records_root).as_posix(),
        "method": result.summary.method,
        "termination": result.summary.termination.kind,
        "success": result.summary.outcome.success,
        "boundary_jump_rad": result.boundary_jump,
        "joint_rmse": None if result.report.joint_rmse is None else result.report.joint_rmse.aggregate,
        "mlflow_run_id": None if tracked is None else tracked.mlflow_run_id,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
