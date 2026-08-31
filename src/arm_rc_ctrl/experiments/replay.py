# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Headless direct-replay baselines (``docs/PLAN.md`` section 6).

The demonstrated trajectory is fed directly to a low-level tracker (PD or
computed torque) in ``skelarm``. Every control step checks the measured state
and the command before use; limit violations, invalid states, and invalid
outputs stop the run with a typed termination. The run is persisted as a
provenance-complete run record and evaluated with the report functions.

Command line::

    python -m arm_rc_ctrl.experiments.replay --scenario configs/tasks/task_1a.toml
        --dataset data/records/processed/<id>.toml --controller configs/controllers/task_1a_pd.toml
        [--exploratory] [--report report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.config import load_config, to_mapping
from arm_rc_ctrl.controllers.reference import DemonstrationReference
from arm_rc_ctrl.controllers.tracking import LimitedTracker, TrackerConfig
from arm_rc_ctrl.data.phases import intervals_from_phases
from arm_rc_ctrl.data.records import ProcessedDatasetRecord, load_record, verify_payload
from arm_rc_ctrl.data.samples import SampleSet, load_samples
from arm_rc_ctrl.experiments.disturbances import ForcePulse
from arm_rc_ctrl.experiments.run_record import (
    LoadedRun,
    RunArrays,
    RunPointerRecord,
    RunSummary,
    load_run,
    record_run_pointer,
    write_run,
)
from arm_rc_ctrl.experiments.simulation import simulate
from arm_rc_ctrl.experiments.termination import (
    Outcome,
    Termination,
)
from arm_rc_ctrl.experiments.tracking import MlflowTracker
from arm_rc_ctrl.metrics.dwell import dwell_metrics
from arm_rc_ctrl.metrics.joint import JointAnglePolicy
from arm_rc_ctrl.metrics.report import RunReport, build_report, report_to_json
from arm_rc_ctrl.provenance import ArtifactReference, collect_provenance, command_line, require_clean_for_confirmatory
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import ScenarioConfig, load_scenario
from arm_rc_ctrl.storage import StorageRoot, open_storage

__all__ = ["ReplayResult", "bind_dataset", "dwell_outcome", "main", "run_replay", "simulate_tracking"]


@dataclass(frozen=True)
class ReplayResult:
    """Outputs of one replay run."""

    pointer: RunPointerRecord
    summary: RunSummary
    directory: Path
    run: LoadedRun
    report: RunReport


def simulate_tracking(
    scenario: ScenarioConfig,
    reference: DemonstrationReference,
    tracker: TrackerConfig,
    *,
    duration_s: float,
    initial_q: tuple[float, ...] | None = None,
    force: ForcePulse | None = None,
) -> tuple[RunArrays, Termination]:
    """Run the tracker against ``reference`` in ``skelarm`` and return the telemetry and termination.

    A ``force`` pulse acts on the endpoint through the Jacobian transpose in addition
    to the tracker's (limited) torque and is logged in the ``ext_force`` array.
    """
    controller = LimitedTracker(reference, tracker, scenario.limits.torque)
    return simulate(scenario, controller, duration_s=duration_s, initial_q=initial_q, force=force)


def dwell_outcome(
    scenario: ScenarioConfig, reference: SampleSet, arrays: RunArrays, termination: Termination
) -> dict[str, bool]:
    """Success criteria of a run: completion plus the scenario's dwell criteria over the reference dwell window."""
    criteria = scenario.task.dwell_criteria
    result = {"completed": termination.is_completed, **dict.fromkeys(criteria.names, False)}
    if not termination.is_completed:
        return result
    intervals = intervals_from_phases(reference.t, reference.phase)
    run_t = cast("NDArray[np.float64]", arrays.arrays["t"])
    if not np.any((run_t >= intervals.dwell[0]) & (run_t <= intervals.dwell[1])):
        return result
    metrics = dwell_metrics(
        run_t,
        cast("NDArray[np.float64]", arrays.arrays["tip"]),
        cast("NDArray[np.float64]", arrays.arrays["dq"]),
        np.asarray(scenario.task.target, dtype=np.float64),
        criteria.tolerance,
        window=(intervals.dwell[0], intervals.dwell[1]),
    )
    result.update(criteria.evaluate(metrics))
    return result


def bind_dataset(
    scenario: ScenarioConfig, scenario_file: Path, dataset: ProcessedDatasetRecord, reference: SampleSet
) -> None:
    """Fail unless the dataset was derived under ``scenario_file`` and matches its record and the scenario."""
    dataset.check_scenario(scenario_file)
    dataset.check_samples(reference)
    if scenario.dof != dataset.dof:
        msg = f"dof mismatch: scenario {scenario.dof}, dataset {dataset.dof}"
        raise ValueError(msg)


def run_replay(
    scenario: ScenarioConfig,
    scenario_file: Path,
    dataset: ProcessedDatasetRecord,
    reference: SampleSet,
    tracker: TrackerConfig,
    *,
    store: StorageRoot,
    exploratory: bool,
    now: datetime | None = None,
    command: str = "python -m arm_rc_ctrl.experiments.replay",
    initial_q: tuple[float, ...] | None = None,
    force: ForcePulse | None = None,
    license_label: str = "LicenseRef-Private",
    access: str = "private",
) -> ReplayResult:
    """Replay the demonstration with the tracker, persist the run, and evaluate it."""
    bind_dataset(scenario, scenario_file, dataset, reference)
    if tracker.dof != scenario.dof:
        msg = f"dof mismatch: scenario {scenario.dof}, tracker {tracker.dof}"
        raise ValueError(msg)
    reference_artifact = dataset.artifact.artifact_id
    payload = dataset.artifact.payload
    resolved = {
        "scenario": to_mapping(scenario),
        "tracker": to_mapping(tracker),
        "reference_artifact": reference_artifact,
        "interpolation": dataset.preprocessing.interpolation,
        "initial_q": list(scenario.task.initial_q if initial_q is None else initial_q),
        "duration_s": float(reference.t[-1]),
        "force": None if force is None else to_mapping(force),
    }
    provenance = collect_provenance(
        resolved,
        seeds={},
        artifacts=[ArtifactReference(payload.uri, payload.sha256, payload.size)],
        exploratory=exploratory,
        now=now,
    )
    require_clean_for_confirmatory(provenance)
    demo = DemonstrationReference.from_samples(reference, cast("Any", dataset.preprocessing.interpolation))
    duration = float(reference.t[-1])
    arrays, termination = simulate_tracking(
        scenario, demo, tracker, duration_s=duration, initial_q=initial_q, force=force
    )
    outcome = Outcome(termination, dwell_outcome(scenario, reference, arrays, termination))
    pointer, summary, directory = write_run(
        store,
        arrays,
        kind="simulation",
        method=f"replay+{tracker.method}",
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
        notes=f"Direct replay of {reference_artifact} with {tracker.method}.",
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
    return ReplayResult(pointer, summary, directory, run, report)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; a thin wrapper around :func:`run_replay`."""
    parser = argparse.ArgumentParser(description="Replay a canonical demonstration with a low-level tracker.")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="processed dataset record (TOML)")
    parser.add_argument("--controller", type=Path, required=True, help="tracker TOML (configs/controllers/*.toml)")
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    parser.add_argument("--records-root", type=Path, default=None, help="repository root for the run pointer record")
    parser.add_argument(
        "--no-pointer", action="store_true", help="do not track the run under data/records/runs (exploratory scratch)"
    )
    parser.add_argument("--no-mlflow", action="store_true", help="skip the mandatory MLflow logging (scratch only)")
    parser.add_argument("--experiment", default=None, help="MLflow experiment name (default: the scenario name)")
    parser.add_argument("--report", type=Path, default=None, help="write the report JSON here")
    args = parser.parse_args(argv)
    store = open_storage()
    scenario = load_scenario(Path(args.scenario))
    record = load_record(Path(args.dataset), ProcessedDatasetRecord)
    samples = load_samples(verify_payload(store, record.artifact))
    tracker = load_config(Path(args.controller), TrackerConfig)
    result = run_replay(
        scenario,
        Path(args.scenario),
        record,
        samples,
        tracker,
        store=store,
        exploratory=args.exploratory,
        now=datetime.now(UTC),
        command=command_line("arm_rc_ctrl.experiments.replay", argv if argv is not None else sys.argv[1:]),
    )
    records_root = repository_root() if args.records_root is None else Path(args.records_root)
    pointer_file = None if args.no_pointer else record_run_pointer(records_root, result.pointer)
    tracked = None
    if not args.no_mlflow:
        experiment = scenario.name if args.experiment is None else str(args.experiment)
        tracked = MlflowTracker(store).log_run(
            result.run, result.report, experiment=experiment, pointer_file=pointer_file
        )
    text = report_to_json(result.report)
    if args.report is not None:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    summary = {
        "run_id": result.pointer.artifact.artifact_id,
        "run_dir": result.directory.relative_to(store.root).as_posix(),
        "pointer": None if pointer_file is None else pointer_file.relative_to(records_root).as_posix(),
        "termination": result.summary.termination.kind,
        "success": result.summary.outcome.success,
        "mlflow_run_id": None if tracked is None else tracked.mlflow_run_id,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
