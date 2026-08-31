# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Running an ESN search study with parent/child MLflow logging (``docs/PLAN.md`` sections 10 and 11; M3-005).

The study (Optuna, ``armrc://optuna/<name>.db``) is mirrored in MLflow as one
parent run per study — the protocol, its digest, the dataset and tracker
identities, provenance, and at the end the study summary and selection — and
one child run per trial holding the point, the scalar objective, every
objective component as its own metric, the running objective as a metric
series, the reason, and the full evaluation as an artifact. Logging is
idempotent per (study, trial), so a resumed study neither duplicates nor
loses trials. The Optuna database and the written report remain the primary
records; MLflow is the browsable mirror.

Command line::

    python -m arm_rc_ctrl.experiments.esn_study --protocol configs/studies/esn_search_1a.toml
        --dataset data/records/processed/<id>.toml --report docs/experiments/task_1a/esn_search.json
        [--max-trials N] [--records-root ROOT] [--exploratory] [--no-mlflow]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.experiments.baselines import frozen_baseline_digest
from arm_rc_ctrl.experiments.esn_objective import TrialContext, TrialEvaluation, make_objective
from arm_rc_ctrl.experiments.esn_search import (
    EsnSearchProtocol,
    TrialPoint,
    enqueue_comparisons,
    load_esn_search,
    point_from_params,
    protocol_digest,
)
from arm_rc_ctrl.experiments.scalars import flatten_scalars
from arm_rc_ctrl.experiments.studies import StudySummary, finished, open_study, run_trials, select_best, summarize
from arm_rc_ctrl.experiments.tracking import MlflowTracker
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    ProvenanceRecord,
    canonical_json,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
)
from arm_rc_ctrl.rc.esn import ensure_single_thread
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import open_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    import optuna

    from arm_rc_ctrl.storage import StorageRoot

__all__ = ["REPORT_SCHEMA_VERSION", "EsnStudyReport", "EsnStudyResult", "main", "run_esn_study"]

REPORT_SCHEMA_VERSION: Final = 1
STUDY_TAG: Final = "armrc.study"
TRIAL_TAG: Final = "armrc.trial"


@dataclass(frozen=True)
class EsnStudyReport:
    """The selection report of a study run (``docs/experiments/<task>/esn_search.json``)."""

    protocol: str
    protocol_file: str
    protocol_sha256: str
    dataset: str
    tracker: str
    tracker_sha256: str
    budget: int
    trials_run: int
    """Trials evaluated by this invocation (the rest were stored by earlier invocations)."""
    summary: StudySummary
    best_point: TrialPoint | None
    n_feasible: int
    provenance: ProvenanceRecord
    mlflow_parent_run: str | None = None
    schema_version: int = field(default=REPORT_SCHEMA_VERSION)


@dataclass(frozen=True)
class EsnStudyResult:
    """Outputs of one study invocation."""

    report: EsnStudyReport
    study: optuna.Study
    evaluations: tuple[TrialEvaluation, ...]
    """The evaluations performed by this invocation, in trial order."""
    child_runs: dict[int, str]
    """MLflow child run ID per trial number logged by this invocation."""


def _study_params(
    protocol: EsnSearchProtocol, digest: str, context: TrialContext, provenance: ProvenanceRecord
) -> dict[str, object]:
    params: dict[str, object] = {}
    flatten_scalars("protocol", to_mapping(protocol), params)
    params["protocol.sha256"] = digest
    params["dataset.artifact_id"] = context.dataset.artifact.artifact_id
    params["dataset.sha256"] = context.dataset.artifact.payload.sha256
    params["tracker.sha256"] = frozen_baseline_digest(protocol.tracker)
    params["provenance.project_commit"] = provenance.project_commit
    params["provenance.project_dirty"] = provenance.project_dirty
    params["provenance.lock_sha256"] = provenance.lock_sha256
    for submodule in provenance.submodules:
        params[f"revisions.{submodule.name}"] = submodule.checked_out or submodule.recorded
    for build in provenance.builds:
        params[f"builds.{build.name}"] = f"{build.version}@{build.source_commit}"
    return params


def _trial_metrics(evaluation: TrialEvaluation) -> dict[str, float]:
    metrics: dict[str, float] = {
        "objective": evaluation.objective,
        "feasible": float(evaluation.feasible),
        "penalized": float(evaluation.penalized),
        "scenarios_evaluated": float(len(evaluation.components)),
        "scenarios_total": float(evaluation.scenarios_total),
    }
    if evaluation.fit_rmse is not None:
        metrics["fit_rmse"] = evaluation.fit_rmse
    for component in evaluation.components:
        prefix = f"component.{component.index}"
        metrics[f"{prefix}.feasible"] = float(component.feasible)
        if component.move_joint_rmse is not None:
            metrics[f"{prefix}.move_joint_rmse"] = component.move_joint_rmse
        if component.saturation_fraction is not None:
            metrics[f"{prefix}.saturation_fraction"] = component.saturation_fraction
        if component.boundary_jump is not None:
            metrics[f"{prefix}.boundary_jump"] = component.boundary_jump
        for name, ok in component.criteria.items():
            metrics[f"{prefix}.criteria.{name}"] = float(ok)
    return metrics


class _StudyLogger:
    """Parent/child MLflow logging of one study invocation."""

    def __init__(self, tracker: MlflowTracker, protocol: EsnSearchProtocol) -> None:
        self.tracker = tracker
        self.protocol = protocol
        self.experiment = tracker.experiment_id(protocol.name)
        self.parent: str | None = None
        self.children: dict[int, str] = {}

    def start(self, digest: str, context: TrialContext, provenance: ProvenanceRecord, protocol_file: Path) -> str:
        """Find or create the parent run of the study (one per study name and protocol digest)."""
        tags = {STUDY_TAG: self.protocol.name, "armrc.kind": "study", "armrc.protocol_sha256": digest}
        existing = self.tracker.find_by_tags(self.experiment, tags)
        if existing is not None:
            self.parent = existing
            return existing
        self.parent = self.tracker.start_run(
            self.experiment,
            name=self.protocol.name,
            tags={**tags, "armrc.project_commit": provenance.project_commit},
            params=_study_params(self.protocol, digest, context, provenance),
        )
        self.tracker.log_text(self.parent, "protocol.toml", protocol_file.read_text(encoding="utf-8"))
        self.tracker.log_text(self.parent, "provenance.json", provenance.to_json() + "\n")
        return self.parent

    def log_trial(self, trial: optuna.Trial, evaluation: TrialEvaluation) -> None:
        """Log one evaluated trial as a child run (skipped when the trial is already mirrored)."""
        assert self.parent is not None  # start() precedes every trial
        tags = {STUDY_TAG: self.protocol.name, TRIAL_TAG: str(trial.number), "armrc.kind": "trial"}
        if self.tracker.find_by_tags(self.experiment, tags) is not None:
            return
        label = trial.user_attrs.get("armrc.comparison")
        tags["mlflow.parentRunId"] = self.parent
        tags["armrc.state"] = "PRUNED" if evaluation.stopped_early else "COMPLETE"
        tags["armrc.feasible"] = str(evaluation.feasible)
        tags["armrc.reason"] = "" if evaluation.reason is None else evaluation.reason
        if label is not None:
            tags["armrc.comparison"] = str(label)
        params: dict[str, object] = {
            "trial.number": trial.number,
            **{f"point.{k}": v for k, v in evaluation.point.params().items()},
        }
        run_id = self.tracker.start_run(self.experiment, name=f"trial-{trial.number}", tags=tags, params=params)
        self.tracker.log_metrics(run_id, _trial_metrics(evaluation))
        self.tracker.log_series(run_id, "objective_running", evaluation.running)
        self.tracker.log_text(run_id, "evaluation.json", canonical_json(to_mapping(evaluation)) + "\n")
        self.tracker.finish(run_id)
        self.children[trial.number] = run_id

    def finish(self, report: EsnStudyReport) -> None:
        """Log the study summary and selection to the parent."""
        assert self.parent is not None  # start() precedes finish()
        summary = report.summary
        metrics = {
            "n_complete": float(summary.n_complete),
            "n_pruned": float(summary.n_pruned),
            "n_feasible": float(report.n_feasible),
            "trials_run": float(report.trials_run),
        }
        if summary.best_value is not None:
            metrics["best_value"] = summary.best_value
        if summary.best_number is not None:
            metrics["best_number"] = float(summary.best_number)
        self.tracker.log_metrics(self.parent, metrics, step=len(summary.trials))
        self.tracker.log_text(self.parent, "study_summary.json", canonical_json(to_mapping(summary)) + "\n")
        self.tracker.log_text(self.parent, "report.json", report_to_json(report) + "\n")
        self.tracker.finish(self.parent)


def report_to_json(report: EsnStudyReport) -> str:
    """Canonical JSON of the report."""
    return canonical_json(to_mapping(report))


def load_report(path: Path) -> EsnStudyReport:
    """Strictly rebuild a report from JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), EsnStudyReport)


def run_esn_study(
    protocol: EsnSearchProtocol,
    protocol_file: Path,
    *,
    store: StorageRoot,
    dataset_file: Path,
    records_root: Path,
    exploratory: bool,
    max_trials: int | None = None,
    tracker: MlflowTracker | None = None,
    now: datetime | None = None,
    command: str = "python -m arm_rc_ctrl.experiments.esn_study",
) -> EsnStudyResult:
    """Create or resume the study, run up to ``max_trials`` new trials (the budget by default), and report."""
    context = TrialContext.load(protocol, store=store, dataset_file=dataset_file, records_root=records_root)
    digest = protocol_digest(protocol)
    dataset = context.dataset
    payload = dataset.artifact.payload
    resolved = {
        "protocol": to_mapping(protocol),
        "protocol_file": _relative(protocol_file),
        "protocol_sha256": digest,
        "dataset": dataset.artifact.artifact_id,
        "tracker_sha256": frozen_baseline_digest(protocol.tracker),
        "max_trials": max_trials,
        "command": command,
    }
    provenance = collect_provenance(
        resolved,
        seeds={"sampler": protocol.sampler.seed},
        artifacts=[ArtifactReference(payload.uri, payload.sha256, payload.size)],
        exploratory=exploratory,
        now=now,
    )
    require_clean_for_confirmatory(provenance)
    study = open_study(store, protocol.name, protocol_sha256=digest, sampler=protocol.sampler, pruner=protocol.pruner)
    enqueue_comparisons(study, protocol)
    logger = None if tracker is None else _StudyLogger(tracker, protocol)
    parent = None if logger is None else logger.start(digest, context, provenance, protocol_file)
    evaluations: list[TrialEvaluation] = []

    def on_evaluation(trial: optuna.Trial, evaluation: TrialEvaluation) -> None:
        evaluations.append(evaluation)
        if logger is not None:
            logger.log_trial(trial, evaluation)

    objective = make_objective(protocol, context, on_evaluation=on_evaluation)
    budget = protocol.budget if max_trials is None else min(protocol.budget, len(finished(study)) + max_trials)
    ran = run_trials(study, objective, budget=budget)
    summary = summarize(study)
    best = None
    if summary.best_number is not None:
        best = point_from_params(protocol.search, select_best(study).params)
    n_feasible = sum(1 for t in summary.trials if t.flags.get("feasible") is True)
    report = EsnStudyReport(
        protocol=protocol.name,
        protocol_file=_relative(protocol_file),
        protocol_sha256=digest,
        dataset=dataset.artifact.artifact_id,
        tracker=protocol.tracker,
        tracker_sha256=frozen_baseline_digest(protocol.tracker),
        budget=protocol.budget,
        trials_run=ran,
        summary=summary,
        best_point=best,
        n_feasible=n_feasible,
        provenance=provenance,
        mlflow_parent_run=parent,
    )
    if logger is not None:
        logger.finish(report)
    return EsnStudyResult(report, study, tuple(evaluations), {} if logger is None else dict(logger.children))


def _relative(path: Path) -> str:
    root = repository_root()
    resolved = path.resolve()
    return resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else path.name


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run (or resume) the ESN hyperparameter search of a task.")
    parser.add_argument(
        "--protocol", type=Path, required=True, help="search protocol (configs/studies/esn_search_*.toml)"
    )
    parser.add_argument("--dataset", type=Path, required=True, help="processed dataset record (TOML)")
    parser.add_argument("--report", type=Path, required=True, help="selection report JSON to write (must not exist)")
    parser.add_argument("--max-trials", type=int, default=None, help="run at most this many new trials, then stop")
    parser.add_argument("--records-root", type=Path, default=None, help="root the dataset record is relative to")
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    parser.add_argument("--no-mlflow", action="store_true", help="skip the MLflow mirror (scratch only)")
    args = parser.parse_args(argv)
    ensure_single_thread()  # before rclib is imported and provenance is collected
    if Path(args.report).exists():
        msg = f"refusing to overwrite {args.report}"
        raise FileExistsError(msg)
    if args.max_trials is not None and args.max_trials < 1:
        msg = "--max-trials must be >= 1"
        raise ValueError(msg)
    store = open_storage()
    protocol = load_esn_search(Path(args.protocol))
    records_root = repository_root() if args.records_root is None else Path(args.records_root)
    result = run_esn_study(
        protocol,
        Path(args.protocol),
        store=store,
        dataset_file=Path(args.dataset),
        records_root=records_root,
        exploratory=bool(args.exploratory),
        max_trials=args.max_trials,
        tracker=None if args.no_mlflow else MlflowTracker(store),
        now=datetime.now(tz=UTC),
        command=command_line("arm_rc_ctrl.experiments.esn_study", sys.argv[1:] if argv is None else argv),
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report_to_json(result.report) + "\n", encoding="utf-8")
    report = result.report
    print(
        json.dumps(
            {
                "study": report.protocol,
                "trials_run": report.trials_run,
                "trials_stored": len(report.summary.trials),
                "budget": report.budget,
                "n_feasible": report.n_feasible,
                "best_number": report.summary.best_number,
                "best_value": report.summary.best_value,
                "mlflow_parent_run": report.mlflow_parent_run,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
