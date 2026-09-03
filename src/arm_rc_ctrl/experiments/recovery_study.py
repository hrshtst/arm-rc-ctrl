# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Running one recovery formulation study with parent/child MLflow logging (M3R-012; recovery plan section 9.2).

One Optuna study per generator formulation (``armrc://optuna/<name>.db``),
mirrored in MLflow exactly like the M3 ESN search: one parent run per study —
the protocol, its digest, the dataset and both frozen tracker identities,
provenance, and at the end the study summary and selection — and one child run
per trial holding the point, the worst-cell objective, every paired component
as its own metric, the running objective as a metric series, the feasibility
reason, and the full evaluation as an artifact. Logging is idempotent per
(study, trial), so a resumed study neither duplicates nor loses trials; the
approved anchors are queued once as comparison trials. The Optuna database and
the written report remain the primary records; MLflow is the browsable mirror.

Command line::

    python -m arm_rc_ctrl.experiments.recovery_study --protocol configs/studies/<study>.toml
        --dataset data/records/processed/<id>.toml --report docs/experiments/<task>/<study>.json
        [--markdown docs/experiments/<task>/<study>.md] [--max-trials N] [--records-root ROOT]
        [--exploratory] [--no-mlflow]
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
from arm_rc_ctrl.experiments.esn_search import PLANNED_PARAMETERS
from arm_rc_ctrl.experiments.esn_study import STUDY_TAG, TRIAL_TAG, is_feasible
from arm_rc_ctrl.experiments.recovery_objective import (
    RecoveryTrialContext,
    RecoveryTrialEvaluation,
    make_recovery_objective,
)
from arm_rc_ctrl.experiments.recovery_search import (
    RECOVERY_TRACKERS,
    RecoverySearchProtocol,
    RecoveryTrialPoint,
    enqueue_recovery_comparisons,
    load_recovery_search,
    point_from_params,
    recovery_protocol_digest,
)
from arm_rc_ctrl.experiments.scalars import flatten_scalars
from arm_rc_ctrl.experiments.studies import (
    StudySummary,
    close_study,
    finished,
    open_study,
    run_trials,
    select_best,
    summarize,
)
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

__all__ = [
    "FEASIBLE_RULE",
    "REPORT_SCHEMA_VERSION",
    "RecoveryStudyReport",
    "RecoveryStudyResult",
    "load_report",
    "main",
    "render_markdown",
    "report_to_json",
    "run_recovery_study",
]

REPORT_SCHEMA_VERSION: Final = 1
FEASIBLE_RULE: Final = "feasible"
"""Selection rule: only trials feasible in every (scenario, tracker) pair can be selected (M3 rule reused)."""


@dataclass(frozen=True)
class RecoveryStudyReport:
    """The selection report of one formulation study run."""

    protocol: str
    protocol_file: str
    protocol_sha256: str
    formulation: str
    dataset: str
    trackers: dict[str, str]
    """SHA-256 of each frozen tracker's gains, by baseline name."""
    budget: int
    trials_run: int
    """Trials evaluated by this invocation (the rest were stored by earlier invocations)."""
    summary: StudySummary
    best_point: RecoveryTrialPoint | None
    n_feasible: int
    provenance: ProvenanceRecord
    mlflow_parent_run: str | None = None
    schema_version: int = field(default=REPORT_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """The selection is re-derived from the stored trials: the feasible count and the best feasible trial."""
        if set(self.trackers) != set(RECOVERY_TRACKERS):
            msg = f"the report must record both frozen trackers {RECOVERY_TRACKERS}, got {sorted(self.trackers)}"
            raise ValueError(msg)
        feasible = [t for t in self.summary.trials if t.flags.get("feasible") is True]
        if self.n_feasible != len(feasible):
            msg = f"n_feasible {self.n_feasible} does not match the {len(feasible)} trials flagged feasible"
            raise ValueError(msg)
        eligible = [t for t in feasible if t.state == "COMPLETE" and t.value is not None]
        if not eligible:
            if self.summary.best_number is not None or self.best_point is not None:
                msg = "a study without a feasible completed trial selects nothing"
                raise ValueError(msg)
            return
        best = min(eligible, key=lambda t: (cast("float", t.value), t.number))
        if self.summary.best_number != best.number or self.summary.best_value != best.value:
            msg = f"the report does not select the best feasible trial ({best.number}, {best.value!r})"
            raise ValueError(msg)
        if self.best_point is None or {k: float(v) for k, v in self.best_point.params().items()} != best.params:
            msg = f"best_point does not equal the parameters of trial {best.number}"
            raise ValueError(msg)


@dataclass(frozen=True)
class RecoveryStudyResult:
    """Outputs of one study invocation."""

    report: RecoveryStudyReport
    study: optuna.Study
    evaluations: tuple[RecoveryTrialEvaluation, ...]
    """The evaluations performed by this invocation, in trial order."""
    child_runs: dict[int, str]
    """MLflow child run ID per trial number logged by this invocation."""


def _study_params(
    protocol: RecoverySearchProtocol, digest: str, context: RecoveryTrialContext, provenance: ProvenanceRecord
) -> dict[str, object]:
    params: dict[str, object] = {}
    flatten_scalars("protocol", to_mapping(protocol), params)
    params["protocol.sha256"] = digest
    params["dataset.artifact_id"] = context.dataset.artifact.artifact_id
    params["dataset.sha256"] = context.dataset.artifact.payload.sha256
    for name in context.trackers:
        params[f"tracker.{name}.sha256"] = frozen_baseline_digest(name)
    params["development.scenarios"] = len(context.scenarios)
    params["provenance.project_commit"] = provenance.project_commit
    params["provenance.project_dirty"] = provenance.project_dirty
    params["provenance.lock_sha256"] = provenance.lock_sha256
    for submodule in provenance.submodules:
        params[f"revisions.{submodule.name}"] = submodule.checked_out or submodule.recorded
    for build in provenance.builds:
        params[f"builds.{build.name}"] = f"{build.version}@{build.source_commit}"
    return params


def _trial_metrics(evaluation: RecoveryTrialEvaluation) -> dict[str, float]:
    metrics: dict[str, float] = {
        "objective": evaluation.objective,
        "feasible": float(evaluation.feasible),
        "penalized": float(evaluation.penalized),
        "pairs_evaluated": float(len(evaluation.components)),
        "pairs_total": float(evaluation.scenarios_total),
    }
    if evaluation.fit_rmse is not None:
        metrics["fit_rmse"] = evaluation.fit_rmse
    for name, value in evaluation.cells.items():
        metrics[f"cells.{name.replace(':', '.')}"] = value
    for component in evaluation.components:
        prefix = f"component.{component.index}.{component.tracker}"
        metrics[f"{prefix}.feasible"] = float(component.feasible)
        for metric in (
            "gap_ratio",
            "early_gap_integral",
            "replay_early_gap_integral",
            "activation_jump_rad",
            "settling_time_s",
            "torque_rms",
            "saturation_fraction",
            "boundary_jump",
        ):
            value = getattr(component, metric)
            if value is not None:
                metrics[f"{prefix}.{metric}"] = float(value)
        for name, ok in component.criteria.items():
            metrics[f"{prefix}.criteria.{name}"] = float(ok)
    return metrics


class _StudyLogger:
    """Parent/child MLflow logging of one study invocation."""

    def __init__(self, tracker: MlflowTracker, protocol: RecoverySearchProtocol) -> None:
        self.tracker = tracker
        self.protocol = protocol
        self.experiment = tracker.experiment_id(protocol.name)
        self.parent: str | None = None
        self.children: dict[int, str] = {}

    def start(
        self, digest: str, context: RecoveryTrialContext, provenance: ProvenanceRecord, protocol_file: Path
    ) -> str:
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

    def log_trial(self, trial: optuna.Trial, evaluation: RecoveryTrialEvaluation) -> None:
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

    def finish(self, report: RecoveryStudyReport) -> None:
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


def report_to_json(report: RecoveryStudyReport) -> str:
    """Canonical JSON of the report."""
    return canonical_json(to_mapping(report))


def load_report(path: Path) -> RecoveryStudyReport:
    """Strictly rebuild a report from JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), RecoveryStudyReport)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def _reason_key(reason: str) -> str:
    if reason.startswith("scenario "):
        return reason.partition("]: ")[2] or reason
    return reason


def render_markdown(report: RecoveryStudyReport) -> str:
    """A Markdown selection report: budget, outcomes, cells, comparison points, and the selected point."""
    summary = report.summary
    trials = summary.trials
    feasible = [t for t in trials if t.flags.get("feasible") is True]
    reasons: dict[str, int] = {}
    for trial in trials:
        if trial.flags.get("feasible") is not True:
            reason = trial.labels.get("reason", "") or "(pruned)"
            key = _reason_key(reason)
            reasons[key] = reasons.get(key, 0) + 1
    names = [*PLANNED_PARAMETERS, "warmup_s"]
    if report.formulation != "no_augmentation":
        names += ["n_synthetic", "sigma_rad", "phi", "gamma"]
    dirty = " (dirty)" if report.provenance.project_dirty else ""
    seed = report.provenance.seeds.get("sampler")
    trackers = ", ".join(f"`{name}` (digest `{digest[:12]}`)" for name, digest in sorted(report.trackers.items()))
    lines = [
        f"# Recovery search `{report.protocol}`",
        "",
        (
            f"- Formulation `{report.formulation}`; protocol `{report.protocol_file}` "
            f"(digest `{report.protocol_sha256[:12]}`), dataset `{report.dataset}`."
        ),
        f"- Frozen trackers: {trackers}.",
        (
            f"- Budget {report.budget}; stored {len(trials)} trials ({summary.n_complete} complete, "
            f"{summary.n_pruned} pruned); {report.n_feasible} feasible; this invocation ran {report.trials_run}."
        ),
        f"- Provenance: commit `{report.provenance.project_commit[:12]}`{dirty}, sampler seed {seed}.",
        "",
        "## Selection",
        "",
    ]
    if report.best_point is None or summary.best_number is None:
        lines.append("No feasible completed trial: nothing is selected.")
    else:
        best = next(t for t in trials if t.number == summary.best_number)
        point = report.best_point.params()
        lines.append(
            f"Trial {best.number} with objective {_fmt(best.value)} "
            "(worst class-by-tracker cell median of the early command-gap ratio)."
        )
        lines.extend(["", "| cell | median gap ratio |", "| --- | --- |"])
        lines.extend(
            f"| {key.removeprefix('cells.')} | {_fmt(value)} |"
            for key, value in sorted(best.metrics.items())
            if key.startswith("cells.")
        )
        lines.extend(["", "| parameter | value |", "| --- | --- |"])
        lines.extend(f"| {name} | {point[name]!r} |" for name in names if name in point)
        lines.extend(
            ["", "Development pairs of the selected trial:", "", "| # | scenario | tracker | kind | gap ratio |"]
        )
        lines.append("| --- | --- | --- | --- | --- |")
        index = 0
        while f"components.{index}.kind" in best.labels:
            prefix = f"components.{index}"
            lines.append(
                f"| {index} | {best.labels.get(prefix + '.scenario_id', '')} "
                f"| {best.labels.get(prefix + '.tracker', '')} | {best.labels[prefix + '.kind']} "
                f"| {_fmt(best.metrics.get(prefix + '.gap_ratio'))} |"
            )
            index += 1
        lines.append("")
    lines.extend(
        [
            "## Comparison points",
            "",
            "| label | trial | objective | feasible | reason |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for trial in trials:
        label = trial.labels.get("armrc.comparison")
        if label is not None:
            reason = trial.labels.get("reason", "")
            lines.append(
                f"| {label} | {trial.number} | {_fmt(trial.value)} | {trial.flags.get('feasible')} | {reason} |"
            )
    lines.extend(["", "## Best feasible trials", "", "| trial | objective | " + " | ".join(names) + " |"])
    lines.append("| --- | --- | " + " | ".join("---" for _ in names) + " |")
    for trial in sorted(feasible, key=lambda t: (t.value if t.value is not None else float("inf"), t.number))[:10]:
        values = " | ".join(f"{trial.params.get(name, float('nan')):.4g}" for name in names)
        lines.append(f"| {trial.number} | {_fmt(trial.value)} | {values} |")
    lines.extend(["", "## Infeasible and pruned trials by reason", "", "| reason | trials |", "| --- | --- |"])
    lines.extend(f"| {reason} | {count} |" for reason, count in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])))
    return "\n".join(lines) + "\n"


def run_recovery_study(
    protocol: RecoverySearchProtocol,
    protocol_file: Path,
    *,
    store: StorageRoot,
    dataset_file: Path,
    records_root: Path,
    exploratory: bool,
    max_trials: int | None = None,
    tracker: MlflowTracker | None = None,
    now: datetime | None = None,
    command: str = "python -m arm_rc_ctrl.experiments.recovery_study",
) -> RecoveryStudyResult:
    """Create or resume the study, run up to ``max_trials`` new trials (the budget by default), and report."""
    context = RecoveryTrialContext.load(protocol, store=store, dataset_file=dataset_file, records_root=records_root)
    digest = recovery_protocol_digest(protocol)
    dataset = context.dataset
    payload = dataset.artifact.payload
    tracker_digests = {name: frozen_baseline_digest(name) for name in context.trackers}
    resolved = {
        "protocol": to_mapping(protocol),
        "protocol_file": _relative(protocol_file),
        "protocol_sha256": digest,
        "formulation": protocol.formulation,
        "dataset": dataset.artifact.artifact_id,
        "trackers": tracker_digests,
        "max_trials": max_trials,
        "command": command,
    }
    provenance = collect_provenance(
        resolved,
        seeds={"sampler": protocol.sampler.seed, "seed_bank": protocol.seed_bank},
        artifacts=[ArtifactReference(payload.uri, payload.sha256, payload.size)],
        exploratory=exploratory,
        now=now,
    )
    require_clean_for_confirmatory(provenance)
    study = open_study(store, protocol.name, protocol_sha256=digest, sampler=protocol.sampler, pruner=protocol.pruner)
    enqueue_recovery_comparisons(study, protocol)
    logger = None if tracker is None else _StudyLogger(tracker, protocol)
    parent = None if logger is None else logger.start(digest, context, provenance, protocol_file)
    evaluations: list[RecoveryTrialEvaluation] = []

    def on_evaluation(trial: optuna.Trial, evaluation: RecoveryTrialEvaluation) -> None:
        evaluations.append(evaluation)
        if logger is not None:
            logger.log_trial(trial, evaluation)

    objective = make_recovery_objective(protocol, context, on_evaluation=on_evaluation)
    budget = protocol.budget if max_trials is None else min(protocol.budget, len(finished(study)) + max_trials)
    try:
        ran = run_trials(study, objective, budget=budget)
        summary = summarize(study, eligible=is_feasible, selection_rule=FEASIBLE_RULE)
        best = None
        if summary.best_number is not None:
            best = point_from_params(protocol, select_best(study, eligible=is_feasible).params)
    finally:
        close_study(study)
    n_feasible = sum(1 for t in summary.trials if t.flags.get("feasible") is True)
    report = RecoveryStudyReport(
        protocol=protocol.name,
        protocol_file=_relative(protocol_file),
        protocol_sha256=digest,
        formulation=str(protocol.formulation),
        dataset=dataset.artifact.artifact_id,
        trackers=tracker_digests,
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
    return RecoveryStudyResult(report, study, tuple(evaluations), {} if logger is None else dict(logger.children))


def _relative(path: Path) -> str:
    root = repository_root()
    resolved = path.resolve()
    return resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else path.name


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run (or resume) one recovery formulation study of a task.")
    parser.add_argument("--protocol", type=Path, required=True, help="recovery search protocol (configs/studies)")
    parser.add_argument("--dataset", type=Path, required=True, help="recovery dataset record (TOML)")
    parser.add_argument("--report", type=Path, required=True, help="selection report JSON to write (must not exist)")
    parser.add_argument(
        "--markdown", type=Path, default=None, help="optional Markdown report to write (must not exist)"
    )
    parser.add_argument("--max-trials", type=int, default=None, help="run at most this many new trials, then stop")
    parser.add_argument("--records-root", type=Path, default=None, help="root the dataset record is relative to")
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    parser.add_argument("--no-mlflow", action="store_true", help="skip the MLflow mirror (scratch only)")
    args = parser.parse_args(argv)
    ensure_single_thread()  # before rclib is imported and provenance is collected
    for target in (args.report, args.markdown):
        if target is not None and Path(target).exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    if args.max_trials is not None and args.max_trials < 1:
        msg = "--max-trials must be >= 1"
        raise ValueError(msg)
    store = open_storage()
    protocol = load_recovery_search(Path(args.protocol))
    records_root = repository_root() if args.records_root is None else Path(args.records_root)
    result = run_recovery_study(
        protocol,
        Path(args.protocol),
        store=store,
        dataset_file=Path(args.dataset),
        records_root=records_root,
        exploratory=bool(args.exploratory),
        max_trials=args.max_trials,
        tracker=None if args.no_mlflow else MlflowTracker(store),
        now=datetime.now(tz=UTC),
        command=command_line("arm_rc_ctrl.experiments.recovery_study", sys.argv[1:] if argv is None else argv),
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report_to_json(result.report) + "\n", encoding="utf-8")
    if args.markdown is not None:
        Path(args.markdown).write_text(render_markdown(result.report), encoding="utf-8")
    report = result.report
    print(
        json.dumps(
            {
                "study": report.protocol,
                "formulation": report.formulation,
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
