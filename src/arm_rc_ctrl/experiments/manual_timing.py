# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: what the manual-demonstration study will cost, projected from a measured subset.

The projection multiplies the study's own counts by the means a smoke check
measured. The counts are derived from the frozen study rather than written
here: six configurations of 31 arms are 186 models, and because replay is
driven through the derivative policy of the configuration it is paired against
(owner decision 2026-09-17), each configuration keeps its own bank per parent,
so the replay side is 60 banks and not the three warm-up banks the repeated
demonstration pilot had.

It is a projection and not a bound. Means measured on one configuration are
multiplied by the maximum run counts, while reservoir sizes, recording lengths
and storage overhead vary; a run that aborts early costs less, and an
infeasible model still costs its fit.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import TYPE_CHECKING, Final, cast

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.execution import ExecutionRecord
from arm_rc_ctrl.experiments.manual_evaluation import ManualModelTiming, ManualRunTiming, prepare_runner
from arm_rc_ctrl.experiments.manual_recipes import ASSIGNMENTS
from arm_rc_ctrl.experiments.manual_study import ARM_COUNT, CONFIGURATION_COUNT, EXPERIMENT_LABEL
from arm_rc_ctrl.provenance import ProvenanceRecord, canonical_json, sha256_file
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.validation import is_hex

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.experiments.manual_study import StudyManifest, StudyModel

__all__ = [
    "BUDGET_ARMS",
    "BUDGET_PARENT",
    "PARENT_COUNT",
    "TIMING_SCHEMA_VERSION",
    "ManualRunStats",
    "ManualStudyProjection",
    "ManualTimingReport",
    "budget_entries",
    "load_timing",
    "main",
    "peak_rss_bytes",
    "project_study",
    "render_timing_markdown",
    "smoke_entries",
    "summarize_timings",
    "timing_to_json",
]

TIMING_SCHEMA_VERSION: Final = 1
PARENT_COUNT: Final = len(ASSIGNMENTS)
"""The ten locked demonstrations; one replay bank per parent per configuration."""
BUDGET_PARENT: Final = "D01"
"""The one parent the budget subset's parent-specific arms all use, so configurations stay comparable."""
BUDGET_ARMS: Final = ("S", "M10", "R10", "C10")
"""Every arm kind, because training cost differs by kind and the projection scales what it measured."""
_SHA256_HEX: Final = 64
_MODULE: Final = "arm_rc_ctrl.experiments.manual_timing"


def budget_entries(manifest: StudyManifest, *, parent: str = BUDGET_PARENT) -> tuple[StudyModel, ...]:
    """The subset the measured budget rests on: every configuration crossed with every arm kind.

    A growing prefix is deterministic but can sit inside one configuration and
    miss the expensive training arms, so a projection built on it would scale
    costs it never measured. This spans both dimensions that drive cost -- the
    six inherited configurations and the four arm kinds -- at one fixed parent,
    which is 24 models and, because the all-ten arm has no single parent while
    the other three share their configuration's, six replay banks.

    Nothing here consults an outcome: the configurations are inherited to avoid
    selecting after seeing results, and the parent is fixed in advance.

    The manifest already guarantees six configurations crossed with the
    approved arms, so the subset cannot come up short without the study itself
    being invalid; only an unknown parent is checked here.
    """
    if parent not in ASSIGNMENTS:
        msg = f"unknown parent {parent!r}; the study's parents are {list(ASSIGNMENTS)}"
        raise ValueError(msg)
    wanted = {kind if kind == "M10" else f"{kind}/{parent}" for kind in BUDGET_ARMS}
    return tuple(entry for entry in manifest.entries if entry.arm.label in wanted)


def smoke_entries(
    manifest: StudyManifest, *, count: int, labels: Sequence[str] | None = None
) -> tuple[StudyModel, ...]:
    """The models a smoke check measures: the first ``count`` of the frozen study, in its own order.

    The subset is taken by position and never by what a label means. Plan
    section 5 inherits the six configurations precisely to avoid choosing after
    seeing results, and requires the deterministic nominal subset to run
    without dropping panel members on performance; a rule that preferred a
    configuration by name would be doing exactly that, on labels this study
    states are not known to predict performance on manual data.

    ``labels`` names a subset by hand for a targeted check, still in manifest
    order. A count larger than the study is an error rather than a quietly
    shorter check, because the projection divides by what was measured.
    """
    if count < 1:
        msg = f"a smoke check measures at least one model, got {count}"
        raise ValueError(msg)
    if labels:
        wanted = set(labels)
        chosen = tuple(entry for entry in manifest.entries if entry.label in wanted)
        unknown = sorted(wanted - {entry.label for entry in chosen})
        if unknown:
            msg = f"unknown model labels {unknown}"
            raise ValueError(msg)
        return chosen
    if count > len(manifest.entries):
        msg = f"the study holds {len(manifest.entries)} models, fewer than the {count} asked for"
        raise ValueError(msg)
    return tuple(manifest.entries[:count])


def peak_rss_bytes() -> tuple[int, int]:
    """Peak resident set size of this process and of its waited-for children (bytes).

    Workers are separate interpreters, so the children's peak is what a bounded
    parallel sweep actually needs beside this process's own.
    """
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return int(own) * 1024, int(children) * 1024  # Linux reports KiB


@dataclass(frozen=True)
class ManualRunStats:
    """Per-run wall-time statistics of one arm (seconds), so a projection built on means can be judged."""

    arm: str
    runs: int
    mean_simulate_s: float
    median_simulate_s: float
    max_simulate_s: float
    mean_persist_s: float
    mean_bytes: float


@dataclass(frozen=True)
class ManualStudyProjection:
    """The full study projected from the measured means of a subset (not a guaranteed bound)."""

    configurations: int
    arms: int
    models: int
    pairs_per_model: int
    parents: int
    replay_banks: int
    """``configurations x parents``: a bank belongs to one parent under one derivative policy."""
    rc_runs: int
    replay_runs: int
    total_runs: int
    rc_run_seconds: float
    replay_run_seconds: float
    fit_seconds: float
    """One fit per model of the whole study, at the mean measured fit cost."""
    total_seconds: float
    storage_bytes: int
    completed_models: int
    """Models whose evidence already exists after the measuring invocation."""
    remaining_seconds: float

    def __post_init__(self) -> None:
        """The counts are consistent and nothing is negative."""
        if self.models != self.configurations * self.arms:
            msg = f"{self.models} models is not {self.configurations} configurations of {self.arms} arms"
            raise ValueError(msg)
        if self.replay_banks != self.configurations * self.parents:
            msg = f"{self.replay_banks} banks is not {self.configurations} configurations of {self.parents} parents"
            raise ValueError(msg)
        if self.total_runs != self.rc_runs + self.replay_runs:
            msg = f"{self.total_runs} runs is not {self.rc_runs} RC and {self.replay_runs} replay"
            raise ValueError(msg)
        if min(self.total_seconds, self.remaining_seconds, self.storage_bytes, self.completed_models) < 0:
            msg = "a projection has no negative figures"
            raise ValueError(msg)


def summarize_timings(runs: Sequence[ManualRunTiming]) -> tuple[ManualRunStats, ...]:
    """Per-arm statistics of the measured runs; an arm with no measurements is omitted."""
    stats: list[ManualRunStats] = []
    for arm in ("replay", "rc"):
        selected = [run for run in runs if run.arm == arm]
        if not selected:
            continue
        stats.append(
            ManualRunStats(
                arm=arm,
                runs=len(selected),
                mean_simulate_s=float(mean(run.simulate_seconds for run in selected)),
                median_simulate_s=float(median(run.simulate_seconds for run in selected)),
                max_simulate_s=float(max(run.simulate_seconds for run in selected)),
                mean_persist_s=float(mean(run.persist_seconds for run in selected)),
                mean_bytes=float(mean(run.run_bytes for run in selected)),
            )
        )
    return tuple(stats)


def project_study(
    models: Sequence[ManualModelTiming],
    runs: Sequence[ManualRunTiming],
    *,
    pairs_per_model: int,
    completed_models: int,
    configurations: int = CONFIGURATION_COUNT,
    arms: int = ARM_COUNT,
    parents: int = PARENT_COUNT,
) -> ManualStudyProjection:
    """Project the whole study from the per-run and per-fit means measured here.

    An arm this invocation did not measure contributes nothing rather than
    dividing by zero, and what the invocation already completed is taken off
    the remaining estimate, so a resumed full run is not quoted its cost twice.
    """
    if pairs_per_model < 0 or completed_models < 0:
        msg = f"pairs_per_model and completed_models are non-negative, got {pairs_per_model} and {completed_models}"
        raise ValueError(msg)
    rc = [run for run in runs if run.arm == "rc"]
    replay = [run for run in runs if run.arm == "replay"]
    rc_seconds = float(mean(run.simulate_seconds + run.persist_seconds for run in rc)) if rc else 0.0
    replay_seconds = float(mean(run.simulate_seconds + run.persist_seconds for run in replay)) if replay else 0.0
    rc_bytes = float(mean(run.run_bytes for run in rc)) if rc else 0.0
    replay_bytes = float(mean(run.run_bytes for run in replay)) if replay else 0.0
    fit_per_model = float(mean(model.fit_seconds for model in models)) if models else 0.0
    total_models = configurations * arms
    replay_banks = configurations * parents
    rc_runs = total_models * pairs_per_model
    replay_runs = replay_banks * pairs_per_model
    fit_seconds = total_models * fit_per_model
    total = rc_runs * rc_seconds + replay_runs * replay_seconds + fit_seconds
    per_model = pairs_per_model * rc_seconds + fit_per_model
    remaining = max(0.0, total - completed_models * per_model - len(replay) * replay_seconds)
    return ManualStudyProjection(
        configurations=configurations,
        arms=arms,
        models=total_models,
        pairs_per_model=pairs_per_model,
        parents=parents,
        replay_banks=replay_banks,
        rc_runs=rc_runs,
        replay_runs=replay_runs,
        total_runs=rc_runs + replay_runs,
        rc_run_seconds=rc_seconds,
        replay_run_seconds=replay_seconds,
        fit_seconds=fit_seconds,
        total_seconds=total,
        storage_bytes=int(rc_runs * rc_bytes + replay_runs * replay_bytes),
        completed_models=completed_models,
        remaining_seconds=remaining,
    )


@dataclass(frozen=True)
class ManualTimingReport:
    """The committed evidence of one timing smoke check."""

    experiment: str
    study_manifest_sha256: str
    evaluation_sha256: str
    entries: tuple[str, ...]
    """The model labels this invocation measured, in the order it measured them."""
    execution: ExecutionRecord
    models: tuple[ManualModelTiming, ...]
    runs: tuple[ManualRunTiming, ...]
    run_stats: tuple[ManualRunStats, ...]
    runs_this_invocation: int
    """Runs this invocation simulated; anything else in ``runs`` was served from the store."""
    replay_banks_built: int
    wall_seconds: float
    peak_rss_bytes: int
    peak_rss_children_bytes: int
    storage_bytes: int
    projection: ManualStudyProjection
    revised_estimate: str
    provenance: ProvenanceRecord
    schema_version: int = field(default=TIMING_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """The report is internally consistent and states what it was written to state."""
        if self.schema_version != TIMING_SCHEMA_VERSION or self.experiment != EXPERIMENT_LABEL:
            msg = f"unsupported timing schema {self.schema_version} or experiment {self.experiment!r}"
            raise ValueError(msg)
        for name in ("study_manifest_sha256", "evaluation_sha256"):
            value = getattr(self, name)
            if not is_hex(value, _SHA256_HEX):
                msg = f"{name} must be 64 lowercase hex characters, got {value!r}"
                raise ValueError(msg)
        if self.runs_this_invocation > len(self.runs):
            msg = f"runs_this_invocation {self.runs_this_invocation} exceeds the {len(self.runs)} runs measured"
            raise ValueError(msg)
        if not self.revised_estimate.strip():
            msg = "the revised estimate must be stated: it is what this report exists to report"
            raise ValueError(msg)
        if min(self.runs_this_invocation, self.replay_banks_built, self.wall_seconds, self.storage_bytes) < 0:
            msg = "a timing report has no negative figures"
            raise ValueError(msg)


def timing_to_json(report: ManualTimingReport) -> str:
    """Canonical JSON of one smoke check."""
    return canonical_json(to_mapping(report))


def load_timing(path: Path) -> ManualTimingReport:
    """Strictly rebuild a timing report from its JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), ManualTimingReport)


def _hours(seconds: float) -> str:
    return f"{seconds / 3600.0:.2f} h"


def _gib(size: float) -> str:
    return f"{size / 2**30:.1f} GiB"


def _fit_source(model: ManualModelTiming) -> str:
    return "cache hit" if model.fit_cache_hit else "fitted now"


def render_timing_markdown(report: ManualTimingReport) -> str:
    """The Markdown rendering of a smoke check, stating what it measured and what it projected."""
    p = report.projection
    lines = [
        "# Task 1-a manual-demonstration timing smoke check (v1)",
        "",
        (
            f"Experiment `{report.experiment}`, study manifest sha256 `{report.study_manifest_sha256[:12]}`, "
            f"evaluation config sha256 `{report.evaluation_sha256[:12]}`, execution identity "
            f"`{report.execution.identity[:12]}` "
            f"({'canonical' if report.execution.canonical else 'NOT canonical'}), project commit "
            f"`{report.provenance.project_commit[:12]}`{' (dirty)' if report.provenance.project_dirty else ''}."
        ),
        "",
        "## Measured cost",
        "",
        (
            f"- Wall time of this invocation: {_hours(report.wall_seconds)} ({report.wall_seconds:.0f} s); "
            f"{report.runs_this_invocation} of {len(report.runs)} runs were simulated by it."
        ),
        (
            f"- Measured {len(report.models)} model(s) over {len(report.entries)} entr(ies) and built "
            f"{report.replay_banks_built} replay bank(s)."
        ),
        (
            f"- Peak resident set size: {_gib(report.peak_rss_bytes)} for this process, "
            f"{_gib(report.peak_rss_children_bytes)} for its waited-for children."
        ),
        f"- Storage of the measured runs and this invocation's manifests: {_gib(report.storage_bytes)}.",
        "",
        "| arm | runs | mean simulate s | median simulate s | max simulate s | mean persist s | mean bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {s.arm} | {s.runs} | {s.mean_simulate_s:.3f} | {s.median_simulate_s:.3f} | {s.max_simulate_s:.3f} "
        f"| {s.mean_persist_s:.3f} | {s.mean_bytes:.0f} |"
        for s in report.run_stats
    )
    lines += [
        "",
        "## Models",
        "",
        "| model | fit | fit s | sweep s | runs |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {m.label} | {_fit_source(m)} | {m.fit_seconds:.2f} | {m.sweep_seconds:.1f} | {m.runs} |"
        for m in report.models
    )
    lines += [
        "",
        "## Full-study projection (measured means scaled to every run; an estimate, not a bound)",
        "",
        (
            f"- {p.configurations} configurations x {p.arms} arms = {p.models:,} models x {p.pairs_per_model} pairs "
            f"= {p.rc_runs:,} RC runs at {p.rc_run_seconds:.2f} s each."
        ),
        (
            f"- {p.configurations} configurations x {p.parents} parents = {p.replay_banks} replay banks x "
            f"{p.pairs_per_model} pairs = {p.replay_runs:,} replay runs at {p.replay_run_seconds:.2f} s each."
        ),
        f"- {p.total_runs:,} runs in total; fits {_hours(p.fit_seconds)}.",
        f"- Projected total: {_hours(p.total_seconds)}; storage about {_gib(p.storage_bytes)}.",
        (
            f"- Already complete after this check: {p.completed_models} model(s); remaining about "
            f"{_hours(p.remaining_seconds)}."
        ),
        "",
        "## Revised estimate",
        "",
        report.revised_estimate,
        "",
        "## Limitations",
        "",
        (
            "- The projection multiplies maximum run counts by means measured on a subset. It is an estimate and "
            "not a guaranteed bound: reservoir sizes, recording lengths and storage overhead vary, an aborted run "
            "costs less, and an infeasible model still costs its fit."
        ),
        (
            "- Timings are wall-clock in the canonical single-threaded execution environment of this machine "
            "(C10); another core type, thread setting, or machine measures differently."
        ),
        "",
    ]
    return "\n".join(lines)


def _smoke(args: argparse.Namespace) -> int:
    """Measure a deterministic subset end to end and write the report and its rendering."""
    output, markdown = Path(cast("str", args.output)), Path(cast("str", args.markdown))
    for target in (output, markdown):
        # Checked before the sweep: evidence is written once, and finding out afterwards would
        # mean paying for the measurement and then throwing it away.
        if target.exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    started = time.perf_counter()
    prepared = prepare_runner(args, role="main", root=repository_root(), module=_MODULE)
    context, runner = prepared.context, prepared.runner
    labels = cast("list[str] | None", args.entries)
    if labels:
        entries = smoke_entries(context.manifest, count=len(labels), labels=labels)
    elif cast("str", args.subset) == "budget":
        entries = budget_entries(context.manifest, parent=cast("str", args.parent))
    else:
        entries = smoke_entries(context.manifest, count=int(cast("int", args.models)))
    for entry in entries:
        runner.evaluate(entry, warmup_s=context.inputs.configuration(entry).warmup_s)
    written = runner.write_pointers(Path(cast("str", args.evidence_dir)))
    runs = runner.run_timings
    models = tuple(runner.model_timings[entry.label] for entry in entries)
    banks = sum(1 for pointer in runner.pointers if pointer.kind == "replay")
    pairs_per_model = len(runner.scenarios) * len(runner.trackers)
    projection = project_study(models, runs, pairs_per_model=pairs_per_model, completed_models=len(entries))
    wall = time.perf_counter() - started
    own, children = peak_rss_bytes()
    estimate = (
        f"Measured {len(models)} of {projection.models} models and {banks} replay bank(s) in {_hours(wall)} on "
        f"{pairs_per_model} pairs per model. Scaling those means to the whole study projects "
        f"{_hours(projection.total_seconds)} of serial simulation and about {_gib(projection.storage_bytes)} of "
        f"run data for {projection.total_runs:,} runs, beside the 2026-09-15 planning estimate of about 13 h and "
        f"26-27 GB rescaled on 2026-09-17 for the revised replay count. Bounded parallel execution reduces "
        f"elapsed time and not storage. This is a projection from a subset, not a guaranteed bound, and "
        f"M3MAN-010 waits for the owner's budget approval."
    )
    report = ManualTimingReport(
        experiment=EXPERIMENT_LABEL,
        study_manifest_sha256=sha256_file(Path(cast("str", args.study))),
        evaluation_sha256=sha256_file(Path(cast("str", args.evaluation))),
        entries=tuple(entry.label for entry in entries),
        execution=prepared.execution,
        models=models,
        runs=runs,
        run_stats=summarize_timings(runs),
        runs_this_invocation=len(runs),
        replay_banks_built=banks,
        wall_seconds=wall,
        peak_rss_bytes=own,
        peak_rss_children_bytes=children,
        storage_bytes=sum(run.run_bytes for run in runs) + runner.manifest_bytes,
        projection=projection,
        revised_estimate=estimate,
        provenance=runner.provenance,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(timing_to_json(report) + "\n", encoding="utf-8")
    markdown.write_text(render_timing_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "models": len(models),
                "runs": len(runs),
                "replay_banks": banks,
                "wall_seconds": round(wall, 1),
                "projected_total_hours": round(projection.total_seconds / 3600.0, 2),
                "projected_storage_gib": round(projection.storage_bytes / 2**30, 1),
                "pointers_written": len(written),
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Timing smoke check of the manual-demonstration study.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    smoke = subparsers.add_parser("smoke", help="measure a deterministic subset and project the whole study")
    smoke.add_argument("--study", type=str, required=True, help="frozen study manifest JSON")
    smoke.add_argument("--evaluation", type=str, required=True, help="manual evaluation config TOML")
    smoke.add_argument("--evidence-dir", type=str, required=True, help="directory of the Git pointer records")
    smoke.add_argument("--output", type=str, required=True, help="timing report JSON to write (must not exist)")
    smoke.add_argument("--markdown", type=str, required=True, help="timing Markdown to write (must not exist)")
    smoke.add_argument(
        "--subset",
        choices=("budget", "prefix"),
        default="budget",
        help="budget: every configuration crossed with every arm kind at one parent (the frozen subset); "
        "prefix: the first --models entries, for a quick implementation check",
    )
    smoke.add_argument("--parent", type=str, default=BUDGET_PARENT, help="the budget subset's fixed parent")
    smoke.add_argument("--models", type=int, default=3, help="models to measure when --subset prefix")
    smoke.add_argument("--entries", type=str, nargs="*", default=None, help="measure these labels instead")
    smoke.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    args.argv = argv
    return _smoke(args)


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
