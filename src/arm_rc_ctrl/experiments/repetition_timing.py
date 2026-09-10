# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Timing smoke check of the repeated-demonstration pilot (M3REP-005; repetition plan sections 9 and 10, C8, C10).

One complete source configuration (``feasible-best``, trial 17) is evaluated
across every applicable behavioral arm and count with the M3REP-004 runner in
the canonical execution environment, and the cost is measured rather than
guessed: wall time of every replay and RC run (simulation and persistence
apart), fit time of every model (from the fit cache record, whether the fit was
produced now or served), the process-cumulative peak resident set size, and
the bytes the runs and manifests occupy. The full-panel projection scales the
measured per-run and per-fit figures to the six entries, three warm-ups, and
every pair, as an upper bound that assumes no early stop, and the report states
the revised engineering estimate the owner's budget decision needs (C8). The
records this check produces belong to the fixed panel and are reused when the
full run resumes.

Command line (launch pinned through ``python -m arm_rc_ctrl.execution run --policy p-cores -- ...``)::

    python -m arm_rc_ctrl.experiments.repetition_timing smoke
        --manifest <docs>/panel_manifest_v1.json --evaluation configs/evaluations/task_1a_repetition_dev_v1.toml
        --validation <docs>/numerical_validation_v1.json --evidence-dir <docs>/evidence
        --output <docs>/timing_smoke_check_v1.json --markdown <docs>/timing_smoke_check_v1.md
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
from arm_rc_ctrl.experiments.repetition_evaluation import (
    MODEL_STATUSES,
    ModelEvidence,
    PilotRunner,
    RunTiming,
    prepare_runner,
)
from arm_rc_ctrl.experiments.repetition_panel import EXPERIMENT_LABEL
from arm_rc_ctrl.experiments.repetition_recipes import panel_arms
from arm_rc_ctrl.provenance import ProvenanceRecord, canonical_json
from arm_rc_ctrl.validation import is_hex

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.experiments.repetition_panel import PanelEntry

__all__ = [
    "PANEL_ENTRIES",
    "PANEL_WARMUPS",
    "TIMING_SCHEMA_VERSION",
    "ModelTiming",
    "PanelProjection",
    "TimingReport",
    "load_timing",
    "main",
    "peak_rss_bytes",
    "project_panel",
    "render_timing_markdown",
    "summarize_timings",
]

TIMING_SCHEMA_VERSION: Final = 1
PANEL_ENTRIES: Final = 6
PANEL_WARMUPS: Final = 3
_SHA256_HEX: Final = 64
_MODULE: Final = "arm_rc_ctrl.experiments.repetition_timing"


def peak_rss_bytes() -> tuple[int, int]:
    """Process-cumulative peak resident set size of this process and of its waited-for children (bytes)."""
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return int(own) * 1024, int(children) * 1024  # Linux reports KiB


@dataclass(frozen=True)
class ModelTiming:
    """The measured cost of one model configuration in this invocation."""

    label: str
    status: str
    fit_identity: str | None
    fit_cache_hit: bool | None
    fit_seconds: float | None
    """Fit time as the fit cache recorded it (measured now for a miss, at its original fitting for a hit)."""
    sweep_seconds: float
    """Wall time of the model's sweep (runs, persistence, and bookkeeping), excluding its replay bank."""
    runs: int
    unexecuted: int
    run_bytes: int

    def __post_init__(self) -> None:
        """Statuses and counts are sane."""
        if self.status not in MODEL_STATUSES or self.sweep_seconds < 0 or self.runs < 0 or self.unexecuted < 0:
            msg = f"malformed model timing for {self.label!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class RunSummaryStats:
    """Per-run wall-time statistics of one arm (seconds)."""

    arm: str
    runs: int
    mean_simulate_s: float
    median_simulate_s: float
    max_simulate_s: float
    mean_persist_s: float
    mean_bytes: float


@dataclass(frozen=True)
class PanelProjection:
    """The full-panel cost scaled from the measured rates (an upper bound: no early stop assumed)."""

    entries: int
    models_per_entry: int
    pairs_per_model: int
    replay_banks: int
    rc_runs: int
    replay_runs: int
    rc_run_seconds: float
    replay_run_seconds: float
    fit_seconds: float
    total_seconds: float
    storage_bytes: int
    completed_models: int
    """Models whose evidence already exists after this smoke check (reused by the full run)."""
    remaining_seconds: float


@dataclass(frozen=True)
class TimingReport:
    """The committed smoke-check evidence (C8)."""

    experiment: str
    panel_label: str
    panel_manifest_sha256: str
    evaluation_sha256: str
    execution: ExecutionRecord
    models: tuple[ModelTiming, ...]
    runs: tuple[RunTiming, ...]
    """Every measured run of the replay bank and the models (a resumed run keeps its original measurement)."""
    run_stats: tuple[RunSummaryStats, ...]
    runs_this_invocation: int
    """Runs simulated by this invocation (the others were served from the store)."""
    replay_bank_seconds: float
    replay_bank_runs: int
    wall_seconds: float
    peak_rss_bytes: int
    peak_rss_children_bytes: int
    storage_bytes: int
    projection: PanelProjection
    revised_estimate: str
    provenance: ProvenanceRecord
    schema_version: int = field(default=TIMING_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Digests and counts are consistent."""
        if self.schema_version != TIMING_SCHEMA_VERSION or self.experiment != EXPERIMENT_LABEL:
            msg = "unsupported timing schema or experiment"
            raise ValueError(msg)
        for name in ("panel_manifest_sha256", "evaluation_sha256"):
            if not is_hex(getattr(self, name), _SHA256_HEX):
                msg = f"{name} must be 64 lowercase hex characters"
                raise ValueError(msg)
        if self.replay_bank_runs != sum(1 for r in self.runs if r.arm == "replay"):
            msg = "replay_bank_runs contradicts the run timings"
            raise ValueError(msg)
        if not self.revised_estimate.strip():
            msg = "the revised estimate must be stated"
            raise ValueError(msg)


def summarize_timings(runs: Sequence[RunTiming]) -> tuple[RunSummaryStats, ...]:
    """Per-arm statistics of the measured runs (empty arms are omitted)."""
    stats: list[RunSummaryStats] = []
    for arm in ("replay", "rc"):
        selected = [r for r in runs if r.arm == arm]
        if not selected:
            continue
        stats.append(
            RunSummaryStats(
                arm=arm,
                runs=len(selected),
                mean_simulate_s=float(mean(r.simulate_seconds for r in selected)),
                median_simulate_s=float(median(r.simulate_seconds for r in selected)),
                max_simulate_s=float(max(r.simulate_seconds for r in selected)),
                mean_persist_s=float(mean(r.persist_seconds for r in selected)),
                mean_bytes=float(mean(r.run_bytes for r in selected)),
            )
        )
    return tuple(stats)


def project_panel(
    models: Sequence[ModelTiming],
    runs: Sequence[RunTiming],
    *,
    models_per_entry: int,
    pairs_per_model: int,
    entries: int = PANEL_ENTRIES,
    replay_banks: int = PANEL_WARMUPS,
    completed_models: int,
) -> PanelProjection:
    """Scale the measured per-run and per-fit costs to the full panel (no early stop assumed)."""
    rc = [r for r in runs if r.arm == "rc"]
    replay = [r for r in runs if r.arm == "replay"]
    rc_seconds = mean(r.simulate_seconds + r.persist_seconds for r in rc) if rc else 0.0
    replay_seconds = mean(r.simulate_seconds + r.persist_seconds for r in replay) if replay else 0.0
    rc_bytes = mean(r.run_bytes for r in rc) if rc else 0.0
    replay_bytes = mean(r.run_bytes for r in replay) if replay else 0.0
    fit_seconds = [m.fit_seconds for m in models if m.fit_seconds is not None]
    fit_per_model = mean(fit_seconds) if fit_seconds else 0.0
    rc_runs = entries * models_per_entry * pairs_per_model
    replay_runs = replay_banks * pairs_per_model
    total = rc_runs * rc_seconds + replay_runs * replay_seconds + entries * models_per_entry * fit_per_model
    per_model = pairs_per_model * rc_seconds + fit_per_model
    remaining = max(0.0, total - completed_models * per_model - len(replay) * replay_seconds)
    return PanelProjection(
        entries=entries,
        models_per_entry=models_per_entry,
        pairs_per_model=pairs_per_model,
        replay_banks=replay_banks,
        rc_runs=rc_runs,
        replay_runs=replay_runs,
        rc_run_seconds=rc_seconds,
        replay_run_seconds=replay_seconds,
        fit_seconds=entries * models_per_entry * fit_per_model,
        total_seconds=total,
        storage_bytes=int(rc_runs * rc_bytes + replay_runs * replay_bytes),
        completed_models=completed_models,
        remaining_seconds=remaining,
    )


def _fit_source(model: ModelTiming) -> str:
    if model.fit_cache_hit is None:
        return "failed"
    return "cache hit" if model.fit_cache_hit else "fitted now"


def _hours(seconds: float) -> str:
    return f"{seconds / 3600.0:.2f} h"


def _mib(size: float) -> str:
    return f"{size / 2**20:.1f} MiB"


def render_timing_markdown(report: TimingReport) -> str:
    """The Markdown rendering of the smoke check."""
    p = report.projection
    lines = [
        "# Task 1-a repetition timing smoke check (v1)",
        "",
        (
            f"Experiment `{report.experiment}`, panel entry `{report.panel_label}` (manifest sha256 "
            f"`{report.panel_manifest_sha256[:12]}`, evaluation config sha256 `{report.evaluation_sha256[:12]}`), "
            f"execution identity `{report.execution.identity[:12]}` "
            f"({'canonical' if report.execution.canonical else 'NOT canonical'}), project commit "
            f"`{report.provenance.project_commit[:12]}`{' (dirty)' if report.provenance.project_dirty else ''}."
        ),
        "",
        "## Measured cost",
        "",
        (
            f"- Wall time of this invocation: {_hours(report.wall_seconds)} ({report.wall_seconds:.0f} s); "
            f"{report.runs_this_invocation} of {len(report.runs)} measured runs were simulated by it."
        ),
        (
            f"- Replay bank: {report.replay_bank_runs} runs in {report.replay_bank_seconds:.0f} s; models: "
            f"{len(report.models)} ({sum(m.runs for m in report.models)} RC runs, "
            f"{sum(m.unexecuted for m in report.models)} unexecuted pairs)."
        ),
        (
            f"- Peak resident set size (process-cumulative): {_mib(report.peak_rss_bytes)}; waited-for children: "
            f"{_mib(report.peak_rss_children_bytes)}."
        ),
        f"- Storage of the measured runs and this invocation's manifests: {_mib(report.storage_bytes)}.",
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
        "| model | status | fit | fit s | sweep s | RC runs | unexecuted | run bytes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {m.label} | {m.status} | {_fit_source(m)} | {'' if m.fit_seconds is None else f'{m.fit_seconds:.2f}'} "
        f"| {m.sweep_seconds:.1f} | {m.runs} | {m.unexecuted} | {m.run_bytes} |"
        for m in report.models
    )
    lines += [
        "",
        "## Full-panel projection (upper bound, no early stop)",
        "",
        (
            f"- {p.entries} entries x {p.models_per_entry} models x {p.pairs_per_model} pairs = {p.rc_runs} RC runs at "
            f"{p.rc_run_seconds:.2f} s each; {p.replay_banks} replay banks x {p.pairs_per_model} = {p.replay_runs} "
            f"replay runs at {p.replay_run_seconds:.2f} s each; fits {_hours(p.fit_seconds)}."
        ),
        f"- Total: {_hours(p.total_seconds)}; storage about {_mib(p.storage_bytes)}.",
        (
            f"- Already complete after this check: {p.completed_models} models and this replay bank; remaining "
            f"about {_hours(p.remaining_seconds)}."
        ),
        "",
        "## Revised engineering estimate",
        "",
        report.revised_estimate,
        "",
        "## Limitations",
        "",
        (
            "- The projection assumes every pair of every model executes; real sweeps stop at their first "
            "infeasible pair, so the bound is loose on the failure side of the panel."
        ),
        (
            "- Timings are wall-clock in the canonical single-threaded execution environment of this machine "
            "(C10); another core type, thread setting, or machine measures differently."
        ),
        "",
    ]
    return "\n".join(lines)


def timing_to_json(report: TimingReport) -> str:
    """Canonical JSON."""
    return canonical_json(to_mapping(report))


def load_timing(path: Path) -> TimingReport:
    """Strictly rebuild a timing report from JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), TimingReport)


def _model_timings(runner: PilotRunner, evidences: Sequence[ModelEvidence]) -> tuple[ModelTiming, ...]:
    timings: list[ModelTiming] = []
    for evidence in evidences:
        label = (
            f"{evidence.fit.panel_label}/{evidence.fit.arm.label}" if evidence.fit is not None else "training-failure"
        )
        sweep = runner.model_timings.get(evidence.evaluation_identity)
        runs = [p for p in evidence.pairs if p.run is not None]
        timings.append(
            ModelTiming(
                label=label,
                status=evidence.status,
                fit_identity=None if evidence.fit is None else evidence.fit.identity,
                fit_cache_hit=None if sweep is None else sweep.fit_cache_hit,
                fit_seconds=None if sweep is None else sweep.fit_seconds,
                sweep_seconds=0.0 if sweep is None else sweep.sweep_seconds,
                runs=len(runs),
                unexecuted=evidence.n_unexecuted,
                run_bytes=sum(p.run.size for p in runs if p.run is not None),
            )
        )
    return tuple(timings)


def _smoke(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    argv = cast("list[str]", args.argv)
    prepared = prepare_runner(args, module=_MODULE, argv=argv)
    runner, context, entries, arms = prepared.runner, prepared.context, prepared.entries, prepared.arms
    if len(entries) != 1:
        msg = "the smoke check evaluates exactly one panel entry (feasible-best, trial 17)"
        raise ValueError(msg)
    entry: PanelEntry = entries[0]
    bank_started = time.perf_counter()
    bank = runner.replay_bank(entry)
    bank_seconds = time.perf_counter() - bank_started
    evidences = runner.run([entry], arms)
    written = runner.write_pointers(Path(cast("str", args.evidence_dir)))
    runs = tuple(p.timing for evidence in (bank, *evidences) for p in evidence.pairs if p.timing is not None)
    rss, rss_children = peak_rss_bytes()
    models = _model_timings(runner, evidences)
    behavioral = [arm for arm in panel_arms() if arm.behavioral]
    projection = project_panel(
        models,
        runs,
        models_per_entry=len(behavioral),
        pairs_per_model=len(bank.pairs),
        completed_models=len(evidences),
    )
    wall = time.perf_counter() - started
    estimate = (
        f"Measured on {entry.label}: {len(evidences)} models and one replay bank in {_hours(wall)}; scaled to the "
        f"six entries and three warm-ups the full panel is at most {_hours(projection.total_seconds)} of wall time "
        f"and about {_mib(projection.storage_bytes)} of run storage in the canonical execution environment, with "
        f"about {_hours(projection.remaining_seconds)} remaining after this check; the earlier engineering estimate "
        "(plan section 10) stands or is revised accordingly, and M3REP-006 waits for the owner's budget approval (C8)."
    )
    report = TimingReport(
        experiment=EXPERIMENT_LABEL,
        panel_label=entry.label,
        panel_manifest_sha256=context.manifest_sha256,
        evaluation_sha256=runner.conditions(entry).replay.evaluation_sha256,
        execution=runner.execution,
        models=models,
        runs=runs,
        run_stats=summarize_timings(runs),
        runs_this_invocation=len(runner.run_timings),
        replay_bank_seconds=bank_seconds,
        replay_bank_runs=sum(1 for r in runs if r.arm == "replay"),
        wall_seconds=wall,
        peak_rss_bytes=rss,
        peak_rss_children_bytes=rss_children,
        storage_bytes=sum(r.run_bytes for r in runs) + runner.manifest_bytes,
        projection=projection,
        revised_estimate=estimate,
        provenance=runner.provenance,
    )
    for target in (args.output, args.markdown):
        if Path(target).exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(timing_to_json(report) + "\n", encoding="utf-8")
    Path(args.markdown).write_text(render_timing_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "models": len(evidences),
                "wall_seconds": round(wall, 1),
                "projected_total_hours": round(projection.total_seconds / 3600.0, 2),
                "pointers_written": len(written),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Timing smoke check of the repeated-demonstration pilot.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    smoke = subparsers.add_parser("smoke", help="evaluate one entry across every behavioral arm and measure the cost")
    smoke.add_argument("--manifest", type=str, required=True, help="frozen panel manifest JSON")
    smoke.add_argument("--evaluation", type=str, required=True, help="pilot evaluation config TOML")
    smoke.add_argument("--validation", type=str, required=True, help="numerical validation JSON (C11 binding)")
    smoke.add_argument("--evidence-dir", type=str, required=True, help="directory of the Git pointer records")
    smoke.add_argument("--entries", type=str, nargs="*", default=["feasible-best"], help="the smoke-check entry")
    smoke.add_argument("--arms", type=str, nargs="*", default=None, help="behavioral arm labels (default: all)")
    smoke.add_argument("--output", type=str, required=True, help="timing report JSON to write (must not exist)")
    smoke.add_argument("--markdown", type=str, required=True, help="timing Markdown to write (must not exist)")
    smoke.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    args.argv = argv
    return _smoke(args)


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
