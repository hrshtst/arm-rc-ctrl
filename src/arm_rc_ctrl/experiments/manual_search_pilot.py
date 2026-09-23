# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MS-004: the bounded timing pilot's preflight, its observed counts and its cost report.

The pilot is the approved search stopped early (``search --stop-at-trials``),
so its trials are the search's own and consume the 100-trial cap. This module
adds the three things the pilot is for:

* a **preflight**, written before the pilot runs, stating what it will
  schedule: trials, the most nominal runs they can take, and what is spent;
* an **observation** of what the retained records, the derived ledger and the
  Optuna study hold afterwards, and the disagreements between the two;
* a **report** dividing each trial's time into what the plan's estimates
  count (simulation) and what they omit (preparing, fitting, persisting,
  worker start-up and the parent's verification), with the parent overhead and
  the bytes the ledger does not charge, and projections for the rest of the
  search and for the comparison that shares its ceiling.

Projections are estimates from a handful of trials, not bounds: the ledger and
its caps remain what stops the work.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from optuna.trial import TrialState

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.experiments.manual_sampled import SampledPoint
from arm_rc_ctrl.experiments.manual_search import NOMINAL_SCENARIOS, REQUIRED_RUNS, load_manual_search, protocol_digest
from arm_rc_ctrl.experiments.manual_search_run import (
    INVOCATIONS_PREFIX,
    STUDY_NAME,
    BudgetLedger,
    SearchInvocation,
    TrialOutcome,
    TrialReservation,
    invocation_records,
    ledger_of,
    pending_reservations,
    pilot_bound_mismatches,
    read_record,
    read_result,
    trial_directories,
)
from arm_rc_ctrl.experiments.studies import PrunerSpec, close_study, open_study, study_uri
from arm_rc_ctrl.provenance import canonical_json
from arm_rc_ctrl.storage import ArtifactUri, StorageRoot, open_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.experiments.manual_search import ManualSearchProtocol

__all__ = [
    "PilotProjection",
    "PilotReport",
    "PilotTrialRow",
    "PilotUncharged",
    "SearchObservation",
    "SearchPreflight",
    "count_disagreements",
    "observe_search",
    "pilot_projection",
    "pilot_report",
    "read_preflight",
    "render_markdown",
    "report_json",
    "search_preflight",
    "write_preflight",
]

_GIB: Final = 1024**3
COMPARISON_RC_RUNS: Final = 12_090
"""Three configurations x 31 learned models x 65 cases x 2 trackers (plan section 5)."""
COMPARISON_REPLAY_RUNS: Final = 3_900
"""Three configurations x 10 parent banks x 65 cases x 2 trackers, before verified reuse."""
COMPARISON_MODELS: Final = 93
"""Three configurations x 31 learned models."""
PLAN_REPLAY_TO_RC: Final = 1.42 / 1.37
"""The measured replay-to-RC cost ratio the plan cites; the pilot runs no replay, so it is carried over."""


@dataclass(frozen=True)
class SearchPreflight:
    """What a search invocation will schedule, stated before it runs."""

    protocol_sha256: str
    scenarios: tuple[str, ...]
    runs_per_trial: int
    trial_cap: int
    stop_at_trials: int | None
    spent_trials: int
    spent_runs: int
    pending_trials: int
    """Reserved trials without an outcome: finished first, and counted among those to schedule."""
    trials_to_schedule: int
    max_rc_runs: int
    hours_cap: float
    gib_cap: float
    spent_seconds: float
    spent_bytes: int


def _outcomes(store: StorageRoot) -> list[TrialOutcome]:
    return [
        read_record(directory / "outcome.json", TrialOutcome)
        for directory in trial_directories(store)
        if (directory / "outcome.json").is_file()
    ]


def search_preflight(
    protocol: ManualSearchProtocol, store: StorageRoot, *, stop_at_trials: int | None
) -> SearchPreflight:
    """The trials and runs a search with this bound would schedule from the store as it is now."""
    mismatches = pilot_bound_mismatches(stop_at_trials, protocol.budget)
    if mismatches:
        raise ValueError("; ".join(mismatches))
    ledger = ledger_of(store)
    target = protocol.budget.trials if stop_at_trials is None else stop_at_trials
    to_schedule = max(0, target - ledger.trials)
    return SearchPreflight(
        protocol_sha256=protocol_digest(protocol),
        scenarios=protocol.objective.scenarios,
        runs_per_trial=REQUIRED_RUNS,
        trial_cap=protocol.budget.trials,
        stop_at_trials=stop_at_trials,
        spent_trials=ledger.trials,
        spent_runs=sum(outcome.runs for outcome in _outcomes(store)),
        pending_trials=len(pending_reservations(store)),
        trials_to_schedule=to_schedule,
        max_rc_runs=REQUIRED_RUNS * to_schedule,
        hours_cap=protocol.budget.hours,
        gib_cap=protocol.budget.gib,
        spent_seconds=ledger.seconds,
        spent_bytes=ledger.stored_bytes,
    )


def write_preflight(path: Path, preflight: SearchPreflight) -> None:
    """Write a preflight once; a stated plan is versioned, never rewritten."""
    if path.exists():
        msg = f"{path} already holds a preflight; a stated plan is never rewritten"
        raise FileExistsError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(to_mapping(preflight)) + "\n", encoding="utf-8")


def read_preflight(path: Path) -> SearchPreflight:
    """Read a preflight strictly."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), SearchPreflight)


@dataclass(frozen=True)
class SearchObservation:
    """What the retained records, the derived ledger and the Optuna study hold."""

    protocol_sha256: str
    finalized: int
    scored: int
    failed: int
    pending: int
    runs: int
    """Nominal runs held by scored trials."""
    failed_runs: int
    """Runs a failed candidate reported before it failed."""
    run_statuses: dict[str, int]
    scenarios: tuple[str, ...]
    """Every scenario a finalized trial's worker reported evaluating."""
    ledger_trials: int
    study_states: dict[str, int]


def observe_search(protocol: ManualSearchProtocol, store: StorageRoot) -> SearchObservation:
    """Count the search as it stands, from each of the places that record it."""
    outcomes = _outcomes(store)
    statuses: Counter[str] = Counter()
    scenarios: dict[str, None] = {}
    for directory in trial_directories(store):
        result = directory / "result.json"
        if (directory / "outcome.json").is_file() and result.is_file():
            report = read_result(result)
            statuses.update(report.statuses)
            scenarios.update(dict.fromkeys(report.scenarios))
    study = open_study(
        store,
        STUDY_NAME,
        protocol_sha256=protocol_digest(protocol),
        sampler=protocol.sampler,
        pruner=PrunerSpec(kind="none"),
        direction="maximize",
    )
    try:
        states = Counter(trial.state.name for trial in study.trials)
    finally:
        close_study(study)
    return SearchObservation(
        protocol_sha256=protocol_digest(protocol),
        finalized=len(outcomes),
        scored=sum(1 for outcome in outcomes if outcome.state == "scored"),
        failed=sum(1 for outcome in outcomes if outcome.state == "failed"),
        pending=len(pending_reservations(store)),
        runs=sum(outcome.runs for outcome in outcomes if outcome.state == "scored"),
        failed_runs=sum(outcome.runs for outcome in outcomes if outcome.state == "failed"),
        run_statuses=dict(sorted(statuses.items())),
        scenarios=tuple(scenarios),
        ledger_trials=ledger_of(store).trials,
        study_states=dict(sorted(states.items())),
    )


def count_disagreements(preflight: SearchPreflight, observation: SearchObservation) -> list[str]:
    """Every way the observed search differs from its preflight or from itself; empty when they agree."""
    found: list[str] = []
    if preflight.protocol_sha256 != observation.protocol_sha256:
        found.append(
            f"the preflight states protocol {preflight.protocol_sha256[:12]}, the search ran "
            f"{observation.protocol_sha256[:12]}"
        )
    planned = preflight.spent_trials + preflight.trials_to_schedule
    if observation.finalized != planned:
        found.append(f"the preflight planned {planned} finalized trials, the records hold {observation.finalized}")
    if observation.pending:
        found.append(f"{observation.pending} reserved trials are still pending")
    if observation.ledger_trials != observation.finalized:
        found.append(f"the ledger counts {observation.ledger_trials} trials, the records {observation.finalized}")
    expected_states = {
        state: count
        for state, count in (
            (TrialState.COMPLETE.name, observation.scored),
            (TrialState.FAIL.name, observation.failed),
            (TrialState.RUNNING.name, observation.pending),
        )
        if count
    }
    if observation.study_states != expected_states:
        found.append(f"the study holds {observation.study_states}, the records imply {expected_states}")
    if observation.runs != REQUIRED_RUNS * observation.scored:
        found.append(f"{observation.scored} scored trials hold {observation.runs} runs, not {REQUIRED_RUNS} each")
    new_runs = observation.runs + observation.failed_runs - preflight.spent_runs
    if new_runs > preflight.max_rc_runs:
        found.append(f"{new_runs} runs were taken, above the preflight's {preflight.max_rc_runs}")
    if sum(observation.run_statuses.values()) != observation.runs + observation.failed_runs:
        found.append(f"the reports give {observation.run_statuses} for {observation.runs} runs")
    if observation.scenarios not in ((), NOMINAL_SCENARIOS) or preflight.scenarios != NOMINAL_SCENARIOS:
        found.append(f"the search evaluated {observation.scenarios}, not {NOMINAL_SCENARIOS} alone")
    return found


@dataclass(frozen=True)
class PilotTrialRow:
    """One finalized trial's point, verdict and where its time went."""

    trial: int
    point: SampledPoint | None
    """``None`` for a trial abandoned before any parameter was recorded: it has only an outcome."""
    state: str
    score: float | None
    runs: int
    charged_seconds: float
    """Every attempt plus the parent's verification: what the ledger counts."""
    worker_seconds: float | None
    verify_seconds: float | None
    prepare_seconds: float | None
    fit_seconds: float | None
    fit_cache_hit: bool | None
    sweep_seconds: float | None
    simulate_seconds: float | None
    persist_seconds: float | None
    simulated_runs: int | None
    """Runs this trial's worker actually simulated; runs served from the store cost nothing measured."""
    startup_seconds: float | None
    """Worker wall clock not inside preparing, fitting or the sweep: interpreter and imports."""
    stored_bytes: int
    run_bytes: int | None
    failure: str | None


def _row(directory: Path) -> PilotTrialRow | None:
    outcome_file = directory / "outcome.json"
    if not outcome_file.is_file():
        return None
    outcome = read_record(outcome_file, TrialOutcome)
    # Reconciliation retains a trial lost mid-sampling as an outcome alone, so a reservation is optional.
    reservation_file = directory / "reservation.json"
    point = read_record(reservation_file, TrialReservation).point if reservation_file.is_file() else None
    result_file = directory / "result.json"
    timing = read_result(result_file).timing if result_file.is_file() else None
    startup = None
    if timing is not None and outcome.worker_seconds is not None:
        inside = timing.prepare_seconds + timing.fit_seconds + timing.sweep_seconds
        startup = max(0.0, outcome.worker_seconds - inside)
    return PilotTrialRow(
        trial=outcome.trial,
        point=point,
        state=outcome.state,
        score=outcome.score,
        runs=outcome.runs,
        charged_seconds=outcome.seconds,
        worker_seconds=outcome.worker_seconds,
        verify_seconds=outcome.verify_seconds,
        prepare_seconds=None if timing is None else timing.prepare_seconds,
        fit_seconds=None if timing is None else timing.fit_seconds,
        fit_cache_hit=None if timing is None else timing.fit_cache_hit,
        sweep_seconds=None if timing is None else timing.sweep_seconds,
        simulate_seconds=None if timing is None else timing.simulate_seconds,
        persist_seconds=None if timing is None else timing.persist_seconds,
        simulated_runs=None if timing is None else timing.simulated_runs,
        startup_seconds=startup,
        stored_bytes=outcome.stored_bytes,
        run_bytes=None if timing is None else timing.run_bytes,
        failure=outcome.failure,
    )


@dataclass(frozen=True)
class PilotUncharged:
    """Elapsed time and bytes the per-trial ledger does not count."""

    invocation_overhead_seconds: float
    """Parent wall clock outside every charged trial: start-up, study bookkeeping, reconciliation."""
    incomplete_invocations: int
    record_bytes: int
    """The trial and invocation records themselves."""
    study_bytes: int
    """The Optuna study database."""


@dataclass(frozen=True)
class PilotProjection:
    """Estimates from the pilot for the rest of the search and the comparison sharing its ceiling."""

    search_trials: int
    seconds_per_trial: float
    search_seconds: float
    """The trial cap at the pilot's charged rate, plus the parent overhead measured so far once."""
    bytes_per_trial: float
    search_bytes: float
    rc_seconds_per_run: float | None
    """Simulating and persisting one nominal run, over the runs the pilot actually simulated."""
    fit_seconds_per_model: float | None
    bytes_per_run: float | None
    comparison_rc_runs: int
    comparison_replay_runs: int
    comparison_models: int
    comparison_seconds: float | None
    """Runs and fits alone at the pilot's nominal rates, replay scaled by the plan's measured ratio: an estimate.

    Perturbed runs can cost more or less than nominal ones, so this bounds nothing.
    """
    comparison_bytes: float | None
    ceiling_seconds: float
    ceiling_bytes: float
    remaining_seconds: float | None
    """The shared ceiling less the projected search and comparison; negative means it would not fit."""
    remaining_bytes: float | None


@dataclass(frozen=True)
class PilotReport:
    """The pilot's evidence: preflight against observation, per-trial time, the uncharged rest, projections."""

    protocol_sha256: str
    preflight: SearchPreflight
    observation: SearchObservation
    disagreements: tuple[str, ...]
    trials: tuple[PilotTrialRow, ...]
    ledger: BudgetLedger
    invocations: tuple[SearchInvocation, ...]
    invocation_seconds: float
    uncharged: PilotUncharged
    projection: PilotProjection


def _files_bytes(root: Path, pattern: str) -> int:
    return sum(path.stat().st_size for path in root.glob(pattern) if path.is_file()) if root.is_dir() else 0


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pilot_projection(
    protocol: ManualSearchProtocol, rows: Sequence[PilotTrialRow], ledger: BudgetLedger, overhead: float
) -> PilotProjection:
    """Project the search and the comparison from the pilot's charged spend and its measured runs.

    Per-run time and bytes are divided by the runs a worker actually
    simulated: a run served from the store was recorded but not measured, and
    counting it would make runs look cheaper than they are. With no measured
    run the cost is unavailable, never zero.
    """
    trials = max(ledger.trials, 1)
    per_trial = ledger.seconds / trials
    bytes_per_trial = ledger.stored_bytes / trials
    search_seconds = per_trial * protocol.budget.trials + overhead
    search_bytes = bytes_per_trial * protocol.budget.trials
    measured = [row for row in rows if row.simulated_runs]
    runs = sum(cast("int", row.simulated_runs) for row in measured)
    rc_per_run = (
        math.fsum(cast("float", row.simulate_seconds) + cast("float", row.persist_seconds) for row in measured) / runs
        if runs
        else None
    )
    bytes_per_run = sum(cast("int", row.run_bytes) for row in measured) / runs if runs else None
    fit = _mean([row.fit_seconds for row in rows if row.fit_seconds is not None and row.fit_seconds > 0.0])
    comparison_seconds = (
        None
        if rc_per_run is None or fit is None
        else rc_per_run * (COMPARISON_RC_RUNS + COMPARISON_REPLAY_RUNS * PLAN_REPLAY_TO_RC) + fit * COMPARISON_MODELS
    )
    comparison_bytes = None if bytes_per_run is None else bytes_per_run * (COMPARISON_RC_RUNS + COMPARISON_REPLAY_RUNS)
    ceiling_seconds, ceiling_bytes = protocol.budget.hours * 3600.0, protocol.budget.gib * _GIB
    return PilotProjection(
        search_trials=protocol.budget.trials,
        seconds_per_trial=per_trial,
        search_seconds=search_seconds,
        bytes_per_trial=bytes_per_trial,
        search_bytes=search_bytes,
        rc_seconds_per_run=rc_per_run,
        fit_seconds_per_model=fit,
        bytes_per_run=bytes_per_run,
        comparison_rc_runs=COMPARISON_RC_RUNS,
        comparison_replay_runs=COMPARISON_REPLAY_RUNS,
        comparison_models=COMPARISON_MODELS,
        comparison_seconds=comparison_seconds,
        comparison_bytes=comparison_bytes,
        ceiling_seconds=ceiling_seconds,
        ceiling_bytes=ceiling_bytes,
        remaining_seconds=None if comparison_seconds is None else ceiling_seconds - search_seconds - comparison_seconds,
        remaining_bytes=None if comparison_bytes is None else ceiling_bytes - search_bytes - comparison_bytes,
    )


def pilot_report(protocol: ManualSearchProtocol, store: StorageRoot, preflight: SearchPreflight) -> PilotReport:
    """Everything the pilot measured, compared against what its preflight stated."""
    observation = observe_search(protocol, store)
    rows = tuple(row for row in (_row(directory) for directory in trial_directories(store)) if row is not None)
    ledger = ledger_of(store)
    invocations = invocation_records(store)
    wall = math.fsum(record.seconds for record in invocations)
    overhead = max(0.0, wall - math.fsum(record.charged_seconds for record in invocations))
    trials_root = trial_directories(store)[0].parent if trial_directories(store) else None
    invocations_root = store.root / ArtifactUri.parse(f"{INVOCATIONS_PREFIX}/x.json").relative_path.parent
    uncharged = PilotUncharged(
        invocation_overhead_seconds=overhead,
        incomplete_invocations=sum(1 for record in invocations if not record.completed),
        record_bytes=(0 if trials_root is None else _files_bytes(trials_root, "trial-*/*.json"))
        + _files_bytes(invocations_root, "*.json"),
        study_bytes=_files_bytes(
            (store.root / ArtifactUri.parse(study_uri(STUDY_NAME)).relative_path).parent, f"{STUDY_NAME}.db*"
        ),
    )
    return PilotReport(
        protocol_sha256=protocol_digest(protocol),
        preflight=preflight,
        observation=observation,
        disagreements=tuple(count_disagreements(preflight, observation)),
        trials=rows,
        ledger=ledger,
        invocations=invocations,
        invocation_seconds=wall,
        uncharged=uncharged,
        projection=pilot_projection(protocol, rows, ledger, overhead),
    )


def report_json(report: PilotReport) -> str:
    """The report as canonical JSON, for committing beside the experiment."""
    return canonical_json(to_mapping(report)) + "\n"


def _s(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _hours(seconds: float | None) -> str:
    return "—" if seconds is None else f"{seconds / 3600.0:.2f} h"


def _gib(value: float | None) -> str:
    return "—" if value is None else f"{value / _GIB:.3f} GiB"


def _neurons(point: SampledPoint | None) -> str:
    return "—" if point is None else str(point.n_neurons)


def _warmup(point: SampledPoint | None) -> str:
    return "—" if point is None else f"{point.warmup_s:g}"


def render_markdown(report: PilotReport) -> str:
    """A human-readable account of the same report; the JSON is the evidence."""
    pre, obs, proj = report.preflight, report.observation, report.projection
    lines = [
        "<!-- Generated by `python -m arm_rc_ctrl.experiments.manual_search_pilot report`; do not edit. -->",
        "",
        "# M3MS-004 timing pilot",
        "",
        (
            f"Protocol `{report.protocol_sha256[:12]}`. The pilot is the approved search stopped at "
            f"{pre.stop_at_trials} trials: its trials consume the {pre.trial_cap}-trial cap and the search "
            "continues after them under the next numbers."
        ),
        "",
        "## Preflight against observation",
        "",
        "| Count | Preflight | Observed |",
        "| --- | ---: | ---: |",
        f"| Finalized trials | {pre.spent_trials + pre.trials_to_schedule} | {obs.finalized} |",
        f"| Pending trials | 0 | {obs.pending} |",
        f"| Scored / failed | — | {obs.scored} / {obs.failed} |",
        f"| Nominal RC runs | at most {pre.max_rc_runs} | {obs.runs + obs.failed_runs} |",
        f"| Ledger trials | — | {obs.ledger_trials} |",
        f"| Study states | — | {', '.join(f'{k} {v}' for k, v in obs.study_states.items()) or 'none'} |",
        f"| Scenarios | {', '.join(pre.scenarios)} | {', '.join(obs.scenarios) or 'none'} |",
        f"| Run statuses | — | {', '.join(f'{k} {v}' for k, v in obs.run_statuses.items()) or 'none'} |",
        "",
        "Disagreements: " + ("none." if not report.disagreements else "; ".join(report.disagreements)),
        "",
        "## Where each trial's time went (seconds)",
        "",
        (
            "Charged = every attempt's worker plus the parent's verification. The sweep encloses simulation and "
            "persistence; start-up is the worker's wall clock outside preparing, fitting and the sweep."
        ),
        "",
        (
            "| Trial | Neurons | Warm-up | State | Score | Charged | Worker | Start-up | Prepare | Fit | Sweep | "
            "Simulate | Persist | Verify | MiB |"
        ),
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {row.trial} | {_neurons(row.point)} | {_warmup(row.point)} | {row.state} | {_s(row.score, 1)} | "
        f"{_s(row.charged_seconds)} | {_s(row.worker_seconds)} | {_s(row.startup_seconds)} | "
        f"{_s(row.prepare_seconds)} | {_s(row.fit_seconds)} | {_s(row.sweep_seconds)} | "
        f"{_s(row.simulate_seconds)} | {_s(row.persist_seconds)} | {_s(row.verify_seconds)} | "
        f"{row.stored_bytes / 1024**2:.2f} |"
        for row in report.trials
    ]
    failures = [row for row in report.trials if row.failure is not None]
    if failures:
        lines += ["", "Failed candidates:", ""]
        lines += [f"- trial {row.trial}: {row.failure}" for row in failures]
    lines += [
        "",
        "## Spend",
        "",
        (
            f"- Ledger: {report.ledger.trials} trials, {report.ledger.seconds:.1f} s, "
            f"{report.ledger.stored_bytes / 1024**2:.2f} MiB."
        ),
        (
            f"- Parent invocations: {len(report.invocations)}, {report.invocation_seconds:.1f} s wall clock; "
            f"{report.uncharged.invocation_overhead_seconds:.1f} s of it outside every charged trial "
            f"({report.uncharged.incomplete_invocations} incomplete)."
        ),
        (
            f"- Not charged: {report.uncharged.record_bytes / 1024:.1f} KiB of trial/invocation records and "
            f"{report.uncharged.study_bytes / 1024:.1f} KiB of Optuna study database."
        ),
        "",
        "## Projections (estimates, not bounds)",
        "",
        (
            f"- Search: {proj.search_trials} trials at {proj.seconds_per_trial:.1f} s charged per trial, plus the "
            f"measured parent overhead once: {_hours(proj.search_seconds)}, {_gib(proj.search_bytes)}."
        ),
        (
            f"- Nominal run (simulate + persist): {_s(proj.rc_seconds_per_run)} s and "
            f"{_s(None if proj.bytes_per_run is None else proj.bytes_per_run / 1024**2)} MiB; fit "
            f"{_s(proj.fit_seconds_per_model)} s per model."
        ),
        (
            f"- Estimated runs-and-fits cost of the comparison: {proj.comparison_rc_runs:,} RC + "
            f"{proj.comparison_replay_runs:,} replay runs "
            f"(replay at the plan's {PLAN_REPLAY_TO_RC:.3f}x RC ratio) and {proj.comparison_models} fits: "
            f"{_hours(proj.comparison_seconds)}, {_gib(proj.comparison_bytes)}. It omits the comparison's own "
            "start-up, verification and reporting, and nominal runs do not bound the cost of perturbed ones."
        ),
        (
            f"- Shared ceiling {_hours(proj.ceiling_seconds)} / {_gib(proj.ceiling_bytes)}; left after both: "
            f"{_hours(proj.remaining_seconds)} / {_gib(proj.remaining_bytes)}."
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: state a preflight, or report a pilot against one."""
    parser = argparse.ArgumentParser(description="Preflight and report the manual ESN search's timing pilot.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    preflight = subparsers.add_parser("preflight", help="state what a bounded search will schedule")
    preflight.add_argument("--protocol", required=True, type=Path)
    preflight.add_argument("--stop-at-trials", type=int, default=None)
    preflight.add_argument("--output", required=True, type=Path, help="a new file; never overwritten")
    report = subparsers.add_parser("report", help="compare the search with its preflight and report its cost")
    report.add_argument("--protocol", required=True, type=Path)
    report.add_argument("--preflight", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path, help="the report as JSON")
    report.add_argument("--markdown", required=True, type=Path, help="the report rendered for reading")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    protocol = load_manual_search(cast("Path", args.protocol))
    store = open_storage()
    if args.subcommand == "preflight":
        stated = search_preflight(protocol, store, stop_at_trials=cast("int | None", args.stop_at_trials))
        write_preflight(cast("Path", args.output), stated)
        print(json.dumps(to_mapping(stated), indent=2))
        return 0
    built = pilot_report(protocol, store, read_preflight(cast("Path", args.preflight)))
    cast("Path", args.output).write_text(report_json(built), encoding="utf-8")
    cast("Path", args.markdown).write_text(render_markdown(built), encoding="utf-8")
    print("\n".join(built.disagreements) or "preflight and observation agree")
    return 1 if built.disagreements else 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
