# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MS-007: derive the machine-readable evidence of the five-arm comparison at the frozen configurations.

The derivation is the closed experiment's own, run over another scope. The
scope comes from the verified freeze (via `ComparisonContext`), not from a
table. It holds the 93 learned models at the three frozen configurations and
the 30 replay banks their conditions key. Every model and bank is read through
its pointer, verified by digest and checked the way a resume checks it. Every
run becomes a measured row (`derive_rows`), and the contrasts, per-parent
contrast summaries and arm summaries are computed from those rows by the
closed experiment's own functions. No second implementation of any of this
exists.

What this experiment adds is its accounting:

* **completeness**: the planned 93 models, 30 banks and 15,990 runs, no run
  unavailable, and every comparison unit's recorded counts equal to the rows
  derived from its evidence;
* **reuse**: which replay banks are the closed experiment's own (identical
  conditions, so identical identities), which fits the comparison was served
  from the search's cache, and whether any nominal run of the search was
  reused (it was not: the search's runs are keyed by its nominal-only scope).

The owner excluded example cases from this task (M3MS-007 decision,
2026-09-24): the reporting assistant chooses any illustrations in M3MS-008,
labelled as post-hoc. The figure inputs here are the per-run and contrast
tables and the summaries, from which every aggregate plot can be rebuilt.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.experiments.manual_comparison_run import (
    MODEL,
    REPLAY,
    ComparisonContext,
    UnitOutcome,
    unit_directory,
)
from arm_rc_ctrl.experiments.manual_contrasts import (
    ManualArmSummary,
    ManualContrastRow,
    ManualContrastSummary,
    arm_summaries,
    contrast_rows,
    contrast_summaries,
)
from arm_rc_ctrl.experiments.manual_evaluation import load_manual_model_evidence, load_manual_pointer
from arm_rc_ctrl.experiments.manual_handoff import ReplayBankKey
from arm_rc_ctrl.experiments.manual_results import (
    ManualResultDocument,
    ManualResultTable,
    ManualRunRow,
    derive_rows,
    evidence_digest,
    load_scoped_evidence,
    store_table,
    table_columns,
    table_to_csv,
    verdicts_of,
)
from arm_rc_ctrl.experiments.manual_search import load_manual_search, protocol_digest
from arm_rc_ctrl.experiments.recovery_search import RECOVERY_TRACKERS
from arm_rc_ctrl.provenance import ProvenanceRecord, canonical_json, sha256_bytes, sha256_file, verify_artifact
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.experiments.manual_study import StudyModel
    from arm_rc_ctrl.storage import StorageRoot

__all__ = [
    "DOCUMENTS",
    "EXPERIMENT",
    "RESULTS_PREFIX",
    "RESULTS_VERSION",
    "SearchComparisonAccounting",
    "SearchComparisonResults",
    "account_comparison",
    "comparison_scope",
    "derive_search_results",
    "load_accounting",
    "load_search_results",
    "render_search_results",
    "stored_rows",
]

EXPERIMENT: Final = "task_1a_manual_esn_search"
RESULTS_VERSION: Final = 1
RESULTS_SCHEMA_VERSION: Final = 1
RESULTS_PREFIX: Final = "armrc://reports/task_1a_manual_search/results"
"""The comparison's stored tables, beside its unit records and apart from the closed experiment's."""
M3MAN_EVIDENCE: Final = "docs/experiments/task_1a_manual_demonstration/evidence"
"""The closed experiment's pointers: a bank named there too is its bank, served under identical conditions."""
DOCUMENTS: Final = {
    "accounting": f"accounting_v{RESULTS_VERSION}.json",
    "arm_summary": f"arm_summary_v{RESULTS_VERSION}.csv",
    "contrast_summary": f"contrast_summary_v{RESULTS_VERSION}.csv",
}
_SIMULATED: Final = ("completed", "infeasible")


def comparison_scope(context: ComparisonContext) -> tuple[tuple[StudyModel, ...], tuple[ReplayBankKey, ...]]:
    """The models and replay banks of the comparison, derived from the verified freeze in unit order."""
    entries = tuple(context.entry(unit) for unit in context.units if unit.kind == MODEL)
    keys: list[ReplayBankKey] = []
    for unit in context.units:
        if unit.kind != REPLAY:
            continue
        configuration = context.configurations[unit.rank]
        keys.append(
            ReplayBankKey(
                configuration=configuration.label,
                assignment=unit.label,
                warmup_s=configuration.warmup_s,
                velocity_cutoff_hz=configuration.velocity_cutoff_hz,
                acceleration_cutoff_hz=configuration.acceleration_cutoff_hz,
            )
        )
    return entries, tuple(keys)


# --- the accounting ------------------------------------------------------------------------------


@dataclass(frozen=True)
class PairCounts:
    """How a unit's pairs divide by verdict."""

    pairs: int
    completed: int
    infeasible: int
    unexecuted: int


@dataclass(frozen=True)
class UnitCounts:
    """One comparison unit's recorded counts beside those its derived rows give."""

    number: int
    label: str
    recorded: PairCounts
    """As the unit's finalized outcome records them."""
    derived: PairCounts


@dataclass(frozen=True)
class ReusedBank:
    """A replay bank the closed experiment already held under identical conditions."""

    configuration: str
    assignment: str
    identity: str
    pointer: str
    """The closed experiment's pointer that names the same bank."""


@dataclass(frozen=True)
class ReusedFit:
    """A comparison model whose fit is the search trial's own, served from the fit cache."""

    configuration: str
    arm: str
    trial: int
    fit_identity: str


@dataclass(frozen=True)
class SearchComparisonAccounting:
    """Completeness against the plan and the reuse the comparison actually made."""

    expected_models: int
    expected_replay_banks: int
    expected_rc_runs: int
    expected_replay_runs: int
    n_models: int
    n_replay_banks: int
    n_rc_runs: int
    n_replay_runs: int
    n_unavailable_runs: int
    units: tuple[UnitCounts, ...]
    reused_banks: tuple[ReusedBank, ...]
    reused_fits: tuple[ReusedFit, ...]
    search_nominal_runs: int
    """Nominal runs the frozen trials' search evidence holds."""
    reused_search_runs: int
    """Of those, runs the comparison's rows name: none, as the search keyed its runs by its nominal-only scope."""

    @property
    def complete(self) -> bool:
        """Every planned model, bank and run is present and simulated, and every unit agrees with its rows."""
        return (
            (self.n_models, self.n_replay_banks, self.n_rc_runs, self.n_replay_runs)
            == (self.expected_models, self.expected_replay_banks, self.expected_rc_runs, self.expected_replay_runs)
            and self.n_unavailable_runs == 0
            and all(unit.recorded == unit.derived for unit in self.units)
        )


def _counts(rows: Sequence[ManualRunRow]) -> PairCounts:
    return PairCounts(
        pairs=len(rows),
        completed=sum(1 for row in rows if row.status == "completed"),
        infeasible=sum(1 for row in rows if row.status == "infeasible"),
        unexecuted=sum(1 for row in rows if row.status == "unexecuted"),
    )


def account_comparison(
    context: ComparisonContext,
    rows: Sequence[ManualRunRow],
    *,
    n_models: int,
    n_banks: int,
    root: Path,
) -> SearchComparisonAccounting:
    """Count the derived rows against the plan and against every unit's outcome, and find what was reused."""
    by_label: dict[str, list[ManualRunRow]] = {}
    for row in rows:
        by_label.setdefault(row.model_label, []).append(row)
    entries = {unit.number: context.entry(unit) for unit in context.units if unit.kind == MODEL}
    units: list[UnitCounts] = []
    for unit in context.units:
        configuration = context.configurations[unit.rank].label
        label = entries[unit.number].label if unit.kind == MODEL else f"{configuration}/replay/{unit.label}"
        outcome = from_mapping(
            json.loads((unit_directory(context.store, unit) / "outcome.json").read_text(encoding="utf-8")),
            UnitOutcome,
        )
        units.append(
            UnitCounts(
                number=unit.number,
                label=label,
                recorded=PairCounts(
                    pairs=outcome.pairs,
                    completed=outcome.completed,
                    infeasible=outcome.infeasible,
                    unexecuted=outcome.unexecuted,
                ),
                derived=_counts(by_label.get(label, [])),
            )
        )
    closed = {
        load_manual_pointer(path).identity: path.name for path in sorted((root / M3MAN_EVIDENCE).glob("replay__*.toml"))
    }
    _, keys = comparison_scope(context)
    reused_banks: list[ReusedBank] = []
    for key in keys:
        identity = _bank_identity(context, key)
        if identity in closed:
            reused_banks.append(
                ReusedBank(
                    configuration=key.configuration,
                    assignment=key.assignment,
                    identity=identity,
                    pointer=closed[identity],
                )
            )
    chosen = {item.configuration: item for item in context.freeze.chosen}
    reused_fits = [
        ReusedFit(
            configuration=entry.configuration,
            arm=entry.arm.label,
            trial=chosen[entry.configuration].trial,
            fit_identity=entry.fit_identity,
        )
        for entry in entries.values()
        if entry.fit_identity == chosen[entry.configuration].fit_identity
    ]
    search_runs: set[str] = set()
    for item in context.freeze.chosen:
        evidence = load_manual_model_evidence(context.store.path(item.evidence, mode="read"))
        search_runs |= {pair.run.artifact_id for pair in evidence.pairs if pair.run is not None}
    compared = {row.run_artifact_id for row in rows if row.run_artifact_id is not None}
    rc = [row for row in rows if row.source == "rc" and row.status in _SIMULATED]
    replay = [row for row in rows if row.source == "replay" and row.status in _SIMULATED]
    per_case = len(context.conditions(min(context.configurations)).pairs)
    return SearchComparisonAccounting(
        expected_models=len(entries),
        expected_replay_banks=len(keys),
        expected_rc_runs=len(entries) * per_case,
        expected_replay_runs=len(keys) * per_case,
        n_models=n_models,
        n_replay_banks=n_banks,
        n_rc_runs=len(rc),
        n_replay_runs=len(replay),
        n_unavailable_runs=sum(1 for row in rows if row.status not in _SIMULATED),
        units=tuple(units),
        reused_banks=tuple(reused_banks),
        reused_fits=tuple(reused_fits),
        search_nominal_runs=len(search_runs),
        reused_search_runs=len(search_runs & compared),
    )


def _bank_identity(context: ComparisonContext, key: ReplayBankKey) -> str:
    conditions = context.runner.conditions(key.warmup_s, (key.velocity_cutoff_hz, key.acceleration_cutoff_hz))
    return sha256_bytes(f"{conditions.identity}:{key.assignment}".encode("ascii"))


# --- the index ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResultInputs:
    """What the derivation read, bound by digest."""

    protocol_sha256: str
    freeze_sha256: str
    study_manifest_sha256: str
    evaluation_sha256: str
    status_sha256: str
    evidence_sha256: str
    n_pointers: int


@dataclass(frozen=True)
class SearchComparisonResults:
    """The index of the comparison's derived evidence: what was read, what was written, and the totals."""

    experiment: str
    version: int
    inputs: SearchResultInputs
    tables: tuple[ManualResultTable, ...]
    documents: tuple[ManualResultDocument, ...]
    n_models: int
    n_replay_banks: int
    n_rc_runs: int
    n_replay_runs: int
    n_unavailable_runs: int
    n_rc_successes: int
    n_replay_successes: int
    complete: bool
    departure_radius_m: float
    command: str
    provenance: ProvenanceRecord
    schema_version: int = RESULTS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """The index names this experiment and its own schema, and never more successes than runs."""
        if (self.experiment, self.schema_version) != (EXPERIMENT, RESULTS_SCHEMA_VERSION):
            msg = f"unsupported results index {self.schema_version} of {self.experiment!r}"
            raise ValueError(msg)
        if self.n_rc_successes > self.n_rc_runs or self.n_replay_successes > self.n_replay_runs:
            msg = "more successes than runs"
            raise ValueError(msg)


def load_search_results(path: Path) -> SearchComparisonResults:
    """Strictly rebuild the index from its JSON."""
    return from_mapping(
        cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), SearchComparisonResults
    )


def render_search_results(results: SearchComparisonResults, accounting: SearchComparisonAccounting) -> str:
    """The index as Markdown: the bound inputs, every output with its digest, the totals and the reuse."""
    inputs = results.inputs
    agree = "equal" if all(u.recorded == u.derived for u in accounting.units) else "do NOT all equal"
    shared = ", ".join(sorted({b.configuration for b in accounting.reused_banks})) or "none"
    lines = [
        "<!-- Generated by `python -m arm_rc_ctrl.experiments.manual_search_results derive`; do not edit. -->",
        "",
        f"# M3MS comparison: derived evidence (v{results.version})",
        "",
        (
            "The per-run table, the contrasts and the summaries of the five-arm comparison at the three frozen "
            "configurations, derived by the closed experiment's own functions from verified evidence."
        ),
        "",
        "| bound input | sha256 |",
        "| --- | --- |",
        f"| search protocol | `{inputs.protocol_sha256}` |",
        f"| verified freeze | `{inputs.freeze_sha256}` |",
        f"| study manifest | `{inputs.study_manifest_sha256}` |",
        f"| evaluation configuration | `{inputs.evaluation_sha256}` |",
        f"| comparison status | `{inputs.status_sha256}` |",
        f"| evidence pointers ({inputs.n_pointers}) | `{inputs.evidence_sha256}` |",
        "",
        "## Stored tables",
        "",
        "| table | record | rows | size (bytes) | sha256 | store location |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {t.name} | `{t.record}` | {t.n_rows} | {t.payload.size} | `{t.payload.sha256}` | `{t.payload.uri}` |"
        for t in results.tables
    ]
    lines += [
        "",
        "## Committed documents",
        "",
        "| file | record | entries | size (bytes) | sha256 |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [f"| `{d.name}` | `{d.record}` | {d.n_entries} | {d.size} | `{d.sha256}` |" for d in results.documents]
    lines += [
        "",
        "## Totals",
        "",
        "| quantity | derived | planned |",
        "| --- | ---: | ---: |",
        f"| models | {results.n_models} | {accounting.expected_models} |",
        f"| replay banks | {results.n_replay_banks} | {accounting.expected_replay_banks} |",
        f"| RC runs | {results.n_rc_runs:,} | {accounting.expected_rc_runs:,} |",
        f"| replay runs | {results.n_replay_runs:,} | {accounting.expected_replay_runs:,} |",
        f"| unavailable runs | {results.n_unavailable_runs} | 0 |",
        f"| RC runs that met every criterion | {results.n_rc_successes:,} | |",
        f"| replay runs that met every criterion | {results.n_replay_successes:,} | |",
        "",
        (
            f"Complete: {'yes' if results.complete else 'no'}. Every comparison unit's recorded counts "
            f"{agree} the rows derived from its evidence ({len(accounting.units)} units)."
        ),
        "",
        "## Reuse",
        "",
        (
            f"- Replay banks shared with the closed experiment (identical conditions, verified before serving): "
            f"{len(accounting.reused_banks)} ({shared})."
        ),
        (
            f"- Fits served from the search's cache: {len(accounting.reused_fits)} "
            f"({', '.join(f'{f.configuration}/{f.arm}' for f in accounting.reused_fits) or 'none'})."
        ),
        (
            f"- Nominal search runs reused: {accounting.reused_search_runs} of {accounting.search_nominal_runs}; the "
            "search keyed its runs by its nominal-only scope, so the comparison simulated its own."
        ),
        "",
        "Generated by:",
        "",
        "```text",
        results.command,
        "```",
        "",
    ]
    return "\n".join(lines)


# --- the derivation --------------------------------------------------------------------------------


def _refuse_existing(output: Path) -> None:
    names = [*DOCUMENTS.values(), f"results_v{RESULTS_VERSION}.json", f"results_v{RESULTS_VERSION}.md"]
    existing = [name for name in names if (output / name).exists()]
    if existing:
        msg = f"refusing to overwrite {existing}: derived evidence is versioned, so derive a new version beside it"
        raise FileExistsError(msg)


def _document(output: Path, name: str, record: str, text: str, entries: int) -> ManualResultDocument:
    data = text.encode("utf-8")
    (output / name).write_bytes(data)
    return ManualResultDocument(name=name, record=record, sha256=sha256_bytes(data), size=len(data), n_entries=entries)


def derive_search_results(
    context: ComparisonContext,
    *,
    evidence_dir: Path,
    status_file: Path,
    output: Path,
    root: Path,
    workers: int = 1,
) -> SearchComparisonResults:
    """Derive every output from the verified evidence, write it once, and return the index."""
    _refuse_existing(output)
    runner, store = context.runner, context.store
    entries, keys = comparison_scope(context)
    loaded = load_scoped_evidence(evidence_dir, store, entries, keys, runner)
    scenarios = tuple((case.scenario_id, str(case.kind)) for case in runner.scenarios)
    rows, radius_m = derive_rows(store, loaded, entries, scenarios, RECOVERY_TRACKERS, workers=workers)
    verdicts = verdicts_of(rows)
    contrasts = contrast_rows(verdicts, scenarios=scenarios)
    accounting = account_comparison(context, rows, n_models=len(loaded.models), n_banks=len(loaded.bank_of), root=root)
    tables = (
        store_table(
            store,
            f"runs_v{RESULTS_VERSION}",
            "ManualRunRow",
            table_to_csv(rows, ManualRunRow),
            len(rows),
            table_columns(ManualRunRow),
            prefix=RESULTS_PREFIX,
        ),
        store_table(
            store,
            f"contrasts_v{RESULTS_VERSION}",
            "ManualContrastRow",
            table_to_csv(contrasts, ManualContrastRow),
            len(contrasts),
            table_columns(ManualContrastRow),
            prefix=RESULTS_PREFIX,
        ),
    )
    output.mkdir(parents=True, exist_ok=True)
    arms = arm_summaries(verdicts, scenarios=scenarios)
    summaries = contrast_summaries(contrasts)
    documents = (
        _document(
            output,
            DOCUMENTS["accounting"],
            "SearchComparisonAccounting",
            canonical_json(to_mapping(accounting)) + "\n",
            len(accounting.units),
        ),
        _document(
            output, DOCUMENTS["arm_summary"], "ManualArmSummary", table_to_csv(arms, ManualArmSummary), len(arms)
        ),
        _document(
            output,
            DOCUMENTS["contrast_summary"],
            "ManualContrastSummary",
            table_to_csv(summaries, ManualContrastSummary),
            len(summaries),
        ),
    )
    evidence_sha256, n_pointers = evidence_digest(evidence_dir)
    rc = [row for row in rows if row.source == "rc" and row.status in _SIMULATED]
    replay = [row for row in rows if row.source == "replay" and row.status in _SIMULATED]
    results = SearchComparisonResults(
        experiment=EXPERIMENT,
        version=RESULTS_VERSION,
        inputs=SearchResultInputs(
            protocol_sha256=protocol_digest(context.protocol),
            freeze_sha256=context.freeze_sha256,
            study_manifest_sha256=sha256_file(context.protocol.study),
            evaluation_sha256=sha256_file(context.protocol.comparison.evaluation),
            status_sha256=sha256_file(status_file),
            evidence_sha256=evidence_sha256,
            n_pointers=n_pointers,
        ),
        tables=tables,
        documents=documents,
        n_models=len(loaded.models),
        n_replay_banks=len(loaded.bank_of),
        n_rc_runs=len(rc),
        n_replay_runs=len(replay),
        n_unavailable_runs=accounting.n_unavailable_runs,
        n_rc_successes=sum(1 for row in rc if row.success),
        n_replay_successes=sum(1 for row in replay if row.success),
        complete=accounting.complete,
        departure_radius_m=radius_m,
        command=runner.command,
        provenance=runner.provenance,
    )
    (output / f"results_v{RESULTS_VERSION}.json").write_bytes(
        (canonical_json(to_mapping(results)) + "\n").encode("utf-8")
    )
    (output / f"results_v{RESULTS_VERSION}.md").write_bytes(render_search_results(results, accounting).encode("utf-8"))
    return results


def load_accounting(path: Path) -> SearchComparisonAccounting:
    """Strictly rebuild the accounting from its JSON."""
    return from_mapping(
        cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), SearchComparisonAccounting
    )


def stored_rows(store: StorageRoot, results: SearchComparisonResults, record: str) -> str:
    """One stored table's text, verified against the digest the index cites."""
    table = next((t for t in results.tables if t.record == record), None)
    if table is None:
        msg = f"the results index names no {record} table"
        raise ValueError(msg)
    return verify_artifact(store, table.payload).read_text(encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: derive the comparison's evidence (new files only)."""
    parser = argparse.ArgumentParser(description="Derive the M3MS comparison's machine-readable evidence.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    derive = subparsers.add_parser("derive", help="derive every output once from the verified evidence")
    derive.add_argument("--protocol", required=True, type=Path)
    derive.add_argument("--freeze", required=True, type=Path)
    derive.add_argument("--evidence-dir", required=True, type=Path)
    derive.add_argument("--status", required=True, type=Path, help="the committed comparison status")
    derive.add_argument("--output", required=True, type=Path)
    derive.add_argument("--workers", type=int, default=1)
    derive.add_argument("--exploratory", action="store_true", help="tolerate a dirty worktree (fixtures)")
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    protocol_file = cast("Path", args.protocol)
    root = repository_root()
    context = ComparisonContext(
        load_manual_search(protocol_file),
        protocol_file,
        cast("Path", args.freeze),
        root=root,
        exploratory=bool(args.exploratory),
        module="arm_rc_ctrl.experiments.manual_search_results",
        argv=argv,
    )
    results = derive_search_results(
        context,
        evidence_dir=cast("Path", args.evidence_dir),
        status_file=cast("Path", args.status),
        output=cast("Path", args.output),
        root=root,
        workers=int(cast("int", args.workers)),
    )
    print(
        json.dumps(
            {
                "complete": results.complete,
                "rc_runs": results.n_rc_runs,
                "replay_runs": results.n_replay_runs,
                "unavailable_runs": results.n_unavailable_runs,
                "tables": [t.payload.uri for t in results.tables],
            },
            indent=2,
        )
    )
    return 0 if results.complete else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
