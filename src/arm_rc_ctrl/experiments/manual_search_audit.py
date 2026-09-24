# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MS-007: audit the comparison's derived evidence against its sources, from a clean checkout.

Every step re-derives what it checks from trusted inputs and compares the whole
value; none compares a label or a stored digest with itself. A step that
cannot read what it needs is recorded as unavailable instead of ending the
audit, and every disagreement is retained in the record.

* ``sources``: the index's bound inputs, committed documents and stored
  tables, each against its digest;
* ``selection``: the freeze, rebuilt from the search's trial records with the
  chosen evidence verified again (`load_verified_freeze`), is the one the
  index binds;
* ``units``: every comparison unit's reservation and outcome against the
  verified freeze and its installed manifest (`check_finalized`), and the
  comparison status re-rendered byte for byte;
* ``manifests``: every model and bank manifest checked as a resume checks it,
  fits and replay pairing included (`verify_manifests`);
* ``payloads_and_metrics``: every stored run verified, judged again from its
  own arrays, and its row rebuilt and compared whole (`rebuild_rows`);
* ``aggregates``: the contrasts, contrast summaries and arm summaries rebuilt
  from the per-run table;
* ``completeness``: the accounting rebuilt and compared whole, against the
  planned 93 models, 30 banks and 15,990 runs.

The owner excluded re-simulation from this audit (M3MS-007 decision,
2026-09-24): stored runs are judged again from their own arrays, but none is
simulated again.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.experiments.manual_audit import EVIDENCE_ERRORS, AuditStep, rebuild_rows, verify_manifests
from arm_rc_ctrl.experiments.manual_comparison_run import (
    ComparisonContext,
    check_finalized,
    comparison_status,
    render_status,
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
from arm_rc_ctrl.experiments.manual_results import (
    ManualRunRow,
    evidence_digest,
    table_from_csv,
    verdicts_of,
)
from arm_rc_ctrl.experiments.manual_search import load_manual_search, protocol_digest
from arm_rc_ctrl.experiments.manual_search_results import (
    DOCUMENTS,
    EXPERIMENT,
    RESULTS_VERSION,
    SearchComparisonResults,
    account_comparison,
    comparison_scope,
    load_accounting,
    render_search_results,
    stored_rows,
)
from arm_rc_ctrl.provenance import ProvenanceRecord, canonical_json, sha256_bytes, sha256_file, verify_artifact
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["AUDIT_STEPS", "AUDIT_VERSION", "SearchComparisonAudit", "audit_search_comparison", "render_audit"]

AUDIT_VERSION: Final = 2
"""v1 trusted the index's headline figures and read it outside the recording boundary (M3MS-007 review)."""
AUDIT_SCHEMA_VERSION: Final = 1
AUDIT_STEPS: Final = (
    "sources",
    "selection",
    "units",
    "manifests",
    "payloads_and_metrics",
    "aggregates",
    "completeness",
)
_LIMIT: Final = 200
_SIMULATED: Final = ("completed", "infeasible")


@dataclass(frozen=True)
class SearchComparisonAudit:
    """The audit record: every step, what it covered and every disagreement it found."""

    experiment: str
    version: int
    results_sha256: str
    freeze_sha256: str
    steps: tuple[AuditStep, ...]
    passed: bool
    command: str
    provenance: ProvenanceRecord
    schema_version: int = AUDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Every step in order, and a verdict that is exactly every step passing."""
        if (self.experiment, self.schema_version) != (EXPERIMENT, AUDIT_SCHEMA_VERSION):
            msg = f"unsupported audit {self.schema_version} of {self.experiment!r}"
            raise ValueError(msg)
        if tuple(step.name for step in self.steps) != AUDIT_STEPS:
            msg = f"the audit runs {AUDIT_STEPS}, got {tuple(step.name for step in self.steps)}"
            raise ValueError(msg)
        if self.passed != all(step.ok for step in self.steps):
            msg = "an audit passes exactly when every step passes"
            raise ValueError(msg)


@dataclass(frozen=True)
class _Audited:
    context: ComparisonContext
    results: SearchComparisonResults | None
    """``None`` when the index could not be read; every step that needs it is then unavailable."""
    index_error: str | None
    results_dir: Path
    evidence_dir: Path
    status_file: Path
    root: Path
    workers: int


def _step(name: str, detail: str, checked: int, failures: Sequence[str], started: float) -> AuditStep:
    return AuditStep(
        name=name,
        ok=not failures,
        checked=checked,
        detail=detail,
        failures=tuple(failures[:_LIMIT]),
        seconds=time.perf_counter() - started,
    )


def _guarded(name: str, run: Callable[[], AuditStep]) -> AuditStep:
    """Run one step; evidence that cannot be read makes it unavailable, never ends the audit."""
    started = time.perf_counter()
    try:
        return run()
    except (*EVIDENCE_ERRORS, LookupError) as error:
        return AuditStep(
            name=name,
            ok=False,
            checked=0,
            detail="the step could not run",
            failures=(f"{type(error).__name__}: {str(error)[:300]}",),
            seconds=time.perf_counter() - started,
            unavailable=True,
        )


def _index(audited: _Audited) -> SearchComparisonResults:
    """The results index, or the reason it could not be read, raised inside the step that needed it."""
    if audited.results is None:
        msg = f"the results index cannot be read: {audited.index_error}"
        raise ValueError(msg)
    return audited.results


def _rows(audited: _Audited) -> tuple[ManualRunRow, ...]:
    return table_from_csv(stored_rows(audited.context.store, _index(audited), "ManualRunRow"), ManualRunRow)


def check_sources(audited: _Audited) -> AuditStep:
    """The index's bound inputs, documents and tables, each against the digest it records."""
    started = time.perf_counter()
    context, results = audited.context, _index(audited)
    inputs = results.inputs
    evidence_sha256, n_pointers = evidence_digest(audited.evidence_dir)
    actual = {
        "search protocol": protocol_digest(context.protocol),
        "study manifest": sha256_file(context.protocol.study),
        "evaluation configuration": sha256_file(context.protocol.comparison.evaluation),
        "comparison status": sha256_file(audited.status_file),
        "evidence pointers": evidence_sha256,
    }
    recorded = {
        "search protocol": inputs.protocol_sha256,
        "study manifest": inputs.study_manifest_sha256,
        "evaluation configuration": inputs.evaluation_sha256,
        "comparison status": inputs.status_sha256,
        "evidence pointers": inputs.evidence_sha256,
    }
    failures = [
        f"{name}: recorded {recorded[name][:12]}, found {actual[name][:12]}"
        for name in actual
        if actual[name] != recorded[name]
    ]
    if n_pointers != inputs.n_pointers:
        failures.append(f"the index binds {inputs.n_pointers} pointers, the directory holds {n_pointers}")
    for document in results.documents:
        path = audited.results_dir / document.name
        if not path.is_file() or sha256_file(path) != document.sha256:
            failures.append(f"{document.name} is missing or not the document the index records")
    try:
        accounting = load_accounting(audited.results_dir / DOCUMENTS["accounting"])
        page = audited.results_dir / f"results_v{RESULTS_VERSION}.md"
        if page.read_bytes() != render_search_results(results, accounting).encode("utf-8"):
            failures.append(f"{page.name} is not the index's rendering")
    except EVIDENCE_ERRORS as error:
        failures.append(f"results_v{RESULTS_VERSION}.md cannot be checked: {type(error).__name__}: {str(error)[:200]}")
    for table in results.tables:
        try:
            verify_artifact(context.store, table.payload)
        except EVIDENCE_ERRORS as error:
            failures.append(f"{table.name}: {type(error).__name__}: {str(error)[:200]}")
    checked = len(actual) + 2 + len(results.documents) + len(results.tables)
    return _step("sources", "bound inputs, committed documents and stored tables by digest", checked, failures, started)


def check_selection(audited: _Audited) -> AuditStep:
    """The freeze rebuilt from the search's records (done when the context was built) is the one the index binds."""
    started = time.perf_counter()
    context = audited.context
    failures: list[str] = []
    bound = _index(audited).inputs.freeze_sha256
    if context.freeze_sha256 != bound:
        failures.append(f"the verified freeze is {context.freeze_sha256[:12]}, the index binds {bound[:12]}")
    # A shortfall is a reported outcome of the freeze, not a fault; the comparison must cover exactly what it chose.
    chosen = {item.rank: item.configuration for item in context.freeze.chosen}
    compared = {rank: configuration.label for rank, configuration in context.configurations.items()}
    if compared != chosen:
        failures.append(f"the comparison covers {compared}, the freeze chose {chosen}")
    return _step(
        "selection", "the freeze rebuilt from the search's trial records and bound by the index", 1, failures, started
    )


def check_units(audited: _Audited) -> AuditStep:
    """Every unit's records against the verified freeze and its manifest, and the status re-rendered."""
    started = time.perf_counter()
    context = audited.context
    failures: list[str] = []
    for unit in context.units:
        if not (unit_directory(context.store, unit) / "outcome.json").is_file():
            failures.append(f"unit {unit.number} has no outcome")
            continue
        try:
            check_finalized(context, unit)
        except EVIDENCE_ERRORS as error:
            failures.append(str(error)[:300])
    status = comparison_status(context)
    rendered = json.dumps(to_mapping(status), sort_keys=True, indent=2) + "\n"
    if rendered.encode("utf-8") != audited.status_file.read_bytes():
        failures.append("the committed comparison status is not the one the unit records give")
    markdown = audited.status_file.with_suffix(".md")
    if markdown.read_bytes() != render_status(status).encode("utf-8"):
        failures.append("the committed status rendering is not the status's")
    return _step(
        "units", "every unit's reservation and outcome, and the status", len(context.units) + 2, failures, started
    )


def check_manifests(audited: _Audited) -> AuditStep:
    """Every model and bank manifest, checked as a resume checks it."""
    started = time.perf_counter()
    context = audited.context
    entries, keys = comparison_scope(context)
    checked, failures = verify_manifests(context.store, context.runner, audited.evidence_dir, entries, keys)
    if checked != len(entries) + len(keys):
        failures.append(f"{checked} manifests were found, the comparison holds {len(entries) + len(keys)}")
    return _step("manifests", "every model and replay bank the comparison holds", checked, failures, started)


def check_payloads_and_metrics(audited: _Audited) -> AuditStep:
    """Every run's payload, judged again from its arrays, and every column of its row."""
    started = time.perf_counter()
    context = audited.context
    entries, keys = comparison_scope(context)
    rows = _rows(audited)
    failures = rebuild_rows(
        context.store, context.runner, audited.evidence_dir, entries, keys, rows, workers=audited.workers
    )
    return _step(
        "payloads_and_metrics",
        "every stored run verified, judged again from its own arrays, and its row rebuilt from that judgement",
        len(rows),
        failures,
        started,
    )


def check_aggregates(audited: _Audited) -> AuditStep:
    """Every contrast and summary rebuilt from the per-run table."""
    started = time.perf_counter()
    context = audited.context
    rows = _rows(audited)
    scenarios = tuple((case.scenario_id, str(case.kind)) for case in context.runner.scenarios)
    verdicts = verdicts_of(rows)
    contrasts = contrast_rows(verdicts, scenarios=scenarios)
    failures: list[str] = []
    stored = table_from_csv(stored_rows(context.store, _index(audited), "ManualContrastRow"), ManualContrastRow)
    if contrasts != stored:
        failures.append("the contrasts rebuilt from the run table are not the stored ones")
    summaries = contrast_summaries(contrasts)
    committed = table_from_csv(
        (audited.results_dir / DOCUMENTS["contrast_summary"]).read_text(encoding="utf-8"), ManualContrastSummary
    )
    if summaries != committed:
        failures.append("the contrast summaries rebuilt from the contrasts are not the committed ones")
    arms = arm_summaries(verdicts, scenarios=scenarios)
    committed_arms = table_from_csv(
        (audited.results_dir / DOCUMENTS["arm_summary"]).read_text(encoding="utf-8"), ManualArmSummary
    )
    if arms != committed_arms:
        failures.append("the arm summaries rebuilt from the run table are not the committed ones")
    return _step(
        "aggregates",
        "every contrast and summary rebuilt from the run table",
        len(contrasts) + len(summaries) + len(arms),
        failures,
        started,
    )


def check_completeness(audited: _Audited) -> AuditStep:
    """The accounting rebuilt from the rows and the unit records, and compared whole."""
    started = time.perf_counter()
    context, results = audited.context, _index(audited)
    rows = _rows(audited)
    # Counted from the rows themselves, never taken from the index being checked.
    n_models = len({row.evidence_identity for row in rows if row.source == "rc" and row.evidence_identity})
    n_banks = len({row.evidence_identity for row in rows if row.source == "replay" and row.evidence_identity})
    rebuilt = account_comparison(context, rows, n_models=n_models, n_banks=n_banks, root=audited.root)
    committed = load_accounting(audited.results_dir / DOCUMENTS["accounting"])
    failures: list[str] = []
    if rebuilt != committed:
        failures.append("the accounting rebuilt from the rows and the unit records is not the committed one")
    if not rebuilt.complete:
        failures.append("the comparison is not complete against its plan")
    rc = [row for row in rows if row.source == "rc" and row.status in _SIMULATED]
    replay = [row for row in rows if row.source == "replay" and row.status in _SIMULATED]
    figures: dict[str, object] = {
        "n_models": n_models,
        "n_replay_banks": n_banks,
        "n_rc_runs": len(rc),
        "n_replay_runs": len(replay),
        "n_unavailable_runs": sum(1 for row in rows if row.status not in _SIMULATED),
        "n_rc_successes": sum(1 for row in rc if row.success),
        "n_replay_successes": sum(1 for row in replay if row.success),
        "complete": rebuilt.complete,
        "departure_radius_m": context.runner.conditions(0.0, (1.0, 1.0)).dwell_tolerance_m,
    }
    recorded = to_mapping(results)
    failures += [
        f"the index records {name}={recorded[name]!r}, the rows give {value!r}"
        for name, value in figures.items()
        if recorded[name] != value
    ]
    table_rows = {table.record: table.n_rows for table in results.tables}
    if table_rows.get("ManualRunRow") != len(rows):
        failures.append(f"the index cites {table_rows.get('ManualRunRow')} run rows, the table holds {len(rows)}")
    return _step(
        "completeness",
        "the accounting, the index's headline figures and the planned totals",
        len(rebuilt.units) + len(figures) + 1,
        failures,
        started,
    )


_CHECKS: Final[dict[str, Callable[[_Audited], AuditStep]]] = {
    "sources": check_sources,
    "selection": check_selection,
    "units": check_units,
    "manifests": check_manifests,
    "payloads_and_metrics": check_payloads_and_metrics,
    "aggregates": check_aggregates,
    "completeness": check_completeness,
}


def audit_search_comparison(
    context: ComparisonContext,
    *,
    results_dir: Path,
    evidence_dir: Path,
    status_file: Path,
    root: Path,
    workers: int = 1,
) -> SearchComparisonAudit:
    """Run every step over the committed derivation and return the record, failures retained."""
    index = results_dir / f"results_v{RESULTS_VERSION}.json"
    # The index is read once, inside the audit: its record and its fingerprint come from the same bytes, so a
    # file that cannot be read is a finding recorded by every step that needed it, never an escaped error.
    results: SearchComparisonResults | None = None
    results_sha256, index_error = "0" * 64, None
    try:
        data = index.read_bytes()
        results_sha256 = sha256_bytes(data)
        results = from_mapping(cast("dict[str, object]", json.loads(data)), SearchComparisonResults)
    except (*EVIDENCE_ERRORS, TypeError, KeyError) as error:
        index_error = f"{type(error).__name__}: {str(error)[:300]}"
    audited = _Audited(
        context=context,
        results=results,
        index_error=index_error,
        results_dir=results_dir,
        evidence_dir=evidence_dir,
        status_file=status_file,
        root=root,
        workers=workers,
    )
    steps = tuple(_guarded(name, lambda check=check: check(audited)) for name, check in _CHECKS.items())
    return SearchComparisonAudit(
        experiment=EXPERIMENT,
        version=AUDIT_VERSION,
        results_sha256=results_sha256,
        freeze_sha256=context.freeze_sha256,
        steps=steps,
        passed=all(step.ok for step in steps),
        command=context.runner.command,
        provenance=context.runner.provenance,
    )


def render_audit(audit: SearchComparisonAudit) -> str:
    """The audit as Markdown; the JSON is the record."""
    lines = [
        "<!-- Generated by `python -m arm_rc_ctrl.experiments.manual_search_audit`; do not edit. -->",
        "",
        f"# M3MS comparison audit (v{audit.version})",
        "",
        (
            f"Results index `{audit.results_sha256[:12]}`, freeze `{audit.freeze_sha256[:12]}`. "
            f"Verdict: **{'passed' if audit.passed else 'failed'}**. No run was simulated again (owner "
            "decision); every stored run was judged again from its own arrays."
        ),
        "",
        "| step | status | checked | seconds | covers |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    lines += [f"| {s.name} | {s.status} | {s.checked:,} | {s.seconds:.1f} | {s.detail} |" for s in audit.steps]
    failing = [s for s in audit.steps if s.failures]
    if failing:
        lines += ["", "## Failures", ""]
        for step in failing:
            lines += [f"### {step.name}", "", *[f"- {failure}" for failure in step.failures], ""]
    lines += ["", "Generated by:", "", "```text", audit.command, "```", ""]
    return "\n".join(lines)


def load_search_audit(path: Path) -> SearchComparisonAudit:
    """Strictly rebuild an audit record from its JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), SearchComparisonAudit)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: audit the committed derivation (new files only); non-zero when it fails."""
    parser = argparse.ArgumentParser(description="Audit the M3MS comparison's derived evidence.")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path, help="the committed results directory")
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="the audit JSON; its Markdown is written beside it")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--exploratory", action="store_true", help="tolerate a dirty worktree (fixtures)")
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    output = cast("Path", args.output)
    markdown = output.with_suffix(".md")
    for path in (output, markdown):
        if path.exists():
            msg = f"{path} already exists; an audit that was run is kept, and a new one is a new version"
            raise FileExistsError(msg)
    protocol_file = cast("Path", args.protocol)
    root = repository_root()
    context = ComparisonContext(
        load_manual_search(protocol_file),
        protocol_file,
        cast("Path", args.freeze),
        root=root,
        exploratory=bool(args.exploratory),
        module="arm_rc_ctrl.experiments.manual_search_audit",
        argv=argv,
    )
    audit = audit_search_comparison(
        context,
        results_dir=cast("Path", args.results),
        evidence_dir=cast("Path", args.evidence_dir),
        status_file=cast("Path", args.status),
        root=root,
        workers=int(cast("int", args.workers)),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((canonical_json(to_mapping(audit)) + "\n").encode("utf-8"))
    markdown.write_bytes(render_audit(audit).encode("utf-8"))
    print(render_audit(audit))
    return 0 if audit.passed else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
