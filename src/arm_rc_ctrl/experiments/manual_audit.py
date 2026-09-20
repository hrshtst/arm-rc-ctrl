# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-011: the clean-checkout audit of the manual study's evidence (plan section 7.1).

The study's evidence is only as good as what an independent reader can check
from a fresh checkout, so this audit re-derives rather than re-reads: every
source file and payload against the digest the committed records keep for it,
every manifest against the study's trusted inputs the way a resume checks it,
every run's metrics recomputed from its own stored trajectories, every
comparison and summary rebuilt from the per-run table, and the frozen
re-simulation subset simulated again and compared with what the sweep stored.

Failures are retained, never raised away: a step records what it checked and
every way the evidence disagreed with it, and the audit ends with a verdict
over all of them. An audit that stopped at the first disagreement would report
one problem and hide the rest.

Re-simulation writes into a scratch store, never the canonical one: an audit
that added runs to the evidence it audits would no longer be auditing the
evidence the study produced. The sample is the subset frozen before execution
(M3MAN-009), because a sample chosen afterwards could be the runs that happened
to reproduce.

Command line::

    python -m arm_rc_ctrl.experiments.manual_audit audit
        --study docs/experiments/task_1a_manual_demonstration/study_manifest_v1.json
        --evaluation configs/evaluations/task_1a_manual_dev_v1.toml
        --evidence-dir docs/experiments/task_1a_manual_demonstration/evidence
        --results docs/experiments/task_1a_manual_demonstration/results
        --docs docs/experiments/task_1a_manual_demonstration
        --output docs/experiments/task_1a_manual_demonstration/audit
        [--workers N] [--gates] [--exploratory]
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.data.manual import ManualDatasetRecord
from arm_rc_ctrl.data.records import load_record, verify_payload
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.experiments.manual_accounting import StudyAccounting, account_study
from arm_rc_ctrl.experiments.manual_contrasts import (
    ManualArmSummary,
    ManualContrastRow,
    ManualContrastSummary,
    arm_summaries,
    contrast_rows,
    contrast_summaries,
)
from arm_rc_ctrl.experiments.manual_evaluation import (
    load_manual_model_evidence,
    load_manual_pointer,
    load_manual_replay_bank,
    load_verified_run,
    manual_pointer_name,
    prepare_runner,
    spawn_worker,
)
from arm_rc_ctrl.experiments.manual_figures import load_figure_inputs, plot_case
from arm_rc_ctrl.experiments.manual_fits import ManualFitStore, recipe_mismatches
from arm_rc_ctrl.experiments.manual_handoff import RepresentativeRule, RunOrdering, load_handoff
from arm_rc_ctrl.experiments.manual_resimulation import ResimulationSubset, resimulation_subset
from arm_rc_ctrl.experiments.manual_results import (
    RESULT_DOCUMENTS,
    ManualResults,
    ManualRunRow,
    ManualSelections,
    evidence_digest,
    load_results,
    run_metrics,
    selections_of,
    table_from_csv,
    verdicts_of,
)
from arm_rc_ctrl.experiments.manual_study import EXPERIMENT_LABEL
from arm_rc_ctrl.provenance import ProvenanceRecord, canonical_json, sha256_bytes, sha256_file, verify_artifact
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import ENV_VAR, StorageError, StorageRoot

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.experiments.manual_evaluation import ManualEvaluationRunner, ManualPairRecord
    from arm_rc_ctrl.experiments.manual_study import StudyManifest

__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "AUDIT_VERSION",
    "AuditStep",
    "AuditTolerances",
    "BundleItem",
    "ManualAudit",
    "ResimulatedRun",
    "audit_to_json",
    "load_audit",
    "main",
    "render_audit_markdown",
]

AUDIT_VERSION: Final = 1
"""Version of this audit's outputs; a re-audit is a new version beside it, never an edit."""
AUDIT_SCHEMA_VERSION: Final = 1
"""Version of the audit record itself, which is its own artifact beside the result schema."""
_MODULE: Final = "arm_rc_ctrl.experiments.manual_audit"
_SIMULATED: Final = ("completed", "infeasible")
_METRIC_FIELDS: Final = tuple(
    f.name for f in dc.fields(ManualRunRow) if f.name.endswith(("_m", "_s", "_rad", "_s2", "_nm", "_rad_s"))
)
"""Every measured column of the per-run table, recomputed here from the run's own arrays."""


@dataclass(frozen=True)
class AuditTolerances:
    """What this audit accepts as agreement, declared before it runs (plan section 7.1)."""

    arrays_bitwise: bool
    """Whether a re-simulated run must reproduce the stored arrays digest exactly."""
    array_abs_tol: float
    """Largest absolute deviation tolerated in a re-simulated channel when the digest differs (rad, m, N m)."""
    metric_abs_tol: float
    """Largest absolute difference tolerated between a committed metric and its recomputation."""
    metric_rel_tol: float
    """Largest relative difference tolerated between a committed metric and its recomputation."""
    aggregates_exact: bool
    """Whether counts, verdicts and aggregates must agree exactly; they are integers and they must."""


DECLARED_TOLERANCES: Final = AuditTolerances(
    arrays_bitwise=True,
    array_abs_tol=1e-9,
    metric_abs_tol=0.0,
    metric_rel_tol=0.0,
    aggregates_exact=True,
)
"""The canonical environment is pinned to the P-cores, where the study reproduced bitwise (plan section 12, C10).

So re-simulation is held to the arrays digest, and recomputed metrics to exact
equality: the same definitions over the same stored arrays on the same machine
are the same floating-point operations. ``array_abs_tol`` is what a failing
run's deviation is reported against, so a disagreement is quantified rather
than merely flagged.
"""


@dataclass(frozen=True)
class AuditStep:
    """One audited property: what it covered, and every way the evidence disagreed with it."""

    name: str
    ok: bool
    checked: int
    detail: str
    failures: tuple[str, ...] = ()
    seconds: float = 0.0

    def __post_init__(self) -> None:
        """A step passes exactly when it retained no failure."""
        if self.ok != (not self.failures):
            msg = f"{self.name}: ok={self.ok} with {len(self.failures)} failures"
            raise ValueError(msg)


@dataclass(frozen=True)
class ResimulatedRun:
    """One frozen-subset run simulated again from the same inputs, against what the study stored."""

    label: str
    scenario_id: str
    tracker: str
    arm: str
    committed_arrays_sha256: str
    rebuilt_arrays_sha256: str
    bitwise: bool
    max_abs_deviation: float | None
    """Largest absolute difference over every channel, measured only when the digests differ."""
    committed_status: str
    rebuilt_status: str


@dataclass(frozen=True)
class BundleItem:
    """One part of the handoff bundle: where it is, what it is, and the digest it is cited by."""

    name: str
    kind: str
    location: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ManualAudit:
    """The audit of the manual study's evidence: what was checked, what disagreed, and under which tolerances."""

    experiment: str
    version: int
    created_at: str
    checkout_commit: str
    checkout_dirty: bool
    validated_commits: tuple[str, ...]
    """The commits whose code produced and then checked this evidence, in order."""
    results_commit: str
    """The commit the derived evidence's own provenance names as its generator."""
    tolerances: AuditTolerances
    steps: tuple[AuditStep, ...]
    resimulated: tuple[ResimulatedRun, ...]
    bundle: tuple[BundleItem, ...]
    n_checked: int
    n_failures: int
    ok: bool
    command: str
    provenance: ProvenanceRecord
    schema_version: int = field(default=AUDIT_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """The verdict re-derives from the steps, so an audit cannot pass while a step failed."""
        if self.experiment != EXPERIMENT_LABEL or self.schema_version != AUDIT_SCHEMA_VERSION:
            msg = f"unsupported audit {self.schema_version} of {self.experiment!r}"
            raise ValueError(msg)
        derived = (
            sum(step.checked for step in self.steps),
            sum(len(step.failures) for step in self.steps),
            all(step.ok for step in self.steps),
        )
        if derived != (self.n_checked, self.n_failures, self.ok):
            msg = "the audit's verdict does not re-derive from its steps"
            raise ValueError(msg)


def audit_to_json(audit: ManualAudit) -> str:
    """Canonical JSON of the audit record, newline-terminated."""
    return canonical_json(to_mapping(audit)) + "\n"


def load_audit(path: Path) -> ManualAudit:
    """Strictly rebuild the audit record from its JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), ManualAudit)


# --- the inputs every step reads ------------------------------------------------------------------


@dataclass(frozen=True)
class _Inputs:
    """Everything the audit was pointed at, resolved once."""

    root: Path
    docs: Path
    results_dir: Path
    evidence_dir: Path
    study_file: Path
    evaluation_file: Path
    runner: ManualEvaluationRunner
    manifest: StudyManifest
    store: StorageRoot
    results: ManualResults
    ordering: RunOrdering
    rule: RepresentativeRule
    rows: tuple[ManualRunRow, ...]
    workers: int


def _step(name: str, detail: str, checked: int, failures: Sequence[str], started: float) -> AuditStep:
    return AuditStep(
        name=name,
        ok=not failures,
        checked=checked,
        detail=detail,
        failures=tuple(failures),
        seconds=time.perf_counter() - started,
    )


def _digest_failures(pairs: Sequence[tuple[str, Path, str]]) -> list[str]:
    """Every named file whose digest is not the one recorded for it, or that is missing."""
    failures: list[str] = []
    for name, path, expected in pairs:
        if not path.is_file():
            failures.append(f"{name}: {path.name} is missing")
        elif sha256_file(path) != expected:
            failures.append(f"{name}: {path.name} has another digest than the one recorded for it")
    return failures


def check_checkout(inputs: _Inputs) -> AuditStep:
    """The audit's own checkout and environment, and the provenance the derived evidence carries."""
    started = time.perf_counter()
    failures: list[str] = []
    provenance = inputs.results.provenance
    execution = inputs.runner.execution
    if not execution.canonical:
        failures.append("the audit is not running in the canonical execution environment")
    if execution.identity != inputs.manifest.execution.identity:
        failures.append(
            f"the audit's execution identity {execution.identity[:12]} is not the study's "
            f"{inputs.manifest.execution.identity[:12]}"
        )
    if provenance.project_dirty:
        failures.append("the derived evidence was produced from a modified checkout")
    if provenance.exploratory:
        failures.append("the derived evidence is marked exploratory")
    return _step(
        "checkout",
        "the audit runs in the study's canonical environment, and the derived evidence was produced cleanly",
        4,
        failures,
        started,
    )


def check_sources(inputs: _Inputs) -> AuditStep:
    """Every configuration and frozen artifact the study and its derived evidence bind by digest."""
    started = time.perf_counter()
    manifest, docs, root = inputs.manifest, inputs.docs, inputs.root
    checks: list[tuple[str, Path, str]] = [
        (f"study source {name}", root / getattr(manifest, name).path, getattr(manifest, name).sha256)
        for name in ("panel", "bank", "scenario", "preprocessing", "model")
    ]
    bound = inputs.results.inputs
    checks += [
        ("study manifest", inputs.study_file, bound.study_manifest_sha256),
        ("evaluation configuration", inputs.evaluation_file, bound.evaluation_sha256),
        ("run ordering", docs / "run_ordering_v1.json", bound.run_ordering_sha256),
        ("representative rule", docs / "representative_rule_v1.json", bound.representative_rule_sha256),
        ("result schema", docs / "result_schema_v3.json", bound.result_schema_sha256),
        ("the ordering the rule was frozen against", docs / "run_ordering_v1.json", inputs.rule.ordering_sha256),
    ]
    failures = _digest_failures(checks)
    digest, pointers = evidence_digest(inputs.evidence_dir)
    if (digest, pointers) != (bound.evidence_sha256, bound.n_pointers):
        failures.append(
            f"the evidence directory holds {pointers} pointers digesting to {digest[:12]}, not the "
            f"{bound.n_pointers} digesting to {bound.evidence_sha256[:12]} the results index bound"
        )
    for role, name in RESULT_DOCUMENTS.items():
        recorded = next((d for d in inputs.results.documents if d.name == name), None)
        if recorded is None:
            failures.append(f"the results index names no {role} document")
            continue
        failures += _digest_failures([(f"{role} document", inputs.results_dir / name, recorded.sha256)])
    return _step(
        "sources",
        "every source, freeze and committed document is the one the evidence binds",
        len(checks) + 1 + len(RESULT_DOCUMENTS),
        failures,
        started,
    )


def check_datasets(inputs: _Inputs) -> AuditStep:
    """The ten locked demonstrations and the raw takes they were derived from."""
    started = time.perf_counter()
    failures: list[str] = []
    checked = 0
    for demonstration in inputs.manifest.demonstrations:
        dataset = demonstration.dataset
        checked += 1
        try:
            record = load_record(inputs.root / dataset.record, ManualDatasetRecord)
            if record.artifact.payload.sha256 != dataset.payload_sha256:
                failures.append(f"{demonstration.assignment}: {dataset.record} records another payload digest")
                continue
            samples = load_samples(verify_payload(inputs.store, record.artifact))
            record.check_samples(samples)
            if samples.n_samples - 1 != demonstration.loss_rows:
                failures.append(f"{demonstration.assignment}: the payload's rows are not the study's loss rows")
            for source in record.artifact.origin.sources:
                checked += 1
                raw = inputs.root / "data" / "records" / "raw" / f"{source}.toml"
                if not raw.is_file():
                    failures.append(f"{demonstration.assignment}: the raw record {source} is not committed")
        except (OSError, ValueError, StorageError) as error:
            failures.append(f"{demonstration.assignment}: {type(error).__name__}: {str(error)[:200]}")
    return _step("datasets", "every locked demonstration and its raw source", checked, failures, started)


def check_manifests(inputs: _Inputs) -> AuditStep:
    """Every model and replay-bank manifest, checked against the study's trusted inputs as a resume checks it."""
    started = time.perf_counter()
    failures: list[str] = []
    checked = 0
    runner = inputs.runner
    for entry in inputs.manifest.entries:
        path = inputs.evidence_dir / manual_pointer_name("model", entry.label)
        if not path.exists():
            continue
        checked += 1
        try:
            pointer = load_manual_pointer(path)
            evidence = load_manual_model_evidence(verify_artifact(inputs.store, pointer.payload))
            runner.verify_stored_model(entry, evidence, where=path.name)
        except (OSError, ValueError, StorageError) as error:
            failures.append(f"{entry.label}: {type(error).__name__}: {str(error)[:200]}")
    banks = {(key.configuration, key.assignment): key for key in inputs.ordering.replay_banks}
    for (configuration, parent), key in banks.items():
        path = _bank_pointer(inputs, configuration, parent)
        if path is None:
            continue
        checked += 1
        try:
            bank = load_manual_replay_bank(verify_artifact(inputs.store, load_manual_pointer(path).payload))
            runner.verify_stored_bank(
                bank,
                assignment=parent,
                warmup_s=key.warmup_s,
                replay_cutoffs=(key.velocity_cutoff_hz, key.acceleration_cutoff_hz),
                where=path.name,
            )
        except (OSError, ValueError, StorageError) as error:
            failures.append(f"{configuration}/{parent}: {type(error).__name__}: {str(error)[:200]}")
    return _step("manifests", "every model and replay bank the study names", checked, failures, started)


def _bank_pointer(inputs: _Inputs, configuration: str, parent: str) -> Path | None:
    """The pointer of the bank the run ordering keys at (configuration, parent), by its trusted identity."""
    key = next(k for k in inputs.ordering.replay_banks if (k.configuration, k.assignment) == (configuration, parent))
    conditions = inputs.runner.conditions(key.warmup_s, (key.velocity_cutoff_hz, key.acceleration_cutoff_hz))
    wanted = sha256_bytes(f"{conditions.identity}:{parent}".encode("ascii"))
    for path in sorted(inputs.evidence_dir.glob("replay__*.toml")):
        if load_manual_pointer(path).identity == wanted:
            return path
    return None


def check_fits(inputs: _Inputs) -> AuditStep:
    """Every cached fit the study's models were produced from: its digests and its construction."""
    started = time.perf_counter()
    fits = ManualFitStore(inputs.store)
    failures: list[str] = []
    entries = [
        entry
        for entry in inputs.manifest.entries
        if (inputs.evidence_dir / manual_pointer_name("model", entry.label)).exists()
    ]
    for entry in entries:
        try:
            record = fits.read_record(entry.fit_identity)
            recipe = fits.read_recipe(record)
            fits.read_weights(record)
            mismatches = recipe_mismatches(entry, recipe, inputs.runner.inputs)
            failures += [f"{entry.label}: {mismatch}" for mismatch in mismatches]
        except (OSError, ValueError, StorageError) as error:
            failures.append(f"{entry.label}: {type(error).__name__}: {str(error)[:200]}")
    return _step(
        "fits", "every present model's fit: its recipe, weights and construction", len(entries), failures, started
    )


# --- payloads and metrics -------------------------------------------------------------------------


def _recompute_row(
    store: StorageRoot,
    pairs: dict[tuple[str, str, str], ManualPairRecord],
    radius_m: float,
    row: ManualRunRow,
) -> list[str]:
    """Verify one run's payload and recompute its measured columns from its own arrays."""
    key = (row.model_label, row.scenario_id, row.tracker)
    pair = pairs.get(key)
    if pair is None:
        return [f"{key}: the evidence holds no pair for this row"]
    try:
        summary, arrays = load_verified_run(store, pair)
    except (OSError, ValueError, StorageError) as error:
        return [f"{key}: {type(error).__name__}: {str(error)[:200]}"]
    if pair.status != row.status or (pair.outcome is not None and pair.outcome.success != row.success):
        return [f"{key}: the table's verdict is not the manifest's"]
    outcome = pair.outcome
    metrics = run_metrics(
        arrays,
        activation_s=cast("float", summary.activation_s),
        target=summary.target,
        final_dwell_samples=outcome.dwell.final_samples if outcome is not None and outcome.dwell.ok else None,
        departure_radius_m=radius_m,
    )
    failures: list[str] = []
    recomputed = dc.asdict(metrics)
    for name in ("n_active_samples", *_METRIC_FIELDS):
        if name not in recomputed:
            continue
        committed, actual = getattr(row, name), recomputed[name]
        if committed is None or actual is None:
            if committed is not actual:
                failures.append(f"{key}.{name}: the table has {committed!r}, the recomputation {actual!r}")
        elif committed != actual:
            failures.append(f"{key}.{name}: the table has {committed!r}, the recomputation {actual!r}")
    return failures


def check_payloads_and_metrics(inputs: _Inputs) -> AuditStep:
    """Every run's payload, and every measured column recomputed from the arrays it stores."""
    started = time.perf_counter()
    pairs = _pairs_by_run(inputs)
    rows = [row for row in inputs.rows if row.status in _SIMULATED]
    work = partial(_recompute_row, inputs.store, pairs, inputs.results.departure_radius_m)
    if inputs.workers == 1:
        results = [work(row) for row in rows]
    else:
        with ThreadPoolExecutor(max_workers=inputs.workers) as pool:
            results = list(pool.map(work, rows))
    failures = [failure for group in results for failure in group]
    return _step(
        "payloads_and_metrics",
        "every stored run verified by digest and its metrics recomputed from its own trajectories",
        len(rows),
        failures[:200],
        started,
    )


def _pairs_by_run(inputs: _Inputs) -> dict[tuple[str, str, str], ManualPairRecord]:
    """Every recorded pair by (model label, scenario, tracker), read from the verified manifests."""
    pairs: dict[tuple[str, str, str], ManualPairRecord] = {}
    for entry in inputs.manifest.entries:
        path = inputs.evidence_dir / manual_pointer_name("model", entry.label)
        if not path.exists():
            continue
        evidence = load_manual_model_evidence(verify_artifact(inputs.store, load_manual_pointer(path).payload))
        for pair in evidence.pairs:
            pairs[entry.label, pair.scenario_id, pair.tracker] = pair
    for key in inputs.ordering.replay_banks:
        path = _bank_pointer(inputs, key.configuration, key.assignment)
        if path is None:
            continue
        bank = load_manual_replay_bank(verify_artifact(inputs.store, load_manual_pointer(path).payload))
        label = f"{key.configuration}/replay/{key.assignment}"
        for pair in bank.pairs:
            pairs[label, pair.scenario_id, pair.tracker] = pair
    return pairs


def check_aggregates(inputs: _Inputs) -> AuditStep:
    """Every comparison, summary and selection rebuilt from the committed per-run table."""
    started = time.perf_counter()
    failures: list[str] = []
    scenarios = tuple((case.scenario_id, str(case.kind)) for case in inputs.runner.scenarios)
    verdicts = verdicts_of(inputs.rows)
    rebuilt_contrasts = contrast_rows(verdicts, scenarios=scenarios)
    committed_contrasts = _stored_table(inputs, "ManualContrastRow", ManualContrastRow)
    if rebuilt_contrasts != committed_contrasts:
        failures.append(_first_difference("contrast", rebuilt_contrasts, committed_contrasts))
    rebuilt_summaries = contrast_summaries(rebuilt_contrasts)
    committed_summaries = table_from_csv(
        (inputs.results_dir / RESULT_DOCUMENTS["contrast_summary"]).read_text(encoding="utf-8"), ManualContrastSummary
    )
    if rebuilt_summaries != committed_summaries:
        failures.append(_first_difference("contrast summary", rebuilt_summaries, committed_summaries))
    rebuilt_arms = arm_summaries(verdicts, scenarios=scenarios)
    committed_arms = table_from_csv(
        (inputs.results_dir / RESULT_DOCUMENTS["arm_summary"]).read_text(encoding="utf-8"), ManualArmSummary
    )
    if rebuilt_arms != committed_arms:
        failures.append(_first_difference("arm summary", rebuilt_arms, committed_arms))
    committed_selections = from_mapping(
        cast(
            "dict[str, object]",
            json.loads((inputs.results_dir / RESULT_DOCUMENTS["selections"]).read_text(encoding="utf-8")),
        ),
        ManualSelections,
    )
    rebuilt_selections = selections_of(
        inputs.rows,
        inputs.rule,
        order=[sid for sid, _ in scenarios],
        rule_sha256=inputs.results.inputs.representative_rule_sha256,
        ordering_sha256=inputs.results.inputs.run_ordering_sha256,
    )
    if rebuilt_selections != committed_selections:
        failures.append("the selections rebuilt from the run table are not the committed ones")
    checked = len(rebuilt_contrasts) + len(rebuilt_summaries) + len(rebuilt_arms) + len(rebuilt_selections.applications)
    return _step(
        "aggregates", "every comparison, summary and selection rebuilt from the run table", checked, failures, started
    )


def _first_difference(kind: str, rebuilt: Sequence[object], committed: Sequence[object]) -> str:
    """Name the first row that differs, so a disagreement is located rather than merely reported."""
    if len(rebuilt) != len(committed):
        return f"the rebuilt {kind} table has {len(rebuilt)} rows, the committed one {len(committed)}"
    for index, (left, right) in enumerate(zip(rebuilt, committed, strict=True)):
        if left != right:
            return f"the rebuilt {kind} row {index} is not the committed one: {left!r} against {right!r}"[:400]
    return f"the {kind} tables differ"


def _stored_table[T](inputs: _Inputs, record: str, cls: type[T]) -> tuple[T, ...]:
    """One table the results index keeps in the store, verified by digest and read back strictly."""
    table = next(t for t in inputs.results.tables if t.record == record)
    return table_from_csv(verify_artifact(inputs.store, table.payload).read_text(encoding="utf-8"), cls)


def check_figures(inputs: _Inputs, scratch: Path) -> AuditStep:
    """The figure inputs' bindings, and that a case still renders from them alone."""
    started = time.perf_counter()
    failures: list[str] = []
    figures = load_figure_inputs(inputs.results_dir / RESULT_DOCUMENTS["figure_inputs"])
    runs = {(row.model_label, row.scenario_id, row.tracker): row for row in inputs.rows}
    checked = 0
    for case in figures.cases:
        for drawn in case.runs:
            checked += 1
            row = runs.get((drawn.label, case.scenario_id, case.tracker))
            if row is None or row.run_sha256 != drawn.run.sha256 or row.arrays_sha256 != drawn.run.arrays_sha256:
                failures.append(f"{case.case_id}: the {drawn.role} run is not the one the per-run table records")
    try:
        plot_case(figures, figures.cases[0].case_id, scratch / "audit-case.png", store=inputs.store, root=inputs.root)
        checked += 1
    except (OSError, ValueError, StorageError) as error:
        failures.append(f"rendering {figures.cases[0].case_id}: {type(error).__name__}: {str(error)[:200]}")
    return _step(
        "figures", "every figure input bound to the run table, and one case rendered", checked, failures, started
    )


def check_completeness(inputs: _Inputs, accounting: StudyAccounting, bundle_missing: Sequence[str]) -> AuditStep:
    """The accounting, the frozen totals, and the per-run table's coverage of what the protocol names."""
    started = time.perf_counter()
    failures: list[str] = []
    expected = inputs.ordering.expected
    if not accounting.complete:
        failures.append(f"the accounting is incomplete: {len(accounting.missing)} models have no evidence")
    if not accounting.all_bind_canonical_execution:
        failures.append("some evidence was keyed in another execution identity than the study's")
    counts = {
        "models": (len([line for line in accounting.models if line.present]), expected.n_models),
        "replay banks": (len(accounting.banks), expected.n_replay_banks),
        "rc runs": (accounting.n_rc_runs, expected.n_rc_runs),
        "replay runs": (accounting.n_replay_runs, expected.n_replay_runs),
        "table rows": (len(inputs.rows), expected.n_runs),
    }
    failures += [
        f"{name}: the evidence holds {found}, the frozen ordering expects {want}"
        for name, (found, want) in counts.items()
        if found != want
    ]
    unavailable = [row for row in inputs.rows if row.status not in _SIMULATED]
    if unavailable:
        failures.append(f"{len(unavailable)} rows name runs that were never simulated")
    failures += list(bundle_missing)
    return _step(
        "completeness", "the accounting, the frozen totals and the bundle's parts", len(counts) + 3, failures, started
    )


# --- re-simulation --------------------------------------------------------------------------------


def _seed_store(inputs: _Inputs, scratch: Path) -> Path:
    """A scratch store holding the ten demonstrations, so a re-simulation never writes into the study's own."""
    store_root = scratch / "store"
    for demonstration in inputs.manifest.demonstrations:
        source = inputs.store.path(f"armrc://processed/{demonstration.dataset.artifact_id}/samples.npz", mode="read")
        target = store_root / "processed" / demonstration.dataset.artifact_id / "samples.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256_file(target) != demonstration.dataset.payload_sha256:
            msg = f"{demonstration.assignment}: the copied demonstration is not the one the study recorded"
            raise ValueError(msg)
    return store_root


def _resimulate(inputs: _Inputs, subset: ResimulationSubset, scratch: Path, *, exploratory: bool) -> tuple[Path, Path]:
    """Re-simulate the frozen subset in a scratch store, in fresh interpreters that inherit this environment."""
    store_root = _seed_store(inputs, scratch)
    evidence = scratch / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, ENV_VAR: str(store_root)}
    scenarios = tuple(case.scenario_id for case in subset.scenarios)
    for model in subset.models:
        spawn_worker(
            model,
            warmup_s=inputs.runner.inputs.configuration(model).warmup_s,
            env=env,
            study_file=inputs.study_file,
            evaluation_file=inputs.evaluation_file,
            root=inputs.root,
            exploratory=exploratory,
            scenario_ids=scenarios,
        )
    return store_root, evidence


def check_resimulation(
    inputs: _Inputs, subset: ResimulationSubset, scratch: Path, *, exploratory: bool = False
) -> tuple[AuditStep, tuple[ResimulatedRun, ...]]:
    """Simulate the frozen subset again and compare every run with what the study stored."""
    started = time.perf_counter()
    failures: list[str] = []
    outcomes: list[ResimulatedRun] = []
    try:
        store_root, _ = _resimulate(inputs, subset, scratch, exploratory=exploratory)
    except (OSError, ValueError, StorageError, subprocess.SubprocessError) as error:
        return _step(
            "resimulation", "the frozen subset re-simulated", 0, [f"{type(error).__name__}: {error}"], started
        ), ()
    rebuilt = StorageRoot(store_root)
    committed = _pairs_by_run(inputs)
    rebuilt_pairs = _index_rebuilt(inputs, rebuilt)
    for label, scenario_id, tracker in subset.run_identities():
        outcomes.append(
            _compare_run(
                inputs,
                rebuilt,
                committed,
                rebuilt_pairs,
                label=label,
                scenario_id=scenario_id,
                tracker=tracker,
                arm="rc",
            )
        )
    parent = subset_parent(subset)
    outcomes.extend(
        _compare_run(
            inputs,
            rebuilt,
            committed,
            rebuilt_pairs,
            label=f"{configuration}/replay/{parent}",
            scenario_id=case.scenario_id,
            tracker=tracker,
            arm="replay",
        )
        for configuration in sorted({model.configuration for model in subset.models})
        for case in subset.scenarios
        for tracker in subset.trackers
    )
    failures += [
        f"{o.label} {o.scenario_id} [{o.tracker}]: re-simulation did not reproduce the stored arrays"
        f" (deviation {o.max_abs_deviation})"
        for o in outcomes
        if not o.bitwise
    ]
    failures += [
        f"{o.label} {o.scenario_id} [{o.tracker}]: re-simulation reached {o.rebuilt_status}, "
        f"the study stored {o.committed_status}"
        for o in outcomes
        if o.committed_status != o.rebuilt_status
    ]
    return _step(
        "resimulation",
        "the frozen 300-run subset simulated again from the same inputs",
        len(outcomes),
        failures,
        started,
    ), tuple(outcomes)


def subset_parent(subset: ResimulationSubset) -> str:
    """The fixed parent the frozen subset's replay banks belong to."""
    parents = {model.arm.assignment for model in subset.models if model.arm.assignment is not None}
    if len(parents) != 1:
        msg = f"the frozen subset spans {sorted(parents)}, not one parent"
        raise ValueError(msg)
    return parents.pop()


def _compare_run(
    inputs: _Inputs,
    rebuilt_store: StorageRoot,
    committed: dict[tuple[str, str, str], ManualPairRecord],
    rebuilt_pairs: dict[tuple[str, str, str], ManualPairRecord],
    *,
    label: str,
    scenario_id: str,
    tracker: str,
    arm: str,
) -> ResimulatedRun:
    """One re-simulated run against the stored one: the arrays digest first, the deviation only if it differs."""
    pair = committed[label, scenario_id, tracker]
    stored_run = cast("Any", pair.run)
    rebuilt = rebuilt_pairs.get((label, scenario_id, tracker))
    if rebuilt is None or rebuilt.run is None:
        return ResimulatedRun(
            label=label,
            scenario_id=scenario_id,
            tracker=tracker,
            arm=arm,
            committed_arrays_sha256=stored_run.arrays_sha256,
            rebuilt_arrays_sha256="",
            bitwise=False,
            max_abs_deviation=None,
            committed_status=pair.status,
            rebuilt_status="missing",
        )
    bitwise = rebuilt.run.arrays_sha256 == stored_run.arrays_sha256
    deviation = None if bitwise else _max_deviation(inputs.store, rebuilt_store, pair=pair, rebuilt=rebuilt)
    return ResimulatedRun(
        label=label,
        scenario_id=scenario_id,
        tracker=tracker,
        arm=arm,
        committed_arrays_sha256=stored_run.arrays_sha256,
        rebuilt_arrays_sha256=rebuilt.run.arrays_sha256,
        bitwise=bitwise,
        max_abs_deviation=deviation,
        committed_status=pair.status,
        rebuilt_status=rebuilt.status,
    )


def _index_rebuilt(inputs: _Inputs, rebuilt_store: StorageRoot) -> dict[tuple[str, str, str], ManualPairRecord]:
    """Every pair the re-simulation produced, by (label, scenario, tracker)."""
    pairs: dict[tuple[str, str, str], ManualPairRecord] = {}
    reports = rebuilt_store.root / "reports" / "task_1a_manual_v1"
    for kind in ("model", "replay"):
        for directory in sorted((reports / kind).glob("*")) if (reports / kind).is_dir() else []:
            manifests = sorted(directory.glob("manifest-*.json"))
            if len(manifests) != 1:
                continue
            if kind == "model":
                evidence = load_manual_model_evidence(manifests[0])
                label, records = evidence.label, evidence.pairs
            else:
                bank = load_manual_replay_bank(manifests[0])
                configuration = _configuration_of_bank(
                    inputs, bank.conditions.warmup_s, bank.conditions.replay_velocity_cutoff_hz
                )
                label, records = f"{configuration}/replay/{bank.assignment}", bank.pairs
            for pair in records:
                pairs[label, pair.scenario_id, pair.tracker] = pair
    return pairs


def _configuration_of_bank(inputs: _Inputs, warmup_s: float, velocity_cutoff_hz: float) -> str:
    """The configuration whose protocol a re-simulated bank belongs to."""
    for key in inputs.ordering.replay_banks:
        if (key.warmup_s, key.velocity_cutoff_hz) == (warmup_s, velocity_cutoff_hz):
            return key.configuration
    msg = f"no configuration runs replay at warm-up {warmup_s} and cutoff {velocity_cutoff_hz}"
    raise ValueError(msg)


def _max_deviation(
    store: StorageRoot, rebuilt_store: StorageRoot, *, pair: ManualPairRecord, rebuilt: ManualPairRecord
) -> float:
    """The largest absolute difference over every shared channel of two runs of the same scenario."""
    _, left = load_verified_run(store, pair)
    _, right = load_verified_run(rebuilt_store, rebuilt)
    worst = 0.0
    for name, values in left.items():
        other = right.get(name)
        if other is None or np.shape(values) != np.shape(other):
            return float("inf")
        difference = np.abs(np.asarray(values, dtype=np.float64) - np.asarray(other, dtype=np.float64))
        finite = difference[np.isfinite(difference)]
        worst = max(worst, float(np.max(finite)) if finite.size else 0.0)
    return worst


# --- the gates and the bundle ---------------------------------------------------------------------


def check_gates(root: Path, *, run: bool) -> AuditStep:
    """The documented quality gates, run here rather than reported from memory."""
    started = time.perf_counter()
    if not run:
        return _step("gates", "not run in this audit; --gates runs them", 0, [], started)
    failures: list[str] = []
    commands = (
        ("nox", ["uv", "run", "--locked", "nox"]),
        ("pre-commit", ["uv", "run", "--locked", "nox", "-s", "pre_commit"]),
    )
    for name, command in commands:
        completed = subprocess.run(command, cwd=root, capture_output=True, check=False)
        if completed.returncode != 0:
            tail = completed.stdout.decode("utf-8", "replace").strip().splitlines()[-5:]
            failures.append(f"{name} exited {completed.returncode}: {' | '.join(tail)}")
    return _step("gates", "uv run --locked nox, then its pre_commit session", len(commands), failures, started)


def _bundle(inputs: _Inputs) -> tuple[tuple[BundleItem, ...], tuple[str, ...]]:
    """The handoff bundle of plan section 7.1 by digest, and the parts of it that are not there.

    A missing part is returned rather than raised, so the audit reports an
    incomplete bundle as a failure beside everything else it found.
    """
    items: list[BundleItem] = []
    missing: list[str] = []

    def add(name: str, kind: str, path: Path) -> None:
        if not path.is_file():
            missing.append(f"the bundle's {name} is missing: {path.name}")
            return
        if not path.resolve().is_relative_to(inputs.root.resolve()):
            # A record never carries a machine path, so a part outside the checkout is reported, not located.
            missing.append(f"the bundle's {name} lies outside the repository")
            return
        items.append(
            BundleItem(
                name=name,
                kind=kind,
                location=path.resolve().relative_to(inputs.root.resolve()).as_posix(),
                sha256=sha256_file(path),
                size=path.stat().st_size,
            )
        )

    docs = inputs.docs
    add("acquisition settings", "acquisition", docs / "acquisition_readiness_v1.md")
    # The bank and the configurations are taken from the paths the study manifest binds, never guessed:
    # a bundle that names a file the study does not bind would cite something else with the same name.
    add("demonstration bank", "acquisition", inputs.root / inputs.manifest.bank.path)
    for name in ("panel", "scenario", "preprocessing", "model"):
        add(f"{name} configuration", "configuration", inputs.root / getattr(inputs.manifest, name).path)
    add("study manifest", "training", inputs.study_file)
    add("numerical validation", "training", docs / "numerical_validation_v1.json")
    add("run ordering", "protocol", docs / "run_ordering_v1.json")
    add("representative rule", "protocol", docs / "representative_rule_v1.json")
    add("re-simulation subset", "protocol", docs / "resimulation_subset_v1.json")
    add("result schema v3", "schema", docs / "result_schema_v3.json")
    add("execution account", "execution", docs / "execution_account_v1.md")
    add("results index", "results", inputs.results_dir / f"results_v{1}.json")
    for role, name in RESULT_DOCUMENTS.items():
        add(role.replace("_", " "), "results", inputs.results_dir / name)
    add("usage documentation", "results", inputs.results_dir / "usage_v1.md")
    items.extend(
        BundleItem(
            name=table.name,
            kind="stored table",
            location=table.payload.uri,
            sha256=table.payload.sha256,
            size=table.payload.size,
        )
        for table in inputs.results.tables
    )
    return tuple(items), tuple(missing)


def _render_steps(audit: ManualAudit) -> list[str]:
    lines = ["| step | checked | verdict | seconds | covers |", "| --- | --- | --- | --- | --- |"]
    lines += [
        f"| `{step.name}` | {step.checked} | {'pass' if step.ok else f'{len(step.failures)} failures'} "
        f"| {step.seconds:.1f} | {step.detail} |"
        for step in audit.steps
    ]
    return lines


def render_audit_markdown(audit: ManualAudit) -> str:
    """The audit as Markdown: the verdict, every step, the tolerances, the sample, and the bundle."""
    bitwise = sum(1 for run in audit.resimulated if run.bitwise)
    lines = [
        f"# Task 1-a manual-demonstration reproduction audit (v{audit.version})",
        "",
        (
            f"Generated by `manual_audit audit`; regenerate rather than edit. Verdict: "
            f"**{'every check passed' if audit.ok else f'{audit.n_failures} failures'}** over "
            f"{audit.n_checked:,} checks, from checkout `{audit.checkout_commit[:12]}`"
            f"{' (modified)' if audit.checkout_dirty else ''}."
        ),
        "",
        (
            f"The derived evidence was produced by `{audit.results_commit[:12]}`; its checks were then "
            f"strengthened in {', '.join(commit[:12] for commit in audit.validated_commits)}, and this audit "
            f"applies them."
        ),
        "",
        "## Steps",
        "",
        *_render_steps(audit),
        "",
        "## Declared tolerances",
        "",
        "| tolerance | value | meaning |",
        "| --- | --- | --- |",
        (
            f"| `arrays_bitwise` | {audit.tolerances.arrays_bitwise} | a re-simulated run must reproduce the "
            f"stored arrays digest |"
        ),
        f"| `array_abs_tol` | {audit.tolerances.array_abs_tol} | deviation a failing run is reported against |",
        f"| `metric_abs_tol` | {audit.tolerances.metric_abs_tol} | recomputed metrics must equal the committed ones |",
        f"| `metric_rel_tol` | {audit.tolerances.metric_rel_tol} | as above, relatively |",
        (
            f"| `aggregates_exact` | {audit.tolerances.aggregates_exact} | counts, verdicts and aggregates are "
            f"integers and must agree exactly |"
        ),
        "",
        "## Re-simulation of the frozen subset",
        "",
        (
            f"{bitwise} of {len(audit.resimulated)} runs reproduced the stored arrays bitwise "
            f"({len(audit.resimulated) - bitwise} did not). The sample was frozen before execution under "
            f"M3MAN-009; only re-simulation is sampled, while verification and recomputation cover every run."
        ),
        "",
    ]
    if any(not run.bitwise for run in audit.resimulated):
        lines += [
            "| run | scenario | tracker | deviation | stored verdict | re-simulated verdict |",
            "| --- | --- | --- | --- | --- | --- |",
            *[
                f"| `{r.label}` | {r.scenario_id} | {r.tracker} | {r.max_abs_deviation} | {r.committed_status} "
                f"| {r.rebuilt_status} |"
                for r in audit.resimulated
                if not r.bitwise
            ],
            "",
        ]
    if audit.n_failures:
        lines += ["## Retained failures", ""]
        for step in audit.steps:
            lines += [f"- `{step.name}`: {failure}" for failure in step.failures]
        lines.append("")
    lines += [
        "## Handoff bundle",
        "",
        "| item | kind | location | size (bytes) | sha256 |",
        "| --- | --- | --- | --- | --- |",
        *[f"| {i.name} | {i.kind} | `{i.location}` | {i.size} | `{i.sha256}` |" for i in audit.bundle],
        "",
        "Generated by:",
        "",
        "```text",
        audit.command,
        "```",
        "",
    ]
    return "\n".join(lines)


# --- the command ----------------------------------------------------------------------------------


def _inputs(args: argparse.Namespace, root: Path) -> tuple[_Inputs, StudyAccounting]:
    """Resolve everything the steps read, through the same trusted path the sweep and the derivation use."""
    docs, results_dir = Path(cast("str", args.docs)), Path(cast("str", args.results))
    evidence_dir = Path(cast("str", args.evidence_dir))
    prepared = prepare_runner(args, role="main", root=root, module=_MODULE)
    results = load_results(results_dir / f"results_v{1}.json")
    runner = prepared.runner
    table = next(t for t in results.tables if t.record == "ManualRunRow")
    rows = table_from_csv(verify_artifact(runner.store, table.payload).read_text(encoding="utf-8"), ManualRunRow)
    inputs = _Inputs(
        root=root,
        docs=docs,
        results_dir=results_dir,
        evidence_dir=evidence_dir,
        study_file=Path(cast("str", args.study)),
        evaluation_file=Path(cast("str", args.evaluation)),
        runner=runner,
        manifest=prepared.context.manifest,
        store=runner.store,
        results=results,
        ordering=load_handoff(docs / "run_ordering_v1.json", RunOrdering),
        rule=load_handoff(docs / "representative_rule_v1.json", RepresentativeRule),
        rows=rows,
        workers=int(cast("int", args.workers)),
    )
    accounting = account_study(
        store=inputs.store, evidence_dir=evidence_dir, manifest=inputs.manifest, provenance=runner.provenance
    )
    return inputs, accounting


def _resimulation_step(
    inputs: _Inputs, scratch: Path, *, requested: bool, exploratory: bool
) -> tuple[AuditStep, tuple[ResimulatedRun, ...]]:
    """Re-simulate the frozen subset, or record that this audit did not."""
    started = time.perf_counter()
    if not requested:
        return _step("resimulation", "skipped by --no-resimulate", 0, [], started), ()
    try:
        subset = resimulation_subset(
            inputs.manifest, scenarios=inputs.runner.scenarios, trackers=tuple(inputs.runner.trackers)
        )
    except ValueError as error:
        # The sample is frozen over the study's own locked cases; an audit of anything else cannot form it.
        return _step("resimulation", "the frozen subset", 0, [f"{type(error).__name__}: {error}"], started), ()
    return check_resimulation(inputs, subset, scratch, exploratory=exploratory)


def _audit(args: argparse.Namespace) -> int:
    """Audit the committed evidence from this checkout and write the record once."""
    root = repository_root()
    output = Path(cast("str", args.output))
    targets = (output / f"reproduction_audit_v{AUDIT_VERSION}.json", output / f"reproduction_audit_v{AUDIT_VERSION}.md")
    for target in targets:
        if target.exists():
            msg = f"refusing to overwrite {target}: an audit is versioned, so write the next one beside it"
            raise FileExistsError(msg)
    if int(cast("int", args.workers)) < 1:
        msg = f"workers must be at least 1, got {args.workers}"
        raise ValueError(msg)
    inputs, accounting = _inputs(args, root)
    bundle, bundle_missing = _bundle(inputs)
    steps: list[AuditStep] = [
        check_checkout(inputs),
        check_sources(inputs),
        check_datasets(inputs),
        check_manifests(inputs),
        check_fits(inputs),
        check_payloads_and_metrics(inputs),
        check_aggregates(inputs),
        check_completeness(inputs, accounting, bundle_missing),
    ]
    with tempfile.TemporaryDirectory(prefix="arm-rc-ctrl-audit-") as scratch_name:
        scratch = Path(scratch_name)
        steps.append(check_figures(inputs, scratch))
        resimulation, outcomes = _resimulation_step(
            inputs, scratch, requested=bool(args.resimulate), exploratory=bool(args.exploratory)
        )
        steps.append(resimulation)
    steps.append(check_gates(root, run=bool(args.gates)))
    audit = ManualAudit(
        experiment=EXPERIMENT_LABEL,
        version=AUDIT_VERSION,
        created_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        checkout_commit=inputs.runner.provenance.project_commit,
        checkout_dirty=inputs.runner.provenance.project_dirty,
        validated_commits=tuple(cast("list[str]", args.validated_commit)),
        results_commit=inputs.results.provenance.project_commit,
        tolerances=DECLARED_TOLERANCES,
        steps=tuple(steps),
        resimulated=outcomes,
        bundle=bundle,
        n_checked=sum(step.checked for step in steps),
        n_failures=sum(len(step.failures) for step in steps),
        ok=all(step.ok for step in steps),
        command=inputs.runner.command,
        provenance=inputs.runner.provenance,
    )
    output.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(audit_to_json(audit), encoding="utf-8")
    targets[1].write_text(render_audit_markdown(audit), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": audit.ok,
                "checked": audit.n_checked,
                "failures": audit.n_failures,
                "steps": {step.name: ("pass" if step.ok else len(step.failures)) for step in audit.steps},
                "resimulated": len(audit.resimulated),
                "bitwise": sum(1 for run in audit.resimulated if run.bitwise),
                "output": str(targets[0]),
            },
            indent=2,
        )
    )
    return 0 if audit.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Audit the manual study's evidence from a clean checkout.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    audit = subparsers.add_parser("audit", help="verify, recompute and re-simulate, then write the audit record")
    for name, text in (
        ("--study", "the frozen study manifest"),
        ("--evaluation", "the evaluation configuration the study ran under"),
        ("--evidence-dir", "the directory of Git pointers the sweep wrote"),
        ("--results", "the directory of derived evidence"),
        ("--docs", "the experiment's documentation directory"),
        ("--output", "the directory the audit record is written to"),
    ):
        audit.add_argument(name, dest=name.removeprefix("--").replace("-", "_"), required=True, help=text)
    audit.add_argument("--workers", type=int, default=1, help="threads verifying and recomputing runs")
    audit.add_argument(
        "--validated-commit",
        action="append",
        default=[],
        help="a commit whose checks this audit applies; repeat for several",
    )
    audit.add_argument("--gates", action="store_true", help="run the documented nox and pre-commit gates")
    audit.add_argument(
        "--no-resimulate",
        dest="resimulate",
        action="store_false",
        help="skip the frozen subset's re-simulation (for a quick verification pass)",
    )
    audit.add_argument("--exploratory", action="store_true", help="tolerate a dirty worktree")
    args = parser.parse_args(argv)
    args.argv = argv
    return _audit(args)


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
