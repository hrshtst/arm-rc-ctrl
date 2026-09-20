# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-011: the clean-checkout audit of the manual study's evidence.

An audit is only useful if it fails when the evidence is wrong and says so
without stopping: a corrupt payload, a missing manifest or an edited document
must each be retained as a failure, with the record still written, so one
problem never hides the others. The audit is exercised here over the fixture
study's own evidence, including one re-simulation of a narrowed frozen subset.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from arm_rc_ctrl.data.manual import ManualDatasetRecord
from arm_rc_ctrl.data.records import load_record, write_record
from arm_rc_ctrl.experiments import manual_audit
from arm_rc_ctrl.experiments.manual_audit import AUDIT_STEPS, AUDIT_VERSION, ManualAudit, load_audit
from arm_rc_ctrl.experiments.manual_evaluation import load_manual_pointer, manual_pointer_name
from arm_rc_ctrl.experiments.manual_fixture import MANUAL_CONFIGURATION as CONFIGURATION
from arm_rc_ctrl.experiments.manual_fixture import MANUAL_SCENARIOS
from arm_rc_ctrl.experiments.manual_resimulation import ResimulationSubset
from arm_rc_ctrl.experiments.manual_results import (
    RESULT_DOCUMENTS,
    ManualRunRow,
    load_results,
    render_results_markdown,
    results_to_json,
    table_from_csv,
    table_to_csv,
)
from arm_rc_ctrl.metrics.recovery import SATURATION_BOUND
from arm_rc_ctrl.provenance import ArtifactReference, sha256_bytes, verify_artifact

if TYPE_CHECKING:
    from collections.abc import Callable

    from arm_rc_ctrl.experiments.manual_evaluation import ManualPairRecord
    from arm_rc_ctrl.experiments.manual_fixture import ManualStudyEvidence
    from arm_rc_ctrl.experiments.manual_study import StudyManifest

AUDIT = f"reproduction_audit_v{AUDIT_VERSION}.json"


def _stub_bundle(docs: Path) -> None:
    """The bundle's acquisition and training parts, which this fixture study has no versioned copies of."""
    for name in ("acquisition_readiness_v1.md", "numerical_validation_v1.json", "execution_account_v1.md"):
        (docs / name).write_text("fixture\n", encoding="utf-8")


def _argv(
    study: ManualStudyEvidence,
    derived: Path,
    docs: Path,
    output: Path,
    *,
    resimulate: bool = False,
    evidence: Path | None = None,
) -> list[str]:
    argv = [
        "audit",
        "--study",
        str(study.f.manifest_file),
        "--evaluation",
        str(study.evaluation),
        "--evidence-dir",
        str(study.evidence if evidence is None else evidence),
        "--results",
        str(derived),
        "--docs",
        str(docs),
        "--output",
        str(output),
        "--validated-commit",
        "abc123def456",
        "--exploratory",
    ]
    return argv if resimulate else [*argv, "--no-resimulate"]


@pytest.fixture
def docs(study: ManualStudyEvidence) -> Path:
    """A documentation directory inside the fixture root, holding the freezes this study was run under."""
    target = study.f.root / "docs" / "experiments" / "task_1a_manual_demonstration"
    target.mkdir(parents=True, exist_ok=True)
    for source in (study.ordering, study.rule, study.schema_v3, study.f.manifest_file):
        copy = target / source.name
        if source != copy:
            shutil.copyfile(source, copy)
    _stub_bundle(target)
    return target


def _audit(
    study: ManualStudyEvidence,
    derived: Path,
    docs: Path,
    tmp_path: Path,
    patch_manual: Callable[..., None],
    **kwargs: bool | Path | None,
) -> tuple[int, ManualAudit]:
    """Run the audit and return its status beside the record it wrote."""
    patch_manual(study, manual_audit)
    output = tmp_path / "audit"
    status = manual_audit.main(_argv(study, derived, docs, output, **cast("Any", kwargs)))
    return status, load_audit(output / AUDIT)


def _failures(record: ManualAudit, step: str) -> tuple[str, ...]:
    return next(s for s in record.steps if s.name == step).failures


# --- a sound study ---------------------------------------------------------------------------------


def test_every_property_checks_out_except_what_this_fixture_study_lacks(
    study: ManualStudyEvidence, derived: Path, docs: Path, tmp_path: Path, patch_manual: Callable[..., None]
) -> None:
    """Sources, manifests, fits, payloads, metrics, aggregates and figures check out over the fixture's evidence.

    Three steps must fail, and each says why: this study evaluated four of the
    manifest's models, its derivation was exploratory, and the fixture keeps no
    committed raw records behind its demonstrations. An audit that passed here
    would be an audit that checks nothing.
    """
    status, record = _audit(study, derived, docs, tmp_path, patch_manual)
    failed = [step.name for step in record.steps if not step.ok]
    assert failed == ["checkout", "datasets", "completeness"]
    assert [step.name for step in record.steps if step.ok] == [
        "sources",
        "manifests",
        "fits",
        "payloads_and_metrics",
        "aggregates",
        "figures",
        "resimulation",
        "gates",
    ]
    assert any("exploratory" in failure for failure in _failures(record, "checkout"))
    assert any("incomplete" in failure for failure in _failures(record, "completeness"))
    assert status == 1
    assert record.n_checked > 0
    assert (
        (tmp_path / "audit" / f"reproduction_audit_v{AUDIT_VERSION}.md")
        .read_text(encoding="utf-8")
        .startswith("# Task 1-a")
    )


def test_the_audit_names_the_commit_that_produced_the_evidence_and_the_ones_that_checked_it(
    study: ManualStudyEvidence, derived: Path, docs: Path, tmp_path: Path, patch_manual: Callable[..., None]
) -> None:
    """A reader must be able to tell which code produced the evidence and which code audited it."""
    _, record = _audit(study, derived, docs, tmp_path, patch_manual)
    assert record.results_commit
    assert record.bundle
    assert {item.kind for item in record.bundle} >= {"results", "protocol", "stored table", "configuration"}
    bound = {study.f.manifest.bank.path, study.f.manifest.scenario.path}
    assert bound <= {item.location for item in record.bundle}, "the bundle cites the paths the study binds"
    assert all(not item.location.startswith("/") for item in record.bundle), "a record never carries a machine path"


def test_the_declared_tolerances_are_recorded_with_the_verdict(
    study: ManualStudyEvidence, derived: Path, docs: Path, tmp_path: Path, patch_manual: Callable[..., None]
) -> None:
    """Tolerances are part of the claim: bitwise arrays, exact metrics, exact aggregates."""
    _, record = _audit(study, derived, docs, tmp_path, patch_manual)
    assert record.tolerances.arrays_bitwise is True
    assert record.tolerances.metric_abs_tol == 0.0
    assert record.tolerances.aggregates_exact is True


# --- evidence that does not agree with itself --------------------------------------------------------


def test_a_corrupt_run_payload_is_retained_as_a_failure(
    study: ManualStudyEvidence, derived: Path, docs: Path, tmp_path: Path, patch_manual: Callable[..., None]
) -> None:
    """The audit records the run that no longer matches its digest, finishes, and fails."""
    arrays = study.arrays_file(study.pair(f"{CONFIGURATION}/M10"))
    backup = tmp_path / "arrays.npz"
    shutil.copyfile(arrays, backup)
    try:
        data = arrays.read_bytes()
        arrays.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))
        status, record = _audit(study, derived, docs, tmp_path, patch_manual)
    finally:
        shutil.copyfile(backup, arrays)
    assert status == 1
    assert not record.ok
    assert any("no longer matches" in failure for failure in _failures(record, "payloads_and_metrics"))
    assert (tmp_path / "audit" / AUDIT).is_file(), "the record is written even when the audit fails"


def test_a_missing_manifest_is_retained_as_a_failure(
    study: ManualStudyEvidence, derived: Path, docs: Path, tmp_path: Path, patch_manual: Callable[..., None]
) -> None:
    """Evidence that is gone is reported by every step that expected it, and the audit still completes."""
    pointer = study.evidence / manual_pointer_name("model", f"{CONFIGURATION}/S/D01")
    body = pointer.read_bytes()
    pointer.unlink()
    try:
        status, record = _audit(study, derived, docs, tmp_path, patch_manual)
    finally:
        pointer.write_bytes(body)
    assert status == 1
    assert _failures(record, "sources"), "the evidence digest no longer matches the results index"
    assert _failures(record, "payloads_and_metrics"), "the rows of the missing model have no pairs"


def test_an_edited_summary_document_is_caught_by_its_digest(
    study: ManualStudyEvidence, derived: Path, docs: Path, tmp_path: Path, patch_manual: Callable[..., None]
) -> None:
    """A committed document that changed after the derivation no longer matches the index that cites it."""
    document = derived / RESULT_DOCUMENTS["arm_summary"]
    original = document.read_text(encoding="utf-8")
    document.write_text(original.replace("feasible-best", "feasible-worst", 1), encoding="utf-8")
    try:
        status, record = _audit(study, derived, docs, tmp_path, patch_manual)
    finally:
        document.write_text(original, encoding="utf-8")
    assert status == 1
    assert any("arm_summary" in failure for failure in _failures(record, "sources"))
    assert _failures(record, "aggregates"), "the rebuilt summaries are not the edited ones"


def test_an_existing_audit_record_is_never_overwritten(
    study: ManualStudyEvidence, derived: Path, docs: Path, tmp_path: Path, patch_manual: Callable[..., None]
) -> None:
    """An audit is versioned like every other piece of evidence here."""
    _audit(study, derived, docs, tmp_path, patch_manual)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        manual_audit.main(_argv(study, derived, docs, tmp_path / "audit"))


# --- what the audit must guarantee (owner review, 2026-09-20) ------------------------------------


def _altered_table(
    study: ManualStudyEvidence, derived: Path, tmp_path: Path, mutate: Callable[[list[ManualRunRow]], None]
) -> Path:
    """A copy of the derived results whose per-run table was altered and re-indexed, as a careful edit would be.

    The index is rebuilt around the altered table, so every digest agrees with
    itself: only a recomputation from the runs themselves can refuse it.
    """
    target = tmp_path / "altered"
    shutil.copytree(derived, target)
    results = load_results(target / "results_v1.json")
    table = next(t for t in results.tables if t.record == "ManualRunRow")
    rows = list(table_from_csv(verify_artifact(study.f.store, table.payload).read_text(encoding="utf-8"), ManualRunRow))
    mutate(rows)
    body = table_to_csv(rows, ManualRunRow).encode("utf-8")
    digest = sha256_bytes(body)
    uri = f"armrc://reports/task_1a_manual_v1/results/altered-{digest[:12]}.csv"
    study.f.store.path(uri, mode="write").write_bytes(body)
    payload = ArtifactReference(uri=uri, sha256=digest, size=len(body))
    altered = replace(table, payload=payload, n_rows=len(rows))
    rebuilt = replace(results, tables=tuple(altered if t.record == "ManualRunRow" else t for t in results.tables))
    (target / "results_v1.json").write_text(results_to_json(rebuilt) + "\n", encoding="utf-8")
    (target / "results_v1.md").write_text(render_results_markdown(rebuilt), encoding="utf-8")
    return target


def _first_simulated(rows: list[ManualRunRow]) -> int:
    return next(i for i, row in enumerate(rows) if row.status in {"completed", "infeasible"})


ALTERATIONS: dict[str, tuple[str, object]] = {
    "endpoint error": ("final_endpoint_error_m", 0.5),
    "dwell duration": ("dwell_final_s", 0.25),
    "torque rms": ("torque_rms_nm", 1.5),
    "saturation fraction": ("saturation_fraction", 0.75),
    "run digest": ("run_sha256", "f" * 64),
    "training sources": ("sources", ("processed-invented",)),
    "scenario class": ("scenario_class", "force"),
}
"""Row fields the first audit never recomputed or compared, so an altered table passed it."""


@pytest.mark.parametrize("field_name", sorted(ALTERATIONS))
def test_every_row_field_is_compared_with_a_trusted_source(
    study: ManualStudyEvidence,
    derived: Path,
    docs: Path,
    tmp_path: Path,
    patch_manual: Callable[..., None],
    field_name: str,
) -> None:
    """A table row that disagrees with the manifest or its own trajectories is refused, whichever field it is."""
    name, value = ALTERATIONS[field_name]

    def mutate(rows: list[ManualRunRow]) -> None:
        index = _first_simulated(rows)
        rows[index] = replace(rows[index], **{name: value})

    altered = _altered_table(study, derived, tmp_path, mutate)
    status, record = _audit(study, altered, docs, tmp_path, patch_manual)
    assert status == 1
    assert any(name in failure for failure in _failures(record, "payloads_and_metrics")), _failures(
        record, "payloads_and_metrics"
    )


def test_a_fabricated_raw_record_is_refused(
    study: ManualStudyEvidence, derived: Path, docs: Path, tmp_path: Path, patch_manual: Callable[..., None]
) -> None:
    """A raw record is loaded and checked against its payload, not merely found on disk."""
    raw = study.f.root / "data" / "records" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for demonstration in study.f.manifest.demonstrations:
        record = load_record(study.f.root / demonstration.dataset.record, ManualDatasetRecord)
        for source in record.artifact.origin.sources:
            path = raw / f"{source}.toml"
            path.write_text("this is not a record\n", encoding="utf-8")
            written.append(path)
    try:
        status, record_json = _audit(study, derived, docs, tmp_path, patch_manual)
    finally:
        for path in written:
            path.unlink()
    assert status == 1
    failures = _failures(record_json, "datasets")
    assert any("raw record" in failure for failure in failures), failures


def test_a_corrupt_manifest_is_recorded_rather_than_crashing_the_audit(
    study: ManualStudyEvidence, derived: Path, docs: Path, tmp_path: Path, patch_manual: Callable[..., None]
) -> None:
    """A manifest whose payload no longer matches its pointer is a recorded failure, and the record is still written."""
    pointer = load_manual_pointer(study.evidence / manual_pointer_name("model", f"{CONFIGURATION}/M10"))
    manifest = study.f.store.path(pointer.payload.uri, mode="read")
    body = manifest.read_bytes()
    manifest.write_bytes(body.replace(b'"n_pairs"', b'"n_Pairs"', 1))
    try:
        status, record = _audit(study, derived, docs, tmp_path, patch_manual)
    finally:
        manifest.write_bytes(body)
    assert (tmp_path / "audit" / AUDIT).is_file(), "the audit record is written even when a manifest is corrupt"
    assert status == 1
    assert any(f"{CONFIGURATION}/M10" in failure for failure in _failures(record, "manifests"))
    dependent = next(s for s in record.steps if s.name == "payloads_and_metrics")
    assert dependent.failures or dependent.unavailable, "a step that cannot read the evidence says so"


def test_unreadable_derived_evidence_leaves_every_dependent_step_marked_unavailable(
    study: ManualStudyEvidence, derived: Path, docs: Path, tmp_path: Path, patch_manual: Callable[..., None]
) -> None:
    """An index that cannot be read is reported as such, with every step it feeds listed and marked unavailable."""
    target = tmp_path / "unreadable"
    shutil.copytree(derived, target)
    (target / "results_v1.json").write_text("this is not an index\n", encoding="utf-8")
    status, record = _audit(study, target, docs, tmp_path, patch_manual)
    assert status == 1
    assert [step.name for step in record.steps] == list(AUDIT_STEPS)
    assert [step.name for step in record.steps if step.unavailable] == [n for n in AUDIT_STEPS if n != "gates"]
    assert all("the derived evidence cannot be read" in failure for failure in _failures(record, "aggregates"))
    assert record.results_commit == "", "no index was read, so no generating commit is claimed"


def test_an_index_missing_a_stored_table_is_recorded_rather_than_crashing_the_audit(
    study: ManualStudyEvidence, derived: Path, docs: Path, tmp_path: Path, patch_manual: Callable[..., None]
) -> None:
    """An index that names no comparison table fails the step that needed it; the other steps still report."""
    target = tmp_path / "without_contrasts"
    shutil.copytree(derived, target)
    results = load_results(target / "results_v1.json")
    kept = tuple(table for table in results.tables if table.record != "ManualContrastRow")
    (target / "results_v1.json").write_text(results_to_json(replace(results, tables=kept)) + "\n", encoding="utf-8")
    status, record = _audit(study, target, docs, tmp_path, patch_manual)
    assert status == 1
    assert [step.name for step in record.steps] == list(AUDIT_STEPS)
    step = next(s for s in record.steps if s.name == "aggregates")
    assert step.unavailable
    assert any("ManualContrastRow" in failure for failure in step.failures), step.failures
    assert next(s for s in record.steps if s.name == "payloads_and_metrics").ok, "one step's gap is not another's"


def _measurement_tampered(
    study: ManualStudyEvidence, tmp_path: Path, label: str, mutate: Callable[[dict[str, Any]], None]
) -> tuple[Path, tuple[str, str]]:
    """A copy of the evidence whose one manifest records a different measurement, under its own correct digest.

    The manifest stays self-consistent: its verdict, criteria and counts are
    untouched, so every check that compares a record with itself still passes
    and only a recomputation from the run's arrays can refuse it.
    """
    evidence = tmp_path / "measurement_evidence"
    shutil.copytree(study.evidence, evidence)
    path = evidence / manual_pointer_name("model", label)
    pointer = load_manual_pointer(path)
    mapping = cast("dict[str, Any]", json.loads(verify_artifact(study.f.store, pointer.payload).read_text("utf-8")))
    pair = next(p for p in mapping["pairs"] if p.get("outcome"))
    mutate(cast("dict[str, Any]", pair["outcome"]))
    body = (json.dumps(mapping, sort_keys=True) + "\n").encode("utf-8")
    digest = sha256_bytes(body)
    uri = f"armrc://reports/task_1a_manual_v1/tampered/manifest-{digest[:12]}.json"
    study.f.store.path(uri, mode="write").write_bytes(body)
    path.unlink()  # a copy in a scratch directory; the study's own pointers are untouched
    write_record(path, replace(pointer, payload=ArtifactReference(uri=uri, sha256=digest, size=len(body))))
    return evidence, (cast("str", pair["scenario_id"]), cast("str", pair["tracker"]))


def _same_side_saturation(outcome: dict[str, Any]) -> float:
    """Another saturation fraction on the same side of the eligibility bound, so no verdict changes.

    A fraction that crossed the bound would flip the recorded criterion and be
    caught by the verdict checks instead, which would not exercise the
    measurement at all.
    """
    original = float(cast("float", outcome["saturation_fraction"]))
    if original > SATURATION_BOUND:
        return 1.0 if original != 1.0 else 0.9
    return 0.0 if original != 0.0 else SATURATION_BOUND


MEASUREMENTS: dict[str, tuple[tuple[str, ...], str, Callable[[dict[str, Any]], object]]] = {
    "dwell duration": (("dwell", "final_duration_s"), "dwell_final_s", lambda _outcome: 0.25),
    "saturation fraction": (("saturation_fraction",), "saturation_fraction", _same_side_saturation),
    "torque rms": (("torque_rms",), "torque_rms_nm", lambda _outcome: 999.0),
}
"""Outcome measurements the manifest records and the row carries, which no verdict or count constrains."""


@pytest.mark.parametrize("measurement", sorted(MEASUREMENTS))
def test_a_measurement_altered_in_manifest_and_table_alike_is_still_refused(
    study: ManualStudyEvidence,
    derived: Path,
    docs: Path,
    tmp_path: Path,
    patch_manual: Callable[..., None],
    measurement: str,
) -> None:
    """Agreement between a manifest and a table proves nothing: these measurements come from the arrays."""
    path, column, chosen = MEASUREMENTS[measurement]
    label = f"{CONFIGURATION}/M10"
    altered_to: list[object] = []

    def alter_outcome(outcome: dict[str, Any]) -> None:
        holder = outcome
        for name in path[:-1]:
            holder = cast("dict[str, Any]", holder[name])
        value = chosen(outcome)
        assert value != holder[path[-1]], "the alteration must change the recorded measurement"
        holder[path[-1]] = value
        altered_to.append(value)

    evidence, (scenario_id, tracker) = _measurement_tampered(study, tmp_path, label, alter_outcome)

    def alter_table(rows: list[ManualRunRow]) -> None:
        index = next(
            i
            for i, row in enumerate(rows)
            if (row.model_label, row.scenario_id, row.tracker) == (label, scenario_id, tracker)
        )
        rows[index] = replace(rows[index], **{column: altered_to[0]})

    altered = _altered_table(study, derived, tmp_path, alter_table)
    status, record = _audit(study, altered, docs, tmp_path, patch_manual, evidence=evidence)
    assert status == 1
    failures = _failures(record, "payloads_and_metrics")
    assert any(path[0] in failure for failure in failures), failures


def test_a_failing_gate_is_recorded_with_the_line_that_names_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nox names the failing session on stderr, so a record that quoted stdout alone said nothing useful."""

    def failed(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args=["uv", "run", "--locked", "nox"],
            returncode=1,
            stdout=b"3239 passed, 1 skipped\n100% tests passed\n",
            stderr=b"nox > Session lint was successful.\nnox > Session type_check failed.\n",
        )

    monkeypatch.setattr(manual_audit.subprocess, "run", failed)
    step = manual_audit.check_gates(Path("/nonexistent"), run=True)
    assert not step.ok
    assert any("type_check failed" in failure for failure in step.failures), step.failures


# --- re-simulation -----------------------------------------------------------------------------------


def test_the_frozen_subset_is_re_simulated_and_compared_run_by_run(
    study: ManualStudyEvidence,
    derived: Path,
    docs: Path,
    tmp_path: Path,
    patch_manual: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-simulation runs in a scratch store and must reproduce the stored arrays bitwise."""

    def narrowed(manifest: StudyManifest, **_kwargs: object) -> ResimulationSubset:
        model = next(e for e in manifest.entries if e.label == f"{CONFIGURATION}/S/D01")
        return ResimulationSubset(models=(model,), scenarios=(MANUAL_SCENARIOS[0],), trackers=("pd_v2",), n_banks=1)

    monkeypatch.setattr(manual_audit, "resimulation_subset", narrowed)
    status, record = _audit(study, derived, docs, tmp_path, patch_manual, resimulate=True)
    assert [run.arm for run in record.resimulated] == ["rc", "replay"]
    assert all(run.bitwise for run in record.resimulated), [run.max_abs_deviation for run in record.resimulated]
    assert all(run.committed_status == run.rebuilt_status for run in record.resimulated)
    assert next(s for s in record.steps if s.name == "resimulation").ok
    assert status == 1, "the fixture study is partial, which the completeness step reports"


def test_a_failed_re_simulation_keeps_its_payloads_and_the_record_says_where(
    study: ManualStudyEvidence,
    derived: Path,
    docs: Path,
    tmp_path: Path,
    patch_manual: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that did not reproduce is the one run worth keeping, so the scratch store is retained and cited."""

    def narrowed(manifest: StudyManifest, **_kwargs: object) -> ResimulationSubset:
        model = next(e for e in manifest.entries if e.label == f"{CONFIGURATION}/S/D01")
        return ResimulationSubset(models=(model,), scenarios=(MANUAL_SCENARIOS[0],), trackers=("pd_v2",), n_banks=1)

    monkeypatch.setattr(manual_audit, "resimulation_subset", narrowed)

    def nothing_rebuilt(*_args: object, **_kwargs: object) -> dict[tuple[str, str, str], ManualPairRecord]:
        return {}

    monkeypatch.setattr(manual_audit, "_index_rebuilt", nothing_rebuilt)
    status, record = _audit(study, derived, docs, tmp_path, patch_manual, resimulate=True)
    assert status == 1
    step = next(s for s in record.steps if s.name == "resimulation")
    assert step.failures
    assert record.retained_resimulation, "the record cites where the re-simulated runs were kept"
    retained = study.f.store.path(record.retained_resimulation, mode="read")
    assert any(retained.rglob("run.json")), "the retained payloads are the runs themselves, not just a note"


def test_a_run_that_did_not_reproduce_cites_a_retained_payload_that_exists(
    study: ManualStudyEvidence,
    derived: Path,
    docs: Path,
    tmp_path: Path,
    patch_manual: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A citation that does not resolve is no better than a deleted payload, so each one is opened here."""

    def narrowed(manifest: StudyManifest, **_kwargs: object) -> ResimulationSubset:
        model = next(e for e in manifest.entries if e.label == f"{CONFIGURATION}/S/D01")
        return ResimulationSubset(models=(model,), scenarios=(MANUAL_SCENARIOS[0],), trackers=("pd_v2",), n_banks=1)

    monkeypatch.setattr(manual_audit, "resimulation_subset", narrowed)
    # The audit's own indexing, reused with one digest disturbed, so the comparison reaches its retained branch.
    indexed = manual_audit._index_rebuilt  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    def mismatched(inputs: object, rebuilt_store: object) -> dict[tuple[str, str, str], ManualPairRecord]:
        rebuilt = cast("Any", indexed)(inputs, rebuilt_store)
        return {
            key: replace(pair, run=replace(pair.run, arrays_sha256="0" * 64))
            for key, pair in cast("dict[tuple[str, str, str], ManualPairRecord]", rebuilt).items()
            if pair.run is not None
        }

    def deviation(*_args: object, **_kwargs: object) -> float:
        return 1.5

    monkeypatch.setattr(manual_audit, "_index_rebuilt", mismatched)
    monkeypatch.setattr(manual_audit, "_max_deviation", deviation)
    _, record = _audit(study, derived, docs, tmp_path, patch_manual, resimulate=True)
    assert record.retained_resimulation
    assert [run.bitwise for run in record.resimulated] == [False, False]
    for run in record.resimulated:
        assert run.retained, "a run that did not reproduce cites the payload it produced"
        assert run.retained != record.retained_resimulation, "the citation names the run, not just the directory"
        assert study.f.store.path(run.retained, mode="read").is_file(), run.retained


def test_a_re_simulation_whose_committed_pair_is_missing_is_recorded_and_keeps_its_runs(
    study: ManualStudyEvidence,
    derived: Path,
    docs: Path,
    tmp_path: Path,
    patch_manual: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nothing to compare against, the comparison is unavailable and the re-simulated runs are kept."""
    evidence = tmp_path / "unpointed_evidence"
    shutil.copytree(study.evidence, evidence)
    (evidence / manual_pointer_name("model", f"{CONFIGURATION}/S/D01")).unlink()

    def narrowed(manifest: StudyManifest, **_kwargs: object) -> ResimulationSubset:
        model = next(e for e in manifest.entries if e.label == f"{CONFIGURATION}/S/D01")
        return ResimulationSubset(models=(model,), scenarios=(MANUAL_SCENARIOS[0],), trackers=("pd_v2",), n_banks=1)

    monkeypatch.setattr(manual_audit, "resimulation_subset", narrowed)
    status, record = _audit(study, derived, docs, tmp_path, patch_manual, resimulate=True, evidence=evidence)
    assert (tmp_path / "audit" / AUDIT).is_file(), "the record is written even when a comparison cannot be made"
    assert status == 1
    step = next(s for s in record.steps if s.name == "resimulation")
    assert step.failures
    assert any(run.committed_status == "unavailable" for run in record.resimulated), [
        (run.label, run.committed_status) for run in record.resimulated
    ]
    assert record.retained_resimulation, "the runs it could not compare are kept, not discarded"
