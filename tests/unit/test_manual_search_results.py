# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MS-007: the comparison's derivation and audit, over real fixture evidence.

A small comparison is built over the fixture study: one frozen configuration
drawn from a real search trial, its D01 replay bank and the four arms paired
against it (S/D01, M10, R10/D01, C10/D01). Each unit is evaluated by the real
worker function and finalized by the real parent verification, and its
pointers are published. The derivation must account for all of it, and the
audit must pass on it untouched and fail, in the step that owns it, on each
kind of tampering.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.experiments import manual_comparison_run, manual_evaluation, manual_search
from arm_rc_ctrl.experiments.manual_comparison_run import (
    ComparisonContext,
    comparison_status,
    comparison_units,
    evaluate_unit,
    publish_pointers,
    render_status,
    unit_directory,
    verified_unit,
)
from arm_rc_ctrl.experiments.manual_fixture import ManualFixture, ManualStudyEvidence, manual_narrowed
from arm_rc_ctrl.experiments.manual_results import ManualRunRow, table_from_csv
from arm_rc_ctrl.experiments.manual_sampled import SampledPoint, sampled_configuration
from arm_rc_ctrl.experiments.manual_search import load_manual_search
from arm_rc_ctrl.experiments.manual_search_audit import AUDIT_STEPS, audit_search_comparison
from arm_rc_ctrl.experiments.manual_search_freeze import (
    FrozenConfiguration,
    ManualSearchFreeze,
    read_freeze,
    write_freeze,
)
from arm_rc_ctrl.experiments.manual_search_results import (
    DOCUMENTS,
    M3MAN_EVIDENCE,
    account_comparison,
    comparison_scope,
    derive_search_results,
    load_accounting,
    load_search_results,
    stored_rows,
)
from arm_rc_ctrl.experiments.manual_search_run import (
    BudgetLedger,
    SearchInputs,
    evaluate_trial,
    reserve_trial,
    write_record,
)
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import ArtifactUri

if TYPE_CHECKING:
    from arm_rc_ctrl.experiments.manual_search import ManualSearchProtocol

POINT = SampledPoint(
    n_neurons=100, spectral_radius=1.0, sparsity=0.9, leak_rate=0.05, input_scaling=0.3, alpha_0=0.01, warmup_s=0.0
)
LABELS = ("D01", "S/D01", "M10", "R10/D01", "C10/D01")
"""The units of the fixture comparison: one bank and the four arms paired against it."""


class _FixtureContext(ComparisonContext):
    """A context over the fixture study: a real runner, the freeze file's one configuration, five units."""

    def __init__(self, protocol: ManualSearchProtocol, f: ManualFixture, freeze_file: Path) -> None:
        configuration = sampled_configuration(protocol, POINT, trial=0)
        args = argparse.Namespace(
            study=str(f.manifest_file),
            evaluation=str(protocol.comparison.evaluation),
            exploratory=True,
            argv=["derive"],
        )
        prepared = manual_evaluation.prepare_runner(args, role="main", root=f.root, sampled=(configuration,))
        self.protocol = protocol
        self.protocol_file = f.root / "configs" / "studies" / "fixture_search.toml"
        self.freeze_file = freeze_file
        self.store = f.store
        self.freeze = read_freeze(freeze_file)
        self.freeze_sha256 = sha256_file(freeze_file)
        self.configurations = {1: configuration}
        self.manifest = prepared.context.manifest
        self.runner = prepared.runner
        self.units = tuple(unit for unit in comparison_units(3) if unit.rank == 1 and unit.label in LABELS)


def _any_scope(*args: object) -> list[str]:
    """The scope check replaced: the fixture's evaluation is narrowed on purpose."""
    del args
    return []


def _freeze(
    protocol: ManualSearchProtocol, *, fit_identity: str, evidence_identity: str, evidence: str
) -> ManualSearchFreeze:
    configuration = sampled_configuration(protocol, POINT, trial=0)
    chosen = FrozenConfiguration(
        rank=1,
        trial=0,
        configuration=configuration.label,
        point=POINT,
        score=1.0,
        successes=2,
        runs=2,
        fit_identity=fit_identity,
        evidence_identity=evidence_identity,
        evidence=evidence,
    )
    return ManualSearchFreeze(
        schema=1,
        protocol_sha256="0" * 64,
        study="manual-esn-search-v1",
        label="highest nominal scores",
        order="descending_score_then_ascending_trial",
        n_configurations=3,
        trial_cap=1,
        finalized=1,
        scored=1,
        failed=0,
        score_counts={"1.0": 1},
        stopped=("the 1-trial cap is spent (1 trials)",),
        ledger=BudgetLedger(trials=1, seconds=10.0, stored_bytes=1000),
        chosen=(chosen,),
        shortfall=2,
    )


class _Built:
    """The fixture comparison: its context, protocol, pointers, status and derived results."""

    def __init__(self, context: _FixtureContext, protocol: ManualSearchProtocol, root: Path, work: Path) -> None:
        self.context = context
        self.protocol = protocol
        self.root = root
        self.evidence_dir = work / "evidence"
        self.status_file = work / "status_v1.json"
        self.results_dir = work / "results"


@pytest.fixture(scope="module")
def built(manual_fixture: ManualFixture, tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Built]:
    """Search trial 0, a freeze of it, the five units evaluated and verified, pointers, status and results."""
    patch = pytest.MonkeyPatch()
    try:
        f = manual_fixture
        for name, value in f.env.items():
            patch.setenv(name, value)
        patch.setattr(manual_search, "repository_root", lambda: f.root)
        patch.setattr(manual_evaluation, "repository_root", lambda: f.root)
        patch.setattr(manual_evaluation, "evaluation_scenarios", manual_narrowed)
        patch.setattr(manual_comparison_run, "comparison_scope_mismatches", _any_scope)
        work = tmp_path_factory.mktemp("search-results")
        study = work / "study"
        study.mkdir()
        evidence = ManualStudyEvidence(f, study)
        protocol = load_manual_search(_fixture_protocol(f, evidence.evaluation))
        trial = evaluate_trial(
            protocol,
            trial=0,
            point=POINT,
            scenarios=("nominal",),
            root=f.root,
            argv=["evaluate-trial"],
            exploratory=True,
        )
        assert trial.evidence is not None
        reservation = reserve_trial(protocol, SearchInputs(protocol, root=f.root, exploratory=True), POINT, trial=0)
        freeze_file = work / "selection_v1.json"
        write_freeze(
            freeze_file,
            _freeze(
                protocol,
                fit_identity=reservation.fit_identity,
                evidence_identity=reservation.evidence_identity,
                evidence=trial.evidence.uri,
            ),
        )
        context = _FixtureContext(protocol, f, freeze_file)
        for unit in context.units:
            result = evaluate_unit(
                protocol,
                freeze_file=freeze_file,
                freeze_sha256=sha256_file(freeze_file),
                number=unit.number,
                root=f.root,
                argv=["evaluate-unit"],
                exploratory=True,
            )
            unit_reservation = context.reservation(unit)
            outcome = verified_unit(context, unit, unit_reservation, result, seconds=1.0)
            directory = unit_directory(f.store, unit)
            write_record(directory / "reservation.json", unit_reservation)
            write_record(directory / "outcome.json", outcome)
        built = _Built(context, protocol, f.root, work)
        publish_pointers(context, built.evidence_dir)
        status = comparison_status(context)
        built.status_file.write_text(json.dumps(to_mapping(status), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        built.status_file.with_suffix(".md").write_text(render_status(status), encoding="utf-8")
        derive_search_results(
            context,
            evidence_dir=built.evidence_dir,
            status_file=built.status_file,
            output=built.results_dir,
            root=f.root,
        )
        yield built
    finally:
        patch.undo()


def _fixture_protocol(f: ManualFixture, evaluation: Path) -> Path:
    """The committed protocol re-pointed at the fixture study (the conftest fixture's own construction)."""
    filters = f.root / "configs" / "evaluations" / "fixture_filters.toml"
    filters.parent.mkdir(parents=True, exist_ok=True)
    filters.write_text(
        'name = "fixture-filters"\ntracker = "../controllers/task_1a_pd_v2.toml"\n\n'
        "[estimator]\nvelocity_cutoff_hz = 20.0\nacceleration_cutoff_hz = 8.0\nmax_dt_ratio = 3.0\n",
        encoding="utf-8",
    )
    body = (repository_root() / "configs/studies/manual_esn_search_v1.toml").read_text(encoding="utf-8")
    replacements = {
        'study = "../../docs/experiments/task_1a_manual_demonstration/study_manifest_v1.json"': (
            f'study = "{f.manifest_file}"'
        ),
        'model = "../models/esn_task_1a_v4.toml"': f'model = "{f.root / f.manifest.model.path}"',
        'scenario = "../tasks/task_1a_manual_v2.toml"': f'scenario = "{f.scenario_file}"',
        'filters = "../evaluations/task_1a_nominal_v4.toml"': f'filters = "{filters}"',
        "velocity_cutoff_hz = 29.980411525699598": "velocity_cutoff_hz = 20.0",
        "acceleration_cutoff_hz = 10.938122239871603": "acceleration_cutoff_hz = 8.0",
        'evaluation = "../evaluations/task_1a_manual_dev_v1.toml"': f'evaluation = "{evaluation}"',
    }
    for old, new in replacements.items():
        body = body.replace(old, new, 1)
    target = f.root / "configs" / "studies" / "fixture_search.toml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


@contextmanager
def _environment(built: _Built, monkeypatch: pytest.MonkeyPatch, f: ManualFixture) -> Generator[None]:
    for name, value in f.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(manual_search, "repository_root", lambda: f.root)
    monkeypatch.setattr(manual_evaluation, "repository_root", lambda: f.root)
    monkeypatch.setattr(manual_evaluation, "evaluation_scenarios", manual_narrowed)
    del built
    yield


@contextmanager
def _tampered(path: Path, content: bytes) -> Generator[None]:
    original = path.read_bytes()
    path.write_bytes(content)
    try:
        yield
    finally:
        path.write_bytes(original)


def _audit(built: _Built) -> dict[str, list[str]]:
    audit = audit_search_comparison(
        built.context,
        results_dir=built.results_dir,
        evidence_dir=built.evidence_dir,
        status_file=built.status_file,
        root=built.root,
    )
    assert tuple(step.name for step in audit.steps) == AUDIT_STEPS
    return {step.name: list(step.failures) for step in audit.steps if step.failures}


# --- the derivation ------------------------------------------------------------------------------


def test_the_derivation_accounts_for_every_unit(built: _Built) -> None:
    """Every planned model, bank and run is present, and every unit's counts are its rows'."""
    accounting = load_accounting(built.results_dir / DOCUMENTS["accounting"])
    per_unit = len(built.context.conditions(1).pairs)
    assert (accounting.n_models, accounting.n_replay_banks) == (4, 1)
    assert (accounting.n_rc_runs, accounting.n_replay_runs) == (4 * per_unit, per_unit)
    assert accounting.n_unavailable_runs == 0
    assert [unit.recorded for unit in accounting.units] == [unit.derived for unit in accounting.units]
    assert accounting.complete


def test_the_reuse_is_found_where_it_happened(built: _Built) -> None:
    """The M10 fit is the search trial's own; no nominal search run is a comparison run."""
    accounting = load_accounting(built.results_dir / DOCUMENTS["accounting"])
    assert [(fit.arm, fit.trial) for fit in accounting.reused_fits] == [("M10", 0)]
    assert accounting.search_nominal_runs == 2
    assert accounting.reused_search_runs == 0, "the search keyed its runs by its nominal-only scope"
    assert accounting.reused_banks == (), "the fixture's closed experiment holds no bank"


def test_a_bank_the_closed_experiment_holds_is_found_as_reused(
    built: _Built, manual_fixture: ManualFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identical conditions give an identical bank identity; the closed experiment's pointer names it too."""
    closed = manual_fixture.root / M3MAN_EVIDENCE
    closed.mkdir(parents=True, exist_ok=True)
    (pointer,) = sorted(built.evidence_dir.glob("replay__*.toml"))
    shutil.copy(pointer, closed / pointer.name)
    try:
        with _environment(built, monkeypatch, manual_fixture):
            results = load_search_results(built.results_dir / "results_v1.json")
            rows = table_from_csv(stored_rows(built.context.store, results, "ManualRunRow"), ManualRunRow)
            accounting = account_comparison(built.context, rows, n_models=4, n_banks=1, root=manual_fixture.root)
        assert [(bank.assignment, bank.pointer) for bank in accounting.reused_banks] == [("D01", pointer.name)]
        assert len(comparison_scope(built.context)[1]) == 1
    finally:
        shutil.rmtree(closed)


def test_derived_evidence_is_never_overwritten(built: _Built, manual_fixture: ManualFixture) -> None:
    """A derivation writes new files only: a second one beside the first is refused."""
    with pytest.raises(FileExistsError, match="overwrite"):
        derive_search_results(
            built.context,
            evidence_dir=built.evidence_dir,
            status_file=built.status_file,
            output=built.results_dir,
            root=manual_fixture.root,
        )


# --- the audit -----------------------------------------------------------------------------------


def test_the_audit_passes_on_untouched_evidence(
    built: _Built, manual_fixture: ManualFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every step passes on what the derivation wrote."""
    with _environment(built, monkeypatch, manual_fixture):
        assert _audit(built) == {}


def test_an_edited_summary_fails_the_aggregates_and_its_source_digest(
    built: _Built, manual_fixture: ManualFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed summary that no longer follows from the rows is refused twice: by digest and by rebuild."""
    path = built.results_dir / DOCUMENTS["arm_summary"]
    lines = path.read_text(encoding="utf-8").splitlines()
    header, first = lines[0].split(","), lines[1].split(",")
    column = header.index("successes")
    first[column] = str(int(first[column]) + 1)
    edited = "\n".join([lines[0], ",".join(first), *lines[2:]]) + "\n"
    with _environment(built, monkeypatch, manual_fixture), _tampered(path, edited.encode("utf-8")):
        failures = _audit(built)
    assert set(failures) >= {"sources", "aggregates"}


def test_an_edited_unit_outcome_fails_the_units_and_the_completeness(
    built: _Built, manual_fixture: ManualFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A unit claiming another breakdown than its manifest is refused where the unit records are checked."""
    unit = built.context.units[1]
    path = unit_directory(manual_fixture.store, unit) / "outcome.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["completed"]:
        data["completed"], data["infeasible"] = data["completed"] - 1, data["infeasible"] + 1
    else:
        data["completed"], data["infeasible"] = 1, data["infeasible"] - 1
    with _environment(built, monkeypatch, manual_fixture), _tampered(path, json.dumps(data).encode("utf-8")):
        failures = _audit(built)
    assert "units" in failures
    assert "completeness" in failures


def test_an_edited_status_fails_the_sources_and_the_units(
    built: _Built, manual_fixture: ManualFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A status that is not what the unit records give is refused by digest and by re-rendering."""
    edited = built.status_file.read_text(encoding="utf-8").replace(
        '"finished_configurations": 1', '"finished_configurations": 3'
    )
    assert edited != built.status_file.read_text(encoding="utf-8")
    with _environment(built, monkeypatch, manual_fixture), _tampered(built.status_file, edited.encode("utf-8")):
        failures = _audit(built)
    assert set(failures) >= {"sources", "units"}


def test_a_run_whose_arrays_no_longer_verify_fails_the_payloads(
    built: _Built, manual_fixture: ManualFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored run's arrays edited after the fact are refused where every run is judged again."""
    results = load_search_results(built.results_dir / "results_v1.json")
    rows = table_from_csv(stored_rows(built.context.store, results, "ManualRunRow"), ManualRunRow)
    row = next(r for r in rows if r.run_uri is not None)
    run_dir = manual_fixture.store.root / ArtifactUri.parse(row.run_uri or "").relative_path.parent
    arrays = next(run_dir.glob("*.npz"))
    with _environment(built, monkeypatch, manual_fixture), _tampered(arrays, arrays.read_bytes() + b"\0"):
        failures = _audit(built)
    assert "payloads_and_metrics" in failures


def test_a_freeze_the_index_does_not_bind_fails_the_selection(
    built: _Built, manual_fixture: ManualFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verified freeze must be the one the derivation bound."""
    with _environment(built, monkeypatch, manual_fixture):
        original = built.context.freeze_sha256
        built.context.freeze_sha256 = "0" * 64
        try:
            failures = _audit(built)
        finally:
            built.context.freeze_sha256 = original
    assert "selection" in failures


def test_a_manifest_that_no_longer_verifies_fails_the_manifests(
    built: _Built, manual_fixture: ManualFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored model manifest edited after it was pointed at is refused where every manifest is checked."""
    pointer = next(built.evidence_dir.glob("model__*.toml"))
    uri = next(line.split('"')[1] for line in pointer.read_text(encoding="utf-8").splitlines() if "armrc://" in line)
    manifest = manual_fixture.store.root / ArtifactUri.parse(uri).relative_path
    with _environment(built, monkeypatch, manual_fixture), _tampered(manifest, manifest.read_bytes() + b" "):
        failures = _audit(built)
    assert "manifests" in failures
    assert "payloads_and_metrics" in failures, "the rows that needed it report themselves too"


# --- the owner's review of M3MS-007 --------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_rc_successes", 1),
        ("n_replay_successes", 1),
        ("n_rc_runs", 999),
        ("n_replay_runs", 999),
        ("n_unavailable_runs", 99),
        ("n_models", 7),
        ("n_replay_banks", 7),
        ("departure_radius_m", 0.5),
    ],
)
def test_every_headline_figure_of_the_index_is_recomputed(
    built: _Built, manual_fixture: ManualFixture, monkeypatch: pytest.MonkeyPatch, field: str, value: float
) -> None:
    """An index claiming another figure than its rows give is refused, whichever figure it is."""
    path = built.results_dir / "results_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data[field] != value
    data[field] = value
    with _environment(built, monkeypatch, manual_fixture), _tampered(path, json.dumps(data).encode("utf-8")):
        failures = _audit(built)
    assert any(field in failure for failure in failures.get("completeness", [])), failures


def test_the_readable_results_are_the_index_rendered(
    built: _Built, manual_fixture: ManualFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A results page that is not the index's rendering is refused, even when the index itself is sound."""
    path = built.results_dir / "results_v1.md"
    with (
        _environment(built, monkeypatch, manual_fixture),
        _tampered(path, b"# Invented results\nAll models were perfect.\n"),
    ):
        failures = _audit(built)
    assert any("results_v1.md" in failure for failure in failures.get("sources", [])), failures


@pytest.mark.parametrize("content", [b"{", b'{"experiment": "task_1a_manual_esn_search"}'])
def test_an_unreadable_index_is_recorded_rather_than_raised(
    built: _Built, manual_fixture: ManualFixture, monkeypatch: pytest.MonkeyPatch, content: bytes
) -> None:
    """Every step is recorded unavailable with the reason, and the audit still returns a failed record."""
    path = built.results_dir / "results_v1.json"
    with _environment(built, monkeypatch, manual_fixture), _tampered(path, content):
        audit = audit_search_comparison(
            built.context,
            results_dir=built.results_dir,
            evidence_dir=built.evidence_dir,
            status_file=built.status_file,
            root=built.root,
        )
    assert not audit.passed
    assert tuple(step.name for step in audit.steps) == AUDIT_STEPS
    needing = {"sources", "selection", "payloads_and_metrics", "aggregates", "completeness"}
    assert {step.name for step in audit.steps if step.unavailable} == needing, "exactly the steps that need it"
    assert all("results index" in step.failures[0] for step in audit.steps if step.unavailable)
    assert all(step.ok for step in audit.steps if step.name not in needing), "the rest still ran and passed"


def test_an_index_that_cannot_be_opened_is_recorded_rather_than_raised(
    built: _Built, manual_fixture: ManualFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present but unreadable index is read once, inside the audit; its fingerprint is not a second read."""
    path = built.results_dir / "results_v1.json"
    mode = path.stat().st_mode
    path.chmod(0)
    try:
        with _environment(built, monkeypatch, manual_fixture):
            audit = audit_search_comparison(
                built.context,
                results_dir=built.results_dir,
                evidence_dir=built.evidence_dir,
                status_file=built.status_file,
                root=built.root,
            )
    finally:
        path.chmod(mode)
    assert not audit.passed
    assert audit.results_sha256 == "0" * 64, "no fingerprint of a file that could not be read"
    unavailable = [step for step in audit.steps if step.unavailable]
    assert unavailable
    assert all("PermissionError" in step.failures[0] for step in unavailable)
