# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MS-006: the comparison's schedule, scope, shared ceiling, resume protocol and verification.

The parent loop is exercised with fake workers and a context whose freeze and
reservations are synthetic, so ordering, the ceiling and every interruption
boundary can be driven exactly. The worker and the parent's verification are
exercised against the fixture study's real evidence: a real replay bank and a
real learned model, served back and recounted, and refused when the report
disagrees with what the store holds.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from arm_rc_ctrl.experiments import manual_comparison_run, manual_evaluation
from arm_rc_ctrl.experiments.manual_comparison_run import (
    ComparisonContext,
    ComparisonUnit,
    SharedLedger,
    UnitOutcome,
    UnitReservation,
    UnitResult,
    check_finalized,
    comparison_scope_mismatches,
    comparison_status,
    comparison_units,
    evaluate_unit,
    locked_scenario_ids,
    publish_pointers,
    serve_unit,
    shared_ledger,
    shared_stopped,
    unit_bytes,
    unit_directory,
    verified_unit,
)
from arm_rc_ctrl.experiments.manual_evaluation import EvidenceIntegrityError
from arm_rc_ctrl.experiments.manual_fits import cache_uri
from arm_rc_ctrl.experiments.manual_fixture import ManualFixture, manual_narrowed
from arm_rc_ctrl.experiments.manual_sampled import SampledPoint, sampled_configuration
from arm_rc_ctrl.experiments.manual_search import load_manual_search
from arm_rc_ctrl.experiments.manual_search_freeze import FrozenConfiguration, ManualSearchFreeze, write_freeze
from arm_rc_ctrl.experiments.manual_search_run import BudgetLedger, SearchInvocation, read_record, write_record
from arm_rc_ctrl.provenance import ArtifactReference, sha256_bytes, sha256_file
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import ENV_VAR, ArtifactUri, StorageRoot

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.experiments.manual_evaluation import ManualRunConditions
    from arm_rc_ctrl.experiments.manual_search import ManualSearchProtocol

ROOT = repository_root()
PROTOCOL_FILE = ROOT / "configs/studies/manual_esn_search_v1.toml"
PAIRS = 130
"""65 cases under two trackers: what every complete unit holds."""
EVIDENCE = ArtifactReference(uri="armrc://reports/task_1a_manual_v1/model/x/manifest-abc.json", sha256="b" * 64, size=1)


@pytest.fixture(scope="module")
def protocol() -> ManualSearchProtocol:
    """The committed search protocol, which carries the comparison's scope and the shared ceiling."""
    return load_manual_search(PROTOCOL_FILE)


def _point(index: int) -> SampledPoint:
    return SampledPoint(
        n_neurons=100 + 50 * index,
        spectral_radius=1.0,
        sparsity=0.9,
        leak_rate=0.05,
        input_scaling=0.3,
        alpha_0=0.01,
        warmup_s=0.0,
    )


def _freeze(protocol: ManualSearchProtocol, *, search_seconds: float = 1000.0, ranks: int = 3) -> ManualSearchFreeze:
    """A valid freeze of ``ranks`` configurations whose search spent ``search_seconds``."""
    chosen = tuple(
        FrozenConfiguration(
            rank=rank,
            trial=rank,
            configuration=sampled_configuration(protocol, _point(rank), trial=rank).label,
            point=_point(rank),
            score=1.0,
            successes=2,
            runs=2,
            fit_identity=sha256_bytes(f"fit{rank}".encode()),
            evidence_identity=sha256_bytes(f"evidence{rank}".encode()),
            evidence=f"armrc://reports/x/{rank}.json",
        )
        for rank in range(1, ranks + 1)
    )
    return ManualSearchFreeze(
        schema=1,
        protocol_sha256="0" * 64,
        study="manual-esn-search-v1",
        label="highest nominal scores",
        order="descending_score_then_ascending_trial",
        n_configurations=3,
        trial_cap=100,
        finalized=100,
        scored=100,
        failed=0,
        score_counts={"1.0": 100},
        stopped=("the 100-trial cap is spent (100 trials)",),
        ledger=BudgetLedger(trials=100, seconds=search_seconds, stored_bytes=1_000_000),
        chosen=chosen,
        shortfall=3 - ranks,
    )


# --- the schedule and the scope ------------------------------------------------------------------


def test_configurations_are_completed_in_selection_order_banks_first() -> None:
    """Ten banks, then the 31 learned models, for rank 1, then rank 2, then rank 3."""
    units = comparison_units(3)
    assert len(units) == 3 * 41
    assert [unit.number for unit in units] == list(range(123))
    for rank in (1, 2, 3):
        block = [unit for unit in units if unit.rank == rank]
        assert [unit.kind for unit in block] == ["replay"] * 10 + ["model"] * 31
        assert [unit.label for unit in block[:10]] == [f"D{index:02d}" for index in range(1, 11)]
        assert block[10].label == "S/D01"
        assert block[20].label == "M10"
    assert [unit.rank for unit in units] == sorted(unit.rank for unit in units)
    assert len({unit.slug for unit in units}) == len(units)


def test_the_comparison_scope_is_every_locked_case_under_both_trackers(protocol: ManualSearchProtocol) -> None:
    """The real evaluation's 65 cases in order, under the two fixed trackers, and nothing else."""
    locked = locked_scenario_ids(protocol.comparison.evaluation)
    trackers = protocol.fixed.trackers
    assert len(locked) == 65
    assert comparison_scope_mismatches(locked, locked, trackers, protocol) == []
    assert comparison_scope_mismatches(locked[:1], locked, trackers, protocol), "a nominal-only sweep is refused"
    assert comparison_scope_mismatches(tuple(reversed(locked)), locked, trackers, protocol), "so is a reordered one"
    assert comparison_scope_mismatches(locked, locked, trackers[:1], protocol), "and a single tracker"
    assert comparison_scope_mismatches(locked, locked, (*trackers, "other"), protocol), "or a third"


# --- the records ---------------------------------------------------------------------------------


def test_a_report_must_add_up_and_carry_evidence_or_a_failure() -> None:
    """A report whose counts do not make its pairs, or that claims both evidence and failure, is refused."""
    good = UnitResult(
        number=0, identity="a" * 64, pairs=2, completed=1, infeasible=1, unexecuted=0, seconds=1.0, evidence=EVIDENCE
    )
    with pytest.raises(ValueError, match="pairs"):
        replace(good, completed=2)
    with pytest.raises(ValueError, match="evidence or a failure"):
        replace(good, failure="also failed")
    with pytest.raises(ValueError, match="evidence or a failure"):
        replace(good, evidence=None)


def test_an_outcome_is_complete_with_evidence_or_otherwise_with_a_reason() -> None:
    """Complete units name their evidence; failed and unavailable ones name why."""
    base = UnitOutcome(
        number=0,
        state="complete",
        identity="a" * 64,
        pairs=0,
        completed=0,
        infeasible=0,
        unexecuted=0,
        seconds=0.0,
        stored_bytes=0,
        evidence="x",
    )
    with pytest.raises(ValueError, match="state"):
        replace(base, evidence=None)
    with pytest.raises(ValueError, match="state"):
        replace(base, state="unavailable")
    replace(base, state="unavailable", evidence=None, failure="its bank did not complete")


def test_a_model_reservation_names_its_fit_and_a_bank_does_not() -> None:
    """A replay bank fits nothing; a model is keyed by its fit."""
    base = UnitReservation(
        number=0,
        rank=1,
        kind="replay",
        label="D01",
        configuration="search-t0001",
        freeze_sha256="f" * 64,
        identity="a" * 64,
        fit_identity=None,
    )
    with pytest.raises(ValueError, match="identities"):
        replace(base, fit_identity="c" * 64)
    with pytest.raises(ValueError, match="identities"):
        replace(base, kind="model")
    replace(base, kind="model", label="M10", fit_identity="c" * 64)


# --- the shared ceiling --------------------------------------------------------------------------


def test_the_search_spend_counts_against_the_shared_ceiling(protocol: ManualSearchProtocol) -> None:
    """The comparison inherits the search's spend; the ceiling is not a fresh allowance."""
    cap = protocol.budget.hours * 3600.0
    fresh = SharedLedger(
        search=BudgetLedger(trials=100, seconds=1000.0, stored_bytes=0), units=0, seconds=0.0, stored_bytes=0
    )
    assert shared_stopped(fresh, protocol.budget) == []
    spent = replace(fresh, seconds=cap - 1000.0)
    assert any("cap is spent" in reason for reason in shared_stopped(spent, protocol.budget))
    tight = replace(fresh, seconds=cap - 1000.0 - 30.0)
    assert any("headroom" in reason for reason in shared_stopped(tight, protocol.budget))
    full = replace(fresh, stored_bytes=int(protocol.budget.gib * 1024**3))
    assert any("GiB" in reason for reason in shared_stopped(full, protocol.budget))


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> StorageRoot:
    """A store of this test's own."""
    root = tmp_path / "store"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(ENV_VAR, str(root))
    return StorageRoot(root)


def test_the_ledger_is_derived_from_the_records_and_counts_unfinished_work(
    protocol: ManualSearchProtocol, store: StorageRoot
) -> None:
    """Finalized units, a pending unit's partial bytes and staged payloads are all charged."""
    freeze = _freeze(protocol)
    units = comparison_units(3)
    write_record(
        unit_directory(store, units[0]) / "outcome.json",
        UnitOutcome(
            number=0,
            state="complete",
            identity="a" * 64,
            pairs=0,
            completed=0,
            infeasible=0,
            unexecuted=0,
            seconds=5.0,
            stored_bytes=100,
            evidence="x",
        ),
    )
    pending = UnitReservation(
        number=10,
        rank=1,
        kind="model",
        label="S/D01",
        configuration="c",
        freeze_sha256="f" * 64,
        identity="d" * 64,
        fit_identity="e" * 64,
    )
    write_record(unit_directory(store, units[10]) / "reservation.json", pending)
    cache = store.root / cache_uri(pending.fit_identity or "").relative_path
    cache.mkdir(parents=True)
    (cache / "weights.npy").write_bytes(b"x" * 300)
    staged = store.root / "runs" / "staging-abc"
    staged.mkdir(parents=True)
    (staged / "arrays.npz").write_bytes(b"y" * 50)
    ledger = shared_ledger(store, freeze, units)
    assert (ledger.units, ledger.seconds) == (1, 5.0)
    assert ledger.stored_bytes == 100 + 300 + 50
    assert unit_bytes(store, pending) == 300
    assert ledger.total_seconds == 1005.0


# --- the parent loop, with fake workers ----------------------------------------------------------


def _identity(unit: ComparisonUnit) -> str:
    """The synthetic evidence identity of a unit; ``_Context.drift`` changes it, as a changed freeze would."""
    return sha256_bytes(f"{unit.slug}{'drifted' if _Context.drift else ''}".encode())


class _Context(ComparisonContext):
    """A context with a synthetic freeze and reservations: the loop without a study behind it."""

    freeze_ranks = 3
    search_seconds = 1000.0
    drift = False

    def __init__(
        self,
        protocol: ManualSearchProtocol,
        protocol_file: Path,
        freeze_file: Path,
        *,
        root: Path,
        exploratory: bool = False,
    ) -> None:
        del root, exploratory
        self.protocol = protocol
        self.protocol_file = protocol_file
        self.freeze_file = freeze_file
        self.store = StorageRoot(Path(os.environ[ENV_VAR]))
        self.freeze = _freeze(protocol, search_seconds=type(self).search_seconds, ranks=type(self).freeze_ranks)
        self.freeze_sha256 = "f" * 64
        self.configurations = {}
        self.units = comparison_units(3)

    def reservation(self, unit: ComparisonUnit) -> UnitReservation:
        return UnitReservation(
            number=unit.number,
            rank=unit.rank,
            kind=unit.kind,
            label=unit.label,
            configuration=f"search-t{unit.rank:04d}",
            freeze_sha256=self.freeze_sha256,
            identity=_identity(unit),
            fit_identity=None if unit.kind == "replay" else sha256_bytes(f"fit-{unit.slug}".encode()),
        )

    def conditions(self, rank: int) -> ManualRunConditions:
        del rank
        return cast("ManualRunConditions", argparse.Namespace(pairs=tuple(range(PAIRS))))

    def installed_evidence(self, unit: ComparisonUnit, reservation: UnitReservation) -> ArtifactReference | None:
        """The synthetic manifest every fake worker reports as installed."""
        del unit, reservation
        return EVIDENCE

    def installed_counts(self, unit: ComparisonUnit, reservation: UnitReservation) -> tuple[int, int, int, int] | None:
        """What the synthetic manifest holds: every pair completed."""
        del unit, reservation
        return (PAIRS, PAIRS, 0, 0)


def _result(number: int, *, identity: str, failure: str | None = None) -> UnitResult:
    if failure is not None:
        return UnitResult(
            number=number,
            identity=identity,
            pairs=0,
            completed=0,
            infeasible=0,
            unexecuted=0,
            seconds=0.1,
            failure=failure,
        )
    return UnitResult(
        number=number,
        identity=identity,
        pairs=PAIRS,
        completed=PAIRS,
        infeasible=0,
        unexecuted=0,
        seconds=0.1,
        evidence=EVIDENCE,
    )


class _Worker:
    """A worker that reports what the test wants: ``"crash"`` writes nothing, ``"fail"`` a refused unit."""

    def __init__(self, behaviour: dict[int, str] | None = None, *, busy: float = 0.0) -> None:
        self.behaviour = behaviour or {}
        self.calls: list[int] = []
        self.timeouts: list[float] = []
        self.busy = busy

    def __call__(self, command: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[bytes]:
        command = list(command)
        number = int(command[command.index("--unit") + 1])
        self.calls.append(number)
        self.timeouts.append(timeout)
        start = time.monotonic()
        while time.monotonic() - start < self.busy:
            pass
        action = self.behaviour.pop(number, "ok")
        if action == "crash":
            return subprocess.CompletedProcess(command, returncode=1, stdout=b"", stderr=b"the worker crashed")
        unit = comparison_units(3)[number]
        report = _result(
            number, identity=_identity(unit), failure="the learner refused it" if action == "fail" else None
        )
        write_record(Path(command[command.index("--output") + 1]), report)
        return subprocess.CompletedProcess(command, returncode=0, stdout=b"", stderr=b"")


def _trusting(
    context: ComparisonContext,
    unit: ComparisonUnit,
    reservation: UnitReservation,
    result: UnitResult,
    *,
    seconds: float,
) -> UnitOutcome:
    """The parent's verdict without evidence behind it: what the worker reported."""
    del context
    return UnitOutcome(
        number=unit.number,
        state="complete" if result.failure is None else "failed",
        identity=reservation.identity,
        pairs=result.pairs,
        completed=result.completed,
        infeasible=result.infeasible,
        unexecuted=result.unexecuted,
        seconds=seconds,
        stored_bytes=0,
        evidence=None if result.evidence is None else result.evidence.uri,
        failure=result.failure,
    )


@pytest.fixture
def loop(monkeypatch: pytest.MonkeyPatch, store: StorageRoot) -> StorageRoot:
    """The parent loop with a synthetic context and a trusting verifier."""
    monkeypatch.setattr(manual_comparison_run, "ComparisonContext", _Context)
    monkeypatch.setattr(manual_comparison_run, "verified_unit", _trusting)
    monkeypatch.setattr(_Context, "freeze_ranks", 3)
    monkeypatch.setattr(_Context, "search_seconds", 1000.0)
    monkeypatch.setattr(_Context, "drift", False)
    return store


def _run(protocol: ManualSearchProtocol, worker: _Worker, monkeypatch: pytest.MonkeyPatch) -> SharedLedger:
    monkeypatch.setattr(manual_comparison_run, "spawn_unit", worker)
    return manual_comparison_run.run_comparison(
        protocol, PROTOCOL_FILE, Path("freeze.json"), root=ROOT, exploratory=True
    )


def _outcome(store: StorageRoot, number: int) -> UnitOutcome | None:
    path = unit_directory(store, comparison_units(3)[number]) / "outcome.json"
    return read_record(path, UnitOutcome) if path.is_file() else None


def test_every_unit_runs_once_in_order(
    protocol: ManualSearchProtocol, loop: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One configuration's whole comparison before the next, each bank before any model."""
    worker = _Worker()
    ledger = _run(protocol, worker, monkeypatch)
    assert worker.calls == list(range(123))
    assert ledger.units == 123
    again = _Worker()
    _run(protocol, again, monkeypatch)
    assert again.calls == [], "a finished comparison schedules nothing"
    assert all(_outcome(loop, n) is not None for n in range(123))


def test_a_failed_bank_makes_its_models_unavailable_without_running_them(
    protocol: ManualSearchProtocol, loop: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model is paired against its parent's bank; without it the model's result is unavailable, not failed."""
    worker = _Worker({0: "fail"})
    _run(protocol, worker, monkeypatch)
    dependants = [10, 21, 31]  # S/D01, R10/D01, C10/D01 at rank 1
    assert not set(dependants) & set(worker.calls)
    for number in dependants:
        outcome = _outcome(loop, number)
        assert outcome is not None
        assert outcome.state == "unavailable"
    assert 20 in worker.calls, "the all-ten arm has no single parent bank and still runs"
    failed = _outcome(loop, 0)
    assert failed is not None
    assert failed.state == "failed"


def test_no_unit_starts_when_the_shared_ceiling_leaves_no_allowance(
    protocol: ManualSearchProtocol, loop: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A search that spent the ceiling leaves the comparison nothing, and nothing is started."""
    monkeypatch.setattr(_Context, "search_seconds", protocol.budget.hours * 3600.0 - 30.0)
    worker = _Worker()
    _run(protocol, worker, monkeypatch)
    assert worker.calls == []
    assert not any(unit_directory(loop, unit).exists() for unit in comparison_units(3))


@pytest.mark.usefixtures("loop")
def test_a_ceiling_reached_midway_retains_the_partial_configuration(
    protocol: ManualSearchProtocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scheduling stops at the ceiling; what ran is kept, what did not is not started, and nothing is traded."""
    monkeypatch.setattr(
        _Context, "search_seconds", protocol.budget.hours * 3600.0 - manual_comparison_run.HEADROOM_S - 0.25
    )
    worker = _Worker(busy=0.1)
    _run(protocol, worker, monkeypatch)
    assert 1 <= len(worker.calls) <= 4
    assert worker.calls == list(range(len(worker.calls))), "a prefix of the schedule, in order"
    assert all(timeout <= 0.25 + 1e-6 for timeout in worker.timeouts), "each worker is bounded by what is left"
    resumed = _Worker()
    _run(protocol, resumed, monkeypatch)
    assert resumed.calls == [], "a resume does not buy more allowance"
    status = comparison_status(cast("Any", _Context(protocol, PROTOCOL_FILE, Path("freeze.json"), root=ROOT)))
    first = status.configurations[0]
    assert not first.finished
    assert first.complete == len(worker.calls)
    assert status.finished_configurations == 0
    assert status.stopped


@pytest.mark.usefixtures("loop")
def test_a_worker_is_given_the_remaining_allowance(
    protocol: ManualSearchProtocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ceiling bounds a running worker, not only the gap between units."""
    worker = _Worker()
    monkeypatch.setattr(_Context, "freeze_ranks", 1)
    _run(protocol, worker, monkeypatch)
    expected = protocol.budget.hours * 3600.0 - 1000.0 - manual_comparison_run.HEADROOM_S
    assert worker.timeouts[0] == pytest.approx(expected, abs=1.0)


def test_an_interrupted_unit_resumes_under_its_own_number_and_every_attempt_is_charged(
    protocol: ManualSearchProtocol, loop: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker that writes no report leaves its unit pending; the resume finishes that unit first."""
    first = _Worker({0: "crash"}, busy=0.05)
    _run(protocol, first, monkeypatch)
    assert first.calls == [0], "an interruption stops the invocation"
    assert _outcome(loop, 0) is None
    resumed = _Worker(busy=0.05)
    _run(protocol, resumed, monkeypatch)
    assert resumed.calls[0] == 0
    outcome = _outcome(loop, 0)
    assert outcome is not None
    assert outcome.seconds >= 0.1, "both attempts are charged"


class _Clock:
    def __init__(self, instants: Sequence[datetime]) -> None:
        self.instants = list(instants)

    def __call__(self) -> datetime:
        return self.instants.pop(0) if len(self.instants) > 1 else self.instants[0]


@pytest.mark.usefixtures("loop")
def test_a_parent_killed_mid_attempt_still_charges_the_time_it_ran(
    protocol: ManualSearchProtocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The attempt is opened on disk before the worker starts, so a resume charges it by the wall clock."""
    opened = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(manual_comparison_run, "_now", _Clock([opened]))

    def killed(command: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[bytes]:
        del command, timeout
        raise KeyboardInterrupt

    monkeypatch.setattr(manual_comparison_run, "spawn_unit", killed)
    with pytest.raises(KeyboardInterrupt):
        manual_comparison_run.run_comparison(protocol, PROTOCOL_FILE, Path("freeze.json"), root=ROOT, exploratory=True)
    monkeypatch.setattr(manual_comparison_run, "_now", _Clock([opened + timedelta(seconds=300)]))
    ledger = _run(protocol, _Worker(), monkeypatch)
    assert ledger.seconds >= 300.0


@pytest.mark.usefixtures("loop")
def test_a_reservation_that_no_longer_derives_is_refused(
    protocol: ManualSearchProtocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume keys its work by what the verified freeze derives now; a recorded reservation that differs stops it."""
    _run(protocol, _Worker({0: "crash"}), monkeypatch)
    monkeypatch.setattr(_Context, "drift", True)
    with pytest.raises(ValueError, match="recorded reservation"):
        _run(protocol, _Worker(), monkeypatch)


def test_every_invocation_is_recorded_even_when_it_raises(
    protocol: ManualSearchProtocol, loop: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent's own time is measured, whether or not it finished."""

    def refusing(*args: object, **kwargs: object) -> UnitOutcome:
        del args, kwargs
        msg = "the evidence was refused"
        raise ValueError(msg)

    monkeypatch.setattr(manual_comparison_run, "verified_unit", refusing)
    with pytest.raises(ValueError, match="refused"):
        _run(protocol, _Worker(), monkeypatch)
    directory = (
        loop.root / ArtifactUri.parse(f"{manual_comparison_run.COMPARISON_PREFIX}/invocations/x").relative_path.parent
    )
    (record,) = [read_record(path, SearchInvocation) for path in directory.glob("*.json")]
    assert not record.completed
    assert record.trials == (0,)


def test_a_worker_is_never_spawned_without_a_deadline() -> None:
    """The guard is in the spawn itself."""
    with pytest.raises(ValueError, match="deadline"):
        manual_comparison_run.spawn_unit(["true"], timeout=0.0)


# --- the worker and the parent's verification, against real evidence -----------------------------


POINT = SampledPoint(
    n_neurons=100, spectral_radius=1.0, sparsity=0.9, leak_rate=0.05, input_scaling=0.3, alpha_0=0.01, warmup_s=0.0
)


def _any_scope(*args: object) -> list[str]:
    """The scope check replaced, for a fixture whose evaluation is narrowed on purpose."""
    del args
    return []


class _FixtureContext(ComparisonContext):
    """A context over the fixture study: a real runner, one configuration, no search behind it."""

    def __init__(self, protocol: ManualSearchProtocol, f: ManualFixture, freeze_file: Path) -> None:
        configuration = sampled_configuration(protocol, POINT, trial=0)
        args = argparse.Namespace(
            study=str(f.manifest_file),
            evaluation=str(protocol.comparison.evaluation),
            exploratory=True,
            argv=["compare"],
        )
        prepared = manual_evaluation.prepare_runner(args, role="main", root=f.root, sampled=(configuration,))
        self.protocol = protocol
        self.protocol_file = PROTOCOL_FILE
        self.freeze_file = freeze_file
        self.store = f.store
        self.freeze = cast("Any", None)
        self.freeze_sha256 = sha256_file(freeze_file)
        self.configurations = {1: configuration}
        self.manifest = prepared.context.manifest
        self.runner = prepared.runner
        self.units = tuple(unit for unit in comparison_units(3) if unit.rank == 1)


@pytest.fixture
def fixture_freeze(fixture_search: ManualSearchProtocol, tmp_path: Path) -> Path:
    """A freeze of one configuration over the fixture study (a shortfall of two)."""
    configuration = sampled_configuration(fixture_search, POINT, trial=0)
    freeze = _freeze(fixture_search, ranks=1)
    chosen = replace(freeze.chosen[0], trial=0, point=POINT, configuration=configuration.label)
    path = tmp_path / "selection.json"
    write_freeze(path, replace(freeze, chosen=(chosen,)))
    return path


def test_a_real_bank_and_model_are_evaluated_and_verified_by_the_parent(
    fixture_search: ManualSearchProtocol,
    manual_fixture: ManualFixture,
    fixture_freeze: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker builds a bank and a paired model; the parent serves both back and recounts them."""
    for name, value in manual_fixture.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(manual_evaluation, "repository_root", lambda: manual_fixture.root)
    monkeypatch.setattr(manual_evaluation, "evaluation_scenarios", manual_narrowed)
    # The fixture's evaluation is narrowed; the scope check itself is held above against the real 65 cases.
    monkeypatch.setattr(manual_comparison_run, "comparison_scope_mismatches", _any_scope)
    digest = sha256_file(fixture_freeze)
    context = _FixtureContext(fixture_search, manual_fixture, fixture_freeze)
    units = {unit.label: unit for unit in context.units}
    for label in ("D01", "S/D01"):
        unit = units[label]
        result = evaluate_unit(
            fixture_search,
            freeze_file=fixture_freeze,
            freeze_sha256=digest,
            number=unit.number,
            root=manual_fixture.root,
            argv=["evaluate-unit"],
            exploratory=True,
        )
        assert result.failure is None
        assert result.pairs == len(context.conditions(1).pairs)
        reservation = context.reservation(unit)
        outcome = verified_unit(context, unit, reservation, result, seconds=1.0)
        assert (outcome.state, outcome.pairs) == ("complete", result.pairs)
        miscounted = (
            replace(result, completed=result.completed - 1, infeasible=result.infeasible + 1)
            if result.completed
            else replace(result, completed=1, infeasible=result.infeasible - 1)
        )
        with pytest.raises(ValueError, match="counts"):
            verified_unit(context, unit, reservation, miscounted, seconds=1.0)
        assert result.evidence is not None
        elsewhere = replace(result, evidence=replace(result.evidence, sha256="0" * 64))
        with pytest.raises(ValueError, match="installed"):
            verified_unit(context, unit, reservation, elsewhere, seconds=1.0)


def test_a_worker_refuses_a_freeze_other_than_the_verified_one(
    fixture_search: ManualSearchProtocol, fixture_freeze: Path, tmp_path: Path
) -> None:
    """The freeze's bytes are checked before anything is read from it."""
    with pytest.raises(ValueError, match="not the freeze the parent verified"):
        evaluate_unit(
            fixture_search,
            freeze_file=fixture_freeze,
            freeze_sha256="0" * 64,
            number=0,
            root=tmp_path,
            argv=["evaluate-unit"],
            exploratory=True,
        )


# --- the owner's review of M3MS-006 --------------------------------------------------------------


@pytest.mark.usefixtures("loop")
def test_a_finished_units_reservation_is_checked_on_resume(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume does not skip a finalized unit on its word: its reservation must still derive."""
    _run(protocol, _Worker(), monkeypatch)
    path = unit_directory(store, comparison_units(3)[0]) / "reservation.json"
    recorded = read_record(path, UnitReservation)
    write_record(path, replace(recorded, freeze_sha256="0" * 64, identity="1" * 64))
    with pytest.raises(ValueError, match="reservation"):
        _run(protocol, _Worker(), monkeypatch)


@pytest.mark.usefixtures("loop")
def test_a_finished_outcome_must_fit_the_trusted_schedule(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An outcome claiming more pairs than every case under both trackers is refused, in status and on resume."""
    _run(protocol, _Worker(), monkeypatch)
    path = unit_directory(store, comparison_units(3)[0]) / "outcome.json"
    write_record(path, replace(read_record(path, UnitOutcome), pairs=999, completed=999))
    context = _Context(protocol, PROTOCOL_FILE, Path("freeze.json"), root=ROOT)
    with pytest.raises(ValueError, match="999"):
        comparison_status(context)
    with pytest.raises(ValueError, match="999"):
        _run(protocol, _Worker(), monkeypatch)


@pytest.mark.usefixtures("loop")
def test_a_finished_outcome_must_name_the_installed_evidence(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete outcome pointing anywhere but the manifest installed under its identity is refused."""
    _run(protocol, _Worker(), monkeypatch)
    path = unit_directory(store, comparison_units(3)[5]) / "outcome.json"
    write_record(path, replace(read_record(path, UnitOutcome), evidence="armrc://reports/elsewhere.json"))
    with pytest.raises(ValueError, match="installed"):
        comparison_status(_Context(protocol, PROTOCOL_FILE, Path("freeze.json"), root=ROOT))


def _fixture_environment(f: ManualFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in f.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(manual_evaluation, "repository_root", lambda: f.root)
    monkeypatch.setattr(manual_evaluation, "evaluation_scenarios", manual_narrowed)
    monkeypatch.setattr(manual_comparison_run, "comparison_scope_mismatches", _any_scope)


def _evaluate(protocol: ManualSearchProtocol, f: ManualFixture, freeze_file: Path, number: int) -> UnitResult:
    """One unit evaluated in this process over the fixture study, as its worker would."""
    return evaluate_unit(
        protocol,
        freeze_file=freeze_file,
        freeze_sha256=sha256_file(freeze_file),
        number=number,
        root=f.root,
        argv=["evaluate-unit"],
        exploratory=True,
    )


def _forbidden(*args: object, **kwargs: object) -> None:
    del args, kwargs
    msg = "a run was simulated outside a worker, its deadline and its budget"
    raise AssertionError(msg)


def test_publication_never_computes_missing_evidence(
    fixture_search: ManualSearchProtocol,
    manual_fixture: ManualFixture,
    fixture_freeze: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Publishing serves what exists; a completed unit whose evidence is gone is refused, not rebuilt."""
    _fixture_environment(manual_fixture, monkeypatch)
    context = _FixtureContext(fixture_search, manual_fixture, fixture_freeze)
    unit = context.units[1]  # D02: its own bank, untouched by the other tests
    result = _evaluate(fixture_search, manual_fixture, fixture_freeze, unit.number)
    reservation = context.reservation(unit)
    outcome = verified_unit(context, unit, reservation, result, seconds=1.0)
    directory = unit_directory(manual_fixture.store, unit)
    write_record(directory / "reservation.json", reservation)
    write_record(directory / "outcome.json", outcome)
    assert result.evidence is not None
    bank = manual_fixture.store.root / ArtifactUri.parse(result.evidence.uri).relative_path.parent
    removed = tmp_path / "removed-bank"
    shutil.move(bank, removed)
    try:
        context = _FixtureContext(fixture_search, manual_fixture, fixture_freeze)
        context.units = (unit,)
        monkeypatch.setattr(context.runner, "simulate", _forbidden, raising=False)
        with pytest.raises(ValueError, match="installed"):
            publish_pointers(context, tmp_path / "pointers")
        with pytest.raises(ValueError, match="its own evidence is not installed"):
            serve_unit(context, unit, reservation)
        assert not bank.exists(), "refusing does not leave an empty directory behind"
    finally:
        shutil.move(removed, bank)
        shutil.rmtree(directory)


def test_a_corrupt_manifest_leaves_the_unit_recoverable(
    fixture_search: ManualSearchProtocol,
    manual_fixture: ManualFixture,
    fixture_freeze: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fault in stored evidence is not a verdict: it propagates, so no failed outcome can be written."""
    _fixture_environment(manual_fixture, monkeypatch)
    result = _evaluate(fixture_search, manual_fixture, fixture_freeze, 2)  # D03's bank, used by no other test
    assert result.evidence is not None
    manifest = manual_fixture.store.root / ArtifactUri.parse(result.evidence.uri).relative_path
    original = manifest.read_bytes()
    manifest.write_bytes(original + b" ")
    try:
        with pytest.raises(EvidenceIntegrityError):
            _evaluate(fixture_search, manual_fixture, fixture_freeze, 2)
    finally:
        manifest.write_bytes(original)


def test_only_a_refused_fit_is_a_failed_model(
    fixture_search: ManualSearchProtocol,
    manual_fixture: ManualFixture,
    fixture_freeze: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The learner refusing a candidate is a result; any other refusal during the sweep propagates."""
    _fixture_environment(manual_fixture, monkeypatch)

    def refused(*args: object, **kwargs: object) -> None:
        del args, kwargs
        msg = "the readout is singular"
        raise ValueError(msg)

    with monkeypatch.context() as patched:
        patched.setattr(manual_comparison_run.ManualFitStore, "fit_or_load", refused)
        result = _evaluate(fixture_search, manual_fixture, fixture_freeze, 20)  # M10: no paired bank
    assert result.failure is not None
    assert "singular" in result.failure
    with monkeypatch.context() as patched:
        patched.setattr(manual_evaluation.ManualEvaluationRunner, "evaluate", refused)
        with pytest.raises(ValueError, match="singular"):
            _evaluate(fixture_search, manual_fixture, fixture_freeze, 20)  # M10: no paired bank


# --- the owner's second review of M3MS-006 -------------------------------------------------------


def test_a_corrupt_cached_fit_is_not_a_failed_model(
    fixture_search: ManualSearchProtocol,
    manual_fixture: ManualFixture,
    fixture_freeze: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached fit that no longer verifies is a fault in the store; only a fresh fit's refusal is a result."""
    _fixture_environment(manual_fixture, monkeypatch)
    _evaluate(fixture_search, manual_fixture, fixture_freeze, 20)  # M10, fitted and cached
    context = _FixtureContext(fixture_search, manual_fixture, fixture_freeze)
    reservation = context.reservation(context.units[20])
    record = manual_fixture.store.root / cache_uri(reservation.fit_identity or "").relative_path / "fit.json"
    original = record.read_bytes()
    data = json.loads(original)
    data["configuration"] = "another-configuration"
    record.write_text(json.dumps(data), encoding="utf-8")
    try:
        with pytest.raises(EvidenceIntegrityError, match="does not verify"):
            _evaluate(fixture_search, manual_fixture, fixture_freeze, 20)
    finally:
        record.write_bytes(original)


def test_finalized_counts_must_be_the_installed_manifests(
    fixture_search: ManualSearchProtocol,
    manual_fixture: ManualFixture,
    fixture_freeze: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving one pair between verdicts keeps the total, and is still refused: the breakdown is recounted."""
    _fixture_environment(manual_fixture, monkeypatch)
    context = _FixtureContext(fixture_search, manual_fixture, fixture_freeze)
    unit = context.units[0]
    result = _evaluate(fixture_search, manual_fixture, fixture_freeze, unit.number)
    reservation = context.reservation(unit)
    outcome = verified_unit(context, unit, reservation, result, seconds=1.0)
    directory = unit_directory(manual_fixture.store, unit)
    write_record(directory / "reservation.json", reservation)
    try:
        write_record(directory / "outcome.json", outcome)
        check_finalized(context, unit)
        moved = (
            replace(outcome, completed=outcome.completed - 1, infeasible=outcome.infeasible + 1)
            if outcome.completed
            else replace(outcome, completed=1, infeasible=outcome.infeasible - 1)
        )
        write_record(directory / "outcome.json", moved)
        with pytest.raises(ValueError, match="breakdown"):
            check_finalized(context, unit)
    finally:
        shutil.rmtree(directory)
