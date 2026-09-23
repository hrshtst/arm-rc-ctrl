# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MS-003: the search's scope guards, its budget accounting and its worker boundary.

The two properties this experiment rests on are tested here rather than
assumed: no perturbed scenario can reach the optimizer, and the budget is spent
once across every invocation that resumes the study. The scope guard is checked
where a worker actually receives its scenarios — through a real subprocess —
because a parent-side preflight would not survive a worker invoked any other
way.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.experiments import manual_evaluation, manual_search_run
from arm_rc_ctrl.experiments.manual_fits import WEIGHTS_FILE, ManualFitStore, cache_uri
from arm_rc_ctrl.experiments.manual_fixture import ManualFixture, manual_narrowed
from arm_rc_ctrl.experiments.manual_numerics import refit_in_subprocess
from arm_rc_ctrl.experiments.manual_sampled import SampledPoint, arm_of, sampled_configuration, sampled_entry
from arm_rc_ctrl.experiments.manual_search import ManualSearchBudget, load_manual_search
from arm_rc_ctrl.experiments.manual_search_freeze import reverify_outcome
from arm_rc_ctrl.experiments.manual_search_run import (
    BudgetLedger,
    TrialOutcome,
    TrialReservation,
    TrialResult,
    TrialSpend,
    budget_complaints,
    evaluate_trial,
    ledger_of,
    nominal_scope_mismatches,
    pending_reservations,
    read_record,
    read_result,
    trial_command,
    trial_directory,
    verified_outcome,
    write_record,
)
from arm_rc_ctrl.experiments.run_record import RUN_SUMMARY_FILE
from arm_rc_ctrl.provenance import ArtifactReference
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import ENV_VAR, StorageRoot

if TYPE_CHECKING:
    from arm_rc_ctrl.experiments.manual_search import ManualSearchProtocol

ROOT = repository_root()
PROTOCOL_FILE = ROOT / "configs/studies/manual_esn_search_v1.toml"
POINT = SampledPoint(
    n_neurons=100,
    spectral_radius=1.0,
    sparsity=0.9,
    leak_rate=0.05,
    input_scaling=0.3,
    alpha_0=0.01,
    warmup_s=0.0,
)


@pytest.fixture(scope="module")
def protocol() -> ManualSearchProtocol:
    """The committed search protocol."""
    return load_manual_search(PROTOCOL_FILE)


# --- the optimizer's scope --------------------------------------------------------------------


def test_the_nominal_scope_accepts_exactly_the_nominal_case() -> None:
    """The scope is one scenario, and the guard says so in the owner's terms."""
    assert nominal_scope_mismatches(["nominal"]) == []


@pytest.mark.parametrize(
    "scenarios",
    [
        ["nominal", "posture-small-20261201-01"],
        ["posture-small-20261201-01"],
        [],
        ["nominal", "nominal"],
    ],
)
def test_any_other_scope_is_refused(scenarios: list[str]) -> None:
    """Widening, replacing or repeating the scenario list all fail the same check."""
    assert nominal_scope_mismatches(scenarios) != []


def test_the_worker_refuses_a_widened_scope_before_it_evaluates_anything(
    protocol: ManualSearchProtocol, tmp_path: Path
) -> None:
    """Called directly, with perturbed scenarios, the worker refuses rather than scoring them."""
    with pytest.raises(ValueError, match="nominal"):
        evaluate_trial(
            protocol,
            trial=0,
            point=POINT,
            scenarios=("nominal", "posture-small-20261201-01"),
            root=tmp_path,
            argv=["evaluate-trial"],
        )


def test_the_real_worker_process_refuses_a_widened_scope(tmp_path: Path) -> None:
    """The guard holds at the entry point a resumed study or a hand-run command would use."""
    command = trial_command(PROTOCOL_FILE, trial=0, point=POINT, output=tmp_path / "result.json", root=ROOT)
    widened = [*command]
    widened[widened.index("--scenarios") + 1] = "posture-small-20261201-01"
    completed = subprocess.run(widened, check=False, capture_output=True, cwd=ROOT)
    assert completed.returncode != 0
    assert b"nominal" in completed.stderr
    assert not (tmp_path / "result.json").exists(), "a refused worker writes no report"


def test_the_worker_command_carries_the_scope_and_the_point(tmp_path: Path) -> None:
    """The worker is told its scenarios and its point, so it can refuse and reproduce independently."""
    command = trial_command(PROTOCOL_FILE, trial=3, point=POINT, output=tmp_path / "r.json", root=ROOT)
    assert command[0] == sys.executable
    assert command[command.index("--scenarios") + 1] == "nominal"
    assert json.loads(command[command.index("--point") + 1]) == to_mapping(POINT)
    assert command[command.index("--trial") + 1] == "3"


# --- what a worker reports ----------------------------------------------------------------------


EVIDENCE = ArtifactReference(
    uri="armrc://reports/task_1a_manual_v1/model/x/manifest-abc.json", sha256="b" * 64, size=10
)


def _result(**changes: object) -> TrialResult:
    base = TrialResult(
        trial=1,
        successes=1,
        runs=2,
        statuses=("completed", "infeasible"),
        scenarios=("nominal",),
        seconds=3.0,
        evidence=EVIDENCE,
    )
    return replace(base, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"successes": 3},
        {"successes": -1},
        {"statuses": ("completed",)},
        {"scenarios": ("nominal", "posture-small-20261201-01")},
        {"trial": -1},
    ],
)
def test_an_inconsistent_or_out_of_scope_report_is_refused(changes: dict[str, object]) -> None:
    """A report that cannot be true is not evidence, whichever way it is wrong."""
    with pytest.raises(ValueError, match=r"trial|nominal"):
        _result(**changes)


def test_a_report_round_trips_through_its_file(tmp_path: Path) -> None:
    """The parent reads the worker's report strictly, as it reads every other record."""
    path = tmp_path / "result.json"
    path.write_text(json.dumps(to_mapping(_result()), sort_keys=True), encoding="utf-8")
    assert read_result(path) == _result()


# --- the budget, across every invocation that resumes ---------------------------------------------


def test_a_ledger_accumulates_rather_than_resets() -> None:
    """A resumed invocation continues the spend; it does not start a fresh allowance."""
    ledger = BudgetLedger(trials=0, seconds=0.0, stored_bytes=0)
    after = ledger.plus(trials=1, seconds=12.0, stored_bytes=1024).plus(trials=1, seconds=8.0, stored_bytes=512)
    assert (after.trials, after.seconds, after.stored_bytes) == (2, 20.0, 1536)


@pytest.mark.parametrize("field", ["trials", "seconds", "stored_bytes"])
def test_a_ledger_refuses_negative_spend(field: str) -> None:
    """Spend only grows; a negative entry would buy back a cap."""
    values: dict[str, object] = {"trials": 0, "seconds": 0.0, "stored_bytes": 0, field: -1}
    with pytest.raises(ValueError, match="non-negative"):
        BudgetLedger(**values)  # pyright: ignore[reportArgumentType] - the invalid value under test


def test_no_cap_is_reached_before_anything_is_spent(protocol: ManualSearchProtocol) -> None:
    """An empty ledger schedules work."""
    assert budget_complaints(BudgetLedger(trials=0, seconds=0.0, stored_bytes=0), protocol.budget) == []


@pytest.mark.parametrize(
    ("ledger", "expected"),
    [
        (BudgetLedger(trials=100, seconds=0.0, stored_bytes=0), "trial"),
        (BudgetLedger(trials=0, seconds=36_000.0, stored_bytes=0), "h cap"),
        (BudgetLedger(trials=0, seconds=0.0, stored_bytes=20 * 1024**3), "GiB"),
    ],
)
def test_a_spent_cap_stops_scheduling(protocol: ManualSearchProtocol, ledger: BudgetLedger, expected: str) -> None:
    """Each approved cap stops the search on its own, and says which one it was."""
    complaints = budget_complaints(ledger, protocol.budget)
    assert any(expected in text for text in complaints), complaints


def test_a_tightened_cap_binds_earlier() -> None:
    """A protocol may spend less than the owner approved, and the ledger honours that."""
    budget = ManualSearchBudget(trials=10, hours=1.0, gib=2.0)
    assert budget_complaints(BudgetLedger(trials=10, seconds=0.0, stored_bytes=0), budget) != []
    assert budget_complaints(BudgetLedger(trials=9, seconds=0.0, stored_bytes=0), budget) == []


@contextmanager
def _corrupted(path: Path, content: bytes | None = None) -> Generator[None]:
    """Rewrite one stored payload for the length of a check, then restore it.

    ``content`` replaces the payload outright, for a file that can no longer be
    decoded at all; the default keeps its length and breaks only its digest.
    The demonstration fixture and its store are built once for the module, so a
    check that a corrupted payload is refused has to leave the store exactly as
    it found it.
    """
    original = path.read_bytes()
    path.write_bytes(original[:-8] + bytes(8) if content is None else content)
    try:
        yield
    finally:
        path.write_bytes(original)


# --- the worker against a real study ------------------------------------------------------------


def test_the_worker_evaluates_the_nominal_case_under_both_trackers(
    fixture_search: ManualSearchProtocol,
    manual_fixture: ManualFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One trial is one candidate's two nominal runs, and nothing else is simulated."""
    for name, value in manual_fixture.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(manual_evaluation, "repository_root", lambda: manual_fixture.root)
    monkeypatch.setattr(manual_evaluation, "evaluation_scenarios", manual_narrowed)
    result = evaluate_trial(
        fixture_search,
        trial=0,
        point=POINT,
        scenarios=("nominal",),
        root=manual_fixture.root,
        argv=["evaluate-trial"],
        exploratory=True,
    )
    assert result.runs == 2, "the two fixed trackers, and only the nominal case"
    assert result.scenarios == ("nominal",)
    assert result.successes <= result.runs
    assert result.evidence is not None, "a scored candidate reports where its evidence was installed"
    timing = result.timing
    assert timing is not None, "the worker says how its time divided, which the estimates omit (M3MS-004)"
    assert timing.simulated_runs == 2
    assert min(timing.prepare_seconds, timing.fit_seconds, timing.simulate_seconds) > 0.0
    assert timing.sweep_seconds >= timing.simulate_seconds + timing.persist_seconds
    assert timing.run_bytes > 0
    directory = manual_fixture.store.path(result.evidence.uri, mode="read").parent
    claimed = manual_evaluation.read_run_claims(directory / manual_evaluation.RUN_CLAIMS_FILE)
    recorded = manual_evaluation.read_progress_runs(directory / manual_evaluation.PROGRESS_FILE)
    assert len(claimed) == 2, "each run was claimed before its payload was published"
    assert {f"armrc://runs/{run}/{RUN_SUMMARY_FILE}" for run in claimed} == set(recorded), (
        "and the claims are the runs the progress record ended up naming"
    )
    # A retried worker is served the evidence the first attempt stored: nothing is swept or simulated,
    # and its report must still be one the parent can read.
    served = evaluate_trial(
        fixture_search,
        trial=0,
        point=POINT,
        scenarios=("nominal",),
        root=manual_fixture.root,
        argv=["evaluate-trial"],
        exploratory=True,
    )
    assert served.timing is not None
    assert (served.timing.simulated_runs, served.timing.sweep_seconds) == (0, 0.0)
    report = tmp_path / "served.json"
    write_record(report, served)
    assert read_result(report) == served


def test_a_scored_candidate_is_checked_against_reconstructed_inputs(
    fixture_search: ManualSearchProtocol, manual_fixture: ManualFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reservation's identities are recorded fields; the model behind them is verified whole.

    The parent rebuilds the study's own fit inputs and runs the same
    model-binding checks a resume and the audit run, so evidence that carries
    the reserved identities but not the construction they name is refused
    instead of scored.
    """
    for name, value in manual_fixture.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(manual_evaluation, "repository_root", lambda: manual_fixture.root)
    monkeypatch.setattr(manual_evaluation, "evaluation_scenarios", manual_narrowed)
    result = evaluate_trial(
        fixture_search,
        trial=0,
        point=POINT,
        scenarios=("nominal",),
        root=manual_fixture.root,
        argv=["evaluate-trial"],
        exploratory=True,
    )
    inputs = manual_search_run.SearchInputs(fixture_search, root=manual_fixture.root, exploratory=True)
    reservation = manual_search_run.reserve_trial(fixture_search, inputs, POINT, trial=0)
    scored = verified_outcome(
        manual_fixture.store, reservation, result, seconds=1.0, protocol=fixture_search, inputs=inputs
    )
    assert (scored.state, scored.runs) == ("scored", 2)
    weights = manual_fixture.store.root / cache_uri(reservation.fit_identity).relative_path / WEIGHTS_FILE
    with _corrupted(weights), pytest.raises(ValueError, match="recorded digest"):
        verified_outcome(manual_fixture.store, reservation, result, seconds=1.0, protocol=fixture_search, inputs=inputs)


_FRESH_PARENT = """
import json, sys
from pathlib import Path

from arm_rc_ctrl.experiments import manual_evaluation, manual_search, manual_search_run
from arm_rc_ctrl.experiments.manual_fixture import manual_narrowed

root = Path(sys.argv[1])
manual_search.repository_root = lambda: root
manual_evaluation.repository_root = lambda: root
manual_evaluation.evaluation_scenarios = manual_narrowed
protocol = manual_search.load_manual_search(Path(sys.argv[2]))
inputs = manual_search_run.SearchInputs(protocol, root=root, exploratory=True)
point = manual_search_run.sampled_point(protocol.space, json.loads(sys.argv[3]))
reservation = manual_search_run.reserve_trial(protocol, inputs, point, trial=0)
outcome = manual_search_run.verified_outcome(
    manual_search_run.open_storage(),
    reservation,
    manual_search_run.read_result(Path(sys.argv[4])),
    seconds=1.0,
    protocol=protocol,
    inputs=inputs,
)
print(json.dumps({"state": outcome.state, "runs": outcome.runs}))
"""
"""A parent that verifies one trial's evidence and nothing else, to be run in a fresh interpreter."""


def test_a_fresh_parent_verifies_a_trials_evidence(
    fixture_search: ManualSearchProtocol,
    manual_fixture: ManualFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The parent's environment has to be the one the study was frozen in, which only a fresh process shows.

    Every in-process test has already loaded the numerical runtimes through
    the fixture, so a parent that probed its environment before loading them
    looks correct here and is refused by the very study it is resuming. The
    successful path is therefore tested where it can fail: in an interpreter
    that starts with nothing loaded.
    """
    for name, value in manual_fixture.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(manual_evaluation, "repository_root", lambda: manual_fixture.root)
    monkeypatch.setattr(manual_evaluation, "evaluation_scenarios", manual_narrowed)
    result = evaluate_trial(
        fixture_search,
        trial=0,
        point=POINT,
        scenarios=("nominal",),
        root=manual_fixture.root,
        argv=["evaluate-trial"],
        exploratory=True,
    )
    report = tmp_path / "result.json"
    write_record(report, result)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _FRESH_PARENT,
            str(manual_fixture.root),
            str(manual_fixture.root / "configs" / "studies" / "fixture_search.toml"),
            json.dumps(to_mapping(POINT)),
            str(report),
        ],
        check=False,
        capture_output=True,
        env=dict(manual_fixture.env),
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert json.loads(completed.stdout) == {"state": "scored", "runs": 2}


@pytest.mark.parametrize("content", [None, b"this is not an npy file at all"])
def test_corrupt_cached_evidence_is_not_a_failed_candidate(
    fixture_search: ManualSearchProtocol,
    manual_fixture: ManualFixture,
    monkeypatch: pytest.MonkeyPatch,
    content: bytes | None,
) -> None:
    """A fault in the store is not a verdict on a candidate: it propagates and the trial stays recoverable.

    Both kinds of fault, because they arrive by different routes: a payload
    whose digest disagrees is caught by a check, while one that cannot be
    decoded at all is whatever the decoder raises, and that was an ordinary
    ``ValueError`` the worker read as a failed fit.
    """
    for name, value in manual_fixture.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(manual_evaluation, "repository_root", lambda: manual_fixture.root)
    monkeypatch.setattr(manual_evaluation, "evaluation_scenarios", manual_narrowed)
    kwargs: dict[str, Any] = {
        "trial": 0,
        "point": POINT,
        "scenarios": ("nominal",),
        "root": manual_fixture.root,
        "argv": ["evaluate-trial"],
        "exploratory": True,
    }
    evaluate_trial(fixture_search, **kwargs)
    configuration = sampled_configuration(fixture_search, POINT, trial=0)
    entry = sampled_entry(manual_fixture.manifest, configuration, arm_of("M10"))
    weights = manual_fixture.store.root / cache_uri(entry.fit_identity).relative_path / WEIGHTS_FILE
    expected = r"recorded digest|cannot be read"
    with _corrupted(weights, content), pytest.raises(manual_evaluation.EvidenceIntegrityError, match=expected):
        evaluate_trial(fixture_search, **kwargs)


def test_a_sampled_fit_reproduces_in_a_fresh_interpreter(
    fixture_search: ManualSearchProtocol, manual_fixture: ManualFixture, tmp_path: Path
) -> None:
    """Reservoir construction reproduces across processes, which is why workers are processes."""
    configuration = sampled_configuration(fixture_search, POINT, trial=1)
    entry = sampled_entry(manual_fixture.manifest, configuration, arm_of("M10"))
    inputs = replace(manual_fixture.inputs, sampled=(configuration,))
    cached = ManualFitStore(manual_fixture.store).fit_or_load(entry, inputs)
    refit = refit_in_subprocess(
        cached.record.identity,
        root=manual_fixture.root,
        parent_identity=manual_fixture.execution.identity,
        output=tmp_path / "refit.json",
        env=manual_fixture.env,
    )
    assert refit.passed, refit
    assert refit.weights_bitwise_equal


# --- the parent loop: reservations, recovery, verification and deadlines ------------------------


class _FakeWorker:
    """A worker that writes the reports the test wants, without fitting or simulating anything."""

    def __init__(self, reports: list[TrialResult | str | None]) -> None:
        self.reports = reports
        self.calls: list[int] = []

    def __call__(self, command: Sequence[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[bytes]:
        command = list(command)
        trial = int(command[command.index("--trial") + 1])
        report = self.reports[len(self.calls)] if len(self.calls) < len(self.reports) else None
        self.calls.append(trial)
        if isinstance(report, str):  # an interruption: nothing written, non-zero exit
            if report == "timeout":
                raise subprocess.TimeoutExpired(command, timeout or 0.0)
            return subprocess.CompletedProcess(command, returncode=1, stdout=b"", stderr=report.encode())
        if report is None:
            return subprocess.CompletedProcess(command, returncode=1, stdout=b"", stderr=b"the worker crashed")
        output = Path(command[command.index("--output") + 1])
        write_record(output, replace(report, trial=trial))
        return subprocess.CompletedProcess(command, returncode=0, stdout=b"", stderr=b"")


def _search(protocol: ManualSearchProtocol, worker: _FakeWorker, monkeypatch: pytest.MonkeyPatch) -> BudgetLedger:
    monkeypatch.setattr(manual_search_run, "spawn_trial", worker)
    return manual_search_run.run_search(protocol, PROTOCOL_FILE, root=ROOT, exploratory=True)


@pytest.fixture
def search_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> StorageRoot:
    """A store of this test's own: two protocols are two studies, and neither is the configured store."""
    root = tmp_path / "store"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(ENV_VAR, str(root))
    return StorageRoot(root)


def _failed_report() -> TrialResult:
    return TrialResult(
        trial=0,
        successes=0,
        runs=0,
        statuses=(),
        scenarios=("nominal",),
        seconds=0.5,
        failure="the fit did not converge",
    )


def test_a_failed_fit_consumes_its_trial_and_is_charged(
    protocol: ManualSearchProtocol, search_store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A candidate that could not be fitted is a failed trial, and what it stored is still charged."""
    tightened = replace(protocol, budget=replace(protocol.budget, trials=1))
    ledger = _search(tightened, _FakeWorker([_failed_report()]), monkeypatch)
    assert ledger.trials == 1
    outcome = read_record(trial_directory(search_store, 0) / "outcome.json", TrialOutcome)
    assert (outcome.state, outcome.score, outcome.failure) == ("failed", None, "the fit did not converge")


@pytest.mark.parametrize("interruption", ["the worker crashed", "timeout"])
def test_an_interrupted_trial_stays_pending_and_resumes_under_its_own_number(
    protocol: ManualSearchProtocol, search_store: StorageRoot, monkeypatch: pytest.MonkeyPatch, interruption: str
) -> None:
    """An interruption is not a failed fit: the trial keeps its number and the resume finishes it."""
    tightened = replace(protocol, budget=replace(protocol.budget, trials=1))
    interrupted = _search(tightened, _FakeWorker([interruption]), monkeypatch)
    assert interrupted.trials == 0, "an interrupted trial has not been spent"
    assert interrupted.seconds > 0.0, "its measured cost is charged even though it has no outcome"
    assert [r.trial for r in pending_reservations(search_store)] == [0]
    resumed_worker = _FakeWorker([_failed_report()])
    resumed = _search(tightened, resumed_worker, monkeypatch)
    assert resumed_worker.calls == [0], "the resume finished trial 0 rather than opening trial 1"
    assert resumed.trials == 1
    assert pending_reservations(search_store) == ()


def test_the_ledger_is_derived_from_the_retained_records(
    protocol: ManualSearchProtocol, search_store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accounting a resume can recover has to be derivable, not carried in a counter that can lag."""
    tightened = replace(protocol, budget=replace(protocol.budget, trials=2))
    _search(tightened, _FakeWorker([_failed_report(), _failed_report()]), monkeypatch)
    derived = ledger_of(search_store)
    assert derived.trials == 2
    (trial_directory(search_store, 1) / "outcome.json").unlink()
    assert ledger_of(search_store).trials == 1, "the ledger follows the records, not a stored total"


@pytest.mark.usefixtures("search_store")
def test_a_spent_cap_schedules_no_worker(protocol: ManualSearchProtocol, monkeypatch: pytest.MonkeyPatch) -> None:
    """A resumed invocation with its budget spent does nothing at all."""
    tightened = replace(protocol, budget=replace(protocol.budget, trials=1))
    _search(tightened, _FakeWorker([_failed_report()]), monkeypatch)
    resumed = _FakeWorker([_failed_report()])
    _search(tightened, resumed, monkeypatch)
    assert resumed.calls == []


@pytest.mark.usefixtures("search_store")
def test_a_worker_is_given_the_remaining_allowance_as_its_timeout(
    protocol: ManualSearchProtocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The elapsed cap has to constrain a running worker, not only the gap between trials."""
    seen: list[float | None] = []

    def recording(command: Sequence[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[bytes]:
        seen.append(timeout)
        return _FakeWorker([_failed_report()])(command, timeout=timeout)

    tightened = replace(protocol, budget=replace(protocol.budget, trials=1, hours=0.5))
    monkeypatch.setattr(manual_search_run, "spawn_trial", recording)
    manual_search_run.run_search(tightened, PROTOCOL_FILE, root=ROOT, exploratory=True)
    assert seen == [pytest.approx(0.5 * 3600.0 - manual_search_run.HEADROOM_S)]


def test_a_report_for_another_trial_is_refused(search_store: StorageRoot) -> None:
    """A report is not evidence: it must answer for the trial that was scheduled."""
    reservation = TrialReservation(
        trial=0,
        configuration="search-t0000",
        point=POINT,
        protocol_sha256="a" * 64,
        fit_identity="c" * 64,
        evidence_identity="d" * 64,
    )
    with pytest.raises(ValueError, match="names trial 999"):
        verified_outcome(search_store, reservation, replace(_failed_report(), trial=999), seconds=1.0)


def test_a_scored_report_must_point_at_evidence() -> None:
    """Counts alone cannot be scored; a scored candidate says where its evidence is."""
    with pytest.raises(ValueError, match="evidence"):
        TrialResult(
            trial=0,
            successes=2,
            runs=2,
            statuses=("completed", "completed"),
            scenarios=("nominal",),
            seconds=1.0,
        )


# --- what the owner's second review of M3MS-003 found -------------------------------------------


class _SlowWorker:
    """A worker that takes a known time and then reports what the test wants."""

    def __init__(self, durations: list[float], report: TrialResult | None) -> None:
        self.durations = durations
        self.report = report
        self.calls = 0
        self.timeouts: list[float] = []

    def __call__(self, command: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[bytes]:
        command = list(command)
        self.timeouts.append(timeout)
        duration = self.durations[min(self.calls, len(self.durations) - 1)]
        self.calls += 1
        monotonic = time.monotonic
        # Charge the attempt without actually sleeping: the parent measures elapsed wall clock.
        start = monotonic()
        while monotonic() - start < duration:
            pass
        if self.report is None or self.calls < len(self.durations):
            return subprocess.CompletedProcess(command, returncode=1, stdout=b"", stderr=b"interrupted")
        write_record(
            Path(command[command.index("--output") + 1]),
            replace(self.report, trial=int(command[command.index("--trial") + 1])),
        )
        return subprocess.CompletedProcess(command, returncode=0, stdout=b"", stderr=b"")


@pytest.mark.usefixtures("search_store")
def test_every_attempt_of_a_retried_trial_is_charged(
    protocol: ManualSearchProtocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry adds to the charge; replacing it would give interrupted work away for free."""
    tightened = replace(protocol, budget=replace(protocol.budget, trials=1))
    worker = _SlowWorker([0.05, 0.05, 0.05], _failed_report())
    monkeypatch.setattr(manual_search_run, "spawn_trial", worker)
    first = manual_search_run.run_search(tightened, PROTOCOL_FILE, root=ROOT, exploratory=True)
    second = manual_search_run.run_search(tightened, PROTOCOL_FILE, root=ROOT, exploratory=True)
    third = manual_search_run.run_search(tightened, PROTOCOL_FILE, root=ROOT, exploratory=True)
    assert second.seconds > first.seconds, "the second attempt added its cost"
    assert third.seconds > second.seconds, "and so did the third, which finalized the trial"
    assert third.trials == 1


@pytest.mark.usefixtures("search_store")
def test_no_worker_starts_without_a_real_deadline(
    protocol: ManualSearchProtocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the allowance inside the persistence headroom, scheduling stops instead of running unbounded."""
    tightened = replace(protocol, budget=replace(protocol.budget, trials=1, hours=0.01))
    worker = _FakeWorker([_failed_report()])
    monkeypatch.setattr(manual_search_run, "spawn_trial", worker)
    ledger = manual_search_run.run_search(tightened, PROTOCOL_FILE, root=ROOT, exploratory=True)
    assert worker.calls == [], "a worker with no allowance is never started"
    assert ledger.trials == 0


def test_a_worker_is_never_spawned_with_a_nonpositive_timeout() -> None:
    """The guard is in the spawn itself, not only in the caller that computes the allowance."""
    with pytest.raises(ValueError, match="deadline"):
        manual_search_run.spawn_trial([sys.executable, "-c", "pass"], timeout=0.0)


def _open(protocol: ManualSearchProtocol, store: StorageRoot) -> object:
    return manual_search_run.open_study(
        store,
        manual_search_run.STUDY_NAME,
        protocol_sha256=manual_search_run.protocol_digest(protocol),
        sampler=protocol.sampler,
        pruner=manual_search_run.PrunerSpec(kind="none"),
        direction="maximize",
    )


def _outcomes(store: StorageRoot) -> set[int]:
    return {
        read_record(directory / "outcome.json", TrialOutcome).trial
        for directory in manual_search_run.trial_directories(store)
        if (directory / "outcome.json").is_file()
    }


def test_a_trial_lost_before_its_reservation_is_adopted_under_its_own_number(
    protocol: ManualSearchProtocol, search_store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interruption between drawing a point and publishing its reservation must not orphan the trial."""
    tightened = replace(protocol, budget=replace(protocol.budget, trials=2))
    study = cast("Any", _open(tightened, search_store))
    trial = study.ask()
    manual_search_run.suggest_sampled_point(tightened.space, trial)  # drawn, then lost
    manual_search_run.close_study(study)
    monkeypatch.setattr(manual_search_run, "spawn_trial", _FakeWorker([_failed_report()]))
    manual_search_run.run_search(tightened, PROTOCOL_FILE, root=ROOT, exploratory=True)
    assert trial.number in _outcomes(search_store), "the lost trial was finished under its own number"


def test_an_outcome_the_study_never_heard_of_is_finalized_on_resume(
    protocol: ManualSearchProtocol, search_store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interruption between publishing an outcome and telling the study leaves no trial running."""
    tightened = replace(protocol, budget=replace(protocol.budget, trials=2))
    monkeypatch.setattr(manual_search_run, "spawn_trial", _FakeWorker([_failed_report()]))
    manual_search_run.run_search(tightened, PROTOCOL_FILE, root=ROOT, exploratory=True)
    study = cast("Any", _open(tightened, search_store))
    lost = study.ask()
    manual_search_run.suggest_sampled_point(tightened.space, lost)
    write_record(
        trial_directory(search_store, lost.number) / "outcome.json",
        TrialOutcome(
            trial=lost.number,
            state="failed",
            score=None,
            successes=0,
            runs=0,
            seconds=1.0,
            stored_bytes=0,
            failure="written before the study was told",
        ),
    )
    manual_search_run.close_study(study)
    monkeypatch.setattr(manual_search_run, "spawn_trial", _FakeWorker([]))
    manual_search_run.run_search(tightened, PROTOCOL_FILE, root=ROOT, exploratory=True)
    study = cast("Any", _open(tightened, search_store))
    try:
        states = {t.number: t.state.name for t in study.trials}
    finally:
        manual_search_run.close_study(study)
    assert states[lost.number] == "FAIL", "the study was reconciled with the retained record"


# --- what the owner's third review of M3MS-003 found --------------------------------------------


class _Clock:
    """A wall clock the test advances by hand, so a lost attempt's charge is exact."""

    def __init__(self, instants: Sequence[datetime]) -> None:
        self.instants = list(instants)
        self.calls = 0

    def __call__(self) -> datetime:
        instant = self.instants[min(self.calls, len(self.instants) - 1)]
        self.calls += 1
        return instant


def _killed_parent(command: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[bytes]:
    """A worker whose parent dies before it can measure the attempt it launched."""
    del command, timeout
    raise KeyboardInterrupt


def test_a_parent_killed_mid_attempt_still_charges_the_time_it_ran(
    protocol: ManualSearchProtocol, search_store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent that never returns cannot measure its own attempt, so the start is persisted first."""
    tightened = replace(protocol, budget=replace(protocol.budget, trials=1))
    opened = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(manual_search_run, "_now", _Clock([opened, opened + timedelta(minutes=5)]))
    monkeypatch.setattr(manual_search_run, "spawn_trial", _killed_parent)
    with pytest.raises(KeyboardInterrupt):
        manual_search_run.run_search(tightened, PROTOCOL_FILE, root=ROOT, exploratory=True)
    spend = read_record(trial_directory(search_store, 0) / "spend.json", TrialSpend)
    assert spend.started_at == opened.isoformat(), "the attempt start is on disk before the worker runs"
    monkeypatch.setattr(manual_search_run, "spawn_trial", _FakeWorker([_failed_report()]))
    resumed = manual_search_run.run_search(tightened, PROTOCOL_FILE, root=ROOT, exploratory=True)
    assert resumed.seconds >= 300.0, "the five minutes the lost attempt was in flight are charged"


def _reservation(trial: int = 0) -> TrialReservation:
    return TrialReservation(
        trial=trial,
        configuration=f"search-t{trial:04d}",
        point=POINT,
        protocol_sha256="a" * 64,
        fit_identity="c" * 64,
        evidence_identity="d" * 64,
    )


def _evidence_directory(store: StorageRoot, reservation: TrialReservation) -> Path:
    uri = manual_evaluation.model_uri(reservation.evidence_identity)
    directory = store.path(f"{uri}/{manual_evaluation.PROGRESS_FILE}", mode="write").parent
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _stored_run(store: StorageRoot, artifact_id: str, *, size: int) -> None:
    payload = store.path(f"armrc://runs/{artifact_id}/{RUN_SUMMARY_FILE}", mode="write")
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_bytes(b"x" * size)


def test_a_run_published_before_its_progress_record_is_still_charged(search_store: StorageRoot) -> None:
    """Ownership is discoverable before publication, so a payload cannot land outside the ceiling."""
    reservation = _reservation()
    directory = _evidence_directory(search_store, reservation)
    artifact_id = "run-20260923-abcdefabcdef"
    _stored_run(search_store, artifact_id, size=4096)
    claims = manual_evaluation.RunClaims(
        search_store, manual_evaluation.model_uri(reservation.evidence_identity), reservation.evidence_identity
    )
    claims.add(artifact_id)
    assert (directory / manual_evaluation.RUN_CLAIMS_FILE).is_file(), "the claim is what makes it discoverable"
    assert manual_search_run.retained_bytes(search_store, reservation) >= 4096


@pytest.mark.parametrize("name", [manual_evaluation.PROGRESS_FILE, manual_evaluation.RUN_CLAIMS_FILE])
def test_an_unreadable_inventory_is_not_an_empty_one(search_store: StorageRoot, name: str) -> None:
    """Bytes that cannot be counted are not bytes that are not there."""
    reservation = _reservation()
    (_evidence_directory(search_store, reservation) / name).write_text("{ truncated", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be read"):
        manual_search_run.retained_bytes(search_store, reservation)


def test_a_trial_abandoned_for_unusable_parameters_still_spends_its_slot(
    protocol: ManualSearchProtocol, search_store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trial lost while it was being sampled is charged, not handed back as a free replacement."""
    tightened = replace(protocol, budget=replace(protocol.budget, trials=1))
    study = cast("Any", _open(tightened, search_store))
    lost = study.ask()  # drawn, then lost before a single parameter was recorded
    manual_search_run.close_study(study)
    worker = _FakeWorker([_failed_report()])
    monkeypatch.setattr(manual_search_run, "spawn_trial", worker)
    ledger = manual_search_run.run_search(tightened, PROTOCOL_FILE, root=ROOT, exploratory=True)
    assert lost.number in _outcomes(search_store), "the abandonment is a retained record, not a silent skip"
    assert ledger.trials == 1, "it consumed the trial it was given"
    assert worker.calls == [], "so the one-trial cap scheduled nothing after it"


def test_bytes_staged_before_any_claim_stay_charged_across_finalization(search_store: StorageRoot) -> None:
    """A payload killed between staging and its claim has no owner, and finalizing a trial does not remove it.

    Charging it only while a reservation is pending would hand the search a
    fresh allowance at every finalization, though the bytes are still there.
    """
    reservation = _reservation()
    directory = trial_directory(search_store, reservation.trial)
    write_record(directory / "reservation.json", reservation)
    staging = search_store.root / "runs" / "staging-0123456789abcdef"
    staging.mkdir(parents=True)
    (staging / "arrays.npz").write_bytes(b"x" * 1_000_000)
    assert manual_search_run.unpublished_bytes(search_store) == 1_000_000
    assert ledger_of(search_store).stored_bytes >= 1_000_000, "the staged payload is charged while it is in flight"
    write_record(
        directory / "outcome.json",
        TrialOutcome(
            trial=reservation.trial,
            state="failed",
            score=None,
            successes=0,
            runs=0,
            seconds=1.0,
            stored_bytes=0,
            failure="the fit did not converge",
        ),
    )
    finalized = ledger_of(search_store)
    assert finalized.trials == 1
    assert finalized.stored_bytes >= 1_000_000, "and it is still charged once the trial is finalized"


# --- M3MS-005: one statement of when the search has stopped -------------------------------------


@pytest.mark.parametrize(
    ("ledger", "expected"),
    [
        (BudgetLedger(trials=99, seconds=0.0, stored_bytes=0), []),
        (BudgetLedger(trials=100, seconds=0.0, stored_bytes=0), ["100-trial cap"]),
        (BudgetLedger(trials=10, seconds=10 * 3600.0 - 30.0, stored_bytes=0), ["headroom"]),
    ],
)
def test_the_search_stops_where_scheduling_stops(
    protocol: ManualSearchProtocol, ledger: BudgetLedger, expected: list[str]
) -> None:
    """A spent cap, or an allowance inside the persistence headroom, is a search that schedules nothing more."""
    stopped = manual_search_run.search_stopped(ledger, protocol.budget)
    assert len(stopped) == len(expected)
    for reason, fragment in zip(stopped, expected, strict=True):
        assert fragment in reason


def test_a_frozen_trial_is_verified_again_against_its_evidence(
    fixture_search: ManualSearchProtocol, manual_fixture: ManualFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The freeze does not take a recorded verdict on trust: scored afresh, the evidence must agree with it."""
    for name, value in manual_fixture.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(manual_evaluation, "repository_root", lambda: manual_fixture.root)
    monkeypatch.setattr(manual_evaluation, "evaluation_scenarios", manual_narrowed)
    result = evaluate_trial(
        fixture_search,
        trial=0,
        point=POINT,
        scenarios=("nominal",),
        root=manual_fixture.root,
        argv=["evaluate-trial"],
        exploratory=True,
    )
    inputs = manual_search_run.SearchInputs(fixture_search, root=manual_fixture.root, exploratory=True)
    reservation = manual_search_run.reserve_trial(fixture_search, inputs, POINT, trial=0)
    outcome = verified_outcome(
        manual_fixture.store, reservation, result, seconds=1.0, protocol=fixture_search, inputs=inputs
    )
    reverify_outcome(manual_fixture.store, reservation, outcome, result, protocol=fixture_search, inputs=inputs)
    flipped = 1 if outcome.successes != 1 else 2
    claimed = replace(outcome, successes=flipped, score=flipped / 2)
    with pytest.raises(ValueError, match="re-verification disagrees"):
        reverify_outcome(manual_fixture.store, reservation, claimed, result, protocol=fixture_search, inputs=inputs)
