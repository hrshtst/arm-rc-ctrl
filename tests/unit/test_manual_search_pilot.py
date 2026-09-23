# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MS-004: the bounded timing pilot is the search's own first trials, measured and counted.

The pilot is not a separate study. It stops the approved search early, so its
trials consume the 100-trial cap and a later invocation continues them under
the next numbers. What it adds is measurement: the worker reports how its time
divided between preparing, fitting and simulating, the parent charges the time
it spends verifying a candidate, and a preflight stated before the pilot is
compared with what the store and the study hold afterwards.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.experiments import manual_search_pilot, manual_search_run
from arm_rc_ctrl.experiments.manual_search import load_manual_search
from arm_rc_ctrl.experiments.manual_search_pilot import (
    SearchPreflight,
    count_disagreements,
    observe_search,
    pilot_report,
    render_markdown,
    search_preflight,
)
from arm_rc_ctrl.experiments.manual_search_run import (
    BudgetLedger,
    SearchInvocation,
    TrialOutcome,
    TrialResult,
    TrialTiming,
    invocation_records,
    read_record,
    trial_directory,
    write_record,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import ENV_VAR, StorageRoot

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.experiments.manual_search import ManualSearchProtocol

ROOT = repository_root()
PROTOCOL_FILE = ROOT / "configs/studies/manual_esn_search_v1.toml"
TIMING = TrialTiming(
    prepare_seconds=0.4,
    fit_seconds=0.9,
    fit_cache_hit=False,
    sweep_seconds=3.0,
    simulate_seconds=2.6,
    persist_seconds=0.2,
    run_bytes=700_000,
    simulated_runs=0,
)


@pytest.fixture(scope="module")
def protocol() -> ManualSearchProtocol:
    """The committed search protocol."""
    return load_manual_search(PROTOCOL_FILE)


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> StorageRoot:
    """A store of this test's own, so each test is its own search."""
    root = tmp_path / "store"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(ENV_VAR, str(root))
    return StorageRoot(root)


def _failed(**changes: object) -> TrialResult:
    report = TrialResult(
        trial=0,
        successes=0,
        runs=0,
        statuses=(),
        scenarios=("nominal",),
        seconds=0.5,
        failure="the fit did not converge",
        timing=TIMING,
    )
    return replace(report, **changes)


class _Worker:
    """A worker that writes the reports the test wants (``None`` is an interruption)."""

    def __init__(self, reports: Sequence[TrialResult | None]) -> None:
        self.reports = list(reports)
        self.calls: list[int] = []

    def __call__(self, command: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[bytes]:
        del timeout
        command = list(command)
        trial = int(command[command.index("--trial") + 1])
        report = self.reports[len(self.calls)] if len(self.calls) < len(self.reports) else _failed()
        self.calls.append(trial)
        if report is None:
            return subprocess.CompletedProcess(command, returncode=1, stdout=b"", stderr=b"the worker crashed")
        write_record(Path(command[command.index("--output") + 1]), replace(report, trial=trial))
        return subprocess.CompletedProcess(command, returncode=0, stdout=b"", stderr=b"")


def _search(
    protocol: ManualSearchProtocol,
    worker: _Worker,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stop_at_trials: int | None = None,
) -> BudgetLedger:
    monkeypatch.setattr(manual_search_run, "spawn_trial", worker)
    return manual_search_run.run_search(
        protocol, PROTOCOL_FILE, root=ROOT, exploratory=True, stop_at_trials=stop_at_trials
    )


def _outcome(store: StorageRoot, trial: int) -> TrialOutcome:
    return read_record(trial_directory(store, trial) / "outcome.json", TrialOutcome)


# --- the pilot is the search's own first trials -------------------------------------------------


@pytest.mark.usefixtures("store")
def test_a_pilot_bound_stops_the_search_inside_the_cap(
    protocol: ManualSearchProtocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pilot of two trials runs exactly two trials of the 100 the cap allows, and stops."""
    worker = _Worker([_failed(), _failed()])
    ledger = _search(protocol, worker, monkeypatch, stop_at_trials=2)
    assert worker.calls == [0, 1]
    assert ledger.trials == 2


@pytest.mark.usefixtures("store")
def test_pilot_trials_consume_the_cap_and_the_search_continues_them(
    protocol: ManualSearchProtocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pilot is not a free allowance: a resume counts its trials and draws the next numbers."""
    tightened = replace(protocol, budget=replace(protocol.budget, trials=3))
    _search(tightened, _Worker([_failed(), _failed()]), monkeypatch, stop_at_trials=2)
    repeated = _Worker([])
    _search(tightened, repeated, monkeypatch, stop_at_trials=2)
    assert repeated.calls == [], "a pilot bound already reached schedules nothing"
    continued = _Worker([_failed()])
    ledger = _search(tightened, continued, monkeypatch)
    assert continued.calls == [2], "the search goes on from the pilot, under the next number"
    assert ledger.trials == 3, "and the cap counts the pilot's trials"


@pytest.mark.parametrize("bound", [0, -1, 101])
@pytest.mark.usefixtures("store")
def test_a_pilot_bound_is_inside_the_cap(
    protocol: ManualSearchProtocol, monkeypatch: pytest.MonkeyPatch, bound: int
) -> None:
    """A bound can only stop the search earlier; it never names more trials than the cap allows."""
    worker = _Worker([])
    with pytest.raises(ValueError, match="pilot"):
        _search(protocol, worker, monkeypatch, stop_at_trials=bound)
    assert worker.calls == []


@pytest.mark.usefixtures("store")
def test_a_bound_above_a_tightened_cap_is_refused(
    protocol: ManualSearchProtocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound is checked against the cap in force, not the approved maximum."""
    tightened = replace(protocol, budget=replace(protocol.budget, trials=5))
    with pytest.raises(ValueError, match="pilot"):
        _search(tightened, _Worker([]), monkeypatch, stop_at_trials=6)


def test_a_pending_pilot_trial_is_finished_before_the_bound_stops_anything(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted pilot trial is still pending work, and a resume under the same bound finishes it."""
    _search(protocol, _Worker([None]), monkeypatch, stop_at_trials=1)
    assert manual_search_run.pending_reservations(store)
    resumed = _Worker([_failed()])
    ledger = _search(protocol, resumed, monkeypatch, stop_at_trials=1)
    assert resumed.calls == [0]
    assert ledger.trials == 1


# --- what the estimates omit is measured and charged --------------------------------------------


def test_the_parents_verification_is_charged_to_the_trial(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifying a candidate is elapsed execution, so the ceiling counts it with the worker's time."""
    original = manual_search_run.verified_outcome

    def slow(*args: object, **kwargs: object) -> TrialOutcome:
        time.sleep(0.2)
        return original(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(manual_search_run, "verified_outcome", slow)
    ledger = _search(protocol, _Worker([_failed()]), monkeypatch, stop_at_trials=1)
    outcome = _outcome(store, 0)
    assert outcome.verify_seconds is not None
    assert outcome.verify_seconds >= 0.2
    assert outcome.worker_seconds is not None
    assert outcome.seconds >= outcome.worker_seconds + outcome.verify_seconds
    assert ledger.seconds >= 0.2


def test_each_invocation_is_recorded_with_its_wall_clock(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent's own start-up and bookkeeping are measured, so the pilot can show what the ledger omits."""
    ledger = _search(protocol, _Worker([_failed(), _failed()]), monkeypatch, stop_at_trials=2)
    records = invocation_records(store)
    assert len(records) == 1
    record = records[0]
    assert isinstance(record, SearchInvocation)
    assert record.trials == (0, 1)
    assert record.stop_at_trials == 2
    assert record.charged_seconds == pytest.approx(ledger.seconds)
    assert record.seconds >= record.charged_seconds, "the invocation encloses every trial it charged"


def test_a_failed_invocation_is_recorded_too(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invocation that raises still spent its time, and the record says it did not complete."""

    def broken(*args: object, **kwargs: object) -> TrialOutcome:
        del args, kwargs
        msg = "the store refused the evidence"
        raise ValueError(msg)

    monkeypatch.setattr(manual_search_run, "verified_outcome", broken)
    with pytest.raises(ValueError, match="refused"):
        _search(protocol, _Worker([_failed()]), monkeypatch, stop_at_trials=1)
    (record,) = invocation_records(store)
    assert not record.completed


# --- the preflight and what was observed --------------------------------------------------------


def test_the_preflight_states_what_the_pilot_will_schedule(protocol: ManualSearchProtocol, store: StorageRoot) -> None:
    """Before anything runs: ten trials, at most twenty nominal runs, and the whole budget left."""
    preflight = search_preflight(protocol, store, stop_at_trials=10)
    assert preflight.scenarios == ("nominal",)
    assert (preflight.spent_trials, preflight.pending_trials, preflight.trials_to_schedule) == (0, 0, 10)
    assert preflight.max_rc_runs == 20
    assert preflight.trial_cap == 100
    assert preflight.spent_seconds == 0.0


def test_the_preflight_counts_what_was_already_spent(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later preflight schedules only what the cap and the bound leave, pending work included."""
    _search(protocol, _Worker([_failed(), _failed(), None]), monkeypatch, stop_at_trials=3)
    preflight = search_preflight(protocol, store, stop_at_trials=10)
    assert (preflight.spent_trials, preflight.pending_trials, preflight.trials_to_schedule) == (2, 1, 8)
    assert search_preflight(protocol, store, stop_at_trials=None).trials_to_schedule == 98


def test_a_preflight_bound_outside_the_cap_is_refused(protocol: ManualSearchProtocol, store: StorageRoot) -> None:
    """The preflight applies the search's own rule, so it cannot plan a pilot the search would refuse."""
    with pytest.raises(ValueError, match="pilot"):
        search_preflight(protocol, store, stop_at_trials=101)


def test_observed_counts_agree_with_the_preflight(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Records, ledger and study all say the same number of trials and runs as the preflight planned."""
    preflight = search_preflight(protocol, store, stop_at_trials=2)
    _search(protocol, _Worker([_failed(), _failed()]), monkeypatch, stop_at_trials=2)
    observation = observe_search(protocol, store)
    assert (observation.finalized, observation.failed, observation.pending) == (2, 2, 0)
    assert observation.study_states == {"FAIL": 2}
    assert count_disagreements(preflight, observation) == []


def test_a_shortfall_against_the_preflight_is_reported(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fewer trials than planned is a disagreement, not a smaller pilot."""
    preflight = search_preflight(protocol, store, stop_at_trials=3)
    _search(protocol, _Worker([_failed(), _failed()]), monkeypatch, stop_at_trials=2)
    disagreements = count_disagreements(preflight, observe_search(protocol, store))
    assert any("3" in item and "2" in item for item in disagreements), disagreements


def test_a_record_the_study_does_not_agree_with_is_reported(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Losing an outcome leaves the study and the records disagreeing, and the check says so."""
    preflight = search_preflight(protocol, store, stop_at_trials=2)
    _search(protocol, _Worker([_failed(), _failed()]), monkeypatch, stop_at_trials=2)
    (trial_directory(store, 1) / "outcome.json").unlink()
    disagreements = count_disagreements(preflight, observe_search(protocol, store))
    assert any("pending" in item for item in disagreements), disagreements
    assert any("study" in item for item in disagreements), disagreements


def test_a_preflight_from_another_protocol_is_refused(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counts compared across two protocols would agree by accident, so the comparison refuses them."""
    preflight = replace(search_preflight(protocol, store, stop_at_trials=1), protocol_sha256="0" * 64)
    _search(protocol, _Worker([_failed()]), monkeypatch, stop_at_trials=1)
    assert any("protocol" in item for item in count_disagreements(preflight, observe_search(protocol, store)))


# --- the report ---------------------------------------------------------------------------------


def test_the_report_divides_each_trials_time_and_projects_the_rest(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-trial phases, the uncharged remainder, and projections for the search and the comparison."""
    preflight = search_preflight(protocol, store, stop_at_trials=2)
    _search(protocol, _Worker([_failed(), _failed()]), monkeypatch, stop_at_trials=2)
    report = pilot_report(protocol, store, preflight)
    assert report.disagreements == ()
    assert [row.trial for row in report.trials] == [0, 1]
    row = report.trials[0]
    assert row.prepare_seconds == TIMING.prepare_seconds
    assert row.fit_seconds == TIMING.fit_seconds
    assert row.verify_seconds is not None
    assert row.worker_seconds is not None
    assert report.invocation_seconds >= report.ledger.seconds
    assert report.projection.search_trials == 100
    assert report.projection.search_seconds == pytest.approx(
        report.ledger.seconds / 2 * 100 + report.uncharged.invocation_overhead_seconds
    ), "the charged rate over the whole cap, plus the parent overhead the ledger does not see"
    assert report.uncharged.record_bytes > 0, "the trial records themselves are not in the ledger"


def test_the_report_is_portable_and_reproducible(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Committed evidence names no machine path, and rendering the same records twice gives the same text."""
    preflight = search_preflight(protocol, store, stop_at_trials=1)
    _search(protocol, _Worker([_failed()]), monkeypatch, stop_at_trials=1)
    first = manual_search_pilot.report_json(pilot_report(protocol, store, preflight))
    second = manual_search_pilot.report_json(pilot_report(protocol, store, preflight))
    assert first == second
    assert str(store.root) not in first
    assert str(Path.home()) not in first
    markdown = render_markdown(pilot_report(protocol, store, preflight))
    assert "| 0 |" in markdown
    assert str(store.root) not in markdown


def test_a_preflight_round_trips_through_its_file(protocol: ManualSearchProtocol, store: StorageRoot) -> None:
    """The preflight is committed before the pilot runs and read back by the report."""
    preflight = search_preflight(protocol, store, stop_at_trials=10)
    path = store.root / "preflight.json"
    manual_search_pilot.write_preflight(path, preflight)
    assert manual_search_pilot.read_preflight(path) == preflight
    assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
    assert isinstance(preflight, SearchPreflight)
