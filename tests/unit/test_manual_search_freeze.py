# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MS-005: the freeze of the three highest nominal scores.

The freeze is refused until the search has stopped and its study agrees with
its records; the selection is the protocol's rule and nothing else; each
chosen trial's evidence is verified again; and the record refuses to exist
when it does not follow the rule, so an edited freeze cannot be loaded.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.experiments import manual_search_freeze, manual_search_run
from arm_rc_ctrl.experiments.manual_search import load_manual_search
from arm_rc_ctrl.experiments.manual_search_freeze import (
    freeze_search,
    read_freeze,
    render_freeze,
    write_freeze,
)
from arm_rc_ctrl.experiments.manual_search_run import (
    BudgetLedger,
    SearchInputs,
    TrialOutcome,
    TrialReservation,
    TrialResult,
    read_record,
    trial_directory,
    write_record,
)
from arm_rc_ctrl.provenance import ArtifactReference
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import ENV_VAR, StorageRoot

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.experiments.manual_search import ManualSearchProtocol

ROOT = repository_root()
PROTOCOL_FILE = ROOT / "configs/studies/manual_esn_search_v1.toml"
EVIDENCE = ArtifactReference(uri="armrc://reports/task_1a_manual_v1/model/x/manifest.json", sha256="b" * 64, size=1)
NO_INPUTS = cast("SearchInputs", object())
"""The freeze's inputs, never used while re-verification is replaced."""


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


def _report(successes: int | None) -> TrialResult:
    """A worker's report: ``None`` is a failed fit, otherwise that many of two nominal runs succeeded."""
    if successes is None:
        return TrialResult(
            trial=0, successes=0, runs=0, statuses=(), scenarios=("nominal",), seconds=0.1, failure="no fit"
        )
    statuses = ("completed",) * successes + ("infeasible",) * (2 - successes)
    return TrialResult(
        trial=0, successes=successes, runs=2, statuses=statuses, scenarios=("nominal",), seconds=0.1, evidence=EVIDENCE
    )


def _unverified(
    store: StorageRoot, reservation: TrialReservation, result: TrialResult, *, seconds: float, **_: object
) -> TrialOutcome:
    """The parent's verdict without real evidence behind it: the counts the worker reported."""
    del store
    if result.failure is not None:
        return TrialOutcome(
            trial=reservation.trial,
            state="failed",
            score=None,
            successes=0,
            runs=0,
            seconds=seconds,
            stored_bytes=0,
            failure=result.failure,
        )
    return TrialOutcome(
        trial=reservation.trial,
        state="scored",
        score=result.successes / 2,
        successes=result.successes,
        runs=2,
        seconds=seconds,
        stored_bytes=0,
        evidence=EVIDENCE.uri,
    )


class _Worker:
    def __init__(self, reports: Sequence[TrialResult]) -> None:
        self.reports = list(reports)
        self.calls = 0

    def __call__(self, command: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[bytes]:
        del timeout
        command = list(command)
        trial = int(command[command.index("--trial") + 1])
        write_record(Path(command[command.index("--output") + 1]), replace(self.reports[self.calls], trial=trial))
        self.calls += 1
        return subprocess.CompletedProcess(command, returncode=0, stdout=b"", stderr=b"")


def _search(
    protocol: ManualSearchProtocol,
    monkeypatch: pytest.MonkeyPatch,
    successes: Sequence[int | None],
    *,
    stop_at_trials: int | None = None,
) -> ManualSearchProtocol:
    """Run a search of ``len(successes)`` trials whose workers report those verdicts."""
    tightened = replace(protocol, budget=replace(protocol.budget, trials=len(successes)))
    monkeypatch.setattr(manual_search_run, "spawn_trial", _Worker([_report(s) for s in successes]))
    monkeypatch.setattr(manual_search_run, "verified_outcome", _unverified)
    manual_search_run.run_search(tightened, PROTOCOL_FILE, root=ROOT, exploratory=True, stop_at_trials=stop_at_trials)
    return tightened


@pytest.fixture
def reverified(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace re-verification with a record of which trials it was asked about."""
    calls: list[int] = []

    def record(store: StorageRoot, reservation: TrialReservation, *args: object, **kwargs: object) -> None:
        del store, args, kwargs
        calls.append(reservation.trial)

    monkeypatch.setattr(manual_search_freeze, "reverify_outcome", record)
    return calls


# --- when a freeze may be taken -----------------------------------------------------------------


@pytest.mark.usefixtures("reverified")
def test_a_search_that_has_not_stopped_is_not_frozen(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pilot's early stop is not the end of the search: no cap is spent, so nothing is frozen."""
    searched = _search(protocol, monkeypatch, [2, 2, 2, 2], stop_at_trials=3)
    with pytest.raises(ValueError, match="no cap is spent"):
        freeze_search(searched, store, inputs=NO_INPUTS)


@pytest.mark.usefixtures("reverified")
def test_pending_work_is_finished_before_a_freeze(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reserved trial without an outcome could still score, so the freeze waits for it."""
    searched = _search(protocol, monkeypatch, [2, 2, 2])
    (trial_directory(store, 2) / "outcome.json").unlink()
    with pytest.raises(ValueError, match="pending"):
        freeze_search(searched, store, inputs=NO_INPUTS)


# --- the rule, and only the rule -----------------------------------------------------------------


def test_the_three_highest_nominal_scores_are_frozen_in_trial_order(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch, reverified: list[int]
) -> None:
    """Descending score, then the earliest trial; a failed fit and lower scores are never chosen."""
    searched = _search(protocol, monkeypatch, [1, 2, None, 2, 0, 2, 2])
    frozen = freeze_search(searched, store, inputs=NO_INPUTS)
    assert [item.trial for item in frozen.chosen] == [1, 3, 5]
    assert [item.rank for item in frozen.chosen] == [1, 2, 3]
    assert {item.score for item in frozen.chosen} == {1.0}
    assert frozen.shortfall == 0
    assert (frozen.finalized, frozen.scored, frozen.failed) == (7, 6, 1)
    assert frozen.score_counts == {"0.0": 1, "0.5": 1, "1.0": 4}
    assert frozen.stopped == ("the 7-trial cap is spent (7 trials)",)
    assert reverified == [1, 3, 5], "exactly the chosen trials are verified again"
    for item in frozen.chosen:
        reservation = read_record(trial_directory(store, item.trial) / "reservation.json", TrialReservation)
        assert (item.point, item.fit_identity, item.evidence_identity) == (
            reservation.point,
            reservation.fit_identity,
            reservation.evidence_identity,
        )


@pytest.mark.usefixtures("reverified")
def test_too_few_scored_configurations_are_a_shortfall(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fewer scored candidates than three is reported, not filled and not a reason to search on."""
    searched = _search(protocol, monkeypatch, [None, 0, None])
    frozen = freeze_search(searched, store, inputs=NO_INPUTS)
    assert [item.trial for item in frozen.chosen] == [1]
    assert frozen.shortfall == 2
    assert "the cap was not enlarged" in render_freeze(frozen)


# --- the study and the records must be the same search -------------------------------------------


@pytest.mark.usefixtures("reverified")
def test_a_score_the_study_did_not_see_is_refused(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An outcome edited after the study was told cannot decide the freeze."""
    searched = _search(protocol, monkeypatch, [2, 0, 2])
    path = trial_directory(store, 1) / "outcome.json"
    edited = replace(read_record(path, TrialOutcome), score=1.0, successes=2)
    write_record(path, edited)
    with pytest.raises(ValueError, match=r"trial 1 has value 0\.0 in the study"):
        freeze_search(searched, store, inputs=NO_INPUTS)


@pytest.mark.usefixtures("reverified")
def test_a_point_the_study_did_not_draw_is_refused(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reservation whose point differs from the study's parameters is not the trial the sampler drew."""
    searched = _search(protocol, monkeypatch, [2, 2, 2])
    path = trial_directory(store, 0) / "reservation.json"
    reservation = read_record(path, TrialReservation)
    moved = replace(reservation.point, warmup_s=2.0 if reservation.point.warmup_s != 2.0 else 0.0)
    write_record(path, replace(reservation, point=moved))
    with pytest.raises(ValueError, match="trial 0 reserved"):
        freeze_search(searched, store, inputs=NO_INPUTS)


# --- the record ---------------------------------------------------------------------------------


def _trusted(*args: object, **kwargs: object) -> None:
    """Re-verification replaced by trust, for tests of the record rather than the evidence."""
    del args, kwargs


@pytest.fixture
def frozen_file(
    protocol: ManualSearchProtocol, store: StorageRoot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    """A freeze written to its file."""
    monkeypatch.setattr(manual_search_freeze, "reverify_outcome", _trusted)
    searched = _search(protocol, monkeypatch, [2, 1, 2, 2])
    path = tmp_path / "selection_v1.json"
    write_freeze(path, freeze_search(searched, store, inputs=NO_INPUTS))
    return path


def test_a_freeze_round_trips_and_is_never_rewritten(frozen_file: Path) -> None:
    """The record reads back whole, and a second write to the same file is refused."""
    frozen = read_freeze(frozen_file)
    assert [item.trial for item in frozen.chosen] == [0, 2, 3]
    with pytest.raises(FileExistsError):
        write_freeze(frozen_file, frozen)


def _edited(path: Path, change: str) -> dict[str, object]:
    data = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    chosen = cast("list[dict[str, object]]", data["chosen"])
    if change == "reordered":
        chosen[0]["rank"], chosen[1]["rank"] = chosen[1]["rank"], chosen[0]["rank"]
        data["chosen"] = [chosen[1], chosen[0], chosen[2]]
    elif change == "best":
        data["label"] = "best configurations"
    elif change == "inflated":
        chosen[0]["successes"] = 1
    elif change == "duplicated":
        chosen[1]["point"] = chosen[0]["point"]
    elif change == "dropped":
        data["chosen"] = chosen[:2]
    elif change == "never stopped":
        data["stopped"] = []
    return data


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("reordered", "descending score, then ascending trial"),
        ("best", "not 'best configurations'"),
        ("inflated", "is not 1 of 2"),
        ("duplicated", "distinct parameter points"),
        ("dropped", "shortfall of 0 is not 3"),
        ("never stopped", "search that stopped"),
    ],
)
def test_an_edited_freeze_is_refused_where_it_is_loaded(
    frozen_file: Path, tmp_path: Path, change: str, reason: str
) -> None:
    """The rule is enforced by the record itself, so an edited file is not a freeze."""
    target = tmp_path / f"{change.replace(' ', '_')}.json"
    target.write_text(json.dumps(_edited(frozen_file, change)), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(reason)):
        read_freeze(target)


def test_the_rendering_names_the_highest_nominal_scores(frozen_file: Path) -> None:
    """The report calls them what the objective supports, and never the best."""
    text = render_freeze(read_freeze(frozen_file))
    assert "highest nominal scores" in text
    assert "best" not in text.lower()


def test_the_ledger_is_carried_into_the_freeze(frozen_file: Path) -> None:
    """The freeze records what the search spent of the shared ceiling, for the comparison that follows."""
    frozen = read_freeze(frozen_file)
    assert isinstance(frozen.ledger, BudgetLedger)
    assert frozen.ledger.trials == frozen.finalized == 4
    assert to_mapping(frozen)["study"] == manual_search_run.STUDY_NAME
