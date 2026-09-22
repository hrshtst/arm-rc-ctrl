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
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.experiments import manual_evaluation, manual_search_run
from arm_rc_ctrl.experiments.manual_fits import ManualFitStore
from arm_rc_ctrl.experiments.manual_fixture import ManualFixture, ManualStudyEvidence, manual_narrowed
from arm_rc_ctrl.experiments.manual_numerics import refit_in_subprocess
from arm_rc_ctrl.experiments.manual_sampled import SampledPoint, arm_of, sampled_configuration, sampled_entry
from arm_rc_ctrl.experiments.manual_search import ManualSearchBudget, load_manual_search
from arm_rc_ctrl.experiments.manual_search_run import (
    BudgetLedger,
    TrialResult,
    budget_complaints,
    evaluate_trial,
    nominal_scope_mismatches,
    read_result,
    trial_command,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import ENV_VAR

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


def _result(**changes: object) -> TrialResult:
    base = TrialResult(
        trial=1,
        successes=1,
        runs=2,
        statuses=("completed", "infeasible"),
        scenarios=("nominal",),
        seconds=3.0,
        stored_bytes=2048,
    )
    return replace(base, **changes)


def test_a_result_scores_the_two_nominal_runs() -> None:
    """The score is the success fraction over the runs the worker actually judged."""
    assert _result().score == 0.5
    assert _result(successes=2).score == 1.0
    assert _result(successes=0).score == 0.0


def test_a_failed_candidate_has_no_score() -> None:
    """A fit failure or an interruption is retained as itself, never read as a zero."""
    assert _result(failure="the fit did not converge").score is None


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


# --- the worker against a real study ------------------------------------------------------------


def _fixture_protocol(f: ManualFixture, evaluation: Path) -> Path:
    """A search protocol over the fixture study, written where the fixture root is the repository."""
    filters = f.root / "configs" / "evaluations" / "fixture_filters.toml"
    filters.parent.mkdir(parents=True, exist_ok=True)
    filters.write_text(
        'name = "fixture-filters"\ntracker = "../controllers/task_1a_pd_v2.toml"\n\n'
        "[estimator]\nvelocity_cutoff_hz = 20.0\nacceleration_cutoff_hz = 8.0\nmax_dt_ratio = 3.0\n",
        encoding="utf-8",
    )
    body = PROTOCOL_FILE.read_text(encoding="utf-8")
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
        assert old in body, old
        body = body.replace(old, new, 1)
    target = f.root / "configs" / "studies" / "fixture_search.toml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


@pytest.fixture
def fixture_search(
    manual_fixture: ManualFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ManualSearchProtocol:
    """The committed protocol, re-pointed at the fixture study, with that root as the repository."""
    base = tmp_path / "study"
    base.mkdir(parents=True, exist_ok=True)
    evidence = ManualStudyEvidence(manual_fixture, base)
    target = _fixture_protocol(manual_fixture, evidence.evaluation)
    # Only the protocol's own view of the repository moves: provenance keeps the real checkout, which
    # is what the fixture study's own evidence was recorded under.
    monkeypatch.setattr("arm_rc_ctrl.experiments.manual_search.repository_root", lambda: manual_fixture.root)
    return load_manual_search(target)


def test_the_worker_evaluates_the_nominal_case_under_both_trackers(
    fixture_search: ManualSearchProtocol, manual_fixture: ManualFixture, monkeypatch: pytest.MonkeyPatch
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
    assert result.score in (0.0, 0.5, 1.0)
    assert result.stored_bytes > 0, "the candidate's runs are retained whatever its verdict"


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


# --- the parent loop, its accounting and its failures --------------------------------------------


class _FakeWorker:
    """A worker that reports what the test wants, without fitting or simulating anything."""

    def __init__(self, results: list[TrialResult | None]) -> None:
        self.results = results
        self.calls = 0

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
        command = list(command)
        result = self.results[self.calls] if self.calls < len(self.results) else None
        self.calls += 1
        if result is None:
            return subprocess.CompletedProcess(command, returncode=1, stdout=b"", stderr=b"the fit did not converge")
        output = Path(command[command.index("--output") + 1])
        trial = int(command[command.index("--trial") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(to_mapping(replace(result, trial=trial)), sort_keys=True), encoding="utf-8")
        return subprocess.CompletedProcess(command, returncode=0, stdout=b"", stderr=b"")


def _search(
    protocol: ManualSearchProtocol, worker: _FakeWorker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> BudgetLedger:
    monkeypatch.setattr(manual_search_run, "spawn_trial", worker)
    return manual_search_run.run_search(
        protocol, PROTOCOL_FILE, root=ROOT, scratch=tmp_path / "trials", exploratory=True
    )


@pytest.fixture
def search_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A store of this test's own: two protocols are two studies, and neither is the configured store."""
    root = tmp_path / "store"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(ENV_VAR, str(root))


@pytest.mark.usefixtures("search_store")
def test_the_parent_spends_its_budget_once_across_resumes(
    protocol: ManualSearchProtocol, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spend is read back rather than restarted, and a spent cap schedules nothing."""
    tightened = replace(protocol, budget=replace(protocol.budget, trials=2))
    scored = _result(successes=2, stored_bytes=1024)
    # One scored candidate, then a worker that fails: both consume a trial of the two-trial cap.
    worker = _FakeWorker([scored])
    first = _search(tightened, worker, tmp_path, monkeypatch)
    assert (first.trials, first.stored_bytes) == (2, 1024)
    assert first.seconds > 0.0
    resumed = _FakeWorker([scored, scored])
    second = _search(tightened, resumed, tmp_path, monkeypatch)
    assert second == first, "the resumed invocation read the spend back instead of starting a fresh allowance"
    assert resumed.calls == 0, "a spent cap schedules no worker at all"


@pytest.mark.usefixtures("search_store")
def test_a_failed_worker_consumes_its_trial(
    protocol: ManualSearchProtocol, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A candidate that could not be fitted is retained as a failed trial, not replaced for free."""
    tightened = replace(protocol, budget=replace(protocol.budget, trials=1))
    ledger = _search(tightened, _FakeWorker([None]), tmp_path, monkeypatch)
    assert ledger.trials == 1
