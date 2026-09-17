# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-008: bounded parallel execution of the sweep (clarification I1).

Workers are separate interpreters, and every one inherits the parent's declared
affinity together with its single-thread settings. That is not a detail: the
execution identity a run is keyed by includes the declared policy and CPU set,
so a worker pinned to its own subset would key its runs to a different
environment and they would not belong to this study at all. Isolation between
workers is one numerical thread each, over the shared canonical CPU set.

This module covers the scheduling and the serving. Constructing the worker
command belongs to the entry point, where the study and configuration paths are
known, and is covered there.
"""

from __future__ import annotations

import shutil
import threading
import time
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments.manual_evaluation import (
    PROGRESS_FILE,
    ManualEvaluationConfig,
    ManualEvaluationRunner,
    evaluate_in_parallel,
    load_manual_evaluation_config,
    replay_bank_uri,
)
from arm_rc_ctrl.experiments.perturbations import RobustnessScenario
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

if TYPE_CHECKING:
    from pathlib import Path

    from arm_rc_ctrl.experiments.manual_fixture import ManualFixture
    from arm_rc_ctrl.experiments.manual_study import StudyModel

REPO_ROOT = repository_root()
DEVELOPMENT_SOURCE = REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml"
TRACKER = TrackerConfig(type="pd", kp=(10.0, 5.0), kd=(1.5, 0.8))
HOLD_S, PULSE_S, HORIZON_S = 0.05, 0.02, 1.0
CONFIGURATION = "feasible-best"
ARMS = ("S/D07", "S/D08")
SHARED_ARMS = ("S/D01", "R10/D01")
"""Two arms of one parent in one configuration: one warm-up, one derivative policy, so one shared bank."""

WARMUP_SERVED = 4.0
WARMUP_TWO = 4.25
WARMUP_ONE = 4.5
WARMUP_SHARED = 4.75
WARMUP_SEPARATE = 5.0
"""A warm-up of its own per test, so each starts from a store holding no evidence of its protocol."""

SCENARIOS = (RobustnessScenario("nominal", "nominal", (0.0, 0.0)),)


def _evaluation(f: ManualFixture) -> tuple[ManualEvaluationConfig, Path]:
    """The fixture's own evaluation configuration (written identically on every call)."""
    evaluations = f.root / "configs" / "evaluations"
    evaluations.mkdir(parents=True, exist_ok=True)
    development = evaluations / DEVELOPMENT_SOURCE.name
    shutil.copyfile(DEVELOPMENT_SOURCE, development)
    scenario = f.scenario_file
    limits = ", ".join(f"{v}" for v in load_manual_scenario(scenario).limits.velocity)
    target = evaluations / "task_1a_manual_dev_fixture.toml"
    target.write_text(
        f'name = "task-1a-manual-dev-fixture"\n'
        f'development = "{development.as_posix()}"\n'
        f'scenario = "{scenario.as_posix()}"\n'
        f"horizon_s = {HORIZON_S}\n\n"
        f"[trigger]\nhold_s = {HOLD_S}\nduration_s = {PULSE_S}\nmagnitude_n = 3.0\n\n"
        f"[simulation]\nvelocity_abort = [{limits}]\n",
        encoding="utf-8",
    )
    return load_manual_evaluation_config(target), target


def _entries(f: ManualFixture, arms: tuple[str, ...] = ARMS) -> tuple[StudyModel, ...]:
    wanted = {(CONFIGURATION, arm) for arm in arms}
    return tuple(e for e in f.manifest.entries if (e.configuration, e.arm.label) in wanted)


def _empty_store(path: Path) -> StorageRoot:
    """A storage root holding nothing, so a sweep into it must produce every piece of its own evidence."""
    path.mkdir(parents=True)
    return StorageRoot(path, repositories=(REPO_ROOT,))


def _runner(f: ManualFixture, store: StorageRoot | None = None) -> ManualEvaluationRunner:
    config, file = _evaluation(f)
    return ManualEvaluationRunner(
        store=f.store if store is None else store,
        inputs=f.inputs,
        config=config,
        evaluation_file=file,
        scenarios=SCENARIOS,
        trackers={"pd_v2": TRACKER, "computed_torque": TRACKER},
        root=f.root,
        execution=f.execution,
        provenance=f.provenance,
    )


class _Observed:
    """A stand-in worker that does the real work and records how many ran at once."""

    def __init__(self, runner: ManualEvaluationRunner, *, dwell_s: float = 0.02) -> None:
        self.runner = runner
        self.dwell_s = dwell_s
        self.lock = threading.Lock()
        self.live = 0
        self.peak = 0
        self.labels: list[str] = []

    def __call__(self, entry: StudyModel, *, warmup_s: float, env: dict[str, str]) -> None:
        del env
        with self.lock:
            self.live += 1
            self.peak = max(self.peak, self.live)
            self.labels.append(entry.label)
        try:
            self.runner.evaluate(entry, warmup_s=warmup_s)
            time.sleep(self.dwell_s)
        finally:
            with self.lock:
                self.live -= 1


# --- what a parallel sweep produces ------------------------------------------------------------


def test_a_parallel_sweep_produces_evidence_the_serial_path_accepts(manual_fixture: ManualFixture) -> None:
    """Evidence produced by workers is what the serial path then serves, unchanged.

    The store keeps completed evidence immutable and serves it, so this cannot
    be a literal "run it twice and diff": a second run of one protocol
    simulates nothing, and two protocols would compare different runs. What it
    establishes is that a parallel sweep's output is accepted by the serial
    path as its own -- identity, counts and pairs -- after the digest
    re-verification that loading progress and manifests performs.
    """
    f = manual_fixture
    entries = _entries(f)
    runner = _runner(f)
    produced = evaluate_in_parallel(
        runner, entries, warmup_s=WARMUP_SERVED, workers=2, env=f.env, spawn=_Observed(_runner(f))
    )
    assert [e.label for e in produced] == [entry.label for entry in entries]
    assert all(e.conditions.execution_identity == f.execution.identity for e in produced)
    serial = [runner.evaluate(entry, warmup_s=WARMUP_SERVED) for entry in entries]
    assert [e.identity for e in serial] == [e.identity for e in produced]
    assert [e.n_completed for e in serial] == [e.n_completed for e in produced]
    assert [[p.scenario_id for p in e.pairs] for e in serial] == [[p.scenario_id for p in e.pairs] for e in produced]


# --- the bound ---------------------------------------------------------------------------------


def test_two_workers_actually_overlap(manual_fixture: ManualFixture) -> None:
    """Bounded is not serial: with two allowed, two run at once."""
    f = manual_fixture
    observed = _Observed(_runner(f))
    evaluate_in_parallel(_runner(f), _entries(f), warmup_s=WARMUP_TWO, workers=2, env=f.env, spawn=observed)
    assert observed.peak == 2
    assert sorted(observed.labels) == sorted(e.label for e in _entries(f))


def test_one_worker_never_overlaps(manual_fixture: ManualFixture) -> None:
    """The bound is honoured in the other direction too, which is what makes it a bound."""
    f = manual_fixture
    observed = _Observed(_runner(f))
    evaluate_in_parallel(_runner(f), _entries(f), warmup_s=WARMUP_ONE, workers=1, env=f.env, spawn=observed)
    assert observed.peak == 1


def test_a_non_positive_worker_count_is_refused(manual_fixture: ManualFixture) -> None:
    """Zero workers would silently do nothing; a negative count is meaningless."""
    f = manual_fixture
    with pytest.raises(ValueError, match="workers"):
        evaluate_in_parallel(
            _runner(f), _entries(f), warmup_s=WARMUP_ONE, workers=0, env=f.env, spawn=_Observed(_runner(f))
        )


# --- the shared replay banks (the one thing two workers could collide over) ---------------------


class _BankWatcher:
    """A stand-in worker that records whether its model's replay bank was already stored when it began."""

    def __init__(self, runner: ManualEvaluationRunner) -> None:
        self.runner = runner
        self.lock = threading.Lock()
        self.seen: list[bool] = []
        self.banks: list[str] = []

    def __call__(self, entry: StudyModel, *, warmup_s: float, env: dict[str, str]) -> None:
        del env
        assignment = entry.arm.assignment
        assert assignment is not None, "this worker is only given models that have a parent"
        uri = replay_bank_uri(self.runner.conditions(warmup_s, self.runner.replay_cutoffs(entry)), assignment)
        directory = self.runner.store.path(f"{uri}/{PROGRESS_FILE}", mode="write").parent
        stored = directory.exists() and any(directory.glob("manifest-*.json"))
        with self.lock:
            self.seen.append(stored)
            self.banks.append(uri)
        self.runner.evaluate(entry, warmup_s=warmup_s)


def test_shared_replay_banks_are_built_before_any_worker_starts(manual_fixture: ManualFixture) -> None:
    """Models of one parent share one bank, and the parent completes it before any worker can race for it."""
    f = manual_fixture
    entries = _entries(f, SHARED_ARMS)
    watcher = _BankWatcher(_runner(f))
    evaluate_in_parallel(_runner(f), entries, warmup_s=WARMUP_SHARED, workers=2, env=f.env, spawn=watcher)
    assert len(watcher.banks) == len(entries)
    assert len(set(watcher.banks)) == 1, "one parent, one warm-up and one policy is one bank"
    assert watcher.seen == [True] * len(entries)


def test_serial_and_parallel_sweeps_agree_in_separate_empty_stores(
    manual_fixture: ManualFixture, tmp_path: Path
) -> None:
    """The same models, evaluated serially and in parallel into stores that start empty, agree completely.

    Separate empty stores are what makes this a comparison rather than a reuse:
    neither sweep can serve anything the other produced, so each simulates every
    run itself and the two sets of evidence are independent witnesses.
    """
    f = manual_fixture
    entries = _entries(f, SHARED_ARMS)
    serial_runner = _runner(f, _empty_store(tmp_path / "serial"))
    serial = [serial_runner.evaluate(entry, warmup_s=WARMUP_SEPARATE) for entry in entries]
    parallel_store = _empty_store(tmp_path / "parallel")
    produced = evaluate_in_parallel(
        _runner(f, parallel_store),
        entries,
        warmup_s=WARMUP_SEPARATE,
        workers=2,
        env=f.env,
        spawn=_Observed(_runner(f, parallel_store)),
    )
    assert [e.identity for e in produced] == [e.identity for e in serial]
    assert [e.replay_bank for e in produced] == [e.replay_bank for e in serial]
    assert [e.status for e in produced] == [e.status for e in serial]
    assert [(e.n_pairs, e.n_completed, e.n_infeasible) for e in produced] == [
        (e.n_pairs, e.n_completed, e.n_infeasible) for e in serial
    ]
    for made, expected in zip(produced, serial, strict=True):
        assert [(p.scenario_id, p.tracker, p.arm, p.status) for p in made.pairs] == [
            (p.scenario_id, p.tracker, p.arm, p.status) for p in expected.pairs
        ]
