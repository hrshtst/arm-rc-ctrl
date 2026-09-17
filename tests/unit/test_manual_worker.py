# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-008: the worker subcommand and the command's worker count (clarification I1).

A worker is a fresh interpreter, so nothing this process patches reaches it:
its scope comes from the configuration it is given, which is how the protocol
defines scope anyway. These tests therefore narrow through a real development
file rather than through the test harness, and check the property that makes a
worker's runs admissible at all -- that it ran in the parent's environment.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments.manual_evaluation import (
    REPORTS_PREFIX,
    load_manual_model_evidence,
    main,
    spawn_worker,
)
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from pathlib import Path

    from arm_rc_ctrl.experiments.manual_evaluation import ManualModelEvidence
    from arm_rc_ctrl.experiments.manual_fixture import ManualFixture
    from arm_rc_ctrl.experiments.manual_study import StudyModel

REPO_ROOT = repository_root()
HOLD_S, PULSE_S, HORIZON_S = 0.05, 0.02, 1.0
CONFIGURATION = "feasible-best"
ARMS = ("S/D09", "S/D10")

WARMUP_WORKER = 5.0
WARMUP_COMMAND = 5.25
"""A warm-up of its own per test, so each starts from a store holding no evidence of its protocol."""

SMALL_DEVELOPMENT = """\
# A deliberately tiny development envelope: one seed, one draw, one direction, so a real worker
# process evaluates five scenarios rather than the locked sixty-five.
name = "task-1a-manual-worker-fixture"
scenario = "../tasks/task_1a.toml"
seeds = [20261301]

[posture]
small_magnitude_rad = 0.05
large_magnitude_rad = 0.1
draws_per_seed = 1

[force]
magnitude_n = 3.0
start_s = 1.0
duration_s = 0.02
directions_deg = [0.0]
"""
EXPECTED_SCENARIOS = 5
"""One nominal, one small posture, one large posture, one force, one combined."""


def _configs(f: ManualFixture) -> tuple[Path, Path]:
    """A small development envelope and an evaluation configuration bound to it, inside the fixture root."""
    evaluations = f.root / "configs" / "evaluations"
    evaluations.mkdir(parents=True, exist_ok=True)
    tasks = f.root / "configs" / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    if not (tasks / "task_1a.toml").exists():
        shutil.copyfile(REPO_ROOT / "configs" / "tasks" / "task_1a.toml", tasks / "task_1a.toml")
    development = evaluations / "worker_development.toml"
    development.write_text(SMALL_DEVELOPMENT, encoding="utf-8")
    scenario = f.scenario_file
    limits = ", ".join(f"{v}" for v in load_manual_scenario(scenario).limits.velocity)
    evaluation = evaluations / "task_1a_manual_worker.toml"
    evaluation.write_text(
        f'name = "task-1a-manual-worker"\n'
        f'development = "{development.as_posix()}"\n'
        f'scenario = "{scenario.as_posix()}"\n'
        f"horizon_s = {HORIZON_S}\n\n"
        f"[trigger]\nhold_s = {HOLD_S}\nduration_s = {PULSE_S}\nmagnitude_n = 3.0\n\n"
        f"[simulation]\nvelocity_abort = [{limits}]\n",
        encoding="utf-8",
    )
    return evaluation, development


def _entry(f: ManualFixture, arm: str) -> StudyModel:
    return next(e for e in f.manifest.entries if (e.configuration, e.arm.label) == (CONFIGURATION, arm))


# --- the worker --------------------------------------------------------------------------------


def test_a_worker_process_evaluates_one_model_in_the_parents_environment(manual_fixture: ManualFixture) -> None:
    """A separate interpreter produces evidence keyed in the environment the parent is in.

    A worker's runs belong to this study only when its execution identity
    equals the parent's, so that equality is what is asserted, not merely that
    the process exited zero.
    """
    f = manual_fixture
    evaluation, _ = _configs(f)
    entry = _entry(f, ARMS[0])
    spawn_worker(
        entry,
        warmup_s=WARMUP_WORKER,
        env=f.env,
        study_file=f.manifest_file,
        evaluation_file=evaluation,
        root=f.root,
        exploratory=True,
    )
    evidence = _stored_evidence(f, entry.label)
    assert evidence is not None
    assert evidence.conditions.execution_identity == f.execution.identity
    assert evidence.conditions.warmup_s == WARMUP_WORKER
    assert evidence.n_pairs == EXPECTED_SCENARIOS * 2


def _stored_evidence(f: ManualFixture, label: str) -> ManualModelEvidence | None:
    """The stored evidence of one model, found by what it says it is rather than by recomputing its key."""
    root = f.store.path(f"{REPORTS_PREFIX}/model", mode="write")
    for manifest in sorted(root.glob("*/manifest-*.json")):
        evidence = load_manual_model_evidence(manifest)
        if evidence.label == label:
            return evidence
    return None


# --- the command's worker count ------------------------------------------------------------------


def test_the_run_command_evaluates_through_workers(
    manual_fixture: ManualFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--workers`` fans the models out across worker processes and still leaves every pointer."""
    f = manual_fixture
    evaluation, _ = _configs(f)
    for name, value in f.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("arm_rc_ctrl.experiments.manual_evaluation.repository_root", lambda: f.root)
    evidence_dir = tmp_path / "evidence"
    argv = [
        "run",
        "--study",
        str(f.manifest_file),
        "--evaluation",
        str(evaluation),
        "--evidence-dir",
        str(evidence_dir),
        "--entries",
        *[f"{CONFIGURATION}/{arm}" for arm in ARMS],
        "--workers",
        "2",
        "--exploratory",
    ]
    assert main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["models"] == len(ARMS)
    assert summary["workers"] == 2
    assert summary["pairs"] == len(ARMS) * EXPECTED_SCENARIOS * 2
    # One pointer per model, plus one per replay bank the models were paired against.
    assert len(sorted(evidence_dir.glob("model__*.toml"))) == len(ARMS)
    assert sorted(evidence_dir.glob("replay__*.toml"))


def test_a_worker_count_below_one_is_refused(
    manual_fixture: ManualFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero workers would silently evaluate nothing."""
    f = manual_fixture
    evaluation, _ = _configs(f)
    for name, value in f.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("arm_rc_ctrl.experiments.manual_evaluation.repository_root", lambda: f.root)
    argv = [
        "run",
        "--study",
        str(f.manifest_file),
        "--evaluation",
        str(evaluation),
        "--evidence-dir",
        str(tmp_path / "evidence"),
        "--workers",
        "0",
        "--exploratory",
    ]
    with pytest.raises(ValueError, match="workers"):
        main(argv)
