# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: the authorized bound reaches the worker processes, not only the process that checked it.

The preflight resolves what this invocation will run and refuses a shape that
was not sanctioned. It does that in the parent, against the parent's runner --
and a worker is a fresh interpreter that rebuilds its scope from the
configuration it is given. A benchmark authorized for 60 nominal runs therefore
passed its own preflight and executed 416 runs before it was stopped, because
the ``--scenarios`` restriction never reached the workers.

Nothing this process patches reaches a worker either, so these tests narrow
through a real development file and spawn real interpreters. That is the only
way the property can be observed at all: the parent's own instrumentation
records what the parent simulated, which under workers is nothing.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments import manual_timing
from arm_rc_ctrl.experiments.manual_evaluation import REPORTS_PREFIX, load_manual_model_evidence
from arm_rc_ctrl.experiments.manual_timing import main
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from pathlib import Path

    from arm_rc_ctrl.experiments.manual_evaluation import ManualModelEvidence
    from arm_rc_ctrl.experiments.manual_fixture import ManualFixture

REPO_ROOT = repository_root()
HOLD_S, PULSE_S, HORIZON_S = 0.05, 0.02, 1.0
MODELS = 2
"""Two models, so the fan-out is real: one worker each under a bound of two."""

SMALL_DEVELOPMENT = """\
# A deliberately tiny development envelope, so a real worker process evaluates five scenarios
# rather than the locked sixty-five and the restriction to one is visible in what it stored.
name = "task-1a-manual-timing-worker-fixture"
scenario = "../tasks/task_1a.toml"
seeds = [20261302]

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
LOCKED_SCENARIOS = 5
"""One nominal, one small posture, one large posture, one force, one combined."""
TRACKERS = 2


def _configs(f: ManualFixture) -> Path:
    """A small development envelope and an evaluation configuration bound to it, inside the fixture root."""
    evaluations = f.root / "configs" / "evaluations"
    evaluations.mkdir(parents=True, exist_ok=True)
    tasks = f.root / "configs" / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    if not (tasks / "task_1a.toml").exists():
        shutil.copyfile(REPO_ROOT / "configs" / "tasks" / "task_1a.toml", tasks / "task_1a.toml")
    development = evaluations / "timing_worker_development.toml"
    development.write_text(SMALL_DEVELOPMENT, encoding="utf-8")
    scenario = f.scenario_file
    limits = ", ".join(f"{v}" for v in load_manual_scenario(scenario).limits.velocity)
    evaluation = evaluations / "task_1a_manual_timing_worker.toml"
    evaluation.write_text(
        f'name = "task-1a-manual-timing-worker"\n'
        f'development = "{development.as_posix()}"\n'
        f'scenario = "{scenario.as_posix()}"\n'
        f"horizon_s = {HORIZON_S}\n\n"
        f"[trigger]\nhold_s = {HOLD_S}\nduration_s = {PULSE_S}\nmagnitude_n = 3.0\n\n"
        f"[simulation]\nvelocity_abort = [{limits}]\n",
        encoding="utf-8",
    )
    return evaluation


def _stored(f: ManualFixture) -> list[ManualModelEvidence]:
    """Every model manifest the store holds, read back rather than taken from this process."""
    root = f.store.path(f"{REPORTS_PREFIX}/model", mode="write")
    return [load_manual_model_evidence(path) for path in sorted(root.glob("*/manifest-*.json"))]


def _argv(f: ManualFixture, tmp_path: Path, *, workers: int) -> list[str]:
    return [
        "smoke",
        "--study",
        str(f.manifest_file),
        "--evaluation",
        str(_configs(f)),
        "--evidence-dir",
        str(tmp_path / "evidence"),
        "--output",
        str(tmp_path / "timing.json"),
        "--markdown",
        str(tmp_path / "timing.md"),
        "--subset",
        "prefix",
        "--models",
        str(MODELS),
        "--scenarios",
        "nominal",
        "--workers",
        str(workers),
        "--exploratory",
    ]


def test_workers_run_only_the_selected_scenarios(
    manual_fixture: ManualFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reproduction of the overrun: under workers, the selection must still bound what runs."""
    f = manual_fixture
    for name, value in f.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(manual_timing, "repository_root", lambda: f.root)
    assert main(_argv(f, tmp_path, workers=2)) == 0
    capsys.readouterr()
    evidences = _stored(f)
    assert len(evidences) == MODELS, "each model left its own evidence"
    for evidence in evidences:
        assert evidence.conditions.scenario_ids == ("nominal",)
        assert evidence.n_pairs == TRACKERS, (
            f"{evidence.label} ran {evidence.n_pairs} pairs; the selection allows one scenario "
            f"under both trackers, and the envelope holds {LOCKED_SCENARIOS}"
        )
