# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-008: the command that runs the manual protocol's evaluation.

A thin entry point over the runner: verify the pinned environment, bind the
frozen study and the evaluation configuration, evaluate the selected models
against their parents' replay baselines, and leave Git pointers to everything
it produced. Re-running it over finished work repeats nothing and writes
nothing new.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments import manual_evaluation
from arm_rc_ctrl.experiments.manual_evaluation import load_manual_pointer, main
from arm_rc_ctrl.experiments.perturbations import RobustnessScenario
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from pathlib import Path

    from arm_rc_ctrl.experiments.manual_fixture import ManualFixture
    from arm_rc_ctrl.experiments.manual_study import StudyManifest, StudyModel

REPO_ROOT = repository_root()
DEVELOPMENT_SOURCE = REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml"
CONFIGURATION, ARM_LABEL = "feasible-best", "S/D06"
HOLD_S, PULSE_S, HORIZON_S = 0.05, 0.02, 1.0

NARROWED = (
    RobustnessScenario("nominal", "nominal", (0.0, 0.0)),
    RobustnessScenario("small-1", "posture_small", (0.02, -0.01), seed=1, draw=0, magnitude_rad=0.05),
)
"""Two cases stand in for the locked sixty-five: the command's own logic is what is under test."""


def _evaluation_file(f: ManualFixture) -> Path:
    """An evaluation configuration inside the fixture root, with the fixture's own task and abort."""
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
    return target


def _entry(f: ManualFixture) -> StudyModel:
    return next(e for e in f.manifest.entries if (e.configuration, e.arm.label) == (CONFIGURATION, ARM_LABEL))


def _narrow(f: ManualFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the real command against the fixture study, over one model and two scenarios."""
    for name, value in f.env.items():
        monkeypatch.setenv(name, value)

    def entries(manifest: StudyManifest, labels: tuple[str, ...] | None = None) -> tuple[StudyModel, ...]:
        del manifest, labels
        return (_entry(f),)

    def scenarios(*_args: object, **_kwargs: object) -> tuple[RobustnessScenario, ...]:
        return NARROWED

    monkeypatch.setattr(manual_evaluation, "repository_root", lambda: f.root)
    monkeypatch.setattr(manual_evaluation, "evaluation_entries", entries)
    monkeypatch.setattr(manual_evaluation, "evaluation_scenarios", scenarios)


def _argv(f: ManualFixture, evidence_dir: Path) -> list[str]:
    return [
        "run",
        "--study",
        str(f.manifest_file),
        "--evaluation",
        str(_evaluation_file(f)),
        "--evidence-dir",
        str(evidence_dir),
        "--exploratory",
    ]


# --- the command ------------------------------------------------------------------------------


def test_the_run_command_evaluates_the_study_and_leaves_pointers(
    manual_fixture: ManualFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One model evaluated against its parent's baselines, with a pointer to each manifest it produced."""
    _narrow(manual_fixture, monkeypatch)
    evidence_dir = tmp_path / "evidence"
    assert main(_argv(manual_fixture, evidence_dir)) == 0
    written = sorted(evidence_dir.glob("*.toml"))
    assert len(written) == 2
    kinds = sorted(load_manual_pointer(path).kind for path in written)
    assert kinds == ["model", "replay"]
    summary = json.loads(capsys.readouterr().out)
    assert summary["models"] == 1
    assert summary["pointers_written"] == 2
    assert summary["pairs"] == len(NARROWED) * 2


def test_running_the_command_again_repeats_nothing(
    manual_fixture: ManualFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finished work is served, not redone, and the pointers already written stay as they are."""
    _narrow(manual_fixture, monkeypatch)
    evidence_dir = tmp_path / "evidence"
    argv = _argv(manual_fixture, evidence_dir)
    assert main(argv) == 0
    capsys.readouterr()
    before = {path: path.read_bytes() for path in sorted(evidence_dir.glob("*.toml"))}
    assert main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["pointers_written"] == 0
    assert {path: path.read_bytes() for path in sorted(evidence_dir.glob("*.toml"))} == before


def test_the_command_records_the_study_and_the_environment_it_ran_under(
    manual_fixture: ManualFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The summary names the study manifest and the execution identity every run was keyed in."""
    _narrow(manual_fixture, monkeypatch)
    assert main(_argv(manual_fixture, tmp_path / "evidence")) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["study_manifest"] == manual_evaluation.sha256_file(manual_fixture.manifest_file)
    assert summary["execution_identity"] == manual_fixture.execution.identity
    assert summary["exploratory"] is True
