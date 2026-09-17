# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: the command that measures what the manual study will cost.

A thin entry point over the evaluator: bind the frozen study in the pinned
environment, evaluate a deterministic subset chosen by position, and write the
timing report and its rendering. It measures the complete path rather than a
simulation of it, so what it reports is what the full execution will do.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments import manual_evaluation, manual_timing
from arm_rc_ctrl.experiments.manual_evaluation import load_manual_pointer
from arm_rc_ctrl.experiments.manual_timing import load_timing, main, render_timing_markdown
from arm_rc_ctrl.experiments.perturbations import RobustnessScenario
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from pathlib import Path

    from arm_rc_ctrl.experiments.manual_fixture import ManualFixture

REPO_ROOT = repository_root()
DEVELOPMENT_SOURCE = REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml"
HOLD_S, PULSE_S, HORIZON_S = 0.05, 0.02, 1.0

NARROWED = (
    RobustnessScenario("nominal", "nominal", (0.0, 0.0)),
    RobustnessScenario("small-1", "posture_small", (0.02, -0.01), seed=1, draw=0, magnitude_rad=0.05),
)
"""Two cases stand in for the locked sixty-five; the command's own logic is what is under test."""


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


def _narrow(f: ManualFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the real command against the fixture study over two scenarios."""
    for name, value in f.env.items():
        monkeypatch.setenv(name, value)

    def scenarios(*_args: object, **_kwargs: object) -> tuple[RobustnessScenario, ...]:
        return NARROWED

    monkeypatch.setattr(manual_timing, "repository_root", lambda: f.root)
    monkeypatch.setattr(manual_evaluation, "evaluation_scenarios", scenarios)


def _argv(f: ManualFixture, tmp_path: Path, *, models: int = 1) -> list[str]:
    return [
        "smoke",
        "--study",
        str(f.manifest_file),
        "--evaluation",
        str(_evaluation_file(f)),
        "--evidence-dir",
        str(tmp_path / "evidence"),
        "--output",
        str(tmp_path / "timing.json"),
        "--markdown",
        str(tmp_path / "timing.md"),
        "--models",
        str(models),
        "--exploratory",
    ]


def test_the_smoke_check_measures_the_study_and_writes_its_report(
    manual_fixture: ManualFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One model measured end to end, with a report that reloads and a rendering that matches it."""
    _narrow(manual_fixture, monkeypatch)
    assert main(_argv(manual_fixture, tmp_path)) == 0
    report = load_timing(tmp_path / "timing.json")
    assert report.experiment == "task_1a_manual_v1"
    assert len(report.entries) == 1
    assert report.runs_this_invocation == len(report.runs) > 0
    assert report.replay_banks_built == 1
    assert report.wall_seconds > 0
    projection = report.projection
    # The study's own counts are exact whatever this check measured; pairs follow the evaluation
    # actually configured, which is narrowed here and is 65 x 2 in the real protocol.
    assert (projection.models, projection.replay_banks) == (186, 60)
    assert projection.pairs_per_model == len(NARROWED) * 2
    assert projection.total_runs == (projection.models + projection.replay_banks) * projection.pairs_per_model
    assert (tmp_path / "timing.md").read_text(encoding="utf-8") == render_timing_markdown(report)
    printed = json.loads(capsys.readouterr().out)
    assert printed["models"] == 1
    assert printed["projected_total_hours"] > 0


def test_the_smoke_check_binds_the_study_and_environment_it_measured(
    manual_fixture: ManualFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The report names the study manifest and the execution identity every measured run was keyed in."""
    _narrow(manual_fixture, monkeypatch)
    assert main(_argv(manual_fixture, tmp_path)) == 0
    capsys.readouterr()
    report = load_timing(tmp_path / "timing.json")
    assert report.study_manifest_sha256 == manual_evaluation.sha256_file(manual_fixture.manifest_file)
    assert report.execution.identity == manual_fixture.execution.identity
    assert report.provenance.exploratory is True
    pointers = sorted((tmp_path / "evidence").glob("*.toml"))
    assert sorted(load_manual_pointer(path).kind for path in pointers) == ["model", "replay"]


def test_the_smoke_check_refuses_to_overwrite_a_committed_report(
    manual_fixture: ManualFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Evidence is written once; a second check writes somewhere else or the first is lost."""
    _narrow(manual_fixture, monkeypatch)
    argv = _argv(manual_fixture, tmp_path)
    assert main(argv) == 0
    capsys.readouterr()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(argv)
