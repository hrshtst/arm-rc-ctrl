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
from typing import TYPE_CHECKING, cast

import pytest

from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments import manual_evaluation, manual_timing
from arm_rc_ctrl.experiments.manual_evaluation import ManualEvaluationRunner, load_manual_pointer
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
    """The prefix subset keeps these tests to one model; the frozen budget subset is covered separately."""
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
        "--subset",
        "prefix",
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
    # The projection is of the locked study whatever this invocation measured: execution is
    # narrowed to the nominal case, but the budget being estimated covers every locked scenario
    # under both trackers -- 65 x 2 = 130 in the real protocol, len(NARROWED) x 2 here.
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


def _budget_argv(f: ManualFixture, tmp_path: Path) -> list[str]:
    """The default invocation: no subset flag, so the frozen budget subset is what runs."""
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
        "--exploratory",
    ]


def test_the_default_subset_is_the_frozen_budget_subset(
    manual_fixture: ManualFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without a flag the command measures the frozen budget subset, not a prefix.

    The rule itself is tested against the real manifest elsewhere; what matters
    here is that the default path consults it, since that is what decides which
    models the measured budget actually runs. One model stands in for the
    twenty-four so this stays a unit test.
    """
    _narrow(manual_fixture, monkeypatch)
    consulted: list[str] = []
    only = manual_fixture.manifest.entries[0]

    def one(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        consulted.append("budget")
        return (only,)

    monkeypatch.setattr(manual_timing, "budget_entries", one)
    # One model stands in for the twenty-four, so the size it runs is stated rather than
    # having the preflight weakened to accommodate a stub.
    argv = [*_budget_argv(manual_fixture, tmp_path), "--scenarios", "nominal", "--expect-runs", "4"]
    assert main(argv) == 0
    capsys.readouterr()
    assert consulted == ["budget"], "the default path must consult the frozen subset"
    assert load_timing(tmp_path / "timing.json").entries == (only.label,)


def test_named_entries_override_the_subset(
    manual_fixture: ManualFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A targeted check measures exactly what it names, whatever the subset rule would have chosen."""
    _narrow(manual_fixture, monkeypatch)
    wanted = manual_fixture.manifest.entries[3].label
    argv = [*_budget_argv(manual_fixture, tmp_path), "--entries", wanted]
    assert main(argv) == 0
    capsys.readouterr()
    assert load_timing(tmp_path / "timing.json").entries == (wanted,)


# --- bounded parallel execution, for the I1 benchmark -------------------------------------------


def test_the_smoke_check_runs_serially_by_default(
    manual_fixture: ManualFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One worker unless asked otherwise, and the report records the count it ran under."""
    _narrow(manual_fixture, monkeypatch)
    assert main(_argv(manual_fixture, tmp_path)) == 0
    capsys.readouterr()
    assert load_timing(tmp_path / "timing.json").workers == 1


def test_the_smoke_check_dispatches_by_warm_up_when_parallel(
    manual_fixture: ManualFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Models are grouped by their inherited warm-up before being fanned out.

    ``evaluate_in_parallel`` takes one warm-up for a whole group because the
    warm-up is part of the conditions a run is keyed by. The budget subset spans
    three of them, so passing a single value would key models to a protocol they
    do not belong to.
    """
    f = manual_fixture
    _narrow(f, monkeypatch)
    groups: list[float] = []
    real = manual_timing.evaluate_in_parallel

    def recording(runner: object, entries: object, **kwargs: object) -> object:
        groups.append(cast("float", kwargs["warmup_s"]))
        return real(runner, entries, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(manual_timing, "evaluate_in_parallel", recording)
    argv = [*_argv(manual_fixture, tmp_path), "--workers", "2"]
    assert main(argv) == 0
    capsys.readouterr()
    report = load_timing(tmp_path / "timing.json")
    assert report.workers == 2
    assert groups, "the parallel path must have been taken"
    assert groups == sorted(set(groups)), "each warm-up is dispatched once, in order"


def test_the_smoke_check_refuses_a_non_positive_worker_count(
    manual_fixture: ManualFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero workers would measure nothing and report a projection divided by nothing."""
    _narrow(manual_fixture, monkeypatch)
    argv = [*_argv(manual_fixture, tmp_path), "--workers", "0"]
    with pytest.raises(ValueError, match="workers"):
        main(argv)


# --- the authorized shape, checked before anything is fitted or simulated -----------------------


def test_only_the_selected_scenarios_are_run(
    manual_fixture: ManualFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--scenarios`` restricts what runs; the report's pairs follow the selection, not the locked set."""
    _narrow(manual_fixture, monkeypatch)
    # Its own model, so this starts from a store holding no evidence of this protocol: a served
    # invocation simulates nothing and would measure no runs at all.
    wanted = manual_fixture.manifest.entries[7].label
    argv = [*_argv(manual_fixture, tmp_path), "--scenarios", "nominal", "--entries", wanted]
    assert main(argv) == 0
    capsys.readouterr()
    report = load_timing(tmp_path / "timing.json")
    assert report.projection.pairs_per_model == len(NARROWED) * 2, (
        "the projection is of the locked study; only what ran was restricted"
    )
    assert report.runs, "this protocol was cold, so the invocation measured its own runs"
    assert {run.scenario_id for run in report.runs} == {"nominal"}


def test_the_command_refuses_before_any_simulation_when_the_shape_is_wrong(
    manual_fixture: ManualFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The preflight fires first: a wrong shape is refused without fitting or simulating anything.

    ``evaluate`` is blocked outright, so if the command reached it the failure
    would be an AssertionError rather than the preflight's ValueError. That is
    what makes this a test of ordering and not merely of the message.
    """
    f = manual_fixture
    _narrow(f, monkeypatch)

    def never(*_args: object, **_kwargs: object) -> object:
        msg = "the preflight must refuse before any model is evaluated"
        raise AssertionError(msg)

    monkeypatch.setattr(ManualEvaluationRunner, "evaluate", never)
    argv = [
        *_budget_argv(f, tmp_path),
        "--scenarios",
        "nominal",
        "--expect-runs",
        "999",
    ]
    with pytest.raises(ValueError, match="authorized measurement"):
        main(argv)
    assert not (tmp_path / "timing.json").exists(), "no report is written for a refused shape"


def test_the_authorized_shape_is_checked_without_an_explicit_expectation(
    manual_fixture: ManualFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no --expect-runs, the sanctioned shape is what a wrong subset is measured against.

    Every other command test states a size explicitly, which leaves untested the
    branch the real benchmark actually takes. A short subset makes the check
    refuse through that default path, and blocking ``evaluate`` proves it
    refused before any model was fitted or simulated.
    """
    f = manual_fixture
    _narrow(f, monkeypatch)

    def never(*_args: object, **_kwargs: object) -> object:
        msg = "the preflight must refuse before any model is evaluated"
        raise AssertionError(msg)

    def short(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        return tuple(f.manifest.entries[:4])

    monkeypatch.setattr(ManualEvaluationRunner, "evaluate", never)
    monkeypatch.setattr(manual_timing, "budget_entries", short)
    argv = [*_budget_argv(f, tmp_path), "--scenarios", "nominal"]
    with pytest.raises(ValueError, match="authorized measurement"):
        main(argv)
    assert not (tmp_path / "timing.json").exists()
