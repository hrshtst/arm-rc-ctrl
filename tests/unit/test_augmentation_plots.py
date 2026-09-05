# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""TOOL-002: recovery augmentation plots use the real generated task-space paths."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from arm_rc_ctrl.data.derivatives import DerivativeConfig
from arm_rc_ctrl.data.recovery import RecoveryDatasetRecord, TaskIntervals
from arm_rc_ctrl.data.samples import SampleSet
from arm_rc_ctrl.data.synthetic import synthetic_task_samples
from arm_rc_ctrl.experiments import augmentation_plots
from arm_rc_ctrl.experiments.augmentation_plots import (
    AUGMENTATION_PLOT_FILES,
    task_space_trajectories,
    write_augmentation_task_space_plots,
)
from arm_rc_ctrl.rc.augment import AugmentationConfig, AugmentationResult, generate_augmentation
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import endpoint_positions, load_scenario

REPO_ROOT = repository_root()
SCENARIO = load_scenario(REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml")
SAMPLES = synthetic_task_samples()
TASK = TaskIntervals(move=(0.0, 0.8), dwell=(0.8, 1.0))
DERIVATIVES = DerivativeConfig(method="central")
CONFIG = AugmentationConfig(
    n_synthetic=16,
    sigma_rad=0.05,
    phi=0.99,
    gamma=1.0,
    seed_bank=1,
    attempt_budget=64,
)


def _result() -> AugmentationResult:
    return generate_augmentation(SAMPLES.t, SAMPLES.q, TASK, SCENARIO, CONFIG, derivatives=DERIVATIVES)


def test_task_space_trajectories_are_forward_kinematics_of_generated_positions() -> None:
    """The visualized paths are exactly the original and generated joint paths mapped by FK."""
    result = _result()
    reference, augmented = task_space_trajectories(result, SCENARIO, family="contractive", count=5)

    assert np.array_equal(reference, endpoint_positions(SCENARIO, result.original.q))
    assert augmented.shape == (5, SAMPLES.n_samples, 2)
    for index in range(5):
        assert np.array_equal(augmented[index], endpoint_positions(SCENARIO, result.episodes[index].contractive.q))


def test_task_space_trajectory_selection_rejects_invalid_family_or_count() -> None:
    """A plot cannot silently substitute an arm or display more episodes than generated."""
    result = _result()
    with pytest.raises(ValueError, match="family"):
        task_space_trajectories(result, SCENARIO, family="unknown", count=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="count"):
        task_space_trajectories(result, SCENARIO, family="contractive", count=0)
    with pytest.raises(ValueError, match="generated 16"):
        task_space_trajectories(result, SCENARIO, family="contractive", count=17)


def test_complete_task_space_plot_set_is_deterministic_and_no_clobber(tmp_path: Path) -> None:
    """All family/scale/contraction figures are written once and regenerate byte-identically."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    written = write_augmentation_task_space_plots(
        SAMPLES,
        TASK,
        SCENARIO,
        DERIVATIVES,
        first,
        displayed_episodes=4,
    )
    regenerated = write_augmentation_task_space_plots(
        SAMPLES,
        TASK,
        SCENARIO,
        DERIVATIVES,
        second,
        displayed_episodes=4,
    )

    assert tuple(path.name for path in written) == AUGMENTATION_PLOT_FILES
    assert all(path.stat().st_size > 0 for path in written)
    assert [path.read_bytes() for path in written] == [path.read_bytes() for path in regenerated]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_augmentation_task_space_plots(
            SAMPLES,
            TASK,
            SCENARIO,
            DERIVATIVES,
            first,
            displayed_episodes=4,
        )


def test_task_space_plot_does_not_replace_a_racing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file created after the early check survives the atomic no-clobber install."""
    output = tmp_path / "plots"
    real_link = os.link

    def racing_link(source: str | Path, target: str | Path) -> None:
        Path(target).write_bytes(b"racer")
        real_link(source, target)

    monkeypatch.setattr(augmentation_plots.os, "link", racing_link)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_augmentation_task_space_plots(
            SAMPLES,
            TASK,
            SCENARIO,
            DERIVATIVES,
            output,
            displayed_episodes=4,
        )

    target = output / AUGMENTATION_PLOT_FILES[0]
    assert target.read_bytes() == b"racer"
    assert not list(output.glob(".*.png"))


def test_command_uses_the_recorded_source_and_reports_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The thin-script entry point validates its record and emits a machine-readable summary."""
    checked: list[Path] = []
    artifact = SimpleNamespace(
        artifact_id="processed-test",
        payload=SimpleNamespace(sha256="a" * 64),
    )

    def check_scenario(path: Path) -> None:
        checked.append(path)

    def check_samples(_samples: object) -> None:
        return

    record = SimpleNamespace(
        artifact=artifact,
        scenario=SimpleNamespace(config_path="unused.toml"),
        crop=SimpleNamespace(task=TASK),
        preprocessing=SimpleNamespace(derivative_method="central-difference"),
        check_scenario=check_scenario,
        check_samples=check_samples,
    )
    scenario_file = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"

    def fake_load_record(_path: Path, _kind: type[object]) -> RecoveryDatasetRecord:
        return cast("RecoveryDatasetRecord", record)

    def fake_verify_payload(_store: object, _artifact: object) -> Path:
        return tmp_path / "samples.npz"

    def fake_load_samples(_path: Path) -> SampleSet:
        return SAMPLES

    def fake_open_storage() -> object:
        return object()

    monkeypatch.setattr(augmentation_plots, "load_record", fake_load_record)
    monkeypatch.setattr(augmentation_plots, "verify_payload", fake_verify_payload)
    monkeypatch.setattr(augmentation_plots, "load_samples", fake_load_samples)
    monkeypatch.setattr(augmentation_plots, "open_storage", fake_open_storage)

    output = tmp_path / "plots"
    assert (
        augmentation_plots.main(
            ["--dataset", str(tmp_path / "record.toml"), "--scenario", str(scenario_file), "--output-dir", str(output)]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["dataset"] == "processed-test"
    assert summary["displayed_episodes"] == 16
    assert checked == [scenario_file]
    assert sorted(path.name for path in output.iterdir()) == sorted(AUGMENTATION_PLOT_FILES)
