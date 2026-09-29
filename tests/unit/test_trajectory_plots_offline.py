# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""CI-001: the curated trajectory-plot writer and its CLI without the external store.

The committed confirmatory suite and its Git-tracked run pointers are real; only
the store read (:func:`load_run`) is replaced by synthetic joint trajectories.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from arm_rc_ctrl.experiments import trajectory_plots
from arm_rc_ctrl.experiments.perturbations import CLASS_ORDER
from arm_rc_ctrl.experiments.robustness import load_suite
from arm_rc_ctrl.experiments.trajectory_plots import (
    ARM_PAIRS,
    main,
    select_representatives,
    write_task_1a_trajectory_plots,
)
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from arm_rc_ctrl.experiments.robustness import RobustnessSuite
    from arm_rc_ctrl.experiments.run_record import LoadedRun, RunPointerRecord
    from arm_rc_ctrl.storage import StorageRoot

ROOT = repository_root()
SUITE_FILE = ROOT / "docs" / "experiments" / "task_1a" / "robustness_confirmatory_v2_recipe_v4.json"
N_SAMPLES = 6


def _store() -> StorageRoot:
    return cast("StorageRoot", SimpleNamespace())


def _fake_run(pointer: RunPointerRecord, *, shift: float = 0.0) -> LoadedRun:
    t = np.linspace(0.0, 5.0, N_SAMPLES, dtype=np.float64) + shift
    q = np.column_stack((0.1 * t, -0.05 * t))
    arrays = {"t": t, "q": q, "q_desired": q + 0.01}
    return cast("LoadedRun", SimpleNamespace(pointer=pointer, arrays=SimpleNamespace(arrays=arrays)))


def _offline(monkeypatch: pytest.MonkeyPatch, *, loaded: list[str] | None = None) -> None:
    def load_run(_store: StorageRoot, pointer: RunPointerRecord) -> LoadedRun:
        if loaded is not None:
            loaded.append(pointer.artifact.artifact_id)
        return _fake_run(pointer)

    monkeypatch.setattr(trajectory_plots, "load_run", load_run)


@pytest.fixture(scope="module")
def suite() -> RobustnessSuite:
    """The committed confirmatory suite (pointers are Git-tracked, payloads are not needed)."""
    return load_suite(SUITE_FILE)


def test_writer_plots_every_class_and_tracker_from_the_verified_pointers(
    suite: RobustnessSuite, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ten figures, one per class and tracker, bound to the selected scenario's pointer-verified runs."""
    loaded: list[str] = []
    _offline(monkeypatch, loaded=loaded)
    manifest = write_task_1a_trajectory_plots(suite, tmp_path, store=_store(), records_root=ROOT)
    classes = cast("dict[str, dict[str, object]]", manifest["classes"])
    assert manifest["selection"] == "closest rc+pd_v2 joint RMSE to class median"
    assert list(classes) == list(CLASS_ORDER)
    selected = select_representatives(suite)
    expected_runs: list[str] = []
    for kind in CLASS_ORDER:
        entry = classes[kind]
        scenario_id = selected[kind].scenario_id
        assert entry["scenario_id"] == scenario_id
        assert entry["plots"] == {tracker: f"trajectories_{kind}_{tracker}.png" for tracker in ARM_PAIRS}
        runs = cast("dict[str, str]", entry["runs"])
        for rc_arm, replay_arm in ARM_PAIRS.values():
            for arm in (rc_arm, replay_arm):
                (match,) = (r for r in suite.runs if r.scenario_id == scenario_id and r.arm == arm)
                assert runs[arm] == match.run_id
            expected_runs += [runs[rc_arm], runs[replay_arm]]
    assert loaded == expected_runs
    pngs = sorted(p.name for p in tmp_path.iterdir())
    assert pngs == sorted(f"trajectories_{kind}_{tracker}.png" for kind in CLASS_ORDER for tracker in ARM_PAIRS)
    assert all((tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for name in pngs)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_task_1a_trajectory_plots(suite, tmp_path, store=_store(), records_root=ROOT)
    replaced = write_task_1a_trajectory_plots(suite, tmp_path, store=_store(), records_root=ROOT, force=True)
    assert replaced == manifest


def test_writer_refuses_rc_and_replay_runs_on_different_clocks(
    suite: RobustnessSuite, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Paired curves are drawn only on one shared clock."""

    def load_run(_store: StorageRoot, pointer: RunPointerRecord) -> LoadedRun:
        rc = pointer.method.startswith("rc")
        return _fake_run(pointer, shift=0.01 if rc else 0.0)

    monkeypatch.setattr(trajectory_plots, "load_run", load_run)
    with pytest.raises(ValueError, match="nominal: RC and replay clocks differ for pd"):
        write_task_1a_trajectory_plots(suite, tmp_path, store=_store(), records_root=ROOT)
    assert not list(tmp_path.iterdir())


def test_writer_refuses_a_pointer_that_names_another_run(
    suite: RobustnessSuite, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pointer file whose artifact is not the suite's run ID is rejected before any store read."""
    _offline(monkeypatch)
    nominal = select_representatives(suite)["nominal"]
    (other,) = (r for r in suite.runs if r.scenario_id == nominal.scenario_id and r.arm == "replay+pd_v2")
    assert nominal.pointer is not None
    assert other.pointer is not None
    records = tmp_path / "records"
    target = records / nominal.pointer
    target.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / other.pointer, target)
    with pytest.raises(ValueError, match=f"points to {other.run_id}, expected {nominal.run_id}"):
        write_task_1a_trajectory_plots(suite, tmp_path / "plots", store=_store(), records_root=records)


def test_writer_refuses_runs_without_a_tracked_pointer_or_a_unique_pair(
    suite: RobustnessSuite, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A representative run must be tracked, and each arm must have exactly one run for the scenario."""
    _offline(monkeypatch)
    nominal = select_representatives(suite)["nominal"]
    untracked = dataclasses.replace(
        suite,
        runs=tuple(dataclasses.replace(r, pointer=None) if r.run_id == nominal.run_id else r for r in suite.runs),
    )
    with pytest.raises(ValueError, match=f"run {nominal.run_id} has no Git-tracked pointer"):
        write_task_1a_trajectory_plots(untracked, tmp_path, store=_store(), records_root=ROOT)
    fake_suite = cast(
        "RobustnessSuite",
        SimpleNamespace(runs=tuple(r for r in suite.runs if not (r.arm == "replay+pd_v2" and r.kind == "nominal"))),
    )
    with pytest.raises(ValueError, match=r"expected one 'replay\+pd_v2' run for 'nominal', found 0"):
        write_task_1a_trajectory_plots(fake_suite, tmp_path, store=_store(), records_root=ROOT)


def test_selection_requires_a_primary_arm_run_in_every_class(suite: RobustnessSuite) -> None:
    """A class without any primary-arm run cannot yield a representative."""
    with pytest.raises(ValueError, match="suite has no 'missing' run with joint RMSE for class 'nominal'"):
        select_representatives(suite, primary_arm="missing")


def test_command_line_writes_the_plots_and_prints_the_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI opens the store, writes the ten figures, and prints the manifest as JSON."""
    _offline(monkeypatch)
    opened: list[bool] = []

    def open_storage() -> StorageRoot:
        opened.append(True)
        return _store()

    monkeypatch.setattr(trajectory_plots, "open_storage", open_storage)
    assert main(["--suite", str(SUITE_FILE), "--output-dir", str(tmp_path), "--records-root", str(ROOT)]) == 0
    assert opened == [True]
    printed = cast("dict[str, object]", json.loads(capsys.readouterr().out))
    classes = cast("dict[str, object]", printed["classes"])
    assert sorted(classes) == sorted(CLASS_ORDER)
    assert len(list(tmp_path.glob("trajectories_*.png"))) == len(CLASS_ORDER) * len(ARM_PAIRS)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(["--suite", str(SUITE_FILE), "--output-dir", str(tmp_path)])
    assert main(["--suite", str(SUITE_FILE), "--output-dir", str(tmp_path), "--force"]) == 0
