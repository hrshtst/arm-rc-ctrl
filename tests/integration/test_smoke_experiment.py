# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M0-012: the headless smoke experiment is deterministic and provenance-complete."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.experiments import smoke
from arm_rc_ctrl.provenance import DirtyWorktreeError, ProvenanceRecord
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

REPO_ROOT = repository_root()
FIXED_TIME = datetime(2026, 8, 29, 12, 0, 0, tzinfo=UTC)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def config() -> smoke.SmokeConfig:
    """The committed smoke configuration."""
    return load_config(REPO_ROOT / smoke.DEFAULT_CONFIG, smoke.SmokeConfig)


def _store(tmp_path: Path, name: str) -> StorageRoot:
    root = tmp_path / name
    root.mkdir()
    return StorageRoot(root, repositories=(REPO_ROOT,))


ARRAY_NAMES = {"t", "q", "dq", "tau", "q_ref", "esn_input", "esn_target", "esn_prediction"}


def _execute(tmp_path: Path, name: str, run_id: str) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Run the command-line entry point in a fresh interpreter against its own storage root."""
    root = tmp_path / name
    root.mkdir()
    env = {**os.environ, "ARM_RC_CTRL_STORAGE_ROOT": str(root), "OMP_NUM_THREADS": "1"}
    subprocess.run(
        [sys.executable, "-m", "arm_rc_ctrl.experiments.smoke", "--run-id", run_id, "--exploratory"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    run_dir = root / "runs" / run_id
    summary = json.loads((run_dir / smoke.SUMMARY_FILE).read_text())
    with np.load(run_dir / smoke.ARRAYS_FILE) as payload:
        arrays = {key: payload[key] for key in payload.files}
    return summary, arrays


def test_two_same_seed_executions_produce_identical_canonical_outputs(tmp_path: Path) -> None:
    """Two fresh-process executions give bitwise-equal arrays and identical metrics/digests."""
    first_summary, first_arrays = _execute(tmp_path, "a", "run-a")
    second_summary, second_arrays = _execute(tmp_path, "b", "run-b")
    assert set(first_arrays) == ARRAY_NAMES
    for name, array in first_arrays.items():
        assert np.array_equal(array, second_arrays[name]), name
    assert first_summary["metrics"] == second_summary["metrics"]
    assert first_summary["arrays"] == second_summary["arrays"]
    assert first_summary["canonical_digest"] == second_summary["canonical_digest"]
    # Only the run identity differs.
    assert first_summary["run_uri"] == "armrc://runs/run-a"
    assert second_summary["run_uri"] == "armrc://runs/run-b"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "UP-005: pinned rclib seeds reservoir weights but starts its spectral-radius power iteration from "
        "Eigen::Random() (C std::rand, never re-seeded), so a second reservoir built in the same process is "
        "scaled differently. Remove this xfail when the rclib pin includes the fix."
    ),
)
def test_in_process_repeat_is_reproducible(tmp_path: Path, config: smoke.SmokeConfig) -> None:
    """Repeating the run inside one interpreter should also reproduce the ESN prediction."""
    first = smoke.run_smoke(config, _store(tmp_path, "a"), "run-a", exploratory=True, now=FIXED_TIME)
    second = smoke.run_smoke(config, _store(tmp_path, "b"), "run-b", exploratory=True, now=FIXED_TIME)
    for name in ("t", "q", "dq", "tau", "q_ref", "esn_input", "esn_target"):
        assert np.array_equal(first.arrays[name], second.arrays[name]), name  # the simulation is deterministic
    assert np.array_equal(first.arrays["esn_prediction"], second.arrays["esn_prediction"])


def test_outputs_are_persisted_with_provenance(tmp_path: Path, config: smoke.SmokeConfig) -> None:
    """arrays.npz and summary.json land under the run URI and can be reloaded and re-validated."""
    store = _store(tmp_path, "s")
    result = smoke.run_smoke(config, store, "run-01", exploratory=True, now=FIXED_TIME)
    run_dir = store.path("armrc://runs/run-01", mode="read")
    assert result.run_dir == run_dir
    assert not (run_dir.parent / "run-01.partial").exists()

    with np.load(run_dir / smoke.ARRAYS_FILE) as payload:
        stored = {name: payload[name] for name in payload.files}
    assert set(stored) == set(result.arrays)
    array_meta = cast("dict[str, dict[str, str]]", result.summary["arrays"])
    for name, array in result.arrays.items():
        assert np.array_equal(stored[name], array)
        assert smoke.array_digest(stored[name]) == array_meta[name]["sha256"]

    summary = json.loads((run_dir / smoke.SUMMARY_FILE).read_text())
    assert summary == result.summary
    record = ProvenanceRecord.from_mapping(summary["provenance"])
    assert record == result.provenance
    assert record.seeds == {"reservoir": config.seed}
    assert record.exploratory is True
    assert cast("dict[str, object]", record.config["esn"])["n_neurons"] == config.esn.n_neurons
    assert record.platform.thread_environment["OMP_NUM_THREADS"] == "1"
    assert [s.name for s in record.submodules] == ["rclib", "skelarm", "rtctrl"]


def test_run_is_physically_and_numerically_sane(tmp_path: Path, config: smoke.SmokeConfig) -> None:
    """The PD loop approaches the target, torques are bounded, and the ESN fits the teacher signal."""
    result = smoke.run_smoke(config, _store(tmp_path, "s"), "run-01", exploratory=True, now=FIXED_TIME)
    metrics = result.summary["metrics"]
    assert isinstance(metrics, dict)
    steps = round(config.simulation.duration / config.simulation.dt)
    assert metrics["samples"] == steps + 1
    assert result.arrays["q"].shape == (steps + 1, 2)
    assert result.arrays["esn_input"].shape == (steps, 4)
    initial_error = float(np.linalg.norm(np.asarray(config.simulation.initial_q) - config.simulation.target_q))
    assert 0.0 <= metrics["final_joint_error_rad"] < initial_error
    assert 0.0 < metrics["max_abs_torque"] < 100.0
    assert 0.0 <= metrics["esn_train_rmse"] < 0.05
    assert np.array_equal(result.arrays["q_ref"], np.tile(config.simulation.target_q, (steps + 1, 1)))


def test_runs_are_immutable(tmp_path: Path, config: smoke.SmokeConfig) -> None:
    """An existing run directory is never overwritten."""
    store = _store(tmp_path, "s")
    smoke.run_smoke(config, store, "run-01", exploratory=True, now=FIXED_TIME)
    with pytest.raises(FileExistsError, match="runs are immutable"):
        smoke.run_smoke(config, store, "run-01", exploratory=True, now=FIXED_TIME)


def test_dirty_worktree_is_rejected_unless_exploratory(
    tmp_path: Path, config: smoke.SmokeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The clean-worktree policy applies before any output is written."""

    def dirty(_root: Path) -> tuple[str, bool]:
        return "0" * 40, True

    monkeypatch.setattr("arm_rc_ctrl.provenance.worktree_state", dirty)
    store = _store(tmp_path, "s")
    with pytest.raises(DirtyWorktreeError, match="project worktree is dirty"):
        smoke.run_smoke(config, store, "run-01", exploratory=False, now=FIXED_TIME)
    assert not (store.root / "runs" / "run-01").exists()
    result = smoke.run_smoke(config, store, "run-01", exploratory=True, now=FIXED_TIME)
    assert result.provenance.project_dirty is True


def test_single_thread_requirement(tmp_path: Path, config: smoke.SmokeConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without OMP_NUM_THREADS=1 the run refuses to start."""
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with pytest.raises(RuntimeError, match="OMP_NUM_THREADS=1"):
        smoke.run_smoke(config, _store(tmp_path, "s"), "run-01", exploratory=True, now=FIXED_TIME)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"simulation": {"kp": (1.0,)}}, "simulation.kp must have 2 entries"),
        ({"simulation": {"dt": -0.1}}, "must be positive"),
        ({"esn": {"washout": 100}}, r"esn.washout must be in \[0, 100\)"),
        ({"seed": -1}, "seed must be non-negative"),
    ],
)
def test_inconsistent_configurations_are_rejected(
    config: smoke.SmokeConfig, changes: dict[str, object], message: str
) -> None:
    """Cross-field checks catch shapes and ranges the type-level loader cannot."""
    updated = config
    for section, value in changes.items():
        if isinstance(value, dict):
            inner = dataclasses.replace(getattr(config, section), **value)
            updated = dataclasses.replace(updated, **{section: inner})
        else:
            updated = dataclasses.replace(updated, **{section: value})
    with pytest.raises(ValueError, match=message):
        smoke.validate_config(updated)


def test_command_line_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``python -m arm_rc_ctrl.experiments.smoke`` resolves the store from the environment and prints metrics."""
    root = tmp_path / "store"
    root.mkdir()
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(root))
    assert smoke.main(["--run-id", "cli-01", "--exploratory"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["run_uri"] == "armrc://runs/cli-01"
    assert (root / "runs" / "cli-01" / smoke.SUMMARY_FILE).is_file()
