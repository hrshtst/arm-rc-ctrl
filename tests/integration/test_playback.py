# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Run export to skelarm StateLogs: verification, channels, metadata, atomicity, and the play wrapper (TOOL-001)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from skelarm import StateLog

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.preprocess import PreprocessResult, preprocess_demonstration
from arm_rc_ctrl.data.records import RawDemonstrationRecord, load_record
from arm_rc_ctrl.experiments import playback
from arm_rc_ctrl.experiments.disturbances import ForcePulse
from arm_rc_ctrl.experiments.playback import export_run_sklog, main_export, main_play, resolve_pointer
from arm_rc_ctrl.experiments.replay import ReplayResult, run_replay
from arm_rc_ctrl.experiments.run_record import record_run_pointer
from arm_rc_ctrl.provenance import ArtifactMismatchError
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
DEV_PD = REPO_ROOT / "configs" / "controllers" / "pd.toml"
RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
FIXED_TIME = datetime(2026, 9, 1, 9, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def replayed(tmp_path_factory: pytest.TempPathFactory) -> tuple[StorageRoot, Path, ReplayResult]:
    """One replay run with a force pulse, its pointer tracked under a records root."""
    base = tmp_path_factory.mktemp("playback")
    root = base / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    store.path(raw.artifact.payload.uri, mode="write").write_bytes(RAW_LOG.read_bytes())
    records = base / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    (records / "data" / "records" / "runs").mkdir(parents=True)
    processed: PreprocessResult = preprocess_demonstration(
        RAW_RECORD, SCENARIO, PREPROCESS, store=store, records_root=records, exploratory=True, now=FIXED_TIME
    )
    result = run_replay(
        load_scenario(SCENARIO),
        SCENARIO,
        processed.record,
        processed.samples,
        load_config(DEV_PD, TrackerConfig),
        store=store,
        exploratory=True,
        now=FIXED_TIME,
        force=ForcePulse.from_polar(0.1, 0.05, 2.0, 90.0),
    )
    record_run_pointer(records, result.pointer)
    return store, records, result


def test_exported_log_carries_state_task_and_identity(
    replayed: tuple[StorageRoot, Path, ReplayResult], tmp_path: Path
) -> None:
    """The log replays the exact measured state with the task, disturbances, and provenance identity embedded."""
    store, records, result = replayed
    run_id = result.pointer.artifact.artifact_id
    out = export_run_sklog(store, resolve_pointer(run_id, records), SCENARIO, tmp_path / "run.sklog.npz")
    log = StateLog.load(out)
    arrays = result.run.arrays.arrays
    scenario = load_scenario(SCENARIO)
    assert len(log) == result.run.arrays.n_samples
    assert np.array_equal(log.times, arrays["t"])
    assert np.array_equal(log.channel("q"), arrays["q"])
    assert np.array_equal(log.channel("dq"), arrays["dq"])
    assert np.array_equal(log.channel("tau"), arrays["tau_applied"])  # applied torque is canonical
    assert np.array_equal(log.channel("tau_applied"), arrays["tau_applied"])
    assert np.array_equal(log.channel("tau_requested"), arrays["tau_requested"])
    assert np.array_equal(log.channel("ext_force"), arrays["ext_force"])
    assert np.array_equal(log.channel("q_desired"), arrays["q_desired"])
    assert np.array_equal(log.channel("saturation"), np.asarray(arrays["saturation"], dtype=np.float64))
    task = log.extra["playback"]["task"]
    assert task["type"] == "reaching"
    assert task["target"]["pos"] == [float(v) for v in scenario.task.target]
    assert task["target"]["tolerance"] == scenario.task.tolerance
    run_meta = log.extra["run"]
    assert run_meta["id"] == run_id
    assert run_meta["method"] == "replay+pd"
    assert run_meta["tau_source"] == "tau_applied"
    assert run_meta["arrays_sha256"] == result.summary.arrays_sha256
    assert run_meta["payload_sha256"] == result.pointer.artifact.payload.sha256
    assert run_meta["project_commit"] == result.summary.provenance.project_commit
    (disturbance,) = run_meta["disturbances"]
    assert disturbance["kind"] == "endpoint_force_pulse"
    assert "source_config" not in log.extra  # playback-only: never advertised as rerunnable
    skeleton = log.build_skeleton()
    assert log.channel("q").shape[1] == scenario.dof
    assert skeleton.q.shape == (scenario.dof,)
    text = out.read_bytes()
    assert b"/home/" not in text
    assert str(tmp_path).encode() not in text


def test_repeated_exports_are_semantically_equal(
    replayed: tuple[StorageRoot, Path, ReplayResult], tmp_path: Path
) -> None:
    """Two exports of the same run carry identical channels and metadata (timestamps aside)."""
    store, records, result = replayed
    pointer = resolve_pointer(result.pointer.artifact.artifact_id, records)
    first = StateLog.load(export_run_sklog(store, pointer, SCENARIO, tmp_path / "a.sklog.npz"))
    second = StateLog.load(export_run_sklog(store, pointer, SCENARIO, tmp_path / "b.sklog.npz"))
    assert first.channel_names == second.channel_names
    for name in first.channel_names:
        assert np.array_equal(first.channel(name), second.channel(name)), name
    assert np.array_equal(first.times, second.times)
    assert first.extra == second.extra
    assert first.producer == second.producer


def test_export_refuses_bad_inputs(replayed: tuple[StorageRoot, Path, ReplayResult], tmp_path: Path) -> None:
    """Overwrites, wrong suffixes, foreign scenarios, unknown runs, and tampered payloads are refused."""
    store, records, result = replayed
    run_id = result.pointer.artifact.artifact_id
    pointer = resolve_pointer(run_id, records)
    out = export_run_sklog(store, pointer, SCENARIO, tmp_path / "run.sklog.npz")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_run_sklog(store, pointer, SCENARIO, out)
    with pytest.raises(ValueError, match="must end with"):
        export_run_sklog(store, pointer, SCENARIO, tmp_path / "run.npz")
    with pytest.raises(ValueError, match="was recorded under scenario"):
        export_run_sklog(store, pointer, REPO_ROOT / "configs" / "tasks" / "task_1a.toml", tmp_path / "x.sklog.npz")
    with pytest.raises(FileNotFoundError, match="no pointer record"):
        resolve_pointer("run-20260901-000000000000", records)
    with pytest.raises(ValueError, match="not a run ID"):
        resolve_pointer("model-20260901-000000000000", records)
    original_pointer = (records / "data" / "records" / "runs" / f"{run_id}.toml").read_text()
    misnamed = tmp_path / "run-20260901-aaaaaaaaaaaa.toml"
    misnamed.write_text(original_pointer)  # a valid record under the wrong file name
    with pytest.raises(ValueError, match="names"):
        export_run_sklog(store, misnamed, SCENARIO, tmp_path / "y.sklog.npz")
    tampered = tmp_path / f"{run_id}.toml"
    tampered.write_text(original_pointer.replace(run_id, "run-20260901-aaaaaaaaaaaa"))
    with pytest.raises(ValueError, match="digest prefix"):  # the record loader rejects an inconsistent ID first
        export_run_sklog(store, tampered, SCENARIO, tmp_path / "y2.sklog.npz")
    payload = store.path(result.pointer.artifact.payload.uri, mode="read")
    original = payload.read_bytes()
    payload.write_bytes(original + b" ")
    try:
        with pytest.raises(ArtifactMismatchError, match=r"size|digest|sha"):
            export_run_sklog(store, pointer, SCENARIO, tmp_path / "z.sklog.npz")
    finally:
        payload.write_bytes(original)
    assert not sorted(tmp_path.glob("*.tmp*"))  # atomic staging leaves nothing behind


def test_export_command_prints_the_player_hint(
    replayed: tuple[StorageRoot, Path, ReplayResult],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The export script resolves the run by ID against the records root and writes the log."""
    store, records, result = replayed
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(store.root))
    out = tmp_path / "cli.sklog.npz"
    argv = [
        "--run", result.pointer.artifact.artifact_id, "--scenario", str(SCENARIO),
        "--records-root", str(records), "--out", str(out),
    ]  # fmt: skip
    assert main_export(argv) == 0
    assert json.loads(capsys.readouterr().out)["out"] == "cli.sklog.npz"
    assert out.is_file()


def test_play_command_exports_to_a_temporary_log_and_invokes_the_pinned_player(
    replayed: tuple[StorageRoot, Path, ReplayResult], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The wrapper forwards the options to the pinned player, propagates its status, and cleans up."""
    store, records, result = replayed
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(store.root))
    calls: list[list[str]] = []

    class _Done:
        returncode = 3

    def fake_run(command: list[str]) -> _Done:
        calls.append(command)
        assert Path(command[2]).is_file()  # the temporary log exists while the player runs
        return _Done()

    monkeypatch.setattr(playback, "_run_player", fake_run)
    argv = [
        "--run", result.pointer.artifact.artifact_id, "--scenario", str(SCENARIO), "--records-root", str(records),
        "--speed", "0.5", "--show-com", "--export", str(tmp_path / "clip.gif"), "--fps", "12",
    ]  # fmt: skip
    assert main_play(argv) == 3  # the player's status propagates
    (command,) = calls
    assert command[1] == str(playback.PLAYER)
    assert command[2].endswith(".sklog.npz")
    assert not Path(command[2]).exists()  # the temporary export is cleaned up
    assert command[3:] == [
        "--speed",
        "0.5",
        "--show-com",
        "--export",
        str((tmp_path / "clip.gif").resolve()),
        "--fps",
        "12",
    ]
