# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-003 (I7): the recorder launcher builds the pinned recorder exactly from the manual protocol's configuration."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from arm_rc_ctrl.data import recording
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.data.recording import (
    RecorderSession,
    RecordingError,
    create_recorder_window,
    load_recording_config,
    main,
    resolve_recorder_session,
    write_session_settings,
)
from arm_rc_ctrl.dependencies import BuildIdentityError, SubmoduleRevision, submodule_revisions, verify_builds
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
SCENARIO = REPO_ROOT / "configs" / "tasks" / "task_1a_manual_v1.toml"
RECORDING = REPO_ROOT / "configs" / "recording" / "task_1a_manual_v1.toml"


@pytest.fixture(scope="module")
def qapp() -> object:
    """Provide a single QApplication instance for the GUI tests."""
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _resolve(
    output_root: Path,
    *,
    session: str = "study-a",
    purpose: str = "study",
    recording: Path = RECORDING,
) -> RecorderSession:
    return resolve_recorder_session(
        SCENARIO, recording, session=session, purpose=purpose, output_root=output_root, repo_root=REPO_ROOT
    )


def test_recording_config_holds_the_d1_and_d2_settings() -> None:
    """IK guidance, both overlays, the 30 s timeout, and the numbered output base."""
    config = load_recording_config(RECORDING)
    assert config.protocol == "task_1a_manual_v1"
    assert config.mode == "ik"
    assert config.timeout_s == 30.0
    assert config.show_tip_trail
    assert config.show_past_trails
    assert config.output_name == "reach.sklog.npz"


def test_session_resolves_the_task_sampling_and_output_layout(tmp_path: Path) -> None:
    """100 Hz from the task period, the timeout as the duration cap, and a per-purpose, per-session directory."""
    session = _resolve(tmp_path)
    assert session.sample_rate_hz == 100.0
    assert session.duration_s == 30.0
    assert session.output_base == tmp_path / "study" / "study-a" / "reach.sklog.npz"
    assert session.multi_take
    assert not session.start_on_grab
    assert session.initial_q == (0.2, 1.2)
    assert session.target == (0.10, 0.45)
    assert session.tolerance == 0.01
    assert session.scenario_path == "configs/tasks/task_1a_manual_v1.toml"
    assert session.scenario_sha256 == sha256_file(SCENARIO)
    assert session.recording_sha256 == sha256_file(RECORDING)
    practice = _resolve(tmp_path, session="practice-a", purpose="practice")
    assert practice.output_base.parent.parent == tmp_path / "practice"


def test_unsafe_or_inconsistent_sessions_are_refused(tmp_path: Path) -> None:
    """Outputs stay outside the repository in an existing absolute directory; names and protocols must match."""
    with pytest.raises(RecordingError, match="outside the repository"):
        _resolve(REPO_ROOT / "docs")
    with pytest.raises(RecordingError, match="absolute"):
        _resolve(tmp_path.relative_to(tmp_path.anchor))
    with pytest.raises(RecordingError, match="exist"):
        _resolve(tmp_path / "missing")
    with pytest.raises(RecordingError, match="session"):
        _resolve(tmp_path, session="Bad Session")
    with pytest.raises(RecordingError, match="purpose"):
        _resolve(tmp_path, purpose="pilot")
    other = tmp_path / "other.toml"
    other.write_text(
        RECORDING.read_text(encoding="utf-8").replace('"task_1a_manual_v1"', '"another_protocol"'), "utf-8"
    )
    with pytest.raises(RecordingError, match="protocol"):
        _resolve(tmp_path, recording=other)


def test_the_recorder_window_matches_the_protocol(qapp: object, tmp_path: Path) -> None:  # noqa: ARG001
    """Exact reset posture, joint limits, target marker, 10 ms tick, first numbered output, and both overlays."""
    session = _resolve(tmp_path)
    config = load_manual_scenario(SCENARIO)
    window = create_recorder_window(session, config, run_timer=False)
    assert np.array_equal(window.skeleton.q, np.asarray(config.task.initial_q))
    assert [link.prop.qmin for link in window.skeleton.links[1:]] == [link.q_min for link in config.robot.links]
    assert [link.prop.qmax for link in window.skeleton.links[1:]] == [link.q_max for link in config.robot.links]
    assert np.array_equal(window.canvas.target, np.asarray(config.task.target))
    assert window.canvas.target_tolerance == config.task.tolerance
    assert window.tick_ms == 10
    assert window.state == "ready"
    assert window.output_path == session.output_base.with_name("reach_001.sklog.npz")
    assert window.checkboxes["tip_trail"].isChecked()
    assert window.checkboxes["past_trails"].isChecked()
    assert window.close()


def test_session_settings_are_portable_and_never_change(tmp_path: Path) -> None:
    """The settings file names configs and the recorder commit without machine paths and refuses a changed session."""
    session = _resolve(tmp_path)
    path = write_session_settings(session, repo_root=REPO_ROOT)
    assert path == session.output_base.parent / "session.json"
    text = path.read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    data = json.loads(text)
    skelarm = next(s for s in submodule_revisions(REPO_ROOT) if s.name == "skelarm")
    assert data["recorder"]["skelarm_commit"] == (skelarm.checked_out or skelarm.recorded)
    assert data["options"]["sample_rate_hz"] == 100.0
    assert write_session_settings(session, repo_root=REPO_ROOT) == path
    changed = tmp_path / "changed.toml"
    changed.write_text(RECORDING.read_text(encoding="utf-8").replace("timeout_s = 30.0", "timeout_s = 20.0"), "utf-8")
    with pytest.raises(RecordingError, match="changed"):
        write_session_settings(_resolve(tmp_path, recording=changed), repo_root=REPO_ROOT)


def test_the_launcher_dry_run_prints_the_session_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dry run resolves and prints the portable settings without creating the session directory."""
    argv = [
        "--scenario", str(SCENARIO), "--recording", str(RECORDING), "--session", "practice-a",
        "--purpose", "practice", "--output-root", str(tmp_path), "--dry-run",
    ]  # fmt: skip
    assert main(argv) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["options"]["duration_s"] == 30.0
    assert data["purpose"] == "practice"
    assert not (tmp_path / "practice").exists()
    script = REPO_ROOT / "scripts" / "record_demo.py"
    result = subprocess.run(  # trusted: our interpreter and script
        [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "--dry-run" in result.stdout


def _drifted(drift: str) -> tuple[SubmoduleRevision, ...]:
    """The repository's submodule revisions with the skelarm checkout dirty, off its pin, or uninitialized."""
    changed: list[SubmoduleRevision] = []
    for revision in submodule_revisions(REPO_ROOT):
        if revision.name != "skelarm":
            changed.append(revision)
        elif drift == "dirty":
            changed.append(replace(revision, dirty=True))
        elif drift == "different_commit":
            changed.append(replace(revision, checked_out="0" * 40))
        else:
            changed.append(replace(revision, checked_out=None, dirty=None))
    return tuple(changed)


@pytest.mark.parametrize("drift", ["dirty", "different_commit", "uninitialized"])
def test_an_unpinned_or_modified_recorder_is_refused_everywhere(
    qapp: object,  # noqa: ARG001
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    drift: str,
) -> None:
    """Resolution, the dry run, and the window all refuse a recorder that is not the clean pinned checkout."""
    session = _resolve(tmp_path)
    changed = _drifted(drift)

    def drifted_revisions(_root: Path | None = None) -> tuple[SubmoduleRevision, ...]:
        return changed

    monkeypatch.setattr(recording, "submodule_revisions", drifted_revisions)
    with pytest.raises(RecordingError, match="skelarm"):
        _resolve(tmp_path)
    with pytest.raises(RecordingError, match="skelarm"):
        create_recorder_window(session, load_manual_scenario(SCENARIO), run_timer=False)
    with pytest.raises(RecordingError, match="skelarm"):
        write_session_settings(session, repo_root=REPO_ROOT)
    argv = [
        "--scenario", str(SCENARIO), "--recording", str(RECORDING), "--session", "study-b",
        "--purpose", "study", "--output-root", str(tmp_path), "--dry-run",
    ]  # fmt: skip
    assert main(argv) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "skelarm" in captured.err
    assert not (tmp_path / "study" / "study-b").exists()


def test_a_stale_installed_build_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The installed skelarm package must be the build of the pinned checkout."""

    def stale(_root: Path | None = None) -> tuple[object, ...]:
        msg = "skelarm: installed Python sources differ from the recorded build"
        raise BuildIdentityError(msg)

    monkeypatch.setattr(recording, "verify_builds", stale)
    with pytest.raises(RecordingError, match="rebuild"):
        _resolve(tmp_path)


def test_session_settings_record_the_verified_build(tmp_path: Path) -> None:
    """The settings carry the installed skelarm build identity that the guard verified."""
    data = json.loads(write_session_settings(_resolve(tmp_path), repo_root=REPO_ROOT).read_text(encoding="utf-8"))
    build = next(b for b in verify_builds(REPO_ROOT) if b.name == "skelarm")
    assert data["recorder"]["skelarm_python_sources_sha256"] == build.python_sources_sha256
    assert data["recorder"]["skelarm_version"] == build.version
