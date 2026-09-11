# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""The installed run-ID playback command (TOOL-003)."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from importlib.metadata import distribution
from pathlib import Path

import pytest

from arm_rc_ctrl.experiments.playback import main_play
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.integration


def test_playback_console_entry_point_uses_the_existing_player() -> None:
    """Packaging exposes the same verified playback path as scripts/play_run.py."""
    project = tomllib.loads((repository_root() / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"]["arm-rc-play-run"] == "arm_rc_ctrl.experiments.playback:main_play"
    entries = [e for e in distribution("arm-rc-ctrl").entry_points if e.name == "arm-rc-play-run"]
    assert len(entries) == 1
    assert entries[0].load() is main_play


def test_installed_playback_help_needs_no_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The executable advertises run lookup, export, and task-clock controls without loading evidence."""
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(tmp_path / "missing-store"))
    command = Path(sys.executable).with_name("arm-rc-play-run")
    completed = subprocess.run([str(command), "--help"], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    for option in ("--run", "--export", "--fps", "--panel", "--task-clock", "--speed"):
        assert option in completed.stdout
    assert "configured store" in " ".join(completed.stdout.split())
