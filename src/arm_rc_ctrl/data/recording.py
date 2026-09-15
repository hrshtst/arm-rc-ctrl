# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Launch the pinned skelarm trajectory recorder for a manual-demonstration session (M3MAN-003, clarification I7).

The recorder reads its own robot schema (link limits in degrees, a task ``type``),
not the arm-rc-ctrl task configuration. This adapter therefore builds the
recorder in-process from the versioned manual task configuration through the
shared scenario conversion, so the reset posture reaches the recorder exactly
(no degree round trip), and verifies the posture, joint limits, target marker,
sampling tick, and numbered output before anything is recorded. Takes go to an
absolute directory outside the repository, separated by purpose (``practice``
or ``study``) and session, next to a portable ``session.json`` that names the
configurations by digest and the recorder by its pinned commit.

Usage::

    uv run python scripts/record_demo.py --scenario configs/tasks/task_1a_manual_v1.toml
    --recording configs/recording/task_1a_manual_v1.toml --session <session> --purpose study
    --output-root <absolute directory outside the repository> [--dry-run]
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
from skelarm import Task

from arm_rc_ctrl.config import ConfigError, load_config
from arm_rc_ctrl.data.manual_scenario import ManualScenarioConfig, load_manual_scenario, manual_build_skeleton
from arm_rc_ctrl.dependencies import submodule_revisions
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

__all__ = [
    "RECORDER",
    "SETTINGS_FILE",
    "RecorderSession",
    "RecordingConfig",
    "RecordingError",
    "create_recorder_window",
    "load_recording_config",
    "main",
    "resolve_recorder_session",
    "session_settings",
    "write_session_settings",
]

RECORDER: Final = repository_root() / "third_party" / "skelarm" / "tools" / "trajectory_recorder.py"
SETTINGS_FILE: Final = "session.json"
SETTINGS_SCHEMA_VERSION: Final = 1
_PURPOSES: Final = ("practice", "study")
_SESSION_RE: Final = re.compile(r"^[a-z0-9][a-z0-9-]{2,31}$")
_LOG_SUFFIX: Final = ".sklog.npz"
_TICK_TOLERANCE: Final = 1e-9


class RecordingError(RuntimeError):
    """A recorder session is unsafe or inconsistent with the manual protocol."""


@dataclass(frozen=True)
class RecordingConfig:
    """Recorder options of one protocol (``configs/recording/*.toml``)."""

    protocol: str
    mode: Literal["ik"]
    """IK guidance (D1): the tip follows the cursor through inverse kinematics."""
    ik_method: str
    timeout_s: float
    """Per-take recording timeout (D2); a reached timeout saves the take and keeps the window open."""
    show_tip_trail: bool
    show_past_trails: bool
    output_name: str
    """Base of the numbered take files, e.g. ``reach.sklog.npz`` -> ``reach_001.sklog.npz``."""

    def __post_init__(self) -> None:
        """Validate the protocol, the timeout, and the output base name."""
        if not self.protocol.strip() or not self.ik_method.strip():
            msg = "protocol and ik_method must not be empty"
            raise ValueError(msg)
        if not (self.timeout_s > 0 and math.isfinite(self.timeout_s)):
            msg = f"timeout_s must be positive and finite, got {self.timeout_s!r}"
            raise ValueError(msg)
        if "/" in self.output_name or not self.output_name.endswith(_LOG_SUFFIX) or self.output_name == _LOG_SUFFIX:
            msg = f"output_name must be a bare file name ending in {_LOG_SUFFIX}, got {self.output_name!r}"
            raise ValueError(msg)


def load_recording_config(path: Path) -> RecordingConfig:
    """Load and validate a recording configuration."""
    return load_config(path, RecordingConfig)


@dataclass(frozen=True)
class RecorderSession:
    """Everything the recorder is started with, resolved from the task and recording configurations."""

    protocol: str
    purpose: str
    session: str
    mode: str
    ik_method: str
    sample_rate_hz: float
    duration_s: float
    output_base: Path
    multi_take: bool
    start_on_grab: bool
    show_tip_trail: bool
    show_past_trails: bool
    initial_q: tuple[float, ...]
    target: tuple[float, ...]
    tolerance: float
    scenario_path: str
    scenario_sha256: str
    recording_path: str
    recording_sha256: str


def _portable(path: Path, repo_root: Path) -> str:
    """Repository-relative POSIX path, or the bare file name for a file outside the repository."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _check_output_root(output_root: Path, repo_root: Path) -> None:
    if not output_root.is_absolute():
        msg = f"--output-root must be an absolute directory, got {output_root}"
        raise RecordingError(msg)
    if not output_root.is_dir():
        msg = f"--output-root {output_root} does not exist; create it first (takes are never written elsewhere)"
        raise RecordingError(msg)
    try:
        output_root.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return
    msg = f"--output-root {output_root} must lie outside the repository; takes are payloads, never Git content"
    raise RecordingError(msg)


def resolve_recorder_session(
    scenario_file: Path,
    recording_file: Path,
    *,
    session: str,
    purpose: str,
    output_root: Path,
    repo_root: Path,
) -> RecorderSession:
    """Resolve and check the recorder options of one session.

    Raises
    ------
    RecordingError
        If the protocols differ, the session or purpose is invalid, the output
        root is not an existing absolute directory outside the repository, or
        the task period is not a whole number of milliseconds.
    """
    scenario = load_manual_scenario(scenario_file)
    recording = load_recording_config(recording_file)
    if recording.protocol != scenario.protocol:
        msg = f"recording protocol {recording.protocol!r} differs from the task protocol {scenario.protocol!r}"
        raise RecordingError(msg)
    if not _SESSION_RE.match(session):
        msg = f"session must match {_SESSION_RE.pattern}, got {session!r}"
        raise RecordingError(msg)
    if purpose not in _PURPOSES:
        msg = f"purpose must be one of {list(_PURPOSES)}, got {purpose!r}"
        raise RecordingError(msg)
    _check_output_root(output_root, repo_root)
    tick_ms = 1000.0 * scenario.timing.dt
    if abs(tick_ms - round(tick_ms)) > _TICK_TOLERANCE:
        msg = f"timing.dt {scenario.timing.dt} s is not a whole number of milliseconds; the recorder cannot sample it"
        raise RecordingError(msg)
    return RecorderSession(
        protocol=scenario.protocol,
        purpose=purpose,
        session=session,
        mode=recording.mode,
        ik_method=recording.ik_method,
        sample_rate_hz=1000.0 / round(tick_ms),
        duration_s=recording.timeout_s,
        output_base=output_root / purpose / session / recording.output_name,
        multi_take=True,
        start_on_grab=False,
        show_tip_trail=recording.show_tip_trail,
        show_past_trails=recording.show_past_trails,
        initial_q=scenario.task.initial_q,
        target=scenario.task.target,
        tolerance=scenario.task.tolerance,
        scenario_path=_portable(scenario_file, repo_root),
        scenario_sha256=sha256_file(scenario_file),
        recording_path=_portable(recording_file, repo_root),
        recording_sha256=sha256_file(recording_file),
    )


@functools.cache
def _recorder_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("arm_rc_ctrl_pinned_trajectory_recorder", path)
    if spec is None or spec.loader is None:
        msg = f"cannot load the pinned recorder from {path}"
        raise RecordingError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their defining module through sys.modules
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[spec.name]
        raise
    return module


def create_recorder_window(  # noqa: ANN201  # the pinned tool's RecorderWindow has no importable type
    session: RecorderSession,
    config: ManualScenarioConfig,
    *,
    run_timer: bool = True,
    recorder_path: Path = RECORDER,
):
    """Build the pinned recorder window for ``session`` and verify it against the protocol.

    Raises
    ------
    RecordingError
        If the posture, joint limits, target marker, or sampling tick of the
        built recorder differ from the configuration.
    """
    skeleton = manual_build_skeleton(config)
    if not np.array_equal(skeleton.q, np.asarray(session.initial_q, dtype=np.float64)):
        msg = (
            f"the recorder skeleton starts at {skeleton.q.tolist()}, not at the reset posture {list(session.initial_q)}"
        )
        raise RecordingError(msg)
    limits = [(link.prop.qmin, link.prop.qmax) for link in skeleton.links[1:]]
    if limits != [(link.q_min, link.q_max) for link in config.robot.links]:
        msg = f"the recorder joint limits {limits} differ from the configured limits (rad)"
        raise RecordingError(msg)
    task = Task.from_dict({"type": "reaching", "target": {"pos": list(session.target), "tolerance": session.tolerance}})
    session.output_base.parent.mkdir(parents=True, exist_ok=True)
    recorder: Any = _recorder_module(recorder_path)
    window: Any = recorder.RecorderWindow(
        skeleton,
        mode=session.mode,
        sample_rate=session.sample_rate_hz,
        duration=session.duration_s,
        output=session.output_base,
        multi_take=session.multi_take,
        start_on_grab=session.start_on_grab,
        show_tip_trail=session.show_tip_trail,
        show_past_trails=session.show_past_trails,
        method=session.ik_method,
        task=task,
        run_timer=run_timer,
    )
    if window.tick_ms != round(1000.0 / session.sample_rate_hz):
        msg = f"the recorder ticks every {window.tick_ms} ms, not every {1000.0 / session.sample_rate_hz} ms"
        raise RecordingError(msg)
    if not np.array_equal(window.canvas.target, np.asarray(session.target, dtype=np.float64)):
        msg = "the recorder's target marker differs from the configured target"
        raise RecordingError(msg)
    return window


def session_settings(session: RecorderSession, *, repo_root: Path) -> dict[str, object]:
    """The portable settings of a session: configurations by digest, the pinned recorder, and the resolved options."""
    skelarm = next(s for s in submodule_revisions(repo_root) if s.name == "skelarm")
    return {
        "settings_schema_version": SETTINGS_SCHEMA_VERSION,
        "protocol": session.protocol,
        "purpose": session.purpose,
        "session": session.session,
        "scenario": {"path": session.scenario_path, "sha256": session.scenario_sha256},
        "recording": {"path": session.recording_path, "sha256": session.recording_sha256},
        "recorder": {
            "tool": "third_party/skelarm/tools/trajectory_recorder.py",
            "skelarm_commit": skelarm.checked_out or skelarm.recorded,
        },
        "options": {
            "mode": session.mode,
            "ik_method": session.ik_method,
            "sample_rate_hz": session.sample_rate_hz,
            "duration_s": session.duration_s,
            "multi_take": session.multi_take,
            "start_on_grab": session.start_on_grab,
            "show_tip_trail": session.show_tip_trail,
            "show_past_trails": session.show_past_trails,
            "output_name": session.output_base.name,
        },
        "task": {"initial_q": list(session.initial_q), "target": list(session.target), "tolerance": session.tolerance},
    }


def write_session_settings(session: RecorderSession, *, repo_root: Path) -> Path:
    """Write ``session.json`` beside the takes; an identical file is kept, a different one refused.

    Raises
    ------
    RecordingError
        If the session directory already holds different settings.
    """
    path = session.output_base.parent / SETTINGS_FILE
    text = json.dumps(session_settings(session, repo_root=repo_root), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            msg = (
                f"{path} holds different settings: the recorder settings changed, but a session's settings "
                "never change, so start a new session"
            )
            raise RecordingError(msg)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the launcher."""
    parser = argparse.ArgumentParser(
        description="Launch the pinned trajectory recorder for a manual-demonstration session."
    )
    parser.add_argument("--scenario", type=Path, required=True, help="manual-protocol task TOML")
    parser.add_argument("--recording", type=Path, required=True, help="recording configuration TOML")
    parser.add_argument("--session", required=True, help="session ID (lowercase letters, digits, hyphens)")
    parser.add_argument(
        "--purpose", choices=_PURPOSES, required=True, help="practice takes are kept apart from study takes"
    )
    parser.add_argument("--output-root", type=Path, required=True, help="absolute directory outside the repository")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved settings and exit without recording")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve the session, write its settings, and run the recorder (or only print the settings with --dry-run)."""
    args = build_parser().parse_args(argv)
    repo_root = repository_root()
    try:
        session = resolve_recorder_session(
            args.scenario,
            args.recording,
            session=args.session,
            purpose=args.purpose,
            output_root=args.output_root,
            repo_root=repo_root,
        )
        if args.dry_run:
            print(json.dumps(session_settings(session, repo_root=repo_root), indent=2, sort_keys=True))
            return 0
        settings = write_session_settings(session, repo_root=repo_root)
        from PyQt6.QtWidgets import QApplication  # the GUI is needed only for a real session

        app = QApplication(sys.argv[:1])
        window = create_recorder_window(session, load_manual_scenario(args.scenario))
    except (RecordingError, ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"recording {session.purpose} session {session.session!r}: takes in {session.output_base.parent}")
    print(f"settings: {settings}")
    window.show()
    window.canvas.setFocus()
    app.exec()
    return 0
