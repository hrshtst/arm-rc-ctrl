# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Launch the pinned skelarm trajectory recorder for a manual-demonstration session (M3MAN-003, clarification I7).

The recorder reads its own robot schema (link limits in degrees, a task ``type``),
not the arm-rc-ctrl task configuration. This adapter therefore builds the
recorder in-process from the versioned manual task configuration through the
shared scenario conversion, so the reset posture reaches the recorder exactly
(no degree round trip), and verifies the posture, joint limits, target marker,
sampling tick, and numbered output before anything is recorded. The sampling
rate comes from the recording configuration when it declares ``sample_rate_hz``
(``task_1a_manual_v2``, clarification I13: 50 Hz acquisition beside the 0.01 s
training grid); a v1 recording configuration keeps the rate of the task period
``timing.dt``. Either way the rate must give a whole-millisecond tick, the task's
training grid must be at least as fine, and the task's acquisition rules must
expect that rate. Takes go to an
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
from arm_rc_ctrl.dependencies import BuildIdentityError, submodule_revisions, verify_builds
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

    from arm_rc_ctrl.dependencies import BuildIdentity, SubmoduleRevision

__all__ = [
    "RECORDER",
    "SETTINGS_FILE",
    "RecorderSession",
    "RecordingConfig",
    "RecordingError",
    "create_recorder_window",
    "load_recording_config",
    "main",
    "require_pinned_recorder",
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
_PERIOD_TOLERANCE_S: Final = 1e-12


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
    sample_rate_hz: float | None = None
    """Acquisition rate (v2, I13); without it the rate follows the task period ``timing.dt`` (v1)."""
    past_trail_history: Literal["all", "last"] | None = None
    """Faint-trail history drawn by the recorder (v2, I12): ``last`` draws only the most recently saved take.

    Without it the recorder keeps its full-history display (v1).
    """

    def __post_init__(self) -> None:
        """Validate the protocol, the timeout, the acquisition rate, and the output base name."""
        if not self.protocol.strip() or not self.ik_method.strip():
            msg = "protocol and ik_method must not be empty"
            raise ValueError(msg)
        if not (self.timeout_s > 0 and math.isfinite(self.timeout_s)):
            msg = f"timeout_s must be positive and finite, got {self.timeout_s!r}"
            raise ValueError(msg)
        if self.sample_rate_hz is not None and not (self.sample_rate_hz > 0 and math.isfinite(self.sample_rate_hz)):
            msg = f"sample_rate_hz must be positive and finite, got {self.sample_rate_hz!r}"
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
    past_trail_history: str | None
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


def require_pinned_recorder(repo_root: Path) -> tuple[SubmoduleRevision, BuildIdentity]:
    """The skelarm checkout the recorder runs from, verified as initialized, clean, at its pin, and installed as built.

    Session settings name the recorder by its commit, so a recording from a
    modified or unpinned recorder must never start.

    Raises
    ------
    RecordingError
        If ``third_party/skelarm`` is uninitialized, has uncommitted changes, is
        checked out away from its pin, or the installed build does not match it.
    """
    skelarm = next((s for s in submodule_revisions(repo_root) if s.name == "skelarm"), None)
    if skelarm is None:
        msg = "skelarm is not a submodule of this repository; the recorder cannot be pinned"
        raise RecordingError(msg)
    if skelarm.checked_out is None:
        msg = "third_party/skelarm is not initialized; run `git submodule update --init third_party/skelarm`"
        raise RecordingError(msg)
    if skelarm.dirty is not False:
        msg = (
            "third_party/skelarm has uncommitted changes; the recorder must run from the clean pinned skelarm checkout"
        )
        raise RecordingError(msg)
    if skelarm.checked_out != skelarm.recorded:
        msg = (
            f"third_party/skelarm is checked out at {skelarm.checked_out[:12]}, not at its pin "
            f"{skelarm.recorded[:12]}; check out the pinned skelarm commit"
        )
        raise RecordingError(msg)
    try:
        builds = verify_builds(repo_root)
    except BuildIdentityError as exc:
        msg = (
            "the installed skelarm build does not match the pinned checkout; run "
            f"`uv run python -m arm_rc_ctrl.dependencies rebuild`: {exc}"
        )
        raise RecordingError(msg) from exc
    build = next((b for b in builds if b.name == "skelarm"), None)
    if build is None:
        msg = "no installed skelarm build identity is recorded; run `uv run python -m arm_rc_ctrl.dependencies rebuild`"
        raise RecordingError(msg)
    return skelarm, build


def _acquisition_rate(scenario: ManualScenarioConfig, recording: RecordingConfig) -> float:
    """The recorder's sampling rate: the recording configuration's ``sample_rate_hz``, else ``1 / timing.dt`` (v1).

    Raises
    ------
    RecordingError
        If the rate is not a whole number of milliseconds per tick, the task's
        training grid is coarser than the rate, or the task's acquisition rules
        expect another rate.
    """
    if recording.sample_rate_hz is None:
        tick_ms = 1000.0 * scenario.timing.dt
        if abs(tick_ms - round(tick_ms)) > _TICK_TOLERANCE:
            msg = (
                f"timing.dt {scenario.timing.dt} s is not a whole number of milliseconds; the recorder cannot sample it"
            )
            raise RecordingError(msg)
        rate = 1000.0 / round(tick_ms)
    else:
        rate = recording.sample_rate_hz
        tick_ms = 1000.0 / rate
        if round(tick_ms) < 1 or abs(tick_ms - round(tick_ms)) > _TICK_TOLERANCE:
            msg = (
                f"sample_rate_hz {rate:g} gives a {tick_ms:.4g} ms period, not a whole number of milliseconds; "
                "the recorder cannot sample it"
            )
            raise RecordingError(msg)
        if scenario.timing.dt > 1.0 / rate + _PERIOD_TOLERANCE_S:
            msg = (
                f"the task's training grid timing.dt {scenario.timing.dt} s is coarser than the acquisition period "
                f"{1.0 / rate} s; the grid must be at least as fine as the acquisition rate"
            )
            raise RecordingError(msg)
    expected = scenario.acquisition_period_s
    if abs(1.0 / rate - expected) > _PERIOD_TOLERANCE_S:
        msg = (
            f"the recorder would sample at {rate:g} Hz but the task's acquisition rules expect {1.0 / expected:g} Hz; "
            "pair the recording configuration with the task configuration of the same acquisition rate"
        )
        raise RecordingError(msg)
    return rate


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
        the acquisition rate cannot be sampled, is finer than the training grid,
        or differs from the task's acquisition rules.
    """
    require_pinned_recorder(repo_root)
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
    rate = _acquisition_rate(scenario, recording)
    return RecorderSession(
        protocol=scenario.protocol,
        purpose=purpose,
        session=session,
        mode=recording.mode,
        ik_method=recording.ik_method,
        sample_rate_hz=rate,
        duration_s=recording.timeout_s,
        output_base=output_root / purpose / session / recording.output_name,
        multi_take=True,
        start_on_grab=False,
        show_tip_trail=recording.show_tip_trail,
        show_past_trails=recording.show_past_trails,
        past_trail_history=recording.past_trail_history,
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
    repo_root: Path | None = None,
):
    """Build the pinned recorder window for ``session`` and verify it against the protocol.

    Raises
    ------
    RecordingError
        If the posture, joint limits, target marker, or sampling tick of the
        built recorder differ from the configuration.
    """
    require_pinned_recorder(repository_root() if repo_root is None else repo_root)
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
        past_trail_history=session.past_trail_history or "all",
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
    expected_history = session.past_trail_history or "all"
    if window.past_trail_history != expected_history:
        msg = f"the recorder draws the {window.past_trail_history!r} saved-trail history, not {expected_history!r}"
        raise RecordingError(msg)
    return window


def _history_option(session: RecorderSession) -> dict[str, str]:
    """The display-history option for ``session.json``; absent for v1 sessions so their settings layout is unchanged."""
    return {} if session.past_trail_history is None else {"past_trail_history": session.past_trail_history}


def session_settings(session: RecorderSession, *, repo_root: Path) -> dict[str, object]:
    """The portable settings of a session: configurations by digest, the pinned recorder, and the resolved options."""
    skelarm, build = require_pinned_recorder(repo_root)
    return {
        "settings_schema_version": SETTINGS_SCHEMA_VERSION,
        "protocol": session.protocol,
        "purpose": session.purpose,
        "session": session.session,
        "scenario": {"path": session.scenario_path, "sha256": session.scenario_sha256},
        "recording": {"path": session.recording_path, "sha256": session.recording_sha256},
        "recorder": {
            "tool": "third_party/skelarm/tools/trajectory_recorder.py",
            "skelarm_commit": skelarm.checked_out,
            "skelarm_version": build.version,
            "skelarm_python_sources_sha256": build.python_sources_sha256,
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
            **_history_option(session),
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
