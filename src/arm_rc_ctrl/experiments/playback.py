# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Export a verified run as a disposable ``skelarm.StateLog`` and play it (``docs/PLAN.md`` 7.5; TOOL-001).

The converter resolves a Git-tracked run pointer, verifies the external
payload against its recorded digest, checks the scenario the run claims,
and writes a ``*.sklog.npz`` the pinned ``skelarm`` player can open: the
robot geometry, the measured trajectory (``time`` and ``q``), the canonical
playback torque (applied when recorded, requested otherwise — both original
channels stay available), every other usable telemetry channel
(visualization-only floating-point copies for integer channels), the task
target and tolerance as playback-only metadata, the disturbances, and the
run's provenance identity. The log is a local, disposable product: it gets
no artifact record, is ignored by Git, must not overwrite anything, is
written atomically, and carries no absolute machine path. This is kinematic
inspection of recorded state — not controller re-execution or simulation.

Command lines::

    python scripts/export_run_sklog.py --run run-20260831-... --scenario configs/tasks/task_1a.toml --out run.sklog.npz
    python scripts/play_run.py --run run-20260831-...
        [--scenario configs/tasks/task_1a.toml] [--speed 0.5] [--show-com] [--panel]
        [--export run.mp4|run.gif] [--fps 24]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import numpy as np

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.data.records import is_artifact_id, load_record, verify_payload, write_record
from arm_rc_ctrl.experiments.run_record import RunPointerRecord, load_run, pointer_from_summary
from arm_rc_ctrl.provenance import canonical_json, portable_config, sha256_file
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import ScenarioConfig, build_skeleton, load_scenario
from arm_rc_ctrl.storage import StorageError, StorageRoot, open_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from arm_rc_ctrl.experiments.run_record import LoadedRun
    from arm_rc_ctrl.storage import StorageRoot

__all__ = ["PLAYER", "export_run_sklog", "main_export", "main_play", "sklog_channels"]

PLAYER: Final = repository_root() / "third_party" / "skelarm" / "tools" / "player.py"
_SUFFIX: Final = ".sklog.npz"
_INTEGER_CHANNELS: Final = ("saturation", "phase")
"""Integer telemetry exported as visualization-only floating-point copies."""
_SKIPPED: Final = ("t",)
"""Arrays that are not exported as channels (time drives the log)."""


def resolve_pointer(
    run_id: str, records_root: Path, *, store: StorageRoot | None = None, scratch: Path | None = None
) -> Path:
    """The Git-tracked pointer record of ``run_id`` (full ID grammar enforced), or an evidence-backed one.

    Runs of the repetition pilot are referenced by their evidence manifests
    rather than by individual pointer records; when no record is tracked and
    ``store`` and ``scratch`` are given, the pointer is reconstructed from the
    run's stored summary (the report's own animation export does the same) and
    written under ``scratch`` for the caller's lifetime.
    """
    if not is_artifact_id(run_id) or not run_id.startswith("run-"):
        msg = f"{run_id!r} is not a run ID (expected run-<yyyymmdd>-<12 hex digits>)"
        raise ValueError(msg)
    pointer_file = records_root / "data" / "records" / "runs" / f"{run_id}.toml"
    if pointer_file.is_file():
        return pointer_file
    if store is None or scratch is None:
        msg = f"no pointer record for {run_id} under data/records/runs"
        raise FileNotFoundError(msg)
    try:
        pointer = pointer_from_summary(store, run_id)
    except (StorageError, FileNotFoundError, ValueError) as exc:
        msg = (
            f"no pointer record for {run_id} under data/records/runs and no stored run summary in the configured "
            "store to reconstruct one from"
        )
        raise FileNotFoundError(msg) from exc
    scratch.mkdir(parents=True, exist_ok=True)
    reconstructed = scratch / f"{run_id}.toml"
    write_record(reconstructed, pointer)
    return reconstructed


def sklog_channels(run: LoadedRun) -> dict[str, NDArray[np.float64]]:
    """The exported channels: canonical ``q``/``dq``/``tau`` plus every other usable telemetry array."""
    arrays = run.arrays.arrays
    channels: dict[str, NDArray[np.float64]] = {}
    tau_source = "tau_applied" if "tau_applied" in arrays else "tau_requested"
    channels["tau"] = np.asarray(arrays[tau_source], dtype=np.float64)
    for name, array in arrays.items():
        if name in _SKIPPED:
            continue
        values = np.asarray(array, dtype=np.float64)
        if name == "task_code" and values.shape[1] == 0:
            continue  # a zero-width task code (task 1-a) carries nothing to draw or plot
        channels[name] = values
    return channels


def _playback_extra(
    run: LoadedRun,
    scenario: ScenarioConfig,
    tau_source: str,
    *,
    task_clock: bool = False,
    scenario_source: str = "provenance",
) -> dict[str, object]:
    summary = run.summary
    provenance = summary.provenance
    clock: dict[str, object] = (
        {
            "kind": "task",
            "activation_s": float(summary.activation_s or 0.0),
            "note": "log times are task-relative (run time minus the activation): warm-up negative, activation at 0",
        }
        if task_clock
        else {"kind": "run", **({} if summary.activation_s is None else {"activation_s": float(summary.activation_s)})}
    )
    return {
        "clock": clock,
        "scenario_source": scenario_source,
        "playback": {
            "task": {
                "type": "reaching",
                "target": {
                    "pos": [float(v) for v in scenario.task.target],
                    "tolerance": float(scenario.task.tolerance),
                },
            }
        },
        "run": {
            "id": run.pointer.artifact.artifact_id,
            "method": summary.method,
            "scenario": summary.scenario,
            "termination": summary.termination.kind,
            "success": summary.outcome.success,
            "duration_s": float(summary.duration_s),
            "control_period_s": float(summary.control_period_s),
            "tau_source": tau_source,
            "arrays_sha256": summary.arrays_sha256,
            "payload_sha256": run.pointer.artifact.payload.sha256,
            "project_commit": provenance.project_commit,
            "config_sha256": provenance.config_sha256,
            "created_at": provenance.created_at,
            "disturbances": [
                {"kind": d.kind, "start_s": float(d.start_s), "end_s": float(d.end_s), "parameters": dict(d.parameters)}
                for d in summary.disturbances
            ],
        },
    }


def _recorded_scenario(run: LoadedRun, scenario_file: Path | None) -> tuple[ScenarioConfig, str]:
    """The scenario the run was recorded under, and where it came from.

    The geometry, timing, limits, and task come from the run's own
    ``provenance.config`` whenever it holds the scenario; a supplied
    ``scenario_file`` is then only verified — it must match the recorded
    scenario completely, so a same-name file with altered links, timing,
    tolerance, or target is refused. Runs whose provenance binds the scenario
    only through a study manifest (the repeated-demonstration pilot's runs)
    carry no scenario mapping; for those the file is required, its name must
    be the run's recorded scenario name, and the source is reported as
    ``file:<sha256>`` in the log so the substitution is never silent.
    """
    stored = run.summary.provenance.config.get("scenario")
    if stored is None:
        if scenario_file is None:
            msg = (
                f"run {run.pointer.artifact.artifact_id} records no scenario in its provenance; pass the scenario "
                "file it was recorded under (verified by name, reported as the log's scenario source)"
            )
            raise ValueError(msg)
        given = load_scenario(scenario_file)
        if given.name != run.summary.scenario:
            msg = (
                f"{scenario_file.name} is the scenario {given.name!r}, but the run was recorded under "
                f"{run.summary.scenario!r}"
            )
            raise ValueError(msg)
        return given, f"file:{sha256_file(scenario_file)}"
    scenario = from_mapping(cast("dict[str, object]", stored), ScenarioConfig)
    if scenario_file is not None:
        given = load_scenario(scenario_file)
        if canonical_json(portable_config(to_mapping(given))) != canonical_json(portable_config(stored)):
            msg = (
                f"{scenario_file.name} does not match the scenario recorded in the run's provenance "
                f"(name {scenario.name!r}): links, timing, limits, or task differ"
            )
            raise ValueError(msg)
    return scenario, "provenance"


def export_run_sklog(
    store: StorageRoot,
    pointer_file: Path,
    out: Path,
    *,
    scenario_file: Path | None = None,
    task_clock: bool = False,
) -> Path:
    """Convert one verified run into a playable ``*.sklog.npz`` (atomic; never overwrites).

    With ``task_clock`` the log's times are the run times minus the recorded
    activation (repetition plan clarification C5): the warm-up is negative and
    activation is at 0 s, so the player's ``t = ...`` label reads task time.
    The run record keeps its run-clock timestamps; the transform is recorded in
    the log's ``clock`` extra. The default export is unchanged.
    """
    from skelarm import StateLog  # deferred: pulls in the full skelarm package

    if not out.name.endswith(_SUFFIX):
        msg = f"the output must end with {_SUFFIX!r}, got {out.name!r}"
        raise ValueError(msg)
    if out.is_symlink() or out.exists():
        msg = f"refusing to overwrite {out}"
        raise FileExistsError(msg)
    pointer = load_record(pointer_file, RunPointerRecord)
    if f"{pointer.artifact.artifact_id}.toml" != pointer_file.name:
        msg = f"pointer file {pointer_file.name} names {pointer.artifact.artifact_id}"
        raise ValueError(msg)
    verify_payload(store, pointer.artifact)
    run = load_run(store, pointer)
    if task_clock and run.summary.activation_s is None:
        msg = f"{pointer.artifact.artifact_id} records no activation; a task-clock export needs one"
        raise ValueError(msg)
    scenario, scenario_source = _recorded_scenario(run, scenario_file)
    arrays = run.arrays.arrays
    tau_source = "tau_applied" if "tau_applied" in arrays else "tau_requested"
    channels = sklog_channels(run)
    dof = run.arrays.dof
    channel_meta: dict[str, dict[str, object]] = {
        "q": {"unit": "rad", "columns": [f"q{j + 1}" for j in range(dof)]},
        "dq": {"unit": "rad/s"},
        "tau": {"unit": "N*m", "label": f"playback torque ({tau_source})"},
        "tip": {"unit": "m", "columns": ["x", "y"]},
        "tracking_error": {"unit": "rad"},
    }
    for name in _INTEGER_CHANNELS:
        if name in channels:
            channel_meta[name] = {"label": f"{name} (float copy of an integer channel, visualization only)"}
    log = StateLog(
        build_skeleton(scenario),
        producer=f"arm-rc-ctrl playback export of {pointer.artifact.artifact_id}",
        channel_meta={name: meta for name, meta in channel_meta.items() if name in channels},
        extra=_playback_extra(run, scenario, tau_source, task_clock=task_clock, scenario_source=scenario_source),
    )
    t = np.asarray(arrays["t"], dtype=np.float64)
    shift = float(run.summary.activation_s or 0.0) if task_clock else 0.0
    for k in range(run.arrays.n_samples):
        log.record(float(t[k]) - shift, **{name: values[k] for name, values in channels.items()})
    out.parent.mkdir(parents=True, exist_ok=True)
    # numpy's savez appends ".npz" to other suffixes, so the staging name must already end with it.
    handle, staged_name = tempfile.mkstemp(prefix=out.stem, suffix=".tmp.npz", dir=out.parent)
    os.close(handle)
    staged = Path(staged_name)
    try:
        log.save(staged)
        try:
            os.link(staged, out)  # atomic no-clobber: a file that appeared meanwhile is never replaced
        except FileExistsError:
            msg = f"refusing to overwrite {out}"
            raise FileExistsError(msg) from None
    finally:
        staged.unlink(missing_ok=True)
    return out


def _export_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="run ID with a Git-tracked pointer under data/records/runs")
    group.add_argument("--pointer", type=Path, help="explicit pointer record (TOML)")
    parser.add_argument(
        "--scenario",
        type=Path,
        default=None,
        help="optional scenario TOML to verify: it must match the run's recorded scenario completely",
    )
    parser.add_argument("--records-root", type=Path, default=None)
    parser.add_argument(
        "--task-clock",
        action="store_true",
        help="shift the log's times by the recorded activation so warm-up is negative and activation is 0 s (C5)",
    )
    return parser


def _pointer_from_args(args: argparse.Namespace, store: StorageRoot, scratch: Path) -> Path:
    records_root = repository_root() if args.records_root is None else Path(args.records_root)
    if args.pointer is not None:
        return Path(args.pointer)
    return resolve_pointer(str(args.run), records_root, store=store, scratch=scratch)


def main_export(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``scripts/export_run_sklog.py``."""
    parser = _export_parser("Export one verified run as a playable skelarm StateLog.")
    parser.add_argument("--out", type=Path, required=True, help=f"output {_SUFFIX} (must not exist)")
    args = parser.parse_args(argv)
    scenario_file = None if args.scenario is None else Path(args.scenario)
    store = open_storage()
    with tempfile.TemporaryDirectory(prefix="arm-rc-ctrl-pointer-") as scratch:
        out = export_run_sklog(
            store,
            _pointer_from_args(args, store, Path(scratch)),
            Path(args.out),
            scenario_file=scenario_file,
            task_clock=bool(args.task_clock),
        )
    print(json.dumps({"out": out.name, "player": "python third_party/skelarm/tools/player.py"}))
    return 0


def main_play(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``scripts/play_run.py``: export to a temporary location and invoke the pinned player."""
    parser = _export_parser("Export one verified run to a temporary log and play it with the pinned skelarm player.")
    parser.add_argument("--speed", type=float, default=None, help="initial playback speed multiplier (> 0)")
    parser.add_argument("--show-com", action="store_true", help="overlay the centers of mass")
    parser.add_argument(
        "--panel", action="store_true", help="include the simulator-style side panel in --export frames"
    )
    parser.add_argument("--export", type=Path, default=None, help="headless .mp4/.gif export instead of the GUI")
    parser.add_argument("--fps", type=float, default=None, help="frame rate of the headless export (> 0)")
    args = parser.parse_args(argv)
    if args.speed is not None and not (math.isfinite(args.speed) and args.speed > 0):
        parser.error(f"--speed must be a finite positive number, got {args.speed!r}")
    if args.fps is not None:
        if args.export is None:
            parser.error("--fps only applies to --export")
        if not (math.isfinite(args.fps) and args.fps > 0):
            parser.error(f"--fps must be a finite positive number, got {args.fps!r}")
    store = open_storage()
    target = None if args.export is None else _video_target(Path(args.export))
    with tempfile.TemporaryDirectory(prefix="arm-rc-ctrl-play-") as scratch:
        pointer_file = _pointer_from_args(args, store, Path(scratch))
        scenario_file = None if args.scenario is None else Path(args.scenario)
        log = export_run_sklog(
            store,
            pointer_file,
            Path(scratch) / f"run{_SUFFIX}",
            scenario_file=scenario_file,
            task_clock=bool(args.task_clock),
        )
        staged_video = None if target is None else _stage_video(target)
        try:
            completed = _run_player(_player_command(args, log, staged_video))
            if target is not None and staged_video is not None and completed.returncode == 0:
                _install_video(staged_video, target)
        finally:
            if staged_video is not None:
                staged_video.unlink(missing_ok=True)
    return int(completed.returncode)


def _player_command(args: argparse.Namespace, log: Path, staged_video: Path | None) -> list[str]:
    """The pinned player's invocation for one exported log."""
    command: list[str] = [sys.executable, str(PLAYER), str(log)]
    if args.speed is not None:
        command += ["--speed", str(args.speed)]
    if args.show_com:
        command += ["--show-com"]
    if args.panel:
        command += ["--panel"]
    if staged_video is not None:
        command += ["--export", str(staged_video)]
    if args.fps is not None:
        command += ["--fps", str(args.fps)]
    return command


def _stage_video(target: Path) -> Path:
    """A staging file the player renders into, next to the target so the install link stays on one filesystem."""
    handle, staged_name = tempfile.mkstemp(prefix=target.stem, suffix=target.suffix, dir=target.parent)
    os.close(handle)
    return Path(staged_name)


def _install_video(staged: Path, target: Path) -> None:
    """Install the finished video atomically without clobbering; a file that raced in survives."""
    try:
        os.link(staged, target)
    except FileExistsError:
        msg = f"refusing to overwrite {target}"
        raise FileExistsError(msg) from None


def _video_target(export: Path) -> Path:
    """A safe headless-export target: the right suffix, not existing, not a symbolic link.

    The check is an early courtesy; the atomic no-clobber installation of the
    finished video is what actually protects a file appearing concurrently.
    """
    if export.suffix.lower() not in (".mp4", ".gif"):
        msg = f"--export must end with .mp4 or .gif, got {export.name!r}"
        raise ValueError(msg)
    if export.is_symlink() or export.exists():
        msg = f"refusing to overwrite {export}"
        raise FileExistsError(msg)
    return export.resolve()


def _run_player(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    """Invoke the pinned player; failures propagate through the exit status."""
    return subprocess.run(command, check=False)  # the interpreter is fixed and the tool path pinned


if __name__ == "__main__":  # pragma: no cover - use the scripts/ entry points
    sys.exit(main_export())
