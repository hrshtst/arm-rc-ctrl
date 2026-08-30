# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Load raw ``skelarm`` demonstrations through the storage resolver.

A raw demonstration is the unchanged ``.sklog.npz`` written by ``skelarm``,
referenced by a :class:`~arm_rc_ctrl.data.records.RawDemonstrationRecord`.
Loading resolves the logical URI under the storage root, verifies size and
SHA-256, parses the archive read-only, and cross-checks the log against the
record. Any failure raises before data is returned; the payload is never
modified or repaired.
"""

from __future__ import annotations

import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from skelarm import StateLog

from arm_rc_ctrl.data.records import RawDemonstrationRecord, verify_payload
from arm_rc_ctrl.storage import StorageRoot

__all__ = [
    "SIMULATED_PERIOD_TOLERANCE_S",
    "WALL_PERIOD_TOLERANCE",
    "RawDemonstration",
    "RawLogError",
    "load_raw_demonstration",
    "read_log_schema_version",
]

SIMULATED_PERIOD_TOLERANCE_S: Final = 1e-9
"""Absolute tolerance on every sample interval when the record declares a simulated clock."""

WALL_PERIOD_TOLERANCE: Final = 0.25
"""Relative tolerance on the median sample interval when the record declares a wall clock."""

_META_MEMBER: Final = "__meta__"
"""Archive member holding skelarm's TOML metadata (the ``.sklog.npz`` file format)."""

_REQUIRED_CHANNELS: Final = ("q",)
"""Only joint positions are mandatory; derivatives are recomputed offline by the preprocessing pipeline."""
_MIN_SAMPLES: Final = 2


class RawLogError(RuntimeError):
    """The payload is not a readable ``skelarm`` log or disagrees with its record."""


def _readonly(array: NDArray[np.float64]) -> NDArray[np.float64]:
    copy = np.array(array, dtype=np.float64, copy=True)
    copy.setflags(write=False)
    return copy


@dataclass(frozen=True)
class RawDemonstration:
    """A verified raw demonstration: record, resolved path, native log, and read-only arrays."""

    record: RawDemonstrationRecord
    path: Path
    log: StateLog
    times: NDArray[np.float64]
    channels: dict[str, NDArray[np.float64]]

    @property
    def n_samples(self) -> int:
        """Number of recorded frames."""
        return int(self.times.shape[0])

    @property
    def dof(self) -> int:
        """Number of joints."""
        return int(self.channels["q"].shape[1])

    @property
    def q(self) -> NDArray[np.float64]:
        """Joint positions ``(N, dof)``."""
        return self.channels["q"]

    @property
    def dq(self) -> NDArray[np.float64] | None:
        """Joint velocities ``(N, dof)`` when the recorder logged them (e.g. skelarm's dynamics mode)."""
        return self.channels.get("dq")


def read_log_schema_version(path: Path) -> int:
    """Return the ``schema_version`` stored in the archive's metadata."""
    with np.load(path, allow_pickle=False) as archive:
        if _META_MEMBER not in archive.files:
            msg = f"{path} has no {_META_MEMBER} member; not a skelarm log"
            raise RawLogError(msg)
        meta = tomllib.loads(str(archive[_META_MEMBER]))
    version = meta.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        msg = f"{path}: metadata schema_version is missing or not an integer"
        raise RawLogError(msg)
    return version


def load_raw_demonstration(store: StorageRoot, record: RawDemonstrationRecord) -> RawDemonstration:
    """Resolve, verify, parse, and cross-check a raw demonstration.

    Raises
    ------
    StorageAccessError
        If the payload is missing or unreadable.
    ArtifactMismatchError
        If its size or digest differs from the record.
    RawLogError
        If it is not a readable ``skelarm`` log or disagrees with the record.
    """
    path = verify_payload(store, record.artifact)
    uri = record.artifact.payload.uri
    try:
        version = read_log_schema_version(path)
        log = StateLog.load(path)
    except (OSError, ValueError, KeyError, TypeError, tomllib.TOMLDecodeError, zipfile.BadZipFile) as exc:
        msg = f"{uri}: not a readable skelarm log: {exc}"
        raise RawLogError(msg) from exc
    problems = _check_against_record(record, version, log)
    if problems:
        msg = f"{uri} disagrees with its record:\n" + "\n".join(problems)
        raise RawLogError(msg)
    times = _readonly(log.times)
    channels = {name: _readonly(log.channel(name)) for name in log.channel_names}
    return RawDemonstration(record=record, path=path, log=log, times=times, channels=channels)


def _check_against_record(record: RawDemonstrationRecord, version: int, log: StateLog) -> list[str]:
    problems: list[str] = []
    expected_version = record.artifact.payload.schema_version
    if version != expected_version:
        problems.append(f"log schema_version {version} != record payload.schema_version {expected_version}")
    problems.extend(_check_channels(record, log))
    if any(name not in log.channel_names for name in _REQUIRED_CHANNELS):
        return problems
    times = log.times
    if times.shape[0] < _MIN_SAMPLES:
        problems.append(f"log has {times.shape[0]} frames; at least {_MIN_SAMPLES} are required")
        return problems
    problems.extend(_check_joints(record, log, times.shape[0]))
    problems.extend(_check_timing(record, times))
    return problems


def _check_channels(record: RawDemonstrationRecord, log: StateLog) -> list[str]:
    problems: list[str] = []
    names = set(log.channel_names)
    problems.extend(f"log has no {name!r} channel" for name in _REQUIRED_CHANNELS if name not in names)
    declared = set(record.sampling.units) - {"t"}
    if declared != names:
        problems.append(f"record units declare channels {sorted(declared)} but the log has {sorted(names)}")
    for name, meta in log.channel_meta.items():
        unit = meta.get("unit")
        if isinstance(unit, str) and record.sampling.units.get(name) != unit:
            problems.append(f"channel {name!r} unit {unit!r} != record unit {record.sampling.units.get(name)!r}")
    return problems


def _check_joints(record: RawDemonstrationRecord, log: StateLog, n: int) -> list[str]:
    problems: list[str] = []
    dof = record.scenario.dof
    for name in (*_REQUIRED_CHANNELS, *(c for c in ("dq", "tau") if c in log.channel_names)):
        array = log.channel(name)
        if array.ndim != 2 or array.shape[1] != dof:  # noqa: PLR2004
            problems.append(f"channel {name!r} has shape {array.shape}; expected ({n}, {dof})")
    try:
        joints = log.build_skeleton().num_joints
    except ValueError:
        joints = None
    if joints is not None and joints != dof:
        problems.append(f"embedded skeleton has {joints} joints; record scenario.dof is {dof}")
    return problems


def _check_timing(record: RawDemonstrationRecord, times: NDArray[np.float64]) -> list[str]:
    problems: list[str] = []
    period = record.sampling.period_s
    if abs(float(times[0])) > SIMULATED_PERIOD_TOLERANCE_S:
        problems.append(f"time must start at 0, got {float(times[0])}")
    steps = np.diff(times)
    if not bool(np.all(steps > 0)):
        problems.append("time is not strictly increasing")
        return problems
    if record.sampling.clock == "simulated":
        worst = float(np.max(np.abs(steps - period)))
        if worst > SIMULATED_PERIOD_TOLERANCE_S:
            problems.append(f"simulated clock: sample interval deviates from {period} s by up to {worst:.3g} s")
    else:
        median = float(np.median(steps))
        if abs(median - period) > WALL_PERIOD_TOLERANCE * period:
            problems.append(f"wall clock: median sample interval {median:.6g} s is not within 25% of {period} s")
    end = float(times[-1])
    if record.duration_s > end + 0.5 * period:
        problems.append(f"record intervals end at {record.duration_s} s but the recording ends at {end:.6g} s")
    return problems
