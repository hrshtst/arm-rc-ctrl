# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Manual-demonstration takes: measurements, hold-anchored smoothing, records, import, and derivation.

Implements the dataset contract of ``task_1a_manual_v1`` (plan section 3 and
clarifications I6 and I9): a take is one full recording of the pinned skelarm
trajectory recorder, kept unchanged from its exact reset posture through the
stationary pre-roll, the natural transient, and the final dwell. Nothing is
cropped, warped, or padded. The processed dataset lives on the task grid with a
measured continuous final dwell, a start check against the reset posture, and
descriptive hold/move/dwell annotations that never decide which samples train.
Saving a take is independent of accepting it: every attempt is imported and
retained, and the offline rules decide.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from zipfile import BadZipFile

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt
from skelarm import StateLog

from arm_rc_ctrl.config import load_config, to_mapping
from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.manual_scenario import (
    ManualScenarioConfig,
    load_manual_scenario,
    manual_endpoint_positions,
    manual_joint_limits,
)
from arm_rc_ctrl.data.preprocess import (
    PENDING_RECORD_FILE,
    PROVENANCE_FILE,
    ResamplingSettings,
    finalize_catalog,
    finalize_payload,
    finalize_record,
)
from arm_rc_ctrl.data.raw import read_log_schema_version
from arm_rc_ctrl.data.records import (
    CANONICAL_UNITS,
    PROCESSED_PAYLOAD_FORMAT,
    PROCESSED_PAYLOAD_NAME,
    RAW_PAYLOAD_FORMAT,
    RAW_PAYLOAD_NAME,
    AccessClass,
    ArraySpec,
    ArtifactRecord,
    Normalization,
    Origin,
    Payload,
    Preprocessing,
    Sampling,
    Scenario,
    array_specs,
    catalog_path,
    expected_array_shapes,
    load_catalog,
    load_record,
    make_artifact_id,
    record_path,
    to_toml,
    verify_payload,
    write_catalog,
    write_record,
)
from arm_rc_ctrl.data.resampling import ResamplingConfig, resample
from arm_rc_ctrl.data.samples import ARRAY_NAMES, SAMPLES_SCHEMA_VERSION, SampleSet, save_samples
from arm_rc_ctrl.data.smoothing import SmoothingConfig
from arm_rc_ctrl.data.validate import DatasetValidationError, ValidationSpec, validate_dataset
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    ProvenanceRecord,
    collect_provenance,
    require_clean_for_confirmatory,
    sha256_file,
)
from arm_rc_ctrl.storage import ArtifactUri, StorageRoot
from arm_rc_ctrl.validation import require_finite

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

__all__ = [
    "MANUAL_PHASE_CODES",
    "MANUAL_RAW_UNITS",
    "MANUAL_SCHEMA_VERSION",
    "DuplicatePayloadError",
    "DwellMeasurement",
    "DwellPredicate",
    "HoldAnchoredResult",
    "HoldSettings",
    "ManualDatasetRecord",
    "ManualDeriveConfig",
    "ManualDeriveResult",
    "ManualImportResult",
    "ManualTake",
    "ManualTakeError",
    "ManualTakeRecord",
    "MotionSummary",
    "RawTiming",
    "RecorderAcquisition",
    "RecorderDisplay",
    "SmoothingCheck",
    "StartCheck",
    "TakeAssessment",
    "assess_take",
    "continuous_dwell",
    "derive_manual_dataset",
    "derive_manual_take",
    "import_manual_take",
    "load_manual_derive_config",
    "load_manual_take",
    "register_manual_records",
    "smooth_hold_anchored",
]

MANUAL_SCHEMA_VERSION = 1
MANUAL_PHASE_CODES: dict[str, int] = {"hold": 0, "move": 1, "dwell": 2}
"""Descriptive annotations of a full recording; every sample enters training regardless of its phase."""
MANUAL_RAW_UNITS: dict[str, str] = {"t": "s", "q": "rad", "tip": "m", "nominal_time": "s"}
_REQUIRED_CHANNELS = frozenset({"q", "tip", "nominal_time"})
_ARCHIVE_META = "__meta__"
_ARCHIVE_TIME = "time"
_REQUIRED_CLOCK = "wall-clock"
_SESSION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,31}$")
_TIME_TOLERANCE_S = 1e-9
_TIP_TOLERANCE_M = 1e-9
_TASK_DIM = 2
_MIN_FRAMES = 2
_MIN_DWELL_SAMPLES = 2
_HOLD_ANCHORED_LABEL = "butterworth-zero-phase-hold-anchored"


class ManualTakeError(RuntimeError):
    """A take cannot be read, does not match its record, or is refused by the dataset contract."""


class DuplicatePayloadError(ManualTakeError):
    """The same bytes are already registered as another attempt; a copy never inherits that attempt's record."""

    def __init__(self, existing: ManualTakeRecord, *, session: str, take: int) -> None:
        self.existing = existing
        super().__init__(
            f"payload identical to {existing.artifact.artifact_id} (attempt {existing.take} of session "
            f"{existing.session!r}): attempt {take} of session {session!r} is a byte-identical copy, not a distinct "
            "recording"
        )


_PLANE = 2


# --- continuous final dwell ---------------------------------------------------------------


@dataclass(frozen=True)
class DwellPredicate:
    """The continuous dwell rule: inside the target radius and slower than the speed bound on every joint."""

    tolerance_m: float
    max_velocity_rad_s: float
    min_duration_s: float
    min_samples: int
    """Consecutive grid samples the final run must span (the duration on the grid plus one)."""

    def __post_init__(self) -> None:
        """Validate positivity and the sample count."""
        for name, value in (
            ("tolerance_m", self.tolerance_m),
            ("max_velocity_rad_s", self.max_velocity_rad_s),
            ("min_duration_s", self.min_duration_s),
        ):
            if not (value > 0 and math.isfinite(value)):
                msg = f"{name} must be positive and finite, got {value!r}"
                raise ValueError(msg)
        if self.min_samples < _MIN_DWELL_SAMPLES:
            msg = f"min_samples must be at least {_MIN_DWELL_SAMPLES}, got {self.min_samples}"
            raise ValueError(msg)


@dataclass(frozen=True)
class DwellMeasurement:
    """What the continuous dwell rule measured on one processed take."""

    predicate: DwellPredicate
    ok: bool
    final_samples: int
    """Consecutive qualifying samples ending at the last sample (0 when the last sample does not qualify)."""
    final_duration_s: float
    start_s: float | None
    end_s: float | None
    max_endpoint_error_m: float | None
    max_joint_speed_rad_s: float | None
    longest_samples: int
    """Longest qualifying run anywhere in the take (diagnostic)."""
    earliest_start_s: float | None
    """Start of the first run that reached the required length, if any (diagnostic)."""


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` index ranges of the True runs of ``mask``."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask.tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, int(mask.shape[0])))
    return runs


def continuous_dwell(
    t: NDArray[np.float64],
    tip: NDArray[np.float64],
    dq: NDArray[np.float64],
    *,
    target: NDArray[np.float64],
    predicate: DwellPredicate,
) -> DwellMeasurement:
    """Measure the uninterrupted final dwell of a take on its grid.

    A sample qualifies when its endpoint lies within ``tolerance_m`` of the
    target (closed bound) and every joint speed is at most
    ``max_velocity_rad_s`` (closed bound); any excursion restarts the run. The
    rule holds when the run ending at the last sample spans at least
    ``min_samples`` samples.

    Raises
    ------
    ValueError
        If the arrays disagree in length or shape or are not finite.
    """
    times = np.asarray(t, dtype=np.float64)
    tips = np.asarray(tip, dtype=np.float64)
    speeds = np.asarray(dq, dtype=np.float64)
    goal = np.asarray(target, dtype=np.float64)
    n = times.shape[0]
    if times.ndim != 1 or tips.shape != (n, _PLANE) or speeds.ndim != _TASK_DIM or speeds.shape[0] != n:
        msg = (
            "t, tip, and dq must share their sample count with shape (N,), (N, 2), (N, dof); "
            f"got {times.shape}, {tips.shape}, {speeds.shape}"
        )
        raise ValueError(msg)
    if goal.shape != (_PLANE,):
        msg = f"target must have shape (2,), got {goal.shape}"
        raise ValueError(msg)
    for name, array in (("t", times), ("tip", tips), ("dq", speeds)):
        if not np.all(np.isfinite(array)):
            msg = f"{name} contains non-finite values"
            raise ValueError(msg)
    distance = np.hypot(tips[:, 0] - goal[0], tips[:, 1] - goal[1])
    speed = np.max(np.abs(speeds), axis=1) if speeds.shape[1] else np.zeros(n)
    inside = (distance <= predicate.tolerance_m) & (speed <= predicate.max_velocity_rad_s)
    runs = _runs(inside)
    final = runs[-1] if runs and runs[-1][1] == n else None
    final_samples = final[1] - final[0] if final is not None else 0
    longest = max((end - start for start, end in runs), default=0)
    earliest = next((float(times[start]) for start, end in runs if end - start >= predicate.min_samples), None)
    if final is None:
        return DwellMeasurement(
            predicate=predicate,
            ok=False,
            final_samples=0,
            final_duration_s=0.0,
            start_s=None,
            end_s=None,
            max_endpoint_error_m=None,
            max_joint_speed_rad_s=None,
            longest_samples=longest,
            earliest_start_s=earliest,
        )
    start, end = final
    return DwellMeasurement(
        predicate=predicate,
        ok=final_samples >= predicate.min_samples,
        final_samples=final_samples,
        final_duration_s=float(times[end - 1] - times[start]),
        start_s=float(times[start]),
        end_s=float(times[end - 1]),
        max_endpoint_error_m=float(np.max(distance[start:end])),
        max_joint_speed_rad_s=float(np.max(speed[start:end])),
        longest_samples=longest,
        earliest_start_s=earliest,
    )


# --- hold-anchored zero-phase smoothing (I9) ----------------------------------------------


@dataclass(frozen=True)
class HoldAnchoredResult:
    """The smoothed take, where the filter started, and how far the first sample moved."""

    values: NDArray[np.float64]
    anchored_from: int
    """First index the filter touched; every earlier sample is the recorded reset posture, bitwise."""
    start_shift_rad: float
    """Largest deviation of the processed first sample from the reset posture (0.0 when anchored later)."""
    onset: int | None
    """First recorded sample that departs from the reset posture, or ``None`` for a stationary take."""


def _lowpass_sos(sample_rate_hz: float, config: SmoothingConfig) -> NDArray[np.float64]:
    if not (sample_rate_hz > 0 and math.isfinite(sample_rate_hz)):
        msg = f"sample_rate_hz must be positive and finite, got {sample_rate_hz!r}"
        raise ValueError(msg)
    nyquist = 0.5 * sample_rate_hz
    if config.cutoff_hz >= nyquist:
        msg = f"cutoff_hz {config.cutoff_hz} must be below the Nyquist frequency {nyquist} Hz"
        raise ValueError(msg)
    return cast(
        "NDArray[np.float64]", butter(config.order, config.cutoff_hz, btype="low", fs=sample_rate_hz, output="sos")
    )


def smooth_hold_anchored(
    values: NDArray[np.float64],
    sample_rate_hz: float,
    config: SmoothingConfig,
    *,
    reset: NDArray[np.float64],
    margin_samples: int,
) -> HoldAnchoredResult:
    """Zero-phase smoothing that keeps the recorded reset hold exact.

    The filter acts on the deviation from ``reset`` and starts ``margin_samples``
    before the first departure, extended with zeros (the hold itself) at its
    start and with its last value at its end; every sample before that point is
    returned bitwise. When the hold is shorter than the margin the filter starts
    at the first sample and the resulting shift of the first sample is the
    onset leakage of the zero-phase filter, reported as ``start_shift_rad`` so
    the caller can bound it instead of snapping.

    Raises
    ------
    ValueError
        If the input is not a finite ``(N, k)`` array, ``reset`` does not have
        one entry per column, the first sample is not exactly ``reset``, or the
        filtered segment is too short for the filter.
    """
    data = np.asarray(values, dtype=np.float64)
    anchor = np.asarray(reset, dtype=np.float64)
    if data.ndim != _TASK_DIM or data.shape[0] == 0:
        msg = f"values must have shape (N, k) with N > 0, got {data.shape}"
        raise ValueError(msg)
    if anchor.shape != (data.shape[1],):
        msg = f"reset must have one entry per column ({data.shape[1]}), got shape {anchor.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(data)):
        msg = "values contain non-finite values; smoothing never repairs data"
        raise ValueError(msg)
    if not np.array_equal(data[0], anchor):
        msg = "the first sample must equal the reset posture exactly; the take does not start at rest"
        raise ValueError(msg)
    if margin_samples < 0:
        msg = f"margin_samples must be non-negative, got {margin_samples}"
        raise ValueError(msg)
    departs = np.any(data != anchor, axis=1)
    onset_indices = np.flatnonzero(departs)
    if onset_indices.size == 0 or config.method == "none":
        onset = int(onset_indices[0]) if onset_indices.size else None
        anchored_from = data.shape[0] if onset is None else onset
        return HoldAnchoredResult(data.copy(), anchored_from, 0.0, onset)
    onset = int(onset_indices[0])
    start = max(0, onset - margin_samples)
    sos = _lowpass_sos(sample_rate_hz, config)
    padlen = 3 * (2 * sos.shape[0] + 1)
    deviation = data[start:] - anchor
    if deviation.shape[0] <= padlen:
        msg = f"the filtered segment has {deviation.shape[0]} samples; the filter needs more than {padlen}"
        raise ValueError(msg)
    padded = np.concatenate([np.zeros((padlen, data.shape[1])), deviation, np.tile(deviation[-1], (padlen, 1))], axis=0)
    filtered = cast("NDArray[Any]", sosfiltfilt(sos, padded, axis=0, padtype=None))[padlen:-padlen]
    out = data.copy()
    out[start:] = anchor + filtered
    shift = float(np.max(np.abs(out[0] - anchor)))
    return HoldAnchoredResult(out, start, shift, onset)


# --- recorder metadata and the raw take record -------------------------------------------


def _required(mapping: dict[str, Any], key: str, kind: type | tuple[type, ...], where: str) -> Any:  # noqa: ANN401
    if key not in mapping:
        msg = f"{where} lacks {key!r}"
        raise ManualTakeError(msg)
    value = mapping[key]
    if isinstance(value, bool) and bool not in (kind if isinstance(kind, tuple) else (kind,)):
        msg = f"{where}[{key!r}] must be {kind}, got a bool"
        raise ManualTakeError(msg)
    if not isinstance(value, kind):
        msg = f"{where}[{key!r}] must be {kind}, got {type(value).__name__}"
        raise ManualTakeError(msg)
    return value


@dataclass(frozen=True)
class RecorderAcquisition:
    """The recorder's ``[extra.acquisition]`` table: its clock and the realized tick timing of the take."""

    clock: str
    mode: str
    tick_period_s: float
    sample_period_s: float
    pose_updates_per_sample: int
    display_period_s: float
    ticks: int
    wall_mean_tick_s: float
    wall_max_tick_s: float
    late_ticks: int
    late_tick_factor: float

    def __post_init__(self) -> None:
        """Validate the clock and the counts."""
        if self.clock != _REQUIRED_CLOCK:
            msg = f"acquisition.clock must be {_REQUIRED_CLOCK!r} (actual sample times), got {self.clock!r}"
            raise ValueError(msg)
        require_finite(
            (
                self.tick_period_s,
                self.sample_period_s,
                self.display_period_s,
                self.wall_mean_tick_s,
                self.wall_max_tick_s,
                self.late_tick_factor,
            ),
            "acquisition",
        )
        if self.tick_period_s <= 0 or self.sample_period_s <= 0 or self.pose_updates_per_sample < 1:
            msg = "acquisition periods must be positive and pose_updates_per_sample at least 1"
            raise ValueError(msg)
        if self.ticks < 0 or self.late_ticks < 0 or self.late_ticks > self.ticks:
            msg = f"acquisition tick counts are inconsistent: ticks={self.ticks}, late_ticks={self.late_ticks}"
            raise ValueError(msg)

    @classmethod
    def from_extra(cls, extra: dict[str, Any]) -> RecorderAcquisition:
        """Read the table the pinned recorder writes; missing or mistyped keys raise :class:`ManualTakeError`."""
        where = "extra.acquisition"
        try:
            return cls(
                clock=_required(extra, "clock", str, where),
                mode=_required(extra, "mode", str, where),
                tick_period_s=float(_required(extra, "tick_period_s", (int, float), where)),
                sample_period_s=float(_required(extra, "sample_period_s", (int, float), where)),
                pose_updates_per_sample=_required(extra, "pose_updates_per_sample", int, where),
                display_period_s=float(_required(extra, "display_period_s", (int, float), where)),
                ticks=_required(extra, "ticks", int, where),
                wall_mean_tick_s=float(_required(extra, "wall_mean_tick_s", (int, float), where)),
                wall_max_tick_s=float(_required(extra, "wall_max_tick_s", (int, float), where)),
                late_ticks=_required(extra, "late_ticks", int, where),
                late_tick_factor=float(_required(extra, "late_tick_factor", (int, float), where)),
            )
        except ValueError as exc:
            raise ManualTakeError(str(exc)) from exc


@dataclass(frozen=True)
class RecorderDisplay:
    """The recorder's ``[extra.display]`` table: overlay settings and the saved takes visible during this take."""

    show_tip_trail: bool
    show_past_trails: bool
    past_trails_shown_during_take: bool
    history_takes: tuple[int, ...]
    visible_source_takes: tuple[int, ...]
    visible_source_files: tuple[str, ...]
    policy_json: str
    """The colour/opacity/width policy as canonical JSON (kept verbatim for provenance)."""

    @classmethod
    def from_extra(cls, extra: dict[str, Any]) -> RecorderDisplay:
        """Read the table the pinned recorder writes; missing or mistyped keys raise :class:`ManualTakeError`."""
        where = "extra.display"
        history = _required(extra, "history_takes", list, where)
        visible = _required(extra, "visible_source_takes", list, where)
        files = _required(extra, "visible_source_files", list, where)
        policy = _required(extra, "policy", dict, where)
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in [*history, *visible]):
            msg = f"{where} take lists must hold integers"
            raise ManualTakeError(msg)
        if not all(isinstance(v, str) for v in files):
            msg = f"{where}['visible_source_files'] must hold strings"
            raise ManualTakeError(msg)
        return cls(
            show_tip_trail=_required(extra, "show_tip_trail", bool, where),
            show_past_trails=_required(extra, "show_past_trails", bool, where),
            past_trails_shown_during_take=_required(extra, "past_trails_shown_during_take", bool, where),
            history_takes=tuple(int(v) for v in history),
            visible_source_takes=tuple(int(v) for v in visible),
            visible_source_files=tuple(str(v) for v in files),
            policy_json=json.dumps(policy, sort_keys=True, separators=(",", ":")),
        )


@dataclass(frozen=True)
class ManualTakeRecord:
    """Git-tracked record of one saved take: the untouched recorder log plus what it declares about itself."""

    artifact: ArtifactRecord
    scenario: Scenario
    sampling: Sampling
    session: str
    take: int
    """Attempt number of the take within its session (the recorder's file number), not a dataset ID."""
    source_file: str
    n_frames: int
    duration_s: float
    reset_q: tuple[float, ...]
    """The first logged posture; the assessment compares it with the configured reset posture."""
    acquisition: RecorderAcquisition
    display: RecorderDisplay
    manual_schema_version: int = MANUAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the envelope, the session and take, and the sampling declaration."""
        if self.manual_schema_version != MANUAL_SCHEMA_VERSION:
            msg = f"unsupported manual_schema_version {self.manual_schema_version}; expected {MANUAL_SCHEMA_VERSION}"
            raise ValueError(msg)
        _check_envelope(self.artifact, kind="raw", payload_format=RAW_PAYLOAD_FORMAT, payload_name=RAW_PAYLOAD_NAME)
        _check_session_and_take(self.session, self.take)
        if not self.source_file.strip() or "/" in self.source_file:
            msg = f"source_file must be a bare file name, got {self.source_file!r}"
            raise ValueError(msg)
        self._check_frames()
        if self.sampling.clock != "wall":
            msg = f"sampling.clock must be 'wall' (actual sample times), got {self.sampling.clock!r}"
            raise ValueError(msg)
        if self.sampling.units != MANUAL_RAW_UNITS:
            msg = f"sampling.units must be exactly {MANUAL_RAW_UNITS}, got {self.sampling.units}"
            raise ValueError(msg)

    def _check_frames(self) -> None:
        if self.n_frames < _MIN_FRAMES:
            msg = f"n_frames must be at least {_MIN_FRAMES}, got {self.n_frames}"
            raise ValueError(msg)
        if not (self.duration_s >= 0 and math.isfinite(self.duration_s)):
            msg = f"duration_s must be finite and non-negative, got {self.duration_s!r}"
            raise ValueError(msg)
        if len(self.reset_q) != self.scenario.dof:
            msg = f"reset_q must have dof={self.scenario.dof} entries, got {len(self.reset_q)}"
            raise ValueError(msg)
        require_finite(self.reset_q, "reset_q")


def _check_envelope(artifact: ArtifactRecord, *, kind: str, payload_format: str, payload_name: str) -> None:
    """Shared envelope checks of the manual records: kind, payload format, and the content-addressed URI."""
    if artifact.kind != kind:
        msg = f"a manual {kind} record must have kind {kind!r}, got {artifact.kind!r}"
        raise ValueError(msg)
    if artifact.payload.format != payload_format:
        msg = f"{kind} payload format must be {payload_format!r}, got {artifact.payload.format!r}"
        raise ValueError(msg)
    bucket = "raw" if kind == "raw" else "processed"
    expected_uri = f"armrc://{bucket}/{artifact.artifact_id}/{payload_name}"
    if artifact.payload.uri != expected_uri:
        msg = f"{kind} payload must be stored at {expected_uri}, got {artifact.payload.uri}"
        raise ValueError(msg)


def _check_session_and_take(session: str, take: int) -> None:
    if not _SESSION_RE.match(session):
        msg = f"session must match {_SESSION_RE.pattern}, got {session!r}"
        raise ValueError(msg)
    if take < 1:
        msg = f"take must be a positive attempt number, got {take}"
        raise ValueError(msg)


@dataclass(frozen=True)
class ManualTake:
    """A loaded take: the record, the log, and read-only float64 views of its arrays."""

    record: ManualTakeRecord
    path: Path
    log: StateLog
    times: NDArray[np.float64]
    nominal_times: NDArray[np.float64]
    q: NDArray[np.float64]
    tip: NDArray[np.float64]

    @property
    def n_frames(self) -> int:
        """Number of logged frames."""
        return int(self.times.shape[0])

    @property
    def dof(self) -> int:
        """Number of joints."""
        return int(self.q.shape[1])


def _check_channels(log: StateLog) -> None:
    channels = frozenset(log.channel_names)
    if channels != _REQUIRED_CHANNELS:
        msg = (
            f"a take must log exactly the channels {sorted(_REQUIRED_CHANNELS)}, got {sorted(channels)} "
            "(tip and nominal_time come from the pinned recorder)"
        )
        raise ManualTakeError(msg)
    for name, unit in MANUAL_RAW_UNITS.items():
        if name == "t":
            continue
        declared = log.channel_meta.get(name, {}).get("unit")
        if declared != unit:
            msg = f"channel {name!r} must declare unit {unit!r}, got {declared!r}"
            raise ManualTakeError(msg)
    for name in ("acquisition", "display"):
        if name not in log.extra or not isinstance(log.extra[name], dict):
            msg = f"the log carries no [extra.{name}] table; record takes with the pinned recorder"
            raise ManualTakeError(msg)


def _check_shapes(
    dof: int, times: NDArray[np.float64], q: NDArray[np.float64], tip: NDArray[np.float64], nominal: NDArray[np.float64]
) -> None:
    n = times.shape[0]
    if n < _MIN_FRAMES:
        msg = f"a take needs at least {_MIN_FRAMES} frames, got {n}"
        raise ManualTakeError(msg)
    if q.shape != (n, dof):
        msg = f"q must have shape (N, {dof}) for the configured joints, got {q.shape}"
        raise ManualTakeError(msg)
    if tip.shape != (n, _TASK_DIM) or nominal.shape != (n,):
        msg = f"tip must have shape (N, 2) and nominal_time shape (N,), got {tip.shape} and {nominal.shape}"
        raise ManualTakeError(msg)
    for name, array in (("time", times), ("q", q), ("tip", tip), ("nominal_time", nominal)):
        if not np.all(np.isfinite(array)):
            msg = f"{name} contains non-finite values; a take is never repaired"
            raise ManualTakeError(msg)


def _check_times(times: NDArray[np.float64]) -> None:
    if abs(float(times[0])) > _TIME_TOLERANCE_S:
        msg = f"the take must start at t = 0 (its first frame is the reset state), got {times[0]!r}"
        raise ManualTakeError(msg)
    if not np.all(np.diff(times) > 0):
        msg = "time must be strictly increasing; a repeated or reordered frame is a recorder fault"
        raise ManualTakeError(msg)


def _read_arrays(
    log: StateLog, dof: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Read and structurally validate a recorder log; failures raise :class:`ManualTakeError`."""
    _check_channels(log)
    times = np.array(log.times, dtype=np.float64)
    q = np.array(log.channel("q"), dtype=np.float64)
    tip = np.array(log.channel("tip"), dtype=np.float64)
    nominal = np.array(log.channel("nominal_time"), dtype=np.float64)
    _check_shapes(dof, times, q, tip, nominal)
    _check_times(times)
    return times, q, tip, nominal


def _check_archive(log_file: Path) -> None:
    """Refuse an archive that is not a state log: metadata, a 1-D time axis, and one row per timestamp in every channel.

    The skelarm loader assumes this structure and fails with arbitrary errors
    otherwise, so the check runs first and names what is wrong.

    Raises
    ------
    ManualTakeError
        If the file is not a readable archive or its members disagree.
    """
    try:
        with np.load(log_file, allow_pickle=False) as archive:
            names = set(archive.files)
            if _ARCHIVE_META not in names or _ARCHIVE_TIME not in names:
                msg = f"{log_file.name} lacks the {_ARCHIVE_META!r} or {_ARCHIVE_TIME!r} member of a state log"
                raise ManualTakeError(msg)
            time_axis = archive[_ARCHIVE_TIME]
            if time_axis.ndim != 1:
                msg = f"{log_file.name}: {_ARCHIVE_TIME!r} must be one-dimensional, got shape {time_axis.shape}"
                raise ManualTakeError(msg)
            rows = int(time_axis.shape[0])
            for name in sorted(names - {_ARCHIVE_META, _ARCHIVE_TIME}):
                shape = archive[name].shape
                if not shape or int(shape[0]) != rows:
                    msg = f"{log_file.name}: channel {name!r} has shape {shape} but {_ARCHIVE_TIME!r} has {rows} rows"
                    raise ManualTakeError(msg)
    except (ValueError, OSError, EOFError, BadZipFile) as exc:
        msg = f"cannot read {log_file.name} as a recorder log: {exc}"
        raise ManualTakeError(msg) from exc


def _sampling(times: NDArray[np.float64]) -> Sampling:
    return Sampling(period_s=float(np.median(np.diff(times))), clock="wall", units=dict(MANUAL_RAW_UNITS))


def _scenario_section(scenario_file: Path, config: ManualScenarioConfig, records_root: Path) -> Scenario:
    try:
        relative = scenario_file.resolve().relative_to(records_root.resolve()).as_posix()
    except ValueError as exc:
        msg = f"scenario file {scenario_file} must live inside the repository {records_root}"
        raise ValueError(msg) from exc
    return Scenario(
        config_path=relative,
        config_sha256=sha256_file(scenario_file),
        robot=config.robot.name,
        task=config.name,
        dof=config.dof,
        initial_q=config.task.initial_q,
        target=config.task.target,
    )


def load_manual_take(store: StorageRoot, record: ManualTakeRecord) -> ManualTake:
    """Verify the stored log against its record and load it.

    Raises
    ------
    ManualTakeError
        If the log is structurally invalid or disagrees with the record.
    """
    path = verify_payload(store, record.artifact)
    log = StateLog.load(path)
    times, q, tip, nominal = _read_arrays(log, record.scenario.dof)
    if times.shape[0] != record.n_frames or abs(float(times[-1]) - record.duration_s) > _TIME_TOLERANCE_S:
        msg = (
            f"the log holds {times.shape[0]} frames over {times[-1]:.4f} s but the record says "
            f"{record.n_frames} over {record.duration_s:.4f} s"
        )
        raise ManualTakeError(msg)
    if not np.array_equal(q[0], np.asarray(record.reset_q, dtype=np.float64)):
        msg = "the log's first posture differs from the record's reset_q"
        raise ManualTakeError(msg)
    acquisition = RecorderAcquisition.from_extra(cast("dict[str, Any]", log.extra["acquisition"]))
    if acquisition != record.acquisition:
        msg = "the log's acquisition metadata differs from the record"
        raise ManualTakeError(msg)
    return ManualTake(record, path, log, times, nominal, q, tip)


@dataclass(frozen=True)
class ManualImportResult:
    """Outcome of :func:`import_manual_take`."""

    record: ManualTakeRecord
    record_file: Path
    payload_file: Path
    resumed: bool


def _identical_payload_present(final_dir: Path, digest: str, artifact_id: str) -> bool:
    if not final_dir.exists():
        return False
    existing = final_dir / RAW_PAYLOAD_NAME
    if not existing.is_file() or sha256_file(existing) != digest:
        msg = f"{artifact_id} exists under {final_dir.parent} with a different payload; inspect it manually"
        raise FileExistsError(msg)
    return True


def import_manual_take(
    log_file: Path,
    scenario_file: Path,
    *,
    store: StorageRoot,
    records_root: Path,
    session: str,
    take: int,
    license_label: str,
    access: AccessClass,
    exploratory: bool,
    notes: str = "",
    now: datetime | None = None,
    command: str = "python -m arm_rc_ctrl.data.manual import",
    register: bool = True,
) -> ManualImportResult:
    """Copy a saved take into the store unchanged, verify it against its new record, and register the record.

    Saving is independent of acceptance: a take that the offline rules will
    reject is imported all the same, so every attempt stays retrievable. With
    ``register=False`` the payload is stored and verified but the Git-tracked
    record and catalog entry are left to :func:`register_manual_records`, so a
    batch can register everything once its takes are processed.

    Raises
    ------
    ManualTakeError
        If the file cannot be read as a recorder take or is structurally invalid.
    DuplicatePayloadError
        If the same bytes are already registered as another session or attempt.
    """
    config = load_manual_scenario(scenario_file)
    _check_archive(log_file)
    try:
        log = StateLog.load(log_file)
        payload_schema = read_log_schema_version(log_file)
    except (ValueError, KeyError, IndexError, TypeError, OSError, EOFError, BadZipFile) as exc:
        msg = f"cannot read {log_file.name} as a recorder log: {exc}"
        raise ManualTakeError(msg) from exc
    times, q, _tip, _nominal = _read_arrays(log, config.dof)
    acquisition = RecorderAcquisition.from_extra(cast("dict[str, Any]", log.extra["acquisition"]))
    display = RecorderDisplay.from_extra(cast("dict[str, Any]", log.extra["display"]))
    resolved = {
        "scenario": to_mapping(config),
        "import": {"session": session, "take": take, "log": log_file.name},
    }
    provenance = collect_provenance(resolved, seeds={}, exploratory=exploratory, now=now)
    require_clean_for_confirmatory(provenance)

    staging = store.root / "raw" / f"staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    created = False
    final_dir: Path | None = None
    try:
        payload = staging / RAW_PAYLOAD_NAME
        shutil.copyfile(log_file, payload)
        digest = sha256_file(payload)
        artifact_id = make_artifact_id("raw", provenance.created_at, digest)
        final_dir = store.path(ArtifactUri("raw", (artifact_id,)), mode="write")
        record = ManualTakeRecord(
            artifact=ArtifactRecord(
                artifact_id=artifact_id,
                kind="raw",
                created_at=provenance.created_at,
                license=license_label,
                access=access,
                payload=Payload(
                    uri=f"armrc://raw/{artifact_id}/{RAW_PAYLOAD_NAME}",
                    sha256=digest,
                    size=payload.stat().st_size,
                    format=RAW_PAYLOAD_FORMAT,
                    schema_version=payload_schema,
                ),
                origin=Origin.from_provenance(provenance, command=command),
                notes=notes,
            ),
            scenario=_scenario_section(scenario_file, config, records_root),
            sampling=_sampling(times),
            session=session,
            take=take,
            source_file=log_file.name,
            n_frames=int(times.shape[0]),
            duration_s=float(times[-1]),
            reset_q=tuple(float(v) for v in q[0]),
            acquisition=acquisition,
            display=display,
        )
        resumed = _identical_payload_present(final_dir, digest, artifact_id)
        if resumed:
            shutil.rmtree(staging, ignore_errors=True)
        else:
            staging.rename(final_dir)
            created = True
        load_manual_take(store, record)  # the payload must satisfy its own record before it is registered
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if created and final_dir is not None:
            shutil.rmtree(final_dir, ignore_errors=True)
        raise

    record, record_file = _settle_take_record(record, records_root, resumed=resumed, register=register)
    return ManualImportResult(record, record_file, final_dir / RAW_PAYLOAD_NAME, resumed)


def _settle_take_record(
    record: ManualTakeRecord, records_root: Path, *, resumed: bool, register: bool
) -> tuple[ManualTakeRecord, Path]:
    """Reuse an existing record of the same attempt, refuse another attempt's record, or write a new one."""
    record_file = record_path(records_root, record.artifact)
    if record_file.exists():
        existing = load_record(record_file, ManualTakeRecord)
        if not resumed or existing.artifact.payload != record.artifact.payload:
            msg = f"{record_file} already exists and does not describe this payload; records are immutable"
            raise FileExistsError(msg)
        if (existing.session, existing.take) != (record.session, record.take):
            raise DuplicatePayloadError(existing, session=record.session, take=record.take)
        record = existing
    elif register:
        record_file.parent.mkdir(parents=True, exist_ok=True)
        write_record(record_file, record)
    if register:
        catalog_file = catalog_path(records_root)
        catalog = load_catalog(catalog_file)
        if catalog.find(record.artifact.artifact_id) is None:
            write_catalog(
                catalog_file, catalog.with_record(record.artifact, record_file.relative_to(records_root).as_posix())
            )
    return record, record_file


def register_manual_records(
    records_root: Path, records: Iterable[ManualTakeRecord | ManualDatasetRecord]
) -> list[Path]:
    """Write the Git-tracked record files and catalog entries of records whose payloads are already stored.

    Idempotent: an existing identical record is accepted, a differing one is a
    ``FileExistsError`` (records are immutable). Returns the record files.
    """
    written: list[Path] = []
    for record in records:
        record_file = record_path(records_root, record.artifact)
        if isinstance(record, ManualTakeRecord):
            finalize_record(record_file, record, schema=ManualTakeRecord, resumed=True)
        else:
            finalize_record(record_file, record, schema=ManualDatasetRecord, resumed=True)
        finalize_catalog(records_root, record.artifact, record_file)
        written.append(record_file)
    return written


# --- assessment ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldSettings:
    """Hold-anchored smoothing: how far before the first departure the filter starts, and the start bound."""

    margin_s: float
    max_start_shift_rad: float

    def __post_init__(self) -> None:
        """Validate positivity."""
        if not (self.margin_s >= 0 and math.isfinite(self.margin_s)):
            msg = f"hold.margin_s must be finite and non-negative, got {self.margin_s!r}"
            raise ValueError(msg)
        if not (self.max_start_shift_rad > 0 and math.isfinite(self.max_start_shift_rad)):
            msg = f"hold.max_start_shift_rad must be positive and finite, got {self.max_start_shift_rad!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ManualDeriveConfig:
    """Versioned derivation settings of manual takes (no normalization is fitted, clarification I8)."""

    smoothing: SmoothingConfig
    resampling: ResamplingSettings
    derivatives: DerivativeConfig
    hold: HoldSettings


def load_manual_derive_config(path: Path) -> ManualDeriveConfig:
    """Load and validate a manual derivation TOML."""
    return load_config(path, ManualDeriveConfig)


@dataclass(frozen=True)
class RawTiming:
    """What the raw frames say about the acquisition: counts, intervals, and raw increment speeds."""

    n_frames: int
    duration_s: float
    median_interval_s: float
    max_interval_s: float
    late_frames: int
    """Intervals slower than the recorder's late-tick factor times its period."""
    max_raw_speed_rad_s: tuple[float, ...]
    """Largest ``|dq| / dt`` between consecutive raw frames, per joint, against the actual interval."""


@dataclass(frozen=True)
class StartCheck:
    """The first logged posture against the configured reset posture."""

    expected: tuple[float, ...]
    observed: tuple[float, ...]
    max_deviation_rad: float
    tolerance_rad: float
    ok: bool


@dataclass(frozen=True)
class SmoothingCheck:
    """How the hold-anchored filter was applied and how far it moved the first sample."""

    method: str
    hold_margin_s: float
    anchored_from_s: float
    onset_s: float | None
    start_shift_rad: float
    tolerance_rad: float
    ok: bool


@dataclass(frozen=True)
class MotionSummary:
    """Descriptive timing and dispersion figures of the processed take (never acceptance criteria)."""

    hold_end_s: float
    dwell_start_s: float
    movement_duration_s: float
    time_to_dwell_s: float
    path_length_rad: float
    """Sum over joints of the total variation of the processed joint angles."""
    peak_speed_rad_s: float
    final_q: tuple[float, ...]
    final_tip: tuple[float, ...]
    final_endpoint_error_m: float


@dataclass(frozen=True)
class TakeAssessment:
    """Every measurement of one take and the verdict of the offline rules."""

    accepted: bool
    problems: tuple[str, ...]
    raw_timing: RawTiming
    start: StartCheck
    smoothing: SmoothingCheck | None
    dwell: DwellMeasurement | None
    motion: MotionSummary | None
    samples: SampleSet | None


def _raw_timing(take: ManualTake) -> RawTiming:
    intervals = np.diff(take.times)
    speeds = np.abs(np.diff(take.q, axis=0)) / intervals[:, None]
    late = float(take.record.acquisition.late_tick_factor) * take.record.acquisition.tick_period_s
    return RawTiming(
        n_frames=take.n_frames,
        duration_s=float(take.times[-1]),
        median_interval_s=float(np.median(intervals)),
        max_interval_s=float(np.max(intervals)),
        late_frames=int(np.count_nonzero(intervals > late)),
        max_raw_speed_rad_s=tuple(float(v) for v in np.max(speeds, axis=0)),
    )


def _raw_problems(take: ManualTake, config: ManualScenarioConfig, timing: RawTiming, start: StartCheck) -> list[str]:
    rules = config.acquisition
    problems: list[str] = []
    if take.n_frames < rules.min_frames:
        problems.append(f"too few frames: {take.n_frames} < min_frames {rules.min_frames}")
    if not start.ok:
        problems.append(
            f"first sample deviates from the reset posture by {start.max_deviation_rad:.3e} rad "
            f"(start_tolerance_rad {start.tolerance_rad:.1e})"
        )
    if timing.max_interval_s > rules.max_sample_gap_s:
        at = float(take.times[int(np.argmax(np.diff(take.times)))])
        problems.append(
            f"sample gap of {timing.max_interval_s:.4f} s after t = {at:.3f} s exceeds max_sample_gap_s "
            f"{rules.max_sample_gap_s} (missing interval)"
        )
    for joint, speed in enumerate(timing.max_raw_speed_rad_s):
        if speed > rules.velocity_bound_rad_s:
            problems.append(
                f"raw increment speed {speed:.3f} rad/s on joint {joint} exceeds velocity_bound_rad_s "
                f"{rules.velocity_bound_rad_s} (jump)"
            )
    problems.extend(_integrity_problems(take, config))
    return problems


def _integrity_problems(take: ManualTake, config: ManualScenarioConfig) -> list[str]:
    """Joint limits, the workspace radius, the logged tip against forward kinematics, and the tick clock."""
    problems: list[str] = []
    for joint, link in enumerate(config.robot.links):
        column = take.q[:, joint]
        if np.any(column < link.q_min) or np.any(column > link.q_max):
            problems.append(f"joint {joint} leaves its limits [{link.q_min}, {link.q_max}]")
    radius = np.hypot(take.tip[:, 0], take.tip[:, 1])
    if np.any(radius > config.limits.endpoint_radius + _TIP_TOLERANCE_M):
        problems.append(f"the endpoint leaves the workspace radius {config.limits.endpoint_radius} m")
    fk = manual_endpoint_positions(config, take.q)
    if np.max(np.abs(fk - take.tip)) > _TIP_TOLERANCE_M:
        problems.append("the logged tip differs from the forward kinematics of the logged joints")
    period = config.timing.dt
    nominal = take.nominal_times
    if abs(float(nominal[0])) > _TIME_TOLERANCE_S or np.max(np.abs(np.diff(nominal) - period)) > _TIME_TOLERANCE_S:
        problems.append(f"nominal_time is not the tick clock of period {period} s")
    return problems


def _phase_codes(t: NDArray[np.float64], hold_end_s: float, dwell_start_s: float) -> NDArray[np.int64]:
    phase = np.full(t.shape[0], MANUAL_PHASE_CODES["move"], dtype=np.int64)
    phase[t < hold_end_s - _TIME_TOLERANCE_S] = MANUAL_PHASE_CODES["hold"]
    phase[t >= dwell_start_s - _TIME_TOLERANCE_S] = MANUAL_PHASE_CODES["dwell"]
    return phase


def assess_take(take: ManualTake, config: ManualScenarioConfig, derive: ManualDeriveConfig) -> TakeAssessment:
    """Measure a take against the frozen offline rules and build its processed samples.

    The raw rules (frame count, exact start, gaps, raw increment speeds, limits,
    workspace, tip integrity, the tick clock) come first. A take that starts at
    rest and moves is then resampled onto the task grid, smoothed with the
    hold-anchored filter, differentiated, and measured for its continuous final
    dwell; the processed speeds and limits are checked again on the grid. Every
    problem names the rule and what was measured; the take is accepted only when
    there is none.
    """
    period = config.timing.dt
    reset = np.asarray(config.task.initial_q, dtype=np.float64)
    timing = _raw_timing(take)
    deviation = float(np.max(np.abs(take.q[0] - reset)))
    start = StartCheck(
        expected=tuple(float(v) for v in reset),
        observed=tuple(float(v) for v in take.q[0]),
        max_deviation_rad=deviation,
        tolerance_rad=config.acquisition.start_tolerance_rad,
        ok=deviation <= config.acquisition.start_tolerance_rad,
    )
    problems = _raw_problems(take, config, timing, start)
    onset_indices = np.flatnonzero(np.any(take.q != take.q[0], axis=1))
    if onset_indices.size == 0:
        problems.append("no movement recorded: the take never departs from its first posture")
    if not start.ok or onset_indices.size == 0:
        return TakeAssessment(
            accepted=False,
            problems=tuple(problems),
            raw_timing=timing,
            start=start,
            smoothing=None,
            dwell=None,
            motion=None,
            samples=None,
        )
    hold_end_s = float(take.times[int(onset_indices[0])])

    interpolation = cast("Literal['linear', 'cubic']", derive.resampling.interpolation)
    grid, q_grid = resample(take.times, take.q, ResamplingConfig(period_s=period, interpolation=interpolation))
    grid = grid - grid[0]
    margin_samples = round(derive.hold.margin_s / period)
    try:
        anchored = smooth_hold_anchored(
            q_grid, 1.0 / period, derive.smoothing, reset=reset, margin_samples=margin_samples
        )
    except ValueError as exc:
        problems.append(f"smoothing refused the take: {exc}")
        return TakeAssessment(
            accepted=False,
            problems=tuple(problems),
            raw_timing=timing,
            start=start,
            smoothing=None,
            dwell=None,
            motion=None,
            samples=None,
        )
    smoothing = SmoothingCheck(
        method=_HOLD_ANCHORED_LABEL if derive.smoothing.method != "none" else "none",
        hold_margin_s=derive.hold.margin_s,
        anchored_from_s=float(anchored.anchored_from * period),
        onset_s=None if anchored.onset is None else float(anchored.onset * period),
        start_shift_rad=anchored.start_shift_rad,
        tolerance_rad=derive.hold.max_start_shift_rad,
        ok=anchored.start_shift_rad <= derive.hold.max_start_shift_rad,
    )
    if not smoothing.ok:
        problems.append(
            f"processed start shift {smoothing.start_shift_rad:.3e} rad exceeds max_start_shift_rad "
            f"{smoothing.tolerance_rad:.1e}: the hold of {hold_end_s:.2f} s is shorter than the smoothing margin "
            f"{derive.hold.margin_s} s, so the onset leaks into the reset state"
        )
    q_s = anchored.values
    dq, ddq = differentiate(q_s, period, derive.derivatives)
    tip_s = manual_endpoint_positions(config, q_s)
    dtip, ddtip = differentiate(tip_s, period, derive.derivatives)
    bound = config.acquisition.velocity_bound_rad_s
    for joint in range(config.dof):
        peak = float(np.max(np.abs(dq[:, joint])))
        if peak > bound:
            problems.append(
                f"processed joint speed {peak:.3f} rad/s on joint {joint} exceeds velocity_bound_rad_s {bound}"
            )
    predicate = DwellPredicate(
        tolerance_m=config.task.tolerance,
        max_velocity_rad_s=config.task.dwell_max_velocity,
        min_duration_s=config.task.dwell_min_duration_s,
        min_samples=config.dwell_min_samples,
    )
    dwell = continuous_dwell(grid, tip_s, dq, target=np.asarray(config.task.target), predicate=predicate)
    if not dwell.ok:
        problems.append(
            f"final dwell of {dwell.final_duration_s:.2f} s ({dwell.final_samples} samples) is shorter than the "
            f"required {predicate.min_duration_s} s ({predicate.min_samples} samples inside {predicate.tolerance_m} m "
            f"at joint speeds <= {predicate.max_velocity_rad_s} rad/s)"
        )
        return TakeAssessment(
            accepted=False,
            problems=tuple(problems),
            raw_timing=timing,
            start=start,
            smoothing=smoothing,
            dwell=dwell,
            motion=None,
            samples=None,
        )
    dwell_start_s = cast("float", dwell.start_s)
    if dwell_start_s <= hold_end_s:
        problems.append("the final dwell begins before the take departs from its reset posture")
        return TakeAssessment(
            accepted=False,
            problems=tuple(problems),
            raw_timing=timing,
            start=start,
            smoothing=smoothing,
            dwell=dwell,
            motion=None,
            samples=None,
        )
    distance = np.hypot(tip_s[:, 0] - config.task.target[0], tip_s[:, 1] - config.task.target[1])
    motion = MotionSummary(
        hold_end_s=hold_end_s,
        dwell_start_s=dwell_start_s,
        movement_duration_s=dwell_start_s - hold_end_s,
        time_to_dwell_s=dwell_start_s,
        path_length_rad=float(np.sum(np.abs(np.diff(q_s, axis=0)))),
        peak_speed_rad_s=float(np.max(np.abs(dq))),
        final_q=tuple(float(v) for v in q_s[-1]),
        final_tip=tuple(float(v) for v in tip_s[-1]),
        final_endpoint_error_m=float(distance[-1]),
    )
    samples = SampleSet(
        t=grid,
        q=q_s,
        dq=dq,
        ddq=ddq,
        tip=tip_s,
        dtip=dtip,
        ddtip=ddtip,
        task_code=np.zeros((grid.shape[0], 0), dtype=np.float64),
        phase=_phase_codes(grid, hold_end_s, dwell_start_s),
    )
    spec = ValidationSpec(
        dof=config.dof, task_dim=_TASK_DIM, task_code_dim=0, period_s=period, limits=manual_joint_limits(config)
    )
    try:
        validate_dataset(samples, spec)
    except DatasetValidationError as exc:
        problems.extend(f"dataset validation: {problem}" for problem in exc.problems)
    return TakeAssessment(
        accepted=not problems,
        problems=tuple(problems),
        raw_timing=timing,
        start=start,
        smoothing=smoothing,
        dwell=dwell,
        motion=motion,
        samples=samples,
    )


# --- the processed dataset record and its derivation ---------------------------------------


@dataclass(frozen=True)
class ManualDatasetRecord:
    """Versioned record of one full-recording manual take on the task grid.

    Mirrors the processed-dataset envelope but carries the manual phase
    annotations, the measured final dwell with its predicate, the start check,
    the hold-anchored smoothing check, the raw timing, and the motion summary.
    No normalization is fitted (clarification I8).
    """

    artifact: ArtifactRecord
    scenario: Scenario
    n_samples: int
    dof: int
    task_dim: int
    task_code_dim: int
    units: dict[str, str]
    phases: dict[str, int]
    preprocessing: Preprocessing
    session: str
    take: int
    raw_timing: RawTiming
    start: StartCheck
    smoothing: SmoothingCheck
    dwell: DwellMeasurement
    motion: MotionSummary
    arrays: dict[str, ArraySpec]
    normalization: Normalization | None = None
    manual_schema_version: int = MANUAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the envelope, dimensions, phases, and the recorded checks."""
        if self.manual_schema_version != MANUAL_SCHEMA_VERSION:
            msg = f"unsupported manual_schema_version {self.manual_schema_version}; expected {MANUAL_SCHEMA_VERSION}"
            raise ValueError(msg)
        _check_envelope(
            self.artifact,
            kind="processed",
            payload_format=PROCESSED_PAYLOAD_FORMAT,
            payload_name=PROCESSED_PAYLOAD_NAME,
        )
        sources = self.artifact.origin.sources
        if not sources or not all(s.startswith("raw-") for s in sources):
            msg = f"origin.sources must name the raw take(s) the dataset came from, got {list(sources)}"
            raise ValueError(msg)
        _check_session_and_take(self.session, self.take)
        self._check_dimensions()
        if not (self.dwell.ok and self.start.ok and self.smoothing.ok):
            msg = "a manual dataset record describes an accepted take: dwell, start, and smoothing checks must hold"
            raise ValueError(msg)
        self._check_arrays()

    def _check_dimensions(self) -> None:
        if self.n_samples < _MIN_FRAMES or self.dof < 1 or self.task_dim != _TASK_DIM or self.task_code_dim != 0:
            msg = "n_samples must be >= 2, dof >= 1, task_dim 2, and task_code_dim 0 for a manual take"
            raise ValueError(msg)
        if self.dof != self.scenario.dof:
            msg = f"dof {self.dof} disagrees with scenario.dof {self.scenario.dof}"
            raise ValueError(msg)
        if self.units != CANONICAL_UNITS:
            msg = f"units must be exactly {CANONICAL_UNITS}, got {self.units}"
            raise ValueError(msg)
        if self.phases != MANUAL_PHASE_CODES:
            msg = f"phases must be exactly {MANUAL_PHASE_CODES}, got {self.phases}"
            raise ValueError(msg)

    def _check_arrays(self) -> None:
        expected_shapes = expected_array_shapes(self.n_samples, self.dof, self.task_dim, self.task_code_dim)
        if tuple(self.arrays) != ARRAY_NAMES:
            msg = f"arrays must be exactly {list(ARRAY_NAMES)} in order, got {list(self.arrays)}"
            raise ValueError(msg)
        for name, spec in self.arrays.items():
            if spec.shape != expected_shapes[name]:
                msg = f"arrays[{name!r}].shape must be {expected_shapes[name]}, got {spec.shape}"
                raise ValueError(msg)

    def check_scenario(self, scenario_file: Path) -> None:
        """Ensure ``scenario_file`` is byte-for-byte the one the dataset was derived under."""
        digest = sha256_file(scenario_file)
        if digest != self.scenario.config_sha256:
            msg = (
                f"scenario {scenario_file} (sha256 {digest[:12]}) differs from the record's "
                f"{self.scenario.config_sha256[:12]}"
            )
            raise ValueError(msg)

    def check_samples(self, samples: SampleSet) -> None:
        """Verify a loaded payload against the record: array specs, the exact start, and the annotations."""
        problems: list[str] = []
        specs = array_specs(samples)
        for name, spec in self.arrays.items():
            if specs.get(name) != spec:
                problems.append(f"array {name!r} differs from its recorded spec")
        if float(samples.t[0]) != 0.0:
            problems.append("t does not start at 0")
        reset = np.asarray(self.start.expected, dtype=np.float64)
        if self.smoothing.start_shift_rad == 0.0:
            if not np.array_equal(samples.q[0], reset):
                problems.append("q[0] is not the reset posture although the hold was anchored")
        elif float(np.max(np.abs(samples.q[0] - reset))) > self.smoothing.tolerance_rad:
            problems.append("q[0] deviates from the reset posture beyond the recorded tolerance")
        expected_phase = _phase_codes(samples.t, self.motion.hold_end_s, self.motion.dwell_start_s)
        if not np.array_equal(samples.phase, expected_phase):
            problems.append("phase annotations disagree with the recorded hold end and dwell start")
        if problems:
            msg = "samples do not match the record:\n" + "\n".join(problems)
            raise ValueError(msg)


@dataclass(frozen=True)
class ManualDeriveResult:
    """Outcome of :func:`derive_manual_dataset`."""

    record: ManualDatasetRecord
    samples: SampleSet
    assessment: TakeAssessment
    record_file: Path
    payload_file: Path
    provenance: ProvenanceRecord
    resumed: bool = False


def _preprocessing_section(derive: ManualDeriveConfig, period: float) -> Preprocessing:
    params = dict(derive.smoothing.parameters())
    label = "none"
    if derive.smoothing.method != "none":
        label = _HOLD_ANCHORED_LABEL
        params["hold_margin_s"] = derive.hold.margin_s
        params["max_start_shift_rad"] = derive.hold.max_start_shift_rad
    return Preprocessing(
        resample_period_s=period,
        smoothing=label,
        smoothing_params=params,
        derivative_method=derive.derivatives.label,
        interpolation=derive.resampling.interpolation,
    )


def _build_record(
    raw: ManualTakeRecord,
    config: ManualScenarioConfig,
    derive: ManualDeriveConfig,
    assessment: TakeAssessment,
    samples: SampleSet,
    *,
    artifact_id: str,
    digest: str,
    size: int,
    provenance: ProvenanceRecord,
    command: str,
    license_label: str,
    access: AccessClass,
) -> ManualDatasetRecord:
    return ManualDatasetRecord(
        artifact=ArtifactRecord(
            artifact_id=artifact_id,
            kind="processed",
            created_at=provenance.created_at,
            license=license_label,
            access=access,
            payload=Payload(
                uri=f"armrc://processed/{artifact_id}/{PROCESSED_PAYLOAD_NAME}",
                sha256=digest,
                size=size,
                format=PROCESSED_PAYLOAD_FORMAT,
                schema_version=SAMPLES_SCHEMA_VERSION,
            ),
            origin=Origin.from_provenance(provenance, command=command, sources=(raw.artifact.artifact_id,)),
        ),
        scenario=raw.scenario,
        n_samples=samples.n_samples,
        dof=samples.dof,
        task_dim=samples.task_dim,
        task_code_dim=samples.task_code_dim,
        units=dict(CANONICAL_UNITS),
        phases=dict(MANUAL_PHASE_CODES),
        preprocessing=_preprocessing_section(derive, config.timing.dt),
        session=raw.session,
        take=raw.take,
        raw_timing=assessment.raw_timing,
        start=assessment.start,
        smoothing=cast("SmoothingCheck", assessment.smoothing),
        dwell=cast("DwellMeasurement", assessment.dwell),
        motion=cast("MotionSummary", assessment.motion),
        arrays=array_specs(samples),
    )


def derive_manual_dataset(
    raw_record_file: Path,
    scenario_file: Path,
    config_file: Path,
    *,
    store: StorageRoot,
    records_root: Path,
    exploratory: bool,
    license_override: str | None = None,
    access_override: AccessClass | None = None,
    now: datetime | None = None,
    command: str = "python -m arm_rc_ctrl.data.manual derive",
) -> ManualDeriveResult:
    """Assess the take of a Git-tracked raw record and persist its dataset (see :func:`derive_manual_take`)."""
    return derive_manual_take(
        load_record(raw_record_file, ManualTakeRecord),
        scenario_file,
        config_file,
        store=store,
        records_root=records_root,
        exploratory=exploratory,
        license_override=license_override,
        access_override=access_override,
        now=now,
        command=command,
    )


def derive_manual_take(
    raw: ManualTakeRecord,
    scenario_file: Path,
    config_file: Path,
    *,
    store: StorageRoot,
    records_root: Path,
    exploratory: bool,
    license_override: str | None = None,
    access_override: AccessClass | None = None,
    now: datetime | None = None,
    command: str = "python -m arm_rc_ctrl.data.manual derive",
    register: bool = True,
) -> ManualDeriveResult:
    """Assess a take and, when it is accepted, persist its full-recording dataset, record, and catalog entry.

    With ``register=False`` the payload is finalized in the store but the record
    file and catalog entry are left to :func:`register_manual_records`.

    Raises
    ------
    ManualTakeError
        If the take is rejected by the rules (nothing is written) or does not load.
    ValueError
        If the raw record was made under another scenario file.
    """
    config = load_manual_scenario(scenario_file)
    derive = load_manual_derive_config(config_file)
    digest = sha256_file(scenario_file)
    if raw.scenario.config_sha256 != digest or raw.scenario.dof != config.dof:
        msg = (
            f"raw record {raw.artifact.artifact_id} was recorded under another scenario "
            f"(sha256 {raw.scenario.config_sha256[:12]} vs {digest[:12]})"
        )
        raise ValueError(msg)
    take = load_manual_take(store, raw)
    assessment = assess_take(take, config, derive)
    if not assessment.accepted or assessment.samples is None:
        msg = f"take {raw.take} ({raw.artifact.artifact_id}) is rejected: " + "; ".join(assessment.problems)
        raise ManualTakeError(msg)
    samples = assessment.samples
    license_label = license_override or raw.artifact.license
    access = access_override or raw.artifact.access
    resolved = {
        "scenario": to_mapping(config),
        "preprocessing": to_mapping(derive),
        "raw_artifact": raw.artifact.artifact_id,
        "record": {"license": license_label, "access": access, "command": command},
    }
    source_ref = ArtifactReference(raw.artifact.payload.uri, raw.artifact.payload.sha256, raw.artifact.payload.size)
    provenance = collect_provenance(resolved, seeds={}, artifacts=[source_ref], exploratory=exploratory, now=now)
    require_clean_for_confirmatory(provenance)

    staging = store.root / "processed" / f"staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        payload_file = staging / PROCESSED_PAYLOAD_NAME
        save_samples(payload_file, samples)
        payload_digest = sha256_file(payload_file)
        size = payload_file.stat().st_size

        def rebuild(
            artifact_id: str, origin: ProvenanceRecord, command_line: str, license_label: str, access: AccessClass
        ) -> ManualDatasetRecord:
            return _build_record(
                raw,
                config,
                derive,
                assessment,
                samples,
                artifact_id=artifact_id,
                digest=payload_digest,
                size=size,
                provenance=origin,
                command=command_line,
                license_label=license_label,
                access=access,
            )

        artifact_id = make_artifact_id("processed", provenance.created_at, payload_digest)
        record = rebuild(artifact_id, provenance, command, license_label, access)
        (staging / PROVENANCE_FILE).write_text(provenance.to_json() + "\n", encoding="utf-8")
        (staging / PENDING_RECORD_FILE).write_text(to_toml(record), encoding="utf-8")
        record, provenance, resumed = finalize_payload(
            store,
            staging,
            record,
            provenance,
            rebuild,
            schema=ManualDatasetRecord,
            requested=(license_override, access_override),
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    artifact_id = record.artifact.artifact_id
    final_dir = store.path(ArtifactUri("processed", (artifact_id,)), mode="write")
    record_file = record_path(records_root, record.artifact)
    if register:
        record = finalize_record(record_file, record, schema=ManualDatasetRecord, resumed=resumed)
        finalize_catalog(records_root, record.artifact, record_file)
    return ManualDeriveResult(
        record, samples, assessment, record_file, final_dir / PROCESSED_PAYLOAD_NAME, provenance, resumed
    )
