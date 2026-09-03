# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Recovery dataset contracts (M3R-002; recovery plan section 4.1).

A recovery training episode is derived from a raw demonstration by consuming
its stationary pre-roll as filtering and derivative context and cropping the
derived episode at the confirmed demonstration motion onset. The contracts here
pin that derivation down, next to the untouched M3 schemas:

- :class:`OnsetAnnotation`: a scripted-known (programmed) or human-confirmed
  motion onset with detector configuration, any human adjustment, and the
  raw-payload digest it was decided on.
- :class:`BaselineCheck`: the pre-roll baseline ``q_pre`` validates the crop; a
  material difference from ``q0_ref`` is flagged for review or rejected at
  derivation time, never silently substituted.
- :class:`TaskIntervals` and :class:`CropWindow`: the task clock starts at zero
  at the confirmed onset and contains movement and dwell only.
- :class:`RecoveryDatasetRecord`: a versioned processed-artifact schema whose
  ``q0_ref`` is the first cropped sample, alongside (never replacing) the M3
  :class:`~arm_rc_ctrl.data.records.ProcessedDatasetRecord`.

Task-phase annotation mirrors :mod:`arm_rc_ctrl.data.phases` for the two-phase
task clock, and :func:`recovery_dataset_problems` extends the semantic dataset
validation with the move/dwell-only rules.

The recovery plan lives at
``docs/experiments/task_1a_state_conditioned_recovery/plan.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.config import ConfigError
from arm_rc_ctrl.data.phases import DEFAULT_TOLERANCE_S, PhaseAnnotationError
from arm_rc_ctrl.data.records import (
    CANONICAL_UNITS,
    PROCESSED_PAYLOAD_FORMAT,
    PROCESSED_PAYLOAD_NAME,
    ArrayDtype,
    ArraySpec,
    ArtifactRecord,
    Normalization,
    Preprocessing,
    ProcessedDatasetRecord,
    Scenario,
    array_specs,
    expected_array_shapes,
    is_artifact_id,
    load_record,
)
from arm_rc_ctrl.data.samples import ARRAY_NAMES, PHASE_DWELL, PHASE_MOVE, PHASE_PRIME, SampleSet
from arm_rc_ctrl.data.validate import DatasetValidationError, JointLimits, ValidationSpec, dataset_problems
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.validation import SHA256_HEX_LENGTH, is_hex, require_finite

__all__ = [
    "RECOVERY_DATASET_SCHEMA_VERSION",
    "TASK_PHASE_CODES",
    "BaselineCheck",
    "CropWindow",
    "OnsetAnnotation",
    "RecoveryDatasetRecord",
    "TaskIntervals",
    "annotate_task_phases",
    "check_task_annotation",
    "load_processed_record",
    "recovery_dataset_problems",
    "recovery_validation_spec",
    "task_intervals_from_phases",
    "validate_recovery_dataset",
]

RECOVERY_DATASET_SCHEMA_VERSION: Final = 1
"""Version of the recovery processed-record schema (independent of the M3 record schema)."""

TASK_PHASE_CODES: Final[dict[str, int]] = {"move": PHASE_MOVE, "dwell": PHASE_DWELL}
"""Task-time phase encoding: the canonical move/dwell integers; prime never appears in task time."""

_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_PAIR = 2
_GRID_TOLERANCE_S: Final = 1e-9
"""Slack for onset seconds against ``sample * period`` (accumulated float error, far below one period)."""
_DURATION_TOLERANCE_S: Final = 1e-9
"""Slack for task-clock durations derived from the same boundaries by different arithmetic."""


# --- onset, baseline, crop --------------------------------------------------------------


@dataclass(frozen=True)
class OnsetAnnotation:
    """Where demonstration motion starts in the raw recording, and who decided.

    A ``scripted`` onset is the programmed motion onset: the detector is the
    literal label ``programmed``, no parameters, no adjustment, confirmed by
    the script. A ``human`` onset is proposed by a configured detector from the
    speed profile and confirmed (possibly adjusted) by a human. Both bind to
    the raw artifact and its exact payload digest.
    """

    kind: Literal["scripted", "human"]
    raw_artifact_id: str
    raw_payload_sha256: str
    """SHA-256 of the raw payload the onset was decided on."""
    detector: str
    """``programmed`` for scripted onsets; the detector label for human onsets."""
    detector_params: dict[str, float]
    sampling_period_s: float
    """Raw recording sample period; onset samples live on this grid."""
    proposed_onset_sample: int
    proposed_onset_s: float
    confirmed_onset_sample: int
    confirmed_onset_s: float
    confirmed_by: Literal["script", "human"]

    def __post_init__(self) -> None:
        """Validate the grid, the raw binding, the detector, and the kind-specific rules."""
        if not (self.sampling_period_s > 0 and self.sampling_period_s < float("inf")):
            msg = f"onset.sampling_period_s must be positive and finite, got {self.sampling_period_s!r}"
            raise ValueError(msg)
        if self.proposed_onset_sample < 1 or self.confirmed_onset_sample < 1:
            msg = (
                "onset samples must leave a non-empty pre-roll (sample index >= 1), got "
                f"{self.proposed_onset_sample} and {self.confirmed_onset_sample}"
            )
            raise ValueError(msg)
        for name, sample, value in (
            ("proposed", self.proposed_onset_sample, self.proposed_onset_s),
            ("confirmed", self.confirmed_onset_sample, self.confirmed_onset_s),
        ):
            require_finite((value,), f"onset.{name}_onset_s")
            if abs(value - sample * self.sampling_period_s) > _GRID_TOLERANCE_S:
                msg = f"onset.{name}_onset_s {value!r} does not lie on the raw sample grid at index {sample}"
                raise ValueError(msg)
        if not is_artifact_id(self.raw_artifact_id) or not self.raw_artifact_id.startswith("raw-"):
            msg = f"onset.raw_artifact_id must be a raw artifact ID, got {self.raw_artifact_id!r}"
            raise ValueError(msg)
        if not is_hex(self.raw_payload_sha256, SHA256_HEX_LENGTH):
            msg = f"onset.raw_payload_sha256 must be 64 lowercase hex characters, got {self.raw_payload_sha256!r}"
            raise ValueError(msg)
        if not _LABEL_RE.match(self.detector):
            msg = f"onset.detector must be a lowercase label, got {self.detector!r}"
            raise ValueError(msg)
        require_finite(self.detector_params.values(), "onset.detector_params")
        if self.kind == "scripted":
            self._validate_scripted()
        else:
            self._validate_human()

    def _validate_scripted(self) -> None:
        if self.detector != "programmed":
            msg = f"a scripted onset must use detector 'programmed', got {self.detector!r}"
            raise ValueError(msg)
        if self.detector_params:
            msg = f"a scripted onset takes no detector parameters, got {sorted(self.detector_params)}"
            raise ValueError(msg)
        if self.confirmed_by != "script":
            msg = f"a scripted onset must be confirmed_by 'script', got {self.confirmed_by!r}"
            raise ValueError(msg)
        if self.confirmed_onset_sample != self.proposed_onset_sample:
            msg = (
                "a scripted onset cannot be adjusted, got proposed sample "
                f"{self.proposed_onset_sample} != confirmed sample {self.confirmed_onset_sample}"
            )
            raise ValueError(msg)

    def _validate_human(self) -> None:
        if self.detector == "programmed":
            msg = "a human onset needs a real detector label, not 'programmed'"
            raise ValueError(msg)
        if self.confirmed_by != "human":
            msg = f"a human onset must be confirmed_by 'human', got {self.confirmed_by!r}"
            raise ValueError(msg)

    @property
    def adjustment_s(self) -> float:
        """The human adjustment, confirmed minus proposed (zero for scripted onsets by construction)."""
        return self.confirmed_onset_s - self.proposed_onset_s


@dataclass(frozen=True)
class BaselineCheck:
    """Pre-roll baseline ``q_pre`` versus the task initial posture ``q0_ref``.

    The baseline only proposes and validates the crop. A deviation beyond the
    tolerance must be recorded as ``flagged`` (or the recording rejected before
    a record exists); it never substitutes ``q_pre`` for ``q0_ref``.
    """

    q_pre: tuple[float, ...]
    """Robust stationary pre-roll baseline (rad)."""
    tolerance_rad: float
    max_deviation_rad: float
    """Largest per-joint deviation ``max_j |q_pre_j - q0_ref_j|`` measured at derivation."""
    status: Literal["passed", "flagged"]

    def __post_init__(self) -> None:
        """Validate the baseline vector and require the status to match the measured deviation."""
        if not self.q_pre:
            msg = "baseline.q_pre must not be empty"
            raise ValueError(msg)
        require_finite(self.q_pre, "baseline.q_pre")
        if not (self.tolerance_rad > 0 and self.tolerance_rad < float("inf")):
            msg = f"baseline.tolerance_rad must be positive and finite, got {self.tolerance_rad!r}"
            raise ValueError(msg)
        if not (self.max_deviation_rad >= 0 and self.max_deviation_rad < float("inf")):
            msg = f"baseline.max_deviation_rad must be non-negative and finite, got {self.max_deviation_rad!r}"
            raise ValueError(msg)
        material = self.max_deviation_rad > self.tolerance_rad
        if self.status == "passed" and material:
            msg = (
                f'baseline status "passed" contradicts max_deviation_rad {self.max_deviation_rad!r} > '
                f"tolerance_rad {self.tolerance_rad!r}; a material difference must be flagged for review, "
                "never silently accepted"
            )
            raise ValueError(msg)
        if self.status == "flagged" and not material:
            msg = (
                f'baseline status "flagged" contradicts max_deviation_rad {self.max_deviation_rad!r} <= '
                f"tolerance_rad {self.tolerance_rad!r}"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class TaskIntervals:
    """Contiguous move/dwell boundaries on the task clock (zero at the confirmed onset; no prime)."""

    move: tuple[float, ...]
    dwell: tuple[float, ...]

    def __post_init__(self) -> None:
        """Require two contiguous, increasing ``[start, end]`` pairs with the move starting at zero."""
        for name, pair in (("move", self.move), ("dwell", self.dwell)):
            if len(pair) != _PAIR:
                msg = f"task.{name} must be a [start, end] pair, got {list(pair)}"
                raise ValueError(msg)
            require_finite(pair, f"task.{name}")
            if not pair[0] < pair[1]:
                msg = f"task.{name} must satisfy start < end, got {list(pair)}"
                raise ValueError(msg)
        if self.move[0] != 0.0:
            msg = f"task.move must start at 0.0 (task time), got {self.move[0]}"
            raise ValueError(msg)
        if self.move[1] != self.dwell[0]:
            msg = "task intervals must be contiguous: move end == dwell start"
            raise ValueError(msg)

    @property
    def duration_s(self) -> float:
        """End of the dwell interval on the task clock."""
        return self.dwell[1]


@dataclass(frozen=True)
class CropWindow:
    """What was cut where: the pre-roll consumed as context and the task clock that remains."""

    pre_roll: tuple[float, ...]
    """``[0.0, confirmed onset]`` on the recording clock, used only as filter/derivative context."""
    source_duration_s: float
    """Full duration of the raw recording."""
    task: TaskIntervals

    def __post_init__(self) -> None:
        """Require a pre-roll from zero to before the recording end that leaves room for the task."""
        if len(self.pre_roll) != _PAIR:
            msg = f"crop.pre_roll must be a [start, end] pair, got {list(self.pre_roll)}"
            raise ValueError(msg)
        require_finite(self.pre_roll, "crop.pre_roll")
        if self.pre_roll[0] != 0.0:
            msg = f"crop.pre_roll must start at 0.0, got {self.pre_roll[0]}"
            raise ValueError(msg)
        if not (self.source_duration_s > 0 and self.source_duration_s < float("inf")):
            msg = f"crop.source_duration_s must be positive and finite, got {self.source_duration_s!r}"
            raise ValueError(msg)
        if not self.pre_roll[0] < self.pre_roll[1]:
            msg = f"crop.pre_roll must satisfy start < end, got {list(self.pre_roll)}"
            raise ValueError(msg)
        if self.pre_roll[1] >= self.source_duration_s:
            msg = (
                f"crop.pre_roll must end before the recording ends, got {self.pre_roll[1]!r} >= "
                f"{self.source_duration_s!r}"
            )
            raise ValueError(msg)
        remainder = self.source_duration_s - self.pre_roll[1]
        if self.task.duration_s > remainder + _DURATION_TOLERANCE_S:
            msg = (
                f"crop.task duration {self.task.duration_s!r} cannot exceed the cropped recording length {remainder!r}"
            )
            raise ValueError(msg)


# --- the recovery dataset record --------------------------------------------------------


@dataclass(frozen=True)
class RecoveryDatasetRecord:
    """Versioned record of one cropped recovery training episode.

    Mirrors the M3 :class:`~arm_rc_ctrl.data.records.ProcessedDatasetRecord`
    envelope but requires move/dwell-only phases, binds the onset annotation,
    the pre-roll baseline check, and the crop window, and fixes ``q0_ref`` as
    the first cropped sample. The strict mapper keeps the two schemas mutually
    exclusive on load.
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
    onset: OnsetAnnotation
    baseline: BaselineCheck
    crop: CropWindow
    q0_ref: tuple[float, ...]
    """Task initial posture: the first cropped sample, authoritative and never replaced by ``q_pre``."""
    arrays: dict[str, ArraySpec]
    normalization: Normalization | None = None
    recovery_schema_version: int = RECOVERY_DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the envelope, dimensions, task phases, and every cross-field binding."""
        if self.recovery_schema_version != RECOVERY_DATASET_SCHEMA_VERSION:
            msg = (
                f"unsupported recovery_schema_version {self.recovery_schema_version}; "
                f"expected {RECOVERY_DATASET_SCHEMA_VERSION}"
            )
            raise ValueError(msg)
        if self.artifact.kind != "processed":
            msg = f"a recovery dataset record must have kind 'processed', got {self.artifact.kind!r}"
            raise ValueError(msg)
        if self.artifact.payload.format != PROCESSED_PAYLOAD_FORMAT:
            msg = f"processed payload format must be {PROCESSED_PAYLOAD_FORMAT!r}, got {self.artifact.payload.format!r}"
            raise ValueError(msg)
        expected_uri = f"armrc://processed/{self.artifact.artifact_id}/{PROCESSED_PAYLOAD_NAME}"
        if self.artifact.payload.uri != expected_uri:
            msg = f"processed payload must be stored at {expected_uri}, got {self.artifact.payload.uri}"
            raise ValueError(msg)
        self._validate_sources()
        self._validate_dimensions()
        if self.units != CANONICAL_UNITS:
            msg = f"units must be exactly {CANONICAL_UNITS}, got {self.units}"
            raise ValueError(msg)
        if self.phases != TASK_PHASE_CODES:
            msg = f"phases must be exactly {TASK_PHASE_CODES}, got {self.phases}"
            raise ValueError(msg)
        self._validate_posture()
        self._validate_arrays()
        self._validate_normalization()
        self._validate_crop()

    def _validate_sources(self) -> None:
        sources = self.artifact.origin.sources
        if not sources or any(not source.startswith("raw-") for source in sources):
            msg = "a recovery dataset must name raw source artifacts in artifact.origin.sources"
            raise ValueError(msg)
        if self.onset.raw_artifact_id not in sources:
            msg = f"origin.sources must include onset.raw_artifact_id {self.onset.raw_artifact_id}"
            raise ValueError(msg)

    def _validate_dimensions(self) -> None:
        if self.n_samples < 2 or self.dof < 1 or self.task_dim < 1 or self.task_code_dim < 0:  # noqa: PLR2004
            msg = (
                "n_samples >= 2, dof >= 1, task_dim >= 1, task_code_dim >= 0 required, got "
                f"{self.n_samples}, {self.dof}, {self.task_dim}, {self.task_code_dim}"
            )
            raise ValueError(msg)
        if self.scenario.dof != self.dof:
            msg = f"scenario.dof {self.scenario.dof} != dof {self.dof}"
            raise ValueError(msg)

    def _validate_posture(self) -> None:
        if len(self.q0_ref) != self.dof:
            msg = f"q0_ref must have dof={self.dof} entries, got {len(self.q0_ref)}"
            raise ValueError(msg)
        require_finite(self.q0_ref, "q0_ref")
        if len(self.baseline.q_pre) != self.dof:
            msg = f"baseline.q_pre must have dof={self.dof} entries, got {len(self.baseline.q_pre)}"
            raise ValueError(msg)
        recomputed = float(np.max(np.abs(np.asarray(self.baseline.q_pre) - np.asarray(self.q0_ref))))
        if recomputed != self.baseline.max_deviation_rad:
            msg = (
                f"baseline.max_deviation_rad {self.baseline.max_deviation_rad!r} does not equal the "
                f"recomputed deviation {recomputed!r} of q_pre from q0_ref"
            )
            raise ValueError(msg)

    def _validate_arrays(self) -> None:
        if tuple(self.arrays) != ARRAY_NAMES:
            msg = f"arrays must be exactly {list(ARRAY_NAMES)} in order, got {list(self.arrays)}"
            raise ValueError(msg)
        expected = expected_array_shapes(self.n_samples, self.dof, self.task_dim, self.task_code_dim)
        for name, spec in self.arrays.items():
            if spec.shape != expected[name]:
                msg = f"arrays.{name}.shape must be {list(expected[name])}, got {list(spec.shape)}"
                raise ValueError(msg)
            wanted: ArrayDtype = "int64" if name == "phase" else "float64"
            if spec.dtype != wanted:
                msg = f"arrays.{name}.dtype must be {wanted!r}, got {spec.dtype!r}"
                raise ValueError(msg)

    def _validate_normalization(self) -> None:
        if self.normalization is None:
            return
        widths = {"q": self.dof, "dq": self.dof, "ddq": self.dof, "tip": self.task_dim, "dtip": self.task_dim}
        widths["ddtip"] = self.task_dim
        widths["task_code"] = self.task_code_dim
        for name, stats in self.normalization.channels.items():
            if len(stats.mean) != widths[name]:
                msg = f"normalization.channels.{name} must have {widths[name]} columns, got {len(stats.mean)}"
                raise ValueError(msg)

    def _validate_crop(self) -> None:
        if self.crop.pre_roll[1] != self.onset.confirmed_onset_s:
            msg = (
                f"crop.pre_roll must end at the confirmed onset {self.onset.confirmed_onset_s!r}, "
                f"got {self.crop.pre_roll[1]!r}"
            )
            raise ValueError(msg)
        period = self.preprocessing.resample_period_s
        expected = (self.n_samples - 1) * period
        if abs(self.crop.task.duration_s - expected) > _DURATION_TOLERANCE_S:
            msg = (
                f"crop.task duration {self.crop.task.duration_s!r} does not match {self.n_samples} samples "
                f"at {period!r} s (expected {expected!r})"
            )
            raise ValueError(msg)

    def check_scenario(self, scenario_file: Path) -> None:
        """Fail unless the dataset was derived under the scenario file's current digest."""
        digest = sha256_file(scenario_file)
        if self.scenario.config_sha256 != digest:
            msg = (
                f"dataset {self.artifact.artifact_id} was derived under scenario digest "
                f"{self.scenario.config_sha256[:12]} but {scenario_file} has digest {digest[:12]}"
            )
            raise ValueError(msg)

    def check_samples(self, samples: SampleSet) -> None:
        """Fail unless ``samples`` matches the record exactly, starts at task time zero, and begins at ``q0_ref``."""
        problems: list[str] = []
        actual = array_specs(samples)
        for name, spec in self.arrays.items():
            if actual[name] != spec:
                problems.append(f"{name}: recorded {spec} != actual {actual[name]}")
        t0 = float(samples.t[0])
        if t0 != 0.0:
            problems.append(f"t must start at 0.0 in task time, got {t0!r}")
        q0 = tuple(float(value) for value in samples.q[0])
        if q0 != self.q0_ref:
            problems.append(
                f"q0_ref {self.q0_ref} is not the first cropped sample {q0}; q0_ref is never replaced "
                "(not by the pre-roll baseline q_pre either)"
            )
        try:
            check_task_annotation(samples.t, samples.phase, self.crop.task)
        except (PhaseAnnotationError, ValueError) as exc:
            problems.append(f"crop.task disagrees with the stored phase transition: {exc}")
        if problems:
            msg = "samples do not match the record:\n" + "\n".join(problems)
            raise ValueError(msg)


def load_processed_record(path: Path) -> ProcessedDatasetRecord | RecoveryDatasetRecord:
    """Load a processed-kind record under whichever processed schema it satisfies (M3 first, then recovery).

    Raises
    ------
    ConfigError
        If the file satisfies neither schema (the recovery schema's error is reported).
    """
    try:
        return load_record(path, ProcessedDatasetRecord)
    except ConfigError:
        return load_record(path, RecoveryDatasetRecord)


# --- task-time phase annotation ---------------------------------------------------------


def annotate_task_phases(
    t: NDArray[np.float64], task: TaskIntervals, *, tolerance_s: float = DEFAULT_TOLERANCE_S
) -> NDArray[np.int64]:
    """Assign move or dwell to every task-time sample; prime never appears.

    Boundaries are half-open, ``[start, end)``, except that the dwell end is
    inclusive (within ``tolerance_s``) so the final sample belongs to the dwell.

    Raises
    ------
    PhaseAnnotationError
        If any sample lies outside the task intervals or an interval has no sample.
    """
    times = np.asarray(t, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        msg = f"t must be a non-empty 1-D array, got shape {times.shape}"
        raise PhaseAnnotationError(msg)
    if not np.all(np.isfinite(times)):
        msg = "t contains non-finite values"
        raise PhaseAnnotationError(msg)
    if tolerance_s < 0:
        msg = f"tolerance_s must be non-negative, got {tolerance_s}"
        raise PhaseAnnotationError(msg)
    start, end = task.move[0], task.dwell[1]
    outside = np.argwhere((times < start - tolerance_s) | (times > end + tolerance_s)).ravel()
    if outside.size:
        first = int(outside[0])
        msg = (
            f"{outside.size} sample(s) fall outside [{start}, {end}] s, first at index {first} "
            f"(t={float(times[first])!r})"
        )
        raise PhaseAnnotationError(msg)
    phase = np.full(times.shape, PHASE_DWELL, dtype=np.int64)
    phase[times < task.dwell[0]] = PHASE_MOVE
    empty = [name for name, code in TASK_PHASE_CODES.items() if not np.any(phase == code)]
    if empty:
        msg = f"interval(s) {empty} contain no samples"
        raise PhaseAnnotationError(msg)
    return phase


def check_task_annotation(
    t: NDArray[np.float64],
    phase: NDArray[np.int64],
    task: TaskIntervals,
    *,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> None:
    """Verify an existing ``phase`` array equals the annotation derived from the task intervals."""
    expected = annotate_task_phases(t, task, tolerance_s=tolerance_s)
    given = np.asarray(phase)
    if given.shape != expected.shape:
        msg = f"phase has shape {given.shape}, expected {expected.shape}"
        raise PhaseAnnotationError(msg)
    mismatch = np.argwhere(given != expected).ravel()
    if mismatch.size:
        first = int(mismatch[0])
        msg = f"{mismatch.size} sample(s) carry a phase that disagrees with the task intervals, first at index {first}"
        raise PhaseAnnotationError(msg)


def task_intervals_from_phases(t: NDArray[np.float64], phase: NDArray[np.int64]) -> TaskIntervals:
    """Recover task-interval boundaries from an ordered move/dwell phase array (sample-grid resolution).

    The move starts at ``t[0]`` (which task time requires to be zero), the
    dwell at its first sample, and the dwell ends at ``t[-1]``.
    """
    times = np.asarray(t, dtype=np.float64)
    codes = np.asarray(phase, dtype=np.int64)
    if times.shape != codes.shape or times.ndim != 1 or times.size == 0:
        msg = f"t and phase must be non-empty 1-D arrays of equal length, got {times.shape} and {codes.shape}"
        raise PhaseAnnotationError(msg)
    allowed = np.array(sorted(TASK_PHASE_CODES.values()), dtype=np.int64)
    if not bool(np.isin(codes, allowed).all()):
        bad = sorted(set(np.unique(codes).tolist()) - set(allowed.tolist()))
        msg = f"task phase array must contain only move/dwell codes, got {bad}"
        raise PhaseAnnotationError(msg)
    if not bool(np.all(np.diff(codes) >= 0)):
        msg = "phase must be ordered move -> dwell"
        raise PhaseAnnotationError(msg)
    starts: list[float] = []
    for code in (PHASE_MOVE, PHASE_DWELL):
        where = np.argwhere(codes == code).ravel()
        if where.size == 0:
            name = next(k for k, v in TASK_PHASE_CODES.items() if v == code)
            msg = f"phase array has no {name!r} samples"
            raise PhaseAnnotationError(msg)
        starts.append(float(times[int(where[0])]))
    end = float(times[-1])
    return TaskIntervals(move=(starts[0], starts[1]), dwell=(starts[1], end))


# --- semantic validation ----------------------------------------------------------------


def recovery_validation_spec(record: RecoveryDatasetRecord, limits: JointLimits, **overrides: float) -> ValidationSpec:
    """Derive the semantic specification from a recovery record; prime is never required."""
    return ValidationSpec(
        dof=record.dof,
        task_dim=record.task_dim,
        task_code_dim=record.task_code_dim,
        period_s=record.preprocessing.resample_period_s,
        limits=limits,
        require_all_phases=False,
        **overrides,
    )


def recovery_dataset_problems(samples: SampleSet, spec: ValidationSpec) -> list[str]:
    """Every violation of ``spec`` plus the move/dwell-only task-phase rules (empty when valid)."""
    problems = dataset_problems(samples, replace(spec, require_all_phases=False))
    phase = samples.phase
    prime = np.argwhere(phase == PHASE_PRIME).ravel()
    if prime.size:
        problems.append(
            f"recovery datasets are move/dwell only: {prime.size} prime sample(s), first at sample {int(prime[0])}"
        )
    missing = [name for name, code in TASK_PHASE_CODES.items() if not bool(np.any(phase == code))]
    if missing:
        problems.append(f"missing task phase interval(s): {missing}")
    return problems


def validate_recovery_dataset(samples: SampleSet, spec: ValidationSpec) -> None:
    """Raise :class:`DatasetValidationError` listing every problem; return silently when valid."""
    problems = recovery_dataset_problems(samples, spec)
    if problems:
        raise DatasetValidationError(problems)
