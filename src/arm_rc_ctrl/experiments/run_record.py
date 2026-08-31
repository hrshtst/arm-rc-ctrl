# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Provenance-complete run records (``docs/PLAN.md`` section 7.4).

A run is stored externally under ``armrc://runs/<run-id>/`` as

- ``arrays.npz`` — measured ``t``, ``q``, ``dq``, ``tip``; desired ``q_desired``,
  raw and filtered desired derivatives; ``tracking_error``; ``tau_requested`` and
  (when the backend exposes it) ``tau_applied``; ``task_code``; ``saturation``;
- ``run.json`` — the :class:`RunSummary`: method, scenario, target, task code,
  disturbances, termination and outcome, per-array specs with digests, seeds,
  and the full :class:`~arm_rc_ctrl.provenance.ProvenanceRecord` (which carries
  the resolved configuration, commits, lock hash, and platform).

The run ID is content-addressed from ``run.json`` (``run-<YYYYMMDD>-<12 hex>``),
so it can only be assigned after the run is complete; the directory is staged
and moved atomically. Git keeps a small :class:`RunPointerRecord`.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.data.arrays import array_digest, load_npz, save_npz
from arm_rc_ctrl.data.records import (
    ArraySpec,
    ArtifactRecord,
    Origin,
    Payload,
    make_artifact_id,
    verify_payload,
)
from arm_rc_ctrl.experiments.termination import Outcome, Termination
from arm_rc_ctrl.provenance import ProvenanceRecord, canonical_json, sha256_file
from arm_rc_ctrl.storage import ArtifactUri, StorageRoot
from arm_rc_ctrl.validation import require_finite

__all__ = [
    "OPTIONAL_ARRAYS",
    "REQUIRED_ARRAYS",
    "RUN_ARRAYS_FILE",
    "RUN_SCHEMA_VERSION",
    "RUN_SUMMARY_FILE",
    "Disturbance",
    "LoadedRun",
    "RunArrays",
    "RunPointerRecord",
    "RunSummary",
    "load_run",
    "write_run",
]

RUN_SCHEMA_VERSION: Final = 1
RUN_ARRAYS_FILE: Final = "arrays.npz"
RUN_SUMMARY_FILE: Final = "run.json"
RUN_PAYLOAD_FORMAT: Final = "run.json"

REQUIRED_ARRAYS: Final[tuple[str, ...]] = (
    "t",
    "q",
    "dq",
    "tip",
    "q_desired",
    "dq_desired_raw",
    "dq_desired",
    "ddq_desired_raw",
    "ddq_desired",
    "tracking_error",
    "tau_requested",
    "task_code",
    "saturation",
)
OPTIONAL_ARRAYS: Final[tuple[str, ...]] = ("tau_applied", "ext_force", "phase", "esn_state_norm")
"""Applied torque, external endpoint force (N) under a disturbance, hold/generate phase, and ESN state norm."""
_JOINT_ARRAYS: Final = (
    "q",
    "dq",
    "q_desired",
    "dq_desired_raw",
    "dq_desired",
    "ddq_desired_raw",
    "ddq_desired",
    "tracking_error",
    "tau_requested",
    "tau_applied",
)
_PLANE: Final = 2
_DIGEST_LENGTH: Final = 64

type RunKind = Literal["simulation", "hardware"]


@dataclass(frozen=True)
class RunArrays:
    """Time series of one run; every array shares the leading sample dimension."""

    arrays: dict[str, NDArray[Any]]

    def __post_init__(self) -> None:
        """Enforce names, dtypes, and consistent shapes; store read-only copies."""
        names = set(self.arrays)
        missing = sorted(set(REQUIRED_ARRAYS) - names)
        unknown = sorted(names - set(REQUIRED_ARRAYS) - set(OPTIONAL_ARRAYS))
        if missing or unknown:
            msg = f"run arrays missing {missing}, unknown {unknown}"
            raise ValueError(msg)
        frozen: dict[str, NDArray[Any]] = {}
        for name, array in self.arrays.items():
            wanted = np.int64 if name in ("saturation", "phase") else np.float64
            if array.dtype != np.dtype(wanted):
                msg = f"{name} must be {np.dtype(wanted)}, got {array.dtype}"
                raise TypeError(msg)
            copy = np.ascontiguousarray(array).copy()
            copy.setflags(write=False)
            frozen[name] = copy
        object.__setattr__(self, "arrays", frozen)
        t = frozen["t"]
        if t.ndim != 1 or t.shape[0] == 0:
            msg = f"t must be a non-empty 1-D array, got shape {t.shape}"
            raise ValueError(msg)
        q = frozen["q"]
        if q.ndim != 2 or q.shape[1] < 1:  # noqa: PLR2004
            msg = f"q must have shape (N, dof) with dof >= 1, got {q.shape}"
            raise ValueError(msg)
        n, dof = t.shape[0], q.shape[1]
        code_dim = frozen["task_code"].shape[1] if frozen["task_code"].ndim == 2 else -1  # noqa: PLR2004
        expected: dict[str, tuple[int, ...]] = dict.fromkeys(_JOINT_ARRAYS, (n, dof))
        expected.update({"tip": (n, _PLANE), "ext_force": (n, _PLANE), "task_code": (n, code_dim), "saturation": (n,)})
        expected.update({"phase": (n,), "esn_state_norm": (n,)})
        for name, array in frozen.items():
            if name != "t" and array.shape != expected[name]:
                msg = f"{name} must have shape {expected[name]}, got {array.shape}"
                raise ValueError(msg)

    @property
    def n_samples(self) -> int:
        """Number of control samples."""
        return int(self.arrays["t"].shape[0])

    @property
    def dof(self) -> int:
        """Number of joints."""
        return int(self.arrays["q"].shape[1])

    def specs(self) -> dict[str, ArraySpec]:
        """Shape, dtype, and digest of every array in canonical order."""
        order = [*REQUIRED_ARRAYS, *(name for name in OPTIONAL_ARRAYS if name in self.arrays)]
        return {
            name: ArraySpec(
                shape=tuple(int(d) for d in self.arrays[name].shape),
                dtype=cast('Literal["float64", "int64"]', str(self.arrays[name].dtype)),
                sha256=array_digest(self.arrays[name]),
            )
            for name in order
        }


@dataclass(frozen=True)
class Disturbance:
    """A disturbance applied during the run."""

    kind: str
    start_s: float
    end_s: float
    parameters: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate kind, timing, and parameters."""
        if not self.kind.strip():
            msg = "disturbance kind must not be empty"
            raise ValueError(msg)
        require_finite((self.start_s, self.end_s), "disturbance timing")
        if self.start_s < 0 or self.end_s < self.start_s:
            msg = f"disturbance timing must satisfy 0 <= start <= end, got {self.start_s}, {self.end_s}"
            raise ValueError(msg)
        require_finite(self.parameters.values(), "disturbance parameters")


@dataclass(frozen=True)
class RunSummary:
    """Everything about a run except the arrays themselves (stored as ``run.json``)."""

    kind: RunKind
    method: str
    scenario: str
    control_period_s: float
    duration_s: float
    target: tuple[float, ...]
    task_code: tuple[float, ...]
    disturbances: tuple[Disturbance, ...]
    termination: Termination
    outcome: Outcome
    arrays: dict[str, ArraySpec]
    arrays_sha256: str
    seeds: dict[str, int]
    provenance: ProvenanceRecord
    notes: str = ""
    schema_version: int = RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate schema, labels, timing, target, arrays, and consistency with the termination."""
        if self.schema_version != RUN_SCHEMA_VERSION:
            msg = f"unsupported run schema_version {self.schema_version}; expected {RUN_SCHEMA_VERSION}"
            raise ValueError(msg)
        if not self.method.strip() or not self.scenario.strip():
            msg = "method and scenario must not be empty"
            raise ValueError(msg)
        self._validate_timing_and_target()
        self._validate_arrays_and_consistency()

    def _validate_timing_and_target(self) -> None:
        if not (self.control_period_s > 0 and np.isfinite(self.control_period_s)):
            msg = f"control_period_s must be positive and finite, got {self.control_period_s!r}"
            raise ValueError(msg)
        if not (self.duration_s > 0 and np.isfinite(self.duration_s)):
            msg = f"duration_s must be positive and finite, got {self.duration_s!r}"
            raise ValueError(msg)
        if len(self.target) != _PLANE:
            msg = f"target must be an [x, y] endpoint, got {list(self.target)}"
            raise ValueError(msg)
        require_finite(self.target, "target")
        require_finite(self.task_code, "task_code")

    def _validate_arrays_and_consistency(self) -> None:
        missing = sorted(set(REQUIRED_ARRAYS) - set(self.arrays))
        if missing:
            msg = f"arrays is missing {missing}"
            raise ValueError(msg)
        if len(self.arrays_sha256) != _DIGEST_LENGTH:
            msg = "arrays_sha256 must be a 64-hex digest"
            raise ValueError(msg)
        if self.outcome.termination != self.termination:
            msg = "outcome.termination must equal termination"
            raise ValueError(msg)
        if self.termination.time_s > self.duration_s + self.control_period_s:
            msg = f"termination at {self.termination.time_s} s lies beyond duration {self.duration_s} s"
            raise ValueError(msg)
        if self.seeds != self.provenance.seeds:
            msg = "seeds must equal provenance.seeds"
            raise ValueError(msg)

    def to_json(self) -> str:
        """Canonical JSON text."""
        return canonical_json(to_mapping(self))

    @classmethod
    def from_json(cls, text: str) -> RunSummary:
        """Strictly rebuild from JSON text."""
        data: object = json.loads(text)
        if not isinstance(data, dict):
            # A document error (the stored text is wrong), not a Python type error.
            msg = "run summary must be a JSON object"
            raise ValueError(msg)  # noqa: TRY004
        return from_mapping(cast("dict[str, object]", data), cls)


@dataclass(frozen=True)
class RunPointerRecord:
    """Git-tracked pointer to an external run with the facts needed without the payload."""

    artifact: ArtifactRecord
    method: str
    scenario: str
    termination_kind: str
    success: bool
    duration_s: float
    n_samples: int
    arrays_sha256: str

    def __post_init__(self) -> None:
        """Validate kind, payload placement, and fields."""
        if self.artifact.kind != "run":
            msg = f"a run pointer must have kind 'run', got {self.artifact.kind!r}"
            raise ValueError(msg)
        expected = f"armrc://runs/{self.artifact.artifact_id}/{RUN_SUMMARY_FILE}"
        if self.artifact.payload.uri != expected:
            msg = f"run payload must be {expected}, got {self.artifact.payload.uri}"
            raise ValueError(msg)
        if self.artifact.payload.format != RUN_PAYLOAD_FORMAT:
            msg = f"run payload format must be {RUN_PAYLOAD_FORMAT!r}"
            raise ValueError(msg)
        if self.n_samples < 1 or not (self.duration_s > 0 and np.isfinite(self.duration_s)):
            msg = "n_samples must be >= 1 and duration_s positive"
            raise ValueError(msg)
        if len(self.arrays_sha256) != _DIGEST_LENGTH:
            msg = "arrays_sha256 must be a 64-hex digest"
            raise ValueError(msg)


@dataclass(frozen=True)
class LoadedRun:
    """A verified run: pointer, summary, arrays, and payload directory."""

    pointer: RunPointerRecord
    summary: RunSummary
    arrays: RunArrays
    directory: Path


def _require_unused(final_dir: Path, artifact_id: str) -> None:
    """Runs are immutable: an existing directory is an error."""
    if final_dir.exists():
        msg = f"{artifact_id} already exists under {final_dir.parent}; runs are immutable"
        raise FileExistsError(msg)


def write_run(
    store: StorageRoot,
    arrays: RunArrays,
    *,
    kind: RunKind,
    method: str,
    scenario: str,
    control_period_s: float,
    duration_s: float,
    target: tuple[float, ...],
    task_code: tuple[float, ...],
    disturbances: tuple[Disturbance, ...],
    termination: Termination,
    outcome: Outcome,
    provenance: ProvenanceRecord,
    license_label: str,
    access: Literal["private", "internal", "public"],
    command: str,
    sources: tuple[str, ...] = (),
    notes: str = "",
) -> tuple[RunPointerRecord, RunSummary, Path]:
    """Persist a run transactionally and return its pointer, summary, and directory."""
    staging = store.root / "runs" / f"staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        arrays_file = staging / RUN_ARRAYS_FILE
        save_npz(arrays_file, arrays.arrays)
        summary = RunSummary(
            kind=kind,
            method=method,
            scenario=scenario,
            control_period_s=control_period_s,
            duration_s=duration_s,
            target=target,
            task_code=task_code,
            disturbances=disturbances,
            termination=termination,
            outcome=outcome,
            arrays=arrays.specs(),
            arrays_sha256=sha256_file(arrays_file),
            seeds=dict(provenance.seeds),
            provenance=provenance,
            notes=notes,
        )
        summary_file = staging / RUN_SUMMARY_FILE
        summary_file.write_text(summary.to_json() + "\n", encoding="utf-8")
        digest = sha256_file(summary_file)
        artifact_id = make_artifact_id("run", provenance.created_at, digest)
        final_dir = store.path(ArtifactUri("runs", (artifact_id,)), mode="write")
        _require_unused(final_dir, artifact_id)
        pointer = RunPointerRecord(
            artifact=ArtifactRecord(
                artifact_id=artifact_id,
                kind="run",
                created_at=provenance.created_at,
                license=license_label,
                access=access,
                payload=Payload(
                    uri=f"armrc://runs/{artifact_id}/{RUN_SUMMARY_FILE}",
                    sha256=digest,
                    size=summary_file.stat().st_size,
                    format=RUN_PAYLOAD_FORMAT,
                    schema_version=RUN_SCHEMA_VERSION,
                ),
                origin=Origin.from_provenance(provenance, command=command, sources=sources, run_id=artifact_id),
                notes=notes,
            ),
            method=method,
            scenario=scenario,
            termination_kind=termination.kind,
            success=outcome.success,
            duration_s=duration_s,
            n_samples=arrays.n_samples,
            arrays_sha256=summary.arrays_sha256,
        )
        staging.rename(final_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return pointer, summary, final_dir


def load_run(store: StorageRoot, pointer: RunPointerRecord) -> LoadedRun:
    """Resolve and verify a run: summary digest, arrays digest, per-array digests, and pointer consistency."""
    summary_file = verify_payload(store, pointer.artifact)
    summary = RunSummary.from_json(summary_file.read_text(encoding="utf-8"))
    directory = summary_file.parent
    arrays_file = directory / RUN_ARRAYS_FILE
    if not arrays_file.is_file():
        msg = f"{arrays_file} is missing"
        raise FileNotFoundError(msg)
    actual = sha256_file(arrays_file)
    if actual != summary.arrays_sha256 or actual != pointer.arrays_sha256:
        msg = f"{arrays_file}: digest {actual[:12]} != recorded {summary.arrays_sha256[:12]}"
        raise ValueError(msg)
    expected_names = tuple(summary.arrays)
    arrays = RunArrays(load_npz(arrays_file, expected_names))
    if arrays.specs() != summary.arrays:
        msg = f"{arrays_file}: array shapes/dtypes/digests differ from run.json"
        raise ValueError(msg)
    if (
        pointer.method != summary.method
        or pointer.scenario != summary.scenario
        or pointer.termination_kind != summary.termination.kind
        or pointer.success != summary.outcome.success
        or pointer.n_samples != arrays.n_samples
    ):
        msg = "pointer record disagrees with run.json"
        raise ValueError(msg)
    return LoadedRun(pointer=pointer, summary=summary, arrays=arrays, directory=directory)
