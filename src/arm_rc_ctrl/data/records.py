# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Versioned artifact records (``docs/PLAN.md`` section 7.2).

Git stores one small TOML record per artifact under
``data/records/{raw,processed,runs,models}/<artifact-id>.toml`` plus the
``data/catalog.toml`` index; payloads live under the external storage root and
are referenced only by logical ``armrc://`` URIs, never by machine paths.

Every record is immutable: a correction creates a new artifact ID that names
the superseded one. Artifact IDs are content-addressed,
``<kind>-<YYYYMMDD>-<first 12 hex of the payload SHA-256>``, so two records
for the same bytes created on the same day collide by construction and a
changed payload can never keep an old ID.

Records are plain frozen dataclasses validated in ``__post_init__`` and
(de)serialized through the strict configuration mapper, so unknown keys,
wrong types, and inconsistent values are rejected on load.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import tomli_w

from arm_rc_ctrl.config import load_config, to_mapping
from arm_rc_ctrl.data.samples import ARRAY_NAMES, PHASE_CODES, SAMPLES_FILE, SampleSet
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    ProvenanceRecord,
    artifact_reference,
    sha256_file,
    verify_artifact,
)
from arm_rc_ctrl.storage import ArtifactUri, StorageRoot
from arm_rc_ctrl.validation import (
    COMMIT_HEX_LENGTH,
    MD5_HEX_LENGTH,
    SHA256_HEX_LENGTH,
    is_hex,
    require_finite,
    validate_utc_timestamp,
)

__all__ = [
    "CANONICAL_UNITS",
    "CATALOG_SCHEMA_VERSION",
    "KIND_BUCKETS",
    "KIND_DIRECTORIES",
    "PROCESSED_PAYLOAD_FORMAT",
    "PROCESSED_PAYLOAD_NAME",
    "RAW_PAYLOAD_FORMAT",
    "RAW_PAYLOAD_NAME",
    "RECORD_SCHEMA_VERSION",
    "AccessClass",
    "ArrayDtype",
    "ArraySpec",
    "ArtifactKind",
    "ArtifactRecord",
    "Catalog",
    "CatalogEntry",
    "ChannelStats",
    "DvcPointer",
    "Intervals",
    "Normalization",
    "Origin",
    "Payload",
    "Preprocessing",
    "ProcessedDatasetRecord",
    "RawDemonstrationRecord",
    "Sampling",
    "Scenario",
    "array_specs",
    "catalog_path",
    "expected_array_shapes",
    "is_artifact_id",
    "load_catalog",
    "load_record",
    "make_artifact_id",
    "payload_from_store",
    "record_path",
    "require_relative_posix",
    "to_toml",
    "verify_payload",
    "write_catalog",
    "write_record",
]

RECORD_SCHEMA_VERSION = 1
CATALOG_SCHEMA_VERSION = 1

type ArtifactKind = Literal["raw", "processed", "run", "model"]
type AccessClass = Literal["private", "internal", "public"]

KIND_BUCKETS: dict[str, str] = {"raw": "raw", "processed": "processed", "run": "runs", "model": "models"}
"""Storage bucket that holds payloads of each artifact kind."""

KIND_DIRECTORIES: dict[str, str] = {"raw": "raw", "processed": "processed", "run": "runs", "model": "models"}
"""Record directory below ``data/records/`` for each artifact kind."""

RAW_PAYLOAD_NAME = "demo.sklog.npz"
RAW_PAYLOAD_FORMAT = "sklog.npz"
PROCESSED_PAYLOAD_NAME = SAMPLES_FILE
PROCESSED_PAYLOAD_FORMAT = "samples.npz"

CANONICAL_UNITS: dict[str, str] = {
    "t": "s",
    "q": "rad",
    "dq": "rad/s",
    "ddq": "rad/s^2",
    "tip": "m",
    "dtip": "m/s",
    "ddtip": "m/s^2",
    "task_code": "1",
    "phase": "code",
}
"""Units every processed dataset must declare, exactly (SI, radians, dimensionless codes)."""

_NORMALIZABLE = ("q", "dq", "ddq", "tip", "dtip", "ddtip", "task_code")
_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")

_ARTIFACT_ID_RE = re.compile(r"^(raw|processed|run|model)-(\d{8})-([0-9a-f]{12})$")
_SESSION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,31}$")
_FORMAT_RE = re.compile(r"^[a-z0-9][a-z0-9.+-]*$")
_PAIR = 2


def is_artifact_id(value: str) -> bool:
    """Whether ``value`` follows the ``<kind>-<YYYYMMDD>-<12 hex>`` grammar."""
    return _ARTIFACT_ID_RE.match(value) is not None


def make_artifact_id(kind: ArtifactKind, created_at: str, sha256: str) -> str:
    """Derive the immutable, content-addressed artifact ID."""
    stamp = validate_utc_timestamp(created_at)
    if not is_hex(sha256, SHA256_HEX_LENGTH):
        msg = "sha256 must be 64 lowercase hex characters"
        raise ValueError(msg)
    return f"{kind}-{stamp:%Y%m%d}-{sha256[:12]}"


def require_relative_posix(value: str, field_name: str) -> None:
    """Fail unless ``value`` is a canonical repository-relative POSIX path (no root, drive, dot segments, backslash)."""
    path = PurePosixPath(value)
    canonical = path.as_posix() == value and not value.endswith("/")
    if (
        not value
        or not canonical
        or path.is_absolute()
        or "\\" in value
        or ":" in path.parts[0]
        or ".." in path.parts
        or "." in path.parts
    ):
        msg = (
            f"{field_name} must be a repository-relative POSIX path without '.' or '..' (canonical form), got {value!r}"
        )
        raise ValueError(msg)


_require_relative_posix = require_relative_posix


# --- common record ----------------------------------------------------------------


@dataclass(frozen=True)
class Payload:
    """Identity of the external payload."""

    uri: str
    sha256: str
    size: int
    format: str
    """Media format label, e.g. ``sklog.npz``."""
    schema_version: int
    """Version of the payload's own schema (e.g. skelarm's log schema)."""

    def __post_init__(self) -> None:
        """Validate URI, digest, size, format label, and schema version."""
        ArtifactUri.parse(self.uri)
        if not is_hex(self.sha256, SHA256_HEX_LENGTH):
            msg = f"payload.sha256 must be 64 lowercase hex characters, got {self.sha256!r}"
            raise ValueError(msg)
        if self.size < 0:
            msg = f"payload.size must be non-negative, got {self.size}"
            raise ValueError(msg)
        if not _FORMAT_RE.match(self.format):
            msg = f"payload.format must be a lowercase label such as 'sklog.npz', got {self.format!r}"
            raise ValueError(msg)
        if self.schema_version < 1:
            msg = f"payload.schema_version must be >= 1, got {self.schema_version}"
            raise ValueError(msg)

    @property
    def parsed_uri(self) -> ArtifactUri:
        """The parsed logical URI."""
        return ArtifactUri.parse(self.uri)


@dataclass(frozen=True)
class Origin:
    """How the payload was produced."""

    command: str
    config_sha256: str
    project_commit: str
    project_dirty: bool
    dependency_commits: dict[str, str]
    sources: tuple[str, ...] = ()
    """Artifact IDs this payload was derived from."""
    run_id: str | None = None
    """Run identifier under ``armrc://runs/`` when a run produced the payload."""

    def __post_init__(self) -> None:
        """Validate command, digests, commits, source IDs, and run ID."""
        if not self.command.strip():
            msg = "origin.command must not be empty"
            raise ValueError(msg)
        if not is_hex(self.config_sha256, SHA256_HEX_LENGTH):
            msg = f"origin.config_sha256 must be 64 lowercase hex characters, got {self.config_sha256!r}"
            raise ValueError(msg)
        if not is_hex(self.project_commit, COMMIT_HEX_LENGTH):
            msg = f"origin.project_commit must be a 40-hex commit, got {self.project_commit!r}"
            raise ValueError(msg)
        for name, commit in self.dependency_commits.items():
            if not name or not is_hex(commit, COMMIT_HEX_LENGTH):
                msg = f"origin.dependency_commits[{name!r}] must be a 40-hex commit, got {commit!r}"
                raise ValueError(msg)
        for source in self.sources:
            if not is_artifact_id(source):
                msg = f"origin.sources entry {source!r} is not an artifact ID"
                raise ValueError(msg)
        if len(set(self.sources)) != len(self.sources):
            msg = "origin.sources must be unique"
            raise ValueError(msg)
        if self.run_id is not None:
            ArtifactUri("runs", (self.run_id,))

    @classmethod
    def from_provenance(
        cls,
        record: ProvenanceRecord,
        *,
        command: str,
        sources: tuple[str, ...] = (),
        run_id: str | None = None,
    ) -> Origin:
        """Summarize a full provenance record."""
        commits = {s.name: s.checked_out for s in record.submodules if s.checked_out is not None}
        return cls(
            command=command,
            config_sha256=record.config_sha256,
            project_commit=record.project_commit,
            project_dirty=record.project_dirty,
            dependency_commits=commits,
            sources=sources,
            run_id=run_id,
        )


@dataclass(frozen=True)
class DvcPointer:
    """DVC target and content hash when DVC manages the payload."""

    target: str
    md5: str

    def __post_init__(self) -> None:
        """Validate the repository-relative target and the MD5 hash."""
        _require_relative_posix(self.target, "dvc.target")
        if not is_hex(self.md5, MD5_HEX_LENGTH):
            msg = f"dvc.md5 must be 32 lowercase hex characters, got {self.md5!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ArtifactRecord:
    """Fields every artifact record carries."""

    artifact_id: str
    kind: ArtifactKind
    created_at: str
    license: str
    """SPDX identifier (or ``LicenseRef-...``) governing the payload."""
    access: AccessClass
    payload: Payload
    origin: Origin
    notes: str = ""
    expires_at: str | None = None
    supersedes: str | None = None
    """Artifact ID this record corrects; the old record and payload are never deleted."""
    dvc: DvcPointer | None = None
    schema_version: int = RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate identity, timestamps, license/access, and payload placement."""
        if self.schema_version != RECORD_SCHEMA_VERSION:
            msg = f"unsupported record schema_version {self.schema_version}; expected {RECORD_SCHEMA_VERSION}"
            raise ValueError(msg)
        created = self._validate_identity()
        if self.expires_at is not None and validate_utc_timestamp(self.expires_at, "expires_at") <= created:
            msg = "expires_at must be later than created_at"
            raise ValueError(msg)
        if not self.license.strip():
            msg = "license must not be empty (use an SPDX identifier or LicenseRef-...)"
            raise ValueError(msg)
        self._validate_placement()
        self._validate_supersedes()

    def _validate_identity(self) -> datetime:
        match = _ARTIFACT_ID_RE.match(self.artifact_id)
        if match is None:
            msg = f"artifact_id {self.artifact_id!r} must match <kind>-<YYYYMMDD>-<12 hex>"
            raise ValueError(msg)
        if match.group(1) != self.kind:
            msg = f"artifact_id {self.artifact_id!r} does not carry kind {self.kind!r}"
            raise ValueError(msg)
        created = validate_utc_timestamp(self.created_at)
        if f"{created:%Y%m%d}" != match.group(2):
            msg = f"artifact_id date {match.group(2)} does not match created_at {self.created_at}"
            raise ValueError(msg)
        if self.payload.sha256[:12] != match.group(3):
            msg = "artifact_id digest prefix does not match payload.sha256"
            raise ValueError(msg)
        return created

    def _validate_placement(self) -> None:
        uri = self.payload.parsed_uri
        expected_bucket = KIND_BUCKETS[self.kind]
        if uri.bucket != expected_bucket:
            msg = f"payload.uri bucket {uri.bucket!r} does not match kind {self.kind!r} (expected {expected_bucket!r})"
            raise ValueError(msg)
        if not uri.segments or uri.segments[0] != self.artifact_id:
            msg = f"payload.uri must live under armrc://{uri.bucket}/{self.artifact_id}/, got {self.payload.uri}"
            raise ValueError(msg)

    def _validate_supersedes(self) -> None:
        if self.supersedes is None:
            return
        if not is_artifact_id(self.supersedes) or not self.supersedes.startswith(f"{self.kind}-"):
            msg = f"supersedes {self.supersedes!r} must be an artifact ID of kind {self.kind!r}"
            raise ValueError(msg)
        if self.supersedes == self.artifact_id:
            msg = "a record cannot supersede itself"
            raise ValueError(msg)


# --- raw demonstration ---------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """Robot/task configuration the demonstration was recorded under."""

    config_path: str
    """Repository-relative path of the versioned scenario TOML."""
    config_sha256: str
    robot: str
    task: str
    dof: int
    initial_q: tuple[float, ...]
    """Initial joint posture (rad)."""
    target: tuple[float, ...]
    """Endpoint target in task space (m)."""

    def __post_init__(self) -> None:
        """Validate the path, digest, names, and posture/target dimensions."""
        _require_relative_posix(self.config_path, "scenario.config_path")
        if not self.config_path.endswith(".toml"):
            msg = f"scenario.config_path must point at a TOML file, got {self.config_path!r}"
            raise ValueError(msg)
        if not is_hex(self.config_sha256, SHA256_HEX_LENGTH):
            msg = "scenario.config_sha256 must be 64 lowercase hex characters"
            raise ValueError(msg)
        if not self.robot.strip() or not self.task.strip():
            msg = "scenario.robot and scenario.task must not be empty"
            raise ValueError(msg)
        if self.dof < 1:
            msg = f"scenario.dof must be >= 1, got {self.dof}"
            raise ValueError(msg)
        if len(self.initial_q) != self.dof:
            msg = f"scenario.initial_q must have dof={self.dof} entries, got {len(self.initial_q)}"
            raise ValueError(msg)
        if not self.target:
            msg = "scenario.target must not be empty"
            raise ValueError(msg)
        require_finite(self.initial_q, "scenario.initial_q")
        require_finite(self.target, "scenario.target")


@dataclass(frozen=True)
class Sampling:
    """Sampling clock and units of the recorded channels."""

    period_s: float
    clock: Literal["simulated", "wall"]
    units: dict[str, str]
    """Unit per channel; ``t`` and ``q`` are mandatory, every other logged channel must be declared too."""

    def __post_init__(self) -> None:
        """Validate the period and mandatory units."""
        if not (self.period_s > 0 and self.period_s < float("inf")):
            msg = f"sampling.period_s must be positive and finite, got {self.period_s!r}"
            raise ValueError(msg)
        missing = sorted({"t", "q"} - set(self.units))
        if missing:
            msg = f"sampling.units is missing {missing}"
            raise ValueError(msg)
        for channel, unit in self.units.items():
            if not channel or not unit.strip():
                msg = f"sampling.units[{channel!r}] must be a non-empty unit"
                raise ValueError(msg)


@dataclass(frozen=True)
class Intervals:
    """Contiguous prime/move/dwell boundaries in seconds from the start of the recording."""

    prime: tuple[float, ...]
    move: tuple[float, ...]
    dwell: tuple[float, ...]

    def __post_init__(self) -> None:
        """Require three contiguous, increasing ``[start, end]`` pairs beginning at zero."""
        for name, pair in (("prime", self.prime), ("move", self.move), ("dwell", self.dwell)):
            if len(pair) != _PAIR:
                msg = f"intervals.{name} must be a [start, end] pair, got {list(pair)}"
                raise ValueError(msg)
            require_finite(pair, f"intervals.{name}")
            if not pair[0] < pair[1]:
                msg = f"intervals.{name} must satisfy start < end, got {list(pair)}"
                raise ValueError(msg)
        if self.prime[0] != 0.0:
            msg = f"intervals.prime must start at 0.0, got {self.prime[0]}"
            raise ValueError(msg)
        if self.prime[1] != self.move[0] or self.move[1] != self.dwell[0]:
            msg = "intervals must be contiguous: prime end == move start and move end == dwell start"
            raise ValueError(msg)

    @property
    def duration_s(self) -> float:
        """End of the dwell interval."""
        return self.dwell[1]


@dataclass(frozen=True)
class RawDemonstrationRecord:
    """Record of one raw ``skelarm`` demonstration retained unchanged as ``demo.sklog.npz``."""

    artifact: ArtifactRecord
    scenario: Scenario
    sampling: Sampling
    session: str
    """Pseudonymous teacher or recording-session identifier."""
    intervals: Intervals
    duration_s: float

    def __post_init__(self) -> None:
        """Validate kind, payload placement/format, session ID, and duration."""
        if self.artifact.kind != "raw":
            msg = f"a raw demonstration record must have kind 'raw', got {self.artifact.kind!r}"
            raise ValueError(msg)
        if self.artifact.payload.format != RAW_PAYLOAD_FORMAT:
            msg = f"raw payload format must be {RAW_PAYLOAD_FORMAT!r}, got {self.artifact.payload.format!r}"
            raise ValueError(msg)
        expected_uri = f"armrc://raw/{self.artifact.artifact_id}/{RAW_PAYLOAD_NAME}"
        if self.artifact.payload.uri != expected_uri:
            msg = f"raw payload must be retained at {expected_uri}, got {self.artifact.payload.uri}"
            raise ValueError(msg)
        if not _SESSION_RE.match(self.session):
            msg = f"session must match [a-z0-9][a-z0-9-]{{2,31}}, got {self.session!r}"
            raise ValueError(msg)
        if self.duration_s != self.intervals.duration_s:
            msg = f"duration_s {self.duration_s} must equal the dwell end {self.intervals.duration_s}"
            raise ValueError(msg)


# --- processed dataset ------------------------------------------------------------------

type ArrayDtype = Literal["float64", "int64"]


@dataclass(frozen=True)
class ArraySpec:
    """Shape, dtype, and digest of one array in ``samples.npz``."""

    shape: tuple[int, ...]
    dtype: ArrayDtype
    sha256: str

    def __post_init__(self) -> None:
        """Validate dimensions and digest."""
        if not self.shape or any(d < 0 for d in self.shape):
            msg = f"array shape must be non-empty with non-negative dimensions, got {list(self.shape)}"
            raise ValueError(msg)
        if not is_hex(self.sha256, SHA256_HEX_LENGTH):
            msg = "array sha256 must be 64 lowercase hex characters"
            raise ValueError(msg)


@dataclass(frozen=True)
class Preprocessing:
    """How the raw demonstration became the canonical dataset."""

    resample_period_s: float
    smoothing: str
    """Filter label, e.g. ``butterworth-zero-phase`` or ``none``."""
    smoothing_params: dict[str, float]
    derivative_method: str
    """Label of the offline derivative scheme, e.g. ``central-difference``."""
    interpolation: str = "linear"
    """Interpolation used when resampling onto the control-period grid."""

    def __post_init__(self) -> None:
        """Validate the period, labels, and parameters."""
        if not (self.resample_period_s > 0 and self.resample_period_s < float("inf")):
            msg = f"preprocessing.resample_period_s must be positive and finite, got {self.resample_period_s!r}"
            raise ValueError(msg)
        labels = (
            ("smoothing", self.smoothing),
            ("derivative_method", self.derivative_method),
            ("interpolation", self.interpolation),
        )
        for name, label in labels:
            if not _LABEL_RE.match(label):
                msg = f"preprocessing.{name} must be a lowercase label, got {label!r}"
                raise ValueError(msg)
        require_finite(self.smoothing_params.values(), "preprocessing.smoothing_params")


@dataclass(frozen=True)
class ChannelStats:
    """Per-column normalization statistics of one array."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    replaced_near_zero: tuple[int, ...] = ()
    """Column indices whose near-zero scale was replaced by 1.0."""

    def __post_init__(self) -> None:
        """Validate lengths, finiteness, positive scales, and replacement indices."""
        if not self.mean or len(self.mean) != len(self.scale):
            msg = f"mean and scale must have the same non-zero length, got {len(self.mean)} and {len(self.scale)}"
            raise ValueError(msg)
        require_finite(self.mean, "mean")
        require_finite(self.scale, "scale")
        if any(s <= 0 for s in self.scale):
            msg = "scale entries must be positive"
            raise ValueError(msg)
        if len(set(self.replaced_near_zero)) != len(self.replaced_near_zero) or any(
            not 0 <= i < len(self.scale) for i in self.replaced_near_zero
        ):
            msg = f"replaced_near_zero must be unique column indices below {len(self.scale)}"
            raise ValueError(msg)
        if any(self.scale[i] != 1.0 for i in self.replaced_near_zero):
            msg = "replaced near-zero scales must equal 1.0"
            raise ValueError(msg)


@dataclass(frozen=True)
class Normalization:
    """Training-only normalization statistics and the artifacts they were fitted on."""

    fitted_on: tuple[str, ...]
    channels: dict[str, ChannelStats]

    def __post_init__(self) -> None:
        """Validate the fitting sources and channel names."""
        if not self.fitted_on or any(not is_artifact_id(a) for a in self.fitted_on):
            msg = "normalization.fitted_on must list at least one artifact ID"
            raise ValueError(msg)
        unknown = sorted(set(self.channels) - set(_NORMALIZABLE))
        if unknown:
            msg = f"normalization.channels has unknown arrays {unknown}; allowed {list(_NORMALIZABLE)}"
            raise ValueError(msg)


def expected_array_shapes(n: int, dof: int, task_dim: int, code_dim: int) -> dict[str, tuple[int, ...]]:
    """Canonical array shapes of a processed dataset with the given dimensions."""
    return {
        "t": (n,),
        "q": (n, dof),
        "dq": (n, dof),
        "ddq": (n, dof),
        "tip": (n, task_dim),
        "dtip": (n, task_dim),
        "ddtip": (n, task_dim),
        "task_code": (n, code_dim),
        "phase": (n,),
    }


def array_specs(samples: SampleSet) -> dict[str, ArraySpec]:
    """Describe every array of a sample set for a processed record."""
    shapes, dtypes, digests = samples.shapes(), samples.dtypes(), samples.digests()
    return {
        name: ArraySpec(shape=shapes[name], dtype=cast("ArrayDtype", dtypes[name]), sha256=digests[name])
        for name in ARRAY_NAMES
    }


@dataclass(frozen=True)
class ProcessedDatasetRecord:
    """Record of one canonical processed dataset stored as ``samples.npz``."""

    artifact: ArtifactRecord
    scenario: Scenario
    """Scenario the source demonstration was recorded under (path, digest, robot, task, posture, target)."""
    n_samples: int
    dof: int
    task_dim: int
    task_code_dim: int
    units: dict[str, str]
    phases: dict[str, int]
    """Integer encoding of the ``phase`` array; must equal the documented codes."""
    preprocessing: Preprocessing
    arrays: dict[str, ArraySpec]
    normalization: Normalization | None = None

    def __post_init__(self) -> None:
        """Validate kind, payload placement, sources, dimensions, units, phases, and array specs."""
        if self.artifact.kind != "processed":
            msg = f"a processed dataset record must have kind 'processed', got {self.artifact.kind!r}"
            raise ValueError(msg)
        if self.artifact.payload.format != PROCESSED_PAYLOAD_FORMAT:
            msg = f"processed payload format must be {PROCESSED_PAYLOAD_FORMAT!r}, got {self.artifact.payload.format!r}"
            raise ValueError(msg)
        expected_uri = f"armrc://processed/{self.artifact.artifact_id}/{PROCESSED_PAYLOAD_NAME}"
        if self.artifact.payload.uri != expected_uri:
            msg = f"processed payload must be stored at {expected_uri}, got {self.artifact.payload.uri}"
            raise ValueError(msg)
        sources = self.artifact.origin.sources
        if not sources or any(not source.startswith("raw-") for source in sources):
            msg = "a processed dataset must name at least one raw source artifact in artifact.origin.sources"
            raise ValueError(msg)
        self._validate_dimensions()
        if self.units != CANONICAL_UNITS:
            msg = f"units must be exactly {CANONICAL_UNITS}, got {self.units}"
            raise ValueError(msg)
        if self.phases != PHASE_CODES:
            msg = f"phases must be exactly {PHASE_CODES}, got {self.phases}"
            raise ValueError(msg)
        self._validate_arrays()
        self._validate_normalization()

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
        """Fail unless ``samples`` has exactly the recorded shapes, dtypes, and digests."""
        problems: list[str] = []
        actual = array_specs(samples)
        for name, spec in self.arrays.items():
            if actual[name] != spec:
                problems.append(f"{name}: recorded {spec} != actual {actual[name]}")
        if problems:
            msg = "samples do not match the record:\n" + "\n".join(problems)
            raise ValueError(msg)


# --- catalog ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogEntry:
    """One line of the append-only catalog."""

    artifact_id: str
    kind: ArtifactKind
    record: str
    """Repository-relative path of the record file."""
    uri: str
    sha256: str
    created_at: str

    def __post_init__(self) -> None:
        """Validate identity and references."""
        if not is_artifact_id(self.artifact_id) or not self.artifact_id.startswith(f"{self.kind}-"):
            msg = f"catalog entry {self.artifact_id!r} is not an artifact ID of kind {self.kind!r}"
            raise ValueError(msg)
        _require_relative_posix(self.record, "catalog record path")
        ArtifactUri.parse(self.uri)
        if not is_hex(self.sha256, SHA256_HEX_LENGTH):
            msg = f"catalog entry {self.artifact_id}: sha256 must be 64 lowercase hex characters"
            raise ValueError(msg)
        validate_utc_timestamp(self.created_at)


@dataclass(frozen=True)
class Catalog:
    """Index of every artifact record tracked by Git."""

    schema_version: int = CATALOG_SCHEMA_VERSION
    artifacts: tuple[CatalogEntry, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Require the supported schema and unique IDs and URIs."""
        if self.schema_version != CATALOG_SCHEMA_VERSION:
            msg = f"unsupported catalog schema_version {self.schema_version}; expected {CATALOG_SCHEMA_VERSION}"
            raise ValueError(msg)
        ids = [e.artifact_id for e in self.artifacts]
        if len(set(ids)) != len(ids):
            msg = "catalog artifact IDs must be unique"
            raise ValueError(msg)
        uris = [e.uri for e in self.artifacts]
        if len(set(uris)) != len(uris):
            msg = "catalog payload URIs must be unique"
            raise ValueError(msg)

    def find(self, artifact_id: str) -> CatalogEntry | None:
        """Return the entry for ``artifact_id`` if present."""
        return next((e for e in self.artifacts if e.artifact_id == artifact_id), None)

    def with_record(self, record: ArtifactRecord, record_file: str) -> Catalog:
        """Return a catalog with ``record`` appended; existing entries are never modified."""
        if self.find(record.artifact_id) is not None:
            msg = f"catalog already contains {record.artifact_id}; records are immutable"
            raise ValueError(msg)
        entry = CatalogEntry(
            artifact_id=record.artifact_id,
            kind=record.kind,
            record=record_file,
            uri=record.payload.uri,
            sha256=record.payload.sha256,
            created_at=record.created_at,
        )
        return Catalog(self.schema_version, (*self.artifacts, entry))


# --- files ----------------------------------------------------------------------------


def _strip_none(value: object) -> object:
    if isinstance(value, dict):
        items = cast("dict[str, object]", value)
        return {k: _strip_none(v) for k, v in items.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none(v) for v in cast("list[object]", value)]
    return value


def to_toml(record: object) -> str:
    """Serialize a record or catalog dataclass to TOML text (``None`` fields are omitted)."""
    mapping = cast("dict[str, object]", _strip_none(to_mapping(record)))
    return tomli_w.dumps(mapping)


def load_record[T](path: Path, schema: type[T]) -> T:
    """Load and strictly validate a record of the given schema."""
    return load_config(path, schema)


def _write_atomic(path: Path, text: str) -> None:
    if not path.parent.is_dir():
        msg = f"record directory {path.parent} does not exist"
        raise FileNotFoundError(msg)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_record(path: Path, record: object) -> None:
    """Write an immutable record file; an existing file is never overwritten."""
    if path.exists():
        msg = f"{path} already exists; records are immutable (create a superseding record instead)"
        raise FileExistsError(msg)
    _write_atomic(path, to_toml(record))


def record_path(repo_root: Path, record: ArtifactRecord) -> Path:
    """Canonical location of a record file."""
    return repo_root / "data" / "records" / KIND_DIRECTORIES[record.kind] / f"{record.artifact_id}.toml"


def catalog_path(repo_root: Path) -> Path:
    """Location of the catalog index."""
    return repo_root / "data" / "catalog.toml"


def load_catalog(path: Path) -> Catalog:
    """Load the catalog; a missing file is an empty catalog."""
    if not path.exists():
        return Catalog()
    return load_config(path, Catalog)


def write_catalog(path: Path, catalog: Catalog) -> None:
    """Write the catalog index (append-only semantics are enforced by :meth:`Catalog.with_record`)."""
    existing = load_catalog(path)
    kept = {e.artifact_id: e for e in existing.artifacts}
    for entry in catalog.artifacts:
        if entry.artifact_id in kept and kept[entry.artifact_id] != entry:
            msg = f"catalog entry {entry.artifact_id} would change; entries are immutable"
            raise ValueError(msg)
    missing = set(kept) - {e.artifact_id for e in catalog.artifacts}
    if missing:
        msg = f"catalog entries {sorted(missing)} would be removed; entries are never deleted"
        raise ValueError(msg)
    _write_atomic(path, to_toml(catalog))


def payload_from_store(store: StorageRoot, uri: str, *, format: str, schema_version: int) -> Payload:  # noqa: A002
    """Digest an existing payload under the storage root."""
    ref = artifact_reference(store, uri)
    return Payload(uri=ref.uri, sha256=ref.sha256, size=ref.size, format=format, schema_version=schema_version)


def verify_payload(store: StorageRoot, record: ArtifactRecord) -> Path:
    """Resolve the payload and fail unless its size and digest match the record."""
    payload = record.payload
    return verify_artifact(store, ArtifactReference(payload.uri, payload.sha256, payload.size))
