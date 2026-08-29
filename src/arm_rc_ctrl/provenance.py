# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Provenance records: everything needed to reproduce a result (``docs/PLAN.md`` section 16).

A :class:`ProvenanceRecord` captures the project commit and dirty state, the
submodule revisions, the ``uv.lock`` digest, the fully resolved configuration
and its digest, the logical URIs and digests of input artifacts, all named
seeds, and platform/package versions. Records are plain data: they round-trip
through :func:`arm_rc_ctrl.config.from_mapping` / :func:`to_mapping` with the
same strict validation used for configuration files.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from arm_rc_ctrl import __version__, dependencies
from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.dependencies import BuildIdentity, SubmoduleRevision
from arm_rc_ctrl.repo import git_output, repository_root
from arm_rc_ctrl.storage import ArtifactUri, StorageRoot

__all__ = [
    "SCHEMA_VERSION",
    "THREAD_ENVIRONMENT_VARIABLES",
    "ArtifactMismatchError",
    "ArtifactReference",
    "DirtyWorktreeError",
    "PlatformInfo",
    "ProvenanceRecord",
    "artifact_reference",
    "canonical_json",
    "collect_provenance",
    "config_digest",
    "require_clean_for_confirmatory",
    "sha256_bytes",
    "sha256_file",
    "verify_artifact",
    "worktree_state",
]

THREAD_ENVIRONMENT_VARIABLES: tuple[str, ...] = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
"""Environment variables that influence numerical determinism and are therefore recorded."""

_SHA256_HEX_LENGTH = 64
_COMMIT_HEX_LENGTH = 40
SCHEMA_VERSION = 1


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(c in "0123456789abcdef" for c in value)


class ArtifactMismatchError(RuntimeError):
    """A payload's size or digest differs from its reference."""


class DirtyWorktreeError(RuntimeError):
    """A confirmatory result was requested from a modified or drifted checkout."""


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    """Deterministic JSON: sorted keys, no whitespace, non-ASCII preserved, NaN/Inf rejected."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def config_digest(config: object) -> tuple[str, str]:
    """Return ``(canonical_json, sha256)`` of a resolved configuration (dataclass or mapping)."""
    mapping = to_mapping(config) if dataclasses.is_dataclass(config) and not isinstance(config, type) else config
    text = canonical_json(mapping)
    return text, sha256_bytes(text.encode("utf-8"))


@dataclass(frozen=True)
class ArtifactReference:
    """Logical location and content identity of one payload."""

    uri: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        """Validate the URI form, digest format, and size."""
        ArtifactUri.parse(self.uri)
        if not _is_hex(self.sha256, _SHA256_HEX_LENGTH):
            msg = f"sha256 must be 64 lowercase hex characters, got {self.sha256!r}"
            raise ValueError(msg)
        if self.size < 0:
            msg = f"size must be non-negative, got {self.size}"
            raise ValueError(msg)


@dataclass(frozen=True)
class PlatformInfo:
    """Interpreter, operating system, hardware, and package versions."""

    python: str
    implementation: str
    system: str
    release: str
    machine: str
    packages: dict[str, str]
    thread_environment: dict[str, str]

    def __post_init__(self) -> None:
        """Reject empty identification fields."""
        for name in ("python", "implementation", "system", "machine"):
            if not getattr(self, name):
                msg = f"platform.{name} must not be empty"
                raise ValueError(msg)
        if not self.packages:
            msg = "platform.packages must not be empty"
            raise ValueError(msg)


@dataclass(frozen=True)
class ProvenanceRecord:
    """Reproducibility metadata attached to every generated result."""

    created_at: str
    project_commit: str
    project_dirty: bool
    submodules: tuple[SubmoduleRevision, ...]
    builds: tuple[BuildIdentity, ...]
    """Verified identity of the packages built from submodules (see ``arm_rc_ctrl.dependencies``)."""
    lock_sha256: str
    config_json: str
    config_sha256: str
    artifacts: tuple[ArtifactReference, ...]
    seeds: dict[str, int]
    platform: PlatformInfo
    exploratory: bool = False
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate integrity: formats, canonical configuration and its digest, timestamp, seeds, uniqueness."""
        if self.schema_version != SCHEMA_VERSION:
            msg = f"unsupported schema_version {self.schema_version}; expected {SCHEMA_VERSION}"
            raise ValueError(msg)
        if not _is_hex(self.project_commit, _COMMIT_HEX_LENGTH):
            msg = f"project_commit must be a 40-hex commit, got {self.project_commit!r}"
            raise ValueError(msg)
        for name in ("lock_sha256", "config_sha256"):
            if not _is_hex(getattr(self, name), _SHA256_HEX_LENGTH):
                msg = f"{name} must be 64 lowercase hex characters, got {getattr(self, name)!r}"
                raise ValueError(msg)
        _validate_config_json(self.config_json, self.config_sha256)
        _validate_timestamp(self.created_at)
        _validate_seeds(self.seeds)
        _validate_uniqueness(self)

    def to_mapping(self) -> dict[str, object]:
        """Plain, serializable form."""
        return to_mapping(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> ProvenanceRecord:
        """Strictly validate and rebuild a record from its plain form."""
        return from_mapping(mapping, cls)

    def to_json(self) -> str:
        """Canonical JSON text."""
        return canonical_json(self.to_mapping())

    @classmethod
    def from_json(cls, text: str) -> ProvenanceRecord:
        """Rebuild a record from JSON text."""
        return cls.from_mapping(json.loads(text))

    @property
    def config(self) -> dict[str, object]:
        """The resolved configuration as a mapping."""
        return json.loads(self.config_json)

    @property
    def is_clean(self) -> bool:
        """Whether the checkout is unmodified, submodules are on their pins, and builds are verifiable."""
        return (
            not self.project_dirty
            and all((s.dirty is not True) and (s.checked_out is None or s.matches_pin) for s in self.submodules)
            and all(not b.editable and not b.source_dirty for b in self.builds)
        )


def _validate_config_json(config_json: str, config_sha256: str) -> None:
    """Require canonical JSON of an object whose digest is ``config_sha256``."""
    try:
        parsed: object = json.loads(config_json)
    except json.JSONDecodeError as exc:
        msg = f"config_json is not valid JSON: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(parsed, dict):
        # A document error (the stored text is wrong), not a Python type error.
        msg = "config_json must encode a JSON object"
        raise ValueError(msg)  # noqa: TRY004
    if canonical_json(cast("dict[str, object]", parsed)) != config_json:
        msg = "config_json is not in canonical form (sorted keys, no whitespace)"
        raise ValueError(msg)
    if sha256_bytes(config_json.encode("utf-8")) != config_sha256:
        msg = "config_sha256 does not match config_json"
        raise ValueError(msg)


def _validate_seeds(seeds: Mapping[str, int]) -> None:
    """Require exact non-negative integers (no bool, no float)."""
    for name, seed in seeds.items():
        if type(seed) is not int or seed < 0:
            msg = f"seed {name!r} must be a non-negative integer, got {seed!r}"
            raise ValueError(msg)


def _validate_uniqueness(record: ProvenanceRecord) -> None:
    """Artifacts, submodules, and builds must be unique, and every build must name a submodule."""
    uris = [a.uri for a in record.artifacts]
    if len(set(uris)) != len(uris):
        msg = "artifacts must have unique URIs"
        raise ValueError(msg)
    for label, names in (
        ("submodules", [s.name for s in record.submodules]),
        ("builds", [b.name for b in record.builds]),
    ):
        if len(set(names)) != len(names):
            msg = f"{label} must have unique names"
            raise ValueError(msg)
    submodule_names = {s.name for s in record.submodules}
    for build in record.builds:
        if build.name not in submodule_names:
            msg = f"build {build.name!r} has no matching submodule entry"
            raise ValueError(msg)


def _validate_timestamp(value: str) -> None:
    """Require an ISO 8601 timestamp that is timezone-aware, in UTC, at second precision."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        msg = f"created_at is not an ISO 8601 timestamp: {value!r}"
        raise ValueError(msg) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        msg = f"created_at must be timezone-aware in UTC, got {value!r}"
        raise ValueError(msg)
    if parsed.isoformat(timespec="seconds") != value:
        msg = f"created_at must have second precision with a +00:00 offset, got {value!r}"
        raise ValueError(msg)


def worktree_state(root: Path) -> tuple[str, bool]:
    """Return ``(HEAD commit, dirty)`` for the repository at ``root``.

    ``dirty`` is true when ``git status --porcelain`` reports anything:
    modified or staged tracked files, untracked files that are not ignored,
    or submodules with new commits, modified content, or untracked content.
    """
    commit = git_output("rev-parse", "HEAD", cwd=root)
    dirty = bool(git_output("status", "--porcelain", cwd=root))
    return commit, dirty


def platform_info(env: Mapping[str, str] | None = None) -> PlatformInfo:
    """Collect interpreter, OS, machine, package, and threading-environment information."""
    env = os.environ if env is None else env
    packages = {"arm-rc-ctrl": __version__, **dependencies.installed_versions()}
    threads = {name: env[name] for name in THREAD_ENVIRONMENT_VARIABLES if name in env}
    return PlatformInfo(
        python=platform.python_version(),
        implementation=platform.python_implementation(),
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        packages=packages,
        thread_environment=threads,
    )


def artifact_reference(store: StorageRoot, uri: str | ArtifactUri) -> ArtifactReference:
    """Digest an existing payload and return its reference."""
    path = store.path(uri, mode="read")
    if not path.is_file():
        msg = f"{uri} is not a file"
        raise ArtifactMismatchError(msg)
    return ArtifactReference(uri=str(uri), sha256=sha256_file(path), size=path.stat().st_size)


def verify_artifact(store: StorageRoot, reference: ArtifactReference) -> Path:
    """Resolve a referenced payload and fail unless its size and digest match."""
    path = store.path(reference.uri, mode="read")
    size = path.stat().st_size
    if size != reference.size:
        msg = f"{reference.uri}: size {size} != recorded {reference.size}"
        raise ArtifactMismatchError(msg)
    digest = sha256_file(path)
    if digest != reference.sha256:
        msg = f"{reference.uri}: sha256 {digest} != recorded {reference.sha256}"
        raise ArtifactMismatchError(msg)
    return path


def collect_provenance(
    config: object,
    *,
    seeds: Mapping[str, int],
    artifacts: Iterable[ArtifactReference] = (),
    root: Path | None = None,
    exploratory: bool = False,
    now: datetime | None = None,
    env: Mapping[str, str] | None = None,
) -> ProvenanceRecord:
    """Capture the current checkout, environment, configuration, inputs, and seeds.

    Parameters
    ----------
    config : object
        Resolved configuration: a dataclass instance or a plain mapping.
    seeds : Mapping[str, int]
        Every random seed used, by name.
    artifacts : Iterable[ArtifactReference], optional
        Input payload references.
    root : Path | None, optional
        Repository root; defaults to the checkout containing this package.
    exploratory : bool, optional
        Mark the result as exploratory so a dirty worktree is tolerated.
    now : datetime | None, optional
        Timestamp override (must be timezone-aware); defaults to the current UTC time.
    env : Mapping[str, str] | None, optional
        Environment override for threading variables.
    """
    root = repository_root() if root is None else root
    stamp = datetime.now(UTC) if now is None else now
    if stamp.tzinfo is None:
        msg = "timestamp must be timezone-aware"
        raise ValueError(msg)
    commit, dirty = worktree_state(root)
    builds = dependencies.verify_builds(root)
    config_json, digest = config_digest(config)
    _validate_seeds(seeds)
    return ProvenanceRecord(
        created_at=stamp.astimezone(UTC).isoformat(timespec="seconds"),
        project_commit=commit,
        project_dirty=dirty,
        submodules=dependencies.submodule_revisions(root),
        builds=builds,
        lock_sha256=sha256_file(root / "uv.lock"),
        config_json=config_json,
        config_sha256=digest,
        artifacts=tuple(artifacts),
        seeds=dict(seeds),
        platform=platform_info(env),
        exploratory=exploratory,
    )


def require_clean_for_confirmatory(record: ProvenanceRecord) -> None:
    """Reject confirmatory use of a record collected from a dirty or drifted checkout.

    Exploratory records pass unconditionally.
    """
    if record.exploratory or record.is_clean:
        return
    problems: list[str] = []
    if record.project_dirty:
        problems.append("project worktree is dirty")
    problems.extend(f"submodule {s.name} is dirty" for s in record.submodules if s.dirty)
    problems.extend(
        f"submodule {s.name} is at {s.checked_out}, not its pin {s.recorded}"
        for s in record.submodules
        if s.checked_out is not None and not s.matches_pin
    )
    problems.extend(f"{b.name} is an editable install (build identity not fixed)" for b in record.builds if b.editable)
    problems.extend(f"{b.name} was built from a dirty submodule" for b in record.builds if b.source_dirty)
    msg = "confirmatory results require a clean checkout: " + "; ".join(problems)
    raise DirtyWorktreeError(msg)
