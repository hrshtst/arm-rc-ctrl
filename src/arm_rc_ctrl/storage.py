# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Machine-local external storage root and logical ``armrc://`` artifact URIs.

Experimental payloads never live in the repository. Every tool resolves one
storage root per machine, in this order (``docs/PLAN.md`` section 7.1):

1. the ``ARM_RC_CTRL_STORAGE_ROOT`` environment variable;
2. ``[storage].root`` in ``${XDG_CONFIG_HOME:-$HOME/.config}/arm-rc-ctrl/storage.toml``;
3. ``/external/arm-rc-ctrl``.

The root must already exist, be a readable directory outside every known
repository worktree, and be writable for operations that produce data. Nothing
here creates the root or falls back to the repository.

Versioned metadata refers to payloads by logical URI, ``armrc://<bucket>/<path>``,
where ``<bucket>`` is one of :data:`BUCKETS`. Resolution canonicalizes the
target and refuses anything that escapes the root, including through symlinks.
"""

from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.repo import GitError, RepositoryNotFoundError, git_output, repository_root

__all__ = [
    "BUCKETS",
    "CONFIG_RELATIVE_PATH",
    "DEFAULT_ROOT",
    "ENV_VAR",
    "URI_SCHEME",
    "ArtifactUri",
    "InvalidArtifactUriError",
    "ResolvedRoot",
    "StorageAccessError",
    "StorageConfig",
    "StorageError",
    "StorageRoot",
    "StorageRootError",
    "StorageSettings",
    "open_storage",
    "resolve_storage_root",
    "storage_config_path",
]

ENV_VAR = "ARM_RC_CTRL_STORAGE_ROOT"
DEFAULT_ROOT = Path("/external/arm-rc-ctrl")
CONFIG_RELATIVE_PATH = Path("arm-rc-ctrl") / "storage.toml"
URI_SCHEME = "armrc"
BUCKETS: tuple[str, ...] = ("raw", "processed", "runs", "models", "mlflow", "optuna", "dvc-cache", "dvc-store")

_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_URI_PREFIX = f"{URI_SCHEME}://"

type AccessMode = Literal["read", "write"]
type RootSource = Literal["environment", "config", "default"]


class StorageError(RuntimeError):
    """Base class for storage configuration and access failures."""


class StorageRootError(StorageError):
    """The configured storage root is unusable."""


class StorageAccessError(StorageError):
    """A payload path cannot be read or written as requested."""


class InvalidArtifactUriError(StorageError, ValueError):
    """A logical artifact URI is malformed."""


@dataclass(frozen=True)
class StorageSettings:
    """The ``[storage]`` table of ``storage.toml``."""

    root: Path


@dataclass(frozen=True)
class StorageConfig:
    """Schema of ``storage.toml``."""

    storage: StorageSettings


@dataclass(frozen=True)
class ResolvedRoot:
    """Where the storage root came from, before validation."""

    path: Path
    source: RootSource
    detail: str


@dataclass(frozen=True)
class ArtifactUri:
    """Parsed ``armrc://<bucket>/<segment>/...`` URI."""

    bucket: str
    segments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the bucket name and every path segment."""
        if self.bucket not in BUCKETS:
            msg = f"unknown bucket {self.bucket!r}; expected one of {', '.join(BUCKETS)}"
            raise InvalidArtifactUriError(msg)
        for segment in self.segments:
            if segment in {".", ".."} or not _SEGMENT_RE.match(segment):
                msg = f"invalid path segment {segment!r} in {self.bucket}/{'/'.join(self.segments)}"
                raise InvalidArtifactUriError(msg)

    @classmethod
    def parse(cls, uri: str) -> ArtifactUri:
        """Parse a logical URI string."""
        if not uri.startswith(_URI_PREFIX):
            msg = f"expected a {_URI_PREFIX} URI, got {uri!r}"
            raise InvalidArtifactUriError(msg)
        rest = uri[len(_URI_PREFIX) :]
        if not rest or rest.startswith("/"):
            msg = f"URI {uri!r} has no bucket"
            raise InvalidArtifactUriError(msg)
        if any(ch in rest for ch in "\\?#") or rest.endswith("/"):
            msg = f"URI {uri!r} contains a backslash, query, fragment, or trailing slash"
            raise InvalidArtifactUriError(msg)
        bucket, *segments = rest.split("/")
        if any(not s for s in segments):
            msg = f"URI {uri!r} contains an empty path segment"
            raise InvalidArtifactUriError(msg)
        return cls(bucket, tuple(segments))

    @property
    def relative_path(self) -> Path:
        """Path of the payload relative to the storage root."""
        return Path(self.bucket, *self.segments)

    def __str__(self) -> str:
        """Render the canonical ``armrc://`` form."""
        return _URI_PREFIX + "/".join((self.bucket, *self.segments))


def storage_config_path(env: Mapping[str, str]) -> Path:
    """Location of the per-machine ``storage.toml`` following the XDG base directory rules."""
    config_home = env.get("XDG_CONFIG_HOME") or ""
    if not config_home:
        home = env.get("HOME") or ""
        if not home:
            msg = "neither XDG_CONFIG_HOME nor HOME is set; cannot locate storage.toml"
            raise StorageRootError(msg)
        config_home = os.path.join(home, ".config")  # noqa: PTH118
    return Path(config_home) / CONFIG_RELATIVE_PATH


def resolve_storage_root(env: Mapping[str, str] | None = None) -> ResolvedRoot:
    """Determine the configured storage root without validating it.

    Raises
    ------
    StorageRootError
        If the environment variable holds a relative path.
    ConfigError
        If ``storage.toml`` exists but is invalid.
    """
    env = os.environ if env is None else env
    value = env.get(ENV_VAR)
    if value:
        path = Path(value)
        if not path.is_absolute():
            msg = f"{ENV_VAR} must be an absolute path, got {value!r}"
            raise StorageRootError(msg)
        return ResolvedRoot(path, "environment", ENV_VAR)
    config_file = storage_config_path(env)
    if config_file.is_file():
        config = load_config(config_file, StorageConfig)
        return ResolvedRoot(config.storage.root, "config", str(config_file))
    return ResolvedRoot(DEFAULT_ROOT, "default", str(DEFAULT_ROOT))


def _known_repositories() -> tuple[Path, ...]:
    """Worktrees a storage root must not live in: this checkout and the current directory's repository."""
    found: list[Path] = []
    with contextlib.suppress(RepositoryNotFoundError):
        found.append(repository_root())
    with contextlib.suppress(GitError):
        found.append(Path(git_output("rev-parse", "--show-toplevel", cwd=Path.cwd())).resolve())
    return tuple(dict.fromkeys(found))


class StorageRoot:
    """A validated storage root that resolves logical URIs to canonical paths beneath it."""

    def __init__(self, root: Path, *, repositories: tuple[Path, ...] | None = None) -> None:
        if not root.is_absolute():
            msg = f"storage root must be absolute, got {root}"
            raise StorageRootError(msg)
        if not root.exists():
            msg = f"storage root {root} does not exist; create it or configure another root"
            raise StorageRootError(msg)
        canonical = root.resolve()
        if not canonical.is_dir():
            msg = f"storage root {root} is not a directory"
            raise StorageRootError(msg)
        if not os.access(canonical, os.R_OK | os.X_OK):
            msg = f"storage root {root} is not readable"
            raise StorageRootError(msg)
        repos = _known_repositories() if repositories is None else tuple(r.resolve() for r in repositories)
        for repo in repos:
            if canonical == repo or canonical.is_relative_to(repo):
                msg = f"storage root {root} lies inside the repository worktree {repo}; payloads must stay external"
                raise StorageRootError(msg)
        self._root = canonical

    @property
    def root(self) -> Path:
        """Canonical (symlink-free) root directory."""
        return self._root

    def path(self, uri: str | ArtifactUri, *, mode: AccessMode) -> Path:
        """Resolve a logical URI to a canonical path under the root.

        Parameters
        ----------
        uri : str | ArtifactUri
            Logical location of the payload.
        mode : {"read", "write"}
            ``read`` requires an existing, readable target. ``write`` requires a
            writable root and creates the target's parent directories.

        Raises
        ------
        InvalidArtifactUriError
            If ``uri`` is malformed.
        StorageAccessError
            If the target escapes the root (e.g. via a symlink) or cannot be
            accessed in the requested mode.
        """
        parsed = ArtifactUri.parse(uri) if isinstance(uri, str) else uri
        target = self._root / parsed.relative_path
        resolved = target.resolve()
        if resolved != self._root and not resolved.is_relative_to(self._root):
            msg = f"{parsed} resolves to {resolved}, outside the storage root {self._root}"
            raise StorageAccessError(msg)
        if mode == "read":
            if not resolved.exists():
                msg = f"{parsed} does not exist under {self._root}"
                raise StorageAccessError(msg)
            if not os.access(resolved, os.R_OK):
                msg = f"{parsed} is not readable"
                raise StorageAccessError(msg)
            return resolved
        if not os.access(self._root, os.W_OK):
            msg = f"storage root {self._root} is not writable"
            raise StorageAccessError(msg)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(resolved.parent, os.W_OK):
            msg = f"directory of {parsed} is not writable"
            raise StorageAccessError(msg)
        return resolved

    def uri_for(self, path: Path) -> ArtifactUri:
        """Return the logical URI of a canonical path beneath the root."""
        resolved = path.resolve()
        if not resolved.is_relative_to(self._root):
            msg = f"{path} is not under the storage root {self._root}"
            raise StorageAccessError(msg)
        bucket, *segments = resolved.relative_to(self._root).parts
        return ArtifactUri(bucket, tuple(segments))


def open_storage(env: Mapping[str, str] | None = None) -> StorageRoot:
    """Resolve and validate the machine-local storage root."""
    resolved = resolve_storage_root(env)
    try:
        return StorageRoot(resolved.path)
    except StorageRootError as exc:
        msg = f"{exc} (configured via {resolved.source}: {resolved.detail})"
        raise StorageRootError(msg) from None
