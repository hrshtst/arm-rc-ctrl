# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Pinned submodule revisions and the Python packages built from them.

``rclib`` and ``skelarm`` are installed into the ``uv`` environment as
non-editable builds of the submodules under ``third_party/``. ``uv`` does not
rebuild a path dependency when only its source files change, so this module
also detects installed copies that no longer match the checked-out sources.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import tomllib
from dataclasses import dataclass
from pathlib import Path

from arm_rc_ctrl.repo import git_output, repository_root

__all__ = [
    "PYTHON_DEPENDENCIES",
    "SUBMODULES",
    "SubmoduleRevision",
    "installed_source_digest",
    "installed_versions",
    "python_source_dir",
    "source_digest",
    "stale_installs",
    "submodule_revisions",
    "submodule_version",
]

SUBMODULES: tuple[str, ...] = ("rclib", "skelarm", "rtctrl")
"""Submodule names under ``third_party/`` in the order they are reported."""

PYTHON_DEPENDENCIES: tuple[str, ...] = ("rclib", "skelarm", "numpy")
"""Distributions whose installed versions are recorded with every result."""

_PYTHON_SOURCE_DIRS: dict[str, str] = {
    "rclib": "python/rclib",
    "skelarm": "src/skelarm",
}


@dataclass(frozen=True)
class SubmoduleRevision:
    """Recorded and checked-out revision of one submodule."""

    name: str
    path: str
    recorded: str
    """Commit recorded in the superproject index (equals ``HEAD`` on a clean checkout)."""
    checked_out: str | None
    """``HEAD`` of the working copy, or ``None`` when the submodule is not initialized."""
    dirty: bool | None
    """Whether the working copy has tracked or untracked changes; ``None`` when not initialized."""

    @property
    def matches_pin(self) -> bool:
        """Whether the checked-out commit is exactly the recorded pin."""
        return self.checked_out == self.recorded


def submodule_revisions(root: Path | None = None) -> tuple[SubmoduleRevision, ...]:
    """Return the recorded and checked-out revisions of every submodule."""
    root = repository_root() if root is None else root
    revisions: list[SubmoduleRevision] = []
    for name in SUBMODULES:
        rel = f"third_party/{name}"
        entry = git_output("ls-files", "--stage", "--", rel, cwd=root)
        if not entry:
            msg = f"{rel} is not a recorded submodule of {root}"
            raise ValueError(msg)
        mode, recorded, _stage, _path = entry.split(maxsplit=3)
        if mode != "160000":
            msg = f"{rel} is not a gitlink (mode {mode})"
            raise ValueError(msg)
        workdir = root / rel
        if (workdir / ".git").exists():
            checked_out: str | None = git_output("rev-parse", "HEAD", cwd=workdir)
            dirty: bool | None = bool(git_output("status", "--porcelain", cwd=workdir))
        else:
            checked_out = None
            dirty = None
        revisions.append(SubmoduleRevision(name, rel, recorded, checked_out, dirty))
    return tuple(revisions)


def installed_versions(distributions: tuple[str, ...] = PYTHON_DEPENDENCIES) -> dict[str, str]:
    """Return installed distribution versions keyed by distribution name."""
    return {name: importlib.metadata.version(name) for name in distributions}


def submodule_version(name: str, root: Path | None = None) -> str:
    """Return the ``[project].version`` declared by a submodule's ``pyproject.toml``."""
    root = repository_root() if root is None else root
    with (root / "third_party" / name / "pyproject.toml").open("rb") as f:
        version = tomllib.load(f)["project"]["version"]
    if not isinstance(version, str):
        msg = f"{name}: project.version is not a string"
        raise TypeError(msg)
    return version


def python_source_dir(name: str, root: Path | None = None) -> Path:
    """Return the Python package directory inside the named submodule."""
    root = repository_root() if root is None else root
    try:
        return root / "third_party" / name / _PYTHON_SOURCE_DIRS[name]
    except KeyError:
        msg = f"{name} provides no Python package"
        raise ValueError(msg) from None


def source_digest(package_dir: Path) -> str:
    """SHA-256 over the relative paths and contents of every ``*.py`` file below ``package_dir``."""
    if not package_dir.is_dir():
        msg = f"not a directory: {package_dir}"
        raise NotADirectoryError(msg)
    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(package_dir).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def installed_source_digest(package: str) -> str:
    """SHA-256 of the installed package's Python sources (located without importing it)."""
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        msg = f"{package} is not an installed package"
        raise ModuleNotFoundError(msg)
    return source_digest(Path(next(iter(spec.submodule_search_locations))))


def stale_installs(root: Path | None = None) -> dict[str, str]:
    """Return installed submodule packages whose sources or versions differ from the checkout.

    The value explains the mismatch. Compiled extensions (rclib's ``_rclib``)
    cannot be compared this way; a pin advance that changes only C++ sources is
    caught by the version comparison only if the version was bumped.
    """
    root = repository_root() if root is None else root
    stale: dict[str, str] = {}
    for name in _PYTHON_SOURCE_DIRS:
        expected_version = submodule_version(name, root)
        installed_version = importlib.metadata.version(name)
        if installed_version != expected_version:
            stale[name] = f"installed version {installed_version} != submodule version {expected_version}"
            continue
        if installed_source_digest(name) != source_digest(python_source_dir(name, root)):
            stale[name] = "installed Python sources differ from the submodule checkout"
    return stale
