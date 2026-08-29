# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Pinned submodule revisions and the build identity of the packages built from them.

``rclib`` and ``skelarm`` are installed into the ``uv`` environment as
non-editable builds of the submodules under ``third_party/``. ``uv`` does not
rebuild a path dependency when only its source files change, and a compiled
extension carries no revision of its own, so this module maintains a **build
manifest** in the environment's ``site-packages``:

- ``python -m arm_rc_ctrl.dependencies rebuild`` reinstalls both packages from
  the checked-out submodules and then records, per package, the submodule
  commit, the installed version, and digests of the installed Python sources
  and compiled extensions (:func:`stamp_builds`);
- :func:`verify_builds` compares that record with the current submodules and
  installed files and raises :class:`BuildIdentityError` when the manifest is
  missing, the pin moved, a submodule is dirty, or an installed file differs.

Provenance collection calls :func:`verify_builds`, so a result can never
report a pin that differs from the binary that produced it. Editable installs
(used only for upstream development) are recorded as such; they are accepted
for exploratory runs and rejected for confirmatory ones.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import sysconfig
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from arm_rc_ctrl.config import ConfigError, from_mapping, to_mapping
from arm_rc_ctrl.repo import git_output, repository_root

__all__ = [
    "BUILT_PACKAGES",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "PYTHON_DEPENDENCIES",
    "SUBMODULES",
    "BuildIdentity",
    "BuildIdentityError",
    "BuildManifest",
    "SubmoduleRevision",
    "check_build",
    "current_build_identity",
    "installed_extension_digest",
    "installed_is_editable",
    "installed_source_digest",
    "installed_versions",
    "main",
    "manifest_path",
    "python_source_dir",
    "read_manifest",
    "site_packages",
    "source_digest",
    "stamp_builds",
    "submodule_revisions",
    "submodule_version",
    "verify_builds",
]

SUBMODULES: tuple[str, ...] = ("rclib", "skelarm", "rtctrl")
"""Submodule names under ``third_party/`` in the order they are reported."""

PYTHON_DEPENDENCIES: tuple[str, ...] = ("rclib", "skelarm", "numpy")
"""Distributions whose installed versions are recorded with every result."""

BUILT_PACKAGES: tuple[str, ...] = ("rclib", "skelarm")
"""Packages installed from submodules and therefore tracked by the build manifest."""

MANIFEST_FILENAME = "arm_rc_ctrl_build_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
REBUILD_COMMAND = "uv run python -m arm_rc_ctrl.dependencies rebuild"

_PYTHON_SOURCE_DIRS: dict[str, str] = {
    "rclib": "python/rclib",
    "skelarm": "src/skelarm",
}
_EXTENSION_SUFFIXES = (".so", ".pyd", ".dylib")
_SHA256_HEX_LENGTH = 64
_ROOT = ""
_SCHEMA_KEY = "schema_version"


class BuildIdentityError(RuntimeError):
    """Installed rclib/skelarm builds cannot be tied to the pinned sources."""


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

    def __post_init__(self) -> None:
        """Validate identifier formats and the initialized/uninitialized consistency."""
        if self.path != f"third_party/{self.name}":
            msg = f"submodule {self.name}: path must be third_party/{self.name}, got {self.path!r}"
            raise ValueError(msg)
        if not _is_hex(self.recorded, 40):
            msg = f"submodule {self.name}: recorded must be a 40-hex commit, got {self.recorded!r}"
            raise ValueError(msg)
        if (self.checked_out is None) != (self.dirty is None):
            msg = f"submodule {self.name}: checked_out and dirty must both be null (uninitialized) or both set"
            raise ValueError(msg)
        if self.checked_out is not None and not _is_hex(self.checked_out, 40):
            msg = f"submodule {self.name}: checked_out must be a 40-hex commit, got {self.checked_out!r}"
            raise ValueError(msg)

    @property
    def matches_pin(self) -> bool:
        """Whether the checked-out commit is exactly the recorded pin."""
        return self.checked_out == self.recorded


@dataclass(frozen=True)
class BuildIdentity:
    """What was built, from which sources, into which installed files."""

    name: str
    version: str
    source_commit: str
    """Submodule ``HEAD`` the installed build derives from."""
    source_dirty: bool
    """Whether the submodule had uncommitted changes (identity is then unknown)."""
    editable: bool
    """Editable install: the code follows the checkout and is never stamped."""
    python_sources_sha256: str
    extension_sha256: str | None
    """Digest of compiled extension modules in the installed package, or ``None`` if there are none."""

    def __post_init__(self) -> None:
        """Validate identifier formats."""
        if not _is_hex(self.source_commit, 40):
            msg = f"{self.name}: source_commit must be a 40-hex commit, got {self.source_commit!r}"
            raise ValueError(msg)
        if not _is_hex(self.python_sources_sha256, _SHA256_HEX_LENGTH):
            msg = f"{self.name}: python_sources_sha256 must be 64 hex characters"
            raise ValueError(msg)
        if self.extension_sha256 is not None and not _is_hex(self.extension_sha256, _SHA256_HEX_LENGTH):
            msg = f"{self.name}: extension_sha256 must be 64 hex characters or null"
            raise ValueError(msg)


@dataclass(frozen=True)
class BuildManifest:
    """On-disk record written by :func:`stamp_builds`."""

    schema_version: int
    created_at: str
    builds: tuple[BuildIdentity, ...]


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(c in "0123456789abcdef" for c in value)


# --- submodules ---------------------------------------------------------------


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


# --- installed packages ---------------------------------------------------------


def installed_versions(distributions: tuple[str, ...] = PYTHON_DEPENDENCIES) -> dict[str, str]:
    """Return installed distribution versions keyed by distribution name."""
    return {name: importlib.metadata.version(name) for name in distributions}


def _installed_package_dir(package: str) -> Path:
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        msg = f"{package} is not an installed package"
        raise ModuleNotFoundError(msg)
    return Path(next(iter(spec.submodule_search_locations)))


def source_digest(package_dir: Path) -> str:
    """SHA-256 over the relative paths and contents of every ``*.py`` file below ``package_dir``."""
    return _tree_digest(package_dir, (".py",))


def _tree_digest(package_dir: Path, suffixes: tuple[str, ...]) -> str:
    """SHA-256 over relative paths and contents of files with the given suffixes, in sorted order."""
    if not package_dir.is_dir():
        msg = f"not a directory: {package_dir}"
        raise NotADirectoryError(msg)
    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*")):
        if "__pycache__" in path.parts or not path.is_file() or path.suffix not in suffixes:
            continue
        digest.update(path.relative_to(package_dir).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def installed_source_digest(package: str) -> str:
    """SHA-256 of the installed package's Python sources (located without importing it)."""
    return source_digest(_installed_package_dir(package))


def installed_extension_digest(package: str) -> str | None:
    """SHA-256 of the installed package's compiled extension modules, or ``None`` if it has none."""
    package_dir = _installed_package_dir(package)
    if not any(p.suffix in _EXTENSION_SUFFIXES for p in package_dir.rglob("*") if p.is_file()):
        return None
    return _tree_digest(package_dir, _EXTENSION_SUFFIXES)


def installed_is_editable(distribution: str) -> bool:
    """Whether the distribution was installed in editable mode (PEP 610 ``direct_url.json``)."""
    text = importlib.metadata.distribution(distribution).read_text("direct_url.json")
    if text is None:
        return False
    info: object = json.loads(text)
    if not isinstance(info, dict):
        return False
    dir_info = cast("dict[str, object]", info).get("dir_info")
    if not isinstance(dir_info, dict):
        return False
    return cast("dict[str, object]", dir_info).get("editable") is True


# --- build manifest -------------------------------------------------------------


def site_packages() -> Path:
    """The ``site-packages`` directory of the running environment."""
    return Path(sysconfig.get_paths()["purelib"])


def manifest_path(site: Path | None = None) -> Path:
    """Location of the build manifest inside ``site``."""
    return (site_packages() if site is None else site) / MANIFEST_FILENAME


def current_build_identity(name: str, root: Path | None = None) -> BuildIdentity:
    """Describe the installed build of ``name`` together with the submodule it should derive from."""
    if name not in BUILT_PACKAGES:
        msg = f"{name} is not built from a submodule"
        raise ValueError(msg)
    root = repository_root() if root is None else root
    revision = next(r for r in submodule_revisions(root) if r.name == name)
    if revision.checked_out is None or revision.dirty is None:
        msg = f"submodule {revision.path} is not initialized"
        raise BuildIdentityError(msg)
    editable = installed_is_editable(name)
    return BuildIdentity(
        name=name,
        version=importlib.metadata.version(name),
        source_commit=revision.checked_out,
        source_dirty=revision.dirty,
        editable=editable,
        python_sources_sha256=installed_source_digest(name),
        extension_sha256=None if editable else installed_extension_digest(name),
    )


def read_manifest(site: Path | None = None) -> BuildManifest | None:
    """Load the manifest, ``None`` if absent; malformed content raises :class:`ConfigError`."""
    path = manifest_path(site)
    if not path.is_file():
        return None
    try:
        data: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"invalid JSON: {exc}"
        raise ConfigError(_ROOT, msg, source=path) from exc
    if not isinstance(data, dict):
        msg = "manifest must be a JSON object"
        raise ConfigError(_ROOT, msg, source=path)
    manifest = from_mapping(cast("dict[str, object]", data), BuildManifest)
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        msg = f"unsupported schema_version {manifest.schema_version}; expected {MANIFEST_SCHEMA_VERSION}"
        raise ConfigError(_SCHEMA_KEY, msg, source=path)
    return manifest


def stamp_builds(root: Path | None = None, site: Path | None = None, *, now: datetime | None = None) -> BuildManifest:
    """Record the identity of the installed builds. Call only right after reinstalling them.

    Refuses dirty submodules, editable installs, and version mismatches, since
    none of those can be tied to a pinned revision.
    """
    root = repository_root() if root is None else root
    identities: list[BuildIdentity] = []
    for name in BUILT_PACKAGES:
        identity = current_build_identity(name, root)
        if identity.editable:
            msg = f"{name} is installed in editable mode; editable installs are never stamped"
            raise BuildIdentityError(msg)
        if identity.source_dirty:
            msg = f"submodule third_party/{name} has uncommitted changes; commit upstream or use an editable override"
            raise BuildIdentityError(msg)
        expected = submodule_version(name, root)
        if identity.version != expected:
            msg = (
                f"{name}: installed version {identity.version} != submodule version {expected}; run `{REBUILD_COMMAND}`"
            )
            raise BuildIdentityError(msg)
        identities.append(identity)
    stamp = datetime.now(UTC) if now is None else now.astimezone(UTC)
    manifest = BuildManifest(MANIFEST_SCHEMA_VERSION, stamp.isoformat(timespec="seconds"), tuple(identities))
    path = manifest_path(site)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(to_mapping(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return manifest


def check_build(recorded: BuildIdentity | None, current: BuildIdentity) -> list[str]:
    """Compare a manifest entry with the present state; return human-readable problems (empty if consistent)."""
    if current.editable:
        return []
    name = current.name
    if recorded is None:
        return [f"{name}: no build manifest entry (unknown build identity); run `{REBUILD_COMMAND}`"]
    problems: list[str] = []
    if current.source_dirty:
        problems.append(f"{name}: submodule has uncommitted changes, so the installed build cannot be identified")
    if recorded.source_commit != current.source_commit:
        problems.append(
            f"{name}: installed build was stamped for commit {recorded.source_commit[:12]} but the submodule is at "
            f"{current.source_commit[:12]} (pin advanced?); run `{REBUILD_COMMAND}`"
        )
    if recorded.version != current.version:
        problems.append(f"{name}: stamped version {recorded.version} != installed version {current.version}")
    if recorded.python_sources_sha256 != current.python_sources_sha256:
        problems.append(f"{name}: installed Python sources differ from the stamped build; run `{REBUILD_COMMAND}`")
    if recorded.extension_sha256 != current.extension_sha256:
        problems.append(f"{name}: installed compiled extension differs from the stamped build; run `{REBUILD_COMMAND}`")
    return problems


def verify_builds(root: Path | None = None, site: Path | None = None) -> tuple[BuildIdentity, ...]:
    """Return the identities of the installed builds, or raise if any cannot be tied to its pin."""
    root = repository_root() if root is None else root
    manifest = read_manifest(site)
    recorded = {b.name: b for b in manifest.builds} if manifest is not None else {}
    problems: list[str] = []
    identities: list[BuildIdentity] = []
    for name in BUILT_PACKAGES:
        current = current_build_identity(name, root)
        problems.extend(check_build(recorded.get(name), current))
        identities.append(current)
    if problems:
        raise BuildIdentityError("\n".join(problems))
    return tuple(identities)


# --- command line -----------------------------------------------------------------


def _rebuild(root: Path) -> None:
    """Reinstall the built packages from the checked-out submodules with the running interpreter."""
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    command = ["uv", "sync", "--locked", f"--python={python}"]
    for name in BUILT_PACKAGES:
        command.extend(["--reinstall-package", name])
    subprocess.run(command, cwd=root, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m arm_rc_ctrl.dependencies {verify,rebuild}``."""
    parser = argparse.ArgumentParser(description="Verify or rebuild the packages built from pinned submodules.")
    parser.add_argument("command", choices=("verify", "rebuild"))
    args = parser.parse_args(argv)
    root = repository_root()
    if args.command == "rebuild":
        _rebuild(root)
        manifest = stamp_builds(root)
        identities = manifest.builds
    else:
        identities = verify_builds(root)
    print(json.dumps([to_mapping(b) for b in identities], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
