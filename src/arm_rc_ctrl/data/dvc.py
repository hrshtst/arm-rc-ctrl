# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Per-machine DVC configuration below the external storage root (``docs/PLAN.md`` section 11).

Git tracks only DVC's portable metadata (``.dvc/config``, ``.dvc/.gitignore``,
``.dvcignore``, ``*.dvc`` metafiles, ``dvc.yaml``/``dvc.lock``). The cache
directory and the default local remote are machine-specific and live in the
Git-ignored ``.dvc/config.local``, resolved to ``<storage-root>/dvc-cache`` and
``<storage-root>/dvc-store``.

Command line::

    python -m arm_rc_ctrl.data.dvc setup    # write .dvc/config.local for this machine
    python -m arm_rc_ctrl.data.dvc verify   # check the mapping and Git portability
"""

from __future__ import annotations

import argparse
import configparser
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from arm_rc_ctrl.repo import git_output, repository_root
from arm_rc_ctrl.storage import StorageRoot, open_storage

__all__ = [
    "CACHE_BUCKET",
    "REMOTE_BUCKET",
    "REMOTE_NAME",
    "DvcConfigError",
    "LocalDvcConfig",
    "configure_local_dvc",
    "main",
    "read_local_config",
    "verify_local_dvc",
]

CACHE_BUCKET = "dvc-cache"
REMOTE_BUCKET = "dvc-store"
REMOTE_NAME = "store"
_LOCAL_CONFIG = Path(".dvc") / "config.local"
_TRACKED_CONFIG = Path(".dvc") / "config"


class DvcConfigError(RuntimeError):
    """The DVC configuration is missing, machine-specific where it must be portable, or mis-mapped."""


@dataclass(frozen=True)
class LocalDvcConfig:
    """Machine-local DVC settings."""

    cache_dir: Path
    remote_name: str
    remote_url: Path


def _dvc(repo_root: Path, *args: str) -> str:
    env = {**os.environ, "DVC_NO_ANALYTICS": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "dvc", *args], cwd=repo_root, env=env, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        msg = f"dvc {' '.join(args)} failed (exit {result.returncode}): {result.stderr.strip()}"
        raise DvcConfigError(msg)
    return result.stdout.strip()


def configure_local_dvc(repo_root: Path, store: StorageRoot) -> LocalDvcConfig:
    """Write ``.dvc/config.local`` mapping the cache and default remote below the storage root."""
    if not (repo_root / _TRACKED_CONFIG).is_file():
        msg = f"{repo_root} is not a DVC repository (run `dvc init` first)"
        raise DvcConfigError(msg)
    cache_dir = store.root / CACHE_BUCKET
    remote_url = store.root / REMOTE_BUCKET
    cache_dir.mkdir(exist_ok=True)
    remote_url.mkdir(exist_ok=True)
    _dvc(repo_root, "config", "--local", "cache.dir", str(cache_dir))
    _dvc(repo_root, "remote", "add", "--local", "-f", REMOTE_NAME, str(remote_url))
    _dvc(repo_root, "remote", "default", "--local", REMOTE_NAME)
    return read_local_config(repo_root)


def read_local_config(repo_root: Path) -> LocalDvcConfig:
    """Parse ``.dvc/config.local``."""
    path = repo_root / _LOCAL_CONFIG
    if not path.is_file():
        msg = f"{path} does not exist; run `python -m arm_rc_ctrl.data.dvc setup`"
        raise DvcConfigError(msg)
    parser = configparser.ConfigParser()
    parser.read(path)
    # DVC writes remote sections as ['remote "name"']; normalize the quoting.
    sections = {name.strip("'"): dict(parser[name]) for name in parser.sections()}
    try:
        cache_dir = Path(sections["cache"]["dir"])
        remote_name = sections["core"]["remote"]
        remote_url = Path(sections[f'remote "{remote_name}"']["url"])
    except KeyError as exc:
        msg = f"{path} lacks {exc}; run `python -m arm_rc_ctrl.data.dvc setup`"
        raise DvcConfigError(msg) from exc
    return LocalDvcConfig(cache_dir=cache_dir, remote_name=remote_name, remote_url=remote_url)


def verify_local_dvc(repo_root: Path, store: StorageRoot) -> LocalDvcConfig:
    """Check the local mapping and that tracked DVC metadata carries no machine paths."""
    config = read_local_config(repo_root)
    problems: list[str] = []
    for label, path, bucket in (
        ("cache.dir", config.cache_dir, CACHE_BUCKET),
        ("remote url", config.remote_url, REMOTE_BUCKET),
    ):
        expected = store.root / bucket
        if not path.is_absolute() or path.resolve() != expected.resolve():
            problems.append(f"{label} is {path}, expected {expected}")
        elif not path.is_dir():
            problems.append(f"{label} {path} does not exist")
    tracked = (repo_root / _TRACKED_CONFIG).read_text(encoding="utf-8")
    if any(line.strip().startswith(("dir", "url")) and "/" in line for line in tracked.splitlines()):
        problems.append(f"{_TRACKED_CONFIG} contains a path; machine-specific settings belong in config.local")
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", "--", _LOCAL_CONFIG.as_posix()],
        cwd=repo_root,
        check=False,
        capture_output=True,
    ).returncode
    if ignored != 0:
        problems.append(f"{_LOCAL_CONFIG} is not ignored by Git")
    if git_output("ls-files", "--", _LOCAL_CONFIG.as_posix(), cwd=repo_root):
        problems.append(f"{_LOCAL_CONFIG} is tracked by Git")
    if problems:
        raise DvcConfigError("; ".join(problems))
    return config


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m arm_rc_ctrl.data.dvc {setup,verify}``."""
    parser = argparse.ArgumentParser(description="Configure or verify per-machine DVC settings.")
    parser.add_argument("command", choices=("setup", "verify"))
    args = parser.parse_args(argv)
    root = repository_root()
    store = open_storage()
    config = configure_local_dvc(root, store) if args.command == "setup" else verify_local_dvc(root, store)
    print(f"cache: {config.cache_dir}\nremote {config.remote_name}: {config.remote_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
