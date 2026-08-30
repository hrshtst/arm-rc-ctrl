# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Locate the project repository and run read-only Git queries against it."""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = ["GitError", "RepositoryNotFoundError", "git_output", "repository_root"]

_ROOT_MARKERS = ("pyproject.toml", "src/arm_rc_ctrl")


class RepositoryNotFoundError(RuntimeError):
    """No enclosing arm-rc-ctrl repository could be located."""


class GitError(RuntimeError):
    """A Git command failed; the message carries the command and its stderr."""


def repository_root(start: Path | None = None) -> Path:
    """Return the repository root that encloses ``start``.

    Parameters
    ----------
    start : Path | None, optional
        Directory (or file) to walk upward from. Defaults to this package's
        source location, which resolves to the checkout when the project is
        installed in editable mode (the ``uv sync`` default).

    Raises
    ------
    RepositoryNotFoundError
        If no ancestor contains both ``pyproject.toml`` and ``src/arm_rc_ctrl``.
    """
    origin = (Path(__file__) if start is None else start).resolve()
    for candidate in (origin, *origin.parents):
        if all((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    msg = f"no arm-rc-ctrl repository root encloses {origin}"
    raise RepositoryNotFoundError(msg)


def git_output(*args: str, cwd: Path) -> str:
    """Run ``git`` with fixed arguments in ``cwd`` and return its stripped stdout."""
    result = subprocess.run(["git", *args], cwd=cwd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        msg = f"git {' '.join(args)} failed in {cwd} (exit {result.returncode}): {result.stderr.strip()}"
        raise GitError(msg)
    return result.stdout.strip()
