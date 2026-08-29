# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Repository quality gate.

Run through the locked project environment: ``uv run --locked nox`` (all default
sessions) or ``uv run --locked nox -s lint``. Sessions execute the tools of that
environment directly and never start a nested ``uv run``: nox strips
``UV_PYTHON`` from session environments, so a nested ``uv`` would silently fall
back to ``.python-version`` and re-create the environment with another
interpreter than the one CI's matrix (or ``UV_PYTHON=3.13 uv run nox``) chose.

Set ``ARM_RC_CTRL_EXPECTED_PYTHON=3.13`` (CI does, from its matrix) to make
every session assert the interpreter it actually runs under.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import nox

nox.options.sessions = ["deps", "lint", "type_check", "tests", "cpp"]
nox.options.default_venv_backend = "none"

REPO_ROOT = Path(__file__).resolve().parent
EXPECTED_PYTHON_VAR = "ARM_RC_CTRL_EXPECTED_PYTHON"


def _require_project_interpreter(session: nox.Session) -> None:
    """Fail unless nox runs inside the project's ``.venv`` under the expected interpreter."""
    actual = f"{sys.version_info.major}.{sys.version_info.minor}"
    session.log(f"interpreter: {sys.executable} (Python {platform.python_version()})")
    venv = (REPO_ROOT / ".venv").resolve()
    if Path(sys.prefix).resolve() != venv:
        session.error(f"nox must run from the locked project environment {venv}; use `uv run --locked nox`")
    expected = os.environ.get(EXPECTED_PYTHON_VAR)
    if expected and expected != actual:
        session.error(
            f"{EXPECTED_PYTHON_VAR}={expected} but this environment is Python {actual}; "
            f"run `UV_PYTHON={expected} uv run --locked nox`"
        )


def _python(session: nox.Session, *args: str) -> None:
    """Run a module of the current interpreter (the project environment)."""
    session.run(sys.executable, *args, external=True)


@nox.session
def deps(session: nox.Session) -> None:
    """Verify that the installed rclib/skelarm builds match the pinned submodules.

    Pass ``-- --rebuild`` to reinstall both packages from the checkout and re-stamp.
    """
    _require_project_interpreter(session)
    command = "rebuild" if "--rebuild" in session.posargs else "verify"
    _python(session, "-m", "arm_rc_ctrl.dependencies", command)


@nox.session
def lint(session: nox.Session) -> None:
    """Run Ruff lint and format checks."""
    _require_project_interpreter(session)
    _python(session, "-m", "ruff", "check", ".")
    _python(session, "-m", "ruff", "format", "--check", ".")


@nox.session
def type_check(session: nox.Session) -> None:
    """Run basedpyright in strict mode."""
    _require_project_interpreter(session)
    _python(session, "-m", "basedpyright")


@nox.session
def tests(session: nox.Session) -> None:
    """Run the test suite with branch coverage reporting (no threshold yet; see M1)."""
    _require_project_interpreter(session)
    _python(session, "-m", "pytest", "--cov", "--cov-report=term-missing", "--cov-report=xml", *session.posargs)


@nox.session
def pre_commit(session: nox.Session) -> None:
    """Run every pre-commit hook against all files."""
    _require_project_interpreter(session)
    _python(session, "-m", "pre_commit", "run", "--all-files", "--show-diff-on-failure")


@nox.session
def cpp(session: nox.Session) -> None:
    """Configure, build, and run the C++ tests (Catch2 is fetched at configure time)."""
    session.run(
        "cmake",
        "-S",
        "cpp",
        "-B",
        "build",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DARM_RC_CTRL_WERROR=ON",
        external=True,
    )
    session.run("cmake", "--build", "build", "-j", external=True)
    session.run("ctest", "--test-dir", "build", "--output-on-failure", external=True)
