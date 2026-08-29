# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Repository quality gate.

Every session runs the tools from the locked project environment via
``uv run --locked`` so local, CI, and reviewer runs use identical versions.
Run everything with ``uv run nox`` or one session with ``uv run nox -s lint``.
Set ``UV_PYTHON=3.13`` to exercise another supported interpreter.
"""

from __future__ import annotations

import nox

nox.options.sessions = ["lint", "type_check", "tests", "cpp"]
nox.options.default_venv_backend = "none"

UV_RUN = ("uv", "run", "--locked")


@nox.session
def lint(session: nox.Session) -> None:
    """Run Ruff lint and format checks."""
    session.run(*UV_RUN, "ruff", "check", ".", external=True)
    session.run(*UV_RUN, "ruff", "format", "--check", ".", external=True)


@nox.session
def type_check(session: nox.Session) -> None:
    """Run basedpyright in strict mode."""
    session.run(*UV_RUN, "basedpyright", external=True)


@nox.session
def tests(session: nox.Session) -> None:
    """Run the test suite with branch coverage reporting (no threshold yet; see M1)."""
    session.run(
        *UV_RUN,
        "pytest",
        "--cov",
        "--cov-report=term-missing",
        "--cov-report=xml",
        *session.posargs,
        external=True,
    )


@nox.session
def pre_commit(session: nox.Session) -> None:
    """Run every pre-commit hook against all files."""
    session.run(*UV_RUN, "pre-commit", "run", "--all-files", "--show-diff-on-failure", external=True)


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
