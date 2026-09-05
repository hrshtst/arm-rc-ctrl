# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M0-004: each quality tool runs and reports a deliberately planted problem."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "quality"
AUDITED_REPRODUCTION_ORCHESTRATOR = "*/arm_rc_ctrl/experiments/reproduce_recovery.py"


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_ruff_reports_planted_unused_import() -> None:
    """Ruff runs with the project configuration and flags F401 in the fixture."""
    result = _run("ruff", "check", "--no-cache", "--output-format=concise", str(FIXTURES / "lint_violation.py"))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "F401" in result.stdout


def test_ruff_format_reports_planted_formatting() -> None:
    """Ruff's formatter check fails on the unformatted fixture."""
    result = _run("ruff", "format", "--check", "--no-cache", str(FIXTURES / "format_violation.py"))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "format_violation.py" in result.stdout


def test_basedpyright_reports_planted_type_error(tmp_path: Path) -> None:
    """Basedpyright runs with the project's strict settings and reports the wrong return type.

    The fixture directory is excluded by ``[tool.basedpyright]`` so the normal
    gate ignores it. This test copies those settings verbatim, replacing only
    the file selection, so the planted error is judged by the real policy.
    """
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        settings = tomllib.load(f)["tool"]["basedpyright"]
    settings["include"] = [str(FIXTURES / "type_violation.py")]
    settings["exclude"] = []
    settings["venvPath"] = str(REPO_ROOT)
    config = tmp_path / "pyrightconfig.json"
    config.write_text(json.dumps(settings))
    result = _run("basedpyright", "--project", str(config), "--outputjson")
    assert result.returncode == 1, result.stdout + result.stderr
    report = json.loads(result.stdout)
    rules = {d.get("rule") for d in report["generalDiagnostics"]}
    assert "reportReturnType" in rules, rules


def test_pytest_executes_planted_failing_assertion(tmp_path: Path) -> None:
    """Pytest runs the planted failing test and coverage measurement is active."""
    env = {"COVERAGE_FILE": str(tmp_path / ".coverage"), "PATH": str(Path(sys.executable).parent)}
    result = _run(
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--cov=arm_rc_ctrl",
        "--cov-report=term",
        str(FIXTURES / "failing_example.py"),
        env=env,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 failed" in result.stdout
    assert "test_deliberate_failure" in result.stdout
    assert "coverage:" in result.stdout.lower()


def test_planted_failing_example_is_not_collected_by_default() -> None:
    """The normal collection never picks up the failing fixture."""
    result = _run("pytest", "--collect-only", "-q", "-p", "no:cacheprovider")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "failing_example.py" not in result.stdout


def test_coverage_excludes_only_the_independently_audited_orchestrator() -> None:
    """The private-payload reproduction command is the sole coverage exclusion."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        coverage_run = tomllib.load(f)["tool"]["coverage"]["run"]

    assert coverage_run["omit"] == [AUDITED_REPRODUCTION_ORCHESTRATOR]
