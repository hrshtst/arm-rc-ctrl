# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M0-002: submodules are pinned by commit, use HTTPS URLs, and match the notice inventory."""

from __future__ import annotations

import configparser
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMODULES = ("rclib", "skelarm", "rtctrl")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"git {' '.join(args)} unavailable here: {result.stderr.strip()}")
    return result.stdout.strip()


def _recorded_gitlink(path: str) -> str:
    """Commit recorded for a submodule path in the superproject index (equals HEAD on a clean checkout)."""
    line = _git("ls-files", "--stage", "--", path)
    assert line, f"{path} is not recorded in the index"
    mode, sha, _stage, _ = line.split(maxsplit=3)
    assert mode == "160000", f"{path} is not a gitlink: {line}"
    return sha


def _notice_pins() -> dict[str, str]:
    """Pinned commits declared in the THIRD_PARTY_NOTICES.md inventory table."""
    text = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    pins: dict[str, str] = {}
    for name in SUBMODULES:
        match = re.search(rf"^\| {name} \|.*`third_party/{name}` \| `([0-9a-f]{{40}})` \|$", text, re.MULTILINE)
        assert match is not None, f"no pinned-commit row for {name} in THIRD_PARTY_NOTICES.md"
        pins[name] = match.group(1)
    return pins


def test_gitmodules_declares_exactly_the_three_https_submodules() -> None:
    """``.gitmodules`` lists rclib, skelarm, and rtctrl under third_party with HTTPS URLs."""
    parser = configparser.ConfigParser()
    parser.read(REPO_ROOT / ".gitmodules")
    sections = {parser[s]["path"]: parser[s]["url"] for s in parser.sections()}
    assert sections == {
        "third_party/rclib": "https://github.com/hrshtst/rclib.git",
        "third_party/skelarm": "https://github.com/hrshtst/skelarm.git",
        "third_party/rtctrl": "https://github.com/hrshtst/rtctrl.git",
    }


@pytest.mark.parametrize("name", SUBMODULES)
def test_recorded_pin_matches_notice_inventory(name: str) -> None:
    """The gitlink recorded in HEAD equals the commit declared in THIRD_PARTY_NOTICES.md."""
    recorded = _recorded_gitlink(f"third_party/{name}")
    assert SHA_RE.match(recorded)
    assert recorded == _notice_pins()[name], f"advance the {name} row in THIRD_PARTY_NOTICES.md with the pin"


@pytest.mark.parametrize("name", SUBMODULES)
def test_checked_out_submodule_matches_recorded_pin(name: str) -> None:
    """The working tree has the pinned commit checked out (not a stale or drifted revision)."""
    path = REPO_ROOT / "third_party" / name
    if not (path / ".git").exists():
        pytest.skip(f"{name} submodule not initialized")
    assert _git("rev-parse", "HEAD", cwd=path) == _recorded_gitlink(f"third_party/{name}")


def test_rclib_nested_submodules_are_initialized() -> None:
    """The Eigen, pybind11, and Catch2 submodules of rclib are present (required to build its wheel)."""
    root = REPO_ROOT / "third_party" / "rclib"
    if not (root / ".git").exists():
        pytest.skip("rclib submodule not initialized")
    for nested in ("eigen", "pybind11", "catch2"):
        assert any((root / "cpp_core" / "third_party" / nested).iterdir()), f"{nested} is empty"
