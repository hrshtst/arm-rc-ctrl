# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M0-005: the C++ skeleton declares a version consistent with Python and pins Catch2 by commit."""

from __future__ import annotations

import re
from pathlib import Path

import arm_rc_ctrl

REPO_ROOT = Path(__file__).resolve().parents[2]
CMAKELISTS = REPO_ROOT / "cpp" / "CMakeLists.txt"


def test_cpp_project_version_matches_python_release_segment() -> None:
    """``project(... VERSION x.y.z)`` equals the release part of the Python version."""
    pattern = r"^project\(arm_rc_ctrl VERSION (\d+\.\d+\.\d+) LANGUAGES CXX\)$"
    match = re.search(pattern, CMAKELISTS.read_text(), re.MULTILINE)
    assert match is not None, "cpp/CMakeLists.txt must declare project(arm_rc_ctrl VERSION x.y.z LANGUAGES CXX)"
    python_release = re.match(r"^\d+\.\d+\.\d+", arm_rc_ctrl.__version__)
    assert python_release is not None
    assert match.group(1) == python_release.group(0)


def test_catch2_is_pinned_by_immutable_commit() -> None:
    """Catch2 is fetched by a 40-hex commit, not only by a movable tag."""
    text = CMAKELISTS.read_text()
    match = re.search(r"FetchContent_Declare\(Catch2\s+GIT_REPOSITORY (\S+)\s+GIT_TAG ([0-9a-f]{40})\)", text)
    assert match is not None, "Catch2 must be declared with GIT_REPOSITORY and a 40-hex GIT_TAG"
    assert match.group(1) == "https://github.com/catchorg/Catch2.git"


def test_cpp_sources_carry_spdx_headers() -> None:
    """Every original C++ source and header carries the GPL-3.0-only SPDX identifier."""
    sources = [*(REPO_ROOT / "cpp").rglob("*.cpp"), *(REPO_ROOT / "cpp").rglob("*.hpp")]
    assert sources
    for path in sources:
        assert "SPDX-License-Identifier: GPL-3.0-only" in path.read_text(), path
