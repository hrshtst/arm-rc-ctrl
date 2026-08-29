# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M0-003: rclib and skelarm are importable from pinned submodule builds, and revisions are recorded."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from arm_rc_ctrl import dependencies
from arm_rc_ctrl.repo import repository_root

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def test_rclib_imports_and_exposes_esn() -> None:
    """The pinned rclib build imports and provides the ESN entry point used by the plan."""
    import rclib

    assert callable(rclib.ESN)
    assert hasattr(rclib.reservoirs, "RandomSparse")
    assert hasattr(rclib.readouts, "Ridge")


def test_skelarm_imports_headless() -> None:
    """The pinned skelarm build imports without a display (PyQt6 offscreen)."""
    import skelarm

    assert callable(skelarm.Skeleton)


def test_installed_versions_match_submodule_pyprojects() -> None:
    """Installed rclib/skelarm versions equal the versions declared by the pinned submodules."""
    versions = dependencies.installed_versions()
    assert set(versions) == {"rclib", "skelarm", "numpy"}
    assert versions["rclib"] == dependencies.submodule_version("rclib")
    assert versions["skelarm"] == dependencies.submodule_version("skelarm")


def test_installed_sources_are_not_stale() -> None:
    """The installed Python sources equal the submodule checkout (else `uv sync --reinstall-package`)."""
    stale = dependencies.stale_installs()
    assert stale == {}, f"stale installs, run `uv sync --reinstall-package rclib --reinstall-package skelarm`: {stale}"


def test_submodule_revisions_are_recorded_and_checked_out(record_property: Callable[[str, object], None]) -> None:
    """Every submodule reports a 40-hex recorded commit and, when initialized, the same checked-out commit."""
    revisions = dependencies.submodule_revisions()
    assert [r.name for r in revisions] == list(dependencies.SUBMODULES)
    for revision in revisions:
        assert SHA_RE.match(revision.recorded), revision
        assert revision.path == f"third_party/{revision.name}"
        if revision.checked_out is not None:
            assert revision.matches_pin, revision
        record_property(f"submodule.{revision.name}", revision.recorded)


def test_source_digest_is_order_independent_and_content_sensitive(tmp_path: Path) -> None:
    """Digests depend on relative paths and contents only."""
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "a.py").write_text("A = 1\n")
    (pkg / "sub" / "b.py").write_text("B = 2\n")
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"ignored")
    first = dependencies.source_digest(pkg)
    (pkg / "__pycache__" / "z.py").write_text("ignored too\n")
    assert dependencies.source_digest(pkg) == first
    (pkg / "sub" / "b.py").write_text("B = 3\n")
    assert dependencies.source_digest(pkg) != first


def test_python_source_dir_rejects_non_python_submodule() -> None:
    """The rtctrl submodule is C++ only and has no Python package directory."""
    with pytest.raises(ValueError, match="rtctrl"):
        dependencies.python_source_dir("rtctrl")


def test_repository_root_finds_checkout() -> None:
    """The repository root is the directory holding pyproject.toml and src/arm_rc_ctrl."""
    root = repository_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "arm_rc_ctrl" / "__init__.py").is_file()
    assert repository_root(root / "tests" / "unit") == root
