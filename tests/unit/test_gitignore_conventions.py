# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M0-007: payloads and machine state stay untracked while records remain trackable."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Representative paths that must never enter Git.
IGNORED = [
    "demo.sklog.npz",
    "data/records/raw/abc123/demo.sklog.npz",
    "data/processed/dataset/samples.npz",
    "tests/scratch/output.npz",
    "runs/run-0001/log.npz",
    "models/recipe/weights.npy",
    "mlflow/mlruns/0/meta.yaml",
    "mlruns/0/meta.yaml",
    "optuna/study.db",
    "optuna/study.sqlite3",
    "dvc-cache/00/abcdef",
    "dvc-store/00/abcdef",
    ".dvc/config.local",
    ".dvc/cache/00/abcdef",
    ".dvc/tmp/lock",
    "configs/storage.toml",
    "storage.toml",
    ".envrc",
    ".venv/bin/python",
    ".nox/tests/bin/python",
    "src/arm_rc_ctrl/__pycache__/x.pyc",
    ".coverage",
    "build/CMakeCache.txt",
    "build-debug/CMakeCache.txt",
    "cpp/build/CMakeCache.txt",
    "dist/arm_rc_ctrl-0.1.0.tar.gz",
]

# Representative paths that must remain trackable.
TRACKABLE = [
    "data/catalog.toml",
    "data/records/raw/abc123.toml",
    "data/records/processed/def456.toml",
    "data/records/runs/run-0001.toml",
    "data/records/models/model-0001.toml",
    "data/records/raw/.gitkeep",
    "data/demo.sklog.npz.dvc",
    "dvc.yaml",
    "dvc.lock",
    "configs/storage.example.toml",
    "configs/robots/planar_2dof.toml",
    "tests/fixtures/tiny_demo.sklog.npz",
    "tests/fixtures/processed/samples.npz",
    "docs/experiments/task_1a/report.md",
    "src/arm_rc_ctrl/__init__.py",
    "cpp/CMakeLists.txt",
    "cpp/tests/version_test.cpp",
]


def _git_ignores(path: str) -> bool:
    """Return whether Git's ignore rules exclude ``path`` (which need not exist)."""
    result = subprocess.run(  # noqa: S603
        ["git", "check-ignore", "-q", "--no-index", "--", path],  # noqa: S607
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        msg = f"git check-ignore failed for {path!r}: {result.stderr.strip()}"
        raise RuntimeError(msg)
    return result.returncode == 0


@pytest.fixture(scope="module", autouse=True)
def _require_git_worktree() -> None:
    result = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "--is-inside-work-tree"],  # noqa: S607
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or result.stdout.strip() != "true":
        pytest.skip("not running inside the project Git worktree")


@pytest.mark.parametrize("path", IGNORED)
def test_payloads_and_machine_state_are_ignored(path: str) -> None:
    """Payloads, DVC data, machine config, experiment state, builds, and runs are ignored."""
    assert _git_ignores(path), f"{path} must be ignored by .gitignore"


@pytest.mark.parametrize("path", TRACKABLE)
def test_records_configs_and_fixtures_remain_trackable(path: str) -> None:
    """Artifact records, DVC metafiles, configs, docs, and test fixtures stay trackable."""
    assert not _git_ignores(path), f"{path} must remain trackable"
