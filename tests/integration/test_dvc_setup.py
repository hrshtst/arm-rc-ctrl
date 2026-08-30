# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-011: DVC cache and remote live below the storage root; Git keeps only portable metadata."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from arm_rc_ctrl.data.dvc import DvcConfigError, configure_local_dvc, read_local_config, verify_local_dvc
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
}


def _run(args: list[str], cwd: Path) -> str:
    env = {**os.environ, **GIT_ENV, "DVC_NO_ANALYTICS": "1", "HOME": str(cwd.parent)}
    result = subprocess.run(args, cwd=cwd, env=env, check=False, capture_output=True, text=True)
    assert result.returncode == 0, f"{' '.join(args)}: {result.stderr}"
    return result.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway Git repository with the project's .gitignore and a fresh `dvc init`."""
    root = tmp_path / "repo"
    root.mkdir()
    _run(["git", "init", "-q", "-b", "main"], root)
    shutil.copy(REPO_ROOT / ".gitignore", root / ".gitignore")
    _run([sys.executable, "-m", "dvc", "init", "-q"], root)
    _run([sys.executable, "-m", "dvc", "config", "core.analytics", "false"], root)
    _run(["git", "add", "-A"], root)
    _run(["git", "commit", "-q", "-m", "init"], root)
    return root


@pytest.fixture
def store(tmp_path: Path) -> StorageRoot:
    """Storage root outside every repository."""
    root = tmp_path / "store"
    root.mkdir()
    return StorageRoot(root, repositories=(REPO_ROOT,))


def test_setup_maps_cache_and_remote_below_the_storage_root(repo: Path, store: StorageRoot) -> None:
    """config.local points the cache and the default remote at <root>/dvc-cache and <root>/dvc-store."""
    config = configure_local_dvc(repo, store)
    assert config.cache_dir == store.root / "dvc-cache"
    assert config.remote_name == "store"
    assert config.remote_url == store.root / "dvc-store"
    assert config.cache_dir.is_dir()
    assert config.remote_url.is_dir()
    assert read_local_config(repo) == config
    assert verify_local_dvc(repo, store) == config
    assert not (repo / ".dvc" / "cache").exists()


def test_git_sees_only_portable_metadata(repo: Path, store: StorageRoot) -> None:
    """After setup, config.local is ignored and untracked; the tracked config has no paths."""
    configure_local_dvc(repo, store)
    status = _run(["git", "status", "--porcelain", "--ignored"], repo)
    assert "!! .dvc/config.local" in status
    assert "?? .dvc/config.local" not in status
    tracked = (repo / ".dvc" / "config").read_text()
    assert str(store.root) not in tracked
    assert "analytics = false" in tracked


def test_dvc_add_uses_the_external_cache_and_keeps_payloads_out_of_git(repo: Path, store: StorageRoot) -> None:
    """Adding a payload leaves only a .dvc metafile for Git; the bytes go to the external cache/remote."""
    configure_local_dvc(repo, store)
    payload = repo / "data" / "demo.sklog.npz"
    payload.parent.mkdir()
    payload.write_bytes(b"payload-bytes")
    _run([sys.executable, "-m", "dvc", "add", "-q", "data/demo.sklog.npz"], repo)
    _run([sys.executable, "-m", "dvc", "push", "-q"], repo)
    assert (repo / "data" / "demo.sklog.npz.dvc").is_file()
    ignored = subprocess.run(["git", "check-ignore", "-q", "data/demo.sklog.npz"], cwd=repo, check=False)
    assert ignored.returncode == 0  # the project .gitignore already excludes payload formats
    cache_files = [p for p in (store.root / "dvc-cache").rglob("*") if p.is_file()]
    remote_files = [p for p in (store.root / "dvc-store").rglob("*") if p.is_file()]
    assert any(p.read_bytes() == b"payload-bytes" for p in cache_files)
    assert any(p.read_bytes() == b"payload-bytes" for p in remote_files)
    _run(["git", "add", "-A"], repo)
    staged = _run(["git", "diff", "--cached", "--name-only"], repo).split()
    assert "data/demo.sklog.npz.dvc" in staged
    assert "data/demo.sklog.npz" not in staged
    assert ".dvc/config.local" not in staged


def test_verify_detects_mismapped_or_missing_configuration(repo: Path, store: StorageRoot, tmp_path: Path) -> None:
    """Verification fails on a missing config.local, a cache elsewhere, or paths in the tracked config."""
    with pytest.raises(DvcConfigError, match="does not exist; run"):
        verify_local_dvc(repo, store)
    configure_local_dvc(repo, store)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _run([sys.executable, "-m", "dvc", "config", "--local", "cache.dir", str(elsewhere)], repo)
    with pytest.raises(DvcConfigError, match=r"cache\.dir is .* expected"):
        verify_local_dvc(repo, store)
    configure_local_dvc(repo, store)
    _run([sys.executable, "-m", "dvc", "config", "cache.dir", str(elsewhere)], repo)  # tracked config, wrong place
    with pytest.raises(DvcConfigError, match="contains a path"):
        verify_local_dvc(repo, store)


def test_setup_requires_an_initialized_dvc_repository(tmp_path: Path, store: StorageRoot) -> None:
    """Without `dvc init` there is nothing to configure."""
    with pytest.raises(DvcConfigError, match="not a DVC repository"):
        configure_local_dvc(tmp_path, store)


def test_project_tracked_dvc_metadata_is_portable() -> None:
    """The committed .dvc/config carries no paths and analytics are off."""
    tracked = (REPO_ROOT / ".dvc" / "config").read_text()
    assert "analytics = false" in tracked
    assert "/" not in tracked.replace("[core]", "")
    assert (REPO_ROOT / ".dvc" / ".gitignore").is_file()
    assert (REPO_ROOT / ".dvcignore").is_file()
