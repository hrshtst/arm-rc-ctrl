# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M0-011: provenance captures commits, dirty state, digests, artifacts, platform, and seeds."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from arm_rc_ctrl import __version__
from arm_rc_ctrl.config import ConfigError
from arm_rc_ctrl.dependencies import SubmoduleRevision
from arm_rc_ctrl.provenance import (
    ArtifactMismatchError,
    ArtifactReference,
    DirtyWorktreeError,
    ProvenanceRecord,
    artifact_reference,
    canonical_json,
    collect_provenance,
    config_digest,
    require_clean_for_confirmatory,
    sha256_bytes,
    sha256_file,
    verify_artifact,
    worktree_state,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageAccessError, StorageRoot

REPO_ROOT = repository_root()
FIXED_TIME = datetime(2026, 8, 29, 12, 0, 0, tzinfo=UTC)


@dataclass(frozen=True)
class Inner:
    """Nested configuration table."""

    gain: float
    path: Path


@dataclass(frozen=True)
class Config:
    """Example resolved configuration."""

    name: str
    steps: int
    inner: Inner
    weights: tuple[float, ...]


CONFIG = Config("smoke", 3, Inner(0.5, Path("/data/x.npz")), (1.0, 2.0))


@pytest.fixture
def store(tmp_path: Path) -> StorageRoot:
    """Empty storage root outside the repository."""
    root = tmp_path / "store"
    root.mkdir()
    return StorageRoot(root, repositories=(REPO_ROOT,))


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A throwaway repository with one committed file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
        "HOME": str(tmp_path),
    }
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True, env=env)
    (repo / "tracked.txt").write_text("v1\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True, env=env)
    return repo


# --- digests ------------------------------------------------------------------


def test_sha256_helpers_agree_with_known_vector(tmp_path: Path) -> None:
    """Both helpers compute the standard SHA-256 (empty input vector)."""
    empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert sha256_bytes(b"") == empty
    path = tmp_path / "empty"
    path.write_bytes(b"")
    assert sha256_file(path) == empty
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_canonical_json_is_order_independent_and_rejects_nan() -> None:
    """Key order does not change the text; NaN/Inf are refused."""
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1}) == '{"a":[1,2],"b":1}'
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json({"x": float("nan")})


def test_config_digest_covers_resolved_dataclass() -> None:
    """The digest is over the plain resolved mapping (paths as strings, tuples as lists)."""
    text, digest = config_digest(CONFIG)
    assert json.loads(text) == {
        "name": "smoke",
        "steps": 3,
        "inner": {"gain": 0.5, "path": "/data/x.npz"},
        "weights": [1.0, 2.0],
    }
    assert digest == sha256_bytes(text.encode())
    assert config_digest(json.loads(text)) == (text, digest)


# --- artifacts ----------------------------------------------------------------


def test_artifact_reference_validates_fields() -> None:
    """URI, digest format, and size are validated on construction."""
    ref = ArtifactReference("armrc://raw/a/demo.sklog.npz", "0" * 64, 0)
    assert ref.size == 0
    with pytest.raises(ValueError, match="sha256 must be 64 lowercase hex"):
        ArtifactReference("armrc://raw/a", "abc", 1)
    with pytest.raises(ValueError, match="size must be non-negative"):
        ArtifactReference("armrc://raw/a", "0" * 64, -1)
    with pytest.raises(ValueError, match="unknown bucket"):
        ArtifactReference("armrc://tmp/a", "0" * 64, 1)


def test_artifact_reference_and_verification(store: StorageRoot) -> None:
    """References are computed from real payloads and verification catches size and content changes."""
    path = store.path("armrc://raw/demo-01/demo.sklog.npz", mode="write")
    path.write_bytes(b"payload")
    ref = artifact_reference(store, "armrc://raw/demo-01/demo.sklog.npz")
    assert ref == ArtifactReference("armrc://raw/demo-01/demo.sklog.npz", sha256_bytes(b"payload"), 7)
    assert verify_artifact(store, ref) == path

    path.write_bytes(b"payloaX")
    with pytest.raises(ArtifactMismatchError, match="sha256"):
        verify_artifact(store, ref)
    path.write_bytes(b"pay")
    with pytest.raises(ArtifactMismatchError, match="size 3 != recorded 7"):
        verify_artifact(store, ref)
    path.unlink()
    with pytest.raises(StorageAccessError, match="does not exist"):
        verify_artifact(store, ref)


# --- git state ----------------------------------------------------------------


def test_worktree_state_detects_tracked_and_untracked_changes(git_repo: Path) -> None:
    """Clean after commit; dirty on tracked modification, staged change, or untracked file."""
    commit, dirty = worktree_state(git_repo)
    assert len(commit) == 40
    assert dirty is False

    (git_repo / "tracked.txt").write_text("v2\n")
    assert worktree_state(git_repo) == (commit, True)
    subprocess.run(["git", "checkout", "-q", "--", "tracked.txt"], cwd=git_repo, check=True)
    assert worktree_state(git_repo)[1] is False

    (git_repo / "untracked.txt").write_text("x\n")
    assert worktree_state(git_repo)[1] is True
    (git_repo / ".gitignore").write_text("untracked.txt\n")
    # An untracked .gitignore itself counts as dirty; ignored files do not.
    assert worktree_state(git_repo)[1] is True


# --- record -------------------------------------------------------------------


def test_collect_provenance_captures_everything(store: StorageRoot) -> None:
    """The record holds commit, dirty flag, submodules, lock/config digests, artifacts, seeds, and platform."""
    path = store.path("armrc://processed/ds-01/samples.npz", mode="write")
    path.write_bytes(b"data")
    ref = artifact_reference(store, "armrc://processed/ds-01/samples.npz")
    record = collect_provenance(
        CONFIG,
        seeds={"reservoir": 1, "scenario": 42},
        artifacts=[ref],
        now=FIXED_TIME,
        env={"OMP_NUM_THREADS": "1", "PATH": "/ignored"},
    )
    assert record.schema_version == 1
    assert record.created_at == "2026-08-29T12:00:00+00:00"
    assert record.project_commit == worktree_state(REPO_ROOT)[0]
    assert isinstance(record.project_dirty, bool)
    assert [s.name for s in record.submodules] == ["rclib", "skelarm", "rtctrl"]
    assert record.lock_sha256 == sha256_file(REPO_ROOT / "uv.lock")
    assert (record.config_json, record.config_sha256) == config_digest(CONFIG)
    assert record.config["inner"] == {"gain": 0.5, "path": "/data/x.npz"}
    assert record.artifacts == (ref,)
    assert record.seeds == {"reservoir": 1, "scenario": 42}
    assert record.platform.packages["arm-rc-ctrl"] == __version__
    assert set(record.platform.packages) >= {"rclib", "skelarm", "numpy"}
    assert record.platform.thread_environment == {"OMP_NUM_THREADS": "1"}
    assert record.platform.python
    assert record.exploratory is False


def test_collect_provenance_rejects_naive_timestamp_and_bad_seeds() -> None:
    """Timestamps must be timezone-aware and seeds non-negative integers."""
    with pytest.raises(ValueError, match="timezone-aware"):
        collect_provenance(CONFIG, seeds={}, now=datetime(2026, 1, 1))  # noqa: DTZ001
    with pytest.raises(ValueError, match="seed 'x' must be a non-negative integer"):
        collect_provenance(CONFIG, seeds={"x": -1}, now=FIXED_TIME)


def test_timestamp_is_normalized_to_utc() -> None:
    """Non-UTC aware timestamps are converted."""
    jst = datetime(2026, 8, 29, 21, 0, 0, tzinfo=timezone(timedelta(hours=9)))
    assert collect_provenance(CONFIG, seeds={}, now=jst).created_at == "2026-08-29T12:00:00+00:00"


def test_record_round_trips_through_mapping_and_json() -> None:
    """to_mapping/from_mapping and to_json/from_json reproduce an equal record."""
    record = collect_provenance(CONFIG, seeds={"a": 7}, now=FIXED_TIME, env={})
    mapping = record.to_mapping()
    assert isinstance(mapping["submodules"], list)
    assert ProvenanceRecord.from_mapping(mapping) == record
    assert ProvenanceRecord.from_json(record.to_json()) == record
    assert record.to_json() == canonical_json(mapping)


def test_record_mapping_is_strictly_validated() -> None:
    """Unknown or mistyped fields in a stored record are rejected on load."""
    record = collect_provenance(CONFIG, seeds={}, now=FIXED_TIME, env={})
    mapping = record.to_mapping()
    with pytest.raises(ConfigError, match=r"unknown key\(s\) 'extra'"):
        ProvenanceRecord.from_mapping({**mapping, "extra": 1})
    with pytest.raises(ConfigError, match=r"seeds\.a: expected integer, got string"):
        ProvenanceRecord.from_mapping({**mapping, "seeds": {"a": "7"}})
    bad = json.loads(record.to_json())
    bad["submodules"][0]["recorded"] = 1
    with pytest.raises(ConfigError, match=r"submodules\[0\]\.recorded: expected string, got integer"):
        ProvenanceRecord.from_mapping(bad)


# --- confirmatory policy ------------------------------------------------------


def _record(*, dirty: bool, submodules: tuple[SubmoduleRevision, ...], exploratory: bool = False) -> ProvenanceRecord:
    base = collect_provenance(CONFIG, seeds={}, now=FIXED_TIME, env={})
    return dataclasses.replace(base, project_dirty=dirty, submodules=submodules, exploratory=exploratory)


def test_clean_record_passes_confirmatory_policy() -> None:
    """Clean project and pinned, clean submodules are accepted."""
    sub = SubmoduleRevision("rclib", "third_party/rclib", "a" * 40, "a" * 40, dirty=False)
    uninitialized = SubmoduleRevision("rtctrl", "third_party/rtctrl", "b" * 40, None, dirty=None)
    record = _record(dirty=False, submodules=(sub, uninitialized))
    assert record.is_clean
    require_clean_for_confirmatory(record)


@pytest.mark.parametrize(
    ("dirty", "sub", "message"),
    [
        (True, SubmoduleRevision("rclib", "third_party/rclib", "a" * 40, "a" * 40, dirty=False), "worktree is dirty"),
        (False, SubmoduleRevision("rclib", "third_party/rclib", "a" * 40, "a" * 40, dirty=True), "rclib is dirty"),
        (False, SubmoduleRevision("rclib", "third_party/rclib", "a" * 40, "c" * 40, dirty=False), "not its pin"),
    ],
)
def test_dirty_or_drifted_record_is_rejected_unless_exploratory(
    *, dirty: bool, sub: SubmoduleRevision, message: str
) -> None:
    """Dirty worktrees and drifted submodules fail confirmatory use; exploratory records pass."""
    record = _record(dirty=dirty, submodules=(sub,))
    assert not record.is_clean
    with pytest.raises(DirtyWorktreeError, match=message):
        require_clean_for_confirmatory(record)
    require_clean_for_confirmatory(_record(dirty=dirty, submodules=(sub,), exploratory=True))
