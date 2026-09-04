# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M0-014: storage-root precedence, armrc:// resolution, access checks, and no repository fallback."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest

from arm_rc_ctrl.config import ConfigError, load_config
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import (
    BUCKETS,
    DEFAULT_ROOT,
    ENV_VAR,
    AccessMode,
    ArtifactUri,
    InvalidArtifactUriError,
    StorageAccessError,
    StorageConfig,
    StorageRoot,
    StorageRootError,
    open_storage,
    resolve_storage_root,
    storage_config_path,
)

REPO_ROOT = repository_root()
IS_ROOT_USER = os.geteuid() == 0


def _write_config(config_home: Path, root: Path, extra: str = "") -> Path:
    path = config_home / "arm-rc-ctrl" / "storage.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'[storage]\nroot = "{root}"\n{extra}')
    return path


@pytest.fixture
def store(tmp_path: Path) -> StorageRoot:
    """An empty, valid storage root outside any repository."""
    root = tmp_path / "store"
    root.mkdir()
    return StorageRoot(root, repositories=(REPO_ROOT,))


# --- precedence ---------------------------------------------------------------


def test_environment_variable_takes_precedence(tmp_path: Path) -> None:
    """A non-empty environment variable wins over an existing config file."""
    _write_config(tmp_path / "xdg", tmp_path / "from-config")
    env = {ENV_VAR: str(tmp_path / "from-env"), "XDG_CONFIG_HOME": str(tmp_path / "xdg")}
    resolved = resolve_storage_root(env)
    assert resolved.path == tmp_path / "from-env"
    assert resolved.source == "environment"


def test_config_file_is_used_when_environment_is_unset_or_empty(tmp_path: Path) -> None:
    """The XDG config is consulted when the variable is absent or empty."""
    config = _write_config(tmp_path / "xdg", tmp_path / "from-config")
    for env in ({"XDG_CONFIG_HOME": str(tmp_path / "xdg")}, {ENV_VAR: "", "XDG_CONFIG_HOME": str(tmp_path / "xdg")}):
        resolved = resolve_storage_root(env)
        assert resolved.path == tmp_path / "from-config"
        assert resolved.source == "config"
        assert resolved.detail == str(config)


def test_default_root_is_used_last(tmp_path: Path) -> None:
    """Without variable or config, the documented default applies (unvalidated)."""
    resolved = resolve_storage_root({"XDG_CONFIG_HOME": str(tmp_path / "empty")})
    assert resolved.path == DEFAULT_ROOT
    assert resolved.source == "default"


def test_xdg_config_home_falls_back_to_home_dot_config(tmp_path: Path) -> None:
    """``storage.toml`` is under ``$XDG_CONFIG_HOME`` or else ``$HOME/.config``."""
    assert storage_config_path({"XDG_CONFIG_HOME": "/xdg"}) == Path("/xdg/arm-rc-ctrl/storage.toml")
    assert storage_config_path({"HOME": str(tmp_path)}) == tmp_path / ".config" / "arm-rc-ctrl" / "storage.toml"
    assert storage_config_path({"XDG_CONFIG_HOME": "", "HOME": "/h"}) == Path("/h/.config/arm-rc-ctrl/storage.toml")
    with pytest.raises(StorageRootError, match="neither XDG_CONFIG_HOME nor HOME"):
        storage_config_path({})


def test_invalid_config_file_is_an_error_not_a_fallback(tmp_path: Path) -> None:
    """A malformed storage.toml fails loudly instead of silently using the default."""
    _write_config(tmp_path / "xdg", tmp_path / "root", extra="extra = 1\n")
    with pytest.raises(ConfigError, match=r"storage: unknown key\(s\) 'extra'"):
        resolve_storage_root({"XDG_CONFIG_HOME": str(tmp_path / "xdg")})


def test_relative_environment_root_is_rejected() -> None:
    """The environment variable must be absolute."""
    with pytest.raises(StorageRootError, match="must be an absolute path"):
        resolve_storage_root({ENV_VAR: "relative/store"})


def test_committed_example_config_matches_schema() -> None:
    """configs/storage.example.toml stays loadable and documents the default root."""
    config = load_config(REPO_ROOT / "configs" / "storage.example.toml", StorageConfig)
    assert config.storage.root == DEFAULT_ROOT


# --- root validation ----------------------------------------------------------


def test_missing_root_fails_and_is_never_created(tmp_path: Path) -> None:
    """An absent root is an error; opening storage does not create it."""
    missing = tmp_path / "absent"
    with pytest.raises(StorageRootError, match="does not exist"):
        StorageRoot(missing, repositories=())
    with pytest.raises(StorageRootError, match=r"does not exist.*configured via environment"):
        open_storage({ENV_VAR: str(missing)})
    assert not missing.exists()


def test_root_must_be_a_directory(tmp_path: Path) -> None:
    """A file cannot serve as the storage root."""
    file_root = tmp_path / "file"
    file_root.write_text("x")
    with pytest.raises(StorageRootError, match="not a directory"):
        StorageRoot(file_root, repositories=())


def test_root_inside_repository_worktree_is_rejected(tmp_path: Path) -> None:
    """Roots inside any known repository worktree are refused, so there is no repository fallback."""
    repo = tmp_path / "repo"
    (repo / "store").mkdir(parents=True)
    with pytest.raises(StorageRootError, match="inside the repository worktree"):
        StorageRoot(repo / "store", repositories=(repo,))
    with pytest.raises(StorageRootError, match="inside the repository worktree"):
        StorageRoot(repo, repositories=(repo,))
    # The real checkout is always among the known repositories.
    with pytest.raises(StorageRootError, match="inside the repository worktree"):
        open_storage({ENV_VAR: str(REPO_ROOT / "data")})


def test_root_symlink_is_canonicalized(tmp_path: Path) -> None:
    """A root given through a symlink resolves to the real directory."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    assert StorageRoot(link, repositories=()).root == real.resolve()


@pytest.mark.skipif(IS_ROOT_USER, reason="permission bits are not enforced for root")
def test_unreadable_root_is_rejected(tmp_path: Path) -> None:
    """A root without read/search permission is unusable."""
    root = tmp_path / "locked"
    root.mkdir()
    root.chmod(0o000)
    try:
        with pytest.raises(StorageRootError, match="not readable"):
            StorageRoot(root, repositories=())
    finally:
        root.chmod(0o700)


# --- URIs ---------------------------------------------------------------------


@pytest.mark.parametrize("bucket", BUCKETS)
def test_every_documented_bucket_parses(bucket: str) -> None:
    """Every documented layout bucket is a valid URI root."""
    uri = ArtifactUri.parse(f"armrc://{bucket}")
    assert uri == ArtifactUri(bucket)
    assert uri.relative_path == Path(bucket)
    assert str(uri) == f"armrc://{bucket}"


def test_uri_round_trip_with_segments() -> None:
    """Segments are preserved and re-rendered identically."""
    uri = ArtifactUri.parse("armrc://raw/demo-01/demo.sklog.npz")
    assert uri == ArtifactUri("raw", ("demo-01", "demo.sklog.npz"))
    assert uri.relative_path == Path("raw/demo-01/demo.sklog.npz")
    assert str(uri) == "armrc://raw/demo-01/demo.sklog.npz"


@pytest.mark.parametrize(
    "uri",
    [
        "file:///raw/x",
        "armrc:/raw/x",
        "armrc://",
        "armrc:///raw/x",
        "armrc://scratch/x",
        "armrc://raw/../processed/x",
        "armrc://raw/./x",
        "armrc://raw//x",
        "armrc://raw/x/",
        "armrc://raw/.hidden",
        "armrc://raw/a b",
        "armrc://raw/x?y=1",
        "armrc://raw/x#frag",
        "armrc://raw/a\\b",
        "/absolute/path",
    ],
)
def test_invalid_uris_are_rejected(uri: str) -> None:
    """Traversal, absolute components, unknown buckets, and foreign schemes are refused."""
    with pytest.raises(InvalidArtifactUriError):
        ArtifactUri.parse(uri)


def test_uri_constructor_validates_too() -> None:
    """Direct construction applies the same rules as parsing."""
    with pytest.raises(InvalidArtifactUriError, match="unknown bucket"):
        ArtifactUri("tmp")
    with pytest.raises(InvalidArtifactUriError, match="invalid path segment"):
        ArtifactUri("raw", ("..",))


# --- path resolution ----------------------------------------------------------


def test_write_creates_bucket_and_parents_under_root(store: StorageRoot) -> None:
    """Write resolution creates the bucket and intermediate directories only."""
    path = store.path("armrc://runs/run-01/log.npz", mode="write")
    assert path == store.root / "runs" / "run-01" / "log.npz"
    assert path.parent.is_dir()
    assert not path.exists()
    assert store.uri_for(path) == ArtifactUri("runs", ("run-01", "log.npz"))


def test_read_requires_existing_readable_target(store: StorageRoot) -> None:
    """Read resolution fails on missing targets and succeeds once the payload exists."""
    with pytest.raises(StorageAccessError, match="does not exist"):
        store.path("armrc://raw/demo/demo.sklog.npz", mode="read")
    target = store.path("armrc://raw/demo/demo.sklog.npz", mode="write")
    target.write_bytes(b"payload")
    assert store.path("armrc://raw/demo/demo.sklog.npz", mode="read") == target


@pytest.mark.skipif(IS_ROOT_USER, reason="permission bits are not enforced for root")
def test_unreadable_payload_and_unwritable_root(store: StorageRoot) -> None:
    """Permission failures are reported as access errors."""
    target = store.path("armrc://raw/demo/demo.sklog.npz", mode="write")
    target.write_bytes(b"payload")
    target.chmod(0o000)
    try:
        with pytest.raises(StorageAccessError, match="not readable"):
            store.path("armrc://raw/demo/demo.sklog.npz", mode="read")
    finally:
        target.chmod(0o600)
    store.root.chmod(0o500)
    try:
        with pytest.raises(StorageAccessError, match="not writable"):
            store.path("armrc://models/new/recipe.toml", mode="write")
    finally:
        store.root.chmod(0o700)


def test_symlink_escaping_root_is_refused(store: StorageRoot, tmp_path: Path) -> None:
    """A bucket or entry symlinked outside the root is refused in both modes."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.npz").write_bytes(b"x")
    (store.root / "raw").symlink_to(outside)
    with pytest.raises(StorageAccessError, match="outside the storage root"):
        store.path("armrc://raw/leak.npz", mode="read")
    with pytest.raises(StorageAccessError, match="outside the storage root"):
        store.path("armrc://raw/new.npz", mode="write")


def test_unknown_access_mode_is_rejected_without_side_effects(store: StorageRoot) -> None:
    """A mode other than read/write is an error and creates nothing."""
    with pytest.raises(ValueError, match="unknown access mode 'append'"):
        store.path("armrc://runs/new/log.npz", mode=cast("AccessMode", "append"))
    assert not (store.root / "runs").exists()


def test_uri_for_rejects_paths_outside_root(store: StorageRoot, tmp_path: Path) -> None:
    """Only paths beneath the root have logical URIs."""
    with pytest.raises(StorageAccessError, match="not under the storage root"):
        store.uri_for(tmp_path / "elsewhere")
    with pytest.raises(InvalidArtifactUriError, match="unknown bucket"):
        store.uri_for(store.root / "scratch" / "x")


def test_open_storage_validates_resolved_root(tmp_path: Path) -> None:
    """open_storage combines resolution and validation."""
    root = tmp_path / "store"
    root.mkdir()
    assert open_storage({ENV_VAR: str(root)}).root == root.resolve()
