# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M0-003/M0-011: installed rclib/skelarm builds are tied to the pinned submodules by a build manifest."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from arm_rc_ctrl import dependencies
from arm_rc_ctrl.config import ConfigError
from arm_rc_ctrl.dependencies import (
    BuildIdentity,
    BuildIdentityError,
    BuildManifest,
    check_build,
    current_build_identity,
    installed_extension_digest,
    installed_is_editable,
    manifest_path,
    read_manifest,
    stamp_builds,
    verify_builds,
)
from arm_rc_ctrl.repo import repository_root

REPO_ROOT = repository_root()
FIXED_TIME = datetime(2026, 8, 29, 12, 0, 0, tzinfo=UTC)
COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
DIGEST_1 = "1" * 64
DIGEST_2 = "2" * 64


def _identity(**changes: object) -> BuildIdentity:
    base = BuildIdentity(
        name="rclib",
        version="0.1.0",
        source_commit=COMMIT_A,
        source_dirty=False,
        editable=False,
        python_sources_sha256=DIGEST_1,
        extension_sha256=DIGEST_2,
    )
    return dataclasses.replace(base, **changes)


# --- pure comparison ----------------------------------------------------------


def test_consistent_entry_has_no_problems() -> None:
    """A stamped identity equal to the present state verifies."""
    assert check_build(_identity(), _identity()) == []


def test_missing_entry_is_unknown_identity() -> None:
    """Without a manifest entry the build identity is unknown and the rebuild command is named."""
    (problem,) = check_build(None, _identity())
    assert "unknown build identity" in problem
    assert "rebuild" in problem


@pytest.mark.parametrize(
    ("current", "fragment"),
    [
        (_identity(source_commit=COMMIT_B), "pin advanced"),
        (_identity(source_dirty=True), "uncommitted changes"),
        (_identity(version="0.2.0"), "stamped version 0.1.0 != installed version 0.2.0"),
        (_identity(python_sources_sha256=DIGEST_2), "Python sources differ"),
        (_identity(extension_sha256=DIGEST_1), "compiled extension differs"),
        (_identity(extension_sha256=None), "compiled extension differs"),
    ],
)
def test_mismatches_are_reported(current: BuildIdentity, fragment: str) -> None:
    """Every field that identifies a build is compared, including a C++-only change."""
    problems = check_build(_identity(), current)
    assert any(fragment in p for p in problems), problems


def test_editable_install_is_never_compared() -> None:
    """Editable installs follow the checkout; they are recorded, not verified."""
    assert check_build(None, _identity(editable=True, extension_sha256=None)) == []


def test_identity_validates_formats() -> None:
    """Commit and digest fields must be hex of the right length."""
    with pytest.raises(ValueError, match="source_commit must be a 40-hex commit"):
        _identity(source_commit="HEAD")
    with pytest.raises(ValueError, match="python_sources_sha256 must be 64 hex"):
        _identity(python_sources_sha256="abc")
    with pytest.raises(ValueError, match="extension_sha256 must be 64 hex"):
        _identity(extension_sha256="abc")


# --- manifest file --------------------------------------------------------------


def test_manifest_round_trip_and_absence(tmp_path: Path) -> None:
    """read_manifest returns None when absent and the stamped manifest otherwise."""
    assert read_manifest(tmp_path) is None
    manifest = stamp_builds(REPO_ROOT, tmp_path, now=FIXED_TIME)
    assert manifest.schema_version == dependencies.MANIFEST_SCHEMA_VERSION
    assert manifest.created_at == "2026-08-29T12:00:00+00:00"
    assert [b.name for b in manifest.builds] == list(dependencies.BUILT_PACKAGES)
    assert read_manifest(tmp_path) == manifest
    assert not manifest_path(tmp_path).with_suffix(".json.tmp").exists()


def test_malformed_manifest_is_rejected(tmp_path: Path) -> None:
    """Invalid JSON, wrong shape, unknown keys, and unsupported schema versions all fail loudly."""
    path = manifest_path(tmp_path)
    path.write_text("{not json")
    with pytest.raises(ConfigError, match="invalid JSON"):
        read_manifest(tmp_path)
    path.write_text("[]")
    with pytest.raises(ConfigError, match="must be a JSON object"):
        read_manifest(tmp_path)
    good = json.loads(json.dumps(dependencies.to_mapping(stamp_builds(REPO_ROOT, tmp_path, now=FIXED_TIME))))
    path.write_text(json.dumps({**good, "extra": 1}))
    with pytest.raises(ConfigError, match=r"unknown key\(s\) 'extra'"):
        read_manifest(tmp_path)
    path.write_text(json.dumps({**good, "schema_version": 99}))
    with pytest.raises(ConfigError, match="unsupported schema_version 99"):
        read_manifest(tmp_path)


# --- against the real environment --------------------------------------------------


def test_installed_builds_have_expected_shape() -> None:
    """The rclib build carries a compiled extension, skelarm does not; neither is editable here."""
    assert installed_extension_digest("rclib") is not None
    assert installed_extension_digest("skelarm") is None
    assert installed_is_editable("rclib") is False
    assert installed_is_editable("skelarm") is False
    identity = current_build_identity("rclib")
    assert identity.version == dependencies.submodule_version("rclib")
    assert identity.source_commit == next(r for r in dependencies.submodule_revisions() if r.name == "rclib").recorded
    with pytest.raises(ValueError, match="not built from a submodule"):
        current_build_identity("rtctrl")


def test_environment_is_stamped_and_verifies() -> None:
    """The development environment carries a manifest that matches (else run the rebuild command)."""
    identities = verify_builds()
    assert [b.name for b in identities] == ["rclib", "skelarm"]


def test_stamp_then_verify_then_tamper(tmp_path: Path) -> None:
    """A fresh stamp verifies; a replaced extension digest, moved pin, or deleted manifest does not."""
    stamp_builds(REPO_ROOT, tmp_path, now=FIXED_TIME)
    verify_builds(REPO_ROOT, tmp_path)

    path = manifest_path(tmp_path)
    tampered = json.loads(path.read_text())
    tampered["builds"][0]["extension_sha256"] = DIGEST_1
    path.write_text(json.dumps(tampered))
    with pytest.raises(BuildIdentityError, match="compiled extension differs"):
        verify_builds(REPO_ROOT, tmp_path)

    tampered["builds"][0]["extension_sha256"] = json.loads(path.read_text())["builds"][0]["extension_sha256"]
    tampered = json.loads(json.dumps(dependencies.to_mapping(read_manifest(tmp_path))))
    tampered["builds"][1]["source_commit"] = COMMIT_B
    path.write_text(json.dumps(tampered))
    with pytest.raises(BuildIdentityError, match="skelarm: installed build was stamped for commit bbbbbbbbbbbb"):
        verify_builds(REPO_ROOT, tmp_path)

    path.unlink()
    with pytest.raises(BuildIdentityError, match="rclib: no build manifest entry"):
        verify_builds(REPO_ROOT, tmp_path)


def _always(*, value: bool) -> Callable[[str], bool]:
    def editable(_distribution: str) -> bool:
        return value

    return editable


def test_stamp_refuses_editable_and_dirty_states(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Editable installs and dirty submodules cannot be stamped."""
    monkeypatch.setattr(dependencies, "installed_is_editable", _always(value=True))
    with pytest.raises(BuildIdentityError, match="editable mode"):
        stamp_builds(REPO_ROOT, tmp_path)
    monkeypatch.setattr(dependencies, "installed_is_editable", _always(value=False))

    real = dependencies.submodule_revisions

    def dirty(root: Path | None = None) -> tuple[dependencies.SubmoduleRevision, ...]:
        return tuple(dataclasses.replace(r, dirty=True) for r in real(root))

    monkeypatch.setattr(dependencies, "submodule_revisions", dirty)
    with pytest.raises(BuildIdentityError, match="uncommitted changes"):
        stamp_builds(REPO_ROOT, tmp_path)
    assert not manifest_path(tmp_path).exists()


def test_manifest_schema_is_strict() -> None:
    """BuildManifest is validated through the strict mapper."""
    with pytest.raises(ConfigError, match="builds: expected array"):
        dependencies.from_mapping({"schema_version": 1, "created_at": "x", "builds": {}}, BuildManifest)


@pytest.mark.parametrize(
    ("names", "got"),
    [
        ((), r"\[\]"),
        (("rclib",), r"\['rclib'\]"),
        (("skelarm", "rclib"), r"\['skelarm', 'rclib'\]"),
        (("rclib", "skelarm", "rclib"), r"\['rclib', 'skelarm', 'rclib'\]"),
        (("rclib", "eigen"), r"\['rclib', 'eigen'\]"),
    ],
)
def test_manifest_requires_exactly_the_built_packages(tmp_path: Path, names: tuple[str, ...], got: str) -> None:
    """Missing, duplicate, unknown, or reordered manifest entries are rejected on load and on construction."""
    entries = [dataclasses.replace(_identity(), name=name) for name in names]
    with pytest.raises(ValueError, match=rf"builds must be exactly \['rclib', 'skelarm'\] in order, got {got}"):
        BuildManifest(1, "2026-08-29T12:00:00+00:00", tuple(entries))
    stamp_builds(REPO_ROOT, tmp_path, now=FIXED_TIME)
    path = manifest_path(tmp_path)
    data = json.loads(path.read_text())
    data["builds"] = [{**data["builds"][0], "name": name} for name in names]
    path.write_text(json.dumps(data))
    with pytest.raises(ConfigError, match="builds must be exactly"):
        read_manifest(tmp_path)
