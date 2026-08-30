# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M0-013: citation metadata is complete and consistent with the package metadata."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast

import yaml

import arm_rc_ctrl

REPO_ROOT = Path(__file__).resolve().parents[2]


def _citation() -> dict[str, object]:
    with (REPO_ROOT / "CITATION.cff").open(encoding="utf-8") as f:
        return cast("dict[str, object]", yaml.safe_load(f))


def _project() -> dict[str, object]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        return cast("dict[str, object]", tomllib.load(f)["project"])


def test_citation_file_has_required_cff_fields() -> None:
    """CFF 1.2.0 requires cff-version, message, title, and authors."""
    cff = _citation()
    assert cff["cff-version"] == "1.2.0"
    assert cff["type"] == "software"
    assert cff["title"] == "arm-rc-ctrl: Reservoir-Computing Control for Robot Arms"
    assert isinstance(cff["message"], str)
    assert cff["message"]
    authors = cast("list[dict[str, str]]", cff["authors"])
    assert authors == [{"family-names": "Atsuta", "given-names": "Hiroshi", "email": "atsuta@ieee.org"}]


def test_citation_matches_package_metadata() -> None:
    """Version, license, repository URL, and author agree with pyproject.toml."""
    cff = _citation()
    project = _project()
    assert cff["version"] == arm_rc_ctrl.__version__ == project["version"]
    assert cff["license"] == project["license"] == "GPL-3.0-only"
    urls = cast("dict[str, str]", project["urls"])
    assert cff["repository-code"] == urls["Repository"]
    authors = cast("list[dict[str, str]]", project["authors"])
    assert authors == [{"name": "Hiroshi Atsuta", "email": "atsuta@ieee.org"}]


def test_no_release_metadata_before_first_archival_release() -> None:
    """Until a tagged Zenodo release exists there is no DOI or release date (docs/PUBLICATION.md)."""
    cff = _citation()
    assert "doi" not in cff
    assert "identifiers" not in cff
    assert "date-released" not in cff
    assert "dev" in str(cff["version"])
    assert (REPO_ROOT / "docs" / "PUBLICATION.md").is_file()
