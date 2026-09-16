# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-002: every committed schema 1 recipe re-serializes byte for byte and keeps its content-addressed identity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.data.records import to_toml
from arm_rc_ctrl.rc.recipe import RECIPE_SCHEMA_VERSION, load_recipe
from arm_rc_ctrl.rc.train import recipe_id
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.regression

RECIPES = sorted((repository_root() / "data" / "records" / "models").glob("model-*.toml"))
HEADER_LINES = 2


@pytest.mark.parametrize("path", RECIPES, ids=[f.stem for f in RECIPES])
def test_committed_recipe_serialization_and_identity_are_unchanged(path: Path) -> None:
    """Loading and re-serializing the frozen recipe reproduces the file, and its ID re-derives from its content."""
    text = path.read_text(encoding="utf-8")
    header = "".join(text.splitlines(keepends=True)[:HEADER_LINES])
    recipe = load_recipe(path)
    assert recipe.schema_version == RECIPE_SCHEMA_VERSION
    assert recipe.validation is None
    assert recipe.training.additional_repeats is None
    assert recipe.training.base_alpha is None
    assert recipe.training.regularization_rule is None
    assert recipe.training.contractive is None
    assert header + to_toml(recipe) == text
    day = path.stem.split("-")[1]
    assert recipe_id(recipe, f"{day[:4]}-{day[4:6]}-{day[6:]}T00:00:00+00:00") == path.stem


def test_three_frozen_recipes_are_locked() -> None:
    """The v2, v3, and v4 task 1-a recipes are the frozen set this lock covers."""
    assert [f.stem for f in RECIPES] == [
        "model-20260831-038a9b2c8432",
        "model-20260831-1b9477aaa246",
        "model-20260831-ea83321eeaa5",
    ]
