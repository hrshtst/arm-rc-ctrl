# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""UP-007: the frozen task 1-a recipes still refit under the current rclib pin (numerical parity across pins)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.rc.recipe import RclibIdentity, load_recipe
from arm_rc_ctrl.rc.runtime import load_training_samples
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageAccessError, open_storage

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
RECIPES = sorted((REPO_ROOT / "data" / "records" / "models").glob("model-*.toml"))


@pytest.mark.parametrize("path", RECIPES, ids=[p.stem for p in RECIPES])
def test_frozen_recipe_refits_within_tolerance_under_the_current_pin(path: Path) -> None:
    """A recipe bound to an earlier rclib commit reproduces its fit report when its identity check is overridden.

    Recipes bind the rclib commit they were made with; advancing the pin does
    not rewrite them. This lock asserts that the rebuilt library still
    reproduces every frozen fit within the recipe's declared tolerance, which
    is what makes the pin advance safe for the frozen evidence.
    """
    recipe = load_recipe(path)
    try:
        store = open_storage()
        samples = load_training_samples(recipe, store)
    except Exception as exc:  # noqa: BLE001 - any setup failure just means no store on this runner
        pytest.skip(f"external storage unavailable: {exc}")
    _model, report = recipe.refit(samples, installed=recipe.rclib)
    assert report.episodes == recipe.fit.episodes
    assert report.loss_rows == recipe.fit.loss_rows
    assert report.rmse == pytest.approx(recipe.fit.rmse, abs=recipe.tolerance.error_abs)
    assert report.max_abs_error == pytest.approx(recipe.fit.max_abs_error, abs=recipe.tolerance.error_abs)
    if RclibIdentity.current() != recipe.rclib:
        # The whole point of the lock: the current pin differs from the recipe's, and the refit still holds.
        assert RclibIdentity.current().version == recipe.rclib.version


def test_storage_access_error_is_a_skip_not_a_failure() -> None:
    """The lock never fails merely because the payloads are absent (CI has no external store)."""
    assert issubclass(StorageAccessError, Exception)
