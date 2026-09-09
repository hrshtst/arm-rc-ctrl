# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""UP-007: the committed old/new rclib build comparison is exact, complete, and reproduced by the current build."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.experiments.build_parity import (
    FIXTURE_CASES,
    load_parity,
    load_probe,
    probe_frozen_recipes,
    render_parity_markdown,
)
from arm_rc_ctrl.rc.recipe import RclibIdentity
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a_repeated_demonstration"
EVIDENCE = DOCS / "rclib_build_parity_v1.json"
MARKDOWN = DOCS / "rclib_build_parity_v1.md"
FROZEN_RECIPES = sorted(p.stem for p in (REPO_ROOT / "data" / "records" / "models").glob("model-*.toml"))


def test_committed_parity_is_bitwise_across_the_pin_advance() -> None:
    """Five cases (three frozen recipes, two fixture recipes) agree bitwise between a015aca and the current pin."""
    report = load_parity(EVIDENCE)
    assert report.all_bitwise_equal
    assert [c.case for c in report.cases] == sorted([*FROZEN_RECIPES, *FIXTURE_CASES])
    assert len(report.cases) == 5
    assert report.rclib_old.commit.startswith("a015aca")
    assert report.rclib_new == RclibIdentity.current()  # the new build is the pinned one
    assert report.rclib_old.commit != report.rclib_new.commit
    assert not report.provenance.project_dirty
    assert not report.provenance.exploratory
    for case in report.cases:
        assert case.predictions_bitwise_equal
        assert case.max_abs_prediction_diff == 0.0
        assert case.fit_report_equal
        assert case.rmse_old == case.rmse_new
        assert (case.exact_old is None) == (case.case in FIXTURE_CASES)
    # The v4 recipe's refit differs from its recorded fit report under both builds (a 6e-16 RMSE difference
    # within its tolerance, unchanged by the pin); v2 and v3 refit bitwise under both.
    exact = {c.case: c.exact_new for c in report.cases if c.exact_new is not None}
    assert sum(1 for value in exact.values() if value) == 2
    assert sum(1 for value in exact.values() if not value) == 1
    assert render_parity_markdown(report) == MARKDOWN.read_text(encoding="utf-8")


def test_current_build_reproduces_the_committed_new_build_predictions(tmp_path: Path) -> None:
    """Probing the frozen and fixture recipes again yields the committed new-build digests (needs the store)."""
    report = load_parity(EVIDENCE)
    try:
        summary = probe_frozen_recipes(tmp_path / "probe")
    except Exception as exc:  # noqa: BLE001 - any setup failure just means no store on this runner
        pytest.skip(f"external storage unavailable: {exc}")
    _summary, predictions = load_probe(tmp_path / "probe")
    assert summary.rclib_installed == report.rclib_new
    from arm_rc_ctrl.data.arrays import array_digest

    for case in report.cases:
        assert array_digest(predictions[case.case]) == case.prediction_sha256_new, case.case
        assert summary.cases[case.case].report.rmse == case.rmse_new
        assert summary.cases[case.case].exact == case.exact_new
