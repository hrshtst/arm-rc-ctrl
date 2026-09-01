# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""The human task 1-a report stays bound to its confirmatory runs and visual assets (DOC-003)."""

from __future__ import annotations

from arm_rc_ctrl.experiments.robustness import load_suite
from arm_rc_ctrl.repo import repository_root

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a"
OVERVIEW = DOCS / "overview.md"
SUITE = DOCS / "robustness_confirmatory_v2_recipe_v4.json"
ANIMATIONS = {
    "nominal_rc_pd.gif": ("nominal", "rc+pd_v2", "run-20260831-8dc3862d7168"),
    "nominal_replay_pd.gif": ("nominal", "replay+pd_v2", "run-20260831-013f6eb247ff"),
    "force_12n_rc_pd.gif": ("force-12N-090deg", "rc+pd_v2", "run-20260831-6a6b01edfccf"),
    "force_12n_rc_ct.gif": ("force-12N-090deg", "rc+computed_torque", "run-20260831-94b08b3e3cc9"),
}


def test_overview_visuals_are_bound_to_confirmatory_runs() -> None:
    """Every named animation exists and cites the exact arm/scenario/run tuple in the locked suite."""
    text = OVERVIEW.read_text(encoding="utf-8")
    suite = load_suite(SUITE)
    actual = {(run.scenario_id, run.arm, run.run_id) for run in suite.runs}
    for filename, identity in ANIMATIONS.items():
        animation = DOCS / "animations" / filename
        assert animation.is_file()
        assert animation.stat().st_size > 0
        assert f"animations/{filename}" in text
        assert identity in actual
        assert identity[2] in text
    for plot in ("rmse_by_class.png", "paired_differences.png", "search_objectives.png"):
        assert (DOCS / "plots" / plot).is_file()
        assert f"plots/{plot}" in text
    assert "260 of 260 runs succeeded" in text
    assert "not yet a demonstration of online learning" in text
