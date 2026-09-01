# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""The human task 1-a report stays bound to its confirmatory runs and visual assets (DOC-003)."""

from __future__ import annotations

from arm_rc_ctrl.experiments.robustness import load_suite
from arm_rc_ctrl.experiments.trajectory_plots import select_representatives
from arm_rc_ctrl.repo import repository_root

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a"
OVERVIEW = DOCS / "overview.md"
SUITE = DOCS / "robustness_confirmatory_v2_recipe_v4.json"
ANIMATIONS = {
    "nominal_rc_pd.gif": ("nominal", "rc+pd_v2", "run-20260831-8dc3862d7168"),
    "nominal_replay_pd.gif": ("nominal", "replay+pd_v2", "run-20260831-013f6eb247ff"),
    "nominal_rc_ct.gif": ("nominal", "rc+computed_torque", "run-20260831-4f4f4934ec19"),
    "nominal_replay_ct.gif": ("nominal", "replay+computed_torque", "run-20260831-8c0569ef8a62"),
    "posture_small_rc_pd.gif": ("posture-small-20260903-03", "rc+pd_v2", "run-20260831-ec1191ddc035"),
    "posture_small_replay_pd.gif": (
        "posture-small-20260903-03",
        "replay+pd_v2",
        "run-20260831-244e793b514f",
    ),
    "posture_small_rc_ct.gif": (
        "posture-small-20260903-03",
        "rc+computed_torque",
        "run-20260831-7fa865cdcc90",
    ),
    "posture_small_replay_ct.gif": (
        "posture-small-20260903-03",
        "replay+computed_torque",
        "run-20260831-4b68d57e2fac",
    ),
    "posture_large_rc_pd.gif": ("posture-large-20260904-03", "rc+pd_v2", "run-20260831-eeaf620bb2c6"),
    "posture_large_replay_pd.gif": (
        "posture-large-20260904-03",
        "replay+pd_v2",
        "run-20260831-23e24d8dfaae",
    ),
    "posture_large_rc_ct.gif": (
        "posture-large-20260904-03",
        "rc+computed_torque",
        "run-20260831-0c1cd4d956c0",
    ),
    "posture_large_replay_ct.gif": (
        "posture-large-20260904-03",
        "replay+computed_torque",
        "run-20260831-144ec91a9a51",
    ),
    "force_12n_rc_pd.gif": ("force-12N-270deg", "rc+pd_v2", "run-20260831-c98f8f1156cc"),
    "force_12n_replay_pd.gif": ("force-12N-270deg", "replay+pd_v2", "run-20260831-c24da9e6c486"),
    "force_12n_rc_ct.gif": ("force-12N-270deg", "rc+computed_torque", "run-20260831-98e177f1a474"),
    "force_12n_replay_ct.gif": (
        "force-12N-270deg",
        "replay+computed_torque",
        "run-20260831-530d72bc46f0",
    ),
    "combined_rc_pd.gif": ("combined-20260901-03-270deg", "rc+pd_v2", "run-20260831-8658997d9ad9"),
    "combined_replay_pd.gif": (
        "combined-20260901-03-270deg",
        "replay+pd_v2",
        "run-20260831-4db678e4956e",
    ),
    "combined_rc_ct.gif": (
        "combined-20260901-03-270deg",
        "rc+computed_torque",
        "run-20260831-0689609c5082",
    ),
    "combined_replay_ct.gif": (
        "combined-20260901-03-270deg",
        "replay+computed_torque",
        "run-20260831-340828190d94",
    ),
}
TRAJECTORY_PLOTS = {
    f"trajectories_{kind}_{tracker}.png"
    for kind in ("nominal", "posture_small", "posture_large", "force", "combined")
    for tracker in ("pd", "computed_torque")
}


def test_overview_visuals_are_bound_to_confirmatory_runs() -> None:
    """Every named animation exists and cites the exact arm/scenario/run tuple in the locked suite."""
    text = OVERVIEW.read_text(encoding="utf-8")
    suite = load_suite(SUITE)
    actual = {(run.scenario_id, run.arm, run.run_id) for run in suite.runs}
    selected_scenarios = {run.scenario_id for run in select_representatives(suite).values()}
    for filename, identity in ANIMATIONS.items():
        animation = DOCS / "animations" / filename
        assert animation.is_file()
        assert animation.stat().st_size > 0
        assert f"animations/{filename}" in text
        assert identity in actual
        assert identity[2] in text
        assert identity[0] in selected_scenarios
    for plot in TRAJECTORY_PLOTS:
        assert (DOCS / "plots" / plot).is_file()
        assert f"plots/{plot}" in text
    for plot in ("rmse_by_class.png", "paired_differences.png", "search_objectives.png"):
        assert (DOCS / "plots" / plot).is_file()
        assert f"plots/{plot}" in text
    assert "260 of 260 runs succeeded" in text
    assert "not yet a demonstration of online learning" in text
    assert "one five-second trajectory (501" in text
    assert "No random-noise injection, perturbed copies, or other data" in text
    assert "not a controlled RC-versus-Deep-RL benchmark" in text
