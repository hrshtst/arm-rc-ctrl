# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Robustness suite aggregation: failures stay counted, effects use both-success pairs, suites validate (M3-009)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from arm_rc_ctrl.experiments.closed_loop import EstimatorSpec
from arm_rc_ctrl.experiments.perturbations import RobustnessScenario
from arm_rc_ctrl.experiments.robustness import (
    Arm,
    ArmRun,
    RobustnessSuite,
    aggregate_runs,
    default_arms,
    paired_effects,
    suite_to_markdown,
)
from arm_rc_ctrl.metrics.joint import JointRmse
from arm_rc_ctrl.metrics.report import ReportWindows, RunReport
from arm_rc_ctrl.provenance import collect_provenance

ARMS = default_arms(["pd_v2"])


def report(run_id: str, *, rmse: float | None, termination: str = "completed", success: bool = True) -> RunReport:
    """A minimal report with a joint RMSE and no dwell/effort metrics."""
    return RunReport(
        run_id=run_id,
        method="x",
        scenario="s",
        reference_artifact="ref",
        termination_kind=termination,
        success=success,
        failed_criteria=() if success else ("completed",),
        windows=ReportWindows((1.0, 4.0), (4.0, 5.0)),
        move_coverage=1.0 if termination == "completed" else 0.2,
        dwell_coverage=1.0 if termination == "completed" else 0.0,
        joint_rmse=None if rmse is None else JointRmse(rmse, (rmse, rmse), 750),
        dwell=None,
        effort=None,
        demand=None,
        effort_source="tau_applied",
    )


SCENARIOS = (
    RobustnessScenario("nominal", "nominal", (0.0, 0.0)),
    RobustnessScenario("posture-small-1-00", "posture_small", (0.05, 0.0), seed=1, draw=0, magnitude_rad=0.05),
    RobustnessScenario("posture-small-1-01", "posture_small", (0.0, 0.05), seed=1, draw=1, magnitude_rad=0.05),
)
RUNS = (
    ArmRun("rc+pd_v2", "nominal", "nominal", "r1", report("r1", rmse=0.02)),
    ArmRun("replay+pd_v2", "nominal", "nominal", "r2", report("r2", rmse=0.001)),
    ArmRun("rc+pd_v2", "posture-small-1-00", "posture_small", "r3", report("r3", rmse=0.03)),
    ArmRun("replay+pd_v2", "posture-small-1-00", "posture_small", "r4", report("r4", rmse=0.002)),
    ArmRun(
        "rc+pd_v2",
        "posture-small-1-01",
        "posture_small",
        "r5",
        report("r5", rmse=None, termination="limit_violation", success=False),
    ),
    ArmRun("replay+pd_v2", "posture-small-1-01", "posture_small", "r6", report("r6", rmse=0.003)),
)


def test_failures_stay_in_the_aggregation() -> None:
    """A failed run counts in its arm and class with its termination kind; medians use successes only."""
    aggregates = aggregate_runs(RUNS, [a.name for a in ARMS])
    assert [(a.arm, a.kind) for a in aggregates] == [
        ("rc+pd_v2", "nominal"),
        ("rc+pd_v2", "posture_small"),
        ("replay+pd_v2", "nominal"),
        ("replay+pd_v2", "posture_small"),
    ]
    rc_small = aggregates[1]
    assert (rc_small.n, rc_small.completed, rc_small.successes) == (2, 1, 1)
    assert rc_small.failures == {"limit_violation": 1}
    assert rc_small.joint_rmse_median == 0.03
    assert rc_small.joint_rmse_max == 0.03
    assert rc_small.saturation_max is None
    replay_small = aggregates[3]
    assert replay_small.failures == {}
    assert replay_small.joint_rmse_median == 0.0025


def test_paired_effects_use_scenarios_where_both_succeeded_and_report_failed_pairs() -> None:
    """The RC-minus-replay difference is taken over both-success pairs; failed pairs are counted, not dropped."""
    effects = paired_effects(RUNS, ARMS)
    by_key = {(e.kind, e.metric): e for e in effects}
    nominal = by_key[("nominal", "joint_rmse")]
    assert (nominal.n_pairs, nominal.n_both_success, nominal.rc_failures, nominal.replay_failures) == (1, 1, 0, 0)
    assert nominal.median_difference == pytest.approx(0.019)
    small = by_key[("posture_small", "joint_rmse")]
    assert (small.n_pairs, small.n_both_success, small.rc_failures, small.replay_failures) == (2, 1, 1, 0)
    assert small.median_difference == pytest.approx(0.028)
    assert small.median_rc == 0.03
    assert small.median_replay == 0.002
    assert by_key[("posture_small", "effort_torque_rms")].median_difference is None
    assert by_key[("posture_small", "effort_torque_rms")].n_both_success == 1
    assert paired_effects(RUNS, [Arm("rc+pd_v2", "rc", "pd_v2")]) == ()  # no replay partner


def test_suite_validates_completeness_and_stored_tables() -> None:
    """Every arm must run every scenario once; stored aggregates/effects must match the recomputed ones."""
    provenance = collect_provenance({"suite": "test"}, seeds={}, artifacts=[], exploratory=True)
    suite = RobustnessSuite(
        label="development",
        protocol="p",
        protocol_file="p.toml",
        scenario="s",
        reference_artifact="ref",
        recipe="recipe",
        recipe_file="recipe.toml",
        estimator=EstimatorSpec(20.0, 20.0, 3.0),
        arms=ARMS,
        scenarios=SCENARIOS,
        runs=RUNS,
        provenance=provenance,
    )
    assert suite.aggregates == aggregate_runs(RUNS, [a.name for a in ARMS])
    assert suite.effects == paired_effects(RUNS, ARMS)
    assert replace(suite, aggregates=suite.aggregates, effects=suite.effects) == suite
    with pytest.raises(ValueError, match="every arm must run every scenario"):
        replace(suite, runs=RUNS[:-1])
    with pytest.raises(ValueError, match="every arm must run every scenario"):
        replace(suite, runs=(*RUNS, RUNS[0]))
    wrong = (replace(suite.aggregates[0], successes=0), *suite.aggregates[1:])
    with pytest.raises(ValueError, match="stored aggregates"):
        replace(suite, aggregates=wrong)
    wrong_effect = (replace(suite.effects[0], n_pairs=9), *suite.effects[1:])
    with pytest.raises(ValueError, match="stored paired effects"):
        replace(suite, effects=wrong_effect)
    with pytest.raises(ValueError, match="unique"):
        replace(suite, scenarios=(*SCENARIOS, SCENARIOS[0]), runs=RUNS)
    text = suite_to_markdown(suite)
    assert "| rc+pd_v2 | posture_small | 2 | 1 | 1 | limit_violation x1 | 0.03 | 0.03 | n/a | n/a |" in text
    assert "## Failed runs (1)" in text
    assert "| rc+pd_v2 | posture-small-1-01 | limit_violation | completed | `r5` |" in text
    with pytest.raises(ValueError, match="must not be empty"):
        Arm(" ", "rc", "pd_v2")
