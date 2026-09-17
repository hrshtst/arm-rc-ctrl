# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-008: the conditions identity that keys every manual evaluation run.

Historical scenario identities and replay caches cannot be reused under the
revised protocol (plan section 6), so the identity must move when anything the
protocol decides moves: the horizon, the dwell rule, the trigger, the abort,
the trackers, the bound files, or the environment. A cache that survived such
a change would serve a run from a different experiment.
"""

from __future__ import annotations

import pytest

from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments.baselines import frozen_baseline_digest
from arm_rc_ctrl.experiments.manual_evaluation import (
    ManualRunConditions,
    load_manual_evaluation_config,
    manual_conditions,
)
from arm_rc_ctrl.experiments.perturbations import load_development_robustness, robustness_scenarios
from arm_rc_ctrl.experiments.recovery_search import RECOVERY_TRACKERS
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root

REPO_ROOT = repository_root()
EVALUATION = REPO_ROOT / "configs" / "evaluations" / "task_1a_manual_dev_v1.toml"
DEVELOPMENT = REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml"
SCENARIO = REPO_ROOT / "configs" / "tasks" / "task_1a_manual_v2.toml"
EXECUTION_IDENTITY = "e" * 64
"""A stand-in environment: the builder binds whatever the caller verified, it does not probe."""

WARMUP_S = 0.25
REPLAY_CUTOFFS = (6.66946, 5.59031)
"""``feasible-best``'s own estimator cutoffs: replay filters exactly as its paired generator does."""

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
TRACKERS = {"pd_v2": DIGEST_A, "computed_torque": DIGEST_B}
SCENARIOS = ("nominal", "force-000deg", "combined-20261201-01-090deg")


def _committed_conditions() -> ManualRunConditions:
    """The conditions of the committed protocol over its own 65 scenarios."""
    config = load_manual_evaluation_config(EVALUATION)
    scenario = load_manual_scenario(config.scenario)
    levels = load_development_robustness(config.development)
    lower = tuple(link.q_min for link in scenario.robot.links)
    upper = tuple(link.q_max for link in scenario.robot.links)
    cases = robustness_scenarios(levels, nominal=scenario.task.initial_q, lower=lower, upper=upper)
    return manual_conditions(
        config,
        EVALUATION,
        scenario_ids=tuple(case.scenario_id for case in cases),
        warmup_s=WARMUP_S,
        replay_cutoffs=REPLAY_CUTOFFS,
        execution_identity=EXECUTION_IDENTITY,
        root=REPO_ROOT,
    )


def _conditions(**changes: object) -> ManualRunConditions:
    """The committed protocol's conditions, with single fields overridden."""
    values: dict[str, object] = {
        "evaluation_name": "task-1a-manual-dev-v1",
        "evaluation_file": "configs/evaluations/task_1a_manual_dev_v1.toml",
        "evaluation_sha256": DIGEST_A,
        "development_file": "configs/evaluations/task_1a_recovery_dev_v1.toml",
        "development_sha256": DIGEST_B,
        "scenario_file": "configs/tasks/task_1a_manual_v2.toml",
        "scenario_sha256": DIGEST_C,
        "horizon_s": 30.0,
        "trigger_hold_s": 0.5,
        "trigger_duration_s": 0.2,
        "trigger_magnitude_n": 12.0,
        "dwell_min_duration_s": 1.0,
        "dwell_tolerance_m": 0.01,
        "dwell_max_velocity_rad_s": 0.05,
        "replay_velocity_cutoff_hz": 6.66946,
        "replay_acceleration_cutoff_hz": 5.59031,
        "velocity_abort": (6.0, 6.0),
        "trackers": dict(TRACKERS),
        "tracker_order": ("pd_v2", "computed_torque"),
        "scenario_ids": SCENARIOS,
        "warmup_s": 0.25,
        "execution_identity": DIGEST_A,
    }
    values.update(changes)
    return ManualRunConditions(**values)  # type: ignore[arg-type]


# --- what the identity must bind -------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("horizon_s", 20.0),
        ("trigger_hold_s", 0.75),
        ("trigger_duration_s", 0.25),
        ("trigger_magnitude_n", 9.0),
        ("dwell_min_duration_s", 1.5),
        ("dwell_tolerance_m", 0.02),
        ("dwell_max_velocity_rad_s", 0.1),
        ("replay_velocity_cutoff_hz", 21.8537),
        ("replay_acceleration_cutoff_hz", 9.33166),
        ("velocity_abort", (12.0, 12.0)),
        ("warmup_s", 0.5),
        ("execution_identity", DIGEST_C),
        ("development_sha256", DIGEST_C),
        ("scenario_sha256", DIGEST_A),
        ("evaluation_sha256", DIGEST_B),
    ],
)
def test_every_decided_quantity_moves_the_identity(field: str, value: object) -> None:
    """Anything the protocol decides is part of the key; a cache must not survive its change."""
    assert _conditions().identity != _conditions(**{field: value}).identity


def test_a_changed_tracker_digest_moves_the_identity() -> None:
    """Gains are bound by value, so re-tuning a tracker invalidates every run it produced."""
    other = dict(TRACKERS)
    other["pd_v2"] = DIGEST_C
    assert _conditions().identity != _conditions(trackers=other).identity


def test_a_changed_scenario_set_moves_the_identity() -> None:
    """The evaluated cases are part of the conditions, not an incidental argument."""
    assert _conditions().identity != _conditions(scenario_ids=(*SCENARIOS, "nominal-2")).identity


def test_the_identity_is_stable_across_equal_conditions() -> None:
    """Two identical descriptions key the same runs, whatever order their mappings were built in."""
    reordered = {"computed_torque": DIGEST_B, "pd_v2": DIGEST_A}
    assert _conditions().identity == _conditions(trackers=reordered).identity
    assert len(_conditions().identity) == 64


# --- ordering and validation -----------------------------------------------------------------


def test_pairs_run_scenario_major_and_tracker_minor() -> None:
    """A fixed, declared order: both trackers of one scenario before the next scenario begins."""
    pairs = _conditions().pairs
    assert pairs == tuple((s, t) for s in SCENARIOS for t in ("pd_v2", "computed_torque"))
    assert len(pairs) == len(SCENARIOS) * 2


def test_the_tracker_order_must_list_every_tracker_once() -> None:
    """The JSON form sorts mapping keys, so the evaluation order is declared explicitly and checked."""
    with pytest.raises(ValueError, match="tracker_order"):
        _conditions(tracker_order=("pd_v2",))


def test_repeated_scenarios_are_refused() -> None:
    """Each case is attempted once per tracker; a repeated id would silently overwrite a run."""
    with pytest.raises(ValueError, match="distinct"):
        _conditions(scenario_ids=("nominal", "nominal"))


def test_a_malformed_digest_is_refused() -> None:
    """Every bound file and the environment are named by a full digest."""
    with pytest.raises(ValueError, match="64 lowercase hex"):
        _conditions(scenario_sha256="abc")


def test_an_abort_that_is_not_per_joint_is_refused() -> None:
    """The abort covers every joint with a positive finite bound."""
    with pytest.raises(ValueError, match="positive finite"):
        _conditions(velocity_abort=(6.0, 0.0))


def test_two_configurations_sharing_a_warm_up_but_not_a_policy_are_different_conditions() -> None:
    """Replay derives its own derivatives, so its filter is part of what produced the run.

    Six of the study's configurations share three warm-ups but each carries its
    own estimator cutoffs, so a bank may only be shared where the whole policy
    matches; keying on warm-up alone would serve one configuration's baselines
    to another whose replay was filtered differently.
    """
    best = _conditions(replay_velocity_cutoff_hz=6.66946, replay_acceleration_cutoff_hz=5.59031)
    dwell = _conditions(replay_velocity_cutoff_hz=28.4575, replay_acceleration_cutoff_hz=5.73043)
    assert best.warmup_s == dwell.warmup_s
    assert best.identity != dwell.identity


def test_the_replay_policy_is_part_of_what_a_run_records() -> None:
    """A stored run states the filter its reference was driven through, not merely a path to a config."""
    conditions = _conditions()
    assert conditions.replay_velocity_cutoff_hz == pytest.approx(6.66946)
    assert conditions.replay_acceleration_cutoff_hz == pytest.approx(5.59031)


def test_a_non_positive_replay_cutoff_is_refused() -> None:
    """A cutoff is a real filter frequency; zero would mean something else entirely."""
    with pytest.raises(ValueError, match="positive and finite"):
        _conditions(replay_velocity_cutoff_hz=0.0)


# --- the conditions the committed protocol produces ------------------------------------------


def test_the_builder_binds_the_committed_files_by_digest() -> None:
    """The conditions name the evaluation, the locked draws, and the task configuration, each by digest."""
    conditions = _committed_conditions()
    assert conditions.evaluation_name == "task-1a-manual-dev-v1"
    assert conditions.evaluation_sha256 == sha256_file(EVALUATION)
    assert conditions.development_sha256 == sha256_file(DEVELOPMENT)
    assert conditions.scenario_sha256 == sha256_file(SCENARIO)
    assert conditions.execution_identity == EXECUTION_IDENTITY


def test_the_builder_records_repository_relative_paths_only() -> None:
    """Recorded evidence never carries a machine path, so the bound files are repository-relative."""
    conditions = _committed_conditions()
    for path in (conditions.evaluation_file, conditions.development_file, conditions.scenario_file):
        assert not path.startswith("/")
        assert "/home/" not in path
        assert path.startswith("configs/")


def test_the_builder_copies_the_dwell_rule_from_the_task_configuration() -> None:
    """The rule a run is judged by is recorded with the run, and it is the acquisition rule."""
    conditions = _committed_conditions()
    scenario = load_manual_scenario(SCENARIO)
    assert conditions.dwell_min_duration_s == scenario.task.dwell_min_duration_s == 1.0
    assert conditions.dwell_tolerance_m == scenario.task.tolerance == 0.01
    assert conditions.dwell_max_velocity_rad_s == scenario.task.dwell_max_velocity == 0.05
    assert conditions.velocity_abort == tuple(scenario.limits.velocity) == (6.0, 6.0)


def test_the_builder_carries_the_revised_horizon_and_trigger() -> None:
    """The protocol's own quantities come from the evaluation configuration."""
    conditions = _committed_conditions()
    config = load_manual_evaluation_config(EVALUATION)
    assert conditions.horizon_s == config.horizon_s == 30.0
    assert conditions.trigger_hold_s == config.trigger.hold_s == 0.5
    assert conditions.trigger_duration_s == config.trigger.duration_s == 0.2
    assert conditions.trigger_magnitude_n == config.trigger.magnitude_n == 12.0


def test_the_builder_binds_both_frozen_trackers_by_their_gains() -> None:
    """Trackers are bound by the digest of their gains, so re-tuning one invalidates its runs."""
    conditions = _committed_conditions()
    assert conditions.tracker_order == RECOVERY_TRACKERS
    assert conditions.trackers == {name: frozen_baseline_digest(name) for name in RECOVERY_TRACKERS}


def test_the_committed_protocol_evaluates_one_hundred_and_thirty_pairs() -> None:
    """Sixty-five development scenarios under both frozen trackers."""
    conditions = _committed_conditions()
    assert len(conditions.scenario_ids) == 65
    assert len(conditions.pairs) == 130
