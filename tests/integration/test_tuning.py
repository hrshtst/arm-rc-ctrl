# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-024: the tuning protocol fixes search spaces, objective, seed, budget, penalty, and dev scenarios."""

from __future__ import annotations

import dataclasses
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from arm_rc_ctrl.config import ConfigError
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.preprocess import preprocess_demonstration
from arm_rc_ctrl.data.records import RawDemonstrationRecord, load_record
from arm_rc_ctrl.data.samples import SampleSet
from arm_rc_ctrl.experiments.tuning import (
    GainRange,
    Objective,
    TuningProtocol,
    evaluate_gains,
    load_protocol,
    run_study,
    sample_gains,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import ScenarioConfig, load_scenario
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
PROTOCOL = REPO_ROOT / "configs" / "studies" / "baseline_gains_1a.toml"
RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
FIXED_TIME = datetime(2026, 8, 30, 11, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[SampleSet, ScenarioConfig]:
    """Processed fixture dataset and its scenario."""
    base = tmp_path_factory.mktemp("tuning")
    root = base / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    store.path(raw.artifact.payload.uri, mode="write").write_bytes(RAW_LOG.read_bytes())
    records = base / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    result = preprocess_demonstration(
        RAW_RECORD, SCENARIO, PREPROCESS, store=store, records_root=records, exploratory=True, now=FIXED_TIME
    )
    return result.samples, load_scenario(SCENARIO)


@pytest.fixture(scope="module")
def protocol() -> TuningProtocol:
    """The committed protocol."""
    return load_protocol(PROTOCOL)


def test_committed_protocol_fixes_everything(protocol: TuningProtocol) -> None:
    """Search spaces, objective, penalty, seed, budget, scenario, and development scenarios are versioned."""
    assert protocol.name == "baseline-gains-1a"
    assert protocol.scenario == REPO_ROOT / "configs" / "tasks" / "task_1a.toml"
    assert load_scenario(protocol.scenario).name == "task-1a-reach"
    assert protocol.budget == 64
    assert protocol.sampler_seed == 20260830
    assert protocol.objective == Objective("median_move_joint_rmse", 10.0)
    assert protocol.search.pd.kp == GainRange(1.0, 300.0, log=True)
    assert protocol.search.computed_torque.kd == GainRange(2.0, 120.0, log=True)
    assert protocol.development.initial_posture_offsets[0] == (0.0, 0.0)
    assert len(protocol.development.initial_posture_offsets) == 4


def test_sampling_is_seeded_log_uniform_and_within_bounds(protocol: TuningProtocol) -> None:
    """The same seed reproduces the same gains; every draw respects the per-tracker bounds."""
    first = [sample_gains(protocol, "pd", 2, np.random.default_rng(1)) for _ in range(20)]
    second = [sample_gains(protocol, "pd", 2, np.random.default_rng(1)) for _ in range(20)]
    assert first == second
    for gains in first:
        assert gains.type == "pd"
        assert all(1.0 <= k <= 300.0 for k in gains.kp)
        assert all(0.05 <= k <= 60.0 for k in gains.kd)
    ct = sample_gains(protocol, "computed_torque", 2, np.random.default_rng(2))
    assert ct.type == "computed_torque"
    assert all(10.0 <= k <= 900.0 for k in ct.kp)
    # Log-uniform: the log of many draws is roughly uniform, so its mean sits near the log-midpoint.
    rng = np.random.default_rng(3)
    logs = np.log([k for _ in range(400) for k in sample_gains(protocol, "pd", 2, rng).kp])
    assert abs(float(np.mean(logs)) - 0.5 * (math.log(1.0) + math.log(300.0))) < 0.3


def test_evaluate_gains_reports_every_component(
    dataset: tuple[SampleSet, ScenarioConfig], protocol: TuningProtocol
) -> None:
    """A feasible trial's objective is the median movement RMSE over the development scenarios."""
    samples, scenario = dataset
    gains = TrackerConfig("computed_torque", (100.0, 100.0), (20.0, 20.0))
    objective, feasible, components = evaluate_gains(protocol, scenario, samples, gains)
    assert len(components) == 4
    assert [c.index for c in components] == [0, 1, 2, 3]
    assert components[0].initial_q == scenario.task.initial_q
    assert all(c.termination == "completed" for c in components)
    rmses = [c.move_joint_rmse for c in components]
    assert all(r is not None and math.isfinite(r) for r in rmses)
    assert all(set(c.criteria) == {"completed", "dwell_in_tolerance", "dwell_stationary"} for c in components)
    if feasible:
        assert objective == pytest.approx(float(np.median([r for r in rmses if r is not None])))
    else:
        assert objective == protocol.objective.infeasible_penalty
        assert any(not c.feasible for c in components)


def test_infeasible_trials_receive_the_documented_penalty(
    dataset: tuple[SampleSet, ScenarioConfig], protocol: TuningProtocol
) -> None:
    """A limit violation in any development scenario yields the penalty, with the cause recorded."""
    samples, scenario = dataset
    strict = dataclasses.replace(scenario, limits=dataclasses.replace(scenario.limits, velocity=(0.3, 0.3)))
    objective, feasible, components = evaluate_gains(
        protocol, strict, samples, TrackerConfig("pd", (20.0, 10.0), (2.0, 1.0))
    )
    assert objective == 10.0
    assert feasible is False
    assert all(c.termination == "limit_violation" for c in components)
    assert all(c.move_joint_rmse is None for c in components)
    assert all(
        c.criteria == {"completed": False, "dwell_in_tolerance": False, "dwell_stationary": False} for c in components
    )


def test_study_is_deterministic_and_selects_the_minimum(
    dataset: tuple[SampleSet, ScenarioConfig], protocol: TuningProtocol
) -> None:
    """Two runs with the same seed give identical trials; the best trial has the lowest objective."""
    samples, scenario = dataset
    small = dataclasses.replace(protocol, budget=4)
    a = run_study(small, samples, "pd", scenario=scenario)
    b = run_study(small, samples, "pd", scenario=scenario)
    assert a == b
    assert a.budget == 4
    assert a.sampler_seed == protocol.sampler_seed
    assert [t.number for t in a.trials] == [0, 1, 2, 3]
    assert a.best.objective == min(t.objective for t in a.trials)
    assert a.feasible_trials == sum(1 for t in a.trials if t.feasible)
    ct = run_study(small, samples, "computed_torque", scenario=scenario)
    assert ct.sampler_seed == a.sampler_seed
    assert ct.budget == a.budget  # equal budget for both trackers


def test_study_result_consistency_is_enforced(
    dataset: tuple[SampleSet, ScenarioConfig], protocol: TuningProtocol
) -> None:
    """Budget must match the trial count and best must be the minimum."""
    samples, scenario = dataset
    result = run_study(dataclasses.replace(protocol, budget=2), samples, "pd", scenario=scenario)
    with pytest.raises(ValueError, match="trials but budget"):
        dataclasses.replace(result, budget=3)
    worst = max(result.trials, key=lambda t: t.objective)
    if worst != result.best:
        with pytest.raises(ValueError, match="lowest objective"):
            dataclasses.replace(result, best=worst)


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("budget = 64", "budget = 0", "budget must be >= 1"),
        ("sampler_seed = 20260830", "sampler_seed = -1", "sampler_seed must be non-negative"),
        ("infeasible_penalty = 10.0", "infeasible_penalty = 0.0", "infeasible_penalty must be positive"),
        (
            "kp = { low = 1.0, high = 300.0, log = true }",
            "kp = { low = 300.0, high = 1.0, log = true }",
            "0 < low < high",
        ),
        (
            "initial_posture_offsets = [[0.0, 0.0], [0.05, -0.05], [-0.05, 0.05], [0.08, 0.0]]",
            "initial_posture_offsets = []",
            "must not be empty",
        ),
    ],
    ids=["budget", "seed", "penalty", "range", "offsets"],
)
def test_protocol_invariants(tmp_path: Path, old: str, new: str, expected: str) -> None:
    """Invalid protocol files fail to load."""
    text = PROTOCOL.read_text().replace(old, new, 1)
    path = tmp_path / "studies" / "bad.toml"
    path.parent.mkdir()
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "task_1a.toml").write_text((REPO_ROOT / "configs" / "tasks" / "task_1a.toml").read_text())
    path.write_text(text)
    with pytest.raises(ConfigError, match=expected):
        load_protocol(path)


def test_dwell_criteria_decide_feasibility(dataset: tuple[SampleSet, ScenarioConfig], protocol: TuningProtocol) -> None:
    """A trial that completes but fails a dwell criterion is infeasible and penalised."""
    samples, scenario = dataset
    gains = TrackerConfig("computed_torque", (100.0, 100.0), (20.0, 20.0))
    lenient = dataclasses.replace(
        scenario, task=dataclasses.replace(scenario.task, dwell_min_fraction=0.0, dwell_max_velocity=10.0)
    )
    demanding = dataclasses.replace(
        scenario, task=dataclasses.replace(scenario.task, dwell_min_fraction=1.0, dwell_max_velocity=1e-6)
    )
    objective_ok, feasible_ok, _ = evaluate_gains(protocol, lenient, samples, gains)
    objective_bad, feasible_bad, components = evaluate_gains(protocol, demanding, samples, gains)
    assert feasible_ok is True
    assert objective_ok < protocol.objective.infeasible_penalty
    assert feasible_bad is False
    assert objective_bad == protocol.objective.infeasible_penalty
    assert all(c.termination == "completed" and c.move_joint_rmse is not None for c in components)
    assert all(c.criteria["completed"] and not c.criteria["dwell_stationary"] for c in components)
