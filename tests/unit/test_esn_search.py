# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""ESN search space: ranges, sampling, points, comparison queueing, and protocol validation (M3-003)."""

from __future__ import annotations

from dataclasses import replace

import optuna
import pytest

from arm_rc_ctrl.experiments.esn_search import (
    PLANNED_PARAMETERS,
    ComparisonPoint,
    EsnObjective,
    EsnSearchProtocol,
    EsnSearchSpace,
    FloatRange,
    IntRange,
    TrialPoint,
    enqueue_comparisons,
    load_esn_search,
    point_from_params,
    protocol_digest,
    suggest_point,
)
from arm_rc_ctrl.provenance import config_digest
from arm_rc_ctrl.rc.train import load_model_config
from arm_rc_ctrl.repo import repository_root

REPO_ROOT = repository_root()
PROTOCOL = REPO_ROOT / "configs" / "studies" / "esn_search_1a.toml"
MODEL_V2 = REPO_ROOT / "configs" / "models" / "esn_task_1a_v2.toml"


@pytest.fixture(scope="module")
def protocol() -> EsnSearchProtocol:
    """The committed search protocol."""
    return load_esn_search(PROTOCOL)


def test_ranges_validate_their_bounds() -> None:
    """Ranges need low < high; log ranges need positive lows; steps exclude log."""
    with pytest.raises(ValueError, match="low < high"):
        FloatRange(1.0, 1.0)
    with pytest.raises(ValueError, match="log range"):
        FloatRange(0.0, 1.0, log=True)
    with pytest.raises(ValueError, match="step"):
        FloatRange(0.1, 1.0, log=True, step=0.1)
    with pytest.raises(ValueError, match="low < high"):
        IntRange(5, 5)
    with pytest.raises(ValueError, match="step"):
        IntRange(1, 10, step=0)
    with pytest.raises(ValueError, match="step"):
        IntRange(1, 10, step=2, log=True)
    with pytest.raises(ValueError, match="log range"):
        IntRange(0, 10, log=True)
    grid = IntRange(100, 800, step=50)
    assert grid.contains(150)
    assert not grid.contains(160)
    assert not grid.contains(850)
    assert FloatRange(0.0, 1.0).contains(1.0)
    assert not FloatRange(0.0, 1.0).contains(1.0001)


def test_space_stays_within_model_validity(protocol: EsnSearchProtocol) -> None:
    """Bounds outside what the reservoir, readout, or estimator accept are rejected at load time."""
    space = protocol.search
    with pytest.raises(ValueError, match=r"search\.sparsity"):
        replace(space, sparsity=FloatRange(0.5, 1.2))
    with pytest.raises(ValueError, match=r"search\.leak_rate"):
        replace(space, leak_rate=FloatRange(0.0, 1.0))
    with pytest.raises(ValueError, match=r"search\.alpha"):
        replace(space, alpha=FloatRange(0.0, 1.0))
    with pytest.raises(ValueError, match=r"search\.seed"):
        replace(space, seed=IntRange(-1, 10))
    with pytest.raises(ValueError, match=r"search\.n_neurons"):
        replace(space, n_neurons=IntRange(0, 10))


def test_suggested_points_lie_in_the_space_and_round_trip(protocol: EsnSearchProtocol) -> None:
    """Sampling draws every planned parameter within bounds; stored parameters rebuild the same point."""
    space = protocol.search
    points: list[TrialPoint] = []

    def objective(trial: optuna.Trial) -> float:
        point = suggest_point(space, trial)
        points.append(point)
        assert dict(trial.params) == point.params()
        return point.alpha

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=3))
    study.optimize(objective, n_trials=20)
    assert len(points) == 20
    for trial, point in zip(study.trials, points, strict=True):
        assert tuple(trial.params) == PLANNED_PARAMETERS
        assert space.contains(point)
        assert point_from_params(space, trial.params) == point
    assert len({p.n_neurons for p in points}) > 1
    assert all((p.n_neurons - 100) % 50 == 0 for p in points)
    with pytest.raises(ValueError, match="missing"):
        point_from_params(space, {"alpha": 0.1})
    with pytest.raises(ValueError, match="outside"):
        point_from_params(space, {**points[0].params(), "alpha": 5.0})


def test_point_builds_the_model_config_and_estimator(protocol: EsnSearchProtocol) -> None:
    """A point overrides only the tuned reservoir/readout fields and the estimator cutoffs."""
    base = load_model_config(MODEL_V2)
    point = TrialPoint(300, 1.1, 0.7, 0.2, 0.8, 5, 0.05, 12.0, 30.0)
    config = point.model_config(base, name="trial-7")
    assert config.name == "trial-7"
    assert config.input_transform == base.input_transform
    assert config.esn.readout.solver == base.esn.readout.solver
    assert config.esn.readout.alpha == 0.05
    reservoir = config.esn.reservoir
    assert (reservoir.n_neurons, reservoir.spectral_radius, reservoir.sparsity) == (300, 1.1, 0.7)
    assert (reservoir.leak_rate, reservoir.input_scaling, reservoir.seed) == (0.2, 0.8, 5)
    assert reservoir.include_bias == base.esn.reservoir.include_bias
    estimator = point.estimator(max_dt_ratio=protocol.max_dt_ratio)
    assert (estimator.velocity_cutoff_hz, estimator.acceleration_cutoff_hz, estimator.max_dt_ratio) == (12.0, 30.0, 3.0)


def test_comparison_points_are_queued_once_and_labelled(protocol: EsnSearchProtocol) -> None:
    """Comparison points run first, carry their label, and are not re-queued on a second call."""
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    assert enqueue_comparisons(study, protocol) == len(protocol.comparison)
    assert enqueue_comparisons(study, protocol) == 0  # already waiting

    def objective(trial: optuna.Trial) -> float:
        return suggest_point(protocol.search, trial).alpha

    study.optimize(objective, n_trials=len(protocol.comparison) + 1)
    assert enqueue_comparisons(study, protocol) == 0  # already stored
    for trial, comparison in zip(study.trials, protocol.comparison, strict=False):
        assert dict(trial.params) == comparison.point.params()
        assert trial.user_attrs["armrc.comparison"] == comparison.label
    assert "armrc.comparison" not in study.trials[-1].user_attrs


def test_protocol_validation(protocol: EsnSearchProtocol) -> None:
    """Budget, labels, comparison membership, and the estimator bound are validated."""
    with pytest.raises(ValueError, match="budget"):
        replace(protocol, budget=len(protocol.comparison))
    duplicate = (*protocol.comparison, replace(protocol.comparison[0], label=protocol.comparison[0].label))
    with pytest.raises(ValueError, match="unique"):
        replace(protocol, comparison=duplicate)
    outside = ComparisonPoint("outside", replace(protocol.comparison[0].point, alpha=5.0))
    with pytest.raises(ValueError, match="outside the search space"):
        replace(protocol, comparison=(*protocol.comparison, outside))
    with pytest.raises(ValueError, match="max_dt_ratio"):
        replace(protocol, max_dt_ratio=0.5)
    with pytest.raises(ValueError, match="label"):
        ComparisonPoint(" ", protocol.comparison[0].point)
    with pytest.raises(ValueError, match="infeasible_penalty"):
        EsnObjective(infeasible_penalty=0.0)
    with pytest.raises(ValueError, match="name"):
        replace(protocol, name=" ")
    assert isinstance(protocol.search, EsnSearchSpace)
    assert protocol.base_model().name == "esn-task-1a-v2"


def test_protocol_digest_identifies_the_protocol(protocol: EsnSearchProtocol) -> None:
    """The digest is stable for equal protocols, changes with any field, and does not depend on the checkout path."""
    digest = protocol_digest(protocol)
    assert len(digest) == 64
    assert digest == protocol_digest(load_esn_search(PROTOCOL))
    assert digest != protocol_digest(replace(protocol, budget=protocol.budget + 1))
    canonical, same = config_digest(protocol)
    assert same == digest
    assert str(REPO_ROOT) not in canonical  # machine-independent: paths inside the repository are relative
    assert '"configs/tasks/task_1a.toml"' in canonical
    assert '"configs/models/esn_task_1a_v2.toml"' in canonical
