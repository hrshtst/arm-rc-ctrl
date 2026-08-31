# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Deterministic robustness scenarios: seeds, norms, bounds, grid/random modes, IDs, force pulses (M3-007, M3-008)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from arm_rc_ctrl.experiments.confirmatory import ConfirmatoryProtocol, load_confirmatory
from arm_rc_ctrl.experiments.perturbations import (
    CLASS_ORDER,
    RobustnessScenario,
    check_offsets,
    force_scenarios,
    posture_grid,
    posture_random,
    robustness_scenarios,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario

REPO_ROOT = repository_root()
CONFIRMATORY = REPO_ROOT / "configs" / "evaluations" / "task_1a_confirmatory_v2.toml"
TASK = REPO_ROOT / "configs" / "tasks" / "task_1a.toml"


@pytest.fixture(scope="module")
def protocol() -> ConfirmatoryProtocol:
    """The locked task 1-a protocol."""
    return load_confirmatory(CONFIRMATORY)


def test_random_offsets_are_seeded_and_have_the_requested_norm() -> None:
    """Same seed and stream reproduce the draws; other seeds or streams differ; every draw has the level's norm."""
    a = posture_random(0.05, 20260901, 2, 4, stream=2)
    assert a == posture_random(0.05, 20260901, 2, 4, stream=2)
    assert a != posture_random(0.05, 20260902, 2, 4, stream=2)
    assert a != posture_random(0.05, 20260901, 2, 4, stream=3)
    assert len(a) == 4
    assert all(math.isclose(float(np.linalg.norm(o)), 0.05, rel_tol=1e-12) for o in a)
    assert len({tuple(round(v, 6) for v in o) for o in a}) == 4
    directions = [math.atan2(o[1], o[0]) for o in a]
    assert max(directions) - min(directions) > 1.0  # not all in one direction
    with pytest.raises(ValueError, match="posture_random"):
        posture_random(0.0, 1, 2, 1, stream=2)


def test_grid_offsets_follow_the_unit_directions() -> None:
    """Grid mode scales normalized directions to the magnitude."""
    grid = posture_grid(0.1, [(1.0, 0.0), (3.0, 4.0), (0.0, -2.0)])
    assert grid[0] == (0.1, 0.0)
    assert grid[1] == pytest.approx((0.06, 0.08))
    assert grid[2] == pytest.approx((0.0, -0.1))
    with pytest.raises(ValueError, match="non-zero"):
        posture_grid(0.1, [(0.0, 0.0)])
    with pytest.raises(ValueError, match="posture_grid"):
        posture_grid(0.1, [])


def test_offsets_are_checked_against_the_joint_limits() -> None:
    """A perturbed posture outside the limits is refused with the joint named."""
    check_offsets([(0.05, -0.05)], (0.5, 0.5), (0.0, 0.0), (1.0, 1.0))
    with pytest.raises(ValueError, match="joint 1"):
        check_offsets([(0.0, 0.6)], (0.5, 0.5), (0.0, 0.0), (1.0, 1.0))
    with pytest.raises(ValueError, match="entries"):
        check_offsets([(0.0,)], (0.5, 0.5), (0.0, 0.0), (1.0, 1.0))


def test_scenario_ids_are_stable_paired_and_ordered(protocol: ConfirmatoryProtocol) -> None:
    """Both methods get the same IDs, offsets, and pulses; the classes come in protocol order."""
    scenario = load_scenario(TASK)
    nominal = scenario.task.initial_q
    lower = [link.q_min for link in scenario.robot.links]
    upper = [link.q_max for link in scenario.robot.links]
    first = robustness_scenarios(protocol, nominal=nominal, lower=lower, upper=upper)
    second = robustness_scenarios(protocol, nominal=nominal, lower=lower, upper=upper)
    assert first == second
    seeds, draws, directions = len(protocol.seeds), protocol.posture.draws_per_seed, len(protocol.force.directions_deg)
    assert len(first) == 1 + 2 * seeds * draws + directions + seeds * draws
    kinds = [s.kind for s in first]
    assert [k for k in CLASS_ORDER if k in kinds] == list(CLASS_ORDER)
    assert kinds == sorted(kinds, key=CLASS_ORDER.index)
    ids = [s.scenario_id for s in first]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "nominal"
    assert ids[1] == f"posture-small-{protocol.seeds[0]}-00"
    assert f"posture-large-{protocol.seeds[-1]}-{draws - 1:02d}" in ids
    assert f"force-{protocol.force.magnitude_n:g}N-090deg" in ids
    assert ids[-1].startswith(f"combined-{protocol.seeds[-1]}-{draws - 1:02d}-")
    small = [s for s in first if s.kind == "posture_small"]
    large = [s for s in first if s.kind == "posture_large"]
    assert all(math.isclose(float(np.linalg.norm(s.offset)), protocol.posture.small_magnitude_rad) for s in small)
    assert all(math.isclose(float(np.linalg.norm(s.offset)), protocol.posture.large_magnitude_rad) for s in large)
    assert all(s.pulse is None for s in small + large)
    assert first[0].offset == (0.0, 0.0)
    assert first[0].initial_q(nominal) == tuple(nominal)
    assert small[0].initial_q(nominal) == pytest.approx(
        tuple(q + d for q, d in zip(nominal, small[0].offset, strict=True))
    )
    combined = [s for s in first if s.kind == "combined"]
    assert [s.offset for s in combined] == [s.offset for s in small]
    assert [s.direction_deg for s in combined[:directions]] == list(protocol.force.directions_deg)
    assert all(s.pulse is not None for s in combined)
    only = robustness_scenarios(protocol, nominal=nominal, classes=("force", "nominal"))
    assert [s.kind for s in only] == ["nominal", *["force"] * directions]
    with pytest.raises(ValueError, match="unknown perturbation classes"):
        robustness_scenarios(protocol, nominal=nominal, classes=("nominal", "shake"))  # type: ignore[arg-type]
    tight = replace(protocol, posture=replace(protocol.posture, large_magnitude_rad=5.0))
    with pytest.raises(ValueError, match="outside"):
        robustness_scenarios(tight, nominal=nominal, lower=lower, upper=upper)


def test_force_scenarios_carry_timing_direction_and_magnitude(protocol: ConfirmatoryProtocol) -> None:
    """Each pulse acts exactly in the locked window with the locked magnitude along its direction."""
    forces = force_scenarios(protocol.force, 2)
    assert [s.direction_deg for s in forces] == list(protocol.force.directions_deg)
    for scenario in forces:
        pulse = scenario.pulse
        assert pulse is not None
        assert pulse.start_s == protocol.force.start_s
        assert pulse.end_s == pytest.approx(protocol.force.start_s + protocol.force.duration_s)
        assert pulse.magnitude_n == pytest.approx(protocol.force.magnitude_n)
        angle = math.radians(scenario.direction_deg or 0.0)
        expected = np.array([math.cos(angle), math.sin(angle)]) * protocol.force.magnitude_n
        assert np.allclose(pulse.at(pulse.start_s), expected, atol=1e-12)
        assert np.allclose(pulse.at(pulse.start_s + pulse.duration_s / 2), expected, atol=1e-12)
        assert not pulse.at(pulse.end_s).any()
        assert not pulse.at(pulse.start_s - 1e-9).any()
        assert scenario.offset == (0.0, 0.0)
    with pytest.raises(ValueError, match="scenario_id"):
        RobustnessScenario(" ", "nominal", (0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        RobustnessScenario("x", "nominal", (float("nan"), 0.0))
    with pytest.raises(ValueError, match="joints"):
        forces[0].initial_q((0.0,))
