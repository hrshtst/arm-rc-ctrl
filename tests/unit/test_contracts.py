# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2-001: RobotState, DesiredJointState, and the TargetGenerator contract."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from arm_rc_ctrl.controllers.contracts import (
    DesiredJointState,
    GeneratorError,
    RobotState,
    TargetGenerator,
    TargetGeneratorBase,
    as_joint_vector,
)


def _state(t: float, q: object, dq: object) -> RobotState:
    """Build a RobotState from plain sequences (the contract converts and validates them)."""
    return RobotState(t, cast("Any", q), cast("Any", dq))


def _desired(q: object, dq: object, ddq: object) -> DesiredJointState:
    """Build a DesiredJointState from plain sequences."""
    return DesiredJointState(cast("Any", q), cast("Any", dq), cast("Any", ddq))


class HoldGenerator(TargetGeneratorBase):
    """Returns the reset posture forever; records what it received."""

    def __init__(self) -> None:
        super().__init__()
        self.posture: np.ndarray | None = None
        self.codes: list[np.ndarray] = []

    def _reset(self, initial_state: RobotState) -> None:
        self.posture = initial_state.q

    def _step(self, state: RobotState, task_code: np.ndarray) -> DesiredJointState:
        del state
        self.codes.append(task_code)
        assert self.posture is not None
        return DesiredJointState.hold(self.posture)


class WrongDofGenerator(HoldGenerator):
    """Returns a target with the wrong joint count."""

    def _step(self, state: RobotState, task_code: np.ndarray) -> DesiredJointState:
        del state, task_code
        return DesiredJointState.hold(np.zeros(3))


class NotAStateGenerator(HoldGenerator):
    """Returns a bare array instead of a DesiredJointState."""

    def _step(self, state: RobotState, task_code: np.ndarray) -> DesiredJointState:
        del state, task_code
        return np.zeros(2)  # type: ignore[return-value]


def test_robot_state_invariants() -> None:
    """Time is finite and non-negative; q and dq are finite float64 vectors of one length, stored read-only."""
    state = _state(0.5, [0.1, 0.2], np.array([1, 2], dtype=np.int64))
    assert state.dof == 2
    assert state.q.dtype == np.float64
    assert state.dq.dtype == np.float64
    assert not state.q.flags.writeable
    assert np.array_equal(state.dq, [1.0, 2.0])
    with pytest.raises(ValueError, match="t must be finite and non-negative"):
        _state(-1e-9, [0.0], [0.0])
    with pytest.raises(ValueError, match="t must be finite and non-negative"):
        _state(math.nan, [0.0], [0.0])
    with pytest.raises(ValueError, match="dq must have 2 entries, got 1"):
        _state(0.0, [0.0, 0.0], [0.0])
    with pytest.raises(ValueError, match="q must be a one-dimensional vector"):
        _state(0.0, np.zeros((2, 1)), np.zeros(2))
    with pytest.raises(ValueError, match="q must be a one-dimensional vector"):
        _state(0.0, [], [])
    with pytest.raises(ValueError, match="dq must be finite"):
        _state(0.0, [0.0], [math.inf])


def test_desired_joint_state_invariants() -> None:
    """q, dq, and ddq share one joint count and are finite; hold() is a stationary target."""
    desired = _desired([0.1, 0.2], [0.0, 0.0], [0.0, 0.0])
    assert desired.dof == 2
    assert not desired.ddq.flags.writeable
    held = DesiredJointState.hold([0.3, -0.4])
    assert np.array_equal(held.q, [0.3, -0.4])
    assert not held.dq.any()
    assert not held.ddq.any()
    with pytest.raises(ValueError, match="ddq must have 2 entries, got 3"):
        _desired([0.0, 0.0], [0.0, 0.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="dq must be finite"):
        _desired([0.0], [math.nan], [0.0])


def test_as_joint_vector_copies_and_freezes() -> None:
    """The caller's array is neither aliased nor mutated."""
    source = np.array([1.0, 2.0])
    vector = as_joint_vector(source, "q", dof=2)
    source[0] = 9.0
    assert vector[0] == 1.0
    with pytest.raises(ValueError, match="read-only"):
        vector[0] = 0.0
    with pytest.raises(ValueError, match="q must have 3 entries, got 2"):
        as_joint_vector(source, "q", dof=3)


def test_generator_must_be_reset_before_stepping() -> None:
    """Stepping an un-reset generator is an error; after reset the base tracks steps and time."""
    generator = HoldGenerator()
    assert isinstance(generator, TargetGenerator)
    assert generator.is_reset is False
    with pytest.raises(GeneratorError, match=r"step\(\) called before reset\(\)"):
        generator.step(_state(0.0, [0.0], [0.0]))
    with pytest.raises(GeneratorError, match="has not been reset"):
        _ = generator.dof
    generator.reset(_state(0.0, [0.2, 1.2], [0.0, 0.0]))
    assert generator.is_reset is True
    assert generator.dof == 2
    first = generator.step(_state(0.0, [0.2, 1.2], [0.0, 0.0]))  # the first step may share the reset time
    assert np.array_equal(first.q, [0.2, 1.2])
    second = generator.step(_state(0.01, [0.21, 1.19], [0.5, -0.5]))
    assert np.array_equal(second.q, [0.2, 1.2])
    assert generator.steps == 2


def test_generator_rejects_bad_states_and_codes() -> None:
    """Joint-count changes, non-advancing time, and malformed task codes are contract violations."""
    generator = HoldGenerator()
    generator.reset(_state(1.0, [0.2, 1.2], [0.0, 0.0]))
    with pytest.raises(GeneratorError, match="time must not go backwards"):
        generator.step(_state(0.5, [0.2, 1.2], [0.0, 0.0]))
    generator.step(_state(1.0, [0.2, 1.2], [0.0, 0.0]))
    with pytest.raises(GeneratorError, match="time must advance"):
        generator.step(_state(1.0, [0.2, 1.2], [0.0, 0.0]))
    with pytest.raises(GeneratorError, match="state has 3 joints but the episode was reset with 2"):
        generator.step(_state(1.01, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))
    with pytest.raises(GeneratorError, match="task_code must be a finite one-dimensional float64 vector"):
        generator.step(_state(1.01, [0.2, 1.2], [0.0, 0.0]), np.array([1, 0]))
    with pytest.raises(GeneratorError, match="task_code must be a finite"):
        generator.step(_state(1.01, [0.2, 1.2], [0.0, 0.0]), np.array([[1.0, 0.0]]))
    generator.step(_state(1.01, [0.2, 1.2], [0.0, 0.0]), np.array([1.0, 0.0]))
    generator.step(_state(1.02, [0.2, 1.2], [0.0, 0.0]))
    assert [c.tolist() for c in generator.codes] == [[], [1.0, 0.0], []]
    assert generator.steps == 3


def test_generator_output_is_validated() -> None:
    """A target with the wrong joint count or type is rejected before it reaches the tracker."""
    wrong = WrongDofGenerator()
    wrong.reset(_state(0.0, [0.2, 1.2], [0.0, 0.0]))
    with pytest.raises(GeneratorError, match="returned 3 joints, expected 2"):
        wrong.step(_state(0.0, [0.2, 1.2], [0.0, 0.0]))
    bad = NotAStateGenerator()
    bad.reset(_state(0.0, [0.2, 1.2], [0.0, 0.0]))
    with pytest.raises(GeneratorError, match="returned ndarray, expected DesiredJointState"):
        bad.step(_state(0.0, [0.2, 1.2], [0.0, 0.0]))


def test_reset_starts_a_fresh_episode() -> None:
    """A second reset discards the step count and time of the first episode."""
    generator = HoldGenerator()
    generator.reset(_state(0.0, [0.0, 0.0], [0.0, 0.0]))
    generator.step(_state(5.0, [0.0, 0.0], [0.0, 0.0]))
    generator.reset(_state(0.0, [1.0, 1.0], [0.0, 0.0]))
    assert generator.steps == 0
    target = generator.step(_state(0.0, [1.0, 1.0], [0.0, 0.0]))
    assert np.array_equal(target.q, [1.0, 1.0])


class CountingGenerator(HoldGenerator):
    """Counts priming samples."""

    def __init__(self) -> None:
        super().__init__()
        self.primed: list[float] = []

    def _prime(self, state: RobotState, task_code: np.ndarray) -> None:
        del task_code
        self.primed.append(state.t)


def test_priming_is_validated_like_stepping() -> None:
    """prime() needs a reset, advances the clock, counts as a step, and reaches the implementation hook."""
    generator = CountingGenerator()
    with pytest.raises(GeneratorError, match=r"prime\(\) called before reset\(\)"):
        generator.prime(_state(0.0, [0.0, 0.0], [0.0, 0.0]))
    generator.reset(_state(0.0, [0.2, 1.2], [0.0, 0.0]))
    generator.prime(_state(0.0, [0.2, 1.2], [0.0, 0.0]))
    generator.prime(_state(0.01, [0.2, 1.2], [0.0, 0.0]))
    with pytest.raises(GeneratorError, match="time must advance"):
        generator.prime(_state(0.01, [0.2, 1.2], [0.0, 0.0]))
    with pytest.raises(GeneratorError, match="time must advance"):
        generator.step(_state(0.01, [0.2, 1.2], [0.0, 0.0]))
    assert generator.primed == [0.0, 0.01]
    assert generator.steps == 2
    generator.step(_state(0.02, [0.2, 1.2], [0.0, 0.0]))
    assert generator.steps == 3
    plain = HoldGenerator()
    plain.reset(_state(0.0, [0.2, 1.2], [0.0, 0.0]))
    plain.prime(_state(0.0, [0.2, 1.2], [0.0, 0.0]))  # the default hook ignores the sample
    assert plain.codes == []
