# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2-007: RcTargetGenerator forms every input from actual feedback and rejects invalid output."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

from arm_rc_ctrl.controllers.contracts import GeneratorError, RobotState
from arm_rc_ctrl.controllers.estimator import CausalDerivativeEstimator, EstimatorConfig
from arm_rc_ctrl.data.normalization import fit_normalization
from arm_rc_ctrl.data.samples import SampleSet
from arm_rc_ctrl.rc.esn import EsnConfig, EsnModel, ReadoutConfig, ReservoirConfig
from arm_rc_ctrl.rc.generator import RcTargetGenerator
from arm_rc_ctrl.rc.teacher_forcing import InputEncoder, InputTransform, build_episode
from arm_rc_ctrl.rc.training import train_readout

DT = 0.01
ESN = EsnConfig(
    reservoir=ReservoirConfig(
        n_neurons=80, spectral_radius=0.8, sparsity=0.9, leak_rate=0.3, input_scaling=0.3, seed=9
    ),
    readout=ReadoutConfig(alpha=1e-6),
)


def _samples(n: int = 300) -> SampleSet:
    t = np.arange(n, dtype=np.float64) * DT
    omega = np.array([2.0, 3.0])
    q = np.sin(omega[None, :] * t[:, None]) * np.array([0.5, 0.3])
    dq = omega[None, :] * np.cos(omega[None, :] * t[:, None]) * np.array([0.5, 0.3])
    phase = np.array([0] * 30 + [1] * 240 + [2] * (n - 270), dtype=np.int64)
    return SampleSet(t, q, dq, np.zeros((n, 2)), q * 0.1, np.zeros((n, 2)), np.zeros((n, 2)), np.zeros((n, 0)), phase)


@pytest.fixture(scope="module")
def trained() -> tuple[EsnModel, InputEncoder, SampleSet]:
    """A model fitted on the sinusoid dataset with its encoder."""
    samples = _samples()
    normalization = fit_normalization(
        samples.arrays(),
        ("q", "dq"),
        fitted_on=("processed-20260830-555555555555",),
        training_rows=np.ones(300, dtype=np.bool_),
    )
    encoder = InputEncoder(
        InputTransform.derive("fixed_scale", normalization, fixed_scales={"q": 0.3, "dq": 4.0}), 2, 0
    )
    model = EsnModel(ESN, input_dim=4, output_dim=2)
    train_readout(model, [build_episode(samples, encoder, source="processed-20260830-555555555555")])
    return model, encoder, samples


def _generator(
    trained: tuple[EsnModel, InputEncoder, SampleSet],
    position_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> RcTargetGenerator:
    model, encoder, _ = trained
    estimator = CausalDerivativeEstimator(EstimatorConfig(DT), 2)
    return RcTargetGenerator(model, encoder, estimator, position_bounds=position_bounds)


def _state(t: float, q: np.ndarray, dq: np.ndarray) -> RobotState:
    return RobotState(t, cast("Any", q), cast("Any", dq))


def _returning(value: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """A stand-in for ``EsnModel.step`` that ignores its input and returns ``value``."""

    def step(_u: np.ndarray) -> np.ndarray:
        return value

    return step


def test_generation_follows_the_demonstration_when_fed_demonstrated_states(
    trained: tuple[EsnModel, InputEncoder, SampleSet],
) -> None:
    """Priming with the hold, then stepping with demonstrated feedback, reproduces the next positions."""
    _, _, samples = trained
    generator = _generator(trained)
    generator.reset(_state(samples.t[0], samples.q[0], samples.dq[0]))
    assert generator.hold_posture is not None
    for k in range(30):
        generator.prime(_state(samples.t[k], samples.q[k], samples.dq[k]))
        assert np.array_equal(generator.last["q_generated"], samples.q[0])  # the hold posture
        assert generator.last["generating"][0] == 0.0
        assert not generator.last["dq_desired"].any()
    errors: list[float] = []
    for k in range(30, 299):
        target = generator.step(_state(samples.t[k], samples.q[k], samples.dq[k]))
        errors.append(float(np.abs(target.q - samples.q[k + 1]).max()))
        assert generator.last["generating"][0] == 1.0
    assert max(errors) < 0.02
    assert generator.last["esn_state_norm"][0] > 0
    assert generator.steps == 299
    assert set(generator.last) >= {"esn_input", "q_generated", "dq_desired_raw", "dq_desired", "ddq_desired"}


def test_inputs_come_from_measured_feedback_not_the_previous_prediction(
    trained: tuple[EsnModel, InputEncoder, SampleSet],
) -> None:
    """When the robot lags the target, the next input encodes the measured state, not the last output."""
    _, encoder, samples = trained
    generator = _generator(trained)
    generator.reset(_state(0.0, samples.q[0], samples.dq[0]))
    generator.prime(_state(0.0, samples.q[0], samples.dq[0]))
    measured_q = samples.q[40] + 0.05  # deliberately off the demonstration
    measured_dq = samples.dq[40] * 0.5
    first = generator.step(_state(DT, measured_q, measured_dq))
    assert np.array_equal(generator.last["esn_input"], encoder.encode(measured_q, measured_dq))
    lagging_q = first.q - 0.03
    second = generator.step(_state(2 * DT, lagging_q, measured_dq))
    assert np.array_equal(generator.last["esn_input"], encoder.encode(lagging_q, measured_dq))
    assert not np.array_equal(generator.last["esn_input"], encoder.encode(first.q, measured_dq))
    assert second.dq == pytest.approx((second.q - first.q) / DT)  # raw estimator output without a cutoff


def test_invalid_predictions_are_rejected(
    trained: tuple[EsnModel, InputEncoder, SampleSet], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-finite, mis-shaped, or out-of-bounds targets raise GeneratorError before reaching the tracker."""
    model, _, samples = trained
    generator = _generator(trained, (np.array([-1.0, -1.0]), np.array([1.0, 1.0])))
    generator.reset(_state(0.0, samples.q[0], samples.dq[0]))
    monkeypatch.setattr(model, "step", _returning(np.array([np.nan, 0.0])))
    with pytest.raises(GeneratorError, match="non-finite target") as non_finite:
        generator.step(_state(0.0, samples.q[0], samples.dq[0]))
    assert non_finite.value.category == "non_finite"
    monkeypatch.setattr(model, "step", _returning(np.array([0.0, 0.0, 0.0])))
    with pytest.raises(GeneratorError, match=r"target of shape \(3,\)") as shape:
        generator.step(_state(DT, samples.q[0], samples.dq[0]))
    assert shape.value.category == "shape"
    monkeypatch.setattr(model, "step", _returning(np.array([1.5, 0.0])))
    with pytest.raises(GeneratorError, match="leaves the joint bounds") as bounds:
        generator.step(_state(2 * DT, samples.q[0], samples.dq[0]))
    assert bounds.value.category == "bounds"
    monkeypatch.undo()


def test_construction_checks(trained: tuple[EsnModel, InputEncoder, SampleSet]) -> None:
    """An unfitted model, mismatched widths, or malformed bounds are refused."""
    model, encoder, _ = trained
    estimator = CausalDerivativeEstimator(EstimatorConfig(DT), 2)
    with pytest.raises(ValueError, match="must be fitted"):
        RcTargetGenerator(EsnModel(ESN, input_dim=4, output_dim=2), encoder, estimator)
    with pytest.raises(ValueError, match="incompatible widths"):
        RcTargetGenerator(model, encoder, CausalDerivativeEstimator(EstimatorConfig(DT), 3))
    with pytest.raises(ValueError, match="lower < upper"):
        RcTargetGenerator(model, encoder, estimator, position_bounds=(np.array([1.0, 1.0]), np.array([0.0, 0.0])))
    with pytest.raises(ValueError, match="must be finite"):
        RcTargetGenerator(model, encoder, estimator, position_bounds=(np.array([-np.inf, 0.0]), np.array([1.0, 1.0])))
    generator = RcTargetGenerator(model, encoder, estimator)
    with pytest.raises(GeneratorError, match=r"step\(\) called before reset\(\)"):
        generator.step(_state(0.0, np.zeros(2), np.zeros(2)))


def test_reset_clears_reservoir_and_estimator_state(trained: tuple[EsnModel, InputEncoder, SampleSet]) -> None:
    """A second episode reproduces the first exactly."""
    _, _, samples = trained
    generator = _generator(trained)

    def episode() -> list[np.ndarray]:
        generator.reset(_state(samples.t[0], samples.q[0], samples.dq[0]))
        for k in range(10):
            generator.prime(_state(samples.t[k], samples.q[k], samples.dq[k]))
        return [generator.step(_state(samples.t[k], samples.q[k], samples.dq[k])).q for k in range(10, 60)]

    first, second = episode(), episode()
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert generator.steps == 60
