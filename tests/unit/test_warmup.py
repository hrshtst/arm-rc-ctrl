# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-006: per-episode zero reset with configurable common warm-up, including the no-warm-up case."""

from __future__ import annotations

import numpy as np
import pytest

from arm_rc_ctrl.controllers.contracts import RobotState
from arm_rc_ctrl.controllers.estimator import CausalDerivativeEstimator, EstimatorConfig
from arm_rc_ctrl.data.samples import SampleSet
from arm_rc_ctrl.data.synthetic import synthetic_task_arrays, synthetic_task_samples
from arm_rc_ctrl.rc.esn import EsnConfig, EsnModel, ReadoutConfig, ReservoirConfig
from arm_rc_ctrl.rc.generator import RcTargetGenerator
from arm_rc_ctrl.rc.teacher_forcing import ChannelTransform, InputEncoder, InputTransform
from arm_rc_ctrl.rc.training import harvest_episode, train_readout
from arm_rc_ctrl.rc.warmup import (
    APPROVED_WARMUPS_S,
    WarmupConfig,
    build_task_episode,
    warmup_inputs,
    warmup_state,
)

DT = 0.01
SOURCE = "processed-20260830-555555555555"
CONFIG = EsnConfig(
    reservoir=ReservoirConfig(
        n_neurons=30, spectral_radius=0.9, sparsity=0.8, leak_rate=0.6, input_scaling=0.5, seed=3
    ),
    readout=ReadoutConfig(alpha=1e-3),
)
TRANSFORM = InputTransform(
    policy="training_std",
    derived_from=(SOURCE,),
    channels={
        "q": ChannelTransform((0.0, 0.0), (1.0, 1.0)),
        "dq": ChannelTransform((0.0, 0.0), (1.0, 1.0)),
    },
)
ENCODER = InputEncoder(TRANSFORM, 2, 0)
SAMPLES = synthetic_task_samples()


def _model() -> EsnModel:
    return EsnModel(CONFIG, input_dim=4, output_dim=2)


@pytest.mark.parametrize("duration", sorted(APPROVED_WARMUPS_S))
def test_approved_warmup_durations_are_accepted(duration: float) -> None:
    """Every approved D2 duration constructs, including the explicit no-warm-up case."""
    assert WarmupConfig(duration).duration_s == duration


def test_off_protocol_warmup_durations_are_rejected() -> None:
    """A duration outside the approved D2 set fails at construction."""
    with pytest.raises(ValueError, match="approved"):
        WarmupConfig(0.3)


@pytest.mark.parametrize(("duration", "rows"), [(0.0, 0), (0.25, 25), (0.5, 50), (1.0, 100), (2.0, 200)])
def test_row_counts_follow_the_control_grid(duration: float, rows: int) -> None:
    """The warm-up consumes exactly duration/period rows; zero consumes none."""
    assert WarmupConfig(duration).n_rows(DT) == rows


def test_off_grid_periods_are_rejected() -> None:
    """A duration that does not land on the control grid cannot be scheduled."""
    with pytest.raises(ValueError, match="grid"):
        WarmupConfig(0.25).n_rows(0.03)
    with pytest.raises(ValueError, match="period"):
        WarmupConfig(0.25).n_rows(0.0)


def test_warmup_times_precede_task_time_zero() -> None:
    """Warm-up rows live on [-T_w, 0): strictly increasing and ending one period before zero."""
    times = WarmupConfig(0.25).times(DT)
    assert times.shape == (25,)
    assert np.all(np.diff(times) > 0)
    assert times[-1] == pytest.approx(-DT)
    assert times[0] == pytest.approx(-0.25)
    assert WarmupConfig(0.0).times(DT).size == 0


def test_task_episode_prepends_the_repeated_initial_state() -> None:
    """Warm-up rows repeat the encoded [q_0, 0] input before teacher forcing and never enter the loss."""
    episode = build_task_episode(SAMPLES, ENCODER, source=SOURCE, warmup=WarmupConfig(0.5), period_s=DT)
    n_w = 50
    n_task = SAMPLES.n_samples - 1
    assert episode.washout_len == n_w
    assert episode.n_rows == n_w + n_task
    assert int(episode.loss_rows.sum()) == n_task
    q0 = SAMPLES.q[0]
    warm_input = ENCODER.encode(q0, np.zeros(2))
    assert np.array_equal(episode.inputs[:n_w], np.tile(warm_input, (n_w, 1)))
    assert np.array_equal(episode.targets[:n_w], np.tile(q0, (n_w, 1)))
    assert np.array_equal(episode.inputs[n_w:], ENCODER.encode_many(SAMPLES.q[:-1], SAMPLES.dq[:-1]))
    assert np.array_equal(episode.targets[n_w:], SAMPLES.q[1:])
    assert episode.t[0] == pytest.approx(-0.5)
    assert episode.t[n_w] == 0.0


def test_zero_warmup_consumes_no_samples() -> None:
    """T_w = 0 adds no rows and drops none: the episode is the pure task pairing."""
    episode = build_task_episode(SAMPLES, ENCODER, source=SOURCE, warmup=WarmupConfig(0.0), period_s=DT)
    assert episode.washout_len == 0
    assert episode.n_rows == SAMPLES.n_samples - 1
    assert bool(episode.loss_rows.all())
    assert episode.t[0] == 0.0
    assert np.array_equal(episode.inputs, ENCODER.encode_many(SAMPLES.q[:-1], SAMPLES.dq[:-1]))


def test_prime_phases_and_shifted_clocks_are_rejected() -> None:
    """A task episode never contains prime samples and always starts at task time zero."""
    arrays = synthetic_task_arrays()
    arrays["phase"] = arrays["phase"].copy()
    arrays["phase"][0] = 0
    with pytest.raises(ValueError, match="prime"):
        build_task_episode(SampleSet.from_arrays(arrays), ENCODER, source=SOURCE, warmup=WarmupConfig(0.0), period_s=DT)
    shifted = synthetic_task_arrays()
    shifted["t"] = shifted["t"] + DT
    with pytest.raises(ValueError, match=r"start at 0.0"):
        build_task_episode(
            SampleSet.from_arrays(shifted), ENCODER, source=SOURCE, warmup=WarmupConfig(0.0), period_s=DT
        )


def test_no_reservoir_state_crosses_episode_boundaries() -> None:
    """Harvesting an episode is independent of whatever episode was harvested before."""
    model = _model()
    first = build_task_episode(SAMPLES, ENCODER, source=SOURCE, warmup=WarmupConfig(0.25), period_s=DT)
    other = build_task_episode(
        synthetic_task_samples(n=81, dwell_start_s=0.6), ENCODER, source=SOURCE, warmup=WarmupConfig(0.5), period_s=DT
    )
    alone = harvest_episode(model, other).states
    harvest_episode(model, first)
    after = harvest_episode(model, other).states
    assert np.array_equal(alone, after)


def test_zero_warmup_state_is_the_zero_reset() -> None:
    """With T_w = 0 the reservoir enters the task in exactly the all-zero reset state."""
    model = _model()
    state = warmup_state(model, ENCODER, SAMPLES.q[0], WarmupConfig(0.0), DT)
    assert np.array_equal(state, np.zeros(model.n_neurons))


def test_ideal_hold_priming_reproduces_the_training_warmup_state() -> None:
    """Training and evaluation share the activation contract: an ideal hold reaches the training state."""
    model = _model()
    warmup = WarmupConfig(0.25)
    episode = build_task_episode(SAMPLES, ENCODER, source=SOURCE, warmup=warmup, period_s=DT)
    train_readout(model, [episode])
    q0 = SAMPLES.q[0]
    expected = warmup_state(model, ENCODER, q0, warmup, DT).copy()
    generator = RcTargetGenerator(model, ENCODER, CausalDerivativeEstimator(EstimatorConfig(DT), 2))
    generator.reset(RobotState(0.0, q0, np.zeros(2)))
    for k in range(1, warmup.n_rows(DT) + 1):
        generator.prime(RobotState(k * DT, q0, np.zeros(2)))
    assert np.array_equal(model.state(), expected)
    inputs = warmup_inputs(ENCODER, q0, warmup.n_rows(DT))
    assert np.array_equal(inputs[0], ENCODER.encode(q0, np.zeros(2)))
