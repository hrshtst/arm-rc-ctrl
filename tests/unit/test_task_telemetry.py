# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-007: versioned task-time generator and warm-up telemetry; hold values are never readout output."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_rc_ctrl.controllers.contracts import RobotState
from arm_rc_ctrl.controllers.estimator import CausalDerivativeEstimator, EstimatorConfig
from arm_rc_ctrl.data.synthetic import synthetic_task_samples
from arm_rc_ctrl.experiments.run_record import OPTIONAL_ARRAYS, REQUIRED_ARRAYS, RunArrays
from arm_rc_ctrl.experiments.simulation import GENERATOR_CHANNELS
from arm_rc_ctrl.rc.esn import EsnConfig, EsnModel, ReadoutConfig, ReservoirConfig
from arm_rc_ctrl.rc.generator import RcTargetGenerator
from arm_rc_ctrl.rc.teacher_forcing import ChannelTransform, InputEncoder, InputTransform
from arm_rc_ctrl.rc.training import train_readout
from arm_rc_ctrl.rc.warmup import WarmupConfig, build_task_episode

N, DOF = 12, 2
DT = 0.01
SOURCE = "processed-20260830-555555555555"


def _arrays(**extra: NDArray[Any]) -> dict[str, NDArray[Any]]:
    t = np.arange(N, dtype=np.float64) * DT
    joint = {
        name: np.random.default_rng(i).standard_normal((N, DOF))
        for i, name in enumerate(REQUIRED_ARRAYS)
        if name not in ("t", "tip", "task_code", "saturation")
    }
    base: dict[str, NDArray[Any]] = {
        "t": t,
        **joint,
        "tip": np.column_stack([np.cos(t), np.sin(t)]),
        "task_code": np.zeros((N, 0)),
        "saturation": np.zeros(N, dtype=np.int64),
        "phase": np.array([0] * 4 + [1] * (N - 4), dtype=np.int64),
    }
    base.update(extra)
    return base


def _masked(active_shape: tuple[int, ...], hold_rows: int, *, active: bool) -> NDArray[np.float64]:
    values = np.random.default_rng(99).standard_normal(active_shape)
    out = np.full(active_shape, np.nan)
    if active:
        out[hold_rows:] = values[hold_rows:]
    else:
        out[:hold_rows] = values[:hold_rows]
    return out


def _telemetry_arrays() -> dict[str, NDArray[Any]]:
    return _arrays(
        generator_output_q=_masked((N, DOF), 4, active=True),
        generator_increment_q=_masked((N, DOF), 4, active=True),
        warmup_state_norm=_masked((N,), 4, active=False),
        warmup_esn_input=_masked((N, 5), 4, active=False),
    )


def test_task_telemetry_arrays_are_accepted_and_ordered() -> None:
    """Masked generator/warm-up channels validate and appear after the M3 optional arrays."""
    arrays = RunArrays(_telemetry_arrays())
    names = list(arrays.specs())
    assert names[: len(REQUIRED_ARRAYS)] == list(REQUIRED_ARRAYS)
    assert set(names) - set(REQUIRED_ARRAYS) == {
        "phase",
        "generator_output_q",
        "generator_increment_q",
        "warmup_state_norm",
        "warmup_esn_input",
    }
    assert "generator_output_q" in OPTIONAL_ARRAYS
    assert "generator_increment_q" in OPTIONAL_ARRAYS
    assert "warmup_state_norm" in OPTIONAL_ARRAYS
    assert "warmup_esn_input" in OPTIONAL_ARRAYS


def test_hold_values_mislabeled_as_readout_are_rejected() -> None:
    """A finite generator_output_q row during the hold phase cannot be stored."""
    arrays = _telemetry_arrays()
    tampered = arrays["generator_output_q"].copy()
    tampered[0] = 0.5  # a hold command mislabeled as readout output
    with pytest.raises(ValueError, match="inactive"):
        RunArrays({**arrays, "generator_output_q": tampered})


def test_missing_readout_values_while_active_are_rejected() -> None:
    """generator_output_q must be finite at every active sample."""
    arrays = _telemetry_arrays()
    tampered = arrays["generator_output_q"].copy()
    tampered[-1] = np.nan
    with pytest.raises(ValueError, match="active"):
        RunArrays({**arrays, "generator_output_q": tampered})


def test_warmup_channels_are_hold_only() -> None:
    """Warm-up state norms and inputs exist only before activation."""
    arrays = _telemetry_arrays()
    tampered = arrays["warmup_state_norm"].copy()
    tampered[-1] = 1.0
    with pytest.raises(ValueError, match="warm"):
        RunArrays({**arrays, "warmup_state_norm": tampered})
    inputs = arrays["warmup_esn_input"].copy()
    inputs[:4] = np.nan
    with pytest.raises(ValueError, match="warm"):
        RunArrays({**arrays, "warmup_esn_input": inputs})


def test_generator_channels_require_the_phase_array() -> None:
    """Without the activation phase the masked channels cannot be interpreted."""
    arrays = _telemetry_arrays()
    del arrays["phase"]
    with pytest.raises(ValueError, match="phase"):
        RunArrays(arrays)


def test_old_m3_run_arrays_are_preserved() -> None:
    """Arrays without any new channel (the M3 shape) still validate exactly as before."""
    base = _arrays()
    del base["phase"]
    assert list(RunArrays(base).specs()) == list(REQUIRED_ARRAYS)


def test_generator_channel_map_covers_the_new_telemetry() -> None:
    """The generator channel map records output and warm-up channels; the increment stays residual-only."""
    assert GENERATOR_CHANNELS.generator_output_q == "generator_output_q"
    assert GENERATOR_CHANNELS.warmup_state_norm == "warmup_state_norm"
    assert GENERATOR_CHANNELS.warmup_esn_input == "warmup_esn_input"
    assert GENERATOR_CHANNELS.generator_increment_q is None


@pytest.fixture(scope="module")
def trained_generator() -> tuple[RcTargetGenerator, EsnModel, np.ndarray]:
    """A generator fitted on a synthetic task episode with an identity encoder."""
    samples = synthetic_task_samples()
    transform = InputTransform(
        policy="training_std",
        derived_from=(SOURCE,),
        channels={
            "q": ChannelTransform((0.0, 0.0), (1.0, 1.0)),
            "dq": ChannelTransform((0.0, 0.0), (1.0, 1.0)),
        },
    )
    encoder = InputEncoder(transform, 2, 0)
    model = EsnModel(
        EsnConfig(
            reservoir=ReservoirConfig(
                n_neurons=30, spectral_radius=0.9, sparsity=0.8, leak_rate=0.6, input_scaling=0.5, seed=3
            ),
            readout=ReadoutConfig(alpha=1e-3),
        ),
        input_dim=4,
        output_dim=2,
    )
    episode = build_task_episode(samples, encoder, source=SOURCE, warmup=WarmupConfig(0.25), period_s=DT)
    train_readout(model, [episode])
    estimator = CausalDerivativeEstimator(EstimatorConfig(DT), 2)
    return RcTargetGenerator(model, encoder, estimator), model, samples.q[0]


def test_priming_never_emits_a_readout_value(
    trained_generator: tuple[RcTargetGenerator, EsnModel, np.ndarray],
) -> None:
    """While the readout is inactive the output channel is NaN and the warm-up channels are populated."""
    generator, _, q0 = trained_generator
    generator.reset(RobotState(0.0, q0, np.zeros(2)))
    generator.prime(RobotState(DT, q0, np.zeros(2)))
    last = generator.last
    assert np.all(np.isnan(last["generator_output_q"]))
    assert np.all(np.isfinite(last["warmup_state_norm"]))
    assert np.all(np.isfinite(last["warmup_esn_input"]))
    assert last["warmup_esn_input"].shape == (4,)
    assert last["generating"][0] == 0.0


def test_stepping_emits_the_readout_and_clears_warmup_channels(
    trained_generator: tuple[RcTargetGenerator, EsnModel, np.ndarray],
) -> None:
    """Once active, generator_output_q is the prediction and the warm-up channels are NaN."""
    generator, _, q0 = trained_generator
    generator.reset(RobotState(0.0, q0, np.zeros(2)))
    generator.prime(RobotState(DT, q0, np.zeros(2)))
    desired = generator.step(RobotState(2 * DT, q0, np.zeros(2)))
    last = generator.last
    assert np.array_equal(last["generator_output_q"], desired.q)
    assert np.array_equal(last["generator_output_q"], last["q_generated"])
    assert np.all(np.isnan(last["warmup_state_norm"]))
    assert np.all(np.isnan(last["warmup_esn_input"]))
    assert last["generating"][0] == 1.0
