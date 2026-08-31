# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2-002: teacher-forcing pairs ``[q_k, dq_k] -> q_(k+1)`` with washout rows excluded from the loss."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from arm_rc_ctrl.data.normalization import fit_normalization
from arm_rc_ctrl.data.records import Normalization
from arm_rc_ctrl.data.samples import PHASE_CODES, SampleSet
from arm_rc_ctrl.rc.teacher_forcing import ChannelTransform, Episode, InputEncoder, InputTransform, build_episode

SOURCE = "processed-20260830-555555555555"


def _samples(n: int = 6, *, code_dim: int = 0, phases: list[int] | None = None) -> SampleSet:
    """A tiny dataset whose values identify their sample index: q = k * [1, 10], dq = k * [0.1, 1]."""
    k = np.arange(n, dtype=np.float64)[:, None]
    phase = np.array(phases if phases is not None else [0, 0, 1, 1, 2, 2][:n], dtype=np.int64)
    return SampleSet(
        t=np.arange(n, dtype=np.float64) * 0.01,
        q=k * np.array([1.0, 10.0]),
        dq=k * np.array([0.1, 1.0]),
        ddq=np.zeros((n, 2)),
        tip=k * np.array([0.01, 0.02]),
        dtip=np.zeros((n, 2)),
        ddtip=np.zeros((n, 2)),
        task_code=np.tile(np.arange(code_dim, dtype=np.float64) + 1.0, (n, 1)) * (k + 1),
        phase=phase,
    )


def _encoder(samples: SampleSet) -> InputEncoder:
    normalization = fit_normalization(
        samples.arrays(), ("q", "dq"), fitted_on=(SOURCE,), training_rows=np.ones(samples.n_samples, dtype=np.bool_)
    )
    return InputEncoder.from_normalization(normalization, dof=samples.dof, task_code_dim=samples.task_code_dim)


def test_rows_pair_each_sample_with_its_successor() -> None:
    """Row k holds the normalized state of sample k and the raw joint position of sample k + 1."""
    samples = _samples()
    encoder = _encoder(samples)
    episode = build_episode(samples, encoder, source=SOURCE)
    assert episode.n_rows == samples.n_samples - 1
    assert (episode.input_dim, episode.dof) == (4, 2)
    assert np.array_equal(episode.t, samples.t[:-1])
    assert np.array_equal(episode.targets, samples.q[1:])  # exact one-step shift, in radians
    assert not np.array_equal(episode.targets, samples.q[:-1])
    transform = encoder.transform
    assert transform.policy == "training_std"
    expected = np.hstack([transform.transform("q", samples.q[:-1]), transform.transform("dq", samples.dq[:-1])])
    assert np.array_equal(episode.inputs, expected)
    assert transform.inverse("q", episode.inputs[:, :2]) == pytest.approx(samples.q[:-1])
    assert episode.source == SOURCE
    assert not episode.inputs.flags.writeable


def test_prime_rows_are_washout_and_excluded_from_the_loss() -> None:
    """Rows whose input sample is in the prime interval drive the reservoir only."""
    samples = _samples()
    episode = build_episode(samples, _encoder(samples), source=SOURCE)
    assert episode.loss_rows.tolist() == [False, False, True, True, True]
    assert episode.washout_len == 2
    # the row pairing the last prime sample with the first movement sample is still washout: its input is prime
    assert samples.phase[1] == PHASE_CODES["prime"]
    assert samples.phase[2] == PHASE_CODES["move"]
    assert episode.loss_rows[1] is np.False_


def test_task_code_is_appended_after_the_normalized_state() -> None:
    """Task-code columns follow the state columns unchanged."""
    samples = _samples(code_dim=2)
    encoder = _encoder(samples)
    episode = build_episode(samples, encoder, source=SOURCE)
    assert (encoder.input_dim, episode.input_dim) == (6, 6)
    assert np.array_equal(episode.inputs[:, 4:], samples.task_code[:-1])
    single = encoder.encode(samples.q[3], samples.dq[3], samples.task_code[3])
    assert np.array_equal(single, episode.inputs[3])


def test_encoder_single_and_batch_agree_without_task_code() -> None:
    """encode() is row-wise encode_many(); a missing task code means zero code columns."""
    samples = _samples()
    encoder = _encoder(samples)
    batch = encoder.encode_many(samples.q, samples.dq)
    for k in range(samples.n_samples):
        assert np.array_equal(encoder.encode(samples.q[k], samples.dq[k]), batch[k])
    assert batch.shape == (samples.n_samples, 4)
    assert batch.dtype == np.float64


def test_encoder_validates_statistics_and_shapes() -> None:
    """Missing channel statistics, joint-count mismatches, and non-finite input are errors."""
    samples = _samples()
    only_q = fit_normalization(samples.arrays(), ("q",), fitted_on=(SOURCE,), training_rows=np.ones(6, dtype=np.bool_))
    with pytest.raises(ValueError, match="lacks statistics for input channel 'dq'"):
        InputEncoder.from_normalization(only_q, dof=2, task_code_dim=0)
    full = fit_normalization(
        samples.arrays(), ("q", "dq"), fitted_on=(SOURCE,), training_rows=np.ones(6, dtype=np.bool_)
    )
    with pytest.raises(ValueError, match="covers 2 joints, expected 3"):
        InputEncoder.from_normalization(full, dof=3, task_code_dim=0)
    with pytest.raises(ValueError, match="dof must be >= 1"):
        InputEncoder(InputTransform.derive("training_std", full), 0, 0)
    encoder = InputEncoder.from_normalization(full, dof=2, task_code_dim=1)
    with pytest.raises(ValueError, match="task_code must have 1 columns, got 2"):
        encoder.encode_many(samples.q, samples.dq, np.zeros((6, 2)))
    with pytest.raises(ValueError, match="dq must have shape \\(6, k\\)"):
        encoder.encode_many(samples.q, samples.dq[:-1])
    with pytest.raises(ValueError, match="q must be finite"):
        encoder.encode(np.array([np.nan, 0.0]), np.zeros(2), np.zeros(1))
    with pytest.raises(ValueError, match="q and dq must have 2 joints"):
        encoder.encode_many(np.zeros((6, 3)), np.zeros((6, 3)), np.zeros((6, 1)))


def test_build_episode_checks_dataset_against_encoder() -> None:
    """A dataset with another joint count or task-code width cannot be encoded silently."""
    samples = _samples(code_dim=1)
    encoder = _encoder(_samples())
    with pytest.raises(ValueError, match="task_code_dim 1; the encoder expects 2 and 0"):
        build_episode(samples, encoder, source=SOURCE)


def test_episode_invariants() -> None:
    """Washout rows form a leading block, at least one loss row exists, and shapes align."""
    t = np.arange(4, dtype=np.float64)
    inputs = np.zeros((4, 3))
    targets = np.zeros((4, 2))
    good = Episode(SOURCE, t, inputs, targets, np.array([False, True, True, True]))
    assert good.washout_len == 1
    assert Episode(SOURCE, t, inputs, targets, np.ones(4, dtype=np.bool_)).washout_len == 0
    with pytest.raises(ValueError, match="leading washout block"):
        Episode(SOURCE, t, inputs, targets, np.array([True, False, True, True]))
    with pytest.raises(ValueError, match="leading washout block"):
        Episode(SOURCE, t, inputs, targets, np.zeros(4, dtype=np.bool_))
    with pytest.raises(ValueError, match=r"targets must have shape \(4, k\)"):
        Episode(SOURCE, t, inputs, targets[:3], np.ones(4, dtype=np.bool_))
    with pytest.raises(ValueError, match="loss_rows boolean"):
        Episode(SOURCE, t, inputs, targets, cast("Any", np.ones(4, dtype=np.int64)))
    with pytest.raises(ValueError, match="strictly increasing"):
        Episode(SOURCE, t[::-1].copy(), inputs, targets, np.ones(4, dtype=np.bool_))
    with pytest.raises(ValueError, match="inputs must be finite"):
        Episode(SOURCE, t, np.full((4, 3), np.inf), targets, np.ones(4, dtype=np.bool_))
    with pytest.raises(ValueError, match="source must name"):
        Episode(" ", t, inputs, targets, np.ones(4, dtype=np.bool_))


def test_encoder_from_a_processed_record_normalization() -> None:
    """The recorded Normalization (as stored in processed records) builds the same encoder."""
    samples = _samples()
    encoder = _encoder(samples)
    stats = fit_normalization(
        samples.arrays(), ("q", "dq"), fitted_on=(SOURCE,), training_rows=np.ones(6, dtype=np.bool_)
    )
    recorded = Normalization(fitted_on=(SOURCE,), channels=dict(stats.channels))
    rebuilt = InputEncoder.from_normalization(recorded, dof=2, task_code_dim=0)
    assert np.array_equal(rebuilt.encode_many(samples.q, samples.dq), encoder.encode_many(samples.q, samples.dq))


def test_fixed_scale_policy_keeps_training_centers_and_shares_one_scale() -> None:
    """fixed_scale centers stay the training means; every joint of a channel gets the same physical scale."""
    samples = _samples()
    stats = fit_normalization(
        samples.arrays(), ("q", "dq"), fitted_on=(SOURCE,), training_rows=np.ones(6, dtype=np.bool_)
    )
    fixed = InputTransform.derive("fixed_scale", stats, fixed_scales={"q": 0.3, "dq": 4.0})
    assert fixed.policy == "fixed_scale"
    assert fixed.derived_from == (SOURCE,)
    assert fixed.fixed_scales == {"q": 0.3, "dq": 4.0}
    assert fixed.channels["q"].center == stats.channels["q"].mean
    assert fixed.channels["q"].scale == (0.3, 0.3)
    assert fixed.channels["dq"].scale == (4.0, 4.0)
    assert fixed.dof == 2
    encoder = InputEncoder(fixed, 2, 0)
    encoded = encoder.encode(samples.q[3], samples.dq[3])
    assert encoded == pytest.approx(
        np.concatenate(
            [(samples.q[3] - stats.channels["q"].mean) / 0.3, (samples.dq[3] - stats.channels["dq"].mean) / 4.0]
        )
    )
    # re-deriving with the same inputs gives an equal transform (what the runtime binding relies on)
    assert InputTransform.derive("fixed_scale", stats, fixed_scales={"q": 0.3, "dq": 4.0}) == fixed
    assert InputTransform.derive("training_std", stats) != fixed


def test_transform_validation() -> None:
    """Policies, channel sets, sources, and scale consistency are validated."""
    samples = _samples()
    stats = fit_normalization(
        samples.arrays(), ("q", "dq"), fitted_on=(SOURCE,), training_rows=np.ones(6, dtype=np.bool_)
    )
    good = InputTransform.derive("training_std", stats)
    with pytest.raises(ValueError, match="policy must be one of"):
        InputTransform("median", good.derived_from, good.channels)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="channels must be exactly"):
        InputTransform("training_std", good.derived_from, {"q": good.channels["q"]})
    with pytest.raises(ValueError, match="derived_from must list the artifact IDs"):
        InputTransform("training_std", ("nope",), good.channels)
    with pytest.raises(ValueError, match="fixed_scale needs fixed_scales"):
        InputTransform("fixed_scale", good.derived_from, good.channels)
    with pytest.raises(ValueError, match="scales must all equal"):
        InputTransform("fixed_scale", good.derived_from, good.channels, {"q": 0.3, "dq": 4.0})
    with pytest.raises(ValueError, match="only meaningful for the fixed_scale policy"):
        InputTransform("training_std", good.derived_from, good.channels, {"q": 0.3, "dq": 4.0})
    with pytest.raises(ValueError, match="fixed_scale needs a scale for channel 'dq'"):
        InputTransform.derive("fixed_scale", stats, fixed_scales={"q": 0.3})
    with pytest.raises(ValueError, match=r"fixed_scales\['q'\] must be positive"):
        InputTransform.derive("fixed_scale", stats, fixed_scales={"q": 0.0, "dq": 4.0})
    with pytest.raises(ValueError, match="scale positive"):
        ChannelTransform((0.0, 0.0), (1.0, 0.0))
    with pytest.raises(ValueError, match="same non-zero length"):
        ChannelTransform((0.0,), (1.0, 1.0))
    with pytest.raises(ValueError, match="expects 2 columns"):
        good.transform("q", np.zeros((3, 3)))
