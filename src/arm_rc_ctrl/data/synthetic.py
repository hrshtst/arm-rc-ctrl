# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Deterministic synthetic sample sets for tests and fixtures (no randomness)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from skelarm import StateLog

from arm_rc_ctrl.data.manual_scenario import ManualScenarioConfig, manual_build_skeleton, manual_endpoint_positions
from arm_rc_ctrl.data.samples import PHASE_DWELL, PHASE_MOVE, PHASE_PRIME, SampleSet

__all__ = [
    "synthetic_arrays",
    "synthetic_demonstration_log",
    "synthetic_manual_take_log",
    "synthetic_samples",
    "synthetic_task_arrays",
    "synthetic_task_samples",
]


def synthetic_arrays(
    n: int = 6, dof: int = 2, task_dim: int = 2, code_dim: int = 0, period_s: float = 0.01
) -> dict[str, NDArray[Any]]:
    """Schema-conforming arrays from analytic signals: first sample prime, last sample dwell, rest move."""
    t: NDArray[Any] = np.arange(n, dtype=np.float64) * period_s
    q: NDArray[Any] = np.stack([np.sin(t + j) for j in range(dof)], axis=1) if dof else np.zeros((n, 0))
    tip: NDArray[Any] = np.stack([np.cos(t + k) for k in range(task_dim)], axis=1) if task_dim else np.zeros((n, 0))
    phase: NDArray[Any] = np.full(n, PHASE_MOVE, dtype=np.int64)
    if n:
        phase[0] = PHASE_PRIME
        phase[-1] = PHASE_DWELL
    arrays: dict[str, NDArray[Any]] = {
        "t": t,
        "q": q,
        "dq": np.gradient(q, axis=0) if n > 1 else np.zeros_like(q),
        "ddq": np.zeros_like(q),
        "tip": tip,
        "dtip": np.gradient(tip, axis=0) if n > 1 else np.zeros_like(tip),
        "ddtip": np.zeros_like(tip),
        "task_code": np.zeros((n, code_dim), dtype=np.float64),
        "phase": phase,
    }
    return arrays


def synthetic_samples(n: int = 6, dof: int = 2, task_dim: int = 2, code_dim: int = 0) -> SampleSet:
    """A valid :class:`SampleSet` built from :func:`synthetic_arrays`."""
    return SampleSet.from_arrays(synthetic_arrays(n, dof, task_dim, code_dim))


def synthetic_task_arrays(
    n: int = 101, dof: int = 2, period_s: float = 0.01, dwell_start_s: float = 0.8
) -> dict[str, NDArray[Any]]:
    """Move/dwell-only task-clock arrays (no prime): a smoothstep reach that holds through the dwell."""
    t: NDArray[Any] = np.arange(n, dtype=np.float64) * period_s
    if not 0.0 < dwell_start_s < float(t[-1]):
        msg = f"dwell_start_s must lie inside the episode (0, {float(t[-1])}), got {dwell_start_s}"
        raise ValueError(msg)
    start = 0.3 + 0.3 * np.arange(dof, dtype=np.float64)
    goal = start + np.array([0.5 - 0.7 * j for j in range(dof)], dtype=np.float64)
    s = np.clip(t / dwell_start_s, 0.0, 1.0)
    blend = s * s * (3.0 - 2.0 * s)
    q: NDArray[Any] = start[None, :] + blend[:, None] * (goal - start)[None, :]
    tip: NDArray[Any] = np.stack([np.cos(t + k) for k in range(2)], axis=1)
    phase: NDArray[Any] = np.where(t < dwell_start_s, PHASE_MOVE, PHASE_DWELL).astype(np.int64)
    return {
        "t": t,
        "q": q,
        "dq": np.gradient(q, period_s, axis=0),
        "ddq": np.zeros_like(q),
        "tip": tip,
        "dtip": np.gradient(tip, period_s, axis=0),
        "ddtip": np.zeros_like(tip),
        "task_code": np.zeros((n, 0), dtype=np.float64),
        "phase": phase,
    }


def synthetic_task_samples(n: int = 101, dof: int = 2, period_s: float = 0.01, dwell_start_s: float = 0.8) -> SampleSet:
    """A valid move/dwell-only :class:`SampleSet` built from :func:`synthetic_task_arrays`."""
    return SampleSet.from_arrays(synthetic_task_arrays(n, dof, period_s, dwell_start_s))


def synthetic_demonstration_log(*, dt: float = 0.01, duration: float = 0.3) -> StateLog:
    """A short, deterministic planar 2-DOF PD reach recorded as a ``skelarm`` log.

    Used to build the committed raw fixture; the log embeds the skeleton and
    the ``q``, ``dq``, ``tau``, ``q_ref``, and ``error`` channels.
    """
    from skelarm import JointPD, LinkProp, SampledJointReference, Skeleton, simulate_controlled

    props = [
        LinkProp(length=0.30, m=1.0, i=0.0075, rgx=0.15, rgy=0.0, qmin=-3.0, qmax=3.0),
        LinkProp(length=0.25, m=0.6, i=0.0031, rgx=0.125, rgy=0.0, qmin=-3.0, qmax=3.0),
    ]
    skeleton = Skeleton(props)
    skeleton.q = np.array([0.3, 0.6])
    skeleton.dq = np.zeros(2)
    target = np.array([0.8, 0.4])
    zeros = np.zeros(2)
    reference = SampledJointReference([0.0, duration], [target, target], [zeros, zeros], [zeros, zeros])
    controller = JointPD(reference, kp=np.array([20.0, 10.0]), kd=np.array([2.0, 1.0]))
    return simulate_controlled(skeleton, controller, duration=duration, dt=dt)


_LATE_TICK_FACTOR = 1.5
_DISPLAY_PERIOD_S = 0.02  # the recorder repaints every 20 ms, in whole ticks


def synthetic_manual_take_log(
    config: ManualScenarioConfig,
    *,
    goal_q: tuple[float, ...],
    hold_s: float = 0.5,
    move_s: float = 0.6,
    dwell_s: float = 0.3,
    sample_period_s: float | None = None,
    jitter_s: float = 0.0,
    seed: int = 0,
    gap_at_s: float | None = None,
    gap_s: float = 0.05,
    jump_at_s: float | None = None,
    start_offset_rad: float = 0.0,
    preroll_amplitude_rad: float = 0.0,
    preroll_frequency_hz: float = 2.0,
    history: tuple[int, ...] = (1, 2),
) -> StateLog:
    """A take in the layout of the pinned trajectory recorder (IK mode) for tests and fixtures.

    Frames are acquired every ``sample_period_s`` (default: the task period
    ``timing.dt``). The arm is at the configured reset posture for ``hold_s``
    (the ``t = 0`` frame included; one sample period leaves only that frame, so
    the take departs at its second sample), reaches ``goal_q`` along a
    minimum-jerk profile over ``move_s``, and rests there for ``dwell_s`` plus one
    frame. ``preroll_amplitude_rad`` adds a natural pre-roll wobble
    ``A sin(2 pi f t)`` (joint pattern ``1, -0.5, ...``) that starts at zero on the
    first frame and fades out with the reach, so the dwell stays exact.
    ``jitter_s`` perturbs the wall-clock frame times uniformly (deterministic in
    ``seed``) and the motion is sampled at those actual times; ``gap_at_s``
    delays every later frame by ``gap_s`` (a missing interval after which the
    motion resumes), ``jump_at_s`` adds an instantaneous 0.5 rad step on the first
    joint that persists, and ``start_offset_rad`` displaces the recorded start from
    the reset posture. The log carries ``q``, ``tip`` (forward kinematics),
    ``nominal_time``, and the recorder's ``[extra.acquisition]`` and
    ``[extra.display]`` tables.
    """
    period = config.timing.dt if sample_period_s is None else sample_period_s
    reset = np.asarray(config.task.initial_q, dtype=np.float64) + np.array(
        [start_offset_rad] + [0.0] * (config.dof - 1), dtype=np.float64
    )
    goal = np.asarray(goal_q, dtype=np.float64)
    n_hold, n_move, n_dwell = (round(x / period) for x in (hold_s, move_s, dwell_s))
    n = n_hold + n_move + n_dwell + 1
    offsets: NDArray[np.float64] = np.zeros(n, dtype=np.float64)
    if jitter_s > 0:
        rng = np.random.default_rng(seed)
        offsets[1:] = rng.uniform(-jitter_s, jitter_s, size=n - 1)
    position = np.arange(n, dtype=np.float64) + offsets / period  # the motion clock in frames at the actual times
    q: NDArray[np.float64] = np.tile(reset, (n, 1))
    fade: NDArray[np.float64] = np.ones(n, dtype=np.float64)
    if n_move:
        tau = np.clip((position[n_hold : n_hold + n_move] - n_hold + 1.0) / n_move, 0.0, 1.0)
        profile = tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)
        q[n_hold : n_hold + n_move] = reset[None, :] + profile[:, None] * (goal - reset)[None, :]
        q[n_hold + n_move :] = goal
        fade[n_hold : n_hold + n_move] = 1.0 - profile
        fade[n_hold + n_move :] = 0.0
    if preroll_amplitude_rad > 0:
        wave = preroll_amplitude_rad * np.sin(2.0 * np.pi * preroll_frequency_hz * position * period) * fade
        pattern = np.array([1.0] + [-0.5] * (config.dof - 1), dtype=np.float64)
        q += wave[:, None] * pattern[None, :]
    if jump_at_s is not None:
        k = round(jump_at_s / period)
        q[k:, 0] += 0.5
    times: NDArray[np.float64] = np.arange(n, dtype=np.float64) * period + offsets
    if gap_at_s is not None:
        times[round(gap_at_s / period) :] += gap_s
    nominal: NDArray[np.float64] = np.arange(n, dtype=np.float64) * period
    tip = manual_endpoint_positions(config, q)
    intervals = np.diff(times)
    joints = [f"j{i + 1}" for i in range(config.dof)]
    log = StateLog(
        manual_build_skeleton(config, q[0]),
        producer="trajectory_recorder",
        channel_meta={
            "q": {"unit": "rad", "label": "joint angle", "columns": joints},
            "tip": {"unit": "m", "label": "tip position", "columns": ["x", "y"]},
            "nominal_time": {"unit": "s", "label": "nominal tick clock (tick index x period)"},
        },
        extra={
            "acquisition": {
                "clock": "wall-clock",
                "time_channel": "seconds since the take started, read when each tick's pose update begins; "
                "dialog pauses excluded",
                "nominal_time_channel": "tick index x tick_period_s",
                "mode": "ik",
                "tick_period_s": period,
                "sample_period_s": period,
                "pose_updates_per_sample": 1,
                "display_period_s": max(1, round(_DISPLAY_PERIOD_S / period)) * period,
                "ticks": n - 1,
                "wall_mean_tick_s": float(np.mean(intervals)),
                "wall_max_tick_s": float(np.max(intervals)),
                "late_ticks": int(np.count_nonzero(intervals > _LATE_TICK_FACTOR * period)),
                "late_tick_factor": _LATE_TICK_FACTOR,
            },
            "display": {
                "show_tip_trail": True,
                "show_past_trails": True,
                "past_trails_shown_during_take": True,
                "history_takes": list(history),
                "visible_source_takes": list(history),
                "visible_source_files": [f"reach_{h:03d}.sklog.npz" for h in history],
                "policy": {
                    "current_color": "#c81e78",
                    "current_alpha": 230,
                    "current_width_px": 2.0,
                    "past_color": "#465aa0",
                    "past_alpha": 55,
                    "past_width_px": 2.0,
                    "order": "saved trails behind the current trail",
                    "source": "logged tip samples (forward kinematics), not the cursor",
                },
            },
        },
    )
    for k in range(n):
        log.record(float(times[k]), q=q[k], tip=tip[k], nominal_time=np.asarray(nominal[k], dtype=np.float64))
    return log
