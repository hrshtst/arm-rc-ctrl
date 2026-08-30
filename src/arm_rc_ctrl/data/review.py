# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Speed-based prime/move/dwell proposal and a review plot for recorded demonstrations.

Human demonstrations carry no explicit intervals. :func:`propose_intervals`
derives them from the joint-speed profile (the first and last samples faster
than a threshold bound the movement); the proposal is written next to a plot
so the boundaries can be confirmed manually before the immutable record is
created (``docs/TASKS.md`` M1-013).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import matplotlib as mpl
import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.data.records import Intervals
from arm_rc_ctrl.data.samples import PHASE_CODES

__all__ = ["DEFAULT_MIN_HOLD_S", "DEFAULT_SPEED_THRESHOLD", "IntervalProposal", "plot_intervals", "propose_intervals"]

DEFAULT_SPEED_THRESHOLD: Final = 0.05
"""Joint speed (rad/s) above which a sample counts as moving."""
DEFAULT_MIN_HOLD_S: Final = 0.2
"""Shortest acceptable prime and dwell hold (s)."""


@dataclass(frozen=True)
class IntervalProposal:
    """Proposed intervals with the evidence they were derived from."""

    intervals: Intervals
    speed_threshold: float
    speed: NDArray[np.float64]
    """Per-sample maximum absolute joint speed (rad/s)."""


def propose_intervals(
    t: NDArray[np.float64],
    q: NDArray[np.float64],
    *,
    dq: NDArray[np.float64] | None = None,
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
    min_hold_s: float = DEFAULT_MIN_HOLD_S,
) -> IntervalProposal:
    """Bound the movement by the first and last samples whose joint speed exceeds the threshold."""
    times = np.asarray(t, dtype=np.float64)
    joints = np.asarray(q, dtype=np.float64)
    if times.ndim != 1 or times.shape[0] < 3 or joints.ndim != 2 or joints.shape[0] != times.shape[0]:  # noqa: PLR2004
        msg = f"t must be (N,) with N >= 3 and q (N, dof), got {times.shape} and {joints.shape}"
        raise ValueError(msg)
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(joints))) or not np.all(np.diff(times) > 0):
        msg = "t and q must be finite and t strictly increasing"
        raise ValueError(msg)
    if not (speed_threshold > 0 and np.isfinite(speed_threshold)) or not (min_hold_s >= 0 and np.isfinite(min_hold_s)):
        msg = "speed_threshold must be positive and min_hold_s non-negative"
        raise ValueError(msg)
    velocity = np.asarray(dq, dtype=np.float64) if dq is not None else np.gradient(joints, times, axis=0)
    if velocity.shape != joints.shape:
        msg = f"dq must have shape {joints.shape}, got {velocity.shape}"
        raise ValueError(msg)
    speed = cast("NDArray[np.float64]", np.max(np.abs(velocity), axis=1))
    moving = np.flatnonzero(speed > speed_threshold)
    if moving.size == 0:
        msg = f"no sample exceeds the speed threshold {speed_threshold} rad/s; nothing moves"
        raise ValueError(msg)
    first, last = int(moving[0]), int(moving[-1])
    if first == 0 or last >= times.shape[0] - 1:
        msg = "movement touches the start or the end of the recording; no hold interval can be proposed"
        raise ValueError(msg)
    move_start = float(times[first])
    move_end = float(times[last + 1])
    prime = (float(times[0]), move_start)
    dwell = (move_end, float(times[-1]))
    for name, (lo, hi) in (("prime", prime), ("dwell", dwell)):
        if hi - lo < min_hold_s:
            msg = f"proposed {name} hold {hi - lo:.3f} s is shorter than the minimum {min_hold_s} s"
            raise ValueError(msg)
    intervals = Intervals(prime=prime, move=(move_start, move_end), dwell=dwell)
    return IntervalProposal(intervals=intervals, speed_threshold=speed_threshold, speed=speed)


def plot_intervals(
    t: NDArray[np.float64],
    q: NDArray[np.float64],
    speed: NDArray[np.float64],
    intervals: Intervals,
    path: Path,
    *,
    speed_threshold: float | None = None,
    title: str = "",
) -> None:
    """Write a PNG showing joint angles and joint speed with the phase boundaries shaded."""
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_q, ax_v) = cast("Any", plt.subplots(2, 1, sharex=True, figsize=(9, 6)))
    for j in range(q.shape[1]):
        ax_q.plot(t, q[:, j], label=f"q{j + 1}")
    ax_v.plot(t, speed, color="black", label="max |dq|")
    if speed_threshold is not None:
        ax_v.axhline(speed_threshold, color="red", linestyle="--", label=f"threshold {speed_threshold} rad/s")
    shades = {"prime": "#dddddd", "move": "#cce5ff", "dwell": "#d5f5d5"}
    for name in PHASE_CODES:
        lo, hi = getattr(intervals, name)
        for ax in (ax_q, ax_v):
            ax.axvspan(lo, hi, color=shades[name], alpha=0.6, label=name if ax is ax_q else None)
    ax_q.set_ylabel("joint angle (rad)")
    ax_v.set_ylabel("joint speed (rad/s)")
    ax_v.set_xlabel("time (s)")
    ax_q.legend(loc="upper right", ncol=3)
    ax_v.legend(loc="upper right")
    if title:
        ax_q.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
