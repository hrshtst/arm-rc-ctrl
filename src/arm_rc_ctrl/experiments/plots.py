# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Curated plots of a run record (headless matplotlib)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from arm_rc_ctrl.experiments.run_record import LoadedRun

__all__ = ["plot_run"]


def plot_run(run: LoadedRun, target: Sequence[float] | None, out: Path, *, title: str | None = None) -> Path:
    """Write a three-panel PNG: joint tracking, endpoint path with the target, and applied torque."""
    arrays = run.arrays.arrays
    t = arrays["t"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
    ax = axes[0]
    for j in range(run.arrays.dof):
        ax.plot(t, arrays["q_desired"][:, j], "--", label=f"q_desired[{j}]")
        ax.plot(t, arrays["q"][:, j], label=f"q[{j}]")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("joint angle (rad)")
    ax.set_title(title or run.pointer.artifact.artifact_id)
    ax.legend(loc="best", fontsize="small")
    ax = axes[1]
    tip = arrays["tip"]
    ax.plot(tip[:, 0], tip[:, 1], label="endpoint")
    ax.plot(tip[0, 0], tip[0, 1], "o", label="start")
    if target is not None:
        ax.plot(target[0], target[1], "x", markersize=10, label="target")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize="small")
    ax = axes[2]
    tau = arrays.get("tau_applied", arrays["tau_requested"])
    for j in range(run.arrays.dof):
        ax.plot(t, tau[:, j], label=f"tau[{j}]")
    if "saturation" in arrays and np.any(arrays["saturation"]):
        saturated = t[arrays["saturation"] > 0]
        ax.plot(saturated, np.zeros_like(saturated), "|", color="red", label="saturated")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("torque (N*m)")
    ax.legend(loc="best", fontsize="small")
    out.parent.mkdir(parents=True, exist_ok=True)
    cast("Any", fig).savefig(out, dpi=100)  # matplotlib's kwargs are untyped
    plt.close(fig)
    return out
