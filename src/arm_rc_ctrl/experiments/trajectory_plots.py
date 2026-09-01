# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Human-facing joint trajectories for representative task 1-a robustness scenarios."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from arm_rc_ctrl.data.records import load_record
from arm_rc_ctrl.experiments.perturbations import CLASS_ORDER, PerturbationClass
from arm_rc_ctrl.experiments.robustness import ArmRun, RobustnessSuite, load_suite
from arm_rc_ctrl.experiments.run_record import RunPointerRecord, load_run
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import open_storage

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from arm_rc_ctrl.experiments.run_record import LoadedRun
    from arm_rc_ctrl.storage import StorageRoot

__all__ = [
    "ARM_PAIRS",
    "CURVE_ORDER",
    "CURVE_STYLES",
    "plot_joint_series",
    "select_representatives",
    "write_task_1a_trajectory_plots",
]

ARM_PAIRS: Final[Mapping[str, tuple[str, str]]] = {
    "pd": ("rc+pd_v2", "replay+pd_v2"),
    "computed_torque": ("rc+computed_torque", "replay+computed_torque"),
}
_CLASS_LABELS: Final[Mapping[PerturbationClass, str]] = {
    "nominal": "Nominal",
    "posture_small": "Small posture fluctuation",
    "posture_large": "Large posture fluctuation",
    "force": "Force",
    "combined": "Combined",
}
_PHASE_BOUNDARIES: Final = (1.0, 4.0)
_MINIMUM_SAMPLES: Final = 2
_TRAJECTORY_DIMENSIONS: Final = 2
CURVE_ORDER: Final = ("reference", "replay_actual", "rc_output", "rc_actual")
"""Legend and draw order for every joint-trajectory panel."""
CURVE_STYLES: Final[Mapping[str, tuple[str, str, float, str]]] = {
    "reference": ("black", "--", 2.2, "teacher/reference"),
    "replay_actual": ("tab:blue", "-", 1.8, "replay actual"),
    "rc_output": ("tab:green", "--", 1.8, "RC output"),
    "rc_actual": ("tab:orange", "-", 1.5, "RC actual"),
}


def _joint_rmse(run: ArmRun) -> float:
    metric = run.report.joint_rmse
    if metric is None:
        msg = f"run {run.run_id} has no joint RMSE"
        raise ValueError(msg)
    return metric.aggregate


def select_representatives(suite: RobustnessSuite, *, primary_arm: str = "rc+pd_v2") -> dict[PerturbationClass, ArmRun]:
    """Select the primary-arm run closest to each class's median joint RMSE.

    The same selected scenario is then shown for every arm. Ties are broken by
    scenario ID so selection is deterministic and cannot depend on suite order.
    """
    selected: dict[PerturbationClass, ArmRun] = {}
    for kind in CLASS_ORDER:
        candidates = [
            run
            for run in suite.runs
            if run.arm == primary_arm and run.kind == kind and run.report.joint_rmse is not None
        ]
        if not candidates:
            msg = f"suite has no {primary_arm!r} run with joint RMSE for class {kind!r}"
            raise ValueError(msg)
        median = statistics.median(_joint_rmse(run) for run in candidates)
        selected[kind] = min(
            candidates,
            key=lambda run: (abs(_joint_rmse(run) - median), run.scenario_id),
        )
    return selected


def _finite_trajectory(array: NDArray[np.float64], name: str, shape: tuple[int, int]) -> NDArray[np.float64]:
    values = np.asarray(array, dtype=np.float64)
    if values.shape != shape or not np.all(np.isfinite(values)):
        msg = f"{name} must be finite with shape {shape}, got {values.shape}"
        raise ValueError(msg)
    return values


def plot_joint_series(
    t: NDArray[np.float64],
    reference: NDArray[np.float64],
    replay_actual: NDArray[np.float64],
    rc_output: NDArray[np.float64],
    rc_actual: NDArray[np.float64],
    out: Path,
    *,
    title: str,
    force: bool = False,
) -> Path:
    """Plot reference, replay actual, RC output, and RC actual positions in the declared order."""
    time = np.asarray(t, dtype=np.float64)
    if time.ndim != 1 or time.size < _MINIMUM_SAMPLES or not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0):
        msg = "t must be a finite, strictly increasing 1-D array with at least two samples"
        raise ValueError(msg)
    reference_values = np.asarray(reference, dtype=np.float64)
    if (
        reference_values.ndim != _TRAJECTORY_DIMENSIONS
        or reference_values.shape[0] != time.size
        or reference_values.shape[1] < 1
    ):
        msg = f"reference must have shape (N, dof), got {reference_values.shape}"
        raise ValueError(msg)
    shape = (time.size, reference_values.shape[1])
    curves = {
        "reference": _finite_trajectory(reference_values, "reference", shape),
        "replay_actual": _finite_trajectory(replay_actual, "replay_actual", shape),
        "rc_output": _finite_trajectory(rc_output, "rc_output", shape),
        "rc_actual": _finite_trajectory(rc_actual, "rc_actual", shape),
    }
    if out.exists() and not force:
        msg = f"refusing to overwrite {out}"
        raise FileExistsError(msg)

    fig, axes = cast(
        "tuple[Any, Any]",
        plt.subplots(
            shape[1],
            1,
            figsize=(10, 2.8 * shape[1]),
            sharex=True,
            squeeze=False,
            constrained_layout=True,
        ),
    )
    panels = axes[:, 0]
    for joint, axis in enumerate(panels):
        for name in CURVE_ORDER:
            color, linestyle, linewidth, label = CURVE_STYLES[name]
            axis.plot(
                time,
                curves[name][:, joint],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label,
            )
        for boundary in _PHASE_BOUNDARIES:
            axis.axvline(boundary, color="0.65", linewidth=0.8, linestyle=":")
        axis.set_ylabel(f"q{joint + 1} (rad)")
        axis.grid(visible=True, alpha=0.3)
    panels[0].set_title(title)
    panels[0].legend(loc="best", fontsize="small", ncol=4)
    panels[-1].set_xlabel("time (s)   |   hold: 0-1   reach: 1-4   dwell: 4-5")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=out.stem, suffix=".tmp.png", dir=out.parent, delete=False) as handle:
        staged = Path(handle.name)
    try:
        fig.savefig(staged, dpi=120, format="png")
        staged.replace(out)
    finally:
        plt.close(fig)
        staged.unlink(missing_ok=True)
    return out


def _arm_run(suite: RobustnessSuite, scenario_id: str, arm: str) -> ArmRun:
    matches = [run for run in suite.runs if run.scenario_id == scenario_id and run.arm == arm]
    if len(matches) != 1:
        msg = f"expected one {arm!r} run for {scenario_id!r}, found {len(matches)}"
        raise ValueError(msg)
    return matches[0]


def _load_arm_run(run: ArmRun, store: StorageRoot, records_root: Path) -> LoadedRun:
    if run.pointer is None:
        msg = f"run {run.run_id} has no Git-tracked pointer"
        raise ValueError(msg)
    pointer_path = records_root / run.pointer
    pointer = load_record(pointer_path, RunPointerRecord)
    if pointer.artifact.artifact_id != run.run_id:
        msg = f"{pointer_path} points to {pointer.artifact.artifact_id}, expected {run.run_id}"
        raise ValueError(msg)
    return load_run(store, pointer)


def write_task_1a_trajectory_plots(
    suite: RobustnessSuite,
    output_dir: Path,
    *,
    store: StorageRoot,
    records_root: Path,
    force: bool = False,
) -> dict[str, object]:
    """Write PD and computed-torque trajectory figures for each representative class scenario."""
    representatives = select_representatives(suite)
    manifest: dict[str, object] = {"selection": "closest rc+pd_v2 joint RMSE to class median", "classes": {}}
    classes = cast("dict[str, object]", manifest["classes"])
    for kind in CLASS_ORDER:
        scenario_id = representatives[kind].scenario_id
        outputs: dict[str, str] = {}
        runs: dict[str, str] = {}
        for tracker, (rc_arm, replay_arm) in ARM_PAIRS.items():
            rc_ref = _arm_run(suite, scenario_id, rc_arm)
            replay_ref = _arm_run(suite, scenario_id, replay_arm)
            rc = _load_arm_run(rc_ref, store, records_root)
            replay = _load_arm_run(replay_ref, store, records_root)
            if not np.array_equal(rc.arrays.arrays["t"], replay.arrays.arrays["t"]):
                msg = f"{scenario_id}: RC and replay clocks differ for {tracker}"
                raise ValueError(msg)
            filename = f"trajectories_{kind}_{tracker}.png"
            plot_joint_series(
                replay.arrays.arrays["t"],
                replay.arrays.arrays["q_desired"],
                replay.arrays.arrays["q"],
                rc.arrays.arrays["q_desired"],
                rc.arrays.arrays["q"],
                output_dir / filename,
                title=f"{_CLASS_LABELS[kind]} — {tracker.replace('_', ' ')} — {scenario_id}",
                force=force,
            )
            outputs[tracker] = filename
            runs[rc_arm] = rc_ref.run_id
            runs[replay_arm] = replay_ref.run_id
        classes[kind] = {"scenario_id": scenario_id, "plots": outputs, "runs": runs}
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Generate all ten curated task 1-a joint-trajectory plots."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--suite", type=Path, required=True, help="locked robustness-suite JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, default=repository_root())
    parser.add_argument("--force", action="store_true", help="replace existing plot files")
    args = parser.parse_args(argv)
    manifest = write_task_1a_trajectory_plots(
        load_suite(args.suite),
        args.output_dir,
        store=open_storage(),
        records_root=args.records_root,
        force=args.force,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - use scripts/ entry point
    raise SystemExit(main())
