# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Human-facing task-space views of the recovery training augmentation (TOOL-002).

The plots call :func:`arm_rc_ctrl.rc.augment.generate_augmentation` directly,
then map its accepted joint trajectories through the scenario's forward
kinematics. They therefore visualize the training data itself, not a separate
illustrative noise process.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from arm_rc_ctrl.data.derivatives import DerivativeConfig
from arm_rc_ctrl.data.records import load_record, verify_payload
from arm_rc_ctrl.data.recovery import RecoveryDatasetRecord
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.experiments.augmentation_validation import ANCHOR
from arm_rc_ctrl.rc.augment import (
    APPROVED_GAMMA,
    APPROVED_SIGMA_RAD,
    SEED_NAMESPACE,
    AugmentationResult,
    EpisodeArrays,
    Family,
    generate_augmentation,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import endpoint_positions, load_scenario
from arm_rc_ctrl.storage import open_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from arm_rc_ctrl.data.recovery import TaskIntervals
    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.scenario import ScenarioConfig

__all__ = [
    "AUGMENTATION_PLOT_FILES",
    "DEFAULT_DISPLAYED_EPISODES",
    "main",
    "task_space_trajectories",
    "write_augmentation_task_space_plots",
]

AUGMENTATION_PLOT_FILES: Final = (
    "augmentation_task_space_families.png",
    "augmentation_task_space_sigma.png",
    "augmentation_task_space_gamma.png",
)
DEFAULT_DISPLAYED_EPISODES: Final = 16
_DEFAULT_DATASET: Final = "processed-20260903-ce343c8ce6a5"
_MAX_DISPLAYED_EPISODES: Final = ANCHOR.n_synthetic
_PLANE: Final = 2


def _derivatives(label: str) -> DerivativeConfig:
    if label == "central-difference":
        return DerivativeConfig(method="central")
    if label == "cubic-spline":
        return DerivativeConfig(method="spline")
    msg = f"unsupported derivative policy {label!r}"
    raise ValueError(msg)


def task_space_trajectories(
    result: AugmentationResult,
    scenario: ScenarioConfig,
    *,
    family: Family,
    count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return reference and accepted augmented endpoint paths used by a plot."""
    if family not in ("non_decaying", "contractive"):
        msg = f"family must be 'non_decaying' or 'contractive', got {family!r}"
        raise ValueError(msg)
    if count < 1:
        msg = f"count must be positive, got {count}"
        raise ValueError(msg)
    if count > len(result.episodes):
        msg = f"count {count} exceeds the generated {len(result.episodes)} episodes"
        raise ValueError(msg)
    reference = endpoint_positions(scenario, result.original.q)
    augmented = np.stack(
        [
            endpoint_positions(scenario, cast("EpisodeArrays", getattr(episode, family)).q)
            for episode in result.episodes[:count]
        ],
        axis=0,
    )
    return np.ascontiguousarray(reference), np.ascontiguousarray(augmented)


def _decorate_task_axis(axis: Axes, scenario: ScenarioConfig, title: str) -> None:
    ax = cast("Any", axis)  # matplotlib's keyword arguments are incompletely typed
    ax.plot(*scenario.task.target, marker="x", color="tab:red", markersize=9, markeredgewidth=2, label="target")
    ax.set_title(title, fontsize="medium")
    ax.set_xlabel("end-effector x (m)")
    ax.set_ylabel("end-effector y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(visible=True, alpha=0.3)


def _draw_paths(
    axis: Axes,
    reference: NDArray[np.float64],
    augmented: NDArray[np.float64],
    scenario: ScenarioConfig,
    *,
    title: str,
    colors: NDArray[np.float64] | None = None,
) -> None:
    ax = cast("Any", axis)  # matplotlib's keyword arguments are incompletely typed
    for index, path in enumerate(augmented):
        color: Any = "tab:blue" if colors is None else colors[index]
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=0.8, alpha=0.42)
    ax.plot(reference[:, 0], reference[:, 1], color="black", linewidth=2.4, label="original demonstration")
    ax.plot(reference[0, 0], reference[0, 1], marker="o", color="black", markersize=5, label="start")
    _decorate_task_axis(axis, scenario, title)


def _save_figure(fig: Figure, target: Path, *, force: bool) -> None:
    if target.is_symlink():
        msg = f"refusing to overwrite symbolic link {target}"
        raise FileExistsError(msg)
    if target.exists() and not force:
        msg = f"refusing to overwrite {target}"
        raise FileExistsError(msg)
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.stem}-", suffix=".png", dir=target.parent, delete=False
    ) as handle:
        staged = Path(handle.name)
    try:
        cast("Any", fig).savefig(staged, dpi=150, format="png", bbox_inches="tight")
        if force:
            staged.replace(target)
        else:
            try:
                os.link(staged, target)
            except FileExistsError as exc:
                msg = f"refusing to overwrite {target}"
                raise FileExistsError(msg) from exc
    finally:
        plt.close(fig)
        staged.unlink(missing_ok=True)


def _result(
    samples: SampleSet,
    task: TaskIntervals,
    scenario: ScenarioConfig,
    derivatives: DerivativeConfig,
    *,
    seed_bank: int,
    sigma_rad: float = ANCHOR.sigma_rad,
    gamma: float = ANCHOR.gamma,
) -> AugmentationResult:
    config = dataclasses.replace(ANCHOR, seed_bank=seed_bank, sigma_rad=sigma_rad, gamma=gamma)
    return generate_augmentation(samples.t, samples.q, task, scenario, config, derivatives=derivatives)


def write_augmentation_task_space_plots(
    samples: SampleSet,
    task: TaskIntervals,
    scenario: ScenarioConfig,
    derivatives: DerivativeConfig,
    output_dir: Path,
    *,
    displayed_episodes: int = DEFAULT_DISPLAYED_EPISODES,
    seed_bank: int = ANCHOR.seed_bank,
    force: bool = False,
) -> tuple[Path, ...]:
    """Write the family, noise-scale, and contraction task-space comparisons."""
    if not 1 <= displayed_episodes <= _MAX_DISPLAYED_EPISODES:
        msg = f"displayed_episodes must be in [1, {_MAX_DISPLAYED_EPISODES}], got {displayed_episodes}"
        raise ValueError(msg)
    if seed_bank < 0:
        msg = f"seed_bank must be non-negative, got {seed_bank}"
        raise ValueError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = tuple(output_dir / name for name in AUGMENTATION_PLOT_FILES)
    for target in targets:
        if target.is_symlink() or (target.exists() and not force):
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)

    anchor = _result(samples, task, scenario, derivatives, seed_bank=seed_bank)
    colors = cast(
        "NDArray[np.float64]",
        plt.get_cmap("viridis")(np.linspace(0.05, 0.9, displayed_episodes)),
    )
    fig, axes = cast(
        "tuple[Any, Any]",
        plt.subplots(1, 2, figsize=(11, 4.7), sharex=True, sharey=True, constrained_layout=True),
    )
    for axis, family, title in zip(
        axes,
        ("non_decaying", "contractive"),
        ("Non-decaying perturbation", "Target-distance contractive perturbation"),
        strict=True,
    ):
        reference, augmented = task_space_trajectories(
            anchor,
            scenario,
            family=family,
            count=displayed_episodes,
        )
        _draw_paths(axis, reference, augmented, scenario, title=title, colors=colors)
    axes[0].legend(loc="best", fontsize="small")
    fig.suptitle(
        f"Matched training trajectories: {displayed_episodes} of {ANCHOR.n_synthetic} episodes "
        f"(sigma={ANCHOR.sigma_rad:g} rad, phi={ANCHOR.phi:g}, gamma={ANCHOR.gamma:g})"
    )
    _save_figure(fig, targets[0], force=force)

    sigma_values = sorted(APPROVED_SIGMA_RAD)
    fig, axes = cast(
        "tuple[Any, Any]",
        plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True, constrained_layout=True),
    )
    for axis, sigma in zip(axes.ravel(), sigma_values, strict=True):
        result = _result(samples, task, scenario, derivatives, seed_bank=seed_bank, sigma_rad=sigma)
        reference, augmented = task_space_trajectories(
            result, scenario, family="non_decaying", count=displayed_episodes
        )
        _draw_paths(axis, reference, augmented, scenario, title=f"sigma = {sigma:g} rad")
    axes.ravel()[0].legend(loc="best", fontsize="small")
    fig.suptitle(
        f"Non-decaying augmentation across approved noise scales "
        f"({displayed_episodes} trajectories; phi={ANCHOR.phi:g})"
    )
    _save_figure(fig, targets[1], force=force)

    gamma_values = sorted(APPROVED_GAMMA)
    fig, axes = cast(
        "tuple[Any, Any]",
        plt.subplots(1, 3, figsize=(14, 4.3), sharex=True, sharey=True, constrained_layout=True),
    )
    for axis, gamma in zip(axes, gamma_values, strict=True):
        result = _result(samples, task, scenario, derivatives, seed_bank=seed_bank, gamma=gamma)
        reference, augmented = task_space_trajectories(result, scenario, family="contractive", count=displayed_episodes)
        _draw_paths(axis, reference, augmented, scenario, title=f"gamma = {gamma:g}")
    axes[0].legend(loc="best", fontsize="small")
    fig.suptitle(
        f"Contractive augmentation across approved envelope exponents "
        f"({displayed_episodes} trajectories; sigma={ANCHOR.sigma_rad:g} rad, phi={ANCHOR.phi:g})"
    )
    _save_figure(fig, targets[2], force=force)
    return targets


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    root = repository_root()
    parser = argparse.ArgumentParser(description="Plot recovery augmentation trajectories in end-effector x-y space.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "data" / "records" / "processed" / f"{_DEFAULT_DATASET}.toml",
        help="recovery processed-record TOML (defaults to the v1 recovery dataset)",
    )
    parser.add_argument("--scenario", type=Path, default=None, help="scenario TOML (defaults to the dataset record)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--displayed-episodes", type=int, default=DEFAULT_DISPLAYED_EPISODES)
    parser.add_argument("--seed-bank", type=int, default=ANCHOR.seed_bank)
    parser.add_argument("--force", action="store_true", help="replace existing regular output files")
    args = parser.parse_args(argv)

    record = load_record(Path(args.dataset), RecoveryDatasetRecord)
    scenario_file = root / record.scenario.config_path if args.scenario is None else Path(args.scenario)
    scenario = load_scenario(scenario_file)
    record.check_scenario(scenario_file)
    samples = load_samples(verify_payload(open_storage(), record.artifact))
    record.check_samples(samples)
    written = write_augmentation_task_space_plots(
        samples,
        record.crop.task,
        scenario,
        _derivatives(record.preprocessing.derivative_method),
        Path(args.output_dir),
        displayed_episodes=args.displayed_episodes,
        seed_bank=args.seed_bank,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "dataset": record.artifact.artifact_id,
                "payload_sha256": record.artifact.payload.sha256,
                "scenario": scenario.name,
                "seed_namespace": SEED_NAMESPACE,
                "seed_bank": args.seed_bank,
                "displayed_episodes": args.displayed_episodes,
                "files": [str(path) for path in written],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the thin script and main tests
    sys.exit(main())
