# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""The structured recovery report on the negative-result path (M3R-018; recovery plan sections 7.2 and 7.3).

Renders the committed machine-readable evidence — the study pointers, the
development ablation, the freeze record, and the curated representative
pairs — into one Markdown report with tables and plots for early gaps,
convergence, dwell, effort, smoothness, paired distributions, every failure
taxonomy, and the limitations. Trajectory figures follow the plan's fixed
curve order (reference, replay actual, dashed RC generated reference, RC
actual) on the run clock with the activation and dwell boundaries marked; the
curated animations are traceable to the same verified run IDs. No confirmatory
data exists: the report presents the accepted negative result and the closed
confirmatory gate.

Command line::

    python -m arm_rc_ctrl.experiments.recovery_report --docs docs/experiments/<task>
        --output <docs>/recovery_report_v1.md
        [--plots-dir <docs>/plots/recovery_report_v1] [--records-root ROOT]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from arm_rc_ctrl.data.records import load_record, verify_payload
from arm_rc_ctrl.data.recovery import RecoveryDatasetRecord, task_intervals_from_phases
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.experiments.augmentation_plots import AUGMENTATION_PLOT_FILES, DEFAULT_DISPLAYED_EPISODES
from arm_rc_ctrl.experiments.evidence import StoredReport, load_report_pointer
from arm_rc_ctrl.experiments.perturbations import (
    RobustnessScenario,
    load_development_robustness,
    robustness_scenarios,
)
from arm_rc_ctrl.experiments.recovery_ablation import LIMITATIONS, AblationReport, load_ablation
from arm_rc_ctrl.experiments.recovery_freeze import FreezeRecord, load_freeze
from arm_rc_ctrl.experiments.recovery_representative import (
    REPRESENTATIVE_CLASSES,
    PairOutcome,
    RepresentativeRecord,
    load_representatives,
)
from arm_rc_ctrl.experiments.run_record import LoadedRun, RunPointerRecord, load_run
from arm_rc_ctrl.metrics.effort import effort_metrics
from arm_rc_ctrl.metrics.joint import JointAnglePolicy, joint_rmse
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario
from arm_rc_ctrl.storage import open_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.storage import StorageRoot

__all__ = [
    "PLOT_FILES",
    "ReportInputs",
    "build_report_inputs",
    "main",
    "plot_recovery_pair",
    "render_recovery_report",
    "write_recovery_plots",
]

PLOT_FILES: Final = (
    "cell_gap_medians.png",
    "cell_jump_medians.png",
    "feasible_by_warmup.png",
    "trajectory_nominal.png",
    "trajectory_posture_small.png",
    "trajectory_posture_large.png",
    "trajectory_force.png",
)
_PRIMARY_TRACKER: Final = "pd_v2"
_DEVELOPMENT_PROTOCOL: Final = Path("configs/evaluations/task_1a_recovery_dev_v1.toml")
_MINIMUM_SAMPLES: Final = 2
_TRAJECTORY_DIMENSIONS: Final = 2
_ANIMATION_PAIR_SIZE: Final = 2
RECOVERY_CURVE_ORDER: Final = ("reference", "replay_actual", "generator_output_q", "rc_actual")
"""Draw order fixed by the recovery plan (section 7.3); the generated reference is dashed."""
RECOVERY_CURVE_STYLES: Final[dict[str, tuple[str, str, float, str]]] = {
    "reference": ("black", "--", 2.2, "teacher/reference"),
    "replay_actual": ("tab:blue", "-", 1.8, "replay actual"),
    "generator_output_q": ("tab:green", "--", 1.8, "RC generated reference"),
    "rc_actual": ("tab:orange", "-", 1.5, "RC actual"),
}
_CLASS_TITLES: Final = {
    "nominal": "Nominal start",
    "posture_small": "Small initial-posture perturbation",
    "posture_large": "Large initial-posture perturbation",
    "force": "External-force pulse",
}


@dataclass(frozen=True)
class ReportInputs:
    """Everything the report renders from, loaded and cross-bound."""

    pointers: dict[str, StoredReport]
    ablation: AblationReport
    freeze: FreezeRecord
    representative: RepresentativeRecord
    reference: SampleSet
    runs: dict[str, LoadedRun]
    """Loaded representative runs by run ID (empty when rendering without the store)."""
    effort: dict[str, float]
    """Applied-torque RMS (N m) per representative run ID over its active segment."""
    torque_peak: dict[str, float] = field(default_factory=dict)
    saturation: dict[str, float] = field(default_factory=dict)
    move_rmse: dict[str, float] = field(default_factory=dict)
    """Movement-window joint RMSE (rad) against the original demonstration, per representative run."""
    scenarios: dict[str, RobustnessScenario] = field(default_factory=dict)
    """Development scenarios keyed by their stable ID for human-facing descriptions."""


def build_report_inputs(docs: Path, *, store: StorageRoot, records_root: Path) -> ReportInputs:
    """Load the committed evidence and every representative run."""
    pointers = {f.name: load_report_pointer(f) for f in sorted(docs.glob("recovery_search_*_v1.toml"))}
    pointers["residual_search_1a_v1.toml"] = load_report_pointer(docs / "residual_search_1a_v1.toml")
    ablation = load_ablation(docs / "development_ablation_v2.json")
    freeze = load_freeze(docs / "model_freeze_v2.json")
    representative = load_representatives(docs / "recovery_representative_v1.json")
    dataset_file = records_root / "data" / "records" / "processed" / f"{representative.dataset}.toml"
    dataset = load_record(dataset_file, RecoveryDatasetRecord)
    reference = load_samples(verify_payload(store, dataset.artifact))
    scenario = load_scenario(records_root / dataset.scenario.config_path)
    torque_limits = scenario.limits.torque
    protocol = load_development_robustness(records_root / _DEVELOPMENT_PROTOCOL)
    scenarios = {case.scenario_id: case for case in robustness_scenarios(protocol, nominal=dataset.q0_ref)}
    missing_scenarios = set(representative.scenarios.values()) - scenarios.keys()
    if missing_scenarios:
        msg = f"representative scenarios are absent from {_DEVELOPMENT_PROTOCOL}: {sorted(missing_scenarios)}"
        raise ValueError(msg)
    runs: dict[str, LoadedRun] = {}
    effort: dict[str, float] = {}
    torque_peak: dict[str, float] = {}
    saturation: dict[str, float] = {}
    move_rmse: dict[str, float] = {}
    task = task_intervals_from_phases(reference.t, reference.phase)
    move_mask = (reference.t >= task.move[0]) & (reference.t < task.move[1])
    policy = JointAnglePolicy.limited(reference.dof)
    for pair in representative.pairs:
        for run_id in (pair.replay_run, pair.rc_run):
            pointer_file = records_root / "data" / "records" / "runs" / f"{run_id}.toml"
            pointer = load_record(pointer_file, RunPointerRecord)
            run = load_run(store, pointer)
            runs[run_id] = run
            arrays = run.arrays.arrays
            run_t = cast("NDArray[np.float64]", arrays["t"])
            source = "tau_applied" if "tau_applied" in arrays else "tau_requested"
            tau = cast("NDArray[np.float64]", arrays[source])
            window = (pair.activation_s, float(run_t[-1]))
            metrics = effort_metrics(run_t, tau, torque_limits, window=window)
            effort[run_id] = float(metrics.torque_rms)
            torque_peak[run_id] = float(metrics.torque_peak)
            saturation[run_id] = float(np.mean(cast("NDArray[np.int64]", arrays["saturation"])))
            active = run_t >= pair.activation_s - 1e-9
            q_active = cast("NDArray[np.float64]", arrays["q"])[active]
            n = min(q_active.shape[0], reference.n_samples)
            if bool(move_mask[:n].any()):
                aligned = joint_rmse(q_active[:n][move_mask[:n]], reference.q[:n][move_mask[:n]], policy)
                move_rmse[run_id] = float(aligned.aggregate)
    return ReportInputs(
        pointers=pointers,
        ablation=ablation,
        freeze=freeze,
        representative=representative,
        reference=reference,
        runs=runs,
        effort=effort,
        torque_peak=torque_peak,
        saturation=saturation,
        move_rmse=move_rmse,
        scenarios=scenarios,
    )


def plot_recovery_pair(
    t: NDArray[np.float64],
    reference: NDArray[np.float64],
    replay_actual: NDArray[np.float64],
    generator_output_q: NDArray[np.float64],
    rc_actual: NDArray[np.float64],
    out: Path,
    *,
    title: str,
    boundaries: Sequence[float],
    xlabel: str,
) -> Path:
    """Plot the plan's fixed curve order on the run clock; the RC output is NaN-masked during the hold."""
    time = np.asarray(t, dtype=np.float64)
    if time.ndim != 1 or time.size < _MINIMUM_SAMPLES or not np.all(np.isfinite(time)):
        msg = "t must be a finite 1-D array with at least two samples"
        raise ValueError(msg)
    curves: dict[str, NDArray[np.float64]] = {}
    for name, values in (
        ("reference", reference),
        ("replay_actual", replay_actual),
        ("generator_output_q", generator_output_q),
        ("rc_actual", rc_actual),
    ):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != _TRAJECTORY_DIMENSIONS or array.shape[0] != time.size or array.shape[1] < 1:
            msg = f"{name} must have shape (N, dof), got {array.shape}"
            raise ValueError(msg)
        if name != "generator_output_q" and not np.all(np.isfinite(array)):
            msg = f"{name} must be finite"
            raise ValueError(msg)
        curves[name] = array
    if out.exists():
        msg = f"refusing to overwrite {out}"
        raise FileExistsError(msg)
    dof = curves["reference"].shape[1]
    fig, axes = cast(
        "tuple[Any, Any]",
        plt.subplots(dof, 1, figsize=(10, 2.8 * dof), sharex=True, squeeze=False, constrained_layout=True),
    )
    panels = axes[:, 0]
    for joint, axis in enumerate(panels):
        for name in RECOVERY_CURVE_ORDER:
            color, linestyle, linewidth, label = RECOVERY_CURVE_STYLES[name]
            axis.plot(time, curves[name][:, joint], color=color, linestyle=linestyle, linewidth=linewidth, label=label)
        for boundary in boundaries:
            axis.axvline(boundary, color="0.65", linewidth=0.8, linestyle=":")
        axis.set_ylabel(f"q{joint + 1} (rad)")
        axis.grid(visible=True, alpha=0.3)
    panels[0].set_title(title)
    panels[0].legend(loc="best", fontsize="small", ncol=4)
    panels[-1].set_xlabel(xlabel)
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


def _reference_on_run_clock(inputs: ReportInputs, run: LoadedRun, activation_s: float) -> NDArray[np.float64]:
    """The task reference lifted onto the run clock: the held start before activation, the task after."""
    arrays = run.arrays.arrays
    run_t = cast("NDArray[np.float64]", arrays["t"])
    q = cast("NDArray[np.float64]", arrays["q"])
    reference = np.empty_like(q)
    hold = run_t < activation_s - 1e-9
    reference[hold] = q[0]
    active = ~hold
    n_active = int(np.count_nonzero(active))
    reference[active] = inputs.reference.q[:n_active]
    return reference


def write_recovery_plots(inputs: ReportInputs, directory: Path) -> list[str]:
    """Write the report's figures; returns the written file names in :data:`PLOT_FILES` order."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    candidates = inputs.ablation.candidates
    cell_names = sorted({name for candidate in candidates for name in candidate.cells})

    fig, axes = cast("tuple[Any, Any]", plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True))
    for axis, cell in zip(axes.ravel(), cell_names, strict=True):
        values = [candidate.cells[cell].gap_median for candidate in candidates]
        axis.hist(values, bins=30, color="tab:blue", alpha=0.8)
        axis.axvline(1.0, color="k", linewidth=0.8)
        axis.set_title(cell, fontsize="small")
        axis.grid(visible=True, alpha=0.3)
    fig.suptitle(f"Early command-gap ratio medians per cell ({len(candidates)} feasible trials; < 1 improves)")
    out = directory / PLOT_FILES[0]
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    written.append(out.name)

    fig, axes = cast("tuple[Any, Any]", plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True))
    for axis, cell in zip(axes.ravel(), cell_names, strict=True):
        values = [candidate.cells[cell].jump_median for candidate in candidates]
        axis.hist(values, bins=30, color="tab:orange", alpha=0.8)
        axis.axvline(1.0, color="k", linewidth=0.8)
        axis.set_title(cell, fontsize="small")
        axis.grid(visible=True, alpha=0.3)
    fig.suptitle("Activation-jump ratio medians per cell (re-derived replay baselines; < 1 improves)")
    out = directory / PLOT_FILES[1]
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    written.append(out.name)

    timing = next(arm for arm in inputs.ablation.arms if arm.formulation == "no_augmentation")
    fig, axis = cast("tuple[Any, Any]", plt.subplots(figsize=(7, 3.6), constrained_layout=True))
    warmups = sorted(timing.feasible_by_warmup, key=float)
    axis.bar(warmups, [timing.feasible_by_warmup[w] for w in warmups], color="tab:green", alpha=0.85)
    axis.set_xlabel("warm-up duration (s)")
    axis.set_ylabel("feasible trials")
    axis.set_title(f"Timing-only arm: feasible trials by warm-up ({timing.n_feasible} of {timing.trials_stored})")
    axis.grid(visible=True, axis="y", alpha=0.3)
    out = directory / PLOT_FILES[2]
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    written.append(out.name)

    task = task_intervals_from_phases(inputs.reference.t, inputs.reference.phase)
    by_pair = {(pair.kind, pair.tracker): pair for pair in inputs.representative.pairs}
    for index, kind in enumerate(REPRESENTATIVE_CLASSES):
        pair = by_pair[(kind, _PRIMARY_TRACKER)]
        rc = inputs.runs[pair.rc_run]
        replay = inputs.runs[pair.replay_run]
        arrays = rc.arrays.arrays
        run_t = cast("NDArray[np.float64]", arrays["t"])
        boundaries = (pair.activation_s, pair.activation_s + task.dwell[0])
        out = plot_recovery_pair(
            run_t,
            _reference_on_run_clock(inputs, rc, pair.activation_s),
            cast("NDArray[np.float64]", replay.arrays.arrays["q"]),
            cast("NDArray[np.float64]", arrays["generator_output_q"]),
            cast("NDArray[np.float64]", arrays["q"]),
            directory / PLOT_FILES[3 + index],
            title=f"{kind}: {pair.scenario_id} (trial {inputs.representative.trial}, {_PRIMARY_TRACKER})",
            boundaries=boundaries,
            xlabel=(
                f"run time (s)   |   hold: 0-{pair.activation_s:g}   "
                f"move: {pair.activation_s:g}-{pair.activation_s + task.dwell[0]:g}   dwell to end"
            ),
        )
        written.append(out.name)
    return written


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}g}"


def _pair_rows(inputs: ReportInputs) -> list[str]:
    scope = (
        "These tables cover the curated representative pairs only; the distribution plots and the "
        "eligibility evaluation cover all feasible development trials."
    )
    lines = [
        scope,
        "",
        "### Paired early metrics and target dwell",
        "",
        (
            "| class | scenario | tracker | RC run | replay run | jump (rad) | early gap (rad s) "
            "| settling (s) | dwell frac | desired vmax (rad/s) |"
        ),
        "| " + " | ".join(["---"] * 10) + " |",
    ]
    for pair in inputs.representative.pairs:
        recovery = pair.recovery
        head = f"| {pair.kind} | {pair.scenario_id} | {pair.tracker} | {pair.rc_run} | {pair.replay_run} "
        if recovery is None:
            lines.append(head + "| n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            head + f"| {_fmt(recovery.activation_jump_rad)} | {_fmt(recovery.command_gap_early.integral)} "
            f"| {_fmt(recovery.reference_settling.settling_time_s)} "
            f"| {_fmt(recovery.generated_dwell.in_tolerance_fraction, 3)} "
            f"| {_fmt(recovery.generated_dwell.velocity_max, 3)} |"
        )
    lines += [
        "",
        "### Original-trajectory RMSE, restoring alignment, and contraction",
        "",
        (
            "| class | tracker | RC move RMSE (rad) | replay move RMSE (rad) | mean cosine "
            "| positive frac | ref deviation early (rad s) | contraction rate (1/s) |"
        ),
        "| " + " | ".join(["---"] * 8) + " |",
    ]
    for pair in inputs.representative.pairs:
        recovery = pair.recovery
        rc_rmse = inputs.move_rmse.get(pair.rc_run)
        replay_rmse = inputs.move_rmse.get(pair.replay_run)
        if recovery is None:
            lines.append(
                f"| {pair.kind} | {pair.tracker} | {_fmt(rc_rmse)} | {_fmt(replay_rmse)} | n/a | n/a | n/a | n/a |"
            )
            continue
        decay = recovery.reference_settling.decay
        lines.append(
            f"| {pair.kind} | {pair.tracker} | {_fmt(rc_rmse)} | {_fmt(replay_rmse)} "
            f"| {_fmt(recovery.alignment.mean_cosine, 3)} | {_fmt(recovery.alignment.positive_fraction, 3)} "
            f"| {_fmt(recovery.reference_deviation_early.integral)} "
            f"| {_fmt(None if decay is None else decay.rate_per_s, 3)} |"
        )
    lines += [
        "",
        "### Smoothness and effort",
        "",
        (
            "| class | tracker | des accel RMS | act accel RMS | des jerk RMS | act jerk RMS "
            "| torque RMS (N m) | torque peak (N m) | saturation |"
        ),
        "| " + " | ".join(["---"] * 9) + " |",
    ]
    for pair in inputs.representative.pairs:
        recovery = pair.recovery
        peak = inputs.torque_peak.get(pair.rc_run)
        saturated = inputs.saturation.get(pair.rc_run)
        torque = inputs.effort.get(pair.rc_run)
        if recovery is None:
            lines.append(
                f"| {pair.kind} | {pair.tracker} | n/a | n/a | n/a | n/a "
                f"| {_fmt(torque)} | {_fmt(peak)} | {_fmt(saturated, 3)} |"
            )
            continue
        lines.append(
            f"| {pair.kind} | {pair.tracker} | {_fmt(recovery.smoothness_desired.accel_rms, 3)} "
            f"| {_fmt(recovery.smoothness_actual.accel_rms, 3)} | {_fmt(recovery.smoothness_desired.jerk_rms, 3)} "
            f"| {_fmt(recovery.smoothness_actual.jerk_rms, 3)} | {_fmt(torque)} | {_fmt(peak)} "
            f"| {_fmt(saturated, 3)} |"
        )
    return lines


def _offset_text(offset: tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{value:+.4f}" for value in offset) + "]"


def _scenario_description(case: RobustnessScenario | None, pair: PairOutcome) -> str:
    """Describe exactly what differs from the nominal representative run."""
    if case is None:
        return f"Scenario `{pair.scenario_id}`; consult the representative table for its recorded identity."
    if case.kind == "nominal":
        return "The arm starts at the cropped demonstration posture; no posture offset or external force is applied."
    if case.kind in ("posture_small", "posture_large"):
        assert case.magnitude_rad is not None
        return (
            f"The initial joints are offset from the cropped demonstration posture by "
            f"$\\Delta q={_offset_text(case.offset)}$ rad (norm {case.magnitude_rad:g} rad). Both arms hold this "
            "perturbed posture; neither corrects it before activation. No external force is applied."
        )
    if case.kind == "force":
        assert case.force_magnitude_n is not None
        assert case.force_start_s is not None
        assert case.force_duration_s is not None
        assert case.direction_deg is not None
        run_start = pair.activation_s + case.force_start_s
        run_end = run_start + case.force_duration_s
        direction = "+x" if case.direction_deg == 0.0 else f"{case.direction_deg:g} degrees"
        return (
            "The arm starts nominally. A "
            f"{case.force_magnitude_n:g} N end-effector pulse acts toward {direction} from task time "
            f"{case.force_start_s:g} to {case.force_start_s + case.force_duration_s:g} s "
            f"(run time {run_start:g} to {run_end:g} s); the red arrow shows the applied force."
        )
    return f"Scenario `{case.scenario_id}` belongs to class `{case.kind}`."


def _animation_observation(inputs: ReportInputs, pair: PairOutcome) -> str:
    """Summarize the numerical evidence needed to interpret one visual pair."""
    recovery = pair.recovery
    if recovery is None:
        return "The RC run did not complete, so no recovery metrics are available."
    rc_rmse = inputs.move_rmse.get(pair.rc_run)
    replay_rmse = inputs.move_rmse.get(pair.replay_run)
    return (
        f"Both simulations completed. RC/replay actual-motion RMSE against the original movement is "
        f"{_fmt(rc_rmse)} / {_fmt(replay_rmse)} rad. The generated reference enters the 0.05 rad band around "
        f"the original after {_fmt(recovery.reference_settling.settling_time_s)} s and spends "
        f"{100 * recovery.generated_dwell.in_tolerance_fraction:.0f}% of dwell within the 1 cm target region."
    )


def _animation_lines(inputs: ReportInputs, animations: Sequence[str]) -> list[str]:
    """Render paired, evidence-linked descriptions for the curated PD animations."""
    available = set(animations)
    pairs = {(pair.kind, pair.tracker): pair for pair in inputs.representative.pairs}
    lines = [
        "## Animations",
        "",
        (
            "These GIFs are kinematic playbacks of the recorded joint positions: the moving links show the "
            "actual simulated robot motion, not the desired trajectory and not a controller re-execution. "
            "The target marker and its 1 cm tolerance ring are task overlays; the time-series plots above "
            "show the commands that cannot be seen in the robot view."
        ),
        "",
        (
            "Each clip is traceable to the run ID in its caption and can be regenerated with "
            "`scripts/play_run.py --run <run-id> --scenario configs/tasks/task_1a.toml --export <file.gif>`."
        ),
        "",
        (
            "Every pair uses timing-only trial 17 and the same frozen PD v2 tracker. Both arms hold their own "
            "initial posture for 0.25 s, then activate together at task time 0. The left clip is driven by the "
            "feedback-conditioned ESN reference; the right clip directly replays the original teacher "
            "trajectory. The side-by-side comparison therefore isolates the reference generator. Computed-"
            "torque runs are quantified in the tables but are not included in this animation set."
        ),
        "",
    ]
    consumed: set[str] = set()
    for kind in REPRESENTATIVE_CLASSES:
        rc_name = f"{kind}_rc_pd.gif"
        replay_name = f"{kind}_replay_pd.gif"
        if rc_name not in available and replay_name not in available:
            continue
        pair = pairs.get((kind, _PRIMARY_TRACKER))
        if pair is None:
            continue
        case = inputs.scenarios.get(pair.scenario_id)
        lines += [
            f"### {_CLASS_TITLES[kind]} — `{pair.scenario_id}`",
            "",
            f"**Setup.** {_scenario_description(case, pair)}",
            "",
            f"**What the result shows.** {_animation_observation(inputs, pair)}",
            "",
        ]
        names = [name for name in (rc_name, replay_name) if name in available]
        if len(names) == _ANIMATION_PAIR_SIZE:
            lines += [
                "| RC-generated reference + PD v2 | Original-reference replay + PD v2 |",
                "| --- | --- |",
                (
                    f"| Actual motion from `{pair.rc_run}`. The unseen desired command is the ESN's "
                    f"feedback-conditioned output. | Actual motion from `{pair.replay_run}`. The unseen "
                    "desired command is the original teacher trajectory. |"
                ),
                (
                    f"| ![{Path(rc_name).stem}](animations/{rc_name}) | "
                    f"![{Path(replay_name).stem}](animations/{replay_name}) |"
                ),
                "",
            ]
        else:
            name = names[0]
            run_id = pair.rc_run if name == rc_name else pair.replay_run
            source = "feedback-conditioned ESN" if name == rc_name else "original teacher trajectory"
            lines += [
                f"`{run_id}` — actual motion commanded by the {source}:",
                "",
                f"![{Path(name).stem}](animations/{name})",
                "",
            ]
        consumed.update(names)
    for name in animations:
        if name not in consumed:
            lines += [f"Additional animation `{name}`:", "", f"![{Path(name).stem}](animations/{name})", ""]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def render_recovery_report(
    inputs: ReportInputs,
    *,
    plots: Sequence[str] = (),
    animations: Sequence[str] = (),
    augmentation_plots: Sequence[str] = AUGMENTATION_PLOT_FILES,
) -> str:
    """The Markdown report of the accepted negative result."""
    ablation = inputs.ablation
    freeze = inputs.freeze
    representative = inputs.representative
    dirty = " (dirty)" if freeze.provenance.project_dirty else ""
    timing = next(arm for arm in ablation.arms if arm.formulation == "no_augmentation")
    lines = [
        "# Task 1-a state-conditioned recovery: development results (v1)",
        "",
        "## Summary",
        "",
        f"- Dataset `{ablation.dataset}`; freeze commit `{freeze.provenance.project_commit[:12]}`{dirty}.",
        (
            f"- **Accepted negative result.** {freeze.n_candidates} feasible development trials, "
            f"{freeze.n_eligible} eligible under the section 7.3 rule; no model is frozen and the "
            "confirmatory suite is not authorized under protocol v1 (`model_freeze_v2`)."
        ),
        (
            f"- Timing-only arm: {timing.n_feasible} of {timing.trials_stored} trials feasible, best "
            f"worst-cell early command-gap ratio {_fmt(timing.best_value)} (trial "
            f"{timing.best_number}); no feasible model was found among the sampled trials of the "
            "augmented arms, and the residual arm is an exploratory negative (D4)."
        ),
        (
            "- Representative pairs below come from the best feasible timing-only trial - a "
            "development-representative point, never a selected model."
        ),
        "",
        "## Study outcomes",
        "",
        "| study | formulation | budget | stored | feasible | best worst-cell gap ratio |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in sorted(inputs.pointers):
        pointer = inputs.pointers[name]
        lines.append(
            f"| {pointer.study} | {pointer.formulation} | {pointer.budget} | {pointer.trials_stored} "
            f"| {pointer.n_feasible} | {_fmt(pointer.best_value)} |"
        )
    lines += [
        "",
        "Full per-trial reports live in the external store behind the committed content-addressed",
        "pointers; the development ablation (`development_ablation_v2`) carries the failure",
        "taxonomies, sampled-coverage figures, and the eligibility evaluation this report renders.",
        "",
        "## Training augmentation in task space",
        "",
        (
            "The colored curves below are accepted synthetic joint trajectories mapped into end-effector "
            "x-y space with the robot's forward kinematics; the black curve is the one original scripted "
            "demonstration. They are generated by the same seeded AR(1) implementation used for training, "
            "not by a separate illustration routine."
        ),
        "",
        (
            f"For readability, each panel shows {DEFAULT_DISPLAYED_EPISODES} of the anchor's 64 synthetic "
            "episodes from seed bank 1. The matched family comparison uses sigma = 0.05 rad, phi = 0.99, "
            "and gamma = 1. The non-decaying family retains the perturbation during movement until the shared "
            "terminal taper; the contractive family additionally scales it by normalized distance to the target. "
            "Both become exactly equal to the original trajectory before dwell."
        ),
        "",
        (
            "The scale figure varies sigma over 0.01, 0.025, 0.05, and 0.10 rad for the non-decaying family. "
            "The contraction figure varies gamma over 0.5, 1, and 2 at sigma = 0.05 rad; larger gamma narrows "
            "the synthetic tube more rapidly as the reference approaches the target. Axes are shared within "
            "each comparison figure so the apparent spread is directly comparable. These plots explain the "
            "training inputs only—the study still found no feasible augmented model among its sampled trials."
        ),
        "",
        "Regenerate these figures with:",
        "",
        "```bash",
        "uv run python scripts/plot_recovery_augmentation.py \\",
        "  --output-dir docs/experiments/task_1a_state_conditioned_recovery/plots/augmentation_strategy_v1 \\",
        "  --force",
        "```",
        "",
        *(f"![{Path(p).stem}](plots/augmentation_strategy_v1/{p})" for p in augmentation_plots),
        "",
        "## Paired distributions (early gaps and activation jumps)",
        "",
        (
            f"Per class-by-tracker cell over all {freeze.n_candidates} feasible trials: the early "
            "command-gap ratio medians and the activation-jump ratio medians (both against paired "
            "replay baselines; values below 1 improve on replay). Small-posture cells miss the "
            "15-of-20 consistency requirement, which is what blocks eligibility."
        ),
        "",
        "## Representative pairs (convergence, dwell, effort, smoothness)",
        "",
        f"Selection: {representative.selection_rule}",
        "",
        *_pair_rows(inputs),
        "",
        "## Failures",
        "",
        "First failing gate of every infeasible trial, per study:",
        "",
    ]
    for arm in ablation.arms:
        lines.append(f"- `{arm.study}`: anchor {arm.anchor_reason or 'feasible'}.")
        lines.extend(f"    - {reason}: {count}" for reason, count in sorted(arm.reasons.items(), key=lambda kv: -kv[1]))
    residual = inputs.pointers.get("residual_search_1a_v1.toml")
    if residual is not None:
        lines.append(
            f"- `{residual.study}` (exploratory, D4): {residual.n_feasible} of {residual.trials_stored} feasible; "
            "the dominant first failure is the joint-velocity limit (see its study report)."
        )
    lines += ["", "## Limitations", ""]
    lines.extend(f"- {text}" for text in LIMITATIONS)
    if plots:
        lines += ["", "## Plots", ""]
        lines.extend(f"![{Path(p).stem}](plots/recovery_report_v1/{p})" for p in plots)
    if animations:
        lines += ["", *_animation_lines(inputs, animations)]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Render the recovery development report from committed evidence.")
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="report Markdown to write (must not exist)")
    parser.add_argument("--plots-dir", type=Path, default=None, help="defaults to <docs>/plots/recovery_report_v1")
    parser.add_argument("--records-root", type=Path, default=None)
    parser.add_argument(
        "--animations", nargs="*", default=(), help="animation file names under <docs>/animations to embed"
    )
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        msg = f"refusing to overwrite {args.output}"
        raise FileExistsError(msg)
    docs = Path(args.docs)
    records_root = repository_root() if args.records_root is None else Path(args.records_root)
    inputs = build_report_inputs(docs, store=open_storage(), records_root=records_root)
    plots_dir = docs / "plots" / "recovery_report_v1" if args.plots_dir is None else Path(args.plots_dir)
    plots = write_recovery_plots(inputs, plots_dir)
    Path(args.output).write_text(
        render_recovery_report(inputs, plots=plots, animations=tuple(args.animations)), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "plots": plots}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
