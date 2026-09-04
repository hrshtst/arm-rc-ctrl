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
        --dataset data/records/processed/<id>.toml --output <docs>/recovery_report_v1.md
        [--plots-dir <docs>/plots/recovery_report_v1] [--records-root ROOT]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from arm_rc_ctrl.data.records import load_record, verify_payload
from arm_rc_ctrl.data.recovery import RecoveryDatasetRecord, task_intervals_from_phases
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.experiments.evidence import StoredReport, load_report_pointer
from arm_rc_ctrl.experiments.recovery_ablation import LIMITATIONS, AblationReport, load_ablation
from arm_rc_ctrl.experiments.recovery_freeze import FreezeRecord, load_freeze
from arm_rc_ctrl.experiments.recovery_representative import (
    REPRESENTATIVE_CLASSES,
    RepresentativeRecord,
    load_representatives,
)
from arm_rc_ctrl.experiments.run_record import LoadedRun, RunPointerRecord, load_run
from arm_rc_ctrl.experiments.trajectory_plots import CURVE_ORDER, CURVE_STYLES
from arm_rc_ctrl.metrics.effort import effort_metrics
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
_MINIMUM_SAMPLES: Final = 2
_TRAJECTORY_DIMENSIONS: Final = 2


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
    torque_limits = load_scenario(records_root / dataset.scenario.config_path).limits.torque
    runs: dict[str, LoadedRun] = {}
    effort: dict[str, float] = {}
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
            effort[run_id] = float(effort_metrics(run_t, tau, torque_limits, window=window).torque_rms)
    return ReportInputs(
        pointers=pointers,
        ablation=ablation,
        freeze=freeze,
        representative=representative,
        reference=reference,
        runs=runs,
        effort=effort,
    )


def plot_recovery_pair(
    t: NDArray[np.float64],
    reference: NDArray[np.float64],
    replay_actual: NDArray[np.float64],
    rc_output: NDArray[np.float64],
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
        ("rc_output", rc_output),
        ("rc_actual", rc_actual),
    ):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != _TRAJECTORY_DIMENSIONS or array.shape[0] != time.size or array.shape[1] < 1:
            msg = f"{name} must have shape (N, dof), got {array.shape}"
            raise ValueError(msg)
        if name != "rc_output" and not np.all(np.isfinite(array)):
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
        for name in CURVE_ORDER:
            color, linestyle, linewidth, label = CURVE_STYLES[name]
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
    header = (
        "| class | scenario | tracker | RC run | replay run | jump (rad) | early gap (rad s) "
        "| settling (s) | dwell frac | desired vmax (rad/s) | torque RMS (N m) | actual jerk RMS |"
    )
    lines = [header, "| " + " | ".join(["---"] * 12) + " |"]
    for pair in inputs.representative.pairs:
        recovery = pair.recovery
        if recovery is None:
            lines.append(
                f"| {pair.kind} | {pair.scenario_id} | {pair.tracker} | {pair.rc_run} | {pair.replay_run} "
                "| n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        torque = inputs.effort.get(pair.rc_run)
        lines.append(
            f"| {pair.kind} | {pair.scenario_id} | {pair.tracker} | {pair.rc_run} | {pair.replay_run} "
            f"| {_fmt(recovery.activation_jump_rad)} | {_fmt(recovery.command_gap_early.integral)} "
            f"| {_fmt(recovery.reference_settling.settling_time_s)} "
            f"| {_fmt(recovery.generated_dwell.in_tolerance_fraction, 3)} "
            f"| {_fmt(recovery.generated_dwell.velocity_max, 3)} | {_fmt(torque)} "
            f"| {_fmt(recovery.smoothness_actual.jerk_rms, 3)} |"
        )
    return lines


def render_recovery_report(inputs: ReportInputs, *, plots: Sequence[str] = (), animations: Sequence[str] = ()) -> str:
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
        lines += [
            "",
            "## Animations",
            "",
            "Each animation renders one verified run listed in the representative table above",
            "(`scripts/play_run.py --run <run-id> --scenario configs/tasks/task_1a.toml --export ...`):",
            "",
        ]
        lines.extend(f"![{Path(a).stem}](animations/{a})" for a in animations)
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
