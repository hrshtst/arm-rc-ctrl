# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-010: figure and animation inputs of the manual study, and the tools that render them (plan section 7.1).

The frozen representative rule chooses which cases to illustrate; this module
records, for each chosen case, exactly which stored runs and which recorded
demonstration a figure draws, each bound by digest, and renders them. A figure
keeps three things visibly apart, because the plan asks for it: the recorded
demonstration on its own recording clock, the reference a controller was
commanded with (the RC readout's generated reference, or the replayed
recording), and the motion the arm actually made. Time is the task clock, zero
at activation, so the negative-time warm-up stays visible.

Rendering is a validation of the inputs and the tools here; presentation
figures for the human-facing report are produced later from the same inputs.

Command lines::

    python -m arm_rc_ctrl.experiments.manual_figures cases --inputs .../results/figure_inputs_v1.json
    python -m arm_rc_ctrl.experiments.manual_figures plot --inputs ... --case <case id> --out case.png
    python -m arm_rc_ctrl.experiments.manual_figures animate --inputs ... --case <case id> --role M10
        --out case.gif [--fps 10] [--stride 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from PIL import Image, ImageSequence
from skelarm import compute_forward_kinematics

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.data.manual import ManualDatasetRecord
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.data.records import load_record, verify_payload
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.experiments.manual_evaluation import ManualRunArtifact, load_verified_payload
from arm_rc_ctrl.experiments.manual_study import EXPERIMENT_LABEL
from arm_rc_ctrl.provenance import canonical_json, sha256_file
from arm_rc_ctrl.rc.recipe import DatasetSource
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import build_robot_skeleton
from arm_rc_ctrl.storage import open_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.experiments.manual_handoff import RepresentativeRule
    from arm_rc_ctrl.experiments.manual_results import ManualRunRow, ManualSelections
    from arm_rc_ctrl.experiments.manual_study import StudyManifest
    from arm_rc_ctrl.scenario import RobotConfig
    from arm_rc_ctrl.storage import StorageRoot

__all__ = [
    "FIGURE_INPUTS_VERSION",
    "ROLE_COLOURS",
    "ManualFigureCase",
    "ManualFigureInputs",
    "ManualFigureRun",
    "animate_case",
    "figure_inputs",
    "figure_inputs_to_json",
    "load_figure_inputs",
    "main",
    "plot_case",
]

FIGURE_INPUTS_VERSION: Final = 1
ROLE_COLOURS: Final = {
    "S": "#1f77b4",
    "M10": "#d62728",
    "R10": "#2ca02c",
    "C10": "#9467bd",
    "replay": "#7f7f7f",
}
"""One colour per role, shared by every figure so a role reads the same everywhere."""
INDISTINGUISHABLE_RAD: Final = 1e-6
"""Joint-angle difference below which two measured trajectories are named as retracing each other in a legend."""
GIF_COLOURS: Final = 32
"""Palette size of an exported animation; more than the handful of colours a frame actually uses."""
ROLE_WIDTHS: Final = {"S": 3.2, "M10": 2.5, "R10": 1.9, "C10": 1.4, "replay": 1.0}
"""Line widths narrowing in role order, so a run that retraces another stays visible as an outline around it.

Identical runs are not rare here: repeating one demonstration ten times at a
correspondingly scaled ridge penalty reproduces the singleton fit.
"""
_REPLAY: Final = "replay"
_DEMONSTRATION: Final = "#000000"


@dataclass(frozen=True)
class ManualFigureRun:
    """One stored run a figure draws, with its role in the comparison and its verdict."""

    role: str
    label: str
    success: bool
    pulse_start_s: float | None
    pulse_end_s: float | None
    run: ManualRunArtifact


@dataclass(frozen=True)
class ManualFigureCase:
    """One illustrated case: the runs to draw beside the demonstration they descend from."""

    case_id: str
    configuration: str
    tracker: str
    scenario_id: str
    scenario_class: str
    categories: tuple[str, ...]
    activation_s: float
    teacher_assignment: str
    teacher: DatasetSource
    runs: tuple[ManualFigureRun, ...]
    missing: tuple[str, ...]
    """Roles the case names whose runs do not exist, stated so a figure's gap is legible."""


@dataclass(frozen=True)
class ManualFigureInputs:
    """Every case the frozen rule chose, with the task configuration its kinematics come from."""

    experiment: str
    scenario_file: str
    scenario_sha256: str
    cases: tuple[ManualFigureCase, ...]
    schema_version: int = field(default=FIGURE_INPUTS_VERSION)

    def __post_init__(self) -> None:
        """Case identifiers are unique, so a command naming one names exactly one."""
        ids = [case.case_id for case in self.cases]
        if len(set(ids)) != len(ids) or self.experiment != EXPERIMENT_LABEL:
            msg = "figure inputs name each case once and belong to this experiment"
            raise ValueError(msg)


def figure_inputs_to_json(inputs: ManualFigureInputs) -> str:
    """Canonical JSON of the figure inputs, newline-terminated."""
    return canonical_json(to_mapping(inputs)) + "\n"


def load_figure_inputs(path: Path) -> ManualFigureInputs:
    """Strictly rebuild the figure inputs from their JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), ManualFigureInputs)


def _figure_run(role: str, row: ManualRunRow) -> ManualFigureRun | None:
    if row.run_uri is None or row.success is None:
        return None
    return ManualFigureRun(
        role=role,
        label=row.model_label,
        success=row.success,
        pulse_start_s=row.pulse_start_s,
        pulse_end_s=row.pulse_end_s,
        run=ManualRunArtifact(
            artifact_id=cast("str", row.run_artifact_id),
            uri=row.run_uri,
            sha256=cast("str", row.run_sha256),
            size=cast("int", row.run_size),
            arrays_sha256=cast("str", row.arrays_sha256),
            sources=cast("tuple[str, ...]", row.sources),
        ),
    )


def figure_inputs(
    selections: ManualSelections,
    rows: Sequence[ManualRunRow],
    *,
    manifest: StudyManifest,
    rule: RepresentativeRule,
    scenario_file: str,
    root: Path,
) -> ManualFigureInputs:
    """The runs and demonstration each selected case draws: the rule's four arms and the parent's replay."""
    index = {(r.configuration, r.tracker, r.arm, r.scenario_id): r for r in rows}
    teacher = next(d for d in manifest.demonstrations if d.assignment == rule.parent)
    roles = (*zip(rule.arms, rule.arm_labels, strict=True), (_REPLAY, f"{_REPLAY}/{rule.parent}"))
    cases: list[ManualFigureCase] = []
    for application in selections.applications:
        for case in application.selection.cases:
            runs: list[ManualFigureRun] = []
            missing: list[str] = []
            for role, arm in roles:
                row = index.get((application.configuration, application.tracker, arm, case.scenario_id))
                drawn = None if row is None else _figure_run(role, row)
                if drawn is None:
                    missing.append(role)
                else:
                    runs.append(drawn)
            activation = {
                r.activation_s
                for _, arm in roles
                if (r := index.get((application.configuration, application.tracker, arm, case.scenario_id)))
                and r.activation_s is not None
            }
            if len(activation) != 1:
                msg = f"{case.scenario_id}: the runs of one case disagree on activation ({sorted(activation)})"
                raise ValueError(msg)
            kind = next(r.scenario_class for r in rows if r.scenario_id == case.scenario_id)
            cases.append(
                ManualFigureCase(
                    case_id=f"{application.configuration}__{application.tracker}__{case.scenario_id}",
                    configuration=application.configuration,
                    tracker=application.tracker,
                    scenario_id=case.scenario_id,
                    scenario_class=kind,
                    categories=case.categories,
                    activation_s=activation.pop(),
                    teacher_assignment=teacher.assignment,
                    teacher=teacher.dataset,
                    runs=tuple(runs),
                    missing=tuple(missing),
                )
            )
    return ManualFigureInputs(
        experiment=EXPERIMENT_LABEL,
        scenario_file=scenario_file,
        scenario_sha256=sha256_file(root / scenario_file),
        cases=tuple(cases),
    )


# --- loading what a case draws -------------------------------------------------------------------


@dataclass(frozen=True)
class _Drawn:
    """One run's arrays on the task clock, with the endpoint of the reference it was commanded with."""

    run: ManualFigureRun
    t: NDArray[np.float64]
    q: NDArray[np.float64]
    q_reference: NDArray[np.float64]
    tip: NDArray[np.float64]
    tip_reference: NDArray[np.float64]


def _points(robot: RobotConfig, q: NDArray[np.float64]) -> NDArray[np.float64]:
    """The origin and every link end of the arm at each posture, the endpoint last; NaN where ``q`` is not finite.

    The skeleton may carry a base link ahead of the configured ones, so the
    point count follows the skeleton rather than the configuration.
    """
    finite = np.all(np.isfinite(q), axis=1)
    first = int(np.argmax(finite)) if bool(np.any(finite)) else None
    skeleton = build_robot_skeleton(robot, np.zeros(q.shape[1]) if first is None else q[first])
    out = np.full((q.shape[0], len(skeleton.links) + 1, 2), np.nan)
    for i in np.flatnonzero(finite):
        skeleton.q = q[i]
        compute_forward_kinematics(skeleton)
        out[i, 0] = (0.0, 0.0)
        for j, link in enumerate(skeleton.links, start=1):
            out[i, j] = (link.xe, link.ye)
    return out


def _load_run(store: StorageRoot, robot: RobotConfig, case: ManualFigureCase, run: ManualFigureRun) -> _Drawn:
    _, arrays = load_verified_payload(store, run.run, where=f"{case.case_id} {run.label}")
    q = np.asarray(arrays["q"], dtype=np.float64)
    q_reference = np.asarray(arrays["q_desired"], dtype=np.float64)
    return _Drawn(
        run=run,
        t=np.asarray(arrays["t"], dtype=np.float64) - case.activation_s,
        q=q,
        q_reference=q_reference,
        tip=np.asarray(arrays["tip"], dtype=np.float64),
        tip_reference=_points(robot, q_reference)[:, -1],
    )


def _load_teacher(store: StorageRoot, root: Path, case: ManualFigureCase) -> SampleSet:
    record = load_record(root / case.teacher.record, ManualDatasetRecord)
    if (record.artifact.artifact_id, record.artifact.payload.sha256) != (
        case.teacher.artifact_id,
        case.teacher.payload_sha256,
    ):
        msg = f"{case.teacher.record} does not describe the demonstration {case.teacher.artifact_id}"
        raise ValueError(msg)
    samples = load_samples(verify_payload(store, record.artifact))
    record.check_samples(samples)
    return samples


def _case(inputs: ManualFigureInputs, case_id: str) -> ManualFigureCase:
    for case in inputs.cases:
        if case.case_id == case_id:
            return case
    msg = f"no case {case_id!r}; the inputs hold {len(inputs.cases)} cases (list them with `cases`)"
    raise KeyError(msg)


def _refuse(out: Path) -> None:
    if out.exists():
        msg = f"refusing to overwrite {out}"
        raise FileExistsError(msg)


def _verdict(run: ManualFigureRun) -> str:
    return "success" if run.success else "failure"


def _legend_label(d: _Drawn, drawn: Sequence[_Drawn]) -> str:
    """The run's label and verdict, naming an earlier run it retraces within a tolerance no plot could show."""
    for other in drawn[: drawn.index(d)]:
        if other.q.shape == d.q.shape:
            gap = float(np.max(np.abs(other.q - d.q)))
            if gap <= INDISTINGUISHABLE_RAD:
                return f"{d.run.label}: {_verdict(d.run)} (within {gap:.1e} rad of {other.run.label.split('/', 1)[1]})"
    return f"{d.run.label}: {_verdict(d.run)}"


def _shade(axis: Any, case: ManualFigureCase, drawn: Sequence[_Drawn]) -> None:  # noqa: ANN401 - a matplotlib axis
    """Mark the warm-up and every pulse window, so a transient is never mistaken for a controller's own motion."""
    start = min(float(d.t[0]) for d in drawn)
    if start < 0:
        axis.axvspan(start, 0.0, color="#eeeeee", zorder=0)
    for d in drawn:
        if d.run.pulse_start_s is not None and d.run.pulse_end_s is not None:
            axis.axvspan(
                d.run.pulse_start_s - case.activation_s,
                d.run.pulse_end_s - case.activation_s,
                color=ROLE_COLOURS[d.run.role],
                alpha=0.12,
                zorder=0,
            )


def plot_case(inputs: ManualFigureInputs, case_id: str, out: Path, *, store: StorageRoot, root: Path) -> Path:
    """Render one case: endpoint paths, distance to the target, and both joints, against the demonstration."""
    _refuse(out)
    case = _case(inputs, case_id)
    scenario = load_manual_scenario(root / inputs.scenario_file)
    if sha256_file(root / inputs.scenario_file) != inputs.scenario_sha256:
        msg = f"{inputs.scenario_file} is not the task configuration the inputs were derived under"
        raise ValueError(msg)
    target = np.asarray(scenario.task.target, dtype=np.float64)
    teacher = _load_teacher(store, root, case)
    drawn = [_load_run(store, scenario.robot, case, run) for run in case.runs]
    # matplotlib's keyword arguments are incompletely typed
    fig, axes = cast("tuple[Any, Any]", plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True))
    space, distance, joint1, joint2 = axes[0][0], axes[0][1], axes[1][0], axes[1][1]
    space.add_patch(Circle((float(target[0]), float(target[1])), scenario.task.tolerance, fill=False, color="k"))
    for d in drawn:
        colour, width = ROLE_COLOURS[d.run.role], ROLE_WIDTHS[d.run.role]
        space.plot(d.tip[:, 0], d.tip[:, 1], "-", color=colour, lw=width)
        space.plot(d.tip_reference[:, 0], d.tip_reference[:, 1], "--", color=colour, lw=width * 0.6)
        distance.plot(d.t, np.hypot(d.tip[:, 0] - target[0], d.tip[:, 1] - target[1]), "-", color=colour, lw=width)
        for axis, joint in ((joint1, 0), (joint2, 1)):
            axis.plot(d.t, d.q[:, joint], "-", color=colour, lw=width)
            axis.plot(d.t, d.q_reference[:, joint], "--", color=colour, lw=width * 0.6)
    # The demonstration is drawn last, so the replay that follows it closely cannot hide it.
    demonstration = {"color": _DEMONSTRATION, "lw": 1.3, "zorder": 5}
    space.plot(teacher.tip[:, 0], teacher.tip[:, 1], ":", **demonstration)
    teacher_distance = np.hypot(teacher.tip[:, 0] - target[0], teacher.tip[:, 1] - target[1])
    distance.plot(teacher.t, teacher_distance, ":", **demonstration)
    for axis, joint in ((joint1, 0), (joint2, 1)):
        axis.plot(teacher.t, teacher.q[:, joint], ":", **demonstration)
    distance.axhline(scenario.task.tolerance, color="k", lw=0.8)
    for axis in (distance, joint1, joint2):
        _shade(axis, case, drawn)
        axis.set_xlabel("task time (s); warm-up shaded, demonstration on its recording clock")
    space.set_aspect("equal", adjustable="datalim")
    space.set(xlabel="x (m)", ylabel="y (m)", title="endpoint path; target radius drawn")
    distance.set(ylabel="distance to target (m)", title="endpoint distance; dwell radius drawn")
    joint1.set(ylabel="joint 1 (rad)", title="joint 1")
    joint2.set(ylabel="joint 2 (rad)", title="joint 2")
    handles = [
        Line2D([], [], color=_DEMONSTRATION, ls=":", label=f"demonstration {case.teacher_assignment} (recorded)"),
        Line2D([], [], color="0.3", ls="--", label="commanded reference (RC generated, or replayed recording)"),
        Line2D([], [], color="0.3", ls="-", label="actual motion"),
        *[
            Line2D([], [], color=ROLE_COLOURS[d.run.role], lw=ROLE_WIDTHS[d.run.role], label=_legend_label(d, drawn))
            for d in drawn
        ],
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8)
    missing = f"; no run for {', '.join(case.missing)}" if case.missing else ""
    fig.suptitle(
        f"{case.configuration} [{case.tracker}] {case.scenario_id} ({case.scenario_class}); "
        f"chosen as {', '.join(case.categories)}{missing}",
        fontsize=10,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90, pil_kwargs={"optimize": True})
    plt.close(fig)
    return out


def animate_case(
    inputs: ManualFigureInputs,
    case_id: str,
    role: str,
    out: Path,
    *,
    store: StorageRoot,
    root: Path,
    fps: int = 10,
    stride: int = 10,
) -> Path:
    """Animate one run of a case: the actual arm, the commanded arm, and the demonstration's endpoint path."""
    _refuse(out)
    if fps <= 0 or stride < 1:
        msg = f"fps must be positive and stride at least 1, got {fps} and {stride}"
        raise ValueError(msg)
    case = _case(inputs, case_id)
    chosen = [run for run in case.runs if run.role == role]
    if len(chosen) != 1:
        msg = f"{case_id} has {len(chosen)} runs in role {role!r}; roles present: {[r.role for r in case.runs]}"
        raise ValueError(msg)
    scenario = load_manual_scenario(root / inputs.scenario_file)
    teacher = _load_teacher(store, root, case)
    d = _load_run(store, scenario.robot, case, chosen[0])
    actual, commanded = _points(scenario.robot, d.q), _points(scenario.robot, d.q_reference)
    frames: NDArray[np.int64] = np.arange(0, int(d.t.shape[0]), stride, dtype=np.int64)
    reach = float(sum(link.length for link in scenario.robot.links))
    fig, axis = cast("tuple[Any, Any]", plt.subplots(figsize=(4.8, 4.8), constrained_layout=True))
    axis.set(xlim=(-reach * 1.05, reach * 1.05), ylim=(-reach * 1.05, reach * 1.05), xlabel="x (m)", ylabel="y (m)")
    axis.set_aspect("equal")
    target = scenario.task.target
    axis.add_patch(Circle((target[0], target[1]), scenario.task.tolerance, fill=False, color="k"))
    axis.plot(teacher.tip[:, 0], teacher.tip[:, 1], ":", color=_DEMONSTRATION, lw=1.2)
    colour = ROLE_COLOURS[role]
    (commanded_line,) = axis.plot([], [], "--o", color=colour, alpha=0.45, lw=1.5, ms=3)
    (actual_line,) = axis.plot([], [], "-o", color=colour, lw=2.5, ms=4)
    (trail,) = axis.plot([], [], "-", color=colour, lw=0.8, alpha=0.7)
    clock = axis.text(0.02, 0.97, "", transform=axis.transAxes, va="top", fontsize=9)
    axis.legend(
        handles=[
            Line2D([], [], color=_DEMONSTRATION, ls=":", label=f"demonstration {case.teacher_assignment} (recorded)"),
            Line2D([], [], color=colour, ls="--", alpha=0.45, label="commanded reference"),
            Line2D([], [], color=colour, ls="-", label=f"actual motion ({_verdict(d.run)})"),
        ],
        loc="lower left",
        fontsize=7,
    )
    axis.set_title(f"{d.run.label} [{case.tracker}] {case.scenario_id}", fontsize=9)
    pulse = (d.run.pulse_start_s, d.run.pulse_end_s)

    def draw(frame: int) -> tuple[Any, ...]:
        i = int(frames[frame])
        actual_line.set_data(actual[i, :, 0], actual[i, :, 1])
        commanded_line.set_data(commanded[i, :, 0], commanded[i, :, 1])
        trail.set_data(d.tip[: i + 1, 0], d.tip[: i + 1, 1])
        now = float(d.t[i])
        phase = "warm-up" if now < 0 else "active"
        if pulse[0] is not None and pulse[1] is not None and pulse[0] <= now + case.activation_s <= pulse[1]:
            phase += ", pulse"
        clock.set_text(f"task time {now:+.2f} s ({phase})")
        return actual_line, commanded_line, trail, clock

    animation = FuncAnimation(fig, draw, frames=len(frames), blit=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = out.with_name(f".{out.name}.staging.gif")
    animation.save(str(staged), writer=PillowWriter(fps=fps), dpi=72)
    plt.close(fig)
    _reduce_palette(staged, out)
    return out


def _reduce_palette(staged: Path, out: Path, colours: int = GIF_COLOURS) -> None:
    """Re-save a GIF on a small undithered palette: a line drawing needs few colours, and the file stays small."""
    with Image.open(staged) as source:
        duration = cast("int", source.info.get("duration", 100))
        frames = [
            frame.convert("RGB").quantize(colors=colours, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
            for frame in ImageSequence.Iterator(source)
        ]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=True)
    staged.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Render the manual study's illustrated cases from their inputs.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    cases = subparsers.add_parser("cases", help="list the case identifiers and the roles each case draws")
    plot = subparsers.add_parser("plot", help="render one case as a static figure")
    animate = subparsers.add_parser("animate", help="animate one run of one case")
    for sub in (cases, plot, animate):
        sub.add_argument("--inputs", type=Path, required=True, help="the committed figure inputs")
    for sub in (plot, animate):
        sub.add_argument("--case", required=True, help="a case identifier, as `cases` lists them")
        sub.add_argument("--out", type=Path, required=True, help="the file to write (must not exist)")
    animate.add_argument("--role", required=True, help="S, M10, R10, C10 or replay")
    animate.add_argument("--fps", type=int, default=10, help="frames per second of the GIF")
    animate.add_argument("--stride", type=int, default=10, help="samples between frames")
    args = parser.parse_args(None if argv is None else list(argv))
    inputs = load_figure_inputs(cast("Path", args.inputs))
    if args.subcommand == "cases":
        for case in inputs.cases:
            print(f"{case.case_id}\t{','.join(case.categories)}\t{','.join(r.role for r in case.runs)}")
        return 0
    root = repository_root()
    store = open_storage()
    if args.subcommand == "plot":
        written = plot_case(inputs, cast("str", args.case), cast("Path", args.out), store=store, root=root)
    else:
        written = animate_case(
            inputs,
            cast("str", args.case),
            cast("str", args.role),
            cast("Path", args.out),
            store=store,
            root=root,
            fps=int(cast("int", args.fps)),
            stride=int(cast("int", args.stride)),
        )
    print(written)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
