# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Reproduce the M3MAN-012 presentation figures; interpretation lives in report/report.md.

The script reads audited summaries without pooling configurations or trackers.
Optional trajectory illustrations reuse M3MAN-010's verified renderer and
frozen case inputs. It never fits a model, simulates a run, or writes to the store.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from arm_rc_ctrl.data.manual import ManualDatasetRecord
from arm_rc_ctrl.data.records import load_record, verify_payload
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.experiments.manual_audit import load_audit
from arm_rc_ctrl.experiments.manual_contrasts import ManualArmSummary, ManualContrastSummary
from arm_rc_ctrl.experiments.manual_figures import animate_case, load_figure_inputs, plot_case
from arm_rc_ctrl.experiments.manual_results import load_results, table_from_csv
from arm_rc_ctrl.experiments.manual_study import StudyManifest, load_study
from arm_rc_ctrl.provenance import sha256_file, worktree_state
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import open_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.storage import StorageRoot

DOCUMENT_LOCATION = Path("docs/experiments/task_1a_manual_demonstration")
TRACKERS = ("pd_v2", "computed_torque")
TRACKER_NAMES = ("PD tracker", "Computed-torque tracker")
ARMS = ("S", "M10", "C10", "replay")
DARK_CELL_THRESHOLD = 55
COLORS = ("#0072B2", "#D55E00", "#009E73", "#6B6B6B")
ARM_NAMES = ("One take (S)", "All ten (M10)", "One + synthetic (C10)", "Replay")
CLASSES = ("nominal", "posture_small", "posture_large", "force", "combined")
CLASS_NAMES = ("Nominal", "Small offset", "Large offset", "Force", "Combined")
CASES = ("feasible-best__pd_v2__nominal", "feasible-middle__pd_v2__posture-small-20261201-01")


@dataclass(frozen=True)
class ReportData:
    """Strictly loaded summaries whose bytes match the issued audit's evidence bundle."""

    study: StudyManifest
    arms: tuple[ManualArmSummary, ...]
    contrasts: tuple[ManualContrastSummary, ...]


def load_report_data(docs: Path) -> ReportData:
    """Refuse modified presentation inputs rather than illustrating unaudited counts."""
    audit = load_audit(docs / "audit/reproduction_audit_v4.json")
    if not audit.ok:
        msg = "the reporting audit did not pass"
        raise ValueError(msg)
    paths = (docs / "study_manifest_v1.json", docs / "results/results_v1.json")
    paths += tuple(docs / "results" / name for name in ("arm_summary_v1.csv", "contrast_summary_v1.csv"))
    for path in (*paths, docs / "results/figure_inputs_v1.json"):
        location = str(DOCUMENT_LOCATION / path.relative_to(docs))
        bindings = [item for item in audit.bundle if item.location == location]
        if len(bindings) != 1:
            msg = f"{location}: expected one exact audit binding, found {len(bindings)}"
            raise ValueError(msg)
        bound = bindings[0]
        if (sha256_file(path), path.stat().st_size) != (bound.sha256, bound.size):
            msg = f"{path.name}: fingerprint differs from audit v4"
            raise ValueError(msg)
    return ReportData(
        load_study(paths[0]),
        table_from_csv(paths[2].read_text(encoding="utf-8"), ManualArmSummary),
        table_from_csv(paths[3].read_text(encoding="utf-8"), ManualContrastSummary),
    )


def _median(record: ManualContrastSummary) -> float:
    """An unavailable comparison has no numeric color; do not render it as a tie."""
    if record.median is None:
        msg = "the report requires complete comparisons, but a median is unavailable"
        raise ValueError(msg)
    return record.median


def _case_counts(data: ReportData) -> dict[str, int]:
    """Read shared panel denominators, refusing inconsistent summaries."""
    counts: dict[str, int] = {}
    for record in (*data.arms, *data.contrasts):
        previous = counts.setdefault(record.scenario_class, record.n_scenarios)
        if previous != record.n_scenarios or previous <= 0:
            msg = f"{record.scenario_class}: inconsistent or nonpositive scenario count"
            raise ValueError(msg)
    if set(counts) != {*CLASSES, "all"} or counts["all"] != sum(counts[c] for c in CLASSES):
        msg = "scenario classes do not sum to the reported panel size"
        raise ValueError(msg)
    return counts


def render_summaries(data: ReportData, out: Path) -> None:
    """Plot every parent, the single M10 model, and class-specific paired medians."""
    counts = _case_counts(data)
    total = counts["all"]
    names = [c.label for c in data.study.configurations]
    labels = [f"{chr(65 + i)}  {name}" for i, name in enumerate(names)]
    lookup = {(r.configuration, r.tracker, r.scenario_class, r.arm_kind): r for r in data.arms}
    fig, axes = cast("tuple[Any, Any]", plt.subplots(1, 2, figsize=(12, 6), sharey=True, layout="constrained"))
    for ti, (tracker, axis) in enumerate(zip(TRACKERS, axes, strict=True)):
        for ai, (arm, color, legend) in enumerate(zip(ARMS, COLORS, ARM_NAMES, strict=True)):
            for ci, config in enumerate(names):
                values = lookup[config, tracker, "all", arm].per_model
                center = ci + (ai - 1.5) * 0.19
                ys = center + np.linspace(-0.065, 0.065, len(values)) if len(values) > 1 else [center]
                axis.scatter(
                    values,
                    ys,
                    color=color,
                    s=24,
                    alpha=0.8,
                    marker="D" if arm == "M10" else "o",
                    label=legend if ci == 0 else None,
                    zorder=3,
                )
        axis.set(
            title=TRACKER_NAMES[ti],
            xlabel=f"Successful scenarios per model (out of {total})",
            xlim=(-0.03 * total, 1.05 * total),
        )
        axis.set_yticks(range(len(names)), labels)
        axis.grid(axis="x", alpha=0.2)
        axis.set_ylim(len(names) - 0.5, -0.6)
    fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=9)
    fig.suptitle("Ten demonstrations do not consistently outperform one", fontsize=15)
    fig.savefig(out / "success_counts.png", dpi=125)
    plt.close(fig)

    contrasts = {(r.configuration, r.tracker, r.scenario_class, r.contrast): r for r in data.contrasts}
    fig, axes = cast("tuple[Any, Any]", plt.subplots(1, 2, figsize=(11, 7.5), layout="constrained"))
    row_labels = [f"{chr(65 + ci)} · {name}" for ci in range(len(names)) for name in ("PD", "Computed torque")]
    for axis, contrast, title in zip(
        axes, ("M10-S", "C10-R10"), ("All ten minus one", "Synthetic minus copies"), strict=True
    ):
        records = [[contrasts[c, tr, cl, contrast] for cl in CLASSES] for c in names for tr in TRACKERS]
        # A cell is a median of ten paired count differences; divide by its own class denominator only.
        values = np.array([[_median(r) / r.n_scenarios * 100 for r in row] for row in records])
        axis.imshow(values, cmap="RdBu", vmin=-100, vmax=100, aspect="auto")
        for y, row in enumerate(records):
            for x, record in enumerate(row):
                axis.text(
                    x,
                    y,
                    f"{record.median:+g}",
                    ha="center",
                    va="center",
                    color="white" if abs(values[y, x]) > DARK_CELL_THRESHOLD else "#111111",
                    fontsize=10,
                )
        class_labels = [
            f"{name}\n{counts[kind]} {'case' if counts[kind] == 1 else 'cases'}"
            for kind, name in zip(CLASSES, CLASS_NAMES, strict=True)
        ]
        axis.set_xticks(range(len(CLASSES)), class_labels, fontsize=9)
        axis.set_yticks(range(len(row_labels)), row_labels, fontsize=9)
        axis.set_title(title)
    fig.suptitle("Paired differences within each scenario class", fontsize=15)
    fig.supxlabel(
        "Numbers: median extra successful cases per parent. Red = worse; blue = better; white = tied.\n"
        "Color uses each class's own denominator. A median tie can conceal gains and losses.",
        fontsize=10,
    )
    fig.savefig(out / "class_comparisons.png", dpi=125)
    plt.close(fig)


def render_trajectories(data: ReportData, docs: Path, out: Path, store: StorageRoot, root: Path) -> None:
    """Display all ten teachers on their actual clocks, plus two preselected contrasting cases."""
    fig, axes = cast("tuple[Any, Any]", plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained"))
    for index, demonstration in enumerate(data.study.demonstrations):
        record = load_record(root / demonstration.dataset.record, ManualDatasetRecord)
        if record.artifact.payload.sha256 != demonstration.dataset.payload_sha256:
            msg = f"{demonstration.assignment}: teacher differs from the frozen study"
            raise ValueError(msg)
        samples = load_samples(verify_payload(store, record.artifact))
        color = plt.get_cmap("tab10")(index)
        axes[0].plot(samples.tip[:, 0], samples.tip[:, 1], color=color, lw=1.1, label=demonstration.assignment)
        for joint in (0, 1):
            axes[joint + 1].plot(samples.t, samples.q[:, joint], color=color, lw=1.1)
    axes[0].add_patch(Circle((0.1, 0.45), 0.01, fill=False, color="black", linestyle="--"))
    axes[0].set(xlabel="Tip x (m)", ylabel="Tip y (m)", title="Ten processed manual teachers")
    axes[0].set_aspect("equal")
    axes[0].legend(ncol=2, fontsize=7)
    for j in (0, 1):
        axes[j + 1].set(xlabel="Recording time (s)", ylabel="Joint angle (rad)", title=f"Joint {j + 1}")
    fig.savefig(out / "demonstrations.png", dpi=125)
    plt.close(fig)
    figures = load_figure_inputs(docs / "results/figure_inputs_v1.json")
    for case, filename in zip(CASES, ("ten_helps.png", "ten_hurts.png"), strict=True):
        plot_case(figures, case, out / filename, store=store, root=root)
    animate_case(figures, CASES[1], "M10", out / "ten_hurts.gif", store=store, root=root, fps=1, stride=100)


def main(argv: Sequence[str] | None = None) -> int:
    """Render to a new directory; --trajectories requires the configured external store."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs", type=Path, default=repository_root() / "docs/experiments/task_1a_manual_demonstration"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectories", action="store_true")
    args = parser.parse_args(argv)
    docs, out = Path(args.docs), Path(args.output)
    data = load_report_data(docs)
    out.mkdir(parents=True, exist_ok=False)
    render_summaries(data, out)
    root = repository_root()
    if args.trajectories:
        render_trajectories(data, docs, out, open_storage(), root)
    commit, dirty = worktree_state(root)
    sources = [
        docs / "audit/reproduction_audit_v4.json",
        docs / "study_manifest_v1.json",
        docs / "results/results_v1.json",
    ]
    sources += [docs / "results" / d.name for d in load_results(docs / "results/results_v1.json").documents]
    manifest = {
        "task": "M3MAN-012",
        "project_commit": commit,
        "project_dirty": dirty,
        "renderer_sha256": sha256_file(Path(__file__)),
        "trajectories": bool(args.trajectories),
        "inputs": {str(p.relative_to(root)): sha256_file(p) for p in sources},
        "outputs": {p.name: {"sha256": sha256_file(p), "size": p.stat().st_size} for p in sorted(out.iterdir())},
        "case_ids": list(CASES) if args.trajectories else [],
    }
    (out / "render_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0
