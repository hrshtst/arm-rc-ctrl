# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only
"""Reproduce the M3MS-008 expert HTML report from audited evidence, without running a simulation."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from arm_rc_ctrl.experiments.manual_contrasts import ManualArmSummary, ManualContrastSummary
from arm_rc_ctrl.experiments.manual_evaluation import ManualRunArtifact
from arm_rc_ctrl.experiments.manual_figures import ManualFigureCase, ManualFigureInputs, ManualFigureRun
from arm_rc_ctrl.experiments.manual_results import ManualRunRow, table_from_csv
from arm_rc_ctrl.experiments.manual_search_audit import load_search_audit
from arm_rc_ctrl.experiments.manual_search_freeze import ManualSearchFreeze, read_freeze
from arm_rc_ctrl.experiments.manual_search_report_plots import (
    ROLES,
    Illustration,
    case_html,
    render_case,
    save_indexed_plot,
    verify_outputs,
)
from arm_rc_ctrl.experiments.manual_search_report_plots import (
    verify_report as verify_fingerprints,
)
from arm_rc_ctrl.experiments.manual_search_results import SearchComparisonResults, load_search_results
from arm_rc_ctrl.experiments.manual_study import load_study
from arm_rc_ctrl.provenance import sha256_file, verify_artifact, worktree_state
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import open_storage

DOCS = Path("docs/experiments/task_1a_manual_esn_search")
TEACHERS = Path("docs/experiments/task_1a_manual_demonstration/study_manifest_v1.json")
TASK = Path("configs/tasks/task_1a_manual_v2.toml")
TRACKERS = ("pd_v2", "computed_torque")
DARK_CELL_THRESHOLD = 0.5
CLASSES = ("nominal", "posture_small", "posture_large", "force", "combined")


@dataclass(frozen=True)
class ReportData:
    """Presentation inputs accepted only through the issued audit's digest bindings."""

    freeze: ManualSearchFreeze
    index: SearchComparisonResults
    arms: tuple[ManualArmSummary, ...]
    contrasts: tuple[ManualContrastSummary, ...]


def load_data(root: Path) -> ReportData:
    """Verify the audit, index, freeze and summary bytes before interpreting them."""
    docs = root / DOCS
    index_path = docs / "results/results_v1.json"
    audit = load_search_audit(docs / "audit/audit_v2.json")
    if not audit.passed or audit.results_sha256 != sha256_file(index_path):
        msg = "results index fingerprint differs from the issued passing audit"
        raise ValueError(msg)
    index = load_search_results(index_path)
    for document in index.documents:
        path = docs / "results" / document.name
        if sha256_file(path) != document.sha256 or path.stat().st_size != document.size:
            msg = f"{document.name}: summary fingerprint differs"
            raise ValueError(msg)
    frozen = docs / "freeze/selection_v1.json"
    if sha256_file(frozen) != audit.freeze_sha256 or audit.freeze_sha256 != index.inputs.freeze_sha256:
        msg = "freeze fingerprint differs from the audit or index"
        raise ValueError(msg)
    return ReportData(
        read_freeze(frozen),
        index,
        table_from_csv((docs / "results/arm_summary_v1.csv").read_text(), ManualArmSummary),
        table_from_csv((docs / "results/contrast_summary_v1.csv").read_text(), ManualContrastSummary),
    )


def signed(value: float | None) -> str:
    """A missing value must remain distinct from a zero difference."""
    return "—" if value is None else f"{value:+g}".replace("-", "\u2212") if value else "0"


def summary_table(data: ReportData) -> str:
    """Generate the primary parent-matched comparison directly from the committed summaries."""
    arms = {(r.configuration, r.tracker, r.scenario_class, r.arm_kind): r for r in data.arms}
    contrasts = {(r.configuration, r.tracker, r.scenario_class, r.contrast): r for r in data.contrasts}
    lines: list[str] = []
    for config in data.freeze.chosen:
        for tracker in TRACKERS:
            single = arms[config.configuration, tracker, "all", "S"]
            multi = arms[config.configuration, tracker, "all", "M10"]
            diff = contrasts[config.configuration, tracker, "all", "M10-S"]
            lines.append(
                f'<tr data-result="{config.configuration}/{tracker}"><th>T{config.trial} / '
                f"{'PD' if tracker == 'pd_v2' else 'Computed torque'}</th>"
                f"<td>{single.median:g} [{single.minimum}\u2013{single.maximum}]</td><td>{multi.successes}</td>"
                f"<td>{signed(diff.median)} [{signed(diff.minimum)} … {signed(diff.maximum)}]</td>"
                f"<td>{diff.parents_improved} / {diff.parents_worsened} / {diff.parents_tied}</td></tr>"
            )
    return "\n".join(lines)


def config_table(data: ReportData) -> str:
    """Expose every searched coordinate of the selected parameter points."""
    lines: list[str] = []
    for c in data.freeze.chosen:
        p = c.point
        values = (
            c.trial,
            p.n_neurons,
            p.spectral_radius,
            p.sparsity,
            p.leak_rate,
            p.input_scaling,
            p.alpha_0,
            p.warmup_s,
        )
        lines.append("<tr>" + "".join(f"<td>{v:.6g}</td>" for v in values) + "</tr>")
    return "\n".join(lines)


def arm_table(data: ReportData) -> str:
    """All five arms' descriptive distributions, counting M10 only once."""
    lines: list[str] = []
    lookup = {(r.configuration, r.tracker, r.arm_kind): r for r in data.arms if r.scenario_class == "all"}
    for c in data.freeze.chosen:
        for tracker in TRACKERS:
            cells = [f"T{c.trial} / {'PD' if tracker == 'pd_v2' else 'Computed torque'}"]
            for arm in ROLES:
                r = lookup[c.configuration, tracker, arm]
                cells.append(f"{r.median:g} [{r.minimum}\u2013{r.maximum}]")
            lines.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    return "\n".join(lines)


def replay_table(data: ReportData) -> str:
    """Expose all four learned-arm versus parent-matched replay contrasts."""
    records = {(r.configuration, r.tracker, r.contrast): r for r in data.contrasts if r.scenario_class == "all"}
    lines: list[str] = []
    for c in data.freeze.chosen:
        for tracker in TRACKERS:
            cells = [f"T{c.trial} / {'PD' if tracker == 'pd_v2' else 'Computed torque'}"]
            for arm in ("S", "M10", "R10", "C10"):
                r = records[c.configuration, tracker, f"{arm}-replay"]
                cells.append(f"{signed(r.median)} [{signed(r.minimum)} … {signed(r.maximum)}]")
            lines.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    return "\n".join(lines)


def render_summaries(data: ReportData, out: Path) -> None:
    """Show all parents and class denominators, with no pooled ranking or uncertainty claims."""
    chosen = data.freeze.chosen
    lookup = {(r.configuration, r.tracker, r.scenario_class, r.arm_kind): r for r in data.arms}
    contrast = {(r.configuration, r.tracker, r.scenario_class, r.contrast): r for r in data.contrasts}
    counts = {kind: lookup[chosen[0].configuration, TRACKERS[0], kind, "S"].n_scenarios for kind in (*CLASSES, "all")}
    if any(r.n_scenarios != counts[r.scenario_class] for r in (*data.arms, *data.contrasts)):
        msg = "inconsistent scenario denominators"
        raise ValueError(msg)
    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#7f7f7f")
    fig, axes = cast("tuple[Any, Any]", plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained"))
    for tracker, ax in zip(TRACKERS, axes, strict=True):
        for i, c in enumerate(chosen):
            for j, (arm, color) in enumerate(zip(ROLES, colors, strict=True)):
                r = lookup[c.configuration, tracker, "all", arm]
                values = np.asarray(r.per_model)
                y = i + (j - 2) * 0.14
                ax.scatter(values, np.full(len(values), y), color=color, s=24, alpha=0.7, label=arm if i == 0 else None)
                ax.plot([r.minimum, r.maximum], [y, y], color=color, lw=1)
        ax.set(
            yticks=range(len(chosen)),
            yticklabels=[f"Trial {c.trial}" for c in chosen],
            xlim=(-1, counts["all"] + 2),
            xlabel=f"Successful cases out of {counts['all']}",
            title=tracker,
        )
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_ylim(len(chosen) - 0.5, -0.5)
    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=8)
    fig.suptitle("One point per model / replay parent; M10 appears once per configuration")
    fig.savefig(out / "success_counts.png", dpi=120)
    plt.close(fig)
    fig, axes = cast("tuple[Any, Any]", plt.subplots(1, 2, figsize=(13, 5), layout="constrained"))
    for tracker, ax in zip(TRACKERS, axes, strict=True):
        matrix = np.asarray(
            [
                [
                    cast("float", contrast[c.configuration, tracker, kind, "M10-S"].median) / counts[kind]
                    for kind in CLASSES
                ]
                for c in chosen
            ],
            dtype=float,
        )
        ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu", aspect="auto")
        for i, c in enumerate(chosen):
            for j, kind in enumerate(CLASSES):
                median = contrast[c.configuration, tracker, kind, "M10-S"].median
                ax.text(
                    j,
                    i,
                    f"{signed(median)} / {counts[kind]}",
                    ha="center",
                    va="center",
                    color="white" if abs(matrix[i, j]) > DARK_CELL_THRESHOLD else "black",
                )
        ax.set(
            xticks=range(len(CLASSES)),
            xticklabels=[k.replace("_", "\n") for k in CLASSES],
            yticks=range(len(chosen)),
            yticklabels=[f"Trial {c.trial}" for c in chosen],
            title=tracker,
        )
    fig.suptitle(
        "M10 \u2212 S: median paired success-count difference / class size\nRed = fewer successes; blue = more"
    )
    fig.savefig(out / "paired_classes.png", dpi=120)
    plt.close(fig)
    render_parent_comparisons(data, out)


def render_parent_comparisons(data: ReportData, out: Path) -> None:
    """Show every parent's contrast instead of letting a median hide a failed teacher."""
    chosen = data.freeze.chosen
    contrast = {(r.configuration, r.tracker, r.scenario_class, r.contrast): r for r in data.contrasts}
    fig, axes = cast(
        "tuple[Any, Any]",
        plt.subplots(len(chosen), len(TRACKERS), figsize=(12, 9), sharex=True, sharey=True, layout="constrained"),
    )
    for i, c in enumerate(chosen):
        for j, tracker in enumerate(TRACKERS):
            ax = axes[i, j]
            for label, color in (("M10-S", "#d62728"), ("C10-R10", "#9467bd"), ("M10-C10", "#b88216")):
                row = contrast[c.configuration, tracker, "all", label]
                ax.plot(range(1, 11), row.differences, "o-", label=label, color=color, lw=1)
            ax.axhline(0, color="black", lw=0.7)
            ax.set(title=f"Trial {c.trial} / {tracker}", xticks=range(1, 11), ylabel="Difference in successes")
            ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("Parent demonstration D01\u2013D10 (M10 is the same model within each panel)")
    fig.savefig(out / "paired_parents.png", dpi=120)
    plt.close(fig)
    for path in out.glob("*.png"):
        save_indexed_plot(path)


CASE_SPECS = (
    Illustration(
        "t1-nominal",
        "search-t0001",
        "pd_v2",
        "D01",
        "nominal",
        "T1 · Nominal feasibility",
        "All five arms pass. This is the type of evidence the nominal objective rewards; it does not measure recovery.",
    ),
    Illustration(
        "t1-offset",
        "search-t0001",
        "pd_v2",
        "D01",
        "posture-small-20261201-02",
        "T1 · Synthetic episodes help locally",
        "A post-hoc example in which C10 succeeds but M10 does not. The aggregate parent plot shows how "
        "uneven this benefit is.",
    ),
    Illustration(
        "t1-force",
        "search-t0001",
        "pd_v2",
        "D01",
        "force-12N-000deg",
        "T1 · A force pulse defeats nominally successful learners",
        "The learned arms fail the final dwell after the pulse; replay recovers. A nominal pass does "
        "not test this transition.",
    ),
    Illustration(
        "t2-nominal",
        "search-t0002",
        "pd_v2",
        "D01",
        "nominal",
        "T2 · M10 passes, synthetic training fails",
        "The selected M10 and singleton pass at reset; C10 fails even here. The search never scored C10.",
    ),
    Illustration(
        "t2-offset",
        "search-t0002",
        "pd_v2",
        "D01",
        "posture-small-20261201-00",
        "T2 · A small initial offset breaks M10",
        "S and replay pass; M10 fails to hold and C10 aborts. This is the same parent and tracker as "
        "the preceding nominal example.",
    ),
    Illustration(
        "t2-torque",
        "search-t0002",
        "computed_torque",
        "D01",
        "nominal",
        "T2 · Failure at the velocity limit",
        "At nominal reset the computed-torque tracker still permits M10 to pass, while C10 aborts. "
        "Safety stops are retained as failures.",
    ),
    Illustration(
        "t4-offset",
        "search-t0004",
        "pd_v2",
        "D01",
        "posture-small-20261201-00",
        "T4 · Close to the ring still fails",
        "M10 ends just outside the strict target region; S, copies, C10 and replay pass. Small endpoint "
        "error is not the full success predicate.",
    ),
    Illustration(
        "t4-parent03",
        "search-t0004",
        "pd_v2",
        "D03",
        "nominal",
        "T4 · The exceptional synthetic parent",
        "C10/D03 fails even nominally. The other nine C10 parents pass all 65 PD cases, so a median of "
        "65 must not hide this parent.",
    ),
    Illustration(
        "t4-force-pd",
        "search-t0004",
        "pd_v2",
        "D01",
        "force-12N-000deg",
        "T4 · Recovery with PD",
        "S, copies, C10 and replay recover; M10 does not. Compare with the identical case under computed torque below.",
    ),
    Illustration(
        "t4-force-ct",
        "search-t0004",
        "computed_torque",
        "D01",
        "force-12N-000deg",
        "T4 · Tracker-dependent aborts",
        "With the fixed computed-torque tracker the learned arms abort at the speed limit, while replay "
        "succeeds. Tracker sensitivity cannot be dismissed.",
    ),
)


def case_inputs(rows: tuple[ManualRunRow, ...], root: Path) -> ManualFigureInputs:
    """Bind post-hoc illustrations to the audited rows, retaining the parent of each baseline."""
    study = load_study(root / TEACHERS)
    if study.scenario.path != str(TASK) or sha256_file(root / TASK) != study.scenario.sha256:
        msg = "task configuration fingerprint differs from the teacher study"
        raise ValueError(msg)
    teachers = {d.assignment: d.dataset for d in study.demonstrations}
    lookup = {(r.configuration, r.tracker, r.arm, r.scenario_id): r for r in rows}
    cases: list[ManualFigureCase] = []
    for spec in CASE_SPECS:
        selected = [
            lookup[spec.configuration, spec.tracker, role if role == "M10" else f"{role}/{spec.parent}", spec.scenario]
            for role in ROLES
        ]
        runs = tuple(
            ManualFigureRun(
                role=role,
                label=r.model_label,
                success=cast("bool", r.success),
                pulse_start_s=r.pulse_start_s,
                pulse_end_s=r.pulse_end_s,
                run=ManualRunArtifact(
                    artifact_id=cast("str", r.run_artifact_id),
                    uri=cast("str", r.run_uri),
                    sha256=cast("str", r.run_sha256),
                    size=cast("int", r.run_size),
                    arrays_sha256=cast("str", r.arrays_sha256),
                    sources=cast("tuple[str, ...]", r.sources),
                ),
            )
            for role, r in zip(ROLES, selected, strict=True)
        )
        first = selected[0]
        cases.append(
            ManualFigureCase(
                spec.case_id,
                spec.configuration,
                spec.tracker,
                spec.scenario,
                first.scenario_class,
                ("post-hoc explanatory selection",),
                cast("float", first.activation_s),
                spec.parent,
                teachers[spec.parent],
                runs,
                (),
            )
        )
    return ManualFigureInputs("task_1a_manual_v1", str(TASK), sha256_file(root / TASK), tuple(cases))


def diagnostic_table(rows: tuple[ManualRunRow, ...]) -> str:
    """List deciding failures for M10; one stored reason per failed run, not all violated criteria."""
    counts = Counter((r.configuration, r.tracker, (r.reason or "pass").split(":")[0]) for r in rows if r.arm == "M10")
    keys = ("pass", "dwell", "trigger", "limit_violation", "generated_reference", "saturation")
    if any(key not in keys for _, _, key in counts):
        msg = "unmapped M10 deciding reason; extend the report table rather than drop failures"
        raise ValueError(msg)
    return "\n".join(
        "<tr><th>" + c + " / " + tr + "</th>" + "".join(f"<td>{counts[c, tr, k]}</td>" for k in keys) + "</tr>"
        for c in dict.fromkeys(r.configuration for r in rows)
        for tr in TRACKERS
    )


def binding_paths(root: Path, data: ReportData) -> tuple[list[Path], list[Path]]:
    """Complete declared input/source inventories, shared by generation and verification."""
    docs, source = root / DOCS, root / DOCS / "report_source"
    sources = [
        Path(__file__),
        Path(__file__).with_name("manual_search_report_plots.py"),
        Path(__file__).with_name("manual_figures.py"),
        root / "scripts/render_manual_search_report.py",
        *sorted(source.iterdir()),
    ]
    inputs = [
        root / TEACHERS,
        root / TASK,
        docs / "audit/audit_v2.json",
        docs / "results/results_v1.json",
        docs / "freeze/selection_v1.json",
        docs / "comparison/status_v1.json",
        root / "configs/studies/manual_esn_search_v1.toml",
        root / "configs/evaluations/task_1a_manual_dev_v1.toml",
        *(docs / "results" / d.name for d in data.index.documents),
    ]
    return sources, inputs


def verify_report(directory: Path, *, root: Path | None = None) -> None:
    """Verify full inventories, fingerprints, bindings and generated tables without the store."""
    root = repository_root() if root is None else root
    data = load_data(root)
    manifest = json.loads((directory / "manifest.json").read_text())
    sources, inputs = binding_paths(root, data)
    for group, paths in (("sources", sources), ("inputs", inputs)):
        if set(manifest[group]) != {str(p.relative_to(root)) for p in paths}:
            msg = f"presentation {group} inventory differs"
            raise ValueError(msg)
    if manifest["cases"] != [s.case_id for s in CASE_SPECS]:
        msg = "presentation case inventory differs"
        raise ValueError(msg)
    table = next(t for t in data.index.tables if t.record == "ManualRunRow")
    if manifest["run_table"] != {"uri": table.payload.uri, "sha256": table.payload.sha256}:
        msg = "presentation table binding differs"
        raise ValueError(msg)
    verify_fingerprints(directory, root=root)
    page = (directory / "index.html").read_text()
    for content in (summary_table(data), config_table(data), arm_table(data), replay_table(data)):
        if content not in page:
            msg = "presentation table differs from the audited evidence"
            raise ValueError(msg)


def build_report(output: Path) -> None:
    """Render a new, portable directory; read the evidence store without modifying it."""
    root = repository_root()
    source = root / DOCS / "report_source"
    data = load_data(root)
    if sha256_file(root / TEACHERS) != data.index.inputs.study_manifest_sha256:
        msg = "teacher study fingerprint differs from the audited index"
        raise ValueError(msg)
    store = open_storage()
    table = next(t for t in data.index.tables if t.record == "ManualRunRow")
    rows = table_from_csv(verify_artifact(store, table.payload).read_text(), ManualRunRow)
    figures = case_inputs(rows, root)
    output.mkdir(parents=True, exist_ok=False)
    assets = output / "assets"
    assets.mkdir()
    render_summaries(data, assets)
    cases: list[dict[str, object]] = []
    for spec, case in zip(CASE_SPECS, figures.cases, strict=True):
        print(f"Rendering {spec.slug}", flush=True)
        cases.append(render_case(case, spec, figures, assets, store, rows))
    duplicates = {(r.configuration, r.tracker, r.parent, r.scenario_id): r for r in rows if r.arm_kind == "R10"}
    paired = [r for r in rows if r.arm_kind == "S"]
    disagreements = sum(
        r.success != duplicates[r.configuration, r.tracker, r.parent, r.scenario_id].success for r in paired
    )
    substitutions = {
        "SUMMARY_TABLE": summary_table(data),
        "CONFIG_TABLE": config_table(data),
        "ARM_TABLE": arm_table(data),
        "REPLAY_TABLE": replay_table(data),
        "FAILURES": diagnostic_table(rows),
        "COPY_PAIRS": f"{len(paired):,}",
        "COPY_DISAGREEMENTS": str(disagreements),
        "RC_SUCCESS": f"{data.index.n_rc_successes:,}",
        "RC_RUNS": f"{data.index.n_rc_runs:,}",
        "REPLAY_SUCCESS": f"{data.index.n_replay_successes:,}",
        "REPLAY_RUNS": f"{data.index.n_replay_runs:,}",
        "CASES": "\n".join(case_html(s, c) for s, c in zip(CASE_SPECS, cases, strict=True)),
        "CASE_SCRIPTS": "\n".join(f'<script src="assets/{s.slug}.js"></script>' for s in CASE_SPECS),
    }
    template = (source / "report.html").read_text()
    for marker, value in substitutions.items():
        template = template.replace("{{" + marker + "}}", value)
    if "{{" in template:
        msg = "unexpanded report template field"
        raise ValueError(msg)
    (output / "index.html").write_text(template)
    (output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    for name in ("player.js", "report.css"):
        shutil.copyfile(source / name, assets / name)
    shutil.copyfile(source / "README.md", output / "README.md")
    commit, dirty = worktree_state(root)
    sources, inputs = binding_paths(root, data)
    manifest = {
        "task": "M3MS-008",
        "project_commit": commit,
        "project_dirty": dirty,
        "sources": {str(p.relative_to(root)): sha256_file(p) for p in sources},
        "inputs": {str(p.relative_to(root)): sha256_file(p) for p in inputs},
        "run_table": {"uri": table.payload.uri, "sha256": table.payload.sha256},
        "display": {"stride": 10, "angle_decimals": 6, "events_and_last_sample_retained": True},
        "cases": [s.case_id for s in CASE_SPECS],
        "outputs": {
            str(p.relative_to(output)): {"sha256": sha256_file(p), "size": p.stat().st_size}
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verify_outputs(output, cast("dict[str, dict[str, object]]", manifest["outputs"]))


def main() -> None:
    """Render from stored evidence, or verify an existing presentation without the store."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_report(cast("Path", args.output))
        print("Presentation, sources and committed inputs verified.")
    else:
        build_report(cast("Path", args.output))


if __name__ == "__main__":
    main()
