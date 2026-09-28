# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Reproducible, offline expert presentation of the closed manual-demonstration study.

Read audited artifacts only. No fitting, simulation, database opening, or store writes.
The browser animation is a decimated display of stored states, never a simulation.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PIL.ImageColor import getrgb

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments.manual_evaluation import load_verified_payload
from arm_rc_ctrl.experiments.manual_figures import ROLE_COLOURS, ManualFigureCase, load_figure_inputs, plot_case
from arm_rc_ctrl.experiments.manual_report import DOCUMENT_LOCATION, ReportData, load_report_data, render_summaries
from arm_rc_ctrl.experiments.manual_results import ManualRunRow, load_results, table_from_csv
from arm_rc_ctrl.provenance import sha256_file, verify_artifact, worktree_state
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot, open_storage

SOURCE = DOCUMENT_LOCATION / "expert_report_source"
JOINT_COUNT = 2
ROLES = ("S", "M10", "R10", "C10", "replay")


@dataclass(frozen=True)
class Illustration:
    """A named subset of the already frozen illustrative cases, with an interpretive caption."""

    slug: str
    case_id: str
    title: str
    caption: str


CASES = (
    Illustration(
        "rescue",
        "feasible-best__pd_v2__nominal",
        "A · A weak singleton is rescued",
        "At the nominal start, D01's singleton and copies miss the target tolerance; all-ten and synthetic "
        "training succeed. This is an example of a gain, not the typical paired effect across all ten "
        "parents.",
    ),
    Illustration(
        "nominal",
        "feasible-middle__pd_v2__nominal",
        "B · Nominal success hides augmentation failure",
        "One take and all ten both pass here. The synthetic model does not settle at the target. This "
        "nominal result alone does not reveal the all-ten model's poor robustness to starting offsets.",
    ),
    Illustration(
        "offset",
        "feasible-middle__pd_v2__posture-small-20261201-01",
        "B · A small posture offset reverses the comparison",
        "The singleton succeeds, all-ten misses by a large distance, and the synthetic run stops at a "
        "joint-speed limit. Compare this with the same configuration's nominal case immediately above.",
    ),
    Illustration(
        "ceiling",
        "feasible-worst__pd_v2__nominal",
        "C · All five methods succeed",
        "All five approaches reach and hold in this nominal case. More broadly, every learned model at C/PD "
        "succeeds in all 65 cases: equal scores here are a ceiling, not proof of equal trajectories.",
    ),
    Illustration(
        "force",
        "feasible-worst__computed_torque__force-12N-000deg",
        "C · Disturbance recovery misses a strict tolerance",
        "With computed torque and a positive-x force pulse, the singleton, copies, synthetic model and "
        "replay recover. All-ten finishes just outside the target radius. Each arm receives its pulse after "
        "its own measured dwell trigger, so pulse times differ.",
    ),
    Illustration(
        "abort",
        "failure-actual-dwell__pd_v2__nominal",
        "D · Learned runs stop at a speed limit",
        "All four learned arms stop early, whereas replay completes. The historical label "
        "'failure-actual-dwell' is not the reason these runs fail. The last stored state can be below the "
        "abort threshold: the violating next step is rejected rather than retained as a normal sample.",
    ),
    Illustration(
        "force_abort",
        "failure-joint-velocity__computed_torque__force-12N-000deg",
        "E · Reach first, then fail under force",
        "The learned arms reach sufficiently well to trigger the pulse, but then encounter joint-speed "
        "aborts. Replay succeeds. This distinguishes recovery failure from never reaching the trigger "
        "condition.",
    ),
    Illustration(
        "slow",
        "failure-generated-dwell__pd_v2__force-12N-000deg",
        "F · Late recovery is still a valid success",
        "The singleton and copies eventually recover and hold, while all-ten and synthetic do not satisfy "
        "the final dwell criterion. A slower transient is not itself a failure: the protocol judges the "
        "final continuous hold within the common horizon.",
    ),
)


def frame_indices(t: NDArray[np.float64], *, stride: int, events: tuple[float, ...] = ()) -> NDArray[np.int64]:
    """Keep native samples, endpoints and the nearest sample at each displayed event; never extrapolate."""
    if stride < 1 or len(t) == 0:
        msg = "stride must be positive and the recorded time axis nonempty"
        raise ValueError(msg)
    indices = {0, len(t) - 1, *range(0, len(t), stride)}
    indices.update(int(np.argmin(np.abs(t - event))) for event in events if t[0] <= event <= t[-1])
    return np.asarray(sorted(indices), dtype=np.int64)


def _signed(number: float) -> str:
    return f"{number:+g}".replace("-", "\u2212") if number else "0"


def summary_table(data: ReportData) -> str:
    """The primary paired table, generated from audited summaries rather than transcribed."""
    arms = {(r.configuration, r.tracker, r.scenario_class, r.arm_kind): r for r in data.arms}
    contrasts = {(r.configuration, r.tracker, r.scenario_class, r.contrast): r for r in data.contrasts}
    rows: list[str] = []
    for i, config in enumerate(data.study.configurations):
        for tracker, name in (("pd_v2", "PD"), ("computed_torque", "Computed torque")):
            single = arms[config.label, tracker, "all", "S"]
            multi = arms[config.label, tracker, "all", "M10"]
            diff = contrasts[config.label, tracker, "all", "M10-S"]
            assert diff.median is not None
            assert diff.minimum is not None
            assert diff.maximum is not None
            rows.append(
                f'<tr data-result="{config.label}/{tracker}"><th>{chr(65 + i)} / {name}</th>'
                f"<td>{single.median:g} [{single.minimum}\u2013{single.maximum}]</td><td>{multi.successes}</td>"
                f"<td>{_signed(diff.median)} [{_signed(diff.minimum)} … {_signed(diff.maximum)}]</td>"
                f"<td>{diff.parents_improved} / {diff.parents_worsened} / {diff.parents_tied}</td></tr>"
            )
    return "\n".join(rows)


def configuration_table(data: ReportData) -> str:
    """Expose every inherited configuration, including its seed and derivative policy."""
    rows: list[str] = []
    for i, c in enumerate(data.study.configurations):
        r = c.reservoir
        values = (
            chr(65 + i),
            c.label,
            str(r.n_neurons),
            f"{r.spectral_radius:.5g}",
            f"{r.sparsity:.5g}",
            f"{r.leak_rate:.5g}",
            f"{r.input_scaling:.5g}",
            f"{c.base_alpha:.5g}",
            str(r.seed),
            f"{c.warmup_s:g}",
            f"{c.velocity_cutoff_hz:.5g} / {c.acceleration_cutoff_hz:.5g}",
        )
        rows.append("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + "</tr>")
    return "\n".join(rows)


def _points(q: NDArray[np.float64], lengths: NDArray[np.float64]) -> NDArray[np.float64]:
    angles = np.cumsum(q, axis=1)
    return np.cumsum(np.stack((np.cos(angles), np.sin(angles)), axis=2) * lengths[None, :, None], axis=1)


def _numbers(values: NDArray[np.float64]) -> object:
    """Six-decimal presentation coordinates; invalid values stay absent, never repaired."""
    rounded = np.round(values, 6).astype(object)
    rounded[~np.isfinite(values)] = None
    return rounded.tolist()


def _case_rows(rows: tuple[ManualRunRow, ...], case: ManualFigureCase) -> dict[str, ManualRunRow]:
    by_role: dict[str, ManualRunRow] = {}
    for role in ROLES:
        arm = "M10" if role == "M10" else f"{role}/{case.teacher_assignment}"
        found = [
            r
            for r in rows
            if (r.configuration, r.tracker, r.scenario_id, r.arm)
            == (case.configuration, case.tracker, case.scenario_id, arm)
        ]
        if len(found) != 1:
            msg = f"{case.case_id}/{role}: expected exactly one audited run row"
            raise ValueError(msg)
        by_role[role] = found[0]
    return by_role


def save_indexed_plot(path: Path) -> None:
    """Compress scientific plots without losing sparsely used legend colours."""
    with Image.open(path) as pixels:
        rgb = pixels.convert("RGB")
        palette = rgb.quantize(colors=248, method=Image.Quantize.MEDIANCUT)
        colours = palette.getpalette()
        assert colours is not None
        colours = list(colours[: 248 * 3])
        for colour in (*ROLE_COLOURS.values(), "#000000", "#ffffff", "#dddddd"):
            colours.extend(getrgb(colour))
        palette.putpalette(colours)
        rgb.quantize(palette=palette, dither=Image.Dither.NONE).save(path, optimize=True)


def render_case(
    case: ManualFigureCase,
    spec: Illustration,
    docs: Path,
    out: Path,
    store: StorageRoot,
    rows: tuple[ManualRunRow, ...],
) -> dict[str, object]:
    """Export full-resolution scientific plots and a compact synchronized browser playback."""
    root = repository_root()
    inputs = load_figure_inputs(docs / "results/figure_inputs_v1.json")
    scenario_file = root / inputs.scenario_file
    if sha256_file(scenario_file) != inputs.scenario_sha256:
        msg = "the figure task configuration differs from its bound digest"
        raise ValueError(msg)
    scenario = load_manual_scenario(scenario_file)
    lengths = np.asarray([link.length for link in scenario.robot.links], dtype=np.float64)
    if len(lengths) != JOINT_COUNT or scenario.robot.base_length != 0:
        msg = "this presentation requires the bound two-link, zero-base planar robot"
        raise ValueError(msg)
    target = np.asarray(scenario.task.target, dtype=np.float64)
    table = _case_rows(rows, case)
    paths_fig, paths_axes = cast("tuple[Any, Any]", plt.subplots(1, 5, figsize=(15, 3.8), layout="constrained"))
    time_fig, axes = cast("tuple[Any, Any]", plt.subplots(4, 1, figsize=(12, 9), sharex=True, layout="constrained"))
    playback: list[dict[str, object]] = []
    for i, run in enumerate(case.runs):
        summary, arrays = load_verified_payload(store, run.run, where=case.case_id)
        row = table[run.role]
        if (row.run_sha256, row.arrays_sha256, row.success) != (run.run.sha256, run.run.arrays_sha256, run.success):
            msg = f"{case.case_id}/{run.role}: table and figure bindings disagree"
            raise ValueError(msg)
        q, qr = np.asarray(arrays["q"], dtype=np.float64), np.asarray(arrays["q_desired"], dtype=np.float64)
        t = np.asarray(arrays["t"], dtype=np.float64) - case.activation_s
        tip = np.asarray(arrays["tip"], dtype=np.float64)
        np.testing.assert_allclose(_points(q, lengths)[:, -1], tip, atol=1e-10, rtol=0)
        reference = _points(qr, lengths)[:, -1]
        color = ROLE_COLOURS[run.role]
        axis = paths_axes[i]
        axis.plot(reference[:, 0], reference[:, 1], "--", color=color, alpha=0.6, lw=0.9)
        axis.plot(tip[:, 0], tip[:, 1], color=color, lw=1.2)
        axis.plot(tip[0, 0], tip[0, 1], "o", color=color, ms=4)
        axis.plot(tip[-1, 0], tip[-1, 1], "x", color=color, ms=5)
        axis.add_patch(Circle((float(target[0]), float(target[1])), scenario.task.tolerance, fill=False, color="black"))
        axis.plot(*target, "+", color="black", ms=5)
        axis.set(
            xlim=(-0.57, 0.57),
            ylim=(-0.57, 0.57),
            xlabel="x (m)",
            ylabel="y (m)",
            title=f"{run.role}: {'pass' if run.success else 'fail'}",
        )
        axis.set_aspect("equal")
        dq = np.asarray(arrays["dq"], dtype=np.float64)
        dqr = np.asarray(arrays["dq_desired"], dtype=np.float64)
        requested = np.asarray(arrays["tau_requested"], dtype=np.float64)
        applied = np.asarray(arrays.get("tau_applied", requested), dtype=np.float64)
        torque_limits = np.asarray(scenario.limits.torque, dtype=np.float64)
        actual_error = np.linalg.norm(tip - target, axis=1) * 1000
        reference_error = np.linalg.norm(reference - target, axis=1) * 1000
        signals = (
            actual_error,
            np.max(np.abs(dq), axis=1),
            np.linalg.norm(q - qr, axis=1),
            np.max(np.abs(applied) / torque_limits, axis=1),
        )
        refs = (reference_error, np.max(np.abs(dqr), axis=1), None, np.max(np.abs(requested) / torque_limits, axis=1))
        for ax, signal, ref in zip(axes, signals, refs, strict=True):
            ax.plot(t, signal, color=color, label=run.role, lw=1.3)
            if ref is not None:
                ax.plot(t, ref, "--", color=color, alpha=0.55, lw=0.8)
            if run.pulse_start_s is not None and run.pulse_end_s is not None:
                ax.axvspan(
                    run.pulse_start_s - case.activation_s, run.pulse_end_s - case.activation_s, color=color, alpha=0.12
                )
        events = (0.0, *(v - case.activation_s for v in (run.pulse_start_s, run.pulse_end_s) if v is not None))
        ix = frame_indices(t, stride=10, events=events)
        playback.append(
            {
                "role": run.role,
                "color": color,
                "success": run.success,
                "reason": row.reason,
                "end": float(t[-1]),
                "t": _numbers(t[ix]),
                "q": _numbers(q[ix]),
                "ref": _numbers(qr[ix]),
                "pulse": list(events[1:]),
                "final_error_mm": row.final_endpoint_error_m * 1000 if row.final_endpoint_error_m is not None else None,
                "dwell_task_s": row.time_to_final_dwell_s,
                "final_dwell_s": row.dwell_final_s,
                "sha256": run.run.sha256,
                "arrays_sha256": run.run.arrays_sha256,
                "uri": run.run.uri,
                "samples": len(t),
                "termination": str(summary.termination.kind),
            }
        )
    paths_fig.suptitle(
        f"{spec.title}\nActual solid · commanded dashed · start circle · final stored point \u00d7 · target ring 10 mm",
        fontsize=11,
    )
    paths_fig.savefig(out / f"{spec.slug}-space.png", dpi=100, pil_kwargs={"optimize": True})
    plt.close(paths_fig)
    axes[0].set_yscale("log")
    axes[0].axhline(10, color="black", ls=":", lw=1)
    axes[0].set(ylabel="Target error (mm)", ylim=(0.1, 2000))
    axes[1].axhline(6, color="black", ls=":", lw=1)
    axes[1].axhline(0.05, color="black", ls=":", lw=0.7)
    axes[1].set_yscale("symlog", linthresh=0.05)
    axes[1].set(ylabel="Max joint speed\n(rad/s)", ylim=(0, None))
    axes[2].set(ylabel="Joint tracking error\nEuclidean norm (rad)")
    axes[3].axhline(1, color="black", ls=":", lw=1)
    axes[3].set_yscale("symlog", linthresh=0.01)
    axes[3].set(ylabel="Max |torque| / limit", ylim=(0, None), xlabel="Task time (s); zero = activation")
    for ax in axes:
        ax.set_xlim(-case.activation_s, 30)
        ax.axvspan(-case.activation_s, 0, color="#ddd", alpha=0.6)
        ax.grid(alpha=0.15)
    axes[0].legend(ncol=5, loc="upper right")
    time_fig.suptitle(
        f"{spec.title}\nSolid: measured / applied. Dashed: commanded / requested. "
        "Colored bands: each run's force pulse.",
        fontsize=11,
    )
    time_fig.savefig(out / f"{spec.slug}-time.png", dpi=100, pil_kwargs={"optimize": True})
    plt.close(time_fig)
    plot_case(inputs, case.case_id, out / f"{spec.slug}-joints.png", store=store, root=root)
    # Deterministic indexed PNGs keep curated figures below the repository's 200 KiB limit.
    for suffix in ("space", "time", "joints"):
        save_indexed_plot(out / f"{spec.slug}-{suffix}.png")
    payload = {
        "slug": spec.slug,
        "title": spec.title,
        "lengths": lengths.tolist(),
        "target": target.tolist(),
        "radius": scenario.task.tolerance,
        "activation": case.activation_s,
        "runs": playback,
    }
    (out / f"{spec.slug}.js").write_text(
        "window.manualCases.push(" + json.dumps(payload, separators=(",", ":"), allow_nan=False) + ");\n"
    )
    return {
        "slug": spec.slug,
        "case_id": case.case_id,
        "title": spec.title,
        "caption": spec.caption,
        "categories": list(case.categories),
        "configuration": case.configuration,
        "tracker": case.tracker,
        "runs": [{k: v for k, v in run.items() if k not in {"t", "q", "ref", "color"}} for run in playback],
    }


def _case_html(spec: Illustration, info: dict[str, object]) -> str:
    runs = cast("list[dict[str, Any]]", info["runs"])
    metric_rows: list[str] = []
    for run in runs:
        verdict = "Pass" if run["success"] else "Fail"
        error = run["final_error_mm"]
        error_text = "—" if error is None else f"{error:.2f}"
        dwell = run["dwell_task_s"]
        dwell_text = "—" if dwell is None else f"{dwell:.2f}"
        metric_rows.append(
            f"<tr><th>{run['role']}</th><td>{verdict}</td><td>{error_text}</td><td>{dwell_text}</td>"
            f"<td>{html.escape(str(run['reason'] or 'All criteria met'))}</td></tr>"
        )
    return f"""<article class="case" id="case-{spec.slug}"><p class="eyebrow">Matched case · D01 parent</p>
<h3>{html.escape(spec.title)}</h3><p>{html.escape(spec.caption)}</p>
<p class="metadata">{html.escape(spec.case_id)} · Frozen categories: {
        html.escape(", ".join(cast("list[str]", info["categories"])))
    }</p>
<div class="player" data-case="{spec.slug}"><div class="controls"><button type="button" class="play">Play</button>
<button type="button" class="reset">Reset</button><label>Speed <select class="speed"><option
value="1">1\u00d7</option><option value="2" selected>2\u00d7</option><option
value="4">4\u00d7</option></select></label>
<label class="scrub">Task time <input class="timeline" type="range" min="0" max="30" step="0.01"
value="0"></label><output class="clock">0.00 s</output></div>
<div class="canvases"></div><p class="hint">Five synchronized stored trajectories. Solid arm: measured; dashed arm:
commanded. A grey pose after a trace ends is only its last stored state. No continuation is simulated.</p></div>
<figure><a href="assets/{spec.slug}-space.png"><img loading="lazy" src="assets/{spec.slug}-space.png" alt="Five
arm-specific task-space paths for {html.escape(spec.title)}"></a><figcaption>Same task-space scale in every panel.
Click a plot for its full-size image. Differences smaller than the target ring are clearer in the error time
series.</figcaption></figure>
<figure><a href="assets/{spec.slug}-time.png"><img loading="lazy" src="assets/{spec.slug}-time.png" alt="Endpoint
error, joint speeds, tracking error and torque over time"></a><figcaption>Full stored sampling, including warm-up
at negative time. Reference/error and speed curves may differ. Torque is normalized per joint before taking the
maximum; this is not torque RMS. Curves end when their recorded run ends.</figcaption></figure>
<details><summary>Joint angles and combined endpoint plot</summary><img loading="lazy"
src="assets/{spec.slug}-joints.png" alt="Joint 1 and joint 2 histories, endpoint path and distance, including the
recorded teacher"><p>The dotted teacher is D01 on its recording clock. All-ten was trained on all ten teachers; D01
is a comparison reference, not its unique teacher.</p></details>
<div class="table-scroll"><table><caption>Audited diagnostic values; error is at the last stored sample, including
early aborts.</caption><thead><tr><th>Arm</th><th>Verdict</th><th>Final error (mm)</th><th>Final hold starts (task
s)</th><th>Deciding reason</th></tr></thead><tbody>{"".join(metric_rows)}</tbody></table></div></article>"""


def verify_outputs(directory: Path, outputs: Mapping[str, Mapping[str, object]]) -> None:
    """Check the complete presentation inventory and each file's bytes, excluding its own manifest."""
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if actual != set(outputs):
        msg = "presentation output inventory differs from its manifest"
        raise ValueError(msg)
    for name, record in outputs.items():
        path = directory / name
        if sha256_file(path) != record["sha256"] or path.stat().st_size != record["size"]:
            msg = f"{name}: presentation fingerprint differs"
            raise ValueError(msg)


def verify_report(directory: Path, *, root: Path | None = None) -> None:
    """Verify presentation, renderer and committed input bytes without needing the store."""
    root = repository_root() if root is None else root
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for group in ("sources", "inputs"):
        for name, digest in manifest[group].items():
            if sha256_file(root / name) != digest:
                msg = f"{name}: presentation input or source fingerprint differs"
                raise ValueError(msg)
    verify_outputs(directory, manifest["outputs"])


def build_report(output: Path) -> None:
    """Create a new portable HTML report and record input, source and output fingerprints."""
    root = repository_root()
    docs, source = root / DOCUMENT_LOCATION, root / SOURCE
    data = load_report_data(docs)
    results = load_results(docs / "results/results_v1.json")
    store = open_storage()
    table = next(t for t in results.tables if t.record == "ManualRunRow")
    table_path = verify_artifact(store, table.payload)
    rows = table_from_csv(table_path.read_text(encoding="utf-8"), ManualRunRow)
    figures = load_figure_inputs(docs / "results/figure_inputs_v1.json")
    output.mkdir(parents=True, exist_ok=False)
    assets = output / "assets"
    assets.mkdir()
    render_summaries(data, assets)
    # The existing teacher figure's source/output inventory is already regression locked.
    old_assets = docs / "report/assets"
    old_manifest = json.loads((old_assets / "render_manifest.json").read_text())
    teacher_binding = old_manifest["outputs"]["demonstrations.png"]
    if sha256_file(old_assets / "demonstrations.png") != teacher_binding["sha256"]:
        msg = "the existing teacher figure differs from its presentation binding"
        raise ValueError(msg)
    shutil.copyfile(old_assets / "demonstrations.png", assets / "demonstrations.png")
    cases: list[dict[str, object]] = []
    for spec in CASES:
        print(f"Rendering {spec.slug}", flush=True)
        case = next(c for c in figures.cases if c.case_id == spec.case_id)
        cases.append(render_case(case, spec, docs, assets, store, rows))
    template = (source / "report.html").read_text(encoding="utf-8")
    for marker, content in {
        "SUMMARY_TABLE": summary_table(data),
        "CONFIG_TABLE": configuration_table(data),
        "CASES": "\n".join(_case_html(s, c) for s, c in zip(CASES, cases, strict=True)),
        "CASE_SCRIPTS": "\n".join(f'<script src="assets/{s.slug}.js"></script>' for s in CASES),
    }.items():
        template = template.replace("{{" + marker + "}}", content)
    if "{{" in template:
        msg = "unexpanded report template field"
        raise ValueError(msg)
    (output / "index.html").write_text(template, encoding="utf-8")
    for name in ("report.css", "player.js"):
        shutil.copyfile(source / name, assets / name)
    (output / "cases.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n")
    shutil.copyfile(source / "README.md", output / "README.md")
    commit, dirty = worktree_state(root)
    sources = [Path(__file__), root / "scripts/render_manual_expert_report.py", *sorted(source.iterdir())]
    inputs = [
        docs / "audit/reproduction_audit_v4.json",
        docs / "study_manifest_v1.json",
        docs / "results/results_v1.json",
        docs / "results/arm_summary_v1.csv",
        docs / "results/contrast_summary_v1.csv",
        docs / "results/figure_inputs_v1.json",
        root / figures.scenario_file,
        old_assets / "render_manifest.json",
        old_assets / "demonstrations.png",
    ]
    sources += [root / "src/arm_rc_ctrl/experiments" / name for name in ("manual_report.py", "manual_figures.py")]
    manifest = {
        "task": "M3MAN-015",
        "project_commit": commit,
        "project_dirty": dirty,
        "sources": {str(p.relative_to(root)): sha256_file(p) for p in sources},
        "inputs": {str(p.relative_to(root)): sha256_file(p) for p in inputs},
        "run_table": {"uri": table.payload.uri, "sha256": table.payload.sha256},
        "display": {"stride": 10, "angle_decimals": 6, "events_and_last_sample_retained": True},
        "outputs": {
            str(p.relative_to(output)): {"sha256": sha256_file(p), "size": p.stat().st_size}
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verify_outputs(output, cast("dict[str, dict[str, object]]", manifest["outputs"]))


def main() -> None:
    """Render a new report directory from the configured, read-only evidence store."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true", help="Verify an existing report without reading the store")
    args = parser.parse_args()
    if args.verify:
        verify_report(cast("Path", args.output))
        print("Presentation, sources and committed inputs verified.")
    else:
        build_report(cast("Path", args.output))


if __name__ == "__main__":
    main()
