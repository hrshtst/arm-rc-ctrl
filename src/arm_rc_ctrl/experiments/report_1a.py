# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Task 1-a results report: tables, plots, paired comparisons, failures, and limitations (``docs/PLAN.md`` 9; M3-011).

The report is rendered from the committed evidence only (the JSON reports
under ``docs/experiments/task_1a``), so it can be regenerated without the
external store and locked by a regression test. The locked confirmatory
suite is the primary evidence; the development suites, the ESN searches,
the seed panel, and the training reports are development evidence and are
labelled as such. Plots are written next to the report; their pixels are
not part of the lock.

Command line::

    python -m arm_rc_ctrl.experiments.report_1a --docs docs/experiments/task_1a
        --output docs/experiments/task_1a/report.md --plots docs/experiments/task_1a/plots [--force]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from arm_rc_ctrl.experiments.esn_stability import StabilityReport, load_stability
from arm_rc_ctrl.experiments.esn_study import EsnStudyReport, load_report
from arm_rc_ctrl.experiments.paired import PairedReport, compare_reports, load_paired_report
from arm_rc_ctrl.experiments.perturbations import CLASS_ORDER
from arm_rc_ctrl.experiments.robustness import ArmRun, RobustnessSuite, load_suite

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from arm_rc_ctrl.metrics.report import RunReport

__all__ = ["ReportInputs", "load_inputs", "main", "render_report", "write_plots"]

CONFIRMATORY_LABEL: Final = "confirmatory"
PLOT_FILES: Final = ("rmse_by_class.png", "paired_differences.png", "search_objectives.png")
_STAT_HEADER: Final = ("arm", "class", "successes", "median", "q25", "q75", "min", "max")
_LIMITATIONS: Final = (
    (
        "Simulation only: all results come from the `skelarm` planar two-link model at 100 Hz with the frozen "
        "tracker gains; no hardware, sensor noise, latency, or model mismatch beyond the configured perturbations "
        "is represented."
    ),
    (
        "One demonstration, one task: the recipe was trained on a single scripted demonstration of task 1-a and "
        "evaluated against it; generalization to other targets, speeds, or postures beyond the perturbation classes "
        "is untested."
    ),
    (
        "The RC generator tracks the demonstration less precisely than the direct replay in every posture class "
        "(median RC minus replay joint RMSE up to about 0.008 rad under PD v2); under the 12 N pulses the difference "
        "vanishes because the pulse dominates both, so the force classes do not separate the methods."
    ),
    (
        "Computed torque absorbs the confirmatory pulse worse than PD v2 (about 0.07 rad for RC and replay alike); "
        "it is a secondary comparison and the ESN objective was tuned with PD v2 only."
    ),
    (
        "Feasibility in tuning is defined by the scenario's dwell criteria and the saturation bound; the selected "
        "recipe sits on several bounds of the v2 search space (spectral radius, sparsity, velocity cutoff high; input "
        "scaling low), which indicates optimization headroom rather than a limitation of the candidate."
    ),
    (
        "Development and confirmatory perturbations differ in levels, timing, directions, and seeds by design; the "
        "confirmatory suite was run exactly once for this study version, so its estimates carry no repeat-run variance."
    ),
    "Reservoir-seed sensitivity was probed with a fixed eight-seed panel on the three leading trials only.",
    (
        "The experiment state that is not in Git (run payloads, models, MLflow and Optuna databases) lives in the "
        "external storage root; the committed pointer records and digests make it verifiable but not self-contained."
    ),
)


@dataclass(frozen=True)
class ReportInputs:
    """The committed evidence the report is rendered from."""

    confirmatory: RobustnessSuite
    confirmatory_file: str
    development: tuple[tuple[str, RobustnessSuite], ...]
    searches: tuple[tuple[str, EsnStudyReport], ...]
    stability: tuple[tuple[str, StabilityReport], ...]
    training: tuple[tuple[str, dict[str, object]], ...]
    nominal: tuple[tuple[str, PairedReport], ...] = ()


def load_inputs(docs: Path) -> ReportInputs:
    """Discover the evidence under ``docs`` (exactly one confirmatory suite)."""
    nominal = [(f.name, load_paired_report(f)) for f in sorted(docs.glob("paired_nominal_*.json"))]
    suites = [(f.name, load_suite(f)) for f in sorted(docs.glob("robustness_*.json"))]
    confirmatory = [(name, s) for name, s in suites if s.label == CONFIRMATORY_LABEL]
    if len(confirmatory) != 1:
        msg = f"expected exactly one confirmatory suite under {docs}, found {len(confirmatory)}"
        raise ValueError(msg)
    searches = [(f.name, load_report(f)) for f in sorted(docs.glob("esn_search*.json"))]
    stability = [(f.name, load_stability(f)) for f in sorted(docs.glob("esn_stability*.json"))]
    training = [
        (f.name, cast("dict[str, object]", json.loads(f.read_text(encoding="utf-8"))))
        for f in sorted(docs.glob("training_*.json"))
    ]
    return ReportInputs(
        confirmatory=confirmatory[0][1],
        confirmatory_file=confirmatory[0][0],
        development=tuple((name, s) for name, s in suites if s.label != CONFIRMATORY_LABEL),
        searches=tuple(searches),
        stability=tuple(stability),
        training=tuple(training),
        nominal=tuple(nominal),
    )


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}g}"


def _row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    return [_row(header), _row(["---"] * len(header)), *(_row(r) for r in rows)]


def _stats(values: Sequence[float]) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """(median, q25, q75, min, max); quartiles need at least two values."""
    if not values:
        return (None, None, None, None, None)
    if len(values) < 2:  # noqa: PLR2004
        return (values[0], None, None, values[0], values[0])
    q = statistics.quantiles(values, n=4, method="inclusive")
    return (statistics.median(values), q[0], q[2], min(values), max(values))


def _groups(suite: RobustnessSuite) -> list[tuple[str, str, list[ArmRun]]]:
    out: list[tuple[str, str, list[ArmRun]]] = []
    for arm in suite.arms:
        for kind in CLASS_ORDER:
            runs = [r for r in suite.runs if r.arm == arm.name and r.kind == kind]
            if runs:
                out.append((arm.name, kind, runs))
    return out


def _stat_rows(suite: RobustnessSuite, pick: Callable[[RunReport], float | None]) -> list[list[object]]:
    rows: list[list[object]] = []
    for arm, kind, runs in _groups(suite):
        values = [v for v in (pick(r.report) for r in runs if r.report.success) if v is not None]
        median, q25, q75, low, high = _stats(values)
        rows.append([arm, kind, len(values), _fmt(median), _fmt(q25), _fmt(q75), _fmt(low), _fmt(high)])
    return rows


def _primary(suite: RobustnessSuite) -> list[str]:
    lines = ["### Joint trajectory RMSE over the movement window (rad)", ""]
    lines.extend(
        _table(_STAT_HEADER, _stat_rows(suite, lambda r: None if r.joint_rmse is None else r.joint_rmse.aggregate))
    )
    dof = max((len(r.report.joint_rmse.per_joint) for r in suite.runs if r.report.joint_rmse is not None), default=0)
    lines.extend(["", "Per-joint RMSE medians over successful runs (rad):", ""])
    header = ["arm", "class", *[f"joint {j}" for j in range(dof)]]
    rows: list[list[object]] = []
    for arm, kind, runs in _groups(suite):
        per_joint = [
            r.report.joint_rmse.per_joint for r in runs if r.report.success and r.report.joint_rmse is not None
        ]
        cells: list[object] = [arm, kind]
        for j in range(dof):
            column = [p[j] for p in per_joint if len(p) > j]
            cells.append(_fmt(statistics.median(column)) if column else "n/a")
        rows.append(cells)
    lines.extend(_table(header, rows))
    return lines


def _median_of(values: Sequence[float]) -> str:
    return _fmt(statistics.median(values)) if values else "n/a"


def _secondary(suite: RobustnessSuite) -> list[str]:
    lines = ["### Dwell-window metrics (medians over successful runs)", ""]
    header = [
        "arm", "class", "endpoint mean (m)", "endpoint RMS (m)", "endpoint max (m)", "endpoint p95 (m)",
        "in-tolerance fraction", "longest in-tolerance (s)", "velocity RMS (rad/s)", "velocity max (rad/s)",
    ]  # fmt: skip
    rows: list[list[object]] = []
    for arm, kind, runs in _groups(suite):
        dwell = [r.report.dwell for r in runs if r.report.success and r.report.dwell is not None]
        rows.append(
            [
                arm, kind,
                _median_of([d.endpoint.mean for d in dwell]), _median_of([d.endpoint.rms for d in dwell]),
                _median_of([d.endpoint.max for d in dwell]), _median_of([d.endpoint.p95 for d in dwell]),
                _median_of([d.in_tolerance_fraction for d in dwell]),
                _median_of([d.longest_in_tolerance_s for d in dwell]),
                _median_of([d.velocity_rms for d in dwell]), _median_of([d.velocity_max for d in dwell]),
            ]
        )  # fmt: skip
    lines.extend(_table(header, rows))
    lines.extend(["", "### Effort over the whole run (medians over successful runs, applied torque)", ""])
    header = ["arm", "class", "torque RMS (N*m)", "torque peak (N*m)", "saturation fraction", "effort (N^2*m^2*s)"]
    rows = []
    for arm, kind, runs in _groups(suite):
        effort = [r.report.effort for r in runs if r.report.success and r.report.effort is not None]
        rows.append(
            [
                arm, kind,
                _median_of([e.torque_rms for e in effort]), _median_of([e.torque_peak for e in effort]),
                _median_of([e.saturation_fraction for e in effort]), _median_of([e.effort for e in effort]),
            ]
        )  # fmt: skip
    lines.extend(_table(header, rows))
    return lines


def _paired_differences(suite: RobustnessSuite) -> list[tuple[str, str, list[float]]]:
    by_key = {(r.arm, r.scenario_id): r for r in suite.runs}
    out: list[tuple[str, str, list[float]]] = []
    for rc_arm in (a for a in suite.arms if a.generator == "rc"):
        replay = next((a for a in suite.arms if a.generator == "replay" and a.tracker == rc_arm.tracker), None)
        if replay is None:
            continue
        for kind in CLASS_ORDER:
            diffs: list[float] = []
            for s in (s for s in suite.scenarios if s.kind == kind):
                rc, rp = by_key.get((rc_arm.name, s.scenario_id)), by_key.get((replay.name, s.scenario_id))
                if rc is None or rp is None or not (rc.report.success and rp.report.success):
                    continue
                c = next(c for c in compare_reports(rc.report, rp.report) if c.name == "joint_rmse")
                if c.rc is not None and c.replay is not None:
                    diffs.append(c.rc - c.replay)
            if diffs:
                out.append((rc_arm.tracker, kind, diffs))
    return out


def _paired(suite: RobustnessSuite) -> list[str]:
    lines = ["### Paired comparisons (RC minus replay, same tracker and scenario)", ""]
    header = [
        "tracker",
        "class",
        "metric",
        "pairs",
        "both succeeded",
        "median RC",
        "median replay",
        "median difference",
    ]
    rows: list[list[object]] = []
    for e in suite.effects:
        unit = f" {e.unit}" if e.unit else ""
        rows.append(
            [
                e.tracker, e.kind, e.metric, e.n_pairs, e.n_both_success,
                f"{_fmt(e.median_rc)}{unit}", f"{_fmt(e.median_replay)}{unit}", f"{_fmt(e.median_difference)}{unit}",
            ]
        )  # fmt: skip
    lines.extend(_table(header, rows))
    lines.extend(["", "Distribution of the per-scenario joint RMSE difference (rad):", ""])
    rows = []
    for tracker, kind, diffs in _paired_differences(suite):
        median, q25, q75, low, high = _stats(diffs)
        rows.append([tracker, kind, len(diffs), _fmt(median), _fmt(q25), _fmt(q75), _fmt(low), _fmt(high)])
    lines.extend(_table(["tracker", "class", "pairs", "median", "q25", "q75", "min", "max"], rows))
    return lines


def _failures(inputs: ReportInputs) -> list[str]:
    lines = ["## Failures", ""]
    rows: list[list[object]] = [
        [name, suite.label, r.arm, r.scenario_id, r.report.termination_kind, ", ".join(r.report.failed_criteria) or "-"]
        for name, suite in ((inputs.confirmatory_file, inputs.confirmatory), *inputs.development)
        for r in suite.runs
        if not r.report.success
    ]
    failed = sum(1 for r in inputs.confirmatory.runs if not r.report.success)
    lines.extend([f"Confirmatory suite: {failed} failed run(s) of {len(inputs.confirmatory.runs)}.", ""])
    if rows:
        lines.extend(_table(["report", "label", "arm", "scenario", "termination", "failed criteria"], rows))
    else:
        lines.append("No failed run in any committed suite.")
    return lines


def _development(inputs: ReportInputs) -> list[str]:
    lines = ["## Development evidence (not confirmatory)", "", "### ESN searches", ""]
    rows: list[list[object]] = [
        [
            name,
            s.protocol,
            s.budget,
            len(s.summary.trials),
            s.summary.n_pruned,
            s.n_feasible,
            "-" if s.summary.best_number is None else s.summary.best_number,
            _fmt(s.summary.best_value),
        ]
        for name, s in inputs.searches
    ]
    header = ["report", "protocol", "budget", "stored", "pruned", "feasible", "best trial", "best objective (rad)"]
    lines.extend(_table(header, rows))
    for name, panel in inputs.stability:
        lines.extend(["", f"### Reservoir-seed panel `{name}`", ""])
        rows = [
            [
                c.trial,
                _fmt(c.own_objective),
                f"{c.feasible_seeds}/{len(c.outcomes)}",
                _fmt(c.objective_median),
                _fmt(c.objective_min),
                _fmt(c.objective_max),
            ]
            for c in panel.configurations
        ]
        header = ["trial", "own objective (rad)", "feasible seeds", "panel median", "panel min", "panel max"]
        lines.extend(_table(header, rows))
    lines.extend(["", "### Training reports", ""])
    rows = []
    for name, t in inputs.training:
        fit = t.get("fit")
        rmse = float(cast("dict[str, float]", fit)["rmse"]) if isinstance(fit, dict) else None
        rows.append(
            [
                name,
                str(t.get("model_config", "")),
                str(t.get("recipe_id", "")),
                _fmt(rmse),
                str(t.get("refit_verified", "")),
            ]
        )
    lines.extend(_table(["report", "model config", "recipe", "fit RMSE (rad)", "refit verified"], rows))
    lines.extend(["", "### Development robustness suites", ""])
    rows = [
        [name, suite.recipe, a.arm, a.kind, f"{a.successes}/{a.n}", _fmt(a.joint_rmse_median), _fmt(a.joint_rmse_max)]
        for name, suite in inputs.development
        for a in suite.aggregates
        if a.arm.startswith("rc")
    ]
    lines.extend(_table(["report", "recipe", "arm", "class", "successes", "RMSE median (rad)", "RMSE max (rad)"], rows))
    return lines


def _playback(inputs: ReportInputs) -> list[str]:
    lines = [
        "Any curated run can be inspected kinematically with the pinned `skelarm` player",
        "(`docs/PLAN.md` section 7.5); the exported log is a local, disposable product.",
        "The nominal paired runs:",
        "",
    ]
    rows: list[list[object]] = [
        [name, report.tracker, report.rc.run_id, report.replay.run_id] for name, report in inputs.nominal
    ]
    lines.extend(_table(["report", "tracker", "RC run", "replay run"], rows))
    rc_pd = next((r.rc.run_id for _n, r in inputs.nominal if r.tracker == "pd"), None)
    if rc_pd is not None:
        lines.extend(
            [
                "",
                "Play the nominal RC+PD v2 run:",
                "",
                "```",
                f"uv run python scripts/play_run.py --run {rc_pd} --scenario configs/tasks/task_1a.toml",
                "```",
            ]
        )
    return lines


def render_report(inputs: ReportInputs, *, plots: Sequence[str] = ()) -> str:
    """The Markdown report."""
    suite = inputs.confirmatory
    dirty = " (dirty)" if suite.provenance.project_dirty else ""
    failed = sum(1 for r in suite.runs if not r.report.success)
    cutoffs = f"{_fmt(suite.estimator.velocity_cutoff_hz)}/{_fmt(suite.estimator.acceleration_cutoff_hz)} Hz"
    seeds = ", ".join(str(v) for v in sorted(set(suite.provenance.seeds.values())))
    nominal = {a.arm: a.joint_rmse_median for a in suite.aggregates if a.kind == "nominal"}
    rc_pd = next((v for k, v in nominal.items() if k.startswith("rc+pd")), None)
    replay_pd = next((v for k, v in nominal.items() if k.startswith("replay+pd")), None)
    lines = [
        "# Task 1-a results",
        "",
        "## Summary",
        "",
        f"- Primary evidence: the locked confirmatory suite `{inputs.confirmatory_file}`",
        f"  (protocol `{suite.protocol_file}`,",
        f"  recipe `{suite.recipe}` from `{suite.recipe_file}`, estimator cutoffs {cutoffs}),",
        f"  run once from commit `{suite.provenance.project_commit[:12]}`{dirty}: {len(suite.scenarios)} scenarios x",
        f"  {len(suite.arms)} arms = {len(suite.runs)} runs, {failed} failed.",
        f"- Confirmatory seeds: {seeds}.",
        f"- Nominal RC+PD v2 joint RMSE: {_fmt(rc_pd)} rad (replay {_fmt(replay_pd)} rad).",
        "",
        "## Primary metric",
        "",
        *_primary(suite),
        "",
        "## Secondary metrics",
        "",
        *_secondary(suite),
        "",
        "## Paired comparisons",
        "",
        *_paired(suite),
        "",
        *_failures(inputs),
        "",
        *_development(inputs),
        "",
        "## Playback",
        "",
        *_playback(inputs),
        "",
        "## Limitations",
        "",
        *[f"- {text}" for text in _LIMITATIONS],
    ]
    if plots:
        lines.extend(["", "## Plots", ""])
        lines.extend(f"![{Path(p).stem}]({p})" for p in plots)
    return "\n".join(lines) + "\n"


def write_plots(inputs: ReportInputs, directory: Path) -> list[Path]:
    """Write the report's figures (joint RMSE by class, paired differences, search objectives)."""
    import matplotlib as mpl  # the backend must be chosen before pyplot loads

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    suite = inputs.confirmatory
    classes = [k for k in CLASS_ORDER if any(s.kind == k for s in suite.scenarios)]
    arms = [a.name for a in suite.arms]
    written: list[Path] = []

    fig, ax = cast("tuple[Any, Any]", plt.subplots(figsize=(9, 4.5)))
    for i, arm in enumerate(arms):
        xs: list[float] = []
        ys: list[float] = []
        for j, kind in enumerate(classes):
            for r in suite.runs:
                if r.arm == arm and r.kind == kind and r.report.success and r.report.joint_rmse is not None:
                    xs.append(j + (i - (len(arms) - 1) / 2) * 0.18)
                    ys.append(r.report.joint_rmse.aggregate)
        ax.scatter(xs, ys, s=14, label=arm)
    ax.set_yscale("log")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_ylabel("joint RMSE (rad)")
    ax.set_title(f"Confirmatory suite: joint RMSE per run ({suite.recipe})")
    ax.legend(fontsize="small")
    ax.grid(visible=True, which="both", alpha=0.3)
    out = directory / PLOT_FILES[0]
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    written.append(out)

    fig, ax = cast("tuple[Any, Any]", plt.subplots(figsize=(9, 4.5)))
    for tracker, kind, diffs in _paired_differences(suite):
        j = classes.index(kind)
        offset = -0.12 if tracker.startswith("pd") else 0.12
        ax.scatter([j + offset] * len(diffs), diffs, s=14, label=tracker if kind == classes[0] else None)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_ylabel("RC minus replay joint RMSE (rad)")
    ax.set_title("Paired differences per scenario")
    ax.legend(fontsize="small")
    ax.grid(visible=True, alpha=0.3)
    out = directory / PLOT_FILES[1]
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    written.append(out)

    fig, ax = cast("tuple[Any, Any]", plt.subplots(figsize=(9, 4.5)))
    for name, s in inputs.searches:
        values = [t.value for t in s.summary.trials if t.flags.get("feasible") is True and t.value is not None]
        if values:
            ax.hist(values, bins=40, alpha=0.6, label=f"{name} ({len(values)} feasible)")
    ax.set_xlabel("median movement joint RMSE over development scenarios (rad)")
    ax.set_ylabel("feasible trials")
    ax.set_title("ESN search objectives")
    ax.legend(fontsize="small")
    ax.grid(visible=True, alpha=0.3)
    out = directory / PLOT_FILES[2]
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    written.append(out)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Render the task 1-a results report from the committed evidence.")
    parser.add_argument("--docs", type=Path, required=True, help="directory holding the committed JSON reports")
    parser.add_argument("--output", type=Path, required=True, help="Markdown report to write")
    parser.add_argument("--plots", type=Path, default=None, help="directory for the figures (default: none)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing report and figures")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists() and not args.force:
        msg = f"refusing to overwrite {output} (use --force)"
        raise FileExistsError(msg)
    inputs = load_inputs(Path(args.docs))
    plot_refs: list[str] = []
    if args.plots is not None:
        plots_dir = Path(args.plots)
        existing = [name for name in PLOT_FILES if (plots_dir / name).exists()]
        if existing and not args.force:
            msg = f"refusing to overwrite {plots_dir / existing[0]} (use --force)"
            raise FileExistsError(msg)
        base = output.parent.resolve()
        for written in write_plots(inputs, plots_dir):
            resolved = written.resolve()
            plot_refs.append(resolved.relative_to(base).as_posix() if resolved.is_relative_to(base) else written.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(inputs, plots=plot_refs), encoding="utf-8")
    print(f"wrote {output}" + (f" and {len(plot_refs)} figures" if plot_refs else ""))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
