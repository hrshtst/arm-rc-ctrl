# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-007: the report tables, task-clock figures, representatives, and animations derive from the evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.execution import load_execution
from arm_rc_ctrl.experiments.repetition_accounting import account_pilot
from arm_rc_ctrl.experiments.repetition_evaluation import C11_CAVEAT, ModelEvidence, ReplayBank
from arm_rc_ctrl.experiments.repetition_fixture import (
    DOCS,
    CraftedSimulator,
    PlanarFixture,
    build_pilot_runner,
    exploratory_provenance,
)
from arm_rc_ctrl.experiments.repetition_numerics import load_validation
from arm_rc_ctrl.experiments.repetition_panel import load_panel
from arm_rc_ctrl.experiments.repetition_recipes import panel_arms
from arm_rc_ctrl.experiments.repetition_report import (
    REPRESENTATIVE_ARMS,
    OutcomeRow,
    RepetitionReport,
    ReportInputs,
    build_report,
    cost_rows,
    equivalence_rows,
    load_report,
    outcome_rows,
    paired_rows,
    plot_outcome_grid,
    plot_peak_speeds,
    plot_trajectory,
    plot_worst_cell_by_count,
    render_report_markdown,
    representatives,
    speed_rows,
    write_animations,
    write_plots,
)
from arm_rc_ctrl.experiments.repetition_timing import load_timing
from arm_rc_ctrl.experiments.run_record import load_run, pointer_from_summary

MANIFEST = DOCS / "panel_manifest_v1.json"


@pytest.fixture(scope="module")
def inputs(fixture: PlanarFixture, tmp_path_factory: pytest.TempPathFactory) -> ReportInputs:
    """Report inputs over the fixture: crafted sweeps of feasible-best for every behavioral arm (one entry present)."""
    f = fixture
    runner = build_pilot_runner(f, velocity_abort=(49.0, 49.0), simulate_fn=CraftedSimulator(f.samples))
    models: dict[str, ModelEvidence] = {}
    for arm in panel_arms():
        if arm.behavioral:
            models[f"{f.entry.label}/{arm.label}"] = runner.evaluate(f.entry, arm)
    evidence_dir = tmp_path_factory.mktemp("report") / "evidence"
    runner.write_pointers(evidence_dir)
    accounting = account_pilot(
        store=f.store,
        evidence_dir=evidence_dir,
        manifest_file=MANIFEST,
        validation_file=DOCS / "numerical_validation_v1.json",
        provenance=exploratory_provenance(),
    )
    bank: ReplayBank = runner.replay_bank(f.entry)
    return ReportInputs(
        docs=DOCS,
        store=f.store,
        root=f.root,
        manifest=load_panel(MANIFEST),
        accounting=accounting,
        validation=load_validation(DOCS / "numerical_validation_v1.json"),
        timing=load_timing(DOCS / "timing_smoke_check_v1.json"),
        execution=load_execution(DOCS / "execution_environment_v1.json"),
        models=models,
        banks={f.entry.warmup_s: bank},
        fit_seconds={},
        sources={"panel_manifest_v1.json": "a" * 64},
    )


def test_tables_derive_from_the_manifests(inputs: ReportInputs) -> None:
    """Outcomes, paired comparisons, equivalences, speeds, and costs follow from the evidence."""
    outcomes = outcome_rows(inputs)
    assert len(outcomes) == 20
    assert all(o.panel_label == "feasible-best" and o.status == "feasible" and o.n_completed == 10 for o in outcomes)
    assert all(o.worst_cell == pytest.approx(0.25, rel=1e-6) for o in outcomes)
    assert all(o.peak_speed is not None and o.peak_speed > 0 for o in outcomes)
    paired = paired_rows(inputs)
    # 6 entries are in the manifest but only feasible-best has evidence: 3 counts x 6 within-formulation pairs
    # plus 1 + 3 + 3 across-formulation comparisons = 25 rows.
    assert len(paired) == 25
    assert all(p.panel_label == "feasible-best" for p in paired)
    assert {p.comparison for p in paired} >= {"absolute/R/K17 vs absolute/S", "residual/S vs absolute/S"}
    first = next(p for p in paired if p.comparison == "absolute/R/K17 vs absolute/S")
    assert first.shared_pairs == 10
    assert not first.verdict_changed
    assert first.signed_difference_median["early_gap_integral"] == pytest.approx(0.0, abs=1e-12)
    assert first.left_worst_cell == pytest.approx(0.25, rel=1e-6)
    equivalence = equivalence_rows(inputs)
    assert len(equivalence) == 72 + 36  # residual comparisons carry two quantities
    assert sum(1 for e in equivalence if e.accepted_exception) == 1
    speeds = speed_rows(inputs)
    assert len(speeds) == 20
    assert all(s.executed_runs == 10 and s.aborts == 0 for s in speeds)
    assert all(set(s.time_above_historical_s) == {"warmup", "movement", "dwell"} for s in speeds)
    costs = cost_rows(inputs)
    assert len(costs) == 26
    behavioral = [c for c in costs if c.arm.split("/")[1] != "S-effective"]
    assert all(c.models == 1 and c.executed_runs_mean == 10.0 and c.evaluation_seconds_mean > 0 for c in behavioral)
    references = [c for c in costs if c.arm.split("/")[1] == "S-effective"]
    assert all(c.models == 6 and c.fits_timed == 0 for c in references)


def test_representatives_follow_the_predeclared_rule(inputs: ReportInputs) -> None:
    """Every evaluated entry's five representative arms name their first evaluated pair, whatever its outcome."""
    reps = representatives(inputs)
    assert [r.arm for r in reps] == [arm.label for arm in REPRESENTATIVE_ARMS]
    assert all(r.panel_label == "feasible-best" for r in reps)
    assert all(r.scenario_id == "nominal" and r.tracker == "pd_v2" for r in reps)
    assert all(r.status == "completed" and r.rc_run is not None and r.replay_run is not None for r in reps)
    assert all(r.plot is None and r.rc_animation is None for r in reps)
    assert {r.warmup_s for r in reps} == {0.25}


def test_figures_render_on_the_task_clock(inputs: ReportInputs, tmp_path: Path) -> None:
    """The grid, worst-cell, peak-speed, and trajectory figures write once and refuse to overwrite."""
    outcomes = outcome_rows(inputs)
    grid = plot_outcome_grid(outcomes, tmp_path / "grid.png", formulation="absolute", entries=["feasible-best"])
    assert grid.stat().st_size > 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        plot_outcome_grid(outcomes, tmp_path / "grid.png", formulation="absolute", entries=["feasible-best"])
    worst = plot_worst_cell_by_count(outcomes, tmp_path / "worst.png", entries=["feasible-best"])
    assert worst.stat().st_size > 0
    peaks = plot_peak_speeds(speed_rows(inputs), tmp_path / "peaks.png", historical=20.0, abort=49.0)
    assert peaks.stat().st_size > 0
    evidence = inputs.models["feasible-best/residual/S"]
    pair = evidence.pairs[0]
    assert pair.run is not None
    rc = load_run(inputs.store, pointer_from_summary(inputs.store, pair.run.artifact_id))
    replay_pair = inputs.banks[0.25].get(pair.scenario_id, pair.tracker)
    assert replay_pair is not None
    assert replay_pair.run is not None
    replay = load_run(inputs.store, pointer_from_summary(inputs.store, replay_pair.run.artifact_id))
    out = plot_trajectory(
        rc,
        replay,
        tmp_path / "trajectory.png",
        title="fixture residual",
        warmup_s=0.25,
        dwell_start_s=0.8,
        historical=(20.0, 20.0),
        abort=(49.0, 49.0),
        force_window=(0.3, 0.4),
    )
    assert out.stat().st_size > 0
    only_replay = plot_trajectory(
        None,
        replay,
        tmp_path / "replay_only.png",
        title="replay",
        warmup_s=0.0,
        dwell_start_s=0.8,
        historical=(20.0, 20.0),
        abort=(49.0, 49.0),
    )
    assert only_replay.stat().st_size > 0


def test_write_plots_and_animations_attach_assets(inputs: ReportInputs, tmp_path: Path) -> None:
    """Plots are written for every representative with a replay run; the animation rule exports via the player."""
    reps = representatives(inputs)
    names, attached = write_plots(inputs, reps, tmp_path / "plots")
    assert names[:4] == ["outcomes_absolute.png", "outcomes_residual.png", "worst_cell_by_count.png", "peak_speed.png"]
    assert len(names) == 4 + len(reps)
    assert all(r.plot is not None and (tmp_path / "plots" / r.plot).is_file() for r in attached)
    calls: list[list[str]] = []

    def fake_player(argv: list[str]) -> int:
        calls.append(list(argv))
        target = Path(argv[argv.index("--export") + 1])
        target.write_bytes(b"GIF89a")
        return 0

    with_animations = write_animations(inputs, attached, tmp_path / "animations", player=cast("Any", fake_player))
    exported = [r for r in with_animations if r.rc_animation is not None]
    assert [r.arm for r in exported] == ["absolute/S", "absolute/R/K65"]
    assert all(r.replay_animation == "feasible-best__replay_pd_v2.gif" for r in exported)
    # Every distinct run is exported once (the crafted RC runs share one identical record, the replay another).
    distinct = {r.rc_run for r in exported} | {r.replay_run for r in exported}
    assert len(calls) == len(distinct) == 2
    assert all("--task-clock" in argv and "--scenario" in argv for argv in calls)

    def failing_player(argv: list[str]) -> int:
        del argv
        return 2

    with pytest.raises(RuntimeError, match="failed to export"):
        write_animations(inputs, attached, tmp_path / "animations-failed", player=cast("Any", failing_player))


def test_report_assembles_roundtrips_and_renders(inputs: ReportInputs, tmp_path: Path) -> None:
    """The assembled report re-derives its counts, carries the C11 caveat verbatim, and renders every table."""
    reps = representatives(inputs)
    report = build_report(inputs, reps=reps, plots=["a.png"], animations=[], provenance=exploratory_provenance())
    assert isinstance(report, RepetitionReport)
    assert report.n_feasible == 20
    assert report.n_rc_gate_failure == 0
    assert report.c11_caveat == C11_CAVEAT
    assert report.velocity_abort == (49.0, 49.0)
    file = tmp_path / "report.json"
    file.write_text(json.dumps(to_mapping(report)), encoding="utf-8")
    assert load_report(file) == report
    markdown = render_report_markdown(report)
    assert "## Paired comparisons" in markdown
    assert "accepted exception (C11)" in markdown
    assert markdown.count("| feasible-best | absolute |") >= 13
    with pytest.raises(ValueError, match="contradict"):
        replace(report, n_feasible=1)
    with pytest.raises(ValueError, match="C11 caveat"):
        replace(report, c11_caveat="other")
    with pytest.raises(ValueError, match="unsupported"):
        replace(report, schema_version=2)
    row = OutcomeRow(**{**to_mapping(report.outcomes[0]), "status": "rc_gate_failure"})  # type: ignore[arg-type]
    assert row.status == "rc_gate_failure"
