# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-005: the timing smoke check measures runs, fits, memory, and storage and projects the full panel."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.experiments import repetition_evaluation
from arm_rc_ctrl.experiments.perturbations import RobustnessScenario
from arm_rc_ctrl.experiments.repetition_evaluation import RunTiming
from arm_rc_ctrl.experiments.repetition_fixture import (
    DOCS,
    PLANAR_DIGESTS,
    PLANAR_SCENARIOS,
    PLANAR_TRACKER,
    CraftedSimulator,
    PlanarFixture,
    build_pilot_runner,
    write_pilot_evaluation_config,
)
from arm_rc_ctrl.experiments.repetition_numerics import PanelContext
from arm_rc_ctrl.experiments.repetition_recipes import ArmSpec
from arm_rc_ctrl.experiments.repetition_timing import (
    ModelTiming,
    TimingReport,
    load_timing,
    main,
    peak_rss_bytes,
    project_panel,
    render_timing_markdown,
    summarize_timings,
)
from arm_rc_ctrl.provenance import sha256_file

if TYPE_CHECKING:
    from arm_rc_ctrl.execution import ExecutionRecord
    from arm_rc_ctrl.storage import StorageRoot


def _timing(arm: str, seconds: float, size: int) -> RunTiming:
    return RunTiming(
        arm=arm,
        label="x",
        scenario_id="s",
        tracker="pd_v2",
        rows=10,
        simulate_seconds=seconds,
        persist_seconds=0.1,
        run_bytes=size,
    )


def test_run_statistics_and_projection_scale_the_measured_rates() -> None:
    """Per-arm statistics and the full-panel projection follow from the measured runs and fits."""
    runs = [_timing("rc", 0.2, 1000), _timing("rc", 0.4, 3000), _timing("replay", 0.1, 500)]
    stats = summarize_timings(runs)
    assert [s.arm for s in stats] == ["replay", "rc"]
    rc = next(s for s in stats if s.arm == "rc")
    assert rc.runs == 2
    assert rc.mean_simulate_s == pytest.approx(0.3)
    assert rc.max_simulate_s == 0.4
    assert rc.mean_bytes == 2000.0
    models = [
        ModelTiming(
            label="a",
            status="feasible",
            fit_identity="a" * 64,
            fit_cache_hit=True,
            fit_seconds=0.5,
            sweep_seconds=10.0,
            runs=2,
            unexecuted=0,
            run_bytes=4000,
        ),
        ModelTiming(
            label="b",
            status="rc_gate_failure",
            fit_identity="b" * 64,
            fit_cache_hit=False,
            fit_seconds=1.5,
            sweep_seconds=5.0,
            runs=1,
            unexecuted=1,
            run_bytes=1000,
        ),
    ]
    projection = project_panel(models, runs, models_per_entry=20, pairs_per_model=130, completed_models=2)
    assert projection.rc_runs == 6 * 20 * 130
    assert projection.replay_runs == 3 * 130
    assert projection.rc_run_seconds == pytest.approx(0.4)  # mean simulate + persist of the rc runs
    assert projection.replay_run_seconds == pytest.approx(0.2)
    assert projection.fit_seconds == pytest.approx(6 * 20 * 1.0)
    assert projection.total_seconds == pytest.approx(15600 * 0.4 + 390 * 0.2 + 120.0)
    assert projection.storage_bytes == (15600 * 2000 + 390 * 500)
    assert projection.remaining_seconds == pytest.approx(projection.total_seconds - 2 * (130 * 0.4 + 1.0) - 0.2)
    empty = project_panel([], [], models_per_entry=20, pairs_per_model=130, completed_models=0)
    assert empty.total_seconds == 0.0
    assert summarize_timings([]) == ()
    with pytest.raises(ValueError, match="malformed"):
        ModelTiming(
            label="a",
            status="other",
            fit_identity=None,
            fit_cache_hit=None,
            fit_seconds=None,
            sweep_seconds=1.0,
            runs=0,
            unexecuted=0,
            run_bytes=0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        _timing("rc", -1.0, 0)
    own, children = peak_rss_bytes()
    assert own > 0
    assert children >= 0


@pytest.mark.usefixtures("pinned_environment")
def test_smoke_command_measures_one_entry_and_writes_the_report(
    fixture: PlanarFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command evaluates the entry across the selected arms on crafted runs and reports every measurement."""
    f = fixture
    evaluation, file = write_pilot_evaluation_config(f.root, velocity_abort=(44.0, 44.0))
    manifest = tmp_path / "panel.json"
    manifest.write_text("{}", encoding="utf-8")

    class FakeConfigs:
        development_sha256 = sha256_file(evaluation.development)
        trackers = PLANAR_DIGESTS

    class FakeRule:
        labels = ("feasible-best",)

    class FakeManifest:
        configs = FakeConfigs()
        rule = FakeRule()
        entries = (f.entry,)

        def entry(self, label: str) -> object:
            assert label == "feasible-best"
            return f.entry

    def fake_load(manifest_file: Path, *, store: StorageRoot, root: Path, execution: ExecutionRecord) -> PanelContext:
        del store, root
        return PanelContext(
            manifest=cast("Any", FakeManifest()),
            manifest_sha256=sha256_file(manifest_file),
            inputs=replace(f.inputs, execution_identity=execution.identity),
            dataset=f.record,
            payload=f.payload,
        )

    def fixture_scenarios(*args: object, **kwargs: object) -> tuple[RobustnessScenario, ...]:
        del args, kwargs
        return PLANAR_SCENARIOS

    def fixture_tracker(name: str) -> TrackerConfig:
        del name
        return PLANAR_TRACKER

    monkeypatch.setattr(PanelContext, "load", fake_load)
    monkeypatch.setattr(repetition_evaluation, "repository_root", lambda: f.root)
    monkeypatch.setattr(repetition_evaluation, "robustness_scenarios", fixture_scenarios)
    monkeypatch.setattr(repetition_evaluation, "load_frozen_baseline", fixture_tracker)
    monkeypatch.setattr(repetition_evaluation, "frozen_baseline_digest", PLANAR_DIGESTS.__getitem__)
    monkeypatch.setattr(repetition_evaluation, "simulate", CraftedSimulator(f.samples))
    output, markdown = tmp_path / "timing.json", tmp_path / "timing.md"
    argv = [
        "smoke",
        "--manifest",
        str(manifest),
        "--evaluation",
        str(file),
        "--validation",
        str(DOCS / "numerical_validation_v1.json"),
        "--evidence-dir",
        str(tmp_path / "evidence"),
        "--arms",
        "absolute/S",
        "absolute/R/K17",
        "--output",
        str(output),
        "--markdown",
        str(markdown),
        "--exploratory",
    ]
    assert main(argv) == 0
    report = load_timing(output)
    assert isinstance(report, TimingReport)
    assert report.panel_label == "feasible-best"
    assert report.replay_bank_runs == 10
    assert [m.label for m in report.models] == ["feasible-best/absolute/S", "feasible-best/absolute/R/K17"]
    assert all(m.status == "feasible" and m.runs == 10 and m.unexecuted == 0 for m in report.models)
    assert all(m.fit_cache_hit is not None and m.fit_seconds is not None for m in report.models)
    assert len(report.runs) == 30
    assert [s.arm for s in report.run_stats] == ["replay", "rc"]
    assert report.wall_seconds > 0
    assert report.peak_rss_bytes > 0
    assert report.storage_bytes > sum(r.run_bytes for r in report.runs)
    assert report.projection.models_per_entry == 20
    assert report.projection.pairs_per_model == 10
    assert report.projection.completed_models == 2
    assert report.projection.total_seconds > 0
    assert "Revised engineering estimate" in markdown.read_text(encoding="utf-8")
    assert markdown.read_text(encoding="utf-8") == render_timing_markdown(report)
    assert sorted(p.name for p in (tmp_path / "evidence").glob("*.toml")) == [
        "model__feasible-best__absolute__R__K17.toml",
        "model__feasible-best__absolute__S.toml",
        "replay__warmup-0.25s.toml",
    ]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(argv)
    with pytest.raises(ValueError, match="exactly one panel entry"):
        main([*argv[:-1], "--entries", "feasible-best", "feasible-best", "--exploratory"])
    with pytest.raises(ValueError, match="revised estimate"):
        replace(report, revised_estimate=" ")
    with pytest.raises(ValueError, match="replay_bank_runs"):
        replace(report, replay_bank_runs=3)


def test_runner_records_timings_for_simulated_runs_only(fixture: PlanarFixture) -> None:
    """Resumed runs are not re-timed; timed runs carry their rows, seconds, and bytes."""
    f = fixture
    runner = build_pilot_runner(f, velocity_abort=(45.0, 45.0), simulate_fn=CraftedSimulator(f.samples))
    evidence = runner.evaluate(f.entry, ArmSpec("residual", "R", 16))
    assert evidence.status == "feasible"
    assert len(runner.run_timings) == 20  # ten replay runs and ten rc runs
    assert all(t.rows == 126 and t.run_bytes > 0 and t.simulate_seconds >= 0 for t in runner.run_timings)
    assert {t.arm for t in runner.run_timings} == {"replay", "rc"}
    assert evidence.evaluation_identity in runner.model_timings
    assert runner.model_timings[evidence.evaluation_identity].fit_cache_hit is False
    assert runner.manifest_bytes > 0
    again = build_pilot_runner(f, velocity_abort=(45.0, 45.0), simulate_fn=CraftedSimulator(f.samples))
    assert again.evaluate(f.entry, ArmSpec("residual", "R", 16)) == evidence
    assert again.run_timings == []
    assert again.model_timings[evidence.evaluation_identity].fit_cache_hit is True
