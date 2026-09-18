# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: the committed form of a timing smoke check.

The report is the evidence the budget decision rests on, so it rebuilds
strictly from its own JSON, renders deterministically from what it holds, and
refuses figures that contradict themselves. It states its own limits too: the
projection multiplies maximum counts by means measured on a subset, which is an
estimate and not a bound.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.experiments.manual_evaluation import ManualModelTiming, ManualRunTiming
from arm_rc_ctrl.experiments.manual_study import EXPERIMENT_LABEL
from arm_rc_ctrl.experiments.manual_timing import (
    ManualTimingReport,
    TimingDerivation,
    load_timing,
    main,
    peak_rss_bytes,
    project_study,
    render_timing_markdown,
    summarize_timings,
    timing_to_json,
)
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from pathlib import Path

    from arm_rc_ctrl.experiments.manual_fixture import ManualFixture

PAIRS_PER_MODEL = 130
DIGEST = "a" * 64
EVALUATION = repository_root() / "configs" / "evaluations" / "task_1a_manual_dev_v1.toml"
"""The real locked protocol: what a derivative must re-derive 65 x 2 = 130 pairs from."""


def _derivation() -> TimingDerivation:
    """A derivation record, for the cases that check where one may appear."""
    return TimingDerivation(
        derived_from_sha256=DIGEST,
        derivation_commit="b" * 40,
        derivation_dirty=False,
        derived_at="2026-09-18T00:00:00+00:00",
        reason="the original projected the subset it measured rather than the study it estimated",
    )


def _runs() -> tuple[ManualRunTiming, ...]:
    return (
        ManualRunTiming(
            arm="rc",
            scenario_id="nominal",
            tracker="pd_v2",
            rows=3001,
            simulate_seconds=1.0,
            persist_seconds=0.2,
            run_bytes=126_000,
        ),
        ManualRunTiming(
            arm="replay",
            scenario_id="nominal",
            tracker="pd_v2",
            rows=3001,
            simulate_seconds=0.8,
            persist_seconds=0.1,
            run_bytes=94_000,
        ),
    )


def _models() -> tuple[ManualModelTiming, ...]:
    return (
        ManualModelTiming(
            label="feasible-best/S/D01",
            fit_seconds=12.5,
            fit_cache_hit=False,
            sweep_seconds=90.0,
            runs=PAIRS_PER_MODEL,
        ),
    )


def _report(f: ManualFixture, **overrides: object) -> ManualTimingReport:
    """A report over the fixture's own environment records, with one field replaceable per case."""
    runs = _runs()
    arguments: dict[str, object] = {
        "experiment": EXPERIMENT_LABEL,
        "study_manifest_sha256": DIGEST,
        "evaluation_sha256": DIGEST,
        "entries": ("feasible-best/S/D01",),
        "execution": f.execution,
        "models": _models(),
        "runs": runs,
        "run_stats": summarize_timings(runs),
        "runs_this_invocation": len(runs),
        "replay_banks_built": 1,
        "workers": 1,
        "wall_seconds": 210.0,
        "peak_rss_bytes": 277_340_160,
        "peak_rss_children_bytes": 277_340_160,
        "storage_bytes": 220_000,
        "projection": project_study(_models(), runs, pairs_per_model=PAIRS_PER_MODEL, completed_models=1),
        "revised_estimate": "Measured here and scaled to the whole study; an estimate, not a bound.",
        "provenance": f.provenance,
    }
    arguments.update(overrides)
    return ManualTimingReport(**arguments)  # type: ignore[arg-type]


# --- the record --------------------------------------------------------------------------------


def test_a_report_rebuilds_from_its_own_json(manual_fixture: ManualFixture, tmp_path: Path) -> None:
    """Every field survives the round trip, including the nested execution and provenance records."""
    report = _report(manual_fixture)
    path = tmp_path / "timing.json"
    path.write_text(timing_to_json(report) + "\n", encoding="utf-8")
    rebuilt = load_timing(path)
    assert to_mapping(rebuilt) == to_mapping(report)
    assert rebuilt.execution.identity == manual_fixture.execution.identity
    assert rebuilt.projection.total_runs == 31_980


def test_a_report_refuses_to_claim_more_measurements_than_it_holds(manual_fixture: ManualFixture) -> None:
    """``runs_this_invocation`` counts what this run simulated, so it cannot exceed what was measured."""
    with pytest.raises(ValueError, match="runs_this_invocation"):
        _report(manual_fixture, runs_this_invocation=99)


def test_a_report_refuses_a_foreign_experiment_or_schema(manual_fixture: ManualFixture) -> None:
    """Evidence names the experiment it belongs to; another label is another experiment."""
    with pytest.raises(ValueError, match=r"experiment|schema"):
        _report(manual_fixture, experiment="task_1a_repetition_v1")
    with pytest.raises(ValueError, match=r"experiment|schema"):
        _report(manual_fixture, schema_version=99)


def test_a_report_requires_well_formed_bindings_and_a_stated_estimate(manual_fixture: ManualFixture) -> None:
    """The digests bind the study and configuration it measured; the estimate is the point of the report."""
    with pytest.raises(ValueError, match="sha256"):
        _report(manual_fixture, study_manifest_sha256="not-a-digest")
    with pytest.raises(ValueError, match="estimate"):
        _report(manual_fixture, revised_estimate="   ")


# --- the rendering -----------------------------------------------------------------------------


def test_the_markdown_is_deterministic_and_states_what_it_projected(manual_fixture: ManualFixture) -> None:
    """The same report renders identically, names the projected totals, and admits it is not a bound."""
    report = _report(manual_fixture)
    first = render_timing_markdown(report)
    assert first == render_timing_markdown(report)
    assert "31,980" in first or "31980" in first
    assert "24,180" in first or "24180" in first
    assert "7,800" in first or "7800" in first
    assert "not a guaranteed bound" in first, "the report must say what its projection is worth"
    assert report.revised_estimate in first


# --- the memory measurement --------------------------------------------------------------------


def test_peak_memory_is_reported_for_this_process_and_its_children() -> None:
    """Workers are separate processes, so their peak is measured beside this one's."""
    own, children = peak_rss_bytes()
    assert own > 0
    assert children >= 0


def test_a_report_refuses_no_negative_figures(manual_fixture: ManualFixture) -> None:
    """A negative measurement is not a small one: it is a broken record, and it is refused."""
    with pytest.raises(ValueError, match="negative"):
        _report(manual_fixture, wall_seconds=-1.0)


def test_a_served_fit_is_rendered_as_a_cache_hit(manual_fixture: ManualFixture) -> None:
    """A model whose fit was served says so, or the report would read as if it had paid for it."""
    served = (
        ManualModelTiming(
            label="feasible-best/S/D01",
            fit_seconds=12.5,
            fit_cache_hit=True,
            sweep_seconds=90.0,
            runs=PAIRS_PER_MODEL,
        ),
    )
    markdown = render_timing_markdown(_report(manual_fixture, models=served))
    assert "cache hit" in markdown
    assert "fitted now" not in markdown


def test_a_report_runs_under_at_least_one_worker(manual_fixture: ManualFixture) -> None:
    """Zero workers would mean the report describes work nobody did."""
    with pytest.raises(ValueError, match="at least one worker"):
        _report(manual_fixture, workers=0)


def test_the_worker_count_survives_the_round_trip(manual_fixture: ManualFixture, tmp_path: Path) -> None:
    """The benchmark reads the count back out of the JSON, so it has to be in there."""
    report = _report(manual_fixture, workers=4)
    path = tmp_path / "timing.json"
    path.write_text(timing_to_json(report) + "\n", encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["workers"] == 4
    assert load_timing(path).workers == 4


# --- the corrected derivative (schema v2) -------------------------------------------------------


def test_a_version_one_report_still_loads(manual_fixture: ManualFixture, tmp_path: Path) -> None:
    """Raising the schema must not strand the reports already written under version 1."""
    path = tmp_path / "timing.json"
    path.write_text(timing_to_json(_report(manual_fixture, schema_version=1)) + "\n", encoding="utf-8")
    rebuilt = load_timing(path)
    assert rebuilt.schema_version == 1
    assert rebuilt.derivation is None


def test_a_derivation_belongs_to_the_schema_that_defines_it(manual_fixture: ManualFixture) -> None:
    """A version 1 record cannot carry a field version 1 never had."""
    derived_from = _derivation()
    with pytest.raises(ValueError, match="derivation"):
        _report(manual_fixture, schema_version=1, derivation=derived_from)


def test_the_derivative_corrects_the_projection_and_names_its_source(
    manual_fixture: ManualFixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The measurement is carried over untouched; only what was computed from it is recomputed.

    The original serial benchmark projected the study it had been restricted
    to rather than the study it was estimating, so its own projection block is
    wrong while every figure it measured is sound. A derivative re-derives the
    locked pair count from the evaluation configuration -- 65 cases under two
    trackers -- rather than trusting either the original or a constant here.
    """
    runs = _runs()
    original = _report(
        manual_fixture,
        schema_version=1,
        evaluation_sha256=sha256_file(EVALUATION),
        projection=project_study(_models(), runs, pairs_per_model=2, completed_models=1),
    )
    source = tmp_path / "original.json"
    source.write_text(timing_to_json(original) + "\n", encoding="utf-8")
    assert original.projection.total_runs == (186 + 60) * 2, "the defect this derivative corrects"

    argv = [
        "derive",
        "--from",
        str(source),
        "--evaluation",
        str(EVALUATION),
        "--output",
        str(tmp_path / "derived.json"),
        "--markdown",
        str(tmp_path / "derived.md"),
    ]
    assert main(argv) == 0
    capsys.readouterr()

    derived = load_timing(tmp_path / "derived.json")
    assert derived.schema_version == 2
    assert derived.derivation is not None
    assert derived.derivation.derived_from_sha256 == sha256_file(source)
    assert derived.projection.pairs_per_model == PAIRS_PER_MODEL
    assert derived.projection.total_runs == 31_980
    # the measurement itself is preserved, which is the whole point of deriving rather than re-running
    assert derived.runs == original.runs
    assert derived.models == original.models
    assert derived.wall_seconds == original.wall_seconds
    assert derived.storage_bytes == original.storage_bytes
    assert to_mapping(derived.execution) == to_mapping(original.execution)
    assert to_mapping(derived.provenance) == to_mapping(original.provenance)


def test_a_derivative_refuses_a_configuration_the_original_did_not_measure(
    manual_fixture: ManualFixture, tmp_path: Path
) -> None:
    """The locked count is re-derived from a configuration, so it must be the measured one."""
    source = tmp_path / "original.json"
    source.write_text(timing_to_json(_report(manual_fixture, schema_version=1)) + "\n", encoding="utf-8")
    argv = [
        "derive",
        "--from",
        str(source),
        "--evaluation",
        str(EVALUATION),
        "--output",
        str(tmp_path / "derived.json"),
        "--markdown",
        str(tmp_path / "derived.md"),
    ]
    with pytest.raises(ValueError, match="evaluation"):
        main(argv)
