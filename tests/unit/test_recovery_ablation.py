# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-013: the ablation core — arm summaries, section 7.3 eligibility, and the rendered comparison."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.experiments.esn_search import TrialPoint
from arm_rc_ctrl.experiments.recovery_ablation import (
    CELL_SCENARIOS,
    MIN_IMPROVING,
    CandidateCell,
    CandidateTrial,
    ablation_to_json,
    build_ablation,
    evaluate_candidates,
    load_ablation,
    render_ablation_markdown,
    summarize_arm,
)
from arm_rc_ctrl.experiments.recovery_search import RecoveryTrialPoint
from arm_rc_ctrl.experiments.recovery_study import RecoveryStudyReport
from arm_rc_ctrl.experiments.studies import StudySummary, TrialRecord
from arm_rc_ctrl.provenance import collect_provenance

if TYPE_CHECKING:
    from pathlib import Path


CELLS = (
    "posture_small:pd_v2",
    "posture_small:computed_torque",
    "posture_large:pd_v2",
    "posture_large:computed_torque",
)


def _trial(
    number: int,
    *,
    feasible: bool,
    value: float | None = None,
    warmup: float = 0.0,
    gap: float = 0.5,
    jump: float = 0.6,
    reason: str | None = None,
    comparison: str | None = None,
    scenarios_per_cell: int = 20,
) -> TrialRecord:
    """A stored trial with ``scenarios_per_cell`` paired components per class-by-tracker cell."""
    labels: dict[str, str] = {}
    metrics: dict[str, float] = {}
    flags: dict[str, bool] = {"feasible": feasible}
    if reason is not None:
        labels["reason"] = reason
    if comparison is not None:
        labels["armrc.comparison"] = comparison
    index = 0
    if feasible:
        for cell in CELLS:
            kind, tracker = cell.split(":")
            for draw in range(scenarios_per_cell):
                prefix = f"components.{index}"
                labels[prefix + ".kind"] = kind
                labels[prefix + ".tracker"] = tracker
                labels[prefix + ".scenario_id"] = f"{kind}-{draw}"
                metrics[prefix + ".gap_ratio"] = gap
                metrics[prefix + ".activation_jump_rad"] = jump * 0.1  # replay jump is 0.1 below
                index += 1
    return TrialRecord(
        number=number,
        state="COMPLETE",
        value=value,
        params={"warmup_s": warmup},
        metrics=metrics,
        flags=flags,
        labels=labels,
    )


def _jumps(scenarios_per_cell: int = 20) -> dict[tuple[str, float, str], float]:
    table: dict[tuple[str, float, str], float] = {}
    for cell in CELLS:
        kind, tracker = cell.split(":")
        for draw in range(scenarios_per_cell):
            for warmup in (0.0, 0.25):
                table[(tracker, warmup, f"{kind}-{draw}")] = 0.1
    return table


def _report(name: str, formulation: str, trials: tuple[TrialRecord, ...]) -> RecoveryStudyReport:
    """A minimal runner-shaped study report; the best feasible trial (if any) carries a matching point."""
    feasible = [t for t in trials if t.flags.get("feasible") is True]
    complete = [t for t in feasible if t.value is not None]
    best = min(complete, key=lambda t: (t.value, t.number)) if complete else None
    best_point = None
    if best is not None:
        # The report re-derives best_point.params() == best.params, so the best trial is rebuilt with
        # the full parameter set of a synthetic point (the ablation itself only reads warmup_s).
        point = RecoveryTrialPoint(
            esn=TrialPoint(100, 0.9, 0.9, 0.1, 0.1, 31, 0.01, 20.0, 20.0),
            warmup_s=float(best.params["warmup_s"]),
            augmentation=None,
        )
        params = {k: float(v) for k, v in point.params().items()}
        patched = TrialRecord(
            number=best.number,
            state=best.state,
            value=best.value,
            params=params,
            metrics=best.metrics,
            flags=best.flags,
            labels=best.labels,
        )
        trials = tuple(patched if t.number == best.number else t for t in trials)
        best_point = point
    summary = StudySummary(
        name=name,
        storage=f"armrc://optuna/{name}.db",
        direction="minimize",
        identity={},
        trials=trials,
        n_complete=sum(1 for t in trials if t.state == "COMPLETE"),
        n_pruned=0,
        best_number=None if best is None else best.number,
        best_value=None if best is None else best.value,
        selection_rule="feasible",
    )
    return RecoveryStudyReport(
        protocol=name,
        protocol_file=f"configs/studies/{name}.toml",
        protocol_sha256="c" * 64,
        formulation=formulation,
        dataset="processed-test",
        trackers={"pd_v2": "a" * 64, "computed_torque": "b" * 64},
        budget=4,
        trials_run=len(trials),
        summary=summary,
        best_point=best_point,
        n_feasible=len(feasible),
        provenance=collect_provenance({}, seeds={}, artifacts=[], exploratory=True),
    )


def test_cell_rule_constants_match_the_protocol() -> None:
    """Protocol v1 fixes exactly 20 scenarios per cell and the 15-of-20 improvement threshold."""
    assert CELL_SCENARIOS == 20
    assert MIN_IMPROVING == 15
    with pytest.raises(ValueError, match="exactly 20"):
        CandidateCell(gap_median=0.5, jump_median=0.5, improving_both=2, n=2, passes=True)


def test_candidates_pass_and_fail_the_cell_rule() -> None:
    """Both medians below 1 with enough improving scenarios pass; a failing jump median disqualifies."""
    good = _trial(1, feasible=True, value=0.5, gap=0.5, jump=0.6)
    bad_jump = _trial(2, feasible=True, value=0.4, gap=0.4, jump=1.2)
    candidates = evaluate_candidates("study-a", (good, bad_jump), _jumps())
    assert [c.number for c in candidates] == [1, 2]
    assert candidates[0].eligible
    assert all(cell.passes for cell in candidates[0].cells.values())
    assert candidates[0].cells["posture_small:pd_v2"].jump_median == pytest.approx(0.6)
    assert not candidates[1].eligible
    assert all(not cell.passes for cell in candidates[1].cells.values())
    assert candidates[1].cells["posture_large:pd_v2"].improving_both == 0


def test_a_missing_replay_jump_is_an_error() -> None:
    """Eligibility never silently skips a paired figure."""
    trial = _trial(3, feasible=True, value=0.5, warmup=1.0)  # jumps table only holds 0.0 and 0.25
    with pytest.raises(ValueError, match="paired"):
        evaluate_candidates("study-a", (trial,), _jumps())


def test_arm_summary_counts_reasons_and_warmups() -> None:
    """Reason heads aggregate the first failing gate; feasible trials split by warm-up; the anchor is kept."""
    trials = (
        _trial(0, feasible=False, value=10.0, reason="scenario 0 [pd_v2]: dwell:dwell_stationary", comparison="anchor"),
        _trial(1, feasible=False, value=10.0, reason="scenario 3 [pd_v2]: limit_violation:joint_velocity"),
        _trial(2, feasible=True, value=0.5, warmup=0.0),
        _trial(3, feasible=True, value=0.7, warmup=0.25),
    )
    arm = summarize_arm("recovery_search_x_v1.json", _report("study-x", "no_augmentation", trials))
    assert arm.reasons == {"dwell": 1, "limit_violation:joint_velocity": 1}
    assert arm.feasible_by_warmup == {"0": 1, "0.25": 1}
    assert arm.anchor_reason == "scenario 0 [pd_v2]: dwell:dwell_stationary"
    assert arm.n_feasible == 2


def test_build_render_and_roundtrip(tmp_path: Path) -> None:
    """The composed report re-derives its counts, renders every required section, and loads back strictly."""
    timing = _report(
        "study-timing",
        "no_augmentation",
        (
            _trial(0, feasible=False, value=10.0, reason="scenario 0 [pd_v2]: dwell:dwell_stationary"),
            _trial(1, feasible=True, value=0.5, warmup=0.0),
        ),
    )
    augmented = _report(
        "study-aug",
        "contractive",
        (_trial(0, feasible=False, value=10.0, reason="scenario 2 [pd_v2]: limit_violation:joint_velocity"),),
    )
    provenance = collect_provenance({}, seeds={}, artifacts=[], exploratory=True)
    report = build_ablation(
        (("timing.json", timing), ("aug.json", augmented)),
        _jumps(),
        dataset="processed-test",
        provenance=provenance,
    )
    assert report.n_eligible == 1
    assert [arm.study for arm in report.arms] == ["study-timing", "study-aug"]
    markdown = render_ablation_markdown(report)
    for required in (
        "# Task 1-a recovery development ablation",
        "## Arms",
        "## Failure taxonomy",
        "## Timing-only arm",
        "## Eligible candidates (section 7.3)",
        "## Limitations",
        "Synthetic-sample-count confound",
        "First-infeasible censoring",
    ):
        assert required in markdown
    file = tmp_path / "ablation.json"
    file.write_text(ablation_to_json(report) + "\n", encoding="utf-8")
    assert load_ablation(file) == report
    tampered = json.loads(file.read_text(encoding="utf-8"))
    tampered["n_eligible"] = 2
    file.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="n_eligible"):
        load_ablation(file)


def test_mismatched_datasets_are_refused() -> None:
    """Every compared study must evaluate the same dataset."""
    timing = _report("study-timing", "no_augmentation", (_trial(1, feasible=True, value=0.5),))
    provenance = collect_provenance({}, seeds={}, artifacts=[], exploratory=True)
    with pytest.raises(ValueError, match="dataset"):
        build_ablation((("timing.json", timing),), _jumps(), dataset="processed-other", provenance=provenance)


def test_candidate_and_cell_invariants_are_enforced() -> None:
    """Cell verdicts and trial eligibility re-derive from their figures."""
    with pytest.raises(ValueError, match="verdict"):
        CandidateCell(gap_median=0.5, jump_median=0.5, improving_both=20, n=20, passes=False)
    with pytest.raises(ValueError, match="verdict"):
        CandidateCell(gap_median=0.5, jump_median=0.5, improving_both=14, n=20, passes=True)
    good = CandidateCell(gap_median=0.5, jump_median=0.5, improving_both=15, n=20, passes=True)
    cells = {
        "posture_small:pd_v2": good,
        "posture_small:computed_torque": good,
        "posture_large:pd_v2": good,
        "posture_large:computed_torque": good,
    }
    with pytest.raises(ValueError, match="contradicts"):
        CandidateTrial(study="s", number=1, value=0.5, warmup_s=0.0, cells=cells, eligible=False)
    with pytest.raises(ValueError, match="exactly"):
        CandidateTrial(
            study="s",
            number=1,
            value=0.5,
            warmup_s=0.0,
            cells={"posture_small:pd_v2": good},
            eligible=True,
        )
