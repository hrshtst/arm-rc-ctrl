# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-019: the recovery reproduction's comparisons, step capture, and audit rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.experiments.recovery_representative import SELECTION_RULE, PairOutcome, RepresentativeRecord
from arm_rc_ctrl.experiments.reproduce_1a import Check, ReproductionError
from arm_rc_ctrl.experiments.reproduce_recovery import (
    STEPS,
    RecoveryReproduction,
    Reproducer,
    _rebuilt_cells,  # pyright: ignore[reportPrivateUsage]
    _stored_components,  # pyright: ignore[reportPrivateUsage]
    animation_names,
    audit_markdown,
    compare_evidence,
)
from arm_rc_ctrl.provenance import collect_provenance

if TYPE_CHECKING:
    from pathlib import Path


def testcompare_evidence_tracks_float_deviations_and_categorical_differences() -> None:
    """Floats accumulate the largest deviation; every other mismatch is categorical."""
    differences: list[str] = []
    worst = compare_evidence(
        "root",
        {"a": 1.0, "b": {"c": [1, 2.5]}, "d": "x", "e": None, "f": True},
        {"a": 1.25, "b": {"c": [1, 2.0]}, "d": "x", "e": None, "f": True},
        differences,
    )
    assert worst == pytest.approx(0.5)
    assert differences == []
    differences = []
    compare_evidence("root", {"a": "x", "b": [1], "c": True}, {"a": "y", "b": [1, 2], "c": False}, differences)
    assert len(differences) == 3
    differences = []
    compare_evidence("root", {"a": 1.0}, {"b": 1.0}, differences)
    assert differences
    assert "keys" in differences[0]


def test_reproduction_ok_requires_every_step() -> None:
    """A truncated or failing run is never ok."""
    passing = tuple(Check(name, ok=True, detail="", elapsed_s=0.0) for name in STEPS)
    good = RecoveryReproduction(
        started_at="t", checks=passing, inputs={}, environment={}, max_deviation=0.0, elapsed_s=1.0
    )
    assert good.ok
    truncated = RecoveryReproduction(
        started_at="t", checks=passing[:-1], inputs={}, environment={}, max_deviation=0.0, elapsed_s=1.0
    )
    assert not truncated.ok
    failing = RecoveryReproduction(
        started_at="t",
        checks=(*passing[:-1], Check(STEPS[-1], ok=False, detail="boom", elapsed_s=0.0)),
        inputs={},
        environment={},
        max_deviation=None,
        elapsed_s=1.0,
    )
    assert not failing.ok


def test_step_captures_failures_by_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising step becomes a named, failed check instead of an exception."""
    reproducer = Reproducer(tmp_path, 0.0, False, None, tmp_path, None)  # noqa: FBT003 - positional dataclass field

    def boom() -> str:
        msg = "missing input"
        raise ReproductionError(msg)

    monkeypatch.setattr(reproducer, "environment", boom)
    check = reproducer.step("environment")
    assert not check.ok
    assert check.name == "environment"
    assert "missing input" in check.detail


def testanimation_names_follow_the_export_order() -> None:
    """The re-rendered report embeds the same animation names the export produced."""
    pairs = tuple(
        PairOutcome(
            scenario_id=f"s-{kind}",
            kind=kind,
            tracker=tracker,
            replay_run=f"run-replay-{kind}-{tracker}",
            rc_run=f"run-rc-{kind}-{tracker}",
            activation_s=0.25,
            replay_completed=True,
            rc_completed=False,
            recovery=None,
        )
        for kind in ("nominal", "posture_small", "posture_large", "force")
        for tracker in ("pd_v2", "computed_torque")
    )
    record = RepresentativeRecord(
        study="s",
        trial=17,
        point_params={"warmup_s": 0.25},
        warmup_s=0.25,
        dataset="processed-test",
        selection_rule=SELECTION_RULE,
        scenarios={
            "nominal": "s-nominal",
            "posture_small": "s-posture_small",
            "posture_large": "s-posture_large",
            "force": "s-force",
        },
        pairs=pairs,
        provenance=collect_provenance({}, seeds={}, artifacts=[], exploratory=True),
    )
    assert animation_names(record) == (
        "nominal_rc_pd.gif",
        "nominal_replay_pd.gif",
        "posture_small_rc_pd.gif",
        "posture_small_replay_pd.gif",
        "posture_large_rc_pd.gif",
        "posture_large_replay_pd.gif",
        "force_rc_pd.gif",
        "force_replay_pd.gif",
    )


def test_audit_markdown_renders_every_check() -> None:
    """The audit note lists the command, every step, and the recorded inputs."""
    result = RecoveryReproduction(
        started_at="2026-09-04T00:00:00+00:00",
        checks=(Check("environment", ok=True, detail="ok", elapsed_s=0.1),),
        inputs={"dataset": "processed-test"},
        environment={"python": "3.12"},
        max_deviation=0.0,
        elapsed_s=1.0,
    )
    text = audit_markdown(result, command="python scripts/reproduce_recovery.py")
    assert text.startswith("# Task 1-a recovery reproduction")
    assert "| environment | True | 0.1 | ok |" in text
    assert "- dataset: `processed-test`" in text
    assert "`python scripts/reproduce_recovery.py`" in text
    assert "Auditor: (to be filled by the auditor)" in text
    assert "Executor machine:" in text


def test_stored_components_parse_the_flattened_trial() -> None:
    """Every persisted per-pair field round-trips out of the flattened trial record."""
    from arm_rc_ctrl.experiments.studies import TrialRecord

    trial = TrialRecord(
        number=17,
        state="COMPLETE",
        value=0.5,
        params={"warmup_s": 0.25},
        metrics={
            "components.0.gap_ratio": 0.5,
            "components.0.activation_jump_rad": 0.03,
            "components.0.settling_time_s": 0.2,
        },
        flags={
            "feasible": True,
            "components.0.feasible": True,
            "components.0.criteria.completed": True,
            "components.0.generated_criteria.generated_dwell_stationary": True,
        },
        labels={
            "components.0.kind": "posture_small",
            "components.0.tracker": "pd_v2",
            "components.0.scenario_id": "s-1",
            "components.0.termination": "completed",
        },
    )
    stored = _stored_components(trial)
    fields = stored[("s-1", "pd_v2")]
    assert fields["kind"] == "posture_small"
    assert fields["feasible"] is True
    assert fields["gap_ratio"] == 0.5
    assert fields["criteria"] == {"completed": True}
    assert fields["generated_criteria"] == {"generated_dwell_stationary": True}
    assert "torque_rms" not in fields


def test_rebuilt_cells_rederive_medians_and_improvement_counts() -> None:
    """The eligibility cells recompute from a re-evaluation plus replay activation jumps."""
    from arm_rc_ctrl.experiments.esn_search import TrialPoint
    from arm_rc_ctrl.experiments.recovery_objective import RecoveryComponent, RecoveryTrialEvaluation
    from arm_rc_ctrl.experiments.recovery_search import RecoveryTrialPoint

    def component(scenario_id: str, kind: str, tracker: str, gap: float, jump: float) -> RecoveryComponent:
        return RecoveryComponent(
            index=0,
            scenario_id=scenario_id,
            kind=kind,  # type: ignore[arg-type]
            tracker=tracker,
            initial_q=(0.0, 0.0),
            termination="completed",
            criteria={"completed": True},
            generated_criteria={},
            feasible=True,
            reason=None,
            gap_ratio=gap,
            activation_jump_rad=jump,
        )

    components = (
        component("a", "posture_small", "pd_v2", 0.4, 0.02),
        component("b", "posture_small", "pd_v2", 0.6, 0.09),
        component("n", "nominal", "pd_v2", 0.0, 0.0),
    )
    evaluation = RecoveryTrialEvaluation(
        point=RecoveryTrialPoint(esn=TrialPoint(100, 0.9, 0.9, 0.1, 0.1, 31, 0.01, 20.0, 20.0), warmup_s=0.25),
        objective=0.5,
        feasible=False,
        penalized=False,
        reason="stopped by the pruner",
        fit_rmse=0.001,
        scenarios_total=6,
        components=components,
        cells={},
        running=(0.5,),
        stopped_early=True,
    )
    jumps = {("pd_v2", "a"): 0.05, ("pd_v2", "b"): 0.05}
    cells = _rebuilt_cells(evaluation, jumps)
    gap_median, jump_median, improving, n = cells["posture_small:pd_v2"]
    assert gap_median == pytest.approx(0.5)
    assert jump_median == pytest.approx((0.02 / 0.05 + 0.09 / 0.05) / 2)
    assert improving == 1  # only scenario "a" improves both metrics
    assert n == 2
