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
