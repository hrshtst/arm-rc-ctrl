# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-008: the direct-replay baselines of one demonstration under the manual protocol.

Every S, R10, and C10 model is compared against replay of its own parent, so
the baselines are shared by every model that shares a parent and are keyed by
the protocol conditions together with that parent. Each scenario is attempted
independently from a fresh reset (D6), and a force case is disturbed when the
arm reaches the target rather than at a fixed task time (plan section 6).
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments.manual_evaluation import (
    ManualEvaluationRunner,
    load_manual_evaluation_config,
    replay_bank_uri,
)
from arm_rc_ctrl.experiments.perturbations import RobustnessScenario
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from pathlib import Path

    from arm_rc_ctrl.experiments.manual_evaluation import ManualEvaluationConfig
    from arm_rc_ctrl.experiments.manual_fixture import ManualFixture

REPO_ROOT = repository_root()
DEVELOPMENT_SOURCE = REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml"

TRACKER = TrackerConfig(type="pd", kp=(10.0, 5.0), kd=(1.5, 0.8))
"""Gains under which the fixture's replay tracks without saturating."""

HOLD_S = 0.05
PULSE_S = 0.02
HORIZON_S = 1.0
WARMUP_S = 0.25

SCENARIOS = (
    RobustnessScenario("nominal", "nominal", (0.0, 0.0)),
    RobustnessScenario("small-1", "posture_small", (0.02, -0.01), seed=1, draw=0, magnitude_rad=0.05),
    RobustnessScenario(
        "force-000deg",
        "force",
        (0.0, 0.0),
        force_magnitude_n=3.0,
        force_start_s=1.0,
        force_duration_s=PULSE_S,
        direction_deg=0.0,
    ),
)
"""A nominal case, a perturbed start, and a force case whose inherited fixed start must be superseded."""


def _evaluation(f: ManualFixture) -> tuple[ManualEvaluationConfig, Path]:
    """A manual evaluation configuration inside the fixture root, binding the locked draws and its own task.

    The fixture root is shared by every test in this module, so this writes the
    same two files every time rather than mutating anything a later test reads
    differently.
    """
    evaluations = f.root / "configs" / "evaluations"
    evaluations.mkdir(parents=True, exist_ok=True)
    development = evaluations / DEVELOPMENT_SOURCE.name
    shutil.copyfile(DEVELOPMENT_SOURCE, development)
    scenario = f.scenario_file
    limits = ", ".join(f"{v}" for v in load_manual_scenario(scenario).limits.velocity)
    target = evaluations / "task_1a_manual_dev_fixture.toml"
    target.write_text(
        f'name = "task-1a-manual-dev-fixture"\n'
        f'development = "{development.as_posix()}"\n'
        f'scenario = "{scenario.as_posix()}"\n'
        f"horizon_s = {HORIZON_S}\n\n"
        f"[trigger]\nhold_s = {HOLD_S}\nduration_s = {PULSE_S}\nmagnitude_n = 3.0\n\n"
        f"[simulation]\nvelocity_abort = [{limits}]\n",
        encoding="utf-8",
    )
    return load_manual_evaluation_config(target), target


def _runner(f: ManualFixture, log: list[str] | None = None) -> ManualEvaluationRunner:
    config, file = _evaluation(f)
    return ManualEvaluationRunner(
        store=f.store,
        inputs=f.inputs,
        config=config,
        evaluation_file=file,
        scenarios=SCENARIOS,
        trackers={"pd_v2": TRACKER, "computed_torque": TRACKER},
        root=f.root,
        execution=f.execution,
        provenance=f.provenance,
        log=(lambda _m: None) if log is None else log.append,
    )


# --- what one parent's bank contains ---------------------------------------------------------


def test_a_bank_covers_every_scenario_and_tracker_in_evaluation_order(manual_fixture: ManualFixture) -> None:
    """Both trackers of one scenario before the next scenario begins, each with a stored run."""
    bank = _runner(manual_fixture).replay_bank("D01", warmup_s=WARMUP_S)
    assert [(p.scenario_id, p.tracker) for p in bank.pairs] == [
        (case.scenario_id, tracker) for case in SCENARIOS for tracker in ("pd_v2", "computed_torque")
    ]
    assert all(pair.arm == "replay" for pair in bank.pairs)
    assert all(pair.run is not None for pair in bank.pairs)
    assert bank.assignment == "D01"


def test_each_scenario_starts_from_its_own_reset_posture(manual_fixture: ManualFixture) -> None:
    """Every case is attempted independently from a fresh reset, at the configured posture plus its offset."""
    bank = _runner(manual_fixture).replay_bank("D01", warmup_s=WARMUP_S)
    nominal = next(p for p in bank.pairs if p.scenario_id == "nominal")
    offset = next(p for p in bank.pairs if p.scenario_id == "small-1")
    assert offset.initial_q != nominal.initial_q
    assert offset.initial_q == pytest.approx(
        tuple(a + b for a, b in zip(nominal.initial_q, (0.02, -0.01), strict=True))
    )


def test_a_force_case_records_the_realised_trigger_not_the_inherited_start(manual_fixture: ManualFixture) -> None:
    """The pulse fires when the measured motion holds the target, so its timestamp is its own, not 1.0 s."""
    bank = _runner(manual_fixture).replay_bank("D01", warmup_s=WARMUP_S)
    forced = [p for p in bank.pairs if p.scenario_id == "force-000deg"]
    assert len(forced) == 2
    for pair in forced:
        assert pair.outcome is not None
        trigger = pair.outcome.trigger
        assert trigger is not None
        # Either it fired, and then at a state-derived time that is never the inherited fixed start,
        # or it never fired, and then the case says so explicitly instead of passing as an ordinary run.
        if trigger.triggered:
            assert pair.pulse_start_s is not None
            assert pair.pulse_start_s == pytest.approx(trigger.pulse_start_s)
            assert pair.pulse_start_s != pytest.approx(WARMUP_S + 1.0)
        else:
            assert pair.pulse_start_s is None
            assert not trigger.ok
            assert trigger.reason is not None
            assert "never" in trigger.reason


def test_the_bank_identity_binds_the_conditions_and_the_parent(manual_fixture: ManualFixture) -> None:
    """Two parents under one protocol are two banks; the same parent under one protocol is one."""
    runner = _runner(manual_fixture)
    first = runner.replay_bank("D01", warmup_s=WARMUP_S)
    second = runner.replay_bank("D02", warmup_s=WARMUP_S)
    assert first.identity != second.identity
    assert first.conditions.identity == second.conditions.identity
    assert replay_bank_uri(first.conditions, "D01") != replay_bank_uri(second.conditions, "D02")
    again = runner.replay_bank("D01", warmup_s=WARMUP_S)
    assert again.identity == first.identity
