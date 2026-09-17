# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-008: what the manual evaluation configuration refuses (plan section 6, clarification I6).

The manual protocol reuses the locked development draws but evaluates them
under its own horizon, dwell rule, force timing, and abort. The configuration
binds those draws by reference rather than re-seeding them, and the loader
refuses a file whose abort or horizon contradicts the protocol it claims to
run. Candidate files are written to a scratch directory with absolute paths,
so no test ever leaves a configuration behind in the repository.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments.manual_evaluation import (
    ManualEvaluationConfig,
    ManualSimulationLimits,
    ManualTriggerRule,
    load_manual_evaluation_config,
    manual_trigger,
)
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from pathlib import Path

REPO_ROOT = repository_root()
DEVELOPMENT_FILE = REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml"
CONFIRMATORY_FILE = REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_confirmatory_v1.toml"
SCENARIO_FILE = REPO_ROOT / "configs" / "tasks" / "task_1a_manual_v2.toml"


def _written(tmp_path: Path, **changes: object) -> Path:
    """A candidate configuration in a scratch directory, overriding single fields of the committed one."""
    values: dict[str, object] = {
        "name": "task-1a-manual-dev-test",
        "development": DEVELOPMENT_FILE.as_posix(),
        "scenario": SCENARIO_FILE.as_posix(),
        "horizon_s": 30.0,
        "hold_s": 0.5,
        "duration_s": 0.2,
        "magnitude_n": 12.0,
        "velocity_abort": [6.0, 6.0],
    }
    values.update(changes)
    text = (
        f'name = "{values["name"]}"\n'
        f'development = "{values["development"]}"\n'
        f'scenario = "{values["scenario"]}"\n'
        f"horizon_s = {values['horizon_s']}\n\n"
        f"[trigger]\nhold_s = {values['hold_s']}\nduration_s = {values['duration_s']}\n"
        f"magnitude_n = {values['magnitude_n']}\n\n"
        f"[simulation]\nvelocity_abort = {values['velocity_abort']}\n"
    )
    target = tmp_path / "evaluation.toml"
    target.write_text(text, encoding="utf-8")
    return target


def _valid() -> ManualEvaluationConfig:
    return ManualEvaluationConfig(
        name="task-1a-manual-dev-test",
        development=DEVELOPMENT_FILE,
        scenario=SCENARIO_FILE,
        horizon_s=30.0,
        trigger=ManualTriggerRule(hold_s=0.5, duration_s=0.2, magnitude_n=12.0),
        simulation=ManualSimulationLimits(velocity_abort=(6.0, 6.0)),
    )


# --- what the configuration itself refuses ---------------------------------------------------


def test_a_confirmatory_development_file_is_refused() -> None:
    """Development diagnostics only: the confirmatory envelope is never evaluated by this pilot."""
    config = _valid()
    with pytest.raises(ValueError, match="development levels only"):
        ManualEvaluationConfig(
            name=config.name,
            development=CONFIRMATORY_FILE,
            scenario=config.scenario,
            horizon_s=config.horizon_s,
            trigger=config.trigger,
            simulation=config.simulation,
        )


def test_an_empty_name_is_refused() -> None:
    """Every versioned configuration identifies itself."""
    config = _valid()
    with pytest.raises(ValueError, match="name must not be empty"):
        ManualEvaluationConfig(
            name="  ",
            development=config.development,
            scenario=config.scenario,
            horizon_s=config.horizon_s,
            trigger=config.trigger,
            simulation=config.simulation,
        )


def test_a_non_positive_abort_is_refused() -> None:
    """The abort is a real per-joint speed bound, not a disabled check."""
    with pytest.raises(ValueError, match="positive finite"):
        ManualSimulationLimits(velocity_abort=(6.0, 0.0))


def test_a_non_positive_trigger_hold_is_refused() -> None:
    """The pulse is armed by a hold of real duration, never immediately on arrival."""
    with pytest.raises(ValueError, match="positive and finite"):
        ManualTriggerRule(hold_s=0.0, duration_s=0.2, magnitude_n=12.0)


# --- what the loader refuses against the scenario --------------------------------------------


def test_the_committed_scenario_loads_with_the_canonical_abort(tmp_path: Path) -> None:
    """The nominal case: the abort equals the task configuration's own per-joint limit."""
    config = load_manual_evaluation_config(_written(tmp_path))
    assert config.simulation.velocity_abort == (6.0, 6.0)
    assert config.development == DEVELOPMENT_FILE.resolve()


def test_a_relaxed_abort_is_refused(tmp_path: Path) -> None:
    """D3: the canonical 6 rad/s applies; the repetition pilot's twofold relaxation does not carry over."""
    with pytest.raises(ValueError, match="canonical"):
        load_manual_evaluation_config(_written(tmp_path, velocity_abort=[12.0, 12.0]))


def test_a_horizon_too_short_for_the_trigger_and_its_dwell_is_refused(tmp_path: Path) -> None:
    """A force case needs the hold, the pulse, and a full final dwell to fit inside the horizon."""
    with pytest.raises(ValueError, match="horizon"):
        load_manual_evaluation_config(_written(tmp_path, horizon_s=1.2))


# --- the trigger the configuration produces --------------------------------------------------


def test_the_trigger_takes_its_predicate_from_the_scenario(tmp_path: Path) -> None:
    """The dwell rule lives in the task configuration, so the trigger cannot drift from acquisition."""
    config = load_manual_evaluation_config(_written(tmp_path))
    scenario = load_manual_scenario(config.scenario)
    trigger = manual_trigger(config, scenario, direction_deg=90.0)
    assert trigger.tolerance_m == scenario.task.tolerance
    assert trigger.max_velocity_rad_s == scenario.task.dwell_max_velocity
    assert trigger.target == tuple(scenario.task.target)
    assert trigger.hold_s == config.trigger.hold_s
    assert trigger.duration_s == config.trigger.duration_s
    assert trigger.force[1] == pytest.approx(config.trigger.magnitude_n)
    # The inherited levels fix a pulse start; a state trigger structurally cannot carry one.
    assert not hasattr(trigger, "start_s")
