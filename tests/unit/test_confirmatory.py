# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-028: the locked confirmatory protocol schema and its consistency checks."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from arm_rc_ctrl.experiments.confirmatory import (
    ConfirmatoryProtocol,
    ForcePulseLevels,
    PostureLevels,
    check_against_pilot,
    load_confirmatory,
)
from arm_rc_ctrl.experiments.perturbation_pilot import PilotReport, Selection, SelectionRules
from arm_rc_ctrl.provenance import collect_provenance

SELECTION = Selection(0.05, 0.06, 12.0, 2.0, 0.2, (0.0, 90.0, 180.0, 270.0))
RULES = SelectionRules(0.1, 1.0, 1.0, 0.01, 0.0)


def _protocol(**overrides: object) -> ConfirmatoryProtocol:
    base = ConfirmatoryProtocol(
        name="task-1a-confirmatory",
        scenario=Path("configs/tasks/task_1a.toml"),
        locked=True,
        pilot_report=Path("docs/experiments/task_1a/perturbation_pilot.json"),
        seeds=(1, 2, 3),
        posture=PostureLevels(0.05, 0.06, 4),
        force=ForcePulseLevels(12.0, 2.0, 0.2, (0.0, 90.0, 180.0, 270.0)),
    )
    return dataclasses.replace(base, **overrides)


def _report(
    selection: Selection = SELECTION, *, dirty: bool = False, scenario_file: str = "configs/tasks/task_1a.toml"
) -> PilotReport:
    provenance = collect_provenance({"unit": True}, seeds={}, artifacts=[], exploratory=True)
    return PilotReport(
        protocol="pilot",
        scenario_file=scenario_file,
        dataset="processed-20260830-feaf73e6663c",
        baselines={},
        rules=RULES,
        cases=(),
        levels=(),
        selection=selection,
        provenance=dataclasses.replace(provenance, project_dirty=dirty),
    )


def test_round_trip_through_toml(tmp_path: Path) -> None:
    """A locked protocol loads back from TOML with paths resolved next to the file."""
    file = tmp_path / "confirmatory.toml"
    file.write_text(
        'name = "t"\nscenario = "task.toml"\nlocked = true\npilot_report = "pilot.json"\nseeds = [7, 8]\n'
        "[posture]\nsmall_magnitude_rad = 0.05\nlarge_magnitude_rad = 0.06\ndraws_per_seed = 4\n"
        "[force]\nmagnitude_n = 12.0\nstart_s = 2.0\nduration_s = 0.2\ndirections_deg = [0.0, 180.0]\n"
    )
    protocol = load_confirmatory(file)
    assert protocol.scenario == tmp_path / "task.toml"
    assert protocol.pilot_report == tmp_path / "pilot.json"
    assert protocol.seeds == (7, 8)
    assert protocol.force.directions_deg == (0.0, 180.0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"locked": False}, "must be locked"),
        ({"seeds": ()}, "distinct non-negative"),
        ({"seeds": (1, 1)}, "distinct non-negative"),
        ({"seeds": (-1,)}, "distinct non-negative"),
        ({"name": ""}, "name must not be empty"),
    ],
)
def test_invalid_protocols(overrides: dict[str, object], message: str) -> None:
    """Lock flag, seeds, and name are validated."""
    with pytest.raises(ValueError, match=message):
        _protocol(**overrides)


def test_invalid_levels() -> None:
    """Posture ordering, draw counts, and force parameters are validated."""
    with pytest.raises(ValueError, match="0 < small <= large"):
        PostureLevels(0.06, 0.05, 4)
    with pytest.raises(ValueError, match="draws_per_seed must be >= 1"):
        PostureLevels(0.05, 0.06, 0)
    with pytest.raises(ValueError, match="magnitude_n > 0"):
        ForcePulseLevels(0.0, 2.0, 0.2, (0.0,))
    with pytest.raises(ValueError, match="non-empty and distinct"):
        ForcePulseLevels(12.0, 2.0, 0.2, (0.0, 0.0))


def test_confirmatory_seeds_must_not_be_development_seeds() -> None:
    """Sharing a seed with a development study is refused."""
    protocol = _protocol(seeds=(20260830, 5))
    protocol.forbid_seeds([1, 2], "nothing")
    with pytest.raises(ValueError, match=r"seeds \[20260830\] are also used by the gain study"):
        protocol.forbid_seeds([20260830], "the gain study")


def test_lock_must_match_the_pilot_selection() -> None:
    """Every locked level equals the pilot's selection; the pilot must be clean and of the same scenario."""
    check_against_pilot(_protocol(), _report())
    with pytest.raises(ValueError, match=r"posture\.large_magnitude_rad = 0\.07 differs"):
        check_against_pilot(_protocol(posture=PostureLevels(0.05, 0.07, 4)), _report())
    with pytest.raises(ValueError, match=r"force\.magnitude_n = 8\.0 differs"):
        check_against_pilot(_protocol(force=ForcePulseLevels(8.0, 2.0, 0.2, SELECTION.force_directions_deg)), _report())
    with pytest.raises(ValueError, match=r"force\.directions_deg"):
        check_against_pilot(_protocol(force=ForcePulseLevels(12.0, 2.0, 0.2, (0.0,))), _report())
    with pytest.raises(ValueError, match="dirty worktree"):
        check_against_pilot(_protocol(), _report(dirty=True))
    with pytest.raises(ValueError, match="differs from the pilot's"):
        check_against_pilot(_protocol(), _report(scenario_file="tests/fixtures/configs/planar_2dof_fixture.toml"))
