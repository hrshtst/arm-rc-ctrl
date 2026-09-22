# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MS-001: the committed search protocol says what the approved plan says.

The plan is the owner's approved protocol and the TOML file is what the search
will actually run under. This lock compares the second with the first, so the
two cannot drift: every searched range, the fixed conditions, the caps and the
selection count are read out of the plan's own tables rather than restated
here.
"""

from __future__ import annotations

import re

import pytest

from arm_rc_ctrl.experiments.manual_search import load_manual_search, protocol_digest, scope_mismatches
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

ROOT = repository_root()
PROTOCOL = ROOT / "configs/studies/manual_esn_search_v1.toml"
PLAN = ROOT / "docs/experiments/task_1a_manual_esn_search/plan.md"


def _plan_rows() -> dict[str, str]:
    """The approved domain table of plan section 2, by parameter name."""
    rows: dict[str, str] = {}
    for line in PLAN.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")] if line.startswith("|") else []
        if len(cells) == 2 and cells[0] not in ("Parameter", "---", "Decision"):
            rows.setdefault(cells[0], cells[1])
    return rows


def _numbers(text: str) -> list[float]:
    return [float(value) for value in re.findall(r"\d+(?:\.\d+)?", text)]


def test_every_searched_range_is_the_one_the_plan_approved() -> None:
    """A range the plan does not carry would search outside the owner's approval."""
    protocol = load_manual_search(PROTOCOL)
    rows = _plan_rows()
    space = protocol.space
    assert _numbers(rows["Reservoir size"]) == [space.n_neurons.low, space.n_neurons.high, space.n_neurons.step]
    assert _numbers(rows["Spectral radius"]) == [space.spectral_radius.low, space.spectral_radius.high]
    assert _numbers(rows["Sparsity"]) == [space.sparsity.low, space.sparsity.high]
    assert _numbers(rows["Leak rate"]) == [space.leak_rate.low, space.leak_rate.high]
    assert _numbers(rows["Input scaling"]) == [space.input_scaling.low, space.input_scaling.high]
    assert _numbers(rows["Base readout regularization `alpha_0`"]) == [space.alpha_0.low, space.alpha_0.high]
    assert _numbers(rows["Warm-up"]) == list(space.warmup_s)
    for name, row in (
        ("leak rate", rows["Leak rate"]),
        ("input scaling", rows["Input scaling"]),
        ("alpha_0", rows["Base readout regularization `alpha_0`"]),
    ):
        assert "logarithmic" in row.lower(), name


def test_the_fixed_conditions_are_the_ones_the_plan_fixes() -> None:
    """Seed, trackers and the inherited filter policy are fixed by the plan, not by this file."""
    protocol = load_manual_search(PROTOCOL)
    plan = PLAN.read_text(encoding="utf-8")
    assert f"seed **{protocol.fixed.reservoir_seed} is fixed**" in plan
    for tracker in protocol.fixed.trackers:
        assert f"`{tracker}`" in plan
    for value in (
        protocol.fixed.velocity_cutoff_hz,
        protocol.fixed.acceleration_cutoff_hz,
        protocol.fixed.max_dt_ratio,
    ):
        assert f"{value}" in plan, value
    assert protocol.fixed.filters.name in plan


def test_the_caps_and_the_selection_count_are_the_approved_ones() -> None:
    """The plan's budget sentence and the protocol's caps are the same numbers."""
    protocol = load_manual_search(PROTOCOL)
    plan = PLAN.read_text(encoding="utf-8")
    assert f"**{protocol.budget.trials} total Optuna trials**" in plan
    assert f"**{protocol.budget.hours:g} hours elapsed execution and {protocol.budget.gib:g} GiB" in plan
    assert protocol.budget.shared_with_comparison is True, "the plan applies one ceiling to search and comparison"
    assert protocol.budget.parallel_trials == 1, "plan section 7 starts serial"
    assert "**three highest-scoring configurations**" in plan
    assert protocol.selection.n_configurations == 3


def test_the_optimizer_cannot_reach_a_perturbed_case() -> None:
    """The committed protocol keeps the separation the experiment depends on."""
    protocol = load_manual_search(PROTOCOL)
    assert protocol.objective.scenarios == ("nominal",)
    assert scope_mismatches(protocol) == []
    assert protocol.comparison.scenarios == 65, "the comparison, and only the comparison, sees all 65 cases"
    assert protocol.comparison.sequential is True


def test_the_protocol_identity_is_recorded_for_the_tasks_that_run_it() -> None:
    """M3MS-003 resumes a study by this identity, so it is pinned here.

    The identity binds the study's content as well as its portable location, so
    a resume cannot land on a different study that merely shares a file name.
    """
    assert (
        protocol_digest(load_manual_search(PROTOCOL))
        == "cf1869098a6324a8a082f8df61238d4cca41692743e1d1d7e1a0606ff837c584"
    )
