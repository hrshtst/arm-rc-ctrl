# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-010: the paired comparisons of the manual study (plan section 6).

Every comparison is made within one configuration and one tracker, over the
scenarios of one class, and reports both measures the owner asked for: the
difference in success counts over the scenarios both arms were run on, with
that denominator stated, and the per-scenario tally of improved, worsened and
tied cases. Across the ten parents the ten differences are listed with their
median and range, and the all-ten arm is counted once in any total, however
many comparisons reuse it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.experiments.manual_contrasts import (
    ALL_CLASSES,
    CONTRASTS,
    PARENTS,
    ScenarioVerdict,
    arm_summaries,
    contrast_rows,
    contrast_summaries,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from arm_rc_ctrl.experiments.manual_contrasts import ManualContrastRow, ManualContrastSummary

CONFIG, TRACKER = "feasible-best", "pd_v2"
SCENARIOS = (
    ("nominal", "nominal"),
    ("small-a", "posture_small"),
    ("small-b", "posture_small"),
    ("force-a", "force"),
)
"""A study in miniature: four scenarios in three classes, in frozen order."""


def _verdicts(
    outcomes: Mapping[tuple[str, str | None], Mapping[str, bool | None]],
    *,
    configuration: str = CONFIG,
    tracker: str = TRACKER,
) -> list[ScenarioVerdict]:
    """``(arm kind, parent) -> {scenario: success}``, filling every omitted arm with success everywhere."""
    classes = dict(SCENARIOS)
    keys: list[tuple[str, str | None]] = [("M10", None)]
    keys += [(kind, parent) for kind in ("S", "R10", "C10", "replay") for parent in PARENTS]
    verdicts: list[ScenarioVerdict] = []
    for kind, parent in keys:
        chosen = outcomes.get((kind, parent), {})
        verdicts.extend(
            ScenarioVerdict(
                configuration=configuration,
                tracker=tracker,
                arm_kind=kind,
                parent=parent,
                scenario_id=scenario_id,
                scenario_class=classes[scenario_id],
                success=chosen.get(scenario_id, True),
            )
            for scenario_id, _ in SCENARIOS
        )
    return verdicts


def _row(rows: Sequence[ManualContrastRow], contrast: str, parent: str, scenario_class: str) -> ManualContrastRow:
    matches = [r for r in rows if (r.contrast, r.parent, r.scenario_class) == (contrast, parent, scenario_class)]
    assert len(matches) == 1, f"expected exactly one {contrast} {parent} {scenario_class} row, got {len(matches)}"
    return matches[0]


def _summary(summaries: Sequence[ManualContrastSummary], contrast: str, scenario_class: str) -> ManualContrastSummary:
    matches = [s for s in summaries if (s.contrast, s.scenario_class) == (contrast, scenario_class)]
    assert len(matches) == 1, f"expected exactly one {contrast} {scenario_class} summary, got {len(matches)}"
    return matches[0]


# --- one paired comparison -----------------------------------------------------------------------


def test_the_all_ten_arm_is_compared_with_each_singleton_on_their_shared_scenarios() -> None:
    """Both measures: the difference in success counts, and which scenarios moved which way."""
    verdicts = _verdicts({("S", "D03"): {"small-a": True, "small-b": False}})
    rows = contrast_rows(verdicts, scenarios=SCENARIOS)
    row = _row(rows, "M10-S", "D03", "posture_small")
    assert (row.arm_a, row.arm_b) == ("M10", "S/D03")
    assert (row.successes_a, row.successes_b, row.difference) == (2, 1, 1)
    assert (row.improved, row.worsened, row.tied_success, row.tied_failure) == (1, 0, 1, 0)


def test_every_count_states_its_denominator() -> None:
    """A difference of one means nothing until the reader knows it is one of two scenarios, or of sixty-five."""
    rows = contrast_rows(_verdicts({}), scenarios=SCENARIOS)
    small = _row(rows, "M10-S", "D01", "posture_small")
    everything = _row(rows, "M10-S", "D01", ALL_CLASSES)
    assert (small.n_scenarios, small.n_shared) == (2, 2)
    assert (everything.n_scenarios, everything.n_shared) == (4, 4)


def test_each_planned_contrast_pairs_the_arms_the_plan_names() -> None:
    """The four arm contrasts, and every arm against the replay of its own parent."""
    rows = contrast_rows(_verdicts({}), scenarios=SCENARIOS)
    pairs = {r.contrast: (r.arm_a, r.arm_b) for r in rows if r.parent == "D07" and r.scenario_class == ALL_CLASSES}
    assert pairs == {
        "M10-S": ("M10", "S/D07"),
        "M10-R10": ("M10", "R10/D07"),
        "C10-R10": ("C10/D07", "R10/D07"),
        "M10-C10": ("M10", "C10/D07"),
        "S-replay": ("S/D07", "replay/D07"),
        "R10-replay": ("R10/D07", "replay/D07"),
        "C10-replay": ("C10/D07", "replay/D07"),
        "M10-replay": ("M10", "replay/D07"),
    }
    assert tuple(pairs) == tuple(name for name, _, _ in CONTRASTS)


def test_configurations_and_trackers_are_never_pooled() -> None:
    """Every row belongs to exactly one configuration and one tracker, and each gets its own rows."""
    verdicts = _verdicts({}) + _verdicts({}, tracker="computed_torque") + _verdicts({}, configuration="feasible-worst")
    rows = contrast_rows(verdicts, scenarios=SCENARIOS)
    keys = {(r.configuration, r.tracker) for r in rows}
    assert keys == {(CONFIG, TRACKER), (CONFIG, "computed_torque"), ("feasible-worst", TRACKER)}
    per_key = len(CONTRASTS) * len(PARENTS) * 4  # three classes and all of them together
    assert len(rows) == 3 * per_key


# --- missing evidence -----------------------------------------------------------------------------


def test_a_missing_run_leaves_a_comparison_partial_rather_than_failed() -> None:
    """An unavailable run is never counted as a failure; the comparison shrinks and says so."""
    verdicts = _verdicts({("S", "D02"): {"small-a": None, "small-b": False}})
    row = _row(contrast_rows(verdicts, scenarios=SCENARIOS), "M10-S", "D02", "posture_small")
    assert row.status == "partial"
    assert (row.n_scenarios, row.n_shared) == (2, 1)
    assert (row.successes_a, row.successes_b, row.improved) == (1, 0, 1)


def test_a_model_with_no_evidence_is_unavailable() -> None:
    """Nothing is compared, and the row says so rather than reporting a tie."""
    verdicts = _verdicts({("C10", "D04"): dict.fromkeys(("nominal", "small-a", "small-b", "force-a"))})
    row = _row(contrast_rows(verdicts, scenarios=SCENARIOS), "C10-R10", "D04", ALL_CLASSES)
    assert row.status == "unavailable"
    assert row.n_shared == 0


# --- across the ten parents -----------------------------------------------------------------------


def test_the_ten_parent_differences_are_listed_with_their_median_and_range() -> None:
    """All ten singletons, in parent order, with the number improved, worsened and tied."""
    # M10 succeeds on both small cases; S/D01-D03 fail both, S/D04-D06 fail one, S/D07-D10 fail none.
    failures: dict[tuple[str, str | None], dict[str, bool | None]] = {}
    for parent in PARENTS[:3]:
        failures["S", parent] = {"small-a": False, "small-b": False}
    for parent in PARENTS[3:6]:
        failures["S", parent] = {"small-a": False}
    summaries = contrast_summaries(contrast_rows(_verdicts(failures), scenarios=SCENARIOS))
    summary = _summary(summaries, "M10-S", "posture_small")
    assert summary.differences == (2, 2, 2, 1, 1, 1, 0, 0, 0, 0)
    assert (summary.median, summary.minimum, summary.maximum) == (1.0, 0, 2)
    assert (summary.parents_improved, summary.parents_worsened, summary.parents_tied) == (6, 0, 4)
    assert (summary.n_parents, summary.n_scenarios) == (10, 2)


def test_a_parent_whose_comparison_is_incomplete_is_left_out_of_the_summary_and_named() -> None:
    """Its difference is absent rather than zero, and the parent count says nine, not ten."""
    verdicts = _verdicts({("S", "D05"): {"small-a": None}})
    summaries = contrast_summaries(contrast_rows(verdicts, scenarios=SCENARIOS))
    summary = _summary(summaries, "M10-S", "posture_small")
    assert summary.differences[4] is None
    assert summary.n_parents == 9


# --- class summaries: the all-ten arm counted once --------------------------------------------------


def test_the_all_ten_arm_is_counted_once_in_the_arm_totals() -> None:
    """M10 is one model: its runs are counted once, however many comparisons reuse it."""
    verdicts = _verdicts({("M10", None): {"small-a": False}, ("S", "D01"): {"small-a": False}})
    summaries = arm_summaries(verdicts, scenarios=SCENARIOS)
    m10 = next(s for s in summaries if (s.arm_kind, s.scenario_class) == ("M10", "posture_small"))
    singles = next(s for s in summaries if (s.arm_kind, s.scenario_class) == ("S", "posture_small"))
    assert (m10.n_models, m10.n_runs, m10.successes, m10.per_model) == (1, 2, 1, (1,))
    assert (singles.n_models, singles.n_runs, singles.successes) == (10, 20, 19)
    assert singles.per_model == (1, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    assert (singles.median, singles.minimum, singles.maximum) == (2.0, 1, 2)


def test_an_arm_total_counts_only_the_runs_that_exist() -> None:
    """Missing runs shrink the denominator and are counted apart, never as failures."""
    verdicts = _verdicts({("replay", "D09"): {"nominal": None}})
    summaries = arm_summaries(verdicts, scenarios=SCENARIOS)
    nominal = next(s for s in summaries if (s.arm_kind, s.scenario_class) == ("replay", "nominal"))
    assert (nominal.n_models, nominal.n_runs, nominal.unavailable_runs, nominal.successes) == (9, 9, 1, 9)
    assert nominal.per_model[8] is None


def test_a_model_missing_part_of_a_class_is_left_out_of_that_class_and_its_runs_are_counted_apart() -> None:
    """Its existing runs are excluded rather than scored against a smaller denominator."""
    verdicts = _verdicts({("R10", "D03"): {"small-a": None}})
    summaries = arm_summaries(verdicts, scenarios=SCENARIOS)
    small = next(s for s in summaries if (s.arm_kind, s.scenario_class) == ("R10", "posture_small"))
    assert (small.n_models, small.n_runs, small.excluded_runs, small.unavailable_runs) == (9, 18, 1, 1)
    assert small.per_model[2] is None
    assert small.median == 2.0


# --- refused input ----------------------------------------------------------------------------------


def test_a_repeated_verdict_is_refused() -> None:
    """Two verdicts for one arm and scenario cannot both be kept, and neither may silently win."""
    verdicts = _verdicts({})
    with pytest.raises(ValueError, match="more than once"):
        contrast_rows([*verdicts, verdicts[0]], scenarios=SCENARIOS)


def test_a_scenario_outside_the_frozen_order_is_refused() -> None:
    """The class and the denominator come from the frozen order, so a stray scenario cannot be placed."""
    stray = ScenarioVerdict(CONFIG, TRACKER, "M10", None, "invented", "nominal", success=True)
    with pytest.raises(ValueError, match="frozen order"):
        contrast_rows([*_verdicts({}), stray], scenarios=SCENARIOS)


def test_a_verdict_whose_class_disagrees_with_the_frozen_order_is_refused() -> None:
    """A mislabelled class would move a scenario into another denominator."""
    wrong = ScenarioVerdict(CONFIG, TRACKER, "M10", None, "force-a", "posture_small", success=True)
    verdicts = [v for v in _verdicts({}) if not (v.arm_kind == "M10" and v.scenario_id == "force-a")]
    with pytest.raises(ValueError, match="class"):
        contrast_rows([*verdicts, wrong], scenarios=SCENARIOS)


@pytest.mark.parametrize(
    ("kind", "parent"),
    [("M10", "D01"), ("S", None), ("replay", None), ("X10", "D01"), ("S", "D11")],
)
def test_an_arm_that_names_the_wrong_parent_is_refused(kind: str, parent: str | None) -> None:
    """The all-ten arm has no parent; every other arm has exactly one of the ten."""
    with pytest.raises(ValueError, match=r"parent|arm"):
        ScenarioVerdict(CONFIG, TRACKER, kind, parent, "nominal", "nominal", success=True)
